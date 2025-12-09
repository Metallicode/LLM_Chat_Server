from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
import requests
import json
import os
from datetime import datetime
import sqlite3
from dotenv import load_dotenv


# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__, static_folder="static")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_DB_PATH = os.path.join(BASE_DIR, "memories.db")


KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge")
RAG_DB_PATH = os.path.join(BASE_DIR, "rag.db")


OLLAMA_URL = os.environ.get("OLLAMA_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")


# Derive embed URL from the chat URL
if "/api/chat" in OLLAMA_URL:
    OLLAMA_BASE = OLLAMA_URL.rsplit("/api/chat", 1)[0]
else:
    OLLAMA_BASE = OLLAMA_URL  # fallback
OLLAMA_EMBED_URL = f"{OLLAMA_BASE}/api/embed"

# Embedding model name (you'll need to `ollama pull` this on the box)
EMBED_MODEL = os.environ.get("EMBED_MODEL") or "nomic-embed-text"

# Tavily configuration
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


# === TAVILY WEB SEARCH =======================================================

def tavily_search(query, topic="general"):
    """
    Call Tavily Search API and return a nicely formatted text block
    plus an error (if any).

    Returns: (text_block: str | None, error_message: str | None)
    """
    if not TAVILY_API_KEY:
        return None, "Tavily API key not configured. Set TAVILY_API_KEY env var."

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    # For cost control: basic depth, small max_results.
    # include_answer=True so Tavily gives a direct answer string.
    body = {
        "query": query,
        "topic": topic,             # "general" or "news" etc
        "search_depth": "basic",    # cheaper; "advanced" is richer but 2 credits
        "max_results": 5,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }

    try:
        r = requests.post(TAVILY_SEARCH_URL, headers=headers, json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"Tavily search failed: {e}"

    # Tavily response example is in docs:
    # { "query": "...", "answer": "...", "results": [ {title,url,content,...}, ...], ... }
    answer = data.get("answer") or ""
    results = data.get("results") or []

    lines = []
    lines.append(f"Tavily web search results for: {query!r}")
    lines.append("Time (UTC): " + datetime.utcnow().isoformat() + "Z")
    lines.append("")

    if answer.strip():
        lines.append("Tavily answer:")
        lines.append(answer.strip())
        lines.append("")

    if results:
        lines.append("Top sources:")
        for res in results[:5]:
            title = res.get("title") or res.get("url") or "Untitled"
            url = res.get("url") or ""
            snippet = (res.get("content") or "").strip()
            # Keep snippet short-ish
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            lines.append(f"- {title}\n  {url}")
            if snippet:
                lines.append(f"  Snippet: {snippet}")
        lines.append("")

    if not answer.strip() and not results:
        lines.append("(No useful web results were found.)")

    return "\n".join(lines), None

# === RAG (docs) SQLite ======================================================

def init_rag_db():
    """Create the RAG docs table if it doesn't exist."""
    conn = sqlite3.connect(RAG_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200):
    """
    Simple character-based chunking with overlap.
    Good enough for small RAG and avoids needing a tokenizer.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == n:
            break
        start = end - overlap  # overlap backwards
        if start < 0:
            start = 0
    return chunks


def clear_rag_index():
    """Delete all rows from docs table."""
    conn = sqlite3.connect(RAG_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM docs")
        conn.commit()
    finally:
        conn.close()


def index_rag_folder():
    """
    Index all text files in KNOWLEDGE_DIR into rag.db.
    Returns the number of chunks indexed.
    """
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    clear_rag_index()

    chunk_count = 0
    conn = sqlite3.connect(RAG_DB_PATH)
    try:
        cur = conn.cursor()
        for root, dirs, files in os.walk(KNOWLEDGE_DIR):
            for name in files:
                if not name.lower().endswith((".txt", ".md")):
                    continue
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, BASE_DIR)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"[WARN] Failed to read {path}: {e}")
                    continue

                chunks = chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    emb = embed_text(chunk)
                    emb_json = json.dumps(emb) if emb is not None else None
                    cur.execute(
                        """
                        INSERT INTO docs (file_path, chunk_index, text, embedding)
                        VALUES (?, ?, ?, ?)
                        """,
                        (rel_path, idx, chunk, emb_json),
                    )
                    chunk_count += 1

        conn.commit()
    finally:
        conn.close()

    print(f"[INFO] RAG indexing complete: {chunk_count} chunks indexed")
    return chunk_count



def search_rag_chunks(query: str, top_k: int = 5, min_sim: float = 0.25):
    """
    Return top-k most semantically similar document chunks to the query.
    Each item: (file_path, chunk_index, text, similarity)
    """
    q_emb = embed_text(query)
    if q_emb is None:
        return []

    conn = sqlite3.connect(RAG_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT file_path, chunk_index, text, embedding FROM docs WHERE embedding IS NOT NULL")
        rows = cur.fetchall()
    finally:
        conn.close()

    scored = []
    for file_path, chunk_index, text, emb_json in rows:
        if not emb_json:
            continue
        try:
            emb = json.loads(emb_json)
        except Exception:
            continue
        sim = cosine_similarity(q_emb, emb)
        if sim >= min_sim:
            scored.append((file_path, chunk_index, text, sim))

    scored.sort(key=lambda t: t[3], reverse=True)
    return scored[:top_k]


def build_rag_context(user_message: str, max_items: int = 5):
    """
    Build an Ollama message (or []) containing relevant doc chunks for the given user message.
    """
    matches = search_rag_chunks(user_message, top_k=max_items)
    if not matches:
        return []

    lines = ["Relevant reference information from your documents:"]
    for file_path, chunk_index, text, sim in matches:
        lines.append(
            f"\n[File: {file_path}, chunk {chunk_index}, similarity {sim:.2f}]\n{text}\n"
        )

    return [{
        "role": "system",
        "content": "\n".join(lines),
    }]

# === RAG API (for UI / manual trigger) ======================================
## Reindex: curl -X POST http://localhost:5000/api/rag/reindex


@app.route("/api/rag/reindex", methods=["POST"])
def api_rag_reindex():
    """Reindex the knowledge folder into rag.db."""
    count = index_rag_folder()
    return jsonify({"success": True, "chunks_indexed": count})



# === MEMORY (SQLite) =========================================================

def init_mem_db():
    """Create the memories table if it doesn't exist, and ensure embedding column."""
    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

        # Ensure 'embedding' column exists
        cur.execute("PRAGMA table_info(memories)")
        cols = [row[1] for row in cur.fetchall()]
        if "embedding" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
            conn.commit()
    finally:
        conn.close()



# === EMBEDDINGS & SEMANTIC MEMORY SEARCH ====================================

def embed_text(text: str):
    """
    Get an embedding vector from Ollama for a given piece of text.
    Returns a list of floats.
    """
    payload = {
        "model": EMBED_MODEL,
        "input": text,
    }
    try:
        r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Ollama /api/embed returns {"embedding": [...]} or {"embeddings": [[...]]}
        if "embedding" in data:
            return data["embedding"]
        elif "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]
        else:
            raise ValueError("Unexpected embed response format")
    except Exception as e:
        print("[WARN] embed_text failed:", e)
        return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two equal-length lists of floats."""
    if not a or not b or len(a) != len(b):
        return 0.0
    # Simple Python loop to avoid importing numpy
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def search_semantic_memories(query: str, top_k: int = 3, min_sim: float = 0.2):
    """
    Return top-k most semantically similar memories to the query.
    Each item: (id, text, created_at, similarity)
    """
    q_emb = embed_text(query)
    if q_emb is None:
        return []

    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, text, created_at, embedding FROM memories WHERE embedding IS NOT NULL"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    scored = []
    for mid, text, created_at, emb_json in rows:
        if not emb_json:
            continue
        try:
            emb = json.loads(emb_json)
        except Exception:
            continue
        sim = cosine_similarity(q_emb, emb)
        if sim >= min_sim:
            scored.append((mid, text, created_at, sim))

    scored.sort(key=lambda t: t[3], reverse=True)
    return scored[:top_k]


def build_memory_context(user_message: str, max_items: int = 3):
    """
    Build an Ollama message (or []) containing relevant memories for the given user message.
    """
    matches = search_semantic_memories(user_message, top_k=max_items)
    if not matches:
        return []

    lines = ["Relevant remembered information:"]
    for mid, text, created_at, sim in matches:
        lines.append(f"- [{mid}] {text} (saved {created_at}, similarity {sim:.2f})")

    return [{
        "role": "system",
        "content": "\n".join(lines),
    }]



def search_memories(query: str, limit: int = 10):
    """Search memories using a simple LIKE match."""
    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        like = f"%{query}%"
        cur.execute(
            """
            SELECT id, text, created_at
            FROM memories
            WHERE text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (like, limit),
        )
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def list_recent_memories(limit: int = 10):
    """List the most recent memories."""
    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, text, created_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def list_all_memories(limit: int = 100):
    """List up to 'limit' most recent memories for the UI."""
    return list_recent_memories(limit)


def add_memory(text: str):
    """Insert a new memory entry and return (id, text, created_at)."""
    created_at = datetime.utcnow().isoformat() + "Z"
    emb = embed_text(text)
    emb_json = json.dumps(emb) if emb is not None else None

    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memories (text, created_at, embedding) VALUES (?, ?, ?)",
            (text, created_at, emb_json),
        )
        conn.commit()
        mem_id = cur.lastrowid
    finally:
        conn.close()
    return mem_id, text, created_at


def update_memory(mem_id: int, new_text: str) -> bool:
    """Update a memory's text and its embedding."""
    emb = embed_text(new_text)
    emb_json = json.dumps(emb) if emb is not None else None

    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE memories SET text = ?, embedding = ? WHERE id = ?",
            (new_text, emb_json, mem_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()



def delete_memory(mem_id: int) -> bool:
    """Delete a memory by id. Returns True if a row was deleted."""
    conn = sqlite3.connect(MEM_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# === MEMORY API (for UI) =====================================================

@app.route("/api/memories", methods=["GET"])
def api_list_memories():
    """Return recent memories as JSON for the UI."""
    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        limit = 100

    rows = list_all_memories(limit=limit)
    memories = [
        {"id": mid, "text": text, "created_at": created_at}
        for (mid, text, created_at) in rows
    ]
    return jsonify({"memories": memories})


@app.route("/api/memories", methods=["POST"])
def api_create_memory():
    """Create a new memory (text only)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Memory text is required"}), 400

    mem_id, mem_text, created_at = add_memory(text)
    return jsonify(
        {"memory": {"id": mem_id, "text": mem_text, "created_at": created_at}}
    )


@app.route("/api/memories/<int:mem_id>", methods=["PUT"])
def api_update_memory(mem_id):
    """Update an existing memory's text."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Memory text is required"}), 400

    ok = update_memory(mem_id, text)
    if not ok:
        return jsonify({"error": "Memory not found"}), 404

    return jsonify({"success": True})


@app.route("/api/memories/<int:mem_id>", methods=["DELETE"])
def api_delete_memory(mem_id):
    """Delete a memory."""
    ok = delete_memory(mem_id)
    if not ok:
        return jsonify({"error": "Memory not found"}), 404

    return jsonify({"success": True})

# === BASIC INDEX & NON-STREAM ENDPOINT ==========================

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Non-streaming endpoint (kept for fallback or tools).
    """
    data = request.get_json()
    user_message = data.get("message", "")
    history = data.get("history", [])

    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        if turn.get("assistant"):
            messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ollama_response = r.json()
    assistant_text = ollama_response["message"]["content"]

    return jsonify({"reply": assistant_text})


# === STREAMING WITH /web SUPPORT (Tavily) ===================================

@app.route("/api/chat_stream", methods=["POST"])
def chat_stream():
    """
    Streaming endpoint: streams text chunks from Ollama to the browser.
    Supports:
    - `/remember <text>`: store a memory (no model call)
    - `/mem <query>`: search memories
    - `/mem`: list recent memories
    - `/web <query>`: Tavily web search + LLM answer
    """
    data = request.get_json()
    user_message = data.get("message", "")
    history = data.get("history", [])

    # Build chat history into Ollama format
    history_messages = []
    for turn in history:
        history_messages.append({"role": "user", "content": turn["user"]})
        if turn.get("assistant"):
            history_messages.append({"role": "assistant", "content": turn["assistant"]})

    # Global system message – later we’ll enrich this with profile/memory/etc.
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful local assistant. "
            "If system messages provide web search results, treat them as your primary "
            "source for answering the user, and mention when you are using web search."
        ),
    }

    # === MEMORY COMMANDS (no model call) =====================================
    if user_message.startswith("/remember"):
        raw = user_message[len("/remember"):].strip()
        if not raw:
            def gen_err():
                yield "ERR:Empty /remember command. Use `/remember something to store`."
            return Response(stream_with_context(gen_err()), mimetype="text/plain")

        add_memory(raw)

        def gen_ok():
            yield "Okay, I'll remember that."
        return Response(stream_with_context(gen_ok()), mimetype="text/plain")

    if user_message.startswith("/mem"):
        raw = user_message[len("/mem"):].strip()

        if raw:
            rows = search_memories(raw, limit=15)
            header = f"Memories matching {raw!r}:\n"
        else:
            rows = list_recent_memories(limit=15)
            header = "Most recent memories:\n"

        def gen_list():
            if not rows:
                yield header + "(No memories found.)"
                return

            yield header
            for mid, text, created_at in rows:
                line = f"- [{mid}] {created_at}: {text}\n"
                yield line

        return Response(stream_with_context(gen_list()), mimetype="text/plain")

    # === /web Tavily SEARCH ==================================================
    web_context_message = None
    final_user_prompt = user_message

    if user_message.startswith("/web"):
        raw_query = user_message[len("/web"):].strip()
        if not raw_query:
            def gen_err():
                yield "ERR:Empty /web query. Use `/web your question here`."
            return Response(stream_with_context(gen_err()), mimetype="text/plain")

        q_lower = raw_query.lower()
        topic = "news" if any(
            kw in q_lower for kw in ["today", "latest", "news", "breaking", "this week"]
        ) else "general"

        summary_text, err = tavily_search(raw_query, topic=topic)
        if err is not None:
            def gen_err2():
                yield "ERR:" + err
            return Response(stream_with_context(gen_err2()), mimetype="text/plain")

        web_context_message = {
            "role": "system",
            "content": summary_text,
        }

        final_user_prompt = (
            "Using the Tavily web search results provided in the system message above, "
            "answer the original question clearly and concisely:\n\n{}".format(raw_query)
        )

    # === NORMAL LLM CHAT (with optional web context) =========================
    # === MEMORY CONTEXT (semantic) ===========================================
    memory_context_messages = []
    # Only build memory context for "real" chat /web queries, not for commands
    if not user_message.startswith(("/remember", "/mem")):
        memory_context_messages = build_memory_context(final_user_prompt)

    # === RAG CONTEXT (docs) ==================================================
    rag_context_messages = []
    if not user_message.startswith(("/remember", "/mem", "/web")):
        rag_context_messages = build_rag_context(final_user_prompt)


    # === NORMAL LLM CHAT (with optional web + memory + RAG context) =========
    messages = [system_msg]
    if web_context_message is not None:
        messages.append(web_context_message)

    messages.extend(memory_context_messages)
    messages.extend(rag_context_messages)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": final_user_prompt})


    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
    }

    def generate():
        try:
            with requests.post(
                OLLAMA_URL, json=payload, stream=True, timeout=600
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        yield "ERR:" + str(data["error"])
                        break

                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content

                    if data.get("done"):
                        break
        except Exception as e:
            yield "ERR:" + str(e)

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    init_mem_db()
    init_rag_db()
    # 0.0.0.0 so you can reach it from other devices on your LAN
    app.run(host="0.0.0.0", port=5000, debug=True)

