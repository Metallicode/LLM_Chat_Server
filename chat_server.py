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

app = Flask(__name__, static_folder="static")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:1b"  # e.g. "llama3.2"

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


# === BASIC INDEX & NON-STREAM ENDPOINT (unchanged) ==========================

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
    Supports `/web <query>` to run a Tavily web search first and give
    the results to Ollama as context.
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

    # Decide if this is a /web query
    web_context_message = None
    final_user_prompt = user_message

    if user_message.startswith("/web"):
        raw_query = user_message[len("/web"):].strip()
        if not raw_query:
            def gen_err():
                yield "ERR:Empty /web query. Use `/web your question here`."
            return Response(stream_with_context(gen_err()), mimetype="text/plain")

        # Pick topic based on query (simple heuristic)
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

    # Build final messages list for Ollama
    messages = [system_msg]
    if web_context_message is not None:
        messages.append(web_context_message)

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
                # Ollama yields one JSON object per line
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
                        yield content  # streaming raw text chunks

                    if data.get("done"):
                        break
        except Exception as e:
            yield "ERR:" + str(e)

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    # 0.0.0.0 so you can reach it from other devices on your LAN
    app.run(host="0.0.0.0", port=5000, debug=True)

