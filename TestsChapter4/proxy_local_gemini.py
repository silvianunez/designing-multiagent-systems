"""
CORS Proxy for Claude.ai widget → Google Gemini API

Usage (Windows):
    set GEMINI_API_KEY=AIzaSy...
    python proxy.py
"""

import json
import os
import sys
from aiohttp import web, ClientSession

API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
PORT = 8080
DEFAULT_MODEL = "gemini-2.0-flash"

print(f"Python: {sys.version}")
print(f"API key set: {'YES' if API_KEY else 'NO — run: set GEMINI_API_KEY=AIzaSy...'}")
if API_KEY:
    print(f"API key ends with: ...{API_KEY[-4:]}")
print(f"Starting at http://localhost:{PORT}/proxy")
print("(keep this window open)\n")


def anthropic_to_gemini(body: dict) -> tuple[str, dict]:
    """Convert Anthropic-style request to Gemini format."""
    model = body.get("model", DEFAULT_MODEL)
    system = body.get("system", "")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 1024)

    contents = []

    # System prompt as first user turn if present
    if system:
        contents.append({
            "role": "user",
            "parts": [{"text": f"[System instructions]: {system}"}]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood. I will follow those instructions."}]
        })

    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if isinstance(content, str):
            contents.append({"role": role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append({"text": block["text"]})
            contents.append({"role": role, "parts": parts})

    gemini_body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        }
    }

    return model, gemini_body


def gemini_to_anthropic(data: dict) -> dict:
    """Convert Gemini response to Anthropic format."""
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = ""
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "model": DEFAULT_MODEL,
    }


async def proxy(request: web.Request) -> web.StreamResponse:
    print(f"[request] {request.method} /proxy")
    try:
        body = await request.json()
        print(f"[request] model={body.get('model', DEFAULT_MODEL)} stream={body.get('stream', False)}")
    except Exception as e:
        print(f"[error] body parse: {e}")
        return web.Response(status=400, text=str(e),
                            headers={"Access-Control-Allow-Origin": "*"})

    if not API_KEY:
        print("[error] API key not set")
        return web.Response(status=500,
                            text='{"error":"GEMINI_API_KEY not set"}',
                            content_type="application/json",
                            headers={"Access-Control-Allow-Origin": "*"})

    is_stream = body.get("stream", False)
    model, gemini_body = anthropic_to_gemini(body)

    # Gemini model name mapping
    if "claude" in model or "sonnet" in model or "haiku" in model or "opus" in model:
        model = DEFAULT_MODEL  # fallback to default Gemini model

    try:
        async with ClientSession() as session:
            if is_stream:
                url = (GEMINI_STREAM_URL.format(model=model)
                       + f"?key={API_KEY}&alt=sse")
                async with session.post(url, json=gemini_body) as resp:
                    print(f"[gemini] stream status={resp.status}")
                    response = web.StreamResponse(
                        status=200,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Access-Control-Allow-Origin": "*",
                        }
                    )
                    await response.prepare(request)

                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:"):
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                                text = (chunk.get("candidates", [{}])[0]
                                        .get("content", {})
                                        .get("parts", [{}])[0]
                                        .get("text", ""))
                                if text:
                                    # Emit in Anthropic SSE format so the widget works unchanged
                                    event = json.dumps({
                                        "type": "content_block_delta",
                                        "delta": {"type": "text_delta", "text": text}
                                    })
                                    await response.write(f"data: {event}\n\n".encode())
                            except Exception:
                                pass

                    await response.write_eof()
                    print("[stream] completed")
                    return response
            else:
                url = GEMINI_URL.format(model=model) + f"?key={API_KEY}"
                async with session.post(url, json=gemini_body) as resp:
                    print(f"[gemini] status={resp.status}")
                    data = await resp.json()
                    if resp.status != 200:
                        print(f"[error] Gemini: {data}")
                        return web.Response(
                            status=resp.status,
                            text=json.dumps(data),
                            content_type="application/json",
                            headers={"Access-Control-Allow-Origin": "*"}
                        )
                    converted = gemini_to_anthropic(data)
                    print(f"[response] ok")
                    return web.Response(
                        text=json.dumps(converted),
                        content_type="application/json",
                        headers={"Access-Control-Allow-Origin": "*"}
                    )

    except Exception as e:
        print(f"[error] Gemini call failed: {e}")
        return web.Response(status=500,
                            text=json.dumps({"error": str(e)}),
                            content_type="application/json",
                            headers={"Access-Control-Allow-Origin": "*"})


async def preflight(request: web.Request) -> web.Response:
    print("[preflight] OPTIONS /proxy")
    return web.Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })


async def health(request: web.Request) -> web.Response:
    return web.Response(text="ok", headers={"Access-Control-Allow-Origin": "*"})


app = web.Application()
app.router.add_post("/proxy", proxy)
app.router.add_route("OPTIONS", "/proxy", preflight)
app.router.add_get("/health", health)

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=PORT)
