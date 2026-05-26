"""
Proxy CORS para el widget de Claude.ai → Anthropic API

Uso (Windows):
    set ANTHROPIC_API_KEY=sk-ant-...
    python proxy.py
"""

import json
import os
import sys
from aiohttp import web, ClientSession

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
PORT = 8080

print(f"Python: {sys.version}")
print(f"API key configurada: {'SI' if API_KEY else 'NO — falta set ANTHROPIC_API_KEY=sk-ant-...'}")
if API_KEY:
    print(f"API key termina en: ...{API_KEY[-4:]}")
print(f"Arrancando en http://localhost:{PORT}/proxy")
print("(deja esta ventana abierta)\n")

async def proxy(request: web.Request) -> web.StreamResponse:
    print(f"[request] {request.method} /proxy")
    try:
        body = await request.json()
        print(f"[request] modelo={body.get('model','?')} stream={body.get('stream',False)}")
    except Exception as e:
        print(f"[error] body parse: {e}")
        return web.Response(status=400, text=str(e),
                            headers={"Access-Control-Allow-Origin": "*"})

    if not API_KEY:
        print("[error] API key no configurada")
        return web.Response(status=500, text='{"error":"ANTHROPIC_API_KEY no configurada"}',
                            content_type="application/json",
                            headers={"Access-Control-Allow-Origin": "*"})

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    is_stream = body.get("stream", False)

    try:
        async with ClientSession() as session:
            async with session.post(ANTHROPIC_URL, json=body, headers=headers) as resp:
                print(f"[anthropic] status={resp.status}")

                if is_stream:
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Access-Control-Allow-Origin": "*",
                        }
                    )
                    await response.prepare(request)
                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                    print("[stream] completado")
                    return response
                else:
                    data = await resp.json()
                    print(f"[response] ok, stop_reason={data.get('stop_reason','?')}")
                    return web.Response(
                        text=json.dumps(data),
                        content_type="application/json",
                        headers={"Access-Control-Allow-Origin": "*"}
                    )
    except Exception as e:
        print(f"[error] llamada a Anthropic: {e}")
        return web.Response(status=500, text=json.dumps({"error": str(e)}),
                            content_type="application/json",
                            headers={"Access-Control-Allow-Origin": "*"})

async def preflight(request: web.Request) -> web.Response:
    print(f"[preflight] OPTIONS /proxy")
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
