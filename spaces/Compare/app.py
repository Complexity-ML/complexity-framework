"""
Compare — Proxy server that calls TR-MoE-400M and Dense-400M Spaces
and returns both results for side-by-side comparison.

Supports streaming (SSE) for all endpoints.

Complexity-ML — 2026
"""

import os
import json
import asyncio
from aiohttp import web, ClientSession

PORT = int(os.environ.get("PORT", 7860))

TR_MOE_URL = os.environ.get("TR_MOE_URL", "https://Pacific-i64-TR-MOE-400M.hf.space")
DENSE_URL = os.environ.get("DENSE_URL", "https://Pacific-i64-Dense-400M.hf.space")


async def stream_model(session, url, prompt, max_tokens, temperature):
    """Stream completions from a model Space, yielding SSE chunks."""
    try:
        async with session.post(
            f"{url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=None,
        ) as resp:
            if resp.status != 200:
                yield {"error": f"status {resp.status}"}
                return
            buffer = ""
            async for raw in resp.content:
                buffer += raw.decode()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        yield {"error": str(e)}


async def call_model(session, url, prompt, max_tokens, temperature):
    """Non-streaming fallback — call a model Space and return the result."""
    try:
        async with session.post(
            f"{url}/v1/completions",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
            timeout=120,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "text": data["choices"][0]["text"],
                    "model": data.get("model", url),
                    "status": "ok",
                }
            else:
                return {"text": "", "model": url, "status": f"error: {resp.status}"}
    except Exception as e:
        return {"text": "", "model": url, "status": f"error: {str(e)}"}


async def handle_compare(request):
    """SSE stream: interleave chunks from both models."""
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    stream = data.get("stream", False)

    if not stream:
        # Non-streaming fallback
        async with ClientSession() as session:
            moe_result, dense_result = await asyncio.gather(
                call_model(session, TR_MOE_URL, prompt, max_tokens, temperature),
                call_model(session, DENSE_URL, prompt, max_tokens, temperature),
            )
        return web.json_response({
            "prompt": prompt,
            "tr_moe": moe_result,
            "dense": dense_result,
        })

    # Streaming mode — SSE
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    await resp.prepare(request)

    async with ClientSession() as session:
        moe_text = ""
        dense_text = ""

        async def drain_stream(tag, url):
            nonlocal moe_text, dense_text
            async for chunk in stream_model(session, url, prompt, max_tokens, temperature):
                if "error" in chunk:
                    event = {"type": tag, "error": chunk["error"]}
                    await resp.write(f"data: {json.dumps(event)}\n\n".encode())
                    return
                text = chunk.get("choices", [{}])[0].get("text", "")
                if not text:
                    continue
                if tag == "tr_moe":
                    moe_text += text
                else:
                    dense_text += text
                event = {"type": tag, "text": text}
                await resp.write(f"data: {json.dumps(event)}\n\n".encode())

        await asyncio.gather(
            drain_stream("tr_moe", TR_MOE_URL),
            drain_stream("dense", DENSE_URL),
        )

    await resp.write(b"data: [DONE]\n\n")
    return resp


async def handle_chat(request):
    """TR-MoE streaming endpoint."""
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    stream = data.get("stream", False)

    if not stream:
        async with ClientSession() as session:
            result = await call_model(session, TR_MOE_URL, prompt, max_tokens, temperature)
        return web.json_response({
            "id": "cmpl-compare",
            "object": "text_completion",
            "model": "Pacific-i64/TR-MoE-400M",
            "choices": [{"index": 0, "text": result["text"], "finish_reason": "length"}],
        })

    # Streaming — forward SSE from TR-MoE
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    await resp.prepare(request)

    async with ClientSession() as session:
        async for chunk in stream_model(session, TR_MOE_URL, prompt, max_tokens, temperature):
            if "error" in chunk:
                await resp.write(f"data: {json.dumps({'error': chunk['error']})}\n\n".encode())
                break
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

    await resp.write(b"data: [DONE]\n\n")
    return resp


async def handle_dense(request):
    """Dense streaming endpoint."""
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    stream = data.get("stream", False)

    if not stream:
        async with ClientSession() as session:
            result = await call_model(session, DENSE_URL, prompt, max_tokens, temperature)
        return web.json_response({
            "id": "cmpl-compare-dense",
            "object": "text_completion",
            "model": "Pacific-i64/Dense-400M",
            "choices": [{"index": 0, "text": result["text"], "finish_reason": "length"}],
        })

    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    await resp.prepare(request)

    async with ClientSession() as session:
        async for chunk in stream_model(session, DENSE_URL, prompt, max_tokens, temperature):
            if "error" in chunk:
                await resp.write(f"data: {json.dumps({'error': chunk['error']})}\n\n".encode())
                break
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

    await resp.write(b"data: [DONE]\n\n")
    return resp


async def handle_health(request):
    return web.json_response({
        "status": "ok",
        "models": {
            "tr_moe": TR_MOE_URL,
            "dense": DENSE_URL,
        }
    })


async def handle_root(request):
    raise web.HTTPFound("https://www.complexity-ai.fr/demo?mode=compare")


app = web.Application()
app.router.add_get("/", handle_root)
app.router.add_post("/v1/compare", handle_compare)
app.router.add_post("/chat", handle_chat)
app.router.add_post("/dense", handle_dense)
app.router.add_get("/health", handle_health)

if __name__ == "__main__":
    print(f"Compare proxy on port {PORT}")
    print(f"  TR-MoE: {TR_MOE_URL}")
    print(f"  Dense:  {DENSE_URL}")
    web.run_app(app, host="0.0.0.0", port=PORT)
