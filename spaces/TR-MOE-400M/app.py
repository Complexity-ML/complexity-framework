"""
TR-MoE-400M — HuggingFace Spaces.
Frontend: complexity-ai.fr/demo (iframe) on port 7860
Backend: vllm on port 8000, proxied with true SSE streaming
"""

import os
import sys
import json
import threading
import subprocess
import http.client

os.environ["VLLM_TARGET_DEVICE"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
API_PORT = 8000
WEB_PORT = int(os.environ.get("PORT", 7860))

# Ensure config.json has required fields for vllm
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
needs_update = False
if "model_type" not in config:
    config["model_type"] = "deep"
    needs_update = True
if "architectures" not in config:
    config["architectures"] = ["DeepForCausalLM"]
    needs_update = True
if config.get("shared_intermediate_size") is None:
    config["shared_intermediate_size"] = 0
    needs_update = True
if needs_update:
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

# Start vllm API server in background on port 8000
def start_vllm():
    subprocess.run([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--trust-remote-code",
        "--port", str(API_PORT),
        "--dtype", "bfloat16",
        "--host", "0.0.0.0",
    ])

threading.Thread(target=start_vllm, daemon=True).start()

# --- Frontend + streaming reverse proxy on port 7860 ---
from http.server import HTTPServer, BaseHTTPRequestHandler

IFRAME_HTML = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TR-MoE-400M</title>
<style>body{margin:0;overflow:hidden}iframe{width:100%;height:100vh;border:none}</style>
</head><body>
<iframe src="https://www.complexity-ai.fr/demo" allow="microphone;camera"></iframe>
</body></html>"""


class Handler(BaseHTTPRequestHandler):

    def _proxy(self, method):
        """Stream proxy to vllm — forwards headers and body chunk by chunk."""
        conn = http.client.HTTPConnection("127.0.0.1", API_PORT)
        try:
            # Read request body if present
            body = None
            cl = self.headers.get("Content-Length")
            if cl:
                body = self.rfile.read(int(cl))

            # Forward to vllm
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()

            # Send status + headers
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding",):
                    self.send_header(k, v)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Stream body chunk by chunk (SSE works!)
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self.send_error(502, str(e))
            except BrokenPipeError:
                pass
        finally:
            conn.close()

    def do_GET(self):
        if self.path.startswith("/v1/"):
            self._proxy("GET")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(IFRAME_HTML)

    def do_POST(self):
        if self.path.startswith("/v1/"):
            self._proxy("POST")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def log_message(self, format, *args):
        pass

print(f"Starting frontend on :{WEB_PORT}, vllm API on :{API_PORT}")
HTTPServer(("0.0.0.0", WEB_PORT), Handler).serve_forever()
