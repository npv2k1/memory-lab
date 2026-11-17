"""Deterministic OpenAI-compatible HTTP fixture, used only in E2E."""
import hashlib
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def embed(text: str) -> list[float]:
    vector = [0.0] * 32
    for word in re.findall(r"\w+", text.lower()):
        vector[int.from_bytes(hashlib.sha256(word.encode()).digest()[:2]) % 32] += 1
    norm = math.sqrt(sum(value * value for value in vector)) or 1
    return [value / norm for value in vector]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(200, {"status": "ok"})

    def do_POST(self):
        if self.path != "/v1/embeddings":
            return self.respond(404, {"error": "not found"})
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.respond(200, {"data": [{"index": i, "embedding": embed(text)} for i, text in enumerate(payload["input"])]})

    def respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8001), Handler).serve_forever()
