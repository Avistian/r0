"""Loopback-only browser transport for the shared explorer actions."""

import json
import secrets
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from r0.modules.explorer.lib.session import ExplorerSession


def explorer_page(token: str) -> str:
    assets = Path(__file__).parent / "web"
    return (
        (assets / "index.html")
        .read_text()
        .replace("/* STYLES */", (assets / "style.css").read_text())
        .replace("/* APPLICATION */", (assets / "app.js").read_text())
        .replace("/* TRANSPORT */", "window.r0token = " + json.dumps(token) + ";")
    )


def serve_explorer(output_dir: Path, *, port: int = 8765) -> None:
    session = ExplorerSession(output_dir)
    token = secrets.token_urlsafe(32)
    page = explorer_page(token).encode()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _local_host(self) -> bool:
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self) -> None:
            if not self._local_host():
                self._reply(403, b"Local host required", "text/plain")
            elif self.path == "/":
                self._reply(200, page, "text/html; charset=utf-8")
            else:
                self._reply(404, b"Not found", "text/plain")

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            if (
                not self._local_host()
                or self.headers.get("X-r0-explorer") != token
                or (origin and urlsplit(origin).netloc != self.headers.get("Host"))
            ):
                self._reply(403, b"Local session required", "text/plain")
                return
            if self.path != "/api":
                self._reply(404, b"Not found", "text/plain")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024**2:
                    raise ValueError("request must be at most 1 MiB")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                payload = cast(dict[str, Any], payload)
                if not isinstance(payload.get("data", {}), dict):
                    raise ValueError("action data must be an object")
                result = session.request(payload["action"], payload.get("data", {}))
                data = json.dumps(result, allow_nan=False).encode()
                self._reply(200, data, "application/json")
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                self._reply(400, json.dumps({"error": str(error)}).encode(), "application/json")

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Dataset explorer: http://127.0.0.1:{port}/", flush=True)
        print(f"Saved generation runs: {session.output_dir}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()
