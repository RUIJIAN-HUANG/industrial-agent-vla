"""Stdlib HTTP entrypoint for the YOLO perception service."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import load_config
from .routes import YoloService


def create_service() -> YoloService:
    return YoloService(load_config())


class YoloRequestHandler(BaseHTTPRequestHandler):
    server: "YoloHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            status, body = self.server.service.health()
        else:
            status, body = HTTPStatus.NOT_FOUND, {"error": "not_found"}
        self._write_json(status, body)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if payload is None:
            return
        if self.path == "/v1/detect":
            status, body = self.server.service.detect(payload)
        elif self.path == "/v1/cancel":
            status, body = self.server.service.cancel(payload)
        else:
            status, body = HTTPStatus.NOT_FOUND, {"error": "not_found"}
        self._write_json(status, body)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _read_json(self) -> Any | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return None

    def _write_json(self, status: int | HTTPStatus, body: Any) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class YoloHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: YoloService,
    ) -> None:
        super().__init__(server_address, YoloRequestHandler)
        self.service = service


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run the YOLO perception service")
    parser.add_argument("--host", default=config["host"])
    parser.add_argument("--port", type=int, default=config["port"])
    args = parser.parse_args()
    server = YoloHTTPServer((args.host, args.port), YoloService(config))
    print(f"YOLO service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
