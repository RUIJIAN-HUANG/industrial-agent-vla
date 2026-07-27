"""Stdlib HTTP entrypoint for the OpenVLA-OFT Arm_B executor service."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import load_config
from .routes import OpenVLAOFTService


def create_service() -> OpenVLAOFTService:
    return OpenVLAOFTService(load_config())


class OpenVLAOFTRequestHandler(BaseHTTPRequestHandler):
    server: "OpenVLAOFTHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        status, body = self.server.service.health()
        self._write_json(status, body)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if payload is None:
            return
        if self.path == "/v1/infer":
            status, body = self.server.service.infer(payload)
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
        except ValueError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_content_length"},
            )
            return None
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return None

    def _write_json(self, status: int | HTTPStatus, body: Any) -> None:
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class OpenVLAOFTHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        service: OpenVLAOFTService,
    ) -> None:
        super().__init__(server_address, OpenVLAOFTRequestHandler)
        self.service = service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenVLA-OFT Arm_B service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    args = parser.parse_args()
    server = OpenVLAOFTHTTPServer((args.host, args.port), create_service())
    print(
        f"OpenVLA-OFT service listening on http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
