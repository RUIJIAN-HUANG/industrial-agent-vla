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
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            maximum = int(self.server.service.config["api"]["max_request_bytes"])
            if not 1 <= length <= maximum:
                self._write_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "request_too_large"},
                )
                return None
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
    parser.add_argument(
        "--print-identity",
        action="store_true",
        help="Print verified deployment digests and exit before loading the model.",
    )
    args = parser.parse_args()
    if args.print_identity:
        print(
            json.dumps(
                {
                    "mock_mode": config["mock_mode"],
                    "checkpoint_sha": config["checkpoint_sha"],
                    "class_map_sha": config["class_map_sha"],
                    "config_sha": config["config_sha"],
                    "device": config["model"]["device"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    service = YoloService(config)
    server = YoloHTTPServer((args.host, args.port), service)
    print(f"YOLO service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
