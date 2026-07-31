from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from time import sleep
from types import TracebackType
from typing import Any
from unittest.mock import Mock, patch

import pytest

from industrial_agent.http_transport import (
    BoundedHTTPTransport,
    HTTPTransportCircuitOpenError,
    HTTPTransportConnectionError,
    HTTPTransportProtocolError,
    HTTPTransportTimeoutError,
)


class _ResponseHandler(BaseHTTPRequestHandler):
    response_status = 200
    response_body = b'{"status":"ok"}'
    seen_methods: list[str] = []

    def _send_configured_response(self) -> None:
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def do_GET(self) -> None:
        self.seen_methods.append("GET")
        self._send_configured_response()

    def do_POST(self) -> None:
        self.seen_methods.append("POST")
        content_length = int(self.headers["Content-Length"])
        request_body = self.rfile.read(content_length)
        decoded = json.loads(request_body.decode("utf-8"))
        if not isinstance(decoded, dict):
            self.send_error(400)
            return
        self._send_configured_response()

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


class _SlowResponseHandler(_ResponseHandler):
    def _sleep_then_respond(self, operation: Any) -> None:
        sleep(0.15)
        try:
            operation()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        self._sleep_then_respond(super().do_GET)

    def do_POST(self) -> None:
        self._sleep_then_respond(super().do_POST)


class _ServerContext:
    def __init__(
        self,
        handler: type[BaseHTTPRequestHandler] = _ResponseHandler,
    ) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def test_request_returns_json_object() -> None:
    _ResponseHandler.response_status = 200
    _ResponseHandler.response_body = b'{"status":"ready","value":3}'
    _ResponseHandler.seen_methods = []
    with _ServerContext() as base_url:
        transport = BoundedHTTPTransport(base_url)
        assert transport.request("/health", {}, 1_000) == {
            "status": "ready",
            "value": 3,
        }
    assert _ResponseHandler.seen_methods == ["GET"]


def test_contract_defined_non_2xx_body_is_returned() -> None:
    _ResponseHandler.response_status = 422
    _ResponseHandler.response_body = (
        b'{"status":"error","error":{"code":"TASK_1001_INVALID"}}'
    )
    _ResponseHandler.seen_methods = []
    with _ServerContext() as base_url:
        transport = BoundedHTTPTransport(base_url)
        response = transport.request("/v1/infer", {"task_id": "task-1"}, 1_000)
        assert response["status"] == "error"
        assert response["error"]["code"] == "TASK_1001_INVALID"
    assert _ResponseHandler.seen_methods == ["POST"]


def test_invalid_json_response_is_rejected() -> None:
    _ResponseHandler.response_status = 200
    _ResponseHandler.response_body = b"not-json"
    with _ServerContext() as base_url:
        transport = BoundedHTTPTransport(base_url)
        with pytest.raises(HTTPTransportProtocolError, match="valid JSON"):
            transport.request("/health", {}, 1_000)


def test_oversized_response_is_rejected() -> None:
    _ResponseHandler.response_status = 200
    _ResponseHandler.response_body = b'{"payload":"' + (b"x" * 128) + b'"}'
    with _ServerContext() as base_url:
        transport = BoundedHTTPTransport(base_url, max_response_bytes=32)
        with pytest.raises(HTTPTransportProtocolError, match="exceeds"):
            transport.request("/health", {}, 1_000)


def test_timeout_is_typed_and_opens_circuit() -> None:
    transport = BoundedHTTPTransport(
        "http://127.0.0.1:8101",
        circuit_failure_threshold=2,
        circuit_reset_after_s=30.0,
    )
    connection = Mock()
    connection.request.side_effect = TimeoutError("socket timed out")
    with patch.object(transport, "_open_connection", return_value=connection):
        with pytest.raises(HTTPTransportTimeoutError):
            transport.request("/health", {}, 10)
        with pytest.raises(HTTPTransportTimeoutError):
            transport.request("/health", {}, 10)
        with pytest.raises(HTTPTransportCircuitOpenError):
            transport.request("/health", {}, 10)


def test_real_socket_response_timeout_is_bounded() -> None:
    _SlowResponseHandler.response_status = 200
    _SlowResponseHandler.response_body = b'{"status":"ready"}'
    with _ServerContext(_SlowResponseHandler) as base_url:
        transport = BoundedHTTPTransport(base_url)
        with pytest.raises(HTTPTransportTimeoutError):
            transport.request("/health", {}, 25)


def test_connection_failure_is_not_silently_swallowed() -> None:
    transport = BoundedHTTPTransport("http://127.0.0.1:8102")
    connection = Mock()
    connection.request.side_effect = ConnectionRefusedError("refused")
    with patch.object(transport, "_open_connection", return_value=connection):
        with pytest.raises(HTTPTransportConnectionError, match="refused"):
            transport.request("/health", {}, 100)


@pytest.mark.parametrize(
    ("route", "payload", "timeout_ms", "error_type"),
    [
        ("health", {}, 100, ValueError),
        ("//other-host/health", {}, 100, ValueError),
        ("/health", None, 100, TypeError),
        ("/health", {"unexpected": True}, 100, ValueError),
        ("/health", {}, 0, ValueError),
        ("/health", {}, True, ValueError),
    ],
)
def test_request_rejects_invalid_boundaries(
    route: str,
    payload: Any,
    timeout_ms: Any,
    error_type: type[BaseException],
) -> None:
    transport = BoundedHTTPTransport("http://127.0.0.1:8101")
    with pytest.raises(error_type):
        transport.request(route, payload, timeout_ms)


def test_base_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="credentials"):
        BoundedHTTPTransport("http://user:password@127.0.0.1:8101")
