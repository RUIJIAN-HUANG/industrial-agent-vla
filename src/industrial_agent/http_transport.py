"""Bounded JSON/HTTP transport for independently hosted Agent services.

The Supervisor keeps model and perception dependencies out of process.  This
module provides the concrete request/reply boundary used by the production
composition root without changing the existing ``ProcessTransport`` or
``PerceptionTransport`` protocols.
"""

from __future__ import annotations

from collections.abc import Mapping
from http.client import (
    HTTPConnection,
    HTTPException,
    HTTPResponse,
    HTTPSConnection,
)
import json
import logging
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlsplit


LOGGER = logging.getLogger(__name__)


class HTTPTransportError(RuntimeError):
    """Base class for bounded transport failures."""


class HTTPTransportTimeoutError(HTTPTransportError, TimeoutError):
    """The remote request or response read exceeded its configured deadline."""


class HTTPTransportConnectionError(HTTPTransportError):
    """The remote service could not be reached or closed the connection."""


class HTTPTransportProtocolError(HTTPTransportError):
    """The remote endpoint returned an invalid JSON response envelope."""


class HTTPTransportCircuitOpenError(HTTPTransportError):
    """Requests are temporarily rejected after repeated transport failures."""


class BoundedHTTPTransport:
    """Thread-safe JSON transport with timeouts, size limits, and a circuit breaker.

    A transport call is attempted once.  Automatic retries are deliberately not
    performed because inference and cancellation requests may have side effects;
    the Supervisor owns all bounded recovery decisions.
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_response_bytes: int = 8 * 1024 * 1024,
        circuit_failure_threshold: int = 3,
        circuit_reset_after_s: float = 5.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if (
            isinstance(circuit_failure_threshold, bool)
            or not isinstance(circuit_failure_threshold, int)
            or circuit_failure_threshold < 1
        ):
            raise ValueError("circuit_failure_threshold must be a positive integer")
        if (
            isinstance(circuit_reset_after_s, bool)
            or not isinstance(circuit_reset_after_s, (int, float))
            or float(circuit_reset_after_s) <= 0.0
        ):
            raise ValueError("circuit_reset_after_s must be positive")

        self.base_url = base_url.rstrip("/")
        self._scheme = parsed.scheme
        self._hostname = parsed.hostname
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self.max_response_bytes = max_response_bytes
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_reset_after_s = float(circuit_reset_after_s)
        self._state_lock = Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _before_request(self) -> None:
        now = monotonic()
        with self._state_lock:
            if self._circuit_open_until > now:
                remaining = self._circuit_open_until - now
                raise HTTPTransportCircuitOpenError(
                    f"transport circuit is open for another {remaining:.3f}s"
                )
            if self._circuit_open_until:
                self._circuit_open_until = 0.0
                self._consecutive_failures = 0

    def _record_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.circuit_failure_threshold:
                self._circuit_open_until = monotonic() + self.circuit_reset_after_s

    @staticmethod
    def _remaining_seconds(deadline: float, *, route: str) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise HTTPTransportTimeoutError(f"{route} exceeded its total deadline")
        return remaining

    def _open_connection(self, timeout_s: float) -> HTTPConnection:
        if self._scheme == "https":
            return HTTPSConnection(self._hostname, self._port, timeout=timeout_s)
        return HTTPConnection(self._hostname, self._port, timeout=timeout_s)

    @staticmethod
    def _set_socket_timeout(connection: HTTPConnection, timeout_s: float) -> None:
        if connection.sock is not None:
            connection.sock.settimeout(timeout_s)

    def _read_response_bytes(
        self,
        response: HTTPResponse,
        *,
        connection: HTTPConnection,
        route: str,
        deadline: float,
    ) -> bytes:
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPTransportProtocolError(
                    f"{route} returned an invalid Content-Length"
                ) from exc
            if declared_length < 0:
                raise HTTPTransportProtocolError(
                    f"{route} returned a negative Content-Length"
                )
            if declared_length > self.max_response_bytes:
                raise HTTPTransportProtocolError(
                    f"{route} response exceeds {self.max_response_bytes} bytes"
                )

        chunks: list[bytes] = []
        received = 0
        try:
            while True:
                remaining = self._remaining_seconds(deadline, route=route)
                self._set_socket_timeout(connection, remaining)
                chunk = response.read(
                    min(64 * 1024, self.max_response_bytes + 1 - received)
                )
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise HTTPTransportProtocolError(
                        f"{route} response body must be bytes, got "
                        f"{type(chunk).__name__}"
                    )
                received += len(chunk)
                if received > self.max_response_bytes:
                    raise HTTPTransportProtocolError(
                        f"{route} response exceeds {self.max_response_bytes} bytes"
                    )
                chunks.append(chunk)
        except TimeoutError as exc:
            raise HTTPTransportTimeoutError(
                f"{route} response read exceeded its total deadline"
            ) from exc
        except (OSError, HTTPException) as exc:
            raise HTTPTransportConnectionError(
                f"{route} response body could not be read: {exc}"
            ) from exc
        raw = b"".join(chunks)
        if not raw:
            raise HTTPTransportProtocolError(f"{route} returned an empty response body")
        return raw

    @staticmethod
    def _decode_response(raw: bytes, *, route: str) -> dict[str, Any]:
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HTTPTransportProtocolError(
                f"{route} response is not valid UTF-8"
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPTransportProtocolError(
                f"{route} response is not valid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise HTTPTransportProtocolError(
                f"{route} response must be a JSON object, got {type(decoded).__name__}"
            )
        return dict(decoded)

    def request(
        self,
        route: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        """POST one bounded JSON request and return a decoded response object."""

        if (
            not isinstance(route, str)
            or not route.startswith("/")
            or route.startswith("//")
        ):
            raise ValueError(
                "route must be an absolute service path beginning with '/'"
            )
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms < 1
        ):
            raise ValueError("timeout_ms must be a positive integer")

        self._before_request()
        method = "GET" if route == "/health" else "POST"
        if method == "GET":
            if payload:
                raise ValueError("/health does not accept a request payload")
            encoded: bytes | None = None
            headers = {"Accept": "application/json"}
        else:
            try:
                encoded = json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError) as exc:
                raise HTTPTransportProtocolError(
                    f"{route} payload is not JSON serializable"
                ) from exc
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(encoded)),
            }

        target = f"{self._base_path}{route}" or "/"
        timeout_s = timeout_ms / 1000.0
        deadline = monotonic() + timeout_s
        connection: HTTPConnection | None = None
        response: HTTPResponse | None = None
        try:
            connection = self._open_connection(
                self._remaining_seconds(deadline, route=route)
            )
            connection.request(
                method,
                target,
                body=encoded,
                headers=headers,
            )
            self._set_socket_timeout(
                connection,
                self._remaining_seconds(deadline, route=route),
            )
            response = connection.getresponse()
            raw = self._read_response_bytes(
                response,
                connection=connection,
                route=route,
                deadline=deadline,
            )
            decoded = self._decode_response(raw, route=route)
        except TimeoutError as exc:
            self._record_failure()
            LOGGER.error(
                "HTTP request timed out route=%s timeout_ms=%d base_url=%s",
                route,
                timeout_ms,
                self.base_url,
            )
            raise HTTPTransportTimeoutError(
                f"{route} request exceeded {timeout_ms}ms"
            ) from exc
        except (
            HTTPTransportConnectionError,
            HTTPTransportProtocolError,
            HTTPTransportTimeoutError,
        ):
            self._record_failure()
            LOGGER.error(
                "invalid HTTP response route=%s base_url=%s",
                route,
                self.base_url,
            )
            raise
        except (OSError, HTTPException) as exc:
            self._record_failure()
            LOGGER.error(
                "HTTP request failed route=%s base_url=%s error=%s",
                route,
                self.base_url,
                exc,
            )
            raise HTTPTransportConnectionError(
                f"{route} connection failed: {exc}"
            ) from exc
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
        self._record_success()
        return decoded
