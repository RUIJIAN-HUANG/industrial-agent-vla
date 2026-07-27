"""Stable service exceptions for the OpenVLA-OFT executor."""

from __future__ import annotations


class ServiceError(Exception):
    """Contract-defined executor error.

    The value in ``code`` intentionally matches the root project's
    ``FailureCode`` strings without importing the supervisor package into this
    standalone service.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
