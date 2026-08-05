"""Errors returned by the YOLO service."""

from __future__ import annotations


class ServiceError(Exception):
    """A controlled error that can be returned through the HTTP API."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
