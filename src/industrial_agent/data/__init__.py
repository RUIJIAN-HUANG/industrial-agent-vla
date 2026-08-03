"""Canonical offline-data contracts.

This package is intentionally independent from the online Supervisor and VLA
transport contracts. Dataset padding and persistence metadata must never add
fields to the frozen online ``ActionChunk`` schema.
"""

from .padding import PaddingPolicy, PaddingResult, PaddingStrategy, pad_actions

__all__ = [
    "PaddingPolicy",
    "PaddingResult",
    "PaddingStrategy",
    "pad_actions",
]
