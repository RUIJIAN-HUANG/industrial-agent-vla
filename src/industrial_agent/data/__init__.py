"""Canonical offline-data contracts.

This package is intentionally independent from the online Supervisor and VLA
transport contracts. Dataset padding and persistence metadata must never add
fields to the frozen online ``ActionChunk`` schema.
"""

from .padding import PaddingPolicy, PaddingResult, PaddingStrategy, pad_actions
from .recorder import CanonicalRecorder, EpisodeMetadata
from .replay import CanonicalEpisodeReader, OfflineEpisodeReplay, OfflineReplayAction

__all__ = [
    "CanonicalEpisodeReader",
    "CanonicalRecorder",
    "EpisodeMetadata",
    "OfflineEpisodeReplay",
    "OfflineReplayAction",
    "PaddingPolicy",
    "PaddingResult",
    "PaddingStrategy",
    "pad_actions",
]
