"""Canonical offline-data contracts.

This package is intentionally independent from the online Supervisor and VLA
transport contracts. Dataset padding and persistence metadata must never add
fields to the frozen online ``ActionChunk`` schema.
"""

from .padding import PaddingPolicy, PaddingResult, PaddingStrategy, pad_actions
from .recorder import CanonicalRecorder, EpisodeMetadata
from .recorder_v2 import CanonicalV2EpisodeMetadata, CanonicalV2Recorder
from .replay import CanonicalEpisodeReader, OfflineEpisodeReplay, OfflineReplayAction
from .split_registry import (
    DataLeakageError,
    DatasetSplit,
    SplitAssignment,
    SplitAssignmentError,
    SplitRegistry,
    SplitRegistryError,
    SplitRegistryIntegrityError,
)

__all__ = [
    "CanonicalEpisodeReader",
    "CanonicalRecorder",
    "CanonicalV2EpisodeMetadata",
    "CanonicalV2Recorder",
    "DataLeakageError",
    "DatasetSplit",
    "EpisodeMetadata",
    "OfflineEpisodeReplay",
    "OfflineReplayAction",
    "PaddingPolicy",
    "PaddingResult",
    "PaddingStrategy",
    "SplitAssignment",
    "SplitAssignmentError",
    "SplitRegistry",
    "SplitRegistryError",
    "SplitRegistryIntegrityError",
    "pad_actions",
]
