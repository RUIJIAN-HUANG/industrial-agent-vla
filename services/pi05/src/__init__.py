"""π0.5 服务核心代码包。

The package initializer is intentionally side-effect free.  In particular, importing
``services.pi05.src.pi05`` must not import the training configuration or initialize
OpenPI.  Runtime services and training tools have different dependency boundaries.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseExecutor",
    "CanonicalActionChunk",
    "ObsPacket",
    "Pi05ContractAdapter",
    "Pi05Executor",
    "PolicyClient",
    "make_policy_client",
]


def __getattr__(name: str) -> Any:
    """Lazily expose compatibility symbols without coupling service startup to training."""

    if name == "CanonicalActionChunk":
        from services.pi05.src.action import CanonicalActionChunk

        return CanonicalActionChunk
    if name == "BaseExecutor":
        from services.pi05.src.base import BaseExecutor

        return BaseExecutor
    if name == "ObsPacket":
        from services.pi05.src.observation import ObsPacket

        return ObsPacket
    if name == "Pi05Executor":
        from services.pi05.src.pi05 import Pi05Executor

        return Pi05Executor
    if name in {"PolicyClient", "make_policy_client"}:
        from services.pi05.src.pi05_client import PolicyClient, make_policy_client

        return {"PolicyClient": PolicyClient, "make_policy_client": make_policy_client}[
            name
        ]
    if name == "Pi05ContractAdapter":
        from services.pi05.src.pi05_contract_adapter import Pi05ContractAdapter

        return Pi05ContractAdapter
    raise AttributeError(name)
