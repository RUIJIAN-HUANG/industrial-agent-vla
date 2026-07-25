"""π0.5 服务核心代码包（方案书 §7.3）。

负责人：E（π0.5/openpi）

包含：
- action: CanonicalActionChunk 协议定义
- observation: ObsPacket 协议定义
- base: BaseExecutor 抽象基类
- pi05: Pi05Executor 业务逻辑
- pi05_client: 策略客户端（本地 JAX / WebSocket）
- pi05_contract_adapter: 体系B 契约适配器
- openpi_service: FastAPI WebSocket+HTTP 服务
"""

from services.pi05.src.action import CanonicalActionChunk
from services.pi05.src.observation import ObsPacket
from services.pi05.src.base import BaseExecutor
from services.pi05.src.pi05 import Pi05Executor
from services.pi05.src.pi05_client import PolicyClient, make_policy_client
from services.pi05.src.pi05_contract_adapter import Pi05ContractAdapter

__all__ = [
    "CanonicalActionChunk",
    "ObsPacket",
    "BaseExecutor",
    "Pi05Executor",
    "PolicyClient",
    "make_policy_client",
    "Pi05ContractAdapter",
]
