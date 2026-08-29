"""执行器抽象基类（方案书 §7.3 仓库结构）。

所有模型专属执行器必须实现此接口。
总控 Agent 依赖此抽象，并通过统一合同调用唯一的 π0.5 执行器。

负责人：A（接口定义）/ E（π0.5 实现）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.pi05.src.action import CanonicalActionChunk
    from services.pi05.src.observation import ObsPacket


class BaseExecutor(ABC):
    """VLA 执行器抽象基类。

    方案书 §3.4：统一观测/动作协议，所有执行器输入 ObsPacket、输出 CanonicalActionChunk。
    """

    @abstractmethod
    def infer(self, obs: ObsPacket) -> CanonicalActionChunk:
        """观测 → 安全动作块。

        Args:
            obs: 统一观测包（原始 RGB + proprio + instruction）。

        Returns:
            含 N×7 动作块与 space_id/frame/control_hz 等协议的 CanonicalActionChunk。
        """
        ...

    @abstractmethod
    def cancel_pending_chunk(self) -> None:
        """失败切换时清空动作队列与客户端缓存（方案书 §3.3.1 Para186）。"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置适配器状态：清空动作队列、episode 缓存、延迟统计。"""
        ...

    @abstractmethod
    def health_check(self) -> dict:
        """返回健康状态（方案书 §7.1）：mode / checkpoint_sha / vram / latency 等。"""
        ...
