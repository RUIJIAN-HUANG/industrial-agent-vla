"""观测包协议定义（方案书 §3.4 ObsPacket v1）。

总 Agent → 执行器的统一观测数据结构。

负责人：A（协议定义）/ E（π0.5 适配）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObsPacket:
    """总 Agent → 执行器的观测包（方案书 §3.4 ObsPacket v1）。

    只包含 ONLINE 白名单字段（RGB + proprio + instruction），
    不含 GT（pose / bbox / mask / success），由 gt_sidecar 隔离（方案书 §2.2）。
    """

    episode_id: str
    step_id: int
    timestamp_ns: int
    rgb_front: np.ndarray  # uint8[H,W,3] 原始 RGB，不做预处理
    rgb_wrist: np.ndarray | None  # uint8[H,W,3]，可选腕部相机
    robot_state: np.ndarray  # float32[d] 本体状态
    instruction: str  # 完整自然语言，不拆槽位
    runtime_flags: dict = field(
        default_factory=dict
    )  # {terminated, truncated, camera_ok}
