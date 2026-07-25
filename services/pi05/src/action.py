"""动作块协议定义（方案书 §3.4 CanonicalActionChunk v1）。

执行器 → 总 Agent 的统一动作块数据结构。

负责人：A（协议定义）/ E（π0.5 适配）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CanonicalActionChunk:
    """执行器 → 总 Agent 的统一动作块（方案书 §3.4 CanonicalActionChunk v1）。

    actions: float32[N,7] [dx,dy,dz,dax,day,daz,gripper]
    space_id: action space 语义标识
    frame: 坐标系
    control_hz: 控制频率
    """

    actions: np.ndarray              # float32[N,7] [dx,dy,dz,dax,day,daz,gripper]
    space_id: str = "eef_delta_xyz_axisangle_gripper_v1"
    frame: str = "robot_base"
    control_hz: int = 10
    generated_step: int = 0
    source_policy: str = "pi05"
    checkpoint_sha: str = ""
    expires_after_ms: int = 1000
