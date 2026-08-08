"""Frozen PI05 integration identities shared by data and training entry points."""

from __future__ import annotations

from typing import Any

import numpy as np


OPENPI_COMMIT: str = "15a9616a00943ada6c20a0f158e3adb39df2ccac"

# π0.5 冻结维度（方案书 §3.4）：模型输出 32D 投影层（pi05_base 兼容），
# 服务/训练契约固定前 7 维 canonical [dx,dy,dz,dax,day,daz,gripper]。
MODEL_ACTION_DIM: int = 32
CANONICAL_ACTION_DIM: int = 7


def project_policy_actions(actions: Any) -> np.ndarray:
    """Project one pinned π0.5 32-D output onto the canonical seven axes.

    Args:
        actions: 模型原始输出，末维必须为 MODEL_ACTION_DIM=32（float32 [N,32]）。

    Returns:
        float32 [N,7] — 取前 CANONICAL_ACTION_DIM 维
        [dx,dy,dz,dax,day,daz,gripper]，与 pi05_base 投影层语义一致。

    Raises:
        ValueError: 末维不是 32 时拒绝（防止用错模型头）。
    """
    array = np.asarray(actions)
    if array.ndim < 1 or array.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(
            "π0.5 base-compatible output must end in "
            f"{MODEL_ACTION_DIM} action dimensions, got {array.shape}"
        )
    return array[..., :CANONICAL_ACTION_DIM]
