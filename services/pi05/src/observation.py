"""观测包协议定义（方案书 §3.4 ObsPacket v1）。

总 Agent → 执行器的统一观测数据结构。

负责人：A（协议定义）/ E（π0.5 适配）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("pi05.observation")

# ── ImageReference 约束（对齐 schemas/executor-infer.schema.json #$defs/imageReference）──
IMAGE_REF_REQUIRED_FIELDS = frozenset(
    {"uri", "image_sha256", "camera_id", "width", "height"}
)
MAX_IMAGE_DIMENSION = 4096  # 单边最大像素，防止恶意请求分配超大数组
_IMAGE_REF_URI_PATTERN = re.compile(r"^cas://sha256/([0-9a-fA-F]{64})$")
_IMAGE_SHA_PATTERN = re.compile(r"^sha256:([0-9a-fA-F]{64})$")


def is_image_reference(value: Any) -> bool:
    """检测 value 是否为框架 ImageReference 字典。

    框架侧 Pi05Adapter / mock / B（模拟器）统一使用
    schemas/executor-infer.schema.json #$defs/imageReference 格式：
      {uri, image_sha256, camera_id, width, height}

    本函数严格校验 5 个字段全部存在（非空、非 None），与旧版 raw pixels
    dict（仅含 pixels/data 字段）严格区分，避免误判。
    （方案书 interface-contracts.md §7.3 / agent-framework.md §11）
    """
    return isinstance(value, dict) and all(
        value.get(key) for key in IMAGE_REF_REQUIRED_FIELDS
    )


def validate_image_reference(
    value: Any,
    *,
    expected_camera_id: str | None = None,
) -> dict[str, Any]:
    """严格校验冻结 ImageReference，并核对 URI 与声明摘要。

    这里只验证纯输入契约，不访问 CAS。真实图像解析、只读鉴权及内容哈希复核
    仍由部署侧 CAS Adapter 负责。
    """
    if not isinstance(value, dict):
        raise ValueError("ImageReference 必须是对象")
    if set(value) != IMAGE_REF_REQUIRED_FIELDS:
        raise ValueError(
            f"ImageReference 必须且只能包含 {sorted(IMAGE_REF_REQUIRED_FIELDS)}"
        )

    uri = value.get("uri")
    image_sha256 = value.get("image_sha256")
    if not isinstance(uri, str):
        raise ValueError("ImageReference.uri 必须是字符串")
    if not isinstance(image_sha256, str):
        raise ValueError("ImageReference.image_sha256 必须是字符串")

    uri_match = _IMAGE_REF_URI_PATTERN.fullmatch(uri)
    sha_match = _IMAGE_SHA_PATTERN.fullmatch(image_sha256)
    if uri_match is None:
        raise ValueError("ImageReference.uri 必须匹配 cas://sha256/<64hex>")
    if sha_match is None:
        raise ValueError("ImageReference.image_sha256 必须匹配 sha256:<64hex>")
    if uri_match.group(1) != sha_match.group(1):
        raise ValueError("ImageReference.uri 摘要与 image_sha256 不一致")

    camera_id = value.get("camera_id")
    if not isinstance(camera_id, str) or not camera_id:
        raise ValueError("ImageReference.camera_id 必须是非空字符串")
    if expected_camera_id is not None and camera_id != expected_camera_id:
        raise ValueError(f"ImageReference.camera_id 必须为 {expected_camera_id!r}")

    for field_name in ("width", "height"):
        dimension = value.get(field_name)
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError(f"ImageReference.{field_name} 必须是整数")
        if dimension < 1 or dimension > MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"ImageReference.{field_name} 必须在 1..{MAX_IMAGE_DIMENSION} 范围内"
            )

    return value


def image_reference_to_placeholder(image_ref: dict[str, Any]) -> np.ndarray:
    """从 ImageReference 尺寸创建零图占位 uint8[H,W,3]。

    适用于 dummy 模式：框架传入 ImageReference 不含原始像素，
    Pi05Executor._infer_mock 不依赖像素内容，按尺寸创建零图即可。
    真实部署需在调用前解析 CAS URI 获取真实像素。

    单边尺寸超过 MAX_IMAGE_DIMENSION (4096) 时记录 warning 并钳制，
    防止恶意超大请求导致内存溢出。
    """
    width = image_ref.get("width", 640)
    height = image_ref.get("height", 480)
    # 类型/值域校验（对齐 _canonical_image_reference 的正整数约束）
    if isinstance(width, bool) or not isinstance(width, int):
        width = 640
    if isinstance(height, bool) or not isinstance(height, int):
        height = 640
    if width < 1:
        width = 640
    if height < 1:
        height = 480
    if width > MAX_IMAGE_DIMENSION:
        logger.warning(
            "ImageReference width=%d 超过上限 %d，钳制", width, MAX_IMAGE_DIMENSION
        )
        width = MAX_IMAGE_DIMENSION
    if height > MAX_IMAGE_DIMENSION:
        logger.warning(
            "ImageReference height=%d 超过上限 %d，钳制", height, MAX_IMAGE_DIMENSION
        )
        height = MAX_IMAGE_DIMENSION
    return np.zeros((height, width, 3), dtype=np.uint8)


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
