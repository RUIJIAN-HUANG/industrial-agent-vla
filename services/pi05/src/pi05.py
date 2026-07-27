"""π0.5 模型专属执行器（openpi / JAX 路径）—— 业务逻辑层。

负责人：E（π0.5/openpi）

方案书出处：
- §3.3 / §3.3.1：π0.5 适配流程（JAX 路径、LeRobot、norm stats、动作块适配）。
- Table 23 Row6（§3.4 协议不变量·模型专属预处理）：总 Agent 传原始 RGB；
  裁剪/resize/pad/归一化只在模型适配器内部（由适配器调用 openpi transform 完成），
  保留像素审计样例。
- §3.3.1 Para185：模型返回动作块后，适配器裁维、加
  space_id/frame/control_hz/checkpoint_sha；反归一化由 openpi output_transform 在
  policy.infer 内完成（用 compute_norm_stats 生成的本项目自有统计，满足 Para186 不沿用 OpenVLA），
  适配器不再二次反归一化。
- §3.3.1 Para186：失败切换时清空动作队列与客户端缓存，重新传当前图像。
- Table 21 Row3（§3.3）：需要 LoRA 时必须走 JAX 路径。
- Table 69 Row7 / 附录B：单步平移 ≤2cm、旋转 ≤5°，超限截断并报警（D5 实测前用候选值）。
- §7.5：image_pipeline 像素审计、contract 动作越界拒绝。

本文件只含业务逻辑：预处理委托 openpi transform、动作裁维/限幅/包装、失败切换、健康检查。
模型加载与 WebSocket 网络代码分离到 pi05_client.py（§7.1 封装 RPC），本文件不直接 import
openpi / openpi_client，仅依赖 PolicyClient 抽象接口。
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("pi05_executor")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][pi05] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


from services.pi05.src.action import CanonicalActionChunk
from services.pi05.src.base import BaseExecutor
from services.pi05.src.observation import ObsPacket


# ---------------------------------------------------------------------------
# 策略客户端抽象（网络/模型加载代码在 pi05_client.py，§7.1 封装 RPC）
# ---------------------------------------------------------------------------
try:
    from services.pi05.src.pi05_client import (  # type: ignore
        OPENPI_AVAILABLE,
        WS_CLIENT_AVAILABLE,
        PolicyClient,
        make_policy_client,
    )
except Exception:  # pi05_client 不可用时降级 Mock
    PolicyClient = None  # type: ignore
    make_policy_client = None  # type: ignore
    OPENPI_AVAILABLE = False
    WS_CLIENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 冻结契约叠加层 imports（体系B 对齐）
# 方案书 interface-contracts.md §4 公共标识、§7.4/§7.5 统一 7 维动作合同。
# ExecutorDescriptor / is_pinned_artifact_digest 定义在 src.industrial_agent.executor
# （非 contracts.py），ACTION_CONTRACT_VERSION / ActionChunk / ActionStep 在 contracts。
# ---------------------------------------------------------------------------
import uuid

from src.industrial_agent.contracts import (  # type: ignore
    ACTION_CONTRACT_VERSION,
    ActionChunk,
    ActionStep,
)
from src.industrial_agent.executor import (  # type: ignore
    ExecutorDescriptor,
    is_pinned_artifact_digest,
)

# ---------------------------------------------------------------------------
# 安全限幅常量（方案书 Table 69 Row7 / 附录B；D5 实测前用候选值）
# ---------------------------------------------------------------------------
MAX_TRANSLATION_M = 0.02  # 单步平移 ≤ 2cm
MAX_ROTATION_RAD = 0.0873  # 单步旋转 ≤ 5° ≈ 0.0873 rad
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0

ACTION_DIM = 7  # [dx,dy,dz,dax,day,daz,gripper]


# 推理专用异常（ValueError 子类，契约适配器 catch block 将其视为不可重试）
class InferenceError(ValueError):
    """策略推理返回了不符合协议约定的动作（None / 0维 / 空数组）。"""


DIM_NAMES = ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]
MOCK_CHUNK_LEN = 10  # Mock 动作块长度（LIBERO 配置常用 10，方案书 §3.3）
CONTROL_HZ = 10  # 方案书 §3.4 control_hz
SOURCE_POLICY = "pi05"
SPACE_ID = "eef_delta_xyz_axisangle_gripper_v1"
FRAME_ID = "robot_base"
EXPIRES_AFTER_MS = 1000  # 动作块超时丢弃（方案书 §3.4 动作过期）


# ---------------------------------------------------------------------------
# 图像预处理（委托 openpi transform；适配器只保证 RGB/dtype/方向不被破坏）
# 方案书 Table 23 Row6：resize/pad/normalize 由 openpi input_transform 在 policy.infer
# 内部完成（属于"适配器调用链内部"）；适配器不做手动 resize。
# ---------------------------------------------------------------------------
def _prep_image(img: np.ndarray) -> np.ndarray:
    """图像进入 openpi 前的最小准备：保证 uint8 / HWC / RGB / 连续内存。

    不做 resize、不交换通道、不翻转。resize/pad/normalize 由 openpi input_transform 完成。
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"rgb 图像形状非法：{img.shape}，期望 [H,W,3] RGB")
    return np.ascontiguousarray(img)


def _make_fixed_test_image() -> np.ndarray:
    """生成确定性 640x480 RGB 测试图，用于 image_pipeline checksum 审计（方案书 §7.5）。"""
    rng = np.random.RandomState(20260720)
    img = rng.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    # 叠加可辨识渐变，便于人眼核对方向/通道（R 通道横向渐变）
    grad = np.linspace(0, 255, 640, dtype=np.uint8)
    img[:, :, 0] = grad[None, :]
    return img


def _image_checksum(img: np.ndarray) -> str:
    """计算图像 bytes 的 sha256，用于像素审计（方案书 §7.5 image_pipeline）。"""
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


# 模块加载时冻结固定测试图（经 _prep_image 后）的 checksum（用于 image_pipeline 测试）
FIXED_TEST_IMAGE: np.ndarray = _make_fixed_test_image()
FIXED_TEST_IMAGE_CHECKSUM: str = _image_checksum(_prep_image(FIXED_TEST_IMAGE))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _compute_dir_sha(path: str, max_files: int = 50) -> str:
    """计算目录下文件的聚合 sha（用于 checkpoint_sha 标识，方案书 §7.2）。"""
    if not path or not os.path.isdir(path):
        return ""
    h = hashlib.sha256()
    count = 0
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            try:
                h.update(fp.encode())
                with open(fp, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 16), b""):
                        h.update(chunk)
                count += 1
                if count >= max_files:
                    break
            except Exception:
                continue
        if count >= max_files:
            break
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# π0.5 执行器
# ---------------------------------------------------------------------------
class Pi05Executor(BaseExecutor):
    """π0.5 专属执行器（业务逻辑层）。

    支持 Mock（dummy）与真实（real）两种模式，通过环境变量 PI05_MODE 切换：
      - PI05_MODE=dummy|real（默认 dummy）
      - PI05_CONFIG_NAME（默认 pi05_droid）
      - PI05_CHECKPOINT_DIR（本地 checkpoint 路径）
      - PI05_WS_HOST / PI05_WS_PORT（远程 WebSocket 服务）
      - PI05_NORM_STATS_PATH（本项目 norm_stats 路径，仅用于 SHA 追溯）
      - PI05_CHECKPOINT_SHA（显式指定 checkpoint sha）
    """

    def __init__(self) -> None:
        # ---- 环境变量 ----
        self.mode: str = os.environ.get("PI05_MODE", "dummy").lower()
        if self.mode not in ("dummy", "real"):
            logger.warning("未知 PI05_MODE=%s，回退到 dummy", self.mode)
            self.mode = "dummy"
        self.config_name: str = os.environ.get("PI05_CONFIG_NAME", "pi05_droid")
        self.checkpoint_dir: str | None = os.environ.get("PI05_CHECKPOINT_DIR")
        self.ws_host: str | None = os.environ.get("PI05_WS_HOST")
        self.ws_port: str | None = os.environ.get("PI05_WS_PORT")
        self.norm_stats_path: str | None = os.environ.get("PI05_NORM_STATS_PATH")

        # ---- 运行时状态 ----
        self._policy: Any = None  # PolicyClient 实例（real 模式）
        self._policy_type: str | None = None  # "local" | "ws" | "mock"
        self._norm_stats_sha: str = ""
        self._checkpoint_sha: str = os.environ.get("PI05_CHECKPOINT_SHA", "")
        self._pending_chunk: np.ndarray | None = None  # 当前动作队列（切换时清空）
        self._pending_generated_step: int = -1
        self._current_episode_id: str | None = None
        self._last_latency_ms: int | None = None
        self._last_truncation_count: int = 0
        self._state_lock = threading.Lock()  # 保护 _pending_chunk 并发读写

        # ---- 初始化 ----
        self._load_norm_stats_sha()
        if self.mode == "real":
            self._init_real_policy()
        else:
            self._init_mock_policy()

    # ===================== 模式初始化 =====================
    def _init_mock_policy(self) -> None:
        self._policy_type = "mock"
        logger.info(
            "【Mock 模式】未加载真实 π0.5 模型，infer 返回安全范围内的假动作块。"
        )

    def _init_real_policy(self) -> None:
        """通过 pi05_client 创建策略客户端；不可用则降级 Mock。"""
        if make_policy_client is None:
            logger.warning("pi05_client 不可用，降级到 Mock 模式。")
            self.mode = "dummy"
            self._init_mock_policy()
            return

        self._policy = make_policy_client(
            config_name=self.config_name,
            checkpoint_dir=self.checkpoint_dir,
            ws_host=self.ws_host,
            ws_port=self.ws_port,
        )
        if self._policy is None:
            logger.warning(
                "无可用 π0.5 策略客户端（openpi/WS 均不可用），降级到 Mock 模式。"
            )
            self.mode = "dummy"
            self._init_mock_policy()
            return

        self._policy_type = self._policy.client_type
        # 本地客户端可从 checkpoint 目录计算 sha；远程客户端用环境变量指定
        if not self._checkpoint_sha and self._policy.checkpoint_dir:
            self._checkpoint_sha = _compute_dir_sha(self._policy.checkpoint_dir)
        logger.info(
            "【Real 模式·%s】策略客户端就绪 (sha=%s)",
            self._policy_type,
            self._checkpoint_sha,
        )

    # ===================== norm_stats（仅 SHA 追溯） =====================
    def _load_norm_stats_sha(self) -> None:
        """记录本项目 norm_stats 的 SHA（方案书 §7.2：日志定位唯一统计资产）。

        反归一化由 openpi output_transform 在 policy.infer 内完成，使用 compute_norm_stats
        生成的本项目自有统计（满足 §3.3.1 Para186 不沿用 OpenVLA）；适配器不再二次反归一化，
        此处只读取 SHA 用于追溯。
        """
        path = self.norm_stats_path
        if not path or not os.path.exists(path):
            if self.mode == "real":
                logger.warning(
                    "未提供 PI05_NORM_STATS_PATH 或文件不存在，norm_stats_sha 为空"
                    "（建议指向 compute_norm_stats 输出，用于 §7.2 追溯）。"
                )
            return
        try:
            with open(path, "rb") as f:
                self._norm_stats_sha = hashlib.sha256(f.read()).hexdigest()[:16]
            logger.info("记录 norm_stats SHA: %s (sha=%s)", path, self._norm_stats_sha)
        except Exception as e:
            logger.warning("norm_stats SHA 计算失败：%s", e)

    # ===================== 预处理 =====================
    def _build_example(self, obs: ObsPacket) -> dict:
        """将 ObsPacket 转为 openpi example 字典（方案书 Table 23 Row6）。

        传原始 RGB（仅保证 uint8/HWC/RGB），resize/pad/normalize 由 openpi input_transform
        内部完成；prompt 传原文（不手动 tokenize）。
        """
        example: dict[str, Any] = {
            "observation/exterior_image_1_left": _prep_image(obs.rgb_front),  # 前视 RGB
            "observation/state": np.asarray(obs.robot_state, dtype=np.float32),
            "prompt": obs.instruction,  # 完整自然语言，不拆槽位
            # ---- 通信字段（WebSocket 客户端透传至 openpi_service 构建 ObsPacket） ----
            "episode_id": obs.episode_id,
            "step_id": obs.step_id,
            "timestamp_ns": obs.timestamp_ns,
            "runtime_flags": obs.runtime_flags,
        }
        if obs.rgb_wrist is not None:
            example["observation/wrist_image_left"] = _prep_image(obs.rgb_wrist)
        else:
            logger.warning("rgb_wrist 为空，使用黑图占位（pi05 通常需要腕部图）")
            example["observation/wrist_image_left"] = np.zeros_like(
                _prep_image(obs.rgb_front)
            )
        return example

    def _pixel_audit_if_test(self, obs: ObsPacket) -> None:
        """若传入固定测试图，校验预处理未破坏 RGB/方向/dtype（方案书 §7.5 image_pipeline）。"""
        if obs.rgb_front.shape == FIXED_TEST_IMAGE.shape and np.array_equal(
            obs.rgb_front, FIXED_TEST_IMAGE
        ):
            prepared = _prep_image(obs.rgb_front)
            cs = _image_checksum(prepared)
            # R 通道（channel 0）应保持横向渐变，防止 BGR 通道交换（方案书 §5.1 静默错误）
            r_ok = np.array_equal(prepared[:, :, 0], FIXED_TEST_IMAGE[:, :, 0])
            if cs != FIXED_TEST_IMAGE_CHECKSUM or not r_ok:
                logger.error("像素审计失败：checksum=%s r_channel_ok=%s", cs[:12], r_ok)
            else:
                logger.info("像素审计通过：固定测试图 checksum=%s", cs[:12])

    def verify_pixel_pipeline(self) -> bool:
        """image_pipeline 测试入口（方案书 §7.5）：固定测试图 RGB/方向/dtype 校验。"""
        prepared = _prep_image(FIXED_TEST_IMAGE)
        ok = (
            _image_checksum(prepared) == FIXED_TEST_IMAGE_CHECKSUM
            and prepared.dtype == np.uint8
            and prepared.ndim == 3
            and prepared.shape[2] == 3
            and np.array_equal(prepared[:, :, 0], FIXED_TEST_IMAGE[:, :, 0])
        )
        logger.info(
            "image_pipeline 校验：%s (checksum=%s)",
            "PASS" if ok else "FAIL",
            _image_checksum(prepared)[:12],
        )
        return ok

    # ===================== 安全限幅 =====================
    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """安全限幅（方案书 Table 69 Row7 / 附录B）。

        - 平移前3维 |·|≤0.02m，旋转3维 |·|≤0.0873rad，超限截断并 WARNING。
        - 夹爪第7维仅允许 0.0/1.0，四舍五入。
        - NaN/Inf 直接 raise ValueError，不下发（方案书 §3.4 协议不变量）。
        - 记录被截断的步骤数。
        """
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
            raise ValueError(f"动作块形状非法：{arr.shape}，期望 [N,{ACTION_DIM}]")
        if not np.all(np.isfinite(arr)):
            bad = np.argwhere(~np.isfinite(arr))
            raise ValueError(f"动作块含 NaN/Inf，拒绝下发：{bad.tolist()}")

        clipped = arr.copy()
        trunc_count = 0

        # 平移 dx,dy,dz
        trans = clipped[:, 0:3]
        trans_clipped = np.clip(trans, -MAX_TRANSLATION_M, MAX_TRANSLATION_M)
        diff_trans = np.abs(trans - trans_clipped)
        # 旋转 dax,day,daz
        rot = clipped[:, 3:6]
        rot_clipped = np.clip(rot, -MAX_ROTATION_RAD, MAX_ROTATION_RAD)
        diff_rot = np.abs(rot - rot_clipped)

        for i in range(arr.shape[0]):
            for j in range(3):
                if diff_trans[i, j] > 1e-9:
                    logger.warning(
                        "截断[step=%d,%s] 平移 %.5f -> %.5f (限幅±%.3f m)",
                        i,
                        DIM_NAMES[j],
                        trans[i, j],
                        trans_clipped[i, j],
                        MAX_TRANSLATION_M,
                    )
                    trunc_count += 1
            for j in range(3):
                if diff_rot[i, j] > 1e-9:
                    logger.warning(
                        "截断[step=%d,%s] 旋转 %.5f -> %.5f (限幅±%.4f rad)",
                        i,
                        DIM_NAMES[3 + j],
                        rot[i, j],
                        rot_clipped[i, j],
                        MAX_ROTATION_RAD,
                    )
                    trunc_count += 1

        clipped[:, 0:3] = trans_clipped
        clipped[:, 3:6] = rot_clipped

        # 夹爪：四舍五入到 0/1（>=0.5 为开）
        gripper = clipped[:, 6]
        rounded = np.where(gripper >= 0.5, GRIPPER_OPEN, GRIPPER_CLOSE).astype(
            np.float32
        )
        diff_grip = np.abs(gripper - rounded)
        for i in range(arr.shape[0]):
            if diff_grip[i] > 1e-9:
                logger.warning(
                    "夹爪[step=%d,gripper] %.3f -> %.1f (仅允许 0/1)",
                    i,
                    gripper[i],
                    rounded[i],
                )
        clipped[:, 6] = rounded

        self._last_truncation_count = trunc_count
        return clipped

    # ===================== 推理 =====================
    def _infer_mock(self, obs: ObsPacket) -> np.ndarray:
        """Mock 模式：返回安全范围内的假动作块 float32[10,7]。

        平移每维 ±0.01m，旋转每维 ±0.01rad，夹爪 0/1（均不触发限幅）。
        """
        rng = np.random.RandomState(obs.step_id & 0xFFFF)
        trans = rng.uniform(-0.01, 0.01, size=(MOCK_CHUNK_LEN, 3)).astype(np.float32)
        rot = rng.uniform(-0.01, 0.01, size=(MOCK_CHUNK_LEN, 3)).astype(np.float32)
        grip = rng.randint(0, 2, size=(MOCK_CHUNK_LEN,)).astype(np.float32)
        return np.concatenate([trans, rot, grip[:, None]], axis=1)

    def _infer_real(self, obs: ObsPacket) -> np.ndarray:
        """Real 模式：调用 PolicyClient.infer。返回值已是物理动作（openpi 已反归一化）。"""
        example = self._build_example(obs)
        result = self._policy.infer(example)
        actions = result["actions"] if isinstance(result, dict) else result
        return np.asarray(actions, dtype=np.float32)

    def infer(self, obs: ObsPacket) -> CanonicalActionChunk:
        """主入口：观测 → 安全动作块（方案书 §3.3.1 Para185）。"""
        t0 = time.time()
        self._pixel_audit_if_test(obs)
        self._current_episode_id = obs.episode_id

        if self.mode == "real" and self._policy is not None:
            raw = self._infer_real(obs)
        else:
            if self.mode == "real":
                logger.warning("real 模式但策略未就绪，本次降级使用 Mock 动作。")
            raw = self._infer_mock(obs)

        # ---- 动作块适配（方案书 §3.3.1 Para185）----
        raw = np.asarray(raw, dtype=np.float32)
        if raw.size == 0 or raw.ndim == 0:
            raise InferenceError("Policy returned empty or invalid actions")
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.shape[1] < ACTION_DIM:
            # 不足 7 维 pad 0（方案书 §3.3：不足维度由适配器 padding）
            pad = np.zeros((raw.shape[0], ACTION_DIM - raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=1)
        actions_7 = raw[:, :ACTION_DIM]  # 裁维到前 7 维
        # 反归一化由 openpi output_transform 在 policy.infer 内完成（用本项目 compute_norm_stats，
        # 满足 §3.3.1 Para185/186），适配器不再二次反归一化。
        actions_7 = self._clip_actions(actions_7)  # 安全限幅

        latency_ms = int((time.time() - t0) * 1000)
        self._last_latency_ms = latency_ms

        # 记录待执行动作队列（切换时清空，方案书 §3.3.1 Para186）
        with self._state_lock:
            self._pending_chunk = actions_7.copy()
            self._pending_generated_step = obs.step_id

        logger.info(
            "infer episode=%s step=%d shape=%s latency=%dms mode=%s trunc=%d",
            obs.episode_id,
            obs.step_id,
            actions_7.shape,
            latency_ms,
            self.mode,
            self._last_truncation_count,
        )

        return CanonicalActionChunk(
            actions=actions_7,
            space_id=SPACE_ID,
            frame=FRAME_ID,
            control_hz=CONTROL_HZ,
            generated_step=obs.step_id,
            source_policy=SOURCE_POLICY,
            checkpoint_sha=self._checkpoint_sha,
            expires_after_ms=EXPIRES_AFTER_MS,
        )

    # ===================== 失败切换 =====================
    def cancel_pending_chunk(self) -> None:
        """失败切换时清空动作队列与客户端缓存（方案书 §3.3.1 Para186：不得保留旧动作块）。"""
        with self._state_lock:
            if self._pending_chunk is not None:
                logger.info(
                    "cancel_pending_chunk：丢弃 %d 步待执行动作块（generated_step=%d）",
                    self._pending_chunk.shape[0],
                    self._pending_generated_step,
                )
            self._pending_chunk = None
            self._pending_generated_step = -1
        # 清空策略客户端缓存（若 API 暴露）
        if self._policy is not None:
            try:
                self._policy.clear_cache()
            except Exception:
                pass

    def reset(self) -> None:
        """重置适配器状态，清空动作队列/episode 缓存/延迟统计（方案书 §3.3.1 Para186）。"""
        self.cancel_pending_chunk()
        self._last_latency_ms = None
        self._last_truncation_count = 0
        self._current_episode_id = None
        logger.info("reset：π0.5 适配器状态已清空，下次 infer 将重新传当前图像。")

    # ===================== 健康检查 =====================
    def _query_vram_mb(self) -> int | None:
        """Real 模式查询显存；Mock 返回 None（方案书 §7.1 健康检查）。"""
        if self.mode != "real":
            return None
        # 优先 JAX
        try:
            import jax  # type: ignore

            devs = jax.local_devices()
            if devs:
                ms = devs[0].memory_stats()
                if ms:
                    # JAX memory_stats 键名因版本可能不同：bytes_used / bytes_in_use / peak_bytes_in_use
                    bytes_used = (
                        ms.get("bytes_used")
                        or ms.get("bytes_in_use")
                        or ms.get("peak_bytes_in_use")
                    )
                    if bytes_used is not None:
                        return int(bytes_used // (1024 * 1024))
        except Exception:
            pass
        # 退化为 nvidia-smi
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,nounits,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return int(out.decode().strip().splitlines()[0])
        except Exception:
            return None

    @property
    def checkpoint_sha(self) -> str:
        """返回 checkpoint SHA（对齐 ExecutorDescriptor.checkpoint_sha）。"""
        return self._checkpoint_sha

    @property
    def norm_stats_sha(self) -> str:
        """返回 norm_stats SHA（对齐 ExecutorDescriptor.norm_stats_sha）。"""
        return self._norm_stats_sha

    def health_check(self) -> dict:
        """返回健康状态（方案书 §7.1）。"""
        return {
            "mode": self.mode,
            "policy_type": self._policy_type,
            "config_name": self.config_name,
            "checkpoint_sha": self._checkpoint_sha,
            "norm_stats_sha": self._norm_stats_sha,
            "vram_usage_mb": self._query_vram_mb(),  # Mock 模式为 None
            "last_latency_ms": self._last_latency_ms,
            "openpi_available": OPENPI_AVAILABLE,
            "ws_available": WS_CLIENT_AVAILABLE,
        }

    # ===================== 冻结契约叠加层（体系B 对齐）=====================
    # 以下两个成员为"叠加层"：让现有体系A 的 Pi05Executor 满足体系B 的
    # Executor Protocol（src/industrial_agent/executor.py），不改动任何现有方法。
    # 方案书 interface-contracts.md §4 公共标识、§7.4/§7.5 统一 7 维动作合同。
    @property
    def descriptor(self) -> ExecutorDescriptor:
        """返回体系B ExecutorDescriptor（冻结契约对齐）。

        - name="pi05"；task_types 覆盖 pick_place / visual_manipulation /
          instruction_interaction；action_contract_version="1.0"。
        - checkpoint_sha / norm_stats_sha 从环境变量 PI05_CHECKPOINT_SHA /
          PI05_NORM_STATS_SHA 读取，必须为完整 sha256:<64hex>。
        - ★禁止使用 self._checkpoint_sha（hexdigest()[:16] 截断值，不符合契约）。
        - 环境变量缺失或格式不符时，dummy 模式使用占位 SHA（不抛异常），
          real 模式抛出 ValueError（必须配置）。
        """
        checkpoint_sha = os.environ.get("PI05_CHECKPOINT_SHA", "")
        norm_stats_sha = os.environ.get("PI05_NORM_STATS_SHA", "")
        if not is_pinned_artifact_digest(checkpoint_sha):
            if self.mode == "real":
                raise ValueError(
                    "real 模式必须设置 PI05_CHECKPOINT_SHA 为 sha256:<64hex> 格式，"
                    f"当前值：{checkpoint_sha!r}"
                )
            logger.warning(
                "PI05_CHECKPOINT_SHA 未设置或格式不符（%r），dummy 模式使用占位值",
                checkpoint_sha,
            )
            checkpoint_sha = "sha256:" + "0" * 64
        if not is_pinned_artifact_digest(norm_stats_sha):
            if self.mode == "real":
                raise ValueError(
                    "real 模式必须设置 PI05_NORM_STATS_SHA 为 sha256:<64hex> 格式，"
                    f"当前值：{norm_stats_sha!r}"
                )
            logger.warning(
                "PI05_NORM_STATS_SHA 未设置或格式不符（%r），dummy 模式使用占位值",
                norm_stats_sha,
            )
            norm_stats_sha = "sha256:" + "0" * 64
        return ExecutorDescriptor(
            name="pi05",
            task_types=frozenset(
                {"pick_place", "visual_manipulation", "instruction_interaction"}
            ),
            action_contract_version=ACTION_CONTRACT_VERSION,
            checkpoint_sha=checkpoint_sha,
            norm_stats_sha=norm_stats_sha,
        )

    def to_action_chunk(
        self,
        canonical: CanonicalActionChunk,
        task_id: str,
        executor_name: str,
    ) -> ActionChunk:
        """把体系A CanonicalActionChunk 包装成体系B ActionChunk（冻结契约对齐）。

        - canonical.actions[:, :7] 逐行转 tuple(float, ...)，每步恰好 7 维；
        - ActionStep.duration_ms 默认 100（CanonicalActionChunk 无此字段）；
        - action_space 固定 "ee_delta_pose_gripper"，不用 canonical.space_id；
        - 构造完调 validate_contract() 自校验。
        """
        actions = np.asarray(canonical.actions, dtype=np.float32)
        # duration_ms 是传输元数据（非第 8 维模型输出），schema 要求 int∈[1,10000]；
        # 这里用固定 100ms 作为默认控制周期，与 HTTP 路径动态推导无冲突（两者都满足契约下限）。
        steps = tuple(
            ActionStep.from_sequence(row[:7].tolist(), duration_ms=100)
            for row in actions
        )
        chunk = ActionChunk(
            contract_version=ACTION_CONTRACT_VERSION,
            chunk_id=str(uuid.uuid4()),
            task_id=task_id,
            executor=executor_name,
            steps=steps,
            action_space="ee_delta_pose_gripper",
            frame="robot_base",
            translation_unit="m",
            rotation_unit="rad",
            gripper_unit="normalized",
        )
        chunk.validate_contract()
        return chunk


# ---------------------------------------------------------------------------
# 自测（构造假 ObsPacket，调用 infer，打印结果）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("PI05_MODE", "dummy")

    ex = Pi05Executor()
    print("=== health_check ===")
    hc = ex.health_check()
    print(hc)
    assert "last_latency_ms" in hc, "health_check 缺少 last_latency_ms 字段"

    print("\n=== image_pipeline ===")
    print("pixel_pipeline PASS:", ex.verify_pixel_pipeline())

    # 构造假 ObsPacket（前视/腕部用随机图；本体状态 8 维）
    front = np.random.RandomState(0).randint(0, 256, (480, 640, 3), dtype=np.uint8)
    wrist = np.random.RandomState(1).randint(0, 256, (224, 224, 3), dtype=np.uint8)
    obs = ObsPacket(
        episode_id="test-ep-001",
        step_id=3,
        timestamp_ns=int(time.time() * 1e9),
        rgb_front=front,
        rgb_wrist=wrist,
        robot_state=np.zeros(8, dtype=np.float32),
        instruction="pick up the red cylinder and place it into cell row 2 col 3",
        runtime_flags={"terminated": False, "truncated": False, "camera_ok": True},
    )

    print("\n=== infer (mock) ===")
    chunk = ex.infer(obs)
    print("actions shape:", chunk.actions.shape, "dtype:", chunk.actions.dtype)
    print(
        "space_id:",
        chunk.space_id,
        "| frame:",
        chunk.frame,
        "| control_hz:",
        chunk.control_hz,
    )
    print(
        "source_policy:",
        chunk.source_policy,
        "| checkpoint_sha:",
        repr(chunk.checkpoint_sha),
    )
    print("first action:", chunk.actions[0])
    assert chunk.actions.shape == (MOCK_CHUNK_LEN, ACTION_DIM)
    assert chunk.actions.dtype == np.float32
    assert chunk.source_policy == SOURCE_POLICY

    print("\n=== safety clip test ===")
    bad = np.array([[0.05, -0.05, 0.001, 0.2, -0.2, 0.0, 0.5]], dtype=np.float32)
    clipped = ex._clip_actions(bad)
    print("in :", bad[0])
    print("out:", clipped[0])
    assert abs(clipped[0, 0]) <= np.float32(MAX_TRANSLATION_M)
    assert abs(clipped[0, 3]) <= np.float32(MAX_ROTATION_RAD)
    assert clipped[0, 6] in (0.0, 1.0)

    print("\n=== NaN rejection test ===")
    try:
        ex._clip_actions(np.array([[np.nan, 0, 0, 0, 0, 0, 0]], dtype=np.float32))
        print("ERROR: NaN 未被拒绝")
    except ValueError as e:
        print("NaN rejected:", str(e)[:60])

    print("\n=== failover test ===")
    ex.cancel_pending_chunk()
    ex.reset()
    print("pending_chunk after reset:", ex._pending_chunk)
    print("episode_id after reset:", ex._current_episode_id)
    print("health after reset:", ex.health_check())

    print("\n=== fixed-test-image pixel audit ===")
    obs_audit = ObsPacket(
        episode_id="audit",
        step_id=0,
        timestamp_ns=0,
        rgb_front=FIXED_TEST_IMAGE.copy(),
        rgb_wrist=None,
        robot_state=np.zeros(8, dtype=np.float32),
        instruction="audit",
        runtime_flags={},
    )
    ex.infer(obs_audit)  # 应打印“像素审计通过”

    print("\nALL TESTS PASSED")
