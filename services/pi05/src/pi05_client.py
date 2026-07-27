"""π0.5 策略客户端：封装 openpi 本地推理与 WebSocket 远程推理。

负责人：E（π0.5/openpi）

方案书出处：
- §3.3：官方支持 WebSocket 远程推理（把 π0.5 部署在独立 GPU/机器，解决仿真显存争用）。
- §7.1：openpi π0.5 服务，官方 WebSocket 或封装 RPC；本模块即"封装 RPC"层。
- Table 21 Row3（§3.3）：需要 LoRA 时必须走 JAX 路径。

本模块把【模型加载 / 网络传输】代码与 pi05.py 业务逻辑分离：
pi05.py 只依赖 PolicyClient 抽象接口，不含任何 WebSocket/HTTP 或 openpi 加载代码。

openpi API 关键事实（官方源码 Policy.infer）：
  infer(obs) 内部依次执行 input_transform（resize 图像 / tokenize prompt / pad state / normalize）
  与 output_transform（Unnormalize 反归一化）。
  因此 create_trained_policy 返回的 policy，其 infer(example)["actions"] 已经是【物理动作】，
  反归一化使用 compute_norm_stats 生成的统计（本项目自有，满足 §3.3.1 Para186 不沿用 OpenVLA）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("pi05_client")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][pi05_client] %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# openpi 本地依赖（JAX 路径）
# ---------------------------------------------------------------------------
try:
    from openpi.policies import policy_config as _openpi_policy_config  # type: ignore
    from openpi.shared import download as _openpi_download  # type: ignore
    from openpi.training import config as _openpi_config  # type: ignore

    OPENPI_AVAILABLE = True
except Exception:
    OPENPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# WebSocket 依赖（用于与 openpi_service.py 协议通信）
# 原 WS_CLIENT_AVAILABLE 为永真占位（try 块无实际操作），
# 现已删除，改为真实 import websockets 检测。
# ---------------------------------------------------------------------------
try:
    import websockets  # type: ignore

    WS_CLIENT_AVAILABLE = True
except Exception:
    WS_CLIENT_AVAILABLE = False

try:
    import msgpack as _msgpack  # type: ignore

    _MSGPACK_AVAILABLE = True
except Exception:
    _MSGPACK_AVAILABLE = False


@runtime_checkable
class PolicyClient(Protocol):
    """策略客户端抽象接口。pi05.py 依赖此接口，不依赖具体网络/模型实现。"""

    def infer(self, example: dict) -> dict:
        """输入 openpi example 字典，返回含 "actions" 的字典。"""
        ...

    def clear_cache(self) -> None:
        """清空客户端/策略缓存（失败切换时调用，方案书 §3.3.1 Para186）。"""
        ...

    @property
    def client_type(self) -> str:
        """客户端类型标识：ws / local。"""
        ...

    @property
    def checkpoint_dir(self) -> str | None:
        """本地客户端返回 checkpoint 路径；远程客户端返回 None。"""
        ...


def _safe_clear_cache(obj: object) -> None:
    """尽力调用 clear_cache/invalidate_cache，避免误调 close/reset 关闭连接或重载模型。"""
    for name in ("clear_cache", "invalidate_cache"):
        m = getattr(obj, name, None)
        if callable(m):
            try:
                m()
            except Exception:
                pass
            return


class WebsocketPolicyClient:
    """WebSocket 远程客户端（与 openpi_service.py ObsPacket 协议对齐）。

    方案书 §3.3：远程推理解决仿真显存争用；总 Agent 只走版本化 RPC。
    发送 ObsPacket 格式数据包，接收 CanonicalActionChunk 响应，
    将 openpi example 键名透明转换为服务端期望的 ObsPacket 字段。
    """

    def __init__(self, host: str, port: int) -> None:
        if not WS_CLIENT_AVAILABLE:
            raise RuntimeError(
                "websockets 库不可用，无法创建 WebSocket 连接。"
                "请安装: pip install websockets"
            )
        self._host = host
        self._port = port
        self._ws_url = f"ws://{host}:{port}"
        self._client: Any = None  # native websockets 连接对象
        self._loop: Any = None
        logger.info("【WS 客户端】目标 %s (待首次 infer 时连接)", self._ws_url)

    def _ensure_connected(self) -> None:
        """建立 / 重建 WebSocket 连接。"""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._loop = loop

        async def _connect():
            self._client = await websockets.connect(
                self._ws_url,
                max_size=10 * 1024 * 1024,  # 10 MB 上限（方案书 §3.4 长度校验）
                ping_interval=20,
                ping_timeout=10,
            )

        if self._client is None or self._client.closed:
            self._loop.run_until_complete(_connect())
            logger.info("【WS 客户端】已连接 %s", self._ws_url)

    @staticmethod
    def _serialize(data: dict[str, Any]) -> bytes:
        """序列化为 openpi_service 兼容格式：msgpack 优先，fallback JSON。"""
        if _MSGPACK_AVAILABLE:
            try:
                return _msgpack.packb(data, use_bin_type=True)
            except Exception as e:
                logger.warning("msgpack 序列化失败，回退 JSON: %s", e)
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _deserialize(raw: bytes) -> dict[str, Any]:
        """反序列化 openpi_service 响应。"""
        if _MSGPACK_AVAILABLE:
            try:
                result = _msgpack.unpackb(raw, raw=False)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        return json.loads(raw.decode("utf-8"))

    def infer(self, example: dict) -> dict:
        """将 openpi example dict 转为 ObsPacket 格式发送，提取 actions 返回。

        example 键名映射（pi05.py _build_example → openpi_service ObsPacket）：
          "observation/exterior_image_1_left"  → "rgb_front"
          "observation/wrist_image_left"       → "rgb_wrist"
          "observation/state"                  → "robot_state"
          "prompt"                             → "instruction"
          "episode_id" / "step_id" / …         → 透传
        """
        self._ensure_connected()

        # ---- 协议转换：openpi example → ObsPacket（方案书 §3.4） ----
        _front = example.get(
            "observation/exterior_image_1_left",
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        _state = example.get("observation/state", np.zeros(8, dtype=np.float32))
        request: dict[str, Any] = {
            "schema_version": "v1",
            "episode_id": str(example.get("episode_id", "unknown")),
            "step_id": int(example.get("step_id", 0)),
            "timestamp_ns": int(example.get("timestamp_ns", int(time.time() * 1e9))),
            "instruction": str(example.get("prompt", "")),
            "rgb_front": (
                {
                    "bytes": _front.tobytes(),
                    "shape": list(_front.shape),
                    "dtype": str(_front.dtype),
                }
                if isinstance(_front, np.ndarray)
                else _front
            ),
            "robot_state": (
                _state.tolist() if isinstance(_state, np.ndarray) else _state
            ),
            "runtime_flags": example.get(
                "runtime_flags",
                {"terminated": False, "truncated": False, "camera_ok": True},
            ),
        }
        wrist = example.get("observation/wrist_image_left")
        if wrist is not None:
            request["rgb_wrist"] = (
                {
                    "bytes": wrist.tobytes(),
                    "shape": list(wrist.shape),
                    "dtype": str(wrist.dtype),
                }
                if isinstance(wrist, np.ndarray)
                else wrist
            )

        # ---- 发送与接收 ----
        async def _send_recv():
            await self._client.send(self._serialize(request))
            raw = await self._client.recv()
            return self._deserialize(raw)

        try:
            response: dict[str, Any] = self._loop.run_until_complete(_send_recv())
        except Exception as e:
            logger.error("WebSocket 请求失败: %s", e)
            self._client = None  # 标记断开，下次 infer 时重连
            raise ConnectionError(f"WebSocket 推理失败 ({self._ws_url}): {e}") from e

        # ---- 校验响应（方案书 §3.4 协议不变量：非法不下发） ----
        if not isinstance(response, dict):
            raise ValueError(f"服务端返回非 dict: {type(response)}")
        if "error" in response:
            raise RuntimeError(f"服务端返回错误: {response['error']}")
        if "actions" not in response:
            raise ValueError(
                f"服务端响应缺少 'actions' 字段，现有 keys: {list(response.keys())[:10]}"
            )
        return {"actions": np.array(response["actions"], dtype=np.float32)}

    def clear_cache(self) -> None:
        """清空客户端连接缓存（失败切换时调用，方案书 §3.3.1 Para186）。"""
        if self._client is not None:
            try:

                async def _close():
                    await self._client.close()

                self._loop.run_until_complete(_close())
            except Exception:
                pass
            self._client = None
            logger.info("【WS 客户端】连接已关闭（clear_cache）")

    @property
    def client_type(self) -> str:
        return "ws"

    @property
    def checkpoint_dir(self) -> str | None:
        return None


class LocalOpenPiPolicyClient:
    """本地 JAX 推理客户端（封装 create_trained_policy）。

    方案书 Table 21 Row3：需要 LoRA 时必须走 JAX 路径。
    create_trained_policy 返回的 policy，其 infer() 已含完整 input/output transform，
    返回物理动作（已反归一化）。
    """

    def __init__(self, config_name: str, checkpoint_dir: str | None) -> None:
        config = _openpi_config.get_config(config_name)
        ckpt = checkpoint_dir
        if not ckpt:
            ckpt = _openpi_download.maybe_download(
                f"gs://openpi-assets/checkpoints/{config_name}"
            )
        self._policy = _openpi_policy_config.create_trained_policy(config, ckpt)
        self._ckpt = ckpt
        self._config_name = config_name
        logger.info("【JAX 客户端】加载 %s @ %s", config_name, ckpt)

    def infer(self, example: dict) -> dict:
        return self._policy.infer(example)

    def clear_cache(self) -> None:
        _safe_clear_cache(self._policy)

    @property
    def client_type(self) -> str:
        return "local"

    @property
    def checkpoint_dir(self) -> str | None:
        return self._ckpt


def make_policy_client(
    config_name: str,
    checkpoint_dir: str | None,
    ws_host: str | None,
    ws_port: str | None,
) -> PolicyClient | None:
    """按优先级创建策略客户端：WebSocket > 本地 JAX。

    均不可用时返回 None，由调用方（pi05.py）降级到 Mock 模式。
    """
    # 优先 WebSocket（§3.3：远程推理解决显存争用）
    if ws_host and ws_port and WS_CLIENT_AVAILABLE:
        try:
            return WebsocketPolicyClient(ws_host, int(ws_port))
        except Exception as e:
            logger.warning("WebSocket 连接失败：%s；尝试本地 JAX 路径", e)

    # 本地 JAX 路径（Table 21 Row3）
    if OPENPI_AVAILABLE:
        try:
            return LocalOpenPiPolicyClient(config_name, checkpoint_dir)
        except Exception as e:
            logger.error("本地 openpi 加载失败：%s", e)

    return None
