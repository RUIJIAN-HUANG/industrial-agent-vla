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

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger("pi05_client")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][pi05_client] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# openpi 本地依赖（JAX 路径）
# ---------------------------------------------------------------------------
try:
    from openpi.training import config as _openpi_config  # type: ignore
    from openpi.policies import policy_config as _openpi_policy_config  # type: ignore
    from openpi.shared import download as _openpi_download  # type: ignore
    OPENPI_AVAILABLE = True
except Exception:
    OPENPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# openpi 远程客户端依赖（WebSocket）
# ---------------------------------------------------------------------------
try:
    from openpi_client import websocket_client_policy as _ws_client  # type: ignore
    WS_CLIENT_AVAILABLE = True
except Exception:
    WS_CLIENT_AVAILABLE = False


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
    def checkpoint_dir(self) -> Optional[str]:
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
    """WebSocket 远程客户端（封装 openpi_client.WebsocketClientPolicy）。

    方案书 §3.3：远程推理解决仿真显存争用；总 Agent 只走版本化 RPC。
    """

    def __init__(self, host: str, port: int) -> None:
        self._client = _ws_client.WebsocketClientPolicy(host=host, port=port)
        self._host = host
        self._port = port
        logger.info("【WS 客户端】连接 openpi WebSocket %s:%s", host, port)

    def infer(self, example: dict) -> dict:
        return self._client.infer(example)

    def clear_cache(self) -> None:
        _safe_clear_cache(self._client)

    @property
    def client_type(self) -> str:
        return "ws"

    @property
    def checkpoint_dir(self) -> Optional[str]:
        return None


class LocalOpenPiPolicyClient:
    """本地 JAX 推理客户端（封装 create_trained_policy）。

    方案书 Table 21 Row3：需要 LoRA 时必须走 JAX 路径。
    create_trained_policy 返回的 policy，其 infer() 已含完整 input/output transform，
    返回物理动作（已反归一化）。
    """

    def __init__(self, config_name: str, checkpoint_dir: Optional[str]) -> None:
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
    def checkpoint_dir(self) -> Optional[str]:
        return self._ckpt


def make_policy_client(
    config_name: str,
    checkpoint_dir: Optional[str],
    ws_host: Optional[str],
    ws_port: Optional[str],
) -> Optional[PolicyClient]:
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
