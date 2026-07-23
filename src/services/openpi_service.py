"""π0.5 网络服务层（openpi_service）。

负责人：E（π0.5/openpi）

方案书出处：
- §3.3 / §3.3.1：π0.5 适配流程；官方支持 WebSocket 远程推理。
- §3.4：ObsPacket v1 / CanonicalActionChunk v1 协议不变量；动作过期丢弃。
- §7.1：服务边界与健康检查（openpi π0.5 服务接口）。
- §7.2：RPC 请求防错字段（schema_version/episode_id/step_id/checkpoint_sha/expires_after_ms）。

本文件为角色 E 的交付物，负责网络服务层：
- WebSocket 连接管理、请求校验、序列化（msgpack 优先，与 openpi 官方兼容；fallback JSON）。
- 调用 src.executors.pi05.Pi05Executor 做模型推理（不在本层直接碰模型）。
- 动作块过期丢弃、episode_id/step_id 防错、健康检查 HTTP 端点。
- PI05_SERVICE_MODE=dummy 时不 import openpi，适合本地全流程联调。

启动：
    uvicorn src.services.openpi_service:app --reload
或：
    python -m src.services.openpi_service
"""

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 第三方依赖：FastAPI / uvicorn 必需；msgpack 可选（fallback JSON）
# ---------------------------------------------------------------------------
try:
    import msgpack  # type: ignore
    _MSGPACK_AVAILABLE = True
except Exception:  # msgpack 不存在时服务仍可启动，回退 JSON
    _MSGPACK_AVAILABLE = False

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn
except Exception as _e:  # 致命依赖缺失，无法启动
    print(f"[openpi_service] 致命错误：FastAPI/uvicorn 未安装：{_e}", file=sys.stderr)
    raise

# starlette 线程池工具：把同步推理放到线程池，避免阻塞事件循环
try:
    from starlette.concurrency import run_in_threadpool  # type: ignore
except Exception:  # 退化为 asyncio 线程池
    run_in_threadpool = None  # type: ignore

import numpy as np

# ---------------------------------------------------------------------------
# 环境变量配置
# ---------------------------------------------------------------------------
SERVICE_MODE = os.environ.get("PI05_SERVICE_MODE", "dummy").lower()
if SERVICE_MODE not in ("dummy", "real"):
    print(f"[openpi_service] 未知 PI05_SERVICE_MODE={SERVICE_MODE}，回退到 dummy",
          file=sys.stderr)
    SERVICE_MODE = "dummy"

# 把服务层模式桥接给执行器（执行器读 PI05_MODE），保证服务层为唯一真相源
os.environ["PI05_MODE"] = SERVICE_MODE

SERVICE_HOST = os.environ.get("PI05_SERVICE_HOST", "0.0.0.0")
try:
    SERVICE_PORT = int(os.environ.get("PI05_SERVICE_PORT", "8000"))
except ValueError:
    SERVICE_PORT = 8000

SCHEMA_VERSION = "v1"
POLICY_ID = "pi05"
DEFAULT_CONTROL_HZ = 10
DEFAULT_EXPIRES_AFTER_MS = 1000

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("openpi_service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s][openpi_service] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# 导入执行器
# 注意：openpi 的 import 已在执行器内部 try/except，本层不再直接 import openpi，
#       dummy 模式下 openpi 不存在时服务仍能启动。
# ---------------------------------------------------------------------------
try:
    from src.executors.pi05 import Pi05Executor, ObsPacket  # type: ignore
    _EXECUTOR_AVAILABLE = True
    _EXECUTOR_IMPORT_ERROR = ""
except Exception as _e:
    _EXECUTOR_AVAILABLE = False
    _EXECUTOR_IMPORT_ERROR = repr(_e)
    Pi05Executor = None  # type: ignore
    ObsPacket = None  # type: ignore
    logger.error("Pi05Executor 导入失败：%s；服务将以错误状态启动", _e)

# ---------------------------------------------------------------------------
# 全局执行器实例 + 运行时状态
# ---------------------------------------------------------------------------
executor: Optional[Any] = None
_START_TIME = time.time()

# 动作块过期丢弃状态（方案书 §3.4 动作过期）
# pending_chunks[episode_id] = {
#     "generated_step": int, "timestamp": float,
#     "expires_after_ms": int, "actions": list
# }
pending_chunks: Dict[str, dict] = {}
current_episode_id: Optional[str] = None
last_step_id: Optional[int] = None


def _init_executor() -> None:
    """根据 PI05_SERVICE_MODE 初始化执行器（dummy/real）。"""
    global executor
    if not _EXECUTOR_AVAILABLE:
        logger.error("执行器不可用，跳过初始化。")
        return
    try:
        executor = Pi05Executor()
        logger.info("Pi05Executor 初始化完成：mode=%s", SERVICE_MODE)
    except Exception as e:
        logger.error("Pi05Executor 初始化异常：%s", e)
        executor = None


def _checkpoint_sha() -> str:
    if executor is not None:
        try:
            return getattr(executor, "_checkpoint_sha", "") or ""
        except Exception:
            return ""
    return ""


def _norm_stats_sha() -> str:
    if executor is not None:
        try:
            return getattr(executor, "_norm_stats_sha", "") or ""
        except Exception:
            return ""
    return ""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="openpi π0.5 Service", version="1.0.0")


@app.on_event("startup")
async def _on_startup() -> None:
    """启动时初始化执行器并打印自测日志。"""
    _init_executor()
    logger.info("===== openpi_service 启动自检 =====")
    logger.info("服务监听：%s:%d", SERVICE_HOST, SERVICE_PORT)
    logger.info("服务模式：%s", SERVICE_MODE)
    logger.info("msgpack 可用：%s", _MSGPACK_AVAILABLE)
    logger.info("执行器可用：%s", _EXECUTOR_AVAILABLE)
    if _EXECUTOR_IMPORT_ERROR:
        logger.info("执行器导入错误：%s", _EXECUTOR_IMPORT_ERROR)
    if executor is not None:
        try:
            logger.info("执行器健康检查：%s", executor.health_check())
        except Exception as e:
            logger.warning("执行器健康检查异常：%s", e)
    logger.info("checkpoint_sha=%s norm_stats_sha=%s",
                _checkpoint_sha(), _norm_stats_sha())
    logger.info("WebSocket 路径：ws://%s:%d/", SERVICE_HOST, SERVICE_PORT)
    logger.info("健康检查：http://%s:%d/health", SERVICE_HOST, SERVICE_PORT)
    logger.info("====================================")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """健康检查端点（方案书 §7.1）。

    返回：status / mode / checkpoint_sha / norm_stats_sha / uptime_seconds
    """
    status = "ok"
    if executor is None:
        status = "error"
    else:
        try:
            executor.health_check()
        except Exception:
            status = "error"
    return {
        "status": status,
        "mode": SERVICE_MODE,
        "checkpoint_sha": _checkpoint_sha(),
        "norm_stats_sha": _norm_stats_sha(),
        "uptime_seconds": round(time.time() - _START_TIME, 3),
    }


# ---------------------------------------------------------------------------
# 序列化辅助：msgpack 优先（与 openpi 官方兼容），fallback JSON
# ---------------------------------------------------------------------------
def _deserialize(raw_bytes: Optional[bytes], raw_text: Optional[str]) -> Dict[str, Any]:
    """反序列化客户端消息：优先 msgpack，fallback JSON。"""
    if raw_bytes is not None:
        if _MSGPACK_AVAILABLE:
            try:
                data = msgpack.unpackb(raw_bytes, raw=False)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning("msgpack 反序列化失败，尝试 JSON：%s", e)
        # 二进制帧也可能是 JSON 文本
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            raise ValueError(f"无法反序列化二进制消息：{e}")
    if raw_text is not None:
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                return data
        except Exception as e:
            raise ValueError(f"无法反序列化文本消息：{e}")
    raise ValueError("空消息")


def _serialize(response: Dict[str, Any]) -> Tuple[bytes, bool]:
    """序列化响应：msgpack 可用返回 (bytes, True)；否则返回 (JSON bytes, True)。

    第二个返回值保留以区分发送方式（统一 send_bytes）。
    """
    if _MSGPACK_AVAILABLE:
        try:
            return msgpack.packb(response, use_bin_type=True), True
        except Exception as e:
            logger.warning("msgpack 序列化失败，回退 JSON：%s", e)
    return json.dumps(response, ensure_ascii=False).encode("utf-8"), True


async def _send(ws: WebSocket, response: Dict[str, Any]) -> None:
    """统一发送：msgpack 二进制 / JSON 文本。"""
    payload, _ = _serialize(response)
    # metadata 用 JSON 文本发送；推理响应统一用二进制（msgpack 或 JSON bytes）
    await ws.send_bytes(payload)


async def _send_error(ws: WebSocket, error: Dict[str, Any]) -> None:
    """发送错误消息（同样走二进制序列化）。"""
    logger.warning("请求错误：%s", error.get("error"))
    await _send(ws, error)


# ---------------------------------------------------------------------------
# 请求校验（方案书 §3.4 协议不变量 / §7.2 防错字段）
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ("episode_id", "step_id", "rgb_front", "instruction")


def _validate_request(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验请求；返回错误 dict 表示失败，None 表示通过。"""
    # 必填字段
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] is None]
    if missing:
        return {"error": "missing_required_fields", "missing": missing,
                "schema_version": SCHEMA_VERSION}

    # episode_id 必须是非空字符串
    ep = data["episode_id"]
    if not isinstance(ep, str) or not ep.strip():
        return {"error": "invalid_episode_id",
                "reason": "episode_id 必须是非空字符串",
                "schema_version": SCHEMA_VERSION}

    # step_id 必须是整数且 >= 0
    sid = data["step_id"]
    if isinstance(sid, bool):
        return {"error": "invalid_step_id", "reason": "step_id 不能是 bool",
                "schema_version": SCHEMA_VERSION}
    if isinstance(sid, int):
        pass
    elif isinstance(sid, float) and sid.is_integer():
        sid = int(sid)
    else:
        try:
            sid = int(sid)  # type: ignore
        except (TypeError, ValueError):
            return {"error": "invalid_step_id", "reason": "step_id 必须是整数",
                    "schema_version": SCHEMA_VERSION}
    if sid < 0:
        return {"error": "invalid_step_id", "reason": "step_id 必须 >= 0",
                "schema_version": SCHEMA_VERSION}
    data["step_id"] = sid

    # instruction 必须是字符串
    if not isinstance(data["instruction"], str):
        return {"error": "invalid_instruction", "reason": "instruction 必须是字符串",
                "schema_version": SCHEMA_VERSION}

    return None


def _check_episode_step(data: Dict[str, Any]) -> None:
    """episode_id / step_id 连续性与动作块过期检查（方案书 §3.4 动作过期）。"""
    global current_episode_id, last_step_id
    ep = data["episode_id"]
    sid = data["step_id"]

    # episode 切换：清空 pending_chunks，重置执行器
    if ep != current_episode_id:
        logger.info("Episode changed, cleared pending chunks (prev=%s new=%s)",
                    current_episode_id, ep)
        pending_chunks.clear()
        if executor is not None:
            try:
                executor.reset()
                executor.cancel_pending_chunk()
            except Exception as e:
                logger.warning("执行器 reset/cancel 异常：%s", e)
        current_episode_id = ep
        last_step_id = None

    # step_id 连续性：同一 episode 内应递增
    if last_step_id is not None and sid != last_step_id + 1:
        logger.warning("step_id 不连续：期望 %d，收到 %d", last_step_id + 1, sid)
    # 同一 episode 内 step_id 应递增（非递增告警）
    if last_step_id is not None and sid <= last_step_id:
        logger.warning("step_id 未递增：last=%d curr=%d", last_step_id, sid)
    last_step_id = sid

    # 检查当前 episode 的 pending chunk 是否过期，过期则丢弃
    chunk = pending_chunks.get(ep)
    if chunk is not None:
        age_ms = (time.time() - chunk["timestamp"]) * 1000.0
        ttl = chunk.get("expires_after_ms", DEFAULT_EXPIRES_AFTER_MS)
        if age_ms > ttl:
            logger.warning("Action chunk expired (episode=%s age_ms=%.0f > %dms)",
                           ep, age_ms, ttl)
            pending_chunks.pop(ep, None)


def _build_obs(data: Dict[str, Any]) -> Any:
    """从请求字典构造 ObsPacket。"""
    if ObsPacket is None:
        raise RuntimeError("ObsPacket 不可用（执行器未导入）")

    rgb_front = np.array(data["rgb_front"], dtype=np.uint8)
    rgb_wrist = None
    if data.get("rgb_wrist") is not None:
        rgb_wrist = np.array(data["rgb_wrist"], dtype=np.uint8)
    robot_state = np.array(data.get("robot_state", []), dtype=np.float32)

    ts = data.get("timestamp_ns")
    if ts is None:
        ts = int(time.time() * 1e9)

    flags = data.get("runtime_flags", {}) or {}

    return ObsPacket(
        episode_id=data["episode_id"],
        step_id=int(data["step_id"]),
        timestamp_ns=int(ts),
        rgb_front=rgb_front,
        rgb_wrist=rgb_wrist,
        robot_state=robot_state,
        instruction=data["instruction"],
        runtime_flags=flags,
    )


# ---------------------------------------------------------------------------
# WebSocket 主入口：ws://host:port/
# ---------------------------------------------------------------------------
@app.websocket("/")
async def ws_infer(ws: WebSocket) -> None:
    await ws.accept()

    # 执行器未就绪：返回错误并关闭
    if executor is None:
        await _send_error(ws, {"error": "executor_not_ready",
                               "reason": "执行器未初始化",
                               "schema_version": SCHEMA_VERSION})
        await ws.close()
        return

    # 连接建立：发送服务端 metadata（JSON 格式）
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "checkpoint_sha": _checkpoint_sha(),
        "control_hz": DEFAULT_CONTROL_HZ,
        "mode": SERVICE_MODE,
        "norm_stats_sha": _norm_stats_sha(),
        "expires_after_ms": DEFAULT_EXPIRES_AFTER_MS,
    }
    await ws.send_text(json.dumps(metadata, ensure_ascii=False))
    logger.info("WebSocket 连接建立，已发送 metadata")

    try:
        while True:
            # 同时兼容二进制（msgpack）与文本（JSON）帧
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            raw_bytes = msg.get("bytes")
            raw_text = msg.get("text")

            # 反序列化
            try:
                data = _deserialize(raw_bytes, raw_text)
            except Exception as e:
                await _send_error(ws, {"error": "deserialize_failed",
                                       "reason": str(e),
                                       "schema_version": SCHEMA_VERSION})
                continue

            # 字段校验
            err = _validate_request(data)
            if err is not None:
                await _send_error(ws, err)
                continue

            # episode/step 连续性与动作过期检查
            _check_episode_step(data)

            # 构造 ObsPacket
            try:
                obs = _build_obs(data)
            except Exception as e:
                await _send_error(ws, {"error": "obs_build_failed",
                                       "reason": str(e),
                                       "schema_version": SCHEMA_VERSION})
                continue

            # 调用执行器（同步推理放线程池，避免阻塞事件循环）
            try:
                if run_in_threadpool is not None:
                    chunk = await run_in_threadpool(executor.infer, obs)
                else:
                    loop = asyncio.get_event_loop()
                    chunk = await loop.run_in_executor(
                        None, lambda: executor.infer(obs))
            except Exception as e:
                logger.error("推理异常：%s", e)
                await _send_error(ws, {"error": "infer_failed",
                                       "reason": str(e),
                                       "schema_version": SCHEMA_VERSION})
                continue

            # 记录 pending chunk，用于后续请求的过期判断
            pending_chunks[data["episode_id"]] = {
                "generated_step": chunk.generated_step,
                "timestamp": time.time(),
                "expires_after_ms": chunk.expires_after_ms,
                "actions": chunk.actions.tolist(),
            }

            # 序列化返回（方案书 §3.4 CanonicalActionChunk v1）
            response = {
                "schema_version": SCHEMA_VERSION,
                "episode_id": data["episode_id"],
                "step_id": data["step_id"],
                "actions": chunk.actions.tolist(),
                "space_id": chunk.space_id,
                "frame": chunk.frame,
                "control_hz": chunk.control_hz,
                "generated_step": chunk.generated_step,
                "source_policy": chunk.source_policy,
                "checkpoint_sha": chunk.checkpoint_sha,
                "expires_after_ms": chunk.expires_after_ms,
            }
            await _send(ws, response)
            logger.info("推理完成 episode=%s step=%d shape=%s",
                        data["episode_id"], data["step_id"],
                        list(chunk.actions.shape))
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception as e:
        logger.error("WebSocket 异常：%s", e)
    finally:
        logger.info("WebSocket 连接关闭")


# ---------------------------------------------------------------------------
# 直接运行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.services.openpi_service:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=False,
    )
