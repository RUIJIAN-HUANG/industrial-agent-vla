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
import re
import sys
import json
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# ---------------------------------------------------------------------------
# 第三方依赖：FastAPI / uvicorn 必需；msgpack 可选（fallback JSON）
# ---------------------------------------------------------------------------
try:
    import msgpack  # type: ignore
    _MSGPACK_AVAILABLE = True
except Exception:  # msgpack 不存在时服务仍可启动，回退 JSON
    _MSGPACK_AVAILABLE = False

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import JSONResponse
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
# 默认 8102，对齐 configs/agent.default.json 的 pi05.base_url（方案书 §2 进程拓扑）
try:
    SERVICE_PORT = int(os.environ.get("PI05_SERVICE_PORT", "8102"))
except ValueError:
    SERVICE_PORT = 8102

# SCHEMA_VERSION 保留 "v1" 仅供现有 WebSocket 端点使用（不影响历史 WS 契约）
SCHEMA_VERSION = "v1"
# CONTRACT_SCHEMA_VERSION 用于 HTTP 路由（/v1/infer、/v1/cancel、/health），
# 对齐 schemas/*.json 的 schema_version pattern "^1\.[0-9]+$"（方案书 §4 公共标识）
CONTRACT_SCHEMA_VERSION = "1.0"
POLICY_ID = "pi05"
DEFAULT_CONTROL_HZ = 10
DEFAULT_EXPIRES_AFTER_MS = 1000

# sha256:<64hex> 校验模式（方案书 §4：checkpoint_sha/norm_stats_sha 必须为完整不可变摘要）
_SHA_PATTERN = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
# HTTP 路由声明的支持任务类型（方案书 §6.2 / Pi05Adapter.descriptor.task_types）
SUPPORTED_TASK_TYPES: List[str] = [
    "pick_place",
    "visual_manipulation",
    "instruction_interaction",
]
SUPPORTED_ACTION_CONTRACTS: List[str] = [CONTRACT_SCHEMA_VERSION]

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

# 稳定错误码枚举（方案书 §13；src/industrial_agent/errors.py 为冻结文件，仅 import）
try:
    from src.industrial_agent.errors import FailureCode  # type: ignore
    _FAILURE_CODE_AVAILABLE = True
except Exception as _e:  # 退化：错误码用字符串常量兜底
    _FAILURE_CODE_AVAILABLE = False
    logger.warning("FailureCode 导入失败：%s；错误码改用字符串常量", _e)

    class _FailureCodeFallback:  # type: ignore[no-redef]
        EXECUTOR_MODEL_REVISION_MISMATCH = "EXEC_2105_MODEL_REVISION_MISMATCH"
        EXECUTOR_UNAVAILABLE = "EXEC_2101_UNAVAILABLE"
        EXECUTOR_RUNTIME = "EXEC_2104_RUNTIME"
        EXECUTOR_TIMEOUT = "EXEC_2102_TIMEOUT"
        EXECUTOR_BACKPRESSURE = "EXEC_2106_BACKPRESSURE"
        EXECUTOR_CANCELLED = "EXEC_2107_CANCELLED"
        INVALID_TASK = "TASK_1001_INVALID"

    FailureCode = _FailureCodeFallback  # type: ignore[assignment, misc]

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
_pending_chunks_lock: asyncio.Lock = asyncio.Lock()  # 并发保护（方案书 §7.1：多 episode 并发安全）
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
# sha 解析（env 优先 + executor 兜底）与格式校验
# 方案书 §4 / §14.2：checkpoint_sha/norm_stats_sha 必须为 sha256:<64hex>；
# env 优先让生产部署能固定 pinned sha，executor 兜底兼容现有 mock 测试。
# ---------------------------------------------------------------------------
def _resolve_checkpoint_sha() -> str:
    """checkpoint_sha：环境变量 PI05_CHECKPOINT_SHA 优先，否则读 executor。"""
    env_sha = os.environ.get("PI05_CHECKPOINT_SHA", "").strip()
    if env_sha:
        return env_sha
    return _checkpoint_sha()


def _resolve_norm_stats_sha() -> str:
    """norm_stats_sha：环境变量 PI05_NORM_STATS_SHA 优先，否则读 executor。"""
    env_sha = os.environ.get("PI05_NORM_STATS_SHA", "").strip()
    if env_sha:
        return env_sha
    return _norm_stats_sha()


def _validate_sha_format(sha: str, field_name: str) -> None:
    """启动时校验 sha 格式；不合规仅 warning 不拒绝启动（dummy/mock 可继续跑）。

    生产 real 模式部署方必须设置合规的 PI05_CHECKPOINT_SHA / PI05_NORM_STATS_SHA。
    """
    if not sha:
        logger.warning("%s 为空（生产部署必须设置 sha256:<64hex> 环境变量）", field_name)
        return
    if not _SHA_PATTERN.fullmatch(sha):
        logger.warning(
            "%s 格式不合规：%s（应匹配 sha256:<64hex>；dummy/mock 模式可继续，"
            "real 模式必须修正）", field_name, sha,
        )


# ---------------------------------------------------------------------------
# /v1/cancel 幂等状态：记录已取消的 task_id（方案书 §8：cancel 幂等）
# ---------------------------------------------------------------------------
_cancelled_tasks: set[str] = set()
_cancelled_tasks_lock: asyncio.Lock = asyncio.Lock()


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
    # sha 解析（env 优先 + executor 兜底）并校验格式（仅警告，不拒绝启动）
    ckpt_sha = _resolve_checkpoint_sha()
    norm_sha = _resolve_norm_stats_sha()
    _validate_sha_format(ckpt_sha, "checkpoint_sha")
    _validate_sha_format(norm_sha, "norm_stats_sha")
    logger.info("checkpoint_sha=%s norm_stats_sha=%s", ckpt_sha, norm_sha)
    logger.info("HTTP 契约 schema_version=%s", CONTRACT_SCHEMA_VERSION)
    logger.info("WebSocket 路径：ws://%s:%d/", SERVICE_HOST, SERVICE_PORT)
    logger.info("健康检查：http://%s:%d/health", SERVICE_HOST, SERVICE_PORT)
    logger.info("推理端点：http://%s:%d/v1/infer", SERVICE_HOST, SERVICE_PORT)
    logger.info("取消端点：http://%s:%d/v1/cancel", SERVICE_HOST, SERVICE_PORT)
    logger.info("====================================")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """健康检查端点（方案书 §6 / schemas/executor-health.schema.json）。

    返回 7 必填字段：schema_version / service / status / checkpoint_sha /
    norm_stats_sha / supported_task_types / supported_action_contracts。
    status ∈ {"ready","loading","degraded"}（schema 不允许 "ok"/"error"）。
    """
    ckpt_sha = _resolve_checkpoint_sha()
    norm_sha = _resolve_norm_stats_sha()
    if executor is None:
        status = "loading"
    else:
        try:
            executor.health_check()
            status = "ready"
        except Exception:
            status = "degraded"
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "service": POLICY_ID,
        "status": status,
        "checkpoint_sha": ckpt_sha,
        "norm_stats_sha": norm_sha,
        "supported_task_types": list(SUPPORTED_TASK_TYPES),
        "supported_action_contracts": list(SUPPORTED_ACTION_CONTRACTS),
        "uptime_ms": int((time.time() - _START_TIME) * 1000),
    }


# ===========================================================================
# HTTP 契约路由（方案书 §6 / §7 / §8）
#   POST /v1/infer   —— 推理（schemas/executor-infer.schema.json）
#   POST /v1/cancel  —— 取消（schemas/executor-cancel.schema.json）
#   GET  /health     —— 已在上方实现
# 现有 WebSocket ws://host:port/ 保留不动，以下为同一 FastAPI app 上的 HTTP 路由。
# ===========================================================================

# /v1/infer 请求 14 必填字段（schemas/executor-infer.schema.json#$defs/request）
_INFER_REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version", "request_id", "trace_id", "episode_id",
    "task_id", "subtask_id", "step_id", "observation_id",
    "deadline_ms", "executor", "checkpoint_sha", "norm_stats_sha",
    "expected_action_contract", "model_input",
)

# /v1/cancel 请求 7 必填字段（schemas/executor-cancel.schema.json#$defs/request）
_CANCEL_REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version", "request_id", "trace_id", "episode_id",
    "task_id", "subtask_id", "reason",
)


def _failure_code_value(code: Any) -> str:
    """从 FailureCode 枚举取字符串值（兼容枚举与字符串回退）。"""
    return code.value if hasattr(code, "value") else str(code)


def _make_infer_error_body(
    req: Dict[str, Any],
    *,
    code: str,
    message: str,
    retryable: bool,
    retry_after_ms: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造 /v1/infer 错误响应信封（status="error"）。

    方案书 §5 / §13：错误响应必须回显关联 ID 与实际 sha，携带 error{code,message,
    retryable[,retry_after_ms,details]}，且不得携带 action_chunk。
    """
    err: Dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if retry_after_ms is not None:
        err["retry_after_ms"] = retry_after_ms
    if details is not None:
        err["details"] = details
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "request_id": req.get("request_id", ""),
        "trace_id": req.get("trace_id", ""),
        "episode_id": req.get("episode_id", ""),
        "task_id": req.get("task_id", ""),
        "subtask_id": req.get("subtask_id", ""),
        "step_id": req.get("step_id", 0),
        "observation_id": req.get("observation_id", ""),
        "executor": POLICY_ID,
        "checkpoint_sha": _resolve_checkpoint_sha(),
        "norm_stats_sha": _resolve_norm_stats_sha(),
        "status": "error",
        "error": err,
    }


def _build_obs_from_model_input(
    model_input: Dict[str, Any], req: Dict[str, Any]
) -> Any:
    """从 model_input 构造 ObsPacket（方案书 §7.3 π0.5 model_input）。

    model_input.observation 为 canonical Observation（含 camera/robot/safety）。
    真实部署图像经共享内存 URI 传递；dummy 模式不依赖像素内容，缺图时用零图占位。
    """
    if ObsPacket is None:
        raise RuntimeError("ObsPacket 不可用（执行器未导入）")

    prompt = model_input.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("model_input.prompt 必须是字符串")

    observation = model_input.get("observation", {}) or {}
    if not isinstance(observation, dict):
        raise ValueError("model_input.observation 必须是对象")

    camera = observation.get("camera", {}) or {}
    if not isinstance(camera, dict):
        raise ValueError("observation.camera 必须是对象")
    full_image = camera.get("full_image", {}) or {}
    if not isinstance(full_image, dict):
        full_image = {}

    # 优先解析直接像素（pixels/data/list）；URI 共享内存场景由部署方解析后注入
    rgb_front: Any = None
    for key in ("pixels", "data", "rgb_front"):
        if key in full_image and full_image[key] is not None:
            try:
                rgb_front = np.array(full_image[key], dtype=np.uint8)
                break
            except Exception as e:
                raise ValueError(f"observation.camera.full_image.{key} 无法转为 uint8 数组：{e}")
    if rgb_front is None:
        # dummy 模式零图占位（Pi05Executor._infer_mock 不依赖像素内容）
        rgb_front = np.zeros((4, 4, 3), dtype=np.uint8)

    rgb_wrist: Any = None
    wrist_image = camera.get("wrist_image", {}) or {}
    if isinstance(wrist_image, dict):
        for key in ("pixels", "data", "rgb_wrist"):
            if key in wrist_image and wrist_image[key] is not None:
                try:
                    rgb_wrist = np.array(wrist_image[key], dtype=np.uint8)
                except Exception:
                    rgb_wrist = None
                break

    robot = observation.get("robot", {}) or {}
    if not isinstance(robot, dict):
        raise ValueError("observation.robot 必须是对象")
    state_src = robot.get("state", robot.get("tcp_pose_m_rad", []))
    try:
        robot_state = np.array(state_src, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"observation.robot.state 无法转为 float32 数组：{e}")

    safety = observation.get("safety", {}) or {}
    if not isinstance(safety, dict):
        safety = {}

    ts_ms = observation.get("timestamp_ms")
    if ts_ms is None or not isinstance(ts_ms, (int, float)):
        ts_ms = int(time.time() * 1000)
    timestamp_ns = int(ts_ms) * 1_000_000

    return ObsPacket(
        episode_id=str(req.get("episode_id", "")),
        step_id=int(req.get("step_id", 0)),
        timestamp_ns=timestamp_ns,
        rgb_front=rgb_front,
        rgb_wrist=rgb_wrist,
        robot_state=robot_state,
        instruction=prompt,
        runtime_flags={"safety": dict(safety)},
    )


def _canonical_chunk_to_action_chunk_dict(
    chunk: Any, task_id: str
) -> Dict[str, Any]:
    """CanonicalActionChunk → action_chunk dict（schemas/action-chunk.schema.json 9 字段）。

    CanonicalActionChunk.actions: float32[N,7] → steps[{values:[7], duration_ms}]。
    duration_ms 从 control_hz 推导（1000/control_hz，默认 100ms）。
    """
    actions = np.asarray(chunk.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"CanonicalActionChunk.actions 形状非法：{actions.shape}，期望 [N,7]"
        )
    if actions.shape[0] < 1 or actions.shape[0] > 32:
        raise ValueError(
            f"action_chunk.steps 数量超限：{actions.shape[0]}，期望 1..32"
        )

    control_hz = int(getattr(chunk, "control_hz", DEFAULT_CONTROL_HZ))
    duration_ms = max(1, int(round(1000.0 / max(1, control_hz))))

    steps: List[Dict[str, Any]] = []
    for i in range(actions.shape[0]):
        values = actions[i].tolist()
        # 夹爪原值透传（Pi05Executor 已限幅为 0/1，值域在 schema [-1,1] 内，不做额外归一化）
        steps.append({"values": values, "duration_ms": duration_ms})

    return {
        "contract_version": CONTRACT_SCHEMA_VERSION,
        "chunk_id": str(uuid4()),
        "task_id": task_id,
        "executor": POLICY_ID,
        "action_space": "ee_delta_pose_gripper",
        "frame": "robot_base",
        "translation_unit": "m",
        "rotation_unit": "rad",
        "gripper_unit": "normalized",
        "steps": steps,
    }


@app.post("/v1/infer")
async def http_infer(request: Request) -> JSONResponse:
    """POST /v1/infer —— π0.5 推理端点（方案书 §7 / schemas/executor-infer.schema.json）。

    流程：14 字段校验 → executor/sha 校验 → 构造 ObsPacket → run_in_threadpool
    推理 → CanonicalActionChunk 转 action_chunk → 构造响应信封。
    """
    t_request_start = time.time()

    # ---- 解析 JSON body ----
    try:
        req = await request.json()
    except Exception as e:
        err_body = _make_infer_error_body(
            {}, code=_failure_code_value(FailureCode.INVALID_TASK),
            message=f"请求 body 不是合法 JSON：{e}", retryable=False,
        )
        return JSONResponse(status_code=400, content=err_body)
    if not isinstance(req, dict):
        err_body = _make_infer_error_body(
            {}, code=_failure_code_value(FailureCode.INVALID_TASK),
            message="请求 body 必须是 JSON 对象", retryable=False,
        )
        return JSONResponse(status_code=400, content=err_body)

    # ---- 14 必填字段校验（缺字段 → 422）----
    missing = [f for f in _INFER_REQUIRED_FIELDS if f not in req or req[f] is None]
    if missing:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.INVALID_TASK),
            message=f"缺少必填字段：{missing}", retryable=False,
            details={"missing": missing},
        )
        return JSONResponse(status_code=422, content=err_body)

    # ---- executor 必须为 pi05（方案书 §14.2：executor name exact match）----
    if req["executor"] != POLICY_ID:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.INVALID_TASK),
            message=f"executor 不匹配：期望 {POLICY_ID!r}，收到 {req['executor']!r}",
            retryable=False,
        )
        return JSONResponse(status_code=400, content=err_body)

    # ---- expected_action_contract 必须为 1.0 ----
    if req.get("expected_action_contract") != CONTRACT_SCHEMA_VERSION:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.INVALID_TASK),
            message=f"expected_action_contract 不兼容：期望 {CONTRACT_SCHEMA_VERSION!r}，"
                    f"收到 {req.get('expected_action_contract')!r}",
            retryable=False,
        )
        return JSONResponse(status_code=400, content=err_body)

    # ---- sha 防篡改校验（方案书 §4 / §14.2：sha 必须与实际加载完全一致）----
    service_ckpt = _resolve_checkpoint_sha()
    service_norm = _resolve_norm_stats_sha()
    if req["checkpoint_sha"] != service_ckpt:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.EXECUTOR_MODEL_REVISION_MISMATCH),
            message=f"checkpoint_sha 不匹配：期望 {service_ckpt!r}，收到 {req['checkpoint_sha']!r}",
            retryable=False,
        )
        return JSONResponse(status_code=409, content=err_body)
    if req["norm_stats_sha"] != service_norm:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.EXECUTOR_MODEL_REVISION_MISMATCH),
            message=f"norm_stats_sha 不匹配：期望 {service_norm!r}，收到 {req['norm_stats_sha']!r}",
            retryable=False,
        )
        return JSONResponse(status_code=409, content=err_body)

    # ---- 执行器可用性 ----
    if executor is None:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.EXECUTOR_UNAVAILABLE),
            message="π0.5 执行器未初始化", retryable=True, retry_after_ms=500,
        )
        return JSONResponse(status_code=503, content=err_body)

    # ---- 构造 ObsPacket ----
    try:
        obs = _build_obs_from_model_input(req["model_input"], req)
    except Exception as e:
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.INVALID_TASK),
            message=f"model_input 解析失败：{e}", retryable=False,
        )
        return JSONResponse(status_code=400, content=err_body)

    # ---- 推理（同步调用放线程池，避免阻塞事件循环）----
    t_infer_start = time.time()
    try:
        if run_in_threadpool is not None:
            chunk = await run_in_threadpool(executor.infer, obs)
        else:
            loop = asyncio.get_event_loop()
            chunk = await loop.run_in_executor(None, lambda: executor.infer(obs))
    except Exception as e:
        logger.error("HTTP /v1/infer 推理异常：%s", e)
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.EXECUTOR_RUNTIME),
            message=f"推理失败：{e}", retryable=False,
        )
        return JSONResponse(status_code=500, content=err_body)
    t_infer_end = time.time()

    # ---- CanonicalActionChunk → action_chunk dict ----
    try:
        action_chunk = _canonical_chunk_to_action_chunk_dict(chunk, req["task_id"])
    except Exception as e:
        logger.error("action_chunk 转换失败：%s", e)
        err_body = _make_infer_error_body(
            req, code=_failure_code_value(FailureCode.EXECUTOR_RUNTIME),
            message=f"action_chunk 转换失败：{e}", retryable=False,
        )
        return JSONResponse(status_code=500, content=err_body)

    # ---- 构造成功响应信封（13 必填 + action_chunk + timing）----
    response: Dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "request_id": req["request_id"],
        "trace_id": req["trace_id"],
        "episode_id": req["episode_id"],
        "task_id": req["task_id"],
        "subtask_id": req["subtask_id"],
        "step_id": req["step_id"],
        "observation_id": req["observation_id"],
        "executor": POLICY_ID,
        "checkpoint_sha": service_ckpt,
        "norm_stats_sha": service_norm,
        "status": "ok",
        "action_chunk": action_chunk,
        "timing": {
            "queue_ms": round((t_infer_start - t_request_start) * 1000, 3),
            "inference_ms": round((t_infer_end - t_infer_start) * 1000, 3),
            "total_ms": round((t_infer_end - t_request_start) * 1000, 3),
        },
    }
    return JSONResponse(status_code=200, content=response)


@app.post("/v1/cancel")
async def http_cancel(request: Request) -> JSONResponse:
    """POST /v1/cancel —— 取消待执行推理（方案书 §8 / schemas/executor-cancel.schema.json）。

    实现：调用 Pi05Executor.cancel_pending_chunk() 清空队列，返回 status="cancelled"。
    幂等：重复 cancel 同一 task_id 返回 "already_completed"。
    """
    # ---- 解析 JSON body ----
    try:
        req = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "schema_version": CONTRACT_SCHEMA_VERSION,
                "request_id": "", "trace_id": "", "task_id": "",
                "status": "error",
                "error": {
                    "code": _failure_code_value(FailureCode.INVALID_TASK),
                    "message": f"请求 body 不是合法 JSON：{e}",
                    "retryable": False,
                },
                "cancelled_request_ids": [],
                "server_context_cleared": False,
            },
        )
    if not isinstance(req, dict):
        return JSONResponse(
            status_code=400,
            content={
                "schema_version": CONTRACT_SCHEMA_VERSION,
                "request_id": "", "trace_id": "", "task_id": "",
                "status": "error",
                "error": {
                    "code": _failure_code_value(FailureCode.INVALID_TASK),
                    "message": "请求 body 必须是 JSON 对象",
                    "retryable": False,
                },
                "cancelled_request_ids": [],
                "server_context_cleared": False,
            },
        )

    # ---- 7 必填字段校验（缺字段 → 422）----
    missing = [f for f in _CANCEL_REQUIRED_FIELDS if f not in req or not req[f]]
    if missing:
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": CONTRACT_SCHEMA_VERSION,
                "request_id": req.get("request_id", ""),
                "trace_id": req.get("trace_id", ""),
                "task_id": req.get("task_id", ""),
                "status": "error",
                "error": {
                    "code": _failure_code_value(FailureCode.INVALID_TASK),
                    "message": f"缺少必填字段：{missing}",
                    "retryable": False,
                    "details": {"missing": missing},
                },
                "cancelled_request_ids": [],
                "server_context_cleared": False,
            },
        )

    task_id = req["task_id"]

    # ---- 幂等取消（方案书 §8：重复 cancel 返回 already_completed）----
    async with _cancelled_tasks_lock:
        if task_id in _cancelled_tasks:
            status = "already_completed"
            cancelled_ids: List[str] = []
            context_cleared = False
        else:
            # 调用 executor.cancel_pending_chunk() 清空待执行动作队列
            if executor is not None:
                try:
                    if run_in_threadpool is not None:
                        await run_in_threadpool(executor.cancel_pending_chunk)
                    else:
                        executor.cancel_pending_chunk()
                except Exception as e:
                    logger.warning("cancel_pending_chunk 异常：%s", e)
            _cancelled_tasks.add(task_id)
            status = "cancelled"
            # 服务端未跟踪 infer request_id（无状态），无法列出被取消的具体 request_id
            cancelled_ids = []
            context_cleared = True

    response: Dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "request_id": req["request_id"],
        "trace_id": req["trace_id"],
        "task_id": task_id,
        "status": status,
        "cancelled_request_ids": cancelled_ids,
        "server_context_cleared": context_cleared,
    }
    return JSONResponse(status_code=200, content=response)


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


def _check_episode_step(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """episode_id / step_id 连续性与动作块过期检查（方案书 §3.4 动作过期）。

    返回值：错误 dict 表示应拒绝请求；None 表示通过。
    pending_chunks 的并发读写由 ws_infer 中的 _pending_chunks_lock 保护。
    """
    global current_episode_id, last_step_id
    ep = data["episode_id"]
    sid = data["step_id"]

    # episode 切换：标记（实际清理由 ws_infer 在异步上下文中加锁完成）
    if ep != current_episode_id:
        logger.info("Episode changed, cleared pending chunks (prev=%s new=%s)",
                    current_episode_id, ep)
        current_episode_id = ep
        last_step_id = None

    # episode 内 step_id 应递增（方案书 §7.2：防止超时返回的旧动作进入新 episode）
    if last_step_id is not None and sid != last_step_id + 1:
        logger.warning("step_id 不连续：期望 %d，收到 %d", last_step_id + 1, sid)
    # 同一 episode 内 step_id 应递增；非递增拒绝执行（方案书 §3.4：step_id 不匹配必须丢弃）
    if last_step_id is not None and sid <= last_step_id:
        return {"error": "stale_step_id",
                "reason": f"step_id 未递增: last={last_step_id}, curr={sid}",
                "schema_version": SCHEMA_VERSION}
    last_step_id = sid

    return None


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

    # 本连接内的 episode 追踪（用于在异步上下文中触发 episode 切换清理）
    _ws_prev_episode_id: Optional[str] = None

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

            # episode/step 连续性与动作过期检查（方案书 §3.4）
            # ---- 加锁保护 global current_episode_id / last_step_id / pending_chunks ----
            # §7.1：多 episode 并发安全，所有全局状态读写必须在锁内完成。
            async with _pending_chunks_lock:
                step_err = _check_episode_step(data)
                if step_err is not None:
                    await _send_error(ws, step_err)
                    continue

                # episode 切换时清空 pending_chunks（方案书 §3.3.1 Para186）
                ep = data["episode_id"]
                if ep != _ws_prev_episode_id:
                    pending_chunks.clear()
                    _ws_prev_episode_id = ep
                # 检查当前 episode 的 pending chunk 是否过期，过期则丢弃
                expired_chunk = pending_chunks.get(ep)
                if expired_chunk is not None:
                    age_ms = (time.time() - expired_chunk["timestamp"]) * 1000.0
                    ttl = expired_chunk.get("expires_after_ms", DEFAULT_EXPIRES_AFTER_MS)
                    if age_ms > ttl:
                        logger.warning("Action chunk expired (episode=%s age_ms=%.0f > %dms)",
                                       ep, age_ms, ttl)
                        pending_chunks.pop(ep, None)

                # episode 切换时重置执行器（reset() 内部已调用 cancel_pending_chunk()）
                if ep != current_episode_id:
                    if executor is not None:
                        try:
                            executor.reset()
                        except Exception as e:
                            logger.warning("执行器 reset 异常：%s", e)

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

            # 记录 pending chunk，用于后续请求的过期判断（加锁保护并发写入）
            async with _pending_chunks_lock:
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
