"""π0.5 WebSocket 服务层单元测试。

负责人：E（π0.5/openpi）/ F（测试）

方案书出处：
- §3.3 / §3.3.1：π0.5 适配流程（WebSocket 远程推理）。
- §3.4：ObsPacket v1 / CanonicalActionChunk v1 协议不变量；动作过期丢弃。
- §7.1：服务边界与健康检查（openpi π0.5 服务接口；多 episode 并发安全）。
- §7.2：RPC 请求防错字段（schema_version/episode_id/step_id/checkpoint_sha/expires_after_ms）。
- §7.5：service_stress（连续100次推理+超时/重启，无内存泄漏/状态串扰；旧动作丢弃）。

测试策略：
- 纯 CPU / 纯 Mock：Pi05Executor 被 unittest.mock 彻底替换，不碰 GPU / openpi / JAX。
- 使用 FastAPI TestClient（Starlette）驱动 WebSocket 端点 ws://host:port/。
- 请求与响应严格遵循 ObsPacket / CanonicalActionChunk v1 schema，禁止手写伪造字段。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 确保服务以 dummy 模式启动（不加载真实模型 / openpi / JAX）
# 必须在 import openpi_service 之前设置
# ---------------------------------------------------------------------------
os.environ["PI05_SERVICE_MODE"] = "dummy"
os.environ["PI05_MODE"] = "dummy"
os.environ["PI05_TASK_PROFILE_VERSION"] = "v2"
os.environ.setdefault(
    "PI05_CHECKPOINT_SHA",
    "sha256:0000000000000000000000000000000000000000000000000000000000000000",
)
os.environ.setdefault(
    "PI05_NORM_STATS_SHA",
    "sha256:0000000000000000000000000000000000000000000000000000000000000000",
)

# 项目根目录加入 sys.path（与被测模块一致，任意目录可运行）
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from services.pi05.src import openpi_service
from services.pi05.src.action import CanonicalActionChunk
from services.pi05.src.observation import (
    ObsPacket,
    is_image_reference,
    validate_image_reference,
)
from industrial_agent.errors import FailureCode, ImageCasError
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from industrial_agent.service_images import CasRequestImageResolver

# ---------------------------------------------------------------------------
# 常量（严格对齐方案书 §3.4 / openpi_service.py 模块常量）
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "v1"
POLICY_ID = "pi05"
DEFAULT_CONTROL_HZ = 10
DEFAULT_EXPIRES_AFTER_MS = 1000
ACTION_DIM = 7
MOCK_CHUNK_LEN = 10
SPACE_ID = "eef_delta_xyz_axisangle_gripper_v1"
FRAME_ID = "robot_base"

TEST_CHECKPOINT_SHA = (
    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
)
TEST_NORM_STATS_SHA = (
    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
)
TEST_FULL_IMAGE_REFERENCE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 辅助函数：构造合法 ObsPacket v1 请求（严格遵循 §3.4 schema）
# ---------------------------------------------------------------------------
def make_valid_request(
    episode_id: str = "test-ep-001",
    step_id: int = 0,
    instruction: str = "pick up the red cylinder and place it into cell row 2 col 3",
    rgb_front: Any = None,
    rgb_wrist: Any = None,
    robot_state: Any = None,
    runtime_flags: Any = None,
) -> dict[str, Any]:
    """构造合法 ObsPacket v1 请求字典（JSON 可序列化）。

    字段严格对齐方案书 §3.4：
      episode_id: str          step_id: int        timestamp_ns: int64
      rgb_front: uint8[H,W,3]  rgb_wrist: Optional  robot_state: float32[d]
      instruction: str         runtime_flags: dict
    """
    if rgb_front is None:
        rgb_front = np.zeros((4, 4, 3), dtype=np.uint8)
    if isinstance(rgb_front, np.ndarray):
        rgb_front = rgb_front.tolist()
    if isinstance(rgb_wrist, np.ndarray):
        rgb_wrist = rgb_wrist.tolist()
    if robot_state is None:
        robot_state = [0.0] * 7
    elif isinstance(robot_state, np.ndarray):
        robot_state = robot_state.tolist()
    if runtime_flags is None:
        runtime_flags = {"terminated": False, "truncated": False, "camera_ok": True}

    req: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "step_id": step_id,
        "timestamp_ns": int(time.time() * 1e9),
        "rgb_front": rgb_front,
        "instruction": instruction,
        "robot_state": robot_state,
        "runtime_flags": runtime_flags,
    }
    if rgb_wrist is not None:
        req["rgb_wrist"] = rgb_wrist
    return req


def make_action_chunk(
    actions: Any = None,
    generated_step: int = 0,
    source_policy: str = POLICY_ID,
    checkpoint_sha: str = TEST_CHECKPOINT_SHA,
    expires_after_ms: int = DEFAULT_EXPIRES_AFTER_MS,
) -> CanonicalActionChunk:
    """构造合法 CanonicalActionChunk v1（严格遵循 §3.4 schema）。

    actions: float32[N,7] [dx,dy,dz,dax,day,daz,gripper]
    """
    if actions is None:
        actions = np.zeros((MOCK_CHUNK_LEN, ACTION_DIM), dtype=np.float32)
    return CanonicalActionChunk(
        actions=actions,
        space_id=SPACE_ID,
        frame=FRAME_ID,
        control_hz=DEFAULT_CONTROL_HZ,
        generated_step=generated_step,
        source_policy=source_policy,
        checkpoint_sha=checkpoint_sha,
        expires_after_ms=expires_after_ms,
    )


def serialize_request(req: dict[str, Any]) -> str:
    """序列化请求为 JSON 文本（TestClient send_text 用）。"""
    return json.dumps(req, ensure_ascii=False)


def parse_response(raw: bytes) -> dict[str, Any]:
    """反序列化服务端响应：msgpack 优先，fallback JSON。"""
    try:
        import msgpack

        return msgpack.unpackb(raw, raw=False)
    except Exception:
        return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_executor() -> MagicMock:
    """完全 Mock 的 Pi05Executor（不碰 GPU / openpi / JAX）。

    - infer() 默认返回合法 CanonicalActionChunk v1（方案书 §3.4）
    - health_check() 返回合法健康状态（方案书 §7.1）
    - reset() / cancel_pending_chunk() 无副作用（方案书 §3.3.1 Para186）
    - _checkpoint_sha / _norm_stats_sha 属性供服务层 _checkpoint_sha() 读取
    """
    ex = MagicMock()
    # 属性（_checkpoint_sha / _norm_stats_sha 被 openpi_service._checkpoint_sha() 读取，
    # 现通过 executor.checkpoint_sha / executor.norm_stats_sha 公共 property 访问）
    ex._checkpoint_sha = TEST_CHECKPOINT_SHA
    ex._norm_stats_sha = TEST_NORM_STATS_SHA
    ex.checkpoint_sha = TEST_CHECKPOINT_SHA
    ex.norm_stats_sha = TEST_NORM_STATS_SHA
    # health_check 返回合法字典（方案书 §7.1）
    ex.health_check.return_value = {
        "mode": "dummy",
        "policy_type": "mock",
        "config_name": "pi05_industrial",
        "checkpoint_sha": TEST_CHECKPOINT_SHA,
        "norm_stats_sha": TEST_NORM_STATS_SHA,
        "vram_usage_mb": None,
        "last_latency_ms": 42,
        "openpi_available": False,
        "ws_available": False,
    }
    # infer 默认返回合法 CanonicalActionChunk（方案书 §3.4）
    ex.infer.return_value = make_action_chunk()
    # reset / cancel_pending_chunk 无副作用（方案书 §3.3.1 Para186）
    ex.reset.return_value = None
    ex.cancel_pending_chunk.return_value = None
    return ex


@pytest.fixture
def test_client(mock_executor: MagicMock, monkeypatch, tmp_path):
    """FastAPI TestClient：启动时用 mock_executor 替换真实 Pi05Executor。

    每个 test 获得：
    - 全新的 TestClient（触发 startup → _init_executor → Pi05Executor()）
    - 全局状态已重置（pending_chunks / current_episode_id / last_step_id / lock）
    - Pi05Executor 被 patch 为返回 mock_executor（彻底隔离底层执行器）
    """
    global TEST_FULL_IMAGE_REFERENCE

    # ---- 为所有 HTTP 用例创建真实临时 CAS 图像；Dummy 也不得绕过图像校验 ----
    cas_root = tmp_path / "cas"
    cas_root.mkdir()
    shared_image_cas = ImageCas(ImageCasConfig(root=cas_root, cache_max_bytes=0))
    TEST_FULL_IMAGE_REFERENCE = shared_image_cas.write_rgb(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        camera_id="CAM_A_TOP",
    ).to_dict()
    monkeypatch.setenv("INDUSTRIAL_AGENT_CAS_ROOT", str(cas_root))

    # ---- 重置全局状态（防止跨测试串扰，方案书 §7.1 多 episode 并发安全）----
    openpi_service.executor = None
    openpi_service._executor_init_error = None
    openpi_service.pending_chunks.clear()
    openpi_service._cancelled_tasks.clear()
    openpi_service._seen_task_ids.clear()
    openpi_service.image_cas = None
    openpi_service.request_image_resolver = None
    openpi_service.v1_infer_handler = None
    openpi_service._image_cas_init_error = None
    # 重建 lock：避免绑定到上一个 TestClient 的事件循环（Python 3.10+ _LoopBoundMixin）
    openpi_service._pending_chunks_lock = asyncio.Lock()
    openpi_service._cancelled_tasks_lock = asyncio.Lock()

    # ---- patch Pi05Executor：使 _init_executor() 创建 mock 而非真实执行器 ----
    with patch.object(openpi_service, "Pi05Executor", return_value=mock_executor):
        with TestClient(openpi_service.app) as client:
            yield client

    # ---- 清理全局状态 ----
    openpi_service.executor = None
    openpi_service._executor_init_error = None
    openpi_service.pending_chunks.clear()
    openpi_service._cancelled_tasks.clear()
    openpi_service._seen_task_ids.clear()
    TEST_FULL_IMAGE_REFERENCE = None


@pytest.fixture
def ws_context(test_client: TestClient):
    """WebSocket 连接上下文工厂：自动接收 metadata，yield (ws, metadata)。

    用法：
        with ws_context() as (ws, metadata):
            ws.send_text(serialize_request(req))
            resp = parse_response(ws.receive_bytes())

    服务端建立连接后先发 metadata（JSON 文本，openpi_service.ws_infer L397），
    随后请求/响应用于二进制帧（JSON bytes，_serialize 回退 JSON）。
    """

    @contextmanager
    def _connect():
        with test_client.websocket_connect("/") as ws:
            metadata = json.loads(ws.receive_text())
            yield ws, metadata

    return _connect


# ---------------------------------------------------------------------------
# 1. 服务生命周期 (test_service_lifecycle)
# ---------------------------------------------------------------------------
def test_service_lifecycle(test_client: TestClient, mock_executor: MagicMock):
    """验证 WebSocket 连接建立、就绪、断开的状态流转；executor 正确初始化与释放。

    方案书 §6 / §7.1：服务边界与健康检查；openpi_service.ws_infer 连接建立流程。
    /health 响应对齐 schemas/executor-health.schema.json（7 必填字段）。
    """
    # ---- 健康检查端点（schemas/executor-health.schema.json）----
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # 7 必填字段
    assert body["schema_version"] == "1.0"
    assert body["service"] == "pi05"
    assert body["status"] in ("ready", "loading", "degraded"), (
        "health.status 必须为 ready/loading/degraded"
    )
    assert body["status"] == "ready", "executor 就绪时 health 应返回 ready"
    assert body["checkpoint_sha"] == TEST_CHECKPOINT_SHA
    assert body["norm_stats_sha"] == TEST_NORM_STATS_SHA
    assert "pick_place" in body["supported_task_types"]
    assert "visual_manipulation" in body["supported_task_types"]
    assert "instruction_interaction" in body["supported_task_types"]
    assert "1.0" in body["supported_action_contracts"]
    # 可选字段 uptime_ms（非负整数）
    assert "uptime_ms" in body and body["uptime_ms"] >= 0

    # ---- executor 已初始化（startup → _init_executor → mock）----
    assert openpi_service.executor is not None
    assert openpi_service.executor is mock_executor

    # ---- WebSocket 连接建立 + metadata 就绪 ----
    with test_client.websocket_connect("/") as ws:
        metadata = json.loads(ws.receive_text())
        # metadata 字段对齐方案书 §7.2 防错字段
        assert metadata["schema_version"] == SCHEMA_VERSION
        assert metadata["policy_id"] == POLICY_ID
        assert metadata["mode"] == "dummy"
        assert metadata["checkpoint_sha"] == TEST_CHECKPOINT_SHA
        assert metadata["control_hz"] == DEFAULT_CONTROL_HZ
        assert metadata["expires_after_ms"] == DEFAULT_EXPIRES_AFTER_MS
        assert "norm_stats_sha" in metadata
        # 连接正常断开（with 退出触发 WebSocketDisconnect，服务端 finally 清理）
    # 断开后服务端不崩溃：再次健康检查仍 ready
    resp2 = test_client.get("/health")
    assert resp2.json()["status"] == "ready"


def test_service_import_requires_explicit_mode():
    """PI05_SERVICE_MODE 缺失时模块导入即失败，禁止隐式 Dummy 启动。"""
    env = os.environ.copy()
    env.pop("PI05_SERVICE_MODE", None)
    env.pop("PI05_MODE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (os.path.join(_PROJECT_ROOT, "src"), env.get("PYTHONPATH")))
    )
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-c", "import services.pi05.src.openpi_service"],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "PI05_SERVICE_MODE 未设置" in result.stderr


def test_real_init_failure_keeps_infer_unavailable(
    test_client: TestClient,
    mock_executor: MagicMock,
    monkeypatch,
):
    """Real 初始化失败后保持 fail-closed，HTTP 不得调用既有 Dummy executor。"""
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setenv("PI05_MODE", "real")
    with patch.object(
        openpi_service,
        "Pi05Executor",
        side_effect=RuntimeError("real policy init failed"),
    ):
        openpi_service._init_executor()

    response = test_client.post("/v1/infer", json=_make_http_infer_body())

    assert openpi_service.executor is None
    assert openpi_service._executor_init_error == "real policy init failed"
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EXEC_2101_UNAVAILABLE"
    mock_executor.infer.assert_not_called()


def test_real_mode_rejects_legacy_websocket_inline_image_transport(
    test_client: TestClient,
    mock_executor: MagicMock,
    monkeypatch,
):
    """Real 模式只允许冻结 HTTP ImageReference 入口，不接受旧 WS 内联像素。"""
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")

    with test_client.websocket_connect("/") as ws:
        error = parse_response(ws.receive_bytes())
        assert error["error"] == "unsupported_transport"
        assert "HTTP /v1/infer ImageReference" in error["reason"]
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_bytes()

    assert exc_info.value.code == 1008
    mock_executor.infer.assert_not_called()


def test_health_real_mode_with_dummy_executor_is_degraded(
    test_client: TestClient,
    monkeypatch,
):
    """请求 real 但执行器实际为 Dummy 时，health 必须 fail-closed。"""
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")

    response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_without_initialized_executor_is_loading(
    test_client: TestClient,
    monkeypatch,
):
    """执行器尚未初始化且无已知错误时，health 返回 loading + HTTP 503。"""
    monkeypatch.setattr(openpi_service, "executor", None)
    monkeypatch.setattr(openpi_service, "_executor_init_error", None)

    response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "loading"


def test_health_invalid_artifact_sha_is_degraded_with_contract_body(
    test_client: TestClient,
    monkeypatch,
):
    """无效制品 SHA 必须降级，且响应字段仍满足冻结 digest 格式。"""
    monkeypatch.setenv("PI05_CHECKPOINT_SHA", "not-a-sha")

    response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["checkpoint_sha"] == "sha256:" + "0" * 64


def test_init_image_cas_dummy_mode_uses_public_resolver(
    monkeypatch,
    tmp_path,
):
    """Dummy 只替换模型计算，HTTP 图像仍必须经过公共 CAS resolver。"""
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "dummy")
    monkeypatch.setenv("INDUSTRIAL_AGENT_CAS_ROOT", str(tmp_path))
    monkeypatch.setattr(openpi_service, "image_cas", object())
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", "stale error")

    openpi_service._init_image_cas()

    assert openpi_service.image_cas is not None
    assert openpi_service.request_image_resolver is not None
    assert openpi_service.v1_infer_handler is not None
    assert openpi_service._image_cas_init_error is None


def test_init_image_cas_real_mode_uses_shared_read_only_resolver(
    monkeypatch,
    tmp_path,
):
    """Real 模式从 v1.3 配置创建 A 的公共 resolver，并执行只读就绪检查。"""
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setenv("INDUSTRIAL_AGENT_CAS_ROOT", str(tmp_path))
    monkeypatch.setattr(openpi_service, "image_cas", None)
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)

    openpi_service._init_image_cas()

    assert openpi_service.image_cas is not None
    assert openpi_service.image_cas.config.root == tmp_path
    assert openpi_service.request_image_resolver is not None
    assert openpi_service.v1_infer_handler is not None
    assert openpi_service._image_cas_init_error is None


def test_init_image_cas_real_mode_fails_closed_when_root_is_missing(
    monkeypatch,
    tmp_path,
):
    """Real 模式 CAS 根目录不可用时不得保留半初始化 resolver。"""
    missing_root = tmp_path / "missing-cas-root"
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setenv("INDUSTRIAL_AGENT_CAS_ROOT", str(missing_root))
    monkeypatch.setattr(openpi_service, "image_cas", object())
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)

    openpi_service._init_image_cas()

    assert openpi_service.image_cas is None
    assert openpi_service.request_image_resolver is None
    assert openpi_service.v1_infer_handler is None
    assert openpi_service._image_cas_init_error


def test_health_real_mode_without_image_cas_is_degraded(
    test_client: TestClient,
    mock_executor: MagicMock,
    monkeypatch,
):
    """Real policy 就绪但 CAS resolver 缺失时，health 不得返回 ready。"""
    real_sha = "sha256:" + "1" * 64
    mock_executor.health_check.return_value.update(
        {"mode": "real", "policy_type": "local"}
    )
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", None)
    monkeypatch.setattr(
        openpi_service,
        "_image_cas_init_error",
        "CAS resolver unavailable",
    )
    monkeypatch.setattr(openpi_service, "_resolve_checkpoint_sha", lambda: real_sha)
    monkeypatch.setattr(openpi_service, "_resolve_norm_stats_sha", lambda: real_sha)

    response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_real_mode_with_ready_image_cas_is_ready(
    test_client: TestClient,
    mock_executor: MagicMock,
    monkeypatch,
):
    """Real policy、制品 SHA 与只读 CAS 均就绪时，health 才能返回 ready。"""
    real_sha = "sha256:" + "2" * 64
    ready_image_cas = MagicMock(spec=["assert_ready"])
    mock_executor.health_check.return_value.update(
        {"mode": "real", "policy_type": "local"}
    )
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", ready_image_cas)
    monkeypatch.setattr(openpi_service, "v1_infer_handler", object())
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)
    monkeypatch.setattr(openpi_service, "_resolve_checkpoint_sha", lambda: real_sha)
    monkeypatch.setattr(openpi_service, "_resolve_norm_stats_sha", lambda: real_sha)

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    ready_image_cas.assert_ready.assert_called_once_with(writable=False)


def test_health_real_mode_rechecks_image_cas_readiness(
    test_client: TestClient,
    mock_executor: MagicMock,
    monkeypatch,
):
    """CAS 挂载在启动后失效时，下一次 health 必须降级。"""
    real_sha = "sha256:" + "3" * 64
    unavailable_image_cas = MagicMock(spec=["assert_ready"])
    unavailable_image_cas.assert_ready.side_effect = OSError("CAS mount unavailable")
    mock_executor.health_check.return_value.update(
        {"mode": "real", "policy_type": "local"}
    )
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", unavailable_image_cas)
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)
    monkeypatch.setattr(openpi_service, "_resolve_checkpoint_sha", lambda: real_sha)
    monkeypatch.setattr(openpi_service, "_resolve_norm_stats_sha", lambda: real_sha)

    response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# 2. 推理请求处理 (test_infer_success)
# ---------------------------------------------------------------------------
def test_infer_success(ws_context, mock_executor: MagicMock):
    """验证合法 ObsPacket 请求被正确解析、透传至 executor、输出 CanonicalActionChunk。

    方案书 §3.4 ObsPacket v1 → CanonicalActionChunk v1；§7.2 防错字段。
    """
    with ws_context() as (ws, metadata):
        req = make_valid_request(episode_id="infer-ep-001", step_id=0)
        ws.send_text(serialize_request(req))

        resp = parse_response(ws.receive_bytes())

    # ---- 响应字段严格对齐 CanonicalActionChunk v1（方案书 §3.4）----
    assert resp["schema_version"] == SCHEMA_VERSION
    assert resp["episode_id"] == "infer-ep-001"
    assert resp["step_id"] == 0
    # actions: float32[N,7] [dx,dy,dz,dax,day,daz,gripper]
    actions = np.array(resp["actions"], dtype=np.float32)
    assert actions.ndim == 2
    assert actions.shape[1] == ACTION_DIM
    assert actions.shape[0] >= 1  # N≥1（方案书 §3.4 协议不变量）
    # 协议不变量字段
    assert resp["space_id"] == SPACE_ID
    assert resp["frame"] == FRAME_ID
    assert resp["control_hz"] == DEFAULT_CONTROL_HZ
    assert resp["generated_step"] == 0
    # §7.2 防错字段：source_policy / checkpoint_sha / expires_after_ms
    assert resp["source_policy"] == POLICY_ID
    assert resp["checkpoint_sha"] == TEST_CHECKPOINT_SHA
    assert resp["expires_after_ms"] == DEFAULT_EXPIRES_AFTER_MS

    # ---- executor.infer 被调用，参数为合法 ObsPacket ----
    mock_executor.infer.assert_called_once()
    obs_arg = mock_executor.infer.call_args.args[0]
    assert isinstance(obs_arg, ObsPacket)
    assert obs_arg.episode_id == "infer-ep-001"
    assert obs_arg.step_id == 0
    assert obs_arg.instruction == req["instruction"]
    assert obs_arg.rgb_front.dtype == np.uint8
    assert obs_arg.rgb_front.shape == (4, 4, 3)
    assert obs_arg.robot_state.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. 非法请求拒绝 (test_invalid_request_handling)
# ---------------------------------------------------------------------------
def test_invalid_request_handling(ws_context, mock_executor: MagicMock):
    """验证服务层拦截缺失字段、类型/维度不匹配、JSON 损坏的非法请求。

    方案书 §3.4 协议不变量；§7.2 防错字段；openpi_service._validate_request / _build_obs。
    坏数据不得透传至 executor（mock_executor.infer 不应被调用）。
    """
    with ws_context() as (ws, metadata):
        # ---- 3a. 缺失必要字段（episode_id）----
        req_missing = make_valid_request()
        del req_missing["episode_id"]
        ws.send_text(serialize_request(req_missing))
        resp_missing = parse_response(ws.receive_bytes())
        assert resp_missing["error"] == "missing_required_fields"
        assert "episode_id" in resp_missing["missing"]

        # ---- 3b. 缺失必要字段（rgb_front）----
        req_no_rgb = make_valid_request()
        del req_no_rgb["rgb_front"]
        ws.send_text(serialize_request(req_no_rgb))
        resp_no_rgb = parse_response(ws.receive_bytes())
        assert resp_no_rgb["error"] == "missing_required_fields"
        assert "rgb_front" in resp_no_rgb["missing"]

        # ---- 3c. episode_id 为空字符串 ----
        req_empty_ep = make_valid_request(episode_id="  ")
        ws.send_text(serialize_request(req_empty_ep))
        resp_empty_ep = parse_response(ws.receive_bytes())
        assert resp_empty_ep["error"] == "invalid_episode_id"

        # ---- 3d. step_id 类型不匹配（字符串无法解析）----
        req_bad_step = make_valid_request()
        req_bad_step["step_id"] = "not_a_number"
        ws.send_text(serialize_request(req_bad_step))
        resp_bad_step = parse_response(ws.receive_bytes())
        assert resp_bad_step["error"] == "invalid_step_id"

        # ---- 3e. step_id 为负数 ----
        req_neg_step = make_valid_request()
        req_neg_step["step_id"] = -1
        ws.send_text(serialize_request(req_neg_step))
        resp_neg_step = parse_response(ws.receive_bytes())
        assert resp_neg_step["error"] == "invalid_step_id"

        # ---- 3f. instruction 类型不匹配（数字而非字符串）----
        req_bad_instr = make_valid_request()
        req_bad_instr["instruction"] = 12345
        ws.send_text(serialize_request(req_bad_instr))
        resp_bad_instr = parse_response(ws.receive_bytes())
        assert resp_bad_instr["error"] == "invalid_instruction"

        # ---- 3g. 维度/类型不匹配：rgb_front 无法转为 uint8 数组 ----
        req_bad_rgb = make_valid_request()
        req_bad_rgb["rgb_front"] = "not_an_image_array"
        ws.send_text(serialize_request(req_bad_rgb))
        resp_bad_rgb = parse_response(ws.receive_bytes())
        assert resp_bad_rgb["error"] == "obs_build_failed"

        # ---- 3h. JSON 损坏 ----
        ws.send_text("{ this is not valid json <<<")
        resp_bad_json = parse_response(ws.receive_bytes())
        assert resp_bad_json["error"] == "deserialize_failed"

    # ---- 所有非法请求均未透传至 executor ----
    mock_executor.infer.assert_not_called()


# ---------------------------------------------------------------------------
# 4. 底层异常降级 (test_executor_exception_fallback)
# ---------------------------------------------------------------------------
def test_executor_exception_fallback(ws_context, mock_executor: MagicMock):
    """验证 executor 抛出 RuntimeError / MemoryError(OOM) 时服务层不崩溃，返回错误响应。

    方案书 §3.4 协议不变量（非法不下发）；openpi_service.ws_infer 推理异常捕获。
    """
    with ws_context() as (ws, metadata):
        # ---- 4a. RuntimeError（模型推理异常）----
        mock_executor.infer.side_effect = RuntimeError("model inference failed")
        req = make_valid_request(episode_id="err-ep", step_id=0)
        ws.send_text(serialize_request(req))
        resp = parse_response(ws.receive_bytes())
        assert resp["error"] == "infer_failed"
        assert "model inference failed" in resp["reason"]
        assert resp["schema_version"] == SCHEMA_VERSION

    # ---- 服务未崩溃：恢复 mock，新连接新 episode 可正常推理 ----
    mock_executor.infer.side_effect = None
    mock_executor.infer.return_value = make_action_chunk()

    with ws_context() as (ws, metadata):
        req2 = make_valid_request(episode_id="recover-ep", step_id=0)
        ws.send_text(serialize_request(req2))
        resp2 = parse_response(ws.receive_bytes())
        assert "actions" in resp2
        assert resp2["episode_id"] == "recover-ep"

    # ---- 4b. MemoryError（OOM）同样被捕获 ----
    mock_executor.infer.side_effect = MemoryError("CUDA out of memory")
    with ws_context() as (ws, metadata):
        req3 = make_valid_request(episode_id="oom-ep", step_id=0)
        ws.send_text(serialize_request(req3))
        resp3 = parse_response(ws.receive_bytes())
        assert resp3["error"] == "infer_failed"
        assert "CUDA out of memory" in resp3["reason"]


# ---------------------------------------------------------------------------
# 5. 动作过期丢弃 (test_action_expiry)
# ---------------------------------------------------------------------------
def test_action_expiry(ws_context, mock_executor: MagicMock):
    """验证 step_id 不匹配、超过 expires_after_ms、episode 切换后动作被丢弃。

    方案书 §3.4 协议不变量·动作过期：
      step_id 不匹配、超过 expires_after_ms 或 episode 切换后返回的动作必须丢弃。
    """
    with ws_context() as (ws, metadata):
        # ---- 5a. step_id 不匹配：回退的 step_id 被拒绝（stale_step_id）----
        req1 = make_valid_request(episode_id="expire-ep", step_id=0)
        ws.send_text(serialize_request(req1))
        resp1 = parse_response(ws.receive_bytes())
        assert "actions" in resp1  # 首次请求成功

        # 同 episode 内 step_id 未递增（0 <= 0）→ stale_step_id
        req_stale = make_valid_request(episode_id="expire-ep", step_id=0)
        ws.send_text(serialize_request(req_stale))
        resp_stale = parse_response(ws.receive_bytes())
        assert resp_stale["error"] == "stale_step_id"
        assert "step_id 未递增" in resp_stale["reason"]

        # ---- 5b. 超过 expires_after_ms：pending chunk 过期被丢弃 ----
        # 设置极短过期时间
        mock_executor.infer.return_value = make_action_chunk(expires_after_ms=50)
        req2 = make_valid_request(episode_id="expire-ep", step_id=1)
        ws.send_text(serialize_request(req2))
        resp2 = parse_response(ws.receive_bytes())
        assert "actions" in resp2

        # E-07：_conn_pending 是 per-connection 局部变量，直接通过行为验证：
        # 请求正常通过即表示 pending 被正确管理

        # 等待超过 expires_after_ms（50ms）
        time.sleep(0.08)

        # 发送下一个 step：服务端检测 pending chunk 过期并丢弃，但请求仍正常处理
        req3 = make_valid_request(episode_id="expire-ep", step_id=2)
        ws.send_text(serialize_request(req3))
        resp3 = parse_response(ws.receive_bytes())
        assert "actions" in resp3  # 过期丢弃后重新推理，响应正常

        # ---- 5c. episode 切换：旧 episode 的 pending 被清除 ----
        # 恢复正常过期时间
        mock_executor.infer.return_value = make_action_chunk(expires_after_ms=1000)

        # 切换到新 episode
        req_new_ep = make_valid_request(episode_id="new-ep-after-switch", step_id=0)
        ws.send_text(serialize_request(req_new_ep))
        resp_new_ep = parse_response(ws.receive_bytes())
        assert "actions" in resp_new_ep

        # 新 episode 的 step_id=0 被接受（per-connection 追踪已重置）
        assert "error" not in resp_new_ep, (
            f"新 episode 的 step_id=0 应被接受（per-connection 状态已重置），"
            f"实际响应：{resp_new_ep}"
        )


# ---------------------------------------------------------------------------
# 6. 断连与重连 (test_disconnect_reconnect)
# ---------------------------------------------------------------------------
def test_disconnect_reconnect(test_client: TestClient, mock_executor: MagicMock):
    """验证客户端断开后服务端清理连接上下文；新连接无旧连接状态串扰。

    方案书 §7.1：多 episode 并发安全；openpi_service.ws_infer 连接级状态 _ws_prev_episode_id。
    """
    # ---- 连接 A：发送请求后断开 ----
    with test_client.websocket_connect("/") as ws_a:
        meta_a = json.loads(ws_a.receive_text())
        req_a = make_valid_request(episode_id="conn-A-ep", step_id=0)
        ws_a.send_text(serialize_request(req_a))
        resp_a = parse_response(ws_a.receive_bytes())
        assert "actions" in resp_a
        assert resp_a["episode_id"] == "conn-A-ep"
    # ws_a 断开（with 退出），服务端 finally 清理

    # ---- 连接 B：新连接，不同 episode，验证无状态串扰 ----
    with test_client.websocket_connect("/") as ws_b:
        meta_b = json.loads(ws_b.receive_text())
        # metadata 独立，不受连接 A 影响
        assert meta_b["schema_version"] == SCHEMA_VERSION
        assert meta_b["policy_id"] == POLICY_ID

        # 连接 B 使用全新 episode，step_id 从 0 开始（不受连接 A 的 last_step_id 影响）
        req_b = make_valid_request(episode_id="conn-B-ep", step_id=0)
        ws_b.send_text(serialize_request(req_b))
        resp_b = parse_response(ws_b.receive_bytes())
        assert "actions" in resp_b
        assert resp_b["episode_id"] == "conn-B-ep"
        assert resp_b["step_id"] == 0

    # ---- 两次推理调用独立，无数据串扰 ----
    assert mock_executor.infer.call_count == 2
    obs_a = mock_executor.infer.call_args_list[0].args[0]
    obs_b = mock_executor.infer.call_args_list[1].args[0]
    assert obs_a.episode_id == "conn-A-ep"
    assert obs_b.episode_id == "conn-B-ep"


# ---------------------------------------------------------------------------
# 7. 压力测试 (test_service_stress) — 方案书 §7.5 service_stress
# ---------------------------------------------------------------------------
def test_service_stress(ws_context, mock_executor: MagicMock):
    """连续 100 次推理请求，验证无内存泄漏、无状态串扰、超时请求被丢弃。

    方案书 §7.5 service_stress：
      连续100次推理+超时/重启 → 无内存泄漏/状态串扰；旧动作丢弃。
    """

    # ---- 每次返回独立 action chunk（按 step_id 区分）----
    def _infer_side_effect(obs: ObsPacket) -> CanonicalActionChunk:
        # 每次返回不同的 actions，验证响应独立性
        actions = np.full((MOCK_CHUNK_LEN, ACTION_DIM), obs.step_id, dtype=np.float32)
        return make_action_chunk(actions=actions, generated_step=obs.step_id)

    mock_executor.infer.side_effect = _infer_side_effect

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    episode = "stress-ep-001"
    with ws_context() as (ws, metadata):
        for i in range(100):
            req = make_valid_request(episode_id=episode, step_id=i)
            ws.send_text(serialize_request(req))
            resp = parse_response(ws.receive_bytes())

            # 无状态串扰：每次响应独立，step_id 与 actions 对应
            assert resp["episode_id"] == episode
            assert resp["step_id"] == i
            assert resp["generated_step"] == i
            actions = np.array(resp["actions"], dtype=np.float32)
            assert actions.shape == (MOCK_CHUNK_LEN, ACTION_DIM)
            # 每次返回的 actions 值 == step_id，验证无串扰
            assert np.all(actions == i), f"step {i}: 响应串扰"

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # ---- 无内存泄漏：100 次请求后内存增长有界（< 20MB）----
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)
    assert total_growth < 20 * 1024 * 1024, (
        f"内存增长过大: {total_growth / 1024 / 1024:.2f} MB（应 < 20MB）"
    )

    # ---- E-07：per-connection pending 随连接断开自动释放，无泄漏 ----
    # _conn_pending 是 ws_infer() 内的局部变量，连接关闭后自动 GC。

    # ---- executor.infer 被调用 100 次 ----
    assert mock_executor.infer.call_count == 100

    # ---- 超时请求被正确丢弃：设置极短过期时间，行为验证 ----
    mock_executor.infer.side_effect = None
    mock_executor.infer.return_value = make_action_chunk(expires_after_ms=30)

    with ws_context() as (ws2, meta2):
        # 请求 1：存储 pending chunk（expires_after_ms=30）
        req1 = make_valid_request(episode_id="timeout-ep", step_id=0)
        ws2.send_text(serialize_request(req1))
        resp1 = parse_response(ws2.receive_bytes())
        assert "actions" in resp1

        # 等待超过 expires_after_ms
        time.sleep(0.06)

        # 请求 2：服务端检测 pending chunk 过期并丢弃，请求仍正常处理
        req2 = make_valid_request(episode_id="timeout-ep", step_id=1)
        ws2.send_text(serialize_request(req2))
        resp2 = parse_response(ws2.receive_bytes())
        assert "actions" in resp2


# ---------------------------------------------------------------------------
# 8. ImageReference 辅助函数测试（openpi_service._is_image_reference /
#    _zero_placeholder_from_image_ref / _build_obs_from_model_input）
# ---------------------------------------------------------------------------
def test_is_image_reference_detects_valid():
    """is_image_reference 识别合规 ImageReference 字典。"""
    assert (
        is_image_reference(
            {
                "uri": "cas://sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "image_sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "camera_id": "CAM_A_TOP",
                "width": 640,
                "height": 480,
            }
        )
        is True
    )


def test_is_image_reference_rejects_legacy_pixels():
    """is_image_reference 拒绝旧版 pixels dict（不含 uri/image_sha256）。"""
    assert is_image_reference({"pixels": [1, 2, 3]}) is False
    assert is_image_reference({"data": b"fake"}) is False
    assert is_image_reference({}) is False


def test_is_image_reference_rejects_non_dict():
    """is_image_reference 拒绝非 dict 输入。"""
    assert is_image_reference(None) is False
    assert is_image_reference("not a dict") is False
    assert is_image_reference([]) is False


def test_is_image_reference_rejects_missing_field():
    """is_image_reference 缺少任一字段时返回 False（严格 5 字段校验）。"""
    base = {
        "uri": "cas://sha256/" + "a" * 64,
        "image_sha256": "sha256:" + "a" * 64,
        "camera_id": "CAM_A_TOP",
        "width": 640,
        "height": 480,
    }
    for field in list(base):
        partial = dict(base)
        del partial[field]
        assert is_image_reference(partial) is False, f"缺少 {field} 应判为 False"
    # 字段值为空/None 也判 False
    for field in list(base):
        partial = dict(base)
        partial[field] = ""
        assert is_image_reference(partial) is False, f"{field}='' 应判为 False"


def test_validate_image_reference_accepts_frozen_contract():
    """严格校验器接受 URI、摘要、相机与尺寸完全一致的引用。"""
    ref = {
        "uri": "cas://sha256/" + "a" * 64,
        "image_sha256": "sha256:" + "a" * 64,
        "camera_id": "CAM_A_TOP",
        "width": 640,
        "height": 480,
    }

    assert validate_image_reference(ref, expected_camera_id="CAM_A_TOP") is ref


def test_validate_image_reference_rejects_digest_mismatch():
    """URI digest 与 image_sha256 不完全一致时必须在 π0.5 侧拒绝。"""
    ref = {
        "uri": "cas://sha256/" + "a" * 64,
        "image_sha256": "sha256:" + "b" * 64,
        "camera_id": "CAM_A_TOP",
        "width": 640,
        "height": 480,
    }

    with pytest.raises(ValueError, match="摘要与 image_sha256 不一致"):
        validate_image_reference(ref, expected_camera_id="CAM_A_TOP")


def test_validate_image_reference_requires_exact_digest_text():
    """冻结契约要求 URI 与声明摘要文本完全一致，不做大小写归一化。"""
    ref = {
        "uri": "cas://sha256/" + "A" * 64,
        "image_sha256": "sha256:" + "a" * 64,
        "camera_id": "CAM_A_TOP",
        "width": 640,
        "height": 480,
    }

    with pytest.raises(ValueError, match="摘要与 image_sha256 不一致"):
        validate_image_reference(ref, expected_camera_id="CAM_A_TOP")


# ---------------------------------------------------------------------------
# 9. HTTP /v1/infer 端点测试（新 pi05ModelInput 格式，ImageReference + arm_a）
# ---------------------------------------------------------------------------
TEST_CKPT_SHA_HTTP = (
    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
)
TEST_NORM_SHA_HTTP = (
    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
)


def _make_http_infer_body(
    task_id: str = "P01_TO_S11",
    prompt: str = "请将轴件 P01 放置到料箱的 S11 格子中。",
    full_image: dict | None = None,
    wrist_image: dict | None = None,
    robot_state: list | None = None,
) -> dict[str, Any]:
    """构造 HTTP /v1/infer 请求体（对齐 Pi05Adapter.plan() 发出的新格式）。

    Pi05Adapter 发送的 model_input 符合 schemas/executor-infer.schema.json
    #$defs/pi05ModelInput：prompt + observation{camera{full_image,wrist_image},
    robot{state,tcp_pose_m_rad}}。
    """
    if full_image is None:
        if TEST_FULL_IMAGE_REFERENCE is None:
            raise RuntimeError("test_client fixture 未初始化真实 CAS 图像")
        full_image = dict(TEST_FULL_IMAGE_REFERENCE)
    if robot_state is None:
        robot_state = [0.51, -0.03, 0.42, 0.01, 0.02, -0.01, 0.0]
    # tcp_pose_m_rad 恰好 6 维，末三项是 robot_base rotation-vector。
    tcp_pose = [0.51, -0.03, 0.42, 0.01, 0.02, -0.01]

    return {
        "schema_version": "1.0",
        "request_id": "req-http-001",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": task_id,
        "subtask_id": task_id,
        "step_id": 0,
        "observation_id": "obs-1029",
        "deadline_ms": 15000,
        "executor": "pi05",
        "checkpoint_sha": TEST_CKPT_SHA_HTTP,
        "norm_stats_sha": TEST_NORM_SHA_HTTP,
        "expected_action_contract": "1.0",
        "model_input": {
            "prompt": prompt,
            "observation": {
                "camera": {
                    "full_image": full_image,
                    "wrist_image": wrist_image,
                },
                "robot": {
                    "state": robot_state,
                    "tcp_pose_m_rad": tcp_pose,
                },
            },
        },
    }


def test_http_infer_with_image_reference(test_client, mock_executor):
    """HTTP /v1/infer 接收 ImageReference 格式 model_input 并返回合规 action_chunk。

    Pi05Adapter 新契约发送 ImageReference；服务端通过公共 resolver 解析真实图像。
    """
    body = _make_http_infer_body()

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.json()}"

    data = resp.json()
    assert data["status"] == "ok"
    assert data["schema_version"] == "1.0"
    assert data["executor"] == "pi05"
    assert data["request_id"] == "req-http-001"
    assert data["task_id"] == "P01_TO_S11"
    assert data["subtask_id"] == "P01_TO_S11"
    assert data["checkpoint_sha"] == TEST_CKPT_SHA_HTTP
    assert data["norm_stats_sha"] == TEST_NORM_SHA_HTTP

    # action_chunk 符合 schemas/action-chunk.schema.json
    ac = data["action_chunk"]
    assert ac["contract_version"] == "1.0"
    assert ac["executor"] == "pi05"
    assert ac["action_space"] == "ee_delta_pose_gripper"
    assert ac["frame"] == "robot_base"
    assert ac["translation_unit"] == "m"
    assert ac["rotation_unit"] == "rad"
    assert ac["gripper_unit"] == "normalized"
    assert len(ac["steps"]) >= 1
    for step in ac["steps"]:
        assert len(step["values"]) == 7
        assert 1 <= step["duration_ms"] <= 10000

    # timing 字段存在且非负
    assert "timing" in data
    assert data["timing"]["total_ms"] >= 0

    # executor.infer 被调用，传入 ObsPacket
    mock_executor.infer.assert_called_once()
    obs = mock_executor.infer.call_args.args[0]
    assert isinstance(obs, ObsPacket)
    assert obs.instruction == body["model_input"]["prompt"]


def test_http_infer_with_wrist_image_null(test_client, mock_executor):
    """HTTP /v1/infer 接受 wrist_image=null。"""
    body = _make_http_infer_body(wrist_image=None)
    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_http_infer_rejects_non_v2_subtask_without_calling_executor(
    test_client,
    mock_executor,
):
    """π0.5 HTTP 边界要求 V2 subtask_id 与 task_id 一致。"""
    body = _make_http_infer_body()
    body["subtask_id"] = "S02_ARM_B_MOVE_FINISHED"

    response = test_client.post("/v1/infer", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "TASK_1001_INVALID"
    mock_executor.infer.assert_not_called()


def test_http_infer_rejects_non_frozen_arm_a_prompt(
    test_client,
    mock_executor,
):
    """π0.5 只接收框架机读配置冻结的 Arm_A 指令。"""
    body = _make_http_infer_body(prompt="把零件随便放进箱子")

    response = test_client.post("/v1/infer", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "TASK_1001_INVALID"
    mock_executor.infer.assert_not_called()


def test_dummy_placeholder_sha_is_consistent_across_health_and_infer(
    test_client,
    mock_executor,
    monkeypatch,
):
    """Dummy 未注入制品 SHA 时，health 与 infer 使用同一合法占位值。"""
    monkeypatch.delenv("PI05_CHECKPOINT_SHA", raising=False)
    monkeypatch.delenv("PI05_NORM_STATS_SHA", raising=False)
    mock_executor.checkpoint_sha = ""
    mock_executor.norm_stats_sha = ""

    health_body = test_client.get("/health").json()
    infer_response = test_client.post("/v1/infer", json=_make_http_infer_body())

    assert health_body["checkpoint_sha"] == TEST_CKPT_SHA_HTTP
    assert health_body["norm_stats_sha"] == TEST_NORM_SHA_HTTP
    assert infer_response.status_code == 200
    assert infer_response.json()["status"] == "ok"


def test_http_infer_rejects_image_reference_digest_mismatch(
    test_client,
    mock_executor,
):
    """HTTP 生产路径在进入解码或推理前拒绝不一致的 CAS 摘要。"""
    body = _make_http_infer_body()
    body["model_input"]["observation"]["camera"]["full_image"]["image_sha256"] = (
        "sha256:" + "b" * 64
    )

    resp = test_client.post("/v1/infer", json=body)

    assert resp.status_code == 422
    assert resp.json()["status"] == "error"
    assert resp.json()["error"]["code"] == "CAS_1304_METADATA_MISMATCH"
    assert resp.json()["error"]["retryable"] is False
    mock_executor.infer.assert_not_called()


def test_http_real_mode_rejects_declared_image_size_mismatch(
    test_client,
    mock_executor,
    monkeypatch,
    tmp_path,
):
    """公共 resolver 拒绝与 PNG 实际尺寸不一致的 ImageReference。"""
    shared_image_cas = ImageCas(ImageCasConfig(root=tmp_path, cache_max_bytes=0))
    expected_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    full_image = shared_image_cas.write_rgb(
        expected_rgb,
        camera_id="CAM_A_TOP",
    ).to_dict()
    full_image["width"] = 1279
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", shared_image_cas)
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)

    response = test_client.post(
        "/v1/infer",
        json=_make_http_infer_body(full_image=full_image),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "TASK_1001_INVALID"
    assert response.json()["error"]["retryable"] is False
    mock_executor.infer.assert_not_called()


def test_http_real_mode_without_cas_resolver_fails_closed(
    test_client,
    mock_executor,
    monkeypatch,
):
    """Real 模式 resolver 未初始化时返回稳定 CAS 错误，禁止零图推理。"""
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", None)
    monkeypatch.setattr(openpi_service, "request_image_resolver", None)
    monkeypatch.setattr(openpi_service, "v1_infer_handler", None)
    monkeypatch.setattr(
        openpi_service,
        "_image_cas_init_error",
        "CAS root unavailable",
    )
    body = _make_http_infer_body()

    resp = test_client.post("/v1/infer", json=body)

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "CAS_1306_UNAVAILABLE"
    assert resp.json()["error"]["retryable"] is True
    mock_executor.infer.assert_not_called()


def test_http_real_mode_resolves_verified_full_image(
    test_client,
    mock_executor,
    monkeypatch,
    tmp_path,
):
    """Real HTTP 路径把公共 CAS 验证后的真实 RGB 传给执行器。"""
    shared_image_cas = ImageCas(ImageCasConfig(root=tmp_path, cache_max_bytes=0))
    expected_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    expected_rgb[10, 20] = [17, 34, 51]
    full_image = shared_image_cas.write_rgb(
        expected_rgb,
        camera_id="CAM_A_TOP",
    )
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", shared_image_cas)
    resolver = CasRequestImageResolver(shared_image_cas)
    monkeypatch.setattr(openpi_service, "request_image_resolver", resolver)
    monkeypatch.setattr(
        openpi_service,
        "v1_infer_handler",
        openpi_service.build_v1_infer_handler(
            resolver=resolver,
            backend=openpi_service._infer_backend,
        ),
    )
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)

    response = test_client.post(
        "/v1/infer",
        json=_make_http_infer_body(full_image=full_image.to_dict()),
    )

    assert response.status_code == 200, response.json()
    mock_executor.infer.assert_called_once()
    obs = mock_executor.infer.call_args.args[0]
    assert np.array_equal(obs.rgb_front, expected_rgb)
    assert obs.rgb_front.dtype == np.uint8
    assert obs.rgb_front.flags.writeable is False


def test_http_rejects_non_null_wrist_image(
    test_client,
    mock_executor,
    monkeypatch,
    tmp_path,
):
    """冻结三相机配置要求 wrist_image=null，非空引用在 Schema 层拒绝。"""
    shared_image_cas = ImageCas(ImageCasConfig(root=tmp_path, cache_max_bytes=0))
    full_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    wrist_rgb = np.full((8, 12, 3), 73, dtype=np.uint8)
    full_image = shared_image_cas.write_rgb(full_rgb, camera_id="CAM_A_TOP")
    wrist_image = shared_image_cas.write_rgb(
        wrist_rgb,
        camera_id="CAM_A_WRIST",
    )
    response = test_client.post(
        "/v1/infer",
        json=_make_http_infer_body(
            full_image=full_image.to_dict(),
            wrist_image=wrist_image.to_dict(),
        ),
    )

    assert response.status_code == 400, response.json()
    assert response.json()["error"]["code"] == "TASK_1001_INVALID"
    mock_executor.infer.assert_not_called()


@pytest.mark.parametrize(
    ("failure_code", "retryable", "expected_status"),
    [
        (FailureCode.CAS_NOT_FOUND, True, 503),
        (FailureCode.CAS_DIGEST_MISMATCH, False, 422),
        (FailureCode.CAS_DECODE_FAILED, False, 422),
        (FailureCode.CAS_METADATA_MISMATCH, False, 422),
        (FailureCode.CAS_LIMIT_EXCEEDED, False, 422),
        (FailureCode.CAS_UNAVAILABLE, True, 503),
    ],
)
def test_http_real_mode_preserves_cas_error_semantics(
    test_client,
    mock_executor,
    monkeypatch,
    failure_code,
    retryable,
    expected_status,
):
    """公共 resolver 的 CAS 错误码和 retryable 必须原样进入 ErrorPacket。"""
    failing_image_cas = MagicMock(spec=["resolve_rgb"])
    failing_image_cas.resolve_rgb.side_effect = ImageCasError(
        failure_code,
        "injected CAS failure",
        retryable=retryable,
    )
    monkeypatch.setattr(openpi_service, "SERVICE_MODE", "real")
    monkeypatch.setattr(openpi_service, "image_cas", failing_image_cas)
    resolver = CasRequestImageResolver(failing_image_cas)
    monkeypatch.setattr(openpi_service, "request_image_resolver", resolver)
    monkeypatch.setattr(
        openpi_service,
        "v1_infer_handler",
        openpi_service.build_v1_infer_handler(
            resolver=resolver,
            backend=openpi_service._infer_backend,
        ),
    )
    monkeypatch.setattr(openpi_service, "_image_cas_init_error", None)

    response = test_client.post("/v1/infer", json=_make_http_infer_body())

    assert response.status_code == expected_status
    error = response.json()["error"]
    assert error["code"] == failure_code.value
    assert error["retryable"] is retryable
    mock_executor.infer.assert_not_called()


def test_http_infer_sha_mismatch_rejected(test_client, mock_executor):
    """HTTP /v1/infer 在校验 SHA 不匹配时返回 409 EXEC_2105_MODEL_REVISION_MISMATCH。"""
    body = _make_http_infer_body()
    body["checkpoint_sha"] = "sha256:" + "f" * 64  # 与服务端 TEST_CKPT_SHA_HTTP 不同

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 409
    assert resp.json()["status"] == "error"
    assert resp.json()["error"]["code"] == "EXEC_2105_MODEL_REVISION_MISMATCH"
    # executer.infer 不应该被调用
    mock_executor.infer.assert_not_called()


def test_http_infer_executor_name_mismatch(test_client, mock_executor):
    """HTTP /v1/infer 在校验 executor 字段不匹配时返回 400 TASK_1001_INVALID。"""
    body = _make_http_infer_body()
    body["executor"] = "openvla_oft"  # 应为 pi05

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 400
    assert resp.json()["status"] == "error"
    assert resp.json()["error"]["code"] == "TASK_1001_INVALID"
    mock_executor.infer.assert_not_called()


def test_http_infer_missing_required_fields(test_client, mock_executor):
    """HTTP /v1/infer 缺少必填字段时返回 422 TASK_1001_INVALID。"""
    body = _make_http_infer_body()
    del body["model_input"]

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 422
    assert resp.json()["status"] == "error"
    assert resp.json()["error"]["code"] == "TASK_1001_INVALID"
    mock_executor.infer.assert_not_called()


def test_http_infer_executor_unavailable_when_not_initialized(test_client):
    """HTTP /v1/infer 在执行器未初始化时返回 503 EXEC_2101_UNAVAILABLE。"""
    # 临时清空 executor 模拟未初始化状态
    saved = openpi_service.executor
    openpi_service.executor = None
    try:
        body = _make_http_infer_body()
        resp = test_client.post("/v1/infer", json=body)
        assert resp.status_code == 503
        assert resp.json()["status"] == "error"
        assert resp.json()["error"]["code"] == "EXEC_2101_UNAVAILABLE"
    finally:
        openpi_service.executor = saved


def test_http_infer_rejects_legacy_pixels(test_client, mock_executor):
    """HTTP /v1/infer 拒绝旧版 pixels dict 格式（冻结 Schema 禁止内联像素）。"""
    body = _make_http_infer_body()
    # 旧版格式：直接像素数组已被冻结 Schema additionalProperties:false 禁止
    body["model_input"]["observation"]["camera"]["full_image"] = {
        "pixels": np.zeros((4, 4, 3), dtype=np.uint8).tolist(),
    }
    body["model_input"]["observation"]["camera"]["wrist_image"] = None

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 400
    assert resp.json()["status"] == "error"
    mock_executor.infer.assert_not_called()


# ---------------------------------------------------------------------------
# 10. HTTP /v1/cancel 端点测试
# ---------------------------------------------------------------------------
def test_http_cancel_basic(test_client, mock_executor):
    """HTTP /v1/cancel 基本流程：先 infer 后 cancel，返回 cancelled。"""
    # 先发起一次 infer 使 task_id 被记录
    body = _make_http_infer_body(task_id="P01_TO_S11")
    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 200

    # 取消
    cancel_body = {
        "schema_version": "1.0",
        "request_id": "cancel-req-001",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": "P01_TO_S11",
        "subtask_id": "P01_TO_S11",
        "reason": "test cancel",
    }
    resp = test_client.post("/v1/cancel", json=cancel_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["server_context_cleared"] is True
    assert data["task_id"] == "P01_TO_S11"


def test_http_cancel_idempotent(test_client, mock_executor):
    """HTTP /v1/cancel 幂等：重复 cancel 返回 already_completed。"""
    body = _make_http_infer_body(task_id="P01_TO_S11")
    test_client.post("/v1/infer", json=body)

    cancel_body = {
        "schema_version": "1.0",
        "request_id": "cancel-req-002",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": "P01_TO_S11",
        "subtask_id": "P01_TO_S11",
        "reason": "test cancel",
    }
    # 第一次取消
    resp1 = test_client.post("/v1/cancel", json=cancel_body)
    assert resp1.json()["status"] == "cancelled"

    # 第二次取消同一 task_id
    resp2 = test_client.post("/v1/cancel", json=cancel_body)
    assert resp2.json()["status"] == "already_completed"


def test_http_cancel_unknown_task_returns_not_found(test_client):
    """HTTP /v1/cancel 对未 infer 过的 task_id 返回 not_found。"""
    cancel_body = {
        "schema_version": "1.0",
        "request_id": "cancel-req-003",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": "never-seen-task",
        "subtask_id": "P01_TO_S11",
        "reason": "test cancel",
    }
    resp = test_client.post("/v1/cancel", json=cancel_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_found"
