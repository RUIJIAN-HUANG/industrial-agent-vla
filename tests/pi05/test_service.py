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
os.environ.setdefault("PI05_SERVICE_MODE", "dummy")
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

from services.pi05.src import openpi_service
from services.pi05.src.action import CanonicalActionChunk
from services.pi05.src.observation import (
    ObsPacket,
    image_reference_to_placeholder,
    is_image_reference,
)

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
        robot_state = [0.0] * 8
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
    # 属性（_checkpoint_sha / _norm_stats_sha 被 openpi_service._checkpoint_sha() 读取）
    ex._checkpoint_sha = TEST_CHECKPOINT_SHA
    ex._norm_stats_sha = TEST_NORM_STATS_SHA
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
def test_client(mock_executor: MagicMock):
    """FastAPI TestClient：启动时用 mock_executor 替换真实 Pi05Executor。

    每个 test 获得：
    - 全新的 TestClient（触发 startup → _init_executor → Pi05Executor()）
    - 全局状态已重置（pending_chunks / current_episode_id / last_step_id / lock）
    - Pi05Executor 被 patch 为返回 mock_executor（彻底隔离底层执行器）
    """
    # ---- 重置全局状态（防止跨测试串扰，方案书 §7.1 多 episode 并发安全）----
    openpi_service.executor = None
    openpi_service.pending_chunks.clear()
    openpi_service._cancelled_tasks.clear()
    openpi_service._seen_task_ids.clear()
    # 重建 lock：避免绑定到上一个 TestClient 的事件循环（Python 3.10+ _LoopBoundMixin）
    openpi_service._pending_chunks_lock = asyncio.Lock()
    openpi_service._cancelled_tasks_lock = asyncio.Lock()

    # ---- patch Pi05Executor：使 _init_executor() 创建 mock 而非真实执行器 ----
    with patch.object(openpi_service, "Pi05Executor", return_value=mock_executor):
        with TestClient(openpi_service.app) as client:
            yield client

    # ---- 清理全局状态 ----
    openpi_service.executor = None
    openpi_service.pending_chunks.clear()
    openpi_service._cancelled_tasks.clear()
    openpi_service._seen_task_ids.clear()


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

        # pending_chunks 应有一条记录
        assert "expire-ep" in openpi_service.pending_chunks
        old_timestamp = openpi_service.pending_chunks["expire-ep"]["timestamp"]

        # 等待超过 expires_after_ms（50ms）
        time.sleep(0.08)

        # 发送下一个 step：服务端检测 pending chunk 过期并丢弃，但请求仍正常处理
        req3 = make_valid_request(episode_id="expire-ep", step_id=2)
        ws.send_text(serialize_request(req3))
        resp3 = parse_response(ws.receive_bytes())
        assert "actions" in resp3  # 过期丢弃后重新推理，响应正常

        # 过期 chunk 被新 chunk 覆盖（timestamp 更新）
        new_timestamp = openpi_service.pending_chunks["expire-ep"]["timestamp"]
        assert new_timestamp > old_timestamp

        # ---- 5c. episode 切换：旧 episode 的 pending_chunks 被清除 ----
        # 恢复正常过期时间
        mock_executor.infer.return_value = make_action_chunk(expires_after_ms=1000)

        # 切换到新 episode
        req_new_ep = make_valid_request(episode_id="new-ep-after-switch", step_id=0)
        ws.send_text(serialize_request(req_new_ep))
        resp_new_ep = parse_response(ws.receive_bytes())
        assert "actions" in resp_new_ep

        # 旧 episode 的 chunk 已被 clear() 清除（方案书 §3.3.1 Para186：切换时清空动作队列）
        assert "expire-ep" not in openpi_service.pending_chunks
        # 新 episode 的 chunk 已存储
        assert "new-ep-after-switch" in openpi_service.pending_chunks
        # step_id 跟踪已重置：新 episode 内 step_id=0 被接受（per-connection 追踪已置 None）
        assert "step_id 未递增" not in str(resp_new_ep), (
            "新 episode 的 step_id=0 应被接受（per-connection 状态已重置）"
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

    # ---- pending_chunks 连接断开后已清理（Bug 7 修复：finally 块 pop）----
    assert len(openpi_service.pending_chunks) == 0, (
        "WebSocket 断开后 pending_chunks 应被清理（非泄漏）"
    )

    # ---- executor.infer 被调用 100 次 ----
    assert mock_executor.infer.call_count == 100

    # ---- 超时请求被正确丢弃：设置极短过期时间，等待后下一请求触发丢弃 ----
    mock_executor.infer.side_effect = None
    mock_executor.infer.return_value = make_action_chunk(expires_after_ms=30)

    with ws_context() as (ws2, meta2):
        # 请求 1：存储 pending chunk（expires_after_ms=30）
        req1 = make_valid_request(episode_id="timeout-ep", step_id=0)
        ws2.send_text(serialize_request(req1))
        resp1 = parse_response(ws2.receive_bytes())
        assert "actions" in resp1

        assert "timeout-ep" in openpi_service.pending_chunks
        old_ts = openpi_service.pending_chunks["timeout-ep"]["timestamp"]

        # 等待超过 expires_after_ms
        time.sleep(0.06)

        # 请求 2：服务端检测 pending chunk 过期并丢弃，请求仍正常处理
        req2 = make_valid_request(episode_id="timeout-ep", step_id=1)
        ws2.send_text(serialize_request(req2))
        resp2 = parse_response(ws2.receive_bytes())
        assert "actions" in resp2

        # 过期 chunk 被新 chunk 覆盖（timestamp 更新，证明旧 chunk 被丢弃）
        new_ts = openpi_service.pending_chunks["timeout-ep"]["timestamp"]
        assert new_ts > old_ts, "过期 chunk 应被丢弃并替换"


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


def test_zero_placeholder_uses_image_ref_dimensions():
    """image_reference_to_placeholder 按 ImageReference 尺寸创建零图。"""
    ref = {
        "uri": "cas://sha256/" + "b" * 64,
        "image_sha256": "sha256:" + "b" * 64,
        "camera_id": "CAM_A_TOP",
        "width": 320,
        "height": 240,
    }
    img = image_reference_to_placeholder(ref)
    assert img.dtype == np.uint8
    assert img.shape == (240, 320, 3)
    assert np.all(img == 0)


def test_zero_placeholder_fallback_dimensions():
    """image_reference_to_placeholder 在尺寸非法时回退默认 640x480。"""
    ref_bad = {
        "uri": "cas://sha256/" + "c" * 64,
        "image_sha256": "sha256:" + "c" * 64,
        "camera_id": "CAM_A_TOP",
        "width": -1,
        "height": 0,
    }
    img = image_reference_to_placeholder(ref_bad)
    assert img.shape == (480, 640, 3)


def test_zero_placeholder_clamps_oversized():
    """image_reference_to_placeholder 尺寸超过 4096 时钳制。"""
    ref_huge = {
        "uri": "cas://sha256/" + "d" * 64,
        "image_sha256": "sha256:" + "d" * 64,
        "camera_id": "CAM_A_TOP",
        "width": 8192,
        "height": 5000,
    }
    img = image_reference_to_placeholder(ref_huge)
    assert img.shape[0] == 4096  # 钳制
    assert img.shape[1] == 4096  # 钳制


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
    task_id: str = "job-1:S01_ARM_A_PACK_HANDOFF",
    prompt: str = "将工作区中的四个红色零件依次装入料箱",
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
        full_image = {
            "uri": "cas://sha256/" + "a" * 64,
            "image_sha256": "sha256:" + "a" * 64,
            "camera_id": "CAM_A_TOP",
            "width": 640,
            "height": 480,
        }
    if robot_state is None:
        robot_state = [0.51, -0.03, 0.42, 0.01, 0.02, -0.01, 0.0]
    # tcp_pose_m_rad 至少 6 维（schemas/executor-infer.schema.json minItems=6）
    tcp_pose = [0.51, -0.03, 0.42, 0.01, 0.02, -0.01]

    return {
        "schema_version": "1.0",
        "request_id": "req-http-001",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": task_id,
        "subtask_id": "S01_ARM_A_PACK_HANDOFF",
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

    Pi05Adapter 新契约发送 ImageReference 而非原始像素；服务端按尺寸创建零图占位。
    """
    body = _make_http_infer_body()

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.json()}"

    data = resp.json()
    assert data["status"] == "ok"
    assert data["schema_version"] == "1.0"
    assert data["executor"] == "pi05"
    assert data["request_id"] == "req-http-001"
    assert data["task_id"] == "job-1:S01_ARM_A_PACK_HANDOFF"
    assert data["subtask_id"] == "S01_ARM_A_PACK_HANDOFF"
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


def test_http_infer_legacy_pixels_fallback(test_client, mock_executor):
    """HTTP /v1/infer 兼容旧版 pixels dict 格式（非 ImageReference）。"""
    body = _make_http_infer_body()
    # 替换为旧版格式：直接像素
    body["model_input"]["observation"]["camera"]["full_image"] = {
        "pixels": np.zeros((4, 4, 3), dtype=np.uint8).tolist(),
    }
    body["model_input"]["observation"]["camera"]["wrist_image"] = None

    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    mock_executor.infer.assert_called_once()


# ---------------------------------------------------------------------------
# 10. HTTP /v1/cancel 端点测试
# ---------------------------------------------------------------------------
def test_http_cancel_basic(test_client, mock_executor):
    """HTTP /v1/cancel 基本流程：先 infer 后 cancel，返回 cancelled。"""
    # 先发起一次 infer 使 task_id 被记录
    body = _make_http_infer_body(task_id="job-cancel-1")
    resp = test_client.post("/v1/infer", json=body)
    assert resp.status_code == 200

    # 取消
    cancel_body = {
        "schema_version": "1.0",
        "request_id": "cancel-req-001",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": "job-cancel-1",
        "subtask_id": "S01_ARM_A_PACK_HANDOFF",
        "reason": "test cancel",
    }
    resp = test_client.post("/v1/cancel", json=cancel_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["server_context_cleared"] is True
    assert data["task_id"] == "job-cancel-1"


def test_http_cancel_idempotent(test_client, mock_executor):
    """HTTP /v1/cancel 幂等：重复 cancel 返回 already_completed。"""
    body = _make_http_infer_body(task_id="job-cancel-idem")
    test_client.post("/v1/infer", json=body)

    cancel_body = {
        "schema_version": "1.0",
        "request_id": "cancel-req-002",
        "trace_id": "trace-9001",
        "episode_id": "episode-17",
        "task_id": "job-cancel-idem",
        "subtask_id": "S01_ARM_A_PACK_HANDOFF",
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
        "subtask_id": "S01_ARM_A_PACK_HANDOFF",
        "reason": "test cancel",
    }
    resp = test_client.post("/v1/cancel", json=cancel_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_found"
