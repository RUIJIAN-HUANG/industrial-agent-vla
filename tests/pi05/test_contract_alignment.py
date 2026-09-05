"""PI05 契约对齐集成测试 —— 进程内路径 + 全链路 E2E + 异常降级。

覆盖三条数据流中唯一未覆盖的进程内路径（Pi05ContractAdapter），
补充 ImageReference 管道连通性、arm_a_rgb 缺失防御、arm_a 缺失降级。

方案书出处：
  - interface-contracts.md §4/§7/§7.3：统一契约与 pi05ModelInput
  - executor.py Pi05Adapter.plan()：HTTP 路径 model_input 结构
  - agent-framework.md §9：统一 7 维动作合同
"""

from __future__ import annotations

import hashlib
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from services.pi05.src.observation import ObsPacket
from services.pi05.src.pi05 import MOCK_CHUNK_LEN, Pi05Executor
from services.pi05.src.pi05_contract_adapter import Pi05ContractAdapter
from industrial_agent.contracts import Observation, TaskSchema
from industrial_agent.errors import ExecutorError, FailureCode
from industrial_agent.executor import ExecutionContext
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from industrial_agent.service_images import CasRequestImageResolver


@pytest.fixture(autouse=True)
def explicit_dummy_mode(monkeypatch):
    """进程内契约测试显式启用 Dummy，不依赖执行器默认值。"""
    monkeypatch.setenv("PI05_MODE", "dummy")


@pytest.fixture
def resolved_camera(tmp_path):
    """Create a real frozen-size CAS frame and the shared request resolver."""

    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(1280, dtype=np.uint16)[None, :] % 256
    pixels[:, :, 1] = 37
    pixels[:, :, 2] = 211
    reference = image_cas.write_rgb(pixels, camera_id="CAM_A_TOP")
    camera = {
        "full_image": reference.to_dict(),
        "arm_a_rgb": reference.to_dict(),
        "wrist_image": None,
    }
    return CasRequestImageResolver(image_cas), camera, pixels


# ── 辅助函数 ──────────────────────────────────────────────────────────────────
def _make_mock_canonical() -> Any:
    """构造 Mock CanonicalActionChunk（替代 Pi05Executor.infer 返回值）。"""
    return types.SimpleNamespace(
        actions=np.zeros((MOCK_CHUNK_LEN, 7), dtype=np.float32),
        space_id="eef_delta_xyz_axisangle_gripper_v1",
        frame="robot_base",
        control_hz=10,
        generated_step=0,
        source_policy="pi05",
        checkpoint_sha="",
        expires_after_ms=1000,
    )


def _make_observation(
    *, camera: dict | None = None, robot: dict | None = None
) -> Observation:
    """构造符合 online-observation.schema.json 的 Observation。

    对齐 mock.py FixedDualArmMockSimulator 产出的 observation 结构：
      camera: {full_image, arm_a_rgb, handoff_rgb, arm_b_rgb, wrist_image?}
      robot: {active_arm, arm_a, arm_b}
      所有 image 字段均为 ImageReference dict。
    """
    arm_a_sha = hashlib.sha256(b"arm_a_rgb").hexdigest()
    wrist_sha = hashlib.sha256(b"wrist").hexdigest()
    default_camera = {
        "full_image": {
            "uri": f"cas://sha256/{arm_a_sha}",
            "image_sha256": f"sha256:{arm_a_sha}",
            "camera_id": "CAM_HANDOFF",
            "width": 1280,
            "height": 720,
        },
        "arm_a_rgb": {
            "uri": f"cas://sha256/{arm_a_sha}",
            "image_sha256": f"sha256:{arm_a_sha}",
            "camera_id": "CAM_A_TOP",
            "width": 1280,
            "height": 720,
        },
        "wrist_image": {
            "uri": f"cas://sha256/{wrist_sha}",
            "image_sha256": f"sha256:{wrist_sha}",
            "camera_id": "CAM_WRIST",
            "width": 224,
            "height": 224,
        },
    }
    default_robot = {
        "active_arm": "Arm_A",
        "arm_a": {
            "tcp_pose_m_rad": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            "state": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.375],
            "retreated": False,
            "gripper_open": False,
            "stationary": True,
        },
        "arm_b": {
            "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
            "state": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
            "retreated": True,
            "gripper_open": True,
            "stationary": True,
        },
    }
    return Observation(
        observation_id="obs-e2e-001",
        timestamp_ms=1_700_000_000_000,
        data={
            "camera": camera or default_camera,
            "robot": robot or default_robot,
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {
                "packed_part_count": 2,
                "bin_at_handoff": False,
                "bin_at_finished": False,
                "bin_speed_m_s": 0.0,
                "status": "pending",
            },
            "quality": {"confidence": 0.99},
        },
    )


def _make_task() -> TaskSchema:
    return TaskSchema(
        task_id="job-1:S01_ARM_A_PACK_HANDOFF",
        instruction="将工作区中的四个红色零件依次装入料箱",
        task_type="pick_place",
        postconditions=(),
    )


def _make_context(*, original_instruction: str | None = None) -> ExecutionContext:
    return ExecutionContext(
        run_id="run-e2e-001",
        strategy_attempt=1,
        replan_index=0,
        step_id=0,
        timeout_ms=15_000,
        original_instruction=original_instruction,
    )


# ── 用例 1：进程内全链路 E2E ───────────────────────────────────────────────────
def test_e2e_adapter_plan_returns_valid_action_chunk(resolved_camera):
    """E2E：Observation(ImageReference) → adapter.plan() → ActionChunk 逐字段校验。

    覆盖：arm_a_rgb 的 ImageReference→共享 CAS 校验与解码→_prep_image→
    _build_example→_infer_mock→_clip_actions→to_action_chunk 的完整管线。
    方案书 agent-framework.md §9 统一 7 维动作合同。
    """
    resolver, camera, expected_pixels = resolved_camera
    captured: dict[str, np.ndarray] = {}

    def capture_infer(obs: ObsPacket):
        captured["rgb"] = obs.rgb_front.copy()
        captured["state"] = obs.robot_state.copy()
        return _make_mock_canonical()

    with patch.object(Pi05Executor, "infer", side_effect=capture_infer):
        adapter = Pi05ContractAdapter(resolver=resolver)
        observation = _make_observation(camera=camera)
        task = _make_task()
        context = _make_context()

        chunk = adapter.plan(task, observation, context)

    # ── ActionChunk 顶层字段 ──
    assert chunk.contract_version == "1.0"
    assert chunk.action_space == "ee_delta_pose_gripper"
    assert chunk.frame == "robot_base"
    assert chunk.translation_unit == "m"
    assert chunk.rotation_unit == "rad"
    assert chunk.gripper_unit == "normalized"
    assert chunk.executor == "pi05"
    assert chunk.task_id == task.task_id
    assert len(chunk.chunk_id) > 0

    # ── steps 结构 ──
    assert len(chunk.steps) == 1
    for step in chunk.steps:
        assert len(step.values) == 7
        assert all(np.isfinite(v) for v in step.values)
        assert 1 <= step.duration_ms <= 10000

    # ── validate_contract 显式调用不抛异常 ──
    chunk.validate_contract()
    np.testing.assert_array_equal(captured["rgb"], expected_pixels)
    assert captured["state"][6] == pytest.approx(0.375)


# ── 用例 2：ImageReference 管道连通性 ───────────────────────────────────────────
def test_image_reference_flow_resolves_verified_cas_pixels(resolved_camera):
    """ImageReference 必须通过共享 CAS resolver 还原为真实、校验后的像素。"""

    resolver, camera, expected_pixels = resolved_camera
    captured: dict[str, np.ndarray] = {}

    def capture_infer(obs: ObsPacket):
        captured["rgb"] = obs.rgb_front.copy()
        return _make_mock_canonical()

    with patch.object(Pi05Executor, "infer", side_effect=capture_infer):
        adapter = Pi05ContractAdapter(resolver=resolver)
        chunk = adapter.plan(
            _make_task(), _make_observation(camera=camera), _make_context()
        )

    assert chunk is not None
    assert len(chunk.steps) >= 1
    np.testing.assert_array_equal(captured["rgb"], expected_pixels)


def test_image_reference_wrist_is_null(resolved_camera):
    """wrist_image=null 时不抛异常。"""
    resolver, camera, _ = resolved_camera

    with patch.object(Pi05Executor, "infer", return_value=_make_mock_canonical()):
        adapter = Pi05ContractAdapter(resolver=resolver)
        chunk = adapter.plan(
            _make_task(), _make_observation(camera=camera), _make_context()
        )

    assert chunk is not None


def test_image_reference_without_resolver_fails_closed():
    """进程内适配器不得把 CAS 引用替换成黑图或绕过公共 resolver。"""

    adapter = Pi05ContractAdapter()
    with pytest.raises(ExecutorError) as exc_info:
        adapter.plan(_make_task(), _make_observation(), _make_context())

    assert exc_info.value.code == FailureCode.CAS_UNAVAILABLE
    assert exc_info.value.retryable is True


# ── 用例 3：异常与降级路径 ──────────────────────────────────────────────────────
def test_missing_arm_a_rgb_raises_executor_error():
    """arm_a_rgb 缺失时抛出 ExecutorError(EXECUTOR_BAD_RESPONSE)。"""
    observation = _make_observation(
        camera={
            "full_image": {
                "uri": f"cas://sha256/{'d' * 64}",
                "image_sha256": f"sha256:{'d' * 64}",
                "camera_id": "CAM_A_TOP",
                "width": 1280,
                "height": 720,
            },
            # arm_a_rgb 故意缺失
        }
    )
    adapter = Pi05ContractAdapter()

    with pytest.raises(ExecutorError) as exc_info:
        adapter.plan(_make_task(), observation, _make_context())

    assert exc_info.value.code == FailureCode.EXECUTOR_BAD_RESPONSE
    assert "arm_a_rgb" in str(exc_info.value)


def test_arm_a_rgb_none_raises_executor_error():
    """arm_a_rgb=None 时抛出 ExecutorError。"""
    observation = _make_observation(camera={"arm_a_rgb": None})
    adapter = Pi05ContractAdapter()

    with pytest.raises(ExecutorError) as exc_info:
        adapter.plan(_make_task(), observation, _make_context())

    assert exc_info.value.code == FailureCode.EXECUTOR_BAD_RESPONSE


def test_arm_a_missing_fails_closed(resolved_camera):
    """arm_a 缺失时禁止零状态兜底，模型不得被调用。"""
    resolver, camera, _ = resolved_camera
    observation = _make_observation(
        camera=camera,
        robot={
            "active_arm": "Arm_A",
            # arm_a 完全缺失
        },
    )

    with patch.object(Pi05Executor, "infer") as infer:
        adapter = Pi05ContractAdapter(resolver=resolver)
        with pytest.raises(ExecutorError) as exc_info:
            adapter.plan(_make_task(), observation, _make_context())

    assert exc_info.value.code == FailureCode.EXECUTOR_BAD_RESPONSE
    assert "cannot produce canonical state_7d" in str(exc_info.value)
    infer.assert_not_called()


# ── 用例 4：prompt 提取优先级 ──────────────────────────────────────────────────
def test_original_instruction_takes_priority_over_task_instruction(resolved_camera):
    """context.original_instruction 存在时优先于 task.instruction。"""
    captured: dict = {}

    def capture_infer(obs: ObsPacket):
        captured["instruction"] = obs.instruction
        return _make_mock_canonical()

    resolver, camera, _ = resolved_camera
    with patch.object(Pi05Executor, "infer", side_effect=capture_infer):
        adapter = Pi05ContractAdapter(resolver=resolver)
        task = _make_task()
        context = _make_context(original_instruction="冻结指令（来自总控 Planner）")
        adapter.plan(task, _make_observation(camera=camera), context)

    assert captured.get("instruction") == "冻结指令（来自总控 Planner）"


def test_task_instruction_fallback_when_original_is_none(resolved_camera):
    """context.original_instruction 为 None 时回退到 task.instruction。"""
    captured: dict = {}

    def capture_infer(obs: ObsPacket):
        captured["instruction"] = obs.instruction
        return _make_mock_canonical()

    resolver, camera, _ = resolved_camera
    with patch.object(Pi05Executor, "infer", side_effect=capture_infer):
        adapter = Pi05ContractAdapter(resolver=resolver)
        task = _make_task()
        context = _make_context(original_instruction=None)
        adapter.plan(task, _make_observation(camera=camera), context)

    assert captured.get("instruction") == task.instruction


# ── 用例 5：ActionChunk 格式自校验 ──────────────────────────────────────────────
def test_to_action_chunk_validates_contract():
    """Pi05Executor.to_action_chunk() 输出通过 validate_contract() 自校验。"""
    ex = Pi05Executor()

    canonical = types.SimpleNamespace(
        actions=np.random.default_rng(42)
        .uniform(-0.01, 0.01, (3, 7))
        .astype(np.float32),
        space_id="eef_delta_xyz_axisangle_gripper_v1",
        frame="robot_base",
        control_hz=10,
        generated_step=5,
        source_policy="pi05",
        checkpoint_sha="",
        expires_after_ms=1000,
    )
    chunk = ex.to_action_chunk(canonical, task_id="task-1", executor_name="pi05")
    assert chunk.executor == "pi05"
    assert chunk.task_id == "task-1"
    assert len(chunk.steps) == 1
    assert chunk.steps[0].duration_ms == 100


# ── 用例 6：arm_a_rgb 解码失败防御 ──────────────────────────────────────────────
def test_arm_a_rgb_not_decodable_raises_executor_error():
    """arm_a_rgb 既不是 ImageReference 也不是可解码的 numpy/pixels dict 时抛异常。"""
    observation = _make_observation(camera={"arm_a_rgb": 42})  # 非 dict 不可解码
    adapter = Pi05ContractAdapter()

    with pytest.raises(ExecutorError) as exc_info:
        adapter.plan(_make_task(), observation, _make_context())

    assert exc_info.value.code == FailureCode.EXECUTOR_BAD_RESPONSE
    assert "frozen ImageReference" in str(exc_info.value)
