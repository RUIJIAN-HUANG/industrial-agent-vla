"""Pi05Executor 单元测试。

被测目标：services/pi05/src/pi05.py
方案书出处：
  - §3.3.1 Para185/186：norm stats / 反归一化由 openpi 在 policy.infer 内部完成，
    适配器不再二次反归一化；失败切换清空动作队列与客户端缓存。
  - Table 69 Row7 / 附录B：单步平移 ≤2cm、旋转 ≤5°（≈0.0873rad），超限截断并报警。
  - §7.5 image_pipeline：固定像素校验图 RGB/尺寸/裁剪/方向 checksum 正确。
  - §3.4：ObsPacket / CanonicalActionChunk 协议不变量，越界动作拒绝下发。

约束：纯 CPU + pytest + unittest.mock，无 GPU 依赖；严禁修改被测源码；
      模拟数据严格对齐 7 维动作 [dx,dy,dz,dax,day,daz,gripper] 与真实协议结构。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.pi05.src import pi05 as pi05_mod
from services.pi05.src.action import CanonicalActionChunk
from services.pi05.src.observation import ObsPacket
from services.pi05.src.pi05 import (
    ACTION_DIM,
    CONTROL_HZ,
    EXPIRES_AFTER_MS,
    FIXED_TEST_IMAGE,
    FIXED_TEST_IMAGE_CHECKSUM,
    FRAME_ID,
    MAX_ROTATION_RAD,
    MAX_TRANSLATION_M,
    MOCK_CHUNK_LEN,
    SOURCE_POLICY,
    SPACE_ID,
    Pi05Executor,
    _image_checksum,
    _prep_image,
)

# 影响执行器初始化的环境变量全集（fixture 统一清空，保证初态确定）
_PI05_ENV_VARS = (
    "PI05_MODE",
    "PI05_CONFIG_NAME",
    "PI05_CHECKPOINT_DIR",
    "PI05_WS_HOST",
    "PI05_WS_PORT",
    "PI05_NORM_STATS_PATH",
    "PI05_CHECKPOINT_SHA",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_pi05_env(monkeypatch):
    """清空所有 PI05_* 环境变量，保证执行器从确定初态初始化。"""
    for var in _PI05_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def mock_executor(clean_pi05_env):
    """Mock 模式（dummy）执行器：未加载真实模型。"""
    return Pi05Executor()


@pytest.fixture
def sample_rgb_front():
    """确定性前视 RGB 图像 uint8[480,640,3]（非固定测试图，避免触发像素审计分支）。"""
    rng = np.random.RandomState(20260724)
    return rng.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_rgb_wrist():
    """确定性腕部 RGB 图像 uint8[224,224,3]。"""
    rng = np.random.RandomState(20260725)
    return rng.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_observation(sample_rgb_front, sample_rgb_wrist):
    """按 ObsPacket schema 构造的合规观测包（方案书 §3.4）。"""
    return ObsPacket(
        episode_id="test-ep-001",
        step_id=7,
        timestamp_ns=1_700_000_000_000_000_000,
        rgb_front=sample_rgb_front,
        rgb_wrist=sample_rgb_wrist,
        robot_state=np.arange(8, dtype=np.float32),
        instruction="pick up the red cylinder and place it into cell row 2 col 3",
        runtime_flags={"terminated": False, "truncated": False, "camera_ok": True},
    )


def _make_minimal_obs(step_id: int = 1, episode_id: str = "mini") -> ObsPacket:
    """构造最小可用 ObsPacket（黑图 + 零状态），供不依赖图像内容的用例复用。"""
    return ObsPacket(
        episode_id=episode_id,
        step_id=step_id,
        timestamp_ns=0,
        rgb_front=np.zeros((480, 640, 3), dtype=np.uint8),
        rgb_wrist=np.zeros((224, 224, 3), dtype=np.uint8),
        robot_state=np.zeros(8, dtype=np.float32),
        instruction=episode_id,
        runtime_flags={"terminated": False, "truncated": False, "camera_ok": True},
    )


# ---------------------------------------------------------------------------
# 用例 1：执行器初始化
# ---------------------------------------------------------------------------
def test_executor_init(mock_executor):
    """用例1：Mock 模式初始化——验证模型挂载状态、设备标识、运行时标志正确。"""
    # Mock 模式未加载真实策略
    assert mock_executor.mode == "dummy"
    assert mock_executor._policy_type == "mock"
    assert mock_executor._policy is None
    # 运行时状态标志处于初态
    assert mock_executor._pending_chunk is None
    assert mock_executor._pending_generated_step == -1
    assert mock_executor._current_episode_id is None
    assert mock_executor._last_latency_ms is None
    assert mock_executor._last_truncation_count == 0
    # 健康检查字段齐全，Mock 模式不查显存
    hc = mock_executor.health_check()
    assert hc["mode"] == "dummy"
    assert hc["policy_type"] == "mock"
    assert hc["config_name"] == "pi05_droid"
    assert hc["vram_usage_mb"] is None
    assert hc["last_latency_ms"] is None


def test_executor_init_real_mode_degrades_when_no_client(clean_pi05_env, monkeypatch):
    """用例1补充：请求 real 模式但无可用策略客户端时降级到 mock（方案书 §3.3）。"""
    monkeypatch.setenv("PI05_MODE", "real")
    with patch.object(pi05_mod, "make_policy_client", return_value=None):
        ex = Pi05Executor()
    assert ex.mode == "dummy"
    assert ex._policy_type == "mock"
    assert ex._policy is None


def test_executor_init_real_mode_mounts_client(clean_pi05_env, monkeypatch):
    """用例1补充：real 模式挂载 mock 策略客户端——验证挂载、类型、checkpoint_sha。"""
    monkeypatch.setenv("PI05_MODE", "real")
    monkeypatch.setenv("PI05_CHECKPOINT_DIR", "/fake/ckpt")
    monkeypatch.setenv("PI05_CHECKPOINT_SHA", "deadbeefcafef00d")

    fake_policy = MagicMock()
    fake_policy.client_type = "local"
    fake_policy.checkpoint_dir = "/fake/ckpt"
    with patch.object(pi05_mod, "make_policy_client", return_value=fake_policy):
        ex = Pi05Executor()

    assert ex.mode == "real"
    assert ex._policy_type == "local"
    assert ex._policy is fake_policy
    # 显式指定的 checkpoint_sha 被记录，不再走 _compute_dir_sha
    assert ex._checkpoint_sha == "deadbeefcafef00d"
    assert ex.health_check()["checkpoint_sha"] == "deadbeefcafef00d"


# ---------------------------------------------------------------------------
# 用例 2：观察量预处理（含像素审计）
# ---------------------------------------------------------------------------
def test_process_observation(mock_executor, sample_observation):
    """用例2：观察量预处理——维度/通道/dtype + 像素审计（RGB 非 BGR、HWC 非 CHW、无翻转）。

    方案书 Table 23 Row6：适配器传原始 RGB，resize/pad/normalize 由 openpi input_transform
    内部完成；适配器只保证 uint8/HWC/RGB/连续内存。
    """
    example = mock_executor._build_example(sample_observation)

    # openpi example 关键字段存在
    assert "observation/exterior_image_1_left" in example
    assert "observation/wrist_image_left" in example
    assert "observation/state" in example
    assert example["prompt"] == sample_observation.instruction

    # state 透传：float32，未做 (state - mean) / std 预归一化
    state = example["observation/state"]
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state, sample_observation.robot_state)

    # 前视 RGB 预处理：uint8 / HWC（ndim=3, shape[2]=3）/ 连续内存
    rgb = example["observation/exterior_image_1_left"]
    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3  # HWC 而非 CHW
    assert rgb.shape[2] == 3  # 3 通道
    assert rgb.flags["C_CONTIGUOUS"]
    # 像素与输入完全一致：无翻转、无通道交换、无 resize
    np.testing.assert_array_equal(rgb, sample_observation.rgb_front)

    # 腕部图同样保持 HWC/RGB
    wrist = example["observation/wrist_image_left"]
    assert wrist.dtype == np.uint8
    assert wrist.ndim == 3 and wrist.shape[2] == 3
    np.testing.assert_array_equal(wrist, sample_observation.rgb_wrist)


def test_process_observation_pixel_audit(mock_executor, caplog):
    """用例2·像素审计：固定测试图 RGB 非 BGR、HWC 非 CHW、无翻转（方案书 §7.5 image_pipeline）。"""
    # 1) verify_pixel_pipeline 入口应通过
    assert mock_executor.verify_pixel_pipeline() is True

    # 2) 固定测试图预处理后 checksum 不变 -> 未被破坏
    prepared = _prep_image(FIXED_TEST_IMAGE)
    assert _image_checksum(prepared) == FIXED_TEST_IMAGE_CHECKSUM
    assert prepared.dtype == np.uint8
    assert prepared.shape == (480, 640, 3)  # HWC 而非 CHW

    # 3) R 通道（index 0）保持横向渐变 -> 未发生 BGR 通道交换
    assert np.array_equal(prepared[:, :, 0], FIXED_TEST_IMAGE[:, :, 0])
    # 未上下/左右翻转：整体像素一致
    np.testing.assert_array_equal(prepared, FIXED_TEST_IMAGE)

    # 4) 传入固定测试图触发 _pixel_audit_if_test，应记录“像素审计通过”
    obs = ObsPacket(
        episode_id="audit",
        step_id=0,
        timestamp_ns=0,
        rgb_front=FIXED_TEST_IMAGE.copy(),
        rgb_wrist=None,
        robot_state=np.zeros(8, dtype=np.float32),
        instruction="audit",
        runtime_flags={},
    )
    with caplog.at_level("INFO", logger="pi05_executor"):
        mock_executor._pixel_audit_if_test(obs)
    assert any("像素审计通过" in r.message for r in caplog.records)


def test_prep_image_rejects_bad_shape():
    """用例2补充：非法图像形状被拒绝（保护像素管线不变量）。"""
    with pytest.raises(ValueError, match="rgb 图像形状非法"):
        _prep_image(np.zeros((480, 640), dtype=np.uint8))  # 缺通道
    with pytest.raises(ValueError, match="rgb 图像形状非法"):
        _prep_image(np.zeros((480, 640, 4), dtype=np.uint8))  # 非 3 通道


# ---------------------------------------------------------------------------
# 用例 3：动作切片与 Chunking
# ---------------------------------------------------------------------------
def test_action_chunking(mock_executor, sample_observation):
    """用例3：Mock 输出 [Batch, Horizon, 7] -> CanonicalActionChunk 切片维度/序列长度正确。"""
    chunk = mock_executor.infer(sample_observation)

    assert isinstance(chunk, CanonicalActionChunk)
    # 维度与序列长度
    assert chunk.actions.shape == (MOCK_CHUNK_LEN, ACTION_DIM)
    assert chunk.actions.dtype == np.float32
    # 协议字段（方案书 §3.4 CanonicalActionChunk v1）
    assert chunk.space_id == SPACE_ID
    assert chunk.frame == FRAME_ID
    assert chunk.control_hz == CONTROL_HZ
    assert chunk.source_policy == SOURCE_POLICY
    assert chunk.generated_step == sample_observation.step_id
    assert chunk.expires_after_ms == EXPIRES_AFTER_MS
    # 待执行动作队列已记录（方案书 §3.3.1 Para186：切换时清空）
    assert mock_executor._pending_chunk is not None
    assert mock_executor._pending_chunk.shape == (MOCK_CHUNK_LEN, ACTION_DIM)
    assert mock_executor._pending_generated_step == sample_observation.step_id
    # Mock 动作均在安全限幅范围内（不触发截断）
    assert mock_executor._last_truncation_count == 0
    assert np.all(np.abs(chunk.actions[:, 0:3]) <= MAX_TRANSLATION_M)
    assert np.all(np.abs(chunk.actions[:, 3:6]) <= MAX_ROTATION_RAD)


def test_action_chunking_pads_short_dim(clean_pi05_env, monkeypatch):
    """用例3补充：模型输出不足 7 维时适配器 pad 0（方案书 §3.3：不足维度由适配器 padding）。"""
    monkeypatch.setenv("PI05_MODE", "real")
    fake_policy = MagicMock()
    fake_policy.client_type = "local"
    fake_policy.checkpoint_dir = None
    fake_policy.infer.return_value = {
        "actions": np.full((4, 5), 0.001, dtype=np.float32)
    }
    with patch.object(pi05_mod, "make_policy_client", return_value=fake_policy):
        ex = Pi05Executor()

    chunk = ex.infer(_make_minimal_obs(step_id=1))
    assert chunk.actions.shape == (4, ACTION_DIM)
    # 后 2 维被 pad 为 0
    np.testing.assert_array_equal(chunk.actions[:, 5:], 0.0)


def test_action_chunking_promotes_1d(clean_pi05_env, monkeypatch):
    """用例3补充：1D 动作输入升维为 [1, 7]。"""
    monkeypatch.setenv("PI05_MODE", "real")
    fake_policy = MagicMock()
    fake_policy.client_type = "local"
    fake_policy.checkpoint_dir = None
    fake_policy.infer.return_value = {
        "actions": np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    }
    with patch.object(pi05_mod, "make_policy_client", return_value=fake_policy):
        ex = Pi05Executor()

    chunk = ex.infer(_make_minimal_obs(step_id=2))
    assert chunk.actions.shape == (1, ACTION_DIM)
    np.testing.assert_allclose(chunk.actions[0, 6], 1.0)  # 夹爪 1.0 保留


# ---------------------------------------------------------------------------
# 用例 4：归一化/反归一化（透传 + 无双重反归一化）
# ---------------------------------------------------------------------------
def test_norm_stats_no_double_denormalization(clean_pi05_env, monkeypatch):
    """用例4：适配器透传，不做二次反归一化（方案书 §3.3.1 Para185/186）。

    归一化（state 减 mean 除 std）由 openpi input_transform 在 policy.infer 内完成；
    反归一化（输出乘 std 加 mean）由 openpi output_transform 在 policy.infer 内完成。
    适配器只透传原始 state、直接使用已反归一化的 actions，严禁二次反归一化。
    """
    monkeypatch.setenv("PI05_MODE", "real")

    # 已知 norm_stats（仅用于断言适配器不应用它们做归一化/反归一化）
    mean = np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5], dtype=np.float32)

    raw_state = np.arange(8, dtype=np.float32)
    captured: dict = {}

    # 模拟 openpi output_transform 已反归一化的物理动作（均在安全限幅内，仅夹爪需取整）
    openpi_actions = np.array(
        [[0.01, -0.01, 0.005, 0.01, -0.01, 0.005, 0.499]], dtype=np.float32
    )

    def fake_infer(example):
        captured["state_passed"] = example["observation/state"].copy()
        return {"actions": openpi_actions.copy()}

    fake_policy = MagicMock()
    fake_policy.client_type = "local"
    fake_policy.checkpoint_dir = None
    fake_policy.infer.side_effect = fake_infer
    with patch.object(pi05_mod, "make_policy_client", return_value=fake_policy):
        ex = Pi05Executor()

    obs = _make_minimal_obs(step_id=3)
    obs.robot_state = raw_state
    chunk = ex.infer(obs)

    # 1) 归一化由 openpi 完成：适配器透传原始 state，未做 (state - mean) / std
    np.testing.assert_array_equal(captured["state_passed"], raw_state)

    # 2) 反归一化由 openpi 完成：适配器直接使用已反归一化输出，未再乘 std 加 mean
    #    期望 = openpi 输出经限幅/取整（夹爪 0.499 < 0.5 -> 0.0），平移/旋转均在限幅内不截断
    expected = openpi_actions.copy()
    expected[:, 6] = 0.0
    np.testing.assert_allclose(chunk.actions, expected)

    # 3) 显式确认不出现双重反归一化：若误做 (out * std + mean)，结果会与本输出明显不同
    double_denorm = openpi_actions * std + mean
    assert not np.allclose(chunk.actions, double_denorm)


# ---------------------------------------------------------------------------
# 用例 5：安全限幅
# ---------------------------------------------------------------------------
def test_safety_clamping(mock_executor, caplog):
    """用例5：平移>2cm、旋转>5° 被截断到安全阈值并触发报警（方案书 Table 69 Row7 / 附录B）。"""
    # 构造超限动作：平移 0.05m(>0.02)、旋转 0.2rad(>0.0873)、夹爪 0.3(非 0/1)
    actions = np.array(
        [
            [0.05, -0.05, 0.001, 0.20, -0.20, 0.000, 0.30],
            [-0.03, 0.04, -0.001, -0.10, 0.15, 0.050, 0.70],
        ],
        dtype=np.float32,
    )
    with caplog.at_level("WARNING", logger="pi05_executor"):
        clipped = mock_executor._clip_actions(actions)

    # 平移截断到 [-0.02, 0.02]（米）
    assert np.all(clipped[:, 0:3] >= -MAX_TRANSLATION_M)
    assert np.all(clipped[:, 0:3] <= MAX_TRANSLATION_M)
    # 旋转截断到 [-0.0873, 0.0873]（弧度 ≈ 5°）
    assert np.all(clipped[:, 3:6] >= -MAX_ROTATION_RAD)
    assert np.all(clipped[:, 3:6] <= MAX_ROTATION_RAD)
    # 夹爪仅允许 0.0 / 1.0
    assert set(np.unique(clipped[:, 6])).issubset({0.0, 1.0})
    # 截断计数 > 0
    assert mock_executor._last_truncation_count > 0
    # 截断时触发报警日志
    assert any("截断" in r.message for r in caplog.records)


def test_safety_clamping_boundary(mock_executor):
    """用例5补充：恰好等于阈值的动作不被截断（边界值合规）。"""
    boundary = np.array(
        [
            [
                MAX_TRANSLATION_M,
                -MAX_TRANSLATION_M,
                0.0,
                MAX_ROTATION_RAD,
                -MAX_ROTATION_RAD,
                0.0,
                1.0,
            ]
        ],
        dtype=np.float32,
    )
    clipped = mock_executor._clip_actions(boundary)
    np.testing.assert_allclose(clipped, boundary)
    assert mock_executor._last_truncation_count == 0


# ---------------------------------------------------------------------------
# 用例 6：越界拒绝
# ---------------------------------------------------------------------------
def test_out_of_bounds_rejection(mock_executor):
    """用例6：NaN/Inf/维度不匹配/shape 错误均被拒绝下发（方案书 §3.4 协议不变量）。"""
    # NaN
    with pytest.raises(ValueError, match="NaN/Inf"):
        mock_executor._clip_actions(
            np.array([[np.nan, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        )
    # Inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        mock_executor._clip_actions(
            np.array([[0, 0, 0, 0, 0, 0, np.inf]], dtype=np.float32)
        )
    # 维度不匹配：列数 != 7
    with pytest.raises(ValueError, match="形状非法"):
        mock_executor._clip_actions(np.zeros((3, 6), dtype=np.float32))
    # shape 错误：1D 输入
    with pytest.raises(ValueError, match="形状非法"):
        mock_executor._clip_actions(np.zeros((7,), dtype=np.float32))
    # shape 错误：3D 输入
    with pytest.raises(ValueError, match="形状非法"):
        mock_executor._clip_actions(np.zeros((2, 7, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# 用例 7：异常处理与重置
# ---------------------------------------------------------------------------
def test_exception_during_inference_propagates(clean_pi05_env, monkeypatch):
    """用例7·异常处理：real 模式推理抛出异常时不被吞没，上抛交由总 Agent 处理。

    方案书 §3.3.1 Para186：失败切换由总 Agent 触发（cancel_pending_chunk + 切换执行器），
    适配器不静默吞掉推理异常，保证 fail-fast。
    """
    monkeypatch.setenv("PI05_MODE", "real")
    fake_policy = MagicMock()
    fake_policy.client_type = "local"
    fake_policy.checkpoint_dir = None
    fake_policy.infer.side_effect = RuntimeError("openpi inference boom")
    with patch.object(pi05_mod, "make_policy_client", return_value=fake_policy):
        ex = Pi05Executor()

    with pytest.raises(RuntimeError, match="openpi inference boom"):
        ex.infer(_make_minimal_obs(step_id=1))


def test_reset_clears_state(clean_pi05_env, monkeypatch):
    """用例7·重置：reset() 后缓存、历史序列、状态向量恢复初始（方案书 §3.3.1 Para186）。"""
    monkeypatch.setenv("PI05_MODE", "dummy")
    with patch.object(pi05_mod, "make_policy_client", return_value=None):
        ex = Pi05Executor()

    # 挂载一个 mock policy 以验证 reset 经 cancel_pending_chunk 调用 clear_cache
    fake_policy = MagicMock()
    ex._policy = fake_policy

    # 产生一次成功推理，积累运行时状态
    obs = _make_minimal_obs(step_id=5, episode_id="reset-ep")
    ex.infer(obs)
    assert ex._pending_chunk is not None
    assert ex._current_episode_id == "reset-ep"
    assert ex._last_latency_ms is not None

    # reset 清空全部状态
    ex.reset()
    assert ex._pending_chunk is None
    assert ex._pending_generated_step == -1
    assert ex._current_episode_id is None
    assert ex._last_latency_ms is None
    assert ex._last_truncation_count == 0
    # reset 经 cancel_pending_chunk 清空了客户端缓存
    fake_policy.clear_cache.assert_called_once()


def test_cancel_pending_chunk_clears_queue(clean_pi05_env, monkeypatch):
    """用例7补充：cancel_pending_chunk 单独清空动作队列与缓存（方案书 §3.3.1 Para186）。"""
    monkeypatch.setenv("PI05_MODE", "dummy")
    with patch.object(pi05_mod, "make_policy_client", return_value=None):
        ex = Pi05Executor()
    fake_policy = MagicMock()
    ex._policy = fake_policy

    ex.infer(_make_minimal_obs(step_id=9, episode_id="cancel-ep"))
    assert ex._pending_chunk is not None
    assert ex._pending_generated_step == 9

    ex.cancel_pending_chunk()
    assert ex._pending_chunk is None
    assert ex._pending_generated_step == -1
    fake_policy.clear_cache.assert_called_once()


# ---------------------------------------------------------------------------
# 用例 8：descriptor 属性（体系B ExecutorDescriptor 对齐）
# ---------------------------------------------------------------------------
def test_descriptor_dummy_without_sha_env(clean_pi05_env, monkeypatch):
    """用例8：dummy 模式缺少 SHA 环境变量时 descriptor 使用占位 SHA 不崩溃。

    方案书 interface-contracts.md §4：checkpoint_sha/norm_stats_sha 必须为
    sha256:<64hex> 格式。dummy 模式允许占位值，real 模式必须拒绝。
    """
    monkeypatch.setenv("PI05_MODE", "dummy")
    with patch.object(pi05_mod, "make_policy_client", return_value=None):
        ex = Pi05Executor()

    desc = ex.descriptor
    assert desc.name == "pi05"
    assert desc.action_contract_version == "1.0"
    assert "pick_place" in desc.task_types
    # 占位 SHA 必须符合 sha256:<64hex> 格式
    from src.industrial_agent.executor import is_pinned_artifact_digest

    assert is_pinned_artifact_digest(desc.checkpoint_sha), (
        f"占位 checkpoint_sha 格式不符：{desc.checkpoint_sha}"
    )
    assert is_pinned_artifact_digest(desc.norm_stats_sha), (
        f"占位 norm_stats_sha 格式不符：{desc.norm_stats_sha}"
    )


def test_descriptor_real_mode_requires_sha(clean_pi05_env, monkeypatch):
    """用例8补充：real 模式缺少 SHA 环境变量时 descriptor 抛 ValueError。

    生产 real 模式必须配置 PI05_CHECKPOINT_SHA / PI05_NORM_STATS_SHA。
    """
    monkeypatch.setenv("PI05_MODE", "real")
    fake_policy = MagicMock()
    fake_policy.client_type = "local"
    fake_policy.checkpoint_dir = None
    with patch.object(pi05_mod, "make_policy_client", return_value=fake_policy):
        ex = Pi05Executor()

    with pytest.raises(ValueError, match="PI05_CHECKPOINT_SHA"):
        _ = ex.descriptor


def test_descriptor_with_valid_sha_env(clean_pi05_env, monkeypatch):
    """用例8补充：设置合规 SHA 环境变量后 descriptor 正常返回。"""
    monkeypatch.setenv("PI05_MODE", "dummy")
    monkeypatch.setenv(
        "PI05_CHECKPOINT_SHA",
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    monkeypatch.setenv(
        "PI05_NORM_STATS_SHA",
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    with patch.object(pi05_mod, "make_policy_client", return_value=None):
        ex = Pi05Executor()

    desc = ex.descriptor
    assert "aaaa" in desc.checkpoint_sha
    assert "bbbb" in desc.norm_stats_sha
