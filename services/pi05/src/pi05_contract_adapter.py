"""π0.5 体系B 契约适配器（叠加层）。

把 src/industrial_agent/executor.py 的 Executor Protocol（体系B）翻译为
services/pi05/src/pi05.py 的 Pi05Executor.infer()（体系A），再把 CanonicalActionChunk
包成体系B 的 ActionChunk。不修改任何冻结文件；结构子类型实现，不显式继承。

方案书出处：interface-contracts.md §4/§7；agent-framework.md §9 统一 7 维动作合同。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from industrial_agent.contracts import ActionChunk, Observation, TaskSchema
from industrial_agent.errors import ExecutorError, FailureCode, ImageCasError
from industrial_agent.executor import ExecutionContext, ExecutorDescriptor
from industrial_agent.service_images import CasRequestImageResolver
from services.pi05.src.observation import ObsPacket, is_image_reference
from services.pi05.src.pi05 import Pi05Executor


class Pi05ContractAdapter:
    """体系B Executor Protocol 的 π0.5 适配实现（结构子类型，不显式继承）。

    职责：把 (TaskSchema, Observation, ExecutionContext) 拆成 ObsPacket，
    调 Pi05Executor.infer()，再把 CanonicalActionChunk 包成 ActionChunk。
    """

    def __init__(
        self,
        executor: Pi05Executor | None = None,
        *,
        resolver: CasRequestImageResolver | None = None,
    ) -> None:
        self._executor: Pi05Executor = executor or Pi05Executor()
        self._resolver = resolver

    @property
    def descriptor(self) -> ExecutorDescriptor:
        return self._executor.descriptor

    def health(self) -> bool:
        try:
            info = self._executor.health_check()
            return info.get("mode") is not None
        except Exception:
            return False

    def plan(
        self, task: TaskSchema, observation: Observation, context: ExecutionContext
    ) -> ActionChunk:
        # 方案书 interface-contracts.md §7.3 / executor.py Pi05Adapter.plan()：
        # Pi05Adapter 从 observation.data.robot.arm_a 提取状态，
        # 从 observation.data.camera.arm_a_rgb 提取图像（camera_id=CAM_A_TOP）。
        # 框架 _phase_vla_inputs 要求 camera.arm_a_rgb 必须存在，不做 fallback。
        camera = observation.data.get("camera", {})
        robot = observation.data.get("robot", {})
        if not isinstance(camera, Mapping):
            camera = {}
        if not isinstance(robot, Mapping):
            robot = {}

        # ---- arm_a_rgb：只接受冻结 ImageReference，并统一通过公共 CAS resolver ----
        raw_front = camera.get("arm_a_rgb")
        if not is_image_reference(raw_front):
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "camera.arm_a_rgb must be a frozen ImageReference",
            )
        if self._resolver is None:
            raise ExecutorError(
                FailureCode.CAS_UNAVAILABLE,
                "Pi05ContractAdapter requires the shared CasRequestImageResolver",
                retryable=True,
            )

        # 冻结三相机配置没有腕部相机；resolver 会拒绝非 null wrist_image。
        raw_wrist = camera.get("wrist_image")
        request = {
            "executor": "pi05",
            "model_input": {
                "observation": {
                    "camera": {
                        "full_image": raw_front,
                        "wrist_image": raw_wrist,
                    }
                }
            },
        }
        try:
            resolved = self._resolver.resolve_vla_request(request)
        except ImageCasError as exc:
            raise ExecutorError(
                exc.code,
                str(exc),
                retryable=exc.retryable,
            ) from exc
        rgb_front = resolved.full_image.rgb
        rgb_wrist = None

        # ---- arm_a.state（框架 _phase_vla_inputs arm_key="arm_a"）----
        arm_a = robot.get("arm_a", {})
        if not isinstance(arm_a, Mapping):
            arm_a = {}
        robot_raw = arm_a.get("state", arm_a.get("tcp_pose_m_rad"))
        if robot_raw is None:
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "robot.arm_a.state is required; zero-state fallback is forbidden",
            )
        robot_state = np.asarray(robot_raw, dtype=np.float32)
        if robot_state.ndim != 1 or robot_state.size == 0:
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "robot.arm_a.state must be a non-empty one-dimensional vector",
            )
        if not np.all(np.isfinite(robot_state)):
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "robot.arm_a.state contains NaN or Infinity",
            )

        # 优先使用 context.original_instruction（FixedDualVLAPlanner 设定的冻结指令）
        # 回退 task.instruction（方案书 executor.py Pi05Adapter.plan() L771）
        instruction = context.original_instruction or task.instruction

        obs_packet = ObsPacket(
            episode_id=context.run_id,
            step_id=context.step_id,
            timestamp_ns=observation.timestamp_ms * 1_000_000,
            rgb_front=rgb_front,  # type: ignore[arg-type]
            rgb_wrist=rgb_wrist,
            robot_state=robot_state,
            instruction=instruction,
            runtime_flags={},
        )
        try:
            canonical = self._executor.infer(obs_packet)
        except (ValueError, TypeError, KeyError) as exc:
            # 输入/参数错误（数据非法、维度不匹配等），重试无效
            raise ExecutorError(
                FailureCode.EXECUTOR_RUNTIME,
                f"π0.5 inference failed (input error): {exc}",
                retryable=False,
            ) from exc
        except Exception as exc:
            # 运行时错误（网络超时、模型 OOM 等），可重试
            raise ExecutorError(
                FailureCode.EXECUTOR_RUNTIME,
                f"π0.5 inference failed (runtime): {exc}",
                retryable=True,
            ) from exc
        return self._executor.to_action_chunk(canonical, task.task_id, "pi05")

    def cancel(self, task_id: str, reason: str) -> None:
        try:
            self._executor.cancel_pending_chunk()
        except Exception:
            pass
