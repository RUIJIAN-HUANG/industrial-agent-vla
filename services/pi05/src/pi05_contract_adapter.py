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
from industrial_agent.sync_contract import canonical_observed_state_7d
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
        # 同一个 π0.5 服务按生命周期子任务服务两只手臂；arm_id 由 Planner
        # 注入 TaskSchema.metadata，缺省只保留单臂 V2 的 Arm_A 兼容路径。
        arm_id = task.metadata.get("arm_id", "Arm_A")
        if arm_id not in {"Arm_A", "Arm_B"}:
            raise ExecutorError(
                FailureCode.INVALID_TASK,
                f"π0.5 task arm_id must be Arm_A or Arm_B, got {arm_id!r}",
            )
        arm_key = "arm_a" if arm_id == "Arm_A" else "arm_b"
        camera_key = "arm_a_rgb" if arm_id == "Arm_A" else "arm_b_rgb"
        camera = observation.data.get("camera", {})
        robot = observation.data.get("robot", {})
        if not isinstance(camera, Mapping):
            camera = {}
        if not isinstance(robot, Mapping):
            robot = {}

        # ---- 当前控制臂的 top-view：只接受冻结 ImageReference ----
        raw_front = camera.get(camera_key)
        if not is_image_reference(raw_front):
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                f"camera.{camera_key} must be a frozen ImageReference",
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
            "arm_id": arm_id,
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

        # ---- 当前控制臂状态 ----
        arm_state = robot.get(arm_key, {})
        if not isinstance(arm_state, Mapping):
            arm_state = {}
        try:
            state_7d = canonical_observed_state_7d(
                arm_state.get("tcp_pose_m_rad"),
                arm_state.get("state"),
                arm_state.get("gripper_open"),
            )
        except (TypeError, ValueError) as exc:
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                f"robot.{arm_key} cannot produce canonical state_7d: {exc}",
            ) from exc
        robot_state = np.asarray(state_7d, dtype=np.float32)

        # 优先使用 context.original_instruction（总控设定的冻结指令）
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
            runtime_flags={"arm_id": arm_id},
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
