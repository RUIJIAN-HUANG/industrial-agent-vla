"""Lazy Isaac Lab Pink IK adapter for the existing Isaac Sim Franka scene.

The collection entry still owns the Isaac ``SingleArticulation`` objects and
the canonical 10 Hz action contract.  This adapter only converts one desired
TCP pose into seven arm-joint position targets.  Imports stay lazy because
Isaac Lab/Pink must be imported after the SimulationApp has started.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from simulation.pink_urdf_compat import prepare_pink_compatible_urdf


_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
_ROLLOUT_MAX_ITERATIONS = 32
_ROLLOUT_JOINT_TOLERANCE_RAD = 1e-5


def _wxyz_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a 3-by-3 rotation matrix for a finite wxyz quaternion."""

    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("orientation must contain four finite wxyz values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("orientation cannot have zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _joint_indices(all_joint_names: list[str]) -> list[int]:
    """Resolve the seven Panda arm joints without assuming USD DOF order."""

    missing = [name for name in _ARM_JOINTS if name not in all_joint_names]
    if missing:
        raise RuntimeError("Pink requires missing Franka joints: " + ", ".join(missing))
    return [all_joint_names.index(name) for name in _ARM_JOINTS]


def _resolve_franka_mesh_root(urdf_path: str) -> str:
    """Locate the Isaac Sim package root used by Franka package:// meshes."""

    urdf = Path(urdf_path).resolve()
    relative_mesh = (
        Path("exts")
        / "isaacsim.asset.importer.urdf"
        / "data"
        / "urdf"
        / "robots"
        / "franka_description"
        / "meshes"
        / "collision"
        / "link0.stl"
    )
    for parent in urdf.parents:
        mesh = parent / relative_mesh
        if mesh.is_file():
            return str(mesh.parents[3])
    raise RuntimeError(
        f"Could not resolve Isaac Sim Franka mesh package root from URDF {urdf_path!r}"
    )


def _resolve_default_franka_urdf() -> str:
    """Find Lula's Franka URDF without importing its conflicting extension."""

    spec = importlib.util.find_spec("isaacsim.core.api")
    origins = (
        []
        if spec is None
        else [spec.origin, *list(spec.submodule_search_locations or [])]
    )
    relative = (
        Path("isaacsim.robot_motion.motion_generation")
        / "motion_policy_configs"
        / "franka"
        / "lula_franka_gen.urdf"
    )
    for origin in origins:
        if origin is None:
            continue
        for parent in Path(origin).resolve().parents:
            if parent.name != "exts":
                continue
            candidate = parent / relative
            if candidate.is_file():
                return str(candidate)
    raise RuntimeError(
        "Could not locate the Isaac Sim Franka URDF without importing Lula"
    )


# Safety envelope for one canonical keyboard action.  The Pink controller is
# iterated predictively below because its native output is one differential
# update.  Without an envelope, those differential updates can accumulate into
# a large joint-space jump while chasing a 30 mm Cartesian target.
_MAX_ACTION_JOINT_DELTA_RAD = 0.12


class PinkFrankaAdapter:
    """One native ``PinkIKController`` per existing Franka articulation."""

    def __init__(
        self,
        *,
        arms: Mapping[str, Any],
        lula_configs: Mapping[str, Mapping[str, Any]] | None = None,
        control_frame_name: str,
        device: str = "cuda:0",
    ) -> None:
        try:
            import pinocchio as pin
            from pink.tasks import DampingTask, FrameTask
            from isaaclab.controllers.pink_ik.null_space_posture_task import (
                NullSpacePostureTask,
            )
            from isaaclab.controllers.pink_ik.pink_ik import PinkIKController
            from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
        except ImportError as exc:
            raise RuntimeError(
                "Pink IK is unavailable. Activate mylab_env and source "
                "/home/xyz/isaacsim/setup_conda_env.sh before collection."
            ) from exc

        self._pin = pin
        self._controllers: dict[str, Any] = {}
        self._frame_tasks: dict[str, Any] = {}
        self._control_frame_names: dict[str, str] = {}
        self._controlled_indices: dict[str, list[int]] = {}
        self._diagnostics: dict[str, dict[str, Any]] = {}
        default_urdf_path = None
        if lula_configs is None:
            default_urdf_path = _resolve_default_franka_urdf()

        for arm_id, arm in arms.items():
            names = [str(name) for name in arm.dof_names]
            indices = _joint_indices(names)
            urdf_path = (
                str(lula_configs[arm_id].get("urdf_path", ""))
                if lula_configs is not None
                else str(default_urdf_path)
            )
            if not urdf_path:
                raise RuntimeError(f"Lula supplied no Franka URDF path for {arm_id}")

            mesh_root = _resolve_franka_mesh_root(urdf_path)

            pink_urdf_path, renamed_fixed_frames = prepare_pink_compatible_urdf(
                urdf_path
            )

            model = pin.buildModelFromUrdf(pink_urdf_path)
            frame_names = {frame.name for frame in model.frames}
            if control_frame_name not in frame_names:
                candidates = sorted(
                    name
                    for name in frame_names
                    if any(
                        word in name.lower() for word in ("hand", "gripper", "finger")
                    )
                )
                raise RuntimeError(
                    f"Pink URDF has no TCP frame {control_frame_name!r}; "
                    f"available candidates={candidates!r}"
                )

            current = np.asarray(arm.get_joint_positions(), dtype=float)
            if current.shape != (len(names),) or not np.all(np.isfinite(current)):
                raise RuntimeError(f"Invalid initial Franka joints for {arm_id}")
            initial_positions = dict(zip(names, current, strict=True))
            robot_cfg = SimpleNamespace(
                init_state=SimpleNamespace(joint_pos=initial_positions)
            )

            frame_task = FrameTask(
                control_frame_name,
                position_cost=20.0,
                orientation_cost=20.0,
                lm_damping=1.0,
                gain=0.7,
            )
            posture_task = NullSpacePostureTask(
                cost=0.35,
                lm_damping=1.0,
                gain=0.25,
                controlled_frames=[control_frame_name],
                controlled_joints=list(_ARM_JOINTS),
            )
            cfg = PinkIKControllerCfg(
                urdf_path=pink_urdf_path,
                mesh_path=mesh_root,
                num_hand_joints=0,
                variable_input_tasks=[frame_task, posture_task],
                fixed_input_tasks=[DampingTask(cost=0.02)],
                joint_names=list(_ARM_JOINTS),
                all_joint_names=names,
                articulation_name=f"v2_collection_{arm_id.lower()}",
                base_link_name="panda_link0",
                show_ik_warnings=True,
                fail_on_joint_limit_violation=True,
            )
            controller = PinkIKController(
                cfg=cfg,
                robot_cfg=robot_cfg,
                device=device,
                controlled_joint_indices=indices,
            )
            controller_tasks = getattr(controller, "_variable_input_tasks", None)
            if controller_tasks is None:
                controller_tasks = controller.cfg.variable_input_tasks
            active_frame_tasks = [
                task
                for task in controller_tasks
                if isinstance(task, FrameTask)
                and getattr(task, "frame", None) == control_frame_name
            ]
            if len(active_frame_tasks) != 1:
                raise RuntimeError(
                    f"Pink controller has {len(active_frame_tasks)} active "
                    f"tasks for frame {control_frame_name!r}"
                )

            self._controllers[arm_id] = controller
            self._frame_tasks[arm_id] = active_frame_tasks[0]
            self._control_frame_names[arm_id] = control_frame_name
            self._controlled_indices[arm_id] = indices
            self._diagnostics[arm_id] = {
                "backend": "pink",
                "urdf_path": urdf_path,
                "pink_compatible_urdf_path": pink_urdf_path,
                "mesh_path": mesh_root,
                "renamed_fixed_frames": [list(item) for item in renamed_fixed_frames],
                "control_frame_name": control_frame_name,
                "controlled_joint_names": list(_ARM_JOINTS),
                "controlled_joint_indices": list(indices),
                "null_space_reference": "episode_initial_joint_positions",
            }

    def controlled_indices(self, arm_id: str) -> list[int]:
        return list(self._controlled_indices[arm_id])

    def diagnostics(self, arm_id: str) -> dict[str, Any]:
        return dict(self._diagnostics[arm_id])

    def control_frame_pose_in_base(
        self,
        *,
        arm_id: str,
        joint_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a virtual Pink FK pose without writing to the articulation."""

        current = np.asarray(joint_positions, dtype=float)
        indices = self._controlled_indices[arm_id]
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise ValueError("joint_positions must be a finite vector")
        if not indices or max(indices) >= current.size:
            raise ValueError("joint_positions does not cover controlled joints")

        return self.frame_pose_in_base(
            arm_id=arm_id,
            frame_name=self._control_frame_names[arm_id],
            joint_positions=current,
        )

    def frame_pose_in_base(
        self,
        *,
        arm_id: str,
        frame_name: str,
        joint_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return any Pink frame pose for a virtual full articulation state."""

        current = np.asarray(joint_positions, dtype=float)
        indices = self._controlled_indices[arm_id]
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise ValueError("joint_positions must be a finite vector")
        if not indices or max(indices) >= current.size:
            raise ValueError("joint_positions does not cover controlled joints")
        controller = self._controllers[arm_id]
        configuration = getattr(controller, "pink_configuration", None)
        if configuration is None:
            raise RuntimeError(f"Pink configuration is unavailable for {arm_id}")
        ordering = np.asarray(
            getattr(controller, "isaac_lab_to_pink_ordering", range(current.size)),
            dtype=int,
        )
        if ordering.shape != (current.size,) or set(ordering.tolist()) != set(
            range(current.size)
        ):
            raise RuntimeError(f"Pink joint ordering is invalid for {arm_id}")
        configuration.update(current[ordering])
        transform = configuration.get_transform_frame_to_world(frame_name)
        position = np.asarray(transform.translation, dtype=float)
        rotation = np.asarray(transform.rotation, dtype=float)
        if (
            position.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(rotation))
        ):
            raise RuntimeError(f"Pink returned an invalid virtual FK pose for {arm_id}")
        return position.copy(), rotation.copy()

    def compute(
        self,
        *,
        arm_id: str,
        current_joint_positions: np.ndarray,
        target_position_base_m: np.ndarray,
        target_orientation_base_wxyz: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        """Compute an absolute seven-joint target for one keyboard action.

        Isaac Lab's Pink controller intentionally returns one differential
        configuration update.  The existing collection controller, however,
        consumes an absolute IK target once per 100 ms canonical action.  Roll
        Pink forward predictively here so a 30 mm keyboard command is not
        reduced to a barely visible single QP integration step.  The rollout
        changes neither the recorded action nor the live articulation state.
        """

        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")

        current = np.asarray(current_joint_positions, dtype=float)
        indices = self._controlled_indices[arm_id]
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise ValueError("current_joint_positions must be a finite vector")
        if not indices or max(indices) >= current.size:
            raise ValueError("current_joint_positions does not cover controlled joints")

        target = self._pin.SE3(
            _wxyz_rotation_matrix(target_orientation_base_wxyz),
            np.asarray(target_position_base_m, dtype=float),
        )
        self._frame_tasks[arm_id].set_target(target)

        controller = self._controllers[arm_id]
        predicted = current.copy()

        # Keep live and predicted configurations inside Pink/Pinocchio limits.
        # A small margin prevents physics overshoot and floating-point noise from
        # leaving the next QP with an invalid starting configuration.
        joint_limit_margin_rad = 0.02
        joint_limit_recovery_tolerance_rad = 1e-3
        joint_limit_state_recovered = False
        joint_limit_target_clamped = False
        safe_lower = None
        safe_upper = None

        pink_configuration = getattr(controller, "pink_configuration", None)
        pink_model = getattr(pink_configuration, "model", None)
        if pink_model is not None:
            controlled_count = len(indices)
            lower = np.asarray(pink_model.lowerPositionLimit, dtype=float).reshape(-1)
            upper = np.asarray(pink_model.upperPositionLimit, dtype=float).reshape(-1)
            if lower.size < controlled_count or upper.size < controlled_count:
                raise RuntimeError(f"Pink joint limit vector is too short for {arm_id}")

            lower = lower[:controlled_count]
            upper = upper[:controlled_count]
            finite_limits = (
                np.isfinite(lower)
                & np.isfinite(upper)
                & ((upper - lower) > 2.0 * joint_limit_margin_rad)
            )
            controlled_current = predicted[indices].copy()
            violation = np.where(
                finite_limits,
                np.maximum(
                    np.maximum(
                        lower - controlled_current,
                        controlled_current - upper,
                    ),
                    0.0,
                ),
                0.0,
            )
            if float(np.max(violation)) > joint_limit_recovery_tolerance_rad:
                raise RuntimeError(
                    f"{arm_id} exceeds Pink joint limits beyond recovery tolerance"
                )

            safe_lower = np.where(
                finite_limits,
                lower + joint_limit_margin_rad,
                -np.inf,
            )
            safe_upper = np.where(
                finite_limits,
                upper - joint_limit_margin_rad,
                np.inf,
            )
            recovered = np.clip(
                controlled_current,
                safe_lower,
                safe_upper,
            )
            joint_limit_state_recovered = not np.allclose(
                recovered,
                controlled_current,
                rtol=0.0,
                atol=1e-12,
            )
            predicted[indices] = recovered

        initial_controlled = predicted[indices].copy()
        iterations = 0
        last_step_rad = float("inf")
        targets = initial_controlled
        for iterations in range(1, _ROLLOUT_MAX_ITERATIONS + 1):
            result = controller.compute(predicted, dt_s)
            if hasattr(result, "detach"):
                result = result.detach()
            if hasattr(result, "cpu"):
                result = result.cpu()
            targets = np.asarray(result, dtype=float)
            if targets.shape != (7,) or not np.all(np.isfinite(targets)):
                raise RuntimeError(f"Pink returned invalid joint targets for {arm_id}")

            if safe_lower is not None and safe_upper is not None:
                bounded_targets = np.clip(targets, safe_lower, safe_upper)
                joint_limit_target_clamped = (
                    joint_limit_target_clamped
                    or not np.allclose(
                        bounded_targets,
                        targets,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                targets = bounded_targets

            last_step_rad = float(np.max(np.abs(targets - predicted[indices])))
            predicted[indices] = targets
            if last_step_rad <= _ROLLOUT_JOINT_TOLERANCE_RAD:
                break

        # Fail closed if predictive rollout asks for an implausibly large
        # joint-space jump for one keyboard action.  Preserve direction while
        # scaling the cumulative update back into the safety envelope.
        cumulative_delta = targets - initial_controlled
        cumulative_max_rad = float(np.max(np.abs(cumulative_delta)))
        rollout_clamped = cumulative_max_rad > _MAX_ACTION_JOINT_DELTA_RAD
        if rollout_clamped:
            targets = initial_controlled + cumulative_delta * (
                _MAX_ACTION_JOINT_DELTA_RAD / cumulative_max_rad
            )
        self._diagnostics[arm_id].update(
            {
                "joint_limit_margin_rad": joint_limit_margin_rad,
                "joint_limit_state_recovered": joint_limit_state_recovered,
                "joint_limit_target_clamped": joint_limit_target_clamped,
                "rollout_clamped": rollout_clamped,
                "rollout_cumulative_delta_rad": cumulative_max_rad,
                "rollout_action_joint_limit_rad": _MAX_ACTION_JOINT_DELTA_RAD,
                "rollout_iterations": iterations,
                "rollout_last_step_rad": last_step_rad,
                "rollout_total_joint_delta_rad": float(
                    np.max(np.abs(targets - initial_controlled))
                ),
            }
        )
        return targets.copy()
