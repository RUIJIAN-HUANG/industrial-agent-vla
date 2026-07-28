"""Run the member-B G0 acceptance checks inside Isaac Sim.

This file must be launched with Isaac Sim's ``python.sh`` rather than the
system Python interpreter. It deliberately keeps every Omniverse import after
``SimulationApp`` starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v1.json"
DEFAULT_SCENE = SCRIPT_DIR / "generated" / "single_bin_scene_v1.usda"

REQUIRED_PRIMS = (
    "/World/Robots/Arm_A",
    "/World/Robots/Arm_B",
    "/World/Parts/P01",
    "/World/Parts/P02",
    "/World/Parts/P03",
    "/World/Parts/P04",
    "/World/Bins/Bin_01",
    "/World/Stations/PACK_STATION",
    "/World/Stations/HANDOFF_CENTER",
    "/World/Stations/FINISHED_01",
    "/World/Cameras/CAM_A_TOP",
    "/World/Cameras/CAM_HANDOFF",
    "/World/Cameras/CAM_B_TOP",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Isaac Sim 5.1 G0 and D03 smoke checks."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--franka-usd", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument(
        "--reset-settle-steps",
        type=int,
        default=120,
        help="Physics steps to wait after every reset (default: 120).",
    )
    parser.add_argument(
        "--capture-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _all_finite(values: Any) -> bool:
    if values is None:
        return False
    flattened = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(flattened, list):
        flattened = [flattened]

    def walk(items: list[Any]) -> bool:
        for item in items:
            if isinstance(item, list):
                if not walk(item):
                    return False
            else:
                try:
                    if not math.isfinite(float(item)):
                        return False
                except (TypeError, ValueError):
                    return False
        return True

    return walk(flattened)


def _write_ppm(
    path: Path,
    rgb: Any,
    *,
    expected_resolution: tuple[int, int] = (1280, 720),
) -> dict[str, Any]:
    """Validate and write one frozen RGB camera frame."""

    import numpy as np

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected HxWx3/4 camera data, got {array.shape!r}")
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    height, width, _channels = array.shape
    if (width, height) != expected_resolution:
        raise ValueError(
            "Camera resolution mismatch: "
            f"got {width}x{height}, expected "
            f"{expected_resolution[0]}x{expected_resolution[1]}"
        )
    channel_ranges = np.ptp(array, axis=(0, 1))
    if not np.any(channel_ranges):
        raise ValueError("Camera frame is spatially uniform (blank/solid color)")
    pixel_sha256 = hashlib.sha256(array.tobytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(array.tobytes())
    return {
        "actual_resolution_px": [width, height],
        "pixel_min": int(array.min()),
        "pixel_max": int(array.max()),
        "pixel_mean": float(array.mean()),
        "channel_ranges": [int(value) for value in channel_ranges],
        "pixel_sha256": pixel_sha256,
    }


def _world_position(stage: Any, prim_path: str) -> list[float]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Required prim is missing: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def _dynamic_snapshot(stage: Any) -> dict[str, list[float]]:
    paths = [f"/World/Parts/P0{index}" for index in range(1, 5)]
    paths.append("/World/Bins/Bin_01")
    return {path: _world_position(stage, path) for path in paths}


def _validate_dynamic_snapshot(
    snapshot: dict[str, list[float]],
    expected: dict[str, list[float]],
) -> list[str]:
    errors: list[str] = []
    for prim_path, position in snapshot.items():
        if not _all_finite(position):
            errors.append(f"{prim_path} contains NaN/Inf: {position}")
            continue
        target = expected[prim_path]
        distance = math.dist(position, target)
        if distance > 0.10:
            errors.append(
                f"{prim_path} drifted {distance:.4f} m from its configured reset pose"
            )
        x, y, z = position
        if not (-1.20 <= x <= 1.20 and -0.65 <= y <= 0.65 and 0.70 <= z <= 1.30):
            errors.append(f"{prim_path} left the workcell bounds: {position}")
    return errors


def _expected_dynamic_positions(config: dict[str, Any]) -> dict[str, list[float]]:
    expected = {
        f"/World/Parts/{item['id']}": [
            float(value) for value in item["pose"]["position_m"]
        ]
        for item in config["parts"]
    }
    expected["/World/Bins/Bin_01"] = [
        float(value) for value in config["bin"]["pose"]["position_m"]
    ]
    return expected


def _robot_state(arm: Any, arm_id: str) -> dict[str, Any]:
    positions = arm.get_joint_positions()
    velocities = arm.get_joint_velocities()
    if not _all_finite(positions):
        raise RuntimeError(f"{arm_id} joint positions contain NaN/Inf")
    if not _all_finite(velocities):
        raise RuntimeError(f"{arm_id} joint velocities contain NaN/Inf")
    joint_names = getattr(arm, "dof_names", None)
    if joint_names is None:
        # Older Isaac Sim wrappers exposed the same information under this
        # name. Isaac Sim 5.1 SingleArticulation uses ``dof_names``.
        joint_names = getattr(arm, "joint_names", None)
    if joint_names is None:
        raise RuntimeError(f"{arm_id} articulation exposes no DOF names")
    joint_names = [str(name) for name in joint_names]
    position_count = len(positions)
    velocity_count = len(velocities)
    if len(joint_names) != position_count or position_count != velocity_count:
        raise RuntimeError(
            f"{arm_id} joint-state size mismatch: {len(joint_names)} names, "
            f"{position_count} positions, {velocity_count} velocities"
        )
    return {
        "arm_id": arm_id,
        "joint_names": joint_names,
        "joint_positions_rad": positions,
        "joint_velocities_rad_s": velocities,
    }


def _capture_cameras(
    simulation_app: Any,
    config: dict[str, Any],
    evidence_dir: Path,
) -> list[dict[str, Any]]:
    import omni.replicator.core as rep

    captures: list[dict[str, Any]] = []
    resources: list[tuple[Any, Any]] = []
    for camera in config["cameras"]:
        camera_id = str(camera["id"])
        prim_path = f"/World/Cameras/{camera_id}"
        resolution = tuple(int(value) for value in camera["resolution_px"])
        if resolution != (1280, 720):
            raise RuntimeError(
                f"{camera_id} must use frozen 1280x720 resolution, got {resolution!r}"
            )
        render_product = rep.create.render_product(
            prim_path,
            resolution,
            name=f"G0_{camera_id}",
        )
        annotator = rep.annotators.get("rgb")
        annotator.attach(render_product)
        resources.append((annotator, render_product))

    for _ in range(8):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=8)

    for camera, (annotator, render_product) in zip(
        config["cameras"], resources, strict=True
    ):
        camera_id = str(camera["id"])
        image_path = evidence_dir / "cameras" / f"{camera_id}.ppm"
        rgb = annotator.get_data()
        image_stats = _write_ppm(
            image_path,
            rgb,
            expected_resolution=tuple(int(value) for value in camera["resolution_px"]),
        )
        captures.append(
            {
                "camera_id": camera_id,
                "prim_path": f"/World/Cameras/{camera_id}",
                "resolution_px": camera["resolution_px"],
                "horizontal_fov_deg": camera["horizontal_fov_deg"],
                "file": image_path,
                "image_stats": image_stats,
                "online_gt_included": False,
            }
        )
        annotator.detach()
        render_product.destroy()
    rep.orchestrator.wait_until_complete()
    return captures


def _run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.resets < 0:
        raise ValueError("--resets cannot be negative")
    if args.reset_settle_steps < 1:
        raise ValueError("--reset-settle-steps must be at least 1")

    args.evidence_dir = args.evidence_dir.expanduser().resolve()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import isaac_compat
    import scene_layout

    config = scene_layout.load_config(args.config)
    errors = scene_layout.validate_scene_config(config)
    if errors:
        raise ValueError("Frozen scene contract failed: " + "; ".join(errors))
    settle_steps = args.reset_settle_steps

    simulation_app = isaac_compat.launch_simulation_app(headless=True)
    result["simulation_app_started"] = True
    try:
        result["isaac_sim_version"] = isaac_compat.require_isaac_sim_51()
        import single_bin_scene_builder
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation

        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
        result["franka_asset"] = franka_asset
        single_bin_scene_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)

        missing = [
            path for path in REQUIRED_PRIMS if not stage.GetPrimAtPath(path).IsValid()
        ]
        if missing:
            raise RuntimeError(f"Required scene prims are missing: {missing}")

        destination = isaac_compat.save_stage_checked(args.output_scene)
        result["scene_file"] = destination
        result["required_prim_count"] = len(REQUIRED_PRIMS)

        physics = config["physics"]
        if World.instance():
            World.instance().clear_instance()
        world = World(
            physics_dt=float(physics["physics_dt_s"]),
            rendering_dt=float(physics["rendering_dt_s"]),
            stage_units_in_meters=1.0,
        )
        arm_a = world.scene.add(
            SingleArticulation(
                prim_path="/World/Robots/Arm_A",
                name="g0_arm_a",
            )
        )
        arm_b = world.scene.add(
            SingleArticulation(
                prim_path="/World/Robots/Arm_B",
                name="g0_arm_b",
            )
        )

        expected_positions = _expected_dynamic_positions(config)
        reset_records: list[dict[str, Any]] = []
        reset_errors: list[str] = []
        reset_count = max(args.resets, 1)
        for reset_index in range(1, reset_count + 1):
            world.reset()
            for _ in range(settle_steps):
                world.step(render=False)
            runtime_stage = isaac_compat.get_current_stage()
            snapshot = _dynamic_snapshot(runtime_stage)
            current_errors = _validate_dynamic_snapshot(snapshot, expected_positions)
            robot_states = [
                _robot_state(arm_a, "Arm_A"),
                _robot_state(arm_b, "Arm_B"),
            ]
            reset_records.append(
                {
                    "reset_index": reset_index,
                    "dynamic_positions_m": snapshot,
                    "robot_states": robot_states,
                    "errors": current_errors,
                }
            )
            reset_errors.extend(
                f"reset {reset_index}: {message}" for message in current_errors
            )
            if args.resets == 0:
                break

        if reset_errors:
            raise RuntimeError("; ".join(reset_errors))

        started = time.monotonic()
        for step_index in range(1, args.steps + 1):
            world.step(render=(step_index % 30 == 0))
            if step_index % 100 == 0 or step_index == args.steps:
                runtime_stage = isaac_compat.get_current_stage()
                snapshot = _dynamic_snapshot(runtime_stage)
                step_errors = _validate_dynamic_snapshot(snapshot, expected_positions)
                if step_errors:
                    raise RuntimeError(f"step {step_index}: " + "; ".join(step_errors))
        elapsed = time.monotonic() - started

        observation = {
            "observation_id": f"g0-{time.time_ns()}",
            "timestamp_unix_ns": time.time_ns(),
            "robot_states": [
                _robot_state(arm_a, "Arm_A"),
                _robot_state(arm_b, "Arm_B"),
            ],
            "online_gt_included": False,
            "note": "Only robot telemetry is present; object ground truth is excluded.",
        }
        _write_json(args.evidence_dir / "robot_observation.json", observation)
        _write_json(args.evidence_dir / "reset_report.json", {"resets": reset_records})

        camera_captures: list[dict[str, Any]] = []
        if args.capture_cameras:
            camera_captures = _capture_cameras(
                simulation_app, config, args.evidence_dir
            )
            if len(camera_captures) != 3:
                raise RuntimeError(
                    f"Expected 3 camera samples, got {len(camera_captures)}"
                )
            _write_json(
                args.evidence_dir / "camera_manifest.json",
                {"cameras": camera_captures, "online_gt_included": False},
            )

        result.update(
            {
                "status": "PASS",
                "headless_steps_requested": args.steps,
                "headless_steps_completed": args.steps,
                "headless_elapsed_seconds": elapsed,
                "steps_per_second": args.steps / elapsed if elapsed else None,
                "resets_requested": args.resets,
                "resets_completed": args.resets,
                "reset_settle_steps": settle_steps,
                "camera_samples": camera_captures,
                "robot_observation_file": args.evidence_dir / "robot_observation.json",
                "reset_report_file": args.evidence_dir / "reset_report.json",
                "online_gt_included": False,
            }
        )
    except BaseException as exc:
        # Isaac Sim/Kit can raise SystemExit during extension startup or
        # shutdown. Persist the real cause before close(), because some Kit
        # builds terminate the interpreter from close() and never return to
        # main().
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(args.evidence_dir / "run_result.json", result)
        simulation_app.close()


def main() -> int:
    args = _parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "FAIL",
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "repository_root": REPO_ROOT,
        "config": args.config,
        "simulation_app_started": False,
    }
    exit_code = 1
    try:
        _run(args, result)
        exit_code = 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr)
    finally:
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(evidence_dir / "run_result.json", result)
    print(f"G0 result: {result['status']}")
    print(f"Evidence: {evidence_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
