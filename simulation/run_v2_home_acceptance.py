"""Visible V2 explicit-HOME acceptance without Cartesian motion or grasping."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visible V2 explicit-HOME acceptance.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--home-cycles", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--review-seconds", type=int, default=30)
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
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    if not 1 <= args.home_cycles <= 20:
        raise ValueError("--home-cycles must be in [1, 20]")
    if not 60 <= args.settle_steps <= 1200:
        raise ValueError("--settle-steps must be in [60, 1200]")
    if not 10 <= args.review_seconds <= 300:
        raise ValueError("--review-seconds must be in [10, 300]")

    for path in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    evidence_dir = args.evidence_dir.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = evidence_dir / "run_result.json"
    result: dict[str, Any] = {
        "status": "ERROR",
        "gate": "V2_VISIBLE_EXPLICIT_HOME_ACCEPTANCE",
        "headless": False,
        "cartesian_motion_performed": False,
        "grasp_performed": False,
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    simulation_app = None
    try:
        import isaac_compat
        from run_g0_acceptance import (
            _capture_cameras,
            _home_readback_errors,
            _robot_state,
            _write_explicit_home,
        )
        from v2_scene_contract import load_config, require_valid_config

        config = load_config(args.config)
        require_valid_config(config)
        simulation_app = isaac_compat.launch_simulation_app(headless=False)

        import single_bin_scene_v2_builder
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        stage = isaac_compat.create_new_stage()
        franka_asset = isaac_compat.resolve_franka_asset(None)
        single_bin_scene_v2_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=True,
        )
        isaac_compat.wait_for_stage_loading(simulation_app, timeout_seconds=180.0)
        scene_file = isaac_compat.save_stage_checked(args.output_scene)

        physics = config["physics"]
        if World.instance():
            World.instance().clear_instance()
        world = World(
            physics_dt=float(physics["physics_dt_s"]),
            rendering_dt=float(physics["rendering_dt_s"]),
            stage_units_in_meters=1.0,
        )
        arms = {
            arm_id: world.scene.add(
                SingleArticulation(
                    prim_path=f"/World/Robots/{arm_id}",
                    name=f"v2_home_{arm_id.lower()}",
                )
            )
            for arm_id in ("Arm_A", "Arm_B")
        }

        cycles: list[dict[str, Any]] = []
        all_errors: list[str] = []
        for cycle_index in range(1, args.home_cycles + 1):
            world.reset()
            targets = {
                arm_id: _write_explicit_home(config, arms[arm_id], arm_id)
                for arm_id in ("Arm_A", "Arm_B")
            }
            for arm_id in ("Arm_A", "Arm_B"):
                arms[arm_id].get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=targets[arm_id])
                )
            for _ in range(args.settle_steps):
                world.step(render=True)
            errors: list[str] = []
            for arm_id in ("Arm_A", "Arm_B"):
                errors.extend(
                    _home_readback_errors(arms[arm_id], arm_id, targets[arm_id])
                )
            cycles.append(
                {
                    "cycle_index": cycle_index,
                    "explicit_home_written": True,
                    "position_hold_commanded": True,
                    "targets": targets,
                    "robot_states": [
                        _robot_state(arms[arm_id], arm_id)
                        for arm_id in ("Arm_A", "Arm_B")
                    ],
                    "errors": errors,
                }
            )
            all_errors.extend(
                f"HOME cycle {cycle_index}: {message}" for message in errors
            )

        _write_json(evidence_dir / "home_cycles.json", {"cycles": cycles})

        deadline = time.monotonic() + args.review_seconds
        while simulation_app.is_running() and time.monotonic() < deadline:
            world.step(render=True)

        final_home_errors: list[str] = []
        final_targets = cycles[-1]["targets"]
        for arm_id in ("Arm_A", "Arm_B"):
            final_home_errors.extend(
                _home_readback_errors(arms[arm_id], arm_id, final_targets[arm_id])
            )
        all_errors.extend(f"final HOME hold: {message}" for message in final_home_errors)
        final_robot_states = [
            _robot_state(arms[arm_id], arm_id) for arm_id in ("Arm_A", "Arm_B")
        ]
        _write_json(
            evidence_dir / "home_hold.json",
            {
                "review_seconds": args.review_seconds,
                "robot_states": final_robot_states,
                "errors": final_home_errors,
            },
        )

        # Replicator camera capture may stop its timeline and invalidate live
        # articulation views. Capture last and never query an arm afterwards.
        camera_captures = _capture_cameras(simulation_app, config, evidence_dir)
        if len(camera_captures) != 3:
            all_errors.append(
                f"expected exactly three camera captures, got {len(camera_captures)}"
            )
        result.update(
            {
                "status": "PASS" if not all_errors else "FAIL",
                "scene_id": config["scene_id"],
                "scene_file": scene_file,
                "franka_asset": franka_asset,
                "home_cycles_requested": args.home_cycles,
                "home_cycles_completed": len(cycles),
                "settle_steps_per_cycle": args.settle_steps,
                "review_seconds": args.review_seconds,
                "camera_captures": camera_captures,
                "final_robot_states": final_robot_states,
                "final_home_errors": final_home_errors,
                "errors": all_errors,
                "online_gt_included": False,
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False), file=sys.stderr)
    finally:
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            _write_json(result_path, result)
        finally:
            if simulation_app is not None:
                simulation_app.close()

    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
