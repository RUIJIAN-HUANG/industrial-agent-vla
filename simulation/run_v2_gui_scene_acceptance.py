"""Visible-GUI V2 build and four-image evidence gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visible V2 scene acceptance.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--review-seconds", type=int, default=45)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _capture_overview(simulation_app: Any, destination: Path) -> dict[str, Any]:
    from omni.kit.viewport.utility import (
        capture_viewport_to_file,
        frame_viewport_prims,
        get_active_viewport,
    )

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active Isaac viewport is unavailable")
    frame_viewport_prims(
        viewport,
        prims=[
            "/World/Environment/Table",
            "/World/Robots",
            "/World/Parts",
            "/World/Bins",
        ],
    )
    for _ in range(30):
        simulation_app.update()
    destination.parent.mkdir(parents=True, exist_ok=True)
    helper = capture_viewport_to_file(viewport, file_path=str(destination))
    task = asyncio.ensure_future(helper.wait_for_result())
    deadline = time.monotonic() + 30.0
    while not task.done():
        if time.monotonic() >= deadline:
            task.cancel()
            raise TimeoutError("viewport overview capture exceeded 30 seconds")
        simulation_app.update()
    task.result()
    for _ in range(10):
        simulation_app.update()
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("viewport overview image was not written")
    return {"file": str(destination), "size_bytes": destination.stat().st_size}


def _review_visible_window(simulation_app: Any, seconds: int) -> None:
    if seconds < 10 or seconds > 300:
        raise ValueError("--review-seconds must be in [10, 300]")
    print(f"Isaac Sim 窗口将保持 {seconds} 秒，请观察场景并截图。")
    deadline = time.monotonic() + seconds
    next_notice = seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        simulation_app.update()
        rounded = int(remaining)
        if rounded <= next_notice - 10:
            next_notice = rounded
            print(f"GUI 检查剩余约 {max(rounded, 0)} 秒")


def main() -> int:
    args = _parse_args()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    root = SCRIPT_DIR.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    evidence_dir = args.evidence_dir.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = evidence_dir / "run_result.json"
    result: dict[str, Any] = {
        "status": "ERROR",
        "gate": "V2_VISIBLE_GUI_SCENE_ACCEPTANCE",
        "headless": False,
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    simulation_app = None
    try:
        import isaac_compat
        from v2_scene_contract import load_config, require_valid_config

        config = load_config(args.config)
        require_valid_config(config)
        simulation_app = isaac_compat.launch_simulation_app(headless=False)

        import single_bin_scene_v2_builder
        from run_g0_acceptance import _capture_cameras

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
        for _ in range(120):
            simulation_app.update()

        camera_captures = _capture_cameras(simulation_app, config, evidence_dir)
        if len(camera_captures) != 3:
            raise RuntimeError("expected exactly three camera captures")
        overview = _capture_overview(
            simulation_app, evidence_dir / "overview" / "scene-overview.png"
        )
        _review_visible_window(simulation_app, int(args.review_seconds))
        result.update(
            {
                "status": "PASS",
                "scene_id": config["scene_id"],
                "scene_file": str(scene_file),
                "franka_asset": str(franka_asset),
                "part_count": len(config["parts"]),
                "slot_count": len(config["bin"]["slots"]),
                "camera_captures": camera_captures,
                "overview_capture": overview,
                "gui_review_seconds": int(args.review_seconds),
                "online_gt_included": False,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            file=sys.stderr,
        )
    finally:
        result["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            _write_json(result_path, result)
        finally:
            if simulation_app is not None:
                simulation_app.close()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
