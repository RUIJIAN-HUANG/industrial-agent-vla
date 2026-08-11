"""Offline V2 contract gate; GUI/physics evidence is a later mandatory stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V2 static scene contract gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from v2_industrial_assets import asset_summary
    from v2_scene_contract import load_config, mass_budget, validate_config

    evidence_dir = args.evidence_dir.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_path = evidence_dir / "run_result.json"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        raw_config = args.config.expanduser().resolve().read_bytes()
        config = load_config(args.config)
        errors = validate_config(config)
        assets = asset_summary(config.get("parts", []))
        errors.extend(assets["errors"])
        budget = mass_budget(config)
        result = {
            "status": "PASS" if not errors else "FAIL",
            "gate": "V2_STATIC_CONTRACT_ONLY",
            "gui_physics_acceptance_required": True,
            "scene_id": config.get("scene_id"),
            "config_path": str(args.config.expanduser().resolve()),
            "config_sha256": hashlib.sha256(raw_config).hexdigest(),
            "part_count": len(config.get("parts", [])),
            "slot_count": len(config.get("bin", {}).get("slots", [])),
            "camera_count": len(config.get("cameras", [])),
            "asset_summary": assets,
            "mass_budget": budget,
            "errors": errors,
            "started_at_local": started,
            "finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    except Exception as exc:
        result = {
            "status": "ERROR",
            "gate": "V2_STATIC_CONTRACT_ONLY",
            "gui_physics_acceptance_required": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_at_local": started,
            "finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"V2 static result: {result_path}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
