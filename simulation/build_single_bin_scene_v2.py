"""Visible-GUI Isaac Sim entry point for the isolated V2 scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "generated" / "single_bin_scene_v2.usda"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V2 dual-Franka, eight-part manual collection scene."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--franka-usd", default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Non-visual diagnostics only; formal V2 scene acceptance must omit this flag.",
    )
    parser.add_argument("--no-robots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.config.expanduser().is_file():
        raise FileNotFoundError(f"V2 scene config does not exist: {args.config}")
    if args.output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("--output must end with .usd, .usda, or .usdc")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    import isaac_compat
    from v2_scene_contract import load_config, require_valid_config

    config = load_config(args.config)
    require_valid_config(config)
    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    try:
        import single_bin_scene_v2_builder

        stage = isaac_compat.create_new_stage()
        franka_asset = None
        if not args.no_robots:
            franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
            print(f"Resolved Franka asset: {franka_asset}")
        single_bin_scene_v2_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=not args.no_robots,
        )
        isaac_compat.wait_for_stage_loading(simulation_app)
        destination = isaac_compat.save_stage_checked(args.output)
        print(f"V2 scene saved successfully: {destination}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
