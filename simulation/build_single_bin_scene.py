"""Standalone Isaac Sim entry point for the frozen single-bin scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v1.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "generated" / "single_bin_scene_v1.usda"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen dual-Franka, four-part, single-bin Isaac Sim "
            "scene as a USD file."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Scene JSON contract (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .usd/.usda file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--franka-usd",
        default=None,
        help="Explicit local path or Omniverse URI for the Franka USD asset.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without opening the graphical interface.",
    )
    parser.add_argument(
        "--no-robots",
        action="store_true",
        help="Build geometry and cameras without resolving Franka assets.",
    )
    return parser.parse_args()


def _local_imports() -> tuple[object, object]:
    """Import helpers in script mode without importing pxr before Kit starts."""

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import isaac_compat
    import single_bin_scene_builder

    return isaac_compat, single_bin_scene_builder


def main() -> int:
    args = _parse_args()
    if not args.config.expanduser().is_file():
        raise FileNotFoundError(f"Scene config does not exist: {args.config}")
    if args.output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("--output must end with .usd, .usda, or .usdc")

    # isaac_compat has no top-level omni/pxr imports, so it is safe to import
    # only that small launcher before SimulationApp exists.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import isaac_compat
    import scene_layout

    config = scene_layout.load_config(args.config)
    contract_errors = scene_layout.validate_scene_config(config)
    if contract_errors:
        formatted = "\n".join(f"  - {error}" for error in contract_errors)
        raise ValueError(
            "Scene config does not match the frozen MVP contract:\n" + formatted
        )

    simulation_app = isaac_compat.launch_simulation_app(headless=args.headless)
    try:
        # All pxr/omni-dependent imports deliberately happen after app startup.
        _compat, scene_builder = _local_imports()
        stage = isaac_compat.create_new_stage()

        franka_asset = None
        if not args.no_robots:
            franka_asset = isaac_compat.resolve_franka_asset(args.franka_usd)
            print(f"Resolved Franka asset: {franka_asset}")

        scene_builder.build_scene(
            stage,
            config,
            franka_asset_path=franka_asset,
            include_robots=not args.no_robots,
        )
        isaac_compat.wait_for_stage_loading(simulation_app)
        destination = isaac_compat.save_stage_checked(args.output)
        print(f"Scene saved successfully: {destination}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
