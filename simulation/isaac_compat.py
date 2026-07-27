"""Small Isaac Sim 4.2/4.5/5.1 compatibility helpers.

This module intentionally has no top-level ``omni`` or ``pxr`` imports.  A
standalone Isaac Sim program must create ``SimulationApp`` before importing
Kit/Omniverse modules.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


FRANKA_ASSET_CANDIDATES = (
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/Franka/franka.usd",
)


def launch_simulation_app(*, headless: bool) -> Any:
    """Launch Isaac Sim, preferring the 5.x namespace."""

    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim Python modules were not found. Run this script with "
                "Isaac Sim's python.bat/python.sh, or from its Python environment."
            ) from exc

    return SimulationApp({"headless": headless})


def _stage_function(name: str) -> Callable[..., Any]:
    """Return a stage utility from the new namespace or the legacy fallback."""

    try:
        from isaacsim.core.utils import stage as stage_utils
    except ImportError:
        try:
            from omni.isaac.core.utils import stage as stage_utils
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim stage utilities are unavailable after SimulationApp startup."
            ) from exc

    function = getattr(stage_utils, name, None)
    if function is None:
        raise RuntimeError(f"Isaac Sim stage utility '{name}' is unavailable.")
    return function


def create_new_stage() -> Any:
    """Create and return a new in-memory USD stage."""

    stage = _stage_function("create_new_stage")()
    # Isaac Sim 5.1 returns a boolean success flag from create_new_stage(),
    # while some older variants return the stage object itself. Never pass the
    # 5.1 boolean into USD APIs as though it were a Usd.Stage.
    if stage is None or isinstance(stage, bool):
        stage = get_current_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not return a current USD stage.")
    return stage


def get_current_stage() -> Any:
    """Return a fresh handle to the currently active USD stage."""

    stage = _stage_function("get_current_stage")()
    if stage is None or isinstance(stage, bool):
        raise RuntimeError("Isaac Sim did not return a valid current USD stage.")
    return stage


def wait_for_stage_loading(
    simulation_app: Any,
    *,
    timeout_seconds: float = 60.0,
) -> None:
    """Pump Kit updates until referenced USD assets finish loading."""

    is_stage_loading = _stage_function("is_stage_loading")
    deadline = time.monotonic() + timeout_seconds
    while is_stage_loading():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Isaac Sim did not finish loading referenced assets within "
                f"{timeout_seconds:.0f} seconds. Check the asset server/cache."
            )
        simulation_app.update()


def save_stage_checked(usd_path: str | Path) -> Path:
    """Save the current stage and require Isaac Sim's boolean success result."""

    destination = Path(usd_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _stage_function("save_stage")(destination.as_posix())
    if not isinstance(result, bool):
        raise RuntimeError(
            "Isaac Sim save_stage returned a non-boolean result "
            f"({result!r}); the stage was not accepted as saved."
        )
    if not result:
        raise RuntimeError(f"Isaac Sim failed to save the stage to: {destination}")
    return destination


def get_assets_root_path() -> str:
    """Resolve the NVIDIA Isaac asset root across supported namespaces."""

    try:
        from isaacsim.storage.native import get_assets_root_path as resolver
    except ImportError:
        try:
            from omni.isaac.core.utils.nucleus import (
                get_assets_root_path as resolver,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Could not import get_assets_root_path from Isaac Sim."
            ) from exc

    root = resolver()
    if not root:
        raise RuntimeError(
            "Isaac Sim returned no asset root. Check the local asset pack or "
            "Nucleus connection, or pass --franka-usd explicitly."
        )
    return str(root).rstrip("/")


def _is_uri(value: str) -> bool:
    return "://" in value


def _asset_exists(asset_path: str) -> bool:
    """Check local paths and Omniverse URLs without accepting unresolved assets."""

    if not _is_uri(asset_path):
        return Path(asset_path).expanduser().is_file()

    try:
        import omni.client
    except ImportError:
        return False

    try:
        result, _entry = omni.client.stat(asset_path)
    except Exception:
        return False

    result_name = getattr(result, "name", str(result)).upper()
    return result_name == "OK" or result_name.endswith(".OK")


def _asset_root_join(root: str, relative_path: str) -> str:
    return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"


def resolve_franka_asset(explicit_path: str | None = None) -> str:
    """Resolve a real Franka USD asset or fail with actionable diagnostics."""

    checked: list[str] = []

    if explicit_path:
        expanded = str(Path(explicit_path).expanduser())
        if _is_uri(explicit_path):
            checked.append(explicit_path)
            if _asset_exists(explicit_path):
                return explicit_path
        elif Path(expanded).is_file():
            return Path(expanded).resolve().as_posix()
        elif explicit_path.startswith("/Isaac/"):
            rooted = _asset_root_join(get_assets_root_path(), explicit_path)
            checked.append(rooted)
            if _asset_exists(rooted):
                return rooted
        else:
            checked.append(str(Path(expanded).resolve()))

        raise FileNotFoundError(
            "The Franka USD supplied via --franka-usd does not exist or cannot "
            f"be reached: {explicit_path}. Checked: {', '.join(checked)}"
        )

    root = get_assets_root_path()
    for relative_path in FRANKA_ASSET_CANDIDATES:
        candidate = _asset_root_join(root, relative_path)
        checked.append(candidate)
        if _asset_exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No supported Franka USD asset was found. Install/mount the Isaac asset "
        "pack, verify the Nucleus connection, or pass --franka-usd. Checked: "
        + ", ".join(checked)
    )
