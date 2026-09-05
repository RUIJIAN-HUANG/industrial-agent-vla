"""Small Isaac Sim 4.2/4.5/5.1 compatibility helpers.

This module intentionally has no top-level ``omni`` or ``pxr`` imports.  A
standalone Isaac Sim program must create ``SimulationApp`` before importing
Kit/Omniverse modules.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


SIMPLIFIED_CHINESE_EXTENSION = "omni.kit.language.simplified_chinese"
SIMPLIFIED_CHINESE_LOCALE_ARG = "--/persistent/app/locale_id=zh_CN"
SIMPLIFIED_CHINESE_LOCALE_ID = "zh_CN"
SIMPLIFIED_CHINESE_PANGRAM = (
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分"
    "对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得"
)


FRANKA_ASSET_CANDIDATES = (
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/Franka/franka.usd",
)


def require_isaac_sim_51() -> dict[str, str]:
    """Return build metadata and fail closed unless Isaac Sim is 5.1.x."""

    try:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.core.version"
        if not extension_manager.is_extension_enabled(extension_id):
            enable_result = extension_manager.set_extension_enabled_immediate(
                extension_id,
                True,
            )
            if enable_result is False or not extension_manager.is_extension_enabled(
                extension_id
            ):
                raise RuntimeError(
                    "Isaac Sim rejected enabling 'isaacsim.core.version'."
                )
        from isaacsim.core.version import get_version
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Isaac Sim version metadata is unavailable. Enable the "
            "'isaacsim.core.version' extension in the 5.1 runtime."
        ) from exc

    raw_version = tuple(str(item) for item in get_version())
    if len(raw_version) < 8:
        raise RuntimeError(
            f"Isaac Sim returned an incomplete version tuple: {raw_version!r}"
        )
    info = {
        "core_version": raw_version[0],
        "prerelease_and_build": raw_version[1],
        "major": raw_version[2],
        "minor": raw_version[3],
        "patch": raw_version[4],
        "prerelease": raw_version[5],
        "build_number": raw_version[6],
        "build_tag": raw_version[7],
    }
    if (info["major"], info["minor"]) != ("5", "1"):
        raise RuntimeError(
            "G0 requires Isaac Sim 5.1.x, but the active runtime reports "
            f"{info['major']}.{info['minor']}.{info['patch']} "
            f"(core={info['core_version']!r})."
        )
    return info


def launch_simulation_app(*, headless: bool, enable_chinese_ui: bool = False) -> Any:
    """Launch Isaac Sim, optionally preloading the Simplified Chinese UI.

    The language extension and locale must be passed to Kit before its font
    atlas is initialized.  Enabling this only for the visible competition UI
    keeps the other headless/smoke entry points unchanged.
    """

    launch_config: dict[str, Any] = {"headless": headless}
    if enable_chinese_ui:
        launch_config["extra_args"] = [
            "--enable",
            SIMPLIFIED_CHINESE_EXTENSION,
            SIMPLIFIED_CHINESE_LOCALE_ARG,
        ]

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

    return SimulationApp(launch_config)


def _language_resource_candidates(
    roots: tuple[Path, ...], *, suffixes: frozenset[str], markers: tuple[str, ...]
) -> tuple[Path, ...]:
    """Find language resources without importing Kit UI modules."""

    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                name = path.name.lower()
                parent = str(path.parent).lower()
                if any(marker in name or marker in parent for marker in markers):
                    candidates.append(path)
        except OSError:
            continue
    return tuple(
        sorted(
            set(candidates),
            key=lambda path: ("regular" not in path.name.lower(), path.as_posix()),
        )
    )


def configure_simplified_chinese_ui() -> dict[str, str]:
    """Register a real Chinese font and glyph regions before creating UI.

    Kit 107's bundled language extension can be enabled while still leaving
    the ImGui atlas without Chinese glyphs on some installations.  Re-register
    ``zh_CN`` explicitly before the first ``omni.ui`` widget is created so the
    atlas receives both the font and its region files.
    """

    try:
        import carb  # type: ignore[import-not-found]
        import omni.kit.app  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Kit language support is unavailable; start the UI with Isaac Sim "
            "and omni.kit.language.simplified_chinese enabled."
        ) from exc

    app = omni.kit.app.get_app()
    extension_manager = app.get_extension_manager()
    if not extension_manager.is_extension_enabled(SIMPLIFIED_CHINESE_EXTENSION):
        enabled = extension_manager.set_extension_enabled_immediate(
            SIMPLIFIED_CHINESE_EXTENSION,
            True,
        )
        if enabled is False or not extension_manager.is_extension_enabled(
            SIMPLIFIED_CHINESE_EXTENSION
        ):
            raise RuntimeError(
                f"Kit rejected enabling {SIMPLIFIED_CHINESE_EXTENSION!r}."
            )

    try:
        import omni.kit.language.core as language_core  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Kit language API is unavailable after enabling the Simplified "
            "Chinese extension."
        ) from exc

    language_root = extension_manager.get_extension_path_by_module(
        SIMPLIFIED_CHINESE_EXTENSION
    )
    imgui_root = extension_manager.get_extension_path_by_module(
        "omni.kit.renderer.imgui"
    )
    roots = tuple(
        Path(value)
        for value in (language_root, imgui_root)
        if isinstance(value, str) and value
    )
    font_candidates = _language_resource_candidates(
        roots
        + (
            Path("/usr/share/fonts/opentype/noto"),
            Path("/usr/share/fonts/truetype/noto"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".local" / "share" / "fonts",
            Path.home() / ".fonts",
        ),
        suffixes=frozenset({".ttf", ".otf"}),
        markers=("cjk", "chinese", "simplified", "sc", "han"),
    )
    region_candidates = _language_resource_candidates(
        roots,
        suffixes=frozenset({".txt"}),
        markers=("zh", "cn", "chinese", "simplified", "sc", "cjk", "han"),
    )
    if not font_candidates or not region_candidates:
        raise RuntimeError(
            "Chinese Kit UI resources are incomplete: expected a CJK TTF/OTF "
            "font and at least one Chinese glyph-region file."
        )

    settings = carb.settings.get_settings()
    settings.set("/persistent/app/locale_id", SIMPLIFIED_CHINESE_LOCALE_ID)
    try:
        if SIMPLIFIED_CHINESE_LOCALE_ID in language_core.get_locales():
            language_core.unregister_language(SIMPLIFIED_CHINESE_LOCALE_ID)
    except (AttributeError, RuntimeError):
        pass
    registered = language_core.register_language(
        (
            SIMPLIFIED_CHINESE_LOCALE_ID,
            "Chinese (Simplified)",
            "简体中文",
        ),
        str(font_candidates[0]),
        1.0,
        [str(path) for path in region_candidates],
        SIMPLIFIED_CHINESE_PANGRAM,
        font_overresolution_size=66,
    )
    if registered is False:
        raise RuntimeError("Kit failed to register the Simplified Chinese font.")
    settings.set("/persistent/app/locale_id", SIMPLIFIED_CHINESE_LOCALE_ID)
    return {
        "locale_id": SIMPLIFIED_CHINESE_LOCALE_ID,
        "font_path": str(font_candidates[0]),
        "region_count": str(len(region_candidates)),
    }


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

    result = _stage_function("create_new_stage")()
    # Isaac Sim 5.1 returns a boolean success flag from create_new_stage(),
    # while older variants return the stage object itself. A False result must
    # fail closed: falling back to get_current_stage() could silently reuse a
    # stale stage left over from a previous run.
    if isinstance(result, bool):
        if not result:
            raise RuntimeError("Isaac Sim failed to create a new USD stage.")
        return get_current_stage()
    return require_usd_stage(result, context="create_new_stage")


def get_current_stage() -> Any:
    """Return a fresh, non-boolean handle to the currently active USD stage."""

    stage = _stage_function("get_current_stage")()
    return require_usd_stage(stage, context="get_current_stage")


def _usd_stage_type() -> type[Any]:
    """Import and return ``pxr.Usd.Stage`` after SimulationApp startup."""

    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "pxr.Usd is unavailable after SimulationApp startup."
        ) from exc
    return Usd.Stage


def require_usd_stage(stage: Any, *, context: str) -> Any:
    """Require an actual ``pxr.Usd.Stage`` rather than a status sentinel."""

    if stage is None or isinstance(stage, bool):
        raise RuntimeError(f"{context} did not return a valid USD Stage.")
    if not isinstance(stage, _usd_stage_type()):
        raise TypeError(
            f"{context} returned {type(stage).__name__}, expected pxr.Usd.Stage."
        )
    return stage


def validate_stage_contract(
    stage: Any,
    *,
    expected_up_axis: str = "Z",
    expected_meters_per_unit: float = 1.0,
    expected_kilograms_per_unit: float = 1.0,
) -> None:
    """Read back and validate the frozen USD stage type, axis, and units."""

    require_usd_stage(stage, context="stage contract validation")
    try:
        from pxr import UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "pxr.UsdGeom/UsdPhysics are unavailable after SimulationApp startup."
        ) from exc

    actual_up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    actual_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    actual_kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    errors: list[str] = []
    if actual_up_axis != expected_up_axis.upper():
        errors.append(
            f"up axis is {actual_up_axis!r}, expected {expected_up_axis.upper()!r}"
        )
    if abs(actual_meters_per_unit - expected_meters_per_unit) > 1e-12:
        errors.append(
            "metersPerUnit is "
            f"{actual_meters_per_unit}, expected {expected_meters_per_unit}"
        )
    if abs(actual_kilograms_per_unit - expected_kilograms_per_unit) > 1e-12:
        errors.append(
            "kilogramsPerUnit is "
            f"{actual_kilograms_per_unit}, "
            f"expected {expected_kilograms_per_unit}"
        )
    if errors:
        raise RuntimeError("Invalid frozen USD stage contract: " + "; ".join(errors))


def configure_and_validate_stage_contract(stage: Any) -> None:
    """Write the frozen Z-up/SI metadata and require an exact readback."""

    require_usd_stage(stage, context="stage contract configuration")
    try:
        from pxr import UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "pxr.UsdGeom/UsdPhysics are unavailable after SimulationApp startup."
        ) from exc

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    validate_stage_contract(stage)


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
