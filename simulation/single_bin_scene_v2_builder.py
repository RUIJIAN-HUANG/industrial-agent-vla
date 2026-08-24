"""Build the isolated V2 manual-collection scene with raw USD APIs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from pxr import Usd, UsdPhysics, UsdShade, Vt

try:
    from simulation import isaac_compat
    from simulation import single_bin_scene_builder as shared
    from simulation.v2_industrial_assets import create_parts
    from simulation.v2_scene_contract import require_valid_config
except ImportError:
    import isaac_compat
    import single_bin_scene_builder as shared
    from v2_industrial_assets import create_parts
    from v2_scene_contract import require_valid_config


def _physics_material(
    stage: Usd.Stage, path: str, spec: Mapping[str, Any]
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(spec["static_friction"]))
    physics.CreateDynamicFrictionAttr(float(spec["dynamic_friction"]))
    physics.CreateRestitutionAttr(float(spec["restitution"]))
    return material


def _bind_physics_material(prim: Any, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _create_materials(
    stage: Usd.Stage, config: Mapping[str, Any]
) -> dict[str, UsdShade.Material]:
    stage.DefinePrim("/World/Materials", "Scope")
    materials = config["materials"]
    return {
        "ordinary": _physics_material(
            stage, "/World/Materials/Ordinary", materials["ordinary"]
        ),
        "carry_grip": _physics_material(
            stage, "/World/Materials/CarryGrip", materials["carry_grip"]
        ),
    }


def _create_bin(
    stage: Usd.Stage,
    bin_config: Mapping[str, Any],
    materials: Mapping[str, UsdShade.Material],
) -> None:
    path = "/World/Bins/Bin_01"
    root = stage.DefinePrim(path, "Xform")
    shared._set_pose(root, bin_config["pose"])
    shared._make_rigid_body(root, float(bin_config["mass_kg"]))
    _bind_physics_material(root, materials["ordinary"])

    size_x, size_y, size_z = (float(value) for value in bin_config["size_m"])
    wall = float(bin_config["wall_thickness_m"])
    divider = float(bin_config["divider_thickness_m"])
    bottom = float(bin_config["bottom_thickness_m"])
    color = shared._color(bin_config.get("color_rgb"), (0.18, 0.20, 0.22))
    interior_x = size_x - 2.0 * wall
    interior_y = size_y - 2.0 * wall
    wall_height = size_z - bottom
    wall_center_z = bottom / 2.0
    bottom_center_z = -size_z / 2.0 + bottom / 2.0

    shared._cube(
        stage,
        f"{path}/Bottom",
        dimensions=(size_x, size_y, bottom),
        position=(0.0, 0.0, bottom_center_z),
        color=color,
        collision=True,
    )
    for name, x in (
        ("Wall_Left", -(size_x - wall) / 2.0),
        ("Wall_Right", (size_x - wall) / 2.0),
    ):
        shared._cube(
            stage,
            f"{path}/{name}",
            dimensions=(wall, size_y, wall_height),
            position=(x, 0.0, wall_center_z),
            color=color,
            collision=True,
        )
    for name, y in (
        ("Wall_Front", -(size_y - wall) / 2.0),
        ("Wall_Back", (size_y - wall) / 2.0),
    ):
        shared._cube(
            stage,
            f"{path}/{name}",
            dimensions=(interior_x, wall, wall_height),
            position=(0.0, y, wall_center_z),
            color=color,
            collision=True,
        )

    for index in range(1, 4):
        x = -interior_x / 2.0 + interior_x * index / 4.0
        shared._cube(
            stage,
            f"{path}/Divider_Column_{index}",
            dimensions=(divider, interior_y, wall_height),
            position=(x, 0.0, wall_center_z),
            color=color,
            collision=True,
        )
    shared._cube(
        stage,
        f"{path}/Divider_Row_1",
        dimensions=(interior_x, divider, wall_height),
        position=(0.0, 0.0, wall_center_z),
        color=color,
        collision=True,
    )

    slots_root = stage.DefinePrim(f"{path}/Slots", "Xform")
    for slot in bin_config["slots"]:
        slot_path = f"{path}/Slots/{slot['id']}"
        slot_prim = stage.DefinePrim(slot_path, "Xform")
        shared._set_xform(slot_prim, position=slot["center_local_m"])
        slot_prim.SetCustomDataByKey("scene:partId", str(slot["part_id"]))
        slot_prim.SetCustomDataByKey("scene:profile", str(slot["profile"]))
    slots_root.SetCustomDataByKey("scene:slotCount", 8)

    handle = bin_config["carry_handle"]
    handle_path = f"{path}/{handle['link']}"
    handle_root = stage.DefinePrim(handle_path, "Xform")
    handle_root.SetCustomDataByKey("scene:frameId", str(handle["frame_id"]))
    support_size = [float(value) for value in handle["support_size_m"]]
    crossbar_size = [float(value) for value in handle["crossbar_size_m"]]
    crossbar_center = [float(value) for value in handle["crossbar_center_local_m"]]
    for index, y in enumerate(handle["support_y_m"], start=1):
        shared._cube(
            stage,
            f"{handle_path}/Support_{index}",
            dimensions=support_size,
            position=(0.0, float(y), crossbar_center[2] / 2.0),
            color=(0.10, 0.11, 0.12),
            collision=True,
        )
    crossbar = shared._cube(
        stage,
        f"{handle_path}/Crossbar",
        dimensions=crossbar_size,
        position=crossbar_center,
        color=(0.10, 0.11, 0.12),
        collision=True,
    )
    _bind_physics_material(crossbar.GetPrim(), materials["carry_grip"])

    tcp = stage.DefinePrim(f"{handle_path}/{handle['frame_id']}", "Xform")
    shared._set_xform(
        tcp,
        position=handle["position_local_m"],
        rpy_deg=handle["rpy_local_deg"],
    )
    tcp.SetCustomDataByKey(
        "scene:approachOffsetM",
        Vt.DoubleArray([float(value) for value in handle["approach_offset_m"]]),
    )
    tcp.SetCustomDataByKey(
        "scene:liftOffsetM",
        Vt.DoubleArray([float(value) for value in handle["lift_offset_m"]]),
    )

    root.SetCustomDataByKey(
        "scene:recipePartIds",
        Vt.StringArray([str(slot["part_id"]) for slot in bin_config["slots"]]),
    )
    root.SetCustomDataByKey("scene:slotCount", 8)
    root.SetCustomDataByKey(
        "scene:designLoadedMassLimitKg",
        float(bin_config["design_loaded_mass_limit_kg"]),
    )
    root.SetCustomDataByKey(
        "scene:acceptanceLoadedMassLimitKg",
        float(bin_config["acceptance_loaded_mass_limit_kg"]),
    )


def build_scene(
    stage: Usd.Stage,
    config: Mapping[str, Any],
    *,
    franka_asset_path: str | None,
    include_robots: bool = True,
    task_id: str | None = None,
) -> Usd.Stage:
    """Populate ``stage`` without touching the V1 builder output."""

    if stage is None:
        raise ValueError("A valid USD stage is required")
    if include_robots and not franka_asset_path:
        raise ValueError("include_robots=True requires a Franka USD asset")
    require_valid_config(config)
    isaac_compat.configure_and_validate_stage_contract(stage)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    world.SetCustomDataByKey("scene:id", str(config["scene_id"]))
    world.SetCustomDataByKey("scene:schemaVersion", str(config["schema_version"]))
    world.SetCustomDataByKey("scene:source", "single_bin_scene_v2_builder.py")

    shared._create_physics_scene(stage, config["physics"])
    shared._create_environment(stage, config)
    shared._create_markers(stage, config)
    materials = _create_materials(stage, config)
    part_configs = deepcopy(config["parts"])
    bin_config = deepcopy(config["bin"])
    if task_id == "BIN01_TO_FINISHED01":
        from simulation.v2_task_initialization import bin01_transport_initial_poses

        initial_poses = bin01_transport_initial_poses(config)
        bin_config["pose"] = initial_poses["/World/Bins/Bin_01"]
        for part in part_configs:
            part["pose"] = initial_poses[f"/World/Parts/{part['id']}"]
    create_parts(stage, part_configs, physics_materials=materials)
    stage.DefinePrim("/World/Bins", "Xform")
    _create_bin(stage, bin_config, materials)
    shared._create_cameras(stage, config["cameras"])
    shared._create_lighting(stage)
    if include_robots:
        assert franka_asset_path is not None
        shared._create_robots(
            stage,
            config["robots"],
            franka_asset_path=franka_asset_path,
        )
    world.SetCustomDataByKey(
        "scene:tokenSequence",
        Vt.StringArray([str(item) for item in config["workflow"]["token_sequence"]]),
    )
    world.SetCustomDataByKey("scene:onlineGtAllowed", False)
    return stage
