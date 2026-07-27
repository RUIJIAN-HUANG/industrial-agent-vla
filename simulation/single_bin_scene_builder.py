"""Build the frozen dual-Franka, single-bin handoff scene with raw USD APIs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt


JsonObject = dict[str, Any]
Color = tuple[float, float, float]


def load_scene_config(config_path: str | Path) -> JsonObject:
    """Load and minimally validate the scene JSON contract."""

    path = Path(config_path).expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Scene config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene config is not valid JSON: {path}: {exc}") from exc

    required = {
        "table",
        "robots",
        "stations",
        "zones",
        "parts",
        "bin",
        "cameras",
        "physics",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Scene config is missing required keys: {missing}")
    if len(config["robots"]) != 2:
        raise ValueError("The frozen scene requires exactly two robots.")
    if len(config["parts"]) != 4:
        raise ValueError("The frozen scene requires exactly four parts.")
    if len(config["cameras"]) != 3:
        raise ValueError("The frozen scene requires exactly three cameras.")
    return config


def _prim_name(identifier: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(identifier))
    if not name:
        raise ValueError("USD prim identifiers cannot be empty.")
    if name[0].isdigit():
        name = f"_{name}"
    return name


def _vec3(values: Sequence[float], *, label: str) -> Gf.Vec3d:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values.")
    return Gf.Vec3d(*(float(value) for value in values))


def _vec3f(values: Sequence[float], *, label: str) -> Gf.Vec3f:
    vector = _vec3(values, label=label)
    return Gf.Vec3f(float(vector[0]), float(vector[1]), float(vector[2]))


def _color(values: Sequence[float] | None, default: Color) -> Color:
    if values is None:
        return default
    if len(values) != 3:
        raise ValueError("color_rgb must contain exactly three values.")
    normalized = tuple(float(value) for value in values)
    if any(value > 1.0 for value in normalized):
        normalized = tuple(value / 255.0 for value in normalized)
    return normalized  # type: ignore[return-value]


def _pose_values(pose: Mapping[str, Any]) -> tuple[Gf.Vec3d, Gf.Vec3f]:
    position = pose.get("position_m", pose.get("position", [0.0, 0.0, 0.0]))
    rotation = pose.get("rpy_deg", [0.0, 0.0, 0.0])
    return (
        _vec3(position, label="pose.position_m"),
        _vec3f(rotation, label="pose.rpy_deg"),
    )


def _set_xform(
    prim: Usd.Prim,
    *,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    rpy_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(_vec3(position, label="position"))
    xformable.AddRotateXYZOp().Set(_vec3f(rpy_deg, label="rpy_deg"))


def _set_pose(prim: Usd.Prim, pose: Mapping[str, Any]) -> None:
    position, rotation = _pose_values(pose)
    _set_xform(prim, position=position, rpy_deg=rotation)


def _set_display(
    geometry: UsdGeom.Gprim,
    color: Color,
    *,
    opacity: float = 1.0,
) -> None:
    geometry.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity < 1.0:
        geometry.CreateDisplayOpacityAttr([float(opacity)])


def _apply_collision(prim: Usd.Prim) -> None:
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(True)


def _cube(
    stage: Usd.Stage,
    path: str,
    *,
    dimensions: Sequence[float],
    position: Sequence[float] = (0.0, 0.0, 0.0),
    rpy_deg: Sequence[float] = (0.0, 0.0, 0.0),
    color: Color = (0.5, 0.5, 0.5),
    collision: bool = False,
    opacity: float = 1.0,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), position=position, rpy_deg=rpy_deg)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        _vec3(dimensions, label=f"{path}.dimensions")
    )
    _set_display(cube, color, opacity=opacity)
    if collision:
        _apply_collision(cube.GetPrim())
    return cube


def _cylinder(
    stage: Usd.Stage,
    path: str,
    *,
    radius: float,
    height: float,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    color: Color = (0.8, 0.05, 0.04),
    collision: bool = False,
) -> UsdGeom.Cylinder:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    _set_xform(cylinder.GetPrim(), position=position)
    _set_display(cylinder, color)
    if collision:
        _apply_collision(cylinder.GetPrim())
    return cylinder


def _make_rigid_body(root: Usd.Prim, mass_kg: float) -> None:
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(mass_kg))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_body.CreateEnableCCDAttr(True)


def _create_physics_scene(stage: Usd.Stage, physics: Mapping[str, Any]) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    gravity_value = physics.get("gravity_m_s2", [0.0, 0.0, -9.81])
    if isinstance(gravity_value, Sequence) and not isinstance(
        gravity_value, (str, bytes)
    ):
        gravity_vector = _vec3(gravity_value, label="physics.gravity_m_s2")
        gravity_magnitude = gravity_vector.GetLength()
        if gravity_magnitude <= 0.0:
            raise ValueError("physics.gravity_m_s2 cannot be a zero vector.")
        gravity_direction = gravity_vector / gravity_magnitude
    else:
        gravity_scalar = float(gravity_value)
        gravity_magnitude = abs(gravity_scalar)
        gravity_direction = Gf.Vec3d(0.0, 0.0, -1.0 if gravity_scalar <= 0.0 else 1.0)
    scene.CreateGravityDirectionAttr(
        Gf.Vec3f(
            float(gravity_direction[0]),
            float(gravity_direction[1]),
            float(gravity_direction[2]),
        )
    )
    scene.CreateGravityMagnitudeAttr(float(gravity_magnitude))

    prim = scene.GetPrim()
    physics_dt = float(physics.get("physics_dt_s", 1.0 / 120.0))
    if physics_dt <= 0.0:
        raise ValueError("physics.physics_dt_s must be positive.")
    steps_per_second = int(round(1.0 / physics_dt))
    if not math.isclose(
        steps_per_second * physics_dt,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "physics.physics_dt_s must be the reciprocal of a whole-number "
            "steps-per-second value."
        )
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx_scene.CreateTimeStepsPerSecondAttr(steps_per_second)
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateEnableCCDAttr(True)

    prim.CreateAttribute(
        "scene:physicsDtSeconds", Sdf.ValueTypeNames.Double, custom=True
    ).Set(physics_dt)
    prim.CreateAttribute(
        "scene:renderingDtSeconds", Sdf.ValueTypeNames.Double, custom=True
    ).Set(float(physics.get("rendering_dt_s", 1.0 / 30.0)))
    prim.CreateAttribute(
        "scene:controlFrequencyHz", Sdf.ValueTypeNames.Double, custom=True
    ).Set(float(physics.get("control_frequency_hz", 60.0)))


def _create_environment(stage: Usd.Stage, config: Mapping[str, Any]) -> None:
    table = config["table"]
    table_position, table_rotation = _pose_values(table["pose"])
    table_size = [float(value) for value in table["size_m"]]
    stage.DefinePrim("/World/Environment", "Xform")
    stage.DefinePrim("/World/Environment/Table", "Xform")
    _cube(
        stage,
        "/World/Environment/Table/Top",
        dimensions=table_size,
        position=table_position,
        rpy_deg=table_rotation,
        color=_color(table.get("color_rgb"), (0.54, 0.56, 0.58)),
        collision=True,
    )

    top_bottom = float(table_position[2]) - table_size[2] / 2.0
    leg_height = max(top_bottom, 0.1)
    leg_xy = 0.075
    inset = 0.09
    for index, (x_sign, y_sign) in enumerate(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
        start=1,
    ):
        x = float(table_position[0]) + x_sign * (table_size[0] / 2.0 - inset)
        y = float(table_position[1]) + y_sign * (table_size[1] / 2.0 - inset)
        _cube(
            stage,
            f"/World/Environment/Table/Leg_{index:02d}",
            dimensions=(leg_xy, leg_xy, leg_height),
            position=(x, y, leg_height / 2.0),
            color=(0.36, 0.38, 0.40),
            collision=True,
        )

    ground_size = (
        max(3.0, table_size[0] + 0.8),
        max(2.0, table_size[1] + 0.8),
        0.05,
    )
    _cube(
        stage,
        "/World/Environment/Ground",
        dimensions=ground_size,
        position=(float(table_position[0]), float(table_position[1]), -0.025),
        color=(0.18, 0.19, 0.20),
        collision=True,
    )


def _create_markers(stage: Usd.Stage, config: Mapping[str, Any]) -> None:
    table_surface = float(config["table"]["surface_z_m"])
    stations_scope = stage.DefinePrim("/World/Stations", "Xform")
    stations_scope.SetCustomDataByKey("collisionPolicy", "visual_only")
    for station in config["stations"]:
        position, rotation = _pose_values(station["pose"])
        footprint = station["footprint_m"]
        station_path = f"/World/Stations/{_prim_name(station['id'])}"
        station_root = stage.DefinePrim(station_path, "Xform")
        _set_xform(station_root, position=position, rpy_deg=rotation)
        station_root.SetCustomDataByKey("scene:kind", str(station["kind"]))
        station_root.SetCustomDataByKey(
            "scene:footprintM",
            Vt.DoubleArray([float(footprint[0]), float(footprint[1])]),
        )
        _cube(
            stage,
            f"{station_path}/Marker",
            dimensions=(float(footprint[0]), float(footprint[1]), 0.0006),
            position=(0.0, 0.0, table_surface + 0.0003 - float(position[2])),
            color=_color(station.get("color_rgb"), (0.20, 0.70, 0.25)),
            opacity=0.35,
        )

    zones_scope = stage.DefinePrim("/World/Zones", "Xform")
    zones_scope.SetCustomDataByKey("collisionPolicy", "visual_only")
    for zone in config["zones"]:
        position, rotation = _pose_values(zone["pose"])
        footprint = zone["footprint_m"]
        zone_path = f"/World/Zones/{_prim_name(zone['id'])}"
        zone_prim = stage.DefinePrim(zone_path, "Xform")
        _set_xform(zone_prim, position=position, rpy_deg=rotation)
        _cube(
            stage,
            f"{zone_path}/Marker",
            dimensions=(float(footprint[0]), float(footprint[1]), 0.0006),
            position=(0.0, 0.0, table_surface + 0.0003 - float(position[2])),
            color=_color(zone.get("color_rgb"), (0.15, 0.45, 0.90)),
            opacity=0.22,
        )
        zone_prim.SetCustomDataByKey(
            "expectedPartCount", int(zone.get("expected_part_count", 0))
        )


def _create_part(stage: Usd.Stage, part: Mapping[str, Any]) -> None:
    path = f"/World/Parts/{_prim_name(part['id'])}"
    root = stage.DefinePrim(path, "Xform")
    _set_pose(root, part["pose"])

    geometry = part["geometry"]
    if str(geometry.get("type", "cylinder")).lower() != "cylinder":
        raise ValueError(f"{part['id']} must use the frozen cylinder geometry.")
    radius = float(geometry["radius_m"])
    height = float(geometry["height_m"])
    mass_kg = float(geometry["mass_kg"])
    color = _color(part.get("color_rgb"), (0.82, 0.04, 0.03))

    _make_rigid_body(root, mass_kg)
    root.SetCustomDataByKey("scene:state", str(part["state"]))
    root.SetCustomDataByKey("scene:zoneId", str(part["zone_id"]))

    _cylinder(
        stage,
        f"{path}/Body",
        radius=radius,
        height=height * 0.82,
        position=(0.0, 0.0, -height * 0.05),
        color=color,
        collision=True,
    )
    cap_height = height * 0.18
    _cylinder(
        stage,
        f"{path}/TopCap",
        radius=radius * 0.78,
        height=cap_height,
        position=(radius * 0.10, 0.0, height / 2.0 - cap_height / 2.0),
        color=(max(color[0] * 0.82, 0.0), color[1], color[2]),
        collision=True,
    )
    _cube(
        stage,
        f"{path}/OrientationTab",
        dimensions=(radius * 0.50, radius * 0.30, cap_height * 0.70),
        position=(
            radius * 0.72,
            0.0,
            height / 2.0 - cap_height * 0.45,
        ),
        color=(0.50, 0.01, 0.01),
        collision=True,
    )


def _create_parts(stage: Usd.Stage, config: Mapping[str, Any]) -> None:
    stage.DefinePrim("/World/Parts", "Xform")
    for part in config["parts"]:
        _create_part(stage, part)


def _create_bin(stage: Usd.Stage, bin_config: Mapping[str, Any]) -> None:
    path = f"/World/Bins/{_prim_name(bin_config['id'])}"
    root = stage.DefinePrim(path, "Xform")
    _set_pose(root, bin_config["pose"])
    _make_rigid_body(root, float(bin_config["mass_kg"]))

    size_x, size_y, size_z = (float(value) for value in bin_config["size_m"])
    wall = float(bin_config["wall_thickness_m"])
    divider = float(bin_config["divider_thickness_m"])
    bottom = float(bin_config["bottom_thickness_m"])
    rows = int(bin_config["grid"]["rows"])
    columns = int(bin_config["grid"]["columns"])
    if rows != 2 or columns != 3:
        raise ValueError("The frozen bin must use a 2 x 3 grid.")
    if min(size_x, size_y, size_z, wall, divider, bottom) <= 0.0:
        raise ValueError("Bin dimensions and thicknesses must be positive.")
    if 2.0 * wall >= min(size_x, size_y) or bottom >= size_z:
        raise ValueError("Bin wall or bottom thickness is too large.")

    color = _color(bin_config.get("color_rgb"), (0.18, 0.20, 0.22))
    wall_height = size_z - bottom
    wall_center_z = bottom / 2.0
    bottom_center_z = -size_z / 2.0 + bottom / 2.0
    interior_x = size_x - 2.0 * wall
    interior_y = size_y - 2.0 * wall

    # The root is the only RigidBody. Every child below is an individual
    # collider, which leaves the top physically open for inserted parts.
    _cube(
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
        _cube(
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
        _cube(
            stage,
            f"{path}/{name}",
            dimensions=(interior_x, wall, wall_height),
            position=(0.0, y, wall_center_z),
            color=color,
            collision=True,
        )

    column_width = interior_x / columns
    for index in range(1, columns):
        x = -interior_x / 2.0 + column_width * index
        _cube(
            stage,
            f"{path}/Divider_Column_{index}",
            dimensions=(divider, interior_y, wall_height),
            position=(x, 0.0, wall_center_z),
            color=color,
            collision=True,
        )

    row_height = interior_y / rows
    for index in range(1, rows):
        y = -interior_y / 2.0 + row_height * index
        _cube(
            stage,
            f"{path}/Divider_Row_{index}",
            dimensions=(interior_x, divider, wall_height),
            position=(0.0, y, wall_center_z),
            color=color,
            collision=True,
        )

    handle_config = bin_config.get("handle", {})
    handle_size = handle_config.get(
        "size_m",
        [max(size_x * 0.38, wall * 4.0), max(wall * 2.0, 0.012), 0.022],
    )
    handle_offset = handle_config.get("offset_m", [0.0, -size_y / 2.0 - wall, 0.015])
    handle_width = float(handle_size[0])
    handle_depth = float(handle_size[1])
    handle_height = float(handle_size[2])
    handle_x = float(handle_offset[0])
    handle_y = float(handle_offset[1])
    handle_z = float(handle_offset[2])
    handle_bar = max(min(handle_depth, handle_height) * 0.55, 0.005)
    for name, x in (
        ("Handle_Mount_Left", handle_x - handle_width / 2.0),
        ("Handle_Mount_Right", handle_x + handle_width / 2.0),
    ):
        _cube(
            stage,
            f"{path}/{name}",
            dimensions=(handle_bar, handle_depth, handle_height),
            position=(x, handle_y, handle_z),
            color=(0.10, 0.11, 0.12),
            collision=True,
        )
    _cube(
        stage,
        f"{path}/Handle_Crossbar",
        dimensions=(handle_width + handle_bar, handle_bar, handle_bar),
        position=(
            handle_x,
            handle_y - handle_depth / 2.0,
            handle_z + handle_height / 2.0,
        ),
        color=(0.10, 0.11, 0.12),
        collision=True,
    )

    root.SetCustomDataByKey(
        "scene:recipePartIds",
        Vt.StringArray([str(item) for item in bin_config["recipe_part_ids"]]),
    )
    root.SetCustomDataByKey(
        "scene:emptySlots",
        int(
            bin_config.get(
                "empty_slots_after_pack",
                bin_config.get("empty_slots", rows * columns),
            )
        ),
    )
    root.SetCustomDataByKey(
        "scene:initialStationId", str(bin_config["initial_station_id"])
    )


def _look_at_matrix(eye: Sequence[float], target: Sequence[float]) -> Gf.Matrix4d:
    eye_point = _vec3(eye, label="camera.position_m")
    target_point = _vec3(target, label="camera.look_at_m")
    if (target_point - eye_point).GetLength() < 1e-6:
        raise ValueError("Camera eye and look-at target cannot be identical.")
    view = Gf.Matrix4d().SetLookAt(
        eye_point,
        target_point,
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    return view.GetInverse()


def _create_cameras(stage: Usd.Stage, cameras: Iterable[Mapping[str, Any]]) -> None:
    stage.DefinePrim("/World/Cameras", "Xform")
    for camera_config in cameras:
        path = f"/World/Cameras/{_prim_name(camera_config['id'])}"
        camera = UsdGeom.Camera.Define(stage, path)
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(
            _look_at_matrix(
                camera_config["pose"]["position_m"],
                camera_config["look_at_m"],
            )
        )

        horizontal_aperture_mm = 20.955
        horizontal_fov_rad = math.radians(
            float(camera_config.get("horizontal_fov_deg", 68.0))
        )
        focal_length_mm = horizontal_aperture_mm / (
            2.0 * math.tan(horizontal_fov_rad / 2.0)
        )
        camera.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        camera.CreateFocalLengthAttr(focal_length_mm)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))

        resolution = camera_config.get("resolution_px", [1280, 720])
        prim = camera.GetPrim()
        prim.CreateAttribute(
            "scene:resolutionX", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(resolution[0]))
        prim.CreateAttribute(
            "scene:resolutionY", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(resolution[1]))
        prim.SetCustomDataByKey(
            "scene:consumers",
            Vt.StringArray([str(item) for item in camera_config["consumers"]]),
        )


def _create_lighting(stage: Usd.Stage) -> None:
    stage.DefinePrim("/World/Lighting", "Xform")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(700.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr(900.0)
    key.CreateAngleAttr(0.7)
    _set_xform(key.GetPrim(), rpy_deg=(-50.0, 20.0, -25.0))


def _create_robots(
    stage: Usd.Stage,
    robots: Iterable[Mapping[str, Any]],
    *,
    franka_asset_path: str,
) -> None:
    stage.DefinePrim("/World/Robots", "Xform")
    for robot in robots:
        path = f"/World/Robots/{_prim_name(robot['id'])}"
        root = stage.DefinePrim(path, "Xform")
        if not root.GetReferences().AddReference(franka_asset_path):
            raise RuntimeError(
                f"Failed to add Franka USD reference for {robot['id']}: "
                f"{franka_asset_path}"
            )
        _set_pose(root, robot["base_pose"])
        root.SetCustomDataByKey("scene:model", str(robot["model"]))
        root.SetCustomDataByKey("scene:executor", str(robot["executor"]))
        root.SetCustomDataByKey("scene:role", str(robot["role"]))
        root.SetCustomDataByKey(
            "scene:softWorkRadiusM", float(robot["soft_work_radius_m"])
        )


def build_scene(
    stage: Usd.Stage,
    config: Mapping[str, Any],
    *,
    franka_asset_path: str | None,
    include_robots: bool = True,
) -> Usd.Stage:
    """Populate ``stage`` with the frozen scene and return it."""

    if stage is None:
        raise ValueError("A valid USD stage is required.")
    if include_robots and not franka_asset_path:
        raise ValueError(
            "include_robots=True requires a previously resolved Franka USD asset."
        )

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    world.SetCustomDataByKey("scene:id", str(config.get("scene_id", "unknown")))
    world.SetCustomDataByKey(
        "scene:schemaVersion", str(config.get("schema_version", "unknown"))
    )

    _create_physics_scene(stage, config["physics"])
    _create_environment(stage, config)
    _create_markers(stage, config)
    _create_parts(stage, config)
    stage.DefinePrim("/World/Bins", "Xform")
    _create_bin(stage, config["bin"])
    _create_cameras(stage, config["cameras"])
    _create_lighting(stage)
    if include_robots:
        assert franka_asset_path is not None
        _create_robots(
            stage,
            config["robots"],
            franka_asset_path=franka_asset_path,
        )

    safety = config.get("safety", {})
    workflow = config.get("workflow", {})
    world.SetCustomDataByKey(
        "scene:handoffZone", str(safety.get("handoff_zone", "HANDOFF_CENTER"))
    )
    world.SetCustomDataByKey(
        "scene:tokenSequence",
        Vt.StringArray([str(token) for token in workflow.get("token_sequence", [])]),
    )
    world.SetCustomDataByKey(
        "scene:handoffReadyEvent",
        str(workflow.get("handoff_ready_event", "handoff_ready")),
    )
    return stage
