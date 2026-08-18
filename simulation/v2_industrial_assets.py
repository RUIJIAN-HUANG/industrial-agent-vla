"""Program-generated V2 shafts, hex nuts, and open-end wrenches."""

from __future__ import annotations

import math
from typing import Any, Mapping


SUPPORTED_PART_TYPES = {"shaft", "nut", "wrench"}


def validate_part_spec(part: Mapping[str, Any]) -> list[str]:
    """Validate one asset without importing Isaac Sim or USD bindings."""

    errors: list[str] = []
    part_id = str(part.get("id", "<missing>"))
    part_type = part.get("part_type")
    if part_type not in SUPPORTED_PART_TYPES:
        return [f"{part_id}.part_type must be one of {sorted(SUPPORTED_PART_TYPES)}"]
    geometry = part.get("geometry")
    if not isinstance(geometry, Mapping):
        return [f"{part_id}.geometry must be an object"]
    required = {
        "shaft": (
            "radius_m",
            "height_m",
            "flange_radius_m",
            "flange_height_m",
            "mass_kg",
        ),
        "nut": ("across_flats_m", "hole_diameter_m", "height_m", "mass_kg"),
        "wrench": (
            "length_m",
            "handle_width_m",
            "head_width_m",
            "thickness_m",
            "mass_kg",
        ),
    }[str(part_type)]
    values: dict[str, float] = {}
    for field in required:
        try:
            value = float(geometry[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{part_id}.geometry.{field} must be numeric")
            continue
        if not math.isfinite(value) or value <= 0.0:
            errors.append(f"{part_id}.geometry.{field} must be positive")
        values[field] = value
    if errors:
        return errors
    if part_type == "shaft":
        if values["flange_radius_m"] <= values["radius_m"]:
            errors.append(f"{part_id} flange must make shaft orientation visible")
        if values["flange_height_m"] >= values["height_m"] / 2.0:
            errors.append(f"{part_id} flange is too tall")
    elif part_type == "nut":
        if values["hole_diameter_m"] >= values["across_flats_m"] * 0.65:
            errors.append(f"{part_id} nut hole leaves insufficient wall thickness")
    else:
        if values["head_width_m"] <= values["handle_width_m"] * 1.5:
            errors.append(f"{part_id} wrench head must be visually distinct")
        if values["length_m"] <= values["head_width_m"] * 2.0:
            errors.append(f"{part_id} wrench handle is too short")
    return errors


def asset_summary(parts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return an auditable summary used by static gates and evidence JSON."""

    errors: list[str] = []
    counts = {part_type: 0 for part_type in sorted(SUPPORTED_PART_TYPES)}
    total_mass = 0.0
    ids: list[str] = []
    for part in parts:
        ids.append(str(part.get("id", "<missing>")))
        errors.extend(validate_part_spec(part))
        part_type = part.get("part_type")
        if part_type in counts:
            counts[str(part_type)] += 1
        try:
            total_mass += float(part["geometry"]["mass_kg"])
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "part_ids": ids,
        "type_counts": counts,
        "total_part_mass_kg": total_mass,
        "all_program_generated": True,
        "external_assets": [],
        "errors": errors,
    }


def _hex_ring_mesh(stage: Any, path: str, geometry: Mapping[str, Any], color: Any) -> None:
    from pxr import Gf, UsdGeom

    across_flats = float(geometry["across_flats_m"])
    hole_radius = float(geometry["hole_diameter_m"]) / 2.0
    height = float(geometry["height_m"])
    outer_radius = across_flats / math.sqrt(3.0)
    z_values = (-height / 2.0, height / 2.0)
    points: list[Any] = []
    for z in z_values:
        for radius in (outer_radius, hole_radius):
            for index in range(6):
                angle = math.radians(60.0 * index + 30.0)
                points.append(
                    Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z)
                )

    def index(layer: int, ring: int, segment: int) -> int:
        return layer * 12 + ring * 6 + segment % 6

    faces: list[list[int]] = []
    for segment in range(6):
        nxt = segment + 1
        faces.append(
            [index(1, 0, segment), index(1, 0, nxt), index(1, 1, nxt), index(1, 1, segment)]
        )
        faces.append(
            [index(0, 1, segment), index(0, 1, nxt), index(0, 0, nxt), index(0, 0, segment)]
        )
        faces.append(
            [index(0, 0, segment), index(0, 0, nxt), index(1, 0, nxt), index(1, 0, segment)]
        )
        faces.append(
            [index(0, 1, nxt), index(0, 1, segment), index(1, 1, segment), index(1, 1, nxt)]
        )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([item for face in faces for item in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _create_shaft(stage: Any, path: str, part: Mapping[str, Any], color: Any) -> None:
    from simulation.single_bin_scene_builder import _cylinder

    geometry = part["geometry"]
    radius = float(geometry["radius_m"])
    height = float(geometry["height_m"])
    flange_radius = float(geometry["flange_radius_m"])
    flange_height = float(geometry["flange_height_m"])
    body_height = height - flange_height
    _cylinder(
        stage,
        f"{path}/Body",
        radius=radius,
        height=body_height,
        position=(0.0, 0.0, -flange_height / 2.0),
        color=color,
        collision=True,
    )
    _cylinder(
        stage,
        f"{path}/Orientation_Flange",
        radius=flange_radius,
        height=flange_height,
        position=(0.0, 0.0, height / 2.0 - flange_height / 2.0),
        color=(min(color[0] * 1.12, 1.0), color[1] * 0.72, color[2] * 0.72),
        collision=True,
    )
    _cylinder(
        stage,
        f"{path}/Orientation_Recess",
        radius=radius * 0.38,
        height=0.001,
        position=(0.0, 0.0, height / 2.0 + 0.0006),
        color=(0.04, 0.04, 0.04),
        collision=False,
    )


def _create_nut(stage: Any, path: str, part: Mapping[str, Any], color: Any) -> None:
    from simulation.single_bin_scene_builder import _cube

    geometry = part["geometry"]
    across_flats = float(geometry["across_flats_m"])
    hole_radius = float(geometry["hole_diameter_m"]) / 2.0
    height = float(geometry["height_m"])
    _hex_ring_mesh(stage, f"{path}/Hex_Ring_Visual", geometry, color)
    outer_apothem = across_flats / 2.0
    radial_thickness = outer_apothem - hole_radius
    collider_depth = across_flats * 0.46
    collider_center = hole_radius + radial_thickness / 2.0
    for index in range(6):
        angle_deg = 60.0 * index
        angle = math.radians(angle_deg)
        _cube(
            stage,
            f"{path}/Collider_{index + 1}",
            dimensions=(radial_thickness, collider_depth, height),
            position=(
                collider_center * math.cos(angle),
                collider_center * math.sin(angle),
                0.0,
            ),
            rpy_deg=(0.0, 0.0, angle_deg),
            color=color,
            collision=True,
            opacity=0.02,
        )


def _create_wrench(stage: Any, path: str, part: Mapping[str, Any], color: Any) -> None:
    from simulation.single_bin_scene_builder import _cube

    geometry = part["geometry"]
    length = float(geometry["length_m"])
    handle_width = float(geometry["handle_width_m"])
    head_width = float(geometry["head_width_m"])
    thickness = float(geometry["thickness_m"])
    head_depth = head_width * 0.62
    handle_length = length - head_depth
    handle_center_y = -head_depth / 2.0
    _cube(
        stage,
        f"{path}/Handle",
        dimensions=(handle_width, handle_length, thickness),
        position=(0.0, handle_center_y, 0.0),
        color=color,
        collision=True,
    )
    jaw_width = max((head_width - handle_width) / 2.0, 0.006)
    head_center_y = length / 2.0 - head_depth / 2.0
    _cube(
        stage,
        f"{path}/Head_Base",
        dimensions=(head_width, head_depth * 0.38, thickness),
        position=(0.0, head_center_y - head_depth * 0.31, 0.0),
        color=color,
        collision=True,
    )
    jaws = (
        ("Left_Jaw", -head_width / 2.0 + jaw_width / 2.0),
        ("Right_Jaw", head_width / 2.0 - jaw_width / 2.0),
    )
    for name, x in jaws:
        _cube(
            stage,
            f"{path}/{name}",
            dimensions=(jaw_width, head_depth, thickness),
            position=(x, head_center_y, 0.0),
            color=color,
            collision=True,
        )


def create_part(stage: Any, part: Mapping[str, Any]) -> Any:
    """Create one dynamic asset under its stable ``/World/Parts/<ID>`` path."""

    errors = validate_part_spec(part)
    if errors:
        raise ValueError("; ".join(errors))
    from simulation.single_bin_scene_builder import (
        _color,
        _make_rigid_body,
        _prim_name,
        _set_pose,
    )

    path = f"/World/Parts/{_prim_name(str(part['id']))}"
    root = stage.DefinePrim(path, "Xform")
    _set_pose(root, part["pose"])
    _make_rigid_body(root, float(part["geometry"]["mass_kg"]))
    root.SetCustomDataByKey("scene:partType", str(part["part_type"]))
    root.SetCustomDataByKey(
        "scene:initialOrientationState", str(part["initial_orientation_state"])
    )
    root.SetCustomDataByKey("scene:zoneId", str(part["zone_id"]))
    color = _color(part.get("color_rgb"), (0.55, 0.55, 0.55))
    creators = {
        "shaft": _create_shaft,
        "nut": _create_nut,
        "wrench": _create_wrench,
    }
    creators[str(part["part_type"])](stage, path, part, color)
    return root


def create_parts(stage: Any, parts: list[Mapping[str, Any]]) -> None:
    stage.DefinePrim("/World/Parts", "Xform")
    for part in parts:
        create_part(stage, part)
