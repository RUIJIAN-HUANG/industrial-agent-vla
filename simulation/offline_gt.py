"""Isolated Isaac ground-truth access for scripted experts and offline QA.

Nothing in this module is imported by the Observation or Canonical Recorder
paths.  Callers may persist its detailed reports only below an ``offline_gt``
artifact directory; Canonical fields receive at most the final episode outcome.
"""

from __future__ import annotations

from itertools import product
from math import acos, cos, radians, sqrt
from typing import Any, Mapping, Sequence

from simulation.v2_terminal_success import vertical_error_rad


def slot_interior_bounds(
    bin_config: Mapping[str, Any],
    slot_id: str,
) -> dict[str, list[float]]:
    """Return one slot's usable XYZ bounds in the bin-local frame."""

    size = [float(value) for value in bin_config["size_m"]]
    if len(size) != 3:
        raise ValueError("bin size must contain three values")
    wall = float(bin_config["wall_thickness_m"])
    divider = float(bin_config["divider_thickness_m"])
    bottom = float(bin_config["bottom_thickness_m"])
    slots = bin_config.get("slots")
    if not isinstance(slots, list):
        raise ValueError("bin slots must be a list")

    centers: dict[str, tuple[float, float]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise ValueError("each bin slot must be an object")
        center = slot.get("center_local_m")
        if not isinstance(center, Sequence) or isinstance(center, (str, bytes)):
            raise ValueError("each bin slot requires center_local_m")
        if len(center) != 3:
            raise ValueError("slot center_local_m must contain three values")
        identifier = str(slot.get("id", ""))
        if not identifier or identifier in centers:
            raise ValueError("bin slot ids must be non-empty and unique")
        centers[identifier] = (float(center[0]), float(center[1]))
    if slot_id not in centers:
        raise ValueError(f"unknown bin slot: {slot_id}")

    x_centers = sorted({center[0] for center in centers.values()})
    y_centers = sorted({center[1] for center in centers.values()})
    target_x, target_y = centers[slot_id]

    def _axis_bounds(
        center: float,
        axis_centers: list[float],
        outer_min: float,
        outer_max: float,
    ) -> tuple[float, float]:
        index = axis_centers.index(center)
        lower = outer_min
        upper = outer_max
        if index > 0:
            lower = (axis_centers[index - 1] + center) / 2.0 + divider / 2.0
        if index + 1 < len(axis_centers):
            upper = (center + axis_centers[index + 1]) / 2.0 - divider / 2.0
        if lower >= upper:
            raise ValueError("slot interior is empty")
        return lower, upper

    x_min, x_max = _axis_bounds(
        target_x,
        x_centers,
        -size[0] / 2.0 + wall,
        size[0] / 2.0 - wall,
    )
    y_min, y_max = _axis_bounds(
        target_y,
        y_centers,
        -size[1] / 2.0 + wall,
        size[1] / 2.0 - wall,
    )
    return {
        "min": [x_min, y_min, -size[2] / 2.0 + bottom],
        "max": [x_max, y_max, size[2] / 2.0],
    }


def p01_s11_task_pass(
    *,
    nearest_slot_id: str,
    inside_target_cell: bool,
    upright: bool,
    containment_axis_pass: Mapping[str, Any],
) -> bool:
    """Apply the atomic task semantics without an invisible wall-clearance gate.

    X/Y are already constrained by membership in the S11 cell.  The complete
    world-aligned part bound remains a useful diagnostic, but its small flange
    overhang must not redefine "put P01 in S11".  Z containment still prevents
    accepting a part that is above or below the bin.
    """

    return bool(
        nearest_slot_id == "S11"
        and inside_target_cell
        and upright
        and containment_axis_pass.get("z") is True
    )


def w01_s14_task_pass(
    *, nearest_slot_id: str, center_inside_target_cell: bool
) -> bool:
    """Apply the group-lead contract: W01 is stably inside S14.

    Final orientation and full-bound containment remain diagnostics only.
    """

    return bool(nearest_slot_id == "S14" and center_inside_target_cell)


class OfflineGtProbe:
    """Read live USD transforms without exposing them to online consumers."""

    def __init__(self, stage: Any) -> None:
        try:
            from pxr import Usd, UsdGeom
            from isaacsim.core.utils.prims import get_prim_at_path
        except ImportError as exc:
            raise RuntimeError(
                "offline_gt requires Isaac Sim USD bindings and prim utilities"
            ) from exc
        if stage is None:
            raise TypeError("stage must not be None")
        self._stage = stage
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._get_prim_at_path = get_prim_at_path

    def _prim(self, path: str) -> Any:
        # Use Isaac Sim's public prim utility instead of calling UsdStage
        # methods directly.  Some 5.1 Python/Boost builds expose a Stage whose
        # GetPrimAtPath/Traverse overloads reject otherwise valid arguments.
        prim = self._get_prim_at_path(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"offline_gt prim is missing: {path}")
        return prim

    def _world_aligned_range(self, path: str) -> Any:
        cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(),
            [self._UsdGeom.Tokens.default_, self._UsdGeom.Tokens.render],
            useExtentsHint=False,
            ignoreVisibility=True,
        )
        return cache.ComputeWorldBound(self._prim(path)).ComputeAlignedRange()

    def _world_matrix(self, path: str) -> Any:
        cache = self._UsdGeom.XformCache(self._Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(self._prim(path))

    def world_position(self, path: str) -> list[float]:
        translation = self._world_matrix(path).ExtractTranslation()
        return [float(value) for value in translation]

    def p01_in_s11(
        self,
        *,
        part_path: str,
        bin_path: str,
        bin_config: Mapping[str, Any],
        upright_tolerance_deg: float = 15.0,
    ) -> dict[str, Any]:
        """Offline-only slot and upright label for the atomic task."""

        try:
            from pxr import Gf
        except ImportError as exc:
            raise RuntimeError("offline_gt requires Isaac Sim Gf bindings") from exc
        tolerance = float(upright_tolerance_deg)
        if tolerance <= 0.0 or tolerance > 45.0:
            raise ValueError("upright_tolerance_deg must be in (0, 45]")

        bin_inverse = self._world_matrix(bin_path).GetInverse()
        part_matrix = self._world_matrix(part_path)
        center = bin_inverse.Transform(part_matrix.ExtractTranslation())
        center_local = [float(center[index]) for index in range(3)]
        slots = {item["id"]: item for item in bin_config["slots"]}
        target = [float(value) for value in slots["S11"]["center_local_m"]]
        nearest = min(
            slots.values(),
            key=lambda item: sum(
                (center_local[index] - float(item["center_local_m"][index])) ** 2
                for index in (0, 1)
            ),
        )["id"]
        size = [float(value) for value in bin_config["size_m"]]
        wall = float(bin_config["wall_thickness_m"])
        divider = float(bin_config["divider_thickness_m"])
        columns = int(bin_config["grid"]["columns"])
        rows = int(bin_config["grid"]["rows"])
        half_x = (size[0] - 2.0 * wall - (columns - 1) * divider) / columns / 2.0
        half_y = (size[1] - 2.0 * wall - (rows - 1) * divider) / rows / 2.0
        inside_target_cell = (
            abs(center_local[0] - target[0]) <= half_x
            and abs(center_local[1] - target[1]) <= half_y
        )

        axis_world = part_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        axis = bin_inverse.TransformDir(axis_world)
        norm = sqrt(sum(float(axis[index]) ** 2 for index in range(3)))
        upright_cosine = float(axis[2]) / norm if norm else -1.0
        upright = upright_cosine >= cos(radians(tolerance))
        contained = self.part_fully_inside_bin(
            part_path=part_path,
            bin_path=bin_path,
            bin_config=bin_config,
        )
        task_pass = p01_s11_task_pass(
            nearest_slot_id=str(nearest),
            inside_target_cell=inside_target_cell,
            upright=upright,
            containment_axis_pass=contained["axis_pass"],
        )
        return {
            "pass": task_pass,
            "part_id": "P01",
            "slot_id": "S11",
            "nearest_slot_id": nearest,
            "center_in_bin_local_m": center_local,
            "inside_target_cell": inside_target_cell,
            "upright": upright,
            "upright_cosine": upright_cosine,
            "upright_tolerance_deg": tolerance,
            "vertical_containment": bool(contained["axis_pass"]["z"]),
            "full_part_inside_bin_diagnostic": bool(contained["pass"]),
            "containment": contained,
        }

    def w01_orientation_errors(
        self, *, part_path: str, bin_path: str
    ) -> tuple[float, float]:
        """Return flatness and unsigned long-axis error for a ``wrench_y`` slot."""

        def angle(left: Sequence[float], right: Sequence[float], *, unsigned: bool) -> float:
            left_norm = sqrt(sum(float(value) ** 2 for value in left))
            right_norm = sqrt(sum(float(value) ** 2 for value in right))
            if not left_norm or not right_norm:
                raise ValueError("orientation direction must not be zero")
            dot = sum(float(a) * float(b) for a, b in zip(left, right)) / (
                left_norm * right_norm
            )
            if unsigned:
                dot = abs(dot)
            return acos(max(-1.0, min(1.0, dot)))

        part_up = self.world_direction(part_path, (0.0, 0.0, 1.0))
        part_long = self.world_direction(part_path, (1.0, 0.0, 0.0))
        bin_up = self.world_direction(bin_path, (0.0, 0.0, 1.0))
        bin_y = self.world_direction(bin_path, (0.0, 1.0, 0.0))
        return angle(part_up, bin_up, unsigned=False), angle(
            part_long, bin_y, unsigned=True
        )

    def w01_in_s14(
        self,
        *,
        part_path: str,
        bin_path: str,
        bin_config: Mapping[str, Any],
        orientation_tolerance_deg: float = 15.0,
    ) -> dict[str, Any]:
        """Accept a stable W01 whose center is inside the S14 cell.

        ``flat_y``/``wrench_y`` describe the frozen initial scene and slot
        geometry.  They are retained as diagnostics, not as terminal task
        requirements for the instruction "把W01放到S14中".
        """

        tolerance = radians(float(orientation_tolerance_deg))
        if tolerance <= 0.0 or tolerance > radians(45.0):
            raise ValueError("orientation_tolerance_deg must be in (0, 45]")
        bin_inverse = self._world_matrix(bin_path).GetInverse()
        part_matrix = self._world_matrix(part_path)
        center = bin_inverse.Transform(part_matrix.ExtractTranslation())
        center_local = [float(center[index]) for index in range(3)]
        allowed = slot_interior_bounds(bin_config, "S14")
        center_inside_target_cell = all(
            allowed["min"][axis] <= center_local[axis] <= allowed["max"][axis]
            for axis in range(3)
        )
        slots = {item["id"]: item for item in bin_config["slots"]}
        nearest = min(
            slots.values(),
            key=lambda item: sum(
                (
                    center_local[index]
                    - float(item["center_local_m"][index])
                )
                ** 2
                for index in (0, 1)
            ),
        )["id"]
        full_containment_diagnostic = self.part_fully_inside_slot(
            part_path=part_path,
            bin_path=bin_path,
            bin_config=bin_config,
            slot_id="S14",
        )
        flat_error, heading_error = self.w01_orientation_errors(
            part_path=part_path, bin_path=bin_path
        )
        orientation_diagnostic_pass = (
            flat_error <= tolerance and heading_error <= tolerance
        )
        return {
            "pass": w01_s14_task_pass(
                nearest_slot_id=str(nearest),
                center_inside_target_cell=center_inside_target_cell,
            ),
            "part_id": "W01",
            "slot_id": "S14",
            "nearest_slot_id": nearest,
            "center_in_bin_local_m": center_local,
            "center_inside_target_cell": center_inside_target_cell,
            "flat_error_rad": flat_error,
            "heading_error_rad": heading_error,
            "orientation_tolerance_deg": float(orientation_tolerance_deg),
            "orientation_required": False,
            "orientation_diagnostic_pass": orientation_diagnostic_pass,
            "full_part_inside_slot_diagnostic": bool(
                full_containment_diagnostic["pass"]
            ),
            "containment": full_containment_diagnostic,
        }

    def local_point_to_world(
        self,
        path: str,
        local_point: Sequence[float],
    ) -> list[float]:
        if len(local_point) != 3:
            raise ValueError("local point must contain three values")
        try:
            from pxr import Gf
        except ImportError as exc:
            raise RuntimeError("offline_gt requires Isaac Sim Gf bindings") from exc
        transformed = self._world_matrix(path).Transform(
            Gf.Vec3d(*(float(value) for value in local_point))
        )
        return [float(value) for value in transformed]

    def world_direction(
        self,
        path: str,
        local_direction: Sequence[float],
    ) -> list[float]:
        """Transform a local direction into world space without translation."""

        if len(local_direction) != 3:
            raise ValueError("local direction must contain three values")
        try:
            from pxr import Gf
        except ImportError as exc:
            raise RuntimeError("offline_gt requires Isaac Sim Gf bindings") from exc
        direction = Gf.Vec3d(*(float(value) for value in local_direction))
        transformed = self._world_matrix(path).TransformDir(direction)
        return [float(value) for value in transformed]

    def part_vertical_error_rad(
        self,
        *,
        part_path: str,
        bin_path: str,
        part_axis_local: Sequence[float] = (0.0, 0.0, 1.0),
        bin_vertical_local: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> float:
        """Measure the directed P01 axis error against the bin vertical."""

        return vertical_error_rad(
            self.world_direction(part_path, part_axis_local),
            self.world_direction(bin_path, bin_vertical_local),
        )

    def part_fully_inside_bin(
        self,
        *,
        part_path: str,
        bin_path: str,
        bin_config: Mapping[str, Any],
        numerical_tolerance_m: float = 0.001,
    ) -> dict[str, Any]:
        """Conservatively require the complete part bound inside bin walls."""

        tolerance = float(numerical_tolerance_m)
        if tolerance < 0.0 or tolerance > 0.001:
            raise ValueError("numerical tolerance must be in [0, 0.001] m")
        size = [float(value) for value in bin_config["size_m"]]
        if len(size) != 3:
            raise ValueError("bin size must contain three values")
        wall = float(bin_config["wall_thickness_m"])
        bottom = float(bin_config["bottom_thickness_m"])

        world_range = self._world_aligned_range(part_path)
        world_min = world_range.GetMin()
        world_max = world_range.GetMax()
        bin_inverse = self._world_matrix(bin_path).GetInverse()

        local_corners: list[list[float]] = []
        for x, y, z in product(
            (float(world_min[0]), float(world_max[0])),
            (float(world_min[1]), float(world_max[1])),
            (float(world_min[2]), float(world_max[2])),
        ):
            try:
                from pxr import Gf
            except ImportError as exc:
                raise RuntimeError("offline_gt requires Isaac Sim Gf bindings") from exc
            point = bin_inverse.Transform(Gf.Vec3d(x, y, z))
            local_corners.append([float(value) for value in point])

        local_min = [min(corner[axis] for corner in local_corners) for axis in range(3)]
        local_max = [max(corner[axis] for corner in local_corners) for axis in range(3)]
        allowed_min = [
            -size[0] / 2.0 + wall,
            -size[1] / 2.0 + wall,
            -size[2] / 2.0 + bottom,
        ]
        allowed_max = [size[0] / 2.0 - wall, size[1] / 2.0 - wall, size[2] / 2.0]
        axis_pass = [
            local_min[axis] >= allowed_min[axis] - tolerance
            and local_max[axis] <= allowed_max[axis] + tolerance
            for axis in range(3)
        ]
        return {
            "pass": all(axis_pass),
            "part_path": part_path,
            "bin_path": bin_path,
            "part_bound_in_bin_local_m": {"min": local_min, "max": local_max},
            "allowed_bin_interior_m": {"min": allowed_min, "max": allowed_max},
            "axis_pass": {"x": axis_pass[0], "y": axis_pass[1], "z": axis_pass[2]},
            "numerical_tolerance_m": tolerance,
        }

    def part_fully_inside_slot(
        self,
        *,
        part_path: str,
        bin_path: str,
        bin_config: Mapping[str, Any],
        slot_id: str,
        numerical_tolerance_m: float = 0.001,
    ) -> dict[str, Any]:
        """Require the complete part bound inside one configured bin slot."""

        tolerance = float(numerical_tolerance_m)
        if tolerance < 0.0 or tolerance > 0.001:
            raise ValueError("numerical tolerance must be in [0, 0.001] m")
        allowed = slot_interior_bounds(bin_config, slot_id)

        world_range = self._world_aligned_range(part_path)
        world_min = world_range.GetMin()
        world_max = world_range.GetMax()
        bin_inverse = self._world_matrix(bin_path).GetInverse()
        try:
            from pxr import Gf
        except ImportError as exc:
            raise RuntimeError("offline_gt requires Isaac Sim Gf bindings") from exc

        local_corners: list[list[float]] = []
        for x, y, z in product(
            (float(world_min[0]), float(world_max[0])),
            (float(world_min[1]), float(world_max[1])),
            (float(world_min[2]), float(world_max[2])),
        ):
            point = bin_inverse.Transform(Gf.Vec3d(x, y, z))
            local_corners.append([float(value) for value in point])

        local_min = [min(corner[axis] for corner in local_corners) for axis in range(3)]
        local_max = [max(corner[axis] for corner in local_corners) for axis in range(3)]
        axis_pass = [
            local_min[axis] >= allowed["min"][axis] - tolerance
            and local_max[axis] <= allowed["max"][axis] + tolerance
            for axis in range(3)
        ]
        return {
            "pass": all(axis_pass),
            "part_path": part_path,
            "bin_path": bin_path,
            "slot_id": slot_id,
            "part_bound_in_bin_local_m": {"min": local_min, "max": local_max},
            "allowed_slot_interior_m": allowed,
            "axis_pass": {"x": axis_pass[0], "y": axis_pass[1], "z": axis_pass[2]},
            "numerical_tolerance_m": tolerance,
        }
