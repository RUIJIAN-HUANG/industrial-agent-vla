"""Isolated Isaac ground-truth access for scripted experts and offline QA.

Nothing in this module is imported by the Observation or Canonical Recorder
paths.  Callers may persist its detailed reports only below an ``offline_gt``
artifact directory; Canonical fields receive at most the final episode outcome.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence


class OfflineGtProbe:
    """Read live USD transforms without exposing them to online consumers."""

    def __init__(self, stage: Any) -> None:
        try:
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise RuntimeError("offline_gt requires Isaac Sim USD bindings") from exc
        if not callable(getattr(stage, "GetPrimAtPath", None)):
            raise TypeError("stage must provide the USD GetPrimAtPath API")
        self._stage = stage
        self._Usd = Usd
        self._UsdGeom = UsdGeom

    def _prim(self, path: str) -> Any:
        prim = next(
            (
                candidate
                for candidate in self._stage.Traverse()
                if str(candidate.GetPath()) == path
            ),
            None,
        )
        if not prim or not prim.IsValid():
            raise RuntimeError(f"offline_gt prim is missing: {path}")
        return prim

    def _world_matrix(self, path: str) -> Any:
        cache = self._UsdGeom.XformCache(self._Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(self._prim(path))

    def world_position(self, path: str) -> list[float]:
        translation = self._world_matrix(path).ExtractTranslation()
        return [float(value) for value in translation]

    def combined_world_bound_by_names(
        self,
        *,
        under_path: str,
        prim_names: Sequence[str],
    ) -> dict[str, Any]:
        """Measure one combined world AABB for uniquely named descendant prims."""

        names = tuple(str(name) for name in prim_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("prim_names must contain unique non-empty names")
        matches: dict[str, list[Any]] = {name: [] for name in names}
        prefix = under_path.rstrip("/") + "/"
        for candidate in self._stage.Traverse():
            path = str(candidate.GetPath())
            name = path.rsplit("/", 1)[-1]
            if path.startswith(prefix) and name in matches:
                matches[name].append(candidate)
        ambiguous = {
            name: [str(prim.GetPath()) for prim in prims]
            for name, prims in matches.items()
            if len(prims) != 1
        }
        if ambiguous:
            raise RuntimeError(
                "offline_gt expected one descendant for every requested prim name: "
                f"{ambiguous}"
            )

        cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(),
            [self._UsdGeom.Tokens.default_, self._UsdGeom.Tokens.render],
            useExtentsHint=False,
            ignoreVisibility=True,
        )
        paths: dict[str, str] = {}
        minima: list[list[float]] = []
        maxima: list[list[float]] = []
        for name in names:
            prim = matches[name][0]
            paths[name] = str(prim.GetPath())
            world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minima.append([float(value) for value in world_range.GetMin()])
            maxima.append([float(value) for value in world_range.GetMax()])
        combined_min = [min(values[axis] for values in minima) for axis in range(3)]
        combined_max = [max(values[axis] for values in maxima) for axis in range(3)]
        center = [(combined_min[axis] + combined_max[axis]) / 2.0 for axis in range(3)]
        return {
            "under_path": under_path,
            "prim_paths": paths,
            "world_aabb_m": {"min": combined_min, "max": combined_max},
            "center_world_m": center,
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

        cache = self._UsdGeom.BBoxCache(
            self._Usd.TimeCode.Default(),
            [self._UsdGeom.Tokens.default_, self._UsdGeom.Tokens.render],
            useExtentsHint=False,
            ignoreVisibility=True,
        )
        world_range = cache.ComputeWorldBound(
            self._prim(part_path)
        ).ComputeAlignedRange()
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
