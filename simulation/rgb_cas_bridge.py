"""Bridge Isaac Sim RGB annotator arrays into the shared image CAS.

This module deliberately has no top-level Isaac Sim imports.  Create the
RenderProduct and RGB annotator after ``SimulationApp`` startup, then pass the
annotator's HxWx3/4 uint8 array to :meth:`IsaacRgbCasPublisher.publish`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from industrial_agent.image_cas import ImageCas
from industrial_agent.perception import ImageReference


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    width: int
    height: int


class IsaacRgbCasPublisher:
    """Validate configured Isaac camera output before publishing a CAS reference."""

    def __init__(self, image_cas: ImageCas, cameras: Mapping[str, CameraSpec]):
        if not cameras:
            raise ValueError("at least one camera specification is required")
        self.image_cas = image_cas
        self.cameras = dict(cameras)

    @classmethod
    def from_scene_config(
        cls,
        image_cas: ImageCas,
        scene_config: Mapping[str, Any],
    ) -> IsaacRgbCasPublisher:
        raw_cameras = scene_config.get("cameras")
        if not isinstance(raw_cameras, list) or not raw_cameras:
            raise ValueError("scene_config.cameras must be a non-empty array")
        cameras: dict[str, CameraSpec] = {}
        for raw in raw_cameras:
            if not isinstance(raw, Mapping):
                raise ValueError("each camera configuration must be an object")
            camera_id = raw.get("id")
            resolution = raw.get("resolution_px")
            if (
                not isinstance(camera_id, str)
                or not camera_id
                or not isinstance(resolution, list)
                or len(resolution) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                    for value in resolution
                )
            ):
                raise ValueError("camera id and resolution_px are invalid")
            if camera_id in cameras:
                raise ValueError(f"duplicate camera id: {camera_id}")
            cameras[camera_id] = CameraSpec(
                camera_id=camera_id,
                width=resolution[0],
                height=resolution[1],
            )
        return cls(image_cas, cameras)

    def publish(self, camera_id: str, annotator_data: Any) -> ImageReference:
        """Publish one fresh RGB/RGBA annotator frame after strict shape checks."""

        spec = self.cameras.get(camera_id)
        if spec is None:
            raise ValueError(f"unknown configured camera: {camera_id}")
        frame = np.asarray(annotator_data)
        expected_rgb = (spec.height, spec.width, 3)
        expected_rgba = (spec.height, spec.width, 4)
        if frame.dtype != np.uint8 or frame.shape not in (expected_rgb, expected_rgba):
            raise ValueError(
                f"{camera_id} frame must be uint8 {expected_rgb} or {expected_rgba}; "
                f"got dtype={frame.dtype}, shape={frame.shape}"
            )
        rgb = np.ascontiguousarray(frame[:, :, :3])
        return self.image_cas.write_rgb(rgb, camera_id=camera_id)
