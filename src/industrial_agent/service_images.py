"""Fail-closed CAS image consumption for VLA and YOLO service handlers.

The Supervisor transports immutable ``ImageReference`` objects. Model-service
entry points must call this module before inference so no service can replace a
missing CAS blob with a black placeholder or decode an unverified path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import OPENVLA_OFT_EXECUTOR_NAME, PI05_EXECUTOR_NAME
from .errors import FailureCode, ImageCasError
from .image_cas import ImageCas, ResolvedRgbFrame
from .observation import FROZEN_IMAGE_HEIGHT, FROZEN_IMAGE_WIDTH


FROZEN_RGB_SIZE = (FROZEN_IMAGE_WIDTH, FROZEN_IMAGE_HEIGHT)
FROZEN_RGB_CAMERA_IDS = frozenset(
    {
        "CAM_A_TOP",
        "CAM_HANDOFF",
        "CAM_B_TOP",
    }
)
VLA_CAMERA_BY_EXECUTOR = {
    PI05_EXECUTOR_NAME: "CAM_A_TOP",
    OPENVLA_OFT_EXECUTOR_NAME: "CAM_B_TOP",
}


@dataclass(frozen=True)
class ResolvedVlaModelImages:
    """Verified pixels ready for one frozen VLA model call."""

    full_image: ResolvedRgbFrame
    wrist_image: None = None


@dataclass(frozen=True)
class ResolvedYoloModelImage:
    """Verified pixels ready for one YOLO model call."""

    image: ResolvedRgbFrame


class CasRequestImageResolver:
    """Resolve service request references through the shared verified CAS."""

    def __init__(self, image_cas: ImageCas):
        self.image_cas = image_cas

    @classmethod
    def from_agent_config(
        cls,
        config: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> CasRequestImageResolver:
        return cls(ImageCas.from_agent_config(config, environ=environ))

    def resolve_vla_request(
        self,
        request: Mapping[str, Any],
    ) -> ResolvedVlaModelImages:
        """Resolve the exact top-view image required by a frozen VLA request."""

        if not isinstance(request, Mapping):
            raise _metadata_error("VLA infer request must be an object")
        executor = request.get("executor")
        if not isinstance(executor, str):
            raise _metadata_error("VLA infer request.executor must be a string")
        expected_camera_id = VLA_CAMERA_BY_EXECUTOR.get(executor)
        if expected_camera_id is None:
            raise _metadata_error(f"unsupported frozen VLA executor: {executor!r}")
        model_input = request.get("model_input")
        if not isinstance(model_input, Mapping):
            raise _metadata_error("VLA infer request.model_input must be an object")

        if executor == OPENVLA_OFT_EXECUTOR_NAME:
            full_image = model_input.get("full_image")
            wrist_image = model_input.get("wrist_image")
        else:
            observation = model_input.get("observation")
            camera = (
                observation.get("camera") if isinstance(observation, Mapping) else None
            )
            if not isinstance(camera, Mapping):
                raise _metadata_error(
                    "π0.5 model_input.observation.camera must be an object"
                )
            full_image = camera.get("full_image")
            wrist_image = camera.get("wrist_image")

        if wrist_image is not None:
            raise _metadata_error(
                "frozen three-camera profile requires wrist_image=null"
            )
        resolved = self.image_cas.resolve_rgb(
            full_image,
            expected_camera_id=expected_camera_id,
            expected_size=FROZEN_RGB_SIZE,
        )
        return ResolvedVlaModelImages(full_image=resolved)

    def resolve_yolo_request(
        self,
        request: Mapping[str, Any],
    ) -> ResolvedYoloModelImage:
        """Resolve one frozen-camera YOLO request without lifecycle data."""

        if not isinstance(request, Mapping) or request.get("detector") != "yolo":
            raise _metadata_error("YOLO detect request.detector must be 'yolo'")
        image = request.get("image")
        if not isinstance(image, Mapping):
            raise _metadata_error("YOLO detect request.image must be an object")
        camera_id = image.get("camera_id")
        if not isinstance(camera_id, str) or camera_id not in FROZEN_RGB_CAMERA_IDS:
            raise _metadata_error(
                f"YOLO image camera_id must be one of "
                f"{sorted(FROZEN_RGB_CAMERA_IDS)}, got {camera_id!r}"
            )
        resolved = self.image_cas.resolve_rgb(
            image,
            expected_camera_id=camera_id,
            expected_size=FROZEN_RGB_SIZE,
        )
        return ResolvedYoloModelImage(image=resolved)


def _metadata_error(message: str) -> ImageCasError:
    return ImageCasError(FailureCode.CAS_METADATA_MISMATCH, message)
