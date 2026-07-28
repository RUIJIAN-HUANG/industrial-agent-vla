"""Transport-neutral production request guards for model-service endpoints.

HTTP/gRPC adapters bind these callables to ``/v1/infer`` or ``/v1/detect``.
Every backend invocation receives verified, immutable RGB pixels in place of
the external CAS reference. A missing or corrupt blob therefore fails before
any model backend can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .contracts import OPENVLA_OFT_EXECUTOR_NAME, PI05_EXECUTOR_NAME
from .errors import FailureCode, ImageCasError
from .service_images import CasRequestImageResolver

ServiceBackend = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class VlaInferRequestHandler:
    """Materialize one frozen VLA request before invoking its model backend."""

    expected_executor: str
    resolver: CasRequestImageResolver
    backend: ServiceBackend

    def __post_init__(self) -> None:
        if self.expected_executor not in {
            PI05_EXECUTOR_NAME,
            OPENVLA_OFT_EXECUTOR_NAME,
        }:
            raise ValueError(
                f"unsupported frozen VLA executor: {self.expected_executor!r}"
            )

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            not isinstance(request, Mapping)
            or request.get("executor") != self.expected_executor
        ):
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                f"service only accepts executor={self.expected_executor!r}",
            )

        images = self.resolver.resolve_vla_request(request)
        prepared = dict(request)
        model_input = request.get("model_input")
        if not isinstance(model_input, Mapping):  # resolver normally rejects first
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                "VLA infer request.model_input must be an object",
            )

        prepared_model_input = dict(model_input)
        if self.expected_executor == OPENVLA_OFT_EXECUTOR_NAME:
            prepared_model_input["full_image"] = images.full_image.rgb
            prepared_model_input["wrist_image"] = None
        else:
            observation = model_input.get("observation")
            if not isinstance(observation, Mapping):
                raise ImageCasError(
                    FailureCode.CAS_METADATA_MISMATCH,
                    "π0.5 model_input.observation must be an object",
                )
            camera = observation.get("camera")
            if not isinstance(camera, Mapping):
                raise ImageCasError(
                    FailureCode.CAS_METADATA_MISMATCH,
                    "π0.5 model_input.observation.camera must be an object",
                )
            prepared_camera = dict(camera)
            prepared_camera["full_image"] = images.full_image.rgb
            prepared_camera["wrist_image"] = None
            prepared_observation = dict(observation)
            prepared_observation["camera"] = prepared_camera
            prepared_model_input["observation"] = prepared_observation

        prepared["model_input"] = prepared_model_input
        return self.backend(prepared)


@dataclass(frozen=True)
class YoloDetectRequestHandler:
    """Materialize one frozen YOLO request before invoking its backend."""

    resolver: CasRequestImageResolver
    backend: ServiceBackend

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        images = self.resolver.resolve_yolo_request(request)
        prepared = dict(request)
        prepared["image"] = images.image.rgb
        return self.backend(prepared)
