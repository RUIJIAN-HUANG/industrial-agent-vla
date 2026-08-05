"""CAS-enforcing entrypoint for YOLO detection."""

from industrial_agent.service_handlers import ServiceBackend, YoloDetectRequestHandler
from industrial_agent.service_images import CasRequestImageResolver


def build_v1_detect_handler(
    *,
    resolver: CasRequestImageResolver,
    backend: ServiceBackend,
) -> YoloDetectRequestHandler:
    return YoloDetectRequestHandler(resolver=resolver, backend=backend)
