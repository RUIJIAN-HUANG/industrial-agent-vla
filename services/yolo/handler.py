"""CAS-enforcing core bound to the YOLO ``POST /v1/detect`` route."""

from industrial_agent.service_handlers import ServiceBackend, YoloDetectRequestHandler
from industrial_agent.service_images import CasRequestImageResolver


def build_v1_detect_handler(
    *,
    resolver: CasRequestImageResolver,
    backend: ServiceBackend,
) -> YoloDetectRequestHandler:
    """Build the mandatory image guard that runs before YOLO inference."""

    return YoloDetectRequestHandler(resolver=resolver, backend=backend)
