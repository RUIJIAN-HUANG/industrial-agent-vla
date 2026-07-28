"""CAS-enforcing core bound to the π0.5 ``POST /v1/infer`` route."""

from industrial_agent.contracts import PI05_EXECUTOR_NAME
from industrial_agent.service_handlers import ServiceBackend, VlaInferRequestHandler
from industrial_agent.service_images import CasRequestImageResolver


def build_v1_infer_handler(
    *,
    resolver: CasRequestImageResolver,
    backend: ServiceBackend,
) -> VlaInferRequestHandler:
    """Build the mandatory image guard that runs before π0.5 inference."""

    return VlaInferRequestHandler(
        expected_executor=PI05_EXECUTOR_NAME,
        resolver=resolver,
        backend=backend,
    )
