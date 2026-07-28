"""CAS-enforcing core bound to the OpenVLA-OFT ``POST /v1/infer`` route."""

from industrial_agent.contracts import OPENVLA_OFT_EXECUTOR_NAME
from industrial_agent.service_handlers import ServiceBackend, VlaInferRequestHandler
from industrial_agent.service_images import CasRequestImageResolver


def build_v1_infer_handler(
    *,
    resolver: CasRequestImageResolver,
    backend: ServiceBackend,
) -> VlaInferRequestHandler:
    """Build the mandatory image guard that runs before OpenVLA inference."""

    return VlaInferRequestHandler(
        expected_executor=OPENVLA_OFT_EXECUTOR_NAME,
        resolver=resolver,
        backend=backend,
    )
