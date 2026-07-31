"""Production composition root for the fixed four-Agent Supervisor.

The platform-specific environment is injected through a factory so this module
does not import Isaac Sim, model weights, or simulator APIs into the Supervisor
process.  A factory may return either an ``ExecutionEnvironment`` or an
``EnvironmentHost`` that pumps an owner-thread runtime such as Isaac Sim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
import importlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from .contracts import TaskSchema
from .environment import ExecutionEnvironment, SafeStopReceipt
from .errors import AgentError
from .executor import (
    ProcessTransport,
    build_executors_from_config,
)
from .http_transport import BoundedHTTPTransport, HTTPTransportError
from .orchestrator import IndustrialAgent, RunResult
from .perception import PerceptionTransport, build_perception_from_config


LOGGER = logging.getLogger(__name__)
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_TASK_BYTES = 1024 * 1024
_T = TypeVar("_T")


class SupervisorShutdownRequested(BaseException):
    """Raised on SIGTERM so the active Agent executes its safe-stop path."""


@runtime_checkable
class EnvironmentHost(Protocol):
    """Own a platform environment and run Supervisor work in the correct thread."""

    environment: ExecutionEnvironment

    def run(self, operation: Callable[[], _T]) -> _T:
        """Execute work while servicing the platform's owner-thread loop."""

    def close(self, reason: str) -> None:
        """Release platform resources after motion has been stopped."""


class DirectEnvironmentHost:
    """Host for environments that do not require an owner-thread pump."""

    def __init__(self, environment: ExecutionEnvironment) -> None:
        if not isinstance(environment, ExecutionEnvironment):
            raise TypeError("environment must implement ExecutionEnvironment")
        self.environment = environment

    def run(self, operation: Callable[[], _T]) -> _T:
        if not callable(operation):
            raise TypeError("operation must be callable")
        return operation()

    def close(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("close reason must be a non-empty string")


def _load_json_object(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} file does not exist: {path}") from exc
    except PermissionError as exc:
        raise PermissionError(f"{label} file is not readable: {path}") from exc
    except OSError as exc:
        raise OSError(f"cannot inspect {label} file {path}: {exc}") from exc
    if size < 1:
        raise ValueError(f"{label} file is empty: {path}")
    if size > max_bytes:
        raise ValueError(f"{label} file exceeds {max_bytes} bytes: {path}")
    try:
        raw = path.read_bytes()
    except PermissionError as exc:
        raise PermissionError(f"{label} file is not readable: {path}") from exc
    except OSError as exc:
        raise OSError(f"cannot read {label} file {path}: {exc}") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} file must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} file is not valid JSON: {path}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return dict(decoded)


def load_agent_config(path: Path) -> dict[str, Any]:
    """Load a bounded production Agent configuration object."""

    return _load_json_object(path, max_bytes=_MAX_CONFIG_BYTES, label="Agent config")


def load_task(path: Path) -> TaskSchema:
    """Load and validate one immutable task contract."""

    raw = _load_json_object(path, max_bytes=_MAX_TASK_BYTES, label="Task")
    return TaskSchema.from_dict(raw)


def build_supervisor(
    config: Mapping[str, Any],
    *,
    transport_factory: (
        Callable[[str, str], ProcessTransport | PerceptionTransport] | None
    ) = None,
) -> IndustrialAgent:
    """Wire both VLA adapters, YOLO, and the Supervisor from one config."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if transport_factory is not None and not callable(transport_factory):
        raise TypeError("transport_factory must be callable")

    def default_transport_factory(
        service_name: str,
        base_url: str,
    ) -> BoundedHTTPTransport:
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError("service_name must be a non-empty string")
        return BoundedHTTPTransport(base_url)

    factory = transport_factory or default_transport_factory
    executors = build_executors_from_config(
        config,
        cast(Callable[[str, str], ProcessTransport], factory),
    )
    perception = build_perception_from_config(
        config,
        cast(Callable[[str, str], PerceptionTransport], factory),
    )
    return IndustrialAgent.from_config(
        executors,
        config,
        perception=perception,
        require_perception=True,
    )


def resolve_environment_host(
    factory_reference: str,
    config: Mapping[str, Any],
) -> EnvironmentHost:
    """Resolve ``module:callable`` and validate the returned platform host."""

    if not isinstance(factory_reference, str) or not factory_reference.strip():
        raise ValueError("environment factory reference must be non-empty")
    module_name, separator, attribute_name = factory_reference.partition(":")
    if (
        separator != ":"
        or not module_name.strip()
        or not attribute_name.strip()
        or ":" in attribute_name
    ):
        raise ValueError(
            "environment factory must use the exact form 'module.path:callable'"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"environment factory module not found: {module_name}"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            f"environment factory module could not be imported: {module_name}"
        ) from exc
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as exc:
        raise AttributeError(
            f"environment factory callable not found: {factory_reference}"
        ) from exc
    if not callable(factory):
        raise TypeError(f"environment factory is not callable: {factory_reference}")

    candidate = factory(config)
    if candidate is None:
        raise TypeError("environment factory returned None")
    if isinstance(candidate, ExecutionEnvironment):
        return DirectEnvironmentHost(candidate)
    if not isinstance(candidate, EnvironmentHost):
        raise TypeError(
            "environment factory must return ExecutionEnvironment or EnvironmentHost"
        )
    if candidate.environment is None or not isinstance(
        candidate.environment, ExecutionEnvironment
    ):
        raise TypeError("environment host exposes an invalid environment")
    return candidate


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_compatible(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"value is not JSON compatible: {type(value).__name__}")


def run_result_to_dict(result: RunResult) -> dict[str, Any]:
    """Convert a RunResult into a stable JSON-compatible process result."""

    if not isinstance(result, RunResult):
        raise TypeError("result must be RunResult")
    converted = _json_compatible(result)
    if not isinstance(converted, dict):
        raise TypeError("RunResult conversion did not produce an object")
    return converted


def _attempt_safe_stop(environment: ExecutionEnvironment, reason: str) -> bool:
    try:
        receipt = environment.safe_stop(reason)
    except (AgentError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        LOGGER.critical("safe-stop request failed reason=%s error=%s", reason, exc)
        return False
    if not isinstance(receipt, SafeStopReceipt):
        LOGGER.critical(
            "safe-stop returned invalid receipt type=%s",
            type(receipt).__name__,
        )
        return False
    if not receipt.confirmed:
        LOGGER.critical(
            "safe-stop was not confirmed stop_epoch=%s",
            receipt.stop_epoch,
        )
        return False
    return True


def _signal_handler(signum: int, frame: Any) -> None:
    del frame
    raise SupervisorShutdownRequested(f"received process signal {signum}")


def _install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed four-Agent Supervisor against a platform host."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--environment-factory",
        default=os.environ.get("INDUSTRIAL_AGENT_ENVIRONMENT_FACTORY"),
        help=(
            "Required module:callable returning ExecutionEnvironment or "
            "EnvironmentHost. May also be set through "
            "INDUSTRIAL_AGENT_ENVIRONMENT_FACTORY."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; never substitutes a mock for a missing platform host."""

    args = _argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.environment_factory is None:
        LOGGER.error(
            "--environment-factory or INDUSTRIAL_AGENT_ENVIRONMENT_FACTORY is required"
        )
        return 2

    host: EnvironmentHost | None = None
    previous_handlers: dict[int, Any] = {}
    try:
        config = load_agent_config(args.config)
        task = load_task(args.task)
        supervisor = build_supervisor(config)
        host = resolve_environment_host(args.environment_factory, config)
        previous_handlers = _install_signal_handlers()
        result = host.run(lambda: supervisor.run(task, host.environment))
        if not isinstance(result, RunResult):
            raise TypeError("environment host returned a non-RunResult value")
        stop_confirmed = _attempt_safe_stop(
            host.environment,
            "Supervisor task completed; revoke motion before process exit",
        )
        output = run_result_to_dict(result)
        output["shutdown_safe_stop_confirmed"] = stop_confirmed
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if not stop_confirmed:
            return 3
        return 0 if result.success else 1
    except (KeyboardInterrupt, SupervisorShutdownRequested) as exc:
        LOGGER.warning("Supervisor shutdown requested: %s", exc)
        if host is None:
            return 130
        return (
            130
            if _attempt_safe_stop(host.environment, "Supervisor process interrupted")
            else 3
        )
    except (
        AgentError,
        AttributeError,
        HTTPTransportError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.exception("Supervisor startup or execution failed: %s", exc)
        if host is not None:
            _attempt_safe_stop(host.environment, "Supervisor failed before clean exit")
        return 2
    finally:
        if previous_handlers:
            _restore_signal_handlers(previous_handlers)
        if host is not None:
            try:
                host.close("Supervisor process is exiting")
            except (
                AgentError,
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                LOGGER.critical("environment host close failed: %s", exc)
                raise


if __name__ == "__main__":
    sys.exit(main())
