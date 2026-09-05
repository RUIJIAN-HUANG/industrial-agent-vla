"""Production composition root for the formal V2 π0.5 Supervisor.

The platform-specific environment is injected through a factory so this module
does not import Isaac Sim, model weights, or simulator APIs into the Supervisor
process.  A factory may return either an ``ExecutionEnvironment`` or an
``EnvironmentHost`` that pumps an owner-thread runtime such as Isaac Sim.

模块说明:
    这是生产环境下三 Agent 运行时（总控、YOLO、π0.5）的组装入口。
    平台相关的环境(如 Isaac Sim)通过工厂函数注入,因此本模块不会
    把模拟器/模型权重等依赖导入 Supervisor 进程。工厂返回的是
    ``ExecutionEnvironment`` 或包装了平台主线程运行时(如 Isaac Sim)
    的 ``EnvironmentHost``。
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
from .executor import EXECUTOR_CONFIG_FIELDS, Pi05Adapter, ProcessTransport
from .http_transport import BoundedHTTPTransport, HTTPTransportError
from .run_result import RunResult
from .v2_supervisor import V2Supervisor


LOGGER = logging.getLogger(__name__)
# 配置文件/任务文件的最大体积限制(1 MB),防止异常大文件拖垮进程
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_TASK_BYTES = 1024 * 1024
# 泛型类型变量,用于 EnvironmentHost.run 返回任意类型
_T = TypeVar("_T")


class SupervisorShutdownRequested(BaseException):
    """Raised on SIGTERM so the active Agent executes its safe-stop path."""

    # 收到 SIGTERM/SIGINT 时抛出该异常,让正在运行的 Agent
    # 有机会执行安全停机(safe-stop)流程而不是被直接强杀。


@runtime_checkable
class EnvironmentHost(Protocol):
    """Own a platform environment and run Supervisor work in the correct thread."""

    # 协议:持有平台环境,并确保 Supervisor 的工作在正确的线程中执行
    # (例如 Isaac Sim 要求某些调用在渲染主线程中进行)。

    environment: ExecutionEnvironment

    def run(self, operation: Callable[[], _T]) -> _T:
        """Execute work while servicing the platform's owner-thread loop."""

        # 执行 operation,同时为平台的"所有者线程"循环提供泵(心跳)服务。

    def close(self, reason: str) -> None:
        """Release platform resources after motion has been stopped."""

        # 运动停止后,释放平台资源(退出码清理阶段调用)。


class DirectEnvironmentHost:
    """Host for environments that do not require an owner-thread pump."""

    # 面向"不需要主线程泵"的环境:直接在调用线程里同步执行操作,
    # close 为空操作。是 EnvironmentHost 的最简单实现。

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
    # 读取并解析 JSON 文件,统一做三类检查:
    # 1. 文件存在性/可读性/大小上限(max_bytes);
    # 2. 编码必须是 UTF-8 且内容为合法 JSON;
    # 3. 根节点必须是 JSON 对象(Mapping)。
    # 任何一步失败都抛出带 label 的明确错误信息,方便定位是哪个文件出错。
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

    # 加载 Agent 配置(JSON 对象),体积上限 _MAX_CONFIG_BYTES。
    return _load_json_object(path, max_bytes=_MAX_CONFIG_BYTES, label="Agent config")


def load_task(path: Path) -> TaskSchema:
    """Load and validate one immutable task contract."""

    # 加载任务契约文件并转成不可变的 TaskSchema(带校验)。
    raw = _load_json_object(path, max_bytes=_MAX_TASK_BYTES, label="Task")
    return TaskSchema.from_dict(raw)


def build_supervisor(
    config: Mapping[str, Any],
    *,
    transport_factory: (Callable[[str, str], ProcessTransport] | None) = None,
) -> V2Supervisor:
    """Wire the only formal runtime: V2 Supervisor + YOLO + π0.5.

    V1 is intentionally rejected here. Historical V1 classes remain importable
    for archived regression tests but cannot be composed by the production
    entry point.
    """

    # 从同一份 V2 配置只装配 π0.5 ProcessTransport 与单臂 Supervisor。
    # transport_factory 可注入自定义传输工厂，便于契约测试替换为 mock。
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if transport_factory is not None and not callable(transport_factory):
        raise TypeError("transport_factory must be callable")

    def default_transport_factory(
        service_name: str,
        base_url: str,
    ) -> BoundedHTTPTransport:
        # 默认工厂:返回带连接数上限的 HTTP 传输(service_name 仅作校验用)。
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError("service_name must be a non-empty string")
        return BoundedHTTPTransport(base_url)

    if str(config.get("config_version", "")).split(".", 1)[0] != "2":
        raise ValueError(
            "V1 is abolished; production build requires config_version 2.x"
        )
    raw_executors = config.get("executors")
    if not isinstance(raw_executors, Mapping) or set(raw_executors) != {"pi05"}:
        raise ValueError("formal V2 config must contain only executors.pi05")
    raw_pi05 = raw_executors.get("pi05")
    if not isinstance(raw_pi05, Mapping) or set(raw_pi05) != EXECUTOR_CONFIG_FIELDS:
        raise ValueError(
            f"executors.pi05 must contain exactly {sorted(EXECUTOR_CONFIG_FIELDS)}"
        )
    if raw_pi05.get("enabled") is not True:
        raise ValueError("formal V2 requires executors.pi05.enabled=true")
    base_url = raw_pi05.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(
        ("http://", "https://")
    ):
        raise ValueError("executors.pi05.base_url must be an HTTP(S) URL")
    factory = transport_factory or default_transport_factory
    transport = cast(ProcessTransport, factory("pi05", base_url))
    executor = Pi05Adapter(
        transport,
        checkpoint_sha=str(raw_pi05.get("checkpoint_sha", "")),
        norm_stats_sha=str(raw_pi05.get("norm_stats_sha", "")),
    )
    return V2Supervisor.from_config(executor, config)


def resolve_environment_host(
    factory_reference: str,
    config: Mapping[str, Any],
) -> EnvironmentHost:
    """Resolve ``module:callable`` and validate the returned platform host."""

    # 把 "module.path:callable" 形式的工厂引用解析成可调用的工厂,
    # 调用后校验返回值的类型:
    #   - 返回 ExecutionEnvironment -> 包一层 DirectEnvironmentHost;
    #   - 返回 EnvironmentHost      -> 校验其 environment 字段后直接使用。
    # 这保证 Supervisor 进程不直接 import 平台 SDK,平台接入点只有这一个。
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
    # 把 RunResult 递归转换为纯 JSON 兼容的数据:
    # Enum -> 取值;dataclass -> 字段字典;Mapping/序列 -> 递归转换。
    # 遇到无法序列化的类型直接抛 TypeError。
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

    # 运行结果转 JSON 字典,供进程退出时打印到 stdout,由外部进程读取。
    if not isinstance(result, RunResult):
        raise TypeError("result must be RunResult")
    converted = _json_compatible(result)
    if not isinstance(converted, dict):
        raise TypeError("RunResult conversion did not produce an object")
    return converted


def _attempt_safe_stop(environment: ExecutionEnvironment, reason: str) -> bool:
    # 向环境请求安全停机(先停运动再释放资源),返回是否确认成功。
    # 失败(异常 / 回执类型不对 / 未被确认)只记录日志并返回 False,
    # 由调用方决定退出码,不会让停机失败反过来抛异常。
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
    # 信号处理器:把 SIGINT/SIGTERM 转成异常抛到主流程里,
    # 这样"收到信号"和"普通异常"走同一套 try/finally 清理路径。
    del frame
    raise SupervisorShutdownRequested(f"received process signal {signum}")


def _install_signal_handlers() -> dict[int, Any]:
    # 安装 SIGINT/SIGTERM 处理器,并返回之前安装的处理器以便恢复。
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _signal_handler)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    # 恢复安装前的信号处理器(保证被嵌入调用时不干扰宿主进程)。
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _argument_parser() -> argparse.ArgumentParser:
    # 命令行参数:
    #   --config / --task            必填:配置文件和任务文件路径;
    #   --environment-factory        平台工厂引用,可用环境变量
    #                                INDUSTRIAL_AGENT_ENVIRONMENT_FACTORY 兜底;
    #   --log-level                  日志级别。
    parser = argparse.ArgumentParser(
        description="Run the formal V2 π0.5/Arm_A Supervisor against a platform host."
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


# ============================ 进程入口 ============================
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; never substitutes a mock for a missing platform host."""

    # 主流程步骤:
    #   1. 解析参数并配置日志;
    #   2. 加载 Agent 配置与任务文件;
    #   3. 装配监督器(两个 VLA 执行器 + YOLO 感知);
    #   4. 解析环境工厂,取得平台 host;
    #   5. 安装信号处理器后,在 host 的线程上下文中运行任务;
    #   6. 任务完成后请求安全停机,把 RunResult 以 JSON 打印到 stdout;
    #   7. 异常/中断时也尽力安全停机,最终恢复信号处理器并关闭 host。
    # 退出码约定:0 成功 / 1 任务执行失败 / 2 启动或运行期错误 /
    #            3 安全停机未确认 / 130 收到中断信号。
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
        # 任务在 host.run 中执行:由 host 决定是否需要在平台主线程泵里运行
        result = host.run(lambda: supervisor.run(task, host.environment))
        if not isinstance(result, RunResult):
            raise TypeError("environment host returned a non-RunResult value")
        # 任务结束后先安全停机(撤销运动),再输出结果
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
        # 用户/系统中断:同样走安全停机,尽量让平台回到安全状态
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
    ):
        # 启动或运行期错误:记录堆栈,若 host 已建立则尽力安全停机
        LOGGER.exception("Supervisor startup or execution failed")
        if host is not None:
            _attempt_safe_stop(host.environment, "Supervisor failed before clean exit")
        return 2
    finally:
        # 无论成功失败:恢复信号处理器、关闭平台 host(关闭失败则抛出,
        # 避免掩盖真实错误时保留关键日志)
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
    # 以脚本方式运行时调用 main,并把返回码传给系统
    sys.exit(main())
