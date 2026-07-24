"""Stable machine-readable failure taxonomy."""

from __future__ import annotations

from enum import Enum


class FailureCode(str, Enum):
    NONE = "NONE"

    # 1xxx: request and contract failures
    INVALID_TASK = "TASK_1001_INVALID"
    UNSUPPORTED_TASK_VERSION = "TASK_1002_UNSUPPORTED_VERSION"
    OBSERVATION_INVALID = "OBS_1101_INVALID"
    OBSERVATION_GT_FORBIDDEN = "OBS_1102_GT_FORBIDDEN"
    OBSERVATION_PAYLOAD_TOO_LARGE = "OBS_1103_PAYLOAD_TOO_LARGE"
    ACTION_CONTRACT_INVALID = "ACT_1201_CONTRACT_INVALID"
    ACTION_NON_FINITE = "ACT_1202_NON_FINITE"
    ACTION_WORKSPACE_BREACH = "ACT_1203_WORKSPACE_BREACH"

    # 2xxx: routing and executor failures
    NO_COMPATIBLE_EXECUTOR = "ROUTE_2001_NO_COMPATIBLE_EXECUTOR"
    EXECUTOR_UNAVAILABLE = "EXEC_2101_UNAVAILABLE"
    EXECUTOR_TIMEOUT = "EXEC_2102_TIMEOUT"
    EXECUTOR_BAD_RESPONSE = "EXEC_2103_BAD_RESPONSE"
    EXECUTOR_RUNTIME = "EXEC_2104_RUNTIME"
    EXECUTOR_MODEL_REVISION_MISMATCH = "EXEC_2105_MODEL_REVISION_MISMATCH"
    EXECUTOR_BACKPRESSURE = "EXEC_2106_BACKPRESSURE"
    EXECUTOR_CANCELLED = "EXEC_2107_CANCELLED"

    # 3xxx: verification and recovery failures
    POSTCONDITION_FAILED = "VERIFY_3001_POSTCONDITION_FAILED"
    VERIFICATION_UNCERTAIN = "VERIFY_3002_UNCERTAIN"
    VERIFIER_UNAVAILABLE = "VERIFY_3003_UNAVAILABLE"
    RECOVERY_EXHAUSTED = "RECOVERY_3101_EXHAUSTED"

    # 9xxx: immediate safe-stop failures
    EMERGENCY_STOP = "SAFE_9001_EMERGENCY_STOP"
    PROTECTIVE_STOP = "SAFE_9002_PROTECTIVE_STOP"
    SYSTEM_FAULT = "SAFE_9003_SYSTEM_FAULT"
    SAFETY_REJECTED = "SAFE_9004_ACTION_REJECTED"


class AgentError(Exception):
    """Base typed exception with a stable failure code."""

    def __init__(self, code: FailureCode, message: str):
        super().__init__(message)
        self.code = code


class ContractError(AgentError):
    pass


class ObservationError(AgentError):
    pass


class ExecutorError(AgentError):
    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        retryable: bool = False,
        retry_after_ms: int | None = None,
    ):
        super().__init__(code, message)
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms


class SafetyError(AgentError):
    pass
