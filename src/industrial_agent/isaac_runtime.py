"""Controlled owner-thread dispatcher for Isaac/Omniverse operations.

The Supervisor deliberately invokes untrusted adapters from deadline worker
threads.  Isaac stage, articulation and physics APIs, however, must be called
from the thread that owns the simulation loop.  This module provides the
bounded bridge between those two execution models without importing Isaac.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, get_ident
from time import monotonic
from typing import Callable, Generic, TypeVar, cast


T = TypeVar("T")


class IsaacGateError(RuntimeError):
    """Base class for controlled-runtime dispatch failures."""


class IsaacGateClosedError(IsaacGateError):
    """The Isaac owner loop closed before a request could complete."""


class IsaacGateBusyError(IsaacGateError):
    """The bounded normal request queue has no free slot."""


class IsaacGateTimeoutError(IsaacGateError, TimeoutError):
    """A request did not receive an owner-thread acknowledgement in time."""

    def __init__(self, label: str, *, started: bool) -> None:
        phase = "after execution started" if started else "before execution started"
        super().__init__(f"{label} timed out {phase}")
        self.label = label
        self.started = started


@dataclass
class _GateRequest(Generic[T]):
    operation: Callable[[], T]
    label: str
    deadline: float
    done: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    started: bool = False
    cancelled: bool = False
    result: T | None = None
    error: BaseException | None = None


class IsaacMainThreadGate:
    """Marshal calls onto the thread that constructed this gate.

    Normal work uses a bounded FIFO.  Safe-stop work uses a separate unbounded
    urgent FIFO.  The caller of :meth:`call_stop` first executes a thread-safe
    stop signal that must not touch Isaac APIs; hold/pause/readback is then
    completed by the owner thread.

    A safe-stop cancels normal work already queued at that instant.  It does
    not permanently disable the gate because post-stop observations still need
    to run.  The execution environment independently quarantines all later
    motion commands.
    """

    def __init__(self, *, max_pending_normal: int = 16) -> None:
        if max_pending_normal < 1:
            raise ValueError("max_pending_normal must be positive")
        self._owner_thread_id = get_ident()
        self._normal: Queue[_GateRequest[object]] = Queue(maxsize=max_pending_normal)
        self._urgent: Queue[_GateRequest[object]] = Queue()
        self._state_lock = Lock()
        self._closed = False
        self._close_reason = ""

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    def _require_owner(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise IsaacGateError("pump() must run on the Isaac owner thread")

    @staticmethod
    def _finish_with_error(
        request: _GateRequest[object],
        error: BaseException,
    ) -> None:
        with request.lock:
            if request.done.is_set():
                return
            request.cancelled = True
            request.error = error
            request.done.set()

    def _cancel_queued_normal_locked(self, reason: str) -> None:
        while True:
            try:
                request = self._normal.get_nowait()
            except Empty:
                return
            self._finish_with_error(request, IsaacGateError(reason))

    @staticmethod
    def _result_or_raise(request: _GateRequest[T]) -> T:
        if request.error is not None:
            raise request.error
        return cast(T, request.result)

    def _wait(
        self,
        request: _GateRequest[T],
        *,
        timeout_s: float,
        on_started_timeout: Callable[[], None] | None,
        cancel_if_not_started: bool,
    ) -> T:
        if request.done.wait(timeout_s):
            return self._result_or_raise(request)

        with request.lock:
            if request.done.is_set():
                return self._result_or_raise(request)
            started = request.started
            if not started and cancel_if_not_started:
                request.cancelled = True

        if started and on_started_timeout is not None:
            on_started_timeout()
        raise IsaacGateTimeoutError(request.label, started=started)

    def call(
        self,
        operation: Callable[[], T],
        *,
        timeout_s: float,
        label: str,
        on_started_timeout: Callable[[], None] | None = None,
    ) -> T:
        """Submit normal work and wait for a bounded owner-thread ACK."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if get_ident() == self._owner_thread_id:
            raise IsaacGateError(
                "call() must be submitted from a worker thread; "
                "owner-thread inline execution would bypass its deadline"
            )

        request: _GateRequest[T] = _GateRequest(
            operation=operation,
            label=label,
            deadline=monotonic() + timeout_s,
        )
        with self._state_lock:
            if self._closed:
                raise IsaacGateClosedError(
                    self._close_reason or "Isaac runtime gate is closed"
                )
            try:
                self._normal.put_nowait(cast(_GateRequest[object], request))
            except Full as exc:
                raise IsaacGateBusyError("normal Isaac request queue is full") from exc
        return self._wait(
            request,
            timeout_s=timeout_s,
            on_started_timeout=on_started_timeout,
            cancel_if_not_started=True,
        )

    def call_stop(
        self,
        *,
        signal_stop: Callable[[], None],
        apply_stop: Callable[[], T],
        timeout_s: float,
        label: str = "Isaac safe-stop",
    ) -> T:
        """Signal stop immediately, then request owner-thread hold/readback."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

        # This callback must only update thread-safe cancellation/lease state.
        # It deliberately runs outside all gate locks and queues.
        signal_stop()

        request: _GateRequest[T] = _GateRequest(
            operation=apply_stop,
            label=label,
            deadline=monotonic() + timeout_s,
        )
        with self._state_lock:
            if self._closed:
                raise IsaacGateClosedError(
                    self._close_reason or "Isaac runtime gate is closed"
                )
            self._cancel_queued_normal_locked(
                "normal Isaac request cancelled by safe-stop"
            )
            self._urgent.put_nowait(cast(_GateRequest[object], request))

        if get_ident() == self._owner_thread_id:
            self._drain_urgent()
        # An urgent request is intentionally not cancelled when its caller times
        # out.  If the owner loop recovers before the process watchdog acts, it
        # must still apply hold/pause.
        return self._wait(
            request,
            timeout_s=timeout_s,
            on_started_timeout=None,
            cancel_if_not_started=False,
        )

    def _execute(self, request: _GateRequest[object], *, urgent: bool) -> None:
        with request.lock:
            if request.done.is_set():
                return
            if request.cancelled or (not urgent and monotonic() >= request.deadline):
                request.cancelled = True
                request.error = IsaacGateTimeoutError(
                    request.label,
                    started=False,
                )
                request.done.set()
                return
            request.started = True

        with self._state_lock:
            closed = self._closed
            close_reason = self._close_reason
        if closed:
            self._finish_with_error(
                request,
                IsaacGateClosedError(close_reason or "Isaac runtime gate is closed"),
            )
            return

        try:
            result = request.operation()
        except BaseException as exc:
            with request.lock:
                request.error = exc
                request.done.set()
        else:
            with request.lock:
                request.result = result
                request.done.set()

    def _drain_urgent(self) -> int:
        processed = 0
        while True:
            try:
                request = self._urgent.get_nowait()
            except Empty:
                return processed
            self._execute(request, urgent=True)
            processed += 1

    def pump(self, *, max_normal: int = 1) -> int:
        """Run queued work on the Isaac owner thread, urgent work first."""

        self._require_owner()
        if max_normal < 0:
            raise ValueError("max_normal cannot be negative")

        processed = self._drain_urgent()
        for _ in range(max_normal):
            try:
                request = self._normal.get_nowait()
            except Empty:
                break
            self._execute(request, urgent=False)
            processed += 1
            # A stop may arrive while the normal operation is in a physics
            # loop. Apply its hold/pause before starting any later normal work.
            processed += self._drain_urgent()
        return processed

    def run_worker_until_complete(
        self,
        operation: Callable[[], T],
        *,
        poll_interval_s: float = 0.001,
        idle_callback: Callable[[], None] | None = None,
    ) -> T:
        """Run a worker operation while this owner thread pumps the gate.

        This helper is intended for the Isaac standalone entry point and smoke
        tests.  Long-lived adapters may instead call :meth:`pump` directly from
        their existing simulation loop.
        """

        self._require_owner()
        if poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        finished = Event()
        result: dict[str, object] = {}

        def worker() -> None:
            try:
                result["value"] = operation()
            except BaseException as exc:
                result["error"] = exc
            finally:
                finished.set()

        thread = Thread(target=worker, daemon=True, name="isaac-gated-worker")
        thread.start()
        while not finished.is_set():
            processed = self.pump(max_normal=1)
            if processed == 0 and idle_callback is not None:
                idle_callback()
            finished.wait(poll_interval_s)
        # A stop caller deliberately leaves its urgent hold queued when its ACK
        # deadline expires. The worker can therefore finish just after queuing
        # that hold. Drain it before returning control to a standalone caller;
        # otherwise this helper could strand the fail-closed operation.
        self.pump(max_normal=0)
        thread.join(timeout=0.0)
        if "error" in result:
            error = result["error"]
            assert isinstance(error, BaseException)
            raise error
        return cast(T, result.get("value"))

    def close(self, reason: str = "Isaac runtime gate closed") -> None:
        """Reject new work and wake every queued waiter."""

        error = IsaacGateClosedError(reason)
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._close_reason = reason
            while True:
                try:
                    request = self._normal.get_nowait()
                except Empty:
                    break
                self._finish_with_error(request, error)
            while True:
                try:
                    request = self._urgent.get_nowait()
                except Empty:
                    break
                self._finish_with_error(request, error)
