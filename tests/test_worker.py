# tests/test_worker.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Any, Callable, Optional

import pytest

import disco.worker as worker_mod
from disco.cluster import WorkerState
from disco.experiments import ExperimentStatus


# -----------------------------
# Fakes / test helpers
# -----------------------------

class FakeCluster:
    def __init__(self) -> None:
        self.meta = object()
        self.register_calls: list[tuple[str, WorkerState]] = []
        self.unregister_calls: list[str] = []
        self.worker_state_calls: list[tuple[str, WorkerState]] = []
        self.worker_info_calls: list[tuple[str, Optional[int], Optional[str], Optional[str]]] = []
        self._desired_cb: dict[str, Callable[[Any], Optional[str]]] = {}

    def register_worker(self, worker: str, state: WorkerState) -> None:
        self.register_calls.append((worker, state))

    def unregister_worker(self, worker: str) -> None:
        self.unregister_calls.append(worker)

    def set_worker_state(self, worker: str, state: WorkerState) -> None:
        self.worker_state_calls.append((worker, state))

    def update_worker_info(
        self,
        *,
        worker: str,
        partition: Optional[int],
        expid: Optional[str],
        repid: Optional[str],
    ) -> None:
        self.worker_info_calls.append((worker, partition, expid, repid))

    def on_desired_state_change(self, worker: str, cb: Callable[[Any], Optional[str]]) -> None:
        self._desired_cb[worker] = cb

    # helper for tests
    def push_desired(self, worker: str, desired: Any) -> Optional[str]:
        return self._desired_cb[worker](desired)


class FakeExperiment:
    def __init__(self, duration: float = 1.0) -> None:
        self.duration = duration


class FakeExperimentStore:
    """
    Minimal stub of ExperimentStore used by Worker:
    - load()
    - set_partition_status()
    - set_partition_exc()
    """
    def __init__(self, meta: Any) -> None:
        self.meta = meta
        self.load_calls: list[str] = []
        self.status_calls: list[tuple[str, str, int, ExperimentStatus]] = []
        self.exc_calls: list[tuple[str, str, int, dict[str, Any], bool]] = []
        self._exp = FakeExperiment(duration=1.0)

    def load(self, expid: str) -> FakeExperiment:
        self.load_calls.append(expid)
        return self._exp

    def set_partition_status(
        self, *, expid: str, repid: str, partition: int, status: ExperimentStatus
    ) -> FakeExperiment:
        self.status_calls.append((expid, repid, partition, status))
        return self._exp

    def set_partition_exc(
        self,
        *,
        expid: str,
        repid: str,
        partition: int,
        exc: dict[str, Any],
        fail_partition: bool,
    ) -> FakeExperiment:
        self.exc_calls.append((expid, repid, partition, exc, fail_partition))
        return self._exp


@dataclass
class FakeDesired:
    request_id: str
    state: WorkerState
    expid: Optional[str] = None
    repid: Optional[str] = None
    partition: Optional[int] = None

    def validate_assignment(self) -> bool:
        # Your current Worker invariant: assignment is only set/required on READY.
        if self.state == WorkerState.READY:
            return self.expid is not None and self.repid is not None and self.partition is not None
        return True

    def validate_state(self) -> bool:
        return True


class DummyTransport:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class DummyRouter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class FakeNodeRuntime:
    """
    Minimal NodeRuntime stub:
    - runner(duration=...) returns a generator (provided at construction time)
    - receive_event / receive_promise record calls
    """
    def __init__(self, gen_factory: Callable[[], Any]) -> None:
        self._gen_factory = gen_factory
        self.events: list[Any] = []
        self.promises: list[Any] = []

    def runner(self, *, duration: float) -> Any:
        return self._gen_factory()

    def receive_event(self, envelope: Any) -> None:
        self.events.append(envelope)

    def receive_promise(self, envelope: Any) -> None:
        self.promises.append(envelope)


def gen_yield_n_then_stop(n: int):
    def _g():
        for _ in range(n):
            yield object()
        return
        yield  # pragma: no cover
    return _g


def gen_raise(exc: BaseException):
    def _g():
        raise exc
        yield  # pragma: no cover
    return _g


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture()
def worker_env(monkeypatch: pytest.MonkeyPatch):
    """
    Provides a Worker instance with all external dependencies patched out.
    """
    # Patch transports/router/grpc server to avoid side effects.
    monkeypatch.setattr(worker_mod, "InProcessTransport", DummyTransport)
    monkeypatch.setattr(worker_mod, "IPCTransport", DummyTransport)
    monkeypatch.setattr(worker_mod, "GrpcTransport", DummyTransport)
    monkeypatch.setattr(worker_mod, "Router", DummyRouter)
    monkeypatch.setattr(worker_mod, "start_grpc_server", lambda *args, **kwargs: object())

    # Patch SessionManager.from_settings (not used in these tests because we patch _setup_run_locked).
    class DummySessionManager:
        @classmethod
        def from_settings(cls, settings: Any) -> "DummySessionManager":
            return cls()

    monkeypatch.setattr(worker_mod, "SessionManager", DummySessionManager)

    # Provide a per-test ExperimentStore instance via class patch.
    store = FakeExperimentStore(meta=object())

    def _store_factory(meta: Any) -> FakeExperimentStore:
        # We ignore meta here but keep the signature compatible.
        return store

    monkeypatch.setattr(worker_mod, "ExperimentStore", _store_factory)

    cluster = FakeCluster()

    addr = "w1"
    ev_q = Queue()
    pr_q = Queue()
    event_queues = {addr: ev_q}
    promise_queues = {addr: pr_q}

    # Minimal AppSettings shape; worker doesn't dereference settings deeply in these patched tests.
    class DummySettings:
        database = object()
        grpc = object()
        model = object()
        data_logger = object()

    w = worker_mod.Worker(
        address=addr,
        cluster=cluster,
        event_queues=event_queues,
        promise_queues=promise_queues,
        settings=DummySettings(),
        name="worker-1",
    )

    # The latest worker code checks `if self._dlogger:` in teardown;
    # if _setup_run_locked is bypassed, ensure it exists.
    w._dlogger = None  # type: ignore[attr-defined]

    return w, cluster, store, ev_q, pr_q


# -----------------------------
# Tests
# -----------------------------

def test_on_desired_state_change_sets_pending_and_kicks(worker_env):
    w, cluster, store, ev_q, pr_q = worker_env

    desired = FakeDesired(
        request_id="r1",
        state=WorkerState.READY,
        expid="e1",
        repid="rep1",
        partition=0,
    )

    assert not w._kick.is_set()
    err = w._on_desired_state_change(desired)
    assert err is None

    assert w._pending_desired is desired
    assert w._kick.is_set()


def test_apply_desired_ready_transitions_and_calls_setup(monkeypatch: pytest.MonkeyPatch, worker_env):
    w, cluster, store, ev_q, pr_q = worker_env

    called = {"setup": 0}

    def fake_setup_locked() -> bool:
        called["setup"] += 1
        # mimic successful setup: status LOADED then INITIALIZED is typically done in real setup
        w._set_partition_status_locked(ExperimentStatus.LOADED)
        w._set_partition_status_locked(ExperimentStatus.INITIALIZED)
        # create two nodes for later ACTIVE tests
        w._nodes["n1"] = FakeNodeRuntime(gen_yield_n_then_stop(1))
        w._nodes["n2"] = FakeNodeRuntime(gen_yield_n_then_stop(2))
        # experiment needed for _start_runners_locked
        w._experiment = FakeExperiment(duration=1.0)
        return True

    monkeypatch.setattr(w, "_setup_run_locked", fake_setup_locked)

    desired_ready = FakeDesired(
        request_id="r-ready",
        state=WorkerState.READY,
        expid="e1",
        repid="rep1",
        partition=0,
    )

    with w._lock:
        w._apply_desired_locked(desired_ready)

    assert called["setup"] == 1
    assert w.state == WorkerState.READY
    assert w._assignment is not None
    assert w._assignment.expid == "e1"
    assert w._assignment.repid == "rep1"
    assert w._assignment.partition == 0

    # Should have published worker state at least INITIALIZING then READY
    assert (w.address, WorkerState.INITIALIZING) in cluster.worker_state_calls
    assert (w.address, WorkerState.READY) in cluster.worker_state_calls

    # Should have reported partition statuses
    assert ("e1", "rep1", 0, ExperimentStatus.LOADED) in store.status_calls
    assert ("e1", "rep1", 0, ExperimentStatus.INITIALIZED) in store.status_calls


def test_active_creates_runners_only_on_ready_to_active(monkeypatch: pytest.MonkeyPatch, worker_env):
    w, cluster, store, ev_q, pr_q = worker_env

    # Prepare READY state with nodes (patch setup).
    def fake_setup_locked() -> bool:
        w._experiment = FakeExperiment(duration=1.0)
        w._nodes["n1"] = FakeNodeRuntime(gen_yield_n_then_stop(1))
        w._nodes["n2"] = FakeNodeRuntime(gen_yield_n_then_stop(1))
        w._set_partition_status_locked(ExperimentStatus.INITIALIZED)
        return True

    monkeypatch.setattr(w, "_setup_run_locked", fake_setup_locked)

    desired_ready = FakeDesired("r-ready", WorkerState.READY, "e1", "rep1", 0)
    with w._lock:
        w._apply_desired_locked(desired_ready)

    assert w.state == WorkerState.READY
    assert w._runners == ()
    assert w._active_runners == []

    # ACTIVE carries NO assignment in your current Worker design.
    desired_active = FakeDesired("r-active", WorkerState.ACTIVE)
    with w._lock:
        w._apply_desired_locked(desired_active)

    assert w.state == WorkerState.ACTIVE
    assert len(w._runners) == 2
    assert w._active_runners == [0, 1]

    # Transitioning ACTIVE again from PAUSED should not recreate runners
    with w._lock:
        w._set_state_locked(WorkerState.PAUSED)
    old_runners = w._runners
    old_active = list(w._active_runners)

    desired_active2 = FakeDesired("r-active2", WorkerState.ACTIVE)
    with w._lock:
        w._apply_desired_locked(desired_active2)

    assert w._runners is old_runners
    assert w._active_runners == old_active


def test_step_runners_once_removes_finished(worker_env):
    w, cluster, store, ev_q, pr_q = worker_env

    # Two runners: first finishes immediately, second yields once then finishes.
    w._runners = (gen_yield_n_then_stop(0)(), gen_yield_n_then_stop(1)())
    w._active_runners = [0, 1]

    # First step: runner 0 finishes, runner 1 yields => only index 1 remains
    w._step_runners_once()
    assert w._active_runners == [1]

    # Second step: runner 1 finishes
    w._step_runners_once()
    assert w._active_runners == []


def test_step_runners_once_exception_fails_partition(monkeypatch: pytest.MonkeyPatch, worker_env):
    w, cluster, store, ev_q, pr_q = worker_env

    # Need an assignment for _fail_partition_locked to work.
    w._assignment = worker_mod.Assignment(expid="e1", repid="rep1", partition=0)
    w._experiment = FakeExperiment(duration=1.0)

    # One runner raises.
    w._runners = (gen_raise(ValueError("boom"))(),)
    w._active_runners = [0]

    # Patch fail_partition to observe call without depending on teardown details.
    called: dict[str, Any] = {}

    def fake_fail_partition_locked(exc: BaseException, *, where: str) -> None:
        called["exc"] = exc
        called["where"] = where
        # simulate that worker becomes AVAILABLE after partition failure
        w._active_runners = []
        w._runners = ()

    monkeypatch.setattr(w, "_fail_partition_locked", fake_fail_partition_locked)

    w._step_runners_once()

    assert isinstance(called["exc"], ValueError)
    assert "runner_index=0" in called["where"]


def test_runner_loop_finishes_partition_and_returns_available(monkeypatch: pytest.MonkeyPatch, worker_env):
    """
    Integration-style test of the runner loop:
    - ACTIVE state
    - runners finish
    - Worker calls _end_run_locked(FINISHED) -> AVAILABLE
    Then we stop the loop with request_stop.
    """
    w, cluster, store, ev_q, pr_q = worker_env

    # Set up assignment and minimal experiment
    w._assignment = worker_mod.Assignment(expid="e1", repid="rep1", partition=0)
    w._experiment = FakeExperiment(duration=1.0)

    # Prepare a fast-finishing runner set
    w._runners = (gen_yield_n_then_stop(1)(), gen_yield_n_then_stop(1)())
    w._active_runners = [0, 1]
    with w._lock:
        w._state = WorkerState.ACTIVE
        w._running = True

    # Run loop in background so we can stop it after it finishes.
    t = threading.Thread(target=w._runner_loop, daemon=True)
    t.start()

    # Wait until it transitions to AVAILABLE (should be quick).
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if w.state == WorkerState.AVAILABLE:
            break
        time.sleep(0.01)

    assert w.state == WorkerState.AVAILABLE
    assert ("e1", "rep1", 0, ExperimentStatus.FINISHED) in store.status_calls

    # Stop the loop cleanly.
    w.request_stop()
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_no_kick_needed_for_ingress_in_ready_or_paused(worker_env):
    """
    Per your latest design note: we do NOT set _kick for every ingress message.
    This test asserts that delivering a message does not implicitly set _kick.
    (Only desired-state changes and request_stop should set it.)
    """
    w, cluster, store, ev_q, pr_q = worker_env

    # Put a dummy message in queue; _kick should remain unchanged until we process control plane.
    w._kick.clear()
    ev_q.put_nowait(worker_mod.IPCEventMsg(  # type: ignore[attr-defined]
        repid="rep1",
        sender_node="a",
        sender_simproc="s",
        target_node="missing",   # will break if delivered, but we won't drain here
        target_simproc="t",
        epoch=0.0,
        data={"x": 1},
        headers={},
        shm_name=None,
        size=0,
    ))
    assert not w._kick.is_set()
