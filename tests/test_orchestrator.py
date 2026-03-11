# tests/test_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import disco.orchestrator as orchestrator_mod
from disco.cluster import WorkerState


# ---------------------------------------------------------------------------
# Small fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeSubmission:
    value: tuple[str, str]
    consumed: bool = False
    released: bool = False

    def consume(self) -> None:
        self.consumed = True

    def release(self) -> None:
        self.released = True


class FakeElection:
    def __init__(self) -> None:
        self.canceled = False

    def run(self, cb: Callable[[], None], abort: Any = None) -> None:
        # Synchronous "leadership": immediately execute callback.
        cb()

    def cancel(self) -> None:
        self.canceled = True


class FakeCluster:
    def __init__(self) -> None:
        self.meta = object()
        self._states: dict[str, WorkerState] = {}
        self._available: tuple[list[str], list[int]] = ([], [])
        self.desired_calls: list[dict[str, Any]] = []
        self.await_available_calls: int = 0
        self._election = FakeElection()

    @property
    def worker_states(self) -> dict[str, WorkerState]:
        return self._states

    def set_worker_state(self, worker: str, state: WorkerState) -> None:
        self._states[worker] = state

    def set_available(self, addresses: list[str], preferred_partitions: list[int] | None = None) -> None:
        self._available = (list(addresses), list(preferred_partitions or []))

    def get_available(self, expid: str) -> tuple[list[str], list[int]]:
        return self._available

    def await_available(self, timeout: float) -> None:
        self.await_available_calls += 1

    def set_desired_state(
        self,
        *,
        worker_address: str,
        state: WorkerState,
        expid: str | None = None,
        repid: str | None = None,
        partition: int | None = None,
    ) -> None:
        self.desired_calls.append(
            dict(
                worker_address=worker_address,
                state=state,
                expid=expid,
                repid=repid,
                partition=partition,
            )
        )
        # Simulate fast worker round-trip for RESERVED: the worker publishes
        # RESERVED immediately (no heavy I/O), so the orchestrator's poll loop
        # sees the transition on its first check.
        if state == WorkerState.RESERVED:
            self._states[worker_address] = WorkerState.RESERVED

    def make_orchestrator_election(self, *, address: str) -> FakeElection:
        return self._election


@dataclass(frozen=True)
class FakeAssignment:
    partition: int
    worker: str


@dataclass
class FakeReplication:
    assignments: dict[int, FakeAssignment]


@dataclass
class FakeExperiment:
    expid: str
    allowed_partitionings: list[str]
    selected_partitioning: str | None = None
    replications: dict[str, FakeReplication] | None = None


class FakeStore:
    """
    Minimal fake ExperimentStore to drive Orchestrator behavior.
    """
    def __init__(self, _meta: object) -> None:
        self._queue: list[FakeSubmission | None] = []
        self.load_map: dict[str, FakeExperiment] = {}
        self.selected: list[tuple[str, str]] = []
        self.assigned: list[tuple[str, str, list[str]]] = []
        self.failed_exc: list[tuple[str, str, dict[str, Any], bool]] = []
        self.failed_status: list[tuple[str, str, Any]] = []

    def push_dequeue(self, item: FakeSubmission | None) -> None:
        self._queue.append(item)

    def dequeue(self, *, timeout: float, force_mode: str) -> FakeSubmission | None:
        if self._queue:
            return self._queue.pop(0)
        return None

    def load(self, expid: str) -> FakeExperiment:
        return self.load_map[expid]

    def select_partitioning(self, expid: str, partitioning_id: str) -> None:
        self.selected.append((expid, partitioning_id))
        self.load_map[expid].selected_partitioning = partitioning_id

    def assign_partitions(self, expid: str, repid: str, assignments: list[str]) -> FakeExperiment:
        self.assigned.append((expid, repid, list(assignments)))
        # Ensure experiment has assignment objects at exp.replications[repid].assignments.values()
        exp = self.load_map[expid]
        exp.replications = exp.replications or {}
        exp.replications[repid] = FakeReplication(
            assignments={i: FakeAssignment(partition=i, worker=w) for i, w in enumerate(assignments)}
        )
        return exp

    def set_replication_exc(self, *, expid: str, repid: str, exc: dict[str, Any], fail_replication: bool) -> None:
        self.failed_exc.append((expid, repid, exc, fail_replication))

    def set_replication_status(self, *, expid: str, repid: str, status: Any) -> None:
        self.failed_status.append((expid, repid, status))


class FakeThread:
    """
    Deterministic Thread replacement (no real threading in tests).
    """
    def __init__(self, *, target: Callable[..., Any], args: tuple[Any, ...], daemon: bool) -> None:
        self._target = target
        self._args = args
        self.daemon = daemon
        self.started = False
        self._alive = False
        self.join_calls: int = 0

    def start(self) -> None:
        self.started = True
        # Do NOT run the target automatically here (we test handover only).
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1
        # Pretend it finishes on join.
        self._alive = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cluster() -> FakeCluster:
    return FakeCluster()


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    # Patch ExperimentStore used by Orchestrator to our FakeStore.
    store = FakeStore(object())
    monkeypatch.setattr(orchestrator_mod, "ExperimentStore", lambda _meta: store)
    return store


@pytest.fixture
def settings() -> Any:
    # Orchestrator only reads .launch_timeout_s.
    return SimpleNamespace(launch_timeout_s=0.5)


@pytest.fixture
def orch(fake_cluster: FakeCluster, fake_store: FakeStore, settings: Any, monkeypatch: pytest.MonkeyPatch) -> orchestrator_mod.Orchestrator:
    # Avoid real threads.
    monkeypatch.setattr(orchestrator_mod, "Thread", FakeThread)

    # Keep Partitioning.load_metadata deterministic.
    monkeypatch.setattr(
        orchestrator_mod.Partitioning,
        "load_metadata",
        staticmethod(lambda _meta, _pid: {"num_partitions": 2}),
    )

    return orchestrator_mod.Orchestrator(
        address="orch-1",
        cluster=fake_cluster,  # type: ignore[arg-type]
        settings=settings,     # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests: leader-loop handling of queue entities
# ---------------------------------------------------------------------------

def test_on_stop_requested_releases_entity_and_exits(
    orch: orchestrator_mod.Orchestrator,
    fake_store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = FakeSubmission(("exp1", "rep1"))
    fake_store.push_dequeue(entity)

    # Make _handle_submission raise _StopRequested to simulate "stop during pre-launch stage".
    monkeypatch.setattr(orch, "_handle_submission", lambda *, entity: (_ for _ in ()).throw(orchestrator_mod._StopRequested()))

    # Run lead callback once. FakeElection calls synchronously.
    orch.run_forever(Event())

    assert entity.released is True
    assert entity.consumed is False
    assert fake_store.failed_exc == []


def test_on_unexpected_exception_marks_failed_and_consumes(
    orch: orchestrator_mod.Orchestrator,
    fake_store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = FakeSubmission(("exp1", "rep1"))

    # Patch dequeue so the leader loop stops after one processed item.
    calls = {"n": 0}

    def dequeue(*, timeout: float, force_mode: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return entity
        orch._stop.set()
        return None

    monkeypatch.setattr(fake_store, "dequeue", dequeue)

    def boom(*, entity: Any) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(orch, "_handle_submission", boom)

    orch.run_forever(Event())

    assert entity.consumed is True
    assert len(fake_store.failed_exc) == 1
    expid, repid, exc, fail_replication = fake_store.failed_exc[0]
    assert (expid, repid) == ("exp1", "rep1")
    assert fail_replication is True
    assert "boom" in exc["description"]


# ---------------------------------------------------------------------------
# Tests: _handle_submission handover rules
# ---------------------------------------------------------------------------

def test_handle_submission_reserves_consumes_and_starts_thread(
    orch: orchestrator_mod.Orchestrator,
    fake_cluster: FakeCluster,
    fake_store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange experiment
    exp = FakeExperiment(expid="exp1", allowed_partitionings=["p1"], selected_partitioning="p1")
    fake_store.load_map["exp1"] = exp

    # Make workers available immediately for 2 partitions.
    fake_cluster.set_available(["w1", "w2"], preferred_partitions=[])

    # Make _await_full_assignment_plan immediate and deterministic.
    monkeypatch.setattr(orch, "_await_full_assignment_plan", lambda *, expid, num_partitions: ["w1", "w2"])

    entity = FakeSubmission(("exp1", "rep1"))

    # Act
    orch._handle_submission(entity=entity)  # type: ignore[arg-type]

    # Assert: reservation desired-state calls were made with assignment.
    reserved_calls = [c for c in fake_cluster.desired_calls if c["state"] == WorkerState.RESERVED]
    assert len(reserved_calls) == 2
    assert {c["worker_address"] for c in reserved_calls} == {"w1", "w2"}
    assert all(c["expid"] == "exp1" and c["repid"] == "rep1" for c in reserved_calls)
    assert {c["partition"] for c in reserved_calls} == {0, 1}

    # Assert: both workers transitioned to RESERVED (via FakeCluster auto-transition).
    assert fake_cluster.worker_states.get("w1") == WorkerState.RESERVED
    assert fake_cluster.worker_states.get("w2") == WorkerState.RESERVED

    # Assert: assignment happened, entity consumed, and a launch thread was started.
    assert fake_store.assigned == [("exp1", "rep1", ["w1", "w2"])]
    assert entity.consumed is True
    assert len(orch._launch_threads) == 1
    t = orch._launch_threads[0]
    assert isinstance(t, FakeThread)
    assert t.started is True


# ---------------------------------------------------------------------------
# Tests: launch thread logic (call directly, no real threads)
# ---------------------------------------------------------------------------

def test_launch_replication_timeout_marks_failed(
    orch: orchestrator_mod.Orchestrator,
    fake_store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prepare exp with assignments
    exp = FakeExperiment(
        expid="exp1",
        allowed_partitionings=["p1"],
        selected_partitioning="p1",
        replications={
            "rep1": FakeReplication(assignments={
                0: FakeAssignment(0, "w1"),
                1: FakeAssignment(1, "w2"),
            })
        },
    )

    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def mark_failed(*, expid: str, repid: str, exc: dict[str, Any] | None = None) -> None:
        calls.append((expid, repid, exc))

    monkeypatch.setattr(orch, "_mark_replication_failed", mark_failed)

    # Force time to advance quickly so deadline is exceeded while waiting for READY.
    t = {"now": 0.0}

    def fake_monotonic() -> float:
        t["now"] += 1.0
        return t["now"]

    monkeypatch.setattr(orchestrator_mod, "monotonic", fake_monotonic)
    orch._settings.launch_timeout_s = 1.0  # deadline ~ now+1, fake_monotonic will exceed rapidly

    # Worker states never become READY -> should time out during initialization.
    orch._launch_replication(exp, "rep1")  # type: ignore[arg-type]

    assert calls, "Expected replication to be marked failed on timeout"
    expid, repid, exc = calls[0]
    assert (expid, repid) == ("exp1", "rep1")
    assert exc is not None
    assert "timeout" in exc["description"].lower()


def test_launch_replication_stop_marks_failed(
    orch: orchestrator_mod.Orchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp = FakeExperiment(
        expid="exp1",
        allowed_partitionings=["p1"],
        selected_partitioning="p1",
        replications={
            "rep1": FakeReplication(assignments={
                0: FakeAssignment(0, "w1"),
                1: FakeAssignment(1, "w2"),
            })
        },
    )

    calls: list[dict[str, Any] | None] = []

    monkeypatch.setattr(
        orch,
        "_mark_replication_failed",
        lambda *, expid, repid, exc=None: calls.append(exc),
    )

    # Stop before launch => should immediately mark failed "stopped during initialization".
    orch._stop.set()
    orch._launch_replication(exp, "rep1")  # type: ignore[arg-type]

    assert calls, "Expected replication to be marked failed on stop"
    assert calls[0] is not None
    assert "stopped" in calls[0]["description"].lower()


# ---------------------------------------------------------------------------
# Tests: reservation
# ---------------------------------------------------------------------------

def test_reserve_workers_timeout_raises_and_cleans_up(
    orch: orchestrator_mod.Orchestrator,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When one worker fails to reach RESERVED within the timeout,
    _reserve_workers should send TERMINATED to workers that did reach
    RESERVED and raise RuntimeError.
    """
    # Disable auto-transition so we can control which workers reach RESERVED.
    original_set = fake_cluster.set_desired_state.__func__  # type: ignore[attr-defined]

    def set_desired_no_auto(self, *, worker_address, state, expid=None, repid=None, partition=None):
        self.desired_calls.append(dict(
            worker_address=worker_address, state=state,
            expid=expid, repid=repid, partition=partition,
        ))
        # Only w1 transitions to RESERVED; w2 stays stuck.
        if state == WorkerState.RESERVED and worker_address == "w1":
            self._states[worker_address] = WorkerState.RESERVED

    monkeypatch.setattr(type(fake_cluster), "set_desired_state", set_desired_no_auto)

    # Ensure w2 is stuck as AVAILABLE.
    fake_cluster._states["w2"] = WorkerState.AVAILABLE

    # Make timeout very short.
    monkeypatch.setattr(orchestrator_mod, "RESERVE_TIMEOUT_S", 0.05)

    with pytest.raises(RuntimeError, match="Timed out"):
        orch._reserve_workers(expid="exp1", repid="rep1", assignments=["w1", "w2"])

    # w1 reached RESERVED, so it should receive a TERMINATED cleanup call.
    terminated_calls = [
        c for c in fake_cluster.desired_calls if c["state"] == WorkerState.TERMINATED
    ]
    assert any(c["worker_address"] == "w1" for c in terminated_calls)


def test_launch_sends_ready_and_active_without_assignment(
    orch: orchestrator_mod.Orchestrator,
    fake_cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    After the RESERVED step, _launch_replication should send READY and ACTIVE
    without assignment parameters (expid/repid/partition are None).
    """
    exp = FakeExperiment(
        expid="exp1",
        allowed_partitionings=["p1"],
        selected_partitioning="p1",
        replications={
            "rep1": FakeReplication(assignments={
                0: FakeAssignment(0, "w1"),
            })
        },
    )

    # Simulate workers reaching READY then ACTIVE instantly.
    fake_cluster._states["w1"] = WorkerState.RESERVED

    def fake_set_desired(*, worker_address, state, expid=None, repid=None, partition=None):
        fake_cluster.desired_calls.append(dict(
            worker_address=worker_address, state=state,
            expid=expid, repid=repid, partition=partition,
        ))
        fake_cluster._states[worker_address] = state

    monkeypatch.setattr(fake_cluster, "set_desired_state", fake_set_desired)

    orch._launch_replication(exp, "rep1")  # type: ignore[arg-type]

    ready_calls = [c for c in fake_cluster.desired_calls if c["state"] == WorkerState.READY]
    active_calls = [c for c in fake_cluster.desired_calls if c["state"] == WorkerState.ACTIVE]

    assert len(ready_calls) == 1
    assert ready_calls[0]["expid"] is None
    assert ready_calls[0]["repid"] is None
    assert ready_calls[0]["partition"] is None

    assert len(active_calls) == 1
    assert active_calls[0]["expid"] is None
    assert active_calls[0]["repid"] is None
    assert active_calls[0]["partition"] is None
