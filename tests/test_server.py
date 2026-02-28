from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, cast

import pytest

import disco.server as server_mod
from disco.config import ConfigError
from disco.worker import WorkerState


# ---------------------------------------------------------------------------
# Minimal fake settings types (explicit attributes, no getattr tricks)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FakeGrpcSettings:
    bind_host: str | None = None
    grace_s: float = 60.0


@dataclass(slots=True)
class FakeZookeeperSettings:
    default_group: str = "default"

@dataclass(slots=True)
class FakeLoggingSettings:
    level: str = "DEBUG"

@dataclass(slots=True)
class FakeAppSettings:
    grace_s: int = 10
    grpc: FakeGrpcSettings = field(default_factory=FakeGrpcSettings)
    zookeeper: FakeZookeeperSettings = field(default_factory=FakeZookeeperSettings)
    logging: FakeLoggingSettings = field(default_factory=FakeLoggingSettings)


# ---------------------------------------------------------------------------
# Fakes for Cluster/Worker and a no-OS "multiprocessing context"
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DesiredStateCall:
    worker_address: str
    state: WorkerState


class FakeCluster:
    def __init__(self) -> None:
        self.calls: List[DesiredStateCall] = []

    def set_desired_state(self, worker_address: str, state: WorkerState) -> None:
        self.calls.append(DesiredStateCall(worker_address=worker_address, state=state))


@contextmanager
def fake_make_cluster(*args: Any, **kwargs: Any) -> Iterator[FakeCluster]:
    _ = (args, kwargs)
    yield FakeCluster()


class FakeQueue:
    # Queue content isn't relevant for these tests; only identity matters.
    pass


class FakeEvent:
    def __init__(self) -> None:
        self._is_set = False

    def set(self) -> None:
        self._is_set = True

    def wait(self, timeout: Optional[float] = None) -> bool:
        _ = timeout
        return self._is_set

    @property
    def is_set(self) -> bool:
        return self._is_set


class FakeProcess:
    """A controlled stand-in for multiprocessing.Process."""

    _pid_seq = 1000

    def __init__(
        self,
        *,
        name: str,
        target: Any,
        args: Tuple[Any, ...],
        daemon: bool,
        autorun: bool,
    ) -> None:
        self.name = name
        self._target = target
        self._args = args
        self.daemon = daemon
        self._autorun = autorun

        self._alive = False
        self._started = False

        self.terminate_called = False
        self.kill_called = False

        FakeProcess._pid_seq += 1
        self.pid = FakeProcess._pid_seq

    def start(self) -> None:
        self._started = True
        self._alive = True

        # For worker processes in tests we often "autorun" to simulate immediate exit.
        if self._autorun:
            try:
                self._target(*self._args)
            finally:
                self._alive = False

    def join(self, timeout: Optional[float] = None) -> None:
        _ = timeout
        # If this is the orchestrator entrypoint, it blocks on a stop_event.
        # We simulate that: if stop_event is set, process can exit.
        if self._alive and self._target is server_mod._orchestrator_process_entry:
            stop_event = cast(FakeEvent, self._args[0])
            if stop_event.is_set:
                self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_called = True
        # Simulate terminate not always working immediately; leave alive until killed or join logic changes it.

    def kill(self) -> None:
        self.kill_called = True
        self._alive = False


class FakeCtx:
    """Simulates a spawn context with Queue/Event/Process factories."""

    def __init__(self, *, autorun_workers: bool) -> None:
        self._autorun_workers = autorun_workers

    def Queue(self) -> FakeQueue:
        return FakeQueue()

    def Event(self) -> FakeEvent:
        return FakeEvent()

    def Process(self, *, name: str, target: Any, args: Tuple[Any, ...], daemon: bool) -> FakeProcess:
        # Only autorun workers; never autorun orchestrator because it waits on stop_event.
        autorun = self._autorun_workers and (target is server_mod._worker_main)
        return FakeProcess(name=name, target=target, args=args, daemon=daemon, autorun=autorun)


class FakeWorker:
    """Captures constructor args and exits immediately in run_forever()."""

    instances: List["FakeWorker"] = []

    def __init__(
        self,
        *,
        address: str,
        cluster: Any,
        event_queues: Mapping[str, Any],
        promise_queues: Mapping[str, Any],
        settings: Any,
        name: Optional[str] = None,
    ) -> None:
        self.address = address
        self.cluster = cluster
        self.event_queues = event_queues
        self.promise_queues = promise_queues
        self.settings = settings
        self.name = name
        FakeWorker.instances.append(self)

    def run_forever(self) -> WorkerState:
        return WorkerState.TERMINATED


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resolve_bind_host_requires_non_loopback_if_not_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = FakeAppSettings(grpc=FakeGrpcSettings(bind_host=None))
    monkeypatch.setattr(server_mod, "_infer_non_loopback_ipv4", lambda: None)

    with pytest.raises(ConfigError):
        _ = server_mod._resolve_bind_host(cast(Any, settings), explicit=None)


def test_resolve_bind_host_rejects_loopback_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = FakeAppSettings(grpc=FakeGrpcSettings(bind_host="127.0.0.1"))
    monkeypatch.setattr(server_mod, "_infer_non_loopback_ipv4", lambda: "10.0.0.12")

    with pytest.raises(ConfigError):
        _ = server_mod._resolve_bind_host(cast(Any, settings), explicit=None)


def test_determine_worker_ports_uses_ports_override() -> None:
    ports = server_mod._determine_worker_ports(bind_host="10.0.0.12", workers=None, ports=[5001, 5002])
    assert ports == [5001, 5002]


def test_build_worker_specs_builds_addresses() -> None:
    specs = server_mod._build_worker_specs("10.0.0.12", [5001, 5002])
    assert [s.address for s in specs] == ["10.0.0.12:5001", "10.0.0.12:5002"]
    assert [s.name for s in specs] == ["worker-0", "worker-1"]


def test_singleton_guard_blocks_second_server(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = FakeAppSettings(grpc=FakeGrpcSettings(bind_host="10.0.0.12"))

    s1 = server_mod.Server(cast(Any, settings), workers=1, orchestrator=False)
    s2 = server_mod.Server(cast(Any, settings), workers=1, orchestrator=False)

    # Make start() fast/no-op for this test.
    monkeypatch.setattr(server_mod.Server, "_run", lambda self: None)

    # Simulate first server already running.
    s1._acquire_singleton_guard()
    try:
        with pytest.raises(RuntimeError):
            s2.start()
    finally:
        s1._release_singleton_guard()


def test_server_start_spawns_workers_and_passes_shared_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeWorker.instances.clear()

    settings = FakeAppSettings(grpc=FakeGrpcSettings(bind_host="10.0.0.12"), grace_s=1)

    # Patch Cluster.make_cluster to yield a FakeCluster.
    monkeypatch.setattr(server_mod.Cluster, "make_cluster", fake_make_cluster)

    # Patch Worker class in module under test.
    monkeypatch.setattr(server_mod, "Worker", FakeWorker)

    # Use FakeCtx where worker processes autorun (exit immediately).
    monkeypatch.setattr(server_mod.Server, "_mp_context", lambda self: FakeCtx(autorun_workers=True))

    srv = server_mod.Server(
        cast(Any, settings),
        ports=[5001, 5002],
        orchestrator=False,
        group="g",
    )
    srv.start()

    # Two workers should have been constructed (in-process autorun simulation).
    assert [w.address for w in FakeWorker.instances] == ["10.0.0.12:5001", "10.0.0.12:5002"]

    # All workers should receive the same shared queue dict objects (same identities).
    assert FakeWorker.instances[0].event_queues is FakeWorker.instances[1].event_queues
    assert FakeWorker.instances[0].promise_queues is FakeWorker.instances[1].promise_queues

    # And queues should include entries for all addresses.
    assert set(FakeWorker.instances[0].event_queues.keys()) == {"10.0.0.12:5001", "10.0.0.12:5002"}
    assert set(FakeWorker.instances[0].promise_queues.keys()) == {"10.0.0.12:5001", "10.0.0.12:5002"}


def test_shutdown_sends_desired_state_then_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = FakeAppSettings(grpc=FakeGrpcSettings(bind_host="10.0.0.12"), grace_s=0)

    srv = server_mod.Server(
        cast(Any, settings),
        ports=[5001, 5002],
        orchestrator=False,
        grace_s=0,
    )
    # Prepopulate worker processes and fake cluster.
    fake_cluster = FakeCluster()
    srv._cluster = cast(Any, fake_cluster)

    p1 = FakeProcess(name="w1", target=lambda: None, args=(), daemon=False, autorun=False)
    p2 = FakeProcess(name="w2", target=lambda: None, args=(), daemon=False, autorun=False)
    p1._alive = True
    p2._alive = True
    srv._worker_procs = {"10.0.0.12:5001": p1, "10.0.0.12:5002": p2}

    # Make shutdown run fast: eliminate sleeps and force monotonic to jump beyond deadlines.
    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def monotonic(self) -> float:
            self.t += 10.0
            return self.t

    clock = _Clock()
    monkeypatch.setattr(server_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(server_mod.time, "monotonic", clock.monotonic)

    srv._shutdown()

    # Desired state must be sent for each worker.
    assert [(c.worker_address, c.state) for c in fake_cluster.calls] == [
        ("10.0.0.12:5001", WorkerState.TERMINATED),
        ("10.0.0.12:5002", WorkerState.TERMINATED),
    ]

    # Terminate attempted, then kill enforced.
    assert p1.terminate_called is True
    assert p2.terminate_called is True
    assert p1.kill_called is True
    assert p2.kill_called is True


def test_orchestrator_started_and_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeWorker.instances.clear()

    settings = FakeAppSettings(grpc=FakeGrpcSettings(bind_host="10.0.0.12"), grace_s=1)

    monkeypatch.setattr(server_mod.Cluster, "make_cluster", fake_make_cluster)
    monkeypatch.setattr(server_mod, "Worker", FakeWorker)

    # Workers autorun, orchestrator does not autorun and waits on stop_event.
    monkeypatch.setattr(server_mod.Server, "_mp_context", lambda self: FakeCtx(autorun_workers=True))

    srv = server_mod.Server(
        cast(Any, settings),
        ports=[5001],
        orchestrator=True,
    )
    srv.start()

    # Orchestrator should have been created and joined (i.e., stopped) as workers exited.
    assert srv._orchestrator_proc is not None
    assert srv._orchestrator_stop is not None
    assert cast(FakeEvent, srv._orchestrator_stop).is_set is True
    assert cast(FakeProcess, srv._orchestrator_proc).is_alive() is False
