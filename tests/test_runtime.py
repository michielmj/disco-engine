from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple, Mapping

import pytest

from disco.envelopes import EventEnvelope, PromiseEnvelope
from disco.exceptions import DiscoRuntimeError
from disco.node import Event, Node, NodeStatus
from disco.runtime import NodeRuntime


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class DummyDataLogger:
    pass


class DummyGraph:
    pass


@dataclass(frozen=True, slots=True)
class DummyGraphData:
    # Keep the same attribute names as the real GraphData
    session_manager: Any
    graph: Any
    orm: Any

    node_name: str
    node_type: str
    node_table: Any

    layer_id_by_simproc: Mapping[str, int]
    node_mask: Any  # Optional[Vector] in real code; Any here is fine


class RecordingRouter:
    """Captures outgoing envelopes and preserves send order."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []  # ("promise"|"event", envelope)

    def send_event(self, env: EventEnvelope) -> None:
        self.calls.append(("event", env))

    def send_promise(self, env: PromiseEnvelope) -> None:
        self.calls.append(("promise", env))


class DummyPartitioning:
    def __init__(
        self,
        preds: Dict[Tuple[str, str], Set[Tuple[str, str]]] | None = None,
        succs: Dict[Tuple[str, str], Set[Tuple[str, str]]] | None = None,
    ) -> None:
        self._preds = preds or {}
        self._succs = succs or {}

    def predecessors(self, node_name: str, simproc_name: str) -> Set[Tuple[str, str]]:
        return set(self._preds.get((node_name, simproc_name), set()))

    def successors(self, node_name: str, simproc_name: str) -> Set[Tuple[str, str]]:
        return set(self._succs.get((node_name, simproc_name), set()))


@dataclass(frozen=True, slots=True)
class DummyNodeInstanceSpec:
    partition: int = 0
    node_name: str = "node"
    node_type: str = "DummyNode"
    distinct_labels: Dict[str, str] | None = None


@dataclass(slots=True)
class DummyModelSpec:
    simprocs: List[str]


class DummyModel:
    def __init__(self, simprocs: List[str], node_cls: type[Node]) -> None:
        self.spec = DummyModelSpec(simprocs=simprocs)
        self._node_cls = node_cls

    def node_factory(self, node_type: str, runtime) -> Node:
        # node_type is ignored in this dummy; tests validate wiring through behavior.
        return self._node_cls(runtime=runtime)


class RecordingNode(Node):
    """A Node used for integration tests; records handler context and can emit outputs."""

    def __init__(self, runtime):
        super().__init__(runtime=runtime)
        self.init_kwargs: Dict[str, Any] | None = None
        self.calls: List[Tuple[str, float | None, str | None]] = []
        self.emit_event: Tuple[str, str, float, Any, Dict[str, str] | None] | None = None
        self.set_wakeup: Tuple[float, bool] | None = (1.0, False)  # default: keep simproc alive

    def initialize(self, **kwargs) -> None:
        self.init_kwargs = dict(kwargs)

    def on_events(self, simproc: str, events: Iterable[Event]) -> None:
        # Record runtime context visible to model code.
        self.calls.append((simproc, self.epoch, self.active_simproc_name))

        # Optionally emit outputs via the runtime facade.
        if self.emit_event is not None:
            tgt_node, tgt_sp, epoch, data, headers = self.emit_event
            self.send_event(tgt_node, tgt_sp, epoch, data, headers=headers)

        if self.set_wakeup is not None:
            delta, hard = self.set_wakeup
            assert self.epoch is not None
            self.wakeup(self.epoch + delta, hard=hard)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_runtime(
    *,
    simprocs: List[str],
    node_cls: type[Node] = RecordingNode,
    preds: Dict[Tuple[str, str], Set[Tuple[str, str]]] | None = None,
    succs: Dict[Tuple[str, str], Set[Tuple[str, str]]] | None = None,
) -> tuple[NodeRuntime, RecordingRouter, RecordingNode]:
    router = RecordingRouter()
    model = DummyModel(simprocs=simprocs, node_cls=node_cls)
    part = DummyPartitioning(preds=preds, succs=succs)

    spec = DummyNodeInstanceSpec(node_name="node", node_type="DummyNode", distinct_labels={})

    graph = DummyGraph()

    data = DummyGraphData(
        session_manager=object(),
        graph=graph,
        orm=object(),
        node_name=spec.node_name,
        node_type=spec.node_type,
        node_table=object(),
        layer_id_by_simproc={s: i for i, s in enumerate(simprocs)},
        node_mask=None,
    )

    rt = NodeRuntime(
        repid="abc",
        spec=spec,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        partitioning=part,  # type: ignore[arg-type]
        router=router,  # type: ignore[arg-type]
        dlogger=DummyDataLogger(),  # type: ignore[arg-type]
        seed_sequence=123,
        graph=graph,  # type: ignore[arg-type]
        data=data,    # type: ignore[arg-type]
    )

    node = rt._node  # type: ignore[attr-defined]
    assert isinstance(node, RecordingNode)
    return rt, router, node


def drain_runner(rt: NodeRuntime, duration: float, max_steps: int = 10_000) -> None:
    gen = rt.runner(duration=duration)
    for _ in range(max_steps):
        try:
            next(gen)
        except StopIteration:
            return
    raise AssertionError("runner did not terminate within max_steps")


# ---------------------------------------------------------------------------
# Tests: lifecycle and API guards
# ---------------------------------------------------------------------------

def test_initialize_forwards_to_node_initialize() -> None:
    rt, _router, node = make_runtime(simprocs=["L0"])
    rt.initialize(foo=1, bar="x")
    assert node.init_kwargs == {"foo": 1, "bar": "x"}
    assert rt._status == NodeStatus.INITIALIZED  # private, OK for tests


def test_send_event_requires_active_status() -> None:
    rt, _router, _node = make_runtime(simprocs=["L0"])
    assert rt._status == NodeStatus.INITIALIZED

    with pytest.raises(DiscoRuntimeError):
        rt.send_event("other", "L0", 1.0, b"data")


# ---------------------------------------------------------------------------
# Tests: runner semantics and active context
# ---------------------------------------------------------------------------

def test_epoch0_invoked_for_all_simprocs_and_context_is_set() -> None:
    # No predecessors; Node schedules a wakeup to avoid "no more events" errors in SimProc.
    rt, _router, node = make_runtime(simprocs=["H", "L"])

    # Only run long enough to execute epoch 0 for both simprocs.
    drain_runner(rt, duration=0.5)

    # Expect two handler invocations: one per simproc at epoch 0.
    assert len(node.calls) == 2
    for simproc_name, epoch, active in node.calls:
        assert epoch == 0.0
        assert active == simproc_name

    assert rt._status == NodeStatus.FINISHED


# ---------------------------------------------------------------------------
# Integration: NodeRuntime ↔ SimProc ↔ Router, promises-first discipline
# ---------------------------------------------------------------------------

def test_node_outputs_forwarded_to_active_simproc_and_promises_sent_before_events() -> None:
    # One simproc with a successor so that promises/events are emitted.
    succs = {("node", "L0"): {("succ", "L0")}}
    rt, router, node = make_runtime(simprocs=["L0"], succs=succs)

    # Emit an event at epoch 1 during the epoch-0 callback.
    node.emit_event = ("succ", "L0", 1.0, b"payload", None)
    node.set_wakeup = (1.0, False)

    drain_runner(rt, duration=0.5)  # run epoch 0 only

    kinds = [k for (k, _env) in router.calls]
    assert "promise" in kinds
    assert "event" in kinds

    # Promises-first: the first promise must be sent before the first event.
    assert kinds.index("promise") < kinds.index("event")

    promise_env = next(env for (k, env) in router.calls if k == "promise")
    assert promise_env.sender_node == "node"
    assert promise_env.sender_simproc == "L0"
    assert promise_env.target_node == "succ"
    assert promise_env.target_simproc == "L0"
    assert promise_env.epoch == 1.0

    event_env = next(env for (k, env) in router.calls if k == "event")
    assert event_env.sender_node == "node"
    assert event_env.sender_simproc == "L0"
    assert event_env.target_node == "succ"
    assert event_env.target_simproc == "L0"
    assert event_env.epoch == 1.0
    assert event_env.data == b"payload"


# ---------------------------------------------------------------------------
# Tests: ingress dispatch (target_simproc) and seqnr forwarding
# ---------------------------------------------------------------------------

class FakeSimProc:
    def __init__(self, name: str, number: int, next_epoch: float | None) -> None:
        self._name = name
        self._number = number
        self._next_epoch = next_epoch
        self.epoch = -1.0
        self.waiting_for = "x/y"
        self.received_events: list[EventEnvelope] = []
        self.received_promises: list[PromiseEnvelope] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def number(self) -> int:
        return self._number

    @property
    def next_epoch(self) -> float | None:
        return self._next_epoch

    def try_next_epoch(self) -> bool:
        return False

    def receive_event(self, sender_node: str, sender_simproc: str, epoch: float, data: Any, headers=None) -> bool:
        self.received_events.append(
            EventEnvelope(
                repid="abc",
                sender_node=sender_node,
                sender_simproc=sender_simproc,
                target_node="node",
                target_simproc=self._name,
                epoch=epoch,
                data=data,
                headers=headers,
            )
        )
        return True

    def receive_promise(self, sender_node: str, sender_simproc: str, seqnr: int, epoch: float, num_events: int) -> bool:
        self.received_promises.append(
            PromiseEnvelope(
                repid="abc",
                sender_node=sender_node,
                sender_simproc=sender_simproc,
                target_node="node",
                target_simproc=self._name,
                seqnr=seqnr,
                epoch=epoch,
                num_events=num_events,
            )
        )
        return True


def test_receive_event_and_promise_dispatch_by_target_simproc() -> None:
    rt, _router, _node = make_runtime(simprocs=["H", "L"])

    fake_h = FakeSimProc(name="H", number=0, next_epoch=0.0)
    fake_l = FakeSimProc(name="L", number=1, next_epoch=0.0)

    # Replace real simprocs with fakes to isolate NodeRuntime dispatch behavior.
    rt._simprocs = (fake_h, fake_l)  # type: ignore[attr-defined]
    rt._simproc_by_name = {"H": 0, "L": 1}  # type: ignore[attr-defined]

    rt.receive_event(
        EventEnvelope(
            repid="abc",
            sender_node="src",
            sender_simproc="H",
            target_node="node",
            target_simproc="L",
            epoch=0.0,
            data=b"x",
            headers={},
        )
    )
    assert len(fake_h.received_events) == 0
    assert len(fake_l.received_events) == 1

    rt.receive_promise(
        PromiseEnvelope(
            repid="abc",
            sender_node="src",
            sender_simproc="H",
            target_node="node",
            target_simproc="H",
            seqnr=7,
            epoch=1.0,
            num_events=3,
        )
    )
    assert len(fake_h.received_promises) == 1
    assert fake_h.received_promises[0].seqnr == 7
    assert len(fake_l.received_promises) == 0
