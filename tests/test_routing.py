from __future__ import annotations

from typing import List, Optional

import pytest

from disco.envelopes import EventEnvelope, PromiseEnvelope
from disco.router import Router, RouterError


class RecordingTransport:
    """
    Fake Transport for Router unit tests.

    - handles_node() returns a configured boolean (or raises).
    - send_event/send_promise record the envelope they received.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        handles: bool = False,
        raises_in_handles: bool = False,
    ) -> None:
        if name is not None:
            self.name = name  # Router will prefer this attribute for display
        self._handles = handles
        self._raises = raises_in_handles

        self.handles_calls: List[tuple[str, str]] = []
        self.sent_events: List[EventEnvelope] = []
        self.sent_promises: List[PromiseEnvelope] = []

    def handles_node(self, repid: str, node: str) -> bool:
        self.handles_calls.append((repid, node))
        if self._raises:
            raise RuntimeError("boom in handles_node")
        return self._handles

    def send_event(self, envelope: EventEnvelope) -> None:
        self.sent_events.append(envelope)

    def send_promise(self, envelope: PromiseEnvelope) -> None:
        self.sent_promises.append(envelope)


def make_event(*, repid: str = "r1", target_node: str = "beta") -> EventEnvelope:
    return EventEnvelope(
        repid=repid,
        sender_node="alpha",
        sender_simproc="main",
        target_node=target_node,
        target_simproc="main",
        epoch=1.0,
        data={"payload": 123},  # Router must treat as opaque
        headers={"k": "v"},
    )


def make_promise(*, repid: str = "r1", target_node: str = "beta") -> PromiseEnvelope:
    return PromiseEnvelope(
        repid=repid,
        sender_node="alpha",
        sender_simproc="main",
        target_node=target_node,
        target_simproc="control",
        seqnr=7,
        epoch=2.0,
        num_events=3,
    )


def test_router_uses_first_matching_transport_for_event() -> None:
    t1 = RecordingTransport(name="t1", handles=False)
    t2 = RecordingTransport(name="t2", handles=True)
    t3 = RecordingTransport(name="t3", handles=True)

    router = Router(transports=[t1, t2, t3])
    env = make_event(repid="r1", target_node="beta")

    router.send_event(env)

    assert t1.sent_events == []
    assert t2.sent_events == [env]
    assert t3.sent_events == []

    assert t1.handles_calls == [("r1", "beta")]
    assert t2.handles_calls == [("r1", "beta")]
    # must not query later transports after first match
    assert t3.handles_calls == []


def test_router_uses_first_matching_transport_for_promise() -> None:
    t1 = RecordingTransport(handles=False)
    t2 = RecordingTransport(handles=True)

    router = Router(transports=[t1, t2])
    env = make_promise(repid="r1", target_node="beta")

    router.send_promise(env)

    assert t1.sent_promises == []
    assert t2.sent_promises == [env]
    assert t1.handles_calls == [("r1", "beta")]
    assert t2.handles_calls == [("r1", "beta")]


def test_router_raises_if_no_transport_handles_node() -> None:
    t1 = RecordingTransport(name="t1", handles=False)
    t2 = RecordingTransport(name="t2", handles=False)

    router = Router(transports=[t1, t2])
    env = make_event(repid="rX", target_node="nope")

    with pytest.raises(RouterError) as ei:
        router.send_event(env)

    msg = str(ei.value)
    assert "No transport available" in msg
    assert "node='nope'" in msg or "node=" in msg
    assert "repid='rX'" in msg or "repid=" in msg
    assert "t1" in msg
    assert "t2" in msg


def test_router_continues_if_transport_raises_in_handles_node() -> None:
    bad = RecordingTransport(name="bad", handles=False, raises_in_handles=True)
    good = RecordingTransport(name="good", handles=True)

    router = Router(transports=[bad, good])
    env = make_event(repid="r1", target_node="beta")

    router.send_event(env)

    assert bad.sent_events == []
    assert good.sent_events == [env]
    assert bad.handles_calls == [("r1", "beta")]
    assert good.handles_calls == [("r1", "beta")]


def test_router_transport_names_prefers_transport_name_attr() -> None:
    t1 = RecordingTransport(name="ipc", handles=False)
    t2 = RecordingTransport(handles=False)  # no name attr => class name

    router = Router(transports=[t1, t2])
    names = router.transport_names()

    assert names[0] == "ipc"
    assert names[1] == "RecordingTransport"


def test_router_transports_returns_transport_objects_in_order() -> None:
    t1 = RecordingTransport(name="t1")
    t2 = RecordingTransport(name="t2")

    router = Router(transports=[t1, t2])
    assert list(router.transports()) == [t1, t2]
