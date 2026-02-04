from __future__ import annotations

from typing import Iterable

import pytest

from disco.envelopes import PromiseEnvelope, EventEnvelope
from disco.node import Event
from disco.simproc import SimProc, DiscoTimingError


class HandleOnEvents:
    def __init__(self):
        self.count = 0
        self.simproc = None
        self.wakeup = None
        self.hard = False
        self.events: list | None = None
        self.advance_promise = None

    def __call__(self, simproc: str, events: Iterable[Event]) -> None:
        self.count += 1

        if self.simproc is not None:

            while self.events is not None and len(self.events) > 0:
                n, s, e, d, h = self.events.pop()
                self.simproc.send_event(target_node=n, target_simproc=s, epoch=e, data=d, headers=h)

            if self.wakeup is not None:
                self.simproc.wakeup(self.simproc.epoch + self.wakeup, self.hard)

            if self.advance_promise is not None:
                self.simproc.advance_promise(*self.advance_promise)


class SendEvent:
    def __init__(self):
        self.count = 0

    def __call__(self, event: EventEnvelope) -> None:
        self.count += 1


class SendPromise:
    def __init__(self):
        self.count = 0
        self.last_promise: PromiseEnvelope | None = None

    def __call__(self, promise: PromiseEnvelope) -> None:
        self.count += 1
        self.last_promise = promise


def test_wakeup():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[],
        successors=[],
    )

    handle_on_events.simproc = sp
    handle_on_events.wakeup = 1.0

    assert sp.try_next_epoch()
    assert sp.next_epoch == 1.
    assert handle_on_events.count == 1


def test_predecessor_promising():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[("predecessor", "1")],
        successors=[],
    )

    sp.receive_promise(sender_node="predecessor", sender_simproc="1", epoch=1.0, seqnr=1, num_events=1)

    assert sp.try_next_epoch()  # epoch 0
    assert sp.next_epoch == 1.0
    assert handle_on_events.count == 1


def test_hard_wakeup():
    """
    The event handler that is run at t=0 sets a hard wakeup at time=2. Even though there is a promise for t=1,
    the simproc will update next_epoch to 2 therefore.
    """

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[("predecessor", "1")],
        successors=[],
    )

    handle_on_events.simproc = sp
    handle_on_events.wakeup = 2.0
    handle_on_events.hard = True

    sp.receive_promise(sender_node="predecessor", sender_simproc="1", epoch=1.0, seqnr=1, num_events=1)
    sp.receive_promise(sender_node="predecessor", sender_simproc="1", epoch=2.0, seqnr=2, num_events=1)

    assert sp.try_next_epoch()   # epoch 0.
    assert sp.epoch == 0         # waiting for event at t=1 (and t=2)
    assert sp.next_epoch == 2.0  # due to hard wakeup

    sp.receive_event(sender_node="predecessor", sender_simproc="1", epoch=1.0, data=b"")
    assert sp._queue.epoch == 1.0
    assert not sp.try_next_epoch()  # waiting for epoch 2.
    assert sp.epoch == 0.0
    assert handle_on_events.count == 1

    sp.receive_event(sender_node="predecessor", sender_simproc="1", epoch=2.0, data=b"")
    assert sp.try_next_epoch()
    assert sp.epoch == 2.0
    assert sp.next_epoch == 4.0
    assert handle_on_events.count == 2


def test_send_event():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[],
        successors=[("successor", "1")],
    )

    handle_on_events.simproc = sp
    handle_on_events.wakeup = 1.0
    handle_on_events.events = [("successor", "1", 1.0, b'', None)]

    assert sp.try_next_epoch()
    assert sp.next_epoch == 1.0
    assert handle_on_events.count == 1

    assert send_event.count == 1
    assert sp.try_next_epoch()
    assert sp.next_epoch == 2.0


def test_advance_promise():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[],
        successors=[("successor", "1")],
    )

    handle_on_events.simproc = sp
    handle_on_events.advance_promise = ("successor", "1", 4.0)
    handle_on_events.wakeup = 1.0

    assert sp.try_next_epoch()
    assert sp.next_epoch == 1.0

    with pytest.raises(DiscoTimingError):
        sp.send_event(target_node="successor", target_simproc="1", epoch=2.0, data=b"")

    assert send_promise is not None
    assert send_promise.last_promise.epoch == 4.0


def test_epoch_zero_2_nodes():

    handle_on_events_a = HandleOnEvents()
    send_event_a = SendEvent()
    send_promise_a = SendPromise()

    eg_a = SimProc(
        name="a-simproc",
        number=0,
        node_name="a",
        repid="abc",
        on_events=handle_on_events_a,
        route_event=send_event_a,
        route_promise=send_promise_a,
        predecessors=[],
        successors=[("b", "1")],
    )

    handle_on_events_a.simproc = eg_a
    handle_on_events_a.wakeup = 1.0
    handle_on_events_a.events = [("b", "1", 1.0, b"", None)]

    handle_on_events_b = HandleOnEvents()
    send_event_b = SendEvent()
    send_promise_b = SendPromise()

    eg_b = SimProc(
        name="b-simproc",
        number=1,
        node_name="b",
        repid="abc",
        on_events=handle_on_events_b,
        route_event=send_event_b,
        route_promise=send_promise_b,
        predecessors=[("a", "1")],
        successors=[],
    )

    assert eg_a.next_epoch == 0.0
    assert eg_a.try_next_epoch()
    assert eg_a.next_epoch == 1.0
    assert send_event_a.count == 1
    assert send_promise_a.last_promise.epoch == 1.0

    handle_on_events_a.events = [("b", "1", 2.0, b"", None)]
    assert eg_a.try_next_epoch()
    assert eg_a.next_epoch == 2.0
    assert send_event_a.count == 2
    assert send_promise_a.last_promise.epoch == 2.0

    assert eg_b.next_epoch == 0.0
    assert not eg_b.try_next_epoch()  # must receive events from predecessor


def test_first_promise_in_future_with_predecessor():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[("predecessor", "1")],
        successors=[],
    )

    assert not sp.try_next_epoch()
    assert sp.epoch == -1.
    assert sp.next_epoch == 0.

    # even thought the queue updates with the first promise, the simproc does not as
    # the first epoch has not been processed yet
    assert not sp.receive_promise(sender_node="predecessor", sender_simproc="1", seqnr=1, epoch=1.0, num_events=1)
    assert sp._queue.epoch == 0.0
    assert sp._queue.next_epoch == 1.0
    assert sp.epoch == -1.0
    assert sp.next_epoch == 0.0

    # simproc first epoch
    assert sp.try_next_epoch()
    assert sp.epoch == 0.0
    assert sp.next_epoch == 1
    assert handle_on_events.count == 1


def test_first_invoke_without_predecessor():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[],
        successors=[],
    )

    handle_on_events.simproc = sp
    handle_on_events.wakeup = 1.0

    # simproc first epoch
    assert sp.try_next_epoch()
    assert sp.epoch == 0.0
    assert sp.next_epoch == 1.0
    assert handle_on_events.count == 1


def test_first_promise_at_time_zero():

    handle_on_events = HandleOnEvents()
    send_event = SendEvent()
    send_promise = SendPromise()

    sp = SimProc(
        name="node-simproc",
        number=0,
        node_name="node",
        repid="abc",
        on_events=handle_on_events,
        route_event=send_event,
        route_promise=send_promise,
        predecessors=[("predecessor", "1")],
        successors=[],
    )

    assert not sp.receive_promise(sender_node="predecessor", sender_simproc="1", seqnr=1, epoch=0.0, num_events=1)
    assert sp._queue.epoch == -1.0
    assert sp._queue.next_epoch == 0.0
    assert sp.epoch == -1.0
    assert sp.next_epoch == 0.0

    # event through the queue is updated to reflect that epoch 0 is complete after the first
    # event is received, the simproc still needs to simproc epoch 0 before it is updated
    assert sp.receive_event(sender_node="predecessor", sender_simproc="1", epoch=0.0, data=b"")

    assert sp._queue.epoch == 0.0
    assert sp._queue.next_epoch is None
    assert sp.epoch == -1.0
    assert sp.next_epoch == 0.0

    # simproc epoch 0
    assert sp.try_next_epoch()
    assert sp.epoch == 0.0
    assert sp.next_epoch is None
    assert handle_on_events.count == 1
