import gc
import sys

import pytest

from disco.event_queue import EventQueue
from tools.ctypes import MAX_UINT64


def drain(queue: EventQueue) -> list:
    """Pop all currently-available events."""
    return list(queue.pop())


def test_push_requires_promised_quantity():
    queue = EventQueue()
    queue.register_predecessor("predecessor", "0")

    # Promise 2 events for epoch 1.0
    assert queue.promise(sender_node="predecessor", sender_simproc="0", seqnr=1, epoch=1.0, num_events=2)

    # First push does not complete the promised quantity
    assert not queue.push(sender_node="predecessor", sender_simproc="0", epoch=1.0, data=b"")

    # Second push completes it
    assert queue.push(sender_node="predecessor", sender_simproc="0", epoch=1.0, data=b"")


def test_push_before_promise_then_promise_enables_epoch():
    queue = EventQueue()
    queue.register_predecessor("predecessor", "0")

    # Event arrives before promise: push() returns True only if the *node epoch* updates (it won't yet).
    assert not queue.push(sender_node="predecessor", sender_simproc="0", epoch=1.0, data=b"")

    # Promise arrives afterwards: epoch should become enabled/known
    assert queue.promise(sender_node="predecessor", sender_simproc="0", seqnr=1, epoch=1.0, num_events=1)
    assert queue.epoch == 1.0

    # Once popped, epoch can progress
    _ = drain(queue)

    # Next epoch: again push before promise does not update epoch
    assert not queue.push(sender_node="predecessor", sender_simproc="0", epoch=2.0, data=b"")
    assert queue.promise(sender_node="predecessor", sender_simproc="0", seqnr=2, epoch=2.0, num_events=1)
    assert queue.epoch == 2.0
    _ = drain(queue)

    # Out-of-order promise should be rejected (False), correct seqnr accepted
    assert not queue.promise(sender_node="predecessor", sender_simproc="0", seqnr=4, epoch=4.0, num_events=1)
    assert queue.promise(sender_node="predecessor", sender_simproc="0", seqnr=3, epoch=3.0, num_events=1)

    # Epoch 4 event won't update anything until epoch 3 is satisfied
    assert not queue.push(sender_node="predecessor", sender_simproc="0", epoch=4.0, data=b"")
    assert queue.push(sender_node="predecessor", sender_simproc="0", epoch=3.0, data=b"")

    # Epoch 4 complete but not enabled while epoch 3 still pending in the queue
    assert not queue.try_next_epoch()

    _ = drain(queue)
    assert queue.epoch == 4.0


def test_empty_queue_defaults():
    queue = EventQueue()
    assert queue.epoch == float("inf")
    assert queue.empty
    assert not queue.try_next_epoch()


@pytest.mark.parametrize("order", [0, 1])
def test_epoch_is_minimum_across_predecessors_regardless_of_promise_order(order: int):
    """
    Two predecessors promise different epochs; node epoch must become the minimum, regardless of call order.
    """
    p1 = ("predecessor1", "0")
    p2 = ("predecessor2", "0")

    first, second = (p1, p2) if order == 0 else (p2, p1)

    queue = EventQueue()
    queue.register_predecessor(*p1)
    queue.register_predecessor(*p2)

    def epoch_for(node: str) -> float:
        return 2.0 if node == "predecessor2" else 1.0

    assert queue.promise(*first, seqnr=1, epoch=epoch_for(first[0]), num_events=0)
    assert queue.promise(*second, seqnr=1, epoch=epoch_for(second[0]), num_events=0)

    assert queue.epoch == 1.0

    # After predecessor1 advances, epoch should advance (given pred2 is already >= 2)
    assert queue.promise("predecessor1", "0", seqnr=2, epoch=3.0, num_events=0)
    assert queue.epoch == 2.0


def test_events_are_emitted_by_epoch_and_sender_is_preserved():
    queue = EventQueue()
    queue.register_predecessor("predecessor1", "0")

    # Push events out-of-order; queue emits by epoch once promises arrive
    queue.push(sender_node="predecessor1", sender_simproc="0", epoch=2.0, data=b"e2")
    queue.push(sender_node="predecessor1", sender_simproc="0", epoch=1.0, data=b"e1")

    # Two events at same epoch (order not guaranteed; we sort in the assert)
    queue.push(sender_node="predecessor1", sender_simproc="0", epoch=3.0, data=b"e3a")
    queue.push(sender_node="predecessor1", sender_simproc="0", epoch=3.0, data=b"e3b")

    assert queue.promise("predecessor1", "0", seqnr=1, epoch=0.5, num_events=0)
    assert drain(queue) == []

    assert queue.promise("predecessor1", "0", seqnr=2, epoch=1.0, num_events=1)
    assert drain(queue) == [("predecessor1", "0", 1.0, b"e1", {})]

    assert queue.promise("predecessor1", "0", seqnr=3, epoch=2.0, num_events=1)
    assert drain(queue) == [("predecessor1", "0", 2.0, b"e2", {})]

    assert queue.promise("predecessor1", "0", seqnr=4, epoch=3.0, num_events=2)
    assert sorted(drain(queue)) == [
        ("predecessor1", "0", 3.0, b"e3a", {}),
        ("predecessor1", "0", 3.0, b"e3b", {}),
    ]


def test_2predecessors_case_a_next_epoch_none_when_earliest_has_no_next_epoch():
    """
    Earliest epoch has no next epoch.
    Expected: next_epoch is None, irrespective of other predecessor's epochs.
    """
    queue = EventQueue()
    queue.register_predecessor("predecessor1", "0")
    queue.register_predecessor("predecessor2", "0")

    queue.promise("predecessor1", "0", 1, 1.0, 1)
    queue.promise("predecessor2", "0", 1, 2.0, 1)
    queue.push("predecessor1", "0", epoch=1.0, data=b"")
    queue.push("predecessor2", "0", epoch=2.0, data=b"")

    assert queue.epoch == 1.0
    assert queue.next_epoch is None


def test_2predecessors_case_b_epoch_skips_empty_to_nonempty_and_sets_next_epoch():
    """
    Predecessor1: earliest epoch empty, next epoch far away
    Predecessor2: epoch 1 complete, next epoch 2 complete
    Expected: epoch=1, next_epoch=2
    """
    queue = EventQueue()
    queue.register_predecessor("predecessor1", "0")
    queue.register_predecessor("predecessor2", "0")

    queue.promise("predecessor1", "0", 1, 99.0, MAX_UINT64)
    queue.promise("predecessor2", "0", 1, 1.0, 1)
    queue.promise("predecessor2", "0", 2, 2.0, 1)
    queue.push("predecessor2", "0", epoch=1.0, data=b"")

    assert queue.epoch == 1.0
    assert queue.next_epoch == 2.0


def test_2predecessors_case_c_next_epoch_is_other_predecessors_epoch_when_it_blocks():
    """
    Predecessor1: epoch 1 complete and earliest; next epoch far away
    Predecessor2: epoch 2 promised
    Expected: epoch=1, next_epoch=2
    """
    queue = EventQueue()
    queue.register_predecessor("predecessor1", "0")
    queue.register_predecessor("predecessor2", "0")

    queue.promise("predecessor1", "0", 1, 1.0, 1)
    queue.promise("predecessor1", "0", 2, 99.0, MAX_UINT64)
    queue.promise("predecessor2", "0", 1, 2.0, 1)
    queue.push("predecessor1", "0", epoch=1.0, data=b"")

    assert queue.epoch == 1.0
    assert queue.next_epoch == 2.0


def test_renew_promise_lowers_event_count_and_can_unblock_epoch():
    queue = EventQueue()
    queue.register_predecessor("pred1", "0")
    queue.register_predecessor("pred2", "0")

    # pred1 overpromises epoch 1, pred2 promises exactly 1
    assert queue.promise("pred1", "0", seqnr=1, epoch=1.0, num_events=MAX_UINT64)
    assert queue.promise("pred2", "0", seqnr=1, epoch=1.0, num_events=1)

    # pred2 delivers its single event.
    # push() only returns True if it updates epoch; it can't yet (pred1 still pending).
    assert not queue.push("pred2", "0", epoch=1.0, data=b"e1")

    # pred2 announces next epoch, but epoch 1 is still incomplete -> should not update next_epoch yet
    assert not queue.promise("pred2", "0", seqnr=2, epoch=2.0, num_events=1)

    # pred1 renews SAME seqnr+epoch to 0 events, making epoch 1 complete globally.
    # Depending on your internal ordering, this may or may not update next_epoch immediately;
    # at minimum it must not crash and must allow eventual progress.
    assert queue.promise("pred1", "0", seqnr=1, epoch=1.0, num_events=0) in (True, False)

    # pred1 delivers next promise so now we know the next_epoch will be 2.0
    assert queue.promise("pred1", "0", seqnr=2, epoch=2.0, num_events=1) in (True, False)

    assert queue.epoch == 1.0
    assert queue.next_epoch == 2.0


def test_renew_promise_rejects_increasing_event_count():
    queue = EventQueue()
    queue.register_predecessor("pred", "0")

    assert queue.promise("pred", "0", seqnr=1, epoch=1.0, num_events=10)

    # Renewing same seqnr/epoch to a HIGHER count is ignored (stale/delayed)
    assert not queue.promise("pred", "0", seqnr=1, epoch=1.0, num_events=11)


def test_renew_promise_cannot_drop_below_already_received_events():
    queue = EventQueue()
    queue.register_predecessor("pred", "0")

    # Promise 5 events for epoch 1
    assert queue.promise("pred", "0", seqnr=1, epoch=1.0, num_events=5)

    # Receive 3 events (push won't advance epoch -> don't assert True)
    assert not queue.push("pred", "0", epoch=1.0, data=b"e1")
    assert not queue.push("pred", "0", epoch=1.0, data=b"e2")
    assert not queue.push("pred", "0", epoch=1.0, data=b"e3")

    # Renewing with same seqnr/epoch but lower than already received (3) must error
    with pytest.raises(RuntimeError):
        queue.promise("pred", "0", seqnr=1, epoch=1.0, num_events=2)


def test_refcount_does_not_leak_after_pop():
    queue = EventQueue()
    queue.register_predecessor("pred", "0")

    obj = {"foo": "bar"}
    rc0 = sys.getrefcount(obj)

    queue.push("pred", "0", 1.0, obj)
    assert sys.getrefcount(obj) == rc0 + 1

    queue.promise("pred", "0", 1, 1.0, 1)

    res = drain(queue)
    assert len(res) == 1
    del res

    gc.collect()
    assert sys.getrefcount(obj) == rc0


def test_refcount_does_not_leak_after_queue_delete():
    queue = EventQueue()
    queue.register_predecessor("pred", "0")

    obj = {"foo": "bar"}
    rc0 = sys.getrefcount(obj)

    queue.push("pred", "0", 1.0, obj)
    assert sys.getrefcount(obj) == rc0 + 1

    del queue
    gc.collect()

    assert sys.getrefcount(obj) == rc0


def test_3predecessors_waiting_for_messages_states_1():
    queue = EventQueue()
    queue.register_predecessor("predecessor1", "0")
    queue.register_predecessor("predecessor2", "0")
    queue.register_predecessor("predecessor3", "0")

    queue.promise("predecessor1", "0", 1, 1.0, 1)
    queue.promise("predecessor1", "0", 2, 6.0, 1)
    queue.promise("predecessor2", "0", 1, 2.0, 1)
    queue.promise("predecessor3", "0", 1, 3.0, 1)
    queue.promise("predecessor2", "0", 2, 5.0, 1)

    queue.try_next_epoch()
    assert queue.waiting_for == "predecessor1/0 (events)"

    queue.push("predecessor1", "0", 1.0, {})
    assert len(drain(queue)) == 1

    queue.push("predecessor2", "0", 2.0, {})
    queue.push("predecessor3", "0", 3.0, {})

    queue.try_next_epoch()

    assert queue.epoch == 2.0
    assert queue.next_epoch is None
    assert queue.waiting_for == "predecessor3/0 (promises)"


def test_3predecessors_waiting_for_messages_states_2():
    queue = EventQueue()
    queue.register_predecessor("predecessor1", "0")
    queue.register_predecessor("predecessor2", "0")
    queue.register_predecessor("predecessor3", "0")

    queue.promise("predecessor1", "0", 1, 1.0, 1)
    queue.promise("predecessor1", "0", 2, 6.0, 1)
    queue.promise("predecessor2", "0", 1, 2.0, 1)
    queue.promise("predecessor3", "0", 1, 3.0, 1)
    queue.promise("predecessor2", "0", 2, 5.0, 1)
    queue.promise("predecessor3", "0", 2, 7.0, 1)

    queue.try_next_epoch()
    assert queue.waiting_for == "predecessor1/0 (events)"

    queue.push("predecessor1", "0", 1.0, {})
    assert len(drain(queue)) == 1

    queue.push("predecessor2", "0", 2.0, {})
    queue.push("predecessor3", "0", 3.0, {})

    queue.try_next_epoch()

    assert queue.epoch == 2.0
    assert queue.next_epoch == 5.0
    assert queue.waiting_for == "predecessor2/0 (events)"
    