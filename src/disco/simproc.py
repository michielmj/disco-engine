"""
A SimProc (Simulation Process) is a core building block of the distributed simulation engine. A simproc can be thought
of as an information layer across the simulation network in with information flows in one direction only. One such
simproc can be the order simproc where downstream nodes place orders at upstream nodes. Another simproc can
be a supply simproc where an upstream node supplies goods to downstream nodes.

These are the basic properties of simprocs:

- The SimProc artifact is actually the SimProc for a single node.
- SimProcs have zero or more predecessors and zero or more successors. Within a simproc, nodes and their
  predecessor-successor relationships must form a directed acyclic graph (DAG).
- The simprocs in a simulation are ordered. SimProcs can send events to lower level simprocs but not to
  higher level simprocs.
- SimProcs make *Promises* to dependent simprocs. A promise tells dependent simprocs at what times they may
  receive events from the simproc. Promises are made automatically by the SimProc and cannot be triggered directly
  in the event handler. A promise has a sequence number, an epoch, and a maximum number of events that
  will be sent by the simproc to that dependency for that epoch. The maximum number of events for an epoch may be
  adjusted downward for an epoch by repromising it. In this way a node can tell its dependencies that they need
  not expect events before that time, thereby creating an opportunity for concurrency. Repromises with a higher
  number of events will be ignored as it is assumed that these have just been overtaken by a later repromise.
- The *epoch* of a simproc is the (simulation) time for the node for which all events have been received and
  all prior events have been processed. If only part of the events have been received or no promise has been made
  for the next epoch yet, then the epoch remains at the earlier time even if all events have been processed. The
  *epoch* is updated just before the *event_handler* is invoked so there will only be events pending for the epoch
  during that. An idle simproc is at an epoch without events pending. The *epoch* is initialized with -1 and updates to
  0 just before the first invocation of the *event_handler*.
- The *next_epoch* of a simproc is the earliest time after *epoch* at which events will be processed. The next epoch
  is defined by a promise or by a wakeup. If neither are available, the next epoch is *None* (undefined).
- The *event_handler* for a simproc is a callback that is invoked when all events for an epoch have been received.
  This is where the actual simulation logic for the node is implemented, where the node's state is updated,
  and where events are sent to dependent simprocs.
- Wakeups may be set for the simproc in the event handler to set the time of a future epoch outside of the event
  promising. A *wakeup* is essentially an event to self. There are soft and hard wakeups. A *soft wakeup*
  inserts an epoch in the sequence of future epochs. There may be future epochs before it so an epoch for soft
  wakeup can only be processed once promises are fully made until at least one epoch after the wakeup (or a hard
  wakeup is scheduled after it). A *hard wakeup* means there will be no future epochs before it (unless there are
  prior hard wakeups). All events and promises prior to a hard wakeup will be postponed. Hard wakeups are particularly
  useful to inform dependent simprocs that they may continue processing their epochs until that time, thereby
  improving concurrency.
- An *advance_promise* can be made to a specific dependent simproc to inform that the node will not send Any events
  before that time. Trying to send an event to a dependency for an earlier epoch after the advance promise has been
  made will lead to an error.
"""

import dataclasses
from heapq import heappush, heappop
from typing import Any, Dict, Tuple, Callable, Iterable, Set

from tools.ctypes import MAX_UINT32

from .exceptions import DiscoRuntimeError
from .event_queue import EventQueue
from .envelopes import EventEnvelope, PromiseEnvelope
from .node import Event


class DiscoTimingError(DiscoRuntimeError):
    pass


class NoMoreEventsError(DiscoRuntimeError):
    """
    A simulation is an infinite chain of events. This error is raised when this chain is known to
    be broken.
    """
    pass


@dataclasses.dataclass(slots=True)
class Successor:
    """
    State container for communication with a SimProc successor.
    """

    epochs: dict  # future epochs -> num_events
    seqnr: int  # sequence number of last promise
    last_promise: float  # last promise sent
    advance_promise: float  # Any advance promise made

    def __str__(self):
        return (
            f"last_promise={str(self.last_promise)}, advance_promise={str(self.advance_promise)}, "
            f"epochs={self.epochs}"
        )


class SimProc:
    def __init__(
        self,
        name: str,
        number: int,
        node_name: str,
        repid: str,
        on_events: Callable[[str, Iterable[Event]], None],
        route_event: Callable[[EventEnvelope], None],
        route_promise: Callable[[PromiseEnvelope], None],
        predecessors: Set[Tuple[str, str]],  # set of source_node, source_process
        successors: Set[Tuple[str, str]],    # set of target_node, target_process
    ):
        """
        :param name: name of the simproc (used for identifying sender of events)

        :param predecessors: list of predecessors (<node>, <simproc>)
        :param successors: list of successor (<node>, <simproc>)
        """

        self._name = name
        self._number = number
        self._node_name = node_name
        self._repid = repid
        self._on_events = on_events
        self._route_event = route_event
        self._route_promise = route_promise

        # epoch 0 is a hard wakeup by default to ensure a the first invocation of the event_handler
        self._wakeup_points = [0.0]
        self._hard_wakeup = {0.0: True}
        self._event_outbox = []
        self._epoch = -1.0
        self._next_epoch = 0.0

        # Predecessors are registered in the EventGraph's EventController
        self._queue = EventQueue()
        for pred_node, pred_simproc in predecessors:
            self._queue.register_predecessor(pred_node, pred_simproc)

        # successor address -> Successor
        self._successors = {succ: Successor({}, 0, -1.0, -1.0) for succ in successors}

    def __str__(self):
        return (
            f"{self._name}(epoch={self._epoch}, "
            f"next_epoch={self._next_epoch}, "
            f"Queue=({str(self._queue)}), "
            f'successors={{{", ".join([f"{n}: ({str(s)})" for n, s in self._successors.items()])}}})'
        )

    def try_next_epoch(self):
        """
        This method does two things:
        - simproc the next epoch if all events have been received and there are no prior events left in the queue
          (this updates epoch)
        - update *next_epoch* (will be None if unknown)

        The method returns true if *epoch* has been updated meaning that the events for the preceding epoch were
        processed.

        :return: True if epoch was updated.
        """
        # If next_epoch is None, this means that the graph has processed the epoch already and the next epoch is not
        # yet enabled. In this case
        if self._next_epoch is None:
            return False

        # cache events
        event_inbox = []
        while self._queue.epoch <= self._next_epoch and not self._queue.empty:
            event_inbox += [Event(n, s, e, d, h) for n, s, e, d, h in self._queue.pop()]

        # Possibility 1: there are predecessors and the queue's epoch is greater than or equal to next_epoch
        # Possibility 2: there are predecessors and the queue's next epoch is greater than next epoch (wakeup)
        # Possibility 3: there are no predecessors so the next_epoch is a wakeup
        if (
            not self._queue.has_predecessors
            or self._queue.epoch >= self._next_epoch
            or (
                self._queue.next_epoch is not None
                and self._queue.next_epoch > self._next_epoch
            )
        ):
            # next epoch becomes now epoch
            self._epoch = self._next_epoch

            # flush wakeups
            is_wakeup = self._epoch == self.next_wakeup
            self._flush_wakeup()

            # SimProc events in cache.
            if is_wakeup or event_inbox:
                self._on_events(self._name, event_inbox)

            # Update next epoch
            self._update_next_epoch()

            # Make promises
            self._promise()

            # Send events in outbox. We are storing events in the outbox such that promises are made before events
            # are sent. This way, successors get information about future events and do not have to wait unnecessarily
            # for the (larger) messages to be received.
            self._send()
            return True

        else:
            return False

    def _update_next_epoch(self):
        """
        Update next_epoch if possible.

        Prerequisites:

        - No events left in the queue for the current epoch.
        - Either a hard wakeup or the next promise known.

        This method is called:

        - after the current epoch was processed;
        - after processing new promises.

         In neither case should there be events left for the current epoch. The only time during which there are
         events to be processed for the current epoch is during event handling (or to be more precise during
         *try_next_epoch*).

         The next epoch is promised based on 3 different inputs:

         - the next epoch promise of the predecessors;
         - the wakeup epoch if set;
         - the promises of Any predecessors.

        :return: True if next_epoch was updated.
        """

        # Possibility 1: a hard wakup or a soft wakeup and no predecessors
        if self.hard_wakeup or not self._queue.has_predecessors:
            next_epoch = self.next_wakeup

            # if the next_wakeup is not set, this means that there will be no more handler invocations which is an
            # error
            if next_epoch is None:
                raise NoMoreEventsError(
                    f"SimProc will not see Any more invocations because there is no wakeup set and "
                    f"there are no predecessors."
                )

        # Possibility 2: no wakeup and the queue has an epoch that is ready to be processed.
        elif self._epoch < self._queue.epoch:
            next_epoch = self._queue.epoch

        # Possibility 3: there are no events to be processed yet but all promises for the next epoch were made.
        # Note: we deal with soft wakeups scheduled at an earlier time in the next step
        elif self._queue.next_epoch is not None:
            next_epoch = self._queue.next_epoch

        # Possibility 4: there is no hard wakeup and the queue next epoch is not yet known. In this case next_epoch
        # is unknown.
        else:
            next_epoch = None

        # Update if soft wake-up is set before queue time. If next epoch is None, this means we do not know if
        # soft wakeup is planned before or after next promise from predecessors.
        if (
            self.next_wakeup is not None
            and next_epoch is not None
            and self.next_wakeup < next_epoch
        ):
            next_epoch = self.next_wakeup

        if self._next_epoch != next_epoch:
            self._next_epoch = next_epoch
            return True
        else:
            return False

    def _promise(self):
        """
        Make promises to successors.

        Prerequisite:

        - *next_epoch* was updated by calling *_update_next_epoch*.

        This method is called:

        - after processing promises from predecessors;
        - after invoking the epoch's *event_handler*.

        The earliest epoch that can be promised to successors is the SimProc' next_epoch. The promise may
        be set to a later epoch if an *advance_promise* was made for the specific successor.


        Make promises to successors (after collecting events, the event controller attempts to proceed to next epoch
        The next epoch is the smaller of the next epoch provided from an event graph with higher prio, and
        The next epoch from the event controller after processing all events.
        """

        # Make promises to all successors that have received events.
        for target, successor in self._successors.items():
            target_node, target_simproc = target

            # Start with next_epoch
            next_epoch = self._next_epoch

            # If an advance promise was made in a procedure this takes precedence if it is after next_epoch or
            # if next_epoch is not yet known.
            if (
                (next_epoch is None and successor.advance_promise > self._epoch)
                or next_epoch is not None
                and successor.advance_promise > next_epoch
            ):
                next_epoch = successor.advance_promise

            # We make promises if next_epoch for the successor is defined.
            if next_epoch is not None:
                # Promise epochs for events that were sent before next epoch.
                #
                # We promise in two steps:
                # 1. all epochs before next_epoch (number of events known)
                # 2. next_epoch (number of events unknown)
                #
                # Notes:
                #
                # - successor.epochs contains all epochs for which there are events (maps epoch to num_events sent
                #   so far)
                # - we now know the next_epoch for this simproc so all events before that time are known and can be
                #   promised
                # - we can also make an advance promise for the next_epoch but we do not yet know the number of events
                for epoch in sorted(successor.epochs):
                    # TODO: how often do we call this? Should we not make sure that epochs remains sorted so we do
                    # not have to sort each time again?

                    # Stop iteration if epoch in future.
                    if epoch >= next_epoch:
                        break

                    # MJa 2022-10-19 it is actually possible that epoch == last_promise if a prepromise was made
                    # assert epoch > successor.last_promise, "Oops, something went terribly wrong. Promised later " \
                    #                                        "already..."
                    assert epoch >= successor.last_promise, (
                        "Oops, something went terribly wrong. Promised later "
                        "already..."
                    )

                    num_events = successor.epochs.pop(epoch)

                    # only update sequence number if this is not a repromise
                    if epoch > successor.last_promise:
                        successor.seqnr += 1

                    self._route_promise(PromiseEnvelope(
                        repid=self._repid,
                        sender_node=self._node_name,
                        sender_simproc=self._name,
                        target_node=target_node,
                        target_simproc=target_simproc,
                        seqnr=successor.seqnr,
                        epoch=epoch,
                        num_events=num_events,
                    ))

                    successor.last_promise = epoch

                # Prepromise next epoch (infinite number of events)
                if next_epoch > successor.last_promise:
                    successor.seqnr += 1

                    self._route_promise(PromiseEnvelope(
                        repid=self._repid,
                        sender_node=self._node_name,
                        sender_simproc=self._name,
                        target_node=target_node,
                        target_simproc=target_simproc,
                        seqnr=successor.seqnr,
                        epoch=next_epoch,
                        num_events=MAX_UINT32,
                    ))

                    successor.last_promise = next_epoch
                    if next_epoch not in successor.epochs:
                        successor.epochs[next_epoch] = 0

    def _send(self):
        """
        Sends the events collected in outbox. Called by try_next_epoch.
        :return:
        """
        for envelope in self._event_outbox:
            self._route_event(envelope)
        self._event_outbox = []

    def send_event(
        self, target_node: str, target_simproc: str, epoch: float, data: any, headers: Dict[str, str] = None
    ):
        """
        Send the events that are collected in the simproc' outbox
        Called from Node (via NodeRuntime) during invocation of the node event handler (on_events).
        :param target_node: reference to receiving node
        :param target_simproc: reference to receiving simproc
        :param epoch:
        :param data:
        :param headers:
        """

        if epoch < self._epoch:
            raise DiscoTimingError("Events cannot be in the past.")

        successor = self._successors[(target_node, target_simproc)]
        if epoch < successor.last_promise:
            raise DiscoTimingError(
                "May not send events before last promise to successor."
            )

        if epoch not in successor.epochs:
            successor.epochs[epoch] = 1
        else:
            successor.epochs[epoch] += 1

        self._event_outbox += [EventEnvelope(
            repid=self._repid,
            sender_node=self._node_name,
            sender_simproc=self._name,
            target_node=target_node,
            target_simproc=target_simproc,
            epoch=epoch,
            data=data,
            headers=headers
        )]

    def wakeup(self, epoch: float, hard: bool = False):
        """
        Set a wakeup for the simproc.
        Called from Node during invocation of event_handler.
        :param epoch: time of the wakeup
        :param hard: true for hard wakeup (default: false)
        :return:
        """

        assert epoch > self._epoch, "Can wake-up in future only."

        if epoch not in self._hard_wakeup:
            heappush(self._wakeup_points, epoch)
        self._hard_wakeup[epoch] = hard

    def _flush_wakeup(self):
        """
        Clean-up past wakeups
        """
        while self._wakeup_points and self._wakeup_points[0] <= self._epoch:
            del self._hard_wakeup[heappop(self._wakeup_points)]

    def advance_promise(self, target_node: str, target_simproc: str, epoch: float):
        """
        Inform a specific successor that no events will be sent before some time in the future.
        :param target_node: receiving node
        :param target_simproc: receiving simproc
        :param epoch: earliest time for future events from this node to target (float)
        :return:
        """
        successor = self._successors[(target_node, target_simproc)]
        assert epoch >= successor.last_promise, "Later promise was already made."

        # If advance promise is the same as last promise, do nothing.
        if epoch == successor.last_promise:
            return

        successor.advance_promise = epoch

    @property
    def name(self) -> str:
        return self._name

    @property
    def number(self) -> int:
        return self._number

    @property
    def next_wakeup(self) -> float:
        """
        Time of next wake-up. None if unset.
        """
        if self._wakeup_points:
            return self._wakeup_points[0]

    @property
    def hard_wakeup(self):
        """
        True if next wake-up set is a hard wake-up.
        """
        epoch = self.next_wakeup
        if epoch is None:
            return False
        else:
            return self._hard_wakeup[epoch]

    @property
    def epoch(self):
        """
        Time of current epoch.
        :return:
        """
        return self._epoch

    @property
    def next_epoch(self):
        """
        Time of next_epoch. None if undefined.
        :return:
        """
        return self._next_epoch

    @property
    def waiting_for(self) -> str:
        """
        Inspection property returning one predecessor that the node is waiting for.
        :return:
        """

        return self._queue.waiting_for

    def receive_event(
        self, sender_node: str, sender_simproc: str, epoch: float, data: Any, headers: Dict[str, str] = None
    ) -> bool:
        """
        Receive an event. Called from the node_controller. Originates from try_next_epoch on a predecessor.
        :param sender_node:
        :param sender_simproc:
        :param headers:
        :param data:
        :param epoch:
        :return:
        """

        if epoch <= self._queue.epoch:
            raise DiscoTimingError(
                f"Epoch in the past: target: {self._node_name}/{self._name}, queue "
                f"epoch: {self._queue.epoch}, event epoch: {epoch}, "
                f"event sender: {sender_node}/{sender_simproc}"
            )

        return self._queue.push(sender_node, sender_simproc, epoch, data, headers)

    def receive_promise(
        self, sender_node: str, sender_simproc: str, seqnr: int, epoch: float, num_events: int
    ) -> bool:
        """
        Receive a promise.

        Called from the node_controller.

        The method returns True if the next_epoch is updated.

        Originates from try_next_epoch or receive_promise on a predecessor.
        :param sender_node:
        :param sender_simproc:
        :param seqnr:
        :param epoch:
        :param num_events:
        :return: true if next_epoch updated
        """

        if (
            self._queue.promise(
                sender_node, sender_simproc, seqnr, epoch, num_events
            )
            and self._next_epoch is None
        ):
            if self._update_next_epoch():
                self._promise()
                return True
            else:
                return False
        else:
            return False
