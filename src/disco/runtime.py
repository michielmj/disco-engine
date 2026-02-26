from __future__ import annotations

from data_logger import DataLogger
from numpy.random import SeedSequence, Generator, default_rng

from .exceptions import DiscoRuntimeError
from .graph import Graph
from .graph_data import GraphData
from .partitioning import NodeInstanceSpec, Partitioning
from .simproc import SimProc

"""
NodeRuntime placeholder implementation.

- Own all simprocs for a single node.
- Provide the public send API used by model code:
    - send_event(...)
    - send_promise(...)
- Decide whether a target is local or remote:
    - Local → deliver directly to local queues (not implemented yet).
    - Remote → build EventEnvelope / PromiseEnvelope and delegate to Router.
- Provide ingress hooks for transports:
    - receive_event(envelope)
    - receive_promise(envelope)
- Expose a step() method used by the Worker runner to advance the node.

This module intentionally does NOT implement:
- EventQueue internals.
- SimProc execution.
Those will be added later; for now, NodeRuntime stores incoming
envelopes in simple lists to keep the system testable.
"""

from typing import Generator, Any, Dict
from tools.mp_logging import getLogger

from .envelopes import EventEnvelope, PromiseEnvelope
from .model import Model
from .router import Router
from .node import NodeStatus, NodeRuntimeLike

logger = getLogger(__name__)

NO_NEWS_SKIP = 10


class NodeRuntime(NodeRuntimeLike):
    """
    Controller for a single logical node in the simulation.

    This is a placeholder / skeleton implementation:

    - Outgoing events:
        - route via SimProc
    - Incoming events/promises:
        - appended to internal lists
        - step() currently just logs and no-ops (no EventQueue yet)

    Once the EventQueue and simproc execution are implemented, the
    receive_* and step() methods should be adapted to use those.
    """

    def __init__(
            self,
            repid: str,
            spec: NodeInstanceSpec,
            model: Model,
            partitioning: Partitioning,
            router: Router,
            dlogger: DataLogger,
            seed_sequence: SeedSequence,
            graph: Graph,
            data: GraphData,
    ) -> None:
        """
        Parameters
        ----------
        repid:
            Replication id.
        spec:
            NodeInstanceSpec.
        model:
            Model specification.
        partitioning:
            Partitioning.
        router:
            Router used for routing non-local envelopes.
        dlogger:
            Data logger for gathering statistics.
        seed_sequence:
            SeedSequence for Monte Carlo simulations.
        graph:
            Graph object with node mask.
        data:
            GraphData object exposing the model data.
        """
        self._repid = repid
        self._name = spec.node_name
        self._router = router
        self._dlogger = dlogger
        self._graph = graph
        self._data = data

        self._rng = default_rng(seed_sequence)
        self._node = model.node_factory(spec.node_type, runtime=self)

        self._simprocs = tuple(SimProc(
            name=s,
            number=i,
            node_name=self._name,
            repid=repid,
            on_events=self._node.on_events,
            route_event=router.send_event,
            route_promise=router.send_promise,
            predecessors=partitioning.predecessors(node_name=self._name, simproc_name=s),
            successors=partitioning.successors(node_name=self._name, simproc_name=s)
        ) for i, s in enumerate(model.spec.simprocs))

        self._status = NodeStatus.INITIALIZED
        self._waiting_for: str | None = None
        self._active_simproc: SimProc | None = None
        self._simproc_by_name = {s: i for i, s in enumerate(model.spec.simprocs)}

        logger.info("Node created for node=%s repid=%s", self._name, self._repid)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._name

    @property
    def repid(self) -> str:
        return self._repid

    # ------------------------------------------------------------------ #
    # Node API (used by Node, implements NodeRuntimeLike)
    # ------------------------------------------------------------------ #

    @property
    def epoch(self) -> float | None:
        if self._active_simproc is None:
            return None
        else:
            return self._active_simproc.epoch

    @property
    def active_simproc_name(self) -> str | None:
        if self._active_simproc is None:
            return None
        else:
            return self._active_simproc.name

    @property
    def active_simproc_number(self) -> int | None:
        if self._active_simproc is None:
            return None
        else:
            return self._active_simproc.number

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def data(self) -> GraphData:
        return self._data

    @property
    def dlogger(self) -> DataLogger:
        return self._dlogger

    @property
    def rng(self) -> Generator:
        return self._rng

    def send_event(
            self,
            target_node: str,
            target_simproc: str,
            epoch: float,
            data: Any,
            headers: Dict[str, str] | None = None,
    ) -> None:
        if self._status == NodeStatus.ACTIVE:
            assert self._active_simproc is not None
            logger.debug(
                "Node[%s] sending event: simproc=%s target_node=%s target_simproc=%s epoch=%s",
                self._name,
                self._active_simproc.name,
                target_node,
                target_simproc,
                epoch,
            )

            self._active_simproc.send_event(
                target_node=target_node, target_simproc=target_simproc, epoch=epoch, data=data, headers=headers
            )

        else:
            raise DiscoRuntimeError(f'NodeRuntime[{self._name}]: `send_event` called on a non-active status: '
                                    f'status={self._status}')

    def wakeup(self, epoch: float, hard: bool) -> None:
        if self._status == NodeStatus.ACTIVE:
            assert self._active_simproc is not None
            logger.debug(
                "Node[%s] setting wakup: simproc=%s epoch=%s hard=%s",
                self._name,
                self._active_simproc.name,
                epoch,
                hard,
            )

            self._active_simproc.wakeup(epoch=epoch, hard=hard)
        else:
            raise DiscoRuntimeError(f'NodeRuntime[{self._name}]: `wakeup` called on a non-active status: '
                                    f'status={self._status}')

    def advance_promise(self, target_node: str, target_simproc: str, epoch: float) -> None:

        if self._status == NodeStatus.ACTIVE:
            assert self._active_simproc is not None
            logger.debug(
                "Node[%s] making advance promise: simproc=%s target_node=%s target_simproc=%s epoch=%s",
                self._name,
                self._active_simproc.name,
                target_node,
                target_simproc,
                epoch,
            )

            self._active_simproc.advance_promise(target_node=target_node, target_simproc=target_simproc, epoch=epoch)
        else:
            raise DiscoRuntimeError(f'NodeRuntime[{self._name}]: `advance_promise` called on a non-active status: '
                                    f'status={self._status}')

    # ------------------------------------------------------------------ #
    # Ingress API (used by transports / Worker)
    # ------------------------------------------------------------------ #

    def receive_event(self, envelope: EventEnvelope) -> None:
        """
        Ingress hook for transports: deliver a remote or IPC event.

        For now, we just store the envelope in a buffer. In the future,
        this will enqueue into an EventQueue.
        """
        logger.debug(
            "NodeRuntime[%s] received event: simproc=%s epoch=%s",
            self._name,
            envelope.target_simproc,
            envelope.epoch,
        )

        self._simprocs[self._simproc_by_name[envelope.target_simproc]].receive_event(
            sender_node=envelope.sender_node,
            sender_simproc=envelope.sender_simproc,
            epoch=envelope.epoch,
            data=envelope.data,
            headers=envelope.headers,
        )

    def receive_promise(self, envelope: PromiseEnvelope) -> None:
        """
        Ingress hook for transports: deliver a remote or IPC promise.

        For now, we just store the envelope in a buffer. In the future,
        this will enqueue into an EventQueue with strict ordering rules.
        """
        logger.debug(
            "NodeRuntime[%s] received promise: simproc=%s seqnr=%s epoch=%s num_events=%s",
            self._name,
            envelope.target_simproc,
            envelope.seqnr,
            envelope.epoch,
            envelope.num_events,
        )

        self._simprocs[self._simproc_by_name[envelope.target_simproc]].receive_promise(
            sender_node=envelope.sender_node,
            sender_simproc=envelope.sender_simproc,
            seqnr=envelope.seqnr,
            epoch=envelope.epoch,
            num_events=envelope.num_events,
        )

    # ------------------------------------------------------------------ #
    # Initialization / Runner hook
    # ------------------------------------------------------------------ #
    def initialize(self, **kwargs) -> None:
        self._node.initialize(**kwargs)
        self._status = NodeStatus.INITIALIZED

    def runner(self, duration: float) -> Generator:
        """
        Called by the Worker runner in ACTIVE state.
        """

        self._status = NodeStatus.ACTIVE

        no_news = 0

        while True:
            if no_news > 0:
                no_news -= 1
                yield

            else:
                # The next epoch is the smallest of next_epochs among the processes. If any simproc has next_epoch equal
                # to None, we de not know what level to simproc next and we must wait for further promises.
                next_epoch: float | None = float("inf")
                for simproc in self._simprocs:
                    if simproc.next_epoch is None:
                        self._active_simproc = simproc
                        self._waiting_for = f"{simproc.name} waiting for {simproc.waiting_for}"
                        next_epoch = None
                        break

                    # SimProcs are iterated in priority order (higher → lower). We select the SimProc with the smallest
                    # next_epoch; ties keep the first encountered, so higher-order SimProcs win ties.
                    elif simproc.next_epoch < next_epoch:
                        next_epoch = simproc.next_epoch
                        self._active_simproc = simproc

                # end for simproc in self._simprocs

                # idle if waiting for promises
                if next_epoch is None:
                    no_news = NO_NEWS_SKIP
                    yield

                # stop if next event is beyond duration
                elif next_epoch >= duration:
                    self._status = NodeStatus.FINISHED
                    return

                # proceed simulation or idle if the active simproc is still waiting for events
                else:
                    assert self._active_simproc is not None
                    if not self._active_simproc.try_next_epoch():
                        self._waiting_for = (
                            f"{self._active_simproc.name} waiting for "
                            f"{self._active_simproc.waiting_for}"
                        )
                        no_news = NO_NEWS_SKIP
                        yield

            # end if no_news
        # end while
