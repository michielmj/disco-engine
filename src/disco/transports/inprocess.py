from __future__ import annotations

"""In-process transport that routes to local NodeRuntimes."""

from dataclasses import dataclass
from typing import Mapping

from tools.mp_logging import getLogger

from ..envelopes import EventEnvelope, PromiseEnvelope
from ..runtime import NodeRuntime
from .base import Transport

logger = getLogger(__name__)


@dataclass(slots=True)
class InProcessTransport(Transport):
    """Deliver envelopes to NodeControllers registered in the same process."""

    nodes: Mapping[str, NodeRuntime]

    def handles_node(self, repid: str, node: str) -> bool:
        if node not in self.nodes:
            return False
        return True

    def send_event(self, envelope: EventEnvelope) -> None:
        try:
            node = self.nodes[envelope.target_node]
        except KeyError as exc:
            # This is a serious internal misconfig → likely BROKEN worker.
            raise KeyError(f"InProcessTransport: unknown node {envelope.target_node!r}") from exc
        logger.debug(
            "InProcess send_event: node=%s simproc=%s epoch=%s",
            envelope.target_node,
            envelope.target_simproc,
            envelope.epoch,
        )
        node.receive_event(envelope)

    def send_promise(self, envelope: PromiseEnvelope) -> None:
        try:
            node = self.nodes[envelope.target_node]
        except KeyError as exc:
            # This is a serious internal misconfig → likely BROKEN worker.
            raise KeyError(f"InProcessTransport: unknown node {envelope.target_node!r}") from exc
        logger.debug(
            "InProcess send_promise: node=%s simproc=%s seqnr=%s epoch=%s num_events=%s",
            envelope.target_node,
            envelope.target_simproc,
            envelope.seqnr,
            envelope.epoch,
            envelope.num_events,
        )
        node.receive_promise(envelope)
