from __future__ import annotations

"""IPC transport for sending envelopes to peer processes."""

from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Mapping, Callable

from ..cluster import Cluster
from ..envelopes import EventEnvelope, PromiseEnvelope
from .base import Transport
from .ipc_messages import IPCEventMsg, IPCPromiseMsg


class IPCTransport(Transport):
    """Transport that delivers envelopes via queues and shared memory."""
    def __init__(
        self,
        cluster: Cluster,
        event_queues: Mapping[str, Queue[IPCEventMsg]],
        promise_queues: Mapping[str, Queue[IPCPromiseMsg]],
        large_payload_threshold: int = 64 * 1024,
        serializer: Callable[[any], tuple[bytes, str]] | None = None  # data -> packed_data, protocol
    ) -> None:
        self._cluster = cluster
        self._event_queues = event_queues
        self._promise_queues = promise_queues
        self._large_payload_threshold = large_payload_threshold
        self._serializer = _default_serializer if serializer is None else serializer

    def handles_node(self, repid: str, node: str) -> bool:
        addr = self._cluster.address_book.get((repid, node))
        if addr is None:
            return False
        return addr in self._event_queues and addr in self._promise_queues

    def send_event(self, envelope: EventEnvelope) -> None:
        try:
            addr = self._cluster.address_book[(envelope.repid, envelope.target_node)]
        except KeyError as exc:
            raise KeyError(
                f"IPCTransport: no address for (repid={envelope.repid!r}, node={envelope.target_node!r})"
            ) from exc
        queue = self._event_queues[addr]

        # serialize data for IPC transport
        headers = {} if envelope.headers is None else envelope.headers
        if isinstance(envelope.data, bytes):
            data = envelope.data
        else:
            data, serialization_protocol = self._serializer(envelope.data)
            headers = headers | {'serialization_protocol': serialization_protocol}

        if len(envelope.data) <= self._large_payload_threshold:
            msg = IPCEventMsg(
                repid=envelope.repid,
                sender_node=envelope.sender_node,
                sender_simproc=envelope.target_simproc,
                target_node=envelope.target_node,
                target_simproc=envelope.target_simproc,
                epoch=envelope.epoch,
                headers=headers,
                data=data,
                shm_name=None,
                size=len(envelope.data),
            )
            queue.put(msg)
        else:
            shm = SharedMemory(create=True, size=len(data))
            try:
                buf = shm.buf
                if buf is None:
                    raise RuntimeError("Shared memory buffer is unavailable")
                buf[: len(data)] = data

                msg = IPCEventMsg(
                    repid=envelope.repid,
                    sender_node="node",
                    sender_simproc="simproc",
                    target_node=envelope.target_node,
                    target_simproc=envelope.target_simproc,
                    epoch=envelope.epoch,
                    headers=headers,
                    data=None,
                    shm_name=shm.name,
                    size=len(data),
                )
                queue.put(msg)
            finally:
                # We *don't* unlink here; that is receiver's job. But if we failed
                # before putting the message on the queue, we should unlink.
                # So we can optionally track success and only unlink on early failure.
                pass

    def send_promise(self, envelope: PromiseEnvelope) -> None:
        addr = self._cluster.address_book[(envelope.repid, envelope.target_node)]
        queue = self._promise_queues[addr]
        msg = IPCPromiseMsg(
            repid=envelope.repid,
            sender_node="node",
            sender_simproc="simproc",
            target_node=envelope.target_node,
            target_simproc=envelope.target_simproc,
            seqnr=envelope.seqnr,
            epoch=envelope.epoch,
            num_events=envelope.num_events,
        )
        queue.put(msg)


def _default_serializer(value: any) -> tuple[bytes, str]:
    """
    Default serializer used when none is provided.
    """
    import pickle

    return pickle.dumps(value), f'PICKLE{pickle.DEFAULT_PROTOCOL}'
