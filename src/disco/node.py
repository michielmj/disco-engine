from __future__ import annotations

from typing import Iterable, Any, Dict, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from numpy.random import Generator

from data_logger import DataLogger

from .graph import Graph
from .graph_data import GraphData


class NodeStatus(IntEnum):
    INITIALIZED = 0
    ACTIVE = 1
    FINISHED = 2
    FAILED = 9


class NodeRuntimeLike(Protocol):
    def send_event(
            self,
            target_node: str,
            target_simproc: str,
            epoch: float,
            data: Any,
            headers: Dict[str, str] | None = None,
    ) -> None: ...

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def epoch(self) -> float | None :
        raise NotImplementedError

    @property
    def rng(self) -> Generator:
        raise NotImplementedError

    @property
    def graph(self) -> Graph:
        raise NotImplementedError

    @property
    def data(self) -> GraphData:
        raise NotImplementedError

    @property
    def dlogger(self) -> DataLogger:
        raise NotImplementedError

    @property
    def active_simproc_name(self) -> str | None:
        raise NotImplementedError

    @property
    def active_simproc_number(self) -> int | None:
        raise NotImplementedError

    def wakeup(self, epoch: float, hard: bool) -> None: ...

    def advance_promise(self, target_node: str, target_simproc: str, epoch: float) -> None: ...


@dataclass(slots=True)
class Event:
    sender_node: str
    sender_simproc: str
    epoch: float
    data: Any
    headers: Dict[str, str]


class Node(ABC):
    """
    Implementation of a Node.
    """

    def __init__(self,
                 runtime: NodeRuntimeLike,
                 ):
        self._runtime = runtime

    @abstractmethod
    def initialize(self, **kwargs) -> None: ...

    @abstractmethod
    def on_events(self, simproc: str, events: Iterable[Event]) -> None: ...

    def send_event(
            self,
            target_node: str,
            target_simproc: str,
            epoch: float,
            data: Any,
            headers: dict[str, str] | None = None
    ):
        self._runtime.send_event(
            target_node=target_node,
            target_simproc=target_simproc,
            epoch=epoch,
            data=data,
            headers=headers
        )

    @property
    def name(self):
        return self._runtime.name

    @property
    def epoch(self):
        return self._runtime.epoch

    @property
    def active_simproc_name(self):
        return self._runtime.active_simproc_name

    @property
    def rng(self) -> Generator:
        return self._runtime.rng

    def graph(self) -> Graph:
        return self._runtime.graph

    @property
    def data(self) -> GraphData:
        return self._runtime.data

    @property
    def dlogger(self) -> DataLogger:
        return self._runtime.dlogger

    def wakeup(self, epoch, hard=False):
        self._runtime.wakeup(epoch=epoch, hard=hard)

    def advance_promise(self, target_node: str, target_simproc: str, epoch: float):
        self._runtime.advance_promise(target_node=target_node, target_simproc=target_simproc, epoch=epoch)
