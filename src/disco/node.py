from __future__ import annotations

from typing import Iterable, Any, Dict, Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from numpy.random import Generator

from data_logger import DataLogger

if TYPE_CHECKING:
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

    __runtime__: NodeRuntimeLike

    def __init__(self, **kwargs):
        pass

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
        self.__runtime__.send_event(
            target_node=target_node,
            target_simproc=target_simproc,
            epoch=epoch,
            data=data,
            headers=headers
        )

    @property
    def name(self):
        return self.__runtime__.name

    @property
    def epoch(self):
        return self.__runtime__.epoch

    @property
    def active_simproc_name(self):
        return self.__runtime__.active_simproc_name

    @property
    def rng(self) -> Generator:
        return self.__runtime__.rng

    @property
    def data(self) -> GraphData:
        return self.__runtime__.data

    @property
    def dlogger(self) -> DataLogger:
        return self.__runtime__.dlogger

    def wakeup(self, epoch, hard=False):
        self.__runtime__.wakeup(epoch=epoch, hard=hard)

    def advance_promise(self, target_node: str, target_simproc: str, epoch: float):
        self.__runtime__.advance_promise(target_node=target_node, target_simproc=target_simproc, epoch=epoch)
