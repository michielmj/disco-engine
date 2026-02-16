from typing import Iterable

from tools.mp_logging import getLogger
import numpy as np
import pandas as pd
import graphblas as gb

from disco import Node
from disco.node import Event

logger = getLogger(__name__)


class StockingPoint(Node):
    """
    This implementation of StockingPoint handles both demand and stock. Events that need to be handled:
    -> order from another node
    -> supply from another node
    -> own demand generation

    Predecessors:
    -> order process from nodes that
    -> supply process on nodes that send goods

    Successors:

    """

    def __init__(self, runtime):
        super().__init__(runtime)
        self.target_ip: gb.Vector = None
        self.mean_demand: gb.Vector = None
        self.std_demand: gb.Vector = None
        self.inv_oh: gb.Vector = None

    def initialize(self, **init_args):

        data = self.data.vertex_data(columns=['ip_target', 'mean_demand', 'std_demand', 'leadtime'])
        self.target_ip = gb.Vector.from_coo(data['index'], data['ip_target'], float, size=data.shape[0])
        demand_data = data.dropna(subset=['mean_demand', 'std_demand'])
        self.mean_demand = gb.Vector.from_coo(
            demand_data['index'],
            demand_data['mean_demand'],
            float,
            size=data.shape[0]
        )
        self.std_demand = gb.Vector.from_coo(
            demand_data['index'],
            demand_data['std_demand'],
            float,
            size=data.shape[0]
        )

    def on_events(self, simproc: str, events: Iterable[Event]) -> None:
        ...
