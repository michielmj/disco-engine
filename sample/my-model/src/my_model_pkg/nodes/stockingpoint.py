import pickle
from typing import Iterable

from tools.mp_logging import getLogger
from toolbox.orderbook import Orderbook
import graphblas as gb

from disco import Node
from disco.node import Event
from my_model_pkg.sampling import get_dists, sample_dists

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

    def initialize(self, **init_args):

        data = self.data.vertex_data(columns=['ip_target', 'dem_dist', 'dem_params'])
        self.ixs = data.index.array
        self.target_ip = data['ip_target'].fillna(0).array
        self.inv_oh = self.target_ip.copy()

        self.demand_dists = get_dists(data, 'dem_dist', 'dem_params')

        self.orderbook = Orderbook()
        self.outbound = self.data.simproc('supply').outbound_map()
        lead_times = self.data.simproc('supply').outbound_edge_data(columns=['lt_mean'])
        lead_times = lead_times.reset_index('target_index', drop=False)
        self.lead_times = [(
            lt,
            gb.Vector.from_coo(ix.array, True, size=self.data.num_vertices)
        ) for lt, ix in lead_times.groupby('lt_mean')['target_index']]


    def on_events(self, simproc: str, events: Iterable[Event]) -> None:
        if self.active_simproc_name == "demand":
            self.handle_demand_events(events)
        elif self.active_simproc_name == "supply":
            self.handle_supply_events(events)

    def handle_demand_events(self, events: Iterable[Event]):
        for event in events:
            assert event.data is gb.Vector
            arr = event.data[self.ixs].to_dense(0.)
            key = pickle.dumps(event.headers.get("delivery_node_idx"))
            self.orderbook.append(key=key, arr=arr)

        self.fulfill_demand()
        self.place_orders()

    def handle_supply_events(self, events: Iterable[Event]):
        for event in events:
            assert event.data is gb.Vector
            arr = event.data[self.ixs].to_dense(0.)
            self.inv_oh += arr

        self.fulfill_demand()

    def fulfill_demand(self):

        for key, arr in self.orderbook.allocate_greedy(self.inv_oh):
            target_node_idx = pickle.loads(key)
            target_node = self.data.node_specs[target_node_idx].node_name
            fulfillment = gb.Vector.from_coo(
                self.ixs,
                arr,
                size=self.data.num_vertices
            )
            outbound = self.data.incidence_matrix[target_node_idx, :] * self.outbound
            delivery = gb.op.plus_times(fulfillment @ outbound).select('!=', 0).new()
            for lt, mask in self.lead_times:
                lt_delivery = delivery.dup(mask=mask)
                if lt_delivery.nvals > 0:
                    self.send_event(
                        target_node=target_node,
                        target_simproc='supply',
                        epoch=self.epoch + lt,
                        data=lt_delivery
                    )

    def place_orders(self):
        sample = sample_dists(
            rng=self.rng,
            dists=self.demand_dists,
            lower=0
        )




