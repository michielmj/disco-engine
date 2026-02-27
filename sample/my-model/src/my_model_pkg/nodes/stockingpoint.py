import pickle
import numpy as np
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

        # Stock data
        self.target_ip = np.asarray(data['ip_target'].fillna(0).array)
        self.inv_oh = self.target_ip.copy()

        # Demand data
        self.orderbook = Orderbook()
        self.demand_dists = get_dists(data, 'dem_dist', 'dem_params')
        self.demand_map = self.data.simproc('demand').outbound_map()
        self.next_order = 0.

        # Supply data
        self.supply_map = self.data.simproc('supply').outbound_map()
        lead_times = self.data.simproc('supply').outbound_edge_data(columns=['lead_time'])
        lead_times = lead_times.reset_index('target_index', drop=False)
        self.lead_times = [(
            lt,
            gb.Vector.from_coo(ix.array, True, size=self.data.num_vertices)
        ) for lt, ix in lead_times.groupby('lead_time')['target_index']]

        # Statistics
        self.dl_inv_pos = self.dlogger.register_periodic_stream({
            'node': self.name
        }, epoch_scale=1e-3, value_scale=1e-6,)

    def on_events(self, simproc: str, events: Iterable[Event]) -> None:
        if self.active_simproc_name == "demand":
            self.handle_demand_events(events)
        elif self.active_simproc_name == "supply":
            self.handle_supply_events(events)

    def handle_demand_events(self, events: Iterable[Event]):
        for event in events:
            assert isinstance(event.data, gb.Vector)
            arr = event.data[self.ixs].to_dense(0.)
            key = pickle.dumps(int(event.headers.get("delivery_node_idx")))
            self.orderbook.append(key=key, arr=arr)

        self.fulfill_demand()
        if self.next_order >= self.epoch:
            self.place_orders()
            self.next_order += 1.
            self.wakeup(self.next_order)

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
            outbound = self.data.incidence_matrix[target_node_idx, :] * self.supply_map
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

        self.dl_inv_pos.record(self.epoch, self.ixs, self.inv_oh)

    def place_orders(self):
        sample = sample_dists(
            rng=self.rng,
            dists=self.demand_dists,
            sample_indices=self.ixs,
            num_vertices=self.data.num_vertices,
            lower=0
        )

        order_qty = gb.op.plus_times(sample @ self.demand_map)
        for node, order in self.data.by_node(order_qty):
            self.send_event(
                target_node=node,
                target_simproc='demand',
                epoch=self.epoch,
                data=order,
                headers={
                    "delivery_node_idx": str(self.data.node_index)
                }
            )
