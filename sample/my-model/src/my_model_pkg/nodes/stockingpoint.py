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

    def on_events(self, simproc: str, events: Iterable[Event]) -> None:
        pass

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

        # Start from target inventory position
        self.inv_oh = self.target_ip.dup()


        # orderbook holds orders that need to be fulfilled
        self.orderbook = Orderbook()  # list of customer orders to fulfill

        # inv_pos (inventory position) is on hand + on order - backlog
        self.inv_pos = self.inv_oh.copy()

        # place demand for backlog
        self.backlog = -1 * self.inv_oh.upper_cap()
        self.orderbook.append(packb(("self", None)), -1 * self.backlog.values)
        self.inv_pos -= self.backlog
        self.inv_oh += self.backlog
        self.on_order = Vector.zeros(self.indices)

        # pre-populate supply orders if not start_from_target
        if start_from_target:
            self.goods_receipts = None
        else:
            self.goods_receipts = self.context.get_goods_receipts()

        # statistics
        self.trace_net_inv = self.collector.get_trace(
            PLA.INV_NET, self.inv_oh - self.backlog, period=1.0
        )

        # optional traces
        if "traces" in init_args:
            traces = init_args["traces"]
            assert isinstance(traces, list)
        else:
            traces = []

        self.trace_supplies = (
            self.collector.get_trace(
                PLA.SUPPLIES, self.indices, period=1.0, accumulate=True
            )
            if PLA.SUPPLIES in traces
            else None
        )
        self.trace_orders = (
            self.collector.get_trace(
                PLA.ORDERS, self.indices, period=1.0, accumulate=True
            )
            if PLA.ORDERS in traces
            else None
        )
        self.trace_requirements = (
            self.collector.get_trace(
                PLA.REQUIREMENTS, self.indices, period=1.0, accumulate=True
            )
            if PLA.REQUIREMENTS in traces
            else None
        )
        self.trace_demand = (
            self.collector.get_trace(
                PLA.DEMAND, self.indices, period=1.0, accumulate=True
            )
            if PLA.DEMAND in traces
            else None
        )
        self.trace_inv_pos = (
            self.collector.get_trace(PLA.INV_POS, self.inv_pos, period=1.0)
            if PLA.INV_POS in traces
            else None
        )
        self.trace_on_order = (
            self.collector.get_trace(PLA.INV_OO, self.on_order, period=1.0)
            if PLA.INV_OO in traces
            else None
        )
        self.trace_backlog = (
            self.collector.get_trace(PLA.INV_BO, self.backlog, period=1.0)
            if PLA.INV_BO in traces
            else None
        )

    def sample_demand(self):
        # There may be nodes where no demand needs to be sampled at all
        if len(self.next_demand_epoch.indices) == 0:
            return

        # During the first epoch, we
        if self.epoch == 0.0:
            self.next_demand_epoch += sample_dists(
                random_state=self.rng,
                dists=self.iat_dists,
                sample_indices=self.next_demand_epoch.indices,
            )

        needed = self.next_demand_epoch.indices[
            self.next_demand_epoch.values <= self.epoch
        ]

        while needed.shape[0] != 0:
            iat = sample_dists(self.rng, self.iat_dists, needed)
            # assert np.all(iat.values > 0.0), f"Non-positive interarrival times: {iat}"

            self.next_demand_epoch += iat

            sample = Vector.zeros(self.indices)
            sample.update(sample_dists(self.rng, self.qty_dists, needed))

            self.orderbook.append(packb(("self", None)), sample.values)
            self.backlog += sample
            self.inv_pos -= sample
            if self.trace_requirements:
                self.trace_requirements.log(self.epoch, sample)
            if self.trace_demand:
                self.trace_demand.log(self.epoch, sample)

            needed = self.next_demand_epoch.indices[
                self.next_demand_epoch.values <= self.epoch
            ]

        next_epoch = np.ceil(self.next_demand_epoch.values.min())
        if next_epoch <= self.epoch:
            next_epoch += 1.0

        try:
            self.wakeup(next_epoch)
        except:
            logger.error(f"{self.name}: epoch={self.epoch}, wakeup epoch={next_epoch}")
            raise

    def handle_orders_events(self, events):
        """
        Receive orders from downstream nodes and place orders to upstream nodes.
        :param events: epoch, data, headers
        :return:
        """

        # process initial goods receipts
        if self.goods_receipts is not None:
            for epoch, group in self.goods_receipts.sort_values(by=EPOCH).groupby(
                by=EPOCH
            ):
                goods_receipt = Vector.from_dataframe(
                    group, indices=TO_INDEX, values=QUANTITY
                )
                self.send_event(
                    f"self/{SUPPLIES_TO}",
                    epoch=epoch,
                    data={"goods_receipt": goods_receipt, "sender": self.name},
                )
                self.inv_pos += goods_receipt
                self.on_order += goods_receipt

            self.goods_receipts = None

        self.sample_demand()

        logger.debug(
            f"{self.name}/order ({self.epoch}): receiving {len(events)} orders at {self.epoch}."
        )
        for sender, epoch, data, headers in events:
            if "order" in data:
                order: Vector = data["order"]
                route = data["route"]
                receiver = data["receiver"]

                order = order.conform_indices(self.inv_oh, strict=True)

                self.orderbook.append(packb((receiver, route)), order.values)
                self.backlog += order

                self.inv_pos -= order
                if self.trace_requirements:
                    self.trace_requirements.log(self.epoch, order)

        self.place_orders()

        # tell supply to wakeup
        if self.orderbook.size != 0:
            self.send_event(f"self/{SUPPLIES_TO}", self.epoch, {})

    def place_orders(self):
        precision = self.config.precision

        reqs = (self.target_ip - self.inv_pos).lower_cap()
        reqs = reqs.minmult(self.ord_min, self.ord_mult)

        if np.any(reqs.values > precision):
            for route, indices in self.routes.items():
                route_reqs = reqs.conform_indices(indices, strict=True)
                if len(route_reqs) > 0:
                    self.on_order += route_reqs

                    # map the requirements to source indices
                    dep_reqs = route_reqs @ self.bom
                    assert (
                        -precision
                        < dep_reqs.values.sum() - route_reqs.values.sum()
                        < precision
                    )

                    # split_by_node splits the requirements by the destination for the order and send
                    for origin, order in self.context.split_by_node(dep_reqs):
                        self.send_event(
                            target=f"{origin}/{ORDERS_AT}",
                            epoch=self.epoch,
                            data={
                                "order": order,
                                "route": route,
                                "receiver": self.name,
                            },
                        )

            # directly replenish requirements without a route
            no_route_reqs = Vector(
                indices=reqs.indices[self.no_route],
                values=reqs.values[self.no_route],
                validate=False,
            )
            self.inv_oh += no_route_reqs

        self.inv_pos += reqs
        if self.trace_orders:
            self.trace_orders.log(self.epoch, reqs)
        if self.trace_inv_pos:
            self.trace_inv_pos.log(self.epoch, self.inv_pos)

    def handle_supplies_events(self, events):
        """
        Receive incoming goods and fulfill orders.
        :param events:
        :return:
        """
        logger.debug(
            f"{self.name}/supply ({self.epoch}): receiving {len(events)} supplies at {self.epoch}."
        )

        # process supplies
        for sender, epoch, data, headers in events:
            if data == {}:
                supply = None
            elif "supply" in data:
                supply = data["supply"] @ self.bom_rt
            elif "goods_receipt" in data:
                supply = data["goods_receipt"]
            else:
                raise SimulationRuntimeError("Unknown supply event.")

            if supply:
                self.inv_oh += supply
                self.on_order -= supply
                if self.trace_supplies:
                    self.trace_supplies.log(self.epoch, supply)

        self.fulfill_orders()

        self.trace_net_inv.log(self.epoch, self.inv_oh - self.backlog)
        if self.trace_on_order:
            self.trace_on_order.log(self.epoch, self.on_order)

    def fulfill_orders(self):
        allocations = self.orderbook.allocate_greedy(self.inv_oh.values)

        for key, values in allocations:
            receiver, route = unpackb(key)
            supply = Vector(indices=self.indices, values=values, validate=False)

            if route is not None:
                lt = self.sample_leadtime(route)
                self.send_event(
                    f"{receiver}/{SUPPLIES_TO}", self.epoch + lt, {"supply": supply}
                )

            elif receiver != "self":
                raise SimulationRuntimeError("Missing route")

            self.backlog -= supply

        if self.trace_backlog:
            self.trace_backlog.log(self.epoch, self.backlog)

    def sample_leadtime(self, route):
        params = self.context.get_route(route)

        try:
            if params is None:
                raise Exception("Unknown route")
            elif params[RouteAttributes.LT_DIST] == "constant":
                return params[RouteAttributes.LT_MEAN]
            elif params[RouteAttributes.LT_DIST] == "truncnorm":
                return max(
                    0.0,
                    self.rng.normal(
                        params[RouteAttributes.LT_MEAN],
                        params[RouteAttributes.LT_VAR] ** 0.5,
                    ),
                )
            else:
                raise Exception("Unknown distribution")

        except KeyError:
            raise Exception("Distribution not fully specified")
