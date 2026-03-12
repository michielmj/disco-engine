# src/disco/partitioner/_simple.py
from __future__ import annotations

from typing import List

import graphblas as gb

from disco.helpers import vstack_rows_from_coo
from disco.graph import Graph
from disco.model import Model
from disco.partitioning import NodeInstanceSpec, Partitioning

from ._protocol import NODE_TYPE
from ._helpers import _iter_node_instances


class SimplePartitioner:
    """
    Baseline partitioner:
    - Always produces exactly 1 partition (partition 0).
    - Respects per-node-type distinct label types by splitting vertices into
      distinct Node instances per distinct-label tuple, separately for each node type.
    - Prerequisite: graph must contain labels for NODE_TYPE and each node type's
      distinct attributes.
    """

    def __init__(self, graph: Graph, model: Model) -> None:
        self._graph = graph
        self._model = model
        self._spec = model.spec

        # Validate that labels are attached
        if self._graph.label_matrix is None:
            raise ValueError("Graph has no labels attached (graph.label_matrix is None)")

        # Validate NODE_TYPE label type
        if NODE_TYPE not in self._graph.label_indices_by_type:
            raise KeyError(f"Label type {NODE_TYPE!r} not found in graph.label_type_vectors")

        # Validate all distinct label types exist (per node type)
        for node_type, nts in self._spec.node_types.items():
            for lt in nts.distinct_nodes:
                if lt not in self._graph.label_indices_by_type:
                    raise KeyError(
                        f"Label type {lt!r} (distinct_nodes for node_type {node_type!r}) "
                        f"not found in graph.label_type_vectors"
                    )

    def partition(self, target_partition_count: int) -> Partitioning:
        if target_partition_count < 1:
            raise ValueError("target_partition_count must be >= 1")

        node_specs: List[NodeInstanceSpec] = []
        row_vectors: List[gb.Vector] = []

        for node_type, label_values, assignment_vector, _ in _iter_node_instances(
            self._graph, self._model
        ):
            node_name = "-".join(["p0", node_type] + label_values)
            node_specs.append(
                NodeInstanceSpec(partition=0, node_name=node_name, node_type=node_type)
            )
            row_vectors.append(assignment_vector)

        if not node_specs:
            raise ValueError(
                "SimplePartitioner produced no nodes. "
                "Check that graph has NODE_TYPE labels matching model.spec.node_types keys, "
                "and that vertices carry those labels."
            )

        incidence = vstack_rows_from_coo(row_vectors, ncols=self._graph.num_vertices)

        return Partitioning.from_node_instance_spec(
            node_specs=node_specs,
            incidence=incidence,
            graph=self._graph,
            model=self._model,
        )
