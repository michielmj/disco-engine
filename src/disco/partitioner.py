# src/disco/partitioner.py
from __future__ import annotations

from typing import List, Protocol

import numpy as np
import graphblas as gb

from disco.helpers import vstack_rows_from_coo

from .graph import Graph
from .model import Model
from .partitioning import NodeInstanceSpec, Partitioning

NODE_TYPE = "node-type"


class Partitioner(Protocol):
    """
    A Partitioner produces a Partitioning for a given (Model, Graph) and target partition count.

    The returned Partitioning.num_partitions must be in [1, target_partition_count].
    """

    def partition(self, target_partition_count: int) -> Partitioning: ...


class SimplePartitioner:
    """
    Baseline partitioner:
    - Always produces exactly 1 partition (partition 0).
    - Respects per-node-type distinct label types by splitting vertices into
      distinct Node instances per distinct-label tuple, separately for each node type.
    - Prerequisite: graph must contain labels for NODE_TYPE and each node type's distinct attributes.
    """

    def __init__(self, graph: Graph, model: Model) -> None:
        self._graph = graph
        self._model = model
        self._spec = model.spec

        # Validate that labels are attached
        if self._graph.label_matrix is None:
            raise ValueError("Graph has no labels attached (graph.label_matrix is None)")

        # Validate NODE_TYPE label type
        if NODE_TYPE not in self._graph.label_type_vectors:
            raise KeyError(f"Label type {NODE_TYPE!r} not found in graph.label_type_vectors")

        # Validate all distinct label types exist (per node type)
        for node_type, nts in self._spec.node_types.items():
            for lt in nts.distinct_nodes:
                if lt not in self._graph.label_type_vectors:
                    raise KeyError(
                        f"Label type {lt!r} (distinct_nodes for node_type {node_type!r}) "
                        f"not found in graph.label_type_vectors"
                    )

    def partition(self, target_partition_count: int) -> Partitioning:
        if target_partition_count < 1:
            raise ValueError("target_partition_count must be >= 1")

        if self._graph.label_matrix is None:
            # defensive (should already be validated in __init__)
            raise ValueError("Graph has no labels attached (graph.label_matrix is None)")

        node_specs: List[NodeInstanceSpec] = []
        row_vectors: List[gb.Vector] = []

        # Build nodes per node_type, each with its own distinct label types.
        for node_type, nts in self._spec.node_types.items():
            distinct = [NODE_TYPE] + list(nts.distinct_nodes)

            combos: np.ndarray = self._graph.by_distinct_labels(distinct=distinct)

            # combos shape: (n_combos, len(distinct))
            # combos[:, 0] corresponds to NODE_TYPE label ids
            for co in combos:
                node_type_label_id = int(co[0])
                lt0, lv0 = self._graph.label_meta[node_type_label_id]
                if lt0 != NODE_TYPE:
                    raise ValueError(
                        "by_distinct_labels returned a combo whose first element is not a NODE_TYPE label id. "
                        "Expected combo[0] to correspond to NODE_TYPE."
                    )
                if lv0 != node_type:
                    # combo belongs to a different node type
                    continue

                # Vertex assignment: vertices that have ALL labels in this combo
                cnt = self._graph.label_matrix[:, co].new(dtype=gb.dtypes.INT64).reduce_rowwise(gb.monoid.plus)
                v = cnt == len(co)

                # .reduce_rowwise("land")
                row_vectors.append(v.new())

                # Node name: p0-<node_type>-<distinct_value_1>-...
                parts: List[str] = [f"p0", str(node_type)]
                for k in range(1, len(distinct)):
                    lbl_id = int(co[k])
                    ltk, lvk = self._graph.label_meta[lbl_id]
                    expected_type = distinct[k]
                    if ltk != expected_type:
                        raise ValueError(
                            f"Combo decoding mismatch at position {k}: expected label_type {expected_type!r}, "
                            f"got {ltk!r}. This implies by_distinct_labels output is not aligned to `distinct` order."
                        )
                    parts.append(str(lvk))

                node_name = "-".join(parts)

                node_specs.append(
                    NodeInstanceSpec(
                        partition=0,
                        node_name=node_name,
                        node_type=str(node_type),
                    )
                )

        if not node_specs:
            raise ValueError(
                "SimplePartitioner produced no nodes. "
                "Check that graph has NODE_TYPE labels matching model.spec.node_types keys, "
                "and that vertices carry those labels."
            )

        # Stack to incidence: (n_nodes x n_vertices). NO transpose needed.
        incidence = vstack_rows_from_coo(row_vectors, ncols=self._graph.num_vertices)

        return Partitioning.from_node_instance_spec(
            node_specs=node_specs,
            incidence=incidence,
            graph=self._graph,
            model=self._model,
        )
