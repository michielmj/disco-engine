# src/disco/partitioner/_helpers.py
"""
Shared helpers for all Partitioner implementations.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import graphblas as gb

from disco.graph import Graph
from disco.model import Model

from ._protocol import NODE_TYPE


def _iter_node_instances(
    graph: Graph,
    model: Model,
) -> Iterator[Tuple[str, List[str], gb.Vector, np.ndarray]]:
    """
    Yield ``(node_type, label_values, assignment_vector, vertex_indices)``
    for each distinct node instance found in the graph.

    Parameters
    ----------
    graph:
        The labelled graph.  Must have ``label_matrix`` attached and a
        label of type ``NODE_TYPE`` for every vertex.
    model:
        The simulation model whose ``spec.node_types`` defines distinct-node
        label attributes.

    Yields
    ------
    node_type : str
        The node-type value for this instance.
    label_values : List[str]
        Distinct label values (in ``distinct_nodes`` order, skipping any
        types not present on the supporting vertices).
    assignment_vector : gb.Vector
        GraphBLAS BOOL vector of size ``graph.num_vertices``; True at each
        vertex belonging to this node instance.
    vertex_indices : np.ndarray
        Sorted int64 array of vertex indices belonging to this node instance.
    """
    if graph.label_matrix is None:
        return

    spec = model.spec
    for node_type, nts in spec.node_types.items():
        distinct = [NODE_TYPE] + list(nts.distinct_nodes)

        combos: List[np.ndarray] = graph.by_distinct_labels(distinct=distinct)

        for co in combos:
            if len(co) == 0:
                continue

            node_type_label_id = int(co[0])
            lt0, lv0 = graph.label_meta[node_type_label_id]

            # Validate the first entry is a NODE_TYPE label
            if lt0 != NODE_TYPE:
                raise ValueError(
                    "by_distinct_labels returned a combo whose first element is not a "
                    f"NODE_TYPE label id (got label type {lt0!r}).  "
                    "Expected combo[0] to correspond to NODE_TYPE."
                )
            if lv0 != node_type:
                # This combo belongs to a different node type; skip it.
                continue

            # Vertex assignment: vertices that carry ALL labels in this combo.
            cnt = (
                graph.label_matrix[:, co]
                .new(dtype=gb.dtypes.INT64)
                .reduce_rowwise(gb.monoid.plus)
            )
            assignment_vector: gb.Vector = (cnt == len(co)).new()

            # Vertex indices
            vertex_indices, _ = assignment_vector.to_coo()

            # Decode distinct label values (skip co[0] which is NODE_TYPE)
            label_values: List[str] = []
            for entry in co[1:]:
                _, lvk = graph.label_meta[int(entry)]
                label_values.append(str(lvk))

            yield node_type, label_values, assignment_vector, vertex_indices.astype(
                np.int64
            )
