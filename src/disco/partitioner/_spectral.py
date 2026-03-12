# src/disco/partitioner/_spectral.py
"""
Spectral Clustering Partitioner.

Partitions a graph's node instances across up to *target_partition_count*
partitions while:
  - balancing per-partition compute weight (sum of vertex_weight), and
  - minimising cross-partition event traffic (sum of edge weights in all layers).

Algorithm
---------
1. Enumerate node instances via :func:`_iter_node_instances` (one instance =
   all vertices sharing the same node-type + distinct-label combination).
2. Build a *SuperGraph* where each node instance is one supervertex, using
   :meth:`Graph._build_supergraph`.  Supervertex weights equal the summed
   ``vertex_weight`` of constituent vertices.
3. Build a symmetric affinity matrix from the SuperGraph layers
   (sum of ``A + Aᵀ`` across all layers) via
   ``graphblas.io.to_scipy_sparse``.
4. Recursively split connected components using
   ``sklearn.cluster.SpectralClustering`` until every cluster weight ≤
   ``target_weight = total_weight / n_partitions + 2 * max_weight``.
5. Greedily merge clusters into *target_partition_count* partitions (always
   combining the lightest cluster with the heaviest remaining unmatched
   cluster) to balance total partition weight.
6. Build ``NodeInstanceSpec`` objects and the incidence matrix, then return
   a ``Partitioning``.
"""
from __future__ import annotations

import heapq
from typing import List, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import sklearn.cluster
from graphblas import io as gb_io

from disco.helpers import vstack_rows_from_coo
from disco.graph import Graph
from disco.graph.core import SuperGraph
from disco.model import Model
from disco.partitioning import NodeInstanceSpec, Partitioning

from ._protocol import NODE_TYPE
from ._helpers import _iter_node_instances


class SpectralClusteringPartitioner:
    """
    Multi-partition partitioner using spectral clustering.

    Groups node instances into at most *target_partition_count* partitions,
    balancing compute weight while minimising cross-partition traffic.

    Parameters
    ----------
    graph:
        A labelled ``Graph``.  Must carry ``NODE_TYPE`` labels and every
        ``distinct_nodes`` label type referenced by the model.
    model:
        The simulation model.
    """

    def __init__(self, graph: Graph, model: Model) -> None:
        self._graph = graph
        self._model = model
        self._spec = model.spec

        if self._graph.label_matrix is None:
            raise ValueError("Graph has no labels attached (graph.label_matrix is None)")

        if NODE_TYPE not in self._graph.label_indices_by_type:
            raise KeyError(
                f"Label type {NODE_TYPE!r} not found in graph.label_type_vectors"
            )

        for node_type, nts in self._spec.node_types.items():
            for lt in nts.distinct_nodes:
                if lt not in self._graph.label_indices_by_type:
                    raise KeyError(
                        f"Label type {lt!r} (distinct_nodes for node_type {node_type!r}) "
                        f"not found in graph.label_type_vectors"
                    )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def partition(self, target_partition_count: int) -> Partitioning:
        """
        Partition the graph into at most *target_partition_count* partitions.

        Returns a :class:`~disco.partitioning.Partitioning` whose
        ``num_partitions`` is in ``[1, target_partition_count]``.
        """
        if target_partition_count < 1:
            raise ValueError("target_partition_count must be >= 1")

        instances = list(_iter_node_instances(self._graph, self._model))

        if not instances:
            raise ValueError(
                "SpectralClusteringPartitioner found no node instances. "
                "Check that graph has NODE_TYPE labels matching model.spec.node_types keys."
            )

        n_instances = len(instances)

        # ---- Step 1: build SuperGraph (one supervertex per node instance) ----
        vertex_map = np.empty(self._graph.num_vertices, dtype=np.int64)
        for sv_idx, (_, _, _, vertex_indices) in enumerate(instances):
            vertex_map[vertex_indices] = sv_idx

        super_graph = self._graph._build_supergraph(vertex_map, n_instances)

        # ---- Step 2: affinity matrix ----
        affinity = self._extract_affinity(super_graph)
        W = super_graph.vertex_weight  # shape (n_instances,)

        # ---- Step 3: cluster + group ----
        if n_instances == 1 or affinity.nnz == 0:
            # Trivial: assign all instances to partition 0.
            partition_assignment = np.zeros(n_instances, dtype=np.int64)
        else:
            target_weight = float(W.sum()) / target_partition_count + 2.0 * float(W.max())
            clusters = self._cluster(affinity, W, target_weight)
            grouped = self._group_clusters(W, clusters, target_partition_count)

            partition_assignment = np.empty(n_instances, dtype=np.int64)
            for part_idx, (_, sv_indices) in enumerate(grouped):
                partition_assignment[sv_indices] = part_idx

        # ---- Step 4: build Partitioning ----
        node_specs: List[NodeInstanceSpec] = []
        row_vectors = []

        for sv_idx, (node_type, label_values, assignment_vector, _) in enumerate(instances):
            part_idx = int(partition_assignment[sv_idx])
            node_name = "-".join([f"p{part_idx}", node_type] + label_values)
            node_specs.append(
                NodeInstanceSpec(
                    partition=part_idx,
                    node_name=node_name,
                    node_type=node_type,
                )
            )
            row_vectors.append(assignment_vector)

        incidence = vstack_rows_from_coo(row_vectors, ncols=self._graph.num_vertices)

        return Partitioning.from_node_instance_spec(
            node_specs=node_specs,
            incidence=incidence,
            graph=self._graph,
            model=self._model,
        )

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_affinity(super_graph: SuperGraph) -> scipy.sparse.csr_array:
        """
        Build a symmetric affinity matrix from the SuperGraph layers.

        Each layer ``A`` contributes ``A + Aᵀ`` so that the result is
        undirected.  GraphBLAS matrices are converted directly to
        ``scipy.sparse.csr_array`` via ``graphblas.io.to_scipy_sparse``.
        """
        n = super_graph.num_vertices
        total: scipy.sparse.csr_array | None = None

        for layer in super_graph.layers:
            A: scipy.sparse.csr_array = gb_io.to_scipy_sparse(layer, format="csr")
            sym = A + A.T
            total = sym if total is None else total + sym

        if total is None:
            return scipy.sparse.csr_array((n, n), dtype=np.float64)

        return total

    @staticmethod
    def _push_component_masks(
        heap: list,
        affinity: scipy.sparse.csr_array,
        weights: np.ndarray,
        submask: np.ndarray | None = None,
    ) -> None:
        """
        Find connected components of ``affinity[submask][:, submask]`` and push
        each component onto *heap* as ``(-component_weight, first_idx, mask)``.
        """
        if submask is None:
            submask = np.arange(affinity.shape[0], dtype=np.int64)

        if submask.size == 0:
            return

        sub = affinity[np.ix_(submask, submask)]
        _, labels = scipy.sparse.csgraph.connected_components(sub, directed=False)

        for component_id in np.unique(labels):
            local_mask = np.where(labels == component_id)[0]
            component_mask = submask[local_mask]
            component_weight = float(weights[component_mask].sum())
            heapq.heappush(
                heap,
                (-component_weight, int(component_mask[0]), component_mask),
            )

    @staticmethod
    def _cluster(
        affinity: scipy.sparse.csr_array,
        weights: np.ndarray,
        target_weight: float,
    ) -> List[np.ndarray]:
        """
        Recursively split connected components until every cluster has weight
        ≤ *target_weight*.

        Returns a list of supervertex-index arrays (one per cluster).
        """
        heap: list = []
        SpectralClusteringPartitioner._push_component_masks(heap, affinity, weights)

        while heap and heap[0][0] < -target_weight:
            neg_w, _, cluster_mask = heapq.heappop(heap)
            cluster_weight = -neg_w

            sub = affinity[np.ix_(cluster_mask, cluster_mask)]
            n_clusters = min(
                cluster_mask.size,
                max(2, int(np.round(cluster_weight / target_weight))),
            )

            if n_clusters == 1:
                heapq.heappush(heap, (0.0, int(cluster_mask[0]), cluster_mask))

            elif cluster_mask.size == 2:
                heapq.heappush(heap, (0.0, int(cluster_mask[0]), cluster_mask[:1]))
                heapq.heappush(heap, (0.0, int(cluster_mask[1]), cluster_mask[1:]))

            else:
                clustering = sklearn.cluster.SpectralClustering(
                    n_clusters=n_clusters,
                    assign_labels="discretize",
                    affinity="precomputed",
                    random_state=0,
                ).fit(sub.toarray())

                for cl_label in np.unique(clustering.labels_):
                    split_mask = cluster_mask[clustering.labels_ == cl_label]
                    SpectralClusteringPartitioner._push_component_masks(
                        heap, affinity, weights, split_mask
                    )

        return [mask for _, _, mask in heap]

    @staticmethod
    def _group_clusters(
        weights: np.ndarray,
        clusters: List[np.ndarray],
        ngroups: int,
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Greedily merge clusters into exactly *ngroups* groups (or fewer if
        there are not enough clusters) by combining the lightest and heaviest
        remaining unmatched clusters.

        Returns a list of ``(total_weight, combined_supervertex_indices)`` tuples.
        """
        weighted: list = sorted(
            [(float(weights[mask].sum()), mask[0], mask) for mask in clusters]
        )

        if len(weighted) <= ngroups:
            return [(w, mask) for w, _, mask in weighted]

        # Seed with the *ngroups* heaviest clusters.
        groups: list = []
        for _ in range(ngroups):
            heapq.heappush(groups, weighted.pop(-1))

        # Merge remaining (smallest first) into the lightest group.
        while weighted:
            lightest_group = heapq.heappop(groups)
            heaviest_remaining = weighted.pop(-1)
            heapq.heappush(
                groups,
                (
                    heaviest_remaining[0] + lightest_group[0],
                    heaviest_remaining[1],
                    np.concatenate([heaviest_remaining[2], lightest_group[2]]),
                ),
            )

        return [(w, mask) for w, _, mask in groups]
