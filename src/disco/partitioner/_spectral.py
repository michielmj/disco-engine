# src/disco/partitioner/_spectral.py
"""
Spectral Clustering Partitioner.

Partitions a graph's node instances across up to *target_partition_count*
partitions while:
  - balancing per-partition compute weight (sum of vertex_weight), and
  - minimising cross-partition event traffic (sum of edge weights in all layers).

Algorithm
---------
1. Build a SuperGraph via :meth:`Graph.compress` using the union of all
   ``same_node`` label types from the model's node types.  Vertices sharing
   a same_node label value are merged into one supervertex, enforcing
   co-location constraints.
2. Enumerate node instances via :func:`_iter_node_instances` on the
   SuperGraph (one instance = all supervertices sharing the same node-type +
   distinct-label combination).
3. Project supervertex-level affinity and weights to the node-instance level
   using a scipy sparse projection matrix P of shape (n_instances × n_sv).
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

import graphblas as gb
import graphblas_algorithms as ga
from sklearn.cluster import SpectralClustering
import numpy as np

from disco.helpers import vstack_rows_from_coo
from disco.graph import Graph
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

        if self._graph.label_matrix_masked is None:
            raise ValueError("Graph has no labels attached (graph.label_matrix is None)")

        if NODE_TYPE not in self._graph.label_indices_by_type:
            raise KeyError(
                f"Label type {NODE_TYPE!r} not found in graph.label_type_vectors"
            )

        distinct_lt = set()
        same_lt = set()
        for node_type, nts in self._spec.node_types.items():
            for lt in nts.distinct_nodes:
                if lt not in self._graph.label_indices_by_type:
                    raise KeyError(
                        f"Label type {lt!r} (distinct_nodes for node_type {node_type!r}) "
                        f"not found in graph.label_type_vectors"
                    )
                else:
                    distinct_lt.add(lt)
            for lt in nts.same_node:
                if lt not in self._graph.label_indices_by_type:
                    raise KeyError(
                        f"Label type {lt!r} (same_node for node_type {node_type!r}) "
                        f"not found in graph.label_type_vectors"
                    )
                else:
                    same_lt.add(lt)

        self._distinct_lt = list(distinct_lt)
        self._same_lt = list(same_lt)

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

        # ---- Step 1: SuperGraph via public compress() ----
        sg = self._graph.compress(self._same_lt)
        n_sv = sg.num_vertices

        # Calculate symmetrical affinity matrix E
        E = gb.Matrix(nrows=n_sv, ncols=n_sv, dtype=gb.dtypes.FP64)
        for layer in sg.layers:
            E(accum=gb.binary.plus) << layer
        E(accum=gb.binary.plus) << E.T

        # ---- Step 2: get connected components ----
        cc = self._get_connected_components(E, sg.vertex_weight)

        # ---- Step 3: split components to maximum weight ----
        # target weight for partitions
        target_weight = sg.vertex_weight.sum() / target_partition_count + sg.vertex_weight.max() * 2

        components = []
        for weight, component in cc:
            if weight <= target_weight:
                components.append((weight, component))
            else:
                components.extend(self._split_component(E, component, sg.vertex_weight, target_weight))

        # ---- Step 4: combine components to number of partitions ----
        final_components = self._combine_components(components, target_partition_count, target_weight)


        # ---- Step 5: build Partitioning ----
        node_specs: List[NodeInstanceSpec] = []
        row_vectors = []

        # DEBUG
        all_idxs = np.array([], dtype=np.uint64)
        for c in final_components:
            idxs = sg.decompress(c)
            if np.intersect1d(all_idxs, idxs).shape[0] != 0:
                raise RuntimeError('Components not distinct.')
            all_idxs = np.union1d(all_idxs, idxs)

        for partition, sv_mask in enumerate(final_components):
            graph_view = sg.get_view(gb.Vector.from_coo(sv_mask, True, size=n_sv))

            for node_type, label_values, assignment_vector, _ in _iter_node_instances(
                    graph_view, self._model
            ):
                # DEBUG
                idxs, _ = assignment_vector.to_coo()
                assert np.setdiff1d(idxs, sv_mask).shape[0] == 0

                node_name = "-".join([f"p{partition}", node_type] + label_values)
                node_specs.append(
                    NodeInstanceSpec(partition=partition, node_name=node_name, node_type=node_type)
                )
                row_vectors.append(sg.decompress(assignment_vector))

        incidence = vstack_rows_from_coo(row_vectors, ncols=self._graph.num_vertices)

        return Partitioning.from_node_instance_spec(
            node_specs=node_specs,
            incidence=incidence,
            graph=self._graph,
            model=self._model,
        )

    @staticmethod
    def _get_connected_components(E: gb.Matrix, weights: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """
        Returns a list of ``(total_weight, sv_indices)`` tuples.
        """

        # each disconnected supervertex is its own component
        nz, _ = E.reduce_rowwise(gb.op.plus).select('!=', 0).new().to_coo()
        zeros = np.setdiff1d(np.arange(E.nrows, dtype=np.int64), nz)
        components = [(weights[v], np.array([v], dtype=int)) for v in zeros]

        G = ga.Graph(E)
        while nz.shape[0] > 0:
            c, _ = ga.components.node_connected_component(G, nz[0]).to_coo()
            w = weights[c].sum()
            components.append((w, c))
            nz = np.setdiff1d(nz, c)

        return components

    @staticmethod
    def _split_component(E: gb.Matrix, component: np.ndarray, vertex_weights: np.ndarray, target_weight: float) -> List[Tuple[float, np.ndarray]]:
        weight = vertex_weights[component].sum()
        n_clusters = min(
            component.shape[0], max(2, int(np.round(weight / target_weight)))
        )
        Es = gb.io.to_scipy_sparse(E[component, component], format="csr")
        Es.indices = Es.indices.astype(np.dtypes.Int32DType)  # SciPy needs 32 bit indices
        Es.indptr = Es.indptr.astype(np.dtypes.Int32DType)

        sc = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels="discretize",
            affinity="precomputed",
            random_state=0,
        )
        sc.fit(Es)

        lbls = sc.labels_
        clusters = np.unique(lbls)

        result = []
        for cl in clusters:
            c = component[lbls == cl]
            w = vertex_weights[c].sum()
            result.append((w, c))

        return result

    @staticmethod
    def _combine_components(
            components: list[tuple[float, np.ndarray]],
            target_partition_count: int,
            target_weight: float,
    ) -> list[np.ndarray]:
        """Combine components into partitions via LPT greedy + local refinement.

        Parameters
        ----------
        components : list of (weight, indices_array)
            Each entry is the total vertex weight and the array of vertex indices
            for one component. Should be sorted descending by weight for
            deterministic LPT behaviour.
        target_partition_count : int
            Desired number of output partitions (k).
        target_weight : float
            Maximum allowed total vertex weight per partition.

        Returns
        -------
        list of ndarray
            Exactly *target_partition_count* arrays of vertex indices.
        """
        k = target_partition_count
        n = len(components)

        comp_weights = [c[0] for c in components]

        # --- Phase 1: LPT greedy assignment --------------------------------
        # Sort descending by weight for longest-processing-time-first.
        order = sorted(range(n), key=lambda i: comp_weights[i], reverse=True)

        # Min-heap: (partition_weight, partition_index)
        part_members: list[list[int]] = [[] for _ in range(k)]
        part_weights = [0.0] * k
        heap = [(0.0, j) for j in range(k)]
        heapq.heapify(heap)

        for i in order:
            w = comp_weights[i]
            skipped: list[tuple[float, int]] = []
            placed = False
            while heap:
                pw, j = heapq.heappop(heap)
                if pw + w <= target_weight:
                    part_members[j].append(i)
                    part_weights[j] = pw + w
                    heapq.heappush(heap, (pw + w, j))
                    placed = True
                    break
                skipped.append((pw, j))
            for item in skipped:
                heapq.heappush(heap, item)
            if not placed:
                # Fallback: place in lightest partition regardless.
                pw, j = heapq.heappop(heap)
                part_members[j].append(i)
                part_weights[j] = pw + w
                heapq.heappush(heap, (pw + w, j))

        # --- Phase 2: local-search refinement --------------------------------
        improved = True
        while improved:
            improved = False
            j_max = max(range(k), key=lambda j: part_weights[j])
            j_min = min(range(k), key=lambda j: part_weights[j])
            spread = part_weights[j_max] - part_weights[j_min]
            if spread < 1e-12:
                break

            # Try single moves: heaviest → lightest
            for idx, ci in enumerate(part_members[j_max]):
                wc = comp_weights[ci]
                new_hi = part_weights[j_max] - wc
                new_lo = part_weights[j_min] + wc
                if new_lo <= target_weight and abs(new_hi - new_lo) < spread:
                    part_members[j_min].append(part_members[j_max].pop(idx))
                    part_weights[j_max] = new_hi
                    part_weights[j_min] = new_lo
                    improved = True
                    break

            if improved:
                continue

            # Try pairwise swaps between heaviest and lightest
            for ia, ca in enumerate(part_members[j_max]):
                for ib, cb in enumerate(part_members[j_min]):
                    delta = comp_weights[ca] - comp_weights[cb]
                    if delta <= 0:
                        continue
                    new_hi = part_weights[j_max] - delta
                    new_lo = part_weights[j_min] + delta
                    if new_lo <= target_weight and abs(new_hi - new_lo) < spread:
                        part_members[j_max][ia] = cb
                        part_members[j_min][ib] = ca
                        part_weights[j_max] = new_hi
                        part_weights[j_min] = new_lo
                        improved = True
                        break
                if improved:
                    break

        # --- Assemble output: concatenate vertex index arrays ----------------
        dtype = components[0][1].dtype if components else np.intp
        return [
            np.concatenate([components[ci][1] for ci in members])
            if members
            else np.empty(0, dtype=dtype)
            for members in part_members
        ]
