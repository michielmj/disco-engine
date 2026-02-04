# src/disco/graph/core.py
from __future__ import annotations

# from typing import Mapping, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Any
from typing import Mapping, Dict, Optional, Tuple, List
from types import MappingProxyType

import numpy as np
import graphblas as gb
from graphblas import Matrix, Vector

from disco.helpers import has_self_loop, matrix_has_cycle
from .graph_mask import GraphMask


class Graph:
    """
    Layered directed graph backed by python-graphblas.

    Structure:
      - Vertices are 0..num_vertices-1 per scenario.
      - Layers: adjacency Matrix per layer_idx (directed, weighted).
      - Optional vertex mask: Vector[BOOL] (wrapped as GraphMask for DB use).
      - Labels:
          * Global label ids 0..num_labels-1 (per scenario).
          * label_matrix: Matrix[BOOL] with shape (num_vertices, num_labels)
                rows  = vertices
                cols  = labels
                True  = vertex has that label
          * label_meta: mapping label_index -> (label_type, label_value)
          * label_type_vectors: mapping label_type -> Vector[BOOL] over labels
                (size num_labels; True where label belongs to that type)
    """

    __slots__ = (
        "_layers",
        "num_vertices",
        "scenario_id",
        "_mask",                # GraphMask | None
        "_label_matrix",        # Matrix[BOOL] | None
        "_label_meta",          # dict[int, tuple[str, str]]
        "_label_type_vectors",  # dict[str, Vector[BOOL]]
        "_node_type_matrix",    # Matrix[BOOL] | None
        "_node_type_meta",      # dict[int, str]
        "num_labels",
    )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        layers: Tuple[Matrix, ...],
        num_vertices: int,
        scenario_id: str = '',
        *,
        mask: Optional[Vector] = None,
        label_matrix: Optional[Matrix] = None,
        label_meta: Optional[Dict[int, Tuple[str, str]]] = None,
        label_type_vectors: Optional[Dict[str, Vector]] = None,
        node_type_matrix: gb.Matrix | None = None,
        node_type_meta: Mapping[int, str] | None = None,
    ) -> None:
        # Avoid unnecessary copies; only wrap if needed
        self._layers = layers

        self.num_vertices = num_vertices
        self.validate(check_cycles=False)

        self.scenario_id = scenario_id

        # mask (GraphMask, internal)
        self._mask: Optional[GraphMask] = None
        if mask is not None:
            self.set_mask(mask)

        # labels
        self._label_matrix: Optional[Matrix] = None
        self._label_meta: Dict[int, Tuple[str, str]] = {}
        self._label_type_vectors: Dict[str, Vector] = {}
        self.num_labels = 0
        if label_matrix is not None:
            self.set_labels(label_matrix, label_meta, label_type_vectors)

        # node types
        # node types
        self._node_type_matrix: Optional[Matrix] = None
        self._node_type_meta: Dict[int, str] = {}
        if node_type_matrix is not None:
            self.set_node_types(node_type_matrix, node_type_meta)

    @classmethod
    def from_edges(
            cls,
            edge_layers: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
            *,
            num_vertices: int,
            scenario_id: str = '',
    ) -> Graph:
        """
        Build a Graph from per-layer edge arrays.

        edge_layers:
            mapping layer_idx -> (source_vertex_indices, target_vertex_indices, weights)
        num_vertices:
            total number of vertices (0..num_vertices-1)
        """
        num_vertices = int(num_vertices)

        layers_dict: Dict[int, Matrix] = {}
        for layer_idx, (src, dst, w) in edge_layers.items():
            src_arr = np.asarray(src, dtype=np.int64)
            dst_arr = np.asarray(dst, dtype=np.int64)
            val_arr = np.asarray(w)

            mat = gb.Matrix.from_coo(
                src_arr,
                dst_arr,
                val_arr,
                nrows=num_vertices,
                ncols=num_vertices,
            )
            layers_dict[int(layer_idx)] = mat

        num_layers = len(layers_dict)
        for idx in layers_dict:
            if idx < 0 or idx >= num_layers:
                raise ValueError('Layer indices must be contiguous.')

        layers = tuple(layers_dict[i] for i in range(num_layers))

        return cls(layers=layers, num_vertices=num_vertices, scenario_id=scenario_id)

    def validate(self, check_cycles: bool = True):
        for idx, layer in enumerate(self._layers):
            if layer.ncols != self.num_vertices:
                raise ValueError(f'Matrix layer {idx} has invalid number of columns.')
            if layer.nrows != self.num_vertices:
                raise ValueError(f'Matrix layer {idx} has invalid number of rows.')
            if has_self_loop(layer):
                raise ValueError(f'Matrix layer {idx} has self loops.')
            if check_cycles:
                if matrix_has_cycle(layer):
                    raise ValueError(f"Matrix layer {idx} appears to have cycles.")

    # ------------------------------------------------------------------ #
    # Mask handling (public API: Vector; internal: GraphMask)
    # ------------------------------------------------------------------ #
    def set_mask(self, mask_vector: Optional[Vector]) -> None:
        """
        Set or clear the vertex mask.

        - mask_vector is a GraphBLAS Vector[BOOL] of size num_vertices.
        - Internally stored as a GraphMask to support DB persistence when needed.
        """
        if mask_vector is None:
            self._mask = None
            return

        if mask_vector.dtype is not gb.dtypes.BOOL:
            raise TypeError(f"Mask vector must have BOOL dtype, got {mask_vector.dtype!r}")
        if mask_vector.size != self.num_vertices:
            raise ValueError(
                f"Mask size ({mask_vector.size}) must match num_vertices ({self.num_vertices})"
            )

        self._mask = GraphMask(mask_vector, scenario_id=self.scenario_id)

    @property
    def mask_vector(self) -> Optional[Vector]:
        """Return the underlying GraphBLAS mask vector, if any."""
        if self._mask is None:
            return None
        return self._mask.vector

    @property
    def graph_mask(self) -> Optional[GraphMask]:
        """
        Return the GraphMask wrapper for this graph, if any.

        This is mainly for internal use by DB/extract helpers, but exposed
        as a read-only property instead of an underscored helper.
        """
        return self._mask

    def set_node_types(
        self,
        node_type_matrix: Matrix,
        node_type_meta: Mapping[int, str] | None = None,
    ) -> None:
        """
        Attach node-type information to the graph.

        Requirements:
          - node_type_matrix.dtype must be BOOL
          - node_type_matrix.nrows == num_vertices
          - node_type_meta must contain an entry for each column index
            in [0, node_type_matrix.ncols)
        """
        # dtype check
        if node_type_matrix.dtype is not gb.dtypes.BOOL:
            raise TypeError(
                f"node_type_matrix must have BOOL dtype, got {node_type_matrix.dtype!r}"
            )

        # shape check
        if node_type_matrix.nrows != self.num_vertices:
            raise ValueError(
                f"node_type_matrix.nrows ({node_type_matrix.nrows}) "
                f"must match num_vertices ({self.num_vertices})"
            )

        ncols = node_type_matrix.ncols

        # metadata presence check
        if node_type_meta is None:
            raise ValueError(
                "node_type_meta must be provided and contain an entry "
                "for each node_type_index (0..ncols-1)"
            )

        for idx in range(ncols):
            if idx not in node_type_meta:
                raise ValueError(
                    f"node_type_meta missing entry for node_type_index {idx}"
                )

        # Optionally you could check there are not *just* extra keys,
        # but for now we allow extras and simply store the mapping.
        self._node_type_matrix = node_type_matrix
        self._node_type_meta = dict(node_type_meta)

    # ------------------------------------------------------------------ #
    # Label handling
    # ------------------------------------------------------------------ #
    def set_labels(
        self,
        label_matrix: Matrix,
        label_meta: Optional[Dict[int, Tuple[str, str]]] = None,
        label_type_vectors: Optional[Dict[str, Vector]] = None,
    ) -> None:
        """
        Attach label structures to the graph.
        """
        if label_matrix.dtype is not gb.dtypes.BOOL:
            raise TypeError(f"label_matrix must have BOOL dtype, got {label_matrix.dtype!r}")
        if label_matrix.nrows != self.num_vertices:
            raise ValueError(
                f"label_matrix.nrows ({label_matrix.nrows}) must match num_vertices ({self.num_vertices})"
            )

        self._label_matrix = label_matrix
        self.num_labels = label_matrix.ncols

        # Keep references; no copies
        self._label_meta = label_meta if label_meta is not None else {}
        self._label_type_vectors = {}

        if label_type_vectors is not None:
            for t, vec in label_type_vectors.items():
                if vec.dtype is not gb.dtypes.BOOL:
                    raise TypeError(f"Label type vector for {t!r} must be BOOL, got {vec.dtype!r}")
                if vec.size != self.num_labels:
                    raise ValueError(
                        f"Label type vector size ({vec.size}) must match num_labels ({self.num_labels})"
                    )
                self._label_type_vectors[t] = vec

    @property
    def label_matrix(self) -> Optional[Matrix]:
        """Sparse boolean matrix of label assignments (vertices x labels)."""
        return self._label_matrix

    @property
    def label_meta(self) -> Mapping[int, Tuple[str, str]]:
        """
        Read-only view of label_index -> (label_type, label_value).
        Mutating the underlying dict is still possible through other references,
        but this property itself cannot be assigned to.
        """
        return MappingProxyType(self._label_meta)

    @property
    def label_type_vectors(self) -> Mapping[str, Vector]:
        """
        Read-only view of label_type -> Vector[BOOL] over label indices.
        """
        return MappingProxyType(self._label_type_vectors)

    def get_vertex_mask_for_label_id(self, label_id: int) -> Vector:
        """
        Return a Vector[BOOL] of size num_vertices indicating which vertices
        have the given label (by label id 0..num_labels-1).
        """
        if self._label_matrix is None:
            raise RuntimeError("Graph has no labels attached (label_matrix is None)")
        if not (0 <= label_id < self.num_labels):
            raise IndexError(f"label_id {label_id} out of range [0, {self.num_labels})")
        return self._label_matrix[:, int(label_id)]

    def get_vertex_mask_for_label_type(self, label_type: str) -> Vector:
        """
        Return a Vector[BOOL] of size num_vertices indicating which vertices
        have ANY label of the given type.
        """
        if self._label_matrix is None:
            raise RuntimeError("Graph has no labels attached (label_matrix is None)")
        if label_type not in self._label_type_vectors:
            raise KeyError(f"Unknown label_type {label_type!r}")

        type_vec = self._label_type_vectors[label_type]
        return self._label_matrix.mxv(type_vec, gb.semiring.lor_land)

    @classmethod
    def _rowwise_and_kron(cls, A: Matrix, B: Matrix) -> Matrix:
        """
        Row-wise Kronecker product for boolean matrices.

        A: (v x a) BOOL, B: (v x b) BOOL  ->  (v x (a*b)) BOOL
        """
        v, a = A.nrows, A.ncols
        _, b = B.nrows, B.ncols

        true_b = gb.Matrix.from_scalar(True, 1, b, dtype=gb.dtypes.BOOL)
        true_a = gb.Matrix.from_scalar(True, 1, a, dtype=gb.dtypes.BOOL)

        left = A.kronecker(true_b, "land")  # (v x a*b)
        right = true_a.kronecker(B, "land")  # (v x a*b)

        return left.ewise_mult(right, "land")

    @classmethod
    def _rowwise_and_kron_many(cls, *Ls: Matrix) -> Matrix:
        """
        Row-wise Kronecker product over multiple label-type matrices.
        """
        M = Ls[0]
        for X in Ls[1:]:
            M = cls._rowwise_and_kron(M, X)
        return M

    def _select_ixs_and_labels(self, label_type: str) -> Tuple[np.ndarray, Matrix]:
        """
        Retrieve (label_indices, label_matrix_for_type) for a specific label type.

        Returns:
            indices: np.ndarray of label indices (columns in the global label matrix)
            submat:  Matrix[BOOL] of shape (num_vertices, len(indices))
        """
        ids = self._label_type_vectors[label_type]
        ixs, vals = ids.to_coo()
        # Keep only indices where the type-vector is True; no sorting or special casing.
        idx = ixs[vals]
        return idx, self._label_matrix[:, idx]

    def by_distinct_labels(self, distinct: List[str]) -> np.ndarray:
        """
        Compute distinct combinations of labels across one or more label types.

        Each row in the returned array represents a combination of *global* label
        indices (one per label type in `distinct`) for which there exists at least
        one vertex that has all of those labels.

        If a vertex has at most one label per label type, no vertex occurs in more
        than one combination.
        """

        # Split the label matrix into one sub-matrix per label type.
        # Each element is (label_indices, label_submatrix) where:
        #   - label_indices: np.ndarray of global label indices for that type
        #   - label_submatrix: gb.Matrix[BOOL] with shape (num_vertices x num_labels_of_type)
        label_indices_and_matrices: List[tuple[np.ndarray, gb.Matrix]] = [
            self._select_ixs_and_labels(label_type) for label_type in distinct
        ]

        # Build an expanded label matrix with all possible combinations of labels
        # across the different label types.
        # S has shape (num_vertices x num_combinations).
        S: gb.Matrix = self._rowwise_and_kron_many(
            *[label_matrix for (_, label_matrix) in label_indices_and_matrices]
        )

        # Determine which combination-columns have at least one supporting vertex.
        combo_support: gb.Vector = S.reduce_columnwise("lor")
        combo_indices, combo_vals = combo_support.to_coo()
        combo_indices = combo_indices[combo_vals]

        # Decode combination indices back into original global label indices.
        rows_per_type: List[np.ndarray] = []
        base = S.ncols
        for label_indices, label_matrix in label_indices_and_matrices:
            d = combo_indices % base
            base //= label_matrix.ncols
            d = d // base
            rows_per_type.append(label_indices[d])

        # Shape: (num_combinations, len(distinct))
        return np.array(rows_per_type).T

    # ------------------------------------------------------------------ #
    # Structural accessors
    # ------------------------------------------------------------------ #
    @property
    def layers(self) -> Tuple[Matrix, ...]:
        """
        Read-only view of the internal layer mapping.

        NOTE:
        - The mapping itself cannot be mutated through this property
          (MappingProxyType).
        - The *Matrix* objects inside are still mutable by design.
        """
        return self._layers

    def get_matrix(self, layer: int) -> Matrix:
        """Return the full adjacency matrix for a layer (no masking applied)."""
        return self._layers[layer]

    def get_out_edges(self, layer: int, vertex_index: int) -> Vector:
        """Return outgoing edges of vertex_index as a Vector."""
        mat = self._layers[layer]
        return mat[vertex_index, :]

    def get_in_edges(self, layer: int, vertex_index: int) -> Vector:
        """Return incoming edges of vertex_index as a Vector."""
        mat = self._layers[layer]
        return mat[:, vertex_index]

    # ------------------------------------------------------------------ #
    # Graph view
    # ------------------------------------------------------------------ #

    def get_view(self, mask: GraphMask | None = None) -> Graph:
        return Graph(
            layers=self._layers,
            num_vertices=self.num_vertices,
            scenario_id=self.scenario_id,
            mask=self._mask if mask is None else mask,
            label_matrix=self._label_matrix,
            label_meta=self._label_meta,
            label_type_vectors=self._label_type_vectors,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"Graph(num_vertices={self.num_vertices}, "
            f"scenario_id={self.scenario_id}, "
            f"num_layers={len(self._layers)}, "
            f"num_labels={self.num_labels})"
        )
