# src/disco/graph/core.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Sequence, Mapping, cast
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
          * per-type metadata for fast access:
                - _labels_by_type: label_type -> {label_value -> label_index}
                - _label_indices_by_type: label_type -> np.ndarray[int64] of label indices
                - _label_index_by_type_value: (label_type, label_value) -> label_index
    """

    __slots__ = (
        "_layers",
        "num_vertices",
        "scenario_id",
        "_vertices",                   # np.ndarray | None  (index i -> vertex key)
        "_mask",                       # GraphMask | None
        "_label_matrix",               # Matrix[BOOL] | None
        "_label_meta",                 # dict[int, tuple[str, str]]
        "_labels_by_type",             # dict[str, dict[str, int]]
        "_label_indices_by_type",      # dict[str, np.ndarray]
        "_label_index_by_type_value",  # dict[tuple[str, str], int]
        "_num_labels",
    )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        layers: Tuple[Matrix, ...],
        num_vertices: int,
        scenario_id: str = "",
        *,
        vertices: Optional[np.ndarray] = None,
        mask: Optional[Vector] = None,
        label_matrix: Optional[Matrix] = None,
        label_meta: Optional[Dict[int, Tuple[str, str]]] = None,
    ) -> None:
        # Avoid unnecessary copies; only wrap if needed
        self._layers: Tuple[Matrix, ...] = layers

        self.num_vertices = num_vertices
        self.scenario_id = scenario_id

        # vertices: 1D array mapping vertex index -> vertex key
        self._vertices: Optional[np.ndarray] = (
            np.asarray(vertices) if vertices is not None else None
        )

        self.validate(check_cycles=False)

        # mask (GraphMask, internal)
        self._mask: Optional[GraphMask] = None
        if mask is not None:
            self.set_mask(mask)

        # labels
        self._label_matrix: Optional[Matrix] = None
        self._label_meta: Dict[int, Tuple[str, str]] = {}
        self._labels_by_type: Dict[str, Dict[str, int]] = {}
        self._label_indices_by_type: Dict[str, np.ndarray] = {}
        self._label_index_by_type_value: Dict[tuple[str, str], int] = {}
        self._num_labels: int = 0
        if label_matrix is not None:
            self.set_labels(label_matrix, label_meta)

    @classmethod
    def from_edges(
        cls,
        edge_layers: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        num_vertices: int,
        scenario_id: str = "",
        vertices: Optional[np.ndarray] = None,
    ) -> Graph:
        """
        Build a Graph from per-layer edge arrays.

        edge_layers:
            mapping layer_idx -> (source_vertex_indices, target_vertex_indices, weights)
        num_vertices:
            total number of vertices (0..num_vertices-1)
        vertices:
            optional 1D array of vertex keys, where position i is the key for vertex i
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
                raise ValueError("Layer indices must be contiguous.")

        layers = tuple(layers_dict[i] for i in range(num_layers))

        return cls(
            layers=layers,
            num_vertices=num_vertices,
            scenario_id=scenario_id,
            vertices=vertices,
        )

    def validate(self, check_cycles: bool = True) -> None:
        if not self.scenario_id:
            raise ValueError("Graph must have a non-empty scenario_id.")

        if self._vertices is not None and len(self._vertices) != self.num_vertices:
            raise ValueError(
                f"vertices array length ({len(self._vertices)}) must match "
                f"num_vertices ({self.num_vertices})"
            )

        for idx, layer in enumerate(self._layers):
            if layer.ncols != self.num_vertices:
                raise ValueError(f"Matrix layer {idx} has invalid number of columns.")
            if layer.nrows != self.num_vertices:
                raise ValueError(f"Matrix layer {idx} has invalid number of rows.")
            if has_self_loop(layer):
                raise ValueError(f"Matrix layer {idx} has self loops.")
            if check_cycles:
                if matrix_has_cycle(layer):
                    raise ValueError(f"Matrix layer {idx} appears to have cycles.")

    # ------------------------------------------------------------------ #
    # Vertices
    # ------------------------------------------------------------------ #
    @property
    def vertices(self) -> Optional[np.ndarray]:
        """
        1D array of vertex keys (position i is the key for vertex i), or None
        if no vertex keys have been attached to this graph.
        """
        return self._vertices

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

    # ------------------------------------------------------------------ #
    # Label handling
    # ------------------------------------------------------------------ #
    def _rebuild_per_type_structures(self) -> None:
        """
        Rebuild per-type metadata from self._label_meta.
        """
        self._labels_by_type.clear()
        self._label_indices_by_type.clear()
        self._label_index_by_type_value.clear()

        for idx, (lt, val) in self._label_meta.items():
            self._label_index_by_type_value[(lt, val)] = idx
            d = self._labels_by_type.setdefault(lt, {})
            d[val] = idx

        for lt, d in self._labels_by_type.items():
            arr = np.fromiter(d.values(), dtype=np.int64)
            arr.sort()
            self._label_indices_by_type[lt] = arr

    def set_labels(
        self,
        label_matrix: Matrix,
        label_meta: Optional[Dict[int, Tuple[str, str]]] = None,
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
        self._num_labels = label_matrix.ncols

        # Global mapping index -> (type, value)
        self._label_meta = label_meta if label_meta is not None else {}

        # Rebuild per-type metadata
        self._rebuild_per_type_structures()

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
    def num_labels(self) -> int:
        """
        Number of labels.
        """
        return self._num_labels

    # ------------------------------ #
    # Label metadata helpers
    # ------------------------------ #

    @property
    def label_indices_by_type(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(self._label_indices_by_type)

    def labels_for_type(self, label_type: str) -> Tuple[np.ndarray, Matrix]:
        """
        Return (label_indices, submatrix) for a label type.

        - label_indices: np.ndarray[int64] of global label indices
        - submatrix: Matrix[BOOL] with shape (num_vertices, len(label_indices))
        """
        if self._label_matrix is None:
            raise RuntimeError("Graph has no labels attached (label_matrix is None)")

        try:
            idxs = self._label_indices_by_type[label_type]
        except KeyError:
            raise KeyError(f"Unknown label_type {label_type!r}") from None

        return idxs, self._label_matrix[:, idxs]

    def label_value_to_index(self, label_type: str) -> Mapping[str, int]:
        """
        Read-only mapping label_value -> label_index for a given label_type.
        """
        mapping = self._labels_by_type.get(label_type)
        if mapping is None:
            raise KeyError(f"Unknown label_type {label_type!r}")
        return MappingProxyType(mapping)

    def label_index_to_value_for_type(self, label_type: str) -> Mapping[int, str]:
        """
        Read-only mapping label_index -> label_value for a given label_type.
        """
        mapping = self._labels_by_type.get(label_type)
        if mapping is None:
            raise KeyError(f"Unknown label_type {label_type!r}")
        inverse = {idx: label for label, idx in mapping.items()}
        return MappingProxyType(inverse)

    def label_info(self, label_index: int) -> Tuple[str, str]:
        """
        Return (label_type, label_value) for a global label index.
        """
        try:
            return self._label_meta[label_index]
        except KeyError:
            raise IndexError(f"label_index {label_index} not found in label_meta") from None

    # ------------------------------ #
    # Label-based masks
    # ------------------------------ #
    def get_vertices_for_label(self, label_id: int) -> np.ndarray:
        """
        Return an array of indices indicating which vertices
        have the given label (by label id 0..num_labels-1).
        """
        if self._label_matrix is None:
            raise RuntimeError("Graph has no labels attached (label_matrix is None)")
        if not (0 <= label_id < self._num_labels):
            raise IndexError(f"label_id {label_id} out of range [0, {self._num_labels})")
        idxs, _ = self._label_matrix[:, int(label_id)].select("!=", 0).new().to_coo()

        return cast(np.ndarray, idxs)

    # ------------------------------ #
    # Incrementally adding labels
    # ------------------------------ #
    def add_labels(
        self,
        label_type: str,
        labels: Mapping[str, np.ndarray | Sequence[int]],
    ) -> None:
        """
        Add or update labels for a given label type.

        Parameters
        ----------
        label_type:
            The label type name (e.g. "echelon", "location").
        labels:
            Mapping from label_value (string) to a sequence/array of vertex indices
            that should receive that label. Assignments are OR'ed into the existing
            label matrix (i.e., union).
        """
        if self.num_vertices == 0:
            # Nothing to do; but we keep semantics simple and just return.
            return

        if self._label_matrix is None:
            # Start from an empty matrix
            base_rows = np.empty(0, dtype=np.int64)
            base_cols = np.empty(0, dtype=np.int64)
            base_vals = np.empty(0, dtype=bool)
        else:
            base_rows, base_cols, base_vals = self._label_matrix.to_coo()
            base_rows = base_rows.astype(np.int64, copy=False)
            base_cols = base_cols.astype(np.int64, copy=False)
            base_vals = base_vals.astype(bool, copy=False)

        new_rows_list: List[np.ndarray] = []
        new_cols_list: List[np.ndarray] = []
        affected_types: set[str] = set()

        # Ensure intermediate per-type structures exist
        if label_type not in self._labels_by_type:
            self._labels_by_type[label_type] = {}

        for value, verts in labels.items():
            verts_arr = np.asarray(verts, dtype=np.int64)
            if verts_arr.size == 0:
                continue

            if (verts_arr < 0).any() or (verts_arr >= self.num_vertices).any():
                raise IndexError(
                    f"Vertex indices for label {label_type!r}:{value!r} "
                    f"must be in [0, {self.num_vertices})"
                )

            key = (label_type, value)
            if key in self._label_index_by_type_value:
                idx = self._label_index_by_type_value[key]
            else:
                idx = self._num_labels
                self._num_labels += 1
                self._label_meta[idx] = (label_type, value)
                self._label_index_by_type_value[key] = idx
                self._labels_by_type[label_type][value] = idx
                affected_types.add(label_type)

            col_arr = np.full(verts_arr.shape, idx, dtype=np.int64)
            new_rows_list.append(verts_arr)
            new_cols_list.append(col_arr)

        if not new_rows_list and self._label_matrix is not None:
            # No new assignments and we already had a matrix; metadata may still
            # have been updated with new labels that have no vertices.
            # Rebuild per-type indices for affected types (if any).
            for lt in affected_types:
                d = self._labels_by_type[lt]
                arr = np.fromiter(d.values(), dtype=np.int64)
                arr.sort()
                self._label_indices_by_type[lt] = arr
            return

        if not new_rows_list and self._label_matrix is None:
            # No labels at all
            self._label_matrix = gb.Matrix.from_coo(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=bool),
                nrows=self.num_vertices,
                ncols=self._num_labels,
            )
        else:
            new_rows = np.concatenate(new_rows_list) if new_rows_list else np.empty(0, dtype=np.int64)
            new_cols = np.concatenate(new_cols_list) if new_cols_list else np.empty(0, dtype=np.int64)
            new_vals = np.ones(new_rows.size, dtype=bool)

            all_rows = np.concatenate([base_rows, new_rows])
            all_cols = np.concatenate([base_cols, new_cols])
            all_vals = np.concatenate([base_vals, new_vals])

            # IMPORTANT: allow duplicate (row, col) and OR them together
            self._label_matrix = gb.Matrix.from_coo(
                all_rows,
                all_cols,
                all_vals,
                nrows=self.num_vertices,
                ncols=self._num_labels,
                dup_op=gb.binary.lor,
            )

        # Rebuild per-type indices for affected label_type (and ensure entry exists)
        for lt in affected_types or {label_type}:
            d = self._labels_by_type.get(lt, {})
            arr = np.fromiter(d.values(), dtype=np.int64) if d else np.empty(0, dtype=np.int64)
            if arr.size:
                arr.sort()
            self._label_indices_by_type[lt] = arr

    # ------------------------------------------------------------------ #
    # Distinct label combinations
    # ------------------------------------------------------------------ #
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

        left = A.kronecker(true_b, "land")   # (v x a*b)
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

    @classmethod
    def _augment_with_unlabelled(
        cls, sub_mat: Matrix, num_vertices: int,
    ) -> Tuple[Matrix, bool]:
        """
        Augment a per-type label sub-matrix with an extra "unlabelled" column.

        The extra column (at index ``sub_mat.ncols``) is ``True`` for every
        vertex that has **no** label of this type (i.e. its row in *sub_mat*
        is entirely empty).

        Returns ``(augmented_matrix, has_unlabelled)`` where *has_unlabelled*
        indicates whether any unlabelled vertices were found.  When all
        vertices carry at least one label of this type the original matrix
        is returned unchanged to avoid unnecessary copies.
        """
        has_label: Vector = sub_mat.reduce_rowwise("lor").new()
        labelled_indices, _ = has_label.to_coo()

        if labelled_indices.size == num_vertices:
            # Every vertex is labelled for this type — nothing to augment.
            return sub_mat, False

        # Identify unlabelled vertices.
        unlabelled_mask = np.ones(num_vertices, dtype=bool)
        unlabelled_mask[labelled_indices] = False
        unlabelled_indices = np.nonzero(unlabelled_mask)[0].astype(np.int64)

        # Build augmented matrix: original columns + one extra column.
        rows, cols, vals = sub_mat.to_coo()
        rows = rows.astype(np.int64, copy=False)
        cols = cols.astype(np.int64, copy=False)
        vals = vals.astype(bool, copy=False)

        extra_col = np.full(unlabelled_indices.size, sub_mat.ncols, dtype=np.int64)
        extra_vals = np.ones(unlabelled_indices.size, dtype=bool)

        aug_mat = gb.Matrix.from_coo(
            np.concatenate([rows, unlabelled_indices]),
            np.concatenate([cols, extra_col]),
            np.concatenate([vals, extra_vals]),
            nrows=num_vertices,
            ncols=sub_mat.ncols + 1,
            dtype=gb.dtypes.BOOL,
        )
        return aug_mat, True

    def by_distinct_labels(self, distinct: Sequence[str]) -> List[np.ndarray]:
        """
        Compute distinct combinations of labels across one or more label types.

        Vertices that lack a label for one or more of the requested types are
        still included: the missing type is simply omitted from that combo.
        Because global label indices are unique across types, each combo is
        unambiguous regardless of length.

        Returns
        -------
        list[np.ndarray]
            Each array contains the *global* label indices (one per label type
            that is present on the supporting vertices) for which at least one
            vertex exists carrying exactly that combination.  Arrays may be
            shorter than ``len(distinct)`` when some vertices lack labels for
            one or more of the requested types, and may be empty if vertices
            exist that carry none of the requested label types.

        If a vertex has at most one label per label type, no vertex occurs in
        more than one combination.
        """
        if self._label_matrix is None or self._num_labels == 0:
            return []

        # Per label type: (global_label_indices, original_width, augmented_matrix)
        label_indices_list: List[np.ndarray] = []
        original_widths: List[int] = []
        augmented_matrices: List[Matrix] = []

        for label_type in distinct:
            idxs, sub_mat = self.labels_for_type(label_type)
            label_indices_list.append(idxs)
            original_widths.append(int(sub_mat.ncols))

            aug_mat, _ = self._augment_with_unlabelled(sub_mat, self.num_vertices)
            augmented_matrices.append(aug_mat)

        # Row-wise AND Kronecker product across all (augmented) type matrices.
        # S has shape (num_vertices × product_of_augmented_widths).
        S: Matrix = self._rowwise_and_kron_many(*augmented_matrices)

        # Determine which combination-columns have at least one supporting vertex.
        combo_support: Vector = S.reduce_columnwise("lor")
        combo_indices, combo_vals = combo_support.to_coo()
        combo_indices = combo_indices[combo_vals]

        if combo_indices.size == 0:
            return []

        # Decode combination indices back into per-type global label indices,
        # using -1 as a sentinel for the unlabelled column.
        rows_per_type: List[np.ndarray] = []
        base = int(S.ncols)
        for label_indices, aug_mat, orig_width in zip(
            label_indices_list, augmented_matrices, original_widths,
        ):
            local = combo_indices % base
            base //= int(aug_mat.ncols)
            local = local // base

            # local < orig_width  → real label;  local == orig_width → unlabelled
            global_indices = np.full(combo_indices.shape, -1, dtype=np.int64)
            real_mask = local < orig_width
            if real_mask.any():
                global_indices[real_mask] = label_indices[local[real_mask]]
            rows_per_type.append(global_indices)

        result = np.stack(rows_per_type, axis=1)
        return [r[r >= 0] for r in result]

    # ------------------------------------------------------------------ #
    # Structural accessors
    # ------------------------------------------------------------------ #
    @property
    def layers(self) -> Tuple[Matrix, ...]:
        """
        Read-only view of the internal layer sequence.
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
    def get_view(self, mask: Vector | None = None) -> Graph:
        """
        Return a shallow view of this graph sharing structure and labels,
        but with an optional different GraphMask.
        """
        mask_vec: Vector | None
        if mask is None:
            mask_vec = self._mask.vector if self._mask is not None else None
        else:
            mask_vec = mask

        return Graph(
            layers=self._layers,
            num_vertices=self.num_vertices,
            scenario_id=self.scenario_id,
            vertices=self._vertices,
            mask=mask_vec,
            label_matrix=self._label_matrix,
            label_meta=self._label_meta,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"Graph(num_vertices={self.num_vertices}, "
            f"scenario_id={self.scenario_id}, "
            f"num_layers={len(self._layers)}, "
            f"num_labels={self._num_labels})"
        )
