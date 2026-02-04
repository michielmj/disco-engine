# src/disco/graph/db.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import graphblas as gb
from sqlalchemy import select, insert, delete, func, literal
from sqlalchemy.orm import Session

from .core import Graph
from .schema import (
    scenarios,
    vertices,
    edges,
    labels,
    vertex_labels,
)


# ---------------------------------------------------------------------------
# Scenario management
# ---------------------------------------------------------------------------


def create_scenario(
    session: Session,
    scenario_id: str,
    *,
    base_scenario_id: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Insert a new scenario row into graph.scenarios and return its scenario_id.

    Assumes the graph.scenarios table has at least:
        - scenario_id (str, primary or unique key)
        - created_at (datetime)
        - base_scenario_id (optional str, FK to scenarios.scenario_id)
        - description (optional str)
    """
    now = datetime.utcnow()

    session.execute(
        insert(scenarios).values(
            scenario_id=scenario_id,
            created_at=now,
            base_scenario_id=base_scenario_id,
            description=description,
        )
    )
    # We treat `scenario_id` (string) as the external identifier everywhere
    return scenario_id


# ---------------------------------------------------------------------------
# Store Graph -> DB (edges + labels)
# ---------------------------------------------------------------------------


def _store_edges_for_scenario(session: Session, graph: Graph) -> None:
    """
    Persist the Graph's structural edges into graph.edges for its scenario.

    - Removes any existing edges for the scenario.
    - Writes rows per layer using Matrix.to_coo().
    """
    scenario_id = graph.scenario_id

    # Remove previous edges for this scenario
    session.execute(
        delete(edges).where(edges.c.scenario_id == literal(scenario_id))
    )

    # Insert new edges per layer
    for layer_idx, mat in enumerate(graph.layers):
        rows, cols, vals = mat.to_coo()
        if len(rows) == 0:
            continue

        rows_to_insert = [
            {
                "scenario_id": scenario_id,
                "layer_idx": int(layer_idx),
                "source_idx": int(r),
                "target_idx": int(c),
                "weight": float(v),
                "name": f"e_{layer_idx}_{int(r)}_{int(c)}",
            }
            for r, c, v in zip(rows, cols, vals)
        ]
        session.execute(insert(edges), rows_to_insert)


def _store_labels_for_scenario(session: Session, graph: Graph) -> None:
    """
    Persist the Graph's label structure into graph.labels and graph.vertex_labels.

    Strategy:
      - If the Graph has no labels (label_matrix is None), do nothing.
      - Otherwise:
          * Delete existing vertex_labels and labels for this scenario.
          * Insert labels based on Graph.label_meta (per label index).
          * Insert vertex_labels based on Graph.label_matrix.to_coo().
    """
    label_matrix = graph.label_matrix
    if label_matrix is None or graph.num_labels == 0:
        # Nothing to store
        return

    scenario_id = graph.scenario_id

    # Clear existing labels and assignments for this scenario
    session.execute(
        delete(vertex_labels).where(vertex_labels.c.scenario_id == literal(scenario_id))
    )
    session.execute(
        delete(labels).where(labels.c.scenario_id == literal(scenario_id))
    )

    # Insert labels from Graph.label_meta; ensure we have metadata for each index
    label_meta = graph.label_meta  # mapping index -> (type, value)
    label_index_to_db_id: Dict[int, int] = {}

    for lbl_idx in range(graph.num_labels):
        if lbl_idx not in label_meta:
            raise ValueError(
                f"Graph.label_meta missing entry for label index {lbl_idx}"
            )
        label_type, label_value = label_meta[lbl_idx]
        result = session.execute(
            insert(labels)
            .values(
                scenario_id=scenario_id,
                type=label_type,
                value=label_value,
            )
            .returning(labels.c.id)
        )
        label_index_to_db_id[lbl_idx] = int(result.scalar_one())

    # Insert vertex_labels based on the boolean label_matrix
    rows, cols, vals = label_matrix.to_coo()
    if len(rows) == 0:
        return

    assignments = []
    for v_idx, lbl_idx, val in zip(rows, cols, vals):
        if not bool(val):
            continue
        db_label_id = label_index_to_db_id[int(lbl_idx)]
        assignments.append(
            {
                "scenario_id": scenario_id,
                "vertex_index": int(v_idx),
                "label_id": db_label_id,
            }
        )

    if assignments:
        session.execute(insert(vertex_labels), assignments)


def store_graph(
    session: Session,
    graph: Graph,
    *,
    store_edges: bool = True,
    store_labels: bool = True,
    # store_node_types: bool = True,
) -> None:
    """
    Persist the Graph structure into the graph schema.

    - By default, stores edges, labels, and node types.
    - Use store_edges / store_labels / store_node_types flags if you want to
      control which parts are written.

    NOTE: This does not manage vertices or scenarios; those should be created
    separately (e.g. via create_scenario and inserts into graph.vertices).
    """
    if store_edges:
        _store_edges_for_scenario(session, graph)
    if store_labels:
        _store_labels_for_scenario(session, graph)


# ---------------------------------------------------------------------------
# Load DB -> Graph (edges + labels)
# ---------------------------------------------------------------------------


def _load_num_vertices(session: Session, scenario_id: str) -> int:
    """
    Infer the number of vertices for a scenario from graph.vertices.

    Uses the naming convention:
      - For vertex rows (graph.vertices), the index column is called 'index'.
    """
    max_idx = session.execute(
        select(func.max(vertices.c.index)).where(
            vertices.c.scenario_id == literal(scenario_id)
        )
    ).scalar_one()
    return int(max_idx) + 1 if max_idx is not None else 0


def _load_edge_layers(
    session: Session,
    scenario_id: str,
    num_vertices: int,
) -> Tuple[gb.Matrix, ...]:
    """
    Load edges for a scenario and build one GraphBLAS Matrix per layer.

    Returns:
        Tuple[Matrix, ...] where each Matrix has shape
        (num_vertices, num_vertices) and contains the edge weights.
    """
    result = session.execute(
        select(
            edges.c.layer_idx,
            edges.c.source_idx,
            edges.c.target_idx,
            edges.c.weight,
        ).where(edges.c.scenario_id == literal(scenario_id))
    )

    layer_sources: Dict[int, list[int]] = {}
    layer_targets: Dict[int, list[int]] = {}
    layer_weights: Dict[int, list[float]] = {}

    for layer_idx, src, tgt, w in result:
        lid = int(layer_idx)
        layer_sources.setdefault(lid, []).append(int(src))
        layer_targets.setdefault(lid, []).append(int(tgt))
        layer_weights.setdefault(lid, []).append(float(w))

    edge_layers: Dict[int, gb.Matrix] = {}
    for layer_idx, src_list in layer_sources.items():
        tgt_list = layer_targets[layer_idx]
        w_list = layer_weights[layer_idx]

        if not src_list:
            # No edges for this layer; skip
            continue

        src_arr = np.asarray(src_list, dtype=np.int64)
        tgt_arr = np.asarray(tgt_list, dtype=np.int64)
        w_arr = np.asarray(w_list, dtype=np.float64)

        mat = gb.Matrix.from_coo(
            src_arr,
            tgt_arr,
            w_arr,
            nrows=num_vertices,
            ncols=num_vertices,
        )
        edge_layers[layer_idx] = mat

    num_layers = len(edge_layers)
    for idx in edge_layers:
        if idx < 0 or idx >= num_layers:
            raise ValueError('Layer indices must be contiguous.')

    layers = tuple(edge_layers[i] for i in range(num_layers))

    return layers


def _load_labels_for_scenario(
    session: Session,
    scenario_id: str,
    num_vertices: int,
) -> tuple[Optional[gb.Matrix], Dict[int, tuple[str, str]], Dict[str, gb.Vector]]:
    """
    Load labels and vertex_labels for a scenario and build:

      - label_matrix: Matrix[BOOL] of shape (num_vertices, num_labels) or None
      - label_meta: label_index -> (label_type, label_value)
      - label_type_vectors: label_type -> Vector[BOOL] over label ids

    Global label ids 0..num_labels-1 are assigned based on the order of rows
    in graph.labels (sorted by id for determinism).
    """
    import graphblas as gb  # local import to avoid cycles

    # Fetch all labels for this scenario
    label_rows = session.execute(
        select(labels.c.id, labels.c.type, labels.c.value).where(
            labels.c.scenario_id == literal(scenario_id)
        )
    ).all()

    if not label_rows:
        # No labels defined
        return None, {}, {}

    # Sort by DB id to get a deterministic label index order
    label_rows_sorted = sorted(label_rows, key=lambda r: int(r[0]))

    label_id_to_index: Dict[int, int] = {}
    label_meta: Dict[int, tuple[str, str]] = {}
    type_to_label_indices: Dict[str, list[int]] = {}

    for idx, (db_id, ltype, lvalue) in enumerate(label_rows_sorted):
        db_id_int = int(db_id)
        label_id_to_index[db_id_int] = idx
        label_meta[idx] = (str(ltype), str(lvalue))
        type_to_label_indices.setdefault(str(ltype), []).append(idx)

    num_labels = len(label_rows_sorted)

    # Build label_type_vectors: type -> Vector[BOOL] over label indices
    label_type_vectors: Dict[str, gb.Vector] = {}
    for ltype, indices in type_to_label_indices.items():
        if not indices:
            continue
        idx_arr = np.asarray(indices, dtype=np.int64)
        val_arr = np.ones(len(indices), dtype=bool)
        vec = gb.Vector.from_coo(
            idx_arr,
            val_arr,
            gb.dtypes.BOOL,
            size=num_labels,
        )
        label_type_vectors[ltype] = vec

    # Fetch vertex-label assignments
    vl_rows = session.execute(
        select(vertex_labels.c.vertex_index, vertex_labels.c.label_id).where(
            vertex_labels.c.scenario_id == literal(scenario_id)
        )
    ).all()

    if not vl_rows:
        # No assignments; return empty matrix
        label_matrix = gb.Matrix.sparse(
            gb.dtypes.BOOL, num_vertices, num_labels
        )
        return label_matrix, label_meta, label_type_vectors

    v_indices: list[int] = []
    l_indices: list[int] = []

    for v_idx, db_label_id in vl_rows:
        v = int(v_idx)
        db_lid = int(db_label_id)
        if db_lid not in label_id_to_index:
            # Label assignment referencing non-existing label row;
            # skip defensively.
            continue
        lidx = label_id_to_index[db_lid]
        v_indices.append(v)
        l_indices.append(lidx)

    if not v_indices:
        label_matrix = gb.Matrix.sparse(
            gb.dtypes.BOOL, num_vertices, num_labels
        )
        return label_matrix, label_meta, label_type_vectors

    v_arr = np.asarray(v_indices, dtype=np.int64)
    l_arr = np.asarray(l_indices, dtype=np.int64)
    vals = np.ones(len(v_indices), dtype=bool)

    label_matrix = gb.Matrix.from_coo(
        v_arr,
        l_arr,
        vals,
        nrows=num_vertices,
        ncols=num_labels,
    )

    return label_matrix, label_meta, label_type_vectors


def load_graph_for_scenario(
    session: Session,
    scenario_id: str,
) -> Graph:
    """
    Load the full Graph (edges + labels) for a scenario.

    - Vertices: inferred from graph.vertices (max index + 1).
    - Edges: from graph.edges.
    - Labels: from graph.labels and graph.vertex_labels, assembled into
      Graph.label_matrix, Graph.label_meta, and Graph.label_type_vectors.
    - Node types: from graph.vertices.node_type, assembled into
      Graph.node_type_matrix and Graph.node_type_meta.
    """
    num_vertices = _load_num_vertices(session, scenario_id)
    edge_layers = _load_edge_layers(session, scenario_id, num_vertices)
    label_matrix, label_meta, label_type_vectors = _load_labels_for_scenario(
        session, scenario_id, num_vertices
    )

    return Graph(
        layers=edge_layers,
        num_vertices=num_vertices,
        scenario_id=scenario_id,
        label_matrix=label_matrix,
        label_meta=label_meta,
        label_type_vectors=label_type_vectors,
    )
