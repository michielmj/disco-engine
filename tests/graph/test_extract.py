# tests/graph/test_extract.py
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert,
)
from sqlalchemy.orm import sessionmaker, Session

from disco.graph import Graph
from disco.graph.schema import (
    create_graph_schema,
    vertices,
    edges,
)
from disco.graph.extract import (
    get_vertex_data,
    get_vertex_numeric_vector,
    get_outbound_edge_data,
    get_inbound_edge_data,
    get_outbound_map,
    get_inbound_map,
)
from graphblas import Vector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_model_tables(engine) -> Tuple[Table, Table]:
    """
    Create key-based model node and edge tables for tests:

      node_data:
        - scenario_id, key, value, capacity

      edge_data:
        - scenario_id, layer_idx, source_key, target_key, cost
    """
    metadata = MetaData()

    node_data = Table(
        "node_data",
        metadata,
        Column("scenario_id", String, primary_key=True),
        Column("key", String, primary_key=True),
        Column("value", Float),
        Column("capacity", Float),
    )

    edge_data = Table(
        "edge_data",
        metadata,
        Column("scenario_id", String, primary_key=True),
        Column("layer_idx", Integer, primary_key=True),  # <-- add layer_idx
        Column("source_key", String, primary_key=True),
        Column("target_key", String, primary_key=True),
        Column("cost", Float),
    )

    metadata.create_all(engine)
    return node_data, edge_data


def _populate_vertices_and_edges(
    session: Session,
    scenario_id: str,
) -> None:
    """
    Insert a small mapping of indices->keys and structural edges:

      vertices (scenario_id, index, key, node_type):
        (S1, 0, "A", "test")
        (S1, 1, "B", "test")
        (S1, 2, "C", "test")

      edges (scenario_id, layer_idx=0, source_idx, target_idx, weight):
        (S1, 0, 0 -> 1, weight=1.5)
        (S1, 0, 1 -> 2, weight=2.5)
    """
    session.execute(
        insert(vertices),
        [
            {
                "scenario_id": scenario_id,
                "index": 0,
                "key": "A",
                "node_type": "test",  # <-- required by current schema
            },
            {
                "scenario_id": scenario_id,
                "index": 1,
                "key": "B",
                "node_type": "test",
            },
            {
                "scenario_id": scenario_id,
                "index": 2,
                "key": "C",
                "node_type": "test",
            },
        ],
    )

    session.execute(
        insert(edges),
        [
            {
                "scenario_id": scenario_id,
                "layer_idx": 0,
                "source_idx": 0,
                "target_idx": 1,
                "weight": 1.5,
                "name": "e_0_0_1",
            },
            {
                "scenario_id": scenario_id,
                "layer_idx": 0,
                "source_idx": 1,
                "target_idx": 2,
                "weight": 2.5,
                "name": "e_0_1_2",
            },
        ],
    )


def engine_session_and_model_tables() -> Iterator[
    tuple[object, sessionmaker, Table, Table]
]:
    """
    Helper generator for tests.

    - Creates in-memory SQLite engine
    - Applies schema_translate_map so "graph" schema is stripped
    - Creates graph schema (scenarios, vertices, edges, ...)
    - Creates key-based model node/edge tables
    """
    # Base in-memory SQLite engine
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    # IMPORTANT: strip the "graph" schema for SQLite
    engine = engine.execution_options(schema_translate_map={"graph": None})

    # Now all DDL/DML using schema="graph" becomes plain tables in "main"
    create_graph_schema(engine)

    node_data, edge_data = _create_model_tables(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)

    try:
        yield engine, SessionLocal, node_data, edge_data
    finally:
        engine.dispose()

# ---------------------------------------------------------------------------
# Tests: vertex data
# ---------------------------------------------------------------------------


def test_get_vertex_data_index_and_key() -> None:
    for _, SessionLocal, node_data, _ in engine_session_and_model_tables():
        session: Session = SessionLocal()
        scenario_id = "S1"

        _populate_vertices_and_edges(session, scenario_id)

        # Node data: rows for A and C only, B is missing -> tests default_fill
        session.execute(
            insert(node_data),
            [
                {
                    "scenario_id": scenario_id,
                    "key": "A",
                    "value": 10.0,
                    "capacity": 100.0,
                },
                {
                    "scenario_id": scenario_id,
                    "key": "C",
                    "value": 30.0,
                    "capacity": 300.0,
                },
            ],
        )
        session.commit()

        # Empty structural graph is fine; we just need num_vertices and scenario_id
        graph = Graph(layers={}, num_vertices=3, scenario_id=scenario_id)

        # Index-based
        df_idx = get_vertex_data(
            session,
            graph,
            node_data,
            columns=[node_data.c.value, node_data.c.capacity],
            index_by="index",
            default_fill=0.0,
        )

        assert list(df_idx.index) == [0, 1, 2]
        assert df_idx.loc[0, "value"] == 10.0
        assert df_idx.loc[1, "value"] == 0.0  # missing -> default_fill
        assert df_idx.loc[2, "value"] == 30.0

        # Key-based
        df_key = get_vertex_data(
            session,
            graph,
            node_data,
            columns=[node_data.c.value, node_data.c.capacity],
            index_by="key",
            default_fill=0.0,
        )

        assert list(df_key.index) == ["A", "B", "C"]
        assert df_key.loc["A", "value"] == 10.0
        assert df_key.loc["B", "value"] == 0.0
        assert df_key.loc["C", "value"] == 30.0

        session.close()


def test_get_vertex_numeric_vector() -> None:
    for _, SessionLocal, node_data, _ in engine_session_and_model_tables():
        session: Session = SessionLocal()
        scenario_id = "S1"

        _populate_vertices_and_edges(session, scenario_id)

        # Node data: only A and C have values
        session.execute(
            insert(node_data),
            [
                {"scenario_id": scenario_id, "key": "A", "value": 1.0, "capacity": 10.0},
                {"scenario_id": scenario_id, "key": "C", "value": 3.0, "capacity": 30.0},
            ],
        )
        session.commit()

        graph = Graph(layers={}, num_vertices=3, scenario_id=scenario_id)

        vec = get_vertex_numeric_vector(
            session,
            graph,
            node_data,
            value_column=node_data.c.value,
            default_value=0.0,
        )

        # Expect indices 0 and 2 populated, 1 is implicitly zero
        idx, vals = vec.to_coo()
        assert set(idx.tolist()) == {0, 2}
        assert set(vals.tolist()) == {1.0, 3.0}

        session.close()


# ---------------------------------------------------------------------------
# Tests: edge data via key-based edge table and index->key mapping
# ---------------------------------------------------------------------------


def test_get_outbound_and_inbound_edge_data_indices_and_keys() -> None:
    for _, SessionLocal, node_data, edge_data in engine_session_and_model_tables():
        session: Session = SessionLocal()
        scenario_id = "S1"

        _populate_vertices_and_edges(session, scenario_id)

        # Edge data: cost defined for first edge only; second is missing
        session.execute(
            insert(edge_data),
            [
                {
                    "scenario_id": scenario_id,
                    "layer_idx": 0,
                    "source_key": "A",
                    "target_key": "B",
                    "cost": 5.0,
                },
                # No row for B->C to test missing edge data
            ],
        )
        session.commit()

        # Graph structure (layers) is not used directly by extract edge-data,
        # but we pass an empty dict for completeness.
        graph = Graph(layers={}, num_vertices=3, scenario_id=scenario_id)

        # Outbound edges, indexed by indices
        df_out_idx = get_outbound_edge_data(
            session,
            graph,
            edge_data,
            columns=[edge_data.c.cost],
            layer_idx=0,
            index_by="indices",
            default_fill=0.0,
        )

        # Edges present structurally: (0,1) and (1,2)
        assert sorted(df_out_idx.index.tolist()) == [(0, 1), (1, 2)]
        assert df_out_idx.loc[(0, 1), "cost"] == 5.0
        # Missing edge_data row -> default_fill
        assert df_out_idx.loc[(1, 2), "cost"] == 0.0

        # Outbound edges, indexed by keys
        df_out_keys = get_outbound_edge_data(
            session,
            graph,
            edge_data,
            columns=[edge_data.c.cost],
            layer_idx=0,
            index_by="keys",
            default_fill=0.0,
        )

        assert sorted(df_out_keys.index.tolist()) == [("A", "B"), ("B", "C")]
        assert df_out_keys.loc[("A", "B"), "cost"] == 5.0
        assert df_out_keys.loc[("B", "C"), "cost"] == 0.0

        # Inbound edges, indexed by indices
        df_in_idx = get_inbound_edge_data(
            session,
            graph,
            edge_data,
            columns=[edge_data.c.cost],
            layer_idx=0,
            index_by="indices",
            default_fill=0.0,
        )

        assert sorted(df_in_idx.index.tolist()) == [(0, 1), (1, 2)]
        assert df_in_idx.loc[(0, 1), "cost"] == 5.0
        assert df_in_idx.loc[(1, 2), "cost"] == 0.0

        session.close()


def test_edge_data_with_mask() -> None:
    for _, SessionLocal, node_data, edge_data in engine_session_and_model_tables():
        session: Session = SessionLocal()
        scenario_id = "S1"

        _populate_vertices_and_edges(session, scenario_id)

        # Edge data for both edges to see masking effect
        session.execute(
            insert(edge_data),
            [
                {
                    "scenario_id": scenario_id,
                    "layer_idx": 0,
                    "source_key": "A",
                    "target_key": "B",
                    "cost": 5.0,
                },
                {
                    "scenario_id": scenario_id,
                    "layer_idx": 0,
                    "source_key": "B",
                    "target_key": "C",
                    "cost": 7.0,
                },
            ],
        )
        session.commit()

        graph = Graph(layers={}, num_vertices=3, scenario_id=scenario_id)

        # Mask for source index == 1 => only edge 1->2 should remain outbound
        mask_vec = Vector.from_coo([1], [True], size=3)
        graph.set_mask(mask_vec)

        df_out_masked = get_outbound_edge_data(
            session,
            graph,
            edge_data,
            columns=[edge_data.c.cost],
            layer_idx=0,
            index_by="indices",
        )

        assert list(df_out_masked.index) == [(1, 2)]
        assert df_out_masked.loc[(1, 2), "cost"] == 7.0

        # Mask for target index == 1 => only edge 0->1 should remain inbound
        mask_vec_in = Vector.from_coo([1], [True], size=3)
        graph.set_mask(mask_vec_in)

        df_in_masked = get_inbound_edge_data(
            session,
            graph,
            edge_data,
            columns=[edge_data.c.cost],
            layer_idx=0,
            index_by="indices",
        )

        assert list(df_in_masked.index) == [(0, 1)]
        assert df_in_masked.loc[(0, 1), "cost"] == 5.0

        session.close()


# ---------------------------------------------------------------------------
# Tests: structural maps (index-based, weight-only)
# ---------------------------------------------------------------------------


def test_outbound_and_inbound_map() -> None:
    for _, SessionLocal, node_data, _ in engine_session_and_model_tables():
        session: Session = SessionLocal()
        scenario_id = "S1"

        _populate_vertices_and_edges(session, scenario_id)
        session.commit()

        # Build a Graph that matches the structural edges (0->1, 1->2)
        src = np.array([0, 1], dtype=np.int64)
        tgt = np.array([1, 2], dtype=np.int64)
        wgt = np.array([1.5, 2.5], dtype=np.float64)
        layers = {0: (src, tgt, wgt)}
        graph = Graph.from_edges(layers, num_vertices=3, scenario_id=scenario_id)

        out_mat = get_outbound_map(session, graph, layer_idx=0)
        in_mat = get_inbound_map(session, graph, layer_idx=0)

        # Check basic structure: non-zero entries in the right places
        rows, cols, vals = out_mat.to_coo()
        entries = {(int(r), int(c)): float(v) for r, c, v in zip(rows, cols, vals)}
        assert entries.get((0, 1)) == 1.5
        assert entries.get((1, 2)) == 2.5

        rows_i, cols_i, vals_i = in_mat.to_coo()
        entries_i = {(int(r), int(c)): float(v) for r, c, v in zip(rows_i, cols_i, vals_i)}
        assert entries_i.get((0, 1)) == 1.5
        assert entries_i.get((1, 2)) == 2.5

        session.close()
