# tests/graph/test_extract.py
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from graphblas import Vector
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)
from sqlalchemy.orm import Session, sessionmaker

from disco.graph import Graph
from disco.graph.extract import (
    get_inbound_edge_data,
    get_inbound_map,
    get_outbound_edge_data,
    get_outbound_map,
    get_vertex_data,
    get_vertex_numeric_vector,
)
from disco.graph.schema import (
    create_graph_schema,
    edges,
    vertices,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_model_tables(engine) -> tuple[Table, Table]:
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

      graph_vertices (scenario_id, index, key):
        (S1, 0, "A")
        (S1, 1, "B")
        (S1, 2, "C")

      graph_edges (scenario_id, layer_idx=0, source_idx, target_idx, weight):
        (S1, 0, 0 -> 1, weight=1.5)
        (S1, 0, 1 -> 2, weight=2.5)
    """
    session.execute(
        insert(vertices),
        [
            {"scenario_id": scenario_id, "index": 0, "key": "A"},
            {"scenario_id": scenario_id, "index": 1, "key": "B"},
            {"scenario_id": scenario_id, "index": 2, "key": "C"},
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
            },
            {
                "scenario_id": scenario_id,
                "layer_idx": 0,
                "source_idx": 1,
                "target_idx": 2,
                "weight": 2.5,
            },
        ],
    )


def engine_session_and_model_tables() -> Iterator[tuple[object, sessionmaker, Table, Table]]:
    """
    Helper generator for tests.

    - Creates in-memory SQLite engine
    - Creates graph tables (graph_scenarios, graph_vertices, graph_edges, ...)
    - Creates key-based model node/edge tables
    """
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

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
    for _, SessionLocal, _node_data, edge_data in engine_session_and_model_tables():
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
    for _, SessionLocal, _node_data, edge_data in engine_session_and_model_tables():
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
    for _, SessionLocal, _node_data, _ in engine_session_and_model_tables():
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

        out_mat = get_outbound_map(session, graph, layer_idx=0, values=edges.c.weight)
        in_mat = get_inbound_map(session, graph, layer_idx=0, values=edges.c.weight)

        # Check basic structure: non-zero entries in the right places
        rows, cols, vals = out_mat.to_coo()
        entries = {(int(r), int(c)): float(v) for r, c, v in zip(rows, cols, vals, strict=True)}
        assert entries.get((0, 1)) == 1.5
        assert entries.get((1, 2)) == 2.5

        rows_i, cols_i, vals_i = in_mat.to_coo()
        entries_i = {(int(r), int(c)): float(v) for r, c, v in zip(rows_i, cols_i, vals_i, strict=True)}
        assert entries_i.get((0, 1)) == 1.5
        assert entries_i.get((1, 2)) == 2.5

        session.close()


# ---------------------------------------------------------------------------
# Tests: default_fill on text columns (pandas 4 raises on a fill of another type)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("error::pandas.errors.Pandas4Warning")
def test_default_fill_of_text_columns_warns_nothing_and_keeps_the_values() -> None:
    for engine, SessionLocal, _node_data, _edge_data in engine_session_and_model_tables():
        metadata = MetaData()
        texts = Table(
            "texts",
            metadata,
            Column("scenario_id", String, primary_key=True),
            Column("key", String, primary_key=True),
            Column("name", String),  # some missing
            Column("note", String),  # all missing
            Column("value", Float),
        )
        edge_texts = Table(
            "edge_texts",
            metadata,
            Column("scenario_id", String, primary_key=True),
            Column("source_key", String, primary_key=True),
            Column("target_key", String, primary_key=True),
            Column("name", String),
            Column("note", String),
        )
        metadata.create_all(engine)

        session: Session = SessionLocal()
        _populate_vertices_and_edges(session, "S1")
        session.execute(
            insert(texts), [{"scenario_id": "S1", "key": "A", "name": "x", "note": None, "value": 1.5}]
        )
        session.execute(
            insert(edge_texts),
            [{"scenario_id": "S1", "source_key": "A", "target_key": "B", "name": "e", "note": None}],
        )
        session.commit()
        graph = Graph(layers={}, num_vertices=3, scenario_id="S1")

        df = get_vertex_data(
            session, graph, texts, [texts.c.name, texts.c.note, texts.c.value], default_fill=0
        )
        assert df["name"].tolist() == ["x", 0, 0]
        assert df["note"].tolist() == [0, 0, 0]
        assert df["value"].tolist() == [1.5, 0.0, 0.0]
        assert df["value"].dtype == np.float64

        for fn in (get_outbound_edge_data, get_inbound_edge_data):
            edf = fn(
                session,
                graph,
                edge_texts,
                [edge_texts.c.name, edge_texts.c.note],
                layer_idx=0,
                default_fill=0,
            )
            assert edf["name"].tolist() == ["e", 0]
            assert edf["note"].tolist() == [0, 0]

        session.close()


# ---------------------------------------------------------------------------
# Tests: a column without missing values is left as it is
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("error::pandas.errors.Pandas4Warning")
@pytest.mark.parametrize("fill", [0, ""])
def test_default_fill_leaves_full_columns_alone(fill: object) -> None:
    for engine, SessionLocal, _node_data, _edge_data in engine_session_and_model_tables():
        metadata = MetaData()
        full = Table(
            "full",
            metadata,
            Column("scenario_id", String, primary_key=True),
            Column("key", String, primary_key=True),
            Column("name", String),
            Column("count", Integer),
            Column("value", Float),
        )
        metadata.create_all(engine)
        session: Session = SessionLocal()
        _populate_vertices_and_edges(session, "S1")
        session.execute(
            insert(full),
            [
                {"scenario_id": "S1", "key": k, "name": k.lower(), "count": i, "value": i + 0.5}
                for i, k in enumerate("ABC")
            ],
        )
        session.commit()
        graph = Graph(layers={}, num_vertices=3, scenario_id="S1")
        cols = [full.c.name, full.c.count, full.c.value]

        filled = get_vertex_data(session, graph, full, cols, default_fill=fill)
        plain = get_vertex_data(session, graph, full, cols)

        pd.testing.assert_frame_equal(filled, plain)
        assert [str(t) for t in filled.dtypes] == [str(t) for t in plain.dtypes]
        session.close()


@pytest.mark.filterwarnings("error::pandas.errors.Pandas4Warning")
def test_default_fill_of_a_datetime_column_warns_nothing() -> None:
    from disco.graph.extract import _fill_missing

    df = pd.DataFrame({"t": pd.to_datetime(["2020-01-01", None])})
    out = _fill_missing(df, 0)
    assert out["t"].tolist() == [pd.Timestamp("2020-01-01"), 0]
    full = pd.DataFrame({"t": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    assert _fill_missing(full, 0)["t"].dtype == full["t"].dtype


@pytest.mark.filterwarnings("error::pandas.errors.Pandas4Warning")
@pytest.mark.parametrize("tz", [None, "UTC"])
def test_default_fill_of_a_datetime_column_with_an_empty_string_keeps_the_dtype(tz: str | None) -> None:
    from disco.graph.extract import _fill_missing

    t = pd.to_datetime(["2020-01-01", None])
    if tz:
        t = t.tz_localize(tz)
    df = pd.DataFrame({"t": t})
    out = _fill_missing(df, "")

    assert out["t"].dtype == df["t"].dtype
    assert out["t"].iloc[0] == df["t"].iloc[0]
    assert pd.isna(out["t"].iloc[1])
