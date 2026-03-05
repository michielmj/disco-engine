# tests/graph/test_graph_factory.py
"""
Tests for disco.graph_factory.graph_from_model.

Strategy:
  - SQLite in-memory database per test (via fixtures).
  - Model tables built inline with SQLAlchemy; no ORM provider needed (reflection mode).
  - Model is built from SimpleNamespace stubs to avoid heavyweight loading.
  - db_validate is exercised for real: tables must satisfy the ORM contracts.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, MetaData, String, Table, UniqueConstraint, insert
from sqlalchemy.engine import Engine

from disco.database import SessionManager
from disco.model.orm import build_orm_bundle
from disco.graph_factory import graph_from_model
from disco.partitioner import NODE_TYPE


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _create_node_table(md: MetaData, name: str, extra_cols: list | None = None) -> Table:
    """Minimal node table satisfying ORM contracts: (scenario_id, key) unique."""
    cols = [
        Column("scenario_id", String, nullable=False),
        Column("key", String, nullable=False),
    ]
    if extra_cols:
        cols.extend(extra_cols)
    return Table(name, md, *cols, UniqueConstraint("scenario_id", "key"))


def _create_edge_table(md: MetaData, name: str) -> Table:
    """Minimal edge table satisfying ORM contracts: (scenario_id, source_key, target_key)."""
    return Table(
        name,
        md,
        Column("scenario_id", String, nullable=False),
        Column("source_key", String, nullable=False),
        Column("target_key", String, nullable=False),
    )


# ---------------------------------------------------------------------------
# Spec / Model helpers
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    node_types: dict[str, dict],
    simprocs: list[str],
    simproc_edge_data_tables: dict[str, str] | None = None,
    default_edge_data_table: str | None = None,
) -> SimpleNamespace:
    """
    Build a minimal spec-like object for build_orm_bundle and graph_from_model.

    node_types keys are node-type names; values are dicts with:
      - "table": str  (node_data_table name)
      - "distinct": list[str]  (optional, distinct_nodes)
    """
    node_type_specs = {
        nt: SimpleNamespace(
            node_data_table=info["table"],
            distinct_nodes=info.get("distinct", []),
        )
        for nt, info in node_types.items()
    }
    return SimpleNamespace(
        orm=None,
        node_types=node_type_specs,
        simproc_edge_data_tables=simproc_edge_data_tables or {},
        default_edge_data_table=default_edge_data_table,
        simprocs=simprocs,
    )


def _make_model(spec: SimpleNamespace, engine: Engine) -> SimpleNamespace:
    """Attach an OrmBundle via reflection and return a minimal Model stub."""
    orm = build_orm_bundle(spec=spec, db=engine)
    return SimpleNamespace(spec=spec, orm=orm)


# ---------------------------------------------------------------------------
# Single-type, single-simproc fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_db():
    """
    In-memory SQLite with:
      - node table "sp_nodes": (scenario_id, key)
      - edge table "sp_edges": (scenario_id, source_key, target_key)

    Seed data for scenario "s1":
      nodes: v0, v1, v2
      edges: v0->v1, v1->v2
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    nodes = _create_node_table(md, "sp_nodes")
    edges = _create_edge_table(md, "sp_edges")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(nodes),
            [
                {"scenario_id": "s1", "key": "v0"},
                {"scenario_id": "s1", "key": "v1"},
                {"scenario_id": "s1", "key": "v2"},
            ],
        )
        conn.execute(
            insert(edges),
            [
                {"scenario_id": "s1", "source_key": "v0", "target_key": "v1"},
                {"scenario_id": "s1", "source_key": "v1", "target_key": "v2"},
            ],
        )

    return engine


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_basic_graph_structure(simple_db: Engine) -> None:
    """
    One node type, one simproc with default edge table.
    Checks vertex count, layer count, scenario_id, and a specific edge.
    """
    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["transport"],
        default_edge_data_table="sp_edges",
    )
    model = _make_model(spec, simple_db)
    db = SessionManager(simple_db)

    graph = graph_from_model(db, "s1", model)

    assert graph.num_vertices == 3
    assert graph.scenario_id == "s1"

    # One simproc -> one layer
    mat = graph.get_matrix(0)
    assert mat.nrows == 3
    assert mat.ncols == 3

    # Edges: v0(0)->v1(1), v1(1)->v2(2) with weight 1.0
    assert mat[0, 1].value == 1.0
    assert mat[1, 2].value == 1.0


def test_scenario_id_propagated(simple_db: Engine) -> None:
    """graph.scenario_id must match the scenario_id argument."""
    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["transport"],
        default_edge_data_table="sp_edges",
    )
    model = _make_model(spec, simple_db)
    db = SessionManager(simple_db)

    graph = graph_from_model(db, "s1", model)
    assert graph.scenario_id == "s1"


def test_node_type_labels_assigned() -> None:
    """
    Two node types -> NODE_TYPE labels should contain one entry per node type
    whose value is the array of vertex indices belonging to that type.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    sp_nodes = _create_node_table(md, "sp_nodes")
    dc_nodes = _create_node_table(md, "dc_nodes")
    edges = _create_edge_table(md, "edges")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(sp_nodes),
            [
                {"scenario_id": "s1", "key": "sp0"},
                {"scenario_id": "s1", "key": "sp1"},
            ],
        )
        conn.execute(
            insert(dc_nodes),
            [{"scenario_id": "s1", "key": "dc0"}],
        )

    spec = _make_spec(
        node_types={
            "sp": {"table": "sp_nodes"},
            "dc": {"table": "dc_nodes"},
        },
        simprocs=["transport"],
        default_edge_data_table="edges",
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    graph = graph_from_model(db, "s1", model)

    assert graph.num_vertices == 3

    # NODE_TYPE labels must be present
    assert NODE_TYPE in graph._labels_by_type

    sp_idx = graph.label_value_to_index(NODE_TYPE)["sp"]
    dc_idx = graph.label_value_to_index(NODE_TYPE)["dc"]

    sp_verts = set(graph.get_vertices_for_label(sp_idx).tolist())
    dc_verts = set(graph.get_vertices_for_label(dc_idx).tolist())

    # sp has 2 vertices (indices 0,1); dc has 1 (index 2)
    assert sp_verts == {0, 1}
    assert dc_verts == {2}


def test_distinct_node_labels() -> None:
    """
    distinct_nodes=["region"] on a node type creates per-region label groups.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    nodes = _create_node_table(md, "sp_nodes", extra_cols=[Column("region", String)])
    edges = _create_edge_table(md, "sp_edges")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(nodes),
            [
                {"scenario_id": "s1", "key": "v0", "region": "N"},
                {"scenario_id": "s1", "key": "v1", "region": "N"},
                {"scenario_id": "s1", "key": "v2", "region": "S"},
            ],
        )

    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes", "distinct": ["region"]}},
        simprocs=["transport"],
        default_edge_data_table="sp_edges",
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    graph = graph_from_model(db, "s1", model)

    region_map = graph.label_value_to_index("region")
    assert set(region_map.keys()) == {"N", "S"}

    n_verts = set(graph.get_vertices_for_label(region_map["N"]).tolist())
    s_verts = set(graph.get_vertices_for_label(region_map["S"]).tolist())
    assert n_verts == {0, 1}
    assert s_verts == {2}


def test_distinct_labels_union_across_node_types() -> None:
    """
    Two node types that share the same distinct label type have their vertex sets
    unioned under the same label value.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    sp_nodes = _create_node_table(md, "sp_nodes", extra_cols=[Column("region", String)])
    dc_nodes = _create_node_table(md, "dc_nodes", extra_cols=[Column("region", String)])
    edges = _create_edge_table(md, "edges")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(sp_nodes),
            [{"scenario_id": "s1", "key": "sp0", "region": "N"}],
        )
        conn.execute(
            insert(dc_nodes),
            [{"scenario_id": "s1", "key": "dc0", "region": "N"}],
        )

    spec = _make_spec(
        node_types={
            "sp": {"table": "sp_nodes", "distinct": ["region"]},
            "dc": {"table": "dc_nodes", "distinct": ["region"]},
        },
        simprocs=["transport"],
        default_edge_data_table="edges",
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    graph = graph_from_model(db, "s1", model)

    region_map = graph.label_value_to_index("region")
    n_verts = set(graph.get_vertices_for_label(region_map["N"]).tolist())
    # sp0 gets index 0, dc0 gets index 1 — both are region "N"
    assert n_verts == {0, 1}


def test_per_simproc_edge_table() -> None:
    """
    When a simproc has its own edge table, edges are loaded from that table
    and placed in the correct layer, independently of the default edge table.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    nodes = _create_node_table(md, "sp_nodes")
    demand_edges = _create_edge_table(md, "demand_edges")
    supply_edges = _create_edge_table(md, "supply_edges")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(nodes),
            [
                {"scenario_id": "s1", "key": "v0"},
                {"scenario_id": "s1", "key": "v1"},
                {"scenario_id": "s1", "key": "v2"},
            ],
        )
        # demand: v0->v1; supply: v1->v2
        conn.execute(insert(demand_edges), [
            {"scenario_id": "s1", "source_key": "v0", "target_key": "v1"},
        ])
        conn.execute(insert(supply_edges), [
            {"scenario_id": "s1", "source_key": "v1", "target_key": "v2"},
        ])

    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["demand", "supply"],
        simproc_edge_data_tables={"demand": "demand_edges", "supply": "supply_edges"},
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    graph = graph_from_model(db, "s1", model)

    demand_mat = graph.get_matrix(0)  # layer 0 = demand
    supply_mat = graph.get_matrix(1)  # layer 1 = supply

    assert demand_mat[0, 1].value == 1.0
    assert demand_mat[1, 2].value is None  # no supply edge in demand layer

    assert supply_mat[1, 2].value == 1.0
    assert supply_mat[0, 1].value is None  # no demand edge in supply layer


def test_multiple_simprocs_multiple_layers(simple_db: Engine) -> None:
    """
    Three simprocs all sharing the default edge table produce a 3-layer graph.
    """
    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["a", "b", "c"],
        default_edge_data_table="sp_edges",
    )
    model = _make_model(spec, simple_db)
    db = SessionManager(simple_db)

    graph = graph_from_model(db, "s1", model)

    # Three layers should exist
    for layer_idx in range(3):
        mat = graph.get_matrix(layer_idx)
        assert mat.nrows == 3


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


def test_missing_default_edge_table_raises() -> None:
    """
    When a simproc has no per-simproc table and there is no default edge table,
    graph_from_model must raise ValueError with a helpful message.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    nodes = _create_node_table(md, "sp_nodes")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(insert(nodes), [{"scenario_id": "s1", "key": "v0"}])

    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["orphan"],
        default_edge_data_table=None,
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    with pytest.raises(ValueError, match="orphan"):
        graph_from_model(db, "s1", model)


def test_empty_scenario() -> None:
    """
    A scenario with no rows in the node table yields a graph with 0 vertices
    and no labels (label_matrix is None).
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    _create_node_table(md, "sp_nodes")
    _create_edge_table(md, "sp_edges")
    md.create_all(engine)

    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["transport"],
        default_edge_data_table="sp_edges",
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    graph = graph_from_model(db, "empty_scenario", model)

    assert graph.num_vertices == 0
    assert graph.label_matrix is None


def test_scenario_isolation() -> None:
    """
    Vertices and edges from a different scenario_id are not included.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    md = MetaData()
    nodes = _create_node_table(md, "sp_nodes")
    edges = _create_edge_table(md, "sp_edges")
    md.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(nodes),
            [
                {"scenario_id": "s1", "key": "v0"},
                {"scenario_id": "s1", "key": "v1"},
                # Different scenario — must NOT appear:
                {"scenario_id": "other", "key": "x0"},
                {"scenario_id": "other", "key": "x1"},
                {"scenario_id": "other", "key": "x2"},
            ],
        )
        conn.execute(
            insert(edges),
            [
                {"scenario_id": "s1", "source_key": "v0", "target_key": "v1"},
                {"scenario_id": "other", "source_key": "x0", "target_key": "x1"},
            ],
        )

    spec = _make_spec(
        node_types={"sp": {"table": "sp_nodes"}},
        simprocs=["transport"],
        default_edge_data_table="sp_edges",
    )
    model = _make_model(spec, engine)
    db = SessionManager(engine)

    graph = graph_from_model(db, "s1", model)

    assert graph.num_vertices == 2
    mat = graph.get_matrix(0)
    assert mat.nvals == 1  # only one edge for s1
