# tests/graph/test_db.py
from typing import Iterator

import numpy as np
import graphblas as gb
from graphblas import Vector
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from disco.graph import (
    Graph,
    create_graph_schema,
    store_graph,
    load_graph_for_scenario,
    get_outbound_map,
    schema,
)
from disco.graph.schema import vertices as vertices_table
from disco.graph.schema import edges as edges_table
from disco.graph.schema import vertex_masks as vertex_masks_table


@pytest.fixture
def engine_and_session_factory() -> Iterator[tuple[object, sessionmaker]]:
    """
    Fixture that provides an in-memory SQLite engine and a sessionmaker.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    # Create graph tables (graph_scenarios, graph_vertices, etc.) in the in-memory database
    create_graph_schema(engine)

    SessionLocal = sessionmaker(bind=engine, future=True)

    yield engine, SessionLocal

    engine.dispose()


def _build_graph_with_vertices(
    num_vertices: int,
    scenario_id: str,
    edge_layers: dict | None = None,
) -> Graph:
    """
    Helper to build a Graph with vertex keys attached.
    """
    vertex_keys = np.array([f"v{i}" for i in range(num_vertices)], dtype=object)
    return Graph.from_edges(
        edge_layers or {},
        num_vertices=num_vertices,
        scenario_id=scenario_id,
        vertices=vertex_keys,
    )


# ---------------------------------------------------------------------------
# store_graph creates scenario + vertices tests
# ---------------------------------------------------------------------------


def test_store_graph_inserts_vertices(
    engine_and_session_factory: tuple[object, sessionmaker]
) -> None:
    _, SessionLocal = engine_and_session_factory
    session: Session = SessionLocal()

    num_vertices = 5
    scenario_id = "scenario_store_test"
    vertex_keys = np.array([f"k{i}" for i in range(num_vertices)], dtype=object)

    graph = Graph.from_edges(
        {},
        num_vertices=num_vertices,
        scenario_id=scenario_id,
        vertices=vertex_keys,
    )
    store_graph(session, graph)

    rows = session.execute(
        select(vertices_table.c.index, vertices_table.c.key)
        .where(vertices_table.c.scenario_id == scenario_id)
        .order_by(vertices_table.c.index)
    ).all()

    assert len(rows) == num_vertices
    for i, (idx, key) in enumerate(rows):
        assert idx == i
        assert key == vertex_keys[i]

    session.close()


def test_store_graph_duplicate_raises(
    engine_and_session_factory: tuple[object, sessionmaker]
) -> None:
    _, SessionLocal = engine_and_session_factory
    session: Session = SessionLocal()

    scenario_id = "dup_scenario"
    keys1 = np.array(["a", "b", "c"], dtype=object)
    keys2 = np.array(["x", "y"], dtype=object)

    graph1 = Graph.from_edges({}, num_vertices=3, scenario_id=scenario_id, vertices=keys1)
    store_graph(session, graph1)

    graph2 = Graph.from_edges({}, num_vertices=2, scenario_id=scenario_id, vertices=keys2)
    with pytest.raises(ValueError):
        store_graph(session, graph2)

    # Ensure vertex rows are still only for the first creation
    rows = session.execute(
        select(vertices_table.c.index, vertices_table.c.key).where(
            vertices_table.c.scenario_id == scenario_id
        )
    ).all()
    assert len(rows) == len(keys1)

    session.close()


def test_store_graph_chunked_insert_large(
    engine_and_session_factory: tuple[object, sessionmaker]
) -> None:
    """
    Sanity-check that chunked inserts work for larger numbers of vertices.
    """
    _, SessionLocal = engine_and_session_factory
    session: Session = SessionLocal()

    num_vertices = 25
    scenario_id = "chunk_test"
    vertex_keys = np.array([f"node_{i}" for i in range(num_vertices)], dtype=object)

    graph = Graph.from_edges(
        {},
        num_vertices=num_vertices,
        scenario_id=scenario_id,
        vertices=vertex_keys,
    )
    # Use small chunk_size so we definitely hit multiple chunks
    store_graph(session, graph, chunk_size=7)

    rows = session.execute(
        select(vertices_table.c.index, vertices_table.c.key)
        .where(vertices_table.c.scenario_id == scenario_id)
        .order_by(vertices_table.c.index)
    ).all()

    assert len(rows) == num_vertices
    for i, (idx, key) in enumerate(rows):
        assert idx == i
        assert key == vertex_keys[i]

    session.close()


def test_store_graph_without_vertices_raises(
    engine_and_session_factory: tuple[object, sessionmaker]
) -> None:
    """store_graph must raise if the graph has no vertices attached."""
    _, SessionLocal = engine_and_session_factory
    session: Session = SessionLocal()

    graph = Graph.from_edges({}, num_vertices=3, scenario_id="no-verts")

    with pytest.raises(ValueError, match="vertices"):
        store_graph(session, graph)

    session.close()


# ---------------------------------------------------------------------------
# Graph roundtrip (edges + labels + mask)
# ---------------------------------------------------------------------------


def test_store_and_load_graph_with_labels_and_mask(
    engine_and_session_factory: tuple[object, sessionmaker]
) -> None:
    _, SessionLocal = engine_and_session_factory
    session: Session = SessionLocal()

    num_vertices = 4
    scenario_id = "roundtrip_test"
    vertex_keys = np.array([f"v{i}" for i in range(num_vertices)], dtype=object)

    # ------------------------------------------------------------------
    # Build in-memory Graph: edges + vertices
    # ------------------------------------------------------------------
    src = np.array([0, 1], dtype=np.int64)
    tgt = np.array([1, 2], dtype=np.int64)
    wgt = np.array([1.0, 2.0], dtype=np.float64)
    edge_layers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        0: (src, tgt, wgt)
    }

    graph = Graph.from_edges(
        edge_layers,
        num_vertices=num_vertices,
        scenario_id=scenario_id,
        vertices=vertex_keys,
    )

    # ------------------------------------------------------------------
    # Attach labels
    #   - 3 labels: indices 0,1,2
    #   - label_meta:
    #       0 -> (type1, A)
    #       1 -> (type2, B)
    #       2 -> (type1, C)
    #   - assignments:
    #       (0, 0), (1, 1), (2, 2), (3, 0)
    # ------------------------------------------------------------------
    v_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    l_idx = np.array([0, 1, 2, 0], dtype=np.int64)
    vals = np.ones(len(v_idx), dtype=bool)

    label_matrix = gb.Matrix.from_coo(
        v_idx,
        l_idx,
        vals,
        nrows=num_vertices,
        ncols=3,
    )

    label_meta = {
        0: ("type1", "A"),
        1: ("type2", "B"),
        2: ("type1", "C"),
    }

    graph.set_labels(label_matrix, label_meta)

    # ------------------------------------------------------------------
    # Set a mask (vertices 1 and 2)
    # NOTE: mask is NOT persisted by store_graph/load_graph,
    #       but will be used in DB queries via graph_vertex_masks.
    # ------------------------------------------------------------------
    mask_vec = Vector.from_coo([1, 2], [True, True], size=num_vertices)
    graph.set_mask(mask_vec)

    # ------------------------------------------------------------------
    # Store graph (scenario + vertices + edges + labels) into DB
    # ------------------------------------------------------------------
    store_graph(session, graph)
    session.commit()

    # Quick sanity: edges really exist in DB
    db_edges = session.execute(
        edges_table.select().where(edges_table.c.scenario_id == scenario_id)
    ).all()
    assert len(db_edges) == 2

    # Quick sanity: vertices really exist in DB
    db_vertices = session.execute(
        vertices_table.select()
        .where(vertices_table.c.scenario_id == scenario_id)
        .order_by(vertices_table.c.index)
    ).all()
    assert len(db_vertices) == num_vertices

    # ------------------------------------------------------------------
    # Load graph back from DB
    # ------------------------------------------------------------------
    loaded = load_graph_for_scenario(session, scenario_id)

    # Structure checks
    assert loaded.num_vertices == num_vertices
    loaded_mat = loaded.get_matrix(0)
    assert loaded_mat.nrows == num_vertices
    assert loaded_mat.ncols == num_vertices
    # Should match the original weights
    assert loaded_mat[0, 1].value == 1.0
    assert loaded_mat[1, 2].value == 2.0

    # Vertices should be restored
    assert loaded.vertices is not None
    assert len(loaded.vertices) == num_vertices
    for i, key in enumerate(vertex_keys):
        assert loaded.vertices[i] == key

    # Label checks
    assert loaded.num_labels == 3
    assert loaded.label_matrix is not None
    assert loaded.label_matrix.nrows == num_vertices
    assert loaded.label_matrix.ncols == 3

    # Meta should match
    assert loaded.label_meta == label_meta

    # Vertices that have label id 0: vertices 0 and 3
    verts_for_label0 = loaded.get_vertices_for_label(0)
    assert set(verts_for_label0.tolist()) == {0, 3}

    # Per-type metadata should be reconstructed correctly:
    # For "type1", we expect indices 0 ("A") and 2 ("C")
    type1_mapping = loaded.label_value_to_index("type1")
    assert type1_mapping["A"] == 0
    assert type1_mapping["C"] == 2

    # ------------------------------------------------------------------
    # Mask + outbound map via DB (vertex_masks join)
    #   - Set mask on loaded graph to vertices {1,2}
    #   - Only edges whose source is in {1,2} should appear
    #   - In our edge set, only edge (1 -> 2) remains
    # ------------------------------------------------------------------
    loaded_mask_vec = Vector.from_coo([1, 2], [True, True], size=num_vertices)
    loaded.set_mask(loaded_mask_vec)

    # Use the free function from disco.graph now that Graph
    # no longer has DB methods on the instance.
    mat_out = get_outbound_map(session, loaded, layer_idx=0, values=schema.edges.c.weight)

    # Expect only one edge: 1 -> 2 with weight 2.0
    rows, cols, vals_out = mat_out.to_coo()
    assert len(rows) == 1
    assert int(rows[0]) == 1
    assert int(cols[0]) == 2
    assert float(vals_out[0]) == 2.0

    # And ensure underlying vertex_masks table was populated for this scenario
    vm_rows = session.execute(
        vertex_masks_table.select().where(
            vertex_masks_table.c.scenario_id == scenario_id
        )
    ).all()
    # We only assert that some rows exist; exact count depends on mask usage
    assert len(vm_rows) > 0

    session.close()
