# tests/graph/test_supergraph.py
"""Tests for Graph.compress() and SuperGraph."""
from __future__ import annotations

import numpy as np
import graphblas as gb
import pytest

from disco.graph.core import Graph, SuperGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_graph(
    num_vertices: int,
    edges: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    scenario_id: str = "test",
) -> Graph:
    """
    Convenience builder: create a Graph with a single layer.

    If *edges* is ``None`` a single layer with no edges is created.
    """
    if edges is None:
        empty = np.empty(0, dtype=np.int64)
        edges = {0: (empty, empty, np.empty(0, dtype=np.float64))}
    return Graph.from_edges(edges, num_vertices=num_vertices, scenario_id=scenario_id)


def _attach_labels(
    graph: Graph,
    label_meta: dict[int, tuple[str, str]],
    assignments: list[tuple[int, int]],
) -> None:
    """
    Attach a label matrix from explicit (vertex_index, label_index) pairs.
    """
    n_labels = max(idx for idx in label_meta) + 1 if label_meta else 0
    if not assignments:
        mat = gb.Matrix.from_coo(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=bool),
            nrows=graph.num_vertices,
            ncols=n_labels,
            dtype=gb.dtypes.BOOL,
        )
    else:
        rows = np.array([r for r, _ in assignments], dtype=np.int64)
        cols = np.array([c for _, c in assignments], dtype=np.int64)
        vals = np.ones(len(assignments), dtype=bool)
        mat = gb.Matrix.from_coo(
            rows, cols, vals,
            nrows=graph.num_vertices, ncols=n_labels, dtype=gb.dtypes.BOOL,
        )
    graph.set_labels(mat, label_meta)


# ---------------------------------------------------------------------------
# Unlabelled vertices get their own supervertex
# ---------------------------------------------------------------------------
def test_all_unlabelled():
    """When no vertex has any of the selected labels, every vertex is 1:1."""
    g = _make_graph(5)
    _attach_labels(
        g,
        label_meta={0: ("color", "red")},
        assignments=[],
    )
    sg = g.compress(["color"])

    assert isinstance(sg, SuperGraph)
    assert sg.num_vertices == 5
    assert sg.num_original_vertices == 5
    assert len(np.unique(sg.vertex_map)) == 5


def test_mixed_labelled_and_unlabelled():
    """
    Vertices 0-2 share a label → 1 supervertex.
    Vertices 3-4 have no label → 2 individual supervertices.
    Total: 3 supervertices.
    """
    g = _make_graph(5)
    _attach_labels(
        g,
        label_meta={0: ("group", "A")},
        assignments=[(0, 0), (1, 0), (2, 0)],
    )
    sg = g.compress(["group"])

    assert sg.num_vertices == 3
    assert sg.vertex_map[0] == sg.vertex_map[1] == sg.vertex_map[2]
    assert sg.vertex_map[3] != sg.vertex_map[4]
    assert sg.vertex_map[3] != sg.vertex_map[0]
    assert sg.vertex_map[4] != sg.vertex_map[0]


def test_unlabelled_count_matches_original():
    """
    The number of supervertices for unlabelled vertices must equal
    the number of unlabelled vertices in the original graph.
    """
    n = 10
    g = _make_graph(n)
    _attach_labels(
        g,
        label_meta={0: ("x", "a"), 1: ("x", "b")},
        assignments=[(0, 0), (1, 1)],
    )
    sg = g.compress(["x"])

    n_labelled_sv = len({sg.vertex_map[0], sg.vertex_map[1]})
    n_unlabelled_sv = sg.num_vertices - n_labelled_sv
    assert n_unlabelled_sv == n - 2


# ---------------------------------------------------------------------------
# Transitive merging of overlapping label sets
# ---------------------------------------------------------------------------
def test_full_chain_collapse():
    """
    Label sub-matrix (4 vertices, 3 labels):

        0     1     2
    0  True
    1        True
    2  True        True
    3        True  True

    Labels 0 and 2 share vertex 2, labels 1 and 2 share vertex 3.
    All four vertices must collapse into one supervertex.
    """
    g = _make_graph(4)
    _attach_labels(
        g,
        label_meta={0: ("t", "a"), 1: ("t", "b"), 2: ("t", "c")},
        assignments=[(0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 2)],
    )
    sg = g.compress(["t"])

    assert sg.num_vertices == 1
    np.testing.assert_array_equal(sg.vertex_map, [0, 0, 0, 0])


def test_spec_example_two_groups():
    """
    8 vertices, 4 labels:

          0     1     2     3
    0  True
    1  True
    2  True
    3  True
    4        True  True
    5        True  True
    6        True        True
    7        True        True

    Label 0 has vertices {0,1,2,3}; labels 1,2,3 share vertices {4,5,6,7}
    via overlapping columns. Result: 2 supervertices.
    """
    g = _make_graph(8)
    _attach_labels(
        g,
        label_meta={
            0: ("t", "a"), 1: ("t", "b"), 2: ("t", "c"), 3: ("t", "d"),
        },
        assignments=[
            (0, 0), (1, 0), (2, 0), (3, 0),
            (4, 1), (4, 2),
            (5, 1), (5, 2),
            (6, 1), (6, 3),
            (7, 1), (7, 3),
        ],
    )
    sg = g.compress(["t"])

    assert sg.num_vertices == 2
    np.testing.assert_array_equal(
        sg.vertex_map, [0, 0, 0, 0, 1, 1, 1, 1],
    )


def test_spec_example_single_group():
    """
    8 vertices, 4 labels:

          0     1     2     3
    0  True
    1  True
    2  True
    3  True        True
    4        True  True
    5        True  True
    6        True        True
    7        True        True

    Vertex 3 bridges label 0 and label 2. Label 2 is shared with
    label 1 (via vertices 4,5). Label 1 bridges to label 3
    (via vertices 6,7). All collapse into one supervertex.
    """
    g = _make_graph(8)
    _attach_labels(
        g,
        label_meta={
            0: ("t", "a"), 1: ("t", "b"), 2: ("t", "c"), 3: ("t", "d"),
        },
        assignments=[
            (0, 0), (1, 0), (2, 0),
            (3, 0), (3, 2),
            (4, 1), (4, 2),
            (5, 1), (5, 2),
            (6, 1), (6, 3),
            (7, 1), (7, 3),
        ],
    )
    sg = g.compress(["t"])

    assert sg.num_vertices == 1
    np.testing.assert_array_equal(
        sg.vertex_map, [0, 0, 0, 0, 0, 0, 0, 0],
    )


def test_two_disjoint_groups():
    """Two labels with disjoint vertex sets → two supervertices."""
    g = _make_graph(4)
    _attach_labels(
        g,
        label_meta={0: ("t", "a"), 1: ("t", "b")},
        assignments=[(0, 0), (1, 0), (2, 1), (3, 1)],
    )
    sg = g.compress(["t"])

    assert sg.num_vertices == 2
    assert sg.vertex_map[0] == sg.vertex_map[1]
    assert sg.vertex_map[2] == sg.vertex_map[3]
    assert sg.vertex_map[0] != sg.vertex_map[2]


# ---------------------------------------------------------------------------
# Large vertex indices (> 1_000_000)
# ---------------------------------------------------------------------------
def test_compress_with_large_indices():
    """
    Graph with 1_500_001 vertices.  Only the two highest-index vertices
    are labelled (same label) → they merge; the rest are individual.
    """
    n = 1_500_001
    v_high_a = 1_000_000
    v_high_b = 1_500_000

    g = _make_graph(n)
    _attach_labels(
        g,
        label_meta={0: ("big", "x")},
        assignments=[(v_high_a, 0), (v_high_b, 0)],
    )
    sg = g.compress(["big"])

    assert sg.num_vertices == 1 + (n - 2)
    assert sg.vertex_map[v_high_a] == sg.vertex_map[v_high_b]
    assert sg.vertex_map[0] != sg.vertex_map[1]
    assert sg.vertex_map[0] != sg.vertex_map[v_high_a]


def test_decompress_with_large_indices():
    """Decompress a vector on a supergraph with large original indices."""
    n = 1_100_000
    v_a, v_b, v_c = 1_000_000, 1_050_000, 1_099_999

    g = _make_graph(n)
    _attach_labels(
        g,
        label_meta={0: ("big", "x")},
        assignments=[(v_a, 0), (v_b, 0), (v_c, 0)],
    )
    sg = g.compress(["big"])

    sv = sg.vertex_map[v_a]
    assert sg.vertex_map[v_b] == sv
    assert sg.vertex_map[v_c] == sv

    vec = gb.Vector.from_coo(
        np.array([sv], dtype=np.int64),
        np.array([42.0]),
        size=sg.num_vertices,
    )
    expanded = sg.decompress(vec)

    assert expanded.size == n
    ixs, vals = expanded.to_coo()
    assert set(ixs.tolist()) == {v_a, v_b, v_c}
    np.testing.assert_array_equal(vals, [42.0, 42.0, 42.0])


# ---------------------------------------------------------------------------
# Edge compression
# ---------------------------------------------------------------------------
def test_edges_sum_and_self_loops_removed():
    """
    4 vertices, 1 layer.
    Edges: 0→2 (w=1), 1→3 (w=2), 2→3 (w=3)
    Labels: vertices {0,1} = group A, vertices {2,3} = group B.

    After compression:
    - sv0 (={0,1}), sv1 (={2,3})
    - 0→2 and 1→3 become sv0→sv1 with weight 1+2 = 3.
    - 2→3 becomes sv1→sv1 (self-loop) and is discarded.
    """
    g = Graph.from_edges(
        {0: (
            np.array([0, 1, 2]),
            np.array([2, 3, 3]),
            np.array([1.0, 2.0, 3.0]),
        )},
        num_vertices=4,
        scenario_id="test",
    )
    _attach_labels(
        g,
        label_meta={0: ("grp", "A"), 1: ("grp", "B")},
        assignments=[(0, 0), (1, 0), (2, 1), (3, 1)],
    )
    sg = g.compress(["grp"])

    assert sg.num_vertices == 2
    mat = sg.get_matrix(0)
    rows, cols, vals = mat.to_coo()

    assert rows.size == 1
    sv0 = sg.vertex_map[0]
    sv1 = sg.vertex_map[2]
    assert int(rows[0]) == sv0
    assert int(cols[0]) == sv1
    assert float(vals[0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------
def test_labels_propagated_via_lor():
    """
    Vertices 0,1 share label "a"; vertex 1 also has label "b".
    After compression the supervertex must carry both labels.
    """
    g = _make_graph(3)
    _attach_labels(
        g,
        label_meta={0: ("t", "a"), 1: ("t", "b")},
        assignments=[(0, 0), (1, 0), (1, 1)],
    )
    sg = g.compress(["t"])

    assert sg.num_vertices == 2
    sv = sg.vertex_map[0]
    lm = sg.label_matrix
    assert lm is not None
    assert lm[sv, 0].new().value is True
    assert lm[sv, 1].new().value is True


# ---------------------------------------------------------------------------
# Decompress
# ---------------------------------------------------------------------------
def test_decompress_vector_round_trip():
    """Values assigned to a supervertex fan out to all constituent vertices."""
    g = _make_graph(6)
    _attach_labels(
        g,
        label_meta={0: ("grp", "A"), 1: ("grp", "B")},
        assignments=[(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)],
    )
    sg = g.compress(["grp"])
    assert sg.num_vertices == 2

    sv_a = sg.vertex_map[0]
    sv_b = sg.vertex_map[3]
    vec = gb.Vector.from_coo(
        np.array([sv_a, sv_b], dtype=np.int64),
        np.array([10.0, 20.0]),
        size=sg.num_vertices,
    )
    expanded = sg.decompress(vec)

    assert expanded.size == 6
    ixs, vals = expanded.to_coo()
    np.testing.assert_array_equal(ixs, [0, 1, 2, 3, 4, 5])
    for v in [0, 1, 2]:
        assert float(vals[v]) == pytest.approx(10.0)
    for v in [3, 4, 5]:
        assert float(vals[v]) == pytest.approx(20.0)


def test_decompress_sparse_vector():
    """Absent supervertices leave their original vertices absent."""
    g = _make_graph(4)
    _attach_labels(
        g,
        label_meta={0: ("grp", "A"), 1: ("grp", "B")},
        assignments=[(0, 0), (1, 0), (2, 1), (3, 1)],
    )
    sg = g.compress(["grp"])

    sv_a = sg.vertex_map[0]
    vec = gb.Vector.from_coo(
        np.array([sv_a], dtype=np.int64),
        np.array([7.0]),
        size=sg.num_vertices,
    )
    expanded = sg.decompress(vec)

    ixs, vals = expanded.to_coo()
    assert set(ixs.tolist()) == {0, 1}
    np.testing.assert_array_equal(vals, [7.0, 7.0])


def test_decompress_indices():
    """Index-based decompress returns original vertex indices."""
    g = _make_graph(6)
    _attach_labels(
        g,
        label_meta={0: ("grp", "A"), 1: ("grp", "B")},
        assignments=[(0, 0), (1, 0), (3, 1), (4, 1), (5, 1)],
    )
    sg = g.compress(["grp"])

    sv_b = sg.vertex_map[3]
    orig = sg.decompress({sv_b})
    np.testing.assert_array_equal(orig, [3, 4, 5])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------
def test_compress_no_labels_attached():
    g = _make_graph(3)
    with pytest.raises(RuntimeError, match="no labels attached"):
        g.compress(["t"])


def test_compress_unknown_label_type():
    g = _make_graph(3)
    _attach_labels(g, label_meta={0: ("x", "a")}, assignments=[(0, 0)])
    with pytest.raises(KeyError, match="nonexistent"):
        g.compress(["nonexistent"])


# ---------------------------------------------------------------------------
# SuperGraph is a Graph
# ---------------------------------------------------------------------------
def test_supergraph_isinstance():
    g = _make_graph(4)
    _attach_labels(g, label_meta={0: ("t", "a")}, assignments=[(0, 0), (1, 0)])
    sg = g.compress(["t"])
    assert isinstance(sg, Graph)
    assert isinstance(sg, SuperGraph)


def test_supergraph_vertices_is_none():
    g = _make_graph(4)
    _attach_labels(g, label_meta={0: ("t", "a")}, assignments=[(0, 0), (1, 0)])
    sg = g.compress(["t"])
    assert sg.vertices is None


def test_supergraph_repr():
    g = _make_graph(4)
    _attach_labels(g, label_meta={0: ("t", "a")}, assignments=[(0, 0), (1, 0)])
    sg = g.compress(["t"])
    r = repr(sg)
    assert "SuperGraph" in r
    assert "num_supervertices=" in r
    assert "num_original_vertices=" in r