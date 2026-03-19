# tests/graph/test_vertex_weight.py
"""
Tests for the vertex_weight attribute on Graph and SuperGraph.
"""
from __future__ import annotations

import numpy as np
import pytest

from disco.graph import Graph, SuperGraph


def _simple_graph(num_vertices: int = 4, vertex_weight: np.ndarray | None = None) -> Graph:
    return Graph(
        layers=tuple(),
        num_vertices=num_vertices,
        scenario_id="s",
        vertex_weight=vertex_weight,
    )


# --------------------------------------------------------------------------- #
# Graph.vertex_weight
# --------------------------------------------------------------------------- #

def test_graph_default_vertex_weight_is_ones() -> None:
    g = _simple_graph(num_vertices=5)
    np.testing.assert_array_equal(g.vertex_weight, np.ones(5))
    assert g.vertex_weight.dtype == np.float64


def test_graph_custom_vertex_weight_stored() -> None:
    w = np.array([1.0, 2.0, 3.0, 4.0])
    g = _simple_graph(num_vertices=4, vertex_weight=w)
    np.testing.assert_array_equal(g.vertex_weight, w)
    assert g.vertex_weight.dtype == np.float64


def test_graph_vertex_weight_coerced_to_float64() -> None:
    w = np.array([1, 2, 3], dtype=np.int32)
    g = _simple_graph(num_vertices=3, vertex_weight=w)
    assert g.vertex_weight.dtype == np.float64
    np.testing.assert_array_equal(g.vertex_weight, [1.0, 2.0, 3.0])


def test_vertex_weight_propagated_in_get_view() -> None:
    w = np.array([0.5, 1.5, 2.5])
    g = _simple_graph(num_vertices=3, vertex_weight=w)
    view = g.get_view()
    np.testing.assert_array_equal(view.vertex_weight, w)


def test_vertex_weight_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="vertex_weight length"):
        _simple_graph(num_vertices=4, vertex_weight=np.ones(3))


# --------------------------------------------------------------------------- #
# SuperGraph.vertex_weight via _build_supergraph
# --------------------------------------------------------------------------- #

def test_supergraph_vertex_weight_is_summed() -> None:
    """
    Two groups: {0,1} → supervertex 0, {2,3,4} → supervertex 1.
    vertex_weight = [1, 2, 3, 4, 5].
    Expected super_weights = [3.0, 12.0].
    """
    g = _simple_graph(
        num_vertices=5,
        vertex_weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    vertex_map = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    sg = g._build_supergraph(vertex_map, num_super=2)

    assert isinstance(sg, SuperGraph)
    np.testing.assert_array_equal(sg.vertex_weight, [3.0, 12.0])
    assert sg.vertex_weight.dtype == np.float64


def test_supergraph_default_weights_summed() -> None:
    """Default weights (all-ones) should sum to group sizes."""
    g = _simple_graph(num_vertices=6)
    vertex_map = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    sg = g._build_supergraph(vertex_map, num_super=3)

    np.testing.assert_array_equal(sg.vertex_weight, [3.0, 2.0, 1.0])


def test_supergraph_inherits_vertex_weight_from_graph() -> None:
    """The SuperGraph produced by Graph.compress carries the vertex_weight attribute."""
    import graphblas as gb

    g = Graph(
        layers=tuple(),
        num_vertices=4,
        scenario_id="s",
        vertex_weight=np.array([2.0, 2.0, 5.0, 5.0]),
    )
    # Attach labels: two groups share label "region=north" (v0,v1) and "region=south" (v2,v3)
    from disco.partitioner import NODE_TYPE
    label_meta = {0: (NODE_TYPE, "A"), 1: ("region", "north"), 2: ("region", "south")}
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 0, 2, 0, 2]
    lm = gb.Matrix.from_coo(rows, cols, [True] * 8, nrows=4, ncols=3, dtype=bool)
    g.set_labels(label_matrix=lm, label_meta=label_meta)

    sg = g.compress(["region"])

    # "north" group: v0 + v1 = 4.0; "south" group: v2 + v3 = 10.0
    # Unlabelled: none
    assert sg.num_vertices == 2
    assert sg.vertex_weight.sum() == pytest.approx(14.0)
