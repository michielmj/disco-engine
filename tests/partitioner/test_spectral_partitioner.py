# tests/partitioner/test_spectral_partitioner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import graphblas as gb
import numpy as np
import pytest

from disco.graph import Graph
from disco.partitioner import SpectralClusteringPartitioner, NODE_TYPE


# ------------------------------------------------------------------ #
# Stubs
# ------------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class NodeTypeSpecStub:
    distinct_nodes: List[str]
    same_node: List[str]
    self_relations: List[Tuple[str, str]]


@dataclass(frozen=True, slots=True)
class ModelSpecStub:
    simprocs: List[str]
    node_types: Dict[str, NodeTypeSpecStub]


@dataclass(frozen=True, slots=True)
class ModelStub:
    spec: ModelSpecStub


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_model(
    node_types: Dict[str, NodeTypeSpecStub] | None = None,
    simprocs: List[str] | None = None
) -> ModelStub:
    """Single node-type 'A' with region as distinct label, no same_node."""
    if node_types is None:
        node_types = {
            "A": NodeTypeSpecStub(
                distinct_nodes=["region"], same_node=[], self_relations=[],
            ),
        }
    if simprocs is None:
        simprocs = ['simproc']
    return ModelStub(spec=ModelSpecStub(simprocs=simprocs, node_types=node_types))


def _coo_indices(v: gb.Vector) -> set[int]:
    idx, _ = v.select("==", True).to_coo()
    return set(idx.tolist())


def _partition_vertex_sets(part) -> Dict[int, set[int]]:
    """Group assigned vertex indices by partition number."""
    result: Dict[int, set[int]] = {}
    for ns in part.node_specs:
        verts = _coo_indices(part.assignment_vector(ns.node_name))
        result.setdefault(ns.partition, set()).update(verts)
    return result


# ------------------------------------------------------------------ #
# Graph factories
# ------------------------------------------------------------------ #


def _make_two_cluster_graph() -> Graph:
    """
    Two disconnected DAG clusters of 3 vertices each, one layer.

    Cluster north: 0→1, 0→2, 1→2
    Cluster south: 3→4, 3→5, 4→5

    Labels:
      node-type=A  on all vertices
      region=north on 0, 1, 2
      region=south on 3, 4, 5
    """
    nv = 6
    src = np.array([0, 0, 1, 3, 3, 4], dtype=np.int64)
    dst = np.array([1, 2, 2, 4, 5, 5], dtype=np.int64)
    w = np.ones(6)

    layer = gb.Matrix.from_coo(src, dst, w, nrows=nv, ncols=nv)
    g = Graph(layers=(layer,), num_vertices=nv, scenario_id="s1")

    label_meta = {
        0: (NODE_TYPE, "A"),
        1: ("region", "north"),
        2: ("region", "south"),
    }
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    cols = [0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2]
    lm = gb.Matrix.from_coo(rows, cols, [True] * 12, nrows=nv, ncols=3, dtype=bool)
    g.set_labels(lm, label_meta)
    return g


def _make_four_pair_graph() -> Graph:
    """
    Four disconnected pairs: (0,1), (2,3), (4,5), (6,7).

    Each pair has one DAG edge and a distinct region label.
    All vertices are node-type=A.
    """
    nv = 8
    src = np.array([0, 2, 4, 6], dtype=np.int64)
    dst = np.array([1, 3, 5, 7], dtype=np.int64)
    w = np.ones(4)

    layer = gb.Matrix.from_coo(src, dst, w, nrows=nv, ncols=nv)
    g = Graph(layers=(layer,), num_vertices=nv, scenario_id="s1")

    label_meta = {
        0: (NODE_TYPE, "A"),
        1: ("region", "r0"),
        2: ("region", "r1"),
        3: ("region", "r2"),
        4: ("region", "r3"),
    }
    rows: List[int] = []
    cols: List[int] = []
    for v in range(nv):
        rows.append(v)
        cols.append(0)
    for pair_idx, (va, vb) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
        for v in (va, vb):
            rows.append(v)
            cols.append(1 + pair_idx)

    lm = gb.Matrix.from_coo(rows, cols, [True] * len(rows), nrows=nv, ncols=5, dtype=bool)
    g.set_labels(lm, label_meta)
    return g


def _make_barbell_graph() -> Graph:
    """
    Two 4-clique DAGs connected by a weak bridge.

    Clique north: vertices 0-3 (DAG edges weight 10.0)
    Clique south: vertices 4-7 (DAG edges weight 10.0)
    Bridge: 3→4 (weight 0.01)

    After symmetrisation the two cliques are strongly intra-connected
    and weakly inter-connected — spectral clustering should split cleanly.
    """
    nv = 8
    src_a = [0, 0, 0, 1, 1, 2]
    dst_a = [1, 2, 3, 2, 3, 3]
    src_b = [4, 4, 4, 5, 5, 6]
    dst_b = [5, 6, 7, 6, 7, 7]

    src = np.array(src_a + src_b + [3], dtype=np.int64)
    dst = np.array(dst_a + dst_b + [4], dtype=np.int64)
    w = np.array([10.0] * 6 + [10.0] * 6 + [0.01])

    layer = gb.Matrix.from_coo(src, dst, w, nrows=nv, ncols=nv)
    g = Graph(layers=(layer,), num_vertices=nv, scenario_id="s1")

    label_meta = {
        0: (NODE_TYPE, "A"),
        1: ("region", "north"),
        2: ("region", "south"),
    }
    rows: List[int] = []
    cols: List[int] = []
    for v in range(nv):
        rows.append(v)
        cols.append(0)
    for v in range(4):
        rows.append(v)
        cols.append(1)
    for v in range(4, 8):
        rows.append(v)
        cols.append(2)

    lm = gb.Matrix.from_coo(rows, cols, [True] * len(rows), nrows=nv, ncols=3, dtype=bool)
    g.set_labels(lm, label_meta)
    return g


# ------------------------------------------------------------------ #
# Constructor validation
# ------------------------------------------------------------------ #


def test_constructor_rejects_graph_without_labels() -> None:
    g = Graph(layers=tuple(), num_vertices=2, scenario_id="s")
    with pytest.raises(ValueError, match="no labels"):
        SpectralClusteringPartitioner(g, _make_model())


def test_constructor_rejects_missing_node_type_label() -> None:
    g = Graph(layers=tuple(), num_vertices=1, scenario_id="s")
    lm = gb.Matrix.from_coo([0], [0], [True], nrows=1, ncols=1, dtype=bool)
    g.set_labels(lm, {0: ("region", "north")})

    with pytest.raises(KeyError):
        SpectralClusteringPartitioner(g, _make_model())


def test_constructor_rejects_missing_distinct_label_type() -> None:
    g = Graph(layers=tuple(), num_vertices=1, scenario_id="s")
    lm = gb.Matrix.from_coo([0], [0], [True], nrows=1, ncols=1, dtype=bool)
    g.set_labels(lm, {0: (NODE_TYPE, "A")})

    with pytest.raises(KeyError, match="region"):
        SpectralClusteringPartitioner(g, _make_model())


def test_constructor_rejects_missing_same_node_label_type() -> None:
    g = Graph(layers=tuple(), num_vertices=1, scenario_id="s")
    lm = gb.Matrix.from_coo([0], [0], [True], nrows=1, ncols=1, dtype=bool)
    g.set_labels(lm, {0: (NODE_TYPE, "A")})

    model = _make_model(
        node_types={
            "A": NodeTypeSpecStub(
                distinct_nodes=[], same_node=["location"], self_relations=[],
            ),
        },
    )
    with pytest.raises(KeyError, match="location"):
        SpectralClusteringPartitioner(g, model)


def test_partition_rejects_zero_target() -> None:
    g = _make_two_cluster_graph()
    p = SpectralClusteringPartitioner(g, _make_model())
    with pytest.raises(ValueError):
        p.partition(target_partition_count=0)


# ------------------------------------------------------------------ #
# Integration: partition()
# ------------------------------------------------------------------ #


def test_single_partition_puts_everything_in_one() -> None:
    g = _make_two_cluster_graph()
    part = SpectralClusteringPartitioner(g, _make_model()).partition(1)

    # Two node instances (A-north, A-south), both in partition 0.
    assert len(part.node_specs) == 2
    assert all(ns.partition == 0 for ns in part.node_specs)

    names = {ns.node_name for ns in part.node_specs}
    assert "p0-A-north" in names
    assert "p0-A-south" in names

    # All 6 vertices accounted for, no overlap.
    all_verts: set[int] = set()
    for ns in part.node_specs:
        verts = _coo_indices(part.assignment_vector(ns.node_name))
        assert not (all_verts & verts)
        all_verts |= verts
    assert all_verts == set(range(6))


def test_two_disconnected_clusters_into_two_partitions() -> None:
    g = _make_two_cluster_graph()
    part = SpectralClusteringPartitioner(g, _make_model()).partition(2)

    assert len(part.node_specs) == 2

    # Each node instance in a different partition.
    partitions = {ns.partition for ns in part.node_specs}
    assert len(partitions) == 2

    # Vertex assignment matches cluster membership.
    for ns in part.node_specs:
        verts = _coo_indices(part.assignment_vector(ns.node_name))
        if "north" in ns.node_name:
            assert verts == {0, 1, 2}
        else:
            assert verts == {3, 4, 5}


def test_four_components_combined_into_two_partitions() -> None:
    """Four equal-weight components are balanced across two partitions."""
    g = _make_four_pair_graph()
    part = SpectralClusteringPartitioner(g, _make_model()).partition(2)

    assert len(part.node_specs) == 4

    pv = _partition_vertex_sets(part)
    assert len(pv) == 2

    # No overlap, full coverage.
    all_verts: set[int] = set()
    for verts in pv.values():
        assert not (all_verts & verts)
        all_verts |= verts
    assert all_verts == set(range(8))

    # Balanced: each partition gets 2 pairs = 4 vertices.
    sizes = sorted(len(v) for v in pv.values())
    assert sizes == [4, 4]


def test_barbell_split_into_two_partitions() -> None:
    """A single large component is split along the weak bridge."""
    g = _make_barbell_graph()
    part = SpectralClusteringPartitioner(g, _make_model()).partition(2)

    assert len(part.node_specs) == 2

    for ns in part.node_specs:
        verts = _coo_indices(part.assignment_vector(ns.node_name))
        if "north" in ns.node_name:
            assert verts == {0, 1, 2, 3}
        else:
            assert verts == {4, 5, 6, 7}

    partitions = {ns.partition for ns in part.node_specs}
    assert len(partitions) == 2


def test_incidence_shape_matches_node_specs_and_graph() -> None:
    g = _make_two_cluster_graph()
    part = SpectralClusteringPartitioner(g, _make_model()).partition(2)

    assert int(part.incidence.nrows) == len(part.node_specs)
    assert int(part.incidence.ncols) == g.num_vertices


# ------------------------------------------------------------------ #
# Unit: _get_connected_components
# ------------------------------------------------------------------ #


def test_get_connected_components_mixed() -> None:
    """One connected pair and two isolated vertices."""
    E = gb.Matrix.from_coo(
        [0, 1], [1, 0], [1.0, 1.0],
        nrows=4, ncols=4, dtype=gb.dtypes.FP64,
    )
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    cc = SpectralClusteringPartitioner._get_connected_components(E, weights)

    comp_map = {frozenset(c.tolist()): w for w, c in cc}
    assert len(comp_map) == 3
    assert comp_map[frozenset({0, 1})] == pytest.approx(3.0)
    assert comp_map[frozenset({2})] == pytest.approx(3.0)
    assert comp_map[frozenset({3})] == pytest.approx(4.0)


def test_get_connected_components_all_isolated() -> None:
    """Empty matrix: every vertex is its own component."""
    E = gb.Matrix(nrows=3, ncols=3, dtype=gb.dtypes.FP64)
    weights = np.array([1.0, 2.0, 3.0])

    cc = SpectralClusteringPartitioner._get_connected_components(E, weights)

    assert len(cc) == 3
    assert all(c.size == 1 for _, c in cc)


def test_get_connected_components_single_component() -> None:
    """Fully connected → one component."""
    E = gb.Matrix.from_coo(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
        [1.0] * 6,
        nrows=3, ncols=3, dtype=gb.dtypes.FP64,
    )
    weights = np.array([1.0, 1.0, 1.0])

    cc = SpectralClusteringPartitioner._get_connected_components(E, weights)

    assert len(cc) == 1
    assert set(cc[0][1].tolist()) == {0, 1, 2}
    assert cc[0][0] == pytest.approx(3.0)


# ------------------------------------------------------------------ #
# Unit: _split_component
# ------------------------------------------------------------------ #


def test_split_component_separates_barbell() -> None:
    """Two triangles joined by a weak bridge split into two clusters."""
    E = gb.Matrix(nrows=6, ncols=6, dtype=gb.dtypes.FP64)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        E[i, j] = 10.0
        E[j, i] = 10.0
    for i, j in [(3, 4), (3, 5), (4, 5)]:
        E[i, j] = 10.0
        E[j, i] = 10.0
    E[2, 3] = 0.01
    E[3, 2] = 0.01

    component = np.arange(6, dtype=np.int64)
    weights = np.ones(6)

    # target_weight=4 → n_clusters = max(2, round(6/4)) = 2
    result = SpectralClusteringPartitioner._split_component(
        E, component, weights, target_weight=4.0,
    )

    assert len(result) == 2
    comp_sets = [set(c.tolist()) for _, c in result]
    assert {0, 1, 2} in comp_sets
    assert {3, 4, 5} in comp_sets


def test_split_component_preserves_total_weight() -> None:
    """All original vertices appear in exactly one output cluster."""
    E = gb.Matrix(nrows=6, ncols=6, dtype=gb.dtypes.FP64)
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
        E[i, j] = 1.0
        E[j, i] = 1.0

    component = np.arange(6, dtype=np.int64)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = SpectralClusteringPartitioner._split_component(
        E, component, weights, target_weight=11.0,
    )

    all_verts = set()
    total_weight = 0.0
    for w, c in result:
        verts = set(c.tolist())
        assert not (all_verts & verts), "overlap between clusters"
        all_verts |= verts
        assert w == pytest.approx(weights[c].sum())
        total_weight += w

    assert all_verts == set(range(6))
    assert total_weight == pytest.approx(21.0)


# ------------------------------------------------------------------ #
# Unit: _combine_components
# ------------------------------------------------------------------ #

_combine = SpectralClusteringPartitioner._combine_components


def test_combine_single_partition() -> None:
    components = [
        (3.0, np.array([0, 1, 2])),
        (2.0, np.array([3, 4])),
    ]
    result = _combine(components, 1, 10.0)

    assert len(result) == 1
    assert set(result[0].tolist()) == {0, 1, 2, 3, 4}


def test_combine_balanced_assignment() -> None:
    """Three components [5, 3, 2] into 2 partitions: expect [5] and [3+2]."""
    components = [
        (5.0, np.array([0, 1, 2])),
        (3.0, np.array([3, 4])),
        (2.0, np.array([5])),
    ]
    result = _combine(components, 2, 10.0)

    assert len(result) == 2
    weights = sorted(
        sum(c[0] for c in components if set(c[1].tolist()) <= set(r.tolist()))
        for r in result
    )
    assert weights == pytest.approx([5.0, 5.0])


def test_combine_more_partitions_than_components() -> None:
    """Excess partitions are empty arrays."""
    components = [(2.0, np.array([0, 1]))]
    result = _combine(components, 3, 10.0)

    assert len(result) == 3
    non_empty = [r for r in result if r.size > 0]
    assert len(non_empty) == 1
    assert set(non_empty[0].tolist()) == {0, 1}


def test_combine_respects_target_weight() -> None:
    """No partition exceeds target_weight when feasible."""
    components = [
        (4.0, np.array([0])),
        (4.0, np.array([1])),
        (3.0, np.array([2])),
        (3.0, np.array([3])),
    ]
    result = _combine(components, 2, 8.0)

    for r in result:
        w = sum(c[0] for c in components if set(c[1].tolist()) <= set(r.tolist()))
        assert w <= 8.0 + 1e-9


def test_combine_all_vertices_present() -> None:
    """Every component vertex appears in exactly one output partition."""
    components = [
        (7.0, np.array([0, 1, 2])),
        (5.0, np.array([3, 4])),
        (3.0, np.array([5, 6])),
        (1.0, np.array([7])),
    ]
    result = _combine(components, 3, 20.0)

    all_verts: set[int] = set()
    for r in result:
        verts = set(r.tolist())
        assert not (all_verts & verts), "vertex appears in multiple partitions"
        all_verts |= verts
    assert all_verts == set(range(8))


def test_combine_swap_does_not_overshoot() -> None:
    """
    Pairwise swap must not worsen the spread.

    LPT produces p0=[8,3]=11, p1=[5,4]=9 (spread 2).
    A swap of 8↔5 would give p0=8, p1=12 (spread 4) — strictly worse.
    The abs() guard in the swap condition must prevent this.

    Without the fix this test would hang (infinite oscillation).
    """
    components = [
        (8.0, np.array([0])),
        (5.0, np.array([1])),
        (4.0, np.array([2])),
        (3.0, np.array([3])),
    ]
    result = _combine(components, 2, 20.0)

    assert len(result) == 2
    p_weights = sorted(
        sum(c[0] for c in components if set(c[1].tolist()) <= set(r.tolist()))
        for r in result
    )
    # LPT gives [11, 9]; no valid refinement can improve this.
    assert p_weights == pytest.approx([9.0, 11.0])
