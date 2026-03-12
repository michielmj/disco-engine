# tests/partitioner/test_spectral_partitioner.py
"""
Tests for SpectralClusteringPartitioner.

Graph topology used in most tests
----------------------------------
8 vertices, 2 node-types (A, B), 2 regions (north, south).
Each node type uses its own region label type to avoid cross-type merging
when compress() is called:

  node instance  | vertices | vertex_weight
  A-north        | 0, 1     | 1.0, 1.0
  A-south        | 2, 3     | 1.0, 1.0
  B-north        | 4, 5     | 1.0, 1.0
  B-south        | 6, 7     | 1.0, 1.0

Label columns:
  0: (NODE_TYPE, "A")
  1: (NODE_TYPE, "B")
  2: ("region-A", "north")   -- only A vertices carry region-A
  3: ("region-A", "south")
  4: ("region-B", "north")   -- only B vertices carry region-B
  5: ("region-B", "south")

compress(["region-A", "region-B"]) produces 4 non-overlapping supervertices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import graphblas as gb
import pytest

from disco.graph import Graph
from disco.partitioner import SpectralClusteringPartitioner, NODE_TYPE


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_model(
    simprocs: List[str] | None = None,
    dn_a: List[str] | None = None,
    sn_a: List[str] | None = None,
    dn_b: List[str] | None = None,
    sn_b: List[str] | None = None,
) -> ModelStub:
    """Model with node types A and B, each using per-type region label types."""
    sp: List[str] = simprocs if simprocs is not None else []
    dn_a = dn_a if dn_a is not None else ["region-A"]
    sn_a = sn_a if sn_a is not None else ["region-A"]
    dn_b = dn_b if dn_b is not None else ["region-B"]
    sn_b = sn_b if sn_b is not None else ["region-B"]
    return ModelStub(
        spec=ModelSpecStub(
            simprocs=sp,
            node_types={
                "A": NodeTypeSpecStub(distinct_nodes=dn_a, same_node=sn_a, self_relations=[]),
                "B": NodeTypeSpecStub(distinct_nodes=dn_b, same_node=sn_b, self_relations=[]),
            },
        )
    )


def _make_labeled_graph(
    num_vertices: int,
    label_rows: List[int],
    label_cols: List[int],
    label_meta: Dict[int, Tuple[str, str]],
    edge_layers: Dict[int, Tuple[List[int], List[int], List[float]]] | None = None,
    vertex_weight: np.ndarray | None = None,
) -> Graph:
    label_matrix = gb.Matrix.from_coo(
        label_rows,
        label_cols,
        [True] * len(label_rows),
        nrows=num_vertices,
        ncols=len(label_meta),
        dtype=bool,
    )
    layers: Tuple[gb.Matrix, ...] = tuple()
    if edge_layers:
        # Build layers sorted by index
        max_layer = max(edge_layers.keys())
        layers = tuple(
            gb.Matrix.from_coo(
                np.asarray(edge_layers[i][0], dtype=np.int64),
                np.asarray(edge_layers[i][1], dtype=np.int64),
                np.asarray(edge_layers[i][2], dtype=np.float64),
                nrows=num_vertices,
                ncols=num_vertices,
            )
            if i in edge_layers
            else gb.Matrix(float, nrows=num_vertices, ncols=num_vertices)
            for i in range(max_layer + 1)
        )

    g = Graph(
        layers=layers,
        num_vertices=num_vertices,
        scenario_id="test-scenario",
        vertex_weight=vertex_weight,
    )
    g.set_labels(label_matrix=label_matrix, label_meta=label_meta)
    return g


def _four_instance_graph(edge_layers: Dict | None = None, vertex_weight: np.ndarray | None = None) -> Graph:
    """
    8 vertices with per-node-type region label types:
      0,1: A-north   label ids: 0=node-type/A, 2=region-A/north
      2,3: A-south   label ids: 0=node-type/A, 3=region-A/south
      4,5: B-north   label ids: 1=node-type/B, 4=region-B/north
      6,7: B-south   label ids: 1=node-type/B, 5=region-B/south
    """
    meta: Dict[int, Tuple[str, str]] = {
        0: (NODE_TYPE, "A"),
        1: (NODE_TYPE, "B"),
        2: ("region-A", "north"),
        3: ("region-A", "south"),
        4: ("region-B", "north"),
        5: ("region-B", "south"),
    }
    rows = []
    cols = []
    # A-north: vertices 0,1  → node-type/A (col 0) + region-A/north (col 2)
    for v in (0, 1):
        rows += [v, v]; cols += [0, 2]
    # A-south: vertices 2,3  → node-type/A (col 0) + region-A/south (col 3)
    for v in (2, 3):
        rows += [v, v]; cols += [0, 3]
    # B-north: vertices 4,5  → node-type/B (col 1) + region-B/north (col 4)
    for v in (4, 5):
        rows += [v, v]; cols += [1, 4]
    # B-south: vertices 6,7  → node-type/B (col 1) + region-B/south (col 5)
    for v in (6, 7):
        rows += [v, v]; cols += [1, 5]

    return _make_labeled_graph(
        num_vertices=8,
        label_rows=rows,
        label_cols=cols,
        label_meta=meta,
        edge_layers=edge_layers,
        vertex_weight=vertex_weight,
    )


def _coo_indices(v: gb.Vector) -> set[int]:
    idx, _ = v.select("==", True).to_coo()
    return set(idx.tolist())


def _all_assigned_vertex_indices(part) -> set[int]:  # type: ignore[no-untyped-def]
    """Union of all vertex indices assigned to any node in the partitioning."""
    result: set[int] = set()
    for ns in part.node_specs:
        result.update(_coo_indices(part.assignment_vector(ns.node_name)))
    return result


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_scp_single_partition_when_target_is_1() -> None:
    graph = _four_instance_graph()
    model = _make_model(simprocs=[])
    part = SpectralClusteringPartitioner(graph=graph, model=model).partition(
        target_partition_count=1
    )
    assert part.num_partitions == 1
    assert part.scenario_id == "test-scenario"
    assert len(part.node_specs) == 4  # 4 node instances


def test_scp_respects_target_partition_count() -> None:
    """num_partitions must be in [1, target_partition_count]."""
    # Build graph with inter-instance edges so clustering can split
    edge_layers = {
        0: (
            [0, 1, 2, 4, 5, 6],  # src
            [2, 3, 4, 5, 6, 7],  # dst
            [10.0, 10.0, 5.0, 5.0, 5.0, 5.0],
        )
    }
    graph = _four_instance_graph(edge_layers=edge_layers)
    model = _make_model(simprocs=["sp0"])

    for target in (1, 2, 3, 4):
        part = SpectralClusteringPartitioner(graph=graph, model=model).partition(
            target_partition_count=target
        )
        assert 1 <= part.num_partitions <= target, (
            f"target={target} → num_partitions={part.num_partitions}"
        )


def test_scp_all_vertices_assigned_exactly_once() -> None:
    """Every vertex must appear in exactly one node instance's incidence row."""
    edge_layers = {
        0: ([0, 2, 4], [2, 4, 6], [5.0, 5.0, 5.0])
    }
    graph = _four_instance_graph(edge_layers=edge_layers)
    model = _make_model(simprocs=["sp0"])

    part = SpectralClusteringPartitioner(graph=graph, model=model).partition(
        target_partition_count=2
    )

    all_assigned = _all_assigned_vertex_indices(part)
    assert all_assigned == set(range(graph.num_vertices))

    # No vertex should appear in two different nodes
    seen: set[int] = set()
    for ns in part.node_specs:
        node_verts = _coo_indices(part.assignment_vector(ns.node_name))
        assert seen.isdisjoint(node_verts), f"Vertex overlap at {ns.node_name}"
        seen.update(node_verts)


def test_scp_heavier_vertices_spread_across_partitions() -> None:
    """
    Two heavy node instances (A-north, B-north, weight=10) and two light ones
    (A-south, B-south, weight=1) with cross-instance edges.
    With target=2, the SCP should produce 2 partitions with roughly balanced weight.
    """
    # Vertex weights: A-north=10,10; A-south=1,1; B-north=10,10; B-south=1,1
    vw = np.array([10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0])

    # Strong edges between all node instances to make them connected
    edge_layers = {
        0: ([0, 2, 4, 0, 4], [2, 4, 6, 6, 2], [1.0, 1.0, 1.0, 1.0, 1.0])
    }
    graph = _four_instance_graph(edge_layers=edge_layers, vertex_weight=vw)
    model = _make_model(simprocs=["sp0"])

    part = SpectralClusteringPartitioner(graph=graph, model=model).partition(
        target_partition_count=2
    )

    assert 1 <= part.num_partitions <= 2

    if part.num_partitions == 2:
        # Compute weight per partition
        weights_per_partition: dict[int, float] = {}
        for ns in part.node_specs:
            node_verts = list(_coo_indices(part.assignment_vector(ns.node_name)))
            w = float(vw[node_verts].sum())
            weights_per_partition[ns.partition] = (
                weights_per_partition.get(ns.partition, 0.0) + w
            )
        # Both partitions should have substantial weight (not all in one)
        min_w = min(weights_per_partition.values())
        total_w = sum(weights_per_partition.values())
        assert min_w / total_w >= 0.2, (
            f"Partitions too unbalanced: {weights_per_partition}"
        )


def test_scp_rejects_zero_target() -> None:
    graph = _four_instance_graph()
    model = _make_model(simprocs=[])
    with pytest.raises(ValueError):
        SpectralClusteringPartitioner(graph=graph, model=model).partition(
            target_partition_count=0
        )


def test_scp_no_edges_produces_valid_partitioning() -> None:
    """Graph with no edges: all node instances are disconnected, still partitions."""
    graph = _four_instance_graph(edge_layers=None)
    model = _make_model(simprocs=[])
    part = SpectralClusteringPartitioner(graph=graph, model=model).partition(
        target_partition_count=2
    )
    assert 1 <= part.num_partitions <= 2
    # All vertices should still be assigned
    all_assigned = _all_assigned_vertex_indices(part)
    assert all_assigned == set(range(graph.num_vertices))


def test_scp_requires_labels() -> None:
    """Construction should fail if graph has no labels attached."""
    g = Graph(layers=tuple(), num_vertices=4, scenario_id="s")
    model = _make_model()
    with pytest.raises(ValueError):
        SpectralClusteringPartitioner(graph=g, model=model)


def test_scp_requires_node_type_label() -> None:
    """Construction should fail if NODE_TYPE label type is missing from graph."""
    g = Graph(layers=tuple(), num_vertices=2, scenario_id="s")
    label_meta = {0: ("region", "north")}
    label_matrix = gb.Matrix.from_coo(
        [0, 1], [0, 0], [True, True], nrows=2, ncols=1, dtype=bool
    )
    g.set_labels(label_matrix=label_matrix, label_meta=label_meta)
    model = _make_model()
    with pytest.raises(KeyError):
        SpectralClusteringPartitioner(graph=g, model=model)


def test_scp_single_instance_always_partition_zero() -> None:
    """If the graph has only one node instance, it goes to partition 0."""
    meta: Dict[int, Tuple[str, str]] = {0: (NODE_TYPE, "A")}
    rows = [0, 1, 2]
    cols = [0, 0, 0]
    label_matrix = gb.Matrix.from_coo(rows, cols, [True] * 3, nrows=3, ncols=1, dtype=bool)
    g = Graph(layers=tuple(), num_vertices=3, scenario_id="s")
    g.set_labels(label_matrix=label_matrix, label_meta=meta)

    model = ModelStub(
        spec=ModelSpecStub(
            simprocs=[],
            node_types={
                "A": NodeTypeSpecStub(
                    distinct_nodes=[],
                    same_node=[NODE_TYPE],
                    self_relations=[],
                ),
            },
        )
    )

    part = SpectralClusteringPartitioner(graph=g, model=model).partition(
        target_partition_count=3
    )
    assert part.num_partitions == 1
    assert all(ns.partition == 0 for ns in part.node_specs)
