# tests/test_partitioner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import graphblas as gb
import pytest

from disco.graph.core import Graph
from disco.partitioner import NODE_TYPE, SimplePartitioner


@dataclass(frozen=True, slots=True)
class NodeTypeSpecStub:
    # Distinct label types (strings) that must be constant within a node instance.
    distinct_nodes: List[str]
    # Intra-node edges between simprocs (not used in these tests; keep empty).
    self_relations: List[Tuple[str, str]]


@dataclass(frozen=True, slots=True)
class ModelSpecStub:
    # IMPORTANT: already ordered by layer (Partitioning expects this contract).
    simprocs: List[str]
    node_types: Dict[str, NodeTypeSpecStub]


@dataclass(frozen=True, slots=True)
class ModelStub:
    spec: ModelSpecStub


def _bool_vec_true_at(size: int, true_indices: List[int]) -> gb.Vector:
    if not true_indices:
        return gb.Vector(bool, size=size)
    return gb.Vector.from_coo(true_indices, [True] * len(true_indices), size=size, dtype=bool)


def _make_graph_two_node_types_same_region() -> Graph:
    """
    Build a graph with 4 vertices and labels:

      vertex 0,1: node-type=A, region=north
      vertex 2,3: node-type=B, region=north

    If NODE_TYPE is included in the distinct label list, we should get two nodes:
      (A,north) and (B,north).
    If NODE_TYPE is NOT included, we'd only get one node (north), which is wrong.
    """
    num_vertices = 4
    g = Graph(layers=tuple(), num_vertices=num_vertices, scenario_id="scenario-1")

    # label ids:
    #   0: (node-type, A)
    #   1: (node-type, B)
    #   2: (region, north)
    label_meta: Dict[int, Tuple[str, str]] = {
        0: (NODE_TYPE, "A"),
        1: (NODE_TYPE, "B"),
        2: ("region", "north"),
    }

    # label_matrix is (num_vertices x num_labels)
    # rows=vertices, cols=label ids
    I: List[int] = []
    J: List[int] = []
    X: List[bool] = []

    # v0,v1 are A + north
    for v in (0, 1):
        I.extend([v, v])
        J.extend([0, 2])
        X.extend([True, True])

    # v2,v3 are B + north
    for v in (2, 3):
        I.extend([v, v])
        J.extend([1, 2])
        X.extend([True, True])

    label_matrix = gb.Matrix.from_coo(I, J, X, nrows=num_vertices, ncols=3, dtype=bool)

    label_type_vectors: Dict[str, gb.Vector] = {
        NODE_TYPE: _bool_vec_true_at(3, [0, 1]),
        "region": _bool_vec_true_at(3, [2]),
    }

    g.set_labels(label_matrix=label_matrix, label_meta=label_meta, label_type_vectors=label_type_vectors)
    return g


def _make_model_distinct_region_for_a_and_b() -> ModelStub:
    """
    Both node types A and B require 'region' as a distinct label.
    The partitioner must also include NODE_TYPE in the distinct list,
    otherwise it would merge A and B nodes that share the same region.
    """
    return ModelStub(
        spec=ModelSpecStub(
            simprocs=[],
            node_types={
                "A": NodeTypeSpecStub(distinct_nodes=["region"], self_relations=[]),
                "B": NodeTypeSpecStub(distinct_nodes=["region"], self_relations=[]),
            },
        )
    )


def _coo_indices(v: gb.Vector) -> set[int]:
    idx, vals = v.select("==", True).to_coo()
    return set(idx.tolist())


def test_simple_partitioner_splits_by_node_type_and_distinct_labels() -> None:
    graph = _make_graph_two_node_types_same_region()
    model = _make_model_distinct_region_for_a_and_b()

    part = SimplePartitioner(graph=graph, model=model).partition(target_partition_count=1)

    # Always 1 partition for SimplePartitioner
    assert part.num_partitions == 1
    assert part.scenario_id == "scenario-1"

    # Must produce two nodes: (A,north) and (B,north)
    assert len(part.node_specs) == 2

    node_types = {ns.node_type for ns in part.node_specs}
    assert node_types == {"A", "B"}

    # Names should include label values; order depends on distinct ordering but should be stable.
    node_names = {ns.node_name for ns in part.node_specs}
    assert "p0-A-north" in node_names
    assert "p0-B-north" in node_names

    # Incidence must be shaped (n_nodes x n_vertices) so assignment_vector works.
    assert int(part.incidence.nrows) == len(part.node_specs)
    assert int(part.incidence.ncols) == graph.num_vertices

    # Check vertex ownership
    a_name = next(ns.node_name for ns in part.node_specs if ns.node_type == "A")
    b_name = next(ns.node_name for ns in part.node_specs if ns.node_type == "B")

    assert _coo_indices(part.assignment_vector(a_name)) == {0, 1}
    assert _coo_indices(part.assignment_vector(b_name)) == {2, 3}


def test_simple_partitioner_requires_label_types_in_graph() -> None:
    """
    If the graph is missing one of the required label_type_vectors, construction should fail.

    This test will catch missing NODE_TYPE handling as well:
    - If SimplePartitioner includes NODE_TYPE in distinct, it must validate NODE_TYPE exists.
    """
    num_vertices = 1
    g = Graph(layers=tuple(), num_vertices=num_vertices, scenario_id="s")

    # Minimal labels with only region type vector (missing NODE_TYPE)
    label_meta = {0: ("region", "north")}
    label_matrix = gb.Matrix.from_coo([0], [0], [True], nrows=num_vertices, ncols=1, dtype=bool)
    label_type_vectors = {"region": _bool_vec_true_at(1, [0])}
    g.set_labels(label_matrix=label_matrix, label_meta=label_meta, label_type_vectors=label_type_vectors)

    model = _make_model_distinct_region_for_a_and_b()

    # We expect a KeyError mentioning the missing label type
    with pytest.raises(KeyError):
        SimplePartitioner(graph=g, model=model)


def test_simple_partitioner_rejects_zero_target_partition_count() -> None:
    """
    Partitioners should not accept nonsensical target counts.

    Even though SimplePartitioner always returns 1 partition, the caller contract
    still requires target_partition_count >= 1.
    """
    graph = _make_graph_two_node_types_same_region()
    model = _make_model_distinct_region_for_a_and_b()

    p = SimplePartitioner(graph=graph, model=model)
    with pytest.raises(ValueError):
        p.partition(target_partition_count=0)
