from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import graphblas as gb
import pytest

from disco.partitioning import NodeInstanceSpec, NodeTopology, Partitioning


class FakeMetastore:
    """
    Minimal in-memory metastore for Partitioning.store/load tests.

    Partitioning only needs update_key() and get_key().
    """

    def __init__(self) -> None:
        self._kv: Dict[str, Any] = {}

    def update_key(self, path: str, value: Any) -> None:
        self._kv[path] = value

    def get_key(self, path: str) -> Any | None:
        return self._kv.get(path)


@dataclass(frozen=True, slots=True)
class NodeTypeSpecStub:
    self_relations: List[Tuple[str, str]]


@dataclass(frozen=True, slots=True)
class ModelSpecStub:
    # Per your contract: list[str] already ordered by layer.
    simprocs: List[str]
    node_types: Dict[str, NodeTypeSpecStub]


@dataclass(frozen=True, slots=True)
class ModelStub:
    spec: ModelSpecStub


@dataclass(frozen=True, slots=True)
class GraphStub:
    scenario_id: str
    num_vertices: int
    layers: List[int]
    matrices: Dict[int, gb.Matrix]

    def get_matrix(self, layer: int) -> gb.Matrix:
        return self.matrices[layer]


def _incidence_from_rows(n_vertices: int, rows: List[Sequence[int]]) -> gb.Matrix:
    # rows[i] are vertex indices assigned to node row i
    I: List[int] = []
    J: List[int] = []
    X: List[bool] = []
    for r, cols in enumerate(rows):
        for c in cols:
            I.append(r)
            J.append(int(c))
            X.append(True)
    return gb.Matrix.from_coo(I, J, X, nrows=len(rows), ncols=n_vertices, dtype=bool)


def _empty_topology(node_name: str, node_type: str) -> NodeTopology:
    return NodeTopology(
        node=node_name,
        node_type=node_type,
        self_relations=[],
        predecessors={},
        successors={},
    )


def _coo_structure(m: gb.Matrix) -> set[tuple[int, int]]:
    r, c, _ = m.to_coo()
    return set(zip(r.tolist(), c.tolist()))


def _coo_indices(v: gb.Vector) -> set[int]:
    idx, _ = v.to_coo()
    return set(idx.tolist())


def _node_indices_for_specs(specs: Sequence[NodeInstanceSpec]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, ns in enumerate(specs):
        if ns.node_name in out:
            raise ValueError(f"duplicate node_name in test node_specs: {ns.node_name}")
        out[ns.node_name] = i
    return out


def test_partitioning_post_init_validates_and_allows_lookup() -> None:
    n_vertices = 5
    nodes = (
        NodeInstanceSpec(partition=0, node_name="p0-A-x", node_type="A"),
        NodeInstanceSpec(partition=1, node_name="p1-A-y", node_type="A"),
        NodeInstanceSpec(partition=0, node_name="p0-B-z", node_type="B"),
    )
    incidence = _incidence_from_rows(n_vertices, rows=[[0, 1], [2], [3, 4]])
    topo = {n.node_name: _empty_topology(n.node_name, n.node_type) for n in nodes}
    node_indices = _node_indices_for_specs(nodes)

    p = Partitioning(
        partitioning_id="pid",
        scenario_id="scenario-1",
        num_partitions=2,
        node_specs=nodes,
        incidence=incidence,
        topology_by_node=topo,
        node_indices=node_indices,
    )

    # Derived lookups
    assert p.node_index("p1-A-y") == 1
    assert p.node_spec("p0-B-z").node_type == "B"
    assert tuple(ns.node_name for ns in p.node_specs) == ("p0-A-x", "p1-A-y", "p0-B-z")


def test_partitioning_post_init_validates_incidence_and_topology() -> None:
    n_vertices = 3
    nodes = (NodeInstanceSpec(partition=0, node_name="p0-A", node_type="A"),)
    node_indices = _node_indices_for_specs(nodes)

    incidence_bad_dtype = gb.Matrix.from_coo([0], [0], [1.0], nrows=1, ncols=n_vertices, dtype=float)
    topo_ok = {"p0-A": _empty_topology("p0-A", "A")}

    with pytest.raises(TypeError, match="incidence matrix must have BOOL dtype"):
        Partitioning(
            partitioning_id="pid",
            scenario_id="s",
            num_partitions=1,
            node_specs=nodes,
            incidence=incidence_bad_dtype,
            topology_by_node=topo_ok,
            node_indices=node_indices,
        )

    incidence_ok = _incidence_from_rows(n_vertices, rows=[[0]])
    topo_missing: Dict[str, NodeTopology] = {}
    with pytest.raises(ValueError, match="missing topology for node"):
        Partitioning(
            partitioning_id="pid",
            scenario_id="s",
            num_partitions=1,
            node_specs=nodes,
            incidence=incidence_ok,
            topology_by_node=topo_missing,
            node_indices=node_indices,
        )

    topo_mismatch = {"p0-A": _empty_topology("p0-A", "WRONG")}
    with pytest.raises(ValueError, match="topology.node_type"):
        Partitioning(
            partitioning_id="pid",
            scenario_id="s",
            num_partitions=1,
            node_specs=nodes,
            incidence=incidence_ok,
            topology_by_node=topo_mismatch,
            node_indices=node_indices,
        )


def test_assignment_vector_slices_incidence_row() -> None:
    n_vertices = 6
    nodes = (
        NodeInstanceSpec(partition=0, node_name="p0-A", node_type="A"),
        NodeInstanceSpec(partition=0, node_name="p0-B", node_type="B"),
    )
    node_indices = _node_indices_for_specs(nodes)

    incidence = _incidence_from_rows(n_vertices, rows=[[1, 3, 5], [0, 2]])
    topo = {n.node_name: _empty_topology(n.node_name, n.node_type) for n in nodes}

    p = Partitioning(
        partitioning_id="pid",
        scenario_id="s",
        num_partitions=1,
        node_specs=nodes,
        incidence=incidence,
        topology_by_node=topo,
        node_indices=node_indices,
    )

    v = p.assignment_vector("p0-A")
    assert _coo_indices(v) == {1, 3, 5}


def test_compute_topology_from_graph_adds_inter_node_and_self_relations() -> None:
    # 5 vertices, 3 nodes:
    #   n0 owns {0,1}
    #   n1 owns {2}
    #   n2 owns {3,4}
    n_vertices = 5
    nodes = (
        NodeInstanceSpec(partition=0, node_name="p0-A-x", node_type="A"),
        NodeInstanceSpec(partition=1, node_name="p1-A-y", node_type="A"),
        NodeInstanceSpec(partition=0, node_name="p0-B-z", node_type="B"),
    )
    incidence = _incidence_from_rows(n_vertices, rows=[[0, 1], [2], [3, 4]])

    # Layer 0 edges: 0->2, 1->3  => n0 has successors n1 and n2 for simproc "s0"
    M0 = gb.Matrix.from_coo([0, 1], [2, 3], [1.0, 1.0], nrows=n_vertices, ncols=n_vertices, dtype=float)
    # Layer 1 edges: 3->0        => n2 has successor n0 for simproc "s1"
    M1 = gb.Matrix.from_coo([3], [0], [1.0], nrows=n_vertices, ncols=n_vertices, dtype=float)

    graph = GraphStub(
        scenario_id="scenario-1",
        num_vertices=n_vertices,
        layers=[0, 1],
        matrices={0: M0, 1: M1},
    )

    # Model: simprocs is already in layer order; node type A has a self-relation s0->s1
    model = ModelStub(
        spec=ModelSpecStub(
            simprocs=["s0", "s1"],
            node_types={
                "A": NodeTypeSpecStub(self_relations=[("s0", "s1")]),
                "B": NodeTypeSpecStub(self_relations=[]),
            },
        )
    )

    computed = Partitioning.compute_topology_from_graph(
        node_specs=nodes,
        incidence=incidence,
        graph=graph,  # structural stub with get_matrix/layers/num_vertices
        model=model,
    )

    n0 = "p0-A-x"
    n1 = "p1-A-y"
    n2 = "p0-B-z"

    # n0 successors on s0: (n1,s0), (n2,s0), plus self relation to (n0,s1)
    assert computed[n0].successors["s0"] == {(n1, "s0"), (n2, "s0"), (n0, "s1")}
    # n1 predecessors on s0: (n0,s0)
    assert computed[n1].predecessors["s0"] == {(n0, "s0")}
    # n2 predecessors on s0: (n0,s0)
    assert computed[n2].predecessors["s0"] == {(n0, "s0")}

    # Layer 1: n2 -> n0
    assert computed[n2].successors["s1"] == {(n0, "s1")}
    assert (n2, "s1") in computed[n0].predecessors["s1"]

    # Self-relation exists for A nodes: predecessor of s1 includes (self, s0)
    assert (n0, "s0") in computed[n0].predecessors["s1"]
    assert (n1, "s0") in computed[n1].predecessors["s1"]

    # B nodes have no self-relations
    assert computed[n2].self_relations == []


def test_store_and_load_round_trip_full() -> None:
    n_vertices = 5
    nodes = (
        NodeInstanceSpec(partition=0, node_name="p0-A-x", node_type="A"),
        NodeInstanceSpec(partition=1, node_name="p1-A-y", node_type="A"),
        NodeInstanceSpec(partition=0, node_name="p0-B-z", node_type="B"),
    )
    node_indices = _node_indices_for_specs(nodes)
    incidence = _incidence_from_rows(n_vertices, rows=[[0, 1], [2], [3, 4]])

    topo = {
        n.node_name: NodeTopology(
            node=n.node_name,
            node_type=n.node_type,
            self_relations=[],
            predecessors={"s0": set()},
            successors={"s0": set()},
        )
        for n in nodes
    }

    p = Partitioning(
        partitioning_id="pid",
        scenario_id="scenario-1",
        num_partitions=2,
        affinity_by_partition={0: "a0", 1: "a1"},
        node_specs=nodes,
        incidence=incidence,
        topology_by_node=topo,
        node_indices=node_indices,
    )

    ms = FakeMetastore()
    p.store(ms)  # FakeMetastore matches the required methods

    # Load requires graph for num_vertices; matrices/layers unused during load
    graph = GraphStub(
        scenario_id="scenario-1",
        num_vertices=n_vertices,
        layers=[],
        matrices={},
    )

    loaded = Partitioning.load(ms, "pid", graph=graph)

    assert loaded.partitioning_id == "pid"
    assert loaded.scenario_id == "scenario-1"
    assert loaded.num_partitions == 2

    assert tuple(ns.node_name for ns in loaded.node_specs) == tuple(ns.node_name for ns in p.node_specs)
    assert _coo_structure(loaded.incidence) == _coo_structure(p.incidence)

    # topology round-trip
    assert loaded.topology_by_node["p0-A-x"].node_type == "A"
