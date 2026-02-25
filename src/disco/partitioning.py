# src/disco/partitioning.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Set, Tuple, Self
from uuid import uuid4

import graphblas as gb

from .exceptions import DiscoError, DiscoRuntimeError
from .graph import Graph
from .metastore import Metastore
from .model import Model

PARTITIONINGS: str = "partitionings"


@dataclass(frozen=True, slots=True)
class NodeInstanceSpec:
    partition: int
    node_name: str
    node_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition": self.partition,
            "node_name": self.node_name,
            "node_type": self.node_type,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> NodeInstanceSpec:
        partition = int(d["partition"])
        node_name = str(d["node_name"])
        node_type = str(d["node_type"])
        return NodeInstanceSpec(partition=partition, node_name=node_name, node_type=node_type)


@dataclass(frozen=True, slots=True)
class NodeTopology:
    node: str
    node_type: str
    self_relations: List[Tuple[str, str]]
    predecessors: Dict[str, Set[Tuple[str, str]]]
    successors: Dict[str, Set[Tuple[str, str]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "node_type": self.node_type,
            "self_relations": [(a, b) for (a, b) in self.self_relations],
            "predecessors": {
                str(k): [(n, s) for (n, s) in sorted(v)] for k, v in self.predecessors.items()
            },
            "successors": {
                str(k): [(n, s) for (n, s) in sorted(v)] for k, v in self.successors.items()
            },
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> NodeTopology:
        node = str(d["node"])
        node_type = str(d["node_type"])

        raw_self = d.get("self_relations", []) or []
        if not isinstance(raw_self, list):
            raise TypeError("self_relations must be a list")
        self_relations: List[Tuple[str, str]] = []
        for pair in raw_self:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise TypeError("self_relations entries must be 2-tuples")
            self_relations.append((str(pair[0]), str(pair[1])))

        def _load_adj(raw: Any, field_name: str) -> Dict[str, Set[Tuple[str, str]]]:
            if raw is None:
                return {}
            if not isinstance(raw, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            out: Dict[str, Set[Tuple[str, str]]] = {}
            for simproc_name, neighbors in raw.items():
                sp = str(simproc_name)
                if neighbors is None:
                    out[sp] = set()
                    continue
                if not isinstance(neighbors, list):
                    raise TypeError(f"{field_name}[{sp}] must be a list of 2-tuples")
                s: Set[Tuple[str, str]] = set()
                for item in neighbors:
                    if not (isinstance(item, (list, tuple)) and len(item) == 2):
                        raise TypeError(f"{field_name}[{sp}] entries must be 2-tuples")
                    s.add((str(item[0]), str(item[1])))
                out[sp] = s
            return out

        predecessors = _load_adj(d.get("predecessors", {}), "predecessors")
        successors = _load_adj(d.get("successors", {}), "successors")

        return NodeTopology(
            node=node,
            node_type=node_type,
            self_relations=self_relations,
            predecessors=predecessors,
            successors=successors,
        )


class PartitioningNotFoundError(KeyError, DiscoError):
    pass


class PartitioningCorruptError(DiscoRuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Partitioning:

    partitioning_id: str
    scenario_id: str
    num_partitions: int

    affinity_by_partition: Mapping[int, str] = field(default_factory=dict)

    # Row order defines incidence row order.
    node_specs: Tuple[NodeInstanceSpec, ...] = field(default_factory=tuple)

    # (n_nodes x n_vertices) BOOL
    incidence: gb.Matrix = field(default_factory=lambda: gb.Matrix(bool))

    topology_by_node: Mapping[str, NodeTopology] = field(default_factory=dict)

    node_indices: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.partitioning_id or not self.partitioning_id.strip():
            raise ValueError("partitioning_id must be non-empty")
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if self.num_partitions <= 0:
            raise ValueError("num_partitions must be > 0")

        # Validate incidence dimensions match nodes
        n_nodes = len(self.node_specs)
        if int(self.incidence.nrows) != n_nodes:
            raise ValueError(
                f"incidence.nrows ({int(self.incidence.nrows)}) must equal number of nodes ({n_nodes})"
            )
        if self.incidence.dtype != gb.dtypes.BOOL:
            raise TypeError("incidence matrix must have BOOL dtype")

        # Validate node indices
        if len(self.node_indices) != n_nodes:
            raise ValueError("node_indices length must equal number of nodes")

        for name, idx in self.node_indices.items():
            if idx < 0 or idx >= n_nodes:
                raise ValueError(f"node_indices[{name}] out of range: {idx}")
            if self.node_specs[idx].node_name != name:
                raise ValueError(f"node_indices inconsistent for '{name}'")

        # Validate nodes + topology consistency
        for i, ns in enumerate(self.node_specs):
            if ns.partition < 0 or ns.partition >= self.num_partitions:
                raise ValueError(
                    f"node '{ns.node_name}' has partition {ns.partition}, outside [0,{self.num_partitions - 1}]"
                )
            if ns.node_name not in self.topology_by_node:
                raise ValueError(f"missing topology for node '{ns.node_name}' in Partitioning object")
            topo = self.topology_by_node[ns.node_name]
            if topo.node != ns.node_name:
                raise ValueError(f"topology.node '{topo.node}' does not match node_name '{ns.node_name}'")
            if topo.node_type != ns.node_type:
                raise ValueError(
                    f"topology.node_type '{topo.node_type}' does not match node_type '{ns.node_type}'"
                )

        for p in self.affinity_by_partition.keys():
            if p < 0 or p >= self.num_partitions:
                raise ValueError(f"affinity_by_partition has invalid partition key: {p}")

    # -------------------------
    # Factories
    # -------------------------

    @classmethod
    def from_node_instance_spec(
            cls,
            node_specs: List[NodeInstanceSpec] | Tuple[NodeInstanceSpec, ...],
            incidence: gb.Matrix,
            graph: Graph,
            model: Model,
    ) -> Self:
        node_specs = tuple(node_specs)
        if not node_specs:
            raise ValueError("node_specs must not be empty")

        scenario_id = graph.scenario_id
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("graph.scenario_id must be a non-empty string")

        max_partition = max(ns.partition for ns in node_specs)
        if max_partition < 0:
            raise ValueError("node.partition must be >= 0")
        num_partitions = max_partition + 1

        node_indices: Dict[str, int] = {}
        for idx, ns in enumerate(node_specs):
            if ns.node_name in node_indices:
                raise ValueError(f"duplicate node_name in node_specs: {ns.node_name}")
            node_indices[ns.node_name] = idx

        for i, ns in enumerate(node_specs):
            if not ns.node_name or not ns.node_name.strip():
                raise ValueError("node_name must be non-empty")
            if not ns.node_type or not ns.node_type.strip():
                raise ValueError("node_type must be non-empty")
            if ns.partition < 0 or ns.partition >= num_partitions:
                raise ValueError(
                    f"node '{ns.node_name}' has partition {ns.partition}, outside [0,{num_partitions - 1}]"
                )

        pid = str(uuid4())

        n_nodes = len(node_specs)
        if int(incidence.nrows) != n_nodes:
            raise ValueError(f"incidence.nrows ({int(incidence.nrows)}) must equal number of nodes ({n_nodes})")
        if incidence.dtype != gb.dtypes.BOOL:
            raise TypeError("incidence matrix must have BOOL dtype")

        topo = cls.compute_topology_from_graph(node_specs=node_specs, incidence=incidence, graph=graph, model=model)

        return cls(
            partitioning_id=pid,
            scenario_id=scenario_id,
            num_partitions=num_partitions,
            affinity_by_partition={},
            node_specs=node_specs,
            incidence=incidence,
            topology_by_node=topo,
            node_indices=node_indices,
        )

    # -------------------------
    # Convenience lookups
    # -------------------------

    def node_spec(self, node_name: str) -> NodeInstanceSpec:
        try:
            idx = self.node_indices[node_name]
            return self.node_specs[idx]
        except KeyError as e:
            raise KeyError(f"unknown node_name '{node_name}' in partitioning manifest") from e

    def node_index(self, node_name: str) -> int:
        try:
            return self.node_indices[node_name]
        except KeyError as e:
            raise KeyError(f"unknown node_name '{node_name}' in partitioning manifest") from e

    def assignment_vector(self, node_name: str) -> gb.Vector:
        i = self.node_index(node_name)
        return self.incidence[i, :].new()

    def predecessors(self, *, node_name: str, simproc_name: str) -> Set[Tuple[str, str]]:
        topo = self.topology_by_node[node_name]
        return set(topo.predecessors.get(simproc_name, set()))

    def successors(self, *, node_name: str, simproc_name: str) -> Set[Tuple[str, str]]:
        topo = self.topology_by_node[node_name]
        return set(topo.successors.get(simproc_name, set()))

    # -------------------------
    # Storage paths
    # -------------------------

    @staticmethod
    def _base_path(partitioning_id: str) -> str:
        return f"{PARTITIONINGS}/{partitioning_id}"

    @staticmethod
    def _metadata_path(partitioning_id: str) -> str:
        return f"{Partitioning._base_path(partitioning_id)}/metadata"

    @staticmethod
    def _manifest_path(partitioning_id: str) -> str:
        return f"{Partitioning._base_path(partitioning_id)}/manifest"

    @staticmethod
    def _node_assignments_path(partitioning_id: str, partition: int, node_name: str) -> str:
        return f"{Partitioning._base_path(partitioning_id)}/{partition}/{node_name}/assignments"

    @staticmethod
    def _node_topology_path(partitioning_id: str, partition: int, node_name: str) -> str:
        return f"{Partitioning._base_path(partitioning_id)}/{partition}/{node_name}/topology"

    # -------------------------
    # Store / Load
    # -------------------------

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "partitioning_id": self.partitioning_id,
            "scenario_id": self.scenario_id,
            "num_partitions": self.num_partitions,
            "affinity_by_partition": {int(k): str(v) for k, v in self.affinity_by_partition.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def manifest_list(self) -> List[Dict[str, Any]]:
        return [n.to_dict() for n in self.node_specs]

    def store(self, metastore: Metastore) -> None:
        metastore.update_key(self._metadata_path(self.partitioning_id), self.metadata_dict())
        metastore.update_key(self._manifest_path(self.partitioning_id), self.manifest_list())

        for i, ns in enumerate(self.node_specs):
            row_vec = self.incidence[i, :].new()
            metastore.update_key(
                self._node_assignments_path(self.partitioning_id, ns.partition, ns.node_name),
                row_vec,
            )
            metastore.update_key(
                self._node_topology_path(self.partitioning_id, ns.partition, ns.node_name),
                self.topology_by_node[ns.node_name].to_dict(),
            )

    @classmethod
    def load(
        cls,
        metastore: Metastore,
        partitioning_id: str,
        graph: Graph,
    ) -> Self:
        meta = metastore.get_key(cls._metadata_path(partitioning_id))
        manifest = metastore.get_key(cls._manifest_path(partitioning_id))

        if meta is None or manifest is None:
            raise PartitioningNotFoundError(partitioning_id)
        if not isinstance(meta, Mapping):
            raise PartitioningCorruptError(f"partitioning '{partitioning_id}': metadata is not a mapping")
        if not isinstance(manifest, list):
            raise PartitioningCorruptError(f"partitioning '{partitioning_id}': manifest is not a list")

        scenario_id = str(meta.get("scenario_id", "") or "")
        num_partitions = int(meta.get("num_partitions", 0) or 0)

        raw_aff = meta.get("affinity_by_partition", {}) or {}
        if not isinstance(raw_aff, Mapping):
            raise PartitioningCorruptError(
                f"partitioning '{partitioning_id}': affinity_by_partition is not a mapping"
            )
        affinity_by_partition: Dict[int, str] = {int(k): str(v) for k, v in raw_aff.items()}

        node_specs: Tuple[NodeInstanceSpec, ...] = tuple(NodeInstanceSpec.from_dict(x) for x in manifest)

        node_indices: Dict[str, int] = {}
        for idx, ns in enumerate(node_specs):
            if ns.node_name in node_indices:
                raise ValueError(f"duplicate node_name in manifest: {ns.node_name}")
            node_indices[ns.node_name] = idx

        n_nodes = len(node_specs)
        n_vertices = int(graph.num_vertices)

        rows: List[int] = []
        cols: List[int] = []

        for i, ns in enumerate(node_specs):
            v = metastore.get_key(cls._node_assignments_path(partitioning_id, ns.partition, ns.node_name))
            if v is None:
                raise PartitioningCorruptError(
                    f"partitioning '{partitioning_id}': missing assignments for node '{ns.node_name}'"
                )
            if not isinstance(v, gb.Vector):
                raise PartitioningCorruptError(
                    f"partitioning '{partitioning_id}': assignments for node '{ns.node_name}' is not a graphblas.Vector"
                )
            idx, _ = v.to_coo()
            if idx.size:
                rows.extend([i] * int(idx.size))
                cols.extend(idx.tolist())

        incidence = gb.Matrix.from_coo(
            rows,
            cols,
            [True] * len(rows),
            nrows=n_nodes,
            ncols=n_vertices,
            dtype=bool,
        )

        topology_by_node: Dict[str, NodeTopology] = {}
        for ns in node_specs:
            t = metastore.get_key(cls._node_topology_path(partitioning_id, ns.partition, ns.node_name))
            if t is None:
                raise PartitioningCorruptError(
                    f"partitioning '{partitioning_id}': missing topology for node '{ns.node_name}'"
                )
            if not isinstance(t, Mapping):
                raise PartitioningCorruptError(
                    f"partitioning '{partitioning_id}': topology for node '{ns.node_name}' is not a mapping"
                )
            topology_by_node[ns.node_name] = NodeTopology.from_dict(t)

        return cls(
            partitioning_id=partitioning_id,
            scenario_id=scenario_id,
            num_partitions=num_partitions,
            affinity_by_partition=affinity_by_partition,
            node_specs=node_specs,
            incidence=incidence,
            topology_by_node=topology_by_node,
            node_indices=node_indices
        )

    @classmethod
    def load_metadata(cls, metastore: Metastore, partitioning_id: str) -> Mapping[str, Any]:
        meta = metastore.get_key(cls._metadata_path(partitioning_id))
        if meta is None:
            raise PartitioningNotFoundError(partitioning_id)
        if not isinstance(meta, Mapping):
            raise PartitioningCorruptError(f"partitioning '{partitioning_id}': metadata is not a mapping")
        return meta

    # -------------------------
    # Topology computation using incidence directly
    # -------------------------
    @classmethod
    def compute_topology_from_graph(
        cls,
        node_specs: List[NodeInstanceSpec] | Tuple[NodeInstanceSpec, ...],
        incidence: gb.Matrix,
        graph: Graph,
        model: Model,
    ) -> Dict[str, NodeTopology]:

        simproc_names_by_order = model.spec.simprocs
        node_names = [ns.node_name for ns in node_specs]

        predecessors: Dict[str, Dict[str, Set[Tuple[str, str]]]] = {name: {} for name in node_names}
        successors: Dict[str, Dict[str, Set[Tuple[str, str]]]] = {name: {} for name in node_names}

        n_simprocs = len(simproc_names_by_order)
        if len(graph.layers) != n_simprocs:
            raise ValueError(f"Number of graph layers does not match number of simprocs.")

        for layer_idx in range(len(graph.layers)):

            simproc_name = simproc_names_by_order[layer_idx]
            M = graph.get_matrix(layer_idx)

            Q = gb.op.plus_times(incidence @ M)
            R = gb.op.plus_times(Q @ incidence.T).select("!=", 0)

            r_rows, r_cols, _ = R.to_coo()
            for rr, cc in zip(r_rows.tolist(), r_cols.tolist()):
                src = node_names[rr]
                dst = node_names[cc]
                successors[src].setdefault(simproc_name, set()).add((dst, simproc_name))
                predecessors[dst].setdefault(simproc_name, set()).add((src, simproc_name))

        for ns in node_specs:
            rels = model.spec.node_types[ns.node_type].self_relations
            for from_sp, to_sp in rels:
                successors[ns.node_name].setdefault(from_sp, set()).add((ns.node_name, to_sp))
                predecessors[ns.node_name].setdefault(to_sp, set()).add((ns.node_name, from_sp))

        out: Dict[str, NodeTopology] = {}
        for ns in node_specs:
            out[ns.node_name] = NodeTopology(
                node=ns.node_name,
                node_type=ns.node_type,
                self_relations=list(model.spec.node_types[ns.node_type].self_relations),
                predecessors=predecessors.get(ns.node_name, {}),
                successors=successors.get(ns.node_name, {}),
            )
        return out
