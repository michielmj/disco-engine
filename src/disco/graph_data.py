from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Iterable

from graphblas import Matrix, Vector
from pandas import DataFrame
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from disco.database import SessionManager
from disco.model import Model, OrmBundle
from disco.partitioning import NodeInstanceSpec

from disco.graph import (
    Graph,
    get_outbound_edge_data,
    get_inbound_edge_data,
    get_outbound_map,
    get_inbound_map,
    get_vertex_data
)


def _normalize_column_elements(
        columns: Sequence[ColumnElement[Any]] | Sequence[str],
        orm_table: Table
) -> Iterable[ColumnElement]:
    for c in columns:
        if isinstance(c, str):
            yield orm_table.columns[c]
        else:
            yield orm_table.columns[c.name]


@dataclass(frozen=True, slots=True)
class SimProcGraphData:
    session_manager: SessionManager
    graph: Graph
    node_table: Table
    edge_table: Table
    layer_idx: int
    node_mask: Optional[Vector]

    def outbound_edge_data(
        self,
        columns: Sequence[ColumnElement[Any]] | Sequence[str],
        *,
        mask: Optional[Vector] = None,
    ) -> DataFrame:
        columns = list(_normalize_column_elements(columns, self.edge_table))

        with self.session_manager.session() as session:
            return get_outbound_edge_data(
                graph=self.graph,
                session=session,
                edge_table=self.edge_table,
                columns=columns,
                layer_idx=self.layer_idx,
                mask=mask if mask is not None else self.node_mask,
            )

    def inbound_edge_data(
        self,
        columns: Sequence[ColumnElement[Any]],
        *,
        mask: Optional[Vector] = None,
    ) -> DataFrame:
        columns = list(_normalize_column_elements(columns, self.edge_table))

        with self.session_manager.session() as session:
            return get_inbound_edge_data(
                graph=self.graph,
                session=session,
                edge_table=self.edge_table,
                columns=columns,
                layer_idx=self.layer_idx,
                mask=mask if mask is not None else self.node_mask,
            )

    def outbound_map(self, *, mask: Optional[Vector] = None) -> Matrix:
        with self.session_manager.session() as session:
            return get_outbound_map(
                graph=self.graph,
                session=session,
                layer_idx=self.layer_idx,
                mask=mask if mask is not None else self.node_mask,
            )

    def inbound_map(self, *, mask: Optional[Vector] = None) -> Matrix:
        with self.session_manager.session() as session:
            return get_inbound_map(
                graph=self.graph,
                session=session,
                layer_idx=self.layer_idx,
                mask=mask if mask is not None else self.node_mask,
            )


@dataclass(frozen=True, slots=True)
class GraphData:
    """
    Node-scoped graph data access facade.

    Pre-binds:
      - Session lifecycle (SessionManager)
      - node_table (from OrmBundle.node_tables via node_type)
      - edge table selection by simproc name (OrmBundle.edge_tables_by_simproc + default fallback)
      - simproc_name -> layer_idx mapping (from Model.spec.simprocs)
      - node assignment mask (partitioning.assignment_vector)
    """
    session_manager: SessionManager
    graph: Graph
    orm: OrmBundle

    node_name: str
    node_type: str
    node_table: Table

    layer_id_by_simproc: Mapping[str, int]
    node_mask: Optional[Vector]

    @classmethod
    def for_node(
        cls,
        *,
        session_manager: SessionManager,
        graph: Graph,
        model: Model,
        spec: NodeInstanceSpec,
        node_mask: Optional[Vector] = None,
    ) -> "GraphData":
        try:
            node_table = model.orm.node_tables[spec.node_type]
        except KeyError as exc:
            raise KeyError(f"OrmBundle.node_tables has no entry for node_type={spec.node_type!r}") from exc

        layer_id_by_simproc = {name: i for i, name in enumerate(model.spec.simprocs)}

        return cls(
            session_manager=session_manager,
            graph=graph,
            orm=model.orm,
            node_name=spec.node_name,
            node_type=spec.node_type,
            node_table=node_table,
            layer_id_by_simproc=layer_id_by_simproc,
            node_mask=node_mask,
        )

    # Node-level
    def vertex_data(
        self,
        columns: Sequence[ColumnElement[Any]] | Sequence[str],
        *,
        mask: Optional[Vector] = None,
    ) -> DataFrame:
        columns = list(_normalize_column_elements(columns, self.node_table))

        with self.session_manager.session() as session:
            return get_vertex_data(
                graph=self.graph,
                session=session,
                vertex_table=self.node_table,
                columns=columns,
                mask=mask if mask is not None else self.node_mask,
            )

    # Expose node-specific ORM slice
    def edge_table_for(self, simproc_name: str) -> Table:
        t = self.orm.edge_tables_by_simproc.get(simproc_name)
        if t is not None:
            return t
        if self.orm.default_edge_table is not None:
            return self.orm.default_edge_table
        raise KeyError(
            f"No edge table for simproc {simproc_name!r} and OrmBundle.default_edge_table is None"
        )

    def simproc(self, simproc_name: str) -> SimProcGraphData:
        try:
            layer_idx = self.layer_id_by_simproc[simproc_name]
        except KeyError as exc:
            raise KeyError(f"Unknown simproc {simproc_name!r} (not in model.spec.simprocs)") from exc

        return SimProcGraphData(
            session_manager=self.session_manager,
            graph=self.graph,
            node_table=self.node_table,
            edge_table=self.edge_table_for(simproc_name),
            layer_idx=layer_idx,
            node_mask=self.node_mask,
        )
