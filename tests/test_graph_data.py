# tests/test_graph_data.py
from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert
from sqlalchemy.orm import Session

from disco.database import SessionManager
from disco.graph import Graph
from disco.graph.schema import create_graph_schema, edges, vertices
from disco.graph_data import GraphData, SimProcGraphData

FILL = {"count": 0, "name": ""}


def _setup() -> tuple[SessionManager, Graph, Table, Table]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    create_graph_schema(engine)
    metadata = MetaData()
    nodes = Table(
        "mixed_nodes",
        metadata,
        Column("scenario_id", String, primary_key=True),
        Column("key", String, primary_key=True),
        Column("name", String),
        Column("count", Integer),
    )
    edge_tbl = Table(
        "mixed_edges",
        metadata,
        Column("scenario_id", String, primary_key=True),
        Column("source_key", String, primary_key=True),
        Column("target_key", String, primary_key=True),
        Column("name", String),
        Column("count", Integer),
    )
    metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(
            insert(vertices),
            [{"scenario_id": "S1", "key": k, "index": i} for i, k in enumerate("ABC")],
        )
        session.execute(
            insert(edges),
            [
                {"scenario_id": "S1", "layer_idx": 0, "source_idx": 0, "target_idx": 1, "weight": 1.5},
                {"scenario_id": "S1", "layer_idx": 0, "source_idx": 1, "target_idx": 2, "weight": 2.5},
            ],
        )
        session.execute(insert(nodes), [{"scenario_id": "S1", "key": "A", "name": "x", "count": 7}])
        session.execute(
            insert(edge_tbl),
            [{"scenario_id": "S1", "source_key": "A", "target_key": "B", "name": "e", "count": 3}],
        )
        session.commit()
    graph = Graph(layers={}, num_vertices=3, scenario_id="S1")
    return SessionManager(engine=engine), graph, nodes, edge_tbl


def _graph_data(sm: SessionManager, graph: Graph, nodes: Table) -> GraphData:
    anything: Any = None
    return GraphData(
        session_manager=sm,
        graph=graph,
        partitioning=anything,
        orm=anything,
        node_name="n",
        node_index=0,
        node_type="t",
        node_table=nodes,
        layer_id_by_simproc={},
        node_mask=None,
    )


def test_vertex_data_passes_a_per_column_mapping_to_the_extract_function() -> None:
    sm, graph, nodes, _ = _setup()
    gd = _graph_data(sm, graph, nodes)
    plain = gd.vertex_data(["name", "count"])
    assert plain["name"].isna().sum() == 2  # no fill: missing rows stay missing
    df = gd.vertex_data(["name", "count"], default_fill=FILL)
    pd.testing.assert_frame_equal(df, plain.fillna(FILL))
    assert df["name"].tolist() == ["x", "", ""]
    assert df["count"].tolist() == [7, 0, 0]


def test_simproc_edge_data_passes_a_per_column_mapping_to_the_extract_function() -> None:
    sm, graph, nodes, edge_tbl = _setup()
    sp = SimProcGraphData(
        session_manager=sm, graph=graph, node_table=nodes, edge_table=edge_tbl, layer_idx=0, node_mask=None
    )
    for fn in (sp.outbound_edge_data, sp.inbound_edge_data):
        plain = fn(["name", "count"])  # type: ignore[operator]
        assert plain["name"].isna().sum() == 1
        df = fn(["name", "count"], default_fill=FILL)  # type: ignore[operator]
        pd.testing.assert_frame_equal(df, plain.fillna(FILL))
        assert df["name"].tolist() == ["e", ""]
        assert df["count"].tolist() == [3, 0]
