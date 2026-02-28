# src/disco/graph/schema.py
from __future__ import annotations

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

scenarios = Table(
    "graph_scenarios",
    metadata,
    Column("scenario_id", String, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("description", String, nullable=True),
)

vertices = Table(
    "graph_vertices",
    metadata,
    Column("scenario_id", String, ForeignKey("graph_scenarios.scenario_id"), primary_key=True),
    Column("index", BigInteger, primary_key=True),  # 0..V-1
    Column("key", String, nullable=False),
)

edges = Table(
    "graph_edges",
    metadata,
    Column("scenario_id", String, ForeignKey("graph_scenarios.scenario_id"), primary_key=True),
    Column("layer_idx", Integer, primary_key=True),
    Column("source_idx", BigInteger, primary_key=True),
    Column("target_idx", BigInteger, primary_key=True),
    Column("weight", Float, nullable=False),
    ForeignKeyConstraint(
        ["scenario_id", "source_idx"],
        ["graph_vertices.scenario_id", "graph_vertices.index"],
        name="fk_edges_source_vertex",
    ),
    ForeignKeyConstraint(
        ["scenario_id", "target_idx"],
        ["graph_vertices.scenario_id", "graph_vertices.index"],
        name="fk_edges_target_vertex",
    ),
)

labels = Table(
    "graph_labels",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("scenario_id", String, ForeignKey("graph_scenarios.scenario_id"), nullable=False),
    Column("type", String, nullable=False),
    Column("value", String, nullable=False),
)

vertex_labels = Table(
    "graph_vertex_labels",
    metadata,
    Column("scenario_id", String, ForeignKey("graph_scenarios.scenario_id"), primary_key=True),
    Column("vertex_index", BigInteger, primary_key=True),
    Column("label_id", Integer, ForeignKey("graph_labels.id"), primary_key=True),
    ForeignKeyConstraint(
        ["scenario_id", "vertex_index"],
        ["graph_vertices.scenario_id", "graph_vertices.index"],
        name="fk_vertex_labels_vertex",
    ),
)

vertex_masks = Table(
    "graph_vertex_masks",
    metadata,
    Column("scenario_id", String, ForeignKey("graph_scenarios.scenario_id"), primary_key=True),
    Column("mask_id", String(36), primary_key=True),  # UUID as string
    Column("vertex_index", BigInteger, primary_key=True),
    Column("updated_at", DateTime, nullable=False),
    ForeignKeyConstraint(
        ["scenario_id", "vertex_index"],
        ["graph_vertices.scenario_id", "graph_vertices.index"],
        name="fk_vertex_masks_vertex",
    ),
)


def create_graph_schema(engine: Engine) -> None:
    """
    Create the graph tables in the default schema.

    All graph infrastructure tables are prefixed with ``graph_``:
    graph_scenarios, graph_vertices, graph_edges, graph_labels,
    graph_vertex_labels, graph_vertex_masks.
    """
    with engine.begin() as conn:
        metadata.create_all(conn)
