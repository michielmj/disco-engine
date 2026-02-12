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
    text, ForeignKeyConstraint,
)
from sqlalchemy.engine import Engine

metadata = MetaData(schema="graph")

scenarios = Table(
    "scenarios",
    metadata,
    Column("scenario_id", String, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("description", String, nullable=True),
)

vertices = Table(
    "vertices",
    metadata,
    Column("scenario_id", String, ForeignKey("graph.scenarios.scenario_id"), primary_key=True),
    Column("index", BigInteger, primary_key=True),  # 0..V-1
    Column("key", String, nullable=False),
    # Column("node_type", String, nullable=False),
    schema="graph",
)

edges = Table(
    "edges",
    metadata,
    Column("scenario_id", String, ForeignKey("graph.scenarios.scenario_id"), primary_key=True),
    Column("layer_idx", Integer, primary_key=True),
    Column("source_idx", BigInteger, primary_key=True),
    Column("target_idx", BigInteger, primary_key=True),
    Column("weight", Float, nullable=False),
    ForeignKeyConstraint(
        ["scenario_id", "source_idx"],
        ["graph.vertices.scenario_id", "graph.vertices.index"],
        name="fk_edges_source_vertex",
    ),
    ForeignKeyConstraint(
        ["scenario_id", "target_idx"],
        ["graph.vertices.scenario_id", "graph.vertices.index"],
        name="fk_edges_target_vertex",
    ),
    schema="graph",
)

labels = Table(
    "labels",
    metadata,
    Column("id", Integer, primary_key=True),  # <-- changed
    Column("scenario_id", String, ForeignKey("graph.scenarios.scenario_id"), nullable=False),
    Column("type", String, nullable=False),
    Column("value", String, nullable=False),
)

vertex_labels = Table(
    "vertex_labels",
    metadata,
    Column("scenario_id", String, ForeignKey("graph.scenarios.scenario_id"), primary_key=True),
    Column("vertex_index", BigInteger, primary_key=True),
    Column("label_id", Integer, ForeignKey("graph.labels.id"), primary_key=True),
    ForeignKeyConstraint(
        ["scenario_id", "vertex_index"],
        ["graph.vertices.scenario_id", "graph.vertices.index"],
        name="fk_vertex_labels_vertex",
    ),
    schema="graph",
)

vertex_masks = Table(
    "vertex_masks",
    metadata,
    Column("scenario_id", String, ForeignKey("graph.scenarios.scenario_id"), primary_key=True),
    Column("mask_id", String(36), primary_key=True),  # UUID as string
    Column("vertex_index", BigInteger, primary_key=True),
    Column("updated_at", DateTime, nullable=False),
    ForeignKeyConstraint(
        ["scenario_id", "vertex_index"],
        ["graph.vertices.scenario_id", "graph.vertices.index"],
        name="fk_vertex_masks_vertex",
    ),
    schema="graph",
)


def create_graph_schema(engine: Engine) -> None:
    """
    Create or update the graph schema.

    - For PostgreSQL: CREATE SCHEMA IF NOT EXISTS graph
    - For others: rely on metadata.create_all; schema='graph' must be supported
      or configured appropriately.
    """
    dialect_name = engine.dialect.name

    with engine.begin() as conn:
        if dialect_name == "postgresql":
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS graph"))
        metadata.create_all(conn)
