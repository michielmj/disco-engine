# tests/model/test_orm.py
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, MetaData, String, Table, UniqueConstraint
from sqlalchemy.engine import Engine

from disco.model.orm import (
    ModelOrmLoadError,
    ModelOrmValidationError,
    build_orm_bundle,
    db_install,
    db_validate,
)


# -----------------------------
# Helpers
# -----------------------------


def _engine() -> Engine:
    # In-memory SQLite is enough for create_all + reflection.
    return sa.create_engine("sqlite+pysqlite:///:memory:")


def _spec(
    *,
    orm: Optional[str] = None,
    node_table: str = "warehouse_data",
    default_edge_table: Optional[str] = "edges_default",
    demand_edge_table: Optional[str] = "demand_edges",
):
    """
    Build a minimal 'spec' object with the attributes expected by disco.model.orm.

    NOTE: build_orm_bundle reads the provider ref from spec.orm.
    """
    node_types = {
        "Warehouse": SimpleNamespace(node_data_table=node_table),
    }

    simproc_edge_data_tables = {}
    if demand_edge_table is not None:
        simproc_edge_data_tables["demand"] = demand_edge_table

    return SimpleNamespace(
        orm=orm,
        node_types=node_types,
        simproc_edge_data_tables=simproc_edge_data_tables,
        default_edge_data_table=default_edge_table,
    )


def _create_valid_node_table(md: MetaData, name: str) -> Table:
    return Table(
        name,
        md,
        Column("scenario_id", String, nullable=False),
        Column("key", String, nullable=False),
        Column("payload", String, nullable=True),
        sa.PrimaryKeyConstraint("scenario_id", "key"),
    )


def _create_valid_edge_table(md: MetaData, name: str) -> Table:
    return Table(
        name,
        md,
        Column("scenario_id", String, nullable=False),
        Column("source_key", String, nullable=False),
        Column("target_key", String, nullable=False),
        Column("payload", String, nullable=True),
    )


def _write_py(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


# -----------------------------
# Tests: reflection mode
# -----------------------------


def test_build_orm_bundle_reflection_success():
    eng = _engine()
    md = MetaData()

    _create_valid_node_table(md, "warehouse_data")
    _create_valid_edge_table(md, "edges_default")
    _create_valid_edge_table(md, "demand_edges")

    md.create_all(eng)

    spec = _spec(orm=None)

    bundle = build_orm_bundle(
        spec=spec,
        db=eng,
        schema=None,
    )

    # bundle has tables (reflected)
    assert "Warehouse" in bundle.node_tables
    assert bundle.node_tables["Warehouse"].name == "warehouse_data"
    assert bundle.default_edge_table is not None
    assert bundle.default_edge_table.name == "edges_default"
    assert "demand" in bundle.edge_tables_by_simproc
    assert bundle.edge_tables_by_simproc["demand"].name == "demand_edges"

    # validate actual DB schema explicitly
    db_validate(bundle, eng, schema=None)


def test_build_orm_bundle_reflection_missing_table_raises():
    eng = _engine()
    md = MetaData()
    _create_valid_node_table(md, "warehouse_data")
    md.create_all(eng)

    spec = _spec(orm=None, default_edge_table="edges_default")  # referenced, but not created

    with pytest.raises(ModelOrmLoadError, match="Required tables are missing"):
        build_orm_bundle(
            spec=spec,
            db=eng,
            schema=None,
        )


# -----------------------------
# Tests: node table validation (via db_validate)
# -----------------------------


def test_node_table_requires_unique_per_scenario():
    eng = _engine()
    md = MetaData()

    # Wrong: primary key only on key, NOT (scenario_id,key)
    Table(
        "warehouse_data",
        md,
        Column("scenario_id", String, nullable=False),
        Column("key", String, nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )

    _create_valid_edge_table(md, "edges_default")
    md.create_all(eng)

    spec = _spec(orm=None, default_edge_table="edges_default", demand_edge_table=None)
    bundle = build_orm_bundle(spec=spec, db=eng, schema=None)

    with pytest.raises(ModelOrmValidationError, match="uniqueness.*\\(scenario_id, key\\)"):
        db_validate(bundle, eng, schema=None)


def test_node_table_unique_constraint_exact_match_is_accepted():
    eng = _engine()
    md = MetaData()

    Table(
        "warehouse_data",
        md,
        Column("scenario_id", String, nullable=False),
        Column("key", String, nullable=False),
        Column("payload", String, nullable=True),
        UniqueConstraint("scenario_id", "key"),
    )

    _create_valid_edge_table(md, "edges_default")
    md.create_all(eng)

    spec = _spec(orm=None, default_edge_table="edges_default", demand_edge_table=None)
    bundle = build_orm_bundle(spec=spec, db=eng, schema=None)

    db_validate(bundle, eng, schema=None)


def test_node_table_requires_scenario_id_not_null():
    eng = _engine()
    md = MetaData()

    Table(
        "warehouse_data",
        md,
        Column("scenario_id", String, nullable=True),  # wrong
        Column("key", String, nullable=False),
        UniqueConstraint("scenario_id", "key"),
    )

    _create_valid_edge_table(md, "edges_default")
    md.create_all(eng)

    spec = _spec(orm=None, default_edge_table="edges_default", demand_edge_table=None)
    bundle = build_orm_bundle(spec=spec, db=eng, schema=None)

    with pytest.raises(ModelOrmValidationError, match="scenario_id.*NOT NULL"):
        db_validate(bundle, eng, schema=None)


# -----------------------------
# Tests: edge table validation (via db_validate)
# -----------------------------


def test_edge_table_requires_source_target_and_scenario_id():
    eng = _engine()
    md = MetaData()

    _create_valid_node_table(md, "warehouse_data")

    # Wrong: missing target_key
    Table(
        "edges_default",
        md,
        Column("scenario_id", String, nullable=False),
        Column("source_key", String, nullable=False),
    )

    md.create_all(eng)

    spec = _spec(orm=None, default_edge_table="edges_default", demand_edge_table=None)
    bundle = build_orm_bundle(spec=spec, db=eng, schema=None)

    with pytest.raises(ModelOrmValidationError, match="missing required column 'target_key'"):
        db_validate(bundle, eng, schema=None)


# -----------------------------
# Tests: Python ORM provider mode
# -----------------------------


def test_provider_mode_db_install_creates_and_validates(tmp_path, monkeypatch):
    """
    Provider mode: build_orm_bundle resolves tables from provider MetaData.
    db_install then creates missing tables and validates the DB.
    """
    eng = _engine()

    _write_py(
        tmp_path / "myormprovider.py",
        """
        import sqlalchemy as sa
        from sqlalchemy import MetaData, Table, Column, String

        metadata = MetaData()

        Table(
            "warehouse_data",
            metadata,
            Column("scenario_id", String, nullable=False),
            Column("key", String, nullable=False),
            sa.PrimaryKeyConstraint("scenario_id", "key"),
        )

        Table(
            "edges_default",
            metadata,
            Column("scenario_id", String, nullable=False),
            Column("source_key", String, nullable=False),
            Column("target_key", String, nullable=False),
        )

        Table(
            "demand_edges",
            metadata,
            Column("scenario_id", String, nullable=False),
            Column("source_key", String, nullable=False),
            Column("target_key", String, nullable=False),
        )

        def get_metadata():
            return metadata
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    spec = _spec(
        orm="myormprovider:get_metadata",
        default_edge_table="edges_default",
        demand_edge_table="demand_edges",
    )

    # provider mode: db is optional at build time
    bundle = build_orm_bundle(spec=spec, db=None, schema=None)

    # deliberate action
    db_install(bundle, eng, schema=None)

    # sanity: tables exist and validate passed
    insp = sa.inspect(eng)
    assert insp.has_table("warehouse_data")
    assert insp.has_table("edges_default")
    assert insp.has_table("demand_edges")


def test_provider_mode_missing_table_in_metadata_raises(tmp_path, monkeypatch):
    eng = _engine()

    # Provider defines only node table, but spec references edge table too.
    _write_py(
        tmp_path / "myormprovider2.py",
        """
        import sqlalchemy as sa
        from sqlalchemy import MetaData, Table, Column, String

        metadata = MetaData()

        Table(
            "warehouse_data",
            metadata,
            Column("scenario_id", String, nullable=False),
            Column("key", String, nullable=False),
            sa.PrimaryKeyConstraint("scenario_id", "key"),
        )

        def get_metadata():
            return metadata
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    spec = _spec(
        orm="myormprovider2:get_metadata",
        default_edge_table="edges_default",  # required by spec but absent from provider metadata
        demand_edge_table=None,
    )

    with pytest.raises(ModelOrmLoadError, match="does not contain required table"):
        build_orm_bundle(spec=spec, db=None, schema=None)
