# tests/model/test_loader.py
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml

from disco.model.loader import (
    ModelLoadError,
    load_model_from_entrypoint,
    load_model_from_package,
    load_model_from_path,
)
from disco.model.orm import db_install, db_validate

try:
    import importlib.metadata as im
except Exception:  # pragma: no cover
    im = None  # type: ignore


def _engine():
    # In-memory SQLite is sufficient for install + reflection.
    return sa.create_engine("sqlite+pysqlite:///:memory:")


def _write_py(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _write_model_yml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _node_module_source(class_name: str) -> str:
    # Create a Node subclass that satisfies any ABC requirements dynamically.
    return f"""
    from disco.node import Node

    def _make_node_cls(name: str):
        attrs = {{}}

        def __init__(self):
            pass
        attrs["__init__"] = __init__

        def initialize(self, **kwargs):
            return None
        attrs["initialize"] = initialize

        # satisfy any abstract methods besides initialize
        for m in getattr(Node, "__abstractmethods__", set()):
            if m in attrs:
                continue
            def _stub(self, *args, **kwargs):
                return None
            _stub.__name__ = m
            attrs[m] = _stub

        return type(name, (Node,), attrs)

    {class_name} = _make_node_cls("{class_name}")
    """


def _orm_provider_source(
    *,
    node_table: str = "warehouse_data",
    default_edge_table: str = "edges_default",
    demand_edge_table: str = "demand_edges",
) -> str:
    # Provides MetaData with required tables and constraints.
    return f"""
    import sqlalchemy as sa
    from sqlalchemy import MetaData, Table, Column, String

    metadata = MetaData()

    Table(
        "{node_table}",
        metadata,
        Column("scenario_id", String, nullable=False),
        Column("key", String, nullable=False),
        sa.PrimaryKeyConstraint("scenario_id", "key"),
    )

    # Default edge table
    Table(
        "{default_edge_table}",
        metadata,
        Column("scenario_id", String, nullable=False),
        Column("source_key", String, nullable=False),
        Column("target_key", String, nullable=False),
    )

    # Simproc-specific edge table
    Table(
        "{demand_edge_table}",
        metadata,
        Column("scenario_id", String, nullable=False),
        Column("source_key", String, nullable=False),
        Column("target_key", String, nullable=False),
    )

    def get_metadata():
        return metadata
    """


def test_load_model_from_path_success(tmp_path):
    eng = _engine()

    # tmp_path/
    #   model.yml
    #   mymodel/
    #     __init__.py
    #     nodes.py
    #     orm.py
    (tmp_path / "mymodel").mkdir()
    _write_py(tmp_path / "mymodel" / "__init__.py", "")
    _write_py(tmp_path / "mymodel" / "nodes.py", _node_module_source("Warehouse"))
    _write_py(tmp_path / "mymodel" / "orm.py", _orm_provider_source())

    model_dict = {
        "name": "test-model",
        "version": "1.0",
        "simprocs": {
            "demand": {"edge-data-table": "demand_edges"},
            "supply": {},
        },
        "orm": "mymodel.orm:get_metadata",
        "node-types": {
            "Warehouse": {
                "class": "mymodel.nodes:Warehouse",
                "node-data-table": "warehouse_data",
                "distinct-nodes": ["site"],
                "self-relations": [["demand", "supply"]],
            }
        },
        "default-edge-data-table": "edges_default",
    }
    _write_model_yml(tmp_path / "model.yml", model_dict)

    m = load_model_from_path(
        tmp_path,
        db=eng,
        import_root=tmp_path,
    )

    # Deliberate DB install + validation
    db_install(m.orm, eng)
    db_validate(m.orm, eng)

    assert m.spec.name == "test-model"
    assert m.spec.simprocs == ["demand", "supply"]
    assert m.spec.simproc_edge_data_tables == {"demand": "demand_edges"}
    assert "Warehouse" in m.node_classes

    node = m.node_factory("Warehouse")
    assert hasattr(node, "initialize")

    # ORM bundle is attached and has the expected tables (DDL/metadata side)
    assert "Warehouse" in m.orm.node_tables
    assert m.orm.node_tables["Warehouse"].name == "warehouse_data"

    assert m.orm.default_edge_table is not None
    assert m.orm.default_edge_table.name == "edges_default"

    assert "demand" in m.orm.edge_tables_by_simproc
    assert m.orm.edge_tables_by_simproc["demand"].name == "demand_edges"
    assert "supply" not in m.orm.edge_tables_by_simproc


def test_load_model_from_path_rejects_non_node_subclass(tmp_path):
    eng = _engine()

    (tmp_path / "mymodel").mkdir()
    _write_py(tmp_path / "mymodel" / "__init__.py", "")
    _write_py(
        tmp_path / "mymodel" / "nodes.py",
        """
        class NotANode:
            def __init__(self):
                pass
            def initialize(self, **kwargs):
                return None
        """,
    )

    model_dict = {
        "simprocs": ["demand"],
        "node-types": {
            "Warehouse": {
                "class": "mymodel.nodes:NotANode",
                "node-data-table": "warehouse_data",
            }
        },
        # no orm/default-edge-data-table needed; this fails before ORM attach
    }
    _write_model_yml(tmp_path / "model.yml", model_dict)

    with pytest.raises(ModelLoadError):
        load_model_from_path(
            tmp_path,
            db=eng,
            import_root=tmp_path,
        )


@pytest.mark.skipif(im is None, reason="importlib.metadata not available")
def test_load_model_from_entrypoint_strict_contract_success(tmp_path, monkeypatch):
    eng = _engine()

    pkg = tmp_path / "mymodelpkg"
    pkg.mkdir()
    _write_py(pkg / "__init__.py", "")
    _write_py(pkg / "nodes.py", _node_module_source("Warehouse"))
    _write_py(pkg / "orm.py", _orm_provider_source())

    model_dict = {
        "name": "pkg-model",
        "version": "1.0",
        "simprocs": {
            "demand": {"edge-data-table": "demand_edges"},
            "supply": {},
        },
        "orm": "mymodelpkg.orm:get_metadata",
        "node-types": {
            "Warehouse": {
                "class": "mymodelpkg.nodes:Warehouse",
                "node-data-table": "warehouse_data",
            }
        },
        "default-edge-data-table": "edges_default",
    }
    _write_model_yml(pkg / "model.yml", model_dict)

    _write_py(
        tmp_path / "epmod.py",
        """
        def model_package() -> str:
            return "mymodelpkg"
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    ep = im.EntryPoint(name="test_model", value="epmod:model_package", group="disco.models")

    import disco.model.loader as loader
    monkeypatch.setattr(loader, "discover_model_entrypoints", lambda group="disco.models": {"test_model": ep})

    m = load_model_from_entrypoint("test_model", db=eng)

    # Deliberate DB install + validation
    db_install(m.orm, eng)
    db_validate(m.orm, eng)

    assert m.spec.name == "pkg-model"
    assert "Warehouse" in m.node_classes
    assert "Warehouse" in m.orm.node_tables
    assert m.orm.default_edge_table is not None
    assert "demand" in m.orm.edge_tables_by_simproc


@pytest.mark.skipif(im is None, reason="importlib.metadata not available")
def test_load_model_from_entrypoint_strict_contract_rejects_non_str(tmp_path, monkeypatch):
    eng = _engine()

    _write_py(
        tmp_path / "epmod2.py",
        """
        def model_package():
            return {"not": "a string"}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    ep = im.EntryPoint(name="bad_model", value="epmod2:model_package", group="disco.models")

    import disco.model.loader as loader
    monkeypatch.setattr(loader, "discover_model_entrypoints", lambda group="disco.models": {"bad_model": ep})

    with pytest.raises(ModelLoadError):
        load_model_from_entrypoint("bad_model", db=eng)


def test_load_model_from_package_reads_resource_model_yml(tmp_path, monkeypatch):
    eng = _engine()

    pkg = tmp_path / "mymodelpkg2"
    pkg.mkdir()
    _write_py(pkg / "__init__.py", "")
    _write_py(pkg / "nodes.py", _node_module_source("Warehouse"))
    _write_py(pkg / "orm.py", _orm_provider_source())

    # Use simple simprocs form (list) to cover "default-only" case.
    model_dict = {
        "simprocs": ["demand"],
        "orm": "mymodelpkg2.orm:get_metadata",
        "node-types": {
            "Warehouse": {
                "class": "mymodelpkg2.nodes:Warehouse",
                "node-data-table": "warehouse_data",
            }
        },
        "default-edge-data-table": "edges_default",
    }
    _write_model_yml(pkg / "model.yml", model_dict)

    monkeypatch.syspath_prepend(str(tmp_path))

    m = load_model_from_package("mymodelpkg2", db=eng)

    # Deliberate DB install + validation
    db_install(m.orm, eng)
    db_validate(m.orm, eng)

    assert m.spec.simprocs == ["demand"]
    assert m.spec.simproc_edge_data_tables == {}
    assert "Warehouse" in m.node_classes
    assert "Warehouse" in m.orm.node_tables
    assert m.orm.default_edge_table is not None
    assert m.orm.default_edge_table.name == "edges_default"
