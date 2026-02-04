# src/disco/model/orm.py

from __future__ import annotations

"""
SQLAlchemy ORM integration for Model loading.

Key design:
- Model loading must NEVER create tables implicitly.
- Table creation is a deliberate action via db_install().

Two modes:
1) Python ORM provider mode (spec.orm is set):
   - Import provider callable -> returns sqlalchemy.MetaData with DDL definitions.
   - Resolve required Table objects from that MetaData (no DB access required).
   - db_validate/db_install operate against a DB handle.

2) Reflection mode (spec.orm is None):
   - Reflect required tables from DB based on names referenced in model.yml.
   - In this mode, tables must already exist.
   - db_install() is NOT supported (no DDL definitions available).

Contracts validated (on the *actual DB schema* in db_validate/db_install):

Nodes:
- Each node type has exactly one node data table (from model.yml)
- Each node table must contain:
  - scenario_id (str-like, NOT NULL)
  - key        (str-like, NOT NULL)
  - uniqueness EXACTLY on (scenario_id, key):
      either PRIMARY KEY exactly (scenario_id, key)
      or a UNIQUE constraint/index exactly (scenario_id, key)

Edges:
- Optional default edge table (default-edge-data-table)
- Optional per-simproc edge tables (ModelSpec.simproc_edge_data_tables mapping)
- Each edge table must contain:
  - scenario_id (str-like, NOT NULL)
  - source_key  (str-like, NOT NULL)
  - target_key  (str-like, NOT NULL)
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Iterable, Mapping, Optional, Sequence, Set, cast

import sqlalchemy as sa
from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import String, Text, Unicode

from disco.database import DbHandle, normalize_db_handle

from .spec import ModelSpec


# -----------------------------
# Errors
# -----------------------------


class ModelOrmError(RuntimeError):
    """Base error for ORM loading/validation failures."""


class ModelOrmLoadError(ModelOrmError):
    """Raised when ORM metadata/tables cannot be loaded/reflected, or install is impossible."""


class ModelOrmValidationError(ModelOrmError):
    """Raised when loaded/reflected tables do not satisfy required constraints."""


# -----------------------------
# Public data bundle
# -----------------------------


@dataclass(frozen=True, slots=True)
class OrmBundle:
    """
    ORM bundle attached to a loaded Model.

    The bundle contains:
    - provider/reflect metadata (mostly useful as a source of Table objects and (optionally) DDL)
    - resolved Table objects for nodes/edges (usable by Graph/scenario readers)
    - the table-name mapping needed to validate/install against a live DB

    ddl_available:
      True  -> provider mode (metadata contains DDL definitions; db_install supported)
      False -> reflection mode (metadata is reflected; db_install NOT supported)
    """

    metadata: MetaData

    # Usable Table objects (resolved from metadata)
    node_tables: Mapping[str, Table]                 # node_type -> Table
    default_edge_table: Optional[Table]              # default edge table, if configured
    edge_tables_by_simproc: Mapping[str, Table]      # simproc -> Table

    # Name mapping used by db_validate/db_install
    node_table_names: Mapping[str, str]              # node_type -> table name
    default_edge_table_name: Optional[str]           # table name
    edge_table_names_by_simproc: Mapping[str, str]   # simproc -> table name
    required_table_names: Set[str]                   # all required table names (no schema prefix)

    ddl_available: bool


# -----------------------------
# Helpers: python ref importing
# -----------------------------


def _parse_python_ref(ref: str) -> tuple[str, str]:
    """
    Parse "pkg.module:attr" or "pkg.module.attr" into (module, attr).
    """
    r = (ref or "").strip()
    if not r:
        raise ModelOrmLoadError("ORM provider reference is empty")

    if ":" in r:
        mod, attr = r.split(":", 1)
        mod, attr = mod.strip(), attr.strip()
    else:
        if "." not in r:
            raise ModelOrmLoadError(
                f"Invalid ORM provider reference '{ref}'. Expected 'pkg.module:callable' or 'pkg.module.callable'."
            )
        mod, attr = r.rsplit(".", 1)
        mod, attr = mod.strip(), attr.strip()

    if not mod or not attr:
        raise ModelOrmLoadError(
            f"Invalid ORM provider reference '{ref}'. Expected 'pkg.module:callable' or 'pkg.module.callable'."
        )
    return mod, attr


def load_metadata_from_provider(provider_ref: str) -> MetaData:
    """
    Import and call a provider function that returns SQLAlchemy MetaData.

    The provider callable must return:
      - sqlalchemy.MetaData

    Example (model.yml):
      orm: "my_model_pkg.orm:get_metadata"
    """
    mod_name, attr_name = _parse_python_ref(provider_ref)
    try:
        mod = import_module(mod_name)
    except Exception as e:
        raise ModelOrmLoadError(f"Failed to import ORM provider module '{mod_name}': {e}") from e

    try:
        provider = getattr(mod, attr_name)
    except AttributeError as e:
        raise ModelOrmLoadError(f"ORM provider module '{mod_name}' has no attribute '{attr_name}'") from e

    if not callable(provider):
        raise ModelOrmLoadError(
            f"ORM provider '{provider_ref}' did not resolve to a callable (got {type(provider)!r})"
        )

    try:
        md = provider()
    except Exception as e:
        raise ModelOrmLoadError(f"ORM provider '{provider_ref}' raised: {e}") from e

    if not isinstance(md, MetaData):
        raise ModelOrmLoadError(
            f"ORM provider '{provider_ref}' must return sqlalchemy.MetaData, got {type(md)!r}"
        )

    return md


# -----------------------------
# Helpers: reflection
# -----------------------------


def _inspector(conn_or_engine: Engine | Connection) -> sa.inspect:
    return sa.inspect(conn_or_engine)


def _table_exists(conn_or_engine: Engine | Connection, table_name: str, schema: Optional[str]) -> bool:
    insp = _inspector(conn_or_engine)
    return bool(insp.has_table(table_name, schema=schema))


def reflect_tables(
    conn_or_engine: Engine | Connection,
    table_names: Iterable[str],
    *,
    schema: Optional[str] = None,
) -> MetaData:
    """
    Reflect a set of tables into a new MetaData.

    Raises ModelOrmLoadError with a clear list if any required tables are missing.
    """
    names = [n.strip() for n in table_names if (n or "").strip()]
    missing = [n for n in names if not _table_exists(conn_or_engine, n, schema)]
    if missing:
        raise ModelOrmLoadError(
            "Required tables are missing from the database: " + ", ".join(sorted(set(missing)))
        )

    md = MetaData()
    for n in names:
        try:
            Table(n, md, schema=schema, autoload_with=conn_or_engine)
        except Exception as e:
            raise ModelOrmLoadError(f"Failed to reflect table '{n}' (schema={schema!r}): {e}") from e
    return md


# -----------------------------
# Helpers: table lookup in metadata
# -----------------------------


def _metadata_table_key(table_name: str, schema: Optional[str]) -> str:
    return f"{schema}.{table_name}" if schema else table_name


def get_table_from_metadata(md: MetaData, table_name: str, *, schema: Optional[str] = None) -> Table:
    """
    Resolve a table from MetaData by name (+ optional schema).
    """
    key = _metadata_table_key(table_name, schema)
    if key in md.tables:
        return md.tables[key]
    # fallback for providers that didn't set schema on tables but caller passed schema
    if schema and table_name in md.tables:
        return md.tables[table_name]
    raise ModelOrmLoadError(
        f"ORM metadata does not contain required table '{table_name}' (schema={schema!r}). "
        f"Known tables: {sorted(md.tables.keys())}"
    )


# -----------------------------
# Validation helpers
# -----------------------------


def _is_str_type(col: Column) -> bool:
    try:
        py_t = col.type.python_type  # may raise NotImplementedError
        if py_t is str:
            return True
    except Exception:
        pass
    return isinstance(col.type, (String, Unicode, Text))


def _require_col(table: Table, name: str) -> Column:
    if name not in table.c:
        raise ModelOrmValidationError(
            f"Table '{table.fullname}' is missing required column '{name}'. "
            f"Columns: {list(table.c.keys())}"
        )
    return table.c[name]


def _require_not_null(col: Column) -> None:
    if getattr(col, "nullable", True):
        raise ModelOrmValidationError(f"Column '{col.table.fullname}.{col.name}' must be NOT NULL.")


def _require_str_col(col: Column, *, what: str) -> None:
    if not _is_str_type(col):
        raise ModelOrmValidationError(
            f"Column '{col.table.fullname}.{col.name}' must be {what} (string-like). "
            f"Got SQLAlchemy type {col.type!r}"
        )


def _iter_index_columns(idx: sa.Index) -> Iterable[sa.Column]:
    cols_attr = getattr(idx, "columns", None)
    if cols_attr is None:
        return ()
    if callable(cols_attr):
        cols_attr = cols_attr()
    return cast(Iterable[sa.Column], cols_attr)


def _colset_equals(cols: Sequence[sa.Column], names: Sequence[str]) -> bool:
    return set(c.name for c in cols) == set(names) and len(cols) == len(set(names))


def _has_unique_on(table: Table, col_names: Sequence[str]) -> bool:
    """
    Check for uniqueness EXACTLY on col_names.
    """
    pk_cols = list(table.primary_key.columns) if table.primary_key is not None else []
    if pk_cols and _colset_equals(pk_cols, col_names):
        return True

    for c in table.constraints:
        if isinstance(c, sa.UniqueConstraint):
            if _colset_equals(list(c.columns), col_names):
                return True

    for idx in table.indexes:
        try:
            if getattr(idx, "unique", False) and _colset_equals(list(_iter_index_columns(idx)), col_names):
                return True
        except Exception:
            pass

    return False


def validate_node_table(table: Table, *, node_type: str) -> None:
    scenario_id = _require_col(table, "scenario_id")
    _require_not_null(scenario_id)
    _require_str_col(scenario_id, what="str")

    key = _require_col(table, "key")
    _require_not_null(key)
    _require_str_col(key, what="str")

    if not _has_unique_on(table, ["scenario_id", "key"]):
        raise ModelOrmValidationError(
            f"Node table '{table.fullname}' for node type '{node_type}' must enforce uniqueness "
            "exactly on (scenario_id, key) via PRIMARY KEY or UNIQUE constraint/index."
        )


def validate_edge_table(table: Table) -> None:
    scenario_id = _require_col(table, "scenario_id")
    _require_not_null(scenario_id)
    _require_str_col(scenario_id, what="str")

    source_key = _require_col(table, "source_key")
    _require_not_null(source_key)
    _require_str_col(source_key, what="str")

    target_key = _require_col(table, "target_key")
    _require_not_null(target_key)
    _require_str_col(target_key, what="str")


# -----------------------------
# Spec -> required names
# -----------------------------


def required_table_names_from_spec(spec: ModelSpec) -> Set[str]:
    """
    Compute the set of tables referenced by model.yml.

    `spec` is expected to be disco.model.spec.ModelSpec, but kept structurally typed.
    """
    names: Set[str] = set()

    for nts in spec.node_types.values():
        names.add(nts.node_data_table)

    if spec.default_edge_data_table:
        names.add(spec.default_edge_data_table)

    # normalized mapping: simproc_name -> edge table name
    for tname in spec.simproc_edge_data_tables.values():
        names.add(tname)

    return {n.strip() for n in names if (n or "").strip()}


# -----------------------------
# Main entry: build bundle (no DB writes)
# -----------------------------


def build_orm_bundle(
    *,
    spec: ModelSpec,
    db: Optional[DbHandle],
    schema: Optional[str] = None,
) -> OrmBundle:
    """
    Build the ORM bundle for a Model.

    - Provider mode (spec.orm set):
        load MetaData from provider, resolve required tables from that MetaData.
        Does NOT validate against DB; use db_validate/db_install.

    - Reflection mode (spec.orm not set):
        requires db handle; reflects required tables from DB.
        Tables must already exist. db_install is not supported.
    """
    required_names = required_table_names_from_spec(spec)
    provider_ref = spec.orm

    if provider_ref:
        md = load_metadata_from_provider(provider_ref)
        ddl_available = True

        # Ensure provider metadata contains all referenced tables
        for tname in required_names:
            _ = get_table_from_metadata(md, tname, schema=schema)

    else:
        if db is None:
            raise ModelOrmLoadError("Reflection mode requires a db handle (Engine|Connection|SessionManager).")

        conn_or_engine = normalize_db_handle(db)
        md = reflect_tables(conn_or_engine, required_names, schema=schema)
        ddl_available = False

    # Resolve tables from md
    node_tables: dict[str, Table] = {}
    node_table_names: dict[str, str] = {}
    for node_type_name, nts in spec.node_types.items():
        node_table_names[node_type_name] = nts.node_data_table
        node_tables[node_type_name] = get_table_from_metadata(md, nts.node_data_table, schema=schema)

    default_edge_table: Optional[Table] = None
    default_edge_table_name: Optional[str] = spec.default_edge_data_table
    if default_edge_table_name:
        default_edge_table = get_table_from_metadata(md, default_edge_table_name, schema=schema)

    edge_tables_by_simproc: dict[str, Table] = {}
    edge_table_names_by_simproc: dict[str, str] = {}
    for simproc_name, table_name in spec.simproc_edge_data_tables.items():
        edge_table_names_by_simproc[simproc_name] = table_name
        edge_tables_by_simproc[simproc_name] = get_table_from_metadata(md, table_name, schema=schema)

    # NOTE: We do NOT validate here against DB.
    # - Provider mode: validate DB later via db_validate/db_install.
    # - Reflection mode: tables are already real DB tables; but we still prefer to validate via db_validate()
    #   for consistent behavior and better error messages.
    return OrmBundle(
        metadata=md,
        node_tables=node_tables,
        default_edge_table=default_edge_table,
        edge_tables_by_simproc=edge_tables_by_simproc,
        node_table_names=node_table_names,
        default_edge_table_name=default_edge_table_name,
        edge_table_names_by_simproc=edge_table_names_by_simproc,
        required_table_names=set(required_names),
        ddl_available=ddl_available,
    )


# -----------------------------
# Deliberate DB actions
# -----------------------------


def db_validate(bundle: OrmBundle, db: DbHandle, *, schema: Optional[str] = None) -> None:
    """
    Validate the actual database schema for this bundle.

    Implementation detail:
    - We reflect the required tables from the DB into a fresh MetaData.
    - We run contract validators against the reflected tables.
    """
    conn_or_engine = normalize_db_handle(db)
    md = reflect_tables(conn_or_engine, bundle.required_table_names, schema=schema)

    # nodes
    for node_type, tname in bundle.node_table_names.items():
        tbl = get_table_from_metadata(md, tname, schema=schema)
        validate_node_table(tbl, node_type=node_type)

    # edges: default
    if bundle.default_edge_table_name:
        tbl = get_table_from_metadata(md, bundle.default_edge_table_name, schema=schema)
        validate_edge_table(tbl)

    # edges: per-simproc
    for simproc_name, tname in bundle.edge_table_names_by_simproc.items():
        _ = simproc_name  # keep name for debugging/breakpoints if needed
        tbl = get_table_from_metadata(md, tname, schema=schema)
        validate_edge_table(tbl)


def db_install(bundle: OrmBundle, db: DbHandle, *, schema: Optional[str] = None) -> None:
    """
    Create missing tables according to bundle.metadata (checkfirst=True), then validate.

    Requires provider mode (DDL definitions must be available).
    """
    if not bundle.ddl_available:
        raise ModelOrmLoadError(
            "db_install is only supported when an ORM provider is configured (DDL definitions available). "
            "In reflection mode, tables must be created externally."
        )

    conn_or_engine = normalize_db_handle(db)
    try:
        bundle.metadata.create_all(conn_or_engine, checkfirst=True)
    except Exception as e:
        raise ModelOrmLoadError(f"Failed to create tables from ORM metadata: {e}") from e

    db_validate(bundle, db, schema=schema)
