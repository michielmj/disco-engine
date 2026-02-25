# src/disco/graph/extract.py

from __future__ import annotations

from typing import Sequence, Mapping, Any, Optional, Literal, cast

import numpy as np
import pandas as pd
import graphblas as gb
from sqlalchemy import select, and_, literal
from sqlalchemy.orm import Session
from sqlalchemy import BigInteger, Boolean, Double, Float, Integer, Numeric, SmallInteger
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.engine import RowMapping

from .core import Graph
from .graph_mask import GraphMask
from .schema import vertex_masks, edges as edges_table, vertices as vertices_table

Backend = Literal["pandas"]  # reserved for future extension
IndexBy = Literal["index", "key"]
EdgeIndexBy = Literal["indices", "keys"]

NumericType = BigInteger | Boolean | Double | Float | Integer | Numeric | SmallInteger


def _rows_to_df(rows: Sequence[Mapping[str, Any]], columns: Sequence[ColumnElement]) -> pd.DataFrame:
    """
    Convert SQLAlchemy RowMapping list (or any mapping sequence) to a pandas DataFrame.
    """
    col_names = [c.name for c in columns]
    if not rows:
        return pd.DataFrame(columns=col_names)
    return pd.DataFrame(rows, columns=col_names)


# ---------------------------------------------------------------------------
# 1. Vertex data via index→key mapping (graph.vertices)
# ---------------------------------------------------------------------------

def get_vertex_data(
        session: Session,
        graph: Graph,
        vertex_table: Table,
        columns: Sequence[ColumnElement[Any]],
        *,
        mask: Optional[GraphMask] = None,
        index_by: IndexBy = "index",
        default_fill: Any | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with vertex-level data from a *model node table*.

    The model node table (vertex_table) is keyed by (scenario_id, key),
    while the Graph structure uses (scenario_id, index). The mapping
    from index -> key is stored in graph.vertices.

    Semantics:
      - We start from graph.vertices (mapping table).
      - Optionally filter by the GraphMask (mask or graph.graph_mask).
      - LEFT OUTER JOIN to the model node table on (scenario_id, key).
      - Return one row per vertex in the mask (or all vertices), with
        requested columns from vertex_table.
      - Missing model rows -> NaN/null, unless default_fill is provided.

    Requirements:
      - graph.vertices must have columns:
            scenario_id, index, key
      - vertex_table must have columns:
            scenario_id, key
    """
    # Validate node table shape
    if not hasattr(vertex_table.c, "scenario_id") or not hasattr(vertex_table.c, "key"):
        raise ValueError(
            "vertex_table must have 'scenario_id' and 'key' columns "
            "with the same semantics as the model node tables."
        )

    vmap = vertices_table
    eff_mask: Optional[GraphMask] = mask if mask is not None else graph.graph_mask

    all_columns = (
        vmap.c.index.label("index"),
        vmap.c.key.label("key"),
        *columns,
    )

    # Base: all vertices in this graph scenario_id
    # We always include index and key in the projection so we can index
    base = (
        select(
            *all_columns,
        )
        .select_from(vmap)
        .outerjoin(
            vertex_table,
            and_(
                vertex_table.c.scenario_id == vmap.c.scenario_id,
                vertex_table.c.key == vmap.c.key,
            ),
        )
        .where(vmap.c.scenario_id == literal(graph.scenario_id))
        .order_by(vmap.c.index)
    )

    # Optional mask: filter vertices by index via vertex_masks
    if eff_mask is not None:
        eff_mask.ensure_persisted(session)
        vm = vertex_masks
        base = base.join(
            vm,
            and_(
                vm.c.scenario_id == vmap.c.scenario_id,
                vm.c.vertex_index == vmap.c.index,
                vm.c.mask_id == literal(eff_mask.mask_id),
            ),
        )

    result = session.execute(base)
    raw_rows: list[RowMapping] = list(result.mappings())
    rows = cast(Sequence[Mapping[str, Any]], raw_rows)
    df = _rows_to_df(rows, all_columns)

    # Index by vertex index (default) or vertex key
    if index_by == "index":
        df = df.set_index("index")
    elif index_by == "key":
        df = df.set_index("key")
    else:
        raise ValueError("index_by must be 'index' or 'key'")

    if default_fill is not None:
        df = df.fillna(default_fill)

    return df


def get_vertex_numeric_vector(
        session: Session,
        graph: Graph,
        vertex_table: Table,
        value_column: ColumnElement[NumericType],
        *,
        mask: Optional[GraphMask] = None,
        default_value: float = 0.0,
) -> gb.Vector:
    """
    Return a GraphBLAS Vector[FP64] of size graph.num_vertices with values taken
    from a single numeric column in a model node table.

    - vertex_table is keyed by (scenario_id, key).
    - Mapping from index to key is via graph.vertices.
    - For vertices without a corresponding node row (or NULL value), we use
      default_value (usually 0.0).
    - If default_value == 0.0, missing values are simply omitted from the sparse
      representation.
    """
    if not hasattr(vertex_table.c, "scenario_id") or not hasattr(vertex_table.c, "key"):
        raise ValueError(
            "vertex_table must have 'scenario_id' and 'key' columns."
        )

    vmap = vertices_table
    eff_mask: Optional[GraphMask] = mask if mask is not None else graph.graph_mask

    base = (
        select(
            vmap.c.index.label("index"),
            value_column.label("val"),
        )
        .select_from(vmap)
        .outerjoin(
            vertex_table,
            and_(
                vertex_table.c.scenario_id == vmap.c.scenario_id,
                vertex_table.c.key == vmap.c.key,
            ),
        )
        .where(vmap.c.scenario_id == literal(graph.scenario_id))
    )

    if eff_mask is not None:
        eff_mask.ensure_persisted(session)
        vm = vertex_masks
        base = base.join(
            vm,
            and_(
                vm.c.scenario_id == vmap.c.scenario_id,
                vm.c.vertex_index == vmap.c.index,
                vm.c.mask_id == literal(eff_mask.mask_id),
            ),
        )

    result = session.execute(base)
    raw_rows: list[RowMapping] = list(result.mappings())

    if not raw_rows:
        return gb.Vector.sparse(gb.dtypes.FP64, size=graph.num_vertices)

    idxs: list[int] = []
    vals: list[float] = []

    for r in raw_rows:
        v_idx = int(r["index"])
        v_val = r["val"]
        if v_val is None:
            if default_value == 0.0:
                # implicit zero; skip in sparse representation
                continue
            v_val = default_value
        idxs.append(v_idx)
        vals.append(float(v_val))

    if not idxs:
        return gb.Vector.sparse(gb.dtypes.FP64, size=graph.num_vertices)

    idx_arr = np.asarray(idxs, dtype=np.int64)
    val_arr = np.asarray(vals, dtype=np.float64)

    return gb.Vector.from_coo(
        idx_arr,
        val_arr,
        gb.dtypes.FP64,
        size=graph.num_vertices,
    )


# ---------------------------------------------------------------------------
# 2. Edge data (outbound / inbound) – key-based tables, index→key mapping
# ---------------------------------------------------------------------------

def _validate_edge_table(edge_table: Table) -> None:
    """
    Validate that a *model* edge table has the expected key-based columns.

    Requirements:
      - scenario_id
      - source_key
      - target_key
    """
    required = ("scenario_id", "source_key", "target_key")
    missing = [name for name in required if not hasattr(edge_table.c, name)]
    if missing:
        raise ValueError(
            f"edge_table must have columns {required}, missing: {missing}"
        )


def get_outbound_edge_data(
        session: Session,
        graph: Graph,
        edge_table: Table,
        columns: Sequence[ColumnElement[Any]],
        *,
        layer_idx: int,
        mask: Optional[GraphMask] = None,
        index_by: EdgeIndexBy = "indices",
        default_fill: Any | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per outbound edge from vertices in the mask
    (or all vertices if no mask), for a specific layer.

    Model edge tables are key-based:
      - scenario_id, source_key, target_key, ...

    Structural edges in the graph schema are index-based:
      - graph.edges with (scenario_id, layer_idx, source_idx, target_idx).

    We:
      - Start from graph.edges (indices).
      - Join vertices twice to map source_idx/target_idx -> source_key/target_key.
      - LEFT OUTER JOIN the model edge table on (scenario_id, source_key, target_key).
      - Optionally filter by GraphMask on the source index.
      - Return a DataFrame indexed by:
          * ('source_index', 'target_index') if index_by == "indices" (default), or
          * ('source_key', 'target_key') if index_by == "keys".

    Missing model edge rows produce NaN/null in the DataFrame, unless default_fill is set.
    """
    _validate_edge_table(edge_table)
    eff_mask: Optional[GraphMask] = mask if mask is not None else graph.graph_mask

    e = edges_table
    v_src = vertices_table.alias("v_src")
    v_tgt = vertices_table.alias("v_tgt")
    ed = edge_table

    all_columns = (
        e.c.source_idx.label("source_index"),
        e.c.target_idx.label("target_index"),
        v_src.c.key.label("source_key"),
        v_tgt.c.key.label("target_key"),
        *columns,
    )

    base = (
        select(
            *all_columns
        )
        .select_from(e)
        .join(
            v_src,
            and_(
                v_src.c.scenario_id == e.c.scenario_id,
                v_src.c.index == e.c.source_idx,
            ),
        )
        .join(
            v_tgt,
            and_(
                v_tgt.c.scenario_id == e.c.scenario_id,
                v_tgt.c.index == e.c.target_idx,
            ),
        )
        .outerjoin(
            ed,
            and_(
                ed.c.scenario_id == e.c.scenario_id,
                ed.c.source_key == v_src.c.key,
                ed.c.target_key == v_tgt.c.key,
            ),
        )
        .where(
            and_(
                e.c.scenario_id == literal(graph.scenario_id),
                e.c.layer_idx == int(layer_idx),
            )
        )
    )

    # Optional mask: filter by source vertex index
    if eff_mask is not None:
        eff_mask.ensure_persisted(session)
        vm = vertex_masks
        base = base.join(
            vm,
            and_(
                vm.c.scenario_id == e.c.scenario_id,
                vm.c.vertex_index == e.c.source_idx,
                vm.c.mask_id == literal(eff_mask.mask_id),
            ),
        )

    result = session.execute(base)
    raw_rows: list[RowMapping] = list(result.mappings())
    rows = cast(Sequence[Mapping[str, Any]], raw_rows)
    df = _rows_to_df(rows, all_columns)

    # Multi-index: indices or keys
    if index_by == "indices":
        df = df.set_index(["source_index", "target_index"])
    elif index_by == "keys":
        df = df.set_index(["source_key", "target_key"])
    else:
        raise ValueError("index_by must be 'indices' or 'keys'")

    if default_fill is not None:
        df = df.fillna(default_fill)

    return df


def get_inbound_edge_data(
        session: Session,
        graph: Graph,
        edge_table: Table,
        columns: Sequence[ColumnElement[Any]],
        *,
        layer_idx: int,
        mask: Optional[GraphMask] = None,
        index_by: EdgeIndexBy = "indices",
        default_fill: Any | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per inbound edge to vertices in the mask
    (or all vertices if no mask), for a specific layer.

    Model edge tables are key-based:
      - scenario_id, source_key, target_key, ...

    Structural edges in the graph schema are index-based:
      - graph.edges with (scenario_id, layer_idx, source_idx, target_idx).

    We:
      - Start from graph.edges (indices).
      - Join vertices twice to map source_idx/target_idx -> source_key/target_key.
      - LEFT OUTER JOIN the model edge table on (scenario_id, source_key, target_key).
      - Optionally filter by GraphMask on the *target* index.
      - Return a DataFrame indexed by:
          * ('source_index', 'target_index') if index_by == "indices" (default), or
          * ('source_key', 'target_key') if index_by == "keys".

    Missing model edge rows produce NaN/null in the DataFrame, unless default_fill is set.
    """
    _validate_edge_table(edge_table)
    eff_mask: Optional[GraphMask] = mask if mask is not None else graph.graph_mask

    e = edges_table
    v_src = vertices_table.alias("v_src")
    v_tgt = vertices_table.alias("v_tgt")
    ed = edge_table

    all_columns = (
        e.c.source_idx.label("source_index"),
        e.c.target_idx.label("target_index"),
        v_src.c.key.label("source_key"),
        v_tgt.c.key.label("target_key"),
        *columns,
    )

    base = (
        select(
            *all_columns,
        )
        .select_from(e)
        .join(
            v_src,
            and_(
                v_src.c.scenario_id == e.c.scenario_id,
                v_src.c.index == e.c.source_idx,
            ),
        )
        .join(
            v_tgt,
            and_(
                v_tgt.c.scenario_id == e.c.scenario_id,
                v_tgt.c.index == e.c.target_idx,
            ),
        )
        .outerjoin(
            ed,
            and_(
                ed.c.scenario_id == e.c.scenario_id,
                ed.c.source_key == v_src.c.key,
                ed.c.target_key == v_tgt.c.key,
            ),
        )
        .where(
            and_(
                e.c.scenario_id == literal(graph.scenario_id),
                e.c.layer_idx == int(layer_idx),
            )
        )
    )

    # Optional mask: filter by target vertex index
    if eff_mask is not None:
        eff_mask.ensure_persisted(session)
        vm = vertex_masks
        base = base.join(
            vm,
            and_(
                vm.c.scenario_id == e.c.scenario_id,
                vm.c.vertex_index == e.c.target_idx,
                vm.c.mask_id == literal(eff_mask.mask_id),
            ),
        )

    result = session.execute(base)
    raw_rows: list[RowMapping] = list(result.mappings())
    rows = cast(Sequence[Mapping[str, Any]], raw_rows)
    df = _rows_to_df(rows, all_columns)

    if index_by == "indices":
        df = df.set_index(["source_index", "target_index"])
    elif index_by == "keys":
        df = df.set_index(["source_key", "target_key"])
    else:
        raise ValueError("index_by must be 'indices' or 'keys'")

    if default_fill is not None:
        df = df.fillna(default_fill)

    return df


# ---------------------------------------------------------------------------
# 3. Map extraction (GraphBLAS matrices, still weight-only from graph.edges)
# ---------------------------------------------------------------------------

def get_outbound_map(
        session: Session,
        graph: Graph,
        *,
        layer_idx: int,
        mask: Optional[GraphMask] = None,
        values: Optional[ColumnElement[NumericType]] = None,
) -> gb.Matrix:
    """
    Return a GraphBLAS Matrix for outbound edges in a given layer,
    using the structural graph.edges table (index-based).

    - Rows: source_idx
    - Columns: target_idx
    - Values: edge weight (from graph.edges table) for now.

    Mask semantics (if provided or set on graph):
    - Only edges whose *source* vertex is in the mask are included.
    """

    eff_mask: Optional[GraphMask] = mask if mask is not None else graph.graph_mask

    e = edges_table
    if values is None:
        values = literal(1.)
    else:
        ...  # TODO

    base = select(
        e.c.source_idx.label("src"),
        e.c.target_idx.label("tgt"),
        values.label("val"),
    ).where(
        and_(
            e.c.scenario_id == literal(graph.scenario_id),
            e.c.layer_idx == int(layer_idx),
        )
    )

    if eff_mask is None:
        stmt = base
    else:
        eff_mask.ensure_persisted(session)
        vm = vertex_masks
        stmt = (
            base.join(
                vm,
                and_(
                    vm.c.scenario_id == e.c.scenario_id,
                    vm.c.vertex_index == e.c.source_idx,
                ),
            )
            .where(vm.c.mask_id == literal(eff_mask.mask_id))
        )

    result = session.execute(stmt)
    raw_rows: list[RowMapping] = list(result.mappings())

    if not raw_rows:
        return gb.Matrix(
            gb.dtypes.FP64, graph.num_vertices, graph.num_vertices
        )

    src = np.fromiter((r["src"] for r in raw_rows), dtype=np.int64)
    tgt = np.fromiter((r["tgt"] for r in raw_rows), dtype=np.int64)
    val = np.fromiter((r["val"] for r in raw_rows), dtype=np.float64)

    return gb.Matrix.from_coo(
        src,
        tgt,
        val,
        nrows=graph.num_vertices,
        ncols=graph.num_vertices,
    )


def get_inbound_map(
        session: Session,
        graph: Graph,
        *,
        layer_idx: int,
        mask: Optional[GraphMask] = None,
        values: Optional[ColumnElement[NumericType]] = None,
) -> gb.Matrix:
    """
    Return a GraphBLAS Matrix for inbound edges in a given layer,
    using the structural graph.edges table (index-based).

    - Rows: source_idx
    - Columns: target_idx
    - Values: edge weight.

    Mask semantics:
    - Only edges whose *target* vertex is in the mask are included.
    """
    eff_mask: Optional[GraphMask] = mask if mask is not None else graph.graph_mask

    e = edges_table
    if values is None:
        values = literal(1.)
    else:
        ...  # TODO

    base = select(
        e.c.source_idx.label("src"),
        e.c.target_idx.label("tgt"),
        values.label("val"),
    ).where(
        and_(
            e.c.scenario_id == literal(graph.scenario_id),
            e.c.layer_idx == int(layer_idx),
        )
    )

    if eff_mask is None:
        stmt = base
    else:
        eff_mask.ensure_persisted(session)
        vm = vertex_masks
        stmt = (
            base.join(
                vm,
                and_(
                    vm.c.scenario_id == e.c.scenario_id,
                    vm.c.vertex_index == e.c.target_idx,
                ),
            )
            .where(vm.c.mask_id == literal(eff_mask.mask_id))
        )

    result = session.execute(stmt)
    raw_rows: list[RowMapping] = list(result.mappings())

    if not raw_rows:
        return gb.Matrix(
            gb.dtypes.FP64, graph.num_vertices, graph.num_vertices
        )

    src = np.fromiter((r["src"] for r in raw_rows), dtype=np.int64)
    tgt = np.fromiter((r["tgt"] for r in raw_rows), dtype=np.int64)
    val = np.fromiter((r["val"] for r in raw_rows), dtype=np.float64)

    return gb.Matrix.from_coo(
        src,
        tgt,
        val,
        nrows=graph.num_vertices,
        ncols=graph.num_vertices,
    )
