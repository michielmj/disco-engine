# src/disco/model/spec.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SimProcSpec(BaseModel):
    """
    SimProc definition (type only, not instances).

    edge_data_table:
        Optional scenario DB table containing edge-instance data for edges associated with this simproc.
        If not provided for a given simproc, callers may fall back to ModelSpec.default_edge_data_table.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    edge_data_table: Optional[str] = Field(default=None, alias="edge-data-table", min_length=1)

    # noinspection PyNestedDecorators
    @field_validator("edge_data_table")
    @classmethod
    def _validate_edge_data_table(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("edge-data-table may not be empty/whitespace")
        return s


class NodeTypeSpec(BaseModel):
    """
    Node type definition (type only, not instances).

    python_class:
        Import path for the Node implementation class.
        Accepts "pkg.module:ClassName" or "pkg.module.ClassName".
    node_data_table:
        Scenario DB table containing node-instance data for this node type.
    distinct_nodes:
        Vertex label attributes used to split vertices into distinct node instances.
    self_relations:
        List of (higher_simproc_name, lower_simproc_name) relations within a node instance.
        "Higher" means smaller simproc order (e.g. 0 -> 1 is OK; 1 -> 0 is not).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    python_class: str = Field(..., alias="class", min_length=1)
    node_data_table: str = Field(..., alias="node-data-table", min_length=1)
    distinct_nodes: List[str] = Field(default_factory=list, alias="distinct-nodes")
    self_relations: List[Tuple[str, str]] = Field(default_factory=list, alias="self-relations")

    # noinspection PyNestedDecorators
    @field_validator("python_class")
    @classmethod
    def _validate_python_class(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("node type class may not be empty/whitespace")
        if ":" not in s and "." not in s:
            raise ValueError("node type class must look like 'pkg.module:ClassName' or 'pkg.module.ClassName'")
        return s

    # noinspection PyNestedDecorators
    @field_validator("node_data_table")
    @classmethod
    def _validate_node_data_table(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("node-data-table may not be empty/whitespace")
        return s

    # noinspection PyNestedDecorators
    @field_validator("distinct_nodes")
    @classmethod
    def _validate_distinct_nodes(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in v:
            s = (item or "").strip()
            if not s:
                raise ValueError("distinct-nodes may not contain empty strings")
            if s in seen:
                raise ValueError(f"distinct-nodes contains duplicate attribute '{s}'")
            seen.add(s)
            cleaned.append(s)
        return cleaned

    # noinspection PyNestedDecorators
    @field_validator("self_relations")
    @classmethod
    def _validate_self_relations_shape(cls, v: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        cleaned: List[Tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for higher, lower in v:
            hgr = (higher or "").strip()
            lwr = (lower or "").strip()
            if not hgr or not lwr:
                raise ValueError("self-relations entries must be (non-empty, non-empty)")
            if hgr == lwr:
                raise ValueError("self-relations may not relate a simproc to itself")
            key = (hgr, lwr)
            if key in seen:
                raise ValueError(f"self-relations contains duplicate entry {key}")
            seen.add(key)
            cleaned.append(key)
        return cleaned


class ModelSpec(BaseModel):
    """
    Full model.yml specification.

    Simprocs may be specified in two forms:

    1) Simple form (list of names):
       simprocs:
         - demand
         - supply

       Use this form when you only need default-edge-data-table.

    2) Rich form (mapping from simproc name to SimProcSpec):
       simprocs:
         demand:
           edge-data-table: demand_edges
         supply:
           edge-data-table: supply_edges

       In this form, simproc order is defined by YAML mapping insertion order.
       Per-simproc edge-data-table is optional; callers may fall back to default-edge-data-table.

    Internally, this spec is normalized to:
      - simprocs: List[str] (ordered simproc names)
      - simproc_edge_data_tables: Dict[str, str] (simproc name -> edge data table) for those simprocs that define one
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)

    # Normalized ordered simproc names
    simprocs: List[str]

    # Internal normalized mapping: simproc_name -> edge-data-table (only those explicitly configured)
    simproc_edge_data_tables: Dict[str, str] = Field(default_factory=dict)

    node_types: Dict[str, NodeTypeSpec] = Field(..., alias="node-types")

    orm: Optional[str] = Field(
        default=None,
        alias="orm",
        description="ORM provider reference. If none is given, ORM will be obtained via database reflection.",
    )

    default_edge_data_table: Optional[str] = Field(
        default=None,
        alias="default-edge-data-table",
        description="Scenario table for default edge data (used when no simproc-specific edge table applies).",
    )

    # -----------------------------
    # Pre-parse normalization for simprocs input shape
    # -----------------------------

    @model_validator(mode="before")
    @classmethod
    def _normalize_simprocs_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("model.yml root must be a mapping/dict")

        raw_simprocs = data.get("simprocs", None)
        if raw_simprocs is None:
            raise ValueError("simprocs is required")

        simproc_list: List[str] = []
        simproc_tables: Dict[str, str] = {}

        # Simple form: list[str]
        if isinstance(raw_simprocs, list):
            if not raw_simprocs:
                raise ValueError("simprocs must not be empty")

            seen: set[str] = set()
            for item in raw_simprocs:
                s = (item or "").strip()
                if not s:
                    raise ValueError("simprocs may not contain empty strings")
                if s in seen:
                    raise ValueError(f"simprocs contains duplicate name '{s}'")
                seen.add(s)
                simproc_list.append(s)

        # Rich form: mapping[str, SimProcSpec]
        elif isinstance(raw_simprocs, Mapping):
            if not raw_simprocs:
                raise ValueError("simprocs must not be empty")

            seen2: set[str] = set()
            # NOTE: mapping insertion order defines simproc order
            for name, cfg in raw_simprocs.items():
                sp_name = (name or "").strip()
                if not sp_name:
                    raise ValueError("simprocs keys may not be empty/whitespace")
                if sp_name in seen2:
                    # Dict keys can't duplicate, but keep a clear message if input was weird
                    raise ValueError(f"simprocs contains duplicate name '{sp_name}'")
                seen2.add(sp_name)
                simproc_list.append(sp_name)

                if cfg is None:
                    continue
                if not isinstance(cfg, Mapping):
                    raise ValueError(
                        f"simprocs['{sp_name}'] must be a mapping (e.g. {{edge-data-table: ...}}) or null"
                    )

                # Let SimProcSpec validate the inner config.
                sp_spec = SimProcSpec.model_validate(cfg)
                if sp_spec.edge_data_table:
                    simproc_tables[sp_name] = sp_spec.edge_data_table

        else:
            raise ValueError("simprocs must be either a list of names or a mapping of simproc specs")

        # Store normalized forms back into the model input
        data = dict(data)
        data["simprocs"] = simproc_list
        data["simproc_edge_data_tables"] = simproc_tables
        return data

    # -----------------------------
    # Field validators
    # -----------------------------

    # noinspection PyNestedDecorators
    @field_validator("default_edge_data_table")
    @classmethod
    def _validate_default_edge_data_table(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("default-edge-data-table may not be empty/whitespace")
        return s

    # noinspection PyNestedDecorators
    @field_validator("orm")
    @classmethod
    def _validate_orm_ref(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("orm may not be empty/whitespace")
        # Keep loose: allow either "pkg.module:callable" or "pkg.module.callable"
        if ":" not in s and "." not in s:
            raise ValueError("orm must look like 'pkg.module:callable' or 'pkg.module.callable'")
        return s

    # -----------------------------
    # Cross-field validation
    # -----------------------------

    @model_validator(mode="after")
    def _validate_cross_refs(self) -> "ModelSpec":
        order_by_name = {name: i for i, name in enumerate(self.simprocs)}

        # node-types keys must be non-empty/trimmed
        for k in self.node_types.keys():
            if not (k or "").strip():
                raise ValueError("node-types keys may not be empty/whitespace")

        # validate self_relations simproc references and ordering (higher => smaller index)
        for node_name, nts in self.node_types.items():
            for higher, lower in nts.self_relations:
                if higher not in order_by_name:
                    raise ValueError(
                        f"node-type '{node_name}' self-relations references unknown simproc '{higher}'"
                    )
                if lower not in order_by_name:
                    raise ValueError(
                        f"node-type '{node_name}' self-relations references unknown simproc '{lower}'"
                    )
                if order_by_name[higher] >= order_by_name[lower]:
                    raise ValueError(
                        f"node-type '{node_name}' self-relations requires higher->lower "
                        f"with smaller->larger simproc order; got "
                        f"({higher}:{order_by_name[higher]}) -> ({lower}:{order_by_name[lower]})"
                    )

        # validate simproc_edge_data_tables keys reference known simprocs
        for sp_name, table_name in self.simproc_edge_data_tables.items():
            if sp_name not in order_by_name:
                raise ValueError(f"simproc edge-data-table references unknown simproc '{sp_name}'")
            if not (table_name or "").strip():
                raise ValueError(f"simproc '{sp_name}' edge-data-table may not be empty/whitespace")

        # optionally: ensure no duplicate table names across simproc_edge_data_tables
        # (helps avoid confusion)
        seen_tables: set[str] = set()
        for sp_name, table_name in self.simproc_edge_data_tables.items():
            if table_name in seen_tables:
                raise ValueError(f"duplicate edge-data-table '{table_name}' used for multiple simprocs")
            seen_tables.add(table_name)

        return self

    def simproc_order(self, simproc_name: str) -> int:
        for i, name in enumerate(self.simprocs):
            if name == simproc_name:
                return i
        raise KeyError(simproc_name)
