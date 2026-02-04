# tests/model/test_spec.py
import pytest
from pydantic import ValidationError

from disco.model.spec import ModelSpec


def _base_model_dict_list_simprocs():
    # Simple simprocs form (list): intended for default-edge-data-table only
    return {
        "name": "test-model",
        "version": "1.0",
        "simprocs": ["demand", "supply"],
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


def _base_model_dict_map_simprocs():
    # Rich simprocs form (mapping): enables per-simproc edge-data-table
    return {
        "name": "test-model",
        "version": "1.0",
        "simprocs": {
            "demand": {"edge-data-table": "demand_edges"},
            "supply": {},
        },
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


def test_modelspec_valid_list_simprocs():
    spec = ModelSpec.model_validate(_base_model_dict_list_simprocs())
    assert spec.simprocs == ["demand", "supply"]
    assert spec.simproc_edge_data_tables == {}
    assert "Warehouse" in spec.node_types
    assert spec.node_types["Warehouse"].self_relations == [("demand", "supply")]
    assert spec.default_edge_data_table == "edges_default"


def test_modelspec_valid_map_simprocs_normalizes():
    spec = ModelSpec.model_validate(_base_model_dict_map_simprocs())
    assert spec.simprocs == ["demand", "supply"]  # preserves mapping insertion order
    assert spec.simproc_edge_data_tables == {"demand": "demand_edges"}
    assert "Warehouse" in spec.node_types
    assert spec.node_types["Warehouse"].self_relations == [("demand", "supply")]
    assert spec.default_edge_data_table == "edges_default"


def test_modelspec_simprocs_empty_rejected_list():
    d = _base_model_dict_list_simprocs()
    d["simprocs"] = []
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_simprocs_duplicate_rejected_list():
    d = _base_model_dict_list_simprocs()
    d["simprocs"] = ["demand", "demand"]
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_simprocs_whitespace_rejected_list():
    d = _base_model_dict_list_simprocs()
    d["simprocs"] = ["demand", "   "]
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_simprocs_empty_rejected_map():
    d = _base_model_dict_map_simprocs()
    d["simprocs"] = {}
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_simprocs_blank_key_rejected_map():
    d = _base_model_dict_map_simprocs()
    d["simprocs"] = {"   ": {"edge-data-table": "x"}}
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_simprocs_edge_data_table_blank_rejected_map():
    d = _base_model_dict_map_simprocs()
    d["simprocs"]["demand"]["edge-data-table"] = "   "
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_simproc_edge_data_table_duplicate_table_name_rejected():
    d = _base_model_dict_map_simprocs()
    d["simprocs"]["supply"] = {"edge-data-table": "demand_edges"}  # duplicate table name
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_self_relations_unknown_simproc_rejected():
    d = _base_model_dict_list_simprocs()
    d["node-types"]["Warehouse"]["self-relations"] = [["demand", "MISSING"]]
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_self_relations_wrong_order_rejected():
    # higher must be earlier in the simprocs list (lower index)
    d = _base_model_dict_list_simprocs()
    d["node-types"]["Warehouse"]["self-relations"] = [["supply", "demand"]]
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)


def test_modelspec_default_edge_data_table_blank_rejected():
    d = _base_model_dict_list_simprocs()
    d["default-edge-data-table"] = "   "
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(d)
