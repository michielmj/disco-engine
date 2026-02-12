# tests/graph/test_graph_basic.py
import numpy as np
import graphblas as gb
from graphblas import Vector
import pytest

from disco.graph import Graph


def test_basic_graph_structure() -> None:
    src = np.array([0, 1, 2], dtype=np.int64)
    tgt = np.array([1, 2, 3], dtype=np.int64)
    wgt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    layers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {0: (src, tgt, wgt)}

    graph = Graph.from_edges(layers, num_vertices=5)

    assert graph.num_vertices == 5
    mat = graph.get_matrix(0)

    # Matrix indexing returns a Scalar; use .value for the Python float
    assert mat[1, 2].value == 2.0

    out_edges = graph.get_out_edges(0, 2)
    # out_edges is a Vector with index 3 containing weight 3.0
    assert out_edges[3].value == 3.0


def test_graph_masking_and_view() -> None:
    src = np.array([0, 1], dtype=np.int64)
    tgt = np.array([1, 2], dtype=np.int64)
    wgt = np.array([1.0, 2.0], dtype=np.float64)
    layers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {0: (src, tgt, wgt)}

    graph = Graph.from_edges(layers, num_vertices=4)

    # Mask vertices 1 and 2
    mask = Vector.from_coo([1, 2], [True, True], size=4)
    graph.set_mask(mask)

    # Mask should be stored as a GraphBLAS Vector[BOOL] of size num_vertices
    mask_vec = graph.mask_vector
    assert mask_vec is not None
    assert mask_vec.size == 4

    idx, vals = mask_vec.to_coo()
    assert set(idx.tolist()) == {1, 2}
    # All values should be True
    assert set(vals.tolist()) == {True}

    # The adjacency matrix itself is not resized by the mask
    mat = graph.get_matrix(0)
    assert mat.nrows == graph.num_vertices
    assert mat.ncols == graph.num_vertices
    assert mat[1, 2].value == 2.0

    # get_view() without explicit mask should reuse the existing mask
    view_same = graph.get_view()
    view_mask = view_same.mask_vector
    assert view_mask is not None
    v_idx, v_vals = view_mask.to_coo()
    assert set(v_idx.tolist()) == {1, 2}
    assert set(v_vals.tolist()) == {True}

    # get_view() with a different mask should override it
    other_mask = Vector.from_coo([0, 3], [True, True], size=4)
    view_other = graph.get_view(mask=other_mask)
    other_view_mask = view_other.mask_vector
    assert other_view_mask is not None
    o_idx, o_vals = other_view_mask.to_coo()
    assert set(o_idx.tolist()) == {0, 3}
    assert set(o_vals.tolist()) == {True}

    # Structure and labels are shared between views and original
    assert view_same.layers is graph.layers
    assert view_other.layers is graph.layers


def test_graph_labels_from_set_labels() -> None:
    """
    Verify that label_matrix and label_meta work together and that we can
    derive vertex sets and per-type metadata.
    """
    num_vertices = 4

    # We define 3 labels with global ids 0, 1, 2:
    #  0: type1 / "A"
    #  1: type2 / "B"
    #  2: type1 / "C"
    #
    # Vertex-label assignments (vertex_index, label_id):
    #  (0, 0), (1, 1), (2, 2), (3, 0)
    v_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    l_idx = np.array([0, 1, 2, 0], dtype=np.int64)
    vals = np.ones(len(v_idx), dtype=bool)

    label_matrix = gb.Matrix.from_coo(
        v_idx,
        l_idx,
        vals,
        nrows=num_vertices,
        ncols=3,
    )

    label_meta = {
        0: ("type1", "A"),
        1: ("type2", "B"),
        2: ("type1", "C"),
    }

    # Empty structural graph; we're testing labels here
    graph = Graph.from_edges({}, num_vertices=num_vertices, scenario_id="test-scenario")
    graph.set_labels(label_matrix, label_meta)

    assert graph.num_labels == 3
    assert graph.label_matrix is not None
    assert graph.label_matrix.nrows == num_vertices
    assert graph.label_matrix.ncols == 3

    # Vertices that have label id 0: vertices 0 and 3
    vertices0 = graph.get_vertices_for_label(0)
    assert set(vertices0.tolist()) == {0, 3}

    # Per-type helpers
    idxs_type1, sub_type1 = graph.labels_for_type("type1")
    assert set(idxs_type1.tolist()) == {0, 2}
    assert sub_type1.nrows == num_vertices
    assert sub_type1.ncols == 2

    value_to_index = graph.label_value_to_index("type1")
    assert value_to_index["A"] == 0
    assert value_to_index["C"] == 2

    index_to_value = graph.label_index_to_value_for_type("type2")
    assert index_to_value[1] == "B"

    assert graph.label_info(0) == ("type1", "A")
    assert graph.label_info(1) == ("type2", "B")

    # Error path: unknown label_type and index
    with pytest.raises(KeyError):
        graph.labels_for_type("does-not-exist")
    with pytest.raises(IndexError):
        graph.label_info(10)


def test_add_labels_basic_and_metadata() -> None:
    """
    add_labels should create new label indices and metadata and assign
    vertices correctly.
    """
    num_vertices = 5
    graph = Graph.from_edges({}, num_vertices=num_vertices, scenario_id="add-basic")

    # Add two labels of the same type
    graph.add_labels(
        "kind",
        {
            "A": [0, 2],
            "B": [3],
        },
    )

    assert graph.num_labels == 2
    assert graph.label_matrix is not None
    assert graph.label_matrix.nrows == num_vertices
    assert graph.label_matrix.ncols == 2

    # Metadata mappings
    assert graph.label_info(0) == ("kind", "A")
    assert graph.label_info(1) == ("kind", "B")

    value_to_index = graph.label_value_to_index("kind")
    assert value_to_index["A"] == 0
    assert value_to_index["B"] == 1

    index_to_value = graph.label_index_to_value_for_type("kind")
    assert index_to_value[0] == "A"
    assert index_to_value[1] == "B"

    # Vertex sets
    verts_A = graph.get_vertices_for_label(0)
    verts_B = graph.get_vertices_for_label(1)
    assert set(verts_A.tolist()) == {0, 2}
    assert set(verts_B.tolist()) == {3}

    # Per-type labels_for_type
    idxs_kind, sub_kind = graph.labels_for_type("kind")
    assert set(idxs_kind.tolist()) == {0, 1}
    assert sub_kind.nrows == num_vertices
    assert sub_kind.ncols == 2


def test_add_labels_union_and_multiple_types() -> None:
    """
    Verify that add_labels ORs assignments (union) and supports multiple types.
    """
    num_vertices = 5
    graph = Graph.from_edges({}, num_vertices=num_vertices, scenario_id="add-union")

    # Initial labels for type "kind"
    graph.add_labels(
        "kind",
        {
            "A": [0, 2],
            "B": [3],
        },
    )

    # Add more vertices to existing label "A" and introduce a new type "region"
    graph.add_labels(
        "kind",
        {
            "A": [2, 4],  # adds vertex 4, keeps 0 and 2
        },
    )
    graph.add_labels(
        "region",
        {
            "N": [0, 2],
            "S": [1],
        },
    )

    # num_labels: 2 from "kind" + 2 from "region"
    assert graph.num_labels == 4

    # Union semantics for label "kind:A" (index 0)
    verts_A = graph.get_vertices_for_label(0)
    assert set(verts_A.tolist()) == {0, 2, 4}

    # "kind:B" (index 1) unchanged
    verts_B = graph.get_vertices_for_label(1)
    assert set(verts_B.tolist()) == {3}

    # Region labels should exist and be mapped
    value_to_index_kind = graph.label_value_to_index("kind")
    value_to_index_region = graph.label_value_to_index("region")

    assert value_to_index_kind["A"] == 0
    assert value_to_index_kind["B"] == 1

    idx_N = value_to_index_region["N"]
    idx_S = value_to_index_region["S"]
    assert {idx_N, idx_S} == {2, 3}

    verts_N = graph.get_vertices_for_label(idx_N)
    verts_S = graph.get_vertices_for_label(idx_S)
    assert set(verts_N.tolist()) == {0, 2}
    assert set(verts_S.tolist()) == {1}

    # labels_for_type for "region"
    idxs_region, sub_region = graph.labels_for_type("region")
    assert set(idxs_region.tolist()) == {idx_N, idx_S}
    assert sub_region.nrows == num_vertices
    assert sub_region.ncols == 2


def test_add_labels_invalid_vertex_indices() -> None:
    """
    add_labels should reject vertex indices outside [0, num_vertices).
    """
    num_vertices = 3
    graph = Graph.from_edges({}, num_vertices=num_vertices, scenario_id="add-invalid")

    with pytest.raises(IndexError):
        graph.add_labels("kind", {"A": [-1]})

    with pytest.raises(IndexError):
        graph.add_labels("kind", {"A": [3]})


def test_by_distinct_labels() -> None:
    """
    Verify that by_distinct_labels_matrix and by_distinct_labels return
    the expected combinations given multiple label types.
    """
    num_vertices = 3
    graph = Graph.from_edges({}, num_vertices=num_vertices, scenario_id="distinct")

    # kind: A,B; region: N,S
    # Vertex assignments:
    #   v0: kind=A, region=N
    #   v1: kind=A, region=S
    #   v2: kind=B, region=N
    graph.add_labels(
        "kind",
        {
            "A": [0, 1],
            "B": [2],
        },
    )
    graph.add_labels(
        "region",
        {
            "N": [0, 2],
            "S": [1],
        },
    )

    combos = graph.by_distinct_labels(["kind", "region"])
    # Each row: [kind_label_index, region_label_index]
    combos_set = {tuple(row.tolist()) for row in combos}

    kind_map = graph.label_value_to_index("kind")
    region_map = graph.label_value_to_index("region")
    kA = kind_map["A"]
    kB = kind_map["B"]
    rN = region_map["N"]
    rS = region_map["S"]

    expected = {
        (kA, rN),  # v0
        (kA, rS),  # v1
        (kB, rN),  # v2
    }
    assert combos_set == expected

    # List-returning variant should be consistent
    combos_list = graph.by_distinct_labels(["kind", "region"])
    assert {tuple(row) for row in combos_list} == expected
