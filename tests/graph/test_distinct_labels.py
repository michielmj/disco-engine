# tests/graph/test_distinct_labels.py
import numpy as np
import graphblas as gb
import pytest

from disco.graph import Graph

ECHELON_LABEL_TYPE = "echelon"
LOCATION_LABEL_TYPE = "location"
NODE_TYPE = "node-type"


def test_by_distinct_labels_and_matrix() -> None:
    label_meta = {
        0: (NODE_TYPE, "StockingPoint"),
        1: (NODE_TYPE, "Factory"),
        2: (ECHELON_LABEL_TYPE, "ech0"),
        3: (ECHELON_LABEL_TYPE, "ech1"),
        4: (ECHELON_LABEL_TYPE, "ech2"),
        5: (LOCATION_LABEL_TYPE, "loc0"),
        6: (LOCATION_LABEL_TYPE, "loc1"),
    }

    # rows = vertices, cols = global label indices
    label_matrix_np = np.array(
        [
            [True, False, True, False, False, True, False],
            [True, False, True, False, False, True, False],
            [True, False, False, True, False, True, False],
            [False, True, False, True, False, True, False],
            [False, True, False, False, True, False, True],
            [True, False, False, False, True, False, True],
            [False, True, False, False, True, False, True],
        ],
        dtype=bool,
    )

    num_vertices = label_matrix_np.shape[0]
    label_matrix = gb.Matrix.from_dense(
        label_matrix_np, nrows=num_vertices, ncols=label_matrix_np.shape[1]
    )

    # ------------------------------------------------------------------
    # Build Graph with labels only (no structural layers)
    # ------------------------------------------------------------------
    g = Graph.from_edges(edge_layers={}, num_vertices=num_vertices, scenario_id="test-scenario")
    g.set_labels(label_matrix, label_meta)

    distinct = [NODE_TYPE, ECHELON_LABEL_TYPE, LOCATION_LABEL_TYPE]

    # ------------------------------------------------------------------
    # Expected distinct combinations of label indices
    # (derived by hand from label_matrix_np)
    # ------------------------------------------------------------------
    expected_combos = {
        (0, 2, 5),  # StockingPoint, ech0, loc0
        (0, 3, 5),  # StockingPoint, ech1, loc0
        (1, 3, 5),  # Factory,       ech1, loc0
        (1, 4, 6),  # Factory,       ech2, loc1
        (0, 4, 6),  # StockingPoint, ech2, loc1
    }

    # ------------------------------------------------------------------
    # Matrix-returning API
    # ------------------------------------------------------------------
    combos_list = g.by_distinct_labels(distinct)
    assert all(c.shape[0] == len(distinct) for c in combos_list)

    combos_matrix_set = {tuple(row) for row in combos_list}
    assert combos_matrix_set == expected_combos


def test_by_distinct_labels_rejects_duplicate_label_per_type() -> None:
    """
    A vertex that carries two labels of the same type (e.g. location=A
    AND location=B) is invalid for by_distinct_labels and must raise.
    """
    label_meta = {
        0: (LOCATION_LABEL_TYPE, "loc0"),
        1: (LOCATION_LABEL_TYPE, "loc1"),
        2: (ECHELON_LABEL_TYPE, "ech0"),
    }

    # Vertex 0 has both loc0 and loc1 — ambiguous.
    label_matrix_np = np.array(
        [
            [True, True, True],
            [True, False, True],
        ],
        dtype=bool,
    )

    num_vertices = label_matrix_np.shape[0]
    label_matrix = gb.Matrix.from_dense(
        label_matrix_np, nrows=num_vertices, ncols=label_matrix_np.shape[1]
    )

    g = Graph.from_edges(edge_layers={}, num_vertices=num_vertices, scenario_id="test-scenario")
    g.set_labels(label_matrix, label_meta)

    with pytest.raises(ValueError, match="more than one label"):
        g.by_distinct_labels([LOCATION_LABEL_TYPE, ECHELON_LABEL_TYPE])


def test_by_distinct_labels_rejects_supergraph_with_merged_locations() -> None:
    """
    Practical scenario: vertices are compressed by a shared resource label,
    but the merged vertices have different locations.  After compression
    the supervertex carries both location labels, making by_distinct_labels
    on the SuperGraph invalid.
    """
    label_meta = {
        0: ("resource", "R1"),
        1: (LOCATION_LABEL_TYPE, "loc0"),
        2: (LOCATION_LABEL_TYPE, "loc1"),
    }

    # Vertex 0: resource=R1, location=loc0
    # Vertex 1: resource=R1, location=loc1
    # Compression on "resource" merges them → supervertex gets both locations.
    label_matrix_np = np.array(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    num_vertices = label_matrix_np.shape[0]
    label_matrix = gb.Matrix.from_dense(
        label_matrix_np, nrows=num_vertices, ncols=label_matrix_np.shape[1]
    )

    g = Graph.from_edges(edge_layers={}, num_vertices=num_vertices, scenario_id="test-scenario")
    g.set_labels(label_matrix, label_meta)

    sg = g.compress(["resource"])

    # The supervertex now has both loc0 and loc1 — by_distinct_labels must reject.
    with pytest.raises(ValueError, match="more than one label"):
        sg.by_distinct_labels([LOCATION_LABEL_TYPE])
