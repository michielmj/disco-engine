# tests/graph/test_distinct_labels.py
import numpy as np
import graphblas as gb

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
    combos_matrix = g.by_distinct_labels(distinct)
    assert combos_matrix.shape[1] == len(distinct)

    combos_matrix_set = {tuple(row) for row in combos_matrix.tolist()}
    assert combos_matrix_set == expected_combos
