import numpy as np
import graphblas as gb
from graphblas import Vector

from disco.graph import Graph

ECHELON_LABEL_TYPE = "echelon"
LOCATION_LABEL_TYPE = "location"
NODE_TYPE = "node-type"


def test_by_distinct_labels_and_matrix() -> None:
    # ------------------------------------------------------------------
    # Label type vectors (over global label indices)
    # ------------------------------------------------------------------
    label_type_vectors = {
        ECHELON_LABEL_TYPE: Vector.from_dense(
            [False, False, True, True, True, False, False]
        ),
        LOCATION_LABEL_TYPE: Vector.from_dense(
            [False, False, False, False, False, True, True]
        ),
        NODE_TYPE: Vector.from_dense(
            [True, True, False, False, False, False, False]
        ),
    }

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
    # Build Graph with labels only
    # ------------------------------------------------------------------
    g = Graph(layers={}, num_vertices=num_vertices, scenario_id="test-scenario")
    g.set_labels(label_matrix, label_meta, label_type_vectors)

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

    combos = g.by_distinct_labels(distinct)
    assert isinstance(combos, np.ndarray)
    combos_set = {tuple(combo) for combo in combos.tolist()}

    assert combos_set == expected_combos


