from __future__ import annotations

import graphblas as gb
import numpy as np
import pytest

from disco.helpers.gb import _longest_path_length, longest_path_length, matrix_has_cycle


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    # _longest_path_length is lru_cache'd on the pickled matrix: no leaks between tests
    _longest_path_length.cache_clear()


def _adj(n: int, edges: list[tuple[int, int]]) -> gb.Matrix:
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    return gb.Matrix.from_coo(rows, cols, [1] * len(edges), nrows=n, ncols=n, dtype=gb.dtypes.INT64)


def test_chain_is_not_a_cycle() -> None:
    assert matrix_has_cycle(_adj(3, [(0, 1), (1, 2)])) is False


def test_cycle_is_a_cycle() -> None:
    assert matrix_has_cycle(_adj(3, [(0, 1), (1, 2), (2, 0)])) is True
    # a cycle behind a source
    assert matrix_has_cycle(_adj(4, [(0, 1), (1, 2), (2, 1), (2, 3)])) is True


def test_self_loop_is_a_cycle() -> None:
    assert matrix_has_cycle(_adj(2, [(0, 1), (1, 1)])) is True
    assert matrix_has_cycle(_adj(1, [(0, 0)])) is True


def test_longest_path_lengths_of_a_dag_with_branches_of_different_length() -> None:
    # 0->1->2->4 (length 3) and 0->3->4 (length 2); 5 is isolated
    v = longest_path_length(_adj(6, [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)]))
    idx, vals = v.to_coo()
    assert dict(zip(idx.tolist(), vals.tolist(), strict=True)) == {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 0}
    assert np.array_equal(idx, np.arange(6))


def test_a_cycle_with_no_source_is_a_cycle() -> None:
    assert matrix_has_cycle(_adj(3, [(0, 1), (1, 2), (2, 0)])) is True
    assert matrix_has_cycle(_adj(2, [(0, 1), (1, 0)])) is True


def test_an_empty_graph_is_acyclic() -> None:
    assert matrix_has_cycle(gb.Matrix(gb.dtypes.INT64, nrows=0, ncols=0)) is False
    assert matrix_has_cycle(gb.Matrix(gb.dtypes.INT64, nrows=3, ncols=3)) is False


@pytest.mark.parametrize("dtype", [gb.dtypes.INT64, gb.dtypes.FP64, gb.dtypes.BOOL])
def test_a_stored_zero_is_not_an_edge(dtype: gb.dtypes.DataType) -> None:
    # 0 -> 1 -> 2 -> 0, but the closing edge 2 -> 0 is a stored zero: no cycle.
    nonzero = True if dtype == gb.dtypes.BOOL else 1
    zero = False if dtype == gb.dtypes.BOOL else 0
    m = gb.Matrix.from_coo([0, 1, 2], [1, 2, 0], [nonzero, nonzero, zero], nrows=3, ncols=3, dtype=dtype)
    assert m.nvals == 3  # the zero is stored
    assert matrix_has_cycle(m) is False
    # a stored zero on the diagonal is not a self-loop either
    z = gb.Matrix.from_coo([0], [0], [zero], nrows=2, ncols=2, dtype=dtype)
    assert matrix_has_cycle(z) is False
