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
