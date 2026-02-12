# src/disco/helpers/gb.py
from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np
from graphblas import Vector, Matrix, dtypes, semiring, monoid, unary, op
import math


def vstack_rows_from_coo(rows: list[Vector], *, ncols: int | None = None) -> Matrix:
    m = len(rows)
    if m == 0:
        raise ValueError("no rows")
    n = ncols if ncols is not None else rows[0].size
    dt = rows[0].dtype

    r_ix = []
    c_ix = []
    vals = []

    for r, v in enumerate(rows):
        idx, vvals = v.to_coo()
        r_ix.append(np.full(idx.size, r, dtype=np.uint64))
        c_ix.append(idx)
        vals.append(vvals)

    R = np.concatenate(r_ix) if r_ix else np.array([], dtype=np.uint64)
    C = np.concatenate(c_ix) if c_ix else np.array([], dtype=np.uint64)
    V = np.concatenate(vals) if vals else np.array([], dtype=dt.np_type)

    return Matrix.from_coo(R, C, V, nrows=m, ncols=n, dtype=dt)


def has_self_loop(M: Matrix) -> bool:
    if M.nrows != M.ncols:
        raise ValueError("Adjacency matrix must be square")
    diag = M.diag()  # Vector over the diagonal entries
    # Non-zero diagonal entries mean self loops
    return diag.nvals > 0


def matrix_has_cycle(M: Matrix) -> bool:
    """
    Return True iff the directed graph represented by adjacency matrix M
    contains a directed cycle (including self-loops).

    - M is n x n (square).
    - Any non-zero entry is treated as an edge (pattern-only).
    """
    try:
        longest_path_length(M)
    except ValueError:
        return True
    else:
        return False


@lru_cache(6)
def _longest_path_length(matrix_serialized: bytes) -> Vector:
    matrix: Matrix = pickle.loads(matrix_serialized)
    n = matrix.nrows
    if matrix.ncols != n:
        raise ValueError("matrix must be square")

    # Keep only real edges; important if matrix was built from_dense (explicit zeros)
    A = matrix.select("!=", 0).apply(unary.one).new(dtype=dtypes.INT64)

    # sources = nodes with indegree 0
    indeg = A.reduce_columnwise(monoid.plus).new(dtype=dtypes.INT64)
    v = Vector(dtypes.INT64, size=n)
    v(mask=~indeg.S) << 0  # only sources are reachable at length 0; others are "missing" (-inf)

    # Relax up to n-1 times (DAG longest path has at most n-1 edges)
    for _ in range(n - 1):
        w = v.dup()
        v(op.max) << semiring.max_plus(v @ A)
        if v.isequal(w):
            return v

    raise ValueError("No convergence in n-1 iterations; graph likely has a directed cycle.")


def longest_path_length(matrix: Matrix) -> Vector:
    """
    Longest path length (number of edges) ending at each vertex in a DAG,
    where paths may start at any source (indegree == 0).

    matrix[i, j] != 0 means edge i -> j. Works for dense inputs (explicit zeros)
    because we drop zeros first.
    """

    return _longest_path_length(pickle.dumps(matrix))


@lru_cache(6)
def _echelon_plus_times(ser: bytes) -> Vector:
    v, M = pickle.loads(ser)

    if M.nrows != M.ncols:
        raise ValueError("M must be square")
    if v.size != M.nrows:
        raise ValueError("v must have size M.nrows")

    A = M.select("!=", 0).new()

    e = v.dup()
    w = v.dup()

    # In a DAG, series ends after at most n-1 steps
    for _ in range(A.nrows - 1):
        w = semiring.plus_times(w @ A).new()   # w = wA  (next-hop accumulated demand)
        if w.nvals == 0:
            break
        e(op.plus) << w                         # e += w
    else:
        # If we didn't terminate in n-1 steps, there's likely a cycle (or zeros stored as edges)
        raise ValueError("No termination within n-1 iterations (cycle or stored zeros?)")

    return e


def echelon_plus_times(v: Vector, M: Matrix) -> Vector:
    """
    e = v + vM + vM^2 + ... (finite for a DAG)
    Assumes plus_times over numeric types.
    """

    return _echelon_plus_times(pickle.dumps((v, M)))
