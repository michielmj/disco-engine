# src/disco/helpers/gb.py
from __future__ import annotations

import numpy as np
import graphblas as gb
import math


def vstack_rows_from_coo(rows: list[gb.Vector], *, ncols: int | None = None) -> gb.Matrix:
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

    return gb.Matrix.from_coo(R, C, V, nrows=m, ncols=n, dtype=dt)


def has_self_loop(M: gb.Matrix) -> bool:
    if M.nrows != M.ncols:
        raise ValueError("Adjacency matrix must be square")
    diag = M.diag()  # Vector over the diagonal entries
    # Non-zero diagonal entries mean self loops
    return diag.nvals > 0


def matrix_has_cycle(M: gb.Matrix) -> bool:
    """
    Return True iff the directed graph represented by adjacency matrix M
    contains a directed cycle (including self-loops).

    - M is n x n (square).
    - Any non-zero entry is treated as an edge (pattern-only).
    """
    nrows, ncols = M.nrows, M.ncols
    if nrows != ncols:
        raise ValueError("Adjacency matrix must be square")

    n = nrows
    if n == 0:
        return False

    # Work on a boolean pattern of M
    if M.dtype == gb.dtypes.BOOL:
        R = M.dup()
    else:
        # Make everything non-zero into True
        rows, cols, _ = M.to_coo()
        R = gb.Matrix.sparse(gb.dtypes.BOOL, n, n)
        if len(rows):
            R[rows, cols] = True

    # Optional quick check: immediate self-loops
    if R.diag().nvals > 0:
        return True

    # R will hold paths of length in [1, L]; we repeatedly expand to [1, 2L]
    # using boolean semiring.
    steps = math.ceil(math.log2(n)) if n > 1 else 0

    for _ in range(steps):
        # If any vertex can reach itself via a path of length <= current L, we have a cycle
        diag = R.diag()
        if diag.nvals > 0:
            return True

        # Expand to include paths of length up to 2*L:
        # R := R OR (R * R) over boolean semiring
        R_sq = R.mxm(R, semiring=gb.semiring.lor_land)
        R = R.eadd(R_sq, semiring=gb.semiring.lor_lor)

        # Small optimization: if matrix becomes empty, no longer any paths at all
        if R.nvals == 0:
            break

    # Final check (in case n == 1 or loop didn't run)
    return R.diag().nvals > 0
