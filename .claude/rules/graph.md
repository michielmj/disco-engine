---
paths:
  - "src/disco/graph/**"
  - "src/disco/partitioner/**"
  - "src/disco/graph_data.py"
  - "src/disco/graph_factory.py"
  - "tests/graph/**"
  - "tests/partitioner/**"
  - "tests/test_graph_factory.py"
  - "tests/test_partitioning.py"
---

# python-graphblas (Chapter 7, Chapter 18)

- Under BOOL, `plus` is logical OR, not addition, and `times` does not exist:
  the `monoid.plus` object and `semiring.plus_times` raise `KeyError`, while
  the `"plus"` string and `op.plus` give OR. To count or sum, cast to INT64
  first with `matrix.dup(dtype=gb.dtypes.INT64)`. `.dup(dtype=...)` and
  `.apply(identity).new(dtype=...)` cast before the operation and behave the
  same.
- An empty sparse slice, such as a label with no vertices, makes
  `np.min`/`np.max` raise on a zero-size array. Guard every reduction with an
  empty check.
- `Graph` (`graph.core`) is database-agnostic: SQLAlchemy belongs in
  `graph.db` and `graph.extract` only, so the graph can be built and tested
  without a database.
