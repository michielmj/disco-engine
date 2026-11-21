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

- The BOOL `plus` monoid is logical OR, not addition. To count, cast to INT64
  first with `matrix.dup(dtype=gb.dtypes.INT64)`; `.new(dtype=...)` and
  `.apply(identity, dtype=...)` do not do the same.
- An empty sparse slice, such as a label with no vertices, makes
  `np.min`/`np.max` raise on a zero-size array. Guard every reduction with an
  empty check.
- `Graph` (`graph.core`) is database-agnostic: SQLAlchemy belongs in
  `graph.db` and `graph.extract` only, so the graph can be built and tested
  without a database.
