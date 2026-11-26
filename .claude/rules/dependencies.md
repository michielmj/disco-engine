---
paths:
  - "src/**/*.py"
  - "src/**/*.pyx"
  - "src/**/*.cpp"
  - "src/**/*.hpp"
---

# Dependencies do their own work (decision 0002)

disco is about stability and performance, and its dependencies (pandas,
numpy, python-graphblas, SQLAlchemy, grpc) are chosen because they are
optimised for both. Code that does their work is slower and breaks on their
upgrades.

- Call the dependency; never re-implement, re-derive or work around its
  behaviour. No hand-written table of its type rules, no copy of its
  coercion, parsing or validation.
- No Python loop over rows, columns, vertices or edges where the library
  has a vectorised operation (`fillna` with a mapping, `astype`, `where`,
  graphblas `apply`/`select`/`mxm`, a SQL `JOIN`).
- A deprecation warning or a behaviour change in a dependency is fixed by
  its documented migration. Read its release notes, its deprecation docs and
  its issue tracker first, and cite the source in your Surfaced decisions.
- If the documented migration changes what a model author or operator sees,
  or does not fit disco's contract, stop and escalate: that is a question
  about disco's interface for the owner, not a shim to write.
- On a data path (graph extraction, partitioning, routing, transports, the
  event loop), measure before and after, and give both numbers and their
  configuration in your Evidence. A slowdown is escalated, not explained.
