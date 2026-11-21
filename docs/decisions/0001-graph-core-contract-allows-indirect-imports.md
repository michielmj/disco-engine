# 0001: The graph-core contract allows indirect imports, until the restructuring

**Status:** accepted, 2026-11-21. Revisit in the restructuring requirement.

## Context

`.claude/rules/graph.md` says `disco.graph.core` is database-agnostic:
SQLAlchemy belongs in `graph.db` and `graph.extract` only, so the graph can be
built and tested without a database. The import-linter contract written for
it on `scaffold-lint-layering` breaks unless it sets
`allow_indirect_imports = True`, because `graph.core` imports `graph_mask`,
which imports `sqlalchemy` (found by the implementer, confirmed by the
review). `graph_mask` and `schema` already broke the rule before the
contract existed.

With indirect imports allowed, the contract still fails on a direct
`sqlalchemy` import in `graph.core`, but not on one that arrives through
another module.

## Decision

Keep the looser contract now, and fix `graph_mask` later, in the planned
restructuring.

The owner's words: "Keep the looser graph contract; fix graph_mask later"

## Consequences

- `graph.md`'s rule is enforced for direct imports only. A reviewer checks
  new indirect paths from `graph.core` to SQLAlchemy by hand until then.
- The restructuring moves the SQLAlchemy use out of `graph_mask` (and
  `schema`), then removes `allow_indirect_imports` from the contract. That
  removal is the proof it is done.
