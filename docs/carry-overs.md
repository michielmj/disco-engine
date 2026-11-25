# Carry-overs

This file lists every known problem that is still open, with the branch that
found it. The owner sets the priorities. Add an item when it is found, keep
its number, and delete the row in the branch that fixes it. `STATUS.md`
points here for the backlog.

Each item was checked when it was recorded. Its evidence is in the Surfaced
decisions commits of the branch named in "Found in".

Priorities set by the owner, 2026-11-25. The architect proposed wave 1 (the
defects and small fixes), then the restructuring, then mypy strict; the
owner's words: "your recommendations". The restructuring's goal, in the
owner's words: "The goal is removal of technical debt as a first step towards
a first stable release."

## Defects

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 8 | py314-deps review | The partitioner fails on a graph whose layer has BOOL weights: `plus_times(incidence @ M)` raises `KeyError`. The same happens on graphblas 2025.2. | `src/disco/partitioning.py:456-457` | 1 (wave 1) |
| 9 | py314-deps review | `matrix_has_cycle` reports a cycle for the acyclic chain 0→1→2: `_longest_path_length` runs n-1 iterations and so never sees convergence. | `src/disco/helpers/gb.py:57,74` | 1 (wave 1) |

## Code that will break, or that can hide a failure

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 11 | py314-deps review | `fillna` with a numeric fill on a `str` column emits `Pandas4Warning`. The result is the same today, but it will break under pandas 4. | `src/disco/graph/extract.py:148,369,483` | 1 (wave 1) |
| 12 | py314-deps | `datetime.utcnow()` is deprecated. PR #44 proposed a fix but was closed without merging, so the fix is ours. | `src/disco/graph/graph_mask.py:80,95,122` | 1 (wave 1) |

## Documented facts that do not hold

| # | Found in | Problem | Decision needed | Priority |
|---|---|---|---|---|
| 14 | py314-deps review | `.claude/rules/graph.md`: the `monoid.plus` object raises on BOOL, and only the `"plus"` string and `op.plus` give OR. `.new(dtype=)` and `.dup(dtype=)` behave the same, although the rule says they differ. Both hold on graphblas 2025.2 and on 2026.11. | Decided (owner, 2026-11-25): "ok" to correcting the wording on the next branch that works on the graph code. | 1 (wave 1) |

## Structure, for the restructuring

| # | Found in | Problem | Priority |
|---|---|---|---|
| 1 | scaffold-adoption | Restructure the package layout. Breaking changes are allowed (owner, 2026-11-21). | 2 (restructuring) |
| 2 | scaffold-adoption | Move mypy to `strict = true`; `.claude/rules/python.md` already asks for it. | 3 (after the restructuring) |
| 3 | scaffold-lint-layering | Move SQLAlchemy out of `graph.graph_mask` and `graph.schema`, then drop `allow_indirect_imports` from the graph-core contract (decision 0001). | with item 1 |
| 6 | scaffold-lint-layering review | The forbidden contracts in `.importlinter` list their source modules by hand, so a new module is not covered until someone adds it. | 2 (restructuring) |
| 7 | scaffold-lint-layering | `grpc_ingress` imports `disco.worker` under `TYPE_CHECKING`. There are two import cycles, transports→runtime→router→transports and node→graph_data→model→node, and the top-level modules have no total layering. | with item 1 |
