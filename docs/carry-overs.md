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

## Code that will break, or that can hide a failure

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 16 | graph-defects | `graph_scenarios.created_at` is written with the deprecated `datetime.utcnow()` (item 12 fixed `graph_mask` only). | `src/disco/graph/db.py:252` | unset |
| 19 | tooling-fixes review | Two guard patterns still backtrack exponentially on repeated options: the initials rule on many `-C`/`-c` git options (22 take 2.2 s), and `INTERPRETER` on many `uv run` options (18 take 24 s). Neither has a realistic trigger. The upstream template has the same patterns. | `.claude/hooks/guard_shell.py` | unset |
| 27 | graphdata-default-fill review | `SimProcGraphData.inbound_edge_data` annotates `columns` as `Sequence[ColumnElement]`, while `outbound_edge_data` also accepts `Sequence[str]`, and both work with `str` at runtime. `tests/test_graph_data.py` calls it with `str` under a `# type: ignore[operator]` that hides the mismatch, without the comment `.claude/rules/python.md` asks for. | `src/disco/graph_data.py:74`, `tests/test_graph_data.py` | unset |

## Documented facts that do not hold

| # | Found in | Problem | Decision needed | Priority |
|---|---|---|---|---|
| 20 | testrun-params review | `ENGINEERING.md` Ch 8 (around line 2662) says a node has a no-argument `__init__` and that a "NodeController" calls `node.initialize(experiment=..., replication=..., resources=..., params=...)`. Ch 11 and `runtime.py:304` say the node's `__init__` receives `params`. | Correct Ch 8 to the code. | unset |
| 21 | testrun-params review | `examples/my-model` declares no disco-engine dependency, although it is a disco model: it runs against whichever disco is installed beside it. Its tests could not run in the cloud container (`data_logger` not importable). | Declare the dependency, or document how the example is installed. | unset |
| 22 | wave 1 | `.python-version` is 3.15, but in the cloud container (Python 3.15 beta) graphblas has no build, so `uv run pytest` fails at collection; every wave 1 branch ran with `uv run --python 3.14`. CI's 3.15 job passed on 2026-11-21. | Check whether this holds on a released 3.15; if so, pin `.python-version` to 3.14 or say so in the run-tests skill. | unset |

## Structure, for the restructuring

| # | Found in | Problem | Priority |
|---|---|---|---|
| 1 | scaffold-adoption | Restructure the package layout. Breaking changes are allowed (owner, 2026-11-21). | 2 (restructuring) |
| 2 | scaffold-adoption | Move mypy to `strict = true`; `.claude/rules/python.md` already asks for it. | 3 (after the restructuring) |
| 3 | scaffold-lint-layering | Move SQLAlchemy out of `graph.graph_mask` and `graph.schema`, then drop `allow_indirect_imports` from the graph-core contract (decision 0001). | with item 1 |
| 6 | scaffold-lint-layering review | The forbidden contracts in `.importlinter` list their source modules by hand, so a new module is not covered until someone adds it. | 2 (restructuring) |
| 7 | scaffold-lint-layering | `grpc_ingress` imports `disco.worker` under `TYPE_CHECKING`. There are two import cycles, transports→runtime→router→transports and node→graph_data→model→node, and the top-level modules have no total layering. | with item 1 |

## Data paths that do a dependency's work (decision 0002)

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 23 | release re-review, 2026-11-26 | The partitioner's topology loops in Python over every pair of `R.to_coo()` (14,253,408 pairs for 1M vertices, 5 layers of 5M edges, 2,000 nodes), and dominates topology time. Older than release #56. | `src/disco/partitioning.py:461-466` | unset |
| 24 | release re-review, 2026-11-26 | `_longest_path_length` is `lru_cache`d on `pickle.dumps(matrix)`: 4.3 ms per call on 500k edges, and up to 6 serialized matrices kept in memory. Older than release #56; after `dependency-fixes` the cycle check no longer calls it. | `src/disco/helpers/gb.py` | unset |
| 25 | dependency-fixes review | The topology's `any_pair` counts a stored `False` in the incidence matrix as a link, where `plus_times` + `select` did not. No producer stores `False` today (`_iter_node_instances` selects `!= False`; the load path writes only `True`), and `Partitioning` checks the incidence dtype but not its values. | `src/disco/partitioning.py:460` | unset |
