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
| 4 | architect, 2026-11-21 | `guard_shell.py` hangs on `git -C <path>`. `WORD` nests `+` inside `+`, which backtracks catastrophically: the check time doubles per path character (23 characters take 6 s), so the hook times out and blocks the command. Fix: `WORD = r"""(?:[^\s"';&\|]\|"[^"]*"\|'[^']*')+"""`, with a test on a long `-C` path. The same fix belongs upstream in `michielmj/scaffolder`. | `.claude/hooks/guard_shell.py` | 1 (wave 1) |
| 10 | py314-deps | `TestRun.initialize(params)` ignores its argument and uses `experiment.params` instead. Decided (owner, 2026-11-25): "follow interface for initialization consistently accross testrun and worker. Use only params." | `src/disco/testrun.py:217` | 1 (wave 1) |

## Code that will break, or that can hide a failure

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 5 | scaffold-lint-layering review | Tests that zip with `strict=False` can silently truncate. | `tests/test_testrun.py:324`, `tests/graph/test_extract.py:418,423`, `tests/test_partitioning.py:83` | 1 (wave 1) |

## Documented facts that do not hold

| # | Found in | Problem | Decision needed | Priority |
|---|---|---|---|---|
| 13 | scaffold-lint-layering | `CLAUDE.md` says CI runs the tests on Linux, macOS and Windows. `test.yml` runs them on Linux only. `build.yml` runs the cibuildwheel tests on all three, but only for PRs to `main` and on dispatch. | Decided (owner, 2026-11-25): "Linux only". Change the sentence; `test.yml` stays as it is. | 1 (wave 1) |

## Structure, for the restructuring

| # | Found in | Problem | Priority |
|---|---|---|---|
| 1 | scaffold-adoption | Restructure the package layout. Breaking changes are allowed (owner, 2026-11-21). | 2 (restructuring) |
| 2 | scaffold-adoption | Move mypy to `strict = true`; `.claude/rules/python.md` already asks for it. | 3 (after the restructuring) |
| 3 | scaffold-lint-layering | Move SQLAlchemy out of `graph.graph_mask` and `graph.schema`, then drop `allow_indirect_imports` from the graph-core contract (decision 0001). | with item 1 |
| 6 | scaffold-lint-layering review | The forbidden contracts in `.importlinter` list their source modules by hand, so a new module is not covered until someone adds it. | 2 (restructuring) |
| 7 | scaffold-lint-layering | `grpc_ingress` imports `disco.worker` under `TYPE_CHECKING`. There are two import cycles, transports→runtime→router→transports and node→graph_data→model→node, and the top-level modules have no total layering. | with item 1 |
