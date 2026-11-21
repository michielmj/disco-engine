# Carry-overs

This file lists every known problem that is still open, with the branch that
found it. The owner sets the priorities. Add an item when it is found, keep
its number, and delete the row in the branch that fixes it. `STATUS.md`
points here for the backlog.

Each item was checked when it was recorded. Its evidence is in the Surfaced
decisions commits of the branch named in "Found in".

## Defects

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 4 | architect, 2026-11-21 | `guard_shell.py` hangs on `git -C <path>`. `WORD` nests `+` inside `+`, which backtracks catastrophically: the check time doubles per path character (23 characters take 6 s), so the hook times out and blocks the command. Fix: `WORD = r"""(?:[^\s"';&\|]\|"[^"]*"\|'[^']*')+"""`, with a test on a long `-C` path. The same fix belongs upstream in `michielmj/scaffolder`. | `.claude/hooks/guard_shell.py` | unset |
| 8 | py314-deps review | The partitioner fails on a graph whose layer has BOOL weights: `plus_times(incidence @ M)` raises `KeyError`. The same happens on graphblas 2025.2. | `src/disco/partitioning.py:456-457` | unset |
| 9 | py314-deps review | `matrix_has_cycle` reports a cycle for the acyclic chain 0→1→2: `_longest_path_length` runs n-1 iterations and so never sees convergence. | `src/disco/helpers/gb.py:57,74` | unset |
| 10 | py314-deps | `TestRun.initialize(params)` ignores its argument and uses `experiment.params` instead. | `src/disco/testrun.py:217` | unset |

## Code that will break, or that can hide a failure

| # | Found in | Problem | Where | Priority |
|---|---|---|---|---|
| 5 | scaffold-lint-layering review | Tests that zip with `strict=False` can silently truncate. | `tests/test_testrun.py:324`, `tests/graph/test_extract.py:418,423`, `tests/test_partitioning.py:83` | unset |
| 11 | py314-deps review | `fillna` with a numeric fill on a `str` column emits `Pandas4Warning`. The result is the same today, but it will break under pandas 4. | `src/disco/graph/extract.py:148,369,483` | unset |
| 12 | py314-deps | `datetime.utcnow()` is deprecated. The external draft PR #44 fixes it. | `src/disco/graph/graph_mask.py:80,95,122` | unset |

## Documented facts that do not hold

| # | Found in | Problem | Decision needed | Priority |
|---|---|---|---|---|
| 13 | scaffold-lint-layering | `CLAUDE.md` says CI runs the tests on Linux, macOS and Windows. `test.yml` runs them on Linux only. `build.yml` runs the cibuildwheel tests on all three, but only for PRs to `main` and on dispatch. | Run the platform matrix on every push or PR, or change the sentence. | unset |
| 14 | py314-deps review | `.claude/rules/graph.md`: the `monoid.plus` object raises on BOOL, and only the `"plus"` string and `op.plus` give OR. `.new(dtype=)` and `.dup(dtype=)` behave the same, although the rule says they differ. Both hold on graphblas 2025.2 and on 2026.11. | Correct the wording on the next branch that works on the graph code (proposed by the architect, not yet decided). | unset |

## Structure, for the restructuring

| # | Found in | Problem | Priority |
|---|---|---|---|
| 1 | scaffold-adoption | Restructure the package layout. Breaking changes are allowed (owner, 2026-11-21). | unset |
| 2 | scaffold-adoption | Move mypy to `strict = true`; `.claude/rules/python.md` already asks for it. | unset |
| 3 | scaffold-lint-layering | Move SQLAlchemy out of `graph.graph_mask` and `graph.schema`, then drop `allow_indirect_imports` from the graph-core contract (decision 0001). | with item 1 |
| 6 | scaffold-lint-layering review | The forbidden contracts in `.importlinter` list their source modules by hand, so a new module is not covered until someone adds it. | unset |
| 7 | scaffold-lint-layering | `grpc_ingress` imports `disco.worker` under `TYPE_CHECKING`. There are two import cycles, transports→runtime→router→transports and node→graph_data→model→node, and the top-level modules have no total layering. | with item 1 |
