# Status

This is where the work stands. Specs say *what* (`specs/`); this file says
*where we are*. Keep it short, and move finished detail into git history.

## In flight

| Milestone | Branch | State | Requirement | Notes |
|---|---|---|---|---|
| Adopt the scaffolder template | `scaffold-adoption`, `scaffold-lint-layering` | landed in `staging` (#47, #48), 2026-11-21 | none (owner's direction, 2026-11-21) | `scaffold-lint-layering` fix round 1: no re-review: prescribed and mutation-verified |
| Python 3.14–3.15, dependency upgrades | `py314-deps` | landed in `staging` (#49), 2026-11-21 | none (owner's direction, 2026-11-21) | round 1: scoped re-review, 0 FAIL; round 2 (example only): no re-review: prescribed, diff checked by the architect |

## Releases

| Version | Date | Contents | Owner verified |
|---|---|---|---|

## Waiting on the owner

- Start Claude Code in the project, accept the workspace-trust dialog, and
  ask it to run `env` and `git push --force`; both must be blocked by
  `guard_shell.py`.
- Check that GitHub Actions runs `test.yml` green on `staging`: the 3.14 and
  3.15 matrix and the static job. (The Linux container suite passed on
  2fc8169, whose code is identical to `staging` at 0281a60: 522 passed,
  1 skipped, Python 3.15.2.)
- Run the wheel builds (`build.yml`, on dispatch) for cp314/cp315 on Linux,
  macOS and Windows, and the example model's tests on 3.14.
- DISCREPANCY (found on `scaffold-lint-layering`): `CLAUDE.md` says CI runs
  the tests on Linux, macOS and Windows. `test.yml` runs Linux only;
  `build.yml` runs the cibuildwheel tests on all three, but only on PRs to
  `main` and on dispatch. Decide: run the platform matrix on every push or
  PR, or change the sentence.
- Decide: correct `.claude/rules/graph.md`'s BOOL wording (the `monoid.plus`
  object raises on BOOL, and only `"plus"`/`op.plus` OR; `.new(dtype=)` and
  `.dup(dtype=)` behave the same) on the next graph branch. The same holds on
  graphblas 2025.2 and 2026.11 (py314-deps review).
- Upstream bug in the scaffolder template: see backlog item 4; the same fix
  belongs in `michielmj/scaffolder`.

## Backlog (carry-overs), prioritised by the owner

| # | Found in | Item | Priority |
|---|---|---|---|
| 1 | scaffold-adoption | Restructure the package layout, breaking changes allowed (owner, 2026-11-21) | unset |
| 2 | scaffold-adoption | Move mypy to `strict = true`; `.claude/rules/python.md` already asks for it | unset |
| 3 | scaffold-lint-layering | Restructuring: move SQLAlchemy out of `graph.graph_mask` and `graph.schema`, then drop `allow_indirect_imports` from the graph-core contract (decision 0001) | with item 1 |
| 4 | architect, 2026-11-21 | `guard_shell.py` hangs on `git -C <path>`: `WORD` nests `+` inside `+` (catastrophic backtracking), so the check time doubles per path character (23 chars: 6 s) and the hook times out, blocking the command. Fix: `WORD = r"""(?:[^\s"';&\|]\|"[^"]*"\|'[^']*')+"""`, with a test on a long `-C` path | unset |
| 5 | scaffold-lint-layering review | Tests that zip with `strict=False` can silently truncate: `test_testrun.py:324`, `graph/test_extract.py:418,423`, `test_partitioning.py:83` | unset |
| 6 | scaffold-lint-layering review | `.importlinter` forbidden contracts list their source modules by hand; a new module is not covered until added | unset |
| 7 | scaffold-lint-layering | Restructuring: `grpc_ingress` imports `disco.worker` under `TYPE_CHECKING`; cycles transports→runtime→router→transports and node→graph_data→model→node; no total layering of the top-level modules | with item 1 |
| 8 | py314-deps review | The partitioner fails on a graph whose layer has BOOL weights: `partitioning.py:456-457` `plus_times(incidence @ M)` raises `KeyError` (the same on graphblas 2025.2) | unset |
| 9 | py314-deps review | `matrix_has_cycle` reports a cycle for the acyclic chain 0→1→2: `_longest_path_length` runs n-1 iterations and never sees convergence | unset |
| 10 | py314-deps | `TestRun.initialize(params)` (`testrun.py`) ignores its argument and uses `experiment.params` | unset |
| 11 | py314-deps review | `graph/extract.py:148,369,483`: `fillna` with a numeric fill on a `str` column emits `Pandas4Warning`; the result is unchanged today, and it will break under pandas 4 | unset |
| 12 | py314-deps | `graph_mask.py:95` uses the deprecated `datetime.utcnow()`; the external draft PR #44 fixes it | unset |
