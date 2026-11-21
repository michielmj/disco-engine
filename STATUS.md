# Status

This is where the work stands. Specs say *what* (`specs/`); this file says
*where we are*. Keep it short, and move finished detail into git history.

## In flight

| Milestone | Branch | State | Requirement | Notes |
|---|---|---|---|---|
| Adopt the scaffolder template | `scaffold-adoption`, `scaffold-lint-layering` | landed in `staging` (#47, #48), 2026-11-21 | none (owner's direction, 2026-11-21) | `scaffold-lint-layering` fix round 1: no re-review: prescribed and mutation-verified |
| Python 3.14–3.15, dependency upgrades | `py314-deps` | reviewed (round 1: 0 FAIL); round 2 waits on the owner's disco-toolbox release | none (owner's direction, 2026-11-21) | stacked on `staging`; Linux suite 522 passed, 1 skipped (3.15.2) |

## Releases

| Version | Date | Contents | Owner verified |
|---|---|---|---|

## Waiting on the owner

- Start Claude Code in the project, accept the workspace-trust dialog, and
  ask it to run `env` and `git push --force`; both must be blocked by
  `guard_shell.py`.
- With Docker running: `bash scripts/test_linux.sh staging`.
- Check that GitHub Actions runs `test.yml` (tests and the new static job)
  green on `staging`.
- DISCREPANCY (found on `scaffold-lint-layering`): `CLAUDE.md` says CI runs
  the tests on Linux, macOS and Windows. `test.yml` runs Linux only;
  `build.yml` runs the cibuildwheel tests on all three, but only on PRs to
  `main` and on dispatch. Decide: run the platform matrix on every push or
  PR, or change the sentence.
- Release a disco-toolbox that allows python-graphblas 2026.x (0.0.5 pins
  `~=2025.2`), then say so: `py314-deps` round 2 (the example model) starts
  then.
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
