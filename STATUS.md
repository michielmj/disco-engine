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
- Review and merge the release PR from `staging` to `main`. `build.yml`
  runs the cibuildwheel tests for cp314/cp315 on Linux, macOS and Windows on
  that PR: check that they pass. Also run the example model's tests on 3.14.
  (`test.yml` passed on `staging` at cd61f39: test (3.14), test (3.15) and
  the static checks.)
- Decide carry-overs 13 and 14 in `docs/carry-overs.md` (the CI platforms
  sentence in `CLAUDE.md`, and the wording of `.claude/rules/graph.md`), and
  prioritise the rest.
- Upstream bug in the scaffolder template: carry-over 4; the same fix
  belongs in `michielmj/scaffolder`.

## Backlog

Every open problem is in [`docs/carry-overs.md`](docs/carry-overs.md),
numbered, with its priority. Add new items there.
