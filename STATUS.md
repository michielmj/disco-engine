# Status

This is where the work stands. Specs say *what* (`specs/`); this file says
*where we are*. Keep it short, and move finished detail into git history.

## In flight

| Milestone | Branch | State | Requirement | Notes |
|---|---|---|---|---|
| Adopt the scaffolder template | `scaffold-adoption`, `scaffold-lint-layering` | landed in `staging` (#47, #48), 2026-11-21 | none (owner's direction, 2026-11-21) | `scaffold-lint-layering` fix round 1: no re-review: prescribed and mutation-verified |
| Python 3.14–3.15, dependency upgrades | `py314-deps` | landed in `staging` (#49), 2026-11-21 | none (owner's direction, 2026-11-21) | round 1: scoped re-review, 0 FAIL; round 2 (example only): no re-review: prescribed, diff checked by the architect |
| Wave 1: defects and small fixes (carry-overs 4, 5, 8–14) | `graph-defects` (8, 9, 11, 12, 14), `tooling-fixes` (4, 13), `testrun-params` (10, 5) | `graph-defects` and `tooling-fixes` briefed and building; `testrun-params` waits on the owner's reading of D4 | none (owner's priorities, 2026-11-25) | 13 is one sentence, so it rides on `tooling-fixes` |
| Restructuring: remove technical debt (carry-overs 1, 3, 6, 7) | none | requirement to be drafted after wave 1 | to be written | owner, 2026-11-25: "The goal is removal of technical debt as a first step towards a first stable release." mypy strict (2) follows it |

## Releases

| Version | Date | Contents | Owner verified |
|---|---|---|---|
| (untagged) | 2026-11-21 | #50 merged `staging` into `main`: scaffolder adoption, Python 3.14–3.15. The last tag is v0.0.7 (2026-05-27). | no: macOS wheel job cancelled, see below |

Next: v0.0.8 after wave 1, once the macOS wheel build passes on `main`
(owner, 2026-11-25, D1: "yes").

## Standing permissions

- Owner, 2026-11-25, for cloud sessions: "Branch permission granted. You are
  free to create and remove your own branches and merge them into staging.
  You may create PRs against main. You may never merge against main
  directly."

## Waiting on the owner

- Remove the three `GIT_CONFIG_*` variables from the Default cloud
  environment (the session runtime overrides them) and add
  `git config --global user.initials jsn` as its setup script, so new
  containers have your initials.

- Delete the merged remote branches `45-unmasked-label-matrix`,
  `py314-deps`, `scaffold-adoption` and `scaffold-lint-layering` (owner,
  2026-11-25: "delete the branches"). All four are in `main`; the cloud
  session's GitHub proxy refuses ref deletion, so do it on GitHub.
- On release PR #50 the Build run's macOS wheel job was cancelled one
  second after it started (run 44929215332); Linux and Windows passed. Run
  `build.yml` by dispatch on `main` and check that macOS passes before
  tagging v0.0.8. Also run the example model's tests on 3.14.
- Start Claude Code in the project, accept the workspace-trust dialog, and
  ask it to run `env` and `git push --force`; both must be blocked by
  `guard_shell.py`.
- Upstream bug in the scaffolder template: carry-over 4; the same fix
  belongs in `michielmj/scaffolder`.

## Backlog

Every open problem is in [`docs/carry-overs.md`](docs/carry-overs.md),
numbered, with its priority. Add new items there.
