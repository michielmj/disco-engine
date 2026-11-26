# Status

This is where the work stands. Specs say *what* (`specs/`); this file says
*where we are*. Keep it short, and move finished detail into git history.

## In flight

| Milestone | Branch | State | Requirement | Notes |
|---|---|---|---|---|
| Adopt the scaffolder template | `scaffold-adoption`, `scaffold-lint-layering` | landed in `staging` (#47, #48), 2026-11-21 | none (owner's direction, 2026-11-21) | `scaffold-lint-layering` fix round 1: no re-review: prescribed and mutation-verified |
| Python 3.14–3.15, dependency upgrades | `py314-deps` | landed in `staging` (#49), 2026-11-21 | none (owner's direction, 2026-11-21) | round 1: scoped re-review, 0 FAIL; round 2 (example only): no re-review: prescribed, diff checked by the architect |
| Wave 1: defects and small fixes (carry-overs 4, 5, 8–14) | `graph-defects` (8, 9, 11, 12, 14), `tooling-fixes` (4, 13), `testrun-params` (10, 5) | landed in `staging`, 2026-11-25: `tooling-fixes` (#52), `testrun-params` (#53), `graph-defects` (#54) | none (owner's priorities, 2026-11-25; D4 answered "A") | `tooling-fixes`: 0 FAIL, round 1 prescribed, no re-review: diff checked by the architect. `testrun-params`: 1 FAIL, round 1, scoped re-review 0 FAIL. `graph-defects`: 3 FAIL, round 1, scoped re-review 2 FAIL, round 2, scoped re-review 0 FAIL. Merged tree of all three: 555 passed, 1 skipped (native Linux, Python 3.14). The reviews found carry-overs 15–22. |
| Dependencies do their own work (decision 0002) | `dependency-fixes` | process change landed (#57), 2026-11-26; branch briefed and building | none (owner's direction, 2026-11-26) | fixes the release re-review's 2 FAIL; closes carry-overs 15, 17, 18 |
| Restructuring: remove technical debt (carry-overs 1, 3, 6, 7) | none | requirement to be drafted next | to be written | owner, 2026-11-25: "The goal is removal of technical debt as a first step towards a first stable release." mypy strict (2) follows it |

## Releases

| Version | Date | Contents | Owner verified |
|---|---|---|---|
| (untagged) | 2026-11-21 | #50 merged `staging` into `main`: scaffolder adoption, Python 3.14–3.15. The last tag is v0.0.7 (2026-05-27). | no: macOS wheel job cancelled, see below |

Next: v0.0.8, wave 1, release PR #56 (`staging` to `main`). **On hold**
(2026-11-26): its full re-review under decision 0002 found 2 FAIL (e):
`_fill_missing` re-implements pandas' fill, and the cycle check
re-implements what scipy does (107 s against 0.9 ms on a 100k-vertex chain).
`dependency-fixes` fixes both; #56 picks it up when it lands. The owner
answered the `default_fill` question "A": pass it straight to pandas, a
per-column mapping allowed. Its `build.yml` run is the cross-platform check,
macOS included (owner, 2026-11-25, D1: "yes"). The owner merges and tags.

## Standing permissions

- Owner, 2026-11-25, for cloud sessions: "Branch permission granted. You are
  free to create and remove your own branches and merge them into staging.
  You may create PRs against main. You may never merge against main
  directly."

## Waiting on the owner

- Your initials reach cloud sessions through the Default environment's
  `GIT_CONFIG_*` variables: they took effect after the container restart on
  2026-11-25 (git reports them with origin "command line"). Keep them; a
  setup script is not needed. Before the restart the session runtime's own
  `GIT_CONFIG_*` values were in place instead, so if a new session refuses
  commits for want of initials, that is the cause.
- Delete the merged remote branches `45-unmasked-label-matrix`,
  `py314-deps`, `scaffold-adoption` and `scaffold-lint-layering` (owner,
  2026-11-25: "delete the branches"). All four are in `main`; the cloud
  session's GitHub proxy refuses ref deletion, so do it on GitHub.
- On release PR #50 the Build run's macOS wheel job was cancelled one
  second after it started (run 44929215332); Linux and Windows passed. On
  the v0.0.8 release PR, check that all three wheel jobs pass, macOS
  included, before you merge and tag.
- Run the example model's tests on 3.14: `examples/my-model` now calls
  `testrun.initialize()` (no argument), which needs this release's
  disco-engine installed beside it (carry-over 21). Its tests could not run
  in the cloud container.
- Mask cleanup on PostgreSQL: `graph_vertex_masks.updated_at` must keep
  naive UTC there too (tested on SQLite only).
- Prioritise carry-overs 15–22, found by the wave 1 reviews.
- Start Claude Code in the project, accept the workspace-trust dialog, and
  ask it to run `env` and `git push --force`; both must be blocked by
  `guard_shell.py`.
- Upstream bug in the scaffolder template: the `WORD` fix of carry-over 4
  (landed in #52) belongs in `michielmj/scaffolder` too, and so does the
  guard test's helper that clears git config passed through the environment.

## Backlog

Every open problem is in [`docs/carry-overs.md`](docs/carry-overs.md),
numbered, with its priority. Add new items there.
