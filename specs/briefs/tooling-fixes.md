# tooling-fixes: the guard hook's hang, and the CI platforms sentence

There is no requirement for this branch: it fixes carry-overs 4 and 13 in
`docs/carry-overs.md`, which the owner made priority 1 on 2026-11-25. Read
both rows first. The other wave 1 branches run in parallel:
`graph-defects` (graph, partitioner, `helpers/gb.py`, `.claude/rules/graph.md`)
and later `testrun-params`.

The brief's statements about the code are only as reliable as the code:
check each one, and record any that proves wrong as a DISCREPANCY.

## What this branch builds

1. **The guard hook no longer hangs** (carry-over 4). `WORD` in
   `.claude/hooks/guard_shell.py:43` nests `+` inside `+`, which backtracks
   catastrophically: a 23-character `git -C <path>` takes 6 s, so the hook
   times out and blocks the command. The prescribed fix is in the
   carry-over row:
   `WORD = r"""(?:[^\s"';&|]|"[^"]*"|'[^']*')+"""`.
   A command with a long `-C` path is checked quickly, and gets the same
   allow or block as before.
2. **`CLAUDE.md` says where CI runs the tests** (carry-over 13). The owner
   decided "Linux only" (2026-11-25): `test.yml` stays as it is. Change the
   sentence in `CLAUDE.md` § Commands ("disco ships to Linux, macOS and
   Windows. The tests must pass on each, and CI runs them on all of
   them; ...") to the facts: `test.yml` runs the tests on Linux on every
   push and PR; `build.yml` runs the cibuildwheel tests on Linux, macOS and
   Windows for PRs to `main` and on dispatch. Check those facts in the
   workflow files before you write them. Keep the pointer to the run-tests
   skill, and check that the skill does not repeat the old claim.

Delete both rows from `docs/carry-overs.md` in this branch.

## Where this is most likely to go wrong

1. **The fix loosens the guard.** The guard is a security control: it
   blocks printing the environment, force-pushes, setting the owner's
   initials, and commits without them. A regex change that stops matching
   is a silent hole. `tests/test_guard_hook.py` exists; make sure every
   block rule is still exercised, and add cases that put the new `WORD` on
   each side of a quote, `;`, `&` and `|`. Show the suite passing on the new
   regex and failing with the old one only on the timing test.
2. **The timing test is flaky.** Bound the long-path check with a generous
   limit that the old regex misses by orders of magnitude (it doubles per
   character), so a slow CI runner does not fail it. Run it on Linux and
   record the times for old and new.

## Constraints

- No new dependency. Change nothing in the guard but `WORD`, unless a test
  proves another pattern has the same defect; record that as a
  DISCREPANCY.
- The same fix belongs upstream in `michielmj/scaffolder`: that is the
  owner's, not this branch's. Do not touch other repositories.
- Your ports and worktree are in your launch prompt; stay out of the other
  branches' areas above.

## Done when

- Item 1 is pinned by a test that fails with the old `WORD` (the
  prove-a-rule skill), shown failing.
- The checks in CLAUDE.md § Commands pass, with tests on Linux.
- Somebody has used it: run the hook as Claude Code does (the JSON on
  stdin that `tests/test_guard_hook.py` uses) on `git -C <a 60-character
  path> status` and on `git push --force`, and report the time and verdict
  of each. Whether it blocks inside a live Claude Code session is "not
  checked, for the owner" (already in `STATUS.md`).
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, each mutation and its result, and what using
  it showed.
