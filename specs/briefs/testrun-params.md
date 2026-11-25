# testrun-params: TestRun initializes from the experiment, as the Worker does

There is no requirement for this branch: it fixes carry-overs 10 and 5 in
`docs/carry-overs.md`, which the owner made priority 1 on 2026-11-25. Read
both rows first. The other wave 1 branches are in review: `graph-defects`
(graph, partitioner, `helpers/gb.py`, including `tests/graph/test_extract.py`
and `tests/test_partitioning.py`) and `tooling-fixes` (guard hook,
`CLAUDE.md`).

Before designing anything, read `ENGINEERING.md` Ch 16 (Test Run) and the
Worker's initialization in Ch 5 and `src/disco/worker.py` (around line 964).

The brief's statements about the code are only as reliable as the code:
check each one, and record any that proves wrong as a DISCREPANCY.

## The owner's decision

Asked where a model's parameters come from in a TestRun, with the two
options shown as code (A: `testrun.initialize()`, from the experiment, as in
production; B: `testrun.initialize({...})`, passed in by the test), the
owner answered "A" (2026-11-25). Their earlier words: "follow interface for
initialization consistently accross testrun and worker. Use only params."

## What this branch builds

1. **`TestRun.initialize()` takes no argument** (carry-over 10). It
   initializes every node with the experiment's params, in
   `partitioning.node_specs` order, exactly as the Worker does: the same
   dict, the same fallback when the experiment has none (the Worker uses
   `experiment.params or {}`). Today `src/disco/testrun.py:206` takes
   `params` and ignores it. `run()`, which initializes when needed, follows.
2. **Tests that pair lists fail on a length mismatch** (carry-over 5). The
   `zip(..., strict=False)` calls in the tests named in the carry-over row
   become `strict=True`. The line numbers there predate `graph-defects`:
   find them by content.
3. **Callers and docs follow.** `tests/test_testrun.py` and
   `examples/my-model/tests/test_my_model.py` call `initialize({})` today.
   Ch 16 says `NodeRuntime.initialize(**experiment.params)`; the Worker
   calls `rt.initialize(params)` with one dict. Bring Ch 16 to what the code
   does, and add a line for model authors to `README.md` if they call
   `initialize` themselves.

Delete rows 10 and 5 from `docs/carry-overs.md` in this branch.

## Where this is most likely to go wrong

1. **TestRun and the Worker drift apart again.** The rule is that both give
   a node the same params for the same experiment. Pin it with a test that
   builds an experiment with params, initializes through TestRun, and checks
   what the nodes received: shown failing when TestRun uses anything but the
   experiment's params. If the two paths can share one function without
   crossing the layering in `.importlinter`, prefer that; if not, record why.
2. **An old call fails silently or confusingly.** A model author's
   `testrun.initialize({})` must fail with a clear `TypeError`, not be
   accepted and ignored. This is a breaking change, which the owner allowed
   for this phase (carry-over 1, 2026-11-21); say so in the README line.
3. **The example runs against another disco.** `examples/my-model` depends
   on `disco-toolbox` and `disco-tools`, not on this checkout directly.
   Find out which disco its tests run against before you change its call,
   and record it; if it runs against a released disco-engine, changing the
   call breaks it until the next release, and that is an ESCALATED item,
   not a silent edit.

## Constraints

- No new dependency. The Worker's behaviour does not change.
- Your ports and worktree are in your launch prompt. Stay out of the other
  branches' files beyond the `strict=True` edits of item 2.
- Graphblas cannot be imported on this container's Python 3.15 beta: run
  the suite with `uv run --python 3.14`, and say so in the Evidence.

## Done when

- Items 1 and 2 are pinned by tests that fail when broken, shown failing
  (the prove-a-rule skill).
- The checks in CLAUDE.md § Commands pass, with tests on Linux.
- Somebody has used it: follow the README's TestRun steps as a model author
  would, with params on the experiment, and report what the nodes received;
  then call `initialize({})` and report the error.
- `ENGINEERING.md` Ch 16 is current (the sync-spec skill).
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, each mutation and its result, and what using
  it showed.

## Round 1 (2026-11-25): pin the Worker's side, and the docs that still say `**params`

The review of 3d42e66 found one FAIL. A re-review, scoped to this round,
follows: the remedy is described, not prescribed.

<finding id="1">
src/disco/worker.py:964, tests/: Only TestRun's side of the rule "TestRun
and the Worker give a node the same params" is pinned. Both paths now read
`Experiment.node_params`, but no test covers the Worker's read:
tests/test_worker.py has no match for `params` or `initialize`, and its
tests replace `_setup_run_locked` with a fake (lines 282, 339). Mutation:
worker.py:964 changed to `params = {}`, full suite: 524 passed, 1 skipped,
1 failed (an unrelated, environment-dependent guard-hook test). So the
Worker can drift from TestRun again and the suite stays green.
</finding>

**Remedy (described):** pin the Worker's side of the rule: for an
experiment with params, and for one without, every NodeRuntime the Worker
builds is initialized with `experiment.node_params`. Show it failing with
the reviewer's mutation (`params = {}` at worker.py:964). Drive the
Worker's real setup path as far as its existing fakes allow
(`tests/test_worker.py`, the Metastore fakes named in
`.claude/rules/testing.md`); if the setup cannot run without a real
cluster, find the narrowest seam that still exercises the line that reads
the params, and record what the test does not cover. Do not change
the Worker's threading or WorkerState (`.claude/rules/worker.md`).

**Also in this round:**
- The docs that still describe `initialize(**params)`: `ENGINEERING.md`
  Ch 5 (around line 862), the docstring at `worker.py:851`, and Ch 11
  (around line 3279, `initialize(**kwargs)`). `NodeRuntime.initialize` takes
  one dict (`runtime.py:303`). Bring all three to that, and Ch 5 to the
  shared `node_params` that Ch 16 now states.
- `examples/my-model`: your ESCALATED reasoning does not hold. The review
  found that neither `disco-toolbox` 0.1.0 nor `disco-tools` 0.2.0 depends
  on disco-engine, and the example does not declare it either: it runs
  against whichever disco is installed beside it. The architect decides:
  the example tracks the engine in this repository (it changed alongside it
  in b9d97e7), so change its call to `testrun.initialize()` on this branch.
  Record the example's missing disco-engine dependency as a DISCREPANCY;
  adding it is not part of this round.
- `tests/test_guard_hook.py::test_the_hook_refuses_a_commit_without_initials`
  fails in this container for a reason outside this branch; it is being
  fixed on `tooling-fixes`. Record it in your Evidence, don't fix it here.

Fresh Evidence lines on the fixed head: the round's test and its mutation,
the full suite (with `uv run --python 3.14`), and the checks.
