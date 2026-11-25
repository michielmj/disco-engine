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
