# <branch>: <title>

Read `specs/requirements/NN-<slug>.md` first, in full. The product owner
approved it on <date>. This branch builds rules x–y. <The other branches of
this requirement, and how they relate: stacked, parallel, or later.>

Before designing anything, read <the sections of `ENGINEERING.md` this
touches>.

Where this brief and the requirement disagree, the requirement wins. The
brief's statements about the code are only as reliable as the code: check
each one, and record any that proves wrong as a DISCREPANCY.

## What this branch builds

1. **<Item>** (rule n). <What must be true when it is done. Don't say how,
   unless the how is a constraint.>

## Responsibility and cost

<!-- Filled in by the architect before the implementer starts (decision 0002).
Leave no line out; write "none" where it does not apply. -->

- **Whose job it is:** <for each behaviour this branch touches: disco's, or a
  dependency's (pandas, numpy, graphblas, SQLAlchemy, grpc). What is a
  dependency's job is called, not re-implemented.>
- **Upstream guidance read:** <the release notes, deprecation docs or issues
  read, with links, and what they recommend. "none" only when no dependency's
  behaviour is involved.>
- **Data path:** <whether the change runs on a data path (graph extraction,
  partitioning, routing, transports, the event loop). If it does: the
  benchmark to run before and after, its configuration, and what slowdown,
  if any, the owner has approved.>

## Where this is most likely to go wrong

<!-- The architect's judgement, and the most valuable part of the brief. Each
item reads as an acceptance test. For example:
1. **The limit is inclusive.** Rule 4 says "up to and including", and the
   existing size check in `app_data` uses `>=`. Prove the boundary both ways:
   `limit` rows succeed, `limit + 1` are refused. -->

1. **<Risk>.** <What to check, and how to prove it.>

## Constraints

- No new dependency. <Name the approved ones, if any.>
- Generated files stay generated: run the regenerate step after any change
  that reaches them.
- Your ports and worktree are in your launch prompt. Other sessions run in
  parallel: <their areas>. Stay out of them.

## Done when

- Each rule above is pinned by a test that fails when the rule is broken,
  shown failing.
- The checks in CLAUDE.md § Commands pass, with tests on Linux.
- Somebody has used it:
  1. <Steps, in the user's own terms.>

  Report what you saw. What needs the owner (a real identity provider,
  production) is recorded as "not checked, for the owner".
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, each mutation and its result, and what using
  it showed.

<!-- Each fix round is appended here as "## Round N (date): <title>", and
committed before the session resumes. -->
