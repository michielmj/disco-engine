# Proportionate effort: effort in proportion to the change, and a reason for every test

- **Date:** 2026-10-28
- **Decided by:** product owner, in the project this template came from;
  adopted with the template
- **Status:** accepted
- **Origin:** decision 0039 of the sndp project, generalised. It comes with
  the template, so it is named rather than numbered: a project's own
  decisions keep their numbering, and `copier update` never collides with
  them.

## Context

A project run this way, by an owner, an architect, implementers and
reviewers, measured three days of its own process and found that effort did
not follow the size of the change:

- **Prescribed changes:** a 6-line change the owner had written out got the
  full path: a brief, an implementer, a reviewer, a full suite run and a
  mutation proof, about 20 minutes and four commits.
- **Small requirements:** one-branch requirements took two and three
  requirement-review rounds. Half the findings were the author's own factual
  errors, which a check against the code would have caught.
- **Reviews:** every review re-ran the full suite and its own mutations,
  because implementers recorded no evidence in git that a reviewer could
  rely on.
- **Numbers:** each new migration changed a hard-coded head revision in
  about six test files, three times in two days. A requirement on hold was
  renumbered three times.
- **Proofs:** each migration got its own hand-built data-loss proof,
  including for data the application recalculates on its next run.
- **Tests:** about a third of the unit tests guarded errors that had never
  occurred. 168 of 234 tests for one error mapping ran every error against
  every error code, a combination never seen; three of those errors had
  occurred live.

## Decision

- **Effort follows the change.**
  - *A prescribed small change* (the owner gives the code, up to about 20
    lines): built with its tests, then the diff is checked against the
    prescription. No reviewer, unless it touches security, personal data or
    data loss.
  - *A small requirement* (one branch, on existing patterns): the author
    checks the draft's facts against the code first, then one review round.
  - *A large requirement:* as before, two to five rounds.
  - *Follow-ups to the owner's own code:* one branch for the whole review,
    with fix rounds, not a branch per point.
- **Check before review.** The author checks every fact against the code or
  a command's output before stating it, and gives a number (PR, commit,
  count, migration) only once the command that produces it has returned.
- **Evidence once, in git.** The implementer records test counts, each
  mutation and its result, and what using it showed, in the Evidence lines of
  its Surfaced decisions commit. The reviewer runs the test files the diff
  touches and relies on those lines; it re-runs the full suite only when they
  are missing or were taken on other code. The full suite runs once per branch,
  by the implementer, and again before a release.
- **A test needs a reason:** a requirement rule's main path, an incident that
  occurred, or a rule on security, personal data or data loss. Other edge
  cases go to the backlog. One case per rule or table entry, not every
  combination. A data-loss proof covers only data that cannot be
  recalculated.
- **Fewer tests to keep in step.** Tests derive facts that change with every
  branch, such as the latest migration, instead of hard-coding them. A proof
  repeated per instance goes through one shared helper.

`CLAUDE.md` (definition of done, how to record), `REVIEW.md` § Proportion,
`.claude/rules/testing.md`, the implementer and reviewer definitions, and the
implement, review, requirement, record-decisions, prove-a-rule and land
skills say so.

## The owner's words

> On the number of test cases: coverage is good, but we should not overdo
> it. Are we testing for hypothetical errors that have not occurred and are
> unlikely to occur?

> Ok. Let's implement all proposals A-E and 1 and 2. Think about how we can
> generalize the learnings for future projects.

## Consequences

- A prescribed change lands in minutes. A small requirement gets one review
  round. The full suite runs about twice per branch instead of four to six
  times.
- Fewer guards against errors that have not happened. When one happens, it
  is fixed then, with a test: a review FAIL of kind (b).
- The evidence lives in git, where the reviewer and the owner can read it. A
  reviewer that finds it missing says so and runs the suite itself.
- Detected by the reviewer: a test without a reason, or missing Evidence
  lines, is an observation in its report.

## What happened that made this necessary

About 2.5 hours of agent time, over three days, on reviews and requirement
reviews of mostly small changes, and three migration-number cascades
through the tests. The general lesson: a process tuned for the large case
taxes every small one, and "a finding needs a live violation" applies to
tests as much as to review findings.
