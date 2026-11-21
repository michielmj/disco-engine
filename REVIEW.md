# REVIEW.md

The standard for reviewing a task branch. A reviewer reads this first, and the
architect writes review prompts against it.

## Scope

- One branch: `git diff origin/staging...origin/<branch>` and its log, after
  `git fetch origin`.
- Work in a detached worktree of the branch (`scripts/new_task.sh --review`).
  You never edit the branch, commit, push or change the PR; a hook enforces
  this.
- Read the branch's brief and the requirement rules it names.

## FAIL or OBSERVATION

Report every finding, then classify it. A FAIL is one of these:
- **(a)** a regression against `origin/staging`;
- **(b)** a defect a real user can hit;
- **(c)** a stated invariant broken on a realistic path, a rule of the brief
  not met, or a test that does not enforce its rule;
- **(d)** a security issue on a reachable path.

Everything else is an OBSERVATION, in one line. A FAIL needs a live
violation; "someone could later write X" is an observation. Report
hypotheticals as observations rather than leaving them out: the architect
decides what goes to the backlog.

<examples>
<example>
FAIL (c): `src/disco_data/export.py:88` checks the limit with `>=`, so an
export of exactly the limit is refused, but rule 4 says "up to and
including". `tests/test_export.py::test_limit` exports `limit - 1` rows and
passes either way: changing `>=` to `>` leaves the suite green.
</example>
<example>
OBSERVATION: `parse_header` would accept a negative version if a future
writer produced one; no writer does today.
</example>
<example>
UNVERIFIED: the "somebody has used it" step needs the real identity
provider. I could not run it, and the implementer recorded it as "not
checked, for the owner".
</example>
</examples>

## Evidence

- A claim of absence needs a command, with its output.
- A claim about behaviour needs the test that asserts it, cited by name, or a
  scratch mutation you ran.
- A test that cannot fail is a FAIL. For a test named after a defect, revert
  the fix and require that test to fail.
- UNVERIFIED is a legitimate verdict. Never upgrade it to tidy a report.

## Proportion

- Review what the diff did; don't audit the rest of the codebase.
- **Rely on the recorded evidence** (the proportionate-effort decision).
  The implementer's Surfaced decisions commit records test counts, each
  mutation and its result, and what using it showed, with the commit they
  ran on. Run the checks and the test files the diff touches. Re-run the
  full suite only when that evidence is missing or was taken on other code
  than you review, and say which. It is the same code when
  `git diff --stat <evidence commit> origin/<branch> -- . ':(exclude)specs/briefs'`
  prints nothing: only empty commits and brief rounds came after it.
- Run a scratch mutation only for a rule the evidence gives none for, or a
  test that looks as if it cannot fail: at most one per branch.
- A test without a reason (a rule's main path, an incident, or a rule on
  security, personal data or data loss) is an observation, not a gap to
  fill. So is an untested edge case, unless a real user can hit it (b).
- A guard earns its place by having failed once, and gets one round.

## Checklist (every branch)

1. Each rule in the brief has a test that fails when the rule is broken.
2. The layering contracts hold (`uv run lint-imports`).
3. Generated files were regenerated, not edited, and nothing drifted.
4. The docs are current: `ENGINEERING.md` for a milestone, `user-guide/` if
   what a user sees changed, `README.md` if what an operator does changed.
5. A message a user sees names no operator material: no command, setting,
   path, API call or exception class.
6. The Surfaced decisions block matches the diff, and its Evidence lines
   were taken on the code you review. A decision made silently is a finding.
7. Nothing reads ambient host state (identity, tools, encoding) without
   supplying it.

## Report format

```
## Verdict: <n> FAIL
### FAILs
- <file:line> — <what> — <evidence>
### OBSERVATIONS
- <one line each>
### UNVERIFIED
- <claim> — <what would settle it>
### What I ran
- <commands and results>
```
