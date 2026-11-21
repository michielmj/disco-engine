# py314-deps: support Python 3.14–3.15, and upgrade the dependencies

There is no requirement file: this is one branch on existing patterns, and the
owner settled the product questions directly (table below). It starts from
`staging`, which now has ruff, import-linter and the CI static job
(`scaffold-lint-layering`, #48). Read that branch's Surfaced decisions
commits (01949f4, 92e988b) for what its checks cover.

## How this was settled

| Round | Question | The owner's words |
|---|---|---|
| 1 (2026-11-21) | — | "I want to stop support for Python 11 and 12. Next, I want to review the list of dependances and see what we can upgrade, and what the latest version of Python is compatible with my dependencies." |
| 2 | Which Python versions should disco support after dropping 3.11 and 3.12? | "3.13–3.15" (superseded in round 3) |
| 2 | Should disco move to pandas 3 (2.3.3 → 3.1.1)? | "Yes, pandas 3 (Recommended)": require pandas ~=3.1, and note the change for model authors. |
| 2 | Which Python should be the local default in .python-version? | "Newest supported (Recommended)": 3.15. |
| 3 | — | "We will also update all disco-... dependencies to the same Python version. Are there any reasons not to have support only for Python 3.14 and 3.15?" |
| 3 | — | "Let's move to 3.14 and 3.15 only" |
| 4 | — | "disco-data-logger and disco-tools have now been updated to match the Python versions." |

The architect's survey (2026-11-21, PyPI): every compiled dependency ships
wheels for 3.14 and 3.15 on Linux x86_64, macOS arm64 and Windows amd64,
except `disco-data-logger` 0.1.5 (wheels up to 3.13 only; on 3.14 it built
from its sdist). With the pins below, `uv lock --upgrade` resolved, and the
suite gave 520 passed, 1 skipped on 3.14. 3.15 was not tried: no 3.15
interpreter was available. `server.py` starts processes with an explicit
`spawn` context, so 3.14's new Linux default start method does not reach it.

The owner has since released both, for 3.14 and 3.15 (round 4):
`disco-tools` 0.2.0 (pure Python, `requires-python >=3.13`) and
`disco-data-logger` 0.2.0 (wheels for cp313–cp315, `requires-python
>=3.13,<3.16`), checked on PyPI by the architect on 2026-11-21.

## What this branch builds

1. **Python 3.14–3.15** (owner, round 3). `requires-python = ">=3.14"`, with
   no upper bound (disco is a library; an upper bound blocks users for no
   reason). Classifiers name 3.14 and 3.15. cibuildwheel builds
   `cp314-* cp315-*`. mypy's `python_version` is the oldest supported, 3.14,
   so code cannot rely on a newer feature unnoticed. `.copier-answers.yml`
   records `python_version: '3.14'`, which is now true.
2. **CI tests both** (owner, round 3). The `test.yml` matrix is 3.14 and
   3.15; the static job runs on the newest.
3. **Local default** (owner, round 2). `.python-version` is `3.15`, which
   `uv sync` and `scripts/test_linux.sh` (its image tag) both read.
4. **Dependencies upgraded** (owner, rounds 1–2). `pandas~=3.1`,
   `numpy~=2.5`, `python-graphblas~=2026.11`, then `uv lock --upgrade` so
   every runtime and dev dependency is at its latest compatible release.
   Floors written as `>=` stay `>=`; raise one only if the lock needs it.
   Raise the floors of `disco-tools` and `disco-data-logger` to their
   3.14/3.15 releases, `>=0.2.0` each (round 4), so no environment resolves
   an older one that lacks them.
5. **The docs say so.** README: Python 3.14 or later, and a short note for
   model authors that disco now uses pandas 3 (copy-on-write, the new
   default string dtype) and what that can change in their own code.

## Where this is most likely to go wrong

1. **pandas 3 changes behaviour silently.** The suite passed in the survey,
   but pandas 3 makes chained assignment a no-op and strings default to the
   `str` dtype. Acceptance: every pandas call site in `src/disco` that
   assigns into a DataFrame or Series, or tests a dtype (`object`,
   `.dtype ==`, `select_dtypes`), is listed in the Evidence with why it is
   correct under pandas 3; and the suite passes with pandas warnings as
   errors (`-W error::pandas.errors.ChainedAssignmentError`, plus the
   FutureWarnings `pyproject.toml` already makes errors).
2. **SuiteSparse 9 → 10 under python-graphblas 2026.11.** The rules in
   `.claude/rules/graph.md` (BOOL `plus` is OR, `.dup(dtype=...)`) are
   assumptions about the library. Acceptance: the graph and partitioner
   tests pass, and each rule in `graph.md` is checked against the new
   version's behaviour with a one-line scratch check; a rule that no longer
   holds is a DISCREPANCY to escalate, not to fix silently.
3. **ruff's and mypy's target moves to 3.14.** ruff infers it from
   `requires-python`, so new UP findings appear (PEP 695 type parameters,
   `type` aliases, PEP 758 `except` without brackets). Rewriting a `TypeVar`
   or a generic pydantic model changes runtime objects. Acceptance: same
   discipline as the parent branch: tool output in its own commit, every
   unsafe or hand fix listed by rule with the reason it preserves behaviour,
   and the suite unchanged.
4. **Deferred annotations (PEP 649) on 3.14.** Annotations are no longer
   evaluated at definition time. Code that reads `__annotations__` or
   evaluates type hints (the CLI's `argparse_model.py`, pydantic models,
   anything using `typing.get_type_hints`) must still see the same types.
   Acceptance: the Evidence lists each annotation-introspection site and how
   the suite covers it.
5. **The disco-* 0.2.0 releases are a minor-version bump.** Their API may
   have changed, not only their Python support. Acceptance: read their
   changes since the versions the current lock holds (their changelogs, or
   the diff of the sdists), list in the Evidence each change that touches a
   name `src/disco` uses, and show the suite covers it. Also,
   `disco-data-logger` declares `<3.16`: record that as the effective upper
   bound for disco, without copying it into disco's own `requires-python`.
6. **3.15 not available, or a pre-release.** The owner installs the 3.15
   interpreter. If `uv python find 3.15` fails, or finds a pre-release
   (`3.15.0a`/`b`/`rc`), stop and report: upgrading uv or installing an
   interpreter is outside the project (`CLAUDE.md` § Boundaries).
7. **Generated protobuf code.** If the lock moves `grpcio-tools` or
   `protobuf`, regenerate the stubs with the command in `CLAUDE.md` and
   check the runtime is not older than the generated code's version check;
   if neither moves, the stubs stay untouched.

## Constraints

- No new dependency, and none dropped. Upgrades within the owner's decisions
  only.
- No change to disco's own behaviour or public API beyond what pandas 3
  brings; an adaptation needed in `src/disco` for pandas 3, SuiteSparse 10 or
  Python 3.14 is in scope, with a test only where the suite does not already
  cover it.
- Your ports and worktree are in your launch prompt.

## Done when

- Items 1–5 hold, and each item above has its acceptance in the Evidence.
- Every command in `CLAUDE.md` § Commands passes. The suite passes natively
  on 3.14 and 3.15 (`uv run --python 3.1x pytest`, separate environments),
  with the counts for each.
- Somebody has used it: from a fresh clone, `uv sync` (3.15), then the
  commands in order; and the example model in `examples/my-model` (its own
  `pyproject.toml` and `tests/`) installs against this branch and its tests
  pass, on 3.15. Whether GitHub runs the matrix green, and the wheel builds,
  is "not checked, for the owner".
- The branch is pushed, with a Surfaced decisions commit.

## Round 1 (2026-11-21): the Linux script, the pandas note, the record, the example

The review of 0a4a4dd found four FAILs. A re-review follows only for
finding 2, whose remedy is described.

### Finding 1: the Linux test script cannot run on this branch

<finding>
scripts/test_linux.sh:32-34 builds the image tag
`ghcr.io/astral-sh/uv:0.12.15-python3.15-trixie` from `.python-version`
(3.15). That tag does not exist ("manifest unknown"). The architect checked
with `docker manifest inspect` that `0.13.9-python3.15-trixie` and
`0.13.9-python3.14-trixie` both exist. A scratch copy of the script with
`UV_VERSION="0.13.9"` ran at 0a4a4dd and gave Python 3.15.2, 522 passed and
1 skipped.
</finding>

Remedy (prescribed): set `UV_VERSION="0.13.9"` in `scripts/test_linux.sh`,
then run `bash scripts/test_linux.sh py314-deps` on the fixed head and record
its counts. Docker is running now. If CI or another script pins the uv
version too, keep them in step and list each one.

### Finding 2: the pandas 3 note says the reverse of what changed

<finding>
README.md:55-63. "Missing text values are `NaN`, as before" is wrong for
frames built from DB rows, which is how `graph/extract.py::_rows_to_df`
builds them. With a `None` text value, pandas 2.3.3 keeps `None` (dtype
object) and pandas 3.1.1 gives `nan` (dtype str), so model code that tests
`is None` changes behaviour. "`df.where(cond, None)` ... no longer turns them
into `None`" is wrong for float columns: 2.3.3 also left `NaN` there
(`[1.0, nan]` on both). The same misstatement is in the comment at
src/disco/database.py:72-73.
</finding>

Remedy (described): rewrite the note and the comment so that each claim
holds on both versions. Check each claim with a scratch script run against
pandas 2.3.3 (staging's `.venv` in the main checkout, `python -I`) and
pandas 3.1.1, and put each claim and both results in the Evidence. Keep the
note short: what a model author's code sees differently, and what to do
about it.

### Finding 3: the recorded BOOL DISCREPANCY is wrong

<finding>
The Surfaced decisions in 0a4a4dd, DISCREPANCY (1), say that `monoid.plus`,
`agg.sum` and `plus_times` on BOOL "now raise KeyError instead of OR-ing".
They raise identically on python-graphblas 2025.2.0 (staging's lock); the
architect confirmed this for `monoid.plus` and `agg.sum`. On both versions
the `"plus"` string and `gb.op.plus` still OR on BOOL. The upgrade changed
nothing here.
</finding>

Remedy (prescribed): an empty commit, `Correct the Surfaced decisions in
0a4a4dd`, which states that the behaviour is the same on 2025.2.0 and
2026.11.0, with your own check on both versions. Do not edit
`.claude/rules/graph.md`: the architect has escalated its wording to the
owner.

### Finding 4: the example model does not install

<finding>
`uv pip compile --python-version 3.15 examples/my-model/pyproject.toml`
fails: disco-toolbox>=0.0.2 requires python-graphblas<2026, and the example
now pins ~=2026.11. With disco-toolbox installed `--no-deps`, the example's
test fails at `testrun.initialize()` with a missing `params` argument;
`testrun.py`'s signature is not touched by this branch.
</finding>

The owner is releasing a disco-toolbox that allows graphblas 2026.x.
Remedy (described):
- If that release is on PyPI, raise the example's `disco-toolbox` floor to
  it, and bring the example's test in line with the current
  `testrun.initialize()` signature. The example teaches model authors the
  API, so it must call it as it is. Then install the example against this
  branch on 3.15 and run its tests.
- If the release is not on PyPI yet, do findings 1–3, record finding 4 as
  ESCALATED (blocking), and report. Do not wait for it.
