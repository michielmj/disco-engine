# scaffold-lint-layering: ruff, import-linter, clang-format and their CI checks

There is no requirement file. This is steps 4 and 5 of adopting the
scaffolder template, which the product owner directed on 2026-11-21 ("First
complete steps 4 and 5"). The parent branch, `scaffold-adoption`, did steps
1–3: read its four commits (`git log origin/staging..origin/scaffold-adoption`)
for what is already in place. This branch is stacked on it and lands after it.

Settled by the owner, not to be reopened here:
- `disco` stays the single top-level package. A full restructuring, breaking
  changes allowed, comes *after* the template is adopted, on its own
  requirement. Don't move modules to satisfy a contract.
- mypy stays non-strict on this branch; `strict = true` is backlog item 2 in
  `STATUS.md`.

Before you start, read `CLAUDE.md` § Commands and § Invariants,
`.claude/rules/python.md`, `.claude/rules/native.md`, and `ENGINEERING.md`
§ 2.8 and Chapter 7's opening.

Where this brief and the code disagree, the code is the fact: check each
statement below, and record any that proves wrong as a DISCREPANCY.

## What this branch builds

When it is done, every command in `CLAUDE.md` § Commands runs and passes,
locally and in CI, and the codebase behaves exactly as it does at the base.

1. **ruff.** The template's configuration (`line-length = 110`, lint
   `select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF"]`), with generated
   files excluded. `uv run ruff format --check .` and `uv run ruff check .`
   pass. Measured at the base with that configuration and the proto stubs
   excluded: 72 files would be reformatted, and about 1,000 lint findings,
   mostly UP006/UP045/UP035 (annotations), E501 (102), E402 (80), I001 (72),
   RUF059 (40), F401 (18), B905 (10).
2. **mypy runs as `uv run mypy`.** Today `[tool.mypy]` names no `files`, so
   the bare command has nothing to check. Give it `files` covering
   `src/disco` and `.claude/hooks`, and `tests` if that passes without
   loosening anything; otherwise record why not. Remove the
   `[[tool.mypy.overrides]] module = "tests.*"` section if it stays unused
   (mypy currently warns about it). No strictness changes.
3. **import-linter.** A `.importlinter` whose contracts describe the layering
   disco *already has*, between its subpackages and modules. At least:
   - only `disco.config` imports `pydantic_settings` (true today:
     `grep -rn pydantic_settings src/disco` finds only `config.py`);
   - `disco.graph.core` imports no `sqlalchemy` (`.claude/rules/graph.md`).

   Survey the real import graph first, then propose the further contracts
   it supports, such as transports not importing the worker or orchestrator,
   or the metastore importing no other disco package. Write only contracts
   the code satisfies today. A layering you would want but the code
   violates goes into your report as a backlog item with the offending
   imports, for the restructuring requirement. Never add an `ignore_imports`
   exception to make a contract pass. `uv run lint-imports` passes.
4. **clang-format.** The template's `.clang-format`: the file
   `template/[% if native in ['nanobind', 'pybind11'] %].clang-format[% endif %]`
   in `git@github.com:michielmj/scaffolder.git` at tag `v0.3.0` (clone it
   into your own scratch directory; the file needs no rendering). The five C++ sources and
   headers under `src/disco/event_queue/` are formatted, and
   `git ls-files "*.cpp" "*.hpp" "*.h" | xargs uv run clang-format --dry-run --Werror`
   passes.
5. **Dev dependencies.** `ruff~=0.17`, `import-linter~=2.15` and
   `clang-format~=23.1` in the `dev` group, and `uv.lock` updated.
6. **CI (step 5).** `.github/workflows/test.yml` also runs ruff check, ruff
   format check, lint-imports and the clang-format check, once per run (a
   separate job, or the 3.12 leg only), alongside the existing pytest and
   mypy steps. Model it on the template's `template/.github/workflows/ci.yml`
   where it fits; leave `build.yml`, `build-windows.yml` and
   `release-publish.yml` alone. `CLAUDE.md` says CI runs the tests on Linux,
   macOS and Windows: check whether the existing workflows actually do so on
   a push or PR, and record the answer (a DISCREPANCY if they don't). Don't
   add a platform matrix on this branch.

## Where this is most likely to go wrong

1. **A lint fix that changes behaviour.** Most findings are mechanical, but
   several rules invite a semantic change: B905 (`zip(..., strict=True)`
   raises where the code silently truncated; keep today's behaviour with
   `strict=False` unless you can show the lengths always match), F841 and
   RUF059 (a removed assignment whose right-hand side has a side effect),
   SIM105/SIM118, and E402 (moving an import above code that it depends on,
   such as a path or warnings setup). Acceptance: the full suite gives the
   same result as at the base, **520 passed, 1 skipped**, and the Evidence
   lines list every non-autofixed change by rule, with the reason it
   preserves behaviour.
2. **Annotation rewrites under Python 3.11.** `requires-python` is `>=3.11`,
   and the pydantic models in `config.py`, `experiments` and elsewhere
   evaluate their annotations at runtime. Acceptance: the suite also passes
   on 3.11 (`uv run --python 3.11 pytest`), the oldest version CI tests.
3. **Generated files edited.** `src/disco/transports/proto/transport_pb2*.py`
   and `src/disco/_version.py` are generated. Exclude them from ruff, and
   from mypy if needed. Acceptance: `git diff origin/scaffold-adoption --
   src/disco/transports/proto/ src/disco/_version.py` is empty.
4. **The format commit hides other changes.** Acceptance: one commit holds
   only the output of `ruff format` (a reviewer re-runs it on the parent and
   gets the same tree), one holds only `ruff check --fix` safe fixes, and
   hand fixes come after, in their own commits. The same goes for
   clang-format.
5. **A contract that cannot fail.** A forbidden-import contract with a typo
   in a module name passes silently. Acceptance: for each contract, a
   scratch mutation that adds the forbidden import makes `uv run
   lint-imports` fail, naming that contract (the prove-a-rule skill);
   revert it afterwards.
6. **A stale native build.** After clang-format, the extension must really
   be rebuilt. Acceptance: `uv sync` rebuilds `disco-engine` (its output
   says so), and the event-queue tests pass.
7. **Ignores that grow into a loophole.** A `# noqa` or a `per-file-ignores`
   entry is allowed only for a specific line or file, with a comment saying
   why. Turning off a rule project-wide is loosening a lint rule: escalate
   it (non-blocking; carry on with the rest) rather than deciding it.

## Constraints

- No new dependency beyond the three in item 5, which the owner approved as
  part of adopting the template.
- No change to behaviour, public API, the wire format or the spec. This
  branch changes no contract, so `ENGINEERING.md` needs no update; if you
  find it must change, that is an escalation.
- Generated files stay generated.
- Your ports and worktree are in your launch prompt. No other session runs in
  parallel.

## Done when

- Items 1–6 hold, and each "most likely to go wrong" item has its
  acceptance shown in the Evidence lines. There is no new test to write:
  the existing suite is the proof of "no behaviour change", and the contract
  mutations are the proof for item 3.
- Every command in `CLAUDE.md` § Commands passes. Run the suite natively
  (`uv run pytest`) on 3.12 and 3.11; run `bash scripts/test_linux.sh
  scaffold-lint-layering` if Docker is available, and otherwise record it
  as "not checked, for the owner".
- Somebody has used it:
  1. From a fresh clone of the branch, `uv sync`, then each command in
     `CLAUDE.md` § Commands, in order, exactly as written there.
  2. Read `test.yml` as a reviewer would: every command it runs exists
     locally and passed in step 1. Whether GitHub runs it green is "not
     checked, for the owner" unless you can see the run.
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, the per-rule list of hand fixes, each contract
  mutation and its result, and what using it showed.

<!-- Each fix round is appended here as "## Round N (date): <title>", and
committed before the session resumes. -->

## Round 1 (2026-11-21): Markdown kept out of ruff format; readable wrapping

The review of 01949f4 found one FAIL and several observations. Fix the items
below; the other observations went to the backlog. No re-review follows:
items 1, 3 and 4 are prescribed, and item 2 changes only comments and
docstrings, so the architect checks this round's diff instead.

1. **ruff format changed the spec.** FAIL (c).
   <finding>
   ENGINEERING.md (71 lines changed, all in b7fc6b5 "Apply ruff format (output
   only)"). The brief says: "No change to ... the spec ... if you find it must
   change, that is an escalation." ruff 0.17.6 formats the Python code blocks
   inside Markdown files, so the format commit reformatted the spec's code
   snippets. `git diff -w` shows only layout changes. ... `ruff format --check .`
   now also gates the Python blocks in README.md, CLAUDE.md, STATUS.md,
   REVIEW.md, specs/** and .claude/skills/**.
   </finding>
   Remedy (prescribed): exclude Markdown from ruff format, in
   `[tool.ruff.format]` (`exclude = ["*.md"]`), so `ruff check` is unchanged;
   and restore `ENGINEERING.md` to its state on `origin/scaffold-adoption`
   (`git checkout origin/scaffold-adoption -- ENGINEERING.md`), in one commit.
   Acceptance: `git diff origin/scaffold-adoption -- '*.md'` shows only this
   brief, and `uv run ruff format --check . -v` no longer lists any `.md`
   file. Reason for excluding rather than accepting: the spec's snippets are
   illustrations at contract level, edited through the sync-spec skill; a
   formatter gate over every document is a workflow change the owner has not
   decided.

2. **The E501 wrapping left orphaned words.** Observation.
   <finding>
   E501 commit 8a74438: about 35 added lines are lone leftover words from
   hard-splitting at column 110 — e.g. the simproc.py module docstring
   ("can be\nthought\nof as…", "may\n  be\n  adjusted…") and worker.py comments
   split mid-code (`("",` / `# logging.INFO)])`).
   </finding>
   Remedy (described): re-wrap every comment and docstring that 8a74438
   split, as normal prose within 110 columns; a commented-out code line goes
   on its own line(s) where it still reads as code, or is removed if it is
   dead. Only comments and docstrings change. Acceptance: no line added by
   8a74438 or this round consists of one or two words left over from a
   split (the reviewer's examples included), and ruff check still passes.

3. **The `Optional[X]` branch of the CLI parser has no test.** Observation;
   this guards the regression the branch itself caused.
   <finding>
   src/disco/cli/argparse_model.py:14: the `origin is Union` branch
   (`Optional[X]`) is no longer exercised. tests/cli DemoModel was rewritten
   to `X | None`, so only the `types.UnionType` branch has a test.
   </finding>
   Remedy (prescribed): one test in `tests/cli/` with a model field written
   `typing.Optional[X]` (keep a `# noqa: UP045` with the reason: the test
   needs the old spelling). Prove it: remove `origin is Union` from the
   condition and show the new test fails, then restore.

4. **`__all__` section comments in `disco/model/__init__.py`.** Observation.
   <finding>
   The RUF022 sort left the `__all__` section comments wrong: the "# orm"
   heading now also covers the loading functions, and "# loading" now covers
   the ORM errors.
   </finding>
   Remedy (prescribed): drop the section comments inside `__all__` (a sorted
   list has no sections).

5. **Surfaced decisions to correct.** The block in 01949f4 says "12 sites"
   for B905; the branch has 10. It says "prose re-wrapped" for 8a74438,
   which item 2 shows was not so. Correct both in the new Surfaced decisions
   commit, naming 01949f4.
