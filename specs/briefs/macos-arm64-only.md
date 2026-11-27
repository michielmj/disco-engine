# macos-arm64-only: macOS wheels for Apple Silicon only

There is no requirement for this branch: it is a prescribed small change,
from the owner's decision of 2026-11-27. Asked "which Macs does disco
support?", with A: "Apple Silicon only. Stop building Intel Mac wheels, and
say in the README that macOS means Apple Silicon", the owner answered "A".
A parallel branch, `graphdata-default-fill`, is in review
(`src/disco/graph_data.py`, Ch 7, README's "Filling missing rows" bullet).

## What this branch builds

1. **`pyproject.toml:113`**, in `[tool.cibuildwheel.macos]`: `archs =
   ["arm64", "x86_64"]` becomes `archs = ["arm64"]`. Nothing else in the
   cibuildwheel settings changes.
2. **`README.md`**, the installation paragraph (line 51, "Disco requires
   **Python 3.14 or later** ..."): one sentence that on macOS disco supports
   Apple Silicon only, because its dependency python-graphblas requires
   numba, which publishes no Intel macOS wheels for Python 3.14 and 3.15.
   Stay out of the "Filling missing rows" bullet: the other branch edits it.

## Responsibility and cost

- **Whose job it is:** which platforms numba (via python-graphblas) supports
  is theirs; disco follows it and does not work around it (decision 0002).
- **Upstream guidance read:** the failed `Build wheels (macos-15)` job of run
  45962694505: "Could not find a version that satisfies the requirement
  llvmlite<0.47,>=0.46.0dev0 (from numba)"; `uv.lock`: python-graphblas
  2026.11.0 requires numba for `python_full_version < '3.16'`, and numba
  0.68.1's and llvmlite's macOS wheels for cp314 are `macosx_12_0_arm64`
  only.
- **Data path:** none.

## Where this is most likely to go wrong

1. **The change cannot be tested here.** The proof is the `Build` run on
   release PR #56 after this lands: the Apple Silicon wheels must build and
   pass their tests. Record it as "not checked, for the owner".
2. **The README sentence overclaims.** Say what is true today and why, not
   that Intel Macs can never be supported.

## Constraints

- No new dependency. No other file changes besides the brief and the
  Surfaced decisions.
- Your ports and worktree are in your launch prompt; stay out of `src/`.

## Done when

- `git diff` shows only the two changes; `pyproject.toml` still parses
  (`uv run --python 3.14 python -c "import tomllib; ..."` or `uv lock
  --check`).
- The branch is pushed, with a Surfaced decisions commit.
