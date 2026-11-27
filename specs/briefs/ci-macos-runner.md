# ci-macos-runner: run the macOS wheel jobs on a runner GitHub still provides

There is no requirement for this branch: it is a prescribed small change by
the architect, 2026-11-27. A parallel branch, `graphdata-default-fill`,
works on `src/disco/graph_data.py`.

## What this branch builds

1. **The macOS wheel jobs run.** In `.github/workflows/build.yml:16` and
   `.github/workflows/release-publish.yml:16`, replace `macos-14` with
   `macos-15` in the `os` matrix. Change nothing else in either file.

Why: every `Build` run since release PR #50 has had its macOS job cancelled
within seconds (runs 44929215332, 45636158090, 45782166817, 45877192784,
45878363127, 45878434579). The job's annotation says: "The configuration
'macos-14-arm64-us-default' is not supported". GitHub no longer provides the
runner, so macOS has not been tested, and a published release would build
no macOS wheels.

## Responsibility and cost

- **Whose job it is:** GitHub's runner images; disco's part is only which
  image it asks for. A pinned version (`macos-15`), not `macos-latest`, so
  that an image change is a commit, not a surprise.
- **Upstream guidance read:** the job annotation above. Read GitHub's
  runner-images documentation for the current macOS labels if you can reach
  it, and record what it says, or that you could not.
- **Data path:** none. Which macOS versions users can install on does not
  change: `pyproject.toml` sets `MACOSX_DEPLOYMENT_TARGET = "10.15"` for the
  cibuildwheel macOS build (lines 115-116), and `archs` stays `arm64` and
  `x86_64`.

## Where this is most likely to go wrong

1. **`x86_64` on an arm64 runner.** `macos-15` is arm64; cibuildwheel builds
   `x86_64` there by cross-compiling, and cannot run the `x86_64` tests
   natively. Read cibuildwheel's documentation on this and record what it
   does on the version this project uses. Do not change `archs` or the test
   settings: if the `x86_64` tests would be skipped, record it as ESCALATED,
   for the owner.
2. **The change cannot be tested here.** A workflow runs only on GitHub. The
   proof is the `Build` run on release PR #56 after this lands: say so in
   your Evidence as "not checked, for the owner".

## Constraints

- No new dependency. No other workflow, job or setting changes.
- Your ports and worktree are in your launch prompt; stay out of `src/`.

## Done when

- The two lines are changed, and `git diff` shows nothing else.
- The YAML of both files still parses (check with a YAML parser from the
  project's environment, `uv run python`).
- The branch is pushed, with a Surfaced decisions commit recording the
  runner-images and cibuildwheel guidance read, or that it could not be.
