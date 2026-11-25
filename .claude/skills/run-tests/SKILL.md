---
name: run-tests
description: How to run this project's tests correctly, on its target platforms (linux, windows, macos). Use whenever you run pytest, investigate a failure that happens only in CI or only sometimes, or touch a memory or performance bound.
---

# Running the tests

## On Linux, in a container

- **The suite:** `bash scripts/test_linux.sh <branch>`. It tests the
  committed state of the branch, so commit first.
- **Slow tests:** `bash scripts/test_linux.sh <branch> -- -m slow`.
- **One file:** `bash scripts/test_linux.sh <branch> -- tests/test_x.py -k name`.

The container runs Linux, as a non-root user, with no git identity and no
host environment. A test that borrows ambient host state fails here, as it
would in CI, instead of passing on your machine.

The script keeps the shared machine usable:
- **One full suite at a time.** A second full suite waits for the first,
  and says whose it is. A run that names test files doesn't wait.
- **A hard limit,** inside the container: `TEST_LINUX_TIMEOUT` seconds
  (default 3600). Raise it for a slow run rather than working around it.
- **A log per run** in `.test-logs/`, named by branch, commit and time, its
  first line naming the branch and commit tested. Read that line before you
  trust a log.

## Natively

The application also ships to Windows, so on a Windows machine run
`uv run pytest` (add `-m slow`, or a path and `-k name`, as needed).
The application also ships to macOS. Unless you are on a Mac, the only place
the suite runs there is the cibuildwheel job in `build.yml` (PRs to `main`, and
dispatch): read its macOS log when it fails.
A test that passes on one platform and fails on another usually depends on
path separators, line endings, file locking, case-insensitive file names or
the default encoding. Supply what the test needs, and never paper over the
difference with a platform check in the test.

## Logs, and other sessions

- Name every log you write uniquely, with the branch, the commit and the
  time, and check its first line (branch and commit) before you trust it.
  Concurrent sessions have overwritten each other's logs in a shared folder.
- Parallel sessions share the machine, but not its heaviest jobs: run one
  full suite at a time. `test_linux.sh` enforces this for its runs.
- Name every container you start after your branch (`<branch>-postgres`),
  so a leftover can be traced, and stopped, by name.

## A test that fails only sometimes

1. Rerun it alone. If it passes alone, it depends on order or load.
2. Loop it five to eight times, with output captured, to get the assertion
   text.
3. Run the same loop on `origin/main` and on your branch, with a throwaway
   local branch that prints the measured values. This A/B separates "I
   introduced this" from "it was already flaky".
4. Record the numbers with their configuration before you touch any bound,
   and fix the defect before you recalibrate the bound.

## A memory or performance bound

Write it as two assertions: a ratio across a 10× change in size, and a
ceiling derived from the configuration, such as
`memory_limit + threads × allowance`. Measure on each target platform, in the
configuration it ships with.

## The tests for one module

While you work, run the tests that cover the module you changed before the
full suite: `uv run pytest tests/test_<module>.py` for a top-level module,
`uv run pytest tests/<subpackage>/` for `metastore`, `graph`, `model`,
`partitioner` and `cli`. Transports are in `test_grpc_transport.py`,
`test_ipc_transport.py` and `test_routing.py`. When one fails, find the cause
before you propose a fix, and say which it is: the change, the test, or
something that was already broken on `origin/staging`.
