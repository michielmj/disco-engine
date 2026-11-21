---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "tests/**/*.py"
---

# Python in this repository

- Use relative imports inside a package and absolute imports across packages,
  so `lint-imports` sees every cross-package dependency.
- Each module defines its own exceptions, and callers catch those rather than
  a broad `Exception`.
- Use Pydantic at the boundaries only (the API, settings, files on disk). The
  domain model is plain dataclasses, which keeps it free of I/O.
- End a dispatcher in `assert_never`, not a silent default: a new case then
  fails type-checking instead of falling through.
- Make a rule a required parameter rather than a convention at each call
  site. A bypass should be a `TypeError` or a failing check, not a review
  finding.
- A message a user sees says what is wrong and "ask your administrator to …".
  The command, path, setting or exception class goes to the server log,
  because users cannot act on operator material.
- `mypy --strict` passes without `# type: ignore`. Where one is unavoidable,
  add a comment saying why.
