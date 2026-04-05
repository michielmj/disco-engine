# C++ / Cython extensions

- event_queue is a pybind11 extension. Any change requires a rebuild:
  `pip install -e .`
- pybind11 variant type binding: always check `bool` before `int` in the
  variant visitor, because Python's bool is a subclass of int.
- Build system: scikit-build-core + CMake. Do not edit generated files.
- Do not add Python-level wrappers that duplicate C++ logic.

# python-graphblas (graph subsystem)

- BOOL `plus` monoid in GraphBLAS is logical OR, not integer addition.
  To count, cast to INT64 first: `matrix.dup(dtype=gb.dtypes.INT64)`.
  Use `.dup(dtype=...)`, not `.new(dtype=...)` or `.apply(identity, dtype=...)`.
- Empty sparse slices (e.g. a label with no assigned vertices) will cause
  np.min/max to raise on a zero-size array. Guard all reductions with
  empty-slice checks.
- Graph is deliberately DB-agnostic. SQLAlchemy belongs in graph.db and
  graph.extract only, never in graph.core.

# Environment constraints

- NEVER install packages system-wide. All pip installs must go to the active
  venv. Verify with `pip --version` — it must resolve inside the project venv
  before installing anything.
- NEVER use sudo for any operation.
- NEVER use `pip install --user` or `--break-system-packages`.
- NEVER install global npm packages (`npm install -g`).
- If the venv is not active (python resolves to a system path), stop and
  report this — do not attempt to activate or create a venv automatically.
- For the C++ extension: always rebuild with `pip install -e ".[dev]"` (inside the
  active venv), never with cmake directly or with system pip.