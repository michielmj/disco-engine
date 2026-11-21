---
paths:
  - "CMakeLists.txt"
  - "cmake/**"
  - "src/**/CMakeLists.txt"
  - "src/**/*.cpp"
  - "src/**/*.hpp"
  - "src/**/*.h"
  - "src/**/*.pyx"
  - "src/**/*.pxd"
  - "src/**/*.pyi"
---

# The native core (C++ with pybind11)

- The native core is `src/disco/event_queue/`: the compiled `_core` module
  (pybind11) and its Python facade in `__init__.py`. The top-level
  `CMakeLists.txt` adds each native subpackage with one `add_subdirectory`
  line, using the macros in `cmake/extension_helpers.cmake`; the build also
  supports Cython targets.
- `uv sync` and `uv run` rebuild `_core` when a native source changes, because
  `[tool.uv] cache-keys` in `pyproject.toml` lists the sources. A new kind of
  source file, or one outside `src/` and `cmake/`, must be added there too, or
  the tests run against a stale build without saying so.
- Every function `_core` exports is declared, with its Python signature, in
  `_core.pyi`; mypy checks the callers against it. Change the stub in the
  same commit as the function.
- In a pybind11 variant visitor, check `bool` before `int`: Python's `bool`
  is a subclass of `int`, so the other order turns `True` into `1`.
- Errors cross into Python as exceptions: throw `std::invalid_argument`
  (ValueError) or `std::out_of_range` (IndexError). Never let a crash reach
  Python, and never return an error code that a caller must remember to check.
- Take inputs by const reference and return values by value. Don't keep a
  pointer or reference to a Python object beyond the call.
- Release the GIL (`py::gil_scoped_release`) only around code that touches
  no Python object.
- Don't add Python wrappers that duplicate logic the C++ already has: two
  implementations drift.
- Prove a rule in native code as anywhere else (prove-a-rule): change the
  source, let `uv run` rebuild it, and require the named test to fail.
- Formatting follows `.clang-format`; CI checks it.
- CI builds a wheel per target platform with cibuildwheel. A change that needs
  a new compiler flag or a system library is an escalation, like a new
  dependency.
