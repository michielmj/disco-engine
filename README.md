# 🧾 Disco Engine

**Disco Engine** is a Python-based *distributed discrete-event simulation core* (“distributed simulation runtime”).
It’s designed to run large, structured simulations (e.g., supply-chain / networked process models) by executing many
independent **NodeRuntimes** that exchange **events** and **promises** over pluggable transports.

- Deterministic single-threaded stepping per worker process
- Scales out via multiple workers and transport backends
- Built for “engine + plugins” (models as Python packages)

[![PyPI](https://img.shields.io/pypi/v/disco-engine.svg)](https://pypi.org/project/disco-engine/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Build](https://github.com/michielmj/disco-engine/actions/workflows/build.yml/badge.svg)](https://github.com/michielmj/disco-engine/actions/workflows/build.yml)
[![Tests](https://github.com/michielmj/disco-engine/actions/workflows/test.yml/badge.svg)](https://github.com/michielmj/disco-engine/actions/workflows/test.yml)

---

## Overview

At a high level, a Disco simulation is:

- a **Layered Graph** (scenario) that defines structure and data access,
- a **Model plugin** that defines simulation logic,
- an **Experiment** that defines parameters and run configuration,
- one or more **Workers** that host NodeRuntimes and step the simulation.

For local debugging and unit tests, Disco also includes a lightweight **TestRun** that executes a single, unpartitioned
run **without** threading, metastore/cluster coordination, or external transports.

---

## ✨ Features

- **Worker lifecycle & state machine** for long-lived simulation processes
- **Routing + transports** (in-process, IPC, gRPC) for node-to-node delivery
- **Partitioning** to distribute nodes across workers (production), while keeping deterministic node order per worker
- **TestRun** for fast local runs without infrastructure (single-process, in-process transport only)
- **Data logging** via an optional `DataLogger`

---

## 🚀 Installation

```bash
pip install disco-engine
```

Disco requires **Python 3.14 or later** (tested on 3.14 and 3.15). Its `disco-data-logger` dependency declares
`<3.16`, so in practice Disco runs on 3.14 and 3.15 until that package supports a newer Python.

### For model authors: pandas 3

Disco uses **pandas 3** (`pandas~=3.1`). The data frames that `disco.graph` and `GraphData` return follow pandas 3
semantics, which can change your own model code:

- **Copy-on-write.** Assigning through a chained expression, such as `df[mask]["x"] = 1`, no longer changes `df`
  (pandas warns with `ChainedAssignmentError`, a warning class). Use `df.loc[mask, "x"] = 1`. A slice or a column you take from a data
  frame is never a view of it, so changing one does not change the other.
- **String dtype.** Text columns have the `str` dtype by default, not `object`. Code that tests
  `df[c].dtype == object` is now false for text columns: use `pd.api.types.is_string_dtype`, or
  `select_dtypes("str")`. Missing text values are `NaN`, as before.
- **Missing values.** `df.where(cond, None)` keeps `NaN` in float and `str` columns; it no longer turns them into
  `None`. Disco's `df_to_table` handles this for the rows it stores.

The pandas 3 release notes list the other changes.

### Development

Development uses [uv](https://docs.astral.sh/uv/). `uv sync` installs the
locked runtime and dev dependencies into `.venv` and builds the C++ event
queue; it rebuilds it when a native source changes.

```bash
uv sync
uv run pytest
uv run mypy src/disco
```

---

## License

This project is licensed under the Apache License, Version 2.0.
See the `LICENSE` file for details.
