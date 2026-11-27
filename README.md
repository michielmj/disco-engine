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

If you call `TestRun.initialize()` yourself, call it with no argument: the nodes receive the experiment's `params`
(set them on the `Experiment`). `initialize({...})` now raises `TypeError` (a breaking change, allowed in this phase).

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

- **Copy-on-write.** A column or slice you take from a data frame is never a view of it: writing to it no longer
  changes the frame (in pandas 2 it could, for a column or a positional slice). Write through the frame, as `df.loc[mask, "x"] = 1`.
- **Text columns.** They have the `str` dtype, not `object`, so `df[c].dtype == object` is false for them: use
  `pd.api.types.is_string_dtype(df[c])`. A missing text value from the database is now `NaN`, where pandas 2 kept
  `None`: replace `value is None` with `pd.isna(value)`. A column whose values are all missing stays `object`
  and keeps `None`; `pd.isna` handles both.
- **Filling missing rows.** `default_fill` of `get_vertex_data` and of the edge-data functions in `disco.graph` is
  handed to `DataFrame.fillna` as it is. For a table of mixed types pass a mapping, `default_fill={"count": 0, "name": ""}`,
  so that each column gets a value of its own type: pandas warns today when a value does not fit a column's
  dtype, and pandas 4 raises. `GraphData.vertex_data` and `SimProcGraphData.outbound_edge_data` / `inbound_edge_data` take the same
  optional `default_fill` keyword and pass it on unchanged.

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
