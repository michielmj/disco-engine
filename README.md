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

---

## License

This project is licensed under the Apache License, Version 2.0.
See the `LICENSE` file for details.
