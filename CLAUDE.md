# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# disco — Distributed Simulation Core Engine

## Purpose
Event-driven simulation platform supporting local, IPC, and gRPC-distributed
execution across Workers. The architecture is strictly separated into:
- **Control plane**: ZooKeeper-backed metadata and desired-state (Cluster/Metastore).
- **Data plane**: events and promises delivered via Router + Transport stack.

## Source layout
```
src/disco/
  cluster.py        # Cluster, WorkerState, DesiredWorkerState, address book
  orchestrator.py   # Leader-elected scheduler; assigns replications to workers
  worker.py         # Worker lifecycle, runner loop, Assignment
  runtime.py        # NodeRuntime — owns SimProcs for one node, routes outbound envelopes
  node.py           # Node, NodeRuntimeLike protocol, NodeStatus
  simproc.py        # SimProc — event/promise semantics per node
  router.py         # Router — selects Transport for each outgoing envelope
  envelopes.py      # EventEnvelope, PromiseEnvelope dataclasses
  config.py         # AppSettings (Pydantic), GrpcSettings, ZookeeperSettings, …
  transports/
    base.py         # Transport protocol (handles_node / send_event / send_promise)
    inprocess.py    # InProcessTransport — same-process delivery
    ipc_egress.py   # IPCTransport — multiprocessing.Queue-based egress
    ipc_receiver.py # IPC ingress
    ipc_messages.py # IPCEventMsg, IPCPromiseMsg
    grpc_transport.py  # GrpcTransport — outbound gRPC (cached channels)
    grpc_ingress.py    # DiscoTransportServicer — inbound gRPC server
    proto/          # transport.proto + generated stubs (transport_pb2*.py)
  metastore/        # ZooKeeper KV store, LeaderElection, ZkConnectionManager
  experiments/      # Experiment, ExperimentStore, ExperimentStatus
  model/            # Model definition, ORM, loader
  partitioner/      # Graph partitioning strategies (simple, spectral)
  graph/
    core.py         # Graph (python-graphblas, DB-agnostic)
    schema.py       # SQLAlchemy metadata (graph.* schema)
    db.py           # store_graph / load_graph
    extract.py      # mask-based data extraction helpers
  event_queue/      # C++ pybind11 extension (EventQueue)
  cli/              # CLI entry points (disco, disco-server)
```

## Data-plane concepts
- **SimProc**: simulation process for one node; owns the event/promise epoch logic.
  Each SimProc calls back into `NodeRuntime.send_event` / `send_promise` to route.
- **NodeRuntime**: controller for one logical node; holds all its SimProcs and
  delegates outbound messages to `Router`.
- **Router**: ordered list of Transports (InProcess → IPC → gRPC); picks the first
  whose `handles_node(repid, node)` returns True.
- **Envelope** (`EventEnvelope` / `PromiseEnvelope`): the wire container; serialized
  exactly once in NodeRuntime before reaching the router.

## Configuration
`AppSettings` (`config.py`) is the single config root. Precedence (high → low):
init kwargs → env vars (`DISCO_*`) → `.env` / `.env.local` → secrets
(`/run/secrets/disco`) → config file (`config.toml` / `config.yaml` in CWD) → defaults.

Use `get_settings()` (cached) in production; pass `AppSettings(...)` kwargs in tests.

## Build & verify
```bash
pip install -e ".[dev]"          # editable install including C++ extension
pytest                            # full test suite
pytest tests/test_worker.py -v   # single file; use :: for individual tests
mypy --strict src/disco           # type check
```

After editing `transport.proto`, regenerate stubs:
```bash
python -m grpc_tools.protoc \
  -I src \
  --python_out=src \
  --grpc_python_out=src \
  src/disco/transports/proto/transport.proto
```

## Key conventions
@.claude/rules/spec.md
@.claude/rules/testing.md
@.claude/rules/transport.md
@.claude/rules/worker.md
@.claude/rules/extensions.md
