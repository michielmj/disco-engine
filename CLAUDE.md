# disco — Distributed Simulation Core Engine

## Purpose
Event-driven simulation platform supporting local, IPC, and gRPC-distributed
execution across Workers. The architecture is strictly separated into:
- **Control plane**: ZooKeeper-backed metadata and desired-state (Cluster/Metastore).
- **Data plane**: events and promises delivered via Router + Transport stack.

## Source layout
```
src/disco/
  cluster/          # Cluster, Orchestrator, address book
  worker/           # Worker, WorkerState, runner loop
  transport/        # Router, Transport implementations
    proto/          # transport.proto + generated stubs
  graph/
    core.py         # Graph (python-graphblas, DB-agnostic)
    schema.py       # SQLAlchemy metadata (graph.* schema)
    db.py           # store_graph / load_graph
    extract.py      # mask-based data extraction helpers
  event_queue/      # C++ pybind11 extension
  config.py         # GrpcSettings and other Pydantic config
```

## Build & verify
- Install (editable, incl. C++ ext): `pip install -e ".[dev]"`
- Tests: `pytest`
- Type check: `mypy --strict src/disco`
- After proto changes: regenerate stubs with `grpc_tools.protoc`

## Key conventions
@.claude/rules/spec.md
@.claude/rules/testing.md
@.claude/rules/transport.md
@.claude/rules/worker.md
@.claude/rules/extensions.md
