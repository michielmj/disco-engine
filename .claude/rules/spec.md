# Spec-driven development

- ENGINEERING_SPEC.md is the authoritative source of truth. Read the relevant
  chapter before changing any module. Chapters map to subsystems:
    Ch 2   — core terminology (envelopes, Router, Transport, WorkerState)
    Ch 3   — Metastore (ZooKeeper-backed KV store)
    Ch 4   — Cluster (address book, desired-state, worker metadata)
    Ch 5   — Worker lifecycle and runner loop
    Ch 6   — Routing and Transports (InProcess, IPC, gRPC)
    Ch 7   — Graph (python-graphblas, schema, DB)
    Ch 8+  — Model definition and runtime blocks

- Every implementation change must be reflected in the spec in the same task.
  Update the exact section(s) affected — no broad rewrites of untouched sections.

- The spec describes *contracts and invariants*, not implementation details.
  When updating it, match that level of abstraction.