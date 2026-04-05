# Worker invariants (Chapter 5)

## State machine
CREATED → AVAILABLE → RESERVED → INITIALIZING → READY → ACTIVE ⇄ PAUSED
Any state → TERMINATED (then back to AVAILABLE) or EXITED or BROKEN.

## Critical invariants — never violate these

1. The Worker is the sole mutator of its own WorkerState.
   Orchestrator sends a desired-state command; the Worker applies it on
   its runner thread and publishes the resulting WorkerState.

2. RESERVED is a lightweight transition (no DB/model I/O). It exists to
   prevent double-assignment races. The Orchestrator waits for RESERVED
   confirmation before proceeding to the next assignment.

3. All simulation stepping and state transitions happen on the runner thread
   (single-threaded for determinism). The desired-state callback runs on a
   separate thread and only stores the update + wakes the runner via _kick.

4. The ACTIVE hot path (drain ingress → step runners) acquires no locks.
   Locks are taken only for: applying desired-state, reporting status, teardown.

## ExperimentStatus vs WorkerState
- WorkerState is published to the Cluster (control plane).
- ExperimentStatus per partition (LOADED, INITIALIZED, ACTIVE, FINISHED,
  PAUSED, CANCELED, FAILED) is maintained via ExperimentStore, separately.
- Workers do not have FINISHED/FAILED states — completion is tracked at the
  partition level.

## Error classification
- Partition-fatal: report exception, set partition FAILED, teardown, back to AVAILABLE.
- Worker-fatal: publish BROKEN, stop runner. Recovery requires process restart.