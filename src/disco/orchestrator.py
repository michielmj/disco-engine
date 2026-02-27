# src/disco/orchestrator.py
from __future__ import annotations

from threading import Event, Thread
from time import monotonic
from typing import Optional, Any

from tools.mp_logging import getLogger

from disco.cluster import Cluster, WorkerState
from disco.config import OrchestratorSettings
from disco.experiments import ExperimentStatus, ExperimentStore, Experiment, Submission
from disco.partitioning import Partitioning

logger = getLogger(__name__)

# How long we wait for the *current* (preferred) partitioning scheme to become runnable
# before considering the next scheme. Keep small and simple for now; can be moved to config.
PARTITIONING_FALLBACK_S = 1.0

# How long the launch thread waits for all workers to reach READY before retrying/logging.
READY_POLL_S = 0.1

# How long we ask dequeue to wait before doing an iteration (e.g. checking stop flag)
DEQUEUE_TIMEOUT_S = 1.0

# How long we wait between subsequent worker availability checks
AVAILABLE_POLL_S = 0.5


class _StopRequested(RuntimeError):
    pass


class Orchestrator:
    """
    Orchestrator: assigns submitted replications to workers and triggers launch.

    Key properties (per requirements):
    - Leader-elected: multiple instances may exist, only the leader is active.
    - Scheduling-only: updates experiment.selected_partitioning and assignment mapping,
      and drives desired worker states (READY -> ACTIVE). No locality awareness yet.
    - FIFO: consumes submitted replications in queue order and does not skip ahead.
    - All-at-once assignment: never assigns a subset of partitions for a replication.
    - Reuse-aware: prefers workers already set up for (expid, partition) via Cluster.get_available(expid).
    """

    def __init__(
            self,
            *,
            address: str,
            cluster: Cluster,
            settings: OrchestratorSettings,
    ) -> None:
        self._address = address
        self._cluster = cluster
        self._settings = settings

        self._stop = Event()

        self._store = ExperimentStore(cluster.meta)

        # Leader election (published leader record is handled by LeaderElection itself).
        self._election = self._cluster.make_orchestrator_election(
            address=address,
        )

        # Launch threads (best-effort bookkeeping for stop/join).
        self._launch_threads: list[Thread] = []

    def request_stop(self, timeout_s: Optional[float] = None) -> None:
        self._stop.set()
        try:
            self._election.cancel()
        except Exception:
            pass

        if timeout_s is not None and timeout_s > 0:
            deadline = monotonic() + timeout_s

            for t in list(self._launch_threads):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                t.join(timeout=remaining)

            still_alive = [t for t in self._launch_threads if t.is_alive()]
            if still_alive:
                logger.warning("Orchestrator shutdown: %d launch threads still alive after timeout.",
                               len(still_alive))
        else:
            for t in list(self._launch_threads):
                t.join()

    def run_forever(self) -> None:
        logger.info("Orchestrator %s starting leader election.", self._address)
        self._election.run(self._on_lead)

    # ------------------------------------------------------------------ #
    # Leader loop
    # ------------------------------------------------------------------ #

    def _on_lead(self) -> None:
        logger.info("Orchestrator %s is leader.", self._address)

        while not self._stop.is_set():
            entity = self._store.dequeue(
                timeout=DEQUEUE_TIMEOUT_S,
                force_mode='raise',
            )

            if entity is None:
                continue

            expid, repid = entity.value
            try:
                # IMPORTANT: keep entity locked until we either:
                # - release() (retry later, stays at head), OR
                # - consume() (remove permanently; success or fail)
                self._handle_submission(entity=entity)

            except _StopRequested:
                # Scenario 1: stop during "waiting for enough workers" or any pre-launch stage
                # -> release so submission remains available at queue head
                entity.release()
                return

            except Exception as exc:
                # Scheduling-side unexpected failure: mark failed so it doesn't go unnoticed,
                # and consume to prevent infinite head-of-queue retries.
                logger.exception(
                    "Orchestrator failed scheduling expid=%s repid=%s: %r",
                    expid,
                    repid,
                    exc,
                )
                self._mark_replication_failed(expid=expid, repid=repid, exc={"description": str(exc)})
                entity.consume()

        logger.info("Orchestrator %s relinquishing leadership.", self._address)

    # ------------------------------------------------------------------ #
    # Submission handling
    # ------------------------------------------------------------------ #

    def _handle_submission(self, *, entity: Submission) -> None:
        """
        Schedule a single submitted replication.

        Policy:
        - Ensure experiment has selected_partitioning (choose once and keep).
        - Ensure seeds for this replication after partitioning is selected.
        - Wait until sufficient AVAILABLE workers exist for *all partitions*.
        - Assign all partitions in one go (reuse-aware).
        - Start a launch thread that waits for all READY then sets ACTIVE.
        """
        if self._stop.is_set():
            raise _StopRequested()

        expid, repid = entity.value

        exp = self._store.load(expid)

        partitioning_id = exp.selected_partitioning
        if partitioning_id is None:
            partitioning_id = self._select_partitioning_for_experiment(exp=exp)
            self._store.select_partitioning(expid, partitioning_id)
            # Keep local variable; no reload required in the single-orchestrator paradigm.

        # Number of partitions for the selected scheme.
        num_partitions = int(Partitioning.load_metadata(self._cluster.meta, partitioning_id)["num_partitions"])

        # Decide which workers will run which partitions (all-at-once).
        assignments = self._await_full_assignment_plan(expid=expid, num_partitions=num_partitions)
        if self._stop.is_set():
            raise _StopRequested()

        # Persist assignments (scheduling-side truth).
        exp = self._store.assign_partitions(expid, repid, assignments)

        # If we get here, assignment succeeded and we consume the entity before handing over to launch.
        entity.consume()

        # Launch in a separate thread (READY barrier then ACTIVE).
        t = Thread(target=self._launch_replication, args=(exp, repid), daemon=True)
        self._launch_threads.append(t)
        t.start()

    def _select_partitioning_for_experiment(self, exp: Experiment) -> str:
        """
        Select a partitioning scheme for an experiment once.

        - Consider allowed_partitionings in order.
        - If enough workers are AVAILABLE to run the first scheme, choose it.
        - If fewer are AVAILABLE but the cluster has enough total workers, wait briefly.
        - After the wait, consider the next scheme (expected to have fewer partitions).
        """
        allowed: list[str] = list(exp.allowed_partitionings)

        if not allowed:
            raise RuntimeError(
                f"Experiment {exp.expid!r} is unpartitioned (no allowed_partitionings) but Worker requires "
                f"selected_partitioning."
            )

        total_workers = len(self._cluster.worker_states)

        for idx, pid in enumerate(allowed):
            num_partitions = Partitioning.load_metadata(self._cluster.meta, pid)["num_partitions"]

            if total_workers < num_partitions:
                # Cannot ever run this scheme with current cluster size; try next.
                continue

            # If enough AVAILABLE right now, pick immediately.
            addresses, _pref = self._cluster.get_available(exp.expid)
            if len(addresses) >= num_partitions:
                logger.info(
                    "Selected partitioning %s for experiment %s (num_partitions=%d, available=%d).",
                    pid,
                    exp.expid,
                    num_partitions,
                    len(addresses),
                )
                return pid

            # Otherwise, wait a little for availability if cluster size is sufficient.
            deadline = monotonic() + PARTITIONING_FALLBACK_S
            while monotonic() < deadline and not self._stop.is_set():
                self._cluster.await_available(timeout=AVAILABLE_POLL_S)
                addresses, _pref = self._cluster.get_available(exp.expid)
                if len(addresses) >= num_partitions:
                    logger.info(
                        "Selected partitioning %s for experiment %s after waiting (num_partitions=%d, available=%d).",
                        pid,
                        exp.expid,
                        num_partitions,
                        len(addresses),
                    )
                    return pid

            # Timed out: try next scheme.
            if idx < len(allowed) - 1:
                logger.info(
                    "Not enough AVAILABLE workers for partitioning %s (need=%d). Considering next scheme.",
                    pid,
                    num_partitions,
                )

        # If we get here, none are runnable with the current cluster size.
        raise RuntimeError(
            f"No partitioning scheme for experiment {exp.expid!r} is runnable with current cluster size "
            f"(total_workers={total_workers})."
        )

    # ------------------------------------------------------------------ #
    # Assignment planning (all-at-once, reuse-aware)
    # ------------------------------------------------------------------ #

    def _await_full_assignment_plan(self, *, expid: str, num_partitions: int) -> list[str]:
        """
        Wait until we can assign *all* partitions for one replication.

        Returns:
            ordered_workers: list[str] of length `num_partitions`, where
                ordered_workers[p] is the worker chosen for partition p.

        Reuse-aware placement:
        - Cluster.get_available(expid) returns AVAILABLE workers ordered by preference:
            1) preferred workers already set up for `expid` (distinct partitions),
            2) other AVAILABLE workers.
          The companion `preferred_partitions` list indicates which partition is already
          loaded on each preferred worker.

        Important: this method never returns a partial plan; it waits until all partitions
        can be assigned at once.
        """
        while not self._stop.is_set():
            addresses, preferred_partitions = self._cluster.get_available(expid)

            if len(addresses) < num_partitions:
                # Wait until availability changes.
                self._cluster.await_available(AVAILABLE_POLL_S)
                continue

            preferred_workers = addresses[: len(preferred_partitions)]

            plan: dict[int, str] = {}
            used_workers: set[str] = set()

            # Keep preferred bindings that fit within [0..num_partitions-1].
            for worker, part in zip(preferred_workers, preferred_partitions):
                if 0 <= part < num_partitions and part not in plan:
                    plan[part] = worker
                    used_workers.add(worker)

            # Fill remaining partitions with remaining workers.
            remaining_workers = [w for w in addresses if w not in used_workers]
            for part in range(num_partitions):
                if part in plan:
                    continue
                if not remaining_workers:
                    break
                worker = remaining_workers.pop(0)
                plan[part] = worker
                used_workers.add(worker)

            if len(plan) == num_partitions:
                return [ass[1] for ass in sorted(plan.items(), key=lambda a: a[0])]

            # This should be rare (e.g., preferred had unusable partitions and we ran out of workers).
            self._cluster.await_available(timeout=AVAILABLE_POLL_S)

        raise _StopRequested()

    # ------------------------------------------------------------------ #
    # Launch sequence (separate thread)
    # ------------------------------------------------------------------ #

    def _launch_replication(self, exp: Experiment, repid: str) -> None:
        """
        Launch a replication:
          1) set desired state to READY for all assigned workers
          2) wait until all workers are READY
          3) set desired state to ACTIVE for all assigned workers
        """

        deadline = monotonic() + self._settings.launch_timeout_s

        assignments = {ass.partition: ass.worker for ass in exp.replications[repid].assignments.values()}

        logger.info(
            "Launching replication expid=%s repid=%s partitions=%d",
            exp.expid,
            repid,
            len(assignments),
        )

        # Step 1: request READY for all partitions (worker performs INITIALIZING internally).
        for partition, worker in assignments.items():
            self._cluster.set_desired_state(
                worker_address=worker,
                state=WorkerState.READY,
                expid=exp.expid,
                repid=repid,
                partition=int(partition),
            )

        # Step 2: READY barrier
        timeout = False
        while not self._stop.is_set() and not (timeout := (monotonic() > deadline)):
            states = self._cluster.worker_states
            if all(states.get(w) == WorkerState.READY for w in assignments.values()):
                break
            self._cluster.await_available(timeout=READY_POLL_S)

        if self._stop.is_set():
            # shutdown requested while initializing
            self._mark_replication_failed(
                expid=exp.expid,
                repid=repid,
                exc={"description": "Orchestrator stopped during initialization."}
            )
            logger.warning("Replication expid=%s repid=%s launch failed during initialization "
                           "due to orchestrator shutdown.", exp.expid, repid)
            return

        if timeout:
            # timeout while initializing
            self._mark_replication_failed(
                expid=exp.expid,
                repid=repid,
                exc={"description": "Launch timeout during initialization."}
            )
            logger.warning("Replication expid=%s repid=%s launch failed during initialization "
                           "due to timeout.", exp.expid, repid)
            return

        # Step 3: request ACTIVE for all partitions.
        for partition, worker in assignments.items():
            self._cluster.set_desired_state(
                worker_address=worker,
                state=WorkerState.ACTIVE,
                expid=exp.expid,
                repid=repid,
                partition=int(partition),
            )

        # Step 4: monitor for successful worker start
        while not self._stop.is_set() and not (timeout := (monotonic() > deadline)):
            states = self._cluster.worker_states
            if all(states.get(w) == WorkerState.ACTIVE for w in assignments.values()):
                break
            self._cluster.await_available(timeout=READY_POLL_S)

        if self._stop.is_set():
            # shutdown requested while initializing
            self._mark_replication_failed(
                expid=exp.expid,
                repid=repid,
                exc={"description": "Orchestrator stopped during simulation start."}
            )
            logger.warning("Replication expid=%s repid=%s launch failed during start "
                           "due to orchestrator shutdown.", exp.expid, repid)
            return

        if timeout:
            # timeout while initializing
            self._mark_replication_failed(
                expid=exp.expid,
                repid=repid,
                exc={"description": "Launch timeout during simulation start."}
            )
            logger.warning("Replication expid=%s repid=%s launch failed during start "
                           "due to timeout.", exp.expid, repid)
            return

    def _mark_replication_failed(self, *, expid: str, repid: str, exc: Optional[dict[str, Any]] = None) -> None:
        try:
            if exc:
                self._store.set_replication_exc(
                    expid=expid,
                    repid=repid,
                    exc=exc,
                    fail_replication=True,
                )
            else:
                self._store.set_replication_status(
                    expid=expid,
                    repid=repid,
                    status=ExperimentStatus.FAILED,
                )
        except Exception:
            logger.exception("Failed to mark replication failed expid=%s repid=%s", expid, repid)
