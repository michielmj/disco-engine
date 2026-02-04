# src/disco/worker.py
from __future__ import annotations

import traceback
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from threading import RLock, Event
from time import monotonic
from typing import Dict, Generator, Mapping, Optional, cast, Any

from sqlalchemy.exc import OperationalError, DBAPIError
from tools.mp_logging import getLogger
from data_logger import DataLogger

from .database import SessionManager
from .exceptions import DiscoError, DiscoRuntimeError
from .cluster import Cluster, WorkerState, DesiredWorkerState
from .config import AppSettings
from .envelopes import EventEnvelope, PromiseEnvelope
from .partitioning import Partitioning, NodeInstanceSpec, PartitioningNotFoundError, PartitioningCorruptError
from .transports.grpc_ingress import start_grpc_server
from .transports.grpc_transport import GrpcTransport
from .transports.inprocess import InProcessTransport
from .transports.ipc_egress import IPCTransport
from .transports.ipc_messages import IPCEventMsg, IPCPromiseMsg

from .experiments import Experiment, ExperimentStore, ExperimentStatus
from .model import Model, load_model
from .graph import Graph, load_graph_for_scenario
from .graph_data import GraphData
from .runtime import NodeRuntime
from .router import Router

logger = getLogger(__name__)


class WorkerError(DiscoError):
    """
    Worker-level errors.

    Conceptually:
    - WorkerError is used for “control-plane” problems: invalid transitions,
      missing assignments, internal invariants violated.
    - “Data/model” problems should usually be captured and turned into partition failures
      (via ExperimentStore.set_partition_exc), not turned into BROKEN unless they prevent
      *any* experiment from being started.
    """
    pass


@dataclass(slots=True)
class Assignment:
    """
    The current (expid, repid, partition) assigned to this worker.

    Lifecycle:
    - Set when we receive desired state READY with a valid assignment.
    - Cleared when the run ends (FINISHED / CANCELED / FAILED) or when we go AVAILABLE.
    - Used for:
        * loading Experiment, Graph, Partitioning
        * reporting partition status/exceptions via ExperimentStore
        * reporting WorkerInfo to Cluster
    """
    expid: str
    repid: str
    partition: int


class Worker:
    """
    Long-lived simulation worker process.

    High-level responsibilities
    ---------------------------
    1) Control plane:
       - Receives desired-state changes from Cluster (via callback thread).
       - Applies state transitions in the *runner thread* (main thread).
       - Publishes worker state + assignment back to Cluster.

    2) Data plane:
       - Receives ingress messages (IPC + gRPC -> IPC queues).
       - Delivers Promise/Event envelopes into NodeRuntime objects.
       - Steps NodeRuntime runners at high frequency (hot path).

    Threading model
    ---------------
    - Runner loop executes in the main thread.
    - Cluster desired-state callback runs on a Kazoo/Metastore callback thread.
    - IPC queues are thread-safe; we intentionally drain them without holding the lock.
    - Shared mutable state between threads is protected by self._lock:
        * self._pending_desired
        * self._running, self._state (though _state is *effectively* runner-owned)
        * transitions, assignment changes, metastore writes
    - self._kick is a wakeup mechanism:
        * Set by callback thread to notify runner “control-plane work is pending”.
        * Also set when request_stop is called, to exit promptly.
        * NOT set for ingress messages (by design): ingress delivery in READY/PAUSED
          can wait until next tick; ACTIVE drains continuously anyway.

    Core lifecycle (one partition run)
    ---------------------------------
    - Desired READY(expid, repid, partition):
        * assignment set
        * INITIALIZING
        * _setup_run_locked(): load experiment/graph/partitioning/model, build nodes
        * set ExperimentStatus.LOADED then INITIALIZED
        * state -> READY
    - Desired ACTIVE:
        * _start_runners_locked(): create runner generators once
        * set ExperimentStatus.ACTIVE
        * state -> ACTIVE
    - ACTIVE hot loop:
        * drain ingress (no lock)
        * step runners (no lock, but fail partition under lock on exception)
        * when all runners finished: end run -> FINISHED, teardown, AVAILABLE
    - Desired PAUSED:
        * state -> PAUSED
        * set ExperimentStatus.PAUSED
        * runner stepping stops (but messages may still be drained)
    - Desired TERMINATED:
        * set ExperimentStatus.CANCELED (when appropriate)
        * teardown + AVAILABLE

    “Fail partition vs break worker” rule of thumb
    ----------------------------------------------
    - Fail partition (keep worker healthy) when:
        * experiment data is inconsistent
        * scenario/graph data is wrong/missing
        * partitioning is corrupt/missing
        * model runtime logic throws during initialize or during next(gen)
    - Break worker when:
        * worker cannot reliably start/operate runs anymore:
            - cannot load fixed model package
            - cannot talk to metastore/cluster to report status/exc
            - cannot access DB due to connectivity/auth issues
            - internal invariants violated that would corrupt future runs
    """

    def __init__(
            self,
            address: str,
            cluster: Cluster,
            event_queues: Mapping[str, Queue[IPCEventMsg]],
            promise_queues: Mapping[str, Queue[IPCPromiseMsg]],
            settings: AppSettings,
            name: Optional[str] = None,
    ) -> None:
        # Identity
        self.address = address
        self._cluster = cluster
        self._name = name or address
        self._settings: AppSettings = settings

        # ExperimentStore is the worker’s “write channel” for partition status/exc.
        # The worker is responsible for advancing ExperimentStatus after ASSIGNED.
        self._exp_store = ExperimentStore(self._cluster.meta)

        # Global IPC queue maps (all workers). We select our own ingress queues from these.
        self._event_queues = event_queues
        self._promise_queues = promise_queues

        # Our own ingress queues must exist in the mappings.
        ingress_event_q = self._event_queues.get(self.address)
        ingress_promise_q = self._promise_queues.get(self.address)
        if ingress_event_q is None or ingress_promise_q is None:
            raise WorkerError(f"No IPC queues configured for worker address {self.address!r}")

        self._ingress_event_queue: Queue[IPCEventMsg] = ingress_event_q
        self._ingress_promise_queue: Queue[IPCPromiseMsg] = ingress_promise_q

        # Synchronization:
        # - _lock protects control-plane state shared with callback thread.
        # - _kick is a low-overhead wakeup signal for the runner to process control-plane work.
        self._lock = RLock()
        self._kick = Event()

        # Worker process state machine. Runner thread is the only mutator of _state.
        self._state: WorkerState = WorkerState.CREATED

        # Current assignment (None when AVAILABLE/CREATED/etc.).
        self._assignment: Assignment | None = None

        # NodeRuntime registry used by InProcessTransport (holds reference to dict).
        # Keep dict identity stable; mutate contents only.
        self._nodes: Dict[str, NodeRuntime] = {}

        # Router and data logger (exist for worker lifetime, but dlogger is per-run).
        self._router: Router
        self._dlogger: DataLogger

        # Cached experiment and expid (to avoid reloading between replications of same exp).
        self._experiment: Experiment | None = None
        self._experiment_expid: str | None = None

        # Cached graph and scenario id (graph tied to scenario).
        self._graph: Graph | None = None
        self._graph_scenario_id: str | None = None

        # Cached partitioning and selection (partitioning tied to graph + selected_partitioning + partition index).
        self._partitioning: Partitioning | None = None
        self._node_specs: Dict[str, NodeInstanceSpec] | None = None
        self._partitioning_id: str | None = None

        # Cached model (fixed plugin/package; loaded once per worker).
        self._model: Model | None = None

        # Runners:
        # - self._runners is immutable during ACTIVE (tuple of generator objects).
        # - self._active_runners is a list of indices into self._runners for runners not finished yet.
        #   This avoids dict lookups and avoids rebuilding tuples during the hot path.
        self._runners: tuple[Generator[object, None, None], ...] = ()
        self._active_runners: list[int] = []

        # Desired-state handoff from callback thread -> runner thread.
        self._pending_desired: DesiredWorkerState | None = None

        # Runner loop enable flag.
        self._running: bool = False

        # DB session manager for loading graph/model-related DB data.
        self._session_manager = SessionManager.from_settings(settings.database)

        # Register worker + install desired-state watch.
        self._register_with_cluster()

        # Build router transports:
        # - InProcessTransport uses _nodes for same-process delivery.
        # - IPCTransport uses queue maps to route between processes.
        # - GrpcTransport supports external clients.
        inproc = InProcessTransport(nodes=self._nodes)
        ipc = IPCTransport(
            cluster=self._cluster,
            event_queues=self._event_queues,
            promise_queues=self._promise_queues,
        )
        grpc = GrpcTransport(cluster=self._cluster, settings=self._settings.grpc)
        self._router = Router(transports=[inproc, ipc, grpc])

        # Start gRPC server; it enqueues IPCEventMsg/IPCPromiseMsg into our ingress queues.
        self._grpc_server = start_grpc_server(
            worker=self,
            event_queue=self._ingress_event_queue,
            promise_queue=self._ingress_promise_queue,
            settings=self._settings.grpc,
        )

        logger.info("Worker %s created with initial state %s", self._name, self._state)

    # ------------------------------------------------------------------ #
    # Runner entrypoint (main thread of worker process)
    # ------------------------------------------------------------------ #

    def run_forever(self) -> WorkerState:
        """
        Entry point called by the worker process main.

        - Sets _running True once (guarded by lock).
        - Runs the runner loop until stop/broken/exit.
        - Returns final WorkerState (acquired under lock to read consistent value).
        """
        with self._lock:
            if self._running:
                raise WorkerError("Worker runner already running")
            self._running = True

        self._runner_loop()

        with self._lock:
            return self._state

    def request_stop(self) -> None:
        """
        Asynchronous stop request (can be called from other thread or signal handler).

        - Sets _running False under lock so runner observes it.
        - Sets _kick so runner wakes promptly even if it is waiting.
        """
        with self._lock:
            self._running = False
        self._kick.set()

    @property
    def state(self) -> WorkerState:
        """
        Exposes current worker state. Intended mainly for diagnostics.
        """
        return self._state

    # ------------------------------------------------------------------ #
    # Cluster / registration
    # ------------------------------------------------------------------ #

    def _register_with_cluster(self) -> None:
        """
        Register worker with the cluster (control plane):
        - Announces existence and initial state.
        - Installs desired-state watch so controller can steer this worker.
        """
        self._cluster.register_worker(self.address, state=self._state)
        self._cluster.on_desired_state_change(self.address, self._on_desired_state_change)

    def _safe_unregister(self) -> None:
        """
        Best-effort unregister. Used when runner loop exits.

        We do not want errors here to mask the root cause that ended the loop.
        """
        try:
            self._cluster.unregister_worker(self.address)
        except Exception as exc:
            logger.warning("Worker %s unregister failed: %r", self._name, exc)

    # ------------------------------------------------------------------ #
    # Desired state handling (Metastore/Kazoo callback thread)
    # ------------------------------------------------------------------ #

    def _on_desired_state_change(self, desired: DesiredWorkerState) -> str | None:
        """
        Callback invoked by Cluster/Metastore watcher thread.

        IMPORTANT:
        - Must not perform heavy work or block.
        - Must not mutate runtime structures directly.
        - Only stores desired state in _pending_desired and “kicks” the runner.

        Return value:
        - None if accepted.
        - Error string if worker is BROKEN and cannot accept new desired states.
        """
        with self._lock:
            if self._state == WorkerState.BROKEN:
                msg = f"Worker {self._name} is BROKEN; cannot process desired-state {desired.request_id}"
                logger.error(msg)
                return msg

            self._pending_desired = desired
            logger.debug(
                "Worker %s received desired-state %s -> target=%s expid=%s repid=%s partition=%s nodes=%s",
                self._name,
                desired.request_id,
                desired.state,
                desired.expid,
                desired.repid,
                desired.partition,
                getattr(desired, "nodes", None),
            )

        # Wake runner thread without holding lock.
        # This is control-plane only; we intentionally do not kick on ingress messages.
        self._kick.set()
        return None

    # ------------------------------------------------------------------ #
    # Runner loop (single-threaded execution in current thread)
    # ------------------------------------------------------------------ #

    def _step_runners_once(self) -> None:
        """
        Hot path: step all currently-active runners exactly once.

        Invariants:
        - Called only in ACTIVE state.
        - self._runners is immutable during ACTIVE.
        - self._active_runners contains indices into self._runners that are not finished yet.
        - No locks are held during stepping to minimize overhead.

        Failure handling:
        - StopIteration: runner completed -> remove it from _active_runners.
        - Any other exception during next(gen):
            * treated as model/runtime “grey zone” failure
            * fail the partition (under lock) and return immediately
            * failing the partition triggers teardown + AVAILABLE, so hot path stops naturally
        """
        runners = self._runners
        active = self._active_runners
        if not active:
            return

        finished_positions: list[int] = []

        # Iterate by position so we can pop later without searching.
        for pos in range(len(active)):
            idx = active[pos]
            try:
                next(runners[idx])
            except StopIteration:
                finished_positions.append(pos)
            except Exception as exc:
                # “Grey zone” rule: any exception during next(gen) fails the partition.
                with self._lock:
                    self._fail_partition_locked(exc, where=f"runner next() failed for runner_index={idx}")
                return

        if finished_positions:
            # Remove finished indices from active in reverse order (safe pops).
            for pos in reversed(finished_positions):
                active.pop(pos)

    def _end_run_locked(self, status: ExperimentStatus) -> None:
        """
        End-of-run sequence for a partition.

        This is the “happy path”/structured path end (e.g. FINISHED),
        distinct from _fail_partition_locked which ends via exception.

        Steps:
        1) persist final partition status (ExperimentStore)
        2) release per-run resources (NodeRuntimes, DataLogger, ingress queues)
        3) clear assignment + publish AVAILABLE to Cluster
        4) clear runner bookkeeping
        """
        # 1) status + bookkeeping
        self._set_partition_status_locked(status)

        # 2) cleanup resources
        self._teardown_run_locked()

        # 3) worker lifecycle bookkeeping
        self._assignment = None
        self._update_worker_info_locked()
        self._set_state_locked(WorkerState.AVAILABLE)

        # 4) clear runner bookkeeping
        self._runners = ()
        self._active_runners = []

    def _runner_loop(self) -> None:
        """
        Main loop executed by the worker process (main thread).

        Design goal:
        - Keep ACTIVE path fast:
            * drain ingress (no lock)
            * step runners (no lock)
        - Control plane work (desired state transitions) is processed:
            * at most once per second (1 Hz tick)
            * or earlier if _kick is set (desired-state change or stop request)

        Time structure:
        - “Control tick” evaluates desired state and transitions the worker state machine.
        - “Hot path” advances simulation and handles completion.

        Exit conditions:
        - _running becomes False (request_stop or EXITED)
        - worker transitions to BROKEN and sets _running False elsewhere (not shown here)
        """
        next_control = monotonic()
        control_period_s = 1.0

        try:
            while True:
                # --- control tick (<= 1 Hz, or earlier if kicked) ---
                now = monotonic()
                if now >= next_control or self._kick.is_set():
                    self._kick.clear()
                    with self._lock:
                        if not self._running:
                            break

                        # Apply desired-state transitions (READY/ACTIVE/PAUSED/TERMINATED/EXITED).
                        self._apply_pending_desired_locked()
                        state = self._state

                    next_control = now + control_period_s
                else:
                    # Fast read without lock:
                    # - runner thread is sole mutator of _state in normal operation
                    # - callback thread only sets _pending_desired + kicks
                    state = self._state
                    if not self._running:
                        break

                # --- hot paths ---
                if state == WorkerState.ACTIVE:
                    # Deliver messages frequently (no lock).
                    self._drain_ingress()

                    # Step all active runners once (no lock).
                    self._step_runners_once()

                    # If all runners finished, end the run as FINISHED.
                    if not self._active_runners:
                        with self._lock:
                            self._end_run_locked(ExperimentStatus.FINISHED)

                    continue

                if state in (WorkerState.READY, WorkerState.PAUSED):
                    # Design note:
                    # - Delivering messages here is not critical, but it is harmless and may reduce
                    #   backlog when switching back to ACTIVE. If you want strictly “no overhead”
                    #   outside ACTIVE, you could move this drain into ACTIVE only.
                    self._drain_ingress()

                    # READY: no runners should exist (created only in READY->ACTIVE).
                    # PAUSED: runners exist but we do not step them.
                    timeout = max(0.0, next_control - monotonic())
                    self._kick.wait(timeout=timeout)
                    continue

                # Any other state: sleep until next tick or kick (control-plane only).
                timeout = max(0.0, next_control - monotonic())
                self._kick.wait(timeout=timeout)

        finally:
            logger.info("Worker %s runner loop exited.", self._name)
            self._safe_unregister()

    # ------------------------------------------------------------------ #
    # Ingress draining (runner thread only)
    # ------------------------------------------------------------------ #

    def _drain_ingress(self) -> None:
        """
        Drain ingress queues without locks.

        Rationale:
        - IPC queues are thread-safe.
        - Draining is on the data-plane hot path and should avoid lock overhead.
        - Delivery into NodeRuntime is not lock-protected; NodeRuntime is assumed
          to be runner-thread-owned.
        """
        # Promises first: they typically unblock simprocs and prevent stalling.
        while True:
            try:
                msg = self._ingress_promise_queue.get_nowait()
            except Empty:
                break
            self._deliver_ingress_promise(msg)

        # Then events: actual simulation events.
        while True:
            try:
                msg = self._ingress_event_queue.get_nowait()
            except Empty:
                break
            self._deliver_ingress_event(msg)

    def _deliver_ingress_event(self, msg: IPCEventMsg) -> None:
        """
        Convert IPCEventMsg -> EventEnvelope and deliver to target NodeRuntime.

        Error policy:
        - Unknown target node / malformed payload: worker BROKEN (routing invariant violated).
        - NodeRuntime.receive_event exception: worker BROKEN (runtime invariants violated).

        Note: These errors are treated as worker-fatal because they indicate corruption
        or a fundamental mismatch in the worker’s runtime setup vs messages being routed.
        """
        node = self._nodes.get(msg.target_node)
        if node is None:
            with self._lock:
                self._transition_to_broken_locked(reason=f"Received event for unknown node {msg.target_node!r}")
            return

        if msg.shm_name is not None:
            with self._lock:
                self._transition_to_broken_locked(
                    reason=f"Unexpected shared-memory event payload for node {msg.target_node!r}"
                )
            return

        if msg.data is None:
            with self._lock:
                self._transition_to_broken_locked(reason=f"IPCEventMsg missing payload for node {msg.target_node!r}")
            return

        envelope = EventEnvelope(
            repid=msg.repid,
            sender_node=msg.sender_node,
            sender_simproc=msg.sender_simproc,
            target_node=msg.target_node,
            target_simproc=msg.target_simproc,
            epoch=msg.epoch,
            data=msg.data,
            headers=msg.headers,
        )

        try:
            node.receive_event(envelope)
        except Exception as exc:
            with self._lock:
                self._transition_to_broken_locked(
                    reason=f"NodeRuntime.receive_event failed for node {msg.target_node!r}: {exc!r}"
                )

    def _deliver_ingress_promise(self, msg: IPCPromiseMsg) -> None:
        """
        Convert IPCPromiseMsg -> PromiseEnvelope and deliver to target NodeRuntime.

        Same policy as events: failures here indicate corrupted routing/invariants.
        """
        node = self._nodes.get(msg.target_node)
        if node is None:
            with self._lock:
                self._transition_to_broken_locked(reason=f"Received promise for unknown node {msg.target_node!r}")
                return

        envelope = PromiseEnvelope(
            repid=msg.repid,
            sender_node=msg.sender_node,
            sender_simproc=msg.sender_simproc,
            target_node=msg.target_node,
            target_simproc=msg.target_simproc,
            seqnr=msg.seqnr,
            epoch=msg.epoch,
            num_events=msg.num_events,
        )

        try:
            node.receive_promise(envelope)
        except Exception as exc:
            with self._lock:
                self._transition_to_broken_locked(
                    reason=f"NodeRuntime.receive_promise failed for node {msg.target_node!r}: {exc!r}"
                )

    def _clear_ingress_queues_locked(self) -> None:
        """
        Best-effort drain of ingress queues during teardown.

        Used to prevent:
        - stale messages from prior runs being delivered into the next run’s NodeRuntimes.
        - memory growth if controller routed messages late.
        """
        while True:
            try:
                self._ingress_promise_queue.get_nowait()
            except Empty:
                break
        while True:
            try:
                self._ingress_event_queue.get_nowait()
            except Empty:
                break

    # ------------------------------------------------------------------ #
    # Desired-state application (runner thread only)
    # ------------------------------------------------------------------ #

    def _apply_pending_desired_locked(self) -> None:
        """
        Apply any pending desired-state change captured by callback thread.

        Invariants:
        - Called only by runner thread under lock.
        - Consumes at most one pending desired state per tick.
          (If multiple arrive quickly, the last one wins because _pending_desired is overwritten.)

        Error handling:
        - Unexpected exceptions are treated as worker-fatal -> BROKEN.
        """
        if self._pending_desired is None:
            return

        desired = self._pending_desired
        self._pending_desired = None

        try:
            self._apply_desired_locked(desired)
        except Exception as exc:
            logger.exception(
                "Worker %s failed to apply desired-state %s: %r",
                self._name,
                desired.request_id,
                exc,
            )
            self._transition_to_broken_locked(reason=f"failed to apply desired-state {desired.request_id}: {exc!r}")

    def _apply_desired_locked(self, desired: DesiredWorkerState) -> None:
        """
        State machine transitions driven by DesiredWorkerState.

        This method is the “control-plane brain”:
        - validates desired state
        - updates assignment (only on READY)
        - calls setup/start/teardown helpers
        - reports worker state to Cluster
        - advances ExperimentStatus via ExperimentStore

        Important invariants:
        - Assignment is set ONLY on READY.
        - Runners are created ONLY on READY -> ACTIVE.
        - ACTIVE hot path does not call control-plane helpers.
        """
        if not desired.validate_assignment():
            raise ValueError(f"Invalid or incomplete assignment expid={desired.expid} repid={desired.repid} "
                             f"partition={desired.partition}.")
        if not desired.validate_state():
            raise ValueError(f"Invalid desired state {desired.state.name}")

        target = desired.state

        new_assignment: Assignment | None
        if desired.expid is None:
            new_assignment = None
        else:
            new_assignment = Assignment(
                expid=cast(str, desired.expid),
                repid=cast(str, desired.repid),
                partition=cast(int, desired.partition),
            )

        logger.info(
            "Worker %s applying desired-state %s: target=%s assignment=%s current=%s",
            self._name,
            desired.request_id,
            target.name,
            new_assignment,
            self._state.name,
        )

        # EXITED: stop the worker process.
        if target == WorkerState.EXITED:
            self._teardown_run_locked()
            self._set_state_locked(WorkerState.EXITED)
            self._running = False
            self._kick.set()
            return

        # READY: load everything needed to later run ACTIVE.
        if target == WorkerState.READY:
            if self._state not in (WorkerState.CREATED, WorkerState.AVAILABLE, WorkerState.TERMINATED):
                raise WorkerError(f"Cannot transition to READY from {self._state.name}")

            if new_assignment is None:
                raise WorkerError("READY requires an assignment")

            self._assignment = new_assignment
            self._set_state_locked(WorkerState.INITIALIZING)

            ok = self._setup_run_locked()
            if not ok:
                # Setup failed but worker remains healthy; partition failure already reported.
                return

            self._update_worker_info_locked()
            # Experiment status updates handled by _setup_run_locked
            self._set_state_locked(WorkerState.READY)
            return

        # ACTIVE: create runners once, then the hot path takes over.
        if target == WorkerState.ACTIVE:
            if self._state not in (WorkerState.READY, WorkerState.PAUSED):
                raise WorkerError(f"Cannot transition to ACTIVE from {self._state.name}")

            if new_assignment is not None:
                raise WorkerError("Assignment can only be set for desired state READY")

            if self._state == WorkerState.READY:
                self._start_runners_locked()

            self._update_worker_info_locked()
            self._set_partition_status_locked(ExperimentStatus.ACTIVE)
            self._set_state_locked(WorkerState.ACTIVE)
            return

        # PAUSED: stop stepping runners (but do not tear down).
        if target == WorkerState.PAUSED:
            if self._state != WorkerState.ACTIVE and self._state != WorkerState.PAUSED:
                raise WorkerError(f"Cannot transition to PAUSED from {self._state.name}")
            self._set_state_locked(WorkerState.PAUSED)
            self._set_partition_status_locked(ExperimentStatus.PAUSED)
            return

        # TERMINATED: cancel and teardown the current run (if any), return to AVAILABLE.
        if target == WorkerState.TERMINATED:
            if self._state not in (WorkerState.ACTIVE, WorkerState.PAUSED, WorkerState.READY):
                raise WorkerError(f"Cannot terminate from {self._state.name}")

            if self._state != WorkerState.READY:
                self._set_partition_status_locked(ExperimentStatus.CANCELED)

            self._set_state_locked(WorkerState.TERMINATED)
            self._teardown_run_locked()
            self._assignment = None
            self._update_worker_info_locked()
            self._set_state_locked(WorkerState.AVAILABLE)
            return

        raise WorkerError(f"Unsupported target state: {target}")

    # ------------------------------------------------------------------ #
    # Run setup/teardown (runner thread only)
    # ------------------------------------------------------------------ #

    def _setup_run_locked(self) -> bool:
        """
        Prepare everything needed to execute a partition run (READY initialization).

        What this does:
        - Ensures no previous run resources remain (_teardown_run_locked).
        - Loads Experiment (metastore).
        - Loads Graph for scenario (DB).
        - Loads Partitioning for selected partitioning ID + partition index (metastore).
        - Loads Model once per worker (fixed module/package).
        - Creates DataLogger for this (expid, repid, partition).
        - Builds NodeRuntimes for all nodes in partition and runs initialize(**params).
        - Advances ExperimentStatus:
            * LOADED: once prerequisites (experiment/graph/partitioning/model) exist
            * INITIALIZED: once NodeRuntimes are successfully initialized

        Return semantics:
        - True: setup OK; worker will go READY.
        - False: partition failed but worker is still healthy (worker stays AVAILABLE).
          The partition failure is reported via ExperimentStore.set_partition_exc.
        - Uncaught exceptions: assumed worker-fatal (-> caller will break worker).
        """

        self._teardown_run_locked()
        assignment = self._require_assignment_locked()

        # --- Experiment ---
        try:
            if self._experiment is None or self._experiment_expid != assignment.expid:
                self._experiment = self._exp_store.load(assignment.expid)
                self._experiment_expid = assignment.expid
            experiment = self._require_experiment_locked()
        except (KeyError, TypeError, ValueError, DiscoRuntimeError) as exc:
            self._fail_partition_locked(exc, where="load experiment")
            return False

        # --- Graph (scenario change) ---
        scenario_id = experiment.scenario_id
        if self._graph is None or self._graph_scenario_id != scenario_id:
            try:
                with self._session_manager.session as session:
                    self._graph = load_graph_for_scenario(session, scenario_id)
                self._graph_scenario_id = scenario_id
                # graph change invalidates partitioning cache
                self._partitioning = None
                self._partitioning_id = None
            except (OperationalError, DBAPIError):
                # DB connectivity/auth is considered worker-fatal: cannot start experiments reliably.
                raise
            except (KeyError, TypeError, ValueError, DiscoRuntimeError) as exc:
                # scenario data is missing/corrupt: fail partition (worker can still run other scenarios).
                self._fail_partition_locked(exc, where="load graph")
                return False

        graph = self._graph

        # --- Partitioning ---
        if experiment.selected_partitioning is None:
            self._fail_partition_locked(
                WorkerError("Experiment has no selected_partitioning"),
                where="select partitioning",
            )
            return False

        try:
            if (
                    self._partitioning is None
                    or self._partitioning_id != experiment.selected_partitioning
            ):
                self._partitioning = Partitioning.load(
                    metastore=self._cluster.meta,
                    partitioning_id=experiment.selected_partitioning,
                    graph=graph,
                )
                self._node_specs = {ns.node_name: ns for ns in self._partitioning.node_specs}
                self._partitioning_id = experiment.selected_partitioning
        except (
                PartitioningNotFoundError, PartitioningCorruptError, KeyError, TypeError, ValueError,
                DiscoRuntimeError) as exc:
            self._fail_partition_locked(exc, where="load partitioning")
            return False

        partitioning = self._partitioning
        node_specs = self._node_specs

        # --- Model (load once per worker) ---
        # If model cannot load, that is typically worker-fatal because model is a fixed package.
        if self._model is None:
            ms = self._settings.model
            self._model = load_model(
                plugin=ms.plugin,
                package=ms.package,
                path=ms.path,
                db=self._session_manager,
                dev_import_root=ms.dev_import_root,
                model_yml=ms.model_yml
            )
        model = self._model

        # At this point: prerequisites exist (and are consistent enough to attempt init).
        self._set_partition_status_locked(ExperimentStatus.LOADED)

        # --- Data logger ---
        # Failure here is treated as worker-fatal (no catch) because it undermines observability and persistence.
        dl = self._settings.data_logger
        dl_path = Path(dl.path) / str(assignment.expid) / str(assignment.repid) / str(assignment.partition)
        self._dlogger = DataLogger(
            segments_dir=dl_path,
            ring_bytes=dl.ring_bytes,
            rotate_bytes=dl.rotate_bytes,
            zstd_level=dl.zstd_level,
        )
        dlogger = self._dlogger

        # --- NodeRuntimes + initialize ---
        seed_sequence = experiment.replications[assignment.repid].get_seed_sequence(assignment.partition)
        params = experiment.params or {}

        logger.info(
            "Worker %s setting up run for expid=%s repid=%s partition=%s nodes=%s scenario=%s partitioning_id=%s",
            self._name,
            assignment.expid,
            assignment.repid,
            assignment.partition,
            list(node_specs),
            experiment.scenario_id,
            experiment.selected_partitioning,
        )

        for node_name, spec in node_specs.items():
            try:
                node_mask = partitioning.assignment_vector(node_name)
                graph_view = graph.get_view(mask=node_mask)

                graph_data = GraphData.for_node(
                    session_manager=self._session_manager,
                    graph=graph_view,
                    model=model,
                    spec=spec,
                )

                rt = NodeRuntime(
                    repid=assignment.repid,
                    spec=spec,
                    model=model,
                    partitioning=partitioning,
                    router=self._router,
                    dlogger=dlogger,
                    seed_sequence=seed_sequence,
                    graph=graph_view,
                    data=graph_data,
                )
                rt.initialize(**params)
                self._nodes[node_name] = rt
            except Exception as exc:
                # Any failure in init is treated as partition-failure (grey-zone rule).
                self._fail_partition_locked(exc, where=f"initialize node={node_name}")
                return False

        # Clear runners; created only when transitioning READY -> ACTIVE.
        self._runners = ()
        self._set_partition_status_locked(ExperimentStatus.INITIALIZED)
        return True

    def _start_runners_locked(self) -> None:
        """
        Create runner generators exactly once when transitioning READY -> ACTIVE.

        Why:
        - Avoid checking/creating runners in the ACTIVE hot path.
        - Keep runner container immutable during ACTIVE for faster stepping and simpler invariants.

        Side effect:
        - Sets ExperimentStatus.ACTIVE here (some call sites also set it; you may want only one).
        """
        if self._runners:
            return

        exp = self._require_experiment_locked()
        duration = exp.duration

        # Stable order = insertion order of dict (Python 3.7+).
        # We intentionally do not keep node names for hot-path speed;
        # failures are reported by runner index.
        runners = [rt.runner(duration=duration) for rt in self._nodes.values()]
        self._runners = tuple(runners)
        self._active_runners = list(range(len(self._runners)))

        self._set_partition_status_locked(ExperimentStatus.ACTIVE)

    def _teardown_run_locked(self) -> None:
        """
        Tear down all per-run resources.

        Called from:
        - setup (to ensure clean slate)
        - end run (FINISHED/CANCELED/FAILED paths)
        - EXITED/TERMINATED transitions

        What it clears:
        - runner bookkeeping (_runners and _active_runners)
        - DataLogger (flush/close)
        - NodeRuntimes (drop references)
        - ingress queues (drop stale messages)
        """
        self._runners = ()
        self._active_runners = []

        if self._dlogger:
            logger.info("Worker %s flushing data logger.", self._name)
            self._dlogger.close()
            self._dlogger = None

        if self._nodes:
            logger.info("Worker %s tearing down %d NodeRuntimes.", self._name, len(self._nodes))
            self._nodes.clear()

        self._clear_ingress_queues_locked()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _require_assignment_locked(self) -> Assignment:
        """
        Ensure an assignment exists.
        Used in any path that must report partition status/exc or load per-run resources.
        """
        if self._assignment is None:
            raise WorkerError("Worker has no assignment")
        return self._assignment

    def _require_experiment_locked(self) -> Experiment:
        """
        Ensure Experiment is loaded in memory.
        Used during setup and when starting runners (duration, params, etc.).
        """
        if self._experiment is None:
            raise WorkerError("Experiment not loaded")
        return self._experiment

    # ------------------------------------------------------------------ #
    # Cluster state helpers / BROKEN transition
    # ------------------------------------------------------------------ #

    def _update_worker_info_locked(self) -> None:
        """
        Publish (expid, repid, partition) to cluster WorkerInfo.

        Called after:
        - assignment set (READY)
        - assignment cleared (AVAILABLE)
        - transitions where controller needs to see current attachment
        """
        expid: str | None = None
        repid: str | None = None
        partition: int | None = None

        if self._assignment is not None:
            expid = self._assignment.expid
            repid = self._assignment.repid
            partition = self._assignment.partition

        try:
            self._cluster.update_worker_info(
                worker=self.address,
                partition=partition,
                expid=expid,
                repid=repid,
            )
        except Exception as exc:
            # If we cannot report worker info, control plane becomes unreliable.
            raise WorkerError(f"Failed to update WorkerInfo in Cluster: {exc!r}") from exc

    def _set_state_locked(self, new_state: WorkerState) -> None:
        """
        Transition worker state and publish it to Cluster.

        - Only runner thread calls this, under lock.
        - If publish fails, we mark worker BROKEN because controller cannot trust the worker state.
        """
        if new_state == self._state:
            return

        logger.info("Worker %s state transition: %s -> %s", self._name, self._state.name, new_state.name)
        self._state = new_state

        try:
            self._cluster.set_worker_state(self.address, new_state)
        except Exception as exc:
            logger.exception("Worker %s failed to publish state %s to Cluster: %r", self._name, new_state.name, exc)
            self._state = WorkerState.BROKEN

    def _set_partition_status_locked(self, status: ExperimentStatus) -> None:
        """
        Advance partition ExperimentStatus via ExperimentStore.

        This is the canonical “status update” path. If this fails, the worker is considered BROKEN
        because it cannot reliably report progress or completion.
        """
        a = self._require_assignment_locked()
        try:
            self._experiment = self._exp_store.set_partition_status(
                expid=a.expid,
                repid=a.repid,
                partition=a.partition,
                status=status,
            )
            self._experiment_expid = a.expid
        except Exception as exc:
            self._transition_to_broken_locked(
                reason=f"failed to update partition status in metastore: {exc!r}"
            )

    def _fail_partition_locked(self, exc: BaseException, *, where: str) -> None:
        """
        Fail the currently assigned partition and return worker to AVAILABLE.

        This is the “partition failure” path (not worker-fatal) and is used for:
        - invalid/missing experiment data
        - invalid/missing/corrupt graph/partitioning/model data
        - runtime exceptions from model logic in next(gen) / initialize / etc.

        What it does:
        1) Creates a structured exc payload (dict[str, Any]) including traceback snippet
        2) Writes it via ExperimentStore.set_partition_exc(..., fail_partition=True)
        3) Tears down run resources
        4) Clears assignment and transitions worker back to AVAILABLE
        """
        a = self._require_assignment_locked()

        tb = traceback.format_exc()
        payload: dict[str, Any] = {
            "where": where,
            "type": type(exc).__name__,
            "message": repr(exc),
            "traceback": tb[-40_000:],
        }

        try:
            self._experiment = self._exp_store.set_partition_exc(
                expid=a.expid,
                repid=a.repid,
                partition=a.partition,
                exc=payload,
                fail_partition=True,
            )
            self._experiment_expid = a.expid
        except Exception as meta_exc:
            # If we cannot report the failure, control-plane consistency is lost -> BROKEN.
            self._transition_to_broken_locked(
                reason=f"failed to report partition failure to metastore: {meta_exc!r}"
            )
            return

        # Cleanly stop this run and go back to AVAILABLE.
        self._teardown_run_locked()
        self._assignment = None
        self._update_worker_info_locked()
        self._set_state_locked(WorkerState.AVAILABLE)

    def _transition_to_broken_locked(self, reason: str) -> None:
        """
        Mark the worker BROKEN.

        This should only happen for worker-fatal situations, where it is not safe to keep running.
        Examples:
        - cannot talk to cluster/metastore
        - internal invariants violated
        - routing corruption detected (unknown node, invalid message shape)
        - persistent infrastructure failure

        Side effects:
        - Sets worker state to BROKEN in Cluster (best effort)
        - Sets _running False so runner loop exits
        """
        logger.error("Worker %s entering BROKEN state: %s", self._name, reason)
        self._state = WorkerState.BROKEN
        try:
            self._cluster.set_worker_state(self.address, WorkerState.BROKEN)
        except Exception as exc:
            logger.exception("Worker %s failed to publish BROKEN state to Cluster: %r", self._name, exc)

        self._running = False
