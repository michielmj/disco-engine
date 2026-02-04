# src/disco/client.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Mapping

from tools.mp_logging import getLogger

from disco.cluster import Cluster, WorkerState, WorkerInfo
from disco.config import AppSettings, get_settings
from disco.experiments import Experiment, ExperimentStore, ExperimentStatus

logger = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """
    Convenience snapshot of a worker's current state and info.

    Note:
        - `state` is derived from Cluster's in-memory watch cache.
        - `info` is loaded from Metastore via Cluster.get_worker_info() and may raise
          if the worker disappears between reads.
    """
    address: str
    state: WorkerState
    info: WorkerInfo | None


class Client:
    """
    User-facing API to interact with a Disco cluster.

    Responsibilities:
      - Submit experiments / replications via ExperimentStore (no direct Metastore usage).
      - Inspect cluster workers via Cluster.
      - Inspect experiment status via ExperimentStore.
    """

    def __init__(self, *, cluster: Cluster) -> None:
        """
        Create a Client bound to an existing Cluster instance.

        Args:
            cluster: A Cluster instance (typically created via Client.make()).
        """
        self.cluster = cluster
        self.store = ExperimentStore(cluster.meta)

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    @contextmanager
    def make(
        cls,
        *,
        settings: AppSettings | None = None,
        group: str | None = None,
    ) -> Iterator["Client"]:
        """
        Create a Client using ZooKeeper settings and yield it as a context manager.

        Args:
            settings:
                Optional AppSettings. If None, uses disco.config.get_settings().
            group:
                Optional ZooKeeper group/chroot namespace. If None, uses
                settings.zookeeper.default_group.

        Yields:
            Client: The constructed client.

        Notes:
            This context manager owns the ZooKeeper connection manager lifecycle.
        """
        if settings is None:
            settings = get_settings()

        with Cluster.make_cluster(settings.zookeeper, group=group) as cluster:
            yield cls(cluster=cluster)

    # ------------------------------------------------------------------ #
    # Submissions
    # ------------------------------------------------------------------ #

    def submit(self, exp: str | Experiment, repid: Optional[str] = None) -> list[tuple[str, str]]:
        """
        Submit replications for scheduling by the orchestrator.

        Submissions are enqueued via ExperimentStore.submit(expid, repid).

        Args:
            exp:
                Either an experiment id (str) or an Experiment object.
                If an Experiment object is provided, it is stored first.
            repid:
                If provided, submit only this replication id.
                If omitted (None), submit ALL replications in the experiment.

        Returns:
            List[(expid, repid)] of submitted items, in the order enqueued.

        Raises:
            KeyError:
                - If expid is unknown (when exp is str).
                - If repid is not found in the experiment.
        """
        if isinstance(exp, Experiment):
            self.store.store(exp)
            expid = exp.expid
            e = exp
        else:
            expid = exp
            e = self.store.load(expid)

        if repid is not None:
            # Let ExperimentStore.submit() validate repid membership.
            self.store.submit(expid, repid)
            return [(expid, repid)]

        submitted: list[tuple[str, str]] = []
        # Submit all replications.
        for r in e.replications.keys():
            self.store.submit(expid, r)
            submitted.append((expid, r))

        return submitted

    # ------------------------------------------------------------------ #
    # Cluster inspection
    # ------------------------------------------------------------------ #

    def worker_states(self) -> Mapping[str, WorkerState]:
        """
        Return a snapshot mapping of worker_address -> WorkerState.

        Notes:
            This is derived from Cluster's watch-maintained in-memory cache.
        """
        return self.cluster.worker_states

    def worker_info(self, worker_address: str) -> WorkerInfo:
        """
        Return WorkerInfo for one worker.

        Args:
            worker_address: The worker address/id.

        Raises:
            KeyError: If worker is not registered or has no info.
        """
        return self.cluster.get_worker_info(worker_address)

    def workers(self) -> list[WorkerSnapshot]:
        """
        Return a combined snapshot for all known workers.

        Each entry includes:
          - address
          - state (from Cluster.worker_states)
          - info (from Cluster.get_worker_info, best-effort)

        Notes:
            WorkerInfo may be None if the worker disappears between state and info reads.
        """
        out: list[WorkerSnapshot] = []
        for addr, state in self.cluster.worker_states.items():
            info: WorkerInfo | None
            try:
                info = self.cluster.get_worker_info(addr)
            except KeyError:
                info = None
            out.append(WorkerSnapshot(address=addr, state=state, info=info))
        return out

    def active_orchestrator(self) -> Any:
        """
        Return the active orchestrator leader record, or None if no leader exists.

        Returns:
            LeaderRecord | None (typed as Any here to avoid importing LeaderRecord in client API surface).
        """
        return self.cluster.get_active_orchestrator()

    # ------------------------------------------------------------------ #
    # Experiment inspection
    # ------------------------------------------------------------------ #

    def get_experiment(self, expid: str) -> Experiment:
        """
        Load a single experiment from the ExperimentStore.

        Args:
            expid: Experiment id.

        Raises:
            KeyError: If experiment does not exist.
        """
        return self.store.load(expid)

    def list_experiments(self) -> dict[str, ExperimentStatus]:
        """
        List experiments and their current status.

        Returns:
            Mapping expid -> ExperimentStatus.

        Notes:
            This loads each experiment blob to read its status. For large numbers of
            experiments, consider adding an index later (but keep client free of
            direct metastore access).
        """
        out: dict[str, ExperimentStatus] = {}
        for expid in self.store.list_expids():  # ExperimentStore provides enumeration
            try:
                exp = self.store.load(expid)
            except KeyError:
                # Experiment disappeared between listing and load; ignore.
                continue
            out[expid] = exp.status
        return out

    def list_replications(self, expid: str) -> dict[str, ExperimentStatus]:
        """
        List replications for an experiment and their current status.

        Args:
            expid: Experiment id.

        Returns:
            Mapping repid -> ExperimentStatus.

        Raises:
            KeyError: If experiment does not exist.
        """
        exp: Experiment = self.store.load(expid)
        return {r.repid: r.status for r in exp.replications.values()}

    # ------------------------------------------------------------------ #
    # Control-plane helpers (pause/resume/cancel)
    # ------------------------------------------------------------------ #

    def pause(self, expid: str, repid: str) -> None:
        """
        Request all assigned partitions of a replication to PAUSE.

        Args:
            expid: Experiment id.
            repid: Replication id.

        Notes:
            This issues desired-state changes to workers. Workers are responsible
            for updating replication/partition status in ExperimentStore.
        """
        exp = self.store.load(expid)
        rep = exp.replications[repid]
        for ass in rep.assignments.values():
            self.cluster.set_desired_state(
                worker_address=ass.worker,
                state=WorkerState.PAUSED,
                expid=expid,
                repid=repid,
                partition=int(ass.partition),
            )

    def resume(self, expid: str, repid: str) -> None:
        """
        Request all assigned partitions of a replication to become ACTIVE.

        Args:
            expid: Experiment id.
            repid: Replication id.
        """
        exp = self.store.load(expid)
        rep = exp.replications[repid]
        for ass in rep.assignments.values():
            self.cluster.set_desired_state(
                worker_address=ass.worker,
                state=WorkerState.ACTIVE,
                expid=expid,
                repid=repid,
                partition=int(ass.partition),
            )

    def cancel(self, expid: str, repid: str) -> None:
        """
        Request all assigned partitions of a replication to TERMINATE.

        Args:
            expid: Experiment id.
            repid: Replication id.
        """
        exp = self.store.load(expid)
        rep = exp.replications[repid]
        for ass in rep.assignments.values():
            self.cluster.set_desired_state(
                worker_address=ass.worker,
                state=WorkerState.TERMINATED,
                expid=expid,
                repid=repid,
                partition=int(ass.partition),
            )
