# src/disco/experiments/store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, cast, Literal

from kazoo.exceptions import NoNodeError

from disco.metastore import Metastore, QueueEntity
from .experiment import (
    Experiment,
    ExperimentStatus,
)

EXPERIMENTS_ROOT = "/simulation/experiments"
SUBMISSIONS_QUEUE = "/simulation/orchestrator/submissions"


@dataclass(frozen=True, slots=True)
class Submission(QueueEntity):
    value: tuple[str, str]  # expid, repid


def experiment_path(expid: str) -> str:
    return f"{EXPERIMENTS_ROOT}/{expid}"


class ExperimentStore:
    """
    Persistence & atomic update helper around Metastore.
    Stores each experiment as a SINGLE key (one znode) to guarantee:
      - load/store/update in a single "transaction" (single znode set/CAS)
      - safe concurrent worker updates via metastore.atomic_update_key()
    """

    def __init__(self, metastore: Metastore):
        self.metastore = metastore
        metastore.ensure_structure([EXPERIMENTS_ROOT, SUBMISSIONS_QUEUE])

    def _path(self, expid: str) -> str:
        return f"{EXPERIMENTS_ROOT}/{expid}"

    # -----------------------------
    # CRUD
    # -----------------------------

    def store(self, experiment: Experiment) -> None:
        self.metastore.update_key(self._path(experiment.expid), experiment.to_dict())

    def load(self, expid: str) -> Experiment:
        raw = self.metastore.get_key(self._path(expid))
        if raw is None:
            raise KeyError(f"Experiment {expid!r} not found")
        if not isinstance(raw, dict):
            raise TypeError(f"Unexpected experiment payload type: {type(raw)}")
        return Experiment.from_dict(raw)

    def reload(self, experiment: Experiment) -> Experiment:
        return self.load(experiment.expid)

    def list_expids(self) -> list[str]:
        """
        List known experiments (child names under EXPERIMENTS_ROOT).
        """
        return sorted(self.metastore.list_members(EXPERIMENTS_ROOT))

    def list_experiments(self) -> list[Experiment]:
        """
        Convenience: load all experiments (best-effort, may be heavy).
        """
        exps: list[Experiment] = []
        for expid in self.list_expids():
            try:
                exps.append(self.load(expid))
            except KeyError:
                # Experiment disappeared between list and load; ignore.
                continue
        return exps

    def submit(self, expid: str, repid: str) -> None:
        exp = self.load(expid)
        if repid not in exp.replications:
            raise KeyError(f"Replication {repid!r} not found for experiment {expid!r}.")
        rep = exp.replications[repid]
        if rep.status != ExperimentStatus.CREATED:
            raise RuntimeError(f"Replication {repid!r} was already submitted.")

        # update status and submit
        self.set_replication_status(expid=expid, repid=repid, status=ExperimentStatus.SUBMITTED)
        self.metastore.enqueue(SUBMISSIONS_QUEUE, (expid, repid))

    def dequeue(self,
                timeout: Optional[float] = None,
                *,
                force_mode: Literal["raise", "release", "consume"] = "raise",
                ) -> Submission:
        return cast(self.metastore.dequeue(
            SUBMISSIONS_QUEUE,
            timeout=timeout,
            force_mode=force_mode
        ), Submission)

    # -----------------------------
    # Atomic updates (worker-safe)
    # -----------------------------

    def atomic_update(self, expid: str, mutator: Callable[[Experiment], None]) -> Experiment:
        """
        Atomically read-modify-write the experiment blob using metastore.atomic_update_key.
        """

        path = self._path(expid)

        def _updater(current: Any) -> Any:
            if current is None:
                raise NoNodeError(path)
            if not isinstance(current, dict):
                raise TypeError(f"Unexpected experiment payload type: {type(current)}")

            e = Experiment.from_dict(current)
            mutator(e)
            e.normalize()
            return e.to_dict()

        updated_raw = self.metastore.atomic_update_key(path, _updater, create_if_missing=False)
        if not isinstance(updated_raw, dict):
            raise TypeError(f"Unexpected updated payload type: {type(updated_raw)}")
        return Experiment.from_dict(updated_raw)

    # -----------------------------
    # Convenience operations (these are what workers/orchestrator call)
    # -----------------------------

    def generate_replications(
            self, expid: str, n_replications: int, *, seeds: Optional[list[int]] = None) -> Experiment:
        def mut(e: Experiment) -> None:
            e.generate_replications(n_replications)

        return self.atomic_update(expid, mut)

    def select_partitioning(self, expid: str, partitioning: Optional[str]) -> Experiment:
        def mut(e: Experiment) -> None:
            e.select_partitioning(partitioning)

        return self.atomic_update(expid, mut)

    def assign_partition(self, expid: str, repid: str, partition: int, worker: str) -> Experiment:
        def mut(e: Experiment) -> None:
            e.assign_partition(repid=repid, partition=partition, worker=worker)

        return self.atomic_update(expid, mut)

    def assign_partitions(self, expid: str, repid: str, assignments: list[str]) -> Experiment:
        def mut(e: Experiment) -> None:
            for partition, worker in enumerate(assignments):
                e.assign_partition(repid=repid, partition=partition, worker=worker)

        return self.atomic_update(expid, mut)

    def assign_replication(self, expid: str, repid: str, worker: str) -> Experiment:
        # unpartitioned shortcut
        return self.assign_partition(expid, repid, 0, worker)

    def set_replication_status(
            self,
            expid: str,
            repid: str,
            status: ExperimentStatus,
    ) -> Experiment:
        def mut(e: Experiment) -> None:
            e.set_replication_status(repid=repid, status=status)

        return self.atomic_update(expid, mut)

    def set_replication_exc(
            self,
            expid: str,
            repid: str,
            exc: dict[str, Any],
            *,
            fail_replication: bool = False,
    ) -> Experiment:
        """
        Update assignment exception message and propagate upward.
        Optionally also mark FAILED in same atomic write (common worker pattern).
        """

        def mut(e: Experiment) -> None:
            e.set_replication_exc(repid=repid, exc=exc)
            if fail_replication and exc:
                e.set_replication_status(repid=repid, status=ExperimentStatus.FAILED)

        return self.atomic_update(expid, mut)

    def set_partition_status(
            self,
            expid: str,
            repid: str,
            partition: int,
            status: ExperimentStatus,
    ) -> Experiment:
        def mut(e: Experiment) -> None:
            e.set_partition_status(repid=repid, partition=partition, status=status)

        return self.atomic_update(expid, mut)

    def set_partition_exc(
            self,
            expid: str,
            repid: str,
            partition: int,
            exc: dict[str, Any],
            *,
            fail_partition: bool = False,
    ) -> Experiment:
        """
        Update assignment exception message and propagate upward.
        Optionally also mark FAILED in same atomic write (common worker pattern).
        """

        def mut(e: Experiment) -> None:
            e.set_partition_exc(repid=repid, partition=partition, exc=exc)
            if fail_partition and exc:
                e.set_partition_status(repid=repid, partition=partition, status=ExperimentStatus.FAILED)

        return self.atomic_update(expid, mut)
