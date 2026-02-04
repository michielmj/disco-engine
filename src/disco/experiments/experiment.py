from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional
from uuid import uuid4
import secrets

from numpy.random import SeedSequence

from disco.exceptions import ExcDict, exc_equals


class ExperimentStatus(IntEnum):
    CREATED = 1
    SUBMITTED = 2
    ASSIGNED = 3
    LOADED = 4
    INITIALIZED = 5
    ACTIVE = 6
    PAUSED = 7
    FINISHED = 8
    CANCELED = 9
    FAILED = 10


def _rand_u32() -> int:
    return secrets.randbits(32) & 0xFFFFFFFF


@dataclass(slots=True)
class Assignment:
    partition: int
    worker: str
    status: ExperimentStatus = ExperimentStatus.CREATED
    exc: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition": self.partition,
            "worker": self.worker,
            "status": int(self.status),
            "exc": self.exc,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Assignment":
        return Assignment(
            partition=int(d["partition"]),
            worker=str(d["worker"]),
            status=ExperimentStatus(int(d["status"])),
            exc=d.get("exc"),
        )


@dataclass(slots=True)
class Replication:
    repno: int
    seed_sequence: SeedSequence
    repid: str = field(default_factory=lambda: str(uuid4()))
    status: ExperimentStatus = ExperimentStatus.CREATED
    assignments: Dict[int, Assignment] = field(default_factory=dict)  # partition -> Assignment
    exc: Optional[dict] = None
    _ss: list[SeedSequence] = field(init=False, repr=False)

    def __post_init__(self):
        self._ss = []

    def normalize(self) -> None:
        self._recompute_status_and_exc()

    # -----------------------------
    # Seed generation
    # -----------------------------

    def get_seed_sequence(self, partition: int) -> SeedSequence:
        if partition >= len(self._ss):
            self._ss += self.seed_sequence.spawn(partition - len(self._ss) + 1)

        return self._ss[partition]

    # -----------------------------
    # Assignments + propagation
    # -----------------------------

    def assign_partition(
        self,
        partition: int,
        worker: str,
        *,
        initial_status: ExperimentStatus = ExperimentStatus.ASSIGNED,
    ) -> Assignment:
        a = Assignment(partition=int(partition), worker=worker, status=initial_status)
        self.assignments[a.partition] = a
        self._recompute_status_and_exc()
        return a

    def set_replication_status(self, status: ExperimentStatus) -> bool:
        if self.status == status:
            return False
        self.status = status
        return True

    def set_replication_exc(self, exc: dict[str, Any]) -> bool:
        if self.exc == exc:
            return False
        self.exc = exc
        return self._recompute_status_and_exc()

    def set_partition_status(self, partition: int, status: ExperimentStatus) -> bool:
        a = self.assignments[int(partition)]
        if a.status == status:
            return False
        a.status = status
        return self._recompute_status_and_exc()

    def set_partition_exc(self, partition: int, exc: dict[str, Any]) -> bool:
        a = self.assignments[int(partition)]
        if a.exc == exc:
            return False
        a.exc = exc
        return self._recompute_status_and_exc()

    def _recompute_status_and_exc(self) -> bool:
        old_status = self.status
        old_exc = self.exc

        # Deterministic ordering: by partition id
        items = [self.assignments[p] for p in sorted(self.assignments)]

        # Aggregate exceptions from assignments
        self.exc = ExcDict().merge_many(a.exc for a in items)

        # Status propagation (guard against empty => don't mark FINISHED)
        if any(a.status == ExperimentStatus.FAILED for a in items):
            self.status = ExperimentStatus.FAILED
        elif any(a.status == ExperimentStatus.CANCELED for a in items):
            self.status = ExperimentStatus.CANCELED
        elif items and all(a.status == ExperimentStatus.FINISHED for a in items):
            self.status = ExperimentStatus.FINISHED
        elif any(a.status == ExperimentStatus.ACTIVE for a in items):
            self.status = ExperimentStatus.ACTIVE

        return (self.status != old_status) or not exc_equals(self.exc, old_exc)

    # -----------------------------
    # (De)serialization
    # -----------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repno": self.repno,
            "repid": self.repid,
            "seed_sequence": self.seed_sequence.state,
            "status": int(self.status),
            "assignments": {str(p): a.to_dict() for p, a in self.assignments.items()},
            "exc": self.exc,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Replication":
        r = Replication(
            repno=int(d["repno"]),
            repid=str(d["repid"]),
            seed_sequence=SeedSequence(**d.get("seed_sequence", {})),
            status=ExperimentStatus(int(d.get("status", int(ExperimentStatus.CREATED)))),
            exc=d.get("exc"),
        )
        assigns_in = d.get("assignments", {}) or {}
        r.assignments = {int(p): Assignment.from_dict(ad) for p, ad in assigns_in.items()}
        r._recompute_status_and_exc()
        return r


@dataclass(slots=True)
class Experiment:
    duration: float
    scenario_id: str
    allowed_partitionings: List[str] = field(default_factory=list)
    master_seed: Optional[int] = None
    expid: str = field(default_factory=lambda: str(uuid4()))
    selected_partitioning: Optional[str] = None  # None => implicit single-partition mode
    replications: Dict[str, Replication] = field(default_factory=dict)  # repid -> Replication
    params: Optional[Dict[str, Any]] = None
    status: ExperimentStatus = ExperimentStatus.CREATED
    exc: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be > 0")

    def normalize(self) -> None:
        # recompute all derived fields and enforce invariants
        for r in self.replications.values():
            r.normalize()
        self._recompute_status_and_exc()

    # -----------------------------
    # Partitioning semantics
    # -----------------------------

    def is_unpartitioned(self) -> bool:
        # If allowed_partitionings is empty, treat as a single partition.
        return not self.allowed_partitionings

    def select_partitioning(self, partitioning: Optional[str]) -> None:
        """
        If allowed_partitionings is empty, partitioning must be None (implicit 1 partition).
        Otherwise it must be one of allowed_partitionings.
        """
        if self.is_unpartitioned():
            if partitioning is not None:
                raise ValueError("No partitionings allowed; use None (implicit single partition).")
            self.selected_partitioning = None
        else:
            if partitioning is None:
                raise ValueError("partitioning must be selected when allowed_partitionings is non-empty")
            if partitioning not in self.allowed_partitionings:
                raise ValueError(f"partitioning {partitioning!r} not in allowed_partitionings")
            self.selected_partitioning = partitioning

    # -----------------------------
    # Replication generation
    # -----------------------------

    def generate_replications(self, n_replications: int) -> None:
        if n_replications <= 0:
            raise ValueError("n_replications must be > 0")

        start_repno = 0
        if self.replications:
            start_repno = max(r.repno for r in self.replications.values()) + 1

        ss = SeedSequence(self.master_seed)
        sss = ss.spawn(n_replications)

        for i in range(n_replications):
            r = Replication(repno=start_repno + i, seed_sequence=sss[i])
            self.replications[r.repid] = r

        self._recompute_status_and_exc()

    # -----------------------------
    # Assignment operations
    # -----------------------------

    def assign_partition(
        self,
        *,
        repid: str,
        partition: int,
        worker: str,
        initial_status: ExperimentStatus = ExperimentStatus.ASSIGNED,
    ) -> Assignment:
        r = self.replications[repid]
        a = r.assign_partition(partition=partition, worker=worker, initial_status=initial_status)
        self._recompute_status_and_exc()
        return a

    def assign_replication(self, *, repid: str, worker: str) -> Assignment:
        # unpartitioned "whole replication" assignment
        return self.assign_partition(
            repid=repid,
            partition=0,
            worker=worker,
            initial_status=ExperimentStatus.ASSIGNED,
        )

    def set_replication_status(self, *, repid: str, status: ExperimentStatus) -> bool:
        changed = self.replications[repid].set_replication_status(status)
        if changed:
            self._recompute_status_and_exc()
        return changed

    def set_replication_exc(self, *, repid: str, exc: dict[str, Any]) -> bool:
        changed = self.replications[repid].set_replication_exc(exc)
        if changed:
            self._recompute_status_and_exc()
        return changed

    def set_partition_status(self, *, repid: str, partition: int, status: ExperimentStatus) -> bool:
        changed = self.replications[repid].set_partition_status(partition, status)
        if changed:
            self._recompute_status_and_exc()
        return changed

    def set_partition_exc(self, *, repid: str, partition: int, exc: dict[str, Any]) -> bool:
        changed = self.replications[repid].set_partition_exc(partition, exc)
        if changed:
            self._recompute_status_and_exc()
        return changed

    # -----------------------------
    # Replication -> Experiment propagation
    # -----------------------------

    def _recompute_status_and_exc(self) -> bool:
        old_status = self.status
        old_exc = self.exc

        # Deterministic ordering: by repno (tie-breaker by repid to be safe)
        reps = sorted(self.replications.values(), key=lambda r: (r.repno, r.repid))

        # Aggregate exceptions from replications
        self.exc = ExcDict().merge_many(r.exc for r in reps)

        if any(r.status == ExperimentStatus.FAILED for r in reps):
            self.status = ExperimentStatus.FAILED
        elif any(r.status == ExperimentStatus.CANCELED for r in reps):
            self.status = ExperimentStatus.CANCELED
        elif reps and all(r.status == ExperimentStatus.FINISHED for r in reps):
            self.status = ExperimentStatus.FINISHED
        elif any(r.status == ExperimentStatus.ACTIVE for r in reps):
            self.status = ExperimentStatus.ACTIVE

        return (self.status != old_status) or not exc_equals(self.exc, old_exc)

    # -----------------------------
    # (De)serialization
    # -----------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "expid": self.expid,
            "duration": float(self.duration),
            "scenario_id": self.scenario_id,
            "master_seed": self.master_seed,
            "allowed_partitionings": list(self.allowed_partitionings),
            "selected_partitioning": self.selected_partitioning,
            "replications": {rid: r.to_dict() for rid, r in self.replications.items()},
            "params": self.params,
            "status": int(self.status),
            "exc": self.exc,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Experiment":
        e = Experiment(
            expid=str(d["expid"]),
            duration=float(d["duration"]),
            scenario_id=str(d["scenario_id"]),
            master_seed=None if d["master_seed"] is None else int(d["master_seed"]),
            allowed_partitionings=[str(x) for x in d.get("allowed_partitionings", [])],
            selected_partitioning=d.get("selected_partitioning"),
            params=d.get("params"),
            status=ExperimentStatus(int(d.get("status", int(ExperimentStatus.CREATED)))),
            exc=d.get("exc"),
        )
        reps_in = d.get("replications", {}) or {}
        e.replications = {rid: Replication.from_dict(rd) for rid, rd in reps_in.items()}
        e._recompute_status_and_exc()
        return e
