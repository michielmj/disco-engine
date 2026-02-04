from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pytest
from kazoo.exceptions import BadVersionError, NoNodeError, NodeExistsError

from disco.experiments import Experiment, ExperimentStatus, ExperimentStore, Replication
from disco.metastore.store import Metastore


# ---------------------------------------------------------------------------
# Minimal fakes for Metastore (Kazoo) used by ExperimentStore tests
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FakeStat:
    version: int


class FakeKazooClient:
    """
    Minimal in-memory KazooClient fake supporting the subset needed for Metastore CAS updates.
    """
    def __init__(self) -> None:
        self.data: Dict[str, bytes] = {}
        self.versions: Dict[str, int] = {}
        self.ensure_calls: List[str] = []

    def ensure_path(self, path: str) -> None:
        self.ensure_calls.append(path)

    def exists(self, path: str) -> FakeStat | None:
        if path in self.data:
            return FakeStat(self.versions.get(path, 0))

        # container semantics: parent exists if it has children
        prefix = path.rstrip("/")
        if not prefix:
            prefix = "/"
        child_prefix = prefix + "/"
        if any(p.startswith(child_prefix) for p in self.data.keys()):
            return FakeStat(self.versions.get(prefix, 0))
        return None

    def get(self, path: str) -> Tuple[bytes, FakeStat]:
        if path not in self.data:
            raise NoNodeError()
        return self.data[path], FakeStat(self.versions.get(path, 0))

    def set(self, path: str, value: bytes, version: int = -1) -> None:
        if path not in self.data:
            raise NoNodeError()

        cur = self.versions.get(path, 0)
        if version != -1 and version != cur:
            raise BadVersionError()

        self.data[path] = value
        self.versions[path] = cur + 1

    def create(self, path: str, value: bytes, makepath: bool = False, ephemeral: bool = False) -> None:
        if path in self.data:
            raise NodeExistsError()
        self.data[path] = value
        self.versions[path] = 0

    def delete(self, path: str, recursive: bool = False) -> None:
        prefix = path.rstrip("/")
        if not prefix:
            prefix = "/"
        to_delete = [p for p in list(self.data.keys()) if p == prefix or p.startswith(prefix + "/")]
        if not to_delete:
            raise NoNodeError()
        for p in to_delete:
            self.data.pop(p, None)
            self.versions.pop(p, None)

    def get_children(self, path: str) -> list[str]:
        prefix = path.rstrip("/")
        if prefix == "":
            prefix = "/"
        p_len = len(prefix)
        children = set()
        for p in self.data.keys():
            if not p.startswith(prefix) or p == prefix:
                continue
            sub = p[p_len:]
            if not sub.startswith("/"):
                continue
            parts = sub.strip("/").split("/", 1)
            if parts[0]:
                children.add(parts[0])
        return sorted(children)


class FakeConnectionManager:
    def __init__(self, client: FakeKazooClient) -> None:
        self._client = client
        self._stopped = False

    @property
    def client(self) -> FakeKazooClient:
        return self._client

    @property
    def stopped(self) -> bool:
        return self._stopped

    def watch_data(self, path: str, callback: Callable[[bytes | None, str], bool]) -> str:
        raise NotImplementedError

    def watch_children(self, path: str, callback: Callable[[list[str] | None, str], bool]) -> str:
        raise NotImplementedError


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
@pytest.fixture
def metastore() -> Metastore:
    client = FakeKazooClient()
    conn = FakeConnectionManager(client)
    return Metastore(connection=conn, group=None, base_structure=["/experiments"])


@pytest.fixture
def store(metastore: Metastore) -> ExperimentStore:
    return ExperimentStore(metastore=metastore)


# ---------------------------------------------------------------------------
# Model tests (dataclasses + normalize)
# ---------------------------------------------------------------------------

def test_experiment_allows_empty_partitionings_and_requires_none_selection() -> None:
    e = Experiment(duration=1.0, scenario_id="s1", allowed_partitionings=[])
    e.select_partitioning(None)  # implicit single-partition mode is OK

    with pytest.raises(ValueError):
        e.select_partitioning("p1")  # not allowed when list is empty


def test_generate_replications_assigns_sequence_numbers() -> None:
    e = Experiment(duration=1.0, scenario_id="s1", allowed_partitionings=[])
    e.generate_replications(3)

    assert len(e.replications) == 3
    repnos = sorted(r.repno for r in e.replications.values())
    assert repnos == [0, 1, 2]


def test_status_and_exception_propagation_via_normalize() -> None:
    e = Experiment(duration=1.0, scenario_id="s1", allowed_partitionings=[])
    e.generate_replications(1)
    repid = next(iter(e.replications.keys()))

    e.assign_partition(repid=repid, partition=0, worker="w1", initial_status=ExperimentStatus.ASSIGNED)

    e.replications[repid].assignments[0].status = ExperimentStatus.ACTIVE
    e.normalize()
    assert e.replications[repid].status == ExperimentStatus.ACTIVE
    assert e.status == ExperimentStatus.ACTIVE

    e.replications[repid].assignments[0].status = ExperimentStatus.FINISHED
    e.normalize()
    assert e.replications[repid].status == ExperimentStatus.FINISHED
    assert e.status == ExperimentStatus.FINISHED

    e.replications[repid].assignments[0].status = ExperimentStatus.FAILED
    e.replications[repid].assignments[0].exc = {"description": "boom"}
    e.normalize()
    assert e.replications[repid].status == ExperimentStatus.FAILED
    assert e.status == ExperimentStatus.FAILED
    assert "description" in e.replications[repid].exc and e.replications[repid].exc["description"] == "boom"
    assert "description" in e.exc and e.exc["description"] == "boom"


# ---------------------------------------------------------------------------
# Store tests (Metastore atomic updates + normalize on commit)
# ---------------------------------------------------------------------------

def test_store_roundtrip_and_atomic_update_normalizes(store: ExperimentStore) -> None:
    e = Experiment(duration=1.0, scenario_id="s1", allowed_partitionings=[])
    e.generate_replications(1)
    repid = next(iter(e.replications.keys()))
    e.assign_partition(repid=repid, partition=0, worker="w1", initial_status=ExperimentStatus.ASSIGNED)

    store.store(e)

    def mut(ex: Experiment) -> None:
        # Deliberately mutate only leaf assignment state (no explicit normalize/recompute).
        ex.replications[repid].assignments[0].status = ExperimentStatus.FINISHED

    updated = store.atomic_update(e.expid, mut)

    assert updated.replications[repid].assignments[0].status == ExperimentStatus.FINISHED
    assert updated.replications[repid].status == ExperimentStatus.FINISHED
    assert updated.status == ExperimentStatus.FINISHED

    reloaded = store.load(e.expid)
    assert reloaded.replications[repid].status == ExperimentStatus.FINISHED
    assert reloaded.status == ExperimentStatus.FINISHED


def test_store_convenience_methods_are_atomic(store: ExperimentStore) -> None:
    e = Experiment(duration=1.0, scenario_id="s1", allowed_partitionings=[])
    store.store(e)

    e2 = store.generate_replications(e.expid, 1)
    repid = next(iter(e2.replications.keys()))

    e3 = store.assign_partition(e.expid, repid, partition=0, worker="w1")
    assert e3.replications[repid].assignments[0].worker == "w1"

    e4 = store.set_partition_status(e.expid, repid, 0, ExperimentStatus.ACTIVE)
    assert e4.replications[repid].status == ExperimentStatus.ACTIVE
    assert e4.status == ExperimentStatus.ACTIVE

    e5 = store.set_partition_exc(
        expid = e.expid,
        repid = repid,
        partition=0,
        exc={"description": "traceback..."},
        fail_partition=True
    )
    assert e5.replications[repid].status == ExperimentStatus.FAILED
    assert e5.status == ExperimentStatus.FAILED
    assert "description" in e5.exc and e5.exc["description"] == "traceback..."
