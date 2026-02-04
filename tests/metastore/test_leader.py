# tests/metastore/test_leader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import pytest
from kazoo.exceptions import NodeExistsError, NoNodeError

from disco.metastore.store import Metastore
from disco.metastore.leader import LeaderElection, LeaderRecord


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeStat:
    # Kazoo stat objects have many fields; we only use ephemeralOwner.
    ephemeralOwner: int = 0


class FakeElection:
    """
    Minimal stand-in for kazoo.recipe.election.Election.

    The real Election.run calls a function (the leader callback) when leadership is acquired.
    We emulate that once.
    """

    def __init__(self, client: Any, path: str, identifier: str) -> None:
        self.client = client
        self.path = path
        self.identifier = identifier
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True

    def run(self, leader_fn: Callable[[Callable[[], None]], None], on_lead: Callable[[], None]) -> None:
        leader_fn(on_lead)


class FakeKazooClient:
    def __init__(self) -> None:
        self.ensure_path_calls: list[str] = []
        self.create_calls: list[dict[str, Any]] = []
        self.set_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.exists_calls: list[str] = []

        # Behavior knobs
        self._create_raises_node_exists: bool = False
        self._exists_stat: Optional[_FakeStat] = None
        self._nodes: set[str] = set()

    def ensure_path(self, path: str) -> None:
        self.ensure_path_calls.append(path)
        self._nodes.add(path)

    def create(self, path: str, value: bytes, *, ephemeral: bool = False, makepath: bool = False) -> None:
        if self._create_raises_node_exists:
            raise NodeExistsError()

        self.create_calls.append(
            {"path": path, "value": value, "ephemeral": ephemeral, "makepath": makepath}
        )
        self._nodes.add(path)

    def set(self, path: str, value: bytes) -> None:
        self.set_calls.append({"path": path, "value": value})
        self._nodes.add(path)

    def delete(self, path: str) -> None:
        self.delete_calls.append(path)
        if path not in self._nodes:
            raise NoNodeError()
        self._nodes.remove(path)

    def exists(self, path: str) -> Optional[_FakeStat]:
        self.exists_calls.append(path)
        return self._exists_stat


class FakeZkConnectionManager:
    def __init__(self, client: FakeKazooClient) -> None:
        self._client = client
        self._stopped = False

    @property
    def client(self) -> FakeKazooClient:
        return self._client

    @property
    def stopped(self) -> bool:
        return self._stopped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_metastore_make_leader_election_ensures_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Metastore.make_leader_election should ensure:
      - root path exists (full path incl. group)
    LeaderElection __init__ should ensure:
      - <root>/election exists
    """
    import disco.metastore.leader as leader_mod

    monkeypatch.setattr(leader_mod, "Election", FakeElection)

    client = FakeKazooClient()
    conn = FakeZkConnectionManager(client)
    meta = Metastore(connection=conn, group="g")

    root_logical = "/simulation/orchestrator/leader_election"
    le = meta.make_leader_election(root_path=root_logical, candidate_id="orch-1", metadata={"a": 1})

    assert isinstance(le, LeaderElection)

    full_root = "/g/simulation/orchestrator/leader_election"
    assert full_root in client.ensure_path_calls
    assert f"{full_root}/election" in client.ensure_path_calls

    assert isinstance(le._election, FakeElection)  # type: ignore[attr-defined]
    assert le._election.path == f"{full_root}/election"  # type: ignore[attr-defined]
    assert le._election.identifier == "orch-1"  # type: ignore[attr-defined]


def test_leader_election_run_publishes_ephemeral_leader_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When leadership is acquired, LeaderElection should create an ephemeral leader node,
    run on_lead, and then delete the leader node on exit.
    """
    import disco.metastore.leader as leader_mod

    monkeypatch.setattr(leader_mod, "Election", FakeElection)

    client = FakeKazooClient()

    def packb(x: Any) -> bytes:
        assert isinstance(x, LeaderRecord)
        return b"LEADER-RECORD"

    le = LeaderElection(
        client=client,
        root_path="/g/simulation/orchestrator/leader_election",
        candidate_id="orch-1",
        metadata={"x": "y"},
        packb=packb,
        retry_delay_s=0.0,
    )

    called: dict[str, bool] = {"on_lead": False}

    def on_lead() -> None:
        called["on_lead"] = True
        le.cancel()  # stop run loop after first leadership

    le.run(on_lead)

    assert called["on_lead"] is True

    leader_path = "/g/simulation/orchestrator/leader_election/leader"
    assert any(c["path"] == leader_path and c["ephemeral"] is True for c in client.create_calls)
    assert leader_path in client.delete_calls


def test_publish_replaces_persistent_leader_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If leader node exists and is persistent (ephemeralOwner == 0), LeaderElection should
    delete it and recreate as ephemeral.
    """
    import disco.metastore.leader as leader_mod

    monkeypatch.setattr(leader_mod, "Election", FakeElection)

    client = FakeKazooClient()
    root = "/g/simulation/orchestrator/leader_election"
    leader_path = f"{root}/leader"

    # Simulate: node exists already (persistent) and create() sees NodeExistsError.
    client._nodes.add(leader_path)
    client._create_raises_node_exists = True
    client._exists_stat = _FakeStat(ephemeralOwner=0)

    def packb(_: Any) -> bytes:
        return b"X"

    le = LeaderElection(
        client=client,
        root_path=root,
        candidate_id="orch-1",
        metadata=None,
        packb=packb,
        retry_delay_s=0.0,
    )

    # Allow the second create to succeed after we "replace" the node.
    original_create = client.create

    def create_side_effect(path: str, value: bytes, *, ephemeral: bool = False, makepath: bool = False) -> None:
        if client._create_raises_node_exists:
            client._create_raises_node_exists = False
            raise NodeExistsError()
        original_create(path, value, ephemeral=ephemeral, makepath=makepath)

    client.create = create_side_effect  # type: ignore[method-assign]

    le._run_as_leader(lambda: None)  # type: ignore[attr-defined]

    # Should inspect, delete, and then recreate ephemerally.
    assert leader_path in client.exists_calls
    assert leader_path in client.delete_calls
    assert any(c["path"] == leader_path and c["ephemeral"] is True for c in client.create_calls)


def test_publish_updates_ephemeral_leader_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If leader node exists and is ephemeral (ephemeralOwner != 0), LeaderElection should
    update it with set().
    """
    import disco.metastore.leader as leader_mod

    monkeypatch.setattr(leader_mod, "Election", FakeElection)

    client = FakeKazooClient()
    root = "/g/simulation/orchestrator/leader_election"
    leader_path = f"{root}/leader"

    # Simulate: node exists already (ephemeral) and create() sees NodeExistsError.
    client._nodes.add(leader_path)
    client._create_raises_node_exists = True
    client._exists_stat = _FakeStat(ephemeralOwner=123)

    def packb(_: Any) -> bytes:
        return b"Y"

    le = LeaderElection(
        client=client,
        root_path=root,
        candidate_id="orch-1",
        metadata=None,
        packb=packb,
        retry_delay_s=0.0,
    )

    le._run_as_leader(lambda: None)  # type: ignore[attr-defined]

    assert leader_path in client.exists_calls
    assert any(c["path"] == leader_path and c["value"] == b"Y" for c in client.set_calls)


def test_cancel_cancels_election_and_deletes_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    import disco.metastore.leader as leader_mod

    monkeypatch.setattr(leader_mod, "Election", FakeElection)

    client = FakeKazooClient()
    root = "/g/simulation/orchestrator/leader_election"
    leader_path = f"{root}/leader"

    client._nodes.add(leader_path)

    le = LeaderElection(
        client=client,
        root_path=root,
        candidate_id="orch-1",
        metadata=None,
        packb=lambda _: b"Z",
        retry_delay_s=0.0,
    )

    le.cancel()

    assert isinstance(le._election, FakeElection)  # type: ignore[attr-defined]
    assert le._election.cancel_called is True  # type: ignore[attr-defined]
    assert leader_path in client.delete_calls
