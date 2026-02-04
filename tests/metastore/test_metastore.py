from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from kazoo.exceptions import BadVersionError, NoNodeError, NodeExistsError

from disco.metastore.store import Metastore, MetastoreConflictError, VersionToken

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLockingQueue:
    """
    Minimal in-memory stand-in for kazoo.recipe.queue.LockingQueue that supports
    Metastore semantics WITHOUT threads, watchers, retry, etc.

    Behavior:
      - put(): append item
      - get(): "locks" the head element and returns it (does not remove it)
      - holds_lock(): True iff an item is currently locked
      - consume(): removes the locked head item
      - release(): unlocks without removing
    """

    def __init__(self) -> None:
        self._items: list[bytes] = []
        self._locked: Optional[bytes] = None

    def put(self, item: bytes) -> None:
        self._items.append(item)

    def get(self, timeout: float | None = None) -> bytes | None:  # timeout ignored for unit tests
        if self._locked is not None:
            # Lock already held: for unit tests return None (Metastore guards holds_lock anyway).
            return None
        if not self._items:
            return None
        self._locked = self._items[0]
        return self._locked

    def holds_lock(self) -> bool:
        return self._locked is not None

    def consume(self) -> None:
        if self._locked is None:
            return
        if self._items and self._items[0] == self._locked:
            self._items.pop(0)
        self._locked = None

    def release(self) -> None:
        self._locked = None


class FakeQueueFactory:
    """
    queue_factory(client, full_path) -> FakeLockingQueue

    Keeps one queue per *full_path*.
    """

    def __init__(self) -> None:
        self.queues: Dict[str, FakeLockingQueue] = {}

    def __call__(self, client: Any, full_path: str) -> FakeLockingQueue:  # client unused
        q = self.queues.get(full_path)
        if q is None:
            q = FakeLockingQueue()
            self.queues[full_path] = q
        return q


@dataclass(slots=True)
class FakeStat:
    """
    Minimal ZnodeStat stand-in.
    Kazoo uses ZnodeStat with many fields; we only need `version`.
    """

    version: int


class FakeKazooClient:
    """
    In-memory fake ZooKeeper client.

    Stores:
      - node payload bytes
      - per-node `version` that increments on every successful `set()`
    """

    def __init__(self) -> None:
        self.data: Dict[str, bytes] = {}
        self.versions: Dict[str, int] = {}
        self.ensure_calls: List[str] = []

    # Basic CRUD ------------------------------------------------------------

    def ensure_path(self, path: str) -> None:
        # ZooKeeper ensure_path creates intermediate nodes.
        # Tests typically just assert it was called.
        self.ensure_calls.append(path)

    def exists(self, path: str) -> FakeStat | None:
        """
        Simulate Kazoo exists():
        - returns a stat-like object if there is a node at `path`,
          or if there are children under `path/…` (container node semantics).
        - returns None otherwise.
        """
        prefix = path.rstrip("/")
        if not prefix:
            prefix = "/"

        # Direct node exists (data stored)
        if prefix in self.data:
            return FakeStat(self.versions.get(prefix, 0))

        # Any child under this node => parent "exists" for our tests
        child_prefix = prefix + "/"
        if any(p.startswith(child_prefix) for p in self.data.keys()):
            return FakeStat(self.versions.get(prefix, 0))

        return None

    def get(self, path: str) -> Tuple[bytes, FakeStat]:
        """
        Kazoo get() raises NoNodeError if missing and returns (data, stat) if present.
        """
        if path not in self.data:
            raise NoNodeError()
        return self.data[path], FakeStat(self.versions.get(path, 0))

    def set(self, path: str, value: bytes, version: int = -1) -> None:
        """
        Kazoo set(path, data, version=...) supports CAS:
          - version == -1 means unconditional set
          - otherwise version must match current stat.version
        On success, node version increments.
        """
        if path not in self.data:
            raise NoNodeError()

        current = self.versions.get(path, 0)

        if version != -1 and version != current:
            raise BadVersionError()

        self.data[path] = value
        self.versions[path] = current + 1

    # noinspection PyUnusedLocal
    def create(
        self,
        path: str,
        value: bytes,
        makepath: bool = False,
        ephemeral: bool = False,  # ignored
        **_kwargs: Any,  # accept sequence=True, acl=..., etc.
    ) -> None:
        """
        Kazoo create raises NodeExistsError if the node already exists.
        """
        if path in self.data:
            raise NodeExistsError()
        self.data[path] = value
        self.versions[path] = 0

    # noinspection PyUnusedLocal
    def delete(self, path: str, recursive: bool = False) -> None:
        """
        Delete node(s). Raise NoNodeError if nothing matched.
        """
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
        """
        Return immediate children under path based on stored nodes.
        """
        prefix = path.rstrip("/")
        if prefix == "":
            prefix = "/"
        p_len = len(prefix)

        children = set()
        for p in self.data.keys():
            if not p.startswith(prefix):
                continue
            if p == prefix:
                continue
            sub = p[p_len:]
            if not sub.startswith("/"):
                continue
            parts = sub.strip("/").split("/", 1)
            if parts[0]:
                children.add(parts[0])
        return sorted(children)


class FakeConnectionManager:
    """
    Minimal fake for ZkConnectionManager used by Metastore tests.
    """

    def __init__(self, client: FakeKazooClient) -> None:
        self._client = client
        self.watch_registrations: list[tuple[str, Callable]] = []
        self.children_watch_registrations: list[tuple[str, Callable]] = []
        self._stopped = False

    @property
    def client(self) -> FakeKazooClient:
        return self._client

    @property
    def stopped(self) -> bool:
        return self._stopped

    def watch_data(self, path: str, callback: Callable[[bytes | None, str], bool]):
        self.watch_registrations.append((path, callback))
        return f"watch-{len(self.watch_registrations)}"

    def watch_children(self, path: str, callback: Callable[[list[str] | None, str], bool]):
        self.children_watch_registrations.append((path, callback))
        return f"child-watch-{len(self.children_watch_registrations)}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client():
    return FakeKazooClient()


@pytest.fixture
def connection(fake_client):
    return FakeConnectionManager(fake_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList, PyUnusedLocal
def test_init_ensures_base_structure_without_group(connection, fake_client):
    _m = Metastore(connection=connection, group=None, base_structure=["/base", "/nested/path"])
    assert "/base" in fake_client.ensure_calls
    assert "/nested/path" in fake_client.ensure_calls


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList, PyUnusedLocal
def test_init_ensures_base_structure_with_group(connection, fake_client):
    _m = Metastore(connection=connection, group="g1", base_structure=["/base", "/nested/path"])
    assert "/g1/base" in fake_client.ensure_calls
    assert "/g1/nested/path" in fake_client.ensure_calls


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_update_and_get_key_default_pickle(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    value = {"a": 1, "b": [1, 2, 3]}
    m.update_key("/foo/bar", value)

    stored = fake_client.data["/foo/bar"]
    assert stored == pickle.dumps(value)

    assert m.get_key("/foo/bar") == value


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_update_and_get_key_with_group(connection, fake_client):
    m = Metastore(connection=connection, group="sim1")

    m.update_key("foo", 42)
    assert "/sim1/foo" in fake_client.data

    assert m.get_key("foo") == 42
    assert m["foo"] == 42


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_custom_serialization(connection, fake_client):
    calls: list[str] = []

    def packb(obj: Any) -> bytes:
        calls.append("pack")
        return str(obj).encode("utf-8")

    def unpackb(data: bytes) -> Any:
        calls.append("unpack")
        return int(data.decode("utf-8"))

    m = Metastore(connection=connection, group=None, packb=packb, unpackb=unpackb)

    m.update_key("/num", 123)
    assert "pack" in calls
    assert fake_client.data["/num"] == b"123"

    val = m.get_key("/num")
    assert "unpack" in calls
    assert val == 123


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_watch_with_callback_wraps_and_registers(connection, fake_client):
    m = Metastore(connection=connection, group="g")

    received: list[tuple[Any, str]] = []

    def user_cb(value: Any, path: str) -> bool:
        received.append((value, path))
        return True

    watch_id = m.watch_with_callback("/foo", user_cb)
    assert watch_id == "watch-1"

    assert len(connection.watch_registrations) == 1
    path, wrapped = connection.watch_registrations[0]
    assert path == "/g/foo"

    wrapped(pickle.dumps({"x": 1}), "/g/foo")
    assert received == [({"x": 1}, "/g/foo")]

    before = len(received)
    keep = wrapped(None, "/g/foo")
    assert keep is False
    assert len(received) == before


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_watch_members_with_callback_wraps_and_registers(connection, fake_client):
    m = Metastore(connection=connection, group="g")

    received: list[tuple[list[str], str]] = []

    def user_cb(children: list[str], path: str) -> bool:
        received.append((children, path))
        return True

    watch_id = m.watch_members_with_callback("root", user_cb)
    assert watch_id == "child-watch-1"

    assert len(connection.children_watch_registrations) == 1
    full_path, wrapped = connection.children_watch_registrations[0]
    assert full_path == "/g/root"
    assert callable(wrapped)

    keep = wrapped(["a", "b"], full_path)
    assert keep is True
    assert received == [(["a", "b"], "/g/root")]

    before = len(received)
    keep2 = wrapped(None, full_path)
    assert keep2 is False
    assert len(received) == before


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_contains_and_list_members(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    m.update_key("/root/a", 1)
    m.update_key("/root/b", 2)
    m.update_key("/root/sub/c", 3)

    assert "/root/a" in fake_client.data
    assert "root/a" in m
    assert "root/x" not in m

    children = m.list_members("root")
    assert set(children) == {"a", "b", "sub"}


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_get_and_update_keys_with_expand_dict(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    data = {
        "replications": {
            "r1": {"assignments": {"a": 1, "b": 2}},
            "r2": {"assignments": {"c": 3}},
        },
        "simple": 99,
    }

    expand = {"replications": {"assignments": None}}

    m.update_keys("meta", data, expand=expand, drop=True)

    assert "/meta/simple" in fake_client.data
    assert "/meta/replications/r1/assignments/a" in fake_client.data

    read = m.get_keys("meta", expand=expand)
    assert read["simple"] == 99
    assert read["replications"]["r1"]["assignments"]["a"] == 1
    assert read["replications"]["r2"]["assignments"]["c"] == 3


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_drop_key(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    m.update_key("/root/a", 1)
    m.update_key("/root/sub/b", 2)

    assert m.drop_key("/root")
    assert not fake_client.data
    assert not m.drop_key("/root")


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_enqueue_and_dequeue(connection, fake_client, monkeypatch):
    # Use fake locking queues (no real Kazoo / ZooKeeper behavior)
    qf = FakeQueueFactory()
    m = Metastore(connection=connection, group=None, queue_factory=qf)

    m.enqueue("/queue", {"x": 1})
    m.enqueue("/queue", {"x": 2})

    e1 = m.dequeue("/queue", timeout=1.0)
    assert e1 is not None
    assert e1.value == {"x": 1}

    # Default force_mode="raise": cannot dequeue again while a lock is held.
    with pytest.raises(RuntimeError):
        m.dequeue("/queue", timeout=0.1)

    # force_mode="release": release the lock, then dequeue should return the same head again.
    e1b = m.dequeue("/queue", timeout=1.0, force_mode="release")
    assert e1b is not None
    assert e1b.value == {"x": 1}
    e1b.consume()  # remove {"x": 1}

    e2 = m.dequeue("/queue", timeout=1.0)
    assert e2 is not None
    assert e2.value == {"x": 2}
    e2.consume()

    e3 = m.dequeue("/queue", timeout=0.1)
    assert e3 is None


# Optional: verify force_mode="consume" drops the currently-locked head and advances.
# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_dequeue_force_mode_consume_drops_locked_head(connection, fake_client, monkeypatch):
    qf = FakeQueueFactory()
    m = Metastore(connection=connection, group=None, queue_factory=qf)

    m.enqueue("/queue", {"x": 1})
    m.enqueue("/queue", {"x": 2})

    e1 = m.dequeue("/queue", timeout=1.0)
    assert e1 is not None
    assert e1.value == {"x": 1}

    e2 = m.dequeue("/queue", timeout=1.0, force_mode="consume")
    assert e2 is not None
    assert e2.value == {"x": 2}
    e2.consume()


# ---------------------------------------------------------------------------
# Tests: versioning / CAS / atomic update
# ---------------------------------------------------------------------------

# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_get_key_with_version_missing_returns_none(connection, fake_client):
    m = Metastore(connection=connection, group=None)
    value, ver = m.get_key_with_version("/missing")
    assert value is None
    assert ver is None


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_get_key_with_version_returns_version_and_increments(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    m.update_key("/k", {"a": 1})
    v1, tok1 = m.get_key_with_version("/k")
    assert v1 == {"a": 1}
    assert tok1 is not None
    assert isinstance(tok1, VersionToken)
    assert tok1.value == 0

    m.update_key("/k", {"a": 2})
    v2, tok2 = m.get_key_with_version("/k")
    assert v2 == {"a": 2}
    assert tok2 is not None
    assert tok2.value == tok1.value + 1


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_update_key_expected_enforces_cas_and_raises_on_stale(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    m.update_key("/k", 1)
    _, tok = m.get_key_with_version("/k")
    assert tok is not None

    m.update_key("/k", 2, expected=tok)
    assert m.get_key("/k") == 2

    with pytest.raises(BadVersionError):
        m.update_key("/k", 3, expected=tok)


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_update_key_expected_raises_if_missing(connection, fake_client):
    m = Metastore(connection=connection, group=None)
    with pytest.raises(NoNodeError):
        m.update_key("/missing", 1, expected=VersionToken(0))


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_compare_and_set_key_success_and_failure(connection, fake_client):
    m = Metastore(connection=connection, group=None)

    m.update_key("/k", "v1")
    _, tok = m.get_key_with_version("/k")
    assert tok is not None

    ok = m.compare_and_set_key("/k", "v2", expected=tok)
    assert ok is True
    assert m.get_key("/k") == "v2"

    ok2 = m.compare_and_set_key("/k", "v3", expected=tok)
    assert ok2 is False
    assert m.get_key("/k") == "v2"

    ok3 = m.compare_and_set_key("/missing", 123, expected=VersionToken(0))
    assert ok3 is False


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_atomic_update_key_updates_existing(connection, fake_client, monkeypatch):
    m = Metastore(connection=connection, group=None)
    m.update_key("/counter", 0)

    monkeypatch.setattr("disco.metastore.store.sleep", lambda *_a, **_kw: None)

    out = m.atomic_update_key("/counter", lambda cur: int(cur) + 1)
    assert out == 1
    assert m.get_key("/counter") == 1


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_atomic_update_key_create_if_missing(connection, fake_client, monkeypatch):
    m = Metastore(connection=connection, group=None)

    monkeypatch.setattr("disco.metastore.store.sleep", lambda *_a, **_kw: None)

    out = m.atomic_update_key(
        "/new",
        lambda cur: {"created": cur is None},
        create_if_missing=True,
    )
    assert out == {"created": True}
    assert m.get_key("/new") == {"created": True}


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_atomic_update_key_missing_raises_if_not_creating(connection, fake_client, monkeypatch):
    m = Metastore(connection=connection, group=None)
    monkeypatch.setattr("disco.metastore.store.sleep", lambda *_a, **_kw: None)

    with pytest.raises(NoNodeError):
        m.atomic_update_key("/missing", lambda cur: 1, create_if_missing=False)


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_atomic_update_key_retries_on_contention_and_succeeds(connection, fake_client, monkeypatch):
    m = Metastore(connection=connection, group=None)
    m.update_key("/k", 0)

    monkeypatch.setattr("disco.metastore.store.sleep", lambda *_a, **_kw: None)

    real_cas = m.compare_and_set_key
    calls = {"n": 0}

    def flaky_cas(path: str, value: Any, *, expected: VersionToken) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            m.update_key(path, 999)  # bump version
            return False
        return real_cas(path, value, expected=expected)

    monkeypatch.setattr(m, "compare_and_set_key", flaky_cas)

    out = m.atomic_update_key("/k", lambda _cur: 123)
    assert out == 123
    assert m.get_key("/k") == 123
    assert calls["n"] >= 2


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_atomic_update_key_fails_after_max_retries(connection, fake_client, monkeypatch):
    m = Metastore(connection=connection, group=None)
    m.update_key("/k", 0)

    monkeypatch.setattr("disco.metastore.store.sleep", lambda *_a, **_kw: None)

    monkeypatch.setattr(m, "compare_and_set_key", lambda *_a, **_kw: False)

    with pytest.raises(MetastoreConflictError):
        m.atomic_update_key("/k", lambda cur: int(cur) + 1, max_retries=3)


# noinspection PyTypeChecker,PyUnresolvedReferences,PyArgumentList
def test_atomic_update_key_create_race_then_retry(connection, fake_client, monkeypatch):
    m = Metastore(connection=connection, group=None)
    monkeypatch.setattr("disco.metastore.store.sleep", lambda *_a, **_kw: None)

    real_create = fake_client.create
    calls = {"n": 0}

    def flaky_create(path: str, value: bytes, makepath: bool = False, ephemeral: bool = False, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            real_create(path, pickle.dumps("other"), makepath=makepath, ephemeral=ephemeral, **kwargs)
            raise NodeExistsError()
        return real_create(path, value, makepath=makepath, ephemeral=ephemeral, **kwargs)

    monkeypatch.setattr(fake_client, "create", flaky_create)

    out = m.atomic_update_key("/race", lambda _cur: "mine", create_if_missing=True)
    assert out == "mine"
    assert m.get_key("/race") == "mine"
