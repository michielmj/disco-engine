# src/disco/metastore/store.py
from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass
from random import random
from time import sleep
from typing import Any, Callable, Optional, cast, Mapping, Literal, Protocol

from kazoo.client import KazooClient
from kazoo.exceptions import NoNodeError, BadVersionError, NodeExistsError
from kazoo.recipe.queue import LockingQueue

from tools.mp_logging import getLogger

from disco.exceptions import DiscoError
from .helpers import ZkConnectionManager
from .leader import LeaderElection

QUEUE_POLLING_S = 0.05  # polling for Kazoo queues

logger = getLogger(__name__)


class MetastoreError(DiscoError):
    """Base exception for Metastore-related errors."""
    pass


class MetastoreStoppedError(MetastoreError):
    """Raised when operations are attempted on a stopped Metastore."""
    pass


class MetastoreConflictError(MetastoreError):
    """Raised when an atomic update fails due to repeated concurrent modifications."""
    pass


@dataclass(frozen=True, slots=True)
class VersionToken:
    """
    Concurrency token for a key. In ZooKeeper this maps to `stat.version`.
    (In etcd this would map to `mod_revision`.)
    """
    value: int


@dataclass(frozen=True, slots=True)
class QueueEntity:
    """
    Represents a locked queue head element.

    - `value` is the decoded payload.
    - `consume()` removes it from the queue (ack).
    - `release()` puts it back (nack).
    """
    value: Any
    _queue: LockingQueue

    def consume(self) -> bool:
        return bool(self._queue.consume())

    def release(self) -> bool:
        return bool(self._queue.release())


class _LockingQueueProto(Protocol):
    def holds_lock(self) -> bool: ...
    def get(self, timeout: float | None = None): ...
    def put(self, value: bytes) -> None: ...
    def consume(self) -> None: ...
    def release(self) -> None: ...


class Metastore:
    """
    High-level metadata store on top of ZooKeeper.

    - Uses ZkConnectionManager (one client per process).
    - Optionally namespaces paths under '/<group>' inside ZooKeeper's chroot.
    - Recovers watchers after session loss via ZkConnectionManager.
    - Supports pluggable serialization (default = pickle).
    """

    def __init__(
            self,
            connection: ZkConnectionManager,
            group: Optional[str] = None,
            packb: Callable[[Any], bytes] = pickle.dumps,
            unpackb: Callable[[bytes], Any] = pickle.loads,
            base_structure: Optional[list[str]] = None,
            *,
            queue_factory: Callable[[KazooClient, str], _LockingQueueProto] | None = None,
    ) -> None:
        self._connection = connection
        self._group = group

        self._packb = packb
        self._unpackb = unpackb

        self._queue_polling_interval = QUEUE_POLLING_S

        # LockingQueue must be reused per queue path to keep the processing element handle.
        self._queue_factory = queue_factory or (lambda client, full_path: LockingQueue(client, full_path))
        self._queues: dict[str, LockingQueue] = {}

        self.ensure_structure(base_structure or [])

    @property
    def group(self) -> Optional[str]:
        return self._group

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _ensure_running(self) -> None:
        """
        Ensure that the underlying connection manager has not been stopped.
        """
        if self.stopped:
            raise MetastoreStoppedError("Metastore connection manager has been stopped.")

    def ensure_structure(self, base_structure: list[str]) -> None:
        """
        Ensure `base_structure` paths exist under the chroot + optional /<group>.
        """
        if not base_structure:
            return

        self._ensure_running()
        for path in base_structure:
            full = self._full_path(path)
            self.client.ensure_path(full)

    @property
    def client(self) -> KazooClient:
        self._ensure_running()
        return self._connection.client

    @property
    def stopped(self) -> bool:
        return self._connection.stopped

    def _full_path(self, path: str) -> str:
        """
        Build the ZooKeeper path inside the chroot configured on the client.
        Only adds '/<group>' if group is configured.
        """
        rel = (path or "").lstrip("/")

        if self._group:
            return f"/{self._group}/{rel}" if rel else f"/{self._group}"
        else:
            return f"/{rel}" if rel else "/"

    def _locking_queue(self, path: str) -> LockingQueue:
        self._ensure_running()
        full_path = self._full_path(path)
        q = self._queues.get(full_path)
        if q is None:
            q = self._queue_factory(self.client, full_path)
            self._queues[full_path] = q
        return q

    # ----------------------------------------------------------------------
    # Watcher registration (recovered after session loss)
    # ----------------------------------------------------------------------

    def watch_with_callback(
            self,
            path: str,
            callback: Callable[[Any, str], bool],
    ) -> uuid.UUID:
        """
        Register a watch on `path`. Callback receives (decoded_value, full_path).
        If callback returns False, the watch is removed.
        """
        self._ensure_running()
        full_path = self._full_path(path)

        def _wrapped(raw: Optional[bytes], p: str) -> bool:
            if raw is None:
                # Node deleted → stop watching.
                return False
            value = self._unpackb(raw)
            return callback(value, p)

        return self._connection.watch_data(full_path, _wrapped)

    def watch_members_with_callback(
            self,
            path: str,
            callback: Callable[[list[str], str], bool],
    ) -> uuid.UUID:
        """
        Register a children watch on the logical `path`.

        - Underlying ZooKeeper watch tracks children of `_full_path(path)`.
        - `callback(children, full_path)` is called when the children list changes.
        - If callback returns False, the watch is removed.
        """
        self._ensure_running()
        full_path = self._full_path(path)

        def _wrapped(children: Optional[list[str]], p: str) -> bool:
            if children is None:
                # Node deleted or no further interest → stop watching by default.
                return False
            return callback(children, p)

        return self._connection.watch_children(full_path, _wrapped)

    # ----------------------------------------------------------------------
    # Key-value operations
    # ----------------------------------------------------------------------

    def get_key_with_version(self, path: str) -> tuple[Any, Optional[VersionToken]]:
        """
        Read a key and return (value, version-token). If the node does not exist,
        returns (None, None).
        """
        self._ensure_running()
        full_path = self._full_path(path)

        try:
            data, stat = self.client.get(full_path)
        except NoNodeError:
            return None, None

        if not data:
            return None, VersionToken(int(stat.version))

        return self._unpackb(data), VersionToken(int(stat.version))

    def get_key(self, path: str) -> Any:
        value, _ver = self.get_key_with_version(path)
        return value

    def __getitem__(self, item: str) -> Any:
        return self.get_key(item)

    def update_key(
            self,
            path: str,
            value: Any,
            ephemeral: bool = False,
            *,
            expected: Optional[VersionToken] = None,
    ) -> None:
        """
        Write a key. If expected is provided, perform a CAS write (set with version).
        If the node doesn't exist and expected is provided, this raises NoNodeError.
        """
        self._ensure_running()
        full_path = self._full_path(path)
        data = self._packb(value)

        if expected is not None:
            # CAS update: only succeeds if version matches
            self.client.set(full_path, data, version=int(expected.value))
            return

        # Non-CAS update (create if missing)
        try:
            self.client.set(full_path, data)
        except NoNodeError:
            self.client.create(full_path, data, makepath=True, ephemeral=ephemeral)

    def compare_and_set_key(self, path: str, value: Any, *, expected: VersionToken) -> bool:
        """
        CAS write: returns True if written, False if version mismatch (or missing node).
        """
        self._ensure_running()
        full_path = self._full_path(path)
        data = self._packb(value)

        try:
            self.client.set(full_path, data, version=int(expected.value))
            return True
        except BadVersionError:
            return False
        except NoNodeError:
            return False

    def atomic_update_key(
            self,
            path: str,
            updater: Callable[[Any], Any],
            *,
            max_retries: int = 20,
            backoff_base_s: float = 0.005,
            backoff_max_s: float = 0.200,
            create_if_missing: bool = False,
            ephemeral: bool = False,
    ) -> Any:
        """
        Atomically update a single key using optimistic concurrency control.

        - Reads (value, version)
        - Computes new_value = updater(value)
        - Writes using CAS (set(..., version=...))
        - Retries on concurrent modification

        If create_if_missing is True and the node doesn't exist, attempts to create it.
        """
        self._ensure_running()
        full_path = self._full_path(path)

        for attempt in range(max_retries):
            current, ver = self.get_key_with_version(path)

            if ver is None:
                if not create_if_missing:
                    raise NoNodeError(full_path)

                # Create path with updater(None) (single create attempt, then retry if raced)
                new_value = updater(None)
                data = self._packb(new_value)
                try:
                    self.client.create(full_path, data, makepath=True, ephemeral=ephemeral)
                    return new_value
                except NodeExistsError:
                    # Lost the race; retry read+CAS
                    pass
            else:
                new_value = updater(current)
                if self.compare_and_set_key(path, new_value, expected=ver):
                    return new_value

            # Backoff with jitter
            delay = min(backoff_max_s, backoff_base_s * (2 ** attempt))
            delay = delay * (0.5 + random())  # jitter in [0.5, 1.5)
            sleep(delay)

        raise MetastoreConflictError(
            f"atomic_update_key({full_path}) failed after {max_retries} retries due to concurrent updates"
        )

    def drop_key(self, path: str) -> bool:
        self._ensure_running()
        full_path = self._full_path(path)
        if self.client.exists(full_path):
            self.client.delete(full_path, recursive=True)
            return True
        return False

    def __contains__(self, item: str) -> bool:
        self._ensure_running()
        return bool(self.client.exists(self._full_path(item)))

    def list_members(self, path: str) -> list[str]:
        self._ensure_running()
        return cast(list[str], self.client.get_children(self._full_path(path)))

    # ----------------------------------------------------------------------
    # Hierarchical get/update
    # ----------------------------------------------------------------------

    def get_keys(
            self,
            path: str,
            expand: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Returns all keys and values found at the logical path `path`.

        `expand` is a nested dict with the same semantics as in update_keys.
        """
        self._ensure_running()
        full_path = self._full_path(path)

        if not self.client.exists(full_path):
            return None

        expand = expand or {}
        data: dict[str, Any] = {}

        base = path.rstrip("/") if path else ""

        for member in self.client.get_children(full_path):
            member_rel = f"{base}/{member}" if base else member
            member_full = self._full_path(member_rel)

            if isinstance(expand, dict) and member in expand:
                cfg = expand[member]  # may be dict or None
                children = self.client.get_children(member_full)
                data[member] = {}

                if isinstance(cfg, dict):
                    # nested expansion allowed: recurse
                    for child in children:
                        child_rel = f"{member_rel}/{child}"
                        data[member][child] = self.get_keys(child_rel, expand=cfg)
                else:
                    # cfg is None -> expand one level, children are leaves
                    for child in children:
                        child_rel = f"{member_rel}/{child}"
                        data[member][child] = self.get_key(child_rel)
            else:
                data[member] = self.get_key(member_rel)

        return data

    def update_keys(
            self,
            path: str,
            members: dict[str, Any],
            expand: dict[str, Any] | None = None,
            drop: bool = False,
    ) -> None:
        """
        Stores all keys and values in `members` under logical path `path`.

        `expand` is a nested dict with the following semantics:

          - If expand[key] is a dict: expand into children and pass that dict down.
          - If expand[key] is None: expand one level, children are leaves.

        Example (relative to `path`):

          expand = {"replications": {"assignments": None}}
          members = {"replications": {"r1": {"assignments": {"a": 1}}}}
          -> "/replications/r1/assignments/a" = 1

          expand = {"replications": None}
          members = {"replications": {"assignments": {"a": 1}}}
          -> "/replications/assignments" = {"a": 1}
        """
        self._ensure_running()
        full_path = self._full_path(path)

        if drop and self.client.exists(full_path):
            self.client.delete(full_path, recursive=True)

        self.client.ensure_path(full_path)

        expand = expand or {}
        base = path.rstrip("/") if path else ""

        for key, value in members.items():
            key_rel = f"{base}/{key}" if base else key

            if isinstance(expand, dict) and key in expand and isinstance(value, dict):
                cfg = expand[key]  # may be dict or None

                # Two cases:
                # - cfg is dict  -> nested expansion allowed
                # - cfg is None  -> expand one level, children are leaves
                for child_key, child_value in value.items():
                    child_rel = f"{key_rel}/{child_key}"

                    if isinstance(cfg, dict) and isinstance(child_value, dict):
                        # further nested expansion
                        self.update_keys(child_rel, child_value, expand=cfg)
                    else:
                        # either cfg is None (one level only) or child_value is scalar
                        self.update_key(child_rel, child_value)
            else:
                self.update_key(key_rel, value)

    # ----------------------------------------------------------------------
    # Queue operations
    # ----------------------------------------------------------------------

    def enqueue(self, path: str, value: Any) -> None:
        self._ensure_running()
        q = self._locking_queue(path)
        q.put(self._packb(value))

    def dequeue(
        self,
        path: str,
        timeout: Optional[float] = None,
        *,
        force_mode: Literal["raise", "release", "consume"] = "raise",
    ) -> QueueEntity | None:
        """
        Lock-and-return the head element as a QueueEntity.

        If an item is already locked in this Metastore instance:
          - force_mode="raise"   -> raise RuntimeError
          - force_mode="release" -> best-effort release() first, then proceed
          - force_mode="consume" -> best-effort consume() first, then proceed
        """
        self._ensure_running()
        q = self._locking_queue(path)

        if q.holds_lock():
            if force_mode == "raise":
                raise RuntimeError(
                    f"Queue {path!r} already holds a lock in this Metastore instance. "
                    f"Call consume() or release() on the previously dequeued QueueEntity first, "
                    f"or pass force_mode='release' or force_mode='consume'."
                )

            try:
                if force_mode == "release":
                    q.release()
                else:  # force_mode == "consume"
                    q.consume()
            except Exception as exc:
                raise RuntimeError(
                    f"Queue {path!r} holds a lock and force_mode={force_mode!r} failed: {exc!r}"
                ) from exc

        raw = q.get(timeout=timeout)
        if raw is None:
            return None

        return QueueEntity(value=self._unpackb(raw), _queue=q)

    # ----------------------------------------------------------------------
    # Leader election
    # ----------------------------------------------------------------------

    def make_leader_election(
        self,
        *,
        root_path: str,
        candidate_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> LeaderElection:
        """
        Create a ZooKeeper leader election under `root_path` (logical metastore path).

        Metastore resolves group/chroot and ensures `root_path` exists. LeaderElection
        will create/use children `<root>/election` and `<root>/leader`.
        """
        self._ensure_running()
        full_root = self._full_path(root_path)
        self.client.ensure_path(full_root)

        return LeaderElection(
            client=self.client,
            root_path=full_root,
            candidate_id=candidate_id,
            metadata=metadata,
            packb=self._packb,
        )
