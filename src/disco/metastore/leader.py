# src/disco/metastore/leader.py
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import sleep
from typing import Any, Callable, Mapping

from kazoo.client import KazooClient
from kazoo.exceptions import KazooException, NodeExistsError, NoNodeError
from kazoo.recipe.election import Election

from tools.mp_logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LeaderRecord:
    candidate_id: str
    metadata: Mapping[str, Any]


class LeaderElection:
    """
    ZooKeeper leader election (Kazoo Election).

    Paths:
      - <root>/election : Kazoo election namespace
      - <root>/leader   : ephemeral leader record (observable for clients)
    """

    def __init__(
        self,
        *,
        client: KazooClient,
        root_path: str,
        candidate_id: str,
        metadata: Mapping[str, Any] | None,
        packb: Callable[[Any], bytes],
        retry_delay_s: float = 0.2,
    ) -> None:
        self._client = client
        self._root = root_path.rstrip("/") or "/"
        self._candidate_id = candidate_id
        self._metadata = dict(metadata or {})
        self._packb = packb
        self._retry_delay_s = retry_delay_s

        self._cancelled = Event()

        self._election_path = f"{self._root}/election"
        self._leader_path = f"{self._root}/leader"
        self._election = Election(self._client, self._election_path, identifier=candidate_id)

        # Ensure election namespace exists (Metastore ensures root exists already).
        self._client.ensure_path(self._election_path)

    @property
    def candidate_id(self) -> str:
        return self._candidate_id

    def cancel(self) -> None:
        self._cancelled.set()
        try:
            self._election.cancel()
        except Exception:
            pass

        # Best-effort cleanup of leader key (ephemeral will disappear on session loss anyway).
        try:
            self._client.delete(self._leader_path)
        except NoNodeError:
            pass
        except Exception:
            pass

    def run(self, on_lead: Callable[[], None]) -> None:
        """
        Block until cancelled; campaign for leadership. When leading, publish leader record and run on_lead().
        """
        while not self._cancelled.is_set():
            try:
                self._election.run(self._run_as_leader, on_lead)
            except KazooException as exc:
                if self._cancelled.is_set():
                    return
                logger.warning(
                    "Leader election error (candidate_id=%s, root=%s): %r",
                    self._candidate_id,
                    self._root,
                    exc,
                )
                sleep(self._retry_delay_s)

    def _run_as_leader(self, on_lead: Callable[[], None]) -> None:
        if self._cancelled.is_set():
            return

        record = LeaderRecord(candidate_id=self._candidate_id, metadata=self._metadata)
        data = self._packb(record)

        # Publish leader record as an ephemeral node.
        try:
            self._client.create(self._leader_path, data, ephemeral=True, makepath=True)
        except NodeExistsError:
            # Should not happen under correct election semantics, but handle stale persistent nodes gracefully.
            try:
                stat = self._client.exists(self._leader_path)
                if stat is not None and getattr(stat, "ephemeralOwner", 0) == 0:
                    # Persistent node: replace with ephemeral.
                    self._client.delete(self._leader_path)
                    self._client.create(self._leader_path, data, ephemeral=True, makepath=True)
                else:
                    # Ephemeral node exists (unexpected) -> just update data.
                    self._client.set(self._leader_path, data)
            except Exception:
                # If we can't publish leader record, it's safer to stop leading work.
                raise

        try:
            on_lead()
        finally:
            # Graceful relinquish.
            try:
                self._client.delete(self._leader_path)
            except NoNodeError:
                pass
            except Exception:
                pass
