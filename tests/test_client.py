# tests/test_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pytest

from disco.client import Client
from disco.experiments import ExperimentStatus


@dataclass
class _FakeReplication:
    repid: str
    status: ExperimentStatus


@dataclass
class _FakeExperiment:
    status: ExperimentStatus
    replications: Dict[str, _FakeReplication]


class _FakeStore:
    def __init__(self, expids: list[str], experiments: dict[str, _FakeExperiment]):
        self._expids = expids
        self._experiments = experiments

    def list_expids(self) -> list[str]:
        return list(self._expids)

    def load(self, expid: str) -> _FakeExperiment:
        if expid not in self._experiments:
            raise KeyError(expid)
        return self._experiments[expid]


def _client_with_store(store: _FakeStore) -> Client:
    # Avoid depending on Client.__init__ signature; we only need `store`.
    c = Client.__new__(Client)
    c.store = store
    return c


def test_list_experiments_returns_mapping() -> None:
    store = _FakeStore(
        expids=["e1", "e2"],
        experiments={
            "e1": _FakeExperiment(status=ExperimentStatus.SUBMITTED, replications={}),
            "e2": _FakeExperiment(status=ExperimentStatus.ACTIVE, replications={}),
        },
    )
    client = _client_with_store(store)

    got = client.list_experiments()

    assert got == {
        "e1": ExperimentStatus.SUBMITTED,
        "e2": ExperimentStatus.ACTIVE,
    }


def test_list_experiments_ignores_missing_between_list_and_load() -> None:
    store = _FakeStore(
        expids=["e1", "e2"],  # e2 will be missing at load time
        experiments={
            "e1": _FakeExperiment(status=ExperimentStatus.SUBMITTED, replications={}),
        },
    )
    client = _client_with_store(store)

    got = client.list_experiments()

    assert got == {"e1": ExperimentStatus.SUBMITTED}


def test_list_replications_returns_mapping() -> None:
    store = _FakeStore(
        expids=["e1"],
        experiments={
            "e1": _FakeExperiment(
                status=ExperimentStatus.SUBMITTED,
                replications={
                    "r1": _FakeReplication(repid="r1", status=ExperimentStatus.ASSIGNED),
                    "r2": _FakeReplication(repid="r2", status=ExperimentStatus.ACTIVE),
                },
            )
        },
    )
    client = _client_with_store(store)

    got = client.list_replications("e1")

    assert got == {
        "r1": ExperimentStatus.ASSIGNED,
        "r2": ExperimentStatus.ACTIVE,
    }


def test_list_replications_raises_for_missing_experiment() -> None:
    store = _FakeStore(expids=[], experiments={})
    client = _client_with_store(store)

    with pytest.raises(KeyError):
        client.list_replications("missing")
