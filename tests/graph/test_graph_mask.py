from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from graphblas import Vector
from sqlalchemy import create_engine, func, insert, select
from sqlalchemy.orm import Session

from disco.graph import graph_mask
from disco.graph.graph_mask import GraphMask
from disco.graph.schema import create_graph_schema, vertex_masks


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    create_graph_schema(engine)
    return Session(engine)


def test_cleanup_deletes_only_masks_older_than_the_limit() -> None:
    # Rows as an earlier version wrote them: naive UTC datetimes.
    session = _session()
    now = _naive_utc_now()
    session.execute(
        insert(vertex_masks),
        [
            {
                "scenario_id": "S",
                "mask_id": "fresh",
                "vertex_index": 0,
                "updated_at": now - timedelta(minutes=50),
            },
            {
                "scenario_id": "S",
                "mask_id": "old",
                "vertex_index": 0,
                "updated_at": now - timedelta(minutes=70),
            },
        ],
    )

    GraphMask.cleanup_old(session, max_age_minutes=60)

    assert set(session.execute(select(vertex_masks.c.mask_id)).scalars()) == {"fresh"}


def test_written_masks_are_stamped_with_naive_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    # SQLite drops tzinfo, so what it returns proves nothing: record what the
    # code stamps, and require that every write takes its stamp from there.
    stamped: list[datetime] = []
    real = graph_mask._utcnow

    def recording() -> datetime:
        now = real()
        stamped.append(now)
        return now

    monkeypatch.setattr(graph_mask, "_utcnow", recording)
    session = _session()
    mask = GraphMask(Vector.from_coo([0, 1], [True, True], size=3), "S", "m1")
    mask.ensure_persisted(session)
    mask.ensure_persisted(session)  # the second call touches updated_at
    GraphMask.cleanup_old(session, max_age_minutes=60)

    assert len(stamped) >= 3
    assert all(s.tzinfo is None for s in stamped)
    assert all(abs(s - _naive_utc_now()) < timedelta(minutes=1) for s in stamped)
    assert session.execute(select(func.count()).select_from(vertex_masks)).scalar_one() == 2
