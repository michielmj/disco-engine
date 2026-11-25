from __future__ import annotations

from datetime import UTC, datetime, timedelta

from graphblas import Vector
from sqlalchemy import create_engine, func, insert, select
from sqlalchemy.orm import Session

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


def test_written_masks_are_stamped_with_naive_utc() -> None:
    session = _session()
    before = _naive_utc_now()
    GraphMask(Vector.from_coo([0, 1], [True, True], size=3), "S", "m1").ensure_persisted(session)
    after = _naive_utc_now()

    stamps = session.execute(select(vertex_masks.c.updated_at)).scalars().all()
    assert len(stamps) == 2
    assert all(before <= s <= after and s.tzinfo is None for s in stamps)
    assert session.execute(select(func.count()).select_from(vertex_masks)).scalar_one() == 2
