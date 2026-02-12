from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Union, cast, Self

import pandas as pd
from sqlalchemy import create_engine, Connection, insert, delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from disco.config import DatabaseSettings


@dataclass(slots=True)
class SessionManager:
    engine: Engine

    @classmethod
    def from_settings(cls, s: DatabaseSettings) -> Self:
        url = s.sqlalchemy_url()
        engine = create_engine(
            url,
            pool_size=s.pool_size,
            max_overflow=s.max_overflow,
            pool_pre_ping=True,
            future=True,
        )
        return cls(engine=engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = Session(self.engine, future=True)
        try:
            yield sess
            # If we reach here without exception -> commit
            sess.commit()
        except Exception:
            # On error -> rollback
            sess.rollback()
            raise
        finally:
            sess.close()


DbHandle = Union[Engine, Connection, SessionManager]


def normalize_db_handle(db: DbHandle) -> Engine | Connection:
    if isinstance(db, SessionManager):
        return db.engine
    return cast(Union[Engine, Connection], db)


def df_to_table(
    session: Session,
    table,
    df: pd.DataFrame,
    *,
    clear: bool = True,
    chunk_size: int = 10_000,
):
    if clear:
        session.execute(delete(table))  # portable "clear table"

    cols = {c.name for c in table.columns}
    df2 = df[[c for c in df.columns if c in cols]].copy()
    df2 = df2.where(pd.notnull(df2), None)  # NaN/NA -> None

    records = df2.to_dict(orient="records")
    stmt = insert(table)

    if chunk_size and len(records) > chunk_size:
        for i in range(0, len(records), chunk_size):
            session.execute(stmt, records[i : i + chunk_size])
    else:
        session.execute(stmt, records)
