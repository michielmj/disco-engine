from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Union, cast

from sqlalchemy import create_engine, Connection
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from disco.config import DatabaseSettings


@dataclass(slots=True)
class SessionManager:
    engine: Engine

    @classmethod
    def from_settings(cls, s: DatabaseSettings) -> "SessionManager":
        engine = create_engine(
            str(s.url),
            pool_size=s.pool_size,
            max_overflow=s.max_overflow,
            pool_pre_ping=True,
            future=True,
        )
        return cls(engine=engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = Session(self.engine)
        try:
            yield sess
        finally:
            sess.close()


DbHandle = Union[Engine, Connection, SessionManager]


def normalize_db_handle(db: DbHandle) -> Engine | Connection:
    if isinstance(db, SessionManager):
        return db.engine
    return cast(Union[Engine, Connection], db)
