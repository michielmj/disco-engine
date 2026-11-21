from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Column, Float, Integer, MetaData, String, Table

from disco.database import df_to_table


class _RecordingSession:
    """Captures the rows df_to_table hands to the database driver."""

    def __init__(self) -> None:
        self.params: list[Any] = []

    def execute(self, stmt: Any, params: Any = None) -> None:
        self.params.append(params)


def test_df_to_table_passes_missing_values_as_none() -> None:
    """pandas 3 keeps NaN in float and str columns under `where(..., None)`; the driver must get None.

    NaN reaches PostgreSQL as the float NaN, or as the text 'nan', not as NULL.
    """
    table = Table(
        "t",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("x", Float, nullable=True),
        Column("s", String, nullable=True),
    )
    session = _RecordingSession()

    df = pd.DataFrame({"id": [1, 2], "x": [1.5, np.nan], "s": ["a", None]})
    df_to_table(session, table, df)  # type: ignore[arg-type]

    assert session.params == [[{"id": 1, "x": 1.5, "s": "a"}, {"id": 2, "x": None, "s": None}]]
