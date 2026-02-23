from sqlalchemy import MetaData, Table, Column, String, Integer, Float, JSON

metadata = MetaData()

stockingpoint_data = Table(
    "stockingpoint_data",
    metadata,
    Column("scenario_id", String, nullable=False),
    Column("key", String, nullable=False),
    Column("echelon", Integer, nullable=False),
    Column("location", String, nullable=False),
    Column("ip_target", Float, nullable=False),
    Column("dem_dist", String, nullable=True),
    Column("dem_params", JSON, nullable=True),
)

demand_data = Table(
    "demand_data",
    metadata,
    Column("scenario_id", String, nullable=False),
    Column("source_key", String, nullable=False),
    Column("target_key", String, nullable=False),
    Column("lead_time", Float, nullable=False),
)

supply_data = Table(
    "supply_data",
    metadata,
    Column("scenario_id", String, nullable=False),
    Column("source_key", String, nullable=False),
    Column("target_key", String, nullable=False),
    Column("lt_mean", Float, nullable=False),
    Column("lt_var", Float, nullable=False),
    # ...
)


def get_metadata() -> MetaData:
    return metadata
