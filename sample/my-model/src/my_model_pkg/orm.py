from sqlalchemy import MetaData, Table, Column, String, Integer, Float

metadata = MetaData()

stockingpoint_data = Table(
    "stockingpoint_data",
    metadata,
    Column("scenario_id", String, nullable=False),
    Column("key", String, nullable=False),
    Column("echelon", Integer, nullable=False),
    Column("location", String, nullable=False),
    Column("ip_target", Float, nullable=False),
    Column("mean_demand", Float, nullable=True),
    Column("std_demand", Float, nullable=True),
)

demand_data = Table(
    "demand_data",
    metadata,
    Column("scenario_id", String, nullable=False),
    Column("source_key", String, nullable=False),
    Column("target_key", String, nullable=False),
    Column("lead_time", String, nullable=False),
)

supply_data = Table(
    "supply_data",
    metadata,
    Column("scenario_id", String, nullable=False),
    Column("source_key", String, nullable=False),
    Column("target_key", String, nullable=False),
    # ...
)


def get_metadata() -> MetaData:
    return metadata
