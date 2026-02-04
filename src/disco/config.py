# src/disco/config.py
from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any, Literal, cast, Optional

from pydantic import BaseModel, Field, PostgresDsn, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Configuration-related error."""
    pass


# ─────────────────────────────────────────────────────────────
# Section configs
# ─────────────────────────────────────────────────────────────


class LoggingSettings(BaseModel):
    level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    format: str = (
        "%(asctime)-20s %(processName)-30s %(name)-40s "
        "%(levelname)-8s: %(message)s"
    )


class GrpcSettings(BaseModel):
    bind_host: str | None = Field(
        default=None,
        description=(
            "Address used for worker gRPC endpoints and worker addresses (host:port). "
            "Should be routable in-cluster (e.g. Pod IP or node IP). "
            "If None, Server will attempt to infer a non-loopback IP."
        ),
    )
    timeout_s: float = Field(
        600.0, description="Default timeout for gRPC calls in seconds."
    )
    max_workers: int = Field(
        10, description="Maximum number of worker threads for the gRPC server."
    )
    grace_s: float = Field(
        60.0, description="Grace period in seconds for server shutdown."
    )

    max_send_message_bytes: int | None = Field(
        default=None,
        description="Maximum send message size in bytes (None = default).",
    )
    max_receive_message_bytes: int | None = Field(
        default=None,
        description="Maximum receive message size in bytes (None = default).",
    )

    keepalive_time_s: float | None = Field(
        default=None,
        description="Time between keepalive pings in seconds (None = disabled).",
    )
    keepalive_timeout_s: float | None = Field(
        default=None,
        description="Timeout for keepalive pings in seconds (None = default).",
    )
    keepalive_permit_without_calls: bool = Field(
        False,
        description="Allow keepalive pings even when there are no active calls.",
    )

    compression: Literal["none", "gzip"] = Field(
        "none",
        description="Compression algorithm for gRPC calls.",
    )

    # Example additional fields (names and exact types may be adjusted in code):
    promise_retry_delays_s: list[float] = Field(
        default_factory=lambda: [0.05, 0.15, 0.5, 1.0, 2.0],
        description="Backoff sequence for promise retries in seconds.",
    )
    promise_retry_max_window_s: float = Field(
        3.0,
        description="Maximum time window for promise delivery retries in seconds.",
    )


class ZookeeperSettings(BaseModel):
    hosts: str = Field(
        "localhost:2181",
        description="Comma-separated host:port pairs for Zookeeper ensemble.",
    )
    chroot: str | None = Field(
        default=None,
        description="Optional chroot path, e.g. /disco.",
    )

    default_group: str = Field(
        "default",
        description="Default logical group / namespace used in the app.",
    )

    session_timeout_s: float = Field(
        10.0,
        description="Zookeeper session timeout in seconds.",
    )
    connection_timeout_s: float = Field(
        5.0,
        description="Initial connection timeout in seconds.",
    )

    max_retries: int = Field(
        5,
        description="Maximum number of retry attempts for failed operations.",
    )
    retry_delay_s: float = Field(
        1.0,
        description="Delay between retries in seconds.",
    )

    auth_scheme: str | None = Field(
        default=None,
        description="Optional auth scheme, e.g. 'digest'.",
    )
    auth_credentials: str | None = Field(
        default=None,
        description="Optional auth credentials, e.g. 'user:password'.",
    )

    use_tls: bool = Field(
        False,
        description="Enable TLS/SSL for Zookeeper connection.",
    )
    ca_cert: str | None = Field(
        default=None,
        description="Path to CA certificate file for TLS, if applicable.",
    )
    client_cert: str | None = Field(
        default=None,
        description="Path to client certificate file for mutual TLS.",
    )
    client_key: str | None = Field(
        default=None,
        description="Path to client private key for mutual TLS.",
    )


class DatabaseSettings(BaseModel):
    """
    DB config as a SQLAlchemy URL.

    In production, override via:
    - env var:     DISCO_DATABASE__URL
    - dotenv:      .env / .env.local
    - secret file: /run/secrets/disco/database__url
    """
    url: PostgresDsn = Field(
        cast(PostgresDsn, "postgresql+psycopg://user:password@localhost:5432/disco"),
        description="SQLAlchemy-style database URL.",
    )
    pool_size: int = 10
    max_overflow: int = 20


class DataLoggerSettings(BaseModel):
    """
    Settings for data logging
    """

    path: DirectoryPath = Field(".", description="Location for storing data logger files.")
    ring_bytes: int = Field(1 << 27, description="Ring buffer size (default=128 MB)")
    rotate_bytes: int = Field(256 << 20, description="Segment file size (default=256 MB)")
    zstd_level: int = Field(1, description="ZSTD Compression level (default=1)")


class ModelSettings(BaseModel):
    """
    Settings for simulation model
    """
    plugin: Optional[str] = Field(None, description='Name of the model entry point defined in the disco.models group. '
                                                    'The entry point should point to a method "epmodule:model_package" '
                                                    'that returns the name of the model package.')
    package: Optional[str] = Field(None, description='Name of a package containing the model.')
    path: Optional[DirectoryPath] = Field(None, description='Load a model from a folder on disk (dev fallback). '
                                                            'Expects path/model.yml.'),
    dev_import_root: Optional[DirectoryPath] = Field(
        None,
        description='In combination with path, temporarily adds `import_root` to sys.path to allow imports. Defaults '
                    'to path.'
    )
    model_yml: str = Field("model.yml", description='Name of the model definition file (default=model.yml).')

    def validate_model(self):
        if len([o is not None for o in [self.plugin, self.package, self.path]]) != 1:
            raise ValueError('Invalid model configuration. Exactly one model option needs to be provided.')


class SerializationSettings(BaseModel):
    """
    Serialization settings for pickle usage.

    protocol:
        - 0–5 in Python 3.8
        - 0–6 in Python 3.12+
    Default: highest available protocol for the running interpreter.
    """

    protocol: int = Field(
        pickle.HIGHEST_PROTOCOL,
        description=(
            "Pickle protocol version. Defaults to the highest supported "
            "by this Python version."
        ),
    )

    def validate_protocol(self) -> None:
        """Ensure user-specified protocol is supported by this interpreter."""
        import pickle as _pickle

        if not (0 <= self.protocol <= _pickle.HIGHEST_PROTOCOL):
            raise ValueError(
                f"Unsupported pickle protocol={self.protocol}. "
                f"This interpreter supports up to protocol={_pickle.HIGHEST_PROTOCOL}."
            )


class OrchestratorSettings(BaseModel):
    launch_timeout_s: float = Field(
        60,
        description="Maximum time allowed for workers to pass from AVAILABLE to ACTIVE after assignment.",
        gt=0,
    )


# ─────────────────────────────────────────────────────────────
# Top-level settings
# ─────────────────────────────────────────────────────────────


class AppSettings(BaseSettings):
    """
    Canonical application configuration for a disco-based service.

    Precedence (highest → lowest):

    1. Init kwargs (tests/overrides)
    2. Environment variables
    3. .env and .env.local
    4. Secret files in /run/secrets/disco
    5. Defaults in this class
    """

    model_config = SettingsConfigDict(
        env_prefix="DISCO_",  # DISCO_LOGGING__LEVEL, DISCO_DATABASE__URL, ...
        env_file=(".env", ".env.local"),  # .env.local overrides .env
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        secrets_dir="/run/secrets/disco",
        extra="ignore",
        validate_default=True,
    )

    app_name: str = "disco"
    grace_s: int = 600  # grace period for SIGTERM
    debug: bool = False

    logging: LoggingSettings = LoggingSettings()
    grpc: GrpcSettings = GrpcSettings()
    zookeeper: ZookeeperSettings = ZookeeperSettings()
    database: DatabaseSettings = DatabaseSettings()
    serialization: SerializationSettings = SerializationSettings()
    data_logger: DataLoggerSettings = DataLoggerSettings()
    model: ModelSettings()
    orchestrator: OrchestratorSettings()


@lru_cache(maxsize=1)
def get_settings(**overrides: Any) -> AppSettings:
    """
    Cached accessor for process-wide settings.

    `overrides` are init kwargs → highest precedence (handy in tests).
    """
    settings = AppSettings(**overrides)
    settings.serialization.validate_protocol()
    settings.model.validate_model()
    return settings
