# src/disco/config.py
from __future__ import annotations

import pickle
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, cast
import contextvars

from pydantic import BaseModel, Field, PostgresDsn, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource


class ConfigError(RuntimeError):
    """Configuration-related error."""
    pass


# ---------------------------------------------------------------------------
# Config file support (context + loader)
# ---------------------------------------------------------------------------

_CONFIG_FILE_CTX: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "DISCO_CONFIG_FILE_CTX",
    default=None,
)


def _find_default_config_file() -> Path | None:
    """Look for config file in current working directory."""
    cwd = Path.cwd()
    for name in ("config.toml", "config.yaml", "config.yml"):
        p = cwd / name
        if p.is_file():
            return p
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib  # Python 3.11 stdlib

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} did not parse into a dict")
    return cast(dict[str, Any], data)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as e:
        raise ConfigError(
            f"YAML config file selected ({path}), but PyYAML is not installed. "
            "Install PyYAML or use config.toml."
        ) from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} did not parse into a dict")
    return cast(dict[str, Any], data)


def _load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return _load_toml(path)
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    raise ConfigError(f"Unsupported config file type: {path} (expected .toml/.yaml/.yml)")


class _ConfigFileSettingsSource(PydanticBaseSettingsSource):
    """Settings source that loads from an optional config file.

    This source is inserted BELOW secrets and ABOVE defaults.
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Not used; we provide a full dict in __call__.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        path = _CONFIG_FILE_CTX.get()
        if path is None:
            # No config file selected/found.
            return {}

        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")

        return _load_config_file(path)


@contextmanager
def _config_file_context(path: Path | None) -> Any:
    token = _CONFIG_FILE_CTX.set(path)
    try:
        yield
    finally:
        _CONFIG_FILE_CTX.reset(token)


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
            "Should be routable in-cluster (e.g. Pod IP). "
            "If None, Server will attempt to infer a non-loopback IP."
        ),
    )
    timeout_s: float = Field(600.0, description="Default timeout for gRPC calls in seconds.")
    max_workers: int = Field(10, description="Maximum number of worker threads for the gRPC server.")
    grace_s: float = Field(60.0, description="Grace period in seconds for server shutdown.")

    max_send_message_bytes: int | None = Field(default=None, description="Maximum send message size in bytes (None = default).")
    max_receive_message_bytes: int | None = Field(default=None, description="Maximum receive message size in bytes (None = default).")

    keepalive_time_s: float | None = Field(default=None, description="Time between keepalive pings in seconds (None = disabled).")
    keepalive_timeout_s: float | None = Field(default=None, description="Timeout for keepalive pings in seconds (None = default).")
    keepalive_permit_without_calls: bool = Field(False, description="Allow keepalive pings even when there are no active calls.")

    compression: Literal["none", "gzip"] = Field("none", description="Compression algorithm for gRPC calls.")

    promise_retry_delays_s: list[float] = Field(
        default_factory=lambda: [0.05, 0.15, 0.5, 1.0, 2.0],
        description="Backoff sequence for promise retries in seconds.",
    )
    promise_retry_max_window_s: float = Field(3.0, description="Maximum time window for promise delivery retries in seconds.")


class ZookeeperSettings(BaseModel):
    hosts: str = Field("localhost:2181", description="Comma-separated host:port pairs for Zookeeper ensemble.")
    chroot: str | None = Field(default=None, description="Optional chroot path, e.g. /disco.")
    default_group: str = Field("default", description="Default logical group / namespace used in the app.")
    session_timeout_s: float = Field(10.0, description="Zookeeper session timeout in seconds.")
    connection_timeout_s: float = Field(5.0, description="Initial connection timeout in seconds.")
    max_retries: int = Field(5, description="Maximum number of retry attempts for failed operations.")
    retry_delay_s: float = Field(1.0, description="Delay between retries in seconds.")
    auth_scheme: str | None = Field(default=None, description="Optional auth scheme, e.g. 'digest'.")
    auth_credentials: str | None = Field(default=None, description="Optional auth credentials, e.g. 'user:password'.")
    use_tls: bool = Field(False, description="Enable TLS/SSL for Zookeeper connection.")
    ca_cert: str | None = Field(default=None, description="Path to CA certificate file for TLS, if applicable.")
    client_cert: str | None = Field(default=None, description="Path to client certificate file for mutual TLS.")
    client_key: str | None = Field(default=None, description="Path to client private key for mutual TLS.")


class DatabaseSettings(BaseModel):
    url: PostgresDsn = Field(
        cast(PostgresDsn, "postgresql+psycopg://user:password@localhost:5432/disco"),
        description="SQLAlchemy-style database URL.",
    )
    pool_size: int = 10
    max_overflow: int = 20


class DataLoggerSettings(BaseModel):
    path: DirectoryPath = Field(".", description="Location for storing data logger files.")
    ring_bytes: int = Field(1 << 27, description="Ring buffer size (default=128 MB)")
    rotate_bytes: int = Field(256 << 20, description="Segment file size (default=256 MB)")
    zstd_level: int = Field(1, description="ZSTD Compression level (default=1)")


class ModelSettings(BaseModel):
    plugin: Optional[str] = Field(None, description="...")
    package: Optional[str] = Field(None, description="...")
    path: Optional[DirectoryPath] = Field(None, description="...")  # fix your trailing comma issue if present in your file
    dev_import_root: Optional[DirectoryPath] = Field(None, description="...")
    model_yml: str = Field("model.yml", description="Name of the model definition file (default=model.yml).")

    def validate_model(self) -> None:
        if len([o is not None for o in [self.plugin, self.package, self.path]]) != 1:
            raise ValueError("Invalid model configuration. Exactly one model option needs to be provided.")


class SerializationSettings(BaseModel):
    protocol: int = Field(
        pickle.HIGHEST_PROTOCOL,
        description="Pickle protocol version.",
    )

    def validate_protocol(self) -> None:
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
    5. Config file
    6. Defaults in this class
    """

    model_config = SettingsConfigDict(
        env_prefix="DISCO_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        secrets_dir="/run/secrets/disco",
        extra="ignore",
        validate_default=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources override later sources.
        # We want: init > env > dotenv > secrets > configfile > defaults
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            _ConfigFileSettingsSource(settings_cls),
        )

    app_name: str = "disco"
    grace_s: int = 600
    debug: bool = False

    logging: LoggingSettings = LoggingSettings()
    grpc: GrpcSettings = GrpcSettings()
    zookeeper: ZookeeperSettings = ZookeeperSettings()
    database: DatabaseSettings = DatabaseSettings()
    serialization: SerializationSettings = SerializationSettings()
    data_logger: DataLoggerSettings = DataLoggerSettings()
    model: ModelSettings = ModelSettings()
    orchestrator: OrchestratorSettings = OrchestratorSettings()


@lru_cache(maxsize=16)
def _get_settings_cached(config_file_str: str | None, overrides_blob: bytes) -> AppSettings:
    overrides = pickle.loads(overrides_blob)
    config_path = Path(config_file_str) if config_file_str is not None else None
    with _config_file_context(config_path):
        settings = AppSettings(**overrides)
    settings.serialization.validate_protocol()
    settings.model.validate_model()
    return settings


def get_settings(*, config_file: str | Path | None = None, **overrides: Any) -> AppSettings:
    """
    Cached accessor for process-wide settings.

    `overrides` are init kwargs → highest precedence (handy in tests).

    If `config_file` is None, we look in CWD for: config.toml, config.yaml, config.yml.
    If none found, config-file source is disabled and defaults apply.
    """
    resolved: Path | None
    if config_file is None:
        resolved = _find_default_config_file()
    else:
        resolved = Path(config_file)

    # Cache key includes config file and overrides.
    overrides_blob = pickle.dumps(overrides, protocol=pickle.HIGHEST_PROTOCOL)
    return _get_settings_cached(str(resolved) if resolved is not None else None, overrides_blob)


def clear_settings_cache() -> None:
    _get_settings_cached.cache_clear()
