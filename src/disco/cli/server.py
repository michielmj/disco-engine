# src/disco/cli/server.py
from typing import Literal, Optional

import pydantic_argparse
from pydantic import Field, IPvAnyAddress, BaseModel


class ServerCommand(BaseModel):
    group: str = Field(description="The group for the Server to run in.")
    workers: Optional[int] = Field(None, description="Number of workers to start.")
    ports: Optional[list[int]] = Field(None, description="Ports to run servers on.")
    bind_host: Optional[IPvAnyAddress] = Field(None, description="Bind host.")
    grace_s: Optional[int] = Field(None, description="Grace duration for shutdown.")
    orchestrator: bool = Field(True, description="Disable orchestrator if false.")
    config_file: Optional[str] = Field(None, description="Optional Disco config file (toml/yaml).")
    loglevel: Optional[Literal[
        "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning", "info", "debug"]] = Field(
        None, description="Logging level override."
    )


def handle_server(command: ServerCommand):
    from disco.config import get_settings
    overrides = {}
    if command.loglevel is not None:
        overrides["logging"] = {"level": command.loglevel.upper()}
    settings = get_settings(config_file=command.config_file, **overrides)

    from disco.server import Server

    server = Server(
        settings=settings,
        workers=command.workers,
        ports=command.ports,
        bind_host=str(command.bind_host) if command.bind_host is not None else None,
        group=command.group,
        grace_s=command.grace_s,
        orchestrator=command.orchestrator
    )

    server.start()


def main() -> None:
    parser = pydantic_argparse.ArgumentParser(model=ServerCommand, prog="disco-server")
    command: ServerCommand = parser.parse_typed_args()
    handle_server(command)
