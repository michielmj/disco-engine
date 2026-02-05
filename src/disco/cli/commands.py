# src/disco/cli/commands.py
from typing import Optional

from pydantic import BaseModel, Field

from disco.exceptions import DiscoError
from .server import ServerCommand, handle_server


class CommandLineError(DiscoError): ...


class Commands(BaseModel):
    server: Optional[ServerCommand] = Field(None, description="Start a Disco Server.")


def handle_command(command: Commands):
    if command.server:
        handle_server(command.server)

    else:
        raise CommandLineError('')
