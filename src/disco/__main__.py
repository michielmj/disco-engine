# src/disco/__main__.py
from __future__ import annotations

import argparse

from disco.cli.argparse_model import add_model_to_parser
from disco.cli.server import ServerCommand, handle_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="disco")
    sub = parser.add_subparsers(dest="command", required=True)

    server_p = sub.add_parser("server", help="Start a Disco Server.")
    add_model_to_parser(server_p, ServerCommand)

    ns = parser.parse_args(argv)

    if ns.command == "server":
        data = vars(ns)
        data.pop("command", None)
        cmd = ServerCommand.model_validate(data)
        handle_server(cmd)
        return

    raise RuntimeError(f"Unknown command: {ns.command}")


if __name__ == "__main__":
    main()
