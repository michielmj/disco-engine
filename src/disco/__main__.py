# src/disco/__main__.py
import pydantic_argparse

from disco.cli.commands import Commands, handle_command

if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(model=Commands, prog="disco")
    command: Commands = parser.parse_typed_args()

    handle_command(command)
