from __future__ import annotations

import argparse
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from disco.cli.argparse_model import add_model_to_parser


class DemoModel(BaseModel):
    # required
    group: str = Field(description="Group name.")

    # optionals
    workers: Optional[int] = Field(None, description="Worker count.")
    ports: Optional[list[int]] = Field(None, description="Port list.")
    orchestrator: bool = Field(True, description="Enable orchestrator.")
    loglevel: Optional[Literal["INFO", "DEBUG"]] = Field(None, description="Log level.")


def test_add_model_to_parser_required_field_enforced() -> None:
    parser = argparse.ArgumentParser(prog="x")
    add_model_to_parser(parser, DemoModel)

    with pytest.raises(SystemExit):
        # missing required --group
        _ = parser.parse_args([])


def test_add_model_to_parser_parses_scalars_and_lists_and_choices() -> None:
    parser = argparse.ArgumentParser(prog="x")
    add_model_to_parser(parser, DemoModel)

    ns = parser.parse_args(["--group", "g1", "--workers", "3", "--ports", "5001", "5002", "--loglevel", "DEBUG"])
    assert ns.group == "g1"
    assert ns.workers == 3
    assert ns.ports == [5001, 5002]
    assert ns.loglevel == "DEBUG"


def test_add_model_to_parser_bool_flag_defaults_and_negation() -> None:
    parser = argparse.ArgumentParser(prog="x")
    add_model_to_parser(parser, DemoModel)

    ns1 = parser.parse_args(["--group", "g1"])
    assert ns1.orchestrator is True  # default from model

    ns2 = parser.parse_args(["--group", "g1", "--no-orchestrator"])
    assert ns2.orchestrator is False

    ns3 = parser.parse_args(["--group", "g1", "--orchestrator"])
    assert ns3.orchestrator is True


def test_add_model_to_parser_literal_choices_rejected() -> None:
    parser = argparse.ArgumentParser(prog="x")
    add_model_to_parser(parser, DemoModel)

    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--group", "g1", "--loglevel", "WARN"])
