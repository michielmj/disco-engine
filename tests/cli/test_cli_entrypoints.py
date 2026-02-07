from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

import disco.__main__ as disco_main
import disco.cli.server as server_cli


@dataclass(slots=True)
class Captured:
    called: bool = False
    args: Any = None


def test_disco_main_server_subcommand_calls_handle_server(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = Captured()

    def fake_handle_server(cmd: Any) -> None:
        cap.called = True
        cap.args = cmd

    # Patch the function used by disco.__main__.main
    monkeypatch.setattr(disco_main, "handle_server", fake_handle_server)

    disco_main.main(
        [
            "server",
            "--group",
            "g1",
            "--workers",
            "2",
            "--ports",
            "5001",
            "5002",
            "--no-orchestrator",
        ]
    )

    assert cap.called is True
    cmd = cap.args
    assert cmd.group == "g1"
    assert cmd.workers == 2
    assert cmd.ports == [5001, 5002]
    assert cmd.orchestrator is False


def test_disco_server_entrypoint_calls_handle_server(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = Captured()

    def fake_handle_server(cmd: Any) -> None:
        cap.called = True
        cap.args = cmd

    monkeypatch.setattr(server_cli, "handle_server", fake_handle_server)

    server_cli.main(
        [
            "--group",
            "g2",
            "--workers",
            "3",
            "--ports",
            "6001",
            "6002",
            "--orchestrator",
        ]
    )

    assert cap.called is True
    cmd = cap.args
    assert cmd.group == "g2"
    assert cmd.workers == 3
    assert cmd.ports == [6001, 6002]
    assert cmd.orchestrator is True


def test_disco_main_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        # argparse should exit because subcommand is required
        disco_main.main([])
