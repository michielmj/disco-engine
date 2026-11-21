"""The stop hook keeps an agent working while something it started still runs, and only then."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_strays", HOOKS / "check_strays.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


strays = _load()

# The implementer's launch prompt, as the /implement skill writes it.
PROMPT = """<task>
branch: m1a-export
worktree: {worktree}
brief: specs/briefs/m1a-export.md
ports: 8110-8119
run settings: none
budget: 90 minutes, until 14:30
</task>
<parallel_sessions>
m1b-import: the importer, ports 8120-8129
</parallel_sessions>
Build the brief, as your agent definition describes."""


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    path = tmp_path / ".claude" / "worktrees" / "m1a-export"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def transcript(tmp_path: Path, worktree: Path) -> Path:
    path = tmp_path / "agent-1.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": PROMPT.format(worktree=worktree)}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "<task>\nworktree: /elsewhere\n</task>"}]},
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_the_task_comes_from_the_launch_prompt(transcript: Path, worktree: Path) -> None:
    task = strays.task_from_transcript(str(transcript))
    assert task is not None
    assert (task.branch, strays.normalize(task.worktree)) == ("m1a-export", strays.normalize(str(worktree)))
    assert task.ports == tuple(range(8110, 8120))


@pytest.mark.parametrize(
    "text",
    ["no task here", "<task>\nbranch: x\n</task>", "<task>\nworktree: <absolute worktree path>\n</task>"],
)
def test_no_usable_task_means_no_task(text: str) -> None:
    assert strays.parse_task(text) is None


def test_a_missing_transcript_means_no_task(tmp_path: Path) -> None:
    assert strays.task_from_transcript(str(tmp_path / "missing.jsonl")) is None


def test_a_path_inside_the_worktree_counts_and_its_neighbour_does_not(worktree: Path) -> None:
    assert strays.inside(str(worktree / "src"), str(worktree))
    assert strays.inside(str(worktree), str(worktree))
    assert not strays.inside(str(worktree) + "-old", str(worktree))
    assert not strays.inside(str(worktree.parent), str(worktree))
    assert not strays.inside("", str(worktree))


@pytest.mark.skipif(os.name != "nt", reason="Git Bash paths exist only on Windows")
def test_a_git_bash_path_is_the_same_path() -> None:
    assert strays.inside(
        "/c/Users/dev/repo/.claude/worktrees/x/src", "C:\\Users\\dev\\repo\\.claude\\worktrees\\x"
    )


def _task(worktree: Path) -> object:
    return strays.Task("m1a-export", str(worktree), tuple(range(8110, 8120)))


def test_containers_of_the_task_are_strays_and_others_are_not(worktree: Path) -> None:
    containers = [
        {"Names": "test-linux-m1a-export-4242", "Labels": "", "Status": "Up 2 minutes"},
        {
            "Names": "test-linux-suite",
            "Labels": f"test-linux.branch=m1a-export,test-linux.worktree={worktree}",
        },
        {"Names": "m1a-export-postgres", "Labels": ""},
        {"Names": "m1a-exporter", "Labels": ""},
        {"Names": "test-linux-suite", "Labels": "test-linux.worktree=/other/worktree"},
        {"Names": "owner-postgres", "Labels": "com.docker.compose.project=owner"},
    ]
    found = strays.stray_containers(containers, _task(worktree))
    assert [s.split()[1] for s in found] == [
        "test-linux-m1a-export-4242",
        "test-linux-suite",
        "m1a-export-postgres",
    ]


def test_processes_of_the_task_are_strays_and_others_are_not(worktree: Path, tmp_path: Path) -> None:
    processes = [
        strays.Proc(101, "python", str(worktree / "src"), "python -m http.server"),
        strays.Proc(102, "node", str(tmp_path), f"node {worktree / 'frontend' / 'vite.js'}"),
        strays.Proc(103, "uvicorn", str(tmp_path), "uvicorn app:main", (8112,)),
        strays.Proc(104, "uvicorn", str(tmp_path), "uvicorn app:main", (8000,)),
        strays.Proc(105, "python", str(tmp_path / "other"), "python other.py", (8121,)),
        strays.Proc(106, "uv", str(worktree), "uv run python check_strays.py"),
    ]
    found = strays.stray_processes(processes, _task(worktree), exclude={106})
    assert [s.split()[1] for s in found] == ["101", "102", "103"]
    assert "listening on 8112" in found[2]


def test_decide_blocks_with_the_list(transcript: Path, worktree: Path) -> None:
    event = {"agent_transcript_path": str(transcript), "stop_hook_active": False}
    reason = strays.decide(
        event,
        lambda: [{"Names": "m1a-export-postgres", "Labels": ""}],
        lambda: ([strays.Proc(101, "python", str(worktree), "python -m http.server")], set()),
    )
    assert reason is not None
    assert "m1a-export-postgres" in reason and "pid 101" in reason
    assert "never by process name" in reason


def test_decide_lets_a_clean_agent_stop(transcript: Path) -> None:
    event = {"agent_transcript_path": str(transcript)}
    assert strays.decide(event, lambda: [], lambda: ([], set())) is None


def test_decide_blocks_only_once(transcript: Path, worktree: Path) -> None:
    event = {"agent_transcript_path": str(transcript), "stop_hook_active": True}
    leftover = [strays.Proc(101, "python", str(worktree), "python")]
    assert strays.decide(event, lambda: [], lambda: (leftover, set())) is None


@pytest.mark.parametrize("payload", ["not json", "[]", "{}", '{"agent_transcript_path": 3}'])
def test_unreadable_input_lets_the_agent_stop(payload: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(HOOKS / "check_strays.py")],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (proc.returncode, proc.stderr) == (0, "")
