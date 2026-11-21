"""Stop check for the implementer and the reviewer: nothing an agent starts outlives it.

It is declared as a Stop hook in their frontmatter, which Claude Code runs as
SubagentStop for that agent only. The agent's branch, worktree and ports come
from the <task> block of its launch prompt, read from its own transcript
(`agent_transcript_path`). Inside a subagent `cd` does not persist, so the
hook's cwd says nothing about where the agent worked.

A stray is:
- a container labelled with the agent's worktree (scripts/test_linux.sh
  labels its own), or named after its branch (run-tests: name the containers
  you start after your branch);
- a process of this user whose working directory or command line is inside
  the worktree, or that listens on one of the agent's ports.

Exit code 2 keeps the agent working, with the list as its next instruction.
It blocks once: when the agent stops again (`stop_hook_active`), the hook lets
it, and the architect's stray check after the report catches what remains.
Unlike the shell guards this fails open: a hook that cannot read its input or
the transcript lets the agent stop, because trapping an agent is worse than
missing a stray.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

TASK = re.compile(r"<task>(.*?)</task>", re.DOTALL)
PORT_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")


@dataclass(frozen=True)
class Task:
    branch: str
    worktree: str
    ports: tuple[int, ...]


@dataclass(frozen=True)
class Proc:
    pid: int
    name: str
    cwd: str
    cmdline: str
    ports: tuple[int, ...] = field(default=())


def parse_ports(text: str) -> tuple[int, ...]:
    """`8110-8119` (new_task.sh's block) or single numbers."""
    ports: list[int] = []
    for start, end in PORT_RANGE.findall(text):
        ports.extend(range(int(start), int(end) + 1))
    if not ports:
        ports = [int(n) for n in re.findall(r"\d+", text)]
    return tuple(ports)


def parse_task(text: str) -> Task | None:
    """The branch, worktree and ports of the first <task> block in `text`."""
    match = TASK.search(text)
    if not match:
        return None
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip().lower(), value.strip())
    worktree = fields.get("worktree", "")
    if not worktree or worktree.startswith("<"):
        return None
    return Task(fields.get("branch", ""), worktree, parse_ports(fields.get("ports", "")))


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def task_from_transcript(path: str) -> Task | None:
    """The task of the agent's launch prompt: the first <task> block in its transcript."""
    try:
        with open(path, encoding="utf-8") as transcript:
            for line in transcript:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                for text in _strings(entry):
                    task = parse_task(text)
                    if task:
                        return task
    except OSError:
        return None
    return None


def normalize(path: str) -> str:
    """One spelling per path: Git Bash's /c/... becomes C:/..., then the OS's case and separators."""
    drive = re.match(r"^/([a-zA-Z])/(.*)$", path)
    if os.name == "nt" and drive:
        path = f"{drive.group(1)}:/{drive.group(2)}"
    return os.path.normcase(os.path.normpath(path))


def inside(path: str, root: str) -> bool:
    if not path:
        return False
    path, root = normalize(path), normalize(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _branch_token(branch: str) -> re.Pattern[str] | None:
    """The branch as test_linux.sh spells it in names, as a whole token.

    `main` must match `test-linux-main-42`, but not `maintenance`.
    """
    if not branch:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", branch)
    return re.compile(rf"(?:^|[-_.]){re.escape(safe)}(?:$|[-_.])")


def stray_containers(containers: list[dict[str, str]], task: Task) -> list[str]:
    """Running containers that belong to the task, as `docker ps --format '{{json .}}'` lists them."""
    token = _branch_token(task.branch)
    strays = []
    for container in containers:
        name = container.get("Names", "")
        labels = dict(
            item.partition("=")[::2] for item in container.get("Labels", "").split(",") if "=" in item
        )
        if inside(labels.get("test-linux.worktree", ""), task.worktree) or (token and token.search(name)):
            strays.append(f"container {name} ({container.get('Status', '')})")
    return strays


def stray_processes(processes: list[Proc], task: Task, exclude: set[int]) -> list[str]:
    """Processes running in the task's worktree or listening on its ports."""
    root = normalize(task.worktree)
    strays = []
    for proc in processes:
        if proc.pid in exclude:
            continue
        listening = sorted(set(proc.ports) & set(task.ports))
        if inside(proc.cwd, task.worktree) or root in normalize(proc.cmdline) or listening:
            where = f", listening on {', '.join(map(str, listening))}" if listening else ""
            strays.append(f"pid {proc.pid} {proc.name}: {proc.cmdline[:120]}{where}")
    return strays


def decide(
    event: dict[str, object],
    list_containers: Callable[..., list[dict[str, str]]],
    list_processes: Callable[..., tuple[list[Proc], set[int]]],
) -> str | None:
    """The reason to keep the agent working, or None to let it stop."""
    transcript = event.get("agent_transcript_path")
    if event.get("stop_hook_active") or not isinstance(transcript, str):
        return None
    task = task_from_transcript(transcript)
    if task is None:
        return None
    processes, exclude = list_processes()
    strays = stray_containers(list_containers(), task) + stray_processes(processes, task, exclude)
    if not strays:
        return None
    return (
        "You started these and they still run. Stop each one by pid or by container name "
        "(never by process name), check that none remains, and list them in your report:\n"
        + "\n".join(f"- {stray}" for stray in strays)
    )


def docker_containers() -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    # Two clauses, not a tuple: ruff formats a tuple differently for Python 3.14 (PEP 758).
    except OSError:
        return []
    except subprocess.TimeoutExpired:
        return []
    containers = []
    for line in proc.stdout.splitlines():
        try:
            containers.append(json.loads(line))
        except ValueError:
            continue
    return containers


def user_processes() -> tuple[list[Proc], set[int]]:
    """This user's processes, with their listening ports, and the hook's own ancestry to leave out."""
    # The hook command supplies psutil; the project does not depend on it, and
    # psutil has no type information, so it is imported untyped.
    psutil = importlib.import_module("psutil")

    me = psutil.Process()
    exclude = {me.pid, *(parent.pid for parent in me.parents())}
    listening: dict[int, set[int]] = {}
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.pid and conn.laddr:
                listening.setdefault(conn.pid, set()).add(conn.laddr.port)
    except psutil.AccessDenied:
        pass  # macOS needs root for this; the cwd and command line still count
    username = me.username()
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cwd", "cmdline", "username"], ad_value=None):
        info = proc.info
        if info["username"] != username:
            continue
        processes.append(
            Proc(
                pid=info["pid"],
                name=info["name"] or "",
                cwd=info["cwd"] or "",
                cmdline=" ".join(info["cmdline"] or []),
                ports=tuple(sorted(listening.get(info["pid"], ()))),
            )
        )
    return processes, exclude


def main() -> int:
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            return 0
    except ValueError:
        return 0
    reason = decide(event, docker_containers, user_processes)
    if reason is None:
        return 0
    print(f"Blocked by .claude/hooks/check_strays.py: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
