"""The shell guards block what CLAUDE.md forbids, and nothing a session needs.

Each BLOCKED case is a form a session has written, or a direct variant of one.
Each ALLOWED case is legitimate work: a guard that blocks real work gets
turned off, so both lists matter equally. When you change a guard, add the
command that prompted the change to one of these lists.
"""

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


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HOOKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard_shell = _load("guard_shell")
guard_readonly = _load("guard_readonly")

BLOCKED = [
    # The environment, in POSIX shells and in PowerShell.
    "env",
    "env | grep TOKEN",
    "env -0",
    "printenv",
    "printenv HOME",
    "export",
    "export -p",
    "set",
    "declare -x",
    "ls -la && env",
    "bash -c env",
    "bash -c 'env | sort'",
    "sudo env",
    "cat /proc/self/environ",
    "Get-ChildItem env:",
    "Get-ChildItem Env:\\",
    "Get-ChildItem -Path env:",
    "gci env: | Sort-Object Name",
    "dir env:",
    "Get-Item env:*",
    "pwsh -Command Get-ChildItem env:",
    "[Environment]::GetEnvironmentVariables()",
    "[System.Environment]::GetEnvironmentVariables()",
    "cmd /c set",
    'python -c "import os; print(os.environ)"',
    "node -e 'console.log(process.env)'",
    # Staging without naming files.
    "git add -A",
    "git add --all",
    "git add .",
    "git add src .",
    "git add -- .",
    "git -C ../other add -A",
    # Force-pushing, in every spelling.
    "git push -f",
    "git push --force",
    "git push origin feature --force",
    "git push --force-with-lease",
    "git push -uf origin feature",
    "git push origin +main",
    # Deleting a remote branch.
    "git push origin --delete staging",
    "git push origin :staging",
    # Rewriting history.
    "git commit --amend",
    "git commit -a --amend --no-edit",
    "git rebase main",
    "git rebase -i HEAD~3",
    "git pull --rebase",
    # Tags.
    "git tag v1.0.0",
    "git tag -a v1.0.0 -m release",
    # A quoted -c value must not hide the command after it.
    'git -c user.name="JSN (Implementer Agent)" commit --amend',
    "git -c 'user.name=JSN (Implementer Agent)' push --force",
    # The owner's initials are set by the owner.
    "git config --global user.initials abc",
    "git config user.initials xyz",
    "git config --global --unset user.initials",
    "git tag -d v1.0.0",
    "git push --tags",
    "git push origin v1.0.0",
    "git push origin refs/tags/v1.0.0",
    # Docker housekeeping that hits other people's work.
    "docker system prune -f",
    "docker volume prune",
    "docker builder prune",
    "docker network prune",
    # Stopping containers that may not be yours.
    "docker stop $(docker ps -q)",
    "docker rm -f $(docker ps -aq)",
    "docker container kill $(docker container ls -q)",
    "docker ps -q | xargs docker stop",
    "docker ps -q | ForEach-Object { docker stop $_ }",
    # Stopping processes by name.
    "pkill -f uvicorn",
    "killall node",
    "taskkill /IM python.exe /F",
    "taskkill /f /im node.exe",
    "Stop-Process -Name python",
    "Stop-Process -Force -Name node",
    "kill -Name python",
    "Get-Process python | Stop-Process",
    "gps node | Stop-Process -Force",
    "kill $(pgrep -f vite)",
    "pgrep -f vite | xargs kill",
    # A script piped into an interpreter through a heredoc.
    "python - <<'EOF'\nprint(1)\nEOF",
    "python3 - <<EOF",
    "python <<EOF",
    "py -3.13 - <<EOF",
    "uv run python - <<'PY'",
    "uv run --with pyyaml python - <<'PY'",
    "node - <<'JS'",
    "bash -s <<EOF",
    "cd tests && python - <<'EOF'",
    "cat <<'EOF' | python -\nprint(1)\nEOF",
    "cat <<EOF | uv run python",
    "python <<< 'print(1)'",
    # Packages fetched on the fly.
    "npx some-random-pkg",
    "cd frontend && npx vite",
    "npm exec eslint",
    "pnpm dlx create-thing",
    "yarn dlx tool",
]

ALLOWED = [
    "set -euo pipefail",
    "env FOO=1 uv run python -m app",
    "env -u HOME bash scripts/x.sh",
    "echo $HOME",
    "$env:PATH",
    "if ($env:APP_MODE) { 'set' }",
    "Get-Item env:APP_MODE",
    "Set-Location src",
    "grep -r 'env' src",
    "git add src/app/a.py tests/test_a.py",
    "git add ./src/a.py",
    "git add .gitignore",
    "git status",
    "git log --format=%H -n 5",
    "git push origin feature",
    "git push -u origin feature",
    'git -c user.name="JSN (Implementer Agent)" commit -m "Fix the export"',
    "git -c 'user.name=JSN (Implementer Agent)' commit --allow-empty -F - <<'EOF'",
    'git -C ../wt -c user.name="JSN (Implementer Agent)" merge origin/staging',
    'git -c user.name="JSN (Implementer Agent)" revert abc123',
    "git merge-base HEAD origin/staging",
    "git log --format='%an %s' -n 5",
    "git config --get user.initials",
    "git config user.initials",
    "git tag",
    "git tag -l 'v*'",
    "git tag --list",
    "git tag --contains HEAD",
    "git tag --sort=-creatordate",
    "git fetch origin",
    "git merge-tree --write-tree origin/staging origin/feature",
    "git worktree add .claude/worktrees/m1a-export -b m1a-export origin/staging",
    "docker run --rm --name test-linux-1 python:3.13 true",
    "docker rm test-linux-1",
    "docker stop test-linux-4242",
    "docker rm -f $(docker ps -aq --filter name=test-linux-4242)",
    "docker ps --filter name=test-linux -q | xargs docker stop",
    "docker ps",
    "kill 12345",
    "kill -9 12345",
    "Stop-Process -Id 4242",
    "Get-Process -Id 4242 | Stop-Process",
    "taskkill /PID 4242 /F",
    "pgrep -af uvicorn",
    "Get-Process node",
    "python scripts/seed.py",
    "python scripts/seed.py <<'EOF'",
    "cat <<'EOF' | python scripts/check.py",
    "cat > tests/data/rows.json <<'EOF'",
    "uv run python -m app",
    "npm run build",
    "npm ci",
    "uv run ruff check src tests",
    "bash scripts/test_linux.sh m1a-export",
]

PYTEST_FORMS = [
    "pytest",
    "pytest -q tests/test_x.py",
    "uv run pytest -q",
    "uv run --frozen pytest",
    "python -m pytest",
    "py -m pytest",
    "py -3.13 -m pytest",
    "uv run python -m pytest -k name",
]


# The name every commit in the lists above and below is checked against.
AUTHOR = "JSN (Implementer Agent)"

# Commits whose author name is missing or wrong.
UNNAMED_COMMITS = [
    "git commit -m x",
    "git commit --allow-empty -F - <<'EOF'",
    "cd ../wt && git commit -m x",
    "git -C ../wt commit -m x",
    "git merge origin/staging",
    "git revert abc123",
    "git cherry-pick abc123",
    'git -c user.name="JSN (Architect Agent)" commit -m x',
    'git -c user.name="ABC (Implementer Agent)" commit -m x',
    'git -c user.name="JSN (Implementer Agent)" -c user.name=other commit -m x',
    'git -c user.name="JSN (Implementer Agent)" commit -m x && git merge origin/staging',
]


@pytest.mark.parametrize("command", BLOCKED)
def test_guard_blocks(command: str) -> None:
    assert guard_shell.check(command, "linux", AUTHOR) is not None


@pytest.mark.parametrize("command", ALLOWED)
def test_guard_allows(command: str) -> None:
    assert guard_shell.check(command, "linux", AUTHOR) is None


@pytest.mark.parametrize("command", UNNAMED_COMMITS)
def test_a_commit_must_carry_its_role(command: str) -> None:
    reason = guard_shell.check(command, "linux", AUTHOR)
    assert reason is not None
    assert f'git -c user.name="{AUTHOR}" commit' in reason


def test_without_initials_a_commit_is_refused_and_other_commands_are_not() -> None:
    reason = guard_shell.check('git -c user.name="JSN (Implementer Agent)" commit -m x', "linux", None)
    assert reason is not None
    assert "git config --global user.initials" in reason
    assert guard_shell.check("git status", "linux", None) is None


@pytest.mark.parametrize(
    ("initials", "agent_type", "expected"),
    [
        ("jsn", "implementer", "JSN (Implementer Agent)"),
        ("JSN", "implementer", "JSN (Implementer Agent)"),
        ("jsn", None, f"JSN ({guard_shell.MAIN_ROLE} Agent)"),
        ("jsn", "data-migrator", "JSN (Data Migrator Agent)"),
        ("jsn", "my-plugin:implementer", "JSN (Implementer Agent)"),
        (None, "implementer", None),
    ],
)
def test_author_name(initials: str | None, agent_type: str | None, expected: str | None) -> None:
    assert guard_shell.author_name(initials, agent_type) == expected


def test_main_role_matches_the_tier() -> None:
    assert guard_shell.MAIN_ROLE in {"Architect", "Developer"}


@pytest.mark.parametrize("command", PYTEST_FORMS)
def test_pytest_on_windows_is_refused_only_when_linux_is_the_only_target(command: str) -> None:
    assert (guard_shell.check(command, "win32", AUTHOR) is not None) == guard_shell.LINUX_ONLY
    assert guard_shell.check(command, "linux", AUTHOR) is None


@pytest.mark.parametrize("command", ["wsl uv run pytest", "docker run --rm img pytest -q"])
def test_pytest_inside_linux_is_allowed_on_windows(command: str) -> None:
    assert guard_shell.check(command, "win32", AUTHOR) is None


READONLY_BLOCKED = [
    "git commit -m x",
    "git push origin HEAD",
    "git merge origin/staging",
    "git rebase main",
    "git cherry-pick abc123",
    "git tag v1",
    "git branch -D feature",
    "git branch --delete feature",
    "git checkout -b scratch",
    "git switch -c scratch",
    "git stash",
    "git stash pop",
    "git reset --hard origin/staging",
    "git update-ref refs/heads/x HEAD",
    'git -c user.name="JSN (Reviewer Agent)" commit -m x',
    "git -c 'user.name=JSN (Reviewer Agent)' merge origin/staging",
    "gh pr merge 12",
    "gh pr comment 12 --body x",
    "gh api repos/o/r/pulls/12/reviews -X POST",
    "az repos pr update --id 12 --status completed",
    "az repos pr set-vote --id 12 --vote approve",
]

READONLY_ALLOWED = [
    "git fetch origin",
    "git diff origin/staging...HEAD",
    "git log --oneline origin/staging..HEAD",
    "git show HEAD --stat",
    "git merge-tree --write-tree origin/staging HEAD",
    "git worktree add --detach .claude/worktrees/review-m1a origin/m1a-export",
    "git worktree remove .claude/worktrees/review-m1a",
    "git branch -a",
    "git branch --contains HEAD",
    "git stash list",
    "git tag -l",
    "git checkout --detach origin/m1a-export",
    "gh pr view 12",
    "gh pr diff 12",
    "az repos pr show --id 12",
    "bash scripts/test_linux.sh m1a-export",
    "sed -i 's/>= limit/> limit/' src/app/rules.py",
]


@pytest.mark.parametrize("command", READONLY_BLOCKED)
def test_readonly_guard_blocks(command: str) -> None:
    assert guard_readonly.check(command) is not None


@pytest.mark.parametrize("command", READONLY_ALLOWED)
def test_readonly_guard_allows(command: str) -> None:
    assert guard_readonly.check(command) is None


def _run(
    hook: str, payload: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS / f"{hook}.py")],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=env,
    )


def _git_identity(tmp_path: Path, initials: str | None) -> dict[str, str]:
    """An environment whose only git configuration is a global file holding `initials`."""
    config = tmp_path / "gitconfig"
    config.write_text(f"[user]\n\tinitials = {initials}\n" if initials else "", encoding="utf-8")
    return {**os.environ, "GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1"}


@pytest.mark.parametrize(
    ("agent_type", "role"), [("implementer", "Implementer"), (None, guard_shell.MAIN_ROLE)]
)
def test_the_hook_names_the_role_from_the_agent_type(
    tmp_path: Path, agent_type: str | None, role: str
) -> None:
    env = _git_identity(tmp_path, "jsn")
    event: dict[str, object] = {"tool_name": "Bash"}
    if agent_type:
        event["agent_type"] = agent_type
    named = {**event, "tool_input": {"command": f'git -c user.name="JSN ({role} Agent)" commit -m x'}}
    assert _run("guard_shell", json.dumps(named), tmp_path, env).returncode == 0
    unnamed = {**event, "tool_input": {"command": "git commit -m x"}}
    blocked = _run("guard_shell", json.dumps(unnamed), tmp_path, env)
    assert blocked.returncode == 2
    assert f"JSN ({role} Agent)" in blocked.stderr


def test_the_hook_refuses_a_commit_without_initials(tmp_path: Path) -> None:
    env = _git_identity(tmp_path, None)
    event = {"tool_name": "Bash", "tool_input": {"command": "git commit -m x"}}
    blocked = _run("guard_shell", json.dumps(event), tmp_path, env)
    assert blocked.returncode == 2
    assert "user.initials" in blocked.stderr


@pytest.mark.parametrize("hook", ["guard_shell", "guard_readonly"])
@pytest.mark.parametrize("tool", ["Bash", "PowerShell"])
def test_protocol(hook: str, tool: str) -> None:
    blocked = _run(hook, json.dumps({"tool_name": tool, "tool_input": {"command": "git push --force"}}))
    assert blocked.returncode == 2
    assert f"Blocked by .claude/hooks/{hook}.py" in blocked.stderr
    allowed = _run(hook, json.dumps({"tool_name": tool, "tool_input": {"command": "git status"}}))
    assert (allowed.returncode, allowed.stderr) == (0, "")


@pytest.mark.parametrize("hook", ["guard_shell", "guard_readonly"])
@pytest.mark.parametrize("payload", ["not json", "{}", '{"tool_input": {"command": 3}}'])
def test_unreadable_input_is_blocked(hook: str, payload: str) -> None:
    assert _run(hook, payload).returncode == 2


def test_an_unreadable_agent_type_is_blocked() -> None:
    assert _run("guard_shell", '{"tool_input": {"command": "ls"}, "agent_type": 3}').returncode == 2
