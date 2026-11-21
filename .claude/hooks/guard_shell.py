"""PreToolUse guard for the Bash and PowerShell tools.

Rules that every brief used to repeat in prose are enforced here instead. A
prose rule decays: it is skipped, misremembered, or obeyed in one shell and
not the next. This hook runs before every shell command, in every session and
every subagent.

It is a guardrail, not a security boundary. It stops the forms a session
actually writes; an interpreter can still print the environment. The real
protection is to keep secrets out of the environment Claude Code starts in
(CHECKLIST.md: Git Credential Manager, `az login`, `gh auth login`).

The protocol: Claude Code sends the tool call as JSON on stdin. Exit code 2
blocks the call and shows stderr to the model; exit code 0 allows it. The
settings entry sets `"onFailure": "block"`, so a guard that cannot start
blocks the command instead of waving it through.

Keep the patterns narrow: a guard that blocks legitimate work gets turned
off. Every rule exists because its failure happened at least once, and every
pattern is pinned by tests/test_guard_hook.py, in both directions.

It also names the author of every commit: `<INITIALS> (<Role> Agent)`, with
the owner's email. The initials are the owner's, set once per machine with
`git config --global user.initials <abc>`; the role is the subagent's name, or
the main session's role when the call comes from no subagent.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

# Where a command can start: the beginning, after a separator or bracket, or
# as the argument of `bash -c` / `pwsh -Command`. Optional wrappers follow.
START = (
    r"(?:^|[;&|(){}\n]\s*"
    r"|\b(?:bash|sh|zsh|pwsh|powershell)(?:\.exe)?\s+(?:-\w+\s+)*-c(?:ommand)?\s+['\"]?)"
    r"(?:(?:sudo|nohup|time|command|builtin|exec)\s+)*"
)
# One shell word, quotes included: `user.name="JSN (Implementer Agent)"` is one word.
WORD = r"""(?:[^\s"';&|]+|"[^"]*"|'[^']*')+"""
# `git` followed by any global options (-C <path>, -c <key=value>, --no-pager).
GIT_OPTIONS = r"(?:\s+(?:-C\s+" + WORD + r"|-c\s+" + WORD + r"|--[\w-]+(?:=\S+)?))*"
GIT = r"\bgit(?:\.exe)?" + GIT_OPTIONS + r"\s+"
# The rest of one command: stops at a separator.
ARGS = r"[^;&|\n]*?"
# An interpreter that can read a script from stdin, run directly or through
# `uv run`, with its options (`-`, `-s`, `-3.13`).
INTERPRETER = (
    r"(?:uv\s+run\s+(?:--?[\w-]+(?:[= ]\S+)?\s+)*)?"
    r"(?:python(?:3(?:\.\d+)?)?|py|node|bash|sh|zsh|ruby|perl|pwsh|powershell)(?:\.exe)?"
    r"(?:\s+-[^\s<|;&]*)*"
)
# A docker command that stops or removes containers.
DOCKER_STOP = r"docker(?:\.exe)?\s+(?:container\s+)?(?:stop|rm|kill)\b"
# Listing containers without a filter: everyone's, the owner's included.
DOCKER_ALL = r"docker(?:\.exe)?\s+(?:container\s+)?(?:ps|ls)\b"


def _any(*alternatives: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{a})" for a in alternatives), flags)


RULES: list[tuple[re.Pattern[str], str]] = [
    (
        _any(
            # POSIX: the commands that list every variable.
            START
            + r"(?:env(?:\s+(?:-0|--null))?|printenv\b[^;&|\n]*|export(?:\s+-[pn])?|set"
            + r"|declare\s+-[a-z]*[px][a-z]*|typeset\s+-[a-z]*x[a-z]*)\s*(?:$|[;&|)'\"\n])",
            r"/proc/[^/\s]+/environ\b",
            # PowerShell and cmd: the env: drive, the .NET call, `cmd /c set`.
            r"\b(?:Get-ChildItem|gci|dir|ls|Get-Item|gi)\s+(?:-(?:Path|LiteralPath)\s+)?"
            + r"['\"]?env:[\\/]?\*?['\"]?(?=\s*(?:$|[;|)}\n]))",
            r"\[(?:System\.)?Environment\]::GetEnvironmentVariables\b",
            r"\bcmd(?:\.exe)?\s+/[ck]\s+set\s*(?:$|[&|\"'\n])",
            # The obvious interpreter one-liners.
            r"\b(?:print|pprint|dict|json\.dumps)\s*\(\s*os\.environ\s*\)",
            r"\bconsole\.(?:log|dir)\s*\(\s*process\.env\s*\)",
            flags=re.IGNORECASE,
        ),
        "Printing the environment is forbidden: the owner's shell can hold live "
        "credentials, and a printed environment once leaked a token into a transcript. "
        "Read the one variable you need without printing its value.",
    ),
    (
        re.compile(GIT + r"add\b" + ARGS + r"\s(?:-[a-zA-Z]*A[a-zA-Z]*|--all|\.|\*|:/)(?=\s|$|[;&|])"),
        "Name every file in `git add`. `-A`, `--all` and `.` stage files you have "
        "not looked at, such as scratch files or secrets.",
    ),
    (
        re.compile(
            GIT
            + r"push\b"
            + ARGS
            + r"(?:\s--force(?:-with-lease|-if-includes)?\b|\s-[a-zA-Z]*f[a-zA-Z]*\b|\s\+\S)"
        ),
        "Never force-push, including --force-with-lease and a +refspec. A reviewed "
        "branch is never rewritten: a correction is a new commit.",
    ),
    (
        re.compile(GIT + r"push\b" + ARGS + r"(?:\s--delete\b|\s-d\b|\s:\S)"),
        "Never delete a remote branch: branches are the owner's to delete, and "
        "`staging` was once deleted by accident.",
    ),
    (
        re.compile(GIT + r"commit\b" + ARGS + r"\s--amend\b"),
        "Never amend. A correction is a new commit, so a reviewer's view of the "
        "branch stays true (CLAUDE.md, How to record).",
    ),
    (
        re.compile(GIT + r"(?:rebase\b|pull\b" + ARGS + r"\s(?:--rebase\b|-r\b))"),
        "Never rebase. Bring a branch up to date by merging origin/staging into it: "
        "a rebase rewrites commits a reviewer may already have seen.",
    ),
    (
        _any(
            # Any `git tag` that is not a listing.
            GIT
            + r"tag\s+(?!(?:-l|--list|-n\d*|--contains|--no-contains|--points-at|--merged|--no-merged"
            + r"|--sort(?:=\S*)?|--format(?:=\S*)?|--column|-v|--verify)(?:\s|$))\S",
            GIT + r"push\b" + ARGS + r"(?:\s--(?:follow-)?tags\b|\srefs/tags/|\s(?:\S+:)?v\d+\.\d+)",
        ),
        "Tags are the product owner's. No session creates, deletes or pushes a tag.",
    ),
    (
        re.compile(
            GIT + r"config\b" + ARGS + r"(?:\buser\.initials\s+[^\s;&|]|--unset(?:-all)?\s+user\.initials\b)",
            re.IGNORECASE,
        ),
        "The initials in user.initials are the owner's, and only the owner sets them. "
        "Stop and ask the owner to run `git config --global user.initials <their initials>`.",
    ),
    (
        re.compile(
            r"\bdocker(?:\.exe)?\s+(?:(?:system|image|container|volume|network|builder|buildx)\s+)?prune\b",
            re.IGNORECASE,
        ),
        "Never prune Docker: it removes the owner's and other sessions' containers, "
        "images and volumes. Remove only what you started, by name.",
    ),
    (
        _any(
            r"\b" + DOCKER_STOP + r"[^;&|\n]*\$\(\s*" + DOCKER_ALL + r"(?![^)]*(?:--filter|\s-f\s))",
            r"\b"
            + DOCKER_ALL
            + r"(?![^|\n]*(?:--filter|\s-f\s))[^|\n]*\|\s*"
            + r"(?:xargs\s+(?:-\S+\s+)*|(?:ForEach-Object|foreach|%)\s*\{\s*)?"
            + DOCKER_STOP,
            flags=re.IGNORECASE,
        ),
        "Stop only the containers you started, by name. An unfiltered `docker ps` lists "
        "the owner's and other sessions' containers too.",
    ),
    (
        _any(
            START + r"(?:pkill|killall)\b",
            r"\btaskkill(?:\.exe)?\b[^;&|\n]*\s/im\b",
            r"\b(?:Stop-Process|spps|kill)\b[^;|\n]*\s-(?:Name|ProcessName)\b",
            r"\b(?:Get-Process|gps)\b(?![^|;\n]*\s-Id\b)[^|;\n]*\|\s*(?:Stop-Process|spps|kill)\b",
            r"\$\(\s*(?:pgrep|pidof)\b",
            r"\b(?:pgrep|pidof)\b[^;&\n]*\|\s*xargs\s+(?:-\S+\s+)*kill\b",
            flags=re.IGNORECASE,
        ),
        "Stop only what you started, by pid: `kill <pid>`, `Stop-Process -Id <pid>` or "
        "`taskkill /PID <pid>`. Stopping by name also stops the owner's and other sessions' "
        "processes.",
    ),
    (
        _any(
            START + INTERPRETER + r"\s*<<",
            r"<<[^|;&\n]*\|\s*" + INTERPRETER + r"\s*(?:$|[;&|)\n])",
            flags=re.IGNORECASE,
        ),
        "Write the script to a file and run the file; never pipe a script into an "
        "interpreter through a heredoc. A heredoc can rewrite the script's `$` and "
        "backslashes, and an inline script leaves nothing to re-run or review.",
    ),
    (
        re.compile(START + r"(?:npx|pnpx|bunx)\b|\b(?:npm|pnpm)\s+(?:exec|x|dlx)\b|\byarn\s+dlx\b"),
        "npx and npm exec can download and run a package that is not in package.json. "
        "Run tools through `npm run <script>`; adding a package is an escalation.",
    ),
]

# When Linux is the only target platform, tests run on Linux: on a Windows host
# a bare pytest is refused, and a pytest inside `wsl` or `docker` is allowed.
# When the project also ships to Windows or macOS, pytest runs natively there.
LINUX_ONLY = False
PYTEST = re.compile(
    START
    + r"(?:uv\s+run\s+(?:--?\S+\s+)*)?(?:(?:python3?|py)(?:\.exe)?(?:\s+-3(?:\.\d+)?)?\s+-m\s+)?pytest\b",
    re.IGNORECASE,
)
PYTEST_REASON = (
    "Tests run on Linux only, the deployment platform. Use `bash scripts/test_linux.sh <branch>`: "
    "it tests the committed branch in a clean container, the way CI does."
)


# Every commit names the role that made it, so `git log` shows which agent
# wrote what. The main session has no agent_type in the hook input.
MAIN_ROLE = "Architect"
COMMITTING = re.compile(
    r"\bgit(?:\.exe)?(?P<options>" + GIT_OPTIONS + r")\s+(?:commit|merge|revert|cherry-pick)(?![-\w])"
)
CONFIG = re.compile(r"-c\s+(" + WORD + r")")
NO_INITIALS_REASON = (
    "Every commit carries the owner's initials, and none are set on this machine. "
    "Stop and ask the owner to run `git config --global user.initials <their initials>`; "
    "never set them yourself."
)


def author_name(initials: str | None, agent_type: str | None) -> str | None:
    """The author name for a commit: `JSN (Implementer Agent)`, or None without initials."""
    if not initials:
        return None
    role = MAIN_ROLE
    if agent_type:
        role = agent_type.rsplit(":", 1)[-1].replace("-", " ").replace("_", " ").title()
    return f"{initials.upper()} ({role} Agent)"


def read_initials() -> str | None:
    """The owner's initials from `git config user.initials`, or None if unset or not letters."""
    try:
        proc = subprocess.run(
            ["git", "config", "--get", "user.initials"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    initials = proc.stdout.strip()
    return initials if re.fullmatch(r"[A-Za-z]+", initials) else None


def _user_name(options: str) -> str | None:
    """The last `-c user.name=` value among git's global options: git uses the last one."""
    name = None
    for match in CONFIG.finditer(options):
        key, _, value = re.sub(r"[\"']", "", match.group(1)).partition("=")
        if key.lower() == "user.name":
            name = value
    return name


def check(command: str, platform: str, author: str | None) -> str | None:
    """Return the reason to block `command`, or None to allow it.

    `author` is the name every commit in `command` must carry, or None when the
    owner's initials are not set.
    """
    for pattern, reason in RULES:
        if pattern.search(command):
            return reason
    for match in COMMITTING.finditer(command):
        if author is None:
            return NO_INITIALS_REASON
        if _user_name(match["options"]) != author:
            return (
                f'Commit as your role: `git -c user.name="{author}" commit ...`, and the same '
                "`-c` for merge, revert and cherry-pick. The name records which role made the "
                "commit; the email stays the owner's."
            )
    if (
        LINUX_ONLY
        and platform == "win32"
        and PYTEST.search(command)
        and not re.search(r"\b(?:wsl|docker)\b", command)
    ):
        return PYTEST_REASON
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
        command = event["tool_input"]["command"]
        if not isinstance(command, str):
            raise TypeError("command is not a string")
        agent_type = event.get("agent_type")
        if agent_type is not None and not isinstance(agent_type, str):
            raise TypeError("agent_type is not a string")
    except (ValueError, KeyError, TypeError) as exc:
        # Fail closed: a command this guard cannot read is a command it cannot vouch for.
        print(
            f"Blocked by .claude/hooks/guard_shell.py: could not read the tool call ({exc}).", file=sys.stderr
        )
        return 2
    # Git is asked for the initials only when the command commits.
    author = author_name(read_initials(), agent_type) if COMMITTING.search(command) else None
    reason = check(command, sys.platform, author)
    if reason is not None:
        print(f"Blocked by .claude/hooks/guard_shell.py: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
