"""PreToolUse guard for the read-only agents: reviewer, requirement-reviewer, surveyor.

Their agent definitions leave out Edit and Write, but Bash and PowerShell can
still commit, push or move a branch. This hook makes "read-only" a mechanism.
It runs only while one of those agents is active (it is declared in their
frontmatter), in addition to guard_shell.py.

Read-only means: nothing this agent does reaches the remote, a branch, a tag
or a PR. A scratch mutation inside the agent's own detached worktree, to prove
that a test fails when its rule is broken, is allowed: it never leaves that
worktree.

The protocol is the same as guard_shell.py: exit code 2 blocks the call.
"""

from __future__ import annotations

import json
import re
import sys

# One shell word, quotes included, so a quoted `-c user.name="..."` can't hide the command after it.
WORD = r"""(?:[^\s"';&|]+|"[^"]*"|'[^']*')+"""
GIT = r"\bgit(?:\.exe)?(?:\s+(?:-C\s+" + WORD + r"|-c\s+" + WORD + r"|--[\w-]+(?:=\S+)?))*\s+"
ARGS = r"[^;&|\n]*?"

FORBIDDEN = re.compile(
    "|".join(
        [
            # Anything that records or publishes: commits, merges, pushes, tags.
            GIT + r"(?:commit|push|merge(?![-\w])|rebase|cherry-pick|revert|am)\b",
            GIT + r"tag\s+(?!-l\b|--list\b)\S",
            # Anything that moves or deletes a branch, or touches the stash:
            # both are shared by every worktree of the repository.
            GIT + r"branch\b" + ARGS + r"\s(?:-[dDmMcCf]\b|--(?:delete|move|copy|force)\b)",
            GIT + r"checkout\b" + ARGS + r"\s-[bB]\b",
            GIT + r"switch\b" + ARGS + r"\s-[cC]\b",
            GIT + r"stash(?!\s+(?:list|show)\b)",
            GIT + r"update-ref\b",
            GIT + r"reset\b" + ARGS + r"\s--hard\s+\S",
            # Anything that changes a pull request.
            r"\bgh\s+pr\s+(?!(?:view|list|diff|checks|status)\b)\w",
            r"\bgh\s+api\b" + ARGS + r"\s(?:-X|--method)\s*(?:POST|PATCH|PUT|DELETE)\b",
            r"\baz\s+repos\s+pr\s+(?!(?:show|list)\b)\w",
        ]
    ),
    re.IGNORECASE,
)

REASON = (
    "You are a read-only agent: you never commit, push, tag, move a branch or change a PR. "
    "Work in your own detached worktree, and report what you found; the architect acts on it."
)


def check(command: str) -> str | None:
    """Return the reason to block `command`, or None to allow it."""
    return REASON if FORBIDDEN.search(command) else None


def main() -> int:
    try:
        command = json.load(sys.stdin)["tool_input"]["command"]
        if not isinstance(command, str):
            raise TypeError("command is not a string")
    except (ValueError, KeyError, TypeError) as exc:
        print(
            f"Blocked by .claude/hooks/guard_readonly.py: could not read the tool call ({exc}).",
            file=sys.stderr,
        )
        return 2
    reason = check(command)
    if reason is not None:
        print(f"Blocked by .claude/hooks/guard_readonly.py: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
