# Status

This is where the work stands. Specs say *what* (`specs/`); this file says
*where we are*. Keep it short, and move finished detail into git history.

## In flight

| Milestone | Branch | State | Requirement | Notes |
|---|---|---|---|---|
| Adopt the scaffolder template | `scaffold-adoption` | building | none (owner's direction, 2026-11-21) | CLAUDE.md, rules, guards, agents, skills, uv, ruff, import-linter, CI |

## Releases

| Version | Date | Contents | Owner verified |
|---|---|---|---|

## Waiting on the owner

- After `scaffold-adoption` lands: start Claude Code in the project, accept
  the workspace-trust dialog, and ask it to run `env` and `git push --force`;
  both must be blocked by `guard_shell.py`.

## Backlog (carry-overs), prioritised by the owner

| # | Found in | Item | Priority |
|---|---|---|---|
| 1 | scaffold-adoption | Restructure the package layout, breaking changes allowed (owner, 2026-11-21) | unset |
| 2 | scaffold-adoption | Move mypy to `strict = true`; `.claude/rules/python.md` already asks for it | unset |
