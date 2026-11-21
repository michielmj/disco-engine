#!/usr/bin/env bash
# Create a worktree for one task, with a block of ports no other session uses.
#
# Usage:  bash scripts/new_task.sh <branch> [<base>]    a new branch from <base> (default origin/staging)
#         bash scripts/new_task.sh --review <branch>    a detached worktree of origin/<branch>, for a reviewer
#
# Worktrees live in .claude/worktrees/: inside the repository, so sessions can
# edit them without extra permissions, and gitignored. Each worktree records
# its port block in .task-ports; removing the worktree frees the block.
DEFAULT_BASE="origin/staging"

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# The owner's ports (8000-8099) are never handed out.
PORT_BASE=8100
BLOCK=10
BLOCKS=20

review=0
if [[ "${1:-}" == "--review" ]]; then review=1; shift; fi
branch="${1:?usage: bash scripts/new_task.sh [--review] <branch> [<base>]}"
base="${2:-$DEFAULT_BASE}"

git fetch -q origin

used="$(git worktree list --porcelain | sed -n 's/^worktree //p' | while read -r wt; do
  cat "$wt/.task-ports" 2>/dev/null || true
done)"
ports=""
for ((i = 0; i < BLOCKS; i++)); do
  start=$((PORT_BASE + i * BLOCK))
  candidate="$start-$((start + BLOCK - 1))"
  if ! grep -qx "$candidate" <<<"$used"; then ports="$candidate"; break; fi
done
[[ -n "$ports" ]] || { echo "new_task: all $BLOCKS port blocks are taken; land or remove a worktree first" >&2; exit 2; }

if ((review)); then
  dir=".claude/worktrees/review-$branch"
  git worktree add -q --detach "$dir" "origin/$branch"
else
  dir=".claude/worktrees/$branch"
  git worktree add -q --no-track -b "$branch" "$dir" "$base"
fi
echo "$ports" >"$dir/.task-ports"

echo "worktree: $(cd "$dir" && pwd -W 2>/dev/null || pwd)"
echo "branch:   $branch$( ((review)) && echo ' (detached, for review)')"
echo "ports:    $ports"
