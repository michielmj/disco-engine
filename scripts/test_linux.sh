#!/usr/bin/env bash
# Run the pytest suite in a Linux container, from a clean clone of a branch.
#
# Tests run on the deployment platform, as a non-root user, with no git
# identity and no host environment. A test that borrows ambient host state
# therefore fails here, the way it would in CI.
#
# Usage:  bash scripts/test_linux.sh [branch] [-- pytest-args...]
#   branch       a local branch; defaults to the current one. Its COMMITTED
#                state is tested, so commit before running.
#   pytest-args  after "--", replace the default arguments, for example
#                -- -m slow    or    -- tests/test_x.py -k name
#
# Needs Docker. The image is pinned; packages are cached in a named volume,
# so only the first run needs the network for long.
#
# The machine is shared, so the script sets its own limits:
# - A full suite runs alone. Its container is always named test-linux-suite,
#   and Docker allows a name only once, so a second full suite waits (up to
#   TEST_LINUX_WAIT seconds, default 3600). A run naming test files is not a
#   full suite and does not wait.
# - The run is stopped inside the container after TEST_LINUX_TIMEOUT seconds
#   (default 3600), where `timeout` works. On Windows, killing this script
#   does not stop the container, so the limit must live in the container.
# - Every run writes .test-logs/test-linux-<branch>-<commit>-<time>-<pid>.log,
#   whose first line names the branch and commit it tested.
# - The container is labelled with the branch and worktree, so a leftover can
#   be traced to the session that started it.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

UV_VERSION="0.12.15"
PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"
IMAGE="ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-trixie"

branch=""
if [[ $# -gt 0 && "$1" != "--" ]]; then branch="$1"; shift; fi
if [[ $# -gt 0 && "$1" == "--" ]]; then shift; fi
branch="${branch:-$(git rev-parse --abbrev-ref HEAD)}"
[[ "$branch" == "HEAD" ]] && { echo "test_linux: detached HEAD - name a branch" >&2; exit 2; }
git rev-parse --verify -q "refs/heads/$branch" >/dev/null || { echo "test_linux: no local branch '$branch'" >&2; exit 2; }
[[ -n "$(git status --porcelain --untracked-files=no)" ]] && \
  echo "test_linux: note - uncommitted changes are NOT tested" >&2

common="$(git rev-parse --path-format=absolute --git-common-dir)"
command -v cygpath >/dev/null 2>&1 && common="$(cygpath -w "$common")"

inner=$(cat <<'INNER'
set -euo pipefail
cd ~
git config --global --add safe.directory '*'
git clone -q /repo.git work && cd work
git checkout -q --detach "origin/$BRANCH"
uv sync -q --locked
echo "test_linux: $BRANCH at $(git rev-parse --short HEAD), $(uv run python --version), uid $(id -u)"
# shellcheck disable=SC2086
uv run pytest -q -p no:cacheprovider ${PYTEST_ARGS:--rf}
INNER
)

outer=$(cat <<'OUTER'
set -euo pipefail
useradd -m -u 1001 -s /bin/bash runner
chown runner /uv-cache
printf '%s\n' "$INNER" > /tmp/inner.sh && chmod 0644 /tmp/inner.sh
exec timeout --kill-after=30 "$TIMEOUT" su runner -s /bin/bash -c "$(printf 'BRANCH=%q PYTEST_ARGS=%q UV_CACHE_DIR=/uv-cache UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never bash /tmp/inner.sh' "$BRANCH" "$PYTEST_ARGS")"
OUTER
)

TIMEOUT="${TEST_LINUX_TIMEOUT:-3600}"
WAIT="${TEST_LINUX_WAIT:-3600}"
full=1
for arg in "$@"; do
  case "$arg" in tests/* | *.py | *::*) full=0 ;; esac
done
safe_branch="${branch//[^A-Za-z0-9_.-]/-}"
if ((full)); then name="test-linux-suite"; else name="test-linux-$safe_branch-$$"; fi

sha="$(git rev-parse --short "refs/heads/$branch")"
mkdir -p .test-logs
log=".test-logs/test-linux-$safe_branch-$sha-$(date +%Y%m%d-%H%M%S)-$$.log"
echo "test_linux: $branch at $sha, $(date '+%Y-%m-%d %H:%M:%S'), pytest ${*:--rf}" >"$log"

# -e passes only what is named here. The host environment never reaches the
# container, so no secret and no APP_* setting leaks into a test.
err="$(mktemp)"
trap 'rm -f "$err"' EXIT
waited=0
until cid="$(MSYS_NO_PATHCONV=1 docker create --rm --name "$name" \
  --label "test-linux.branch=$branch" \
  --label "test-linux.worktree=$(pwd -W 2>/dev/null || pwd)" \
  -v "$common:/repo.git:ro" \
  -v test-linux-uv-cache:/uv-cache \
  -e BRANCH="$branch" -e PYTEST_ARGS="$*" -e INNER="$inner" -e TIMEOUT="$TIMEOUT" \
  "$IMAGE" bash -c "$outer" 2>"$err")"; do
  grep -q "already in use" "$err" || { cat "$err" >&2; exit 2; }
  if ((waited == 0)); then
    holder="$(docker inspect -f '{{index .Config.Labels "test-linux.branch"}} ({{.State.Status}})' "$name" 2>/dev/null || true)"
    echo "test_linux: another full suite is running, for ${holder:-an unknown branch}; waiting up to ${WAIT}s." >&2
    if [[ "$holder" == *"(created)" ]]; then
      echo "test_linux: a holder that stays 'created' never started: its owner removes it with 'docker rm $name'." >&2
    fi
  fi
  ((waited >= WAIT)) && { echo "test_linux: gave up after waiting ${WAIT}s (TEST_LINUX_WAIT)" >&2; exit 3; }
  sleep 15
  waited=$((waited + 15))
done
# From here the container is ours: stop it however this script ends.
trap 'docker stop -t 5 "$cid" >/dev/null 2>&1 || true; rm -f "$err"' EXIT INT TERM

set +e
docker start -a "$cid" 2>&1 | tee -a "$log"
status="${PIPESTATUS[0]}"
set -e
if [[ "$status" == 124 || "$status" == 137 ]]; then
  echo "test_linux: stopped at the ${TIMEOUT}s limit (TEST_LINUX_TIMEOUT)" | tee -a "$log" >&2
fi
echo "test_linux: log at $log" >&2
exit "$status"
