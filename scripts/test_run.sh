#!/usr/bin/env bash
# Back-compat shim — the runner was renamed to run.sh. Forwards all args + env
# (PROFILE, TEST_*, AUTO_STOP, …). Prefer `scripts/run.sh` going forward.
exec "$(dirname "$0")/run.sh" "$@"
