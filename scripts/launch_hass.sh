#!/bin/sh
# Launch Home Assistant from the local venv with the local config dir.
# Resolves paths relative to this script's location, so it works from
# the main checkout or any worktree (cd to the project, then exec hass).

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_DIR"

# shellcheck source=/dev/null
. ./venv/bin/activate

exec ./venv/bin/hass -c config "$@"
