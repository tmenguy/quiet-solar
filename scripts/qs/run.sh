#!/usr/bin/env bash
# Convenience runner for QS scripts.
# Usage: bash scripts/qs/run.sh <script_name> [args...]
# Example: bash scripts/qs/run.sh quality_gate --fix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_name> [args...]"
    echo "Available scripts:"
    for f in "${SCRIPT_DIR}"/*.py; do
        name="$(basename "$f" .py)"
        [ "$name" = "__init__" ] || [ "$name" = "utils" ] && continue
        echo "  $name"
    done
    exit 1
fi

SCRIPT="$1"
shift

# Activate venv if available
if [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
    source "${REPO_ROOT}/venv/bin/activate"
fi

cd "${SCRIPT_DIR}"
exec python "${SCRIPT}.py" "$@"
