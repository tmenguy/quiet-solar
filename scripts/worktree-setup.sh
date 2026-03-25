#!/bin/bash
# Creates a git worktree for parallel story/bugfix development.
# Symlinks venv, config, and non-git custom_components from the main worktree.
#
# Usage: bash scripts/worktree-setup.sh <issue_number>
# Result: ../<repo>-worktrees/QS_<issue_number>/ ready for development

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <issue_number>"
    exit 1
fi

ISSUE="$1"

# P4: Validate issue number is numeric only
if ! [[ "$ISSUE" =~ ^[0-9]+$ ]]; then
    echo "Error: Issue number must be numeric, got '${ISSUE}'"
    exit 1
fi

BRANCH="QS_${ISSUE}"

# P1: Resolve main worktree reliably (works even if run from inside a worktree)
MAIN_DIR="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"

# P2: Derive directory name from main worktree path (not hardcoded)
MAIN_BASENAME="$(basename "$MAIN_DIR")"
WORKTREES_DIR="${MAIN_DIR}/../${MAIN_BASENAME}-worktrees"
WORKTREE_DIR="${WORKTREES_DIR}/${BRANCH}"

# Check branch doesn't already exist
if git -C "$MAIN_DIR" show-ref --verify --quiet "refs/heads/${BRANCH}" 2>/dev/null; then
    echo "Error: Branch ${BRANCH} already exists"
    exit 1
fi

# Check worktree doesn't already exist
if [ -d "$WORKTREE_DIR" ]; then
    echo "Error: Worktree directory ${WORKTREE_DIR} already exists"
    exit 1
fi

# Create worktrees parent directory if needed
mkdir -p "$WORKTREES_DIR"

# Create worktree with new branch from main
echo "Creating worktree at ${WORKTREE_DIR} on branch ${BRANCH}..."
git -C "$MAIN_DIR" worktree add "$WORKTREE_DIR" -b "$BRANCH"

# P2: Symlink venv using derived basename (not hardcoded)
if [ -d "${MAIN_DIR}/venv" ]; then
    ln -s "../../${MAIN_BASENAME}/venv" "${WORKTREE_DIR}/venv"
    echo "Symlinked venv"
fi

# P2: Symlink config contents using derived basename
# config/ may already exist as a directory (git creates it for tracked files like configuration.yaml).
# Symlink each non-git item inside config/ individually, same pattern as custom_components/.
if [ -d "${MAIN_DIR}/config" ]; then
    mkdir -p "${WORKTREE_DIR}/config"
    TRACKED_CONFIG="$(git -C "$MAIN_DIR" ls-files -- config/)"
    for item in "${MAIN_DIR}/config"/*; do
        [ -e "$item" ] || continue
        name="$(basename "$item")"
        target="${WORKTREE_DIR}/config/${name}"
        # Skip items that are tracked in git (already present in worktree)
        if echo "$TRACKED_CONFIG" | grep -q "^config/${name}$"; then
            continue
        fi
        if [ ! -e "$target" ]; then
            ln -s "../../../${MAIN_BASENAME}/config/${name}" "$target"
            echo "Symlinked config/${name}"
        fi
    done
fi

# Symlink non-git custom_components using derived basename
# quiet_solar is in git and already present in the worktree — skip it
if [ -d "${MAIN_DIR}/custom_components" ]; then
    for dir in "${MAIN_DIR}/custom_components"/*/; do
        [ -d "$dir" ] || continue
        name="$(basename "$dir")"
        target="${WORKTREE_DIR}/custom_components/${name}"
        if [ "$name" != "quiet_solar" ] && [ "$name" != "__pycache__" ] && [ ! -e "$target" ]; then
            ln -s "../../../${MAIN_BASENAME}/custom_components/${name}" "$target"
            echo "Symlinked custom_components/${name}"
        fi
    done
fi

echo ""
echo "Worktree ready: ${WORKTREE_DIR}"
echo "Branch: ${BRANCH}"
echo "To start working: cd ${WORKTREE_DIR} && source venv/bin/activate"
