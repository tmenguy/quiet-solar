#!/bin/bash
# Removes a git worktree after a story/bugfix is merged.
# Symlinks inside the worktree are removed with the directory.
# The targets (main venv, config, custom_components) are NOT affected.
#
# Usage: bash scripts/worktree-cleanup.sh <issue_number>

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
WORKTREE_DIR="${MAIN_DIR}/../${MAIN_BASENAME}-worktrees/${BRANCH}"

# Check worktree exists
if [ ! -d "$WORKTREE_DIR" ]; then
    echo "Error: Worktree directory ${WORKTREE_DIR} does not exist"
    exit 1
fi

# P5: Check for uncommitted tracked changes before --force removal
if git -C "$WORKTREE_DIR" diff --quiet HEAD 2>/dev/null && \
   git -C "$WORKTREE_DIR" diff --cached --quiet HEAD 2>/dev/null; then
    : # Clean — no uncommitted changes
else
    echo "Warning: Worktree ${BRANCH} has uncommitted changes!"
    echo "These will be lost. Press Ctrl+C to abort, or Enter to continue."
    read -r
fi

# Remove worktree (--force needed because symlinks and generated files are untracked)
echo "Removing worktree at ${WORKTREE_DIR}..."
git -C "$MAIN_DIR" worktree remove --force "$WORKTREE_DIR"

# P3: Delete the local branch (only if fully merged, -d not -D)
if git -C "$MAIN_DIR" show-ref --verify --quiet "refs/heads/${BRANCH}" 2>/dev/null; then
    if git -C "$MAIN_DIR" branch -d "$BRANCH" 2>/dev/null; then
        echo "Deleted local branch ${BRANCH}"
    else
        echo "Note: Local branch ${BRANCH} not fully merged — kept for safety"
    fi
fi

# Prune stale worktree references
git -C "$MAIN_DIR" worktree prune

echo "Worktree ${BRANCH} removed."
echo "To update main: cd ${MAIN_DIR} && git checkout main && git pull"
