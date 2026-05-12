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

# Check worktree doesn't already exist
if [ -d "$WORKTREE_DIR" ]; then
    echo "Error: Worktree directory ${WORKTREE_DIR} already exists"
    exit 1
fi

# Fetch latest so origin/main is up to date (safe even if main is not checked out)
echo "Fetching latest from origin..."
git -C "$MAIN_DIR" fetch origin

# Create worktrees parent directory if needed
mkdir -p "$WORKTREES_DIR"

# Create worktree — reuse existing branch or create new one
if git -C "$MAIN_DIR" show-ref --verify --quiet "refs/heads/${BRANCH}" 2>/dev/null; then
    # F17: Warn if existing branch has diverged from main
    if ! git -C "$MAIN_DIR" merge-base --is-ancestor "$BRANCH" main 2>/dev/null; then
        echo "Warning: Branch ${BRANCH} has diverged from main. Consider rebasing."
    fi
    echo "Creating worktree at ${WORKTREE_DIR} using existing branch ${BRANCH}..."
    git -C "$MAIN_DIR" worktree add "$WORKTREE_DIR" "$BRANCH"
else
    echo "Creating worktree at ${WORKTREE_DIR} on new branch ${BRANCH}..."
    git -C "$MAIN_DIR" worktree add "$WORKTREE_DIR" -b "$BRANCH" --no-track origin/main
fi

# Belt-and-suspenders: if the branch ended up tracking origin/main —
# either because --no-track was silently ignored on some git version,
# or because we re-attached to an existing branch that was created
# under the old buggy setup — drop the upstream so `git push` won't
# try to push to main.
if upstream="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    if [ "$upstream" = "origin/main" ]; then
        echo "Detected upstream origin/main on ${BRANCH}; unsetting."
        git -C "$WORKTREE_DIR" branch --unset-upstream
    fi
fi

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
    # F4: Enable dotglob so hidden files (.storage, .HA_VERSION, etc.) are included
    shopt -s dotglob
    for item in "${MAIN_DIR}/config"/*; do
        [ -e "$item" ] || continue
        name="$(basename "$item")"
        target="${WORKTREE_DIR}/config/${name}"
        # F15: Use fixed-string match instead of regex to avoid . metacharacter issues
        if echo "$TRACKED_CONFIG" | grep -qF "config/${name}"; then
            continue
        fi
        if [ ! -e "$target" ]; then
            ln -s "../../../${MAIN_BASENAME}/config/${name}" "$target"
            echo "Symlinked config/${name}"
        fi
    done
    shopt -u dotglob
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

# Verify the worktree's HEAD landed on the expected branch. A downstream
# session (e.g. Claude Desktop auto-isolation) that creates its own
# worktree on top of this one inherits HEAD — if HEAD is wrong here, the
# user ends up needing to "change the branch" manually after opening.
ACTUAL_BRANCH="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref HEAD)"
if [ "$ACTUAL_BRANCH" != "$BRANCH" ]; then
    echo "Error: worktree HEAD is on '${ACTUAL_BRANCH}', expected '${BRANCH}'"
    echo "Attempting recovery with explicit checkout..."
    if ! git -C "$WORKTREE_DIR" checkout "$BRANCH"; then
        echo "Recovery failed. Worktree at ${WORKTREE_DIR} is in an inconsistent state."
        exit 1
    fi
    ACTUAL_BRANCH="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref HEAD)"
    if [ "$ACTUAL_BRANCH" != "$BRANCH" ]; then
        echo "Recovery still failed. HEAD is on '${ACTUAL_BRANCH}'."
        exit 1
    fi
fi

echo ""
echo "Worktree ready: ${WORKTREE_DIR}"
echo "Branch: ${BRANCH} (HEAD verified)"
echo "To start working: cd ${WORKTREE_DIR} && source venv/bin/activate"
