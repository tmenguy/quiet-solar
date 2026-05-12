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
WORKTREES_DIR_RAW="${MAIN_DIR}/../${MAIN_BASENAME}-worktrees"
# Canonicalize: git records the resolved path in `worktree list --porcelain`,
# so subsequent string-matching against that listing must use the same form
# (no `..` segments). mkdir + cd resolves the parent reliably without
# requiring GNU realpath / readlink -f (portability for macOS).
mkdir -p "$WORKTREES_DIR_RAW"
WORKTREES_DIR="$(cd "$WORKTREES_DIR_RAW" && pwd -P)"
WORKTREE_DIR="${WORKTREES_DIR}/${BRANCH}"

# Worktree directory present? Distinguish fully-set-up (idempotent no-op)
# from partially-set-up (recover by removing and continuing). The previous
# unconditional `exit 1` here was a partial-state trap: any failure later
# in the script (e.g. a non-fast-forward `push -u`, network blip, or
# missing remote) left the worktree directory on disk and blocked all
# subsequent retries.
if [ -d "$WORKTREE_DIR" ]; then
    # Treat as fully set up only if git agrees this is a tracked worktree
    # whose HEAD is on $BRANCH AND whose upstream is origin/$BRANCH.
    # `grep -qxF` requires a full-line match (porcelain emits one entry per
    # line with the exact path) so a sibling prefix collision — e.g.
    # checking QS_17 while QS_173 also exists — cannot false-positive.
    IS_TRACKED_WORKTREE="no"
    if git -C "$MAIN_DIR" worktree list --porcelain \
        | grep -qxF "worktree ${WORKTREE_DIR}"; then
        IS_TRACKED_WORKTREE="yes"
    fi
    if [ "$IS_TRACKED_WORKTREE" = "yes" ]; then
        ACTUAL_HEAD="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
        ACTUAL_UPSTREAM="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo "")"
        if [ "$ACTUAL_HEAD" = "$BRANCH" ] && [ "$ACTUAL_UPSTREAM" = "origin/${BRANCH}" ]; then
            echo "Worktree ${WORKTREE_DIR} already set up on ${BRANCH} tracking origin/${BRANCH}."
            exit 0
        fi
        echo "Worktree ${WORKTREE_DIR} exists but is in a partial state"
        echo "  (HEAD='${ACTUAL_HEAD}', upstream='${ACTUAL_UPSTREAM}')."
        # Refuse to silently destroy uncommitted work or stashes. The
        # recovery branch runs `worktree remove --force` which drops
        # uncommitted edits without warning — guard with an explicit
        # cleanliness check before touching disk.
        DIRTY=""
        STASHES=""
        if [ -d "${WORKTREE_DIR}/.git" ] || [ -f "${WORKTREE_DIR}/.git" ]; then
            DIRTY="$(git -C "$WORKTREE_DIR" status --porcelain 2>/dev/null || true)"
            STASHES="$(git -C "$WORKTREE_DIR" stash list 2>/dev/null || true)"
        fi
        if [ -n "$DIRTY" ] || [ -n "$STASHES" ]; then
            echo "Error: worktree ${WORKTREE_DIR} has uncommitted changes or stashes."
            echo "  Status: ${DIRTY:-(none)}"
            echo "  Stashes: ${STASHES:-(none)}"
            echo "Refusing to destructively recover. Resolve manually:"
            echo "  cd ${WORKTREE_DIR} && git status"
            exit 1
        fi
        echo "Removing tracked worktree to recover..."
        # Surface a clear remediation message if `worktree remove --force`
        # fails (locked worktree, corrupted .git file, permission). Without
        # this, `set -e` aborts with no actionable hint.
        if ! git -C "$MAIN_DIR" worktree remove --force "$WORKTREE_DIR"; then
            echo "Error: 'git worktree remove --force' failed on ${WORKTREE_DIR}."
            echo "The worktree may be locked. Try:"
            echo "  git worktree unlock ${WORKTREE_DIR}"
            echo "  git worktree remove --force ${WORKTREE_DIR}"
            echo "…then re-run this script."
            exit 1
        fi
    else
        echo "Directory ${WORKTREE_DIR} exists but is not a tracked git worktree."
        echo "Removing stale directory to recover..."
        rm -rf "$WORKTREE_DIR"
    fi
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
    git -C "$MAIN_DIR" worktree add "$WORKTREE_DIR" -b "$BRANCH" origin/main
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

# Detect non-fast-forward divergence BEFORE attempting `push -u`. Without
# this, a relaxed re-run after a failed push against a diverged
# origin/${BRANCH} always re-attempts the push and re-hits the same
# failure ad infinitum until the operator intervenes manually. Use
# `ls-remote` rather than a full `fetch` — cheaper and avoids mutating
# local refs in a setup script.
REMOTE_TIP="$(git -C "$WORKTREE_DIR" ls-remote origin "$BRANCH" 2>/dev/null | awk '{print $1}')"
if [ -n "$REMOTE_TIP" ]; then
    LOCAL_TIP="$(git -C "$WORKTREE_DIR" rev-parse "$BRANCH")"
    if [ "$REMOTE_TIP" != "$LOCAL_TIP" ] && \
       ! git -C "$WORKTREE_DIR" merge-base --is-ancestor "$REMOTE_TIP" "$LOCAL_TIP" 2>/dev/null; then
        echo "Error: origin/${BRANCH} has commits not in local ${BRANCH}."
        echo "  Remote tip: ${REMOTE_TIP}"
        echo "  Local tip:  ${LOCAL_TIP}"
        echo "Refusing to push (would be non-fast-forward)."
        echo "Resolve manually:"
        echo "  cd ${WORKTREE_DIR} && git fetch && git rebase origin/${BRANCH}"
        echo "…then re-run this script (or just 'git push')."
        exit 1
    fi
fi

# Publish the branch to origin and set upstream → origin/${BRANCH}.
# This makes `git push` / `git pull` Just Work without flags and guarantees
# the upstream is the feature branch, never origin/main (regardless of
# git's branch.autoSetupMerge default). Idempotent: if origin/${BRANCH}
# already exists and matches the local tip, push is a no-op but still
# (re)sets the upstream — repairing any pre-existing branch that was
# created with the old buggy origin/main upstream.
# Positioned AFTER the HEAD verification block so we never publish content
# from an unexpected HEAD to origin/${BRANCH}.
echo "Publishing ${BRANCH} to origin and setting upstream..."
git -C "$WORKTREE_DIR" push -u origin "${BRANCH}"

echo ""
echo "Worktree ready: ${WORKTREE_DIR}"
echo "Branch: ${BRANCH} (HEAD verified)"
echo "To start working: cd ${WORKTREE_DIR} && source venv/bin/activate"
