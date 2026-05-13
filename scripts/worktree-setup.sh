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
mkdir -p "$WORKTREES_DIR_RAW"
# Use the verbatim form for `git worktree add` — git records this exact
# path in `worktree list --porcelain`. Also compute the canonical form
# (`pwd -P` resolves symlinks like macOS's /var → /private/var). The
# membership check below accepts either representation, because git's
# own canonicalization rules vary across versions and platforms — and
# `pwd -P` alone diverges from `git worktree add`'s recorded path on
# macOS, breaking the membership match and triggering an unsafe
# `rm -rf` against a live tracked worktree.
WORKTREES_DIR="$WORKTREES_DIR_RAW"
WORKTREES_DIR_CANON="$(cd "$WORKTREES_DIR_RAW" && pwd -P)"
WORKTREE_DIR="${WORKTREES_DIR}/${BRANCH}"
WORKTREE_DIR_CANON="${WORKTREES_DIR_CANON}/${BRANCH}"

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
    # Accept either the verbatim or canonical form: `git worktree add`'s
    # recorded path may match either, depending on git version and
    # platform symlink layout (macOS /var → /private/var). Multiple `-e`
    # patterns to `grep -F` are portable on macOS bash 3.2 and Linux.
    if git -C "$MAIN_DIR" worktree list --porcelain \
        | grep -E "^worktree " \
        | sed 's/^worktree //' \
        | grep -qxF -e "$WORKTREE_DIR" -e "$WORKTREE_DIR_CANON"; then
        IS_TRACKED_WORKTREE="yes"
    fi
    # FIX #04 MF-1: the cleanliness gate must run only on the destructive
    # arms, NOT before the happy-path no-op detection. A developer with
    # uncommitted edits in a healthy QS_<N> worktree re-running setup
    # (re-symlink, re-verify upstream) was previously blocked by the
    # cleanliness check before reaching the "already set up" early-return.
    # Resolve happy path / detached-HEAD first; only then gate destructive
    # recovery on cleanliness.
    if [ "$IS_TRACKED_WORKTREE" = "yes" ]; then
        ACTUAL_HEAD="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
        # `rev-parse --abbrev-ref HEAD` returns the literal string 'HEAD'
        # in detached state — the equality check below would silently
        # fail and fall through to destructive recovery, but detached
        # HEAD is often intentional inspection. Refuse and ask the
        # operator to resolve.
        if [ "$ACTUAL_HEAD" = "HEAD" ]; then
            echo "Error: worktree ${WORKTREE_DIR} is in detached-HEAD state."
            echo "Refusing to recover automatically — this may be intentional inspection."
            echo "Resolve manually:"
            echo "  cd ${WORKTREE_DIR} && git checkout ${BRANCH}   # or git switch -"
            echo "…then re-run this script."
            exit 1
        fi
        ACTUAL_UPSTREAM="$(git -C "$WORKTREE_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo "")"
        if [ "$ACTUAL_HEAD" = "$BRANCH" ] && [ "$ACTUAL_UPSTREAM" = "origin/${BRANCH}" ]; then
            echo "Worktree ${WORKTREE_DIR} already set up on ${BRANCH} tracking origin/${BRANCH}."
            exit 0
        fi
    fi
    # FIX #04 SF-5: refuse to proceed when $WORKTREE_DIR is unreadable.
    # Otherwise `find` emits nothing, HAS_NONGIT_CONTENT stays empty, and
    # recovery falls through to `rm -rf` which then fails messily.
    if ! { [ -r "$WORKTREE_DIR" ] && [ -x "$WORKTREE_DIR" ]; }; then
        echo "Error: ${WORKTREE_DIR} exists but is not readable/executable."
        echo "Permission issue (possibly sudo-owned from a prior run). Resolve manually:"
        echo "  ls -la ${WORKTREE_DIR}"
        echo "  sudo chown -R \$(whoami) ${WORKTREE_DIR}   # if appropriate"
        exit 1
    fi
    # Refuse to silently destroy uncommitted work, stashes, or stray
    # user-created files. Both destructive recovery arms below
    # (`worktree remove --force` and `rm -rf`) drop content without
    # warning — guard with an explicit cleanliness check before touching
    # disk. The check has to cope with both a real tracked worktree
    # (`.git` present, `git status`/`git stash list` work) and an
    # untracked stale directory (may have no `.git`, may just contain
    # user-created files).
    DIRTY=""
    STASHES=""
    HAS_NONGIT_CONTENT=""
    if [ -d "${WORKTREE_DIR}/.git" ] || [ -f "${WORKTREE_DIR}/.git" ]; then
        # `if ! ...` rather than `2>/dev/null || true`: a real `git
        # status` failure (corrupt index, broken .git file) must surface
        # — not be swallowed and mistaken for "clean".
        if ! DIRTY="$(git -C "$WORKTREE_DIR" status --porcelain 2>/dev/null)"; then
            echo "Error: 'git status' failed in ${WORKTREE_DIR}."
            echo "The worktree's git metadata may be corrupted. Resolve manually:"
            echo "  cd ${WORKTREE_DIR} && git status   # inspect error"
            exit 1
        fi
        if ! STASHES="$(git -C "$WORKTREE_DIR" stash list 2>/dev/null)"; then
            # stash list is repo-wide; a failure here means the git command
            # itself broke. Leave STASHES empty rather than abort — the
            # status check above already gates real metadata corruption.
            STASHES=""
        fi
    fi
    # FIX #04 MF-2 / SF-1 / SF-3: detect visible content in the directory,
    # capturing the first few offending filenames into the diagnostic.
    # Exclusions:
    #   - `.git` (exact): the tracked worktree's git dir/file.
    #   - `.DS_Store`, `.idea`, `.vscode`, `.envrc`, `.python-version`:
    #     common macOS/IDE/tool dotfiles that aren't user-meaningful.
    # Do NOT exclude `.gitignore`, `.gitattributes`, `.gitmodules` —
    # if those are loose in a stale directory, the operator probably
    # wants to know before we wipe them.
    if [ -d "$WORKTREE_DIR" ]; then
        OFFENDING="$(find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 \
            ! -name '.git' \
            ! -name '.DS_Store' \
            ! -name '.idea' \
            ! -name '.vscode' \
            ! -name '.envrc' \
            ! -name '.python-version' \
            -print 2>/dev/null | head -5)"
        if [ -n "$OFFENDING" ]; then
            HAS_NONGIT_CONTENT="$OFFENDING"
        fi
    fi
    if [ -n "$DIRTY" ] || [ -n "$STASHES" ] || \
       { [ -z "$DIRTY" ] && [ -z "$STASHES" ] && [ "$IS_TRACKED_WORKTREE" = "no" ] && [ -n "$HAS_NONGIT_CONTENT" ]; }; then
        echo "Error: ${WORKTREE_DIR} has uncommitted changes, stashes, or non-git content."
        echo "  Status:  ${DIRTY:-(none)}"
        echo "  Stashes: ${STASHES:-(none)}"
        echo "  Other:   ${HAS_NONGIT_CONTENT:-(none)}"
        echo "Refusing to destructively recover. Resolve manually:"
        echo "  ls -la ${WORKTREE_DIR}"
        exit 1
    fi
    if [ "$IS_TRACKED_WORKTREE" = "yes" ]; then
        echo "Worktree ${WORKTREE_DIR} exists but is in a partial state"
        echo "  (HEAD='${ACTUAL_HEAD}', upstream='${ACTUAL_UPSTREAM}')."
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
# FIX #04 SF-4: previously `2>/dev/null` suppressed network/auth errors,
# leaving REMOTE_TIP empty and silently turning the divergence guard
# into a no-op. Capture stderr alongside stdout and check the exit code
# explicitly so a genuine ls-remote failure is surfaced. An empty
# REMOTE_TIP now unambiguously means "branch doesn't exist on origin"
# rather than "we couldn't tell".
REMOTE_TIP_OUTPUT=""
LS_REMOTE_RC=0
REMOTE_TIP_OUTPUT="$(git -C "$WORKTREE_DIR" ls-remote --heads origin "$BRANCH" 2>&1)" || LS_REMOTE_RC=$?
if [ "$LS_REMOTE_RC" -ne 0 ]; then
    echo "Error: 'git ls-remote --heads origin ${BRANCH}' failed."
    echo "  Output: ${REMOTE_TIP_OUTPUT}"
    echo "Cannot verify divergence before push. Resolve manually:"
    echo "  cd ${WORKTREE_DIR} && git ls-remote --heads origin ${BRANCH}"
    exit 1
fi
REMOTE_TIP="$(echo "$REMOTE_TIP_OUTPUT" | awk '{print $1}')"
if [ -n "$REMOTE_TIP" ]; then
    LOCAL_TIP="$(git -C "$WORKTREE_DIR" rev-parse "$BRANCH")"
    if [ "$REMOTE_TIP" != "$LOCAL_TIP" ]; then
        # `merge-base --is-ancestor` errors (not "false") if REMOTE_TIP
        # isn't reachable locally. The outer `!` inversion would then
        # misclassify "unfetched" as "divergence". Guarantee reachability
        # by fetching the branch once if needed.
        if ! git -C "$WORKTREE_DIR" cat-file -e "${REMOTE_TIP}^{commit}" 2>/dev/null; then
            echo "Fetching origin/${BRANCH} to compare ancestry..."
            if ! git -C "$WORKTREE_DIR" fetch origin "$BRANCH" 2>/dev/null; then
                echo "Error: failed to fetch origin/${BRANCH} for divergence check."
                echo "Network or remote-access issue. Resolve manually:"
                echo "  cd ${WORKTREE_DIR} && git fetch origin ${BRANCH}"
                exit 1
            fi
        fi
        if ! git -C "$WORKTREE_DIR" merge-base --is-ancestor "$REMOTE_TIP" "$LOCAL_TIP" 2>/dev/null; then
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
# Wrap the push with explicit failure handling: under bare `set -e` the
# script aborts with no actionable hint on auth expiry, network drop,
# pre-push hook reject, or credential prompt hang.
# FIX #04 SF-2: previous wording said "Worktree IS fully set up locally
# — only the publish step failed", which misled operators into thinking
# a re-run would skip straight past setup. In fact the partial-state
# recovery arm rebuilds the worktree from scratch on the next run.
# Point at the manual retry as the primary remediation; note re-running
# as a slower-but-safe alternative.
if ! git -C "$WORKTREE_DIR" push -u origin "${BRANCH}"; then
    echo "Error: 'git push -u origin ${BRANCH}' failed."
    echo "Common causes: auth expired, network drop, pre-push hook reject."
    echo "The worktree is built locally and the failure is just in the"
    echo "publish step. The simplest fix is to retry the push directly:"
    echo "  cd ${WORKTREE_DIR} && git push -u origin ${BRANCH}"
    echo "Re-running this script would rebuild the worktree from scratch"
    echo "(slower, but harmless if you prefer that)."
    exit 1
fi

echo ""
echo "Worktree ready: ${WORKTREE_DIR}"
echo "Branch: ${BRANCH} (HEAD verified)"
echo "To start working: cd ${WORKTREE_DIR} && source venv/bin/activate"
