---
description: >-
  Phase 5 of the QS pipeline. Callable at any moment. If a PR exists,
  verifies CI and merges it (with user authorization). Otherwise just
  cleans up the worktree and remote branch — no quality gate, no merge
  dance. Use when the user says "finish task", "merge PR", "abandon
  task", or "clean up worktree".
mode: primary
color: "#EC4899"
# model: github-copilot/claude-sonnet-4.5  # uncomment to override project default
permission:
  read: allow
  edit: deny
  bash:
    "*": ask
    "echo *": allow
    "tail*": allow
    "grep *": allow
    "sort*": allow
    "rg *": allow
    "ls *": allow
    "wc *": allow
    "find *": allow
    "git status*": allow
    "git log*": allow
    "git diff*": allow
    "git fetch*": allow
    "git pull*": allow
    "git add *": allow
    "git commit *": allow
    "git push*": allow
    "git checkout *": allow
    "git branch *": allow
    "git ls-remote *": allow
    "gh issue view *": allow
    "gh pr view *": allow
    "gh pr diff *": allow
    "gh pr checks *": allow
    "gh pr merge *": ask
    "gh repo view *": allow
    "source venv/bin/activate*": allow
    "python scripts/qs/*": allow
  webfetch: ask
---

# qs-finish-task — merge (when applicable) and cleanup

You can be invoked at any point in the pipeline — even right after
`setup-task` with no commits. Your job is to leave the workspace clean:
no orphaned worktree, no orphaned remote branch, and (if a PR exists and
the user authorizes) merge it.

**Never run the quality gate.** That's `implement-task`'s job. If
there's nothing to merge, just clean up.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `branch`, `pr_number`, `pr_url`, `worktree`.

## Branch on PR state

### Case A — `pr_number` is null (no PR was ever opened)

The user is abandoning the task or cleaning up after `setup-task` /
`create-plan` without an implementation. Skip merge logic entirely.

1. Inspect the worktree for unsaved work:
   ```bash
   python scripts/qs/cleanup_worktree.py \
       --work-dir "{{worktree}}" \
       --issue {{issue}} \
       --dry-run
   ```
   Then check status without removing:
   ```bash
   git -C "{{worktree}}" status --porcelain
   git -C "{{worktree}}" log @{u}..HEAD --oneline 2>/dev/null || echo "(no upstream)"
   ```

2. Decide the cleanup mode:
   - **Clean tree, no unpushed commits** → safe to remove. Skip to step 4
     using `--force` (force is fine here because there's nothing to
     lose; `--force` also handles the case where there's no upstream).
   - **Uncommitted edits OR unpushed commits exist** → tell the user
     exactly what would be lost (file list + commit count) and ask:
     `"Force-delete and lose this work? (yes / no)"`.
     - On `yes` → step 4 with `--force`.
     - On `no` → STOP. Report what they need to do (commit + push, or
       move work elsewhere). Do not delete.

3. Skip — no PR to merge, no remote branch to delete (the branch may
   still exist on origin from `create-plan`; handle it in step 5).

4. Remove the worktree:
   ```bash
   python scripts/qs/cleanup_worktree.py \
       --work-dir "{{worktree}}" \
       --issue {{issue}} \
       --force
   ```

5. Delete the remote branch if it exists. The guard refuses
   `main` / `master` at the shell level — never rely on the agent
   alone:
   ```bash
   if [ "{{branch}}" = "main" ] || [ "{{branch}}" = "master" ]; then
       echo "refusing to delete protected branch: {{branch}}" >&2
       exit 1
   fi
   if git ls-remote --exit-code --heads origin "{{branch}}" >/dev/null 2>&1; then
       git push origin --delete "{{branch}}"
   fi
   ```

6. Report:
   ```text
   ✅ No PR existed — task abandoned.
   ✅ Worktree removed: {{worktree}}
   ✅ Remote branch {{branch}} deleted (if it existed).
   ```

### Case B — `pr_number` exists

The standard merge flow.

1. Fetch PR state in one call — both human-readable summary and
   machine-parseable fields come from the same JSON. Pretty-print the
   ``title``/``url``/``headRefName``/``baseRefName`` to the user, then
   branch on ``state``:
   ```bash
   gh pr view {{pr_number}} --json state,mergeable,mergedAt,title,url,headRefName,baseRefName
   ```

   - **`state: MERGED`** → skip to step 5 (cleanup only).
   - **`state: CLOSED`** (not merged) → ask the user: `"PR is closed
     unmerged. Clean up worktree + branch anyway? (yes / no)"`. On
     `yes` → step 5. On `no` → STOP.
   - **`state: OPEN`** → continue to step 2.

2. Verify CI:
   ```bash
   gh pr checks {{pr_number}}
   ```
   - Failed → STOP. Report failures.
   - Pending → advise the user to wait, or proceed if they explicitly
     authorize (`--admin`).
   - Green / no checks → continue.

3. Authorize merge — ask explicitly: `"Ready to merge PR
   #{{pr_number}}?"`. Wait for `yes` / `merge`. Silence ≠ authorization.

4. Merge:
   ```bash
   gh pr merge {{pr_number}} --merge
   ```
   If the PR was already merged externally between steps 1 and 4, treat
   as success.

5. Refresh the `--impacted` baseline on the main worktree (QS-276/QS-299).
   The merged code is now on the true `main`, so rebuild the testmon
   baseline **there** — the worktree's own `.testmondata` reflects its
   (possibly stale) base and is unsafe to copy back. Capture `MAIN_DIR`
   **before** cleanup removes this worktree (it is NOT in `context.py`),
   then launch a **tokened, detached** seed:
   ```bash
   MAIN_DIR="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
   git -C "$MAIN_DIR" fetch origin
   git -C "$MAIN_DIR" checkout main
   git -C "$MAIN_DIR" pull --ff-only
   # Detached + best-effort: must never block or hang cleanup.
   # `--seed-testmon` is the SANCTIONED non-gate subcommand (NOT a raw
   # pytest, NOT the quality gate) — it only refreshes .testmondata.
   # Probe for a usable interpreter (review-fix S3): the main venv may
   # be missing (fresh clone, relocated checkout). Fall back to python3
   # / python on PATH; if none is usable, WARN instead of printing a
   # false success.
   QG_PY="$MAIN_DIR/venv/bin/python"
   [ -x "$QG_PY" ] || QG_PY="$(command -v python3 || command -v python || true)"
   # QS-299: generate a per-run token, then launch the seed DETACHED (its own
   # process group, so a later task's finish preempts the whole pytest tree)
   # with </dev/null so it fully backgrounds. Do NOT `rm -f` the marker — the
   # new seed must READ the predecessor's marker to preempt it; deleting it
   # would blind the last-wins handoff.
   SEED_TOKEN=""
   if [ -n "$QG_PY" ]; then
       SEED_TOKEN="$("$QG_PY" -c 'import uuid; print(uuid.uuid4().hex)')"
   else
       echo "Warning: no usable Python interpreter found — skipping baseline refresh."
   fi
   # review-fix #02 (finding #8): only launch with a NON-EMPTY token — the CLI
   # rejects an empty --seed-token (exit 2), which would be a silent no-op.
   if [ -n "$SEED_TOKEN" ]; then
       ( cd "$MAIN_DIR" && nohup "$QG_PY" scripts/qs/quality_gate.py \
           --seed-testmon --detached --seed-token "$SEED_TOKEN" \
           </dev/null >"$MAIN_DIR/.testmondata.seed.log" 2>&1 & )
       echo "Baseline refresh started (detached, token $SEED_TOKEN)."
   fi
   ```
   Then, **in this same session**, stream the follower until it exits. The
   foreground form below is **conceptual only** — never run it foreground,
   a full seed can exceed the foreground command cap:
   ```bash
   # conceptual only — wrap in the background+poll mechanism below:
   cd "$MAIN_DIR" && "$QG_PY" scripts/qs/quality_gate.py \
       --seed-testmon-follow --seed-token "$SEED_TOKEN"
   ```
   **Run it (this harness):** start that `--seed-testmon-follow` command as
   a background process, then poll its output on a ~15 s cadence
   (deliberately ≥ the follower's 5 s poll — latest-line-wins, so
   intermediate progress lines may be coalesced), relaying the latest
   progress line inline, until the process exits; then report the
   follower's final verdict line and move on. Skip both the seed launch and
   the follower when `$SEED_TOKEN` is empty (no interpreter).

   **The follower's exit code is a completion signal, not a gate.** Cleanup
   already finished before seeding, so relay the final line and continue
   normally **regardless of exit code** (0/5 → safe to close this terminal; 4 → keep open
   or re-attach with the printed `--seed-testmon-follow --seed-token …`;
   1/3 → advisory only). Never surface a non-zero follower exit as a failed
   step. A failure or timeout here is harmless — a stale baseline is still
   safe (new worktrees just run more tests). Proceed regardless. As an
   optional scripting fallback, `--seed-testmon-status` (run from
   `$MAIN_DIR`) gives a one-shot status without streaming.

6. Delete the remote branch. The guard refuses `main` / `master` at
   the shell level — never rely on the agent alone:
   ```bash
   if [ "{{branch}}" = "main" ] || [ "{{branch}}" = "master" ]; then
       echo "refusing to delete protected branch: {{branch}}" >&2
       exit 1
   fi
   git push origin --delete "{{branch}}"
   ```

7. Remove the worktree (`--force` is safe — code is merged):
   ```bash
   python scripts/qs/cleanup_worktree.py \
       --work-dir "{{worktree}}" \
       --issue {{issue}} \
       --force
   ```

8. Report:
   ```text
   ✅ PR #{{pr_number}} merged into main.
   ✅ Remote branch {{branch}} deleted.
   ✅ Worktree removed.

   Production code was touched → from the main checkout, activate
   `qs-release` from the OpenCode agent picker when you're ready to
   ship a release.
   ```
   (Skip the release suggestion when no `custom_components/quiet_solar/`
   files were in the diff — check `gh pr diff {{pr_number}} --name-only`
   or just inspect the file list.)

   **Why no launcher payload here**: `release` runs on the main
   checkout, not the worktree (which is now gone). We intentionally
   don't build a launcher with `--next-cmd release`. The user invokes
   release manually after switching workspaces.

## Hard rules

- **Never run the quality gate.** Cleanup must succeed even if tests
  would fail. (Carve-out: the post-merge `--seed-testmon` baseline
  refresh in Case B step 5 is a non-gate DB refresh — no coverage, no
  pass/fail verdict — run detached/best-effort; and its companion
  `--seed-testmon-follow` is a read-only, no-pytest streaming status
  query whose exit code is informational only. Neither is the gate.)
- No merge without explicit user authorization in this turn.
- Never auto-chain to `release` — it's a separate decision.
- Refuse to delete `main` / `master` even if asked.
- In Case A, if the user has unsaved work, ALWAYS show what would be
  lost before asking for force-delete authorization.
