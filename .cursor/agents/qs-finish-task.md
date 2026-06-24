---
name: qs-finish-task
description: >-
  Phase 5 of the QS pipeline. Verifies CI passes, merges the PR (with
  user authorization), cleans up worktree + remote branch.
model: inherit
readonly: false
is_background: false
---

# qs-finish-task — merge (when applicable) and cleanup

You can be invoked at any point in the pipeline — even right after
`/setup-task` with no commits. Your job is to leave the workspace clean:
no orphaned worktree, no orphaned remote branch, and (if a PR exists and
the user authorizes) merge it.

**Never run the quality gate.** That's `/implement-task`'s job. If
there's nothing to merge, just clean up.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `branch`, `pr_number`, `pr_url`, `worktree`.

## Branch on PR state

### Case A — `pr_number` is null (no PR was ever opened)

The user is abandoning the task or cleaning up after `/setup-task` /
`/create-plan` without an implementation. Skip merge logic entirely.

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
   still exist on origin from `/create-plan`; handle it in step 5).

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

1. Show PR summary:
   ```bash
   gh pr view {{pr_number}}
   ```

2. Inspect PR state from the JSON:
   ```bash
   gh pr view {{pr_number}} --json state,mergeable,mergedAt
   ```

   - **`state: MERGED`** → skip to step 6 (cleanup only).
   - **`state: CLOSED`** (not merged) → ask the user: `"PR is closed
     unmerged. Clean up worktree + branch anyway? (yes / no)"`. On
     `yes` → step 6. On `no` → STOP.
   - **`state: OPEN`** → continue to step 3.

3. Verify CI:
   ```bash
   gh pr checks {{pr_number}}
   ```
   - Failed → STOP. Report failures.
   - Pending → advise the user to wait, or proceed if they explicitly
     authorize (`--admin`).
   - Green / no checks → continue.

4. Authorize merge — ask explicitly: `"Ready to merge PR
   #{{pr_number}}?"`. Wait for `yes` / `merge`. Silence ≠ authorization.

5. Merge:
   ```bash
   gh pr merge {{pr_number}} --merge
   ```
   If the PR was already merged externally between steps 2 and 5, treat
   as success.

6. Refresh the `--impacted` baseline on the main worktree (QS-276). The
   merged code is now on the true `main`, so rebuild the testmon
   baseline **there** — the worktree's own `.testmondata` reflects its
   (possibly stale) base and is unsafe to copy back. Capture `MAIN_DIR`
   **before** cleanup removes this worktree (it is NOT in `context.py`):
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
   if [ -n "$QG_PY" ]; then
       ( cd "$MAIN_DIR" && nohup "$QG_PY" \
           scripts/qs/quality_gate.py --seed-testmon >/dev/null 2>&1 & )
       echo "Baseline refresh started (detached, best-effort)."
   else
       echo "Warning: no usable Python interpreter found — skipping baseline refresh."
   fi
   ```
   A failure or timeout here is harmless — a stale baseline is still
   safe (new worktrees just run more tests). Proceed regardless. The
   first refresh after a large merge may approach a near-full run;
   acceptable because it is detached.

7. Delete the remote branch. The guard refuses `main` / `master` at
   the shell level — never rely on the agent alone:
   ```bash
   if [ "{{branch}}" = "main" ] || [ "{{branch}}" = "master" ]; then
       echo "refusing to delete protected branch: {{branch}}" >&2
       exit 1
   fi
   git push origin --delete "{{branch}}"
   ```

8. Remove the worktree (`--force` is safe — code is merged):
   ```bash
   python scripts/qs/cleanup_worktree.py \
       --work-dir "{{worktree}}" \
       --issue {{issue}} \
       --force
   ```

9. Report:
    ```text
    ✅ PR #{{pr_number}} merged into main.
    ✅ Remote branch {{branch}} deleted.
    ✅ Worktree removed.

    Production code was touched → from the main checkout, select
    qs-release from the Cursor agent picker when you're ready to
    ship a release.
    ```
    (Skip the release suggestion when no `custom_components/quiet_solar/`
    files were in the diff — check `gh pr diff {{pr_number}} --name-only`
    or just inspect the file list.)

    **Why no launcher payload here**: `/release` runs on the main
    checkout, not the worktree (which is now gone). We intentionally
    don't build a launcher with `--next-cmd release` — see QS-175 OUT OF
    SCOPE. The user invokes release manually after switching workspaces.

## Hard rules

- **Never run the quality gate.** Cleanup must succeed even if tests
  would fail. (Carve-out: the post-merge `--seed-testmon` baseline
  refresh in Case B step 6 is a non-gate DB refresh — no coverage, no
  pass/fail verdict — run detached/best-effort. It is not the gate.)
- No merge without explicit user authorization in this turn.
- Never auto-chain to `/release` — it's a separate decision.
- Refuse to delete `main` / `master` even if asked.
- In Case A, if the user has unsaved work, ALWAYS show what would be
  lost before asking for force-delete authorization.
