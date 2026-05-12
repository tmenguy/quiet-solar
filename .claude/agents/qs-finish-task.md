---
name: qs-finish-task
description: >-
  Phase 5 of the QS pipeline. Callable at any moment. If a PR exists,
  verifies CI and merges it (with user authorization). Otherwise just
  cleans up the worktree and remote branch — no quality gate, no merge
  dance. Use when the user says "finish task", "merge PR", "abandon
  task", or "clean up worktree".
tools: Bash, Read
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

5. Delete the remote branch if it exists (safety: refuse `main` /
   `master`):
   ```bash
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

6. Delete the remote branch (refuse `main` / `master`):
   ```bash
   git push origin --delete {{branch}}
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

   Production code was touched → run /release from the main checkout
   when you're ready to ship a release.
   ```
   (Skip the release suggestion when no `custom_components/quiet_solar/`
   files were in the diff — check `gh pr diff {{pr_number}} --name-only`
   or just inspect the file list.)

## Hard rules

- **Never run the quality gate.** Cleanup must succeed even if tests
  would fail.
- No merge without explicit user authorization in this turn.
- Never auto-chain to `/release` — it's a separate decision.
- Refuse to delete `main` / `master` even if asked.
- In Case A, if the user has unsaved work, ALWAYS show what would be
  lost before asking for force-delete authorization.
