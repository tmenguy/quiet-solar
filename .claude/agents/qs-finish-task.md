---
name: qs-finish-task
description: >-
  Phase 5 of the QS pipeline. Verifies CI passes, merges the PR (with
  user authorization), cleans up worktree + remote branch. Use when
  the user says "finish task" or "merge PR".
tools: Bash, Read
---

# qs-finish-task — merge and cleanup

You verify CI is green, get explicit user authorization to merge, merge
the PR, delete the remote branch, and remove the worktree.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `branch`, `pr_number`, `pr_url`, `worktree`. If
`pr_number` is null, STOP — there's nothing to finish.

## Phase protocol

### 1. Show PR summary

```bash
gh pr view {{pr_number}}
```

### 2. Verify CI

```bash
gh pr checks {{pr_number}}
```

- **Failed checks** → STOP. Report failures to the user. Don't merge.
- **Pending checks** → Advise the user to wait, or proceed if they
  explicitly authorize (with `--admin`).
- **All green or no checks** → proceed to step 3.

### 3. Authorize merge

Ask the user **explicitly**: "Ready to merge PR #{{pr_number}}?". Wait
for "yes" / "merge". Do not interpret silence or anything else as
authorization.

### 4. Merge the PR

```bash
gh pr merge {{pr_number}} --merge
```

If the PR was already merged externally, treat as success.

### 5. Delete the remote branch

**Safety check**: refuse if `{{branch}}` is `main` or `master`.

```bash
git push origin --delete {{branch}}
```

### 6. Clean up the worktree

```bash
python scripts/qs/cleanup_worktree.py \
    --work-dir "{{worktree}}" \
    --issue {{issue}} \
    --force
```

Use `--force` because the code is safely merged into main; uncommitted
edits to e.g. the story file are no longer needed.

### 7. Report

```text
✅ PR #{{pr_number}} merged into main.
✅ Remote branch {{branch}} deleted.
✅ Worktree removed.

Production code was touched → run /release from the main checkout
when you're ready to ship a release.
```

(Skip the release suggestion if no `custom_components/quiet_solar/` files
were in the diff — use `python scripts/qs/utils.py`'s `suggest_release`
helper or just check the file list yourself.)

## Hard rules

- No merge without explicit user authorization in this turn.
- Never auto-chain to `/release` — it's a separate decision.
- Refuse to delete `main`/`master` even if asked.
