---
name: qs-finish-task
description: >-
  Phase 5 of the QS pipeline. Verifies CI passes, merges the PR (with
  user authorization), cleans up worktree + remote branch.
model: inherit
readonly: false
is_background: false
---

# qs-finish-task — merge and cleanup

You verify CI green, get explicit merge authorization, merge the PR,
delete the remote branch, and remove the worktree.

## Discover the task context

```bash
python scripts/qs/context.py
```

If `pr_number` is null, STOP.

## Phase protocol

### 1. Show PR summary
```bash
gh pr view {{pr_number}}
```

### 2. Verify CI
```bash
gh pr checks {{pr_number}}
```
- **Failed** → STOP, report.
- **Pending** → advise wait.
- **Green** → proceed.

### 3. Authorize merge

Ask **explicitly**: "Ready to merge PR #{{pr_number}}?". Wait for
"yes" / "merge".

### 4. Merge the PR
```bash
gh pr merge {{pr_number}} --merge
```

### 5. Delete the remote branch

Refuse if `{{branch}}` is `main` or `master`.

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

### 7. Report

```text
✅ PR #{{pr_number}} merged.
✅ Branch deleted.
✅ Worktree removed.

(If production code was touched) Run /qs-release from main when ready.
```

## Hard rules

- No merge without explicit user authorization.
- Never auto-chain to release.
- Refuse to delete `main`/`master`.
