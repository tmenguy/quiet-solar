# /finish-story

Final quality gate, merge PR, clean up worktree, update epics.

## Input

- `--pr N` (required): PR number
- `--story-key X.Y` (optional): story key to mark as DONE in epics.md

## Steps

### 0.5. Mandatory doc-sync gate

Before anything else, run the automated doc-sync check:

```bash
python scripts/qs/doc_sync.py {{story_file}} --base-branch main
```

This compares the story artifact's tasks, acceptance criteria, and dev notes against the actual git diff. Review its output, then:

1. **For each discrepancy**: present it to the user with context
2. **User resolves each**: update the doc, or explain why the discrepancy is acceptable
3. **Apply approved changes** and stage them (`_bmad-output/` is included in the commit step)

Also do a manual read of the story artifact to catch anything the script can't — e.g., ACs whose intent doesn't match the implementation, or dev notes that are stale.

This gate is **mandatory** — do NOT proceed to merge until all discrepancies are resolved.

### 1. Final quality gate

```bash
python scripts/qs/quality_gate.py
```

If it fails, fix issues before proceeding. Do NOT skip.

### 2. Push any remaining changes

```bash
git add custom_components/ tests/ _bmad-output/ _qsprocess/ scripts/ && git status
```

If there are uncommitted changes, ask the user to confirm, then commit and push:
```bash
git commit -m "fix: address review feedback"
git push
```

### 3. Merge and clean up

```bash
python scripts/qs/finish_story.py {{pr_number}} --story-key "{{story_key}}"
```

This script handles:
- Merging the PR (merge commit, not squash)
- Deleting the remote branch
- Cleaning up the worktree
- Switching main to latest
- Marking the story as DONE in epics.md

### 4. Commit epics update

If epics.md was updated:
```bash
cd {{main_worktree}}
git add _bmad-output/planning-artifacts/epics.md
git commit -m "docs: mark story {{story_key}} as done"
git push origin main
```

### 5. Report

Show the user:
- PR merge status
- Worktree cleanup status
- Epics update status
- Suggest `/release` if appropriate
