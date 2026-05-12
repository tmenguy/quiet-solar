---
description: Verify CI green, merge PR with user authorization, delete remote branch, remove worktree.
---

Use the **qs-finish-task** subagent to handle this. The subagent
discovers PR + branch + worktree from the current branch name.

Expected outcome:
- CI status verified (`gh pr checks <PR>`); blocked if checks failed.
- User explicitly authorizes merge.
- `gh pr merge --merge` succeeds.
- Remote branch deleted (refuses to delete `main`/`master`).
- Worktree removed via `python scripts/qs/cleanup_worktree.py --force`.
- If production code was touched, user is reminded to run `/release`
  from main.

User request:
$ARGUMENTS
