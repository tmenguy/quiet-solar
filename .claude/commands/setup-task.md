---
description: Create GitHub issue + feature branch QS_<N> + worktree, then print the launcher for the new session.
---

Use the **qs-setup-task** subagent to handle this. Pass the user's full
request verbatim as the task description.

Expected outcome:
- Issue created (or fetched if `--issue N` was passed).
- Branch `QS_<N>` created from `origin/main`.
- Worktree created at `../<repo>-worktrees/QS_<N>/` (unless
  `--no-worktree`).
- Launcher command printed for the user to open a new session on the
  worktree. The next phase is `/create-plan`.

User request:
$ARGUMENTS
