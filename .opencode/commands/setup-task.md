---
description: >-
  Phase 1 — set up a new Quiet Solar task: create a GitHub issue, feature
  branch, worktree, and render the per-task qs-create-plan-QS-<N>.md agent
  into the worktree. Emits a launcher for a new OpenCode session on the
  worktree. Runs in the home OpenCode on main; never touches main's checkout.
agent: qs-setup-task
subtask: true
---

Delegate to the `qs-setup-task` subagent. Pass the user's full request
(issue number, story key, free-text description, `--plan` path, or flags)
verbatim. The subagent owns the full setup-task protocol.

Expected outcome:
- GitHub issue created or reused.
- Feature branch `QS_<N>` and worktree created via `scripts/worktree-setup.sh`.
- Per-task agent file `qs-create-plan-QS-<N>.md` **rendered into the new
  worktree** with issue-specific context and narrow permissions.
- Launcher payload printed (terminal + optional PyCharm variants) for the
  user to start a fresh OpenCode session on the worktree.

Handoff model: this is the **only** slash command in the OpenCode pipeline.
All subsequent phases are activated by name inside the worktree's OpenCode
session (the agent files are rendered per-task with full context baked in).
Do NOT chain to any other phase from this session.
