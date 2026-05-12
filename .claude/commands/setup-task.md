---
description: Create GitHub issue + feature branch QS_<N> + worktree, then print the launcher for the new session.
---

> **Preferred entry**: open a fresh terminal in the main checkout and run
> `claude --agent qs-setup-task` (interactive session — you can converse
> with the persona mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-setup-task** subagent to handle this. Pass the user's full
request verbatim as the task description.

Expected outcome:
- Issue created (or fetched if `--issue N` was passed).
- Branch `QS_<N>` created from `origin/main`.
- Worktree created at `../<repo>-worktrees/QS_<N>/` (unless
  `--no-worktree`).
- Launcher command printed for the user to open a fresh interactive
  `claude --agent qs-create-plan` session on the worktree (preferred),
  with `/create-plan` as the slash-command fallback for Claude Desktop.

User request:
$ARGUMENTS
