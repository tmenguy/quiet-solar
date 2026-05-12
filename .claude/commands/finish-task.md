---
description: Callable any time. If a PR exists, merge it (with auth) then clean up. Otherwise just clean up the worktree + remote branch.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-finish-task` (interactive session — you can answer
> "Ready to merge PR #N?" and authorize the force-delete prompts
> mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-finish-task** subagent to handle this. The subagent
discovers PR + branch + worktree from the current branch name and
branches behavior on whether a PR exists.

Expected outcome:

- **PR exists, open** — CI verified, user authorizes, `gh pr merge`,
  remote branch deleted, worktree removed.
- **PR exists, already merged** — skip merge, just clean up branch +
  worktree.
- **PR exists, closed unmerged** — user authorizes cleanup, then clean
  up.
- **No PR** — fast-path cleanup. If worktree has uncommitted/unpushed
  work, user is shown what would be lost and asked for force-delete
  authorization. Otherwise removed immediately.

Hard rules: never runs the quality gate, refuses to delete
`main`/`master`, never auto-runs `/release`.

User request:
$ARGUMENTS
