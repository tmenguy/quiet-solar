---
description: Draft the story artifact with acceptance criteria, run adversarial review with 4 parallel sub-agents, commit and push.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-create-plan` (interactive session — you can answer
> the persona's clarifying questions in step 2 of its protocol).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-create-plan** subagent to handle this. The subagent will
discover the current task context from the branch name (`QS_<N>`) and
the GitHub issue.

Expected outcome:
- Story written to `docs/stories/QS-<N>.story.md` with
  acceptance criteria (Given/When/Then) and a task breakdown.
- 4-reviewer adversarial review run in parallel; findings triaged
  interactively.
- Story committed and pushed.
- Next-phase command printed: launcher form (`claude --agent
  qs-implement-task` or `claude --agent qs-implement-setup-task`)
  plus slash-command fallback (`/implement-task` or
  `/implement-setup-task`).

User request:
$ARGUMENTS
