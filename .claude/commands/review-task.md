---
description: Run adversarial review on the PR with 4 parallel reviewer sub-agents, drive interactive triage, generate a fix plan if needed.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-review-task` (interactive session — you can drive
> "fix all / skip all / one by one?" triage mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-review-task** subagent to handle this. The subagent
discovers PR + story file from the branch name.

Expected outcome:
- 4 reviewer subagents spawned in parallel (blind-hunter,
  edge-case-hunter, acceptance-auditor, coderabbit).
- Findings consolidated and triaged interactively with the user.
- If findings remain, a fix plan written to
  `docs/stories/QS-<N>.story_review_fix_#NN.md`, plus the launcher form
  (`claude --agent qs-implement-task`) and slash-command fallback
  (`/implement-task`) for the user to apply it.
- If clean, next-phase command printed: launcher form (`claude --agent
  qs-finish-task`) plus slash-command fallback (`/finish-task`).

User request:
$ARGUMENTS
