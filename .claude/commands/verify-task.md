---
description: Run fix-verification on the bug × product lane PR with 3 parallel reviewer sub-agents (edge-case-hunter, coderabbit, regression-proof), drive interactive triage, generate a fix plan if needed.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-verify-task` (interactive session — you can drive
> "fix all / skip all / one by one?" triage mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-verify-task** subagent to handle this. The subagent
discovers PR + story file from the branch name and reads its lane file
(`docs/workflow/lanes/bug-product.md`).

Expected outcome:
- 3 reviewer subagents spawned in parallel (edge-case-hunter,
  coderabbit, regression-proof). `qs-review-blind-hunter` and
  `qs-review-acceptance-auditor` do NOT run in this lane.
- Findings consolidated and triaged interactively with the user.
- If findings remain, a fix plan written to
  `docs/stories/QS-<N>.story_review_fix_#NN.md`, plus the launcher form
  (`claude --agent qs-implement-task`) and the matching slash-command
  fallback (`/implement-task`); then re-run `/verify-task`.
- If clean, next-phase command printed: launcher form (`claude --agent
  qs-finish-task`) plus slash-command fallback (`/finish-task`).

User request:
$ARGUMENTS
