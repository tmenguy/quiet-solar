---
description: TDD implementation of the story under custom_components/quiet_solar/, pass full quality gate, open PR.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-implement-task` (interactive session — you can
> answer "Ready to run the quality gate?" and drive the implementation
> mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-implement-task** subagent to handle this. The subagent
discovers task context from the branch name.

Expected outcome:
- TDD-implemented code under `custom_components/quiet_solar/` and tests.
- `python scripts/qs/quality_gate.py` passes (pytest 100% cov + ruff +
  mypy + translations).
- Auto-committed, pushed, PR opened.
- Next-phase command printed: launcher form (`claude --agent
  qs-review-task`) plus slash-command fallback (`/review-task`).

User request:
$ARGUMENTS
