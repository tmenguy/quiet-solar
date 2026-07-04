---
description: TDD implementation for dev-environment changes (scripts/, .claude/, .cursor/, .opencode/, legacy/, docs/, .github/, top-level config). Pass the impacted quality gate (--impacted; coverage-vacuous on dev-only trees), open PR.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-implement-setup-task` (interactive session — you
> can answer "Ready to run the quality gate?" and drive the
> implementation mid-flight).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-implement-setup-task** subagent to handle this. The
subagent discovers task context from the branch name.

Expected outcome:
- TDD-implemented changes scoped strictly to dev-environment paths.
- `python scripts/qs/quality_gate.py --impacted` passes (dev-only
  changes carry no product-coverage delta, so that side is a fast
  no-op; the tooling's own testmon-selected tests still run).
- For md-only edits pinned by `tests/qs`,
  `python scripts/qs/quality_gate.py --quick tests/qs` also passes
  before commit (testmon cannot select doc-pinning tests for markdown
  changes, so `--impacted` alone is vacuous there).
- Auto-committed, pushed, PR opened.
- Next-phase command printed: launcher form (`claude --agent
  qs-review-task`) plus slash-command fallback (`/review-task`).

User request:
$ARGUMENTS
