---
description: TDD implementation for dev-environment changes (scripts/, .claude/, .cursor/, .opencode/, _qsprocess_opencode/, docs/, .github/, top-level config). Quality gate runs in dev-only fast path.
---

Use the **qs-implement-setup-task** subagent to handle this. The
subagent discovers task context from the branch name.

Expected outcome:
- TDD-implemented changes scoped strictly to dev-environment paths.
- `python scripts/qs/quality_gate.py` passes (dev-only fast path:
  modified test files only; ruff/mypy/coverage skipped).
- Auto-committed, pushed, PR opened.
- Next-phase command printed (`/review-task`).

User request:
$ARGUMENTS
