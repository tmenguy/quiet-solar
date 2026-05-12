---
description: TDD implementation of the story under custom_components/quiet_solar/, pass full quality gate, open PR.
---

Use the **qs-implement-task** subagent to handle this. The subagent
discovers task context from the branch name.

Expected outcome:
- TDD-implemented code under `custom_components/quiet_solar/` and tests.
- `python scripts/qs/quality_gate.py` passes (pytest 100% cov + ruff +
  mypy + translations).
- Auto-committed, pushed, PR opened.
- Next-phase command printed (`/review-task`).

User request:
$ARGUMENTS
