---
name: qs-implement-task
description: >-
  Phase 3 of the QS pipeline. TDD implementation of the story under
  custom_components/quiet_solar/, must pass the full quality gate,
  opens a PR.
model: inherit
readonly: false
is_background: false
---

# qs-implement-task — TDD implementation (production code scope)

You implement the story under `custom_components/quiet_solar/` and
`tests/`, run the full quality gate, and open a PR.

## Discover the task context

```bash
python scripts/qs/context.py
```

Capture `issue`, `title`, `branch`, `story_file`. Read
`docs/workflow/project-rules.md` and
`docs/workflow/project-context.md` if you haven't this session.

## Phase protocol

### 1. Load context

- Read `{{story_file}}` completely.
- Confirm branch state.

### 2. TDD implementation

Red → green → refactor:
1. Write failing tests under `tests/` that encode every acceptance
   criterion.
2. Implement the minimum code under `custom_components/quiet_solar/` to
   make them pass.
3. Refactor while keeping tests green.

Edit scope: `custom_components/quiet_solar/**`, `tests/**`, plus the
story file. All other paths are out of scope.

### 3. Implementation summary

Present a summary (files, design decisions, risks). Ask "Ready to run
the quality gate?". Wait.

### 4. Quality gate (non-negotiable)

```bash
python scripts/qs/quality_gate.py
```

Must exit 0. Fix autonomously on failure. Escalate after 2–3 attempts.

### 5. Commit, push, open PR (automatic)

Once the gate is green, proceed **without asking**:

```bash
git add custom_components/quiet_solar/ tests/ docs/stories/
git commit -m "QS-{{issue}}: {{short summary}}"
git push origin {{branch}}

python scripts/qs/create_pr.py \
    --title "QS-{{issue}}: {{title}}" \
    --summary "{{1-3 bullet summary}}" \
    --issue {{issue}}
```

### 6. Tell the user the next command

```text
✅ Implementation complete — quality gate passed.
✅ Committed and pushed.
✅ PR #{{pr_number}} opened.

Next: type in this session.
  → /qs-review-task
```

## Hard rules

- No code without a failing test first.
- No commit without a green quality gate.
- After a green gate, commit + push + PR are automatic — no prompts.
- Coverage below 100% is a hard block.
- Do NOT edit `_qsprocess_opencode/**` (except the story file),
  `.opencode/**`, `scripts/qs_opencode/**`, `.claude/agents/**`,
  `.cursor/agents/**`.
