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

Red → green → refactor. For every cycle:

1. Write failing tests under `tests/` for each story acceptance criterion.
2. Implement the minimum code under `custom_components/quiet_solar/`
   to make them pass.
3. Refactor while keeping tests green.

Verify with `python scripts/qs/quality_gate.py --quick <path>` during
the inner loop (the canonical TDD command; accepts files,
directories, or both). The full quality gate runs at step 4 before
commit. See the `## Commands` section of
[docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
for the full command grammar and the forbidden-vs-allowed raw-pytest
rule.

Edit scope: `custom_components/quiet_solar/**`, `tests/**`, plus the
story file (for progress notes). Stop and escalate if you need to
edit elsewhere.

### 3. Implementation summary

Present a summary (files, design decisions, risks). Ask "Ready to run
the quality gate?". Wait.

### 4. Quality gate (non-negotiable)

```bash
python scripts/qs/quality_gate.py
```

Must exit 0. Fix autonomously on failure. Escalate after 2–3 attempts.

**Doc-maintenance pre-commit sub-step.** Before staging, run

```bash
python scripts/qs/check_doc_drift.py
```

on the staged diff. If exit 1, either update the listed
`docs/agents/` docs and re-stage, or include a justification
paragraph in the PR body under a `## Doc maintenance` heading
explaining why the docs are unaffected. See
`docs/workflow/project-rules.md` "Doc maintenance".

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
- Do NOT edit `legacy/**`, `.opencode/agents/**`, `.claude/agents/**`,
  `.cursor/agents/**` (and `legacy/**` is frozen historical code).
