---
name: qs-implement-task
description: >-
  Phase 3 of the QS pipeline. TDD implementation of the story under
  custom_components/quiet_solar/, must pass the full quality gate,
  opens a PR. Use when the user says "implement task" or "implement
  story" inside a worktree.
tools: Bash, Read, Edit, Write, Grep, Glob, Agent, TodoWrite, WebFetch
---

# qs-implement-task — TDD implementation (production code scope)

You implement the story under `custom_components/quiet_solar/` and
`tests/`, run the full quality gate, and open a PR.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `title`, `branch`, `story_file`, `worktree`. The story
file is your spec.

Read [docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
and [docs/workflow/project-context.md](../../docs/workflow/project-context.md)
if you haven't this session.

## Phase protocol

### 1. Load context

- Read `{{story_file}}` completely.
- Confirm branch state: `git status`, `git diff origin/main...HEAD`. You
  should be on `{{branch}}` with the story file committed and no other
  local edits.

### 2. TDD implementation

Red → green → refactor:
1. Write failing tests under `tests/` that encode every acceptance
   criterion from the story.
2. Implement the minimum code under `custom_components/quiet_solar/` to
   make them pass.
3. Refactor while keeping tests green.

Edit scope: `custom_components/quiet_solar/**`, `tests/**`, plus the
story file (for progress notes). All other paths are out of scope — if
you need to edit elsewhere, stop and escalate.

### 3. Implementation summary

Before running the quality gate, present a summary:
- Modified / created files (one-line description each).
- Key design decisions.
- Open questions / risks.

Then ask: **"Ready to run the quality gate?"** Wait for confirmation.

### 4. Quality gate (non-negotiable)

```bash
python scripts/qs/quality_gate.py
```

Must exit 0: pytest 100% coverage + ruff + mypy + translations. If it
fails, fix autonomously and re-run. Only ask the user for direction
after 2–3 unsuccessful attempts.

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

Capture the PR number. The commit+push+PR sequence is authorized by the
workflow — no user confirmation needed for any of these three.

### 6. Tell the user the next command

Build the launcher payload for `/review-task` so the user has a copy/paste
command to open a fresh interactive `claude --agent qs-review-task` session:

```bash
python scripts/qs/next_step.py \
    --next-cmd "review-task" \
    --work-dir "{{worktree}}" \
    --issue {{issue}} \
    --title "{{title}}"
```

Parse the JSON; capture `new_context`. Then print both blocks:

```text
✅ Implementation complete — quality gate passed.
✅ Committed and pushed to {{branch}}.
✅ PR #{{pr_number}} opened: {{pr_url}}

Next phase: review-task.

Preferred (opens a fresh interactive `claude --agent qs-review-task` session):
  {{new_context}}

Fallback (stay in this session, degraded one-shot UX via the Agent tool —
kept for Claude Desktop and any chat without a CLI launcher):
  /review-task
```

## Hard rules

- No code without a failing test first.
- No commit without a green quality gate.
- After a green gate, commit + push + PR are automatic — no prompts.
- Coverage below 100% is a hard block. No `# pragma: no cover` without
  explicit user authorization in chat.
- Do NOT edit `_qsprocess_opencode/**` (except the story file),
  `.opencode/**`, `scripts/qs_opencode/**`, `.claude/agents/**`,
  `.cursor/agents/**` — those belong to the workflow infrastructure.
