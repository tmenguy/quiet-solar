---
name: qs-implement-task
description: >-
  Phase 3 of the QS pipeline. TDD implementation of the story under
  custom_components/quiet_solar/, runs the impacted quality gate
  (--impacted) before commit/PR (the whole-repo gate is CI's job),
  opens a PR. Use when the user says "implement task" or "implement
  story" inside a worktree.
tools: Bash, Read, Edit, Write, Grep, Glob, Agent, TodoWrite, WebFetch, LSP
---

# qs-implement-task — TDD implementation (production code scope)

You implement the story under `custom_components/quiet_solar/` and
`tests/`, run the **impacted** quality gate
(`python scripts/qs/quality_gate.py --impacted`) before commit/PR, and
open a PR. The whole-repo gate is CI's job — never run the full gate
locally.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `title`, `branch`, `story_file`, `worktree`,
`latest_review_fix`. The story file is your spec (unless a review fix plan is active — see step 1b).

Read [docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
and [docs/workflow/project-context.md](../../docs/workflow/project-context.md)
if you haven't this session.

## Phase protocol

### 1. Load context

- Read `{{story_file}}` completely.
- Confirm branch state: `git status`, `git diff origin/main...HEAD`. You
  should be on `{{branch}}` with the story file committed and no other
  local edits.

### 1b. Check for review fix plan

If `latest_review_fix` from `context.py` is non-empty:

1. Read the fix-plan file at that path.
2. The fix plan is your **primary work list** — each finding marked
   "fix" is a task. The story file (`{{story_file}}`) provides
   background context only.
3. After implementing all findings, proceed to step 3 (implementation
   summary) as usual.

If `latest_review_fix` is empty, skip this step and implement the
story from scratch as normal.

**Mid-session re-entry.** If the user says "review done, implement
it" (or similar) during an existing session, re-run
`python scripts/qs/context.py`, pick up the new `latest_review_fix`,
read it, and begin implementing its findings.

### 2. TDD implementation

Red → green → refactor. For every cycle:

1. Write failing tests under `tests/` for each story acceptance criterion.
2. Implement the minimum code under `custom_components/quiet_solar/`
   to make them pass.
3. Refactor while keeping tests green.

Verify with `python scripts/qs/quality_gate.py --quick <path>` during
the inner loop (the canonical TDD command; accepts files,
directories, or both). The `--impacted` gate runs at step 4 before
commit (changed lines 100%); the whole-repo 100% gate is enforced in
CI on every PR. See the `## Commands` section of
[docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
for the full command grammar and the forbidden-vs-allowed raw-pytest
rule.

Edit scope: `custom_components/quiet_solar/**`, `tests/**`, plus the
story file (for progress notes). Stop and escalate if you need to
edit elsewhere.

### 3. Implementation summary

Before running the quality gate, present a summary:
- Modified / created files (one-line description each).
- Key design decisions.
- Open questions / risks.

Then ask: **"Ready to run the quality gate?"** Wait for confirmation.

### 4. Quality gate (non-negotiable)

**ALWAYS** run the impacted inner-loop gate before commit/PR. Do
**not** run, or substitute, the full gate locally:

```bash
python scripts/qs/quality_gate.py --impacted
```

`--impacted` runs the testmon-selected tests and verifies the lines
**you changed** are 100% covered (exit 0 required). It self-heals a
drifted testmon baseline automatically (purges + rebuilds + re-checks
on a changed-line miss) — no manual file deletion is ever required.
The whole-repo 100% gate (pytest + ruff + mypy + translations) stays
authoritative in **CI** on every PR — that is what guarantees full
coverage, including any coverage lost in *unchanged* code, which is
**CI's exclusive job**. The only local full-gate run is an explicit
user request:

```bash
python scripts/qs/quality_gate.py        # full gate, on EXPLICIT request only
```

If `--impacted` fails, fix the **code/tests** — never switch to the
full gate to diagnose. `--impacted` self-heals; a manual
`--seed-testmon` + re-run is an escalation-only last resort for the
residual case (e.g. two consecutive killed runs), after which escalate
to the user. The full gate is not an inner-loop debugging tool. Only
ask the user for direction after 2–3 unsuccessful attempts.

**Doc-maintenance pre-commit sub-step.** After staging your
intended changes (`git add` first so the diff is populated), run

```bash
python scripts/qs/check_doc_drift.py
```

against the staged diff. If exit 1, either update the listed
`docs/agents/` docs and re-stage, or include a justification
paragraph in the PR body under a `## Doc maintenance` heading
explaining why the docs are unaffected. See
[docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
"Doc maintenance".

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
    --title "{{title}}" \
    --harness claude-code
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
- Do NOT edit `legacy/**`, `.opencode/agents/**`, `.claude/agents/**`,
  `.cursor/agents/**` — those belong to the workflow infrastructure
  (and `legacy/**` is frozen historical code).
