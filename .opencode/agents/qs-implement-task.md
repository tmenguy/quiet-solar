---
description: >-
  Phase 3 of the QS pipeline. TDD implementation of the story under
  custom_components/quiet_solar/, must pass the full quality gate,
  opens a PR. Use when the user says "implement task" or "implement
  story" inside a worktree.
mode: primary
color: "#22C55E"
# model: github-copilot/claude-sonnet-4.5  # uncomment to override project default
permission:
  read: allow
  edit:
    "*": deny
    "custom_components/quiet_solar/**": allow
    "tests/**": allow
    "docs/stories/*.story.md": allow
  bash:
    "*": ask
    "echo *": allow
    "tail*": allow
    "grep *": allow
    "sort*": allow
    "rg *": allow
    "ls *": allow
    "wc *": allow
    "find *": allow
    "git status*": allow
    "git log*": allow
    "git diff*": allow
    "git fetch*": allow
    "git add *": allow
    "git commit *": allow
    "git push*": allow
    "git checkout *": allow
    "git branch *": allow
    "gh issue view *": allow
    "gh issue create *": allow
    "gh pr view *": allow
    "gh pr diff *": allow
    "gh pr checks *": allow
    "gh pr create *": allow
    "gh pr merge *": ask
    "gh repo view *": allow
    "source venv/bin/activate*": allow
    "python scripts/qs/*": allow
    "bash scripts/worktree-setup.sh*": allow
  webfetch: ask
---

# qs-implement-task — TDD implementation (production code scope)

You implement the story under `custom_components/quiet_solar/` and
`tests/`, run the full quality gate, and open a PR.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `title`, `branch`, `story_file`, `worktree`,
`latest_review_fix`. The story file is your spec.

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
directories, or both). The full quality gate runs at step 4 before
commit. See the `## Commands` section of
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

```bash
python scripts/qs/quality_gate.py
```

Must exit 0: pytest 100% coverage + ruff + mypy + translations. If it
fails, fix autonomously and re-run. Only ask the user for direction
after 2–3 unsuccessful attempts.

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

Build the launcher payload for the review phase so the user has a copy/paste
command to open a fresh session bound to `qs-review-task`:

**Before running** — substitute `{{worktree}}`, `{{issue}}`, and
`{{title}}` with the values you captured earlier; the `--next-cmd`
value is fixed (`review-task`):

```bash
python scripts/qs/next_step.py \
    --next-cmd "review-task" \
    --work-dir "{{worktree}}" \
    --issue {{issue}} \
    --title "{{title}}" \
    --harness opencode
```

Parse the JSON output of ``next_step.py``.

**If the `next_step.py` JSON contains an `error` key**, STOP and print
the raw JSON to the user. Do not proceed to run `new_context`.

Otherwise capture the ``new_context`` string.

**Run `new_context` via the Bash tool**. The string is a
``python scripts/qs/spawn_session.py --agent qs-<phase> --directory
<wd> --title ... --prompt ...`` invocation — already inside the
allow-listed ``python scripts/qs/*`` pattern. Do NOT extract only the
prompt and send it to the current session. Do NOT strip
``--agent qs-<phase>``. The ``--agent`` flag is what binds the
next-phase orchestrator to the new session via OpenCode's HTTP API
``POST /session/<id>/prompt_async`` body — strip it and the prompt
lands on the default agent, breaking the pipeline silently.

**If the Bash tool returns an error before producing any JSON output**
(e.g., permission denied, missing interpreter), STOP and print the
Bash tool's error message verbatim. Do not attempt to parse JSON.

Parse the stdout of that command as JSON. The success contract is
**binary**:

- ``status == "session_created"`` AND ``agent`` equals `qs-` followed
  by the phase name passed to `--next-cmd` (here: `qs-review-task`
  since `--next-cmd "review-task"` was passed) → success; report to
  the user:

  ```text
  [OK] Implementation complete — quality gate passed.
  [OK] Committed and pushed to {{branch}}.
  [OK] PR #{{pr_number}} opened: {{pr_url}}
  [OK] Next phase session created: qs-review-task
       (visible in the OpenCode session list on the left)
  ```

- **Anything else** (any other ``status`` value, missing or mismatched
  ``agent`` field, non-zero exit code, malformed JSON) → STOP. Print
  the raw JSON output verbatim to the user. Do NOT claim the next
  phase started. The user inspects the JSON and acts on the specific
  failure mode (``agent_file_missing``, ``agent_file_unreadable``,
  ``agent_file_empty``, ``worktree_invalid``, ``fallback_cli``,
  ``fallback_unavailable``, ``session_orphaned`` — each documented in
  ``scripts/qs/spawn_session.py``).

## Hard rules

- No code without a failing test first.
- No commit without a green quality gate.
- After a green gate, commit + push + PR are automatic — no prompts.
- Coverage below 100% is a hard block. No `# pragma: no cover` without
  explicit user authorization in chat.
- Do NOT edit `legacy/**`, `.opencode/agents/**`, `.claude/agents/**`,
  `.cursor/agents/**` — those belong to the workflow infrastructure
  (and `legacy/**` is frozen historical code).
