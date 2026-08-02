---
name: qs-implement-setup-task
description: >-
  Phase 3 variant for dev-environment changes only (scripts/, .claude/,
  .cursor/, .opencode/, legacy/, docs/, .github/, top-level
  config). Same TDD flow as qs-implement-task but narrower edit scope;
  the pre-commit gate is --impacted (coverage-vacuous on dev-only
  trees, and a no-op entirely when the change set contains no .py file).
  Use when /create-plan selected implement-setup-task as the next phase.
tools: Bash, Read, Edit, Write, Grep, Glob, Agent, TodoWrite, WebFetch, LSP
---

# qs-implement-setup-task — TDD implementation (dev-env scope)

Narrower-scoped variant of `qs-implement-task`. Edits only dev-environment
paths. The pre-commit gate is `--impacted` (coverage-vacuous on
dev-only trees). Read step 4 before trusting a green: when the change
set contains a `.py` file the tooling's own testmon-selected tests run,
but a change set with no `.py` file at all is checked by nothing.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `title`, `branch`, `story_file`, `worktree`,
`latest_review_fix`.

Read [docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
if you haven't this session.

## Phase protocol

### 1. Load context

- Read `{{story_file}}` completely.
- Confirm branch state.

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

### 2. TDD implementation (dev-env)

Red → green → refactor, scoped to dev-environment paths:

- `scripts/qs/**`, top-level `scripts/*.sh`
- `.claude/**`, `.cursor/**`, `.opencode/**`
- `legacy/**` — frozen historical code (`git mv` operations INTO this
  directory are permitted when the story requires it; in-place edits
  are forbidden)
- `docs/**`
- `.github/**`
- Top-level config: `pyproject.toml`, `requirements*.txt`, `CLAUDE.md`,
  `AGENTS.md`, `.cursorrules`, `.gitignore`, `setup.cfg`

If you need to edit `custom_components/quiet_solar/` or `tests/` (other
than dev tooling tests), STOP — this should have been routed to
`/implement-task`.

### 3. Implementation summary

Present a summary, ask "Ready to run the quality gate?". Wait.

### 4. Quality gate (impacted inner loop)

**ALWAYS** run the impacted gate before commit/PR. Do **not** run, or
substitute, the full gate locally:

```bash
python scripts/qs/quality_gate.py --impacted
```

When the change set contains a `.py` file it runs the testmon-selected
tests and verifies any changed lines under
`custom_components/quiet_solar/` are 100% covered, self-healing a
drifted testmon baseline automatically (no manual file deletion ever).
Dev-only changes (`scripts/`, `docs/`, agent files) carry no delta to
`custom_components/quiet_solar/` coverage specifically, so that side of
`--impacted` is a fast no-op.

`.py` presence decides whether the gate *runs* anything — it does not by
itself tell you the run verified anything:

- **Change set contains a `.py` file.** The pass runs, and the
  gate/tooling's own correctness is guarded to the extent testmon
  selects tests for it (plus the whole-repo gate in CI). Two caveats: a
  `.py` edit can still select **zero** tests (testmon fingerprints AST
  blocks, so a comment-only or formatting-only edit selects nothing),
  and the impacted pass **excludes `tests/test_quality_gate.py`** by
  path — so a change to `scripts/qs/quality_gate.py` is *never* covered
  by `--impacted`. For quality-gate changes always also run
  `python scripts/qs/quality_gate.py --quick tests/test_quality_gate.py`.
- **Change set has no `.py` file at all** — docs-only, agent-file-only, a
  commands-only edit. `--impacted` exits early and checks **nothing**
  (QS-290): testmon could never select a test and diff-cover has no
  Python lines to score. That green is honest but empty, so the
  `--quick tests/qs` run below is your ONLY real verification, not a
  supplement.
- **A git probe failed.** The early exit **fails closed** — no exit, the
  full pipeline runs. So "no output about the exit" never means "the
  exit silently fired".

The whole-repo 100% gate is enforced in **CI** on every PR — the
only local full-gate run is an explicit user request:

```bash
python scripts/qs/quality_gate.py        # full gate, on EXPLICIT request only
```

Pass on a green gate; fix on red.

**Changes to `tests/qs`-pinned non-Python files.** For change sets
touching any `tests/qs`-pinned non-Python file (agent files, commands,
workflow docs, `.claude/settings.json`) — even when Python files
changed too — also run

```bash
python scripts/qs/quality_gate.py --quick tests/qs
```

before commit — testmon cannot see non-Python files (its blindness is
per-file, not per-changeset), so `--impacted` alone is blind there;
the first failure would otherwise surface only in CI.

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

```bash
git add scripts/ tests/qs/ tests/test_quality_gate.py .claude/ .cursor/ .opencode/ legacy/ docs/ .github/ CLAUDE.md AGENTS.md .cursorrules opencode.json
git commit -m "QS-{{issue}}: {{short summary}}"
git push origin {{branch}}

python scripts/qs/create_pr.py \
    --title "QS-{{issue}}: {{title}}" \
    --summary "{{1-3 bullet summary}}" \
    --issue {{issue}} \
    --risk LOW
```

(Adjust `git add` to only the paths you actually touched — don't stage
empty directories.)

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

Parse the JSON; capture `new_context` and `phase_agent_pinned` (a `false`
there means the GUI pin was skipped). Then print both blocks:

On `false` the worktree may still carry the **previous** phase's pin, which
`false` cannot distinguish from no pin at all — so drop the GUI block
entirely (pin sentence and bullets) and route the user to the Preferred
`--agent` line, which is correct either way.

```text
✅ Implementation complete — quality gate passed.
✅ Committed and pushed to {{branch}}.
✅ PR #{{pr_number}} opened: {{pr_url}}

Next phase: review-task.

Preferred (opens a fresh interactive `claude --agent qs-review-task` session):
  {{new_context}}

Fallback (stay in this session, degraded one-shot UX via the Agent tool —
kept for any chat without a CLI launcher; the GUI can instead run the phase
agent directly, see `docs/workflow/harness.md`):
  /review-task

[Claude Code GUI] the worktree should now be pinned to `qs-review-task` in
`.claude/settings.local.json` (the payload's `phase_agent_pinned` reports
whether that write happened — it is always skipped on a main checkout).
The GUI displays the active agent nowhere, so if the phase looks wrong,
use the Preferred line above, where `--agent` always wins.
  • **New session** (not a restored one — the GUI reopens the last session)
  • Select directory `{{worktree}}`
  • Name it `QS_{{issue}} review-task`
  • See `docs/workflow/harness.md` →
    "GUI launch surface (Claude Code Desktop)".
```

## Hard rules

- Edit scope is **strictly** dev-environment paths. If you find yourself
  touching `custom_components/quiet_solar/`, that's a scope violation —
  re-route to `/implement-task`.
- Same TDD discipline as `qs-implement-task`: no code without a failing
  test first, no commit without a green gate.
- After green gate, commit + push + PR are automatic — no prompts.
