---
description: >-
  Phase 3 variant for dev-environment changes only (scripts/, .claude/,
  .cursor/, .opencode/, legacy/, docs/, .github/, top-level config).
  Same TDD flow as qs-implement-task but narrower edit scope; the
  pre-commit gate is --impacted (coverage-vacuous on dev-only trees).
  Use when create-plan selected implement-setup-task as the next
  phase.
mode: primary
color: "#16A34A"
# model: github-copilot/claude-sonnet-4.5  # uncomment to override project default
permission:
  read: allow
  edit:
    "*": deny
    "scripts/**": allow
    ".claude/**": allow
    ".cursor/**": allow
    ".opencode/**": allow
    "docs/**": allow
    ".github/**": allow
    "tests/**": allow
    "CLAUDE.md": allow
    "AGENTS.md": allow
    ".cursorrules": allow
    ".gitignore": allow
    "pyproject.toml": allow
    "requirements*.txt": allow
    "setup.cfg": allow
    "opencode.json": allow
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
    "git mv *": allow
    "git rm --cached*": allow
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
    "mkdir -p *": allow
  webfetch: ask
---

# qs-implement-setup-task — TDD implementation (dev-env scope)

Narrower-scoped variant of `qs-implement-task`. Edits only dev-environment
paths. The pre-commit gate is `--impacted` (coverage-vacuous on
dev-only trees; the tooling's own testmon-selected tests still run —
see step 4).

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
  `AGENTS.md`, `.cursorrules`, `.gitignore`, `setup.cfg`, `opencode.json`

If you need to edit `custom_components/quiet_solar/` or `tests/` (other
than dev tooling tests), STOP — this should have been routed to
`implement-task`.

### 3. Implementation summary

Present a summary, ask "Ready to run the quality gate?". Wait.

### 4. Quality gate (impacted inner loop)

**ALWAYS** run the impacted gate before commit/PR. Do **not** run, or
substitute, the full gate locally:

```bash
python scripts/qs/quality_gate.py --impacted
```

It runs the testmon-selected tests and verifies any changed lines
under `custom_components/quiet_solar/` are 100% covered, self-healing a
drifted testmon baseline automatically (no manual file deletion ever).
Dev-only changes (`scripts/`, `docs/`, agent files) carry no delta to
`custom_components/quiet_solar/` coverage specifically, so that side of
`--impacted` is a fast no-op — but the gate/tooling's own correctness is
still guarded by its testmon-selected tests under `--impacted` (and the
whole-repo gate in CI). The whole-repo 100% gate is enforced in **CI** on every PR — the
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
git add scripts/ .claude/ .cursor/ .opencode/ legacy/ docs/ .github/ CLAUDE.md AGENTS.md .cursorrules opencode.json
git commit -m "QS-{{issue}}: {{short summary}}"
git push origin {{branch}}

python scripts/qs/create_pr.py \
    --title "QS-{{issue}}: {{title}}" \
    --summary "{{1-3 bullet summary}}" \
    --issue {{issue}} \
    --risk LOW
```

(Adjust `git add` to only the paths you actually touched — don't stage
empty directories. `legacy/` is only staged when you've performed a
`git mv` into it; this phase never edits files already under
`legacy/`.)

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

## Code intelligence (LSP)

OpenCode defaults to pyright (`"lsp": true` in `opencode.json`) but, per
opencode.ai/docs/lsp, exposes LSP to the agent **only as diagnostics** —
no go-to-definition / find-references navigation. Because navigation is
the larger ergonomics win and the diagnostics-only mode is not worth
dedicated wiring, LSP is intentionally **not** enabled for this agent;
use grep/glob for code navigation here. The Claude twin carries an
explicit `LSP` tool (diagnostics + navigation). See
[docs/agents/lsp-evaluation.md](../../docs/agents/lsp-evaluation.md).

## Hard rules

- Edit scope is **strictly** dev-environment paths. If you find yourself
  touching `custom_components/quiet_solar/`, that's a scope violation —
  re-route to `implement-task`.
- Same TDD discipline as `qs-implement-task`: no code without a failing
  test first, no commit without a green gate.
- After green gate, commit + push + PR are automatic — no prompts.
- Do NOT edit `legacy/**` — frozen historical code (`git mv`
  operations INTO this directory are permitted when the story requires
  it).
