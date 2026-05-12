---
name: qs-implement-setup-task
description: >-
  Phase 3 variant for dev-environment changes only (scripts/, .claude/,
  .cursor/, .opencode/, _qsprocess_opencode/, docs/, .github/, top-level
  config). Same TDD flow as qs-implement-task but narrower edit scope
  and the fast-path quality gate. Use when /create-plan selected
  implement-setup-task as the next phase.
tools: Bash, Read, Edit, Write, Grep, Glob, Agent, TodoWrite, WebFetch
---

# qs-implement-setup-task — TDD implementation (dev-env scope)

Narrower-scoped variant of `qs-implement-task`. Edits only dev-environment
paths. The quality gate runs in its dev-only fast path
(`quality_gate.py` auto-detects this when production code is untouched).

## Discover the task context first

```bash
python scripts/qs/context.py
```

Capture `issue`, `title`, `branch`, `story_file`, `worktree`.

Read [docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
if you haven't this session.

## Phase protocol

### 1. Load context

- Read `{{story_file}}` completely.
- Confirm branch state.

### 2. TDD implementation (dev-env)

Red → green → refactor, scoped to dev-environment paths:

- `scripts/qs/**`, `scripts/qs_opencode/**`, top-level `scripts/*.sh`
- `.claude/**`, `.cursor/**`, `.opencode/**`
- `_qsprocess_opencode/**` (stories + product docs)
- `docs/**`
- `.github/**`
- Top-level config: `pyproject.toml`, `requirements*.txt`, `CLAUDE.md`,
  `AGENTS.md`, `.cursorrules`, `.gitignore`, `setup.cfg`

If you need to edit `custom_components/quiet_solar/` or `tests/` (other
than dev tooling tests), STOP — this should have been routed to
`/implement-task`.

### 3. Implementation summary

Present a summary, ask "Ready to run the quality gate?". Wait.

### 4. Quality gate (dev-only fast path)

```bash
python scripts/qs/quality_gate.py
```

Smart scope detection skips the full suite when only dev files changed —
runs the modified test files only. To force the full suite, use
`--full`. Pass on a green gate; fix on red.

### 5. Commit, push, open PR (automatic)

```bash
git add scripts/ .claude/ .cursor/ .opencode/ _qsprocess_opencode/ docs/ .github/ CLAUDE.md AGENTS.md .cursorrules
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

```text
✅ Implementation complete — quality gate passed.
✅ Committed and pushed to {{branch}}.
✅ PR #{{pr_number}} opened: {{pr_url}}

Next: type in this session.
  → /review-task
```

## Hard rules

- Edit scope is **strictly** dev-environment paths. If you find yourself
  touching `custom_components/quiet_solar/`, that's a scope violation —
  re-route to `/implement-task`.
- Same TDD discipline as `qs-implement-task`: no code without a failing
  test first, no commit without a green gate.
- After green gate, commit + push + PR are automatic — no prompts.
