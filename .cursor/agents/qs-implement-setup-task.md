---
name: qs-implement-setup-task
description: >-
  Phase 3 variant for dev-environment changes only (scripts/, .claude/,
  .cursor/, .opencode/, _qsprocess_opencode/, docs/, .github/, top-level
  config). Same TDD flow with narrower edit scope and fast-path quality
  gate.
model: inherit
readonly: false
is_background: false
---

# qs-implement-setup-task — TDD implementation (dev-env scope)

Narrower-scoped variant of `qs-implement-task`. Edits only
dev-environment paths.

## Discover the task context

```bash
python scripts/qs/context.py
```

Read `docs/workflow/project-rules.md` if not already loaded.

## Phase protocol

### 1. Load context

- Read `{{story_file}}` completely.
- Confirm branch state.

### 2. TDD implementation (dev-env)

Red → green → refactor, scoped to:

- `scripts/qs/**`, `scripts/qs_opencode/**`, top-level `scripts/*.sh`
- `.claude/**`, `.cursor/**`, `.opencode/**`
- `_qsprocess_opencode/**`
- `docs/**`
- `.github/**`
- Top-level config: `pyproject.toml`, `requirements*.txt`, `CLAUDE.md`,
  `AGENTS.md`, `.cursorrules`, `.gitignore`, `setup.cfg`

If you need to edit `custom_components/quiet_solar/` or product `tests/`,
STOP — re-route to `/qs-implement-task`.

### 3. Implementation summary

Present, ask "Ready to run the quality gate?".

### 4. Quality gate (dev-only fast path)

```bash
python scripts/qs/quality_gate.py
```

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

### 6. Tell the user the next command

```text
✅ Implementation complete.
✅ Committed and pushed.
✅ PR #{{pr_number}} opened.

Next: type in this session.
  → /qs-review-task
```

## Hard rules

- Edit scope is **strictly** dev-environment paths.
- Same TDD discipline as qs-implement-task.
