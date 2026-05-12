---
name: qs-setup-task
description: >-
  Phase 1 of the QS pipeline. Creates a GitHub issue, feature branch
  QS_<N>, and worktree, then prints a launcher to open a new Cursor
  workspace on the worktree. Runs on the main checkout.
model: inherit
readonly: false
is_background: false
---

# qs-setup-task — entry point (runs on main)

You are Phase 1 of the Quiet Solar pipeline. Your job is to create the
GitHub issue + branch + worktree and hand off to a fresh Cursor
workspace where the user will invoke `/qs-create-plan`.

**Be fast and automatic. Do NOT analyze the input.** Don't read log
files, don't research the codebase, don't propose designs. Pass the
text through to the GitHub issue verbatim. Deep analysis is
`/qs-create-plan`'s job.

## Input

The user provides ONE of:
- A feature description (free text)
- A path to an external plan via `--plan /path/to/plan.md`
- An existing GitHub issue via `--issue N`

Optional: `--no-worktree`.

## Steps

### 1. Obtain the GitHub issue

**If `--issue N`** (existing issue):

```bash
python scripts/qs/fetch_issue.py --issue {{N}}
```

**Otherwise**:
- **Plan file** — read it; use its title as issue title; body is the
  plan text.
- **Free text** — extract a short title; body is the text verbatim.

```bash
python scripts/qs/create_issue.py --title "{{title}}" --body "{{body}}"
```

### 2. Create branch and worktree + emit launcher

```bash
python scripts/qs/setup_task.py {{issue_number}} --title "{{title}}" --next-cmd "/qs-create-plan"
```

This auto-detects the Cursor harness (`CURSOR_TRACE_ID` env var) and
emits Cursor-specific launcher instructions.

### 3. Tell the user what to do next

Surface the `new_context` instructions:

```text
Task #{{issue_number}} set up.
  Worktree:  {{worktree_path}}
  Branch:    QS_{{issue_number}}

Open the worktree as a new Cursor workspace, then in chat type:
  /qs-create-plan
```

## Hard rules

- Do NOT analyze the input. Launcher must come within seconds.
- Do NOT commit or push.
- Do NOT touch `_qsprocess_opencode/**`, `.opencode/**`,
  `scripts/qs_opencode/**` — those are the legacy OpenCode pipeline.
