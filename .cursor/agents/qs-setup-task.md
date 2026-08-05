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
GitHub issue + branch + worktree and hand off to a fresh session on the
worktree where the user will invoke `/create-plan`.

**Be fast and automatic, *except* the single lane/epic question when
the declaration is missing (step 1b). Do NOT analyze the input.** Don't
read log files, don't research the codebase, don't propose designs.
Pass the text through to the GitHub issue verbatim. Deep analysis is
`/create-plan`'s job.

## Input

The user provides ONE of:
- A feature description (free text — may include logs / error traces)
- A path to an external plan via `--plan /path/to/plan.md`
- An existing GitHub issue via `--issue N`

Optional: `--no-worktree` (create branch only, skip worktree).

## Steps

### 1. Obtain the GitHub issue

**If `--issue N`** (existing issue):

```bash
python scripts/qs/fetch_issue.py --issue {{N}}
```

Capture `issue_number`, `title`, `body`, `labels`, plus the lane axes
`kind` / `target` / `scale` / `lane`, `parent_epic`, and
`declaration_complete` from the JSON. Lane handling is step 1b.

**Otherwise** (new issue):
- **Plan file** — read it; use its title as issue title; body is the full
  plan text.
- **Free text** — extract a short title (first sentence or ~80 chars);
  body is the full text verbatim.

First resolve the lane (step 1b), then create the issue with its
labels — every task is born fully declared:

```bash
python scripts/qs/create_issue.py --title "{{title}}" --body "{{body}}" --labels "{{labels}}"
```

e.g. `--labels "kind:bug,target:product,scale:task"`. If a parent epic
was declared, include a `Refs #{{epic}}` line in the issue body.

Capture `issue_number` from the JSON output.

### 1b. Lane declaration (QS-332 — the one permitted question)

Every issue carries exactly one lane: `kind:*` + `target:*` +
`scale:task` for tasks, `target:*` + `scale:epic` (and **no kind**) for
epics. `scale:task` is the implicit default — never ask "task or
epic?" separately — but it is always applied explicitly as a label.

**Existing issue (`--issue N`):** if `declaration_complete` is `true`,
use the declaration as-is and ask nothing (epic linkage included:
whatever the body says, or nothing). If it is `false`, ask the user for
exactly the missing axes in ONE question, and — since a question is
being asked anyway — include the optional "part of an epic? (#N /
none)" iff the body carries no parent-epic declaration. Then backfill
the full label set onto the issue before proceeding:

```bash
gh issue edit {{N}} --add-label "{{missing labels}}"
```

(plus a `Refs #{{epic}}` body line via `gh issue edit {{N}} --body ...`
if an epic was given). The issue on GitHub always ends up fully
declared — `setup_task.py` refuses to proceed otherwise.

**New issue (free text / plan file):** ask ONE question for the lane —
the 6 options: `bug` / `feature` (product), `harness bug` /
`harness feature` (factory), `epic product` / `epic factory` — plus the
optional "part of an epic? (#N / none)". Skip the question **only when
the user's request contains an explicit lane name or explicit axis
values** ("bug", "harness feature", "kind:bug target:factory",
"epic product", ...); anything else — however suggestive the text —
means ask. This is a bright line, not text analysis, so the "do NOT
analyze the input" rule stands.

### 2. Set up branch and worktree + emit launcher

One command does it all:

```bash
python scripts/qs/setup_task.py {{issue_number}} --title "{{title}}" --next-cmd "/create-plan" --harness cursor
```

For `--no-worktree`, pass `--no-worktree`. The script:
- creates branch `QS_{{issue_number}}` from `origin/main`
- creates the worktree at `../<repo>-worktrees/QS_{{issue_number}}/`
- detects the harness and emits the appropriate launcher

Capture `worktree_path`, `branch`, and the launcher payload
(`new_context`, `same_context`, plus optional `pycharm_context`).

### 3. Tell the user what to do next

> Launch surfaces for the Claude harness (including the GUI) are
> documented in
> [docs/workflow/harness.md](../../docs/workflow/harness.md).
> That doc's GUI phase pin is best-effort: the Claude payload
> reports the outcome as `phase_agent_pinned`, and no other harness
> reads it.

The worktree already has `HEAD` on `QS_{{issue_number}}` (verified by
`scripts/worktree-setup.sh`). Surface the launcher for the next phase.

```text
Task #{{issue_number}} set up.
  Worktree:  {{worktree_path}}
  Branch:    QS_{{issue_number}}  (HEAD already checked out)

Next phase: create-plan.
Open the worktree as a new Cursor workspace, select qs-create-plan
from the agent picker, then paste:
  {{new_context}}
```

Do NOT attempt to spawn the next agent in this session — the ergonomic
flow is one session per phase.

## Code intelligence (LSP)

Cursor provides editor-native LSP (2.4+): pyright diagnostics,
go-to-definition, find-references, and hover types are surfaced
in-session by the editor itself, not as a separate agent tool. There is
nothing to enable in this agent file — type errors and navigation are
ambient as you read and edit. The Claude twin wires an explicit `LSP`
tool over the same pyright backend; Cursor's equivalent is implicit, so
no `tools:` change is needed here. See
[docs/agents/lsp-evaluation.md](../../docs/agents/lsp-evaluation.md).

## Hard rules

- Do NOT analyze the input. The launcher must come fast and
  automatically, *except* the single lane/epic question when the
  declaration is missing (step 1b). Only an explicit lane name or
  explicit axis values in the user's request count as an answer —
  never inferred ones.
- Do NOT commit or push — setup-task only creates branches/worktrees.
- Do NOT touch `legacy/**` — that's the retired per-task-rendering
  OpenCode pipeline (frozen historical code).
- If any step fails, abort and report; do not auto-heal.
