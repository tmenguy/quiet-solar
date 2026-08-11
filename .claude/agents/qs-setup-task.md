---
name: qs-setup-task
description: >-
  Phase 1 of the QS pipeline. Creates a GitHub issue, feature branch
  QS_<N>, and worktree, then prints a launcher command to open a new
  session on the worktree. Runs on the main checkout. Use when the user
  says "setup task", "new task", "work on issue #N", or describes a new
  feature to start.
tools: Bash, Read, Glob, Grep, LSP
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

### 1c. Epic lanes stop here (no branch, no worktree)

If the resolved lane is `epic-product` or `epic-factory`, the issue is
the deliverable: an epic has **no implement phase and no branch,
worktree, or PR** (see
[docs/epics/QS-321.md](../../docs/epics/QS-321.md)). Its output is a
rationale document on `main` plus **child issues**.

So for an epic: create/label the issue as above, then **skip step 2
entirely** — do NOT run `setup_task.py`. Report the epic issue number
and tell the user the next move is to decompose it into child tasks
(each child is its own task lane, carrying `Refs #<epic>`; run
setup-task on those). `setup_task.py` refuses an epic outright, so a
slip here fails loudly rather than silently cutting a worktree.

### 2. Set up branch and worktree + emit launcher (tasks only)

**Resolve the next phase from the lane (QS-335).** The next phase is
`create-plan` for every lane **except** `bug-product`, whose
diagnose-first flow replaces it with `diagnose-task`. Set
`NEXT_PHASE = create-plan`, or `NEXT_PHASE = diagnose-task` when the
resolved lane (step 1b) is `bug-product`, and substitute that value in
the `--next-cmd` flag below and in every handoff site in step 3.

One command does it all (**substitute `{{NEXT_PHASE}}`** with the value
you just resolved):

```bash
python scripts/qs/setup_task.py {{issue_number}} --title "{{title}}" --next-cmd "/{{NEXT_PHASE}}" --harness claude-code
```

For `--no-worktree`, pass `--no-worktree`. The script:
- creates branch `QS_{{issue_number}}` from `origin/main`
- creates the worktree at `../<repo>-worktrees/QS_{{issue_number}}/`
- detects the harness and emits the appropriate launcher

Capture `worktree_path`, `branch`, and the launcher payload
(`new_context`, `same_context`, `phase_agent_pinned`, plus optional
`pycharm_context`).

### 3. Tell the user what to do next

The worktree already has `HEAD` on `QS_{{issue_number}}` (verified by
`scripts/worktree-setup.sh`). Surface the launcher (preferred path — an
interactive `claude --agent qs-create-plan` session) and the slash-command
fallback (degraded one-shot UX; the GUI can instead run the phase agent
directly, see [docs/workflow/harness.md](../../docs/workflow/harness.md)).

The launcher attempts to pin `qs-{{NEXT_PHASE}}` into the new worktree's
`.claude/settings.local.json`, so a Claude Code GUI session opened on that
directory boots as the next-phase orchestrator without any `--agent` flag. Check
the payload's `phase_agent_pinned` before promising it: with
`--no-worktree` the work dir **is** the main checkout, which is never
pinned, so that run always reports `false`.

On `false` the worktree may still carry the **previous** phase's pin, which
`false` cannot distinguish from no pin at all — so drop the GUI block
entirely (pin sentence and bullets) and route the user to the Preferred
`--agent` line, which is correct either way.

```text
Task #{{issue_number}} set up.
  Worktree:  {{worktree_path}}
  Branch:    QS_{{issue_number}}  (HEAD already checked out)

Next phase: {{NEXT_PHASE}}.

Preferred (opens a fresh interactive `claude --agent qs-{{NEXT_PHASE}}` session):
  {{new_context}}

Fallback (stay in this session, degraded one-shot UX via the Agent tool —
kept for any chat without a CLI launcher; the GUI can instead run the phase
agent directly, see `docs/workflow/harness.md`):
  /create-plan — or `/diagnose-task` when the lane is `bug-product`

[Claude Code GUI] the worktree should now be pinned to `qs-{{NEXT_PHASE}}` in
`.claude/settings.local.json` (the payload's `phase_agent_pinned` reports
whether that write happened — it is always skipped on a main checkout).
The GUI displays the active agent nowhere, so if the phase looks wrong,
use the Preferred line above, where `--agent` always wins.
  • **New session** (not a restored one — the GUI reopens the last session)
  • Select directory `{{worktree_path}}`
  • Name it `QS_{{issue_number}} {{NEXT_PHASE}}`
  • See `docs/workflow/harness.md` →
    "GUI launch surface (Claude Code Desktop)".
```

If `pycharm_context` is present in the payload, mention it as a bridge
for IDE-embedded terminals (clipboard / AppleScript helpers).

Do NOT attempt to spawn the next agent in this session — the ergonomic
flow is one session per phase.

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
