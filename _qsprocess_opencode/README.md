# `_qsprocess_opencode/` — OpenCode-flavored workflow sibling

This directory is the OpenCode-native sibling of `_qsprocess/`. It exists so the
existing Claude Code / Cursor workflow in `_qsprocess/` and `.claude/` stays
bit-identical while OpenCode gets first-class subagent support for the same
six-phase pipeline — using a **per-task agent** architecture instead of a
fixed set of static subagents.

> **Hard constraint**: nothing under `_qsprocess/`, `scripts/qs/`,
> `docs/development-workflow-guide.md`, `.claude/`, `CLAUDE.md`, `.cursor/`,
> or `.cursorrules` is ever modified. The authoritative rules still live in
> `_qsprocess/rules/` and `CLAUDE.md`; this tree pulls them in by reference.

## What lives here

```
_qsprocess_opencode/
├── PLAN.md                               # implementation plan (source of truth)
├── README.md                             # this file
├── SMOKE_TEST.md                         # end-to-end verification runbook
├── agent_templates/                      # per-task agent templates ({{VAR}} syntax)
│   ├── qs-create-plan.md.tmpl
│   ├── qs-implement-task.md.tmpl
│   ├── qs-review-task.md.tmpl
│   ├── qs-review-blind-hunter.md.tmpl
│   ├── qs-review-edge-case-hunter.md.tmpl
│   ├── qs-review-acceptance-auditor.md.tmpl
│   ├── qs-review-coderabbit.md.tmpl
│   ├── qs-finish-task.md.tmpl
```

Templates are rendered by `scripts/qs_opencode/render_agent.py` into each
worktree's `.opencode/agents/` folder as `qs-<phase>-QS-<N>.md`. There is
**no shared per-task agent location** — every task's agents live only in
its own worktree and are cleaned up at finish-task time.

## How this relates to `_qsprocess/`

| Concern | `_qsprocess/` | `_qsprocess_opencode/` |
| --- | --- | --- |
| Tools served | Claude Code, Cursor | OpenCode |
| Phase materialization | static skill files | **per-task agent files rendered from templates** |
| Handoff between phases | terminal launcher or instruction text | Phase 1: launcher for new OpenCode session on worktree. Phases 2–6: Task tool spawn of rendered sibling agent. |
| Source of project rules | `_qsprocess/rules/project-rules.md` (authoritative) | references `_qsprocess/rules/` — no duplication |
| Code-style rule set | `_bmad-output/project-context.md` (authoritative) | references same file |
| Quality gate | `python scripts/qs/quality_gate.py` | same command — no fork |
| Phase scripts | `scripts/qs/` | `scripts/qs_opencode/` (sibling, no import) |

**Renames vs. Claude/Cursor phase names** (intentional, OpenCode-only):

- `implement-story` → `implement-task`
- `review-story` → `review-task`
- `finish-story` → `finish-task`

`/release` is narrowly scoped to tag + GitHub Release and is **decoupled** from
`finish-task` (the PR merge happens in `finish-task`).

## Architecture — two static agents, eight rendered per task

### Static (in main checkout)

| File | Purpose |
| --- | --- |
| `.opencode/agents/qs-setup-task.md` | Task setup — creates issue, branch, worktree. Runs on `main`. |
| `.opencode/agents/qs-release.md` | Release — tags, bumps version, creates GitHub Release. Runs on `main`, independent of any task. |
| `.opencode/commands/setup-task.md` | Slash command delegating to `qs-setup-task`. |
| `.opencode/commands/release.md` | Slash command delegating to `qs-release`. |

### Per-task (rendered into each worktree's `.opencode/agents/`)

| Rendered agent | Rendered by |
| --- | --- |
| `qs-create-plan-QS-<N>` | `qs-setup-task` (before emitting launcher) |
| `qs-implement-task-QS-<N>` | `qs-create-plan-QS-<N>` (at end of phase) |
| `qs-review-task-QS-<N>` | `qs-implement-task-QS-<N>` |
| `qs-review-blind-hunter-QS-<N>` (hidden) | `qs-implement-task-QS-<N>` |
| `qs-review-edge-case-hunter-QS-<N>` (hidden) | `qs-implement-task-QS-<N>` |
| `qs-review-acceptance-auditor-QS-<N>` (hidden) | `qs-implement-task-QS-<N>` |
| `qs-review-coderabbit-QS-<N>` (hidden) | `qs-implement-task-QS-<N>` |
| `qs-finish-task-QS-<N>` | `qs-review-task-QS-<N>` |

`implement-task` renders five review agents in one transition so the
orchestrator can Task-spawn its sub-roles in parallel with no further I/O.

## Template placeholders

Templates use a tiny `{{VAR}}` substitution — no Jinja dependency. The
renderer raises on any undefined or leftover placeholder.

Standard placeholders available in every template:

- `PHASE` — phase slug (e.g. `create-plan`)
- `ISSUE`, `ISSUE_NUMBER` — issue number (both resolve to the same value)
- `BRANCH` — feature branch (defaults to `QS_<issue>`)
- `TITLE` — issue / story title
- `WORK_DIR` — absolute worktree path
- `STORY_FILE` — path to the story artifact (empty string if not yet known)
- `PR_NUMBER` — PR number (empty string if not yet opened)
- `AGENT_NAME` — full `qs-<phase>-QS-<N>` name

Additional custom variables can be passed with `--extra KEY=VALUE` (must be
`UPPER_SNAKE_CASE`).

## Handoff scripts

All phase transitions go through `scripts/qs_opencode/`:

1. **Render the next agent(s)** via `render_agent.py`.
2. **Emit handoff JSON** via `next_step.py` (contains exact
   `render_commands`, `spawn_prompt`, and `next_agent` name).
3. **Spawn the next agent** via the Task tool.
4. At finish-task, **remove all rendered agents** via `cleanup_agents.py`
   before the worktree is deleted.

Phase 1's handoff is the exception: instead of Task-spawn, it prints a
launcher command (via `launch_opencode.py`) for the user to start a fresh
OpenCode session on the new worktree. The `--preload-command` is a
natural-language instruction like "Activate agent
`qs-create-plan-QS-<N>` and run its phase protocol" — there is no
`/create-plan` slash command anymore.

## Open TODOs carried forward

- Exact `opencode` CLI flags to preload a slash command / agent-activation
  instruction on a new session (used by Phase 1 launcher). Resolved:
  `opencode [project] --agent <name> --prompt <text>` (confirmed via
  `opencode --help`).
- Permission allowlists per agent — starting sketches only; expect tuning
  after the first pipeline run.

## Merging back later

When the user is ready to drop the Claude/Cursor integration, they can
promote this tree to `_qsprocess/` and retire the scripts. Until then,
keep both trees in parallel. Any rule changes belong in
`_qsprocess/rules/project-rules.md` — this sibling **only references**.
