# Quiet Solar — OpenCode Rules

This file is the OpenCode entry point. Claude Code reads `CLAUDE.md` directly
and ignores this file, so the Claude/Cursor workflow is untouched.

## Required reading (load with the Read tool on startup)

OpenCode does not auto-parse `@file` references in `AGENTS.md`, so explicitly
load these before doing any substantive work:

- `CLAUDE.md` — base project rules (shared with Claude Code)
- `_qsprocess/rules/project-rules.md` — full project rules
- `_bmad-output/project-context.md` — 42-rule code style set
- `_qsprocess_opencode/README.md` — this workflow's directory layout and conventions

## Workflow — one static agent, nine rendered per task

The OpenCode pipeline has **exactly one static agent and one slash command**:

| Command        | Subagent         | Phase                                                 |
| -------------- | ---------------- | ----------------------------------------------------- |
| `/setup-task`  | `qs-setup-task`  | Create issue + branch + worktree; render first per-task agent; print launcher |

Every downstream phase is a **per-task agent** rendered on demand from a
template into the new worktree's `.opencode/agents/` folder, named
`qs-<phase>-QS-<N>.md`:

- `qs-create-plan-QS-<N>` — write story artifact, commit, push
- `qs-implement-task-QS-<N>` — TDD implementation + quality gate + PR
- `qs-review-task-QS-<N>` — orchestrate parallel role-based review + triage
- `qs-review-blind-hunter-QS-<N>` (hidden sub-role)
- `qs-review-edge-case-hunter-QS-<N>` (hidden sub-role)
- `qs-review-acceptance-auditor-QS-<N>` (hidden sub-role)
- `qs-review-coderabbit-QS-<N>` (hidden sub-role)
- `qs-finish-task-QS-<N>` — final gate + merge PR + worktree cleanup
- `qs-release-QS-<N>` — tag + GitHub Release (optional)

Each rendered agent has the issue-specific context (issue number, title,
branch, worktree, story file) baked into its system prompt and a narrow
tool/permission allowlist tuned to that phase.

## Handoff model

- **Phase 1 → 2**: new OpenCode session on the new worktree — explicit
  launcher printed by `qs-setup-task` (not a Task spawn). Before emitting
  the launcher, `qs-setup-task` renders `qs-create-plan-QS-<N>.md` into
  the worktree.
- **Phases 2 → 3 → 4 → 5 (→ 6)**: sibling-spawn via the Task tool inside
  a single OpenCode instance running in the worktree. Each phase ends by
  calling `scripts/qs_opencode/next_step.py` which emits a handoff JSON
  payload. The payload contains:
  - `render_commands` — one or more `render_agent.py` invocations the
    finishing agent must execute to materialize the next agent file(s).
  - `spawn_prompt` — exact prompt to pass to the next `Task(...)` call.
  - `next_agent` — the `qs-<phase>-QS-<N>` name to spawn.

`implement-task → review-task` is special: it renders **five** agents in
one transition (review orchestrator + four reviewer sub-roles), so the
orchestrator can Task-spawn its reviewers in parallel without further I/O.

There are **no slash commands for phases 2–6** because OpenCode command
frontmatter pins a static `agent:` name and cannot dispatch to
dynamically-named agents. If the Task tool cannot spawn dynamically-named
agents in the running OpenCode version, the finishing agent asks the user
to activate the rendered agent by name.

See `docs/opencode-workflow-guide.md` for the full description.

## Review architecture

`qs-review-task-QS-<N>` is an orchestrator that Task-spawns four hidden
reviewer subagents in parallel:

- `qs-review-blind-hunter-QS-<N>` — diff only, no repo context
- `qs-review-edge-case-hunter-QS-<N>` — diff + repo read-only
- `qs-review-acceptance-auditor-QS-<N>` — diff + story file
- `qs-review-coderabbit-QS-<N>` — wraps the existing CodeRabbit flow

The orchestrator consolidates and triages findings before presenting them
to the user.

## Rendering and cleanup

- `scripts/qs_opencode/render_agent.py --phase <p> --work-dir <w> --issue <N> --title <t> [--story-file ...] [--pr ...] [--extra KEY=VALUE ...]` renders one template into `<w>/.opencode/agents/qs-<p>-QS-<N>.md`. Refuses to overwrite without `--overwrite`. Templates live under `_qsprocess_opencode/agent_templates/*.md.tmpl` and use `{{VAR}}` substitution.
- `scripts/qs_opencode/cleanup_agents.py --work-dir <w> --issue <N>` removes every `qs-*-QS-<N>.md` under `<w>/.opencode/agents/`. Called by `qs-finish-task-QS-<N>` just before the worktree is deleted.

## Quality gate (non-negotiable)

All code work must pass `python scripts/qs/quality_gate.py` before commit.
The gate runs pytest with 100% coverage, ruff, mypy, and translations
validation.

## Commit discipline

Never commit or push without explicit user authorization. Phase prompts
present diffs and wait for "commit" / "push" / "open PR" / "merge" before
performing those git operations.

## Hands-off areas

Do **not** modify these while working in OpenCode — they are the
Claude/Cursor source of truth and must remain bit-identical:

- `_qsprocess/**`
- `scripts/qs/**`
- `docs/development-workflow-guide.md`
- `.claude/**`
- `CLAUDE.md`
- `.cursor/**`, `.cursorrules`

OpenCode-flavored siblings live under `_qsprocess_opencode/`,
`scripts/qs_opencode/`, `.opencode/`, and `docs/opencode-workflow-guide.md`.
