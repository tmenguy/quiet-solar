# Quiet Solar OpenCode Workflow Guide

This guide describes the Quiet Solar development workflow **as it runs in
OpenCode** (the legacy per-task-rendered pipeline). The Claude Code /
Cursor reference is [docs/workflow/overview.md](workflow/overview.md) —
the static-agent pipeline. Both pipelines coexist; the OpenCode one is
slated for retirement once the static-agent one is proven.

Both workflows drive the same lifecycle (setup → plan → implement → review
→ finish → release) against the same git worktree layout, the same quality
gate (`python scripts/qs/quality_gate.py`), and the same story artifacts
under `docs/stories/`. The difference is purely in how phases are
materialized and hand off to each other.

## Architecture — per-task agents

The OpenCode flow has **one static agent** and **nine per-task agents**
generated on demand.

- **Static**: `.opencode/agents/qs-setup-task.md` — the only always-present
  subagent, plus its slash command `.opencode/commands/setup-task.md`. Runs
  in the home OpenCode on `main`.
- **Per-task**: every downstream phase is rendered from a template into the
  new worktree's `.opencode/agents/` folder with all issue-specific context
  baked into the system prompt and narrow tool/permission allowlists tuned
  to that phase.

Per-task agent files are named `qs-<phase>-QS-<issue>.md`, e.g.
`qs-create-plan-QS-42.md`, and live only inside the worktree for that task.
They are cleaned up at `/finish-task` time before the worktree is removed.

### Why per-task instead of static

- Each agent ships with the specific issue number, title, branch, worktree
  path, and story-file path hard-coded into its prompt — no runtime
  parameter juggling.
- Tool and bash allowlists can be narrowed to exactly what that phase of
  that task needs.
- Multiple in-flight tasks cannot cross-contaminate: each worktree has its
  own isolated agent definitions.

### Templates

Templates live under `_qsprocess_opencode/agent_templates/`:

```
qs-create-plan.md.tmpl
qs-implement-task.md.tmpl
qs-review-task.md.tmpl
qs-review-blind-hunter.md.tmpl
qs-review-edge-case-hunter.md.tmpl
qs-review-acceptance-auditor.md.tmpl
qs-review-coderabbit.md.tmpl
qs-finish-task.md.tmpl
qs-release.md.tmpl
```

Templates use a deliberately tiny `{{VAR}}` substitution syntax (no Jinja
dependency). The renderer raises on any undefined or leftover placeholder,
so every variable must be provided.

Known placeholders: `PHASE`, `ISSUE`, `ISSUE_NUMBER`, `BRANCH`, `TITLE`,
`WORK_DIR`, `STORY_FILE`, `PR_NUMBER`, `AGENT_NAME`. Additional
key/value pairs can be passed with `--extra KEY=VALUE`.

### Rendering

`scripts/qs_opencode/render_agent.py` renders one phase template into
`<work_dir>/.opencode/agents/qs-<phase>-QS-<issue>.md`. It refuses to
overwrite an existing file unless `--overwrite` is passed.

`scripts/qs_opencode/cleanup_agents.py` removes every
`qs-*-QS-<issue>.md` under `<work_dir>/.opencode/agents/` at finish-task
time. It never touches files outside the given worktree.

## Handoff model

Only `/setup-task` is a slash command. Phases 2–6 have **no slash
commands** because OpenCode command frontmatter pins a static `agent:`
name and cannot dispatch to dynamically-named agents. Instead, each phase
ends by rendering the next phase's agent file(s) and then spawning the
next agent via the Task tool (or instructing the user to activate it by
name if the Task tool can't dispatch dynamically-named agents).

| Transition              | Mechanism                                                               |
| ----------------------- | ----------------------------------------------------------------------- |
| Phase 1 → 2             | **Launcher** — new OpenCode session on the new worktree (not a Task)    |
| Phase 2 → 3, 3 → 4, 4 → 5 | Task tool — sibling-spawn the next `qs-<phase>-QS-<N>` agent in worktree |
| Phase 5 → 6 (optional)  | Task tool — sibling-spawn `qs-release-QS-<N>` **with explicit user consent** |

### Phase 1 → 2 (setup-task → create-plan)

`qs-setup-task` runs in the home OpenCode on `main`:

1. Follow [`docs/workflow/phase-protocols.md` — `setup-task`](workflow/phase-protocols.md#setup-task-agent-qs-setup-task)
   to create/reuse the issue, branch, and worktree.
2. Render `qs-create-plan-QS-<N>.md` into the worktree:

   ```bash
   python scripts/qs_opencode/render_agent.py \
       --phase create-plan \
       --work-dir "<worktree>" \
       --issue <N> \
       --title "<title>" \
       --story-file "<expected_story_path>"
   ```

3. Emit the launcher via `scripts/qs_opencode/launch_opencode.py` whose
   `--preload-command` is a natural-language instruction telling the new
   session to activate `qs-create-plan-QS-<N>`.

The **user** runs the launcher to start a fresh OpenCode cd'd into the
worktree. The new session then activates the rendered agent by name.

### Phases 2 → 3 → 4 → 5 (→ 6): Task-spawn inside the worktree

At the end of each phase, the running agent:

1. Invokes `scripts/qs_opencode/next_step.py --phase <finishing_phase> …`
   to get a handoff JSON payload. The payload contains:
   - `render_commands` — one `render_agent.py` invocation per agent that
     must exist before the Task spawn.
   - `spawn_prompt` — exact prompt to pass to `Task(subagent_type=..., prompt=...)`.
   - `next_agent` — the `qs-<phase>-QS-<N>` name to spawn.
   - `instructions_for_current_agent` — human-readable summary.
   - `optional` — `true` for the finish-task → release transition (release
     only happens with explicit user consent).
2. Executes the `render_commands` sequentially to materialize the next
   agent file(s).
3. Spawns the next agent with the `Task` tool.

`implement-task → review-task` is special: it renders **five** agent
files in one transition (the review orchestrator plus the four reviewer
sub-roles), so `qs-review-task-QS-<N>` can Task-spawn its reviewers in
parallel without any further rendering.

If the Task tool in the running OpenCode version cannot spawn
dynamically-named agents, the agent asks the user to activate the
freshly-rendered agent by name instead.

## Review architecture

`qs-review-task-QS-<N>` is an **orchestrator**. It doesn't review itself;
it spawns four hidden reviewer subagents in parallel via the Task tool,
each with a deliberately different lens:

| Subagent                              | Input                | Purpose                                       |
| ------------------------------------- | -------------------- | --------------------------------------------- |
| `qs-review-blind-hunter-QS-<N>`       | Diff only (no repo)  | Catches problems visible without repo context |
| `qs-review-edge-case-hunter-QS-<N>`   | Diff + repo (RO)     | Branching / boundary conditions exhaustively  |
| `qs-review-acceptance-auditor-QS-<N>` | Diff + story file    | Acceptance-criteria traceability matrix      |
| `qs-review-coderabbit-QS-<N>`         | Wraps CodeRabbit     | Keeps the existing CodeRabbit flow in the loop |

The orchestrator consolidates and triages findings (must-fix, should-fix,
nice-to-have, invalid) before presenting them to the user for interactive
fix / defer / reject decisions.

## Models

All agents inherit the project-wide default model set in `opencode.json`
(`"model": "github-copilot/claude-opus-4.6"`). To override for a specific
phase, add a `model:` key to the template's YAML frontmatter.

Guidance for per-agent overrides (if needed later):

- **Opus-class** recommended for `qs-create-plan` and `qs-implement-task`
  (high-stakes writing / TDD).
- **Small/fast** for `qs-setup-task`, `qs-review-task` (orchestrator),
  `qs-finish-task`, `qs-release`, and the hidden review sub-roles.

## Permissions (moderate lockdown)

Every agent declares an explicit `permission` block in its YAML
frontmatter:

- `edit`: narrow allowlist of paths the phase is expected to touch
  (e.g. create-plan: only `docs/stories/QS-<N>.story.md`;
  review-*: `deny`; setup-task: `deny`).
- `bash`: known-script allowlist (`python scripts/qs/*`,
  `python scripts/qs_opencode/*`, `git *`, `gh *` as each phase needs).
  Destructive / scope-changing operations (`rm *`, `git push --force`,
  `gh pr merge`, `git tag`, release ops) fall back to `ask` via the
  default `"*": ask` rule.
- `read`: broad (all phases need to read the repo).
- `webfetch`: `deny` unless the phase needs it.

## Commit discipline

Never commit, push, open a PR, merge, or tag without explicit user
authorization. Each phase presents the diff or command and waits for the
user's explicit keyword ("commit", "push", "open PR", "merge", "release")
before running the corresponding git/gh command. This mirrors the
Claude/Cursor discipline.

## Directory map

```
AGENTS.md                                 # OpenCode entry point (Claude ignores)
opencode.json                             # instructions list

.opencode/
  agent/
    qs-setup-task.md                      # STATIC — only always-present agent
  command/
    setup-task.md                         # STATIC — only slash command

_qsprocess_opencode/                      # OpenCode-owned templates + history
  README.md
  PLAN.md
  SMOKE_TEST.md
  SESSION_LOG.md
  agent_templates/                        # per-task agent templates
    qs-create-plan.md.tmpl
    qs-implement-task.md.tmpl
    qs-implement-setup-task.md.tmpl
    qs-review-task.md.tmpl
    qs-review-blind-hunter.md.tmpl
    qs-review-edge-case-hunter.md.tmpl
    qs-review-acceptance-auditor.md.tmpl
    qs-review-coderabbit.md.tmpl
    qs-plan-critic.md.tmpl
    qs-plan-concrete-planner.md.tmpl
    qs-plan-dev-proxy.md.tmpl
    qs-plan-scope-guardian.md.tmpl
    qs-finish-task.md.tmpl
  product/product-brief.md                # historical (canonical at docs/product/)
  tests/                                  # pipeline test suite

scripts/qs_opencode/                      # OpenCode-specific machinery
  __init__.py
  utils.py                                # self-contained; no import from scripts/qs
  render_agent.py                         # template → per-task agent file
  cleanup_agents.py                       # remove qs-*-QS-<N>.md at finish-task
  next_step.py                            # phase → render + Task-spawn payload
  launch_opencode.py                      # Phase 1 launcher payload

# Inside each worktree (created per task):
<worktree>/.opencode/agents/
  qs-create-plan-QS-<N>.md                # rendered by setup-task
  qs-implement-task-QS-<N>.md             # rendered by create-plan
  qs-review-task-QS-<N>.md                # rendered by implement-task
  qs-review-blind-hunter-QS-<N>.md        # rendered by implement-task
  qs-review-edge-case-hunter-QS-<N>.md    # rendered by implement-task
  qs-review-acceptance-auditor-QS-<N>.md  # rendered by implement-task
  qs-review-coderabbit-QS-<N>.md          # rendered by implement-task
  qs-finish-task-QS-<N>.md                # rendered by review-task
  qs-release-QS-<N>.md                    # rendered by finish-task (optional)
```

## Quality gate

Unchanged from the Claude/Cursor workflow:

```bash
python scripts/qs/quality_gate.py
```

Runs pytest with 100% coverage, ruff (check + format), mypy, and
translations validation. Must be green before any commit.

## Hands-off areas

OpenCode must not modify any of these (they belong to the new
static-agent pipeline, which is the path forward):

- `.claude/**`, `CLAUDE.md`
- `.cursor/**`, `.cursorrules`
- `scripts/qs/**` (excluding shared utilities both pipelines read)

These are the static-agent pipeline's source of truth.

OpenCode owns: `_qsprocess_opencode/agent_templates/`,
`scripts/qs_opencode/`, `.opencode/`, `AGENTS.md`, `opencode.json`, and
this document.

Shared (read-only from both pipelines): `docs/workflow/`, `docs/stories/`,
`docs/product/`, `scripts/qs/quality_gate.py`, the worktree layout.

## Cross-reference

- Static-agent pipeline reference: [docs/workflow/overview.md](workflow/overview.md)
- Phase protocol source (canonical for both flows): [docs/workflow/phase-protocols.md](workflow/phase-protocols.md)
- OpenCode-specific deltas: embedded directly in the rendered per-task
  agents (see `_qsprocess_opencode/agent_templates/*.md.tmpl`)
- Full implementation plan: `_qsprocess_opencode/PLAN.md`
- Smoke test runbook: `_qsprocess_opencode/SMOKE_TEST.md`
