# OpenCode Workflow Integration — Plan

Status: architecture rebuilt around per-task agents; ready for final smoke
test and commit.

Constraint: **do not modify any file** under `_qsprocess/**`,
`scripts/qs/**`, `docs/development-workflow-guide.md`, `.claude/**`,
`CLAUDE.md`, `.cursor/**`, or `.cursorrules`. The existing Claude / Cursor
setup must remain bit-identical.

## Goal

Adapt the quiet-solar lifecycle (`/setup-task → /create-plan →
/implement-task → /review-task → /finish-task → /release`) to run natively
in OpenCode with:

- **One static agent and one slash command** (`/setup-task` →
  `qs-setup-task`). Everything else is generated per task.
- **Twelve per-task agents** rendered on demand from templates into the new
  worktree's `.opencode/agents/` folder, with issue-specific context baked
  into the system prompt and narrow tool/permission allowlists.

Keep `_qsprocess/` as the Claude/Cursor source of truth and add
`_qsprocess_opencode/` as the OpenCode-flavored sibling (references only,
no duplicated rules).

## Key Decisions (locked)

1. **`/setup-task` is the ONLY slash command and `qs-setup-task` is the
   ONLY static agent.** OpenCode command frontmatter pins a static
   `agent:` name and cannot dispatch to dynamically-named agents, so
   phases 2–6 deliberately have no slash command.
2. **Per-task agent files** named `qs-<phase>-QS-<N>.md` are rendered
   from templates in `_qsprocess_opencode/agent_templates/*.md.tmpl` into
   the worktree's `.opencode/agents/`. Each ships with issue number, title,
   branch, worktree path, and story-file path hard-coded into its prompt.
3. **Template engine**: simple `{{VAR}}` substitution in
   `scripts/qs_opencode/render_agent.py`. No Jinja dependency. Missing or
   leftover placeholders raise.
4. **Handoff model**:
   - Phase 1 → 2: `qs-setup-task` renders `qs-create-plan-QS-<N>.md`,
     then prints a launcher. The **user** starts a fresh OpenCode on the
     worktree and activates the rendered agent by name.
   - Phases 2 → 3 → 4 → 5 (→ 6): each finishing agent calls
     `scripts/qs_opencode/next_step.py`, which emits a handoff JSON
     containing `render_commands`, `spawn_prompt`, and `next_agent`. The
     agent runs the render commands and then Task-spawns the next agent.
   - `implement-task → review-task` renders **five** agents in one
     transition (review orchestrator + four reviewer sub-roles).
   - `finish-task → release` is marked `optional` — release only happens
     with explicit user consent.
5. **Cleanup**: `scripts/qs_opencode/cleanup_agents.py --work-dir <w>
   --issue <N>` removes every `qs-*-QS-<N>.md` from
   `<w>/.opencode/agents/` at finish-task time, before the worktree is
   deleted.
6. **Permissions (moderate lockdown)**: each template declares a narrow
   `permission` block — edit allowlist per phase, known-script bash
   allowlist, `webfetch: deny` unless needed. Destructive ops fall back
   to `"*": ask`.
7. **Models**: agents inherit the project-wide default from `opencode.json`
   (`"model": "github-copilot/claude-opus-4.6"`). Per-agent overrides can
   be added to individual templates when needed.
8. **Renames vs. Claude/Cursor** (intentional, OpenCode-only):
   - `implement-story` → `implement-task`
   - `review-story` → `review-task`
   - `finish-story` → `finish-task`

## File Layout

### Static (checked in, in main checkout)

```
AGENTS.md                                 # OpenCode entry point
opencode.json                             # instructions refs

.opencode/
  agent/
    qs-setup-task.md                      # ONLY static agent
  command/
    setup-task.md                         # ONLY slash command

_qsprocess_opencode/
  PLAN.md                                 # this file
  README.md                               # directory map + conventions
  SMOKE_TEST.md                           # end-to-end verification runbook
  agent_templates/                        # per-task agent templates
    qs-create-plan.md.tmpl
    qs-plan-critic.md.tmpl
    qs-plan-concrete-planner.md.tmpl
    qs-plan-dev-proxy.md.tmpl
    qs-plan-scope-guardian.md.tmpl
    qs-implement-task.md.tmpl
    qs-review-task.md.tmpl
    qs-review-blind-hunter.md.tmpl
    qs-review-edge-case-hunter.md.tmpl
    qs-review-acceptance-auditor.md.tmpl
    qs-review-coderabbit.md.tmpl
    qs-finish-task.md.tmpl
    qs-release.md.tmpl

scripts/qs_opencode/
  __init__.py
  utils.py                                # self-contained helpers
  render_agent.py                         # template → per-task agent file
  cleanup_agents.py                       # remove qs-*-QS-<N>.md at finish
  next_step.py                            # phase → render + Task-spawn payload
  launch_opencode.py                      # Phase 1 launcher payload

docs/
  opencode-workflow-guide.md              # sibling of development-workflow-guide.md
```

### Per-task (rendered into each worktree)

```
<worktree>/.opencode/agents/
  qs-create-plan-QS-<N>.md                # rendered by qs-setup-task
  qs-plan-critic-QS-<N>.md               # rendered by qs-create-plan (Phase 1)
  qs-plan-concrete-planner-QS-<N>.md     # rendered by qs-create-plan (Phase 1)
  qs-plan-dev-proxy-QS-<N>.md            # rendered by qs-create-plan (Phase 1)
  qs-plan-scope-guardian-QS-<N>.md       # rendered by qs-create-plan (Phase 1)
  qs-implement-task-QS-<N>.md             # rendered by qs-create-plan
  qs-review-task-QS-<N>.md                # rendered by qs-implement-task
  qs-review-blind-hunter-QS-<N>.md        # rendered by qs-implement-task
  qs-review-edge-case-hunter-QS-<N>.md    # rendered by qs-implement-task
  qs-review-acceptance-auditor-QS-<N>.md  # rendered by qs-implement-task
  qs-review-coderabbit-QS-<N>.md          # rendered by qs-implement-task
  qs-finish-task-QS-<N>.md                # rendered by qs-review-task
  qs-release-QS-<N>.md                    # rendered by qs-finish-task (optional)
```

## Pipeline Flow

```
[Home OpenCode, main worktree]
 user: /setup-task 3.2
   └─ qs-setup-task (static):
       1. Follow _qsprocess/skills/setup-task.md (issue, branch, worktree)
       2. render_agent.py --phase create-plan ...
          → writes <worktree>/.opencode/agents/qs-create-plan-QS-<N>.md
       3. launch_opencode.py --preload-command "Activate qs-create-plan-QS-<N>..."
          → prints launcher (terminal + optional PyCharm)

 user: runs the printed launcher

[New OpenCode, QS_<N> worktree]
 user: "Activate qs-create-plan-QS-<N> and begin."
   └─ qs-create-plan-QS-<N>:
       - Author + commit + push story artifact (with user authorization)
       - next_step.py --phase create-plan ...
       - render qs-implement-task-QS-<N>
       - Task(qs-implement-task-QS-<N>, spawn_prompt)

   └─ qs-implement-task-QS-<N>:
       - TDD loop, quality gate, commit, open PR (with user authorization)
       - next_step.py --phase implement-task ...
       - render 5 agents: qs-review-task + 4 reviewer sub-roles
       - Task(qs-review-task-QS-<N>, spawn_prompt)

   └─ qs-review-task-QS-<N> (orchestrator):
       ├─ Task(qs-review-blind-hunter-QS-<N>)        ┐
       ├─ Task(qs-review-edge-case-hunter-QS-<N>)    │ parallel
       ├─ Task(qs-review-acceptance-auditor-QS-<N>)  │
       ├─ Task(qs-review-coderabbit-QS-<N>)          ┘
       - Consolidate + triage, interactive fix/defer/reject
       - next_step.py --phase review-task ...
       - render qs-finish-task-QS-<N>
       - Task(qs-finish-task-QS-<N>, spawn_prompt)

   └─ qs-finish-task-QS-<N>:
       - Final quality gate, merge PR (with user authorization)
       - cleanup_agents.py --work-dir <w> --issue <N>
       - Worktree cleanup, epics update
       - Optionally: render qs-release-QS-<N> and Task-spawn it
```

## Template Placeholders

Standard placeholders available in every template (populated by
`render_agent.py` from CLI args):

- `PHASE`, `ISSUE`, `ISSUE_NUMBER`, `BRANCH`, `TITLE`, `WORK_DIR`,
  `STORY_FILE`, `PR_NUMBER`, `AGENT_NAME`.

Additional variables may be passed with `--extra KEY=VALUE` (must be
`UPPER_SNAKE_CASE`). The renderer raises on any undefined or leftover
placeholder, so templates are self-validating.

## Per-agent Permission Sketches

| Agent | edit | bash allowlist | webfetch |
|---|---|---|---|
| qs-setup-task | deny | `gh *`, `git *`, `python scripts/qs/*`, `python scripts/qs_opencode/*`, `scripts/worktree-setup.sh` | deny |
| qs-create-plan-QS-<N> | narrow (story file path only) | `git *`, `gh issue view *`, `python scripts/qs_opencode/*` | ask |
| qs-implement-task-QS-<N> | allow | `git *`, `pytest *`, `ruff *`, `mypy *`, `python scripts/qs/quality_gate.py`, `python scripts/qs_opencode/*`, `gh pr create *` | ask |
| qs-review-task-QS-<N> | deny | `gh pr *`, `git diff *`, `git log *`, `python scripts/qs_opencode/*` | ask |
| qs-review-blind-hunter-QS-<N> | deny | `git diff *` only | deny |
| qs-review-edge-case-hunter-QS-<N> | deny | `git diff *`, `git log *`, `grep *` | deny |
| qs-review-acceptance-auditor-QS-<N> | deny | `git diff *`; reads story file | deny |
| qs-review-coderabbit-QS-<N> | deny | `gh api *`, `gh pr view *` | deny |
| qs-finish-task-QS-<N> | narrow (`_bmad-output/planning-artifacts/epics.md`) | `gh pr merge *`, `git *`, `scripts/worktree-cleanup.sh`, `python scripts/qs/quality_gate.py`, `python scripts/qs_opencode/cleanup_agents.py *` | deny |
| qs-release-QS-<N> | deny | `git tag *`, `git push --tags`, `python scripts/qs/release.py`, `gh release *` | deny |

Starting sketches; expect tuning after first pipeline run.

## CodeRabbit Integration

The existing CodeRabbit flow (auto-review on PR push) is untouched.
`qs-review-coderabbit-QS-<N>`:

1. Fetches CodeRabbit comments via `gh api repos/.../pulls/<N>/comments`
   filtered by author.
2. Normalizes them into the same findings shape the other reviewers use.
3. Returns structured findings to `qs-review-task-QS-<N>` → same
   interactive fix/discuss/reject loop.

## Implementation Status

- [x] Delete obsolete static agents and slash commands for phases 2-6.
- [x] `scripts/qs_opencode/render_agent.py` — template renderer.
- [x] `scripts/qs_opencode/cleanup_agents.py` — per-task agent cleanup.
- [x] `scripts/qs_opencode/next_step.py` — rewritten for new handoff model.
- [x] `.opencode/agents/qs-setup-task.md` — rewritten to render
      create-plan agent before launcher.
- [x] `.opencode/commands/setup-task.md` — updated.
- [x] Thirteen templates under `_qsprocess_opencode/agent_templates/` (including 4 plan reviewer sub-agents).
- [x] `docs/opencode-workflow-guide.md` — rewritten for new architecture.
- [x] `AGENTS.md` — rewritten.
- [x] `_qsprocess_opencode/README.md` — rewritten.
- [x] `_qsprocess_opencode/SMOKE_TEST.md` — updated for new architecture.
- [x] `_qsprocess_opencode/PLAN.md` — this file, updated.
- [ ] Final verification pass (step 11).
- [ ] User-authorized commit (step 12).

## Open TODOs for Iteration

- **Model IDs** — agents inherit the project default
  (`github-copilot/claude-opus-4.6` in `opencode.json`). Add per-agent
  `model:` overrides to templates if a phase benefits from a different model.
- **OpenCode CLI preload flags** — confirmed via `opencode --help`:
  `opencode [project] --agent <name> --prompt <text>`. The launcher uses
  both flags and also prints a banner with the intended agent + prompt
  as a visible fallback.
- **Task tool + dynamic agents** — whether OpenCode hot-reloads
  newly-rendered `.opencode/agents/` files mid-session is version-
  dependent and unverified empirically. The pipeline handles both
  cases: `next_step.py` emits a `spawn_prompt` for Task() AND a
  `launcher_command` fallback. Finishing agents try Task-spawn first;
  on failure they present the launcher command, which starts a fresh
  OpenCode session where the rendered agent is guaranteed to load.
- **Permission allowlists** — starting sketches only; expect tuning
  after first full pipeline run.

## Entry Point for the Next Context

To continue implementing or verifying this plan in a fresh context, say:

> "Read `_qsprocess_opencode/PLAN.md` and run the smoke-test checks in
> `_qsprocess_opencode/SMOKE_TEST.md`."
