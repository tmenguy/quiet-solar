# OpenCode Workflow — Session Log

Resumable context for debugging and iterating on the OpenCode port of the
Quiet Solar development workflow. Last updated at the end of the
bring-up session that produced the initial commit.

---

## Goal (locked)

Port the Claude / Cursor pipeline
(`/setup-task → /create-plan → /implement-task → /review-task → /finish-task → /release`)
to OpenCode **without touching** any of the Claude/Cursor source of
truth. Protected paths (bit-identical):

- `_qsprocess/**`
- `scripts/qs/**`
- `docs/development-workflow-guide.md`
- `.claude/**`, `CLAUDE.md`
- `.cursor/**`, `.cursorrules`

All OpenCode-flavored siblings live under:

- `_qsprocess_opencode/` (templates + docs, this file included)
- `scripts/qs_opencode/` (python helpers)
- `.opencode/` (static entry point only)
- `docs/opencode-workflow-guide.md`
- `AGENTS.md` (OpenCode entry point, equivalent of CLAUDE.md)

---

## Architecture (locked)

### One static agent, N rendered per task

Only `qs-setup-task` is a permanent agent file. Every downstream phase is
a **per-task agent** rendered on demand from a template into the new
worktree's `.opencode/agent/` directory, named `qs-<phase>-QS-<N>.md`
with issue-specific context (issue #, title, branch, worktree, story
file, PR #) baked directly into the system prompt, and with a narrow
`permission` allowlist tuned to that phase.

The 10 rendered agents per task (9 templates, implement renders 5):

| Phase | Agent | Template |
|---|---|---|
| 2 | `qs-create-plan-QS-<N>` | `qs-create-plan.md.tmpl` |
| 3 | `qs-implement-task-QS-<N>` | `qs-implement-task.md.tmpl` |
| 4 | `qs-review-task-QS-<N>` | `qs-review-task.md.tmpl` |
| 4a | `qs-review-blind-hunter-QS-<N>` (hidden) | `qs-review-blind-hunter.md.tmpl` |
| 4b | `qs-review-edge-case-hunter-QS-<N>` (hidden) | `qs-review-edge-case-hunter.md.tmpl` |
| 4c | `qs-review-acceptance-auditor-QS-<N>` (hidden) | `qs-review-acceptance-auditor.md.tmpl` |
| 4d | `qs-review-coderabbit-QS-<N>` (hidden) | `qs-review-coderabbit.md.tmpl` |
| 5 | `qs-finish-task-QS-<N>` | `qs-finish-task.md.tmpl` |
| 6 | `qs-release-QS-<N>` (optional) | `qs-release.md.tmpl` |

### Why no slash commands for phases 2–6

OpenCode command frontmatter pins a static `agent:` name and cannot
dispatch to dynamically-named agents. So there is exactly **one** slash
command (`/setup-task`) and one static agent (`qs-setup-task`).
Everything downstream is file-based agent activation.

### Handoff model

**Phase 1 → 2**: new OpenCode session on the new worktree, launched via
an explicit `sh /tmp/qs_oc_launch_<N>.sh` one-liner that invokes:

```
opencode <worktree> --agent qs-create-plan-QS-<N> --prompt "Begin your phase protocol."
```

(Both `--agent` and `--prompt` are confirmed-working top-level OpenCode
CLI flags. Verified via `opencode --help` in this session.)

**Phases 2 → 3 → 4 → 5 → (6)**: dual-path handoff. Each finishing
agent:

1. Renders the next agent file(s) via `render_agent.py`.
2. Calls `next_step.py` which emits JSON containing both a Task
   `spawn_prompt` and a full `launcher_command` / `launcher_payload`.
3. **Tries** `Task(subagent_type='qs-<next>-QS-<N>', prompt=...)`.
4. **If Task fails** (OpenCode may not hot-reload new `.opencode/agent/`
   files mid-session — this is version-dependent and unverified
   empirically), the agent presents the `launcher_command` to the user
   and stops. The user runs it; a fresh OpenCode starts on the same
   worktree (or main checkout, for release) with the next agent
   pre-activated.

`implement-task → review-task` is special: it renders **five** agents in
one transition (orchestrator + 4 reviewer sub-roles) so the orchestrator
can Task-spawn its reviewers in parallel with no further I/O.

### Quality gate

All code work must pass `python scripts/qs/quality_gate.py` (pytest 100%
coverage + ruff + mypy + translations validation) before commit. The
gate is called by `implement-task` and `finish-task`.

---

## Files shipped

### Static OpenCode entry point

- `.opencode/agent/qs-setup-task.md` — the only permanent subagent file.
  Renders `qs-create-plan-QS-<N>.md` into the worktree, then emits the
  Phase 1 → 2 launcher.
- `.opencode/command/setup-task.md` — the only slash command (invokes
  `qs-setup-task`).
- `opencode.json` — project-level OpenCode config.

### Python helpers (`scripts/qs_opencode/`)

- `utils.py` — self-contained helpers (PyCharm detection, git worktree
  helpers, launcher builders). `opencode_launch_command(work_dir, issue,
  title, *, agent=..., preload_command=...)` is the canonical launcher
  builder used by both setup-task and every phase transition.
- `launch_opencode.py` — CLI wrapper around `build_launcher_payload`
  used by setup-task. Accepts `--agent` and `--preload-command`.
- `render_agent.py` — renders one template into
  `<work_dir>/.opencode/agent/qs-<phase>-QS-<N>.md`. `{{VAR}}`
  substitution. Refuses overwrite without `--overwrite`.
- `next_step.py` — emits the handoff JSON at end-of-phase. Contains the
  `PHASE_TRANSITIONS` map (create-plan → implement-task, etc.) including
  the 5-render fan-out for implement → review, and the `optional: true`
  flag on finish → release. **Every transition's JSON now includes both
  `spawn_prompt` (for Task) and `launcher_command` / `launcher_payload`
  (for the hot-reload-fails fallback).**
- `cleanup_agents.py` — removes all `qs-*-QS-<N>.md` from a worktree's
  `.opencode/agent/`. Called by `finish-task` before worktree removal.
- `__init__.py` — empty.

### Templates (`_qsprocess_opencode/agent_templates/`)

Nine `*.md.tmpl` files, all with:

- `mode: subagent`
- `model: TODO/confirm-per-agent` (user will tune per-agent post-commit)
- `permission` blocks scoped narrowly (e.g., create-plan can only edit
  the story file; implement-task edits only `custom_components/quiet_solar/**`
  and `tests/**`; review agents are edit-deny)
- YAML frontmatter validated via `yaml.safe_load`
- Reference `_qsprocess/skills/<phase>.md` as authoritative protocol
- Reviewer sub-roles have `hidden: true`

### Documentation

- `AGENTS.md` — OpenCode equivalent of `CLAUDE.md`; the OpenCode entry
  point. Describes the one-static-agent architecture.
- `docs/opencode-workflow-guide.md` — full walkthrough.
- `_qsprocess_opencode/README.md` — directory layout and conventions.
- `_qsprocess_opencode/PLAN.md` — implementation plan + open TODOs.
- `_qsprocess_opencode/SMOKE_TEST.md` — verification runbook.
- `_qsprocess_opencode/SESSION_LOG.md` — this file.

---

## Key technical decisions

### 1. Agent file naming includes the issue number

`qs-<phase>-QS-<N>.md` so multiple tasks can coexist. `cleanup_agents.py`
removes only files matching the current issue's pattern.

### 2. Protected-paths discipline

Every template's "Hard rules" section forbids edits to protected paths.
Every template's `permission.edit` allowlist is scoped to the minimum
surface for that phase. Defense in depth: prompt + permission.

### 3. Launcher uses `--agent` + `--prompt`

Confirmed via `opencode --help`:
```
opencode [project] [options]
  --agent   agent to use
  --prompt  prompt to use
```

Generated launcher scripts look like:

```sh
echo '── Activating agent: qs-create-plan-QS-42 ──' && \
echo '── Initial prompt (paste manually if preload fails) ──' && \
echo 'Begin your phase protocol.' && \
printf '\033]0;%s\007' 'QS_42: ...' && \
opencode /path/to/worktree --agent qs-create-plan-QS-42 --prompt 'Begin your phase protocol.'
```

Banner lines print the agent + prompt so the user can activate manually
if flag syntax ever changes.

### 4. Dual-path phase transitions (Task + launcher fallback)

Added this round. Critical insight: if OpenCode does not hot-reload
`.opencode/agent/` mid-session, mid-session Task-spawn of a
newly-rendered agent will fail with "unknown agent". Rather than pick
one path, every transition emits BOTH:

- `spawn_prompt` → for `Task(subagent_type=..., prompt=...)`
- `launcher_command` → `sh /tmp/qs_oc_launch_<N>.sh` that runs
  `opencode <worktree> --agent <next> --prompt "Begin your phase protocol."`

Finishing agent tries Task first; on failure presents the launcher.
Works for any OpenCode version, gracefully.

### 5. Models left as placeholders

Every rendered agent ships with `model: TODO/confirm-per-agent`. User
will tune per-agent after first commit based on cost/quality profile of
each phase (e.g., reviewer sub-roles might run on cheaper models).

---

## Verification status

All completed and passing:

- [x] All 5 Python scripts parse (AST check).
- [x] All 9 templates render without leftover `{{...}}`.
- [x] All 9 rendered YAML frontmatters valid (`yaml.safe_load`).
- [x] All frontmatters include: `description`, `mode: subagent`,
  `model: TODO/confirm-per-agent`, `permission`. Reviewer sub-roles
  additionally have `hidden: true`.
- [x] Templates reference the right scripts in the right counts
  (next_step.py, render_agent.py, cleanup_agents.py, quality_gate.py).
- [x] `next_step.py` handoff JSON verified for every transition,
  including the 5-render fan-out at implement→review and the
  `optional: true` flag on finish→release.
- [x] Every transition's JSON now includes both `spawn_prompt` and
  `launcher_command` / `launcher_payload`.
- [x] `render_agent.py` refuses overwrite without `--overwrite`.
- [x] `cleanup_agents.py` dry-run and real-delete work.
- [x] End-to-end smoke (render → next_step → launcher → cleanup) passes.
- [x] Hands-off git status clean (zero changes under protected paths).

---

## Known-open items / risks

### 1. Task-tool dynamic-agent behavior (empirically unverified)

Whether OpenCode hot-reloads `.opencode/agent/` files mid-session
(required for in-session Task-spawn of rendered agents to succeed) is
**version-dependent and not yet empirically tested**. The dual-path
design makes this non-blocking — worst case is 4 OpenCode restarts per
task, each a single paste — but confirming behavior on your installed
version would let you prefer one path.

**Test recipe** (when you want to verify):
1. Run `/setup-task` to create a worktree + Phase 1 handoff.
2. Run the printed launcher. Fresh OpenCode opens in Phase 2.
3. In Phase 2, complete the story and run the render + next_step + Task
   spawn sequence. Observe whether Task() for `qs-implement-task-QS-<N>`
   succeeds or fails with "unknown agent".
4. If it fails, the launcher fallback kicks in automatically. Either way
   the pipeline completes.

### 2. Per-agent model selection

All 10 agents currently say `model: TODO/confirm-per-agent`. Pick once
you know your cost ceiling. Suggestions to consider:
- create-plan: medium-high (story quality matters)
- implement-task: highest (TDD + quality gate)
- review-task orchestrator: medium
- reviewer sub-roles: low-medium (they're narrow, fast)
- finish-task: medium (merge decisions)
- release: medium

### 3. Permission allowlists

Starting-sketch only. Expect tuning after the first real task run —
watch for `ask`-prompt fatigue on commands that should be `allow`, and
for over-broad `allow` rules that should be narrowed.

### 4. Phase 2→3 `--overwrite` edge case

If create-plan re-runs the render command (the handoff JSON re-emits
it), `render_agent.py` will refuse to overwrite the existing
`qs-implement-task-QS-<N>.md`. Template notes this is harmless (the
file is already there from step 4). If it ever becomes a real issue,
add `--overwrite` to the handoff re-emission OR skip it when the file
exists.

### 5. finish-task → release worktree-gone issue

By step 6 of finish-task, `git worktree remove` has already run, so
`$(pwd)` inside the (deleted) worktree is invalid. Template tells the
agent to resolve `--work-dir` to "whichever checkout is now current"
which is the main checkout. Verified this works in template text, but
runtime behavior depends on the user's shell cd-ing out before the
worktree removal — if it doesn't, the agent will need to `cd` back to
the repo root explicitly. Watch for this on first real release.

### 6. CI hook for `scripts/qs_opencode/*.py`

The Claude/Cursor `scripts/qs/*.py` are covered by the quality gate.
The OpenCode siblings are **not**. If we want coverage, add them to the
gate's pytest/mypy/ruff targets. Deferred for now — they're workflow
glue, not product code.

---

## Resumption checklist

If you come back to this workflow later and need to continue:

1. Read this file.
2. Read `AGENTS.md` for the current-state entry point summary.
3. Read `_qsprocess_opencode/PLAN.md` section "Open TODOs" for any
   items deferred.
4. Read `_qsprocess_opencode/SMOKE_TEST.md` and run it before making
   changes — it catches regressions in render / handoff / launcher in
   ~2 minutes.
5. If changing the handoff contract, update **both**:
   - `scripts/qs_opencode/next_step.py` (JSON shape)
   - All 4 phase templates (create-plan, implement-task, review-task,
     finish-task) that consume the JSON
6. Never edit protected paths. `git status --short -- _qsprocess/
   scripts/qs/ docs/development-workflow-guide.md .claude/ CLAUDE.md
   .cursor/ .cursorrules` should always be empty.

---

## First-commit contents

This session produced the first commit, which adds:

- `.opencode/agent/qs-setup-task.md`
- `.opencode/command/setup-task.md`
- `opencode.json`
- `AGENTS.md`
- `scripts/qs_opencode/` (6 files: __init__, utils, render_agent,
  cleanup_agents, next_step, launch_opencode)
- `_qsprocess_opencode/` (README, PLAN, SMOKE_TEST, SESSION_LOG,
  agent_templates/ with 9 templates)
- `docs/opencode-workflow-guide.md`

Zero changes to protected paths.
