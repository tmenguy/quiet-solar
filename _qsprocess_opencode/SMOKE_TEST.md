# OpenCode Workflow — Smoke-Test Checklist

End-to-end verification runbook for the **per-task agent** OpenCode
workflow. Run after the user has assigned real model IDs to the static
`qs-setup-task` agent and to the templates under
`_qsprocess_opencode/agent_templates/`.

Each row describes one verification step, its expected outcome, and what
to inspect if it fails.

---

## 0. Static sanity (no OpenCode required)

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 0.1 | Python scripts parse | `python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('scripts/qs_opencode').glob('*.py')]"` | No output, exit 0 |
| 0.2 | `render_agent.py` dry-render into a tempdir | `TMP=$(mktemp -d); mkdir -p $TMP/.opencode/agent; python scripts/qs_opencode/render_agent.py --phase create-plan --work-dir $TMP --issue 999 --title "smoke test" --story-file _qsprocess_opencode/stories/QS-999.story.md` | Valid JSON with `agent_file`, `agent_name`; file exists and has no `{{...}}` leftovers |
| 0.3 | `render_agent.py` refuses overwrite without flag | Re-run 0.2 | Exits non-zero with "already exists" message |
| 0.4 | `render_agent.py` works with `--overwrite` | Re-run 0.2 with `--overwrite` | Valid JSON, file rewritten |
| 0.5 | `render_agent.py` covers all 9 phases | Loop phases: create-plan, implement-task, review-task, review-blind-hunter, review-edge-case-hunter, review-acceptance-auditor, review-coderabbit, finish-task, release | 9 agent files rendered, all free of `{{...}}` leftovers |
| 0.6 | `cleanup_agents.py --dry-run` lists files | `python scripts/qs_opencode/cleanup_agents.py --work-dir $TMP --issue 999 --dry-run` | JSON with `removed` listing all 9 rendered files, nothing actually deleted |
| 0.7 | `cleanup_agents.py` deletes files | Run without `--dry-run` | 9 files removed; re-run returns empty `removed` list |
| 0.8 | `next_step.py` JSON shape (create-plan → implement-task) | `python scripts/qs_opencode/next_step.py --phase create-plan --work-dir /tmp/x --title "QS_42 test" --issue 42 --story-file _qsprocess_opencode/stories/QS-42.story.md` | JSON with `next_agent` = `qs-implement-task-QS-42`, `render_commands` (list of 1), `spawn_prompt`, `instructions_for_current_agent` |
| 0.9 | `next_step.py` fan-out (implement-task → review-task) | Same for `--phase implement-task --pr 5 ...` | JSON with 5 `render_commands` (review-task + 4 sub-roles) |
| 0.10 | `next_step.py` marks release optional | `--phase finish-task ...` | JSON with `"optional": true` |
| 0.11 | `next_step.py` terminal on release | `--phase release ...` | JSON with `next_agent: null` |
| 0.12 | `launch_opencode.py` JSON shape | `python scripts/qs_opencode/launch_opencode.py --work-dir /tmp/x --title "QS_42 test" --issue 42 --preload-command "Activate qs-create-plan-QS-42 and begin."` | Valid JSON with `tool`, `same_context`, `new_context`, optional `pycharm_context` |
| 0.13 | Static agent/command files have frontmatter | Visual: `.opencode/agent/qs-setup-task.md` and `.opencode/command/setup-task.md` start/end with `---` | Pass |
| 0.14 | Only one static agent, one command | `ls .opencode/agent/qs-*.md` ; `ls .opencode/command/*.md` | Exactly `qs-setup-task.md` and `setup-task.md` |
| 0.15 | No model placeholder left | `grep -rl "TODO/confirm-per-agent" .opencode/agent/qs-setup-task.md _qsprocess_opencode/agent_templates/*.md.tmpl \| wc -l` | `0` |
| 0.16 | No accidental edits to hands-off areas | `git status _qsprocess/ scripts/qs/ docs/development-workflow-guide.md .claude/ CLAUDE.md .cursor/ .cursorrules` | Nothing modified |

## 1. OpenCode discovery

Open the repo in OpenCode on main.

| # | Check | How | Expected |
|---|-------|-----|----------|
| 1.1 | AGENTS.md loaded | Start session, ask: "What rules are you following?" | Agent mentions `CLAUDE.md`, `_qsprocess/rules/project-rules.md`, `_qsprocess_opencode/README.md` |
| 1.2 | Slash commands listed | Type `/` in the CLI | **Only `/setup-task`** appears (no `/create-plan`, `/implement-task`, `/review-task`, `/finish-task`, `/release`) |
| 1.3 | Subagents registered | Ask: "List your qs-* subagents" | Exactly `qs-setup-task` — no per-task agents yet (they're rendered per-task into worktrees) |

## 2. Phase 1 — `/setup-task`

| # | Check | How | Expected |
|---|-------|-----|----------|
| 2.1 | Issue + branch + worktree | `/setup-task 42` (throwaway issue in test repo) | Issue fetched/created, branch `QS_42`, worktree under `../quiet-solar-worktrees/QS_42/` |
| 2.2 | Create-plan agent rendered | `ls <worktree>/.opencode/agent/` | Contains `qs-create-plan-QS-42.md`, no `{{...}}` leftovers |
| 2.3 | Rendered agent has issue context | `grep -E "QS-42\|#42" <worktree>/.opencode/agent/qs-create-plan-QS-42.md` | Multiple matches |
| 2.4 | Launcher emitted | Agent's final message | Terminal command (and PyCharm command if detected) printed |
| 2.5 | Launcher preloads agent activation | Inspect the `--preload-command` | Instructs new session to activate `qs-create-plan-QS-42` by name |
| 2.6 | No Task spawn to create-plan | Session log | Agent did NOT call the Task tool — handoff is an explicit new-session launch |
| 2.7 | Main checkout untouched | `git -C <main repo root> status` | Clean, still on main |

## 3. Phase 1 → 2 handoff

| # | Check | How | Expected |
|---|-------|-----|----------|
| 3.1 | New OpenCode opens on worktree | Run the printed launcher | OpenCode CLI starts with cwd = worktree |
| 3.2 | Rendered agent discoverable | Ask: "List your qs-* subagents" | `qs-create-plan-QS-42` present |
| 3.3 | User activates rendered agent | Say "Activate qs-create-plan-QS-42 and begin." | Agent starts its phase protocol |

## 4. Phase 2 — qs-create-plan-QS-42

| # | Check | How | Expected |
|---|-------|-----|----------|
| 4.1 | Story artifact written | After agent completes | File at the path baked into the agent's prompt, committed on `QS_42` |
| 4.2 | Commit required explicit auth | Transcript | Agent presented diff and waited for "commit" |
| 4.3 | Push required explicit auth | Transcript | Agent presented push plan and waited for "push" |
| 4.4 | next_step.py called | Transcript | `scripts/qs_opencode/next_step.py --phase create-plan ...` invoked |
| 4.5 | render_agent.py called for implement-task | Transcript | `render_agent.py --phase implement-task ...` invoked; file exists |
| 4.6 | Task spawn to qs-implement-task-QS-42 | Session log | Task tool invoked with `subagent_type="qs-implement-task-QS-42"` and `spawn_prompt` from next_step.py |
| 4.7 | Fallback if dynamic spawn unsupported | If Task rejected | Agent asked user to activate `qs-implement-task-QS-42` manually |

## 5. Phase 3 — qs-implement-task-QS-42

| # | Check | How | Expected |
|---|-------|-----|----------|
| 5.1 | TDD order | Commits / transcript | Failing tests first, then implementation |
| 5.2 | Quality gate green | `python scripts/qs/quality_gate.py` run by agent | Exit 0; pytest 100% cov, ruff clean, mypy clean, translations ok |
| 5.3 | PR opened | `gh pr list --head QS_42` | PR exists with quality checklist filled in |
| 5.4 | 5-way render for review | Transcript | `render_agent.py` invoked 5 times: review-task, review-blind-hunter, review-edge-case-hunter, review-acceptance-auditor, review-coderabbit |
| 5.5 | Review agent files exist | `ls <worktree>/.opencode/agent/` | 5 new `qs-review-*-QS-42.md` files |
| 5.6 | Task spawn to qs-review-task-QS-42 | Session log | Orchestrator invoked with spawn_prompt |

## 6. Phase 4 — qs-review-task-QS-42

| # | Check | How | Expected |
|---|-------|-----|----------|
| 6.1 | Parallel reviewer spawns | Session log | 4 concurrent Task calls to `qs-review-blind-hunter-QS-42`, `qs-review-edge-case-hunter-QS-42`, `qs-review-acceptance-auditor-QS-42`, `qs-review-coderabbit-QS-42` |
| 6.2 | Reviewers are hidden | `/` menu | Sub-roles not exposed as slash commands (none exist anyway); if OpenCode UI shows subagents, the 4 sub-roles are marked hidden |
| 6.3 | Consolidated findings | Orchestrator's final message | Single triaged list: must-fix / should-fix / nice-to-have / invalid |
| 6.4 | Interactive triage | Transcript | For each must-fix: fix / defer / reject decision taken with user |
| 6.5 | Handoff to finish-task | Session log | `render_agent.py --phase finish-task ...` then Task spawn to `qs-finish-task-QS-42` |

## 7. Phase 5 — qs-finish-task-QS-42

| # | Check | How | Expected |
|---|-------|-----|----------|
| 7.1 | Final gate re-run | Transcript | `quality_gate.py` run once more on PR head, green |
| 7.2 | Merge authorized | Transcript | Agent waited for explicit "merge" before `gh pr merge` |
| 7.3 | cleanup_agents.py called | Transcript | `python scripts/qs_opencode/cleanup_agents.py --work-dir <w> --issue 42` invoked |
| 7.4 | Per-task agent files removed | `ls <worktree>/.opencode/agent/` before worktree deletion | No `qs-*-QS-42.md` files remain |
| 7.5 | Branch + worktree cleanup | `git branch -a \| grep QS_42` ; `git worktree list` | Gone |
| 7.6 | Epics index updated | Diff on `_bmad-output/planning-artifacts/epics.md` (or equivalent) | Story marked done |
| 7.7 | Release prompted (not auto) | Session log | If release warranted, user asked; `qs-release-QS-42` only rendered/spawned on explicit "release" |

## 8. Phase 6 — qs-release-QS-42 (optional)

| # | Check | How | Expected |
|---|-------|-----|----------|
| 8.1 | Version bump proposed | Transcript | Semver bump type confirmed with user |
| 8.2 | Tag authorized | Transcript | Waited for "release" / "tag" before `git tag` / `git push --tags` |
| 8.3 | GitHub Release created | `gh release list` | Release present with auto-generated notes |

## 9. Failure-mode spot checks

| # | Check | How | Expected |
|---|-------|-----|----------|
| 9.1 | Permission-denied on edits in review phase | Ask `qs-review-*-QS-42` agent to edit a file | Refused / permission prompt |
| 9.2 | Permission-denied on push in setup | Ask `qs-setup-task` to `git push --force` | Falls through to `"*": ask` |
| 9.3 | Hands-off areas respected | Ask any rendered agent to edit `scripts/qs/utils.py` | Refuses, points to `scripts/qs_opencode/` |
| 9.4 | Render-agent refuses overwrite mid-pipeline | If a phase re-renders an existing agent file | Non-zero exit; agent reports and aborts (never silently clobbers) |
| 9.5 | Recovery from killed session mid-phase | Kill OpenCode during implement-task, restart on the worktree | Per-task agents still on disk; user can re-activate `qs-implement-task-QS-42` by name |

## 10. Cross-check against Claude/Cursor

| # | Check | Expected |
|---|-------|----------|
| 10.1 | `git diff` on hands-off paths after full run | Empty (no changes to `_qsprocess/**`, `scripts/qs/**`, `.claude/**`, `CLAUDE.md`, `.cursor/**`, `.cursorrules`, `docs/development-workflow-guide.md`) |
| 10.2 | Claude Code still works | Open same repo in Claude Code, run `/setup-task` | Works identically to before OpenCode integration |

---

## What to do when something fails

- **Template placeholder error** (`Template references undefined variables...`):
  add the missing key to `render_agent.py`'s context dict OR pass it via
  `--extra KEY=VALUE`. Never silently default missing context.
- **Launcher preload doesn't work** (step 3.2/3.3): `opencode [project]
  --agent <name> --prompt <text>` is the confirmed invocation (see
  `opencode --help`). If a future OpenCode version changes flag names,
  update `opencode_launch_command()` in `scripts/qs_opencode/utils.py`.
  The launcher also prints a banner with the intended agent + prompt so
  the user can activate manually as a fallback.
- **Task spawn fails on dynamic agent** (steps 4.6, 5.6, 6.5, 7.7):
  this is the expected outcome if OpenCode does not hot-reload
  `.opencode/agent/` files mid-session. `next_step.py` emits a
  `launcher_command` in its JSON payload for every transition; the
  finishing agent presents that command to the user, who runs it to
  start a fresh OpenCode session with the newly-rendered agent
  pre-activated via `--agent` + `--prompt`. Same worktree, fresh
  process = guaranteed visibility.
- **Quality gate fails**: never override. Fix in the current phase and
  re-run before proceeding.
- **Permission prompt fatigue**: tighten the `permission.bash` allowlist
  in the offending template's frontmatter — never broaden to `"*": allow`.
- **Model too weak for a phase**: edit the `model:` line in that single
  template (affects future tasks) and/or the already-rendered agent file
  for the in-flight task.
- **Per-task agent files linger after finish-task**: run
  `python scripts/qs_opencode/cleanup_agents.py --work-dir <w> --issue <N>`
  manually against any stale worktree.
