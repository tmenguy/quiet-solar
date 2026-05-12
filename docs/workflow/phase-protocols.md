# Phase protocols

Each phase has a static agent under `.claude/agents/` (and
`.cursor/agents/`). Agents discover task context at runtime via
`python scripts/qs/context.py`. This document captures the contract for
each phase — inputs, outputs, hand-off, hard rules.

Each phase is invoked **as an interactive session** via
`claude --agent qs-<phase>` (the launcher form — preferred). The
slash-command form (`/<phase>`) is kept as a **degraded fallback** for
Claude Desktop and any chat without a CLI launcher. See
[overview.md](overview.md) section "Orchestrators are interactive
sessions; sub-agents are parallel fan-out" for the full rationale —
this document does not duplicate it.

---

## `setup-task` (agent: `qs-setup-task`)

**Runs on**: main checkout.
**Inputs**: free text describing a feature, OR `--issue N` to use an
existing GitHub issue, OR `--plan /path/to/plan.md`.
**Side effects**: creates GitHub issue, creates branch `QS_<N>`, creates
worktree at `<repo>-worktrees/QS_<N>/`.
**Output**: launcher command (`scripts/qs/launchers/<harness>.py`-generated)
the user runs to open a new session on the worktree.
**Next phase**: `claude --agent qs-create-plan` in the worktree
(preferred — fresh interactive session), or `/create-plan` as fallback.

**Hard rules**:
- Do NOT analyze, diagnose, or interpret user input. Pass the text
  through to the GitHub issue verbatim. Deep analysis is `/create-plan`'s
  job.
- Do NOT switch the main checkout's branch.
- Do NOT commit or push — only branch/worktree creation.

---

## `create-plan` (agent: `qs-create-plan`)

**Runs on**: worktree.
**Inputs**: branch `QS_<N>` (issue resolves from there).
**Side effects**: writes story file to
`docs/stories/QS-<N>.story.md`, commits and pushes.
**Output**: story file with acceptance criteria + task breakdown +
adversarial review notes appended.
**Next phase**: `claude --agent qs-implement-task` or
`claude --agent qs-implement-setup-task` in the worktree (preferred —
fresh interactive session), or `/implement-task` /
`/implement-setup-task` as fallback. The agent decides which based on
the file paths in its task breakdown.

**Phase protocol**:
1. Read the issue (`gh issue view`). Read `docs/workflow/project-rules.md`
   and `docs/workflow/project-context.md`. Glob the relevant code areas.
2. Present a scope/risk summary; ask clarifying questions; wait for user
   answers.
3. Draft the plan in memory — acceptance criteria (Given/When/Then) and
   task breakdown.
4. **Adversarial review**: spawn the 4 plan-reviewer subagents in
   parallel (one message, 4 sub-agent invocations). See
   [adversarial-review.md](adversarial-review.md).
5. Synthesize and triage findings interactively with the user. Max 3
   review rounds before forcing finalization.
6. Determine `NEXT_PHASE`: `implement-setup-task` if all touched files
   are in `scripts/`, `.claude/`, `.cursor/`, `.opencode/`,
   `_qsprocess_opencode/`, `docs/`, `.github/`, or top-level config;
   otherwise `implement-task`.
7. Write the story file. Append "Adversarial Review Notes". Commit and
   push.
8. Tell the user the next command — emit the launcher payload (preferred,
   `claude --agent qs-implement-task` or `claude --agent
   qs-implement-setup-task`) plus the slash-command fallback
   (`/implement-task` or `/implement-setup-task`).

**Hard rules**:
- Do not write code in this phase.
- Edit scope is the story file only.
- Never skip the adversarial review.

---

## `implement-task` / `implement-setup-task` (agents: `qs-implement-task`, `qs-implement-setup-task`)

**Runs on**: worktree.
**Inputs**: story file from `create-plan`.
**Side effects**: writes code under `custom_components/quiet_solar/` and
`tests/` (or `scripts/`, `.claude/`, etc. for `implement-setup-task`);
auto-commits, pushes, opens PR after green quality gate.
**Output**: PR opened with quality checklist and risk assessment.
**Next phase**: `claude --agent qs-review-task` in the worktree
(preferred — fresh interactive session), or `/review-task` as fallback.

**Phase protocol**:
1. Read story file. Confirm branch state.
2. TDD: write failing tests → implement minimum code → refactor.
3. Present implementation summary (files modified, design decisions,
   risks). Ask "Ready to run the quality gate?".
4. Run quality gate (`python scripts/qs/quality_gate.py`). Fix
   autonomously on failure; escalate only after 2–3 attempts.
5. Auto-commit, push, open PR. No confirmation prompt — authorized by
   the workflow.

**Edit scope**:
- `qs-implement-task`: `custom_components/quiet_solar/**`, `tests/**`,
  plus the story file (for progress notes).
- `qs-implement-setup-task`: `scripts/qs/**`, `.claude/**`, `.cursor/**`,
  `.opencode/**`, `_qsprocess_opencode/**`, `docs/**`, `.github/**`,
  top-level config files, plus the story file.

**Hard rules**:
- No code without a failing test first.
- No commit without a green quality gate.
- Coverage below 100% is a hard block. No `# pragma: no cover` without
  explicit user authorization.

---

## `review-task` (agent: `qs-review-task`)

**Runs on**: worktree.
**Inputs**: PR number (resolved from branch).
**Side effects**: writes fix-plan files under
`docs/stories/QS-<N>.story_review_fix_#NN.md` (if fixes
are needed).
**Output**: triaged findings; either "ready for finish-task" or a fix
plan + instructions for the user to apply fixes.
**Next phase**: `claude --agent qs-finish-task` in the worktree
(preferred — fresh interactive session), or `/finish-task` as fallback.
When fixes are needed, re-run `claude --agent qs-implement-task` (or
`/implement-task`) and then `claude --agent qs-review-task` (or
`/review-task`).

**Phase protocol**:
1. Fetch PR diff.
2. **Spawn 4 reviewer subagents in parallel** (one message, 4
   invocations):
   - `qs-review-blind-hunter` — diff only, no repo context
   - `qs-review-edge-case-hunter` — diff + repo read-only
   - `qs-review-acceptance-auditor` — diff + story file
   - `qs-review-coderabbit` — wraps CodeRabbit's review
3. Consolidate into must-fix / should-fix / nice-to-have / invalid.
4. **Zero-findings fast path**: if no must-fix or should-fix findings,
   emit the launcher payload (preferred, `claude --agent
   qs-finish-task`) plus the slash-command fallback (`/finish-task`).
5. Interactive triage: present table → ask "fix all / skip all / one by
   one?" → collect decisions → confirm.
6. If fixes needed, write fix-plan file (auto-incremented suffix
   `#01`, `#02`, …) and emit the launcher payload (`claude --agent
   qs-implement-task`) plus the slash-command fallback (`/implement-task`)
   for the user to apply the fix plan.
7. When fixes pushed, the user re-runs `claude --agent qs-review-task`
   (or `/review-task` as fallback) — loop back to step 1 until clean.

**Hard rules**:
- This agent is an orchestrator — do not review code yourself. Always
  delegate to the 4 subagents.
- Edit scope is the fix-plan files only.
- Sub-agent spawning must be parallel, not serial.

---

## `finish-task` (agent: `qs-finish-task`)

**Runs on**: worktree (until cleanup, then transitions out).
**Inputs**: PR number.
**Side effects**: merges PR, deletes branch on origin, removes worktree.
**Output**: confirmation; user is directed to `/release` if appropriate.
**Next phase**: `/release` from the main checkout (independent), or
`claude --agent qs-release` if the user prefers the interactive form.
Note that `qs-finish-task` does **not** emit a launcher payload for
release — it lives on the main checkout, which is a different
workspace, and the user invokes it manually (see QS-175 OUT OF SCOPE).

**Phase protocol**:
1. Show PR summary.
2. Verify CI status: `gh pr checks <PR>`. If pending, advise wait. If
   failed, STOP.
3. Ask user for explicit merge authorization.
4. Merge PR: `gh pr merge --merge`.
5. Delete remote branch (safety check: refuse if branch is
   `main` / `master`).
6. Run `python scripts/qs/cleanup_worktree.py --work-dir <wd>
   --issue <N> --force` (force because code is safely on main).
7. Report. If merged and production code touched, tell the user to run
   `/release` from main.

**Hard rules**:
- No merge without explicit user authorization.
- Never auto-chain to `/release` — it's a separate decision.

---

## `release` (agent: `qs-release`)

**Runs on**: main checkout. Independent of any task.
**Inputs**: none (uses current date for tag derivation).
**Side effects**: bumps `manifest.json` version, commits, tags, pushes
tag.
**Output**: tag `vYYYY.MM.DD.N`; GitHub Actions runs the release
pipeline.
**Next phase**: terminal.

**Phase protocol**:
1. Confirm clean main: `git checkout main`, `git pull`, `git status`.
2. Dry run: `python scripts/qs/release.py --dry-run` → show proposed
   tag → ask for confirmation.
3. Run real: `python scripts/qs/release.py` → bumps manifest, commits,
   pushes, tags.
4. Report tag and version; note that GitHub Actions handles the rest.

**Hard rules**:
- Always dry-run first. Get user confirmation before the real run.
- Refuse to run if `git status` is not clean.
