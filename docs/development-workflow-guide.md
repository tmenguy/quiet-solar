# Quiet Solar Development Workflow Guide

This document describes the end-to-end development workflow for the quiet-solar project, from idea to release. All workflows are driven by natural language commands through Claude Code or Cursor — no manual GitHub operations needed.

## Quick Reference

| You say | What happens |
|---------|-------------|
| "Fix this bug where..." | Issue → worktree → quick-dev → quality gates → PR |
| "I want to add a feature that..." | create-story → commit → issue → worktree → dev-story → quality gates → PR |
| "Create story 3.2" | create-story → commit story file. Stops here. |
| "Implement story 3.2" | Issue → worktree → dev-story → quality gates → PR |
| "Work on issue #N" | Fetches issue → routes to bug or feature flow |
| "Process PR feedback" | Pull review comments → interactive fix/discuss/reject |
| "Merge PR #N" | Merge commit + delete branch + worktree cleanup |
| "Create a release" | Tag, version bump, GitHub Release |

---

## The Development Lifecycle

### Phase 1: Setup

Every piece of work starts with:

1. **Story creation** (features only): `/bmad-create-story` generates a comprehensive story file with acceptance criteria, tasks, dev notes, and guardrails. The story file is committed to main.

2. **GitHub issue**: Created automatically with a link to the story file.

3. **Worktree**: A git worktree is created at `../quiet-solar-worktrees/QS_<issue_number>/` with shared venv and config via symlinks. This lets you work on multiple stories in parallel. The main worktree stays on main.

### Phase 2: Development

Inside the worktree, the agent follows red-green-refactor:
1. Write failing tests
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Run quality gates (all 4 must pass):

```bash
source venv/bin/activate
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing  # 100% mandatory
ruff check custom_components/quiet_solar/                                     # zero violations
ruff format --check custom_components/quiet_solar/                            # all formatted
mypy custom_components/quiet_solar/                                           # no issues
```

### Phase 3: Completion

After development:

1. **Commit and push** from the worktree
2. **Create PR** with quality checklist and risk assessment
3. **Code review** (see below)
4. **Process review feedback** (see below)
5. **Merge** (user-triggered): merge commit + branch deletion + worktree cleanup + main update

### Phase 4: Release

Tag-based releases: `vYYYY.MM.DD.XX`. The CI pipeline runs all quality gates, validates HACS compatibility, and creates a GitHub Release with auto-generated changelog.

---

## Code Review and PR Feedback

The review workflow has two parts that work together:

### Step 1: Run the code review

After a PR is created, run `/bmad-code-review`. This launches 3 parallel adversarial reviewers:

- **Blind Hunter** — reviews the diff with no project context, finds issues purely from the code
- **Edge Case Hunter** — reviews with project access, finds boundary conditions and unhandled scenarios
- **Acceptance Auditor** — reviews against the story spec, checks every acceptance criterion

The findings are triaged into categories: patch (fixable), defer (pre-existing), intent gap (spec incomplete), bad spec (spec wrong), or rejected as noise.

For best results, use a **different LLM** than the one that implemented the story.

### Step 2: Process review feedback

After review comments exist on the PR (from the code review, a human reviewer, GitHub Copilot, or any other source), run `/bmad-pr-review-feedback` to process them interactively.

The skill:
1. Detects the open PR for your current branch
2. Pulls all unresolved review threads from GitHub via GraphQL
3. Also fetches top-level review bodies and PR conversation comments
4. Presents each thread one at a time with:
   - File path and line number
   - Code context (~5 lines before/after)
   - All comments in the thread with author attribution
5. For each thread, you choose an action:

| Action | What happens |
|--------|-------------|
| **Fix** | Agent implements the suggestion, runs quality gates, stages changes. You confirm the commit. Reply posted + thread resolved. |
| **Discuss** | You compose a reply. Posted to the PR thread, which stays open for further discussion. |
| **Reject** | You provide a rationale. Posted to the PR thread + thread resolved. |
| **Skip** | Deferred to later. No action taken. |

After all threads are processed, you get a summary (counts of each action) and quality gates re-run if any fixes were made.

**Important:** The skill never auto-commits. You always confirm before any commit or push happens.

### Example session

```
you> process PR feedback

PR #36: feat: PR review feedback skill (Story 1.8)
URL: https://github.com/tmenguy/quiet-solar/pull/36
Review decision: CHANGES_REQUESTED

Would you like to:
1. Process existing feedback
2. Run local review first

you> 1

Found 3 unresolved review thread(s) on PR #36.

Thread 1/3 — workflow.md:200
[code context shown]
Review comment by @reviewer: "Wrong mutation name here..."

Actions: [F]ix | [D]iscuss | [R]eject | [S]kip

you> F
[agent implements fix, runs quality gates]
Fix implemented and quality gates pass. Changes staged:
  workflow.md | 2 +-
Commit and push this fix? [Y/N]

you> Y
[committed, pushed, replied on PR, thread resolved]

Thread 2/3 — ...
```

---

## Worktrees

Every development workflow uses git worktrees by default. This means:

- **Main worktree** always stays on `main` — never gets dirty
- **Feature worktrees** live at `../quiet-solar-worktrees/QS_<N>/`
- **Shared resources**: venv, config/, and non-git custom_components are symlinked (not duplicated)
- **Parallel work**: multiple stories can be in-progress simultaneously

Say "no worktree" if you want the old single-directory branch workflow instead.

### Worktree lifecycle

```
Start work    → scripts/worktree-setup.sh <issue_number>
Do work       → all changes happen in the worktree
Merge PR      → gh pr merge <N> --merge --delete-branch
Cleanup       → scripts/worktree-cleanup.sh <issue_number>
Update main   → git checkout main && git pull
```

---

## Quality Gates

All four must pass before any PR can be created or merged:

| Gate | Command | Threshold |
|------|---------|-----------|
| Tests + coverage | `pytest tests/ --cov=... --cov-report=term-missing` | 100% coverage, zero failures |
| Lint | `ruff check custom_components/quiet_solar/` | Zero violations |
| Format | `ruff format --check custom_components/quiet_solar/` | All files formatted |
| Type check | `mypy custom_components/quiet_solar/` | No issues |

---

## Available Skills

| Skill | When to use |
|-------|-------------|
| `/bmad-create-story` | Plan a story with comprehensive context for the dev agent |
| `/bmad-dev-story` | Implement a story following its task list |
| `/bmad-quick-dev-new-preview` | Bug fixes and small changes |
| `/bmad-code-review` | Adversarial 3-layer code review |
| `/bmad-pr-review-feedback` | Process PR review comments interactively |
| `/bmad-sprint-status` | Check sprint progress |
| `/bmad-correct-course` | Manage scope changes mid-sprint |

---

## File Locations

| What | Where |
|------|-------|
| Project rules | `_qsprocess/rules/project-rules.md` |
| Development lifecycle | `_qsprocess/workflows/development-lifecycle.md` |
| Code-level rules (42 rules) | `_bmad-output/project-context.md` |
| Architecture & patterns | `_bmad-output/planning-artifacts/architecture.md` |
| Story files | `_bmad-output/implementation-artifacts/` |
| Failure mode catalog | `docs/failure-mode-catalog.md` |
| Worktree scripts | `scripts/worktree-setup.sh`, `scripts/worktree-cleanup.sh` |
