# Story 1.8: AI-Assisted PR Review with Interactive Feedback Loop

Status: review

## Story

As TheDev,
I want PRs to be reviewed by an AI reviewer (or human) on GitHub, with the system pulling review comments back into the local workflow so I can discuss, fix, or reject feedback interactively — a true PR back-and-forth,
So that code review is integrated into the agentic workflow without requiring manual GitHub context-switching.

## Acceptance Criteria

1. **Given** a PR is open on GitHub
   **When** TheDev initiates a review cycle
   **Then** an AI reviewer (configuration TBD) or human posts review comments on the PR

2. **Given** review comments exist on the PR
   **When** TheDev asks to process review feedback
   **Then** the system pulls all unresolved PR comments from GitHub into the local workflow
   **And** each comment is presented with its diff context
   **And** TheDev can choose per comment: fix (implement the suggestion), discuss (reply on the PR), or reject (dismiss with rationale)

3. **Given** TheDev chooses to fix a comment
   **When** the fix is implemented
   **Then** the fix is committed, pushed, and a reply is posted on the PR resolving the comment

4. **Given** TheDev chooses to discuss a comment
   **When** TheDev provides a response
   **Then** the response is posted as a PR reply and the comment remains open for further discussion

5. **Given** TheDev chooses to reject a comment
   **When** TheDev provides a rationale
   **Then** the rationale is posted as a PR reply and the comment is resolved

6. **Given** all comments are processed
   **When** the review cycle completes
   **Then** the system reports a summary of fixes, discussions, and rejections
   **And** quality gates are re-run if any fixes were made

## Tasks / Subtasks

- [x] Task 1: Create `/bmad-pr-review-feedback` skill for interactive PR comment processing (AC: #2, #3, #4, #5, #6)
  - [x] 1.1 Create skill directory at `.claude/skills/bmad-pr-review-feedback/` with `workflow.md`
  - [x] 1.2 Implement Step 1 — detect open PR for current branch (use `gh pr view --json number,url,reviewDecision`)
  - [x] 1.3 Implement Step 2 — pull unresolved review comments via `gh api` (GraphQL for thread resolution status)
  - [x] 1.4 Implement Step 3 — present each comment with file path, line, diff hunk context; prompt TheDev for action (fix / discuss / reject / skip)
  - [x] 1.5 Implement Step 4 — handle "fix" action: implement change, run quality gates, commit, push, post reply resolving comment
  - [x] 1.6 Implement Step 5 — handle "discuss" action: compose reply, post via `gh api`, leave comment open
  - [x] 1.7 Implement Step 6 — handle "reject" action: compose rationale, post via `gh api`, resolve comment thread
  - [x] 1.8 Implement Step 7 — summary report: count of fixes/discussions/rejections, quality gate re-run if any fixes made
- [x] Task 2: Mirror skill to Cursor (AC: all)
  - [x] 2.1 Copy skill to `.cursor/skills/bmad-pr-review-feedback/` and `_bmad/bmm/workflows/4-implementation/bmad-pr-review-feedback/`
- [x] Task 3: Update development lifecycle Phase 3d to reference the new skill (AC: #1, #2)
  - [x] 3.1 Update `_qsprocess/workflows/development-lifecycle.md` Phase 3d to mention `/bmad-pr-review-feedback` as the tool for processing review comments after `/bmad-code-review` generates them
  - [x] 3.2 Update `_qsprocess/rules/project-rules.md` workflow routing table to add "Process PR feedback" intent
- [x] Task 4: Document the AI reviewer strategy decision (AC: #1)
  - [x] 4.1 Document in the skill workflow that the initial review source is the existing `/bmad-code-review` skill — it already runs 3 parallel adversarial review agents locally
  - [x] 4.2 The skill should support BOTH local `/bmad-code-review` findings AND GitHub PR review comments from any source (human, Copilot, future CI-based reviewer)
  - [x] 4.3 Added Phase A option in workflow Step 2 that bridges `/bmad-code-review` output to GitHub PR comments

## Dev Notes

### This is a process/tooling story — no production Python code changes

This story creates a new BMad skill and updates process documentation. It does NOT modify any code in `custom_components/quiet_solar/` or `tests/`. The deliverable is a new agent skill that integrates GitHub PR review comments into the local agentic workflow.

### Existing infrastructure to build on

**Local code review (already exists):**
- `/bmad-code-review` skill at `.claude/skills/bmad-code-review/workflow.md`
- 4-step process: gather context → 3 parallel review subagents → triage → present
- Review subagents: Blind Hunter (adversarial), Edge Case Hunter (boundary analysis), Acceptance Auditor (spec validation)
- Currently presents findings locally but does NOT post them to GitHub

**GitHub CLI (`gh`) is the integration API:**
- Already installed and authenticated (prerequisite in development-lifecycle.md)
- All GitHub operations use `gh` CLI — no custom GitHub Actions or API tokens needed for this story

**Key `gh` commands for PR review interaction:**

```bash
# Get PR for current branch
gh pr view --json number,url,title,reviewDecision,state

# List PR review comments (unresolved)
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments

# List PR reviews
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews

# Post a review comment reply
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies -f body="..."

# Post a new PR review with comments
gh pr review {pr_number} --comment --body "..."

# Resolve a review thread (GraphQL needed)
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "..."}) { thread { isResolved } } }'

# Unresolve a review thread (GraphQL needed)
gh api graphql -f query='mutation { unresolveReviewThread(input: {threadId: "..."}) { thread { isResolved } } }'
```

**Important:** Thread resolution requires the GraphQL API — the REST API does not support resolving/unresolving review threads. The skill must use `gh api graphql` for resolve/unresolve operations.

### Workflow design: two-phase approach

**Phase A — Post review to GitHub (bridge `/bmad-code-review` → GitHub):**
1. Run `/bmad-code-review` locally (existing skill)
2. Take the triaged findings and post them as PR review comments via `gh api`
3. Each finding becomes a review comment anchored to the relevant file/line

**Phase B — Process feedback interactively (the core of this story):**
1. Pull ALL unresolved review comments from the PR (from any source: local review, human, Copilot)
2. Present each with diff context
3. Interactive per-comment loop: fix / discuss / reject
4. Summary + quality gate re-run

This two-phase design means the skill works with ANY review source, not just the local `/bmad-code-review`. A human reviewer's comments are processed identically.

### Quality gates re-run after fixes

If any "fix" actions were taken, the full quality gate suite must pass before the review cycle is considered complete:
```bash
source venv/bin/activate
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing
ruff check custom_components/quiet_solar/
ruff format --check custom_components/quiet_solar/
mypy custom_components/quiet_solar/
```

### Open questions resolved

**Q: Which AI reviewer to use?**
A: Start with the existing `/bmad-code-review` skill as the primary AI reviewer. It already runs 3 parallel adversarial agents. The skill should also support processing comments from any external reviewer (human, Copilot, future GitHub Action). The "which reviewer" question becomes a configuration choice, not a blocker.

**Q: How to configure reviewer selection?**
A: Not needed for v1. The skill processes whatever review comments exist on the PR, regardless of source. Reviewer configuration (auto-requesting Copilot review, triggering a Claude-based GitHub Action) is a future enhancement.

### Project Structure Notes

```
.claude/skills/bmad-pr-review-feedback/     (new — the skill)
  workflow.md                                (main workflow)
  steps/                                     (optional, if steps are complex enough to warrant separate files)

.cursor/skills/bmad-pr-review-feedback/      (new — mirror for Cursor)
  workflow.md

_bmad/bmm/workflows/*/bmad-pr-review-feedback/  (new — mirror for BMad core)
  workflow.md

_qsprocess/workflows/development-lifecycle.md   (modified — Phase 3d update)
_qsprocess/rules/project-rules.md               (modified — routing table update)
```

### What NOT to do

- Do NOT modify any production Python code in `custom_components/quiet_solar/`
- Do NOT modify test code in `tests/`
- Do NOT create a new GitHub Actions workflow — this is a local skill, not CI
- Do NOT require new API tokens or authentication — `gh` CLI is already authenticated
- Do NOT auto-merge or auto-approve — the skill is advisory and interactive
- Do NOT delete or replace the existing `/bmad-code-review` skill — extend the workflow to bridge its output to GitHub

### Previous story intelligence

**Story 1.7 (worktrees):** Established the pattern of process-only stories that create scripts and update `_qsprocess/` docs without touching production code. Used shell scripts in `scripts/` for reusable operations.

**Story 1.4 (PR templates):** Created `.github/PULL_REQUEST_TEMPLATE.md` with quality checklist and risk assessment. Created `.github/CODEOWNERS`. Created issue templates.

**Story 1.1 (agentic workflow):** Established `_qsprocess/` as the canonical location for project process docs shared between Claude and Cursor.

**Phase 3d (code review):** Currently says "run `/bmad-code-review` after PR is created" and recommends using a different LLM. This story extends Phase 3d with the feedback processing loop.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.8] — acceptance criteria and open questions
- [Source: _bmad-output/planning-artifacts/prd.md#NFR23] — minimal manual steps for developer workflows
- [Source: _bmad-output/planning-artifacts/architecture.md#CI/CD Pipeline] — planned auto-review.yml and coverage-report
- [Source: _qsprocess/workflows/development-lifecycle.md#Phase 3d] — current code review phase
- [Source: .claude/skills/bmad-code-review/workflow.md] — existing 4-step review process
- [Source: _bmad-output/implementation-artifacts/1-7-parallel-story-development-worktrees.md] — previous story patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Created `/bmad-pr-review-feedback` skill with 5-step workflow: detect PR, offer Phase A/B choice, pull unresolved threads via GraphQL, interactive per-comment processing (fix/discuss/reject/skip), summary with quality gate re-run
- Used GraphQL API for thread operations (resolution status, posting replies, resolving threads) — REST API lacks thread resolution support
- Mirrored skill to `.cursor/skills/` and `_bmad/bmm/workflows/4-implementation/`
- Updated Phase 3d in development-lifecycle.md with "Processing Review Feedback" subsection
- Added "Process PR feedback" intent to both quick reference table and routing table in project-rules.md
- Skill works with any review source (local /bmad-code-review, human, Copilot, future CI reviewers)
- Code review findings addressed (15 items): fixed wrong GraphQL mutation (P1), added skip_count tracking (P2), removed auto-commit in favor of user confirmation (P3), scoped git add to changed files (P4), simplified Phase A to honest 2-option menu (P5/P6), added error handling on all GraphQL calls (P7), added 3-attempt escape hatch for fix loops (P8), added missing SKILL.md to BMad mirror (P9), documented heredoc approach for shell metacharacters (P10), added active branch check (P11), added null-line and deleted-file handling (P12), added pagination support (D1), added top-level review body fetching (D2), added branch protection resolve warning (D3)

### Change Log

- 2026-03-24: Story 1.8 implemented — PR review feedback skill, lifecycle docs, routing updates
- 2026-03-24: Addressed 15 code review findings (P1-P12, D1-D3) — see completion notes

### File List

New files:
- `.claude/skills/bmad-pr-review-feedback/SKILL.md`
- `.claude/skills/bmad-pr-review-feedback/bmad-skill-manifest.yaml`
- `.claude/skills/bmad-pr-review-feedback/workflow.md`
- `.cursor/skills/bmad-pr-review-feedback/SKILL.md`
- `.cursor/skills/bmad-pr-review-feedback/bmad-skill-manifest.yaml`
- `.cursor/skills/bmad-pr-review-feedback/workflow.md`
- `_bmad/bmm/workflows/4-implementation/bmad-pr-review-feedback/SKILL.md`
- `_bmad/bmm/workflows/4-implementation/bmad-pr-review-feedback/bmad-skill-manifest.yaml`
- `_bmad/bmm/workflows/4-implementation/bmad-pr-review-feedback/workflow.md`

Modified files:
- `_qsprocess/workflows/development-lifecycle.md` (Phase 3d expanded with feedback loop)
- `_qsprocess/rules/project-rules.md` (routing table + quick reference)
- `_bmad-output/implementation-artifacts/1-8-ai-assisted-pr-review.md` (status + tasks + record)
