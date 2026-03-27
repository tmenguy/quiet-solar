# Story 1.11: Living Documentation Sync During Dev Lifecycle

Status: dev-complete

issue: 46
branch: "QS_46"

## Story

As TheDev,
I want the dev workflow skills to detect when user direction, review feedback, or adversarial findings impact the story specification (or other project docs), proactively propose doc changes for my approval, and run a formal doc-sync checkpoint at key lifecycle boundaries,
So that the story artifact, architecture, and project rules stay aligned with reality throughout implementation and review — not just at creation time.

## Acceptance Criteria

1. **Given** TheDev gives direction or information during `/implement-story` execution
   **When** the direction impacts the story's acceptance criteria, tasks, scope, or technical approach
   **Then** the agent flags the impact, proposes specific edits to the story artifact (`_bmad-output/implementation-artifacts/*.md`), and waits for TheDev's approval before applying
   **And** if the change also impacts `architecture.md` or `_qsprocess/rules/project-rules.md`, the agent mentions this and proposes those changes separately

2. **Given** TheDev gives direction or information during `/review-story` execution
   **When** the direction impacts the story specification or project docs
   **Then** the same inline detection and proposal mechanism from AC#1 applies

3. **Given** a GitHub PR comment or adversarial review finding surfaces during `/review-story`
   **When** the finding has implications beyond the code fix (e.g., reveals a missing acceptance criterion, an architecture gap, or a rule that should be added)
   **Then** the agent flags the doc impact alongside the code fix proposal
   **And** TheDev can choose "doc-update" as an action (in addition to fix/discuss/reject/skip) to update the relevant document

4. **Given** `/implement-story` reaches the end (before PR creation)
   **When** the compound doc-sync step runs
   **Then** the agent reviews all user direction and implementation decisions made during the session
   **And** proposes a consolidated set of story artifact updates (acceptance criteria adjustments, task changes, dev notes additions)
   **And** optionally proposes architecture or project-rules updates if warranted
   **And** TheDev approves or rejects each proposed change
   **And** approved changes are committed alongside the implementation

5. **Given** `/review-story` reaches the end (after processing all comments)
   **When** the compound doc-sync step runs
   **Then** the same consolidated review from AC#4 applies, also incorporating review findings

6. **Given** `/finish-story` starts
   **When** the mandatory doc-sync gate runs (before merge)
   **Then** the agent reads the story artifact and compares it against the actual implementation (files changed, tests written, behavior delivered)
   **And** flags any discrepancies (e.g., AC not tested, task marked done but not implemented, scope that changed without doc update)
   **And** TheDev resolves each discrepancy before the merge proceeds

7. **Given** doc-sync proposes changes to any document
   **When** TheDev approves the changes
   **Then** the changes are applied to the document in place
   **And** the modified documents are staged for the next commit

## Tasks / Subtasks

- [x] Task 1: Add inline doc-impact detection instructions to `/implement-story` skill (AC: #1)
  - [x] 1.1 Edit `_qsprocess/skills/implement-story.md` to add an "Inline Doc-Sync" guidance section between Step 2 (BMad dev-story) and Step 3 (Final quality gate)
  - [x] 1.2 The guidance instructs the agent: when the user provides direction that changes scope, acceptance criteria, technical approach, or tasks — flag it, propose edits to the story artifact, wait for approval, apply if accepted
  - [x] 1.3 Secondary docs (architecture.md, project-rules.md, project-context.md) are mentioned only when the change is structural, not for every small adjustment

- [x] Task 2: Add inline doc-impact detection instructions to `/review-story` skill (AC: #2, #3)
  - [x] 2.1 Edit `_qsprocess/skills/review-story.md` Step 4 (Process each unresolved comment) to add "doc-update" as a fifth action alongside fix/discuss/reject/skip
  - [x] 2.2 The "doc-update" action: agent proposes specific edits to the story artifact (or other docs), TheDev approves, changes are applied and committed
  - [x] 2.3 Add guidance for adversarial review findings (from `/bmad-code-review`): when a finding reveals a spec gap, the agent flags the doc impact alongside the code finding

- [x] Task 3: Add compound doc-sync step to `/implement-story` (AC: #4, #7)
  - [x] 3.1 Add a new Step 3.5 "Compound Doc-Sync" between current Step 3 (quality gate) and Step 4 (commit)
  - [x] 3.2 The step reviews all user direction given during the session, all implementation decisions that deviated from the original spec, and any new information discovered
  - [x] 3.3 Proposes a consolidated list of story artifact updates (AC adjustments, task modifications, dev notes)
  - [x] 3.4 Optionally proposes architecture.md or project-rules.md updates if implementation revealed structural gaps
  - [x] 3.5 TheDev approves/rejects each item; approved changes are applied and included in the commit

- [x] Task 4: Add compound doc-sync step to `/review-story` (AC: #5, #7)
  - [x] 4.1 Add a new Step 4.5 "Compound Doc-Sync" after Step 4 (process comments) and before Step 5 (output finish command)
  - [x] 4.2 Same mechanism as Task 3 but also incorporates review findings (GitHub comments, adversarial review, Copilot feedback)
  - [x] 4.3 TheDev approves/rejects; approved changes committed and pushed

- [x] Task 5: Add mandatory doc-sync gate to `/finish-story` (AC: #6, #7)
  - [x] 5.1 Add a new Step 0.5 "Mandatory Doc-Sync Gate" before Step 1 (quality gate) in `_qsprocess/skills/finish-story.md`
  - [x] 5.2 The gate uses `scripts/qs/doc_sync.py` to compare story artifact against git diff, plus manual agent review
  - [x] 5.3 Flags discrepancies: ACs not covered by tests, tasks marked done but not implemented, scope changes not reflected in the doc
  - [x] 5.4 TheDev must resolve each discrepancy (update doc or explain why it's fine) before merge proceeds
  - [x] 5.5 Include `_bmad-output/`, `_qsprocess/`, `scripts/` in the git add for the commit step so doc and script changes are captured

- [x] Task 6: Update project-context.md development lifecycle section (AC: all)
  - [x] 6.1 Add a brief note in the Development Lifecycle section of `_bmad-output/project-context.md` mentioning that doc-sync is built into implement/review/finish skills
  - [x] 6.2 Keep it concise — one or two sentences, not a full explanation

- [x] Task 7: Mirror skill changes to Cursor (AC: all)
  - [x] 7.1 Ensure any skill file changes in `_qsprocess/skills/` are the canonical source (Claude and Cursor both read from there) — confirmed, no BMad skills were modified
  - [x] 7.2 If any BMad skill workflows were modified (in `.claude/skills/`), mirror to `.cursor/skills/` for dual-tool compatibility — N/A, no BMad skills modified

## Dev Notes

### This is a process/tooling story — no production Python code changes

This story modifies skill definition files in `_qsprocess/skills/` and possibly `_bmad-output/project-context.md`. It does NOT modify any code in `custom_components/quiet_solar/` or `tests/`. The deliverables are updated skill instructions that the agent follows during workflow execution.

### Files to modify

- `_qsprocess/skills/implement-story.md` — add inline detection guidance + compound doc-sync step
- `_qsprocess/skills/review-story.md` — add doc-update action + inline detection + compound doc-sync step
- `_qsprocess/skills/finish-story.md` — add mandatory doc-sync gate at start
- `_bmad-output/project-context.md` — brief lifecycle note
- `_qsprocess/skills/setup-story.md` — updated to use `claude_launch_command()`
- `_qsprocess/workflows/development-lifecycle.md` — updated with doc-sync references
- `scripts/qs/setup_worktree.py` — updated to use `claude_launch_command()`
- `scripts/qs/utils.py` — added `CLAUDE_LAUNCH_OPTS` constant and `claude_launch_command()` helper
- `scripts/worktree-setup.sh` — fixed per-item config symlink loop, branch reuse support

### Design principles

1. **Story artifact is primary target** — always propose story doc updates first. Architecture and project-rules only when the change is structural.
2. **User always approves** — never auto-apply doc changes. Flag, propose, wait for approval.
3. **Inline + compound = belt and suspenders** — inline catches changes as they happen, compound catches anything missed at the end.
4. **Finish gate is mandatory** — the only gate that blocks. Implement/review compound steps are "always run but advisory".
5. **Lightweight instructions** — these are agent instructions, not code. Keep them concise and actionable. Avoid over-engineering the skill text.

### Previous story patterns (from 1.8)

Story 1.8 (AI-Assisted PR Review) followed the same pattern: process/tooling story that modified skill files and added new agent actions. The "doc-update" action in review-story mirrors how 1.8 added fix/discuss/reject actions. Use the same concise instruction style.

### Project Structure Notes

- Skill files live in `_qsprocess/skills/` (canonical source per project rules)
- Claude reads from `_qsprocess/skills/` directly (per CLAUDE.md routing)
- `.claude/skills/` and `.cursor/skills/` contain BMad skills (bmad-dev-story, bmad-code-review, etc.) — these are different from the QS process skills
- Dual-tool compatibility requirement: changes must work for both Claude Code and Cursor

### References

- [Source: _qsprocess/skills/implement-story.md] — current implement skill (6 steps)
- [Source: _qsprocess/skills/review-story.md] — current review skill (5 steps)
- [Source: _qsprocess/skills/finish-story.md] — current finish skill (5 steps)
- [Source: _bmad-output/implementation-artifacts/1-8-ai-assisted-pr-review.md] — previous process/tooling story pattern
- [Source: _bmad-output/project-context.md#Development Lifecycle] — lifecycle docs to update

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
N/A — process/tooling story, no code debugging needed

### Completion Notes List
- Added inline doc-sync (Step 2.5) to both implement-story and review-story skills
- Added compound doc-sync (Steps 3.5 and 4.5) to implement-story and review-story respectively
- Added mandatory doc-sync gate (Step 0.5) to finish-story skill
- Added "doc-update" as fifth action in review-story comment processing
- Added adversarial review doc-impact guidance to review-story
- Created `scripts/qs/doc_sync.py` script for automated story-vs-implementation comparison (used by finish-story gate)
- Updated git add in implement-story and finish-story to include `_qsprocess/` and `scripts/`
- Updated project-context.md with doc-sync lifecycle note
- User direction: favored scriptable checks over pure instructions where possible

### File List
- `_qsprocess/skills/implement-story.md` — added Step 2.5 (inline doc-sync) + Step 3.5 (compound doc-sync) + expanded git add
- `_qsprocess/skills/review-story.md` — added Step 2.5 (inline doc-sync) + doc-update action + adversarial guidance + Step 4.5 (compound doc-sync)
- `_qsprocess/skills/finish-story.md` — added Step 0.5 (mandatory doc-sync gate with script) + expanded git add + added --story-file input
- `_qsprocess/skills/setup-story.md` — updated to use `claude_launch_command()`
- `_qsprocess/workflows/development-lifecycle.md` — updated with doc-sync references
- `scripts/qs/doc_sync.py` — new script: parses story artifact, compares against git diff, reports discrepancies
- `scripts/qs/setup_worktree.py` — updated to use `claude_launch_command()` from utils
- `scripts/qs/utils.py` — added `CLAUDE_LAUNCH_OPTS`, `claude_launch_command()`, `shlex` import
- `scripts/worktree-setup.sh` — per-item config symlink loop with dotglob, branch reuse with divergence warning
- `_bmad-output/project-context.md` — added doc-sync note to Development Lifecycle section
- `_bmad-output/implementation-artifacts/1-11-living-documentation-sync.md` — updated tasks to done + dev agent record
