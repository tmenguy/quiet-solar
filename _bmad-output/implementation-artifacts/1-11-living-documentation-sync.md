# Story 1.11: Living Documentation Sync During Dev Lifecycle

Status: ready-for-dev

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

- [ ] Task 1: Add inline doc-impact detection instructions to `/implement-story` skill (AC: #1)
  - [ ] 1.1 Edit `_qsprocess/skills/implement-story.md` to add an "Inline Doc-Sync" guidance section between Step 2 (BMad dev-story) and Step 3 (Final quality gate)
  - [ ] 1.2 The guidance instructs the agent: when the user provides direction that changes scope, acceptance criteria, technical approach, or tasks — flag it, propose edits to the story artifact, wait for approval, apply if accepted
  - [ ] 1.3 Secondary docs (architecture.md, project-rules.md, project-context.md) are mentioned only when the change is structural, not for every small adjustment

- [ ] Task 2: Add inline doc-impact detection instructions to `/review-story` skill (AC: #2, #3)
  - [ ] 2.1 Edit `_qsprocess/skills/review-story.md` Step 4 (Process each unresolved comment) to add "doc-update" as a fifth action alongside fix/discuss/reject/skip
  - [ ] 2.2 The "doc-update" action: agent proposes specific edits to the story artifact (or other docs), TheDev approves, changes are applied and committed
  - [ ] 2.3 Add guidance for adversarial review findings (from `/bmad-code-review`): when a finding reveals a spec gap, the agent flags the doc impact alongside the code finding

- [ ] Task 3: Add compound doc-sync step to `/implement-story` (AC: #4, #7)
  - [ ] 3.1 Add a new Step 3.5 "Compound Doc-Sync" between current Step 3 (quality gate) and Step 4 (commit)
  - [ ] 3.2 The step reviews all user direction given during the session, all implementation decisions that deviated from the original spec, and any new information discovered
  - [ ] 3.3 Proposes a consolidated list of story artifact updates (AC adjustments, task modifications, dev notes)
  - [ ] 3.4 Optionally proposes architecture.md or project-rules.md updates if implementation revealed structural gaps
  - [ ] 3.5 TheDev approves/rejects each item; approved changes are applied and included in the commit

- [ ] Task 4: Add compound doc-sync step to `/review-story` (AC: #5, #7)
  - [ ] 4.1 Add a new Step 4.5 "Compound Doc-Sync" after Step 4 (process comments) and before Step 5 (output finish command)
  - [ ] 4.2 Same mechanism as Task 3 but also incorporates review findings (GitHub comments, adversarial review, Copilot feedback)
  - [ ] 4.3 TheDev approves/rejects; approved changes committed and pushed

- [ ] Task 5: Add mandatory doc-sync gate to `/finish-story` (AC: #6, #7)
  - [ ] 5.1 Add a new Step 0.5 "Mandatory Doc-Sync Gate" before Step 1 (quality gate) in `_qsprocess/skills/finish-story.md`
  - [ ] 5.2 The gate reads the story artifact and compares acceptance criteria against actual implementation (files changed, tests present, behavior delivered)
  - [ ] 5.3 Flags discrepancies: ACs not covered by tests, tasks marked done but not implemented, scope changes not reflected in the doc
  - [ ] 5.4 TheDev must resolve each discrepancy (update doc or explain why it's fine) before merge proceeds
  - [ ] 5.5 Include `_bmad-output/` in the git add for the commit step so doc changes are captured

- [ ] Task 6: Update project-context.md development lifecycle section (AC: all)
  - [ ] 6.1 Add a brief note in the Development Lifecycle section of `_bmad-output/project-context.md` mentioning that doc-sync is built into implement/review/finish skills
  - [ ] 6.2 Keep it concise — one or two sentences, not a full explanation

- [ ] Task 7: Mirror skill changes to Cursor (AC: all)
  - [ ] 7.1 Ensure any skill file changes in `_qsprocess/skills/` are the canonical source (Claude and Cursor both read from there)
  - [ ] 7.2 If any BMad skill workflows were modified (in `.claude/skills/`), mirror to `.cursor/skills/` for dual-tool compatibility

## Dev Notes

### This is a process/tooling story — no production Python code changes

This story modifies skill definition files in `_qsprocess/skills/` and possibly `_bmad-output/project-context.md`. It does NOT modify any code in `custom_components/quiet_solar/` or `tests/`. The deliverables are updated skill instructions that the agent follows during workflow execution.

### Files to modify

- `_qsprocess/skills/implement-story.md` — add inline detection guidance + compound doc-sync step
- `_qsprocess/skills/review-story.md` — add doc-update action + inline detection + compound doc-sync step
- `_qsprocess/skills/finish-story.md` — add mandatory doc-sync gate at start
- `_bmad-output/project-context.md` — brief lifecycle note

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

### Debug Log References

### Completion Notes List

### File List
