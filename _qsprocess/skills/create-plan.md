# /create-plan

Create a story implementation artifact through adversarial multi-perspective review. Runs inside the worktree (or on the QS_N branch in --no-worktree mode).

## Input

- `--issue N` (required): GitHub issue number. Branch is `QS_N`.
- `--story-key X` (optional): Epic story key (e.g., "3.2") for looking up `epics.md`.
- `--plan /path` (optional): Path to an external plan `.md` file (e.g., from Cursor). Used as both synthesis input and adversarial comparison.

If neither `--story-key` nor `--plan` is provided, the issue title and body are used as the primary source of requirements.

## Prerequisites

You should be inside the worktree directory (`../quiet-solar-worktrees/QS_N/`) or on branch `QS_N` (if `--no-worktree` was used in `/setup-task`).

If not on `QS_N`, check out the branch:
```bash
git checkout QS_{{issue_number}}
```

Verify: `git branch --show-current` should return `QS_N`.

## Persona Definitions

Four adversarial reviewers are defined in `_qsprocess/plan-reviewers/`:

| Persona | File | Gets | Challenges |
|---------|------|------|------------|
| **The Critic** | `critic.md` | Plan ONLY | Soundness — holes, contradictions, vague scope |
| **The Concrete Planner** | `concrete-planner.md` | Plan + file tree + snippets | Specificity — missing file paths, vague tasks |
| **The Dev Agent Proxy** | `dev-proxy.md` | Plan + project rules | Implementability — rule violations, missing guardrails |
| **The External Challenger** | `external-challenger.md` | Plan + external plan | Alternative comparison (only when `--plan` provided) |

---

## Phase 1: Gather & Analyze

**Goal:** Build a compact Analysis Brief that all later phases consume. This phase does the heavy codebase reading — later phases work from the brief.

### 1.1 Fetch issue details

```bash
python scripts/qs/fetch_issue.py --issue {{N}}
```

Capture `title`, `body`, `labels`, and `story_type` from the JSON output.

### 1.2 Parallel exploration

Launch 2-3 parallel **Explore** subagents:

**Agent 1 — Issue Explorer:**
- Read the issue body thoroughly (this is the user's intent — do NOT paraphrase or skip it)
- Read `_bmad-output/planning-artifacts/epics.md` for story context
- If `--story-key` provided: extract the specific epic/story details
- If `--plan` provided: read and summarize the external plan
- Identify story type and construct the story key (see "Story Key Construction" below)

**Agent 2 — Codebase Explorer:**
- From the issue body and title, identify affected areas of the codebase
- Find relevant source files, understand current implementation
- Map module dependencies and boundaries
- Note code patterns and conventions in affected areas

**Agent 3 — Context Explorer:**
- Read `_bmad-output/project-context.md` — extract rules relevant to this issue's scope
- Read `_bmad-output/planning-artifacts/architecture.md` — extract applicable constraints
- Find recent implementation artifacts in the same epic (if applicable) for continuity
- Check test patterns in affected areas

### 1.3 Compile Analysis Brief

From the three agents' outputs, compile a concise Analysis Brief (hold in memory, do NOT write to disk):

```
## Analysis Brief for #{{N}}: {{title}}

### Issue Summary
- Title: ...
- Body: (full body preserved)
- Labels: ...
- Story type: ...
- Story key: ...

### Affected Codebase Areas
- Files: ...
- Modules: ...
- Dependencies: ...

### Architecture Constraints
- (relevant rules extracted from project-context.md)

### Similar Past Work
- (previous stories in same epic, if any)

### Risk Assessment
- Level: CRITICAL | HIGH | MEDIUM | LOW
- Reason: ...
- Complexity indicators: (module count, ambiguity count)

### Open Questions
- (questions discovered during analysis)
```

### 1.4 Determine complexity escalation

Set `escalate_to_draft_review = true` if ANY of:
- Risk level is CRITICAL or HIGH
- Open questions count > 3
- Affected modules count > 5
- `--plan` was provided (user should see how plans were merged)

---

## Phase 2: Present & Clarify

**Goal:** Validate scope and resolve ambiguities before drafting.

### 2.1 Present analysis summary

Show the user a concise version of the Analysis Brief:
- Scope, risk level, story type
- Key findings (3-5 bullets)
- Open questions

### 2.2 Ask targeted questions

Use `AskUserQuestion` for structured questions about:
- Scope boundaries (if unclear from the issue)
- Priority trade-offs (if multiple approaches exist)
- Constraints (timeline, backward compat, risk tolerance)
- Ambiguities found during analysis

Wait for user answers before proceeding.

---

## Phase 3: Draft Plan

**Goal:** Generate a complete story draft from the Analysis Brief + user answers.

### 3.1 Generate story draft

Create a full story following the `bmad-create-story` template format:

```markdown
# Story {{epic_num}}.{{story_num}}: {{story_title}}

Status: draft

## Story
As a {{role}},
I want {{action}},
so that {{benefit}}.

## Acceptance Criteria
1. Given ... When ... Then ...

## Tasks / Subtasks
- [ ] Task 1 (AC: #)
  - [ ] Subtask 1.1

## Dev Notes
- Architecture constraints
- File structure
- Test requirements

### Project Structure Notes
### References
```

Requirements for the draft:
- **Use the issue body as primary source** — do not replace the user's intent with your own analysis
- **Acceptance Criteria must be BDD-style** (Given/When/Then)
- **Tasks must reference which AC they satisfy**
- **Dev Notes must include** relevant project-context rules, affected files, test patterns

#### When `--plan` is provided

1. Read the external plan to understand its analysis, root causes, and fix plan
2. Generate the draft using both the analysis brief AND the external plan
3. The external plan is **primary authority** — follow its structure, root cause analysis, and fix plan
4. Only diverge if your analysis reveals something clearly missing or incorrect
5. Preserve the external plan's ordering, priorities, and file/line references

#### When `--story-key` is provided

Look up the story in `_bmad-output/planning-artifacts/epics.md` and use it as context.

Hold the draft in memory — do NOT write to disk yet.

### 3.2 Complexity check

If `escalate_to_draft_review` is true, proceed to Phase 3.5. Otherwise, skip to Phase 4.

---

## Phase 3.5: Draft Review (Conditional)

**Only triggered when complexity escalation is active.**

Present the draft to the user with a summary:
- Story statement
- AC count and key ACs
- Task count and task list overview
- Key design decisions made
- If `--plan`: highlight how the external plan was integrated

Ask the user: "Review before adversarial challenge? You can approve to proceed, or request changes."

If the user requests changes, incorporate them and update the draft.

---

## Phase 4: Adversarial Challenge

**Goal:** Subject the draft to parallel adversarial review by distinct personas.

### 4.1 Prepare reviewer contexts

Build the minimal context each reviewer needs:

- **Critic:** The plan draft only (nothing else)
- **Concrete Planner:** The plan draft + a file tree of affected directories + key source snippets
- **Dev Agent Proxy:** The plan draft + relevant project-context rules + architecture constraints
- **External Challenger (if `--plan`):** The plan draft + the original external plan file

### 4.2 Launch parallel reviewers

Launch 3-4 parallel **Agent** subagents. Each agent prompt:

1. "Read and follow the persona instructions in `{project-root}/_qsprocess/plan-reviewers/{{persona}}.md`"
2. Provide the reviewer's designated context
3. "Return your findings in the format specified by the persona file"

```
Parallel:
  Agent → critic.md     + plan draft
  Agent → concrete-planner.md + plan draft + file tree + snippets
  Agent → dev-proxy.md  + plan draft + project rules
  Agent → external-challenger.md + plan draft + external plan   (only if --plan)
```

### 4.3 Collect findings

Gather all reviewer outputs. If a reviewer fails or returns no findings, note it but continue with available results.

---

## Phase 5: Synthesize & Triage

**Goal:** Normalize, deduplicate, and classify all findings.

### 5.1 Normalize findings

All reviewers return findings with `[category]` tags. Normalize into a unified list:
```
- Source: {{reviewer name}}
- Category: critical | redesign | improve | clarify | dismiss
- Finding: ...
- Evidence/Location: ...
- Suggestion: ...
```

### 5.2 Deduplicate

If multiple reviewers flagged the same issue:
- Keep the strongest/most specific version
- Note which reviewers agree (agreement strengthens the finding)

### 5.3 Classify

For each finding, validate the reviewer's category assignment:
- `critical` — Blocks implementation. Contradictions, missing info, impossible requirements, rule violations.
- `redesign` — Better approach exists. The plan works but there's a fundamentally better way.
- `improve` — Good approach, can be better. Will auto-incorporate unless user objects.
- `clarify` — Ambiguous, needs user decision. Can't be resolved without user input.
- `dismiss` — False alarm, already handled, noise. Silently drop.

---

## Phase 6: Present & Resolve

**Goal:** Walk through findings with the user and get decisions.

### 6.1 Present findings summary

```
## Adversarial Review — Round {{N}}

**Reviewers:** Critic, Concrete Planner, Dev Proxy{{, External Challenger}}
**Findings:** {{total}} ({{critical}} critical, {{redesign}} redesign,
              {{improve}} improve, {{clarify}} clarify, {{dismissed}} dismissed)

### Critical (must resolve):
1. [Critic] {{finding}} → Proposed fix: ...

### Redesign proposals (your decision):
1. [Concrete Planner] {{finding}} → Alternative: ...

### Improvements (incorporating unless you object):
1. [Dev Proxy] {{finding}}

### Needs clarification:
1. {{question}}
```

### 6.2 Resolve findings

Use `AskUserQuestion` for:
- `clarify` items — user must provide the answer
- `redesign` items — user chooses: accept the redesign or keep current approach

`critical` items: present the proposed fix and confirm.
`improve` items: list them as "incorporating unless you object."

### 6.3 Decision: approve or revise

After resolving findings, ask: **"Approve plan and finalize, or revise and re-review?"**

- **Approve** → Incorporate all accepted findings into the draft, proceed to Phase 7
- **Revise** → Incorporate feedback, update draft, loop back to Phase 4 (adversarial review only — no re-analysis). Maximum 3 total rounds (initial + 2 revisions). If max rounds reached, force finalization with a warning.

---

## Phase 7: Finalize

**Goal:** Write the final story file, commit, and push.

### 7.1 Build the story key

Construct the story key with GitHub issue number. Always embed `Github-#N`:

- Epic stories: `{{epic}}-{{story}}-Github-#{{issue_number}}-{{slug}}`
- Bug fixes: `bug-Github-#{{issue_number}}-{{slug}}`
- Other: `{{prefix}}-Github-#{{issue_number}}-{{slug}}`

When using `--issue N` without `--story-key`, the `story_type` from `fetch_issue.py` determines the prefix:
- `"bug"` → `bug-Github-#{{issue_number}}-{{slug}}`
- `"feature"` → `feature-Github-#{{issue_number}}-{{slug}}` (or epic-style if story key also provided)

### 7.2 Write the story file

Write the final draft to: `_bmad-output/implementation-artifacts/{{story_key}}.md`

Update the story:
- Set `Status: ready-for-dev`
- Incorporate all accepted findings (better ACs, better tasks, better dev notes)

### 7.3 Add frontmatter

After the title line, add:

```yaml
issue: {{issue_number}}
branch: "QS_{{issue_number}}"
```

### 7.4 Append Adversarial Review Notes

Add a section at the end of the story file:

```markdown
## Adversarial Review Notes

**Reviewers:** Critic, Concrete Planner, Dev Proxy{{, External Challenger}}
**Rounds:** {{N}}

### Key findings incorporated:
- {{finding}} → {{resolution}}

### Decisions made:
- {{redesign decision}} — Rationale: {{user's reason}}

### Known risks acknowledged:
- {{risk from reviewers that wasn't actionable but worth noting}}
```

### 7.5 Commit and push

```bash
git add _bmad-output/
git commit -m "story: create {{story_key}} (Github-#{{issue_number}})"
git push -u origin QS_{{issue_number}}
```

### 7.6 Output next-step command

Run `next_step.py` to generate command options:

```bash
python scripts/qs/next_step.py --skill implement-story --issue {{issue_number}} --work-dir "$(pwd)" --title "{{title}}"
```

Parse the JSON output and tell the user:

```
Story created on branch QS_{{issue_number}}.

**Option A — New context:**
  {{new_context}}

**Option B — Same context:**
  {{same_context}}
```

If the JSON includes `pycharm_context`, also show:

```
**Option C — PyCharm + clipboard:**
  {{pycharm_context}}
  (Opens PyCharm on the worktree. In PyCharm: ⌥F12 → ⌘V → Enter)
```

If the JSON includes `pycharm_applescript_context`, also show:

```
**Option D — PyCharm + auto-type (experimental):**
  {{pycharm_applescript_context}}
  (Like C but tries to auto-type via AppleScript. Needs Accessibility permissions.)
```

---

## Story Key Construction Reference

This section consolidates the story key logic for all input combinations:

| Input | Story key format | Example |
|-------|-----------------|---------|
| `--issue N --story-key 3.2` | `{{epic}}-{{story}}-Github-#{{N}}-{{slug}}` | `3-2-Github-#60-power-calc` |
| `--issue N` (bug label) | `bug-Github-#{{N}}-{{slug}}` | `bug-Github-#61-fix-overflow` |
| `--issue N` (feature) | `feature-Github-#{{N}}-{{slug}}` | `feature-Github-#62-add-export` |
| `--issue N --plan /path` | Same as above based on type | (plan doesn't change key format) |
