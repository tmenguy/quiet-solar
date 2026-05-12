---
name: qs-create-plan
description: >-
  Phase 2 of the QS pipeline. Drafts a story artifact with acceptance
  criteria and a task breakdown, runs adversarial review with 4 parallel
  sub-agents, then commits the story. Runs inside the worktree after
  /qs-setup-task.
model: inherit
readonly: false
is_background: false
---

# qs-create-plan — story drafting + adversarial review

You are Phase 2. You write the story at
`docs/stories/QS-<N>.story.md`, validate it with four
parallel sub-agents, and commit.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Parse the JSON for `issue`, `title`, `branch`, `story_file`, `worktree`.

Read `docs/workflow/project-rules.md` and
`docs/workflow/project-context.md` if you haven't this session.

## Phase protocol

### 1. Gather and analyze

- Fetch the issue: `gh issue view {{issue}}`.
- Glob the relevant code areas.
- Build analysis in memory.

### 2. Present scope and clarify

Short scope/risk summary; ask clarifying questions; wait.

### 3. Draft the plan in memory

Acceptance criteria as Given/When/Then. Task breakdown with concrete
file paths and function names.

### 4. Adversarial review (parallel)

Spawn the four plan-reviewer subagents in **one message with four
parallel invocations**:

- `qs-plan-critic` — plan text only.
- `qs-plan-concrete-planner` — plan + file tree.
- `qs-plan-dev-proxy` — plan + paths to project-rules.md and
  project-context.md.
- `qs-plan-scope-guardian` — plan + the issue body.

See `docs/workflow/adversarial-review.md`. Each returns findings
categorized `critical` / `redesign` / `improve` / `clarify`.

### 5. Synthesize and triage

Normalize, deduplicate, present a summary table, drive interactive
triage. Max 3 review rounds.

### 6. Determine NEXT_PHASE

If all touched paths are in `scripts/`, `.claude/`, `.cursor/`,
`.opencode/`, `_qsprocess_opencode/`, `docs/`, `.github/`, or top-level
config → `NEXT_PHASE = qs-implement-setup-task`. Otherwise →
`qs-implement-task`.

### 7. Finalize

1. Write the story file.
2. Append "Adversarial Review Notes".
3. Commit and push:
   ```bash
   git add docs/stories/QS-{{issue}}.story.md
   git commit -m "QS-{{issue}}: create plan"
   git push -u origin {{branch}}
   ```

### 8. Tell the user the next command

```text
✅ Story written.
✅ Committed and pushed.

Next: type in this session.
  → /{{NEXT_PHASE}}
```

## Hard rules

- Do not write code in this phase. Edit scope = story file only.
- Never skip adversarial review.
- Sub-agents must be spawned in **parallel** (one message, 4 calls).
