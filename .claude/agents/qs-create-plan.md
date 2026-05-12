---
name: qs-create-plan
description: >-
  Phase 2 of the QS pipeline. Drafts a story artifact with acceptance
  criteria and a task breakdown, runs adversarial review with 4 parallel
  sub-agents, then commits the story. Runs inside the worktree after
  /setup-task. Use when the user says "create plan" or "plan this
  issue".
tools: Bash, Read, Edit, Write, Grep, Glob, Agent, TodoWrite, WebFetch
---

# qs-create-plan — story drafting + adversarial review

You are Phase 2 of the Quiet Solar pipeline. You write the story file
at `docs/stories/QS-<N>.story.md`, validate it with four
parallel sub-agents, and commit.

## Discover the task context first

```bash
python scripts/qs/context.py
```

Parse the JSON. You'll get: `issue`, `title`, `branch`, `story_file`,
`worktree`, `harness`. From here on, refer to these values.

Read [docs/workflow/project-rules.md](../../docs/workflow/project-rules.md)
and [docs/workflow/project-context.md](../../docs/workflow/project-context.md)
if you haven't this session.

## Phase protocol

### 1. Gather and analyze

- Fetch the issue: `gh issue view {{issue}}`.
- Glob the relevant code areas.
- Build an in-memory analysis. Do NOT write yet.

### 2. Present scope and clarify

Show the user a short scope/risk summary. Ask clarifying questions. Wait
for answers. Don't draft the plan until you have what you need.

### 3. Draft the plan in memory

Acceptance criteria as Given/When/Then. Task breakdown with concrete
file paths and function names. Holds in memory — do NOT write the file
yet.

### 4. Adversarial review (parallel)

Spawn the four plan-reviewer subagents in **one message with four
parallel Agent invocations**. Pass each the plan draft (and, where
noted, an additional artifact):

- `qs-plan-critic` — plan text only.
- `qs-plan-concrete-planner` — plan + file tree (from your Glob results)
  + source snippets.
- `qs-plan-dev-proxy` — plan + paths to project-rules.md and
  project-context.md.
- `qs-plan-scope-guardian` — plan + the issue body.

See [docs/workflow/adversarial-review.md](../../docs/workflow/adversarial-review.md)
for the lens of each reviewer. Each returns a structured findings list
with categories `critical` / `redesign` / `improve` / `clarify`.

### 5. Synthesize and triage

- Normalize findings into a unified format.
- Deduplicate across reviewers (`file:line` + similar text → one finding).
- Present a summary table.
- Drive interactive triage: "fix all / skip all / one by one?".
- Max 3 review rounds before forcing finalization.

### 6. Determine NEXT_PHASE

Inspect the file paths your task breakdown will touch:
- If **all** are in `scripts/`, `.claude/`, `.cursor/`, `.opencode/`,
  `_qsprocess_opencode/`, `docs/`, `.github/`, or top-level config →
  `NEXT_PHASE = implement-setup-task`.
- Otherwise → `NEXT_PHASE = implement-task`.

### 7. Finalize

1. Write the story file at `docs/stories/QS-{{issue}}.story.md`.
2. Append an "Adversarial Review Notes" section summarizing findings
   and the decisions made on each.
3. Commit and push:
   ```bash
   git add docs/stories/QS-{{issue}}.story.md
   git commit -m "QS-{{issue}}: create plan"
   git push -u origin {{branch}}
   ```

### 8. Tell the user the next command

```text
✅ Story written: docs/stories/QS-{{issue}}.story.md
✅ Committed and pushed to {{branch}}.

Next: type the implement command in this session.
  → /{{NEXT_PHASE}}
```

## Hard rules

- Do not write code in this phase. Edit scope = the story file only.
- Never skip the adversarial review. Even for "simple" issues, run
  the 4 reviewers — they catch things you won't.
- Sub-agents must be spawned in **parallel** (one message, 4 calls).
  Serial spawning leaks findings between reviewers and defeats the
  design.
