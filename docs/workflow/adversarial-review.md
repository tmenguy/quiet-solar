# Adversarial review — the 4-reviewer pattern

Two phases — `create-plan` and `review-task` — fan out into four parallel
sub-agents, each examining the same input through a different lens. The
goal is to catch issues a single reviewer would miss because lens bias is
real: a "completeness" reviewer doesn't think about scope creep; a "scope"
reviewer doesn't enumerate edge cases.

## Why parallel, not serial

All four subagents must be spawned in **one message with four parallel
invocations**, not one-by-one in sequence. Two reasons:

1. **Independence**: serial spawning leaks earlier findings into later
   reviewers' context. They start agreeing instead of disagreeing.
2. **Speed**: 4 parallel ~30s reviews beat 4 sequential ~30s reviews
   every time.

## How to spawn in parallel

In Claude Code: emit one assistant message with four `Agent` tool calls,
each with a different `subagent_type`. The harness runs them
concurrently.

In Cursor: emit one message that invokes the four subagents (Cursor
supports this natively as of 2.4).

In OpenCode (legacy pipeline): four parallel `Task` calls in one message.

## Plan reviewers (used by `qs-create-plan`)

Spawned with the plan draft (and, where noted, an additional artifact).
Each returns a findings list with categories `critical` / `redesign` /
`improve` / `clarify`.

### `qs-plan-critic`
**Lens**: cynical, blunt. **Input**: plan draft text **only** (no
codebase, no issue body, no rules). **Looks for**: logical
contradictions, hand-waving, missing failure modes, untestable acceptance
criteria. **Hard rule**: never read repo files. If the plan can't stand
alone, that's a finding.

### `qs-plan-concrete-planner`
**Lens**: file-level concreteness. **Input**: plan + file tree + source
snippets. **Looks for**: vague file references ("update the handler"),
non-existent paths, vague test specs ("write tests"). **Hard rule**: no
bash; read-only.

### `qs-plan-dev-proxy`
**Lens**: implementation simulator. **Input**: plan +
[project-rules.md](project-rules.md) + [project-context.md](project-context.md).
**Looks for**: layer-boundary violations, async anti-patterns, missing
context the implementer will need, coverage infeasibility. **Hard rule**:
never read source code. Simulate not having codebase access.

### `qs-plan-scope-guardian`
**Lens**: scope vs. issue. **Input**: plan + the GitHub issue body.
**Looks for**: over-engineering, gold-plating, silent requirement
expansion, abstractions not justified by the issue. **Hard rule**: never
read repo. Compare plan's scope to issue's stated scope exactly.

## Code reviewers (used by `qs-review-task`)

Spawned with the PR diff. Each returns findings categorized as
`must-fix` / `should-fix` / `nice-to-have`.

### `qs-review-blind-hunter`
**Lens**: diff-only. **Input**: `gh pr diff <N>` output. **Looks for**:
obvious bugs visible from the diff alone — off-by-one, swapped args,
broken string literals, missing error handling, security smells, lint
violations. **Hard rule**: NEVER read repo files. NEVER fetch the issue
body or story file.

### `qs-review-edge-case-hunter`
**Lens**: exhaustive boundaries. **Input**: PR diff + repo read-only.
**Looks for**: unhandled edge cases — empty/None/0/negative/max inputs,
cold start, concurrent state, partial failure, retry, cache miss, etc.
Walks every branching path. **Hard rule**: never duplicate blind-hunter
findings; stay in the edge-case lane.

### `qs-review-acceptance-auditor`
**Lens**: AC traceability. **Input**: PR diff + story file. **Looks
for**: ACs from the story not implemented or not tested. Builds a
traceability matrix. **Hard rule**: don't re-litigate design; just
verify the PR does what the story says.

### `qs-review-coderabbit`
**Lens**: external auto-review. **Input**: PR number. **Looks for**:
whatever CodeRabbit flags, normalized to must-fix / should-fix /
nice-to-have. **Hard rule**: never do your own analysis; if CodeRabbit
didn't flag it, don't manufacture a finding.

## Triage model

The orchestrator (the parent agent, not the reviewers) consolidates
findings:

1. **Normalize**: deduplicate identical findings across reviewers
   (`file:line` + similar text).
2. **Bucket**: critical → must-fix; vague/inapplicable → invalid.
3. **Present**: show a summary table to the user.
4. **Triage**: ask "fix all / skip all / one by one?". If one by one,
   walk each finding and ask "fix or skip?". Collect all decisions, then
   confirm before acting.
5. **Loop** (review only): if fixes were applied, re-run the review
   (loop back to step 1). Loop until clean or user says "proceed".

## Don't confuse the patterns

The plan reviewers and the code reviewers are **disjoint sets**. The
plan reviewers never see code; the code reviewers never see the plan
draft (the acceptance-auditor sees the *final* story file, not the
draft). This separation prevents the plan reviewers from drifting into
implementation suggestions and the code reviewers from re-litigating
design.
