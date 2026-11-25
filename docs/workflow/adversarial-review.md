# Adversarial review — the parallel-reviewer pattern

Two phases — `create-plan` and `review-task` — fan out into parallel
sub-agents, each examining the same input through a different lens. The
goal is to catch issues a single reviewer would miss because lens bias is
real: a "completeness" reviewer doesn't think about scope creep; a "scope"
reviewer doesn't enumerate edge cases.

`review-task` always fans out into **four** code reviewers, then runs a
single **finding verifier** over their consolidated output (QS-453 —
see "Finding verification and the review ledger" below).
`create-plan` is a **repeatable, on-demand** review inside its mode loop:
**round 1** runs the **four global plan reviewers**; **round 2+** runs
those four **plus a fifth, diff-aware** reviewer
(`qs-plan-delta-auditor`) — see "Round asymmetry" below.

The **bug × product** lane (QS-335) runs dedicated minimal rosters
instead — 2 diagnosis reviewers (`diagnose-task`) and 3 fix-verification
reviewers (`verify-task`); see
[lanes/bug-product.md](lanes/bug-product.md).

## Why parallel, not serial

All the round's reviewers must be spawned in **one message with parallel
invocations** (four for a code review or a round-1 plan review; five for
a round-2+ plan review), not one-by-one in sequence. The one carve-out is
the code-review **finding verifier** (`qs-review-finding-verifier`): it
is not a reviewer but a sequential post-pass that runs once, *after* the
fan-out and the consolidation — it needs their output. Two reasons for
the parallel rule:

1. **Independence**: serial spawning leaks earlier findings into later
   reviewers' context. They start agreeing instead of disagreeing.
2. **Speed**: 4 parallel ~30s reviews beat 4 sequential ~30s reviews
   every time.

## How to spawn in parallel

In Claude Code: emit one assistant message with one `Agent` tool call
per reviewer, each with a different `subagent_type`. The harness runs
them concurrently.

In OpenCode: the same fan-out — all of the round's reviewers invoked in
parallel in one message.

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

### `qs-plan-delta-auditor` (round 2+ only)
**Lens**: diff-aware regression/resolution. **Input**: a unified diff
between the previously-reviewed plan and the current plan (computed
**in-context** by the orchestrator — no snapshot files), plus the prior
round's accepted findings. **Looks for**: (a) whether each prior
accepted finding was actually resolved by the edits, and (b) new
contradictions, regressions, or scope drift the edits introduced.
**Hard rule**: operates purely on the prompt input (read-only tooling —
e.g. `tools: Read` on Claude); never diffs anything itself, never
reads repo files, stays strictly on the delta (the global four own
whole-plan coverage). Returns "No delta to review." on an empty diff.

### Round asymmetry (plan reviews only)

- **Round 1**: the **4 global reviewers** above, each on the whole plan.
- **Round 2+**: the same 4 global reviewers (re-run on the whole plan)
  **plus `qs-plan-delta-auditor`**. The orchestrator holds both the
  previous-reviewed plan text and the current text in-session, computes
  the unified diff **in-context**, and pastes that diff + the prior
  round's accepted-findings list into the delta-auditor's prompt.

### Finding-state model

The orchestrator keeps light per-finding state —
`open` / `resolved` / `rejected` — in the story's "Adversarial Review
Notes". Re-runs **dedupe against this state**: a finding the user
explicitly **rejected** does not resurface as new; a `resolved` finding
the delta-auditor reports as still present flips back to `open`. The
orchestrator presents **deltas first** (new / changed / resolved) with
the full global list collapsed underneath.

## Code reviewers (used by `qs-review-task`)

Spawned with the PR diff. Each returns findings categorized as
`must-fix` / `should-fix` / `nice-to-have`.

### `qs-review-blind-hunter`
**Lens**: diff-only. **Input**: `gh pr diff <N>` output (in a delta
round, the delta diff text pasted by the orchestrator). **Looks for**:
obvious bugs visible from the diff alone — off-by-one, swapped args,
broken string literals, missing error handling, security smells.
**Muzzled (QS-453)**: reports only `must-fix` / `should-fix`, every
finding with a `Reproduces when:` scenario (no scenario → not reported);
TODO/FIXME and style/lint are left to ruff/mypy. Its independent
diff-only lens stays — it still finds must-fix issues no other reviewer
reports. **Hard rule**: NEVER read repo files. NEVER fetch the issue
body or story file.

### `qs-review-edge-case-hunter`
**Lens**: exhaustive boundaries. **Input**: PR diff + repo read-only.
**Looks for**: unhandled edge cases — empty/None/0/negative/max inputs,
cold start, concurrent state, partial failure, retry, cache miss, etc.
Walks every branching path; in a delta round it reviews the diff since
`base` but still enumerates the boundaries of every function the delta
touches. **Hard rule**: never duplicate blind-hunter findings; stay in
the edge-case lane.

### `qs-review-acceptance-auditor`
**Lens**: AC traceability. **Input**: PR diff + story file. **Looks
for**: ACs from the story not implemented or not tested. Builds a
traceability matrix. **Hard rule**: don't re-litigate design; just
verify the PR does what the story says.

### `qs-review-coderabbit`
**Lens**: external auto-review. **Input**: PR number. **Looks for**:
whatever CodeRabbit flags, normalized to must-fix / should-fix /
nice-to-have — parsing only comments **authored by CodeRabbit** (never
the orchestrators' own PR comments). **Hard rule**: never do your own
analysis; if CodeRabbit didn't flag it, don't manufacture a finding.

## Finding verification and the review ledger (QS-453)

Both code-review orchestrators (`qs-review-task` and, in the bug ×
product lane, `qs-verify-task`) run the same round protocol around their
reviewer fan-out:

1. **Open the round** — `review_ledger.py start-round` on the PR head.
   Rounds 1–2 are **full**; from round 3 on a round is a **delta**
   against the last reviewed head (forced full by `--force-full`, a
   non-ancestor base, or a merge commit). In a delta round only the
   diff-scoped reviewers (blind-hunter, edge-case-hunter) see the delta;
   the whole-PR reviewers (acceptance-auditor, coderabbit,
   regression-proof) stay full. Nothing changed since the last round →
   no round.
2. **Fan out** the reviewers in parallel and **consolidate** (keys
   `C-1…`).
3. **Verify** — `qs-review-finding-verifier` (read-only, `deep` class)
   reads the code and returns one JSON object: per finding a verdict
   (`CONFIRMED` / `PLAUSIBLE` / `REFUTED`, evidence `file:line`), a
   concrete `reproduces_when`, a `severity_action` (`keep`; `demote` a
   verified must/should finding with no scenario; `promote` a
   nice-to-have it verified) and a **relation** to the ledger (`NEW` /
   `DUPLICATE` / `REOPENS` / `FLIP-FLOP`); plus a `RESOLVED` /
   `STILL-PRESENT` check of every `accepted` finding. If the verifier
   fails, the orchestrator records an **unverified** fallback round
   instead (nothing auto-decided).
4. **Record** — `review_ledger.py add` validates the object and
   reconciles it deterministically against the ledger.
5. **Triage** in six buckets, first match wins: `FLIP-FLOP` (explicit
   decision), `Reopened` (fix), `Unverified` (explicit decision),
   `Dropped by verifier` (reject, overridable), `Blocking` (fix),
   `Nice-to-have` (reject — opt-in). "fix all" covers `Reopened` +
   `Blocking`. A round with only dropped / nice-to-have findings takes
   the zero-blocking fast path.
6. **Decide and commit** — `review_ledger.py set` per decision,
   `check-round`, then the fix plan (headings `### [must-fix] F-003 —
   …`) and the ledger are committed together.

**The ledger** — `docs/stories/QS-<N>.review_ledger.json`, written only
by `scripts/qs/review_ledger.py` (the orchestrator is the single writer;
implement phases never stage it). Findings get stable IDs (`F-001`…)
and a state history: `open` → `accepted` / `rejected` (by `user`,
`default-skip` or `verifier`); `accepted` → `resolved` (check) or back
to `open` (still present / reopened); `rejected` by the user → `open`
when a reviewer challenges it in changed code; `accepted` / `resolved`
→ `superseded` when a FLIP-FLOP that undoes it is accepted. Repeats of
decided findings are suppressed as duplicates, and **flip-flops** — a
round asking to undo an earlier fix, or re-raising what the user
rejected — are surfaced for an explicit decision. The round lifecycle is
git: a committed ledger is a finished round; a dirty ledger is an
unfinished one, discarded and redone by the next `start-round`.

**Why not `/code-review`.** Claude Code's built-in `/code-review` was
considered as a reviewer and rejected on harness parity alone: it is
Claude-only, while both harnesses render the same reviewer roster. Its
useful ideas — a CONFIRMED / PLAUSIBLE verdict and a concrete failure
scenario per finding — are what the finding verifier reproduces.

## Triage model

The orchestrator (the parent agent, not the reviewers) consolidates
findings:

1. **Normalize**: deduplicate identical findings across reviewers
   (`file:line` + similar text). For plan reviews, also dedupe against
   the **finding-state model** (`open`/`resolved`/`rejected`) so a
   rejected finding doesn't resurface and a regressed `resolved` finding
   flips back to `open`.
2. **Bucket**: critical → must-fix; vague/inapplicable → invalid. Code
   reviews then run the finding verifier and bucket by its verdicts
   (see "Finding verification and the review ledger").
3. **Present**: show a summary table to the user. For plan reviews,
   present **deltas first** (new / changed / resolved).
4. **Triage**: ask "fix all / skip all / one by one?". If one by one,
   walk each finding and ask "fix or skip?". Collect all decisions, then
   confirm before acting.
5. **Loop**: re-running review is **on-demand and repeatable**, not a
   capped counter. In `create-plan` the user re-enters REVIEW whenever
   they want; FINALIZE is an **advisory** gate that never hard-blocks.
   In `review-task` the loop runs until clean or the user says
   "proceed".

## Don't confuse the patterns

The plan reviewers and the code reviewers are **disjoint sets**. The
plan reviewers never see code; the code reviewers never see the plan
draft (the acceptance-auditor sees the *final* story file, not the
draft). This separation prevents the plan reviewers from drifting into
implementation suggestions and the code reviewers from re-litigating
design.
