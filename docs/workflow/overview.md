# Workflow overview — static-agent pipeline

This document describes the development pipeline for Quiet Solar. It works
identically across Claude Code, Cursor, OpenCode, and Codex;
harness-specific machinery (session spawning, launcher emission) is
isolated in `scripts/qs/launchers/`.

## The six phases

```text
  setup-task → create-plan → implement-task → review-task → finish-task → release
                                                                            (independent)
```

The **bug × product** lane (QS-335) diverges into a diagnose-first flow:
`create-plan` is replaced by `diagnose-task` and `review-task` by
`verify-task` for that lane only
(`setup → diagnose → fix → verify → finish`). See
[lanes/bug-product.md](lanes/bug-product.md).

| Phase            | Where it runs        | What it produces                            |
| ---------------- | -------------------- | ------------------------------------------- |
| `setup-task`     | main checkout        | issue, branch `QS_<N>`, worktree            |
| `create-plan`    | worktree             | story file at `docs/stories/QS-<N>.story.md` via an interactive discuss/review/finalize loop, committed at finalize |
| `implement-task` | worktree             | TDD code, green quality gate, PR opened     |
| `review-task`    | worktree             | parallel adversarial review, fix-plan loop  |
| `finish-task`    | worktree             | PR merged, worktree removed                 |
| `release`        | main checkout        | tag, manifest bump, GitHub Release          |

Two phases (`setup-task`, `release`) are entered from the main checkout;
the rest run in the worktree.

## Static agents — no rendering

There is exactly **one agent file per phase**, checked in to `.claude/agents/`
(mirrored in `.cursor/agents/` and `.opencode/agents/`). Agents discover
task context at runtime via
`python scripts/qs/context.py`, which reads `git branch --show-current`
(`QS_<N>`) and resolves the issue, title, story file, and PR number from
there.

This replaces the older per-task rendering model (`qs-implement-task-QS-42.md`
generated from `.tmpl` files). The previous OpenCode pipeline now lives
under `legacy/`; this is the only supported model going forward.

## Adversarial review (parallel sub-agents)

Two phases fan out into parallel sub-agents that each look at the same
input through a different lens:

- **create-plan** is an on-demand, repeatable review inside its mode
  loop. **Round 1** spawns the 4 global plan reviewers —
  `qs-plan-critic`, `qs-plan-concrete-planner`, `qs-plan-dev-proxy`,
  `qs-plan-scope-guardian`. **Round 2+** spawns those four **plus
  `qs-plan-delta-auditor`**, a fifth diff-aware reviewer fed an
  in-context diff of the plan's edits + the prior round's accepted
  findings.
- **review-task** always spawns four: `qs-review-blind-hunter`,
  `qs-review-edge-case-hunter`, `qs-review-acceptance-auditor`,
  `qs-review-coderabbit`.

All of a round's reviewers must be spawned in **one message with
parallel sub-agent invocations** (four, or five for a round-2+ plan
review) — serial spawning defeats the design (later reviewers see
earlier findings and conform to them). See
[adversarial-review.md](adversarial-review.md) for the full pattern.

## Orchestrators are interactive sessions; sub-agents are parallel fan-out

The pipeline runs two fundamentally different kinds of agent, and the
launcher distinction matters.

**Phase orchestrators** (`qs-setup-task`, `qs-create-plan`,
`qs-diagnose-task`, `qs-implement-task`, `qs-implement-setup-task`,
`qs-review-task`, `qs-verify-task`, `qs-finish-task`, `qs-release`;
`qs-diagnose-task` / `qs-verify-task` run only in the bug × product
lane) are meant to run as **interactive
`claude --agent qs-<phase>` sessions**. Claude Code launches a fresh
session whose system prompt IS the agent body, and the user converses
with the persona mid-flight — answering clarifying questions in
`qs-create-plan` step 2, authorizing the quality gate in
`qs-implement-task` step 4, driving "fix all / skip all" triage in
`qs-review-task`, and so on. One phase = one session (the system prompt
is immutable mid-session). The launcher (`scripts/qs/launchers/`) emits
a `claude --agent qs-<phase>` invocation in the new_context for exactly
this reason.

**Sub-agents** are the plan reviewers spawned by `qs-create-plan` (4;
5 in round 2+ with the delta-auditor) and the 4 code reviewers spawned
by `qs-review-task`. They **stay as
`Agent`-tool fan-out** — non-interactive, parallel, returning a final
findings report. That's the right shape for them: they're independent
parallel workers, which is exactly what the `Agent` tool is good at.
Making them interactive would defeat the design (later reviewers would
see earlier findings and conform).

**Slash commands stay as a degraded fallback.** Running `/create-plan`
(or any `/qs-*` slash) from the default Claude Code session spawns the
phase orchestrator via the `Agent` tool — non-interactive, one-shot,
returning a final summary with no way for the user to interject. The
persona body executes, but the per-phase "answer mid-flight" UX is
gone. This is broken-by-design UX kept **only as a fallback path** for
environments without the CLI launcher (notably Claude Desktop). The
slash command files under `.claude/commands/` are reworded — not
deleted — to flag this distinction.

### Claude Desktop limitation

Claude Desktop has no equivalent to `claude --agent`: no URL scheme, no
CLI argument pass-through, and no UI gesture that pre-loads an agent
persona *by itself*. Read with the **cold-start** qualifier, all three
remain true — there is still no way to cold-start a GUI session on a
directory programmatically.

Both qualifiers are load-bearing:

- `/desktop` *can* move an already-running CLI session to the GUI on the
  same directory and branch, persona included — see
  [harness.md](harness.md) → "Hybrid: `/desktop`". What has no
  programmatic entry point is creating a GUI session from nothing.
- **New session** on a directory whose `.claude/settings.local.json`
  carries an `agent` key *does* boot the persona. That is a UI gesture
  which pre-loads one — but only because a settings file was written
  first, which is why the claim is "no gesture by itself" rather than
  "no gesture".

**What does not follow is that GUI users are stuck with the
slash-command fallback.** A fourth mechanism exists, and it is the one
the pipeline uses: the `agent` key in `.claude/settings.local.json`. The
launcher writes it into the worktree at every handoff, so a GUI session
the user opens there boots as the phase orchestrator with no `--agent`
flag involved. The gesture (**New session** → select directory → name
it), the traps, and the `/desktop` hybrid are documented in
[harness.md](harness.md) → "GUI launch surface (Claude Code Desktop)".

The existing `pycharm_context` (clipboard / AppleScript)
helpers in `scripts/qs/launchers/claude.py` remain the suggested
bridge for IDE-embedded terminals: setup-task emits a `pycharm_context`
shell command that copies
the launcher invocation to the clipboard and opens PyCharm on the
worktree — users then paste into PyCharm's embedded terminal to get
the interactive path. This is honest about the remaining limitation, not
an attempt to automate the GUI with brittle clipboard tricks.

## Phase routing

`create-plan` chooses between two implement-phase variants based on the
files it expects to touch:

- **`implement-setup-task`** — all touched files are in dev-environment
  paths (`scripts/`, `.claude/`, `.cursor/`, `.opencode/`,
  `legacy/`, `docs/`, `.github/`, top-level config). Narrower
  edit scope; the quality gate runs the dev-only fast path.
- **`implement-task`** — production code under
  `custom_components/quiet_solar/` is touched. Full quality gate
  (pytest 100% + ruff + mypy + translations).

**Lanes (QS-332).** Every task is born in exactly one of 6 lanes —
{bug, feature, epic} × {product, factory} — declared at setup as GitHub
labels (`kind:*`, `target:*`, `scale:*`) and exposed by
`python scripts/qs/context.py` as the `lane` field. Each lane has a
protocol file `docs/workflow/lanes/<lane>.md`; orchestrator agents read
their lane file early in the session (falling back to
[phase-protocols.md](phase-protocols.md) when the lane is undeclared —
pre-existing worktrees only). The lane files start as byte-identical
copies of `phase-protocols.md` (enforced by
`tests/qs/docs/test_lanes.py`); they diverge one PR per lane (#335–#340).
The quality gate enforces the declaration (missing declaration fails)
and surfaces cross-target diffs as a loud warning — purpose, not path,
is the classifier, so a crossing never fails the gate.

## Harness abstraction

Everything harness-specific lives in `scripts/qs/launchers/*.py` and is
selected by `scripts/qs/harness.py::detect()`. The agent bodies share
an aligned core protocol across harnesses; the frontmatter (`tools:`
for Claude Code, `readonly:` for Cursor, `permission:` for OpenCode)
and the declared harness-specific sections differ — see
[harness.md](harness.md).

See [harness.md](harness.md) for how to add a new harness.
See [project-rules.md](project-rules.md) § "Harness sync" for the cross-harness body-parity rule.
Code intelligence is opt-in per harness: Claude agents carry a native pyright `LSP` tool — see [harness.md](harness.md) § "Code intelligence (LSP)" and [../agents/lsp-evaluation.md](../agents/lsp-evaluation.md).

## Required reading

- [project-rules.md](project-rules.md) — architecture constraints, commands, workflow routing
- [project-context.md](project-context.md) — 42-rule code style set
- [phase-protocols.md](phase-protocols.md) — what each phase does, in detail
- [adversarial-review.md](adversarial-review.md) — the 4-reviewer pattern
- [harness.md](harness.md) — adding a new harness

## Quality gate

`python scripts/qs/quality_gate.py` — pytest 100% coverage + ruff + mypy
+ translations. Smart scope detection skips the full suite when only
dev-infrastructure files changed (**dev-only fast path**), or only
UI assets under `custom_components/quiet_solar/ui/` changed (**ui-only
fast path** — runs only `tests/test_dashboard_rendering.py`).
