# Workflow overview — static-agent pipeline

This document describes the development pipeline for Quiet Solar. It works
identically across Claude Code, Cursor, OpenCode (via the legacy
`_qsprocess_opencode/` pipeline) and Codex; harness-specific machinery
(session spawning, launcher emission) is isolated in
`scripts/qs/launchers/`.

## The six phases

```text
  setup-task → create-plan → implement-task → review-task → finish-task → release
                                                                            (independent)
```

| Phase            | Where it runs        | What it produces                            |
| ---------------- | -------------------- | ------------------------------------------- |
| `setup-task`     | main checkout        | issue, branch `QS_<N>`, worktree            |
| `create-plan`    | worktree             | story file at `docs/stories/QS-<N>.story.md`, committed |
| `implement-task` | worktree             | TDD code, green quality gate, PR opened     |
| `review-task`    | worktree             | parallel adversarial review, fix-plan loop  |
| `finish-task`    | worktree             | PR merged, worktree removed                 |
| `release`        | main checkout        | tag, manifest bump, GitHub Release          |

Two phases (`setup-task`, `release`) are entered from the main checkout;
the rest run in the worktree.

## Static agents — no rendering

There is exactly **one agent file per phase**, checked in to `.claude/agents/`
(or `.cursor/agents/`). Agents discover task context at runtime via
`python scripts/qs/context.py`, which reads `git branch --show-current`
(`QS_<N>`) and resolves the issue, title, story file, and PR number from
there.

This replaces the older per-task rendering model (`qs-implement-task-QS-42.md`
generated from `.tmpl` files). There is no `render_agent.py`, no
`cleanup_agents.py`, no template directory in this pipeline. The old
`_qsprocess_opencode/` rendering pipeline is still functional and unchanged;
it will be retired once this static-agent pipeline is proven.

## Adversarial review (parallel sub-agents)

Two phases fan out into four parallel sub-agents that each look at the
same input through a different lens:

- **create-plan** spawns: `qs-plan-critic`, `qs-plan-concrete-planner`,
  `qs-plan-dev-proxy`, `qs-plan-scope-guardian`.
- **review-task** spawns: `qs-review-blind-hunter`,
  `qs-review-edge-case-hunter`, `qs-review-acceptance-auditor`,
  `qs-review-coderabbit`.

All four must be spawned in **one message with four parallel sub-agent
invocations** — serial spawning defeats the design (later reviewers see
earlier findings and conform to them). See
[adversarial-review.md](adversarial-review.md) for the full pattern.

## Phase routing

`create-plan` chooses between two implement-phase variants based on the
files it expects to touch:

- **`implement-setup-task`** — all touched files are in dev-environment
  paths (`scripts/`, `.claude/`, `.cursor/`, `.opencode/`,
  `_qsprocess_opencode/`, `docs/`, `.github/`, top-level config). Narrower
  edit scope; the quality gate runs the dev-only fast path.
- **`implement-task`** — production code under
  `custom_components/quiet_solar/` is touched. Full quality gate
  (pytest 100% + ruff + mypy + translations).

## Harness abstraction

Everything harness-specific lives in `scripts/qs/launchers/*.py` and is
selected by `scripts/qs/harness.py::detect()`. The agent bodies are
nearly identical across harnesses; only the frontmatter differs
(`tools:` for Claude Code, `readonly:` for Cursor).

See [harness.md](harness.md) for how to add a new harness.

## Required reading

- [project-rules.md](project-rules.md) — architecture constraints, commands, workflow routing
- [project-context.md](project-context.md) — 42-rule code style set
- [phase-protocols.md](phase-protocols.md) — what each phase does, in detail
- [adversarial-review.md](adversarial-review.md) — the 4-reviewer pattern
- [harness.md](harness.md) — adding a new harness

## Quality gate

`python scripts/qs/quality_gate.py` — pytest 100% coverage + ruff + mypy
+ translations. Smart scope detection skips the full suite when only
dev-infrastructure files changed.
