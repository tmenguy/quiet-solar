---
title: TheDev
slug: the-dev
kind: persona
last_verified: 2026-05-19
---

# TheDev

## TL;DR

TheDev is the developer working on quiet-solar — often the same person
as TheAdmin. They find joy in solving real family problems through
code. Two modes: **building** (smooth pipeline, no manual GitHub ops,
expressive tests) and **exploring** (agentic pairing, fast feedback,
green/red TDD loops). They are **allergic to bureaucracy** — every
extra ceremony, every duplicate doc, every "fill out this form before
you commit" is friction that erodes velocity.

## When you need this persona

- Designing a workflow phase, slash command, or static agent.
- Adding a quality-gate check, a test fixture, or a CI job.
- Writing documentation under `docs/workflow/` or `docs/agents/`.
- Choosing between "build a tool" vs "add a rule to docs".

## Core idea

TheDev demands:

- **TDD discipline without ceremony**: `--quick` for the inner loop,
  full gate before commit, 100% coverage. No exceptions, no
  `# pragma: no cover` without authorization.
- **Pipelined releases**: setup → plan → implement → review → finish.
  Each phase is a single command. Agents auto-commit, auto-push,
  auto-open PRs.
- **No GitHub UI for routine work**: PR creation, fix-plan iteration,
  release tagging — all happen from the terminal.
- **Hostile to gold-plating**: a feature that needs three meetings to
  justify is a feature that shouldn't exist.

## Characteristic interaction path

```text
Idea → /setup-task → worktree on QS_<N>
→ /create-plan → story + 4-reviewer adversarial review
→ /implement-task or /implement-setup-task → TDD loop, quality gate, PR
→ /review-task → 4-reviewer adversarial review of the PR
→ /finish-task → merge, cleanup, return to main
```

## Common mistakes

- Adding a workflow step that requires TheDev to fill in a template
  twice. They will route around it.
- Writing a doc that paraphrases another doc — duplication is friction.
  The drift checker (`scripts/qs/check_doc_drift.py`) plus the
  `covers:` frontmatter is the structural defence.
- Building a feature that requires manual git ops. TheDev's pipeline
  is the contract; if it bypasses the pipeline, it doesn't ship.

## See also

- [the-admin.md](the-admin.md) — the user TheDev is ultimately
  serving.
- [../../workflow/overview.md](../../workflow/overview.md) — the
  static-agent pipeline TheDev runs against.
- [../../workflow/project-rules.md](../../workflow/project-rules.md)
  — the rules TheDev's code must follow.
