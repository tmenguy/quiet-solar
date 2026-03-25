# CLAUDE.md

Read `_qsprocess/rules/project-rules.md` before doing any work.

Use these skills for development workflows:
- `/create-story` — create story artifact + GitHub issue + feature branch
- `/setup-story` — set up worktree, output launch command for implementation
- `/implement-story` — TDD implementation with enforced quality gates
- `/review-story` — code review + Copilot + process feedback
- `/finish-story` — merge PR, cleanup worktree, update epics
- `/release` — tag and ship

Quality gates: `python scripts/qs/quality_gate.py` (runs pytest 100% cov + ruff + mypy + translations).
Code rules: `_bmad-output/project-context.md`
