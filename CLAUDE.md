# CLAUDE.md

Read `_qsprocess/rules/project-rules.md` before doing any work.

Use these skills for development workflows:
- `/setup-task` — create GitHub issue + feature branch + worktree (never touches main checkout)
- `/create-plan` — create story artifact in worktree, commit and push
- `/implement-story` — TDD implementation with enforced quality gates
- `/review-story` — code review + CodeRabbit + process feedback
- `/finish-story` — merge PR, cleanup worktree, update epics
- `/release` — tag and ship

Quality gates: `python scripts/qs/quality_gate.py` (runs pytest 100% cov + ruff + mypy + translations).
Code rules: `_qsprocess_opencode/project-context.md`
