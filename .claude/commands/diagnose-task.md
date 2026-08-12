---
description: Drive the interactive bug-diagnosis loop (diagnose / review / finalize) for the bug × product lane — evidence before hypotheses, root cause before fix plan, red-test spec — then commit the diagnosis story and route to the fix.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-diagnose-task` (interactive session — you can answer
> the persona's evidence-gathering questions in its DIAGNOSE loop).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-diagnose-task** subagent to handle this. The subagent will
discover the current task context from the branch name (`QS_<N>`) and
the GitHub issue, and read its lane file
(`docs/workflow/lanes/bug-product.md`).

Note: the diagnose loop (DIAGNOSE / REVIEW / TRIAGE / FINALIZE) is built
for the interactive launcher path above. In this one-shot fallback the
persona still persists the diagnosis story and can run a review, but the
open-ended evidence gathering — asking you for production data, pushing
back on a hypothesis — is exactly the UX this fallback can't offer.

Expected outcome:
- Diagnosis story written to `docs/stories/QS-<N>.story.md` as the
  diagnosis converges (root cause in one sentence with file/function
  references, fix plan, red-test spec), readable in the editor before
  being committed.
- Adversarial diagnosis review available on demand: round 1 runs
  `qs-diag-root-cause-skeptic` + `qs-diag-fix-minimalist` in parallel;
  round 2+ adds `qs-plan-delta-auditor`.
- At FINALIZE the story is committed; the handoff routes to one of three
  exits (fix → `/implement-task`; iceberg close-as-superseded or
  no-defect → `/finish-task`).

User request:
$ARGUMENTS
