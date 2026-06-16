---
description: Drive the interactive plan-mode loop (discuss / review / finalize), persisting the story file as it converges and running adversarial review on demand, then commit and push.
---

> **Preferred entry**: open a fresh terminal in the worktree and run
> `claude --agent qs-create-plan` (interactive session — you can answer
> the persona's clarifying questions in step 2 of its protocol).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-create-plan** subagent to handle this. The subagent will
discover the current task context from the branch name (`QS_<N>`) and
the GitHub issue.

Note: the plan-mode loop (DISCUSS / REVIEW / TRIAGE / FINALIZE) is built
for the interactive launcher path above. In this one-shot fallback the
persona still persists the story and can run a review, but the
open-ended back-and-forth — pushing back on the draft, asking for
another review round — is exactly the UX this fallback can't offer.

Expected outcome:
- Story written to `docs/stories/QS-<N>.story.md` as the discussion
  converges (acceptance criteria + task breakdown), readable in the
  editor before being committed.
- Adversarial review available on demand: round 1 runs the 4 global
  plan reviewers in parallel; round 2+ adds `qs-plan-delta-auditor`
  fed an in-context diff. Findings are triaged interactively and folded
  back into the story.
- At FINALIZE (advisory gate, never hard-blocked) the story is committed
  and pushed.
- Next-phase command printed: launcher form (`claude --agent
  qs-implement-task` or `claude --agent qs-implement-setup-task`)
  plus slash-command fallback (`/implement-task` or
  `/implement-setup-task`).

User request:
$ARGUMENTS
