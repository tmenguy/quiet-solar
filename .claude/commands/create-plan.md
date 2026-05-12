---
description: Draft the story artifact with acceptance criteria, run adversarial review with 4 parallel sub-agents, commit and push.
---

Use the **qs-create-plan** subagent to handle this. The subagent will
discover the current task context from the branch name (`QS_<N>`) and
the GitHub issue.

Expected outcome:
- Story written to `docs/stories/QS-<N>.story.md` with
  acceptance criteria (Given/When/Then) and a task breakdown.
- 4-reviewer adversarial review run in parallel; findings triaged
  interactively.
- Story committed and pushed.
- Next-phase command printed (`/implement-task` or
  `/implement-setup-task`).

User request:
$ARGUMENTS
