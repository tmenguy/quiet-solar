---
description: Run adversarial review on the PR with 4 parallel reviewer sub-agents, drive interactive triage, generate a fix plan if needed.
---

Use the **qs-review-task** subagent to handle this. The subagent
discovers PR + story file from the branch name.

Expected outcome:
- 4 reviewer subagents spawned in parallel (blind-hunter,
  edge-case-hunter, acceptance-auditor, coderabbit).
- Findings consolidated and triaged interactively with the user.
- If findings remain, a fix plan written to
  `docs/stories/QS-<N>.story_review_fix_#NN.md` with
  a ready-to-copy `/implement-task` prompt.
- If clean, next-phase command printed (`/finish-task`).

User request:
$ARGUMENTS
