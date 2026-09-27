---
description: Drive the interactive epic decompose loop (discuss / review / finalize) on docs/epics/QS-<N>.md, then file the children whose turn it is, land the document on main and sync the epic issue.
---

> **Preferred entry**: open a fresh terminal in the epic's worktree and
> run `claude --agent qs-decompose-epic` (interactive session — you can
> discuss the decomposition, push back on the draft and ask for another
> review round).
>
> **This slash command is the degraded fallback** — kept for Claude
> Desktop and any chat without a CLI launcher. It spawns a one-shot
> non-interactive `Agent`-tool sub-process; the persona runs to
> completion and returns a final summary, and you cannot interject. This
> is the broken-by-design UX that QS-175 mitigates — we keep the slash
> command **only as a fallback**, not as the primary flow.

Use the **qs-decompose-epic** subagent to handle this. The subagent will
discover the current task context from the branch name (`QS_<N>`) and
the GitHub issue, and refuses unless the issue is `scale:epic` ×
`target:factory`.

Note: the decompose loop (DISCUSS / REVIEW / TRIAGE / FINALIZE) is built
for the interactive launcher path above. In this one-shot fallback the
persona can still write the document and run a review, but the
open-ended back-and-forth is exactly the UX this fallback can't offer.

Expected outcome:
- Epic document written to `docs/epics/QS-<N>.md` as the discussion
  converges, readable in the editor before it lands.
- Adversarial review on demand: the 4 global plan reviewers in
  parallel; round 2+ adds `qs-plan-delta-auditor`.
- At FINALIZE: the children whose turn it is filed, the document landed
  on `main` by `python scripts/qs/epic_doc.py land`, the epic issue
  synced by `python scripts/qs/epic_doc.py sync-issue` (it stays open).
- Next-phase command printed: launcher form (`claude --agent
  qs-finish-task`) plus slash-command fallback (`/finish-task`).

User request:
$ARGUMENTS
