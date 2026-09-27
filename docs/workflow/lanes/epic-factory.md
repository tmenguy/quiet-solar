# Phase protocols — epic × factory lane

This lane **diverges** from
[phase-protocols.md](../phase-protocols.md): an epic is an umbrella
over several tasks, not a task. It has **no implement phase and no
PR**. Its output is a rationale document on `main`
(`docs/epics/QS-<N>.md`) plus the **child issues** whose turn has come.
The flow is:

`setup → decompose → finish`

`setup-task` cuts a **short-lived docs-only worktree** `QS_<N>`,
`decompose-epic` (agent `qs-decompose-epic`) drafts and reviews the
document, files the children, lands the document on `main` and syncs
the epic issue, and `finish-task` only discards the worktree. The epic
issue stays **open** for the epic's lifetime (`scale:epic` is exempt
from the stale bot) — nothing in this lane ever closes it.
`create-plan`, `implement-*` and `review-task` do **not** run in this
lane; each child is its own task, in its own lane.

The same flow is **re-entered** to file the next wave of children or to
amend the document (see "Re-entry" below).

Each phase is invoked **as an interactive session** via
`claude --agent qs-<phase>` (the launcher form — preferred). The
slash-command form (`/<phase>`) is kept as a **degraded fallback**. See
[overview.md](../overview.md) for the rationale.

---

## `setup-task` (agent: `qs-setup-task`)

**Runs on**: main checkout. Unchanged from the shared contract except
routing: **next phase is `decompose-epic`**, and the worktree it cuts is
a short-lived docs-only one (no PR is ever opened from it).
**Side effects**: creates (or reuses) the epic issue, branch `QS_<N>`,
worktree. `setup_task.py` allows `scale:epic` only for
`--next-cmd decompose-epic` with `target:factory` and a worktree
(`--no-worktree` is refused).
**Next phase**: `claude --agent qs-decompose-epic` in the worktree
(preferred), or `/decompose-epic` as fallback.

**Hard rules**: do NOT analyze or interpret user input — pass it
through to the issue verbatim; decomposition is `decompose-epic`'s job.

---

## `decompose-epic` (agent: `qs-decompose-epic`)

**Runs on**: the epic's worktree.
**Inputs**: branch `QS_<N>` (the epic issue resolves from there).
**Side effects**: writes `docs/epics/QS-<N>.md` (the **only** file it
writes — no story file); at FINALIZE, files child issues, lands the
document on `main` by one direct commit, and edits the epic issue body.
**Next phase**: `claude --agent qs-finish-task` (preferred), or
`/finish-task` as fallback — only after `epic_doc.py land` succeeded.

### Mode selection

At session start run
`python scripts/qs/epic_doc.py status --issue <N> --sync` (it fetches
`origin/main` first, and fast-forwards a reused worktree that is behind
and has nothing local):

| local doc | on `origin/main` | mode |
|---|---|---|
| absent | absent | **DECOMPOSE** — new draft |
| present | absent | **DECOMPOSE** — continue the draft left by an unfinished session; never overwrite it blindly |
| — | present | **RESUME** — continue from the local version when `local_modified` (show `diff`), else from the `origin/main` version |

### DECOMPOSE — the mode loop

The same loop as `create-plan` (DISCUSS / REVIEW / TRIAGE / FINALIZE,
the same banner and the same three intents), on the epic document:

- **DISCUSS** — read the epic issue, glob the relevant areas, discuss
  goal, evidence, design, decisions and the decomposition (each child:
  title, `kind`, `wave`, one-paragraph scope). Write the document from
  the template below as soon as the first round converges; overwrite it
  on every change; it stays uncommitted until FINALIZE. Maintainer
  decisions go into its **Decisions** section, not only into chat.
- **REVIEW** — round 1 the 4 global plan reviewers (`qs-plan-critic`,
  `qs-plan-concrete-planner`, `qs-plan-dev-proxy`,
  `qs-plan-scope-guardian`) on the document, in parallel; round 2+ plus
  `qs-plan-delta-auditor` with the in-context diff. The scope-guardian
  gets the epic issue body.
- **TRIAGE** — the `open` / `resolved` / `rejected` finding states,
  recorded in the document's **Adversarial review record**.
- **FINALIZE** — the create-plan advisory gate (the document changed
  since the last review, or `critical` findings are still `open` → ask
  "ship anyway?"; never hard-block), then the procedure below.

### RESUME

Show the decomposition table (filed vs. `not filed`) and ask what to do:
file the next child(ren), amend the document (a new decision, a
re-plan), or both — then FINALIZE. Review in RESUME runs **only on the
explicit REVIEW intent**; it is never automatic.

### FINALIZE procedure

1. **Preflight**: `python scripts/qs/epic_doc.py land --issue <N>
   --message "…" --dry-run` must pass before any issue is created.
2. **File the children whose turn is now** —
   **File children just in time**: children are not exempt from the
   stale bot, so a child is filed when its turn comes and the rest stay
   `not filed` rows. Default:
   every `not filed` row of the lowest `wave` that still has one; with
   no `wave` column (legacy documents) the maintainer names the rows.
   **Only rows marked `not filed` are filed**, and the document is
   rewritten with the new number **after each issue is created**. Before
   creating one, search for an existing issue with that exact title
   (`gh issue list --state all --search "\"<title>\" in:title"`) whose
   body declares `### Parent epic` `#<N>`, and adopt it instead (covers a
   crash between create and rewrite). Each child is created with
   `python scripts/qs/create_issue.py --title … --body … --labels
   "kind:<k>,target:factory,scale:task[,<the epic's area:* labels>]"` —
   the target is always inherited from the epic, never set per child —
   and a body in the child form of the conventions below.
3. **Land the document**: `python scripts/qs/epic_doc.py land --issue
   <N> --message "…"` — one direct commit on `main`, landing **only the
   epic's own `docs/epics/QS-<N>.md`**; any other changed path is refused
   (`out-of-scope`).
4. **Sync the epic issue**: `python scripts/qs/epic_doc.py sync-issue
   --issue <N>`; its JSON must report `state: OPEN`.
5. **Hand off** to `finish-task`, which discards the worktree.

If step 3 fails after step 2 filed children, those children link a
document that is not on `main` yet: stop and report. Re-running
FINALIZE is safe — filed rows are skipped and `land` is idempotent.
Last-resort fallback, documented and never automated: a hand-opened PR
carrying `Refs #<N>` — never through `create_pr.py`, whose
`Fixes #<N>` would close the epic.

### The landing rule

An epic document reaches `main` by **direct commit, landing only the
epic's own `docs/epics/QS-<N>.md`**, through `epic_doc.py land` only —
never through `create_pr.py`. `land` refuses a non-epic issue, any other
changed path (`out-of-scope`), a stale `docs/agents/` doc
(`check_doc_drift.py`), and a
path that `main` changed since the worktree's base (the `conflict`
refusal carries the current `origin/main` content; merge it into the
local file, then re-run with `--merged <path>=<blob>`). The working
tree is never modified before the push is verified, so a failed run
never loses the draft. An amendment to an existing epic document
follows the same rule, or rides a child's own PR when that child changes
the design. See [project-rules.md](../project-rules.md) → "Epic
documents".

### Epic doc template

```markdown
# Epic QS-<N> — <title>

> **Status: <decisions recorded / implementation in progress / …>.**
> `docs/workflow/` remains authoritative for current agent behaviour
> until the children below land.

| | |
|---|---|
| **Issue** | [#<N>](<issue url>) — stays **open** for the epic's lifetime (`scale:epic` is exempt from the stale bot) |
| **Classification** | `scale:epic` × `target:factory` — no `kind` of its own; children carry their own kinds |
| **Branch / worktree / PR** | no PR; a short-lived docs-only worktree, discarded after this document reaches `main` |
| **Live state** | issue #<N>: filed children as `- [ ] #NNN` task-list lines |
| **This document** | the durable rationale: goal, evidence, design, decisions, decomposition |
| **Builds on** | <related epics or documents, or "—"> |

---

## Goal

## Evidence

## Design

## Decisions

<!-- maintainer decisions taken during decomposition, dated -->

## Decomposition

| # | child | kind | wave | issue |
|---|---|---|---|---|
| 1 | <child title> | feature | 1 | not filed |

## Adversarial review record

<!-- per round: reviewers, findings and their open / resolved / rejected state -->

## Doc maintenance
```

The **Decomposition** table is what `epic_doc.py sync-issue` parses:
every table under `## Decomposition` (its `###` subsections included)
whose header has `child` and `issue` columns (`#`, `kind`, `wave`
optional). An issue cell is `#123`, `[#123](url)` or `not filed`;
anything else makes `sync-issue` refuse and write nothing.

### Issue-body conventions

- **Epic issue**: the first line links the rationale document
  (`**Rationale document:** [docs/epics/QS-<N>.md](…/blob/main/docs/epics/QS-<N>.md)`);
  filed children are `- [ ] #N — <child>` task-list lines; unfiled
  children are `- (not filed) <child>` lines under `## Children`.
  `epic_doc.py sync-issue` is the only writer: it adds, never deletes or
  rewrites a line it does not own (ticks and trailing text are kept).
- **Child issue**, in order: a link to `docs/epics/QS-<N>.md` on `main`
  as the first line; the child's scope; a `### Parent epic` section
  whose next line is `#<N>`.
- **No other `Refs #…` line anywhere in a child's body.** When the
  `### Parent epic` section is absent, `targets.parse_parent_epic` falls
  back to the **first** `Refs #N` in the body and reads it as the parent
  epic — that is how #370 was wrongly attached to #369.

### Re-entry

To file the next wave, or to amend the document: `setup-task --issue
<N>` (reuses the worktree if one survived) → `decompose-epic` (RESUME)
→ `finish-task`.

**Hard rules**:
- Refuse unless the task is `scale:epic` × `target:factory`.
- The only file written is `docs/epics/QS-<N>.md`.
- Always offer, and recommend, at least one REVIEW round before the first
  land; the FINALIZE gate is advisory (never a hard block).
- Sub-agents are spawned in parallel (one message).
- Hand off to `finish-task` only after `epic_doc.py land` succeeded.

---

## `finish-task` (agent: `qs-finish-task`)

**Runs on**: the epic's worktree. An epic never has a PR, so Case A
(no PR) is always the path. `python scripts/qs/epic_doc.py status
--issue <N>` decides, branching on the probe `status`: if it is **not
`ok`** (offline `git-error`, `undecodable`, `lookup-failed`) → STOP,
show the JSON, and never offer force-delete without a successful probe;
`status: ok` with `safe_to_discard: true` → the landed document is not
"unpushed work", clean up; `status: ok` with `safe_to_discard: false` →
show what would be lost — the `changed` paths and `unpushed_commits`
count the probe reports (notably an unlanded `docs/epics/QS-<N>.md`, but
also any other uncommitted edit or local commit) — and ask. Cleanup runs
`cleanup_worktree.py … --force --delete-branch` so a
re-entry starts fresh from `origin/main`; the remote `QS_<N>` branch is
deleted as for any task. Branch on the cleanup JSON `status`:
`removed-branch-kept` or `branch-checked-out-elsewhere` mean the local
branch was **kept** (read `branch_delete_error`) — report that honestly,
never claim it was removed.
**Output**: "epic session closed — doc on `main`, issue #N stays open".
**Next phase**: terminal until the next re-entry.

**Hard rules**:
- Never close the epic issue.
- Refuse to delete `main` / `master`.

---

## `release` (agent: `qs-release`)

Not part of this lane — an epic ships no code.
