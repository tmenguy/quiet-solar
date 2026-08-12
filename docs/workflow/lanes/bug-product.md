# Phase protocols — bug × product lane

This lane **diverges** from
[phase-protocols.md](../phase-protocols.md): fixing a bug is not
building a feature. The flow is **diagnose-first**:

`setup → diagnose → fix (implement) → verify → finish`

`create-plan` and `review-task` do **not** run in this lane. They are
replaced by `diagnose-task` (agent `qs-diagnose-task`) and `verify-task`
(agent `qs-verify-task`). Every other phase follows the shared contract;
only the divergent details are spelled out below. Each phase has a
static agent under `.claude/agents/` (mirrored in `.cursor/agents/` and
`.opencode/agents/`); agents discover task context at runtime via
`python scripts/qs/context.py`.

Each phase is invoked **as an interactive session** via
`claude --agent qs-<phase>` (the launcher form — preferred). The
slash-command form (`/<phase>`) is kept as a **degraded fallback**. See
[overview.md](../overview.md) for the rationale.

---

## `setup-task` (agent: `qs-setup-task`)

**Runs on**: main checkout. Unchanged from the shared contract except
routing: **next phase is `diagnose-task`**, not `create-plan`.
**Side effects**: creates GitHub issue, branch `QS_<N>`, worktree.
**Next phase**: `claude --agent qs-diagnose-task` in the worktree
(preferred), or `/diagnose-task` as fallback.

**Hard rules**: do NOT analyze or interpret user input — pass it
through to the issue verbatim; diagnosis is `diagnose-task`'s job.

---

## `diagnose-task` (agent: `qs-diagnose-task`)

**Runs on**: worktree.
**Inputs**: branch `QS_<N>` (issue resolves from there).
**Side effects**: writes the diagnosis story to
`docs/stories/QS-<N>.story.md` (written as soon as the first diagnosis
round converges, **committed only at FINALIZE**). May create a
superseding issue via `gh issue create` (iceberg), and at FINALIZE may
close this issue via `gh issue close --comment`.
**Output**: a diagnosis story — root cause + fix plan + red-test spec.
**Next phase**: `implement-task` (normal fix) or `finish-task` (iceberg
close-as-superseded / no-defect) — resolved at FINALIZE.

A **diagnose-first** mode loop (DIAGNOSE / REVIEW / TRIAGE / FINALIZE):
evidence before hypotheses, hypotheses before cause, cause before plan.

### DIAGNOSE (default)

- **Evidence gathering** — ask the user for production data before
  theorising. Pick from the checklists below per bug; they are not a
  rigid form.
- **Root-cause analysis** — read the code, form hypotheses, confirm or
  eliminate each against the evidence. **Hard rule: no fix plan until
  the root cause is stated in one sentence with file/function
  references.** Insufficient evidence → ask, don't guess; staying in
  DIAGNOSE across sessions is normal. A fresh session that finds the
  story file already existing and carrying the bug-template sections
  reads it first and adopts it as the current diagnosis state
  (resume, don't restart).
- **Reproduction** — **demonstrate when feasible**: the agent may run
  throwaway, uncommitted scripts/snippets via Bash to show the
  hypothesis live (nothing is committed in this phase). Always produce
  the **red test spec**: exact test file, fixture data derived from the
  evidence, the assertion that fails today. **Sanctioned fallback**
  when a unit test cannot reproduce the bug (timing, hardware,
  cloud-API dependent): the story states *why* and names the
  alternative proof, and carries the mandatory acceptance line —
  `Fallback accepted: <reason>` — recorded when the human accepts it
  in-session.
- **Fix plan** — short, produced *by* the diagnosis. **Minimum-diff
  rule**: the fix plan lists the files it expects to touch; a
  blast-radius statement is mandatory. Amendment path: implement may
  extend the list with a reasoned progress note in the story; verify
  flags **unexplained** excess as must-fix.
- **Iceberg check** — is the root cause local, or the tip of something
  generic? If iceberg → create a new issue labelled
  `kind:feature,target:product,scale:task` (or, for an epic,
  `target:product,scale:epic` and **no kind**) via `gh issue create`,
  carrying the full diagnosis
  and a back-link; then a **per-case human decision**: (a) close this
  bug as superseded, or (b) ship a minimal containment fix here and
  link.

#### Generic evidence checklist

- exact HA + integration versions
- timeline; expected vs observed; frequency / determinism
- recent changes (upgrades, config edits, HA core bumps)
- debug-level logs around the incident window

#### Quiet-solar evidence checklist

- config-entry options / device setup
- entity histories for charger / car / solar / grid sensors
- `custom_components.quiet_solar` debug log capture
- solver inputs around the incident window

### Bug story template

The diagnosis story carries these sections:
**Symptom** · **Evidence** · **Root cause** · **Repro strategy** ·
**Fix plan** · **Iceberg check** · **Acceptance** (the red test(s), or
the fallback proof + the `Fallback accepted:` acceptance line).

### REVIEW / TRIAGE (invoked, on demand)

Same loop mechanics as create-plan, with the **bug diagnosis roster**:
round 1 = `qs-diag-root-cause-skeptic` + `qs-diag-fix-minimalist` in
parallel; round 2+ adds `qs-plan-delta-auditor`. Review stays
**on-demand** (user-invoked, offered once per stable version). Findings
use the `critical/redesign/improve/clarify` categories and the same
finding-state triage model as create-plan.

### FINALIZE (three exits, all human-confirmed)

Commit the story (skip the commit if no story file was written —
early exits 2/3 — or if the story is already committed and
unchanged; on exit 1 an unwritten story is **written now first** — a
confirmed fix exit implies the diagnosis converged, and the fix loop
needs the story on disk), then route via one of **three exits**:

1. **fix** (normal) → `implement-task`.
2. **close-as-superseded** (iceberg): the diagnose agent runs
   `gh issue close --comment` linking the superseding issue, then →
   `finish-task`. The superseding issue's body is the durable record
   of the diagnosis.
3. **no-defect / cannot-diagnose** (works-as-intended, duplicate, or
   evidence exhausted): the human decides close (agent closes with the
   rationale) or leave open awaiting evidence. **Either way the
   diagnosis-so-far is posted as an issue comment before cleanup** →
   `finish-task`.

On exits 2/3 with an open fix PR (`pr_number` non-null from
`context.py` — a fix loop already ran and this diagnosis abandons that
fix), the diagnose agent closes the PR first
(`gh pr close <pr_number> --comment "superseded — see issue"`) so
finish-task lands on its CLOSED-unmerged cleanup branch instead of
offering to merge the superseded fix. Only when no fix PR exists is
the handoff the Case A no-PR cleanup.

---

## `implement-task` (agent: `qs-implement-task`)

**Runs on**: worktree. Fixes the diagnosed bug under
`custom_components/quiet_solar/` and `tests/` (the regression test the
red-test protocol writes). Follows the shared implement contract
with the **red-test protocol** below.

### Red-test protocol

1. Write **each** spec'd regression test **first**, run each red with
   the sanctioned `::`-form (`pytest <file>::<test> -v`), and **record
   the failure output** (story progress note + PR body) — each must
   fail for the diagnosed reason, not an import/collection error.
2. Apply the minimal fix; re-run the test green.
3. **Scope constraint**: deliver the fix at the scope diagnosed — no
   drive-by refactors, no opportunistic cleanups. Anything bigger goes
   through the iceberg escalation, never into the bug PR. File-list
   amendments carry a reasoned progress note (minimum-diff rule).
4. **Fallback path**: with an accepted `Fallback accepted:` line in the
   story, implement per plan and document the alternative evidence in
   the PR body.
5. Run the impacted gate before commit; next-phase handoff routes to
   **`verify-task`**.

**Next phase**: `claude --agent qs-verify-task` (preferred), or
`/verify-task` as fallback.

---

## `verify-task` (agent: `qs-verify-task`)

**Runs on**: worktree.
**Inputs**: PR number (resolved from branch).
**Side effects**: writes fix-plan files under
`docs/stories/QS-<N>.story_review_fix_#NN.md` (if fixes are needed).
**Output**: triaged findings; either "ready for finish-task" or a fix
plan. `qs-review-task` does **not** run in this lane.
**Next phase**: `finish-task` (clean), or `implement-task` then re-run
`verify-task` (fixes).

Orchestrator — spawns the **fix-verification roster** in parallel (one
message): `qs-review-edge-case-hunter` (regression is the dominant
risk) + `qs-review-coderabbit` + `qs-review-regression-proof`. Consolidate
into must-fix / should-fix / nice-to-have / invalid, drive interactive
triage, and either fast-path to `finish-task` (zero findings) or write a
fix plan and loop through `implement-task`.

`qs-review-blind-hunter` and `qs-review-acceptance-auditor` do **not**
run in this lane — for a bug, acceptance *is* the red test, which
`qs-review-regression-proof` audits.

---

## `finish-task` (agent: `qs-finish-task`)

Unchanged from the shared contract. Bug × product closes on merge; the
diagnose-task iceberg / no-defect exits land as pure cleanup — Case A
when no fix PR was ever opened, the CLOSED-unmerged cleanup branch when
diagnose-task closed a superseded fix PR (either way finish-task does
not touch the issue — the diagnose agent already did).

---

## `release` (agent: `qs-release`)

Unchanged from the shared contract. Runs on the main checkout,
independent of any task.

---

## Adversarial review

See [adversarial-review.md](../adversarial-review.md). This lane runs
**dedicated minimal rosters**: 2 diagnosis reviewers
(`qs-diag-root-cause-skeptic`, `qs-diag-fix-minimalist`; round 2+ adds
`qs-plan-delta-auditor`) at diagnose time, and 3 fix-verification
reviewers (`qs-review-edge-case-hunter`, `qs-review-coderabbit`,
`qs-review-regression-proof`) at verify time.
