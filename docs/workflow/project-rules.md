# Quiet Solar — Project Rules

## Process authority

All workflow rules, phase protocols, and code-style rules live under
`docs/workflow/`. Harness-specific config (`.claude/`, `.cursor/`,
`.opencode/`) references these docs; it never duplicates them.

## Project overview

Quiet Solar is a Home Assistant custom component that optimizes solar
energy self-consumption through a constraint-based solver.

## Commands

Activate `source venv/bin/activate` for all Python commands.
`scripts/qs/quality_gate.py` is the **single test entry point** —
it owns the cache, `pytest-xdist` parallelization,
`COVERAGE_CORE=sysmon`, and scope detection. Raw `pytest` bypasses
all four; use it only for ad-hoc single-node debugging.

```bash
# Impacted-tests inner loop (QS-276) — the implement-phase pre-commit
# gate. The implement phase ALWAYS runs it before commit/PR and never
# substitutes the full gate locally. Runs only the testmon-selected
# tests under --cov=<package>, then diff-cover --fail-under=100 on the
# CHANGED lines. Guarantees the lines YOU changed are 100% covered in
# ~seconds. QS-278: coverage ACCUMULATES across runs (--cov-append), so
# a no-op re-run (testmon selects 0 tests) or a single-file edit (small
# subset) still has every changed-vs-origin line covered — no spurious
# FAIL, and the run stays fast. The accumulated data is reset only on a
# fresh select-all baseline (missing/rebuilt .testmondata).
# QS-283: --impacted is self-healing. It reaps orphaned .coverage.*
# shards and stale .testmondata WAL/SHM sidecars on entry, and on a
# changed-line miss from an INCREMENTAL run it automatically rebuilds
# the baseline and re-checks once — recovering from a drifted baseline
# left by a killed run with NO manual file deletion ever required.
# QS-290: a change set with NO .py file at all EXITS EARLY — no pytest,
# no diff-cover, no `git fetch` — a handful of local git calls instead
# of 10-24 s of work.
# testmon fingerprints AST blocks in .py files only, so such a run could
# never select a test, and diff-cover has no Python lines to score: the
# verdict was ALREADY vacuous, the exit only stops paying for it. The
# change set is computed against the MERGE-BASE (committed + staged +
# unstaged + untracked), a superset of diff-cover's three-dot range, and
# any git failure fails CLOSED (no exit). See "non-Python change sets"
# below for what to run instead.
# QS-290: the `origin/main` fetch is TTL-cached (10 min) on FETCH_HEAD,
# so a TDD burst stops re-fetching. A stale base is OLDER, so the
# changed-line set is a superset — a TTL hit can only turn a PASS into a
# FAIL, never a FAIL into a PASS. The skip is deliberately conservative:
# it requires FETCH_HEAD to be non-empty (a FAILED fetch truncates it to
# 0 bytes while still bumping the mtime), to actually record a `main`
# fetch (a `gh pr checkout` or another-branch fetch rewrites it without
# advancing origin/main), and to carry a non-future mtime (clock skew).
# Any of those being off just means "fetch as usual". CI always bypasses
# the TTL entirely.
python scripts/qs/quality_gate.py --impacted

# Full quality gate (pytest 100% cov + ruff + mypy + translations).
# Authoritative full-suite gate — enforced in CI on every PR; the only
# local run is an EXPLICIT user request. Detecting coverage lost in
# UNCHANGED code is CI's exclusive job (--impacted cannot see it); never
# reach for the full gate locally as an inner-loop diagnostic.
python scripts/qs/quality_gate.py

# Caching for repeated FULL-gate runs — skips gates when git state
# matches a previous pass on a clean tree.
python scripts/qs/quality_gate.py --cache

# Auto-fix formatting and lint.
python scripts/qs/quality_gate.py --fix

# JSON output for scripts.
python scripts/qs/quality_gate.py --json

# Fast iteration on one or more EXPLICIT test paths (files or dirs).
# Uses sysmon, skips coverage / ruff / mypy / translations.
# QS-290: xdist is used only ABOVE a small-run threshold (50 collected
# tests). At or below it the run is single-process — xdist's per-worker
# collection re-imports the HA-heavy conftest and costs more than the
# parallelism saves (43 tests: serial 11.5 s vs xdist 16.8 s). The
# decision is announced on stderr. `QS_QG_PYTEST_WORKERS=auto` forces
# xdist back; an explicit value of that var always wins over the
# threshold, in both directions. Large targets (e.g. `tests/qs`, 680+
# tests) keep `-n auto` unchanged.
# The canonical TDD red/green/refactor command while you iterate on a
# known test target; --impacted is the pre-commit gate that finds the
# impacted tests for you.
python scripts/qs/quality_gate.py --quick tests/test_solver.py
python scripts/qs/quality_gate.py --quick tests/ha_tests
python scripts/qs/quality_gate.py --quick tests/test_solver.py tests/test_constraints.py

# Refresh the testmon baseline (no coverage, no verdict). Sanctioned
# non-gate subcommand — used by finish-task after a merge. QS-283: a
# true from-scratch rebuild — purges .testmondata (+WAL/SHM) and clears
# the accumulated coverage data first, so a reseed against an advanced
# baseline fully re-fingerprints (never a "0 changed" dead end).
python scripts/qs/quality_gate.py --seed-testmon

# QS-286: companion read-only query — report the detached --seed-testmon
# run's completion status (no pytest/coverage/testmon import). Exit codes:
# 0 = ok/safe to close, 4 = still running, 1 = rerun, 3 = no readable status.
python scripts/qs/quality_gate.py --seed-testmon-status

# QS-299: companion read-only FOLLOWER — stream the detached --seed-testmon
# run's progress inline (a `N/M tests (pct%)` line per poll) until it reaches
# a terminal state; used by finish-task after a merge. Requires --seed-token.
# Exit codes: 0 = ok/safe to close, 5 = superseded by a fresher refresh,
# 1 = rerun, 4 = still running after 45m, 3 = no readable status.
python scripts/qs/quality_gate.py --seed-testmon-follow --seed-token "$TOKEN"

# Ad-hoc single-node pytest — debugging only.
source venv/bin/activate && pytest tests/test_solver.py::test_function_name -v
```

**Local-vs-CI coverage invariant (QS-276).** Local commits run
`--impacted` (the lines you changed are 100% covered); the **full-suite
100% coverage of `custom_components/quiet_solar/`** requirement is
enforced in **CI on every PR** and is
what actually guarantees full coverage. The implement phase ALWAYS
runs `--impacted` before commit/PR and never substitutes the full gate
locally (the only local full-gate run is an explicit user request). For
change sets touching any `tests/qs`-pinned non-Python file (agent
files, commands, workflow docs, `.claude/settings.json`) — even when
Python files changed too — also run
`python scripts/qs/quality_gate.py --quick tests/qs` before commit
(testmon cannot see non-Python files). The
three iteration commands relate as: `--impacted` is the mandatory
pre-commit gate (finds + runs the impacted tests, checks changed-line
coverage, self-heals a drifted baseline); `--quick` is for hammering
an explicit test path you already know; `--cache` accelerates repeated
*full*-gate runs. `--impacted` is mutually exclusive with
`--quick`/`--cache`/`--no-cache`/`--full`/`--fix`.

**Non-Python change sets — `--quick` is the ONLY real check (QS-290).**
When a change set contains no `.py` file at all, `--impacted` exits
early and checks **nothing**: testmon fingerprints AST blocks in `.py`
files only, so it could never select a test, and diff-cover has no
Python lines to score. Its green is honest but empty — it was equally
empty before the early exit, which merely stopped paying 10–24 s for it.
So a `--quick` run is not a supplement there, it is **the**
verification, and it stays mandatory. Which target depends on what
changed — matching the hint the gate itself prints:

| Non-Python change | Run |
| --- | --- |
| agent files, commands, workflow docs, `.claude/settings.json` | `python scripts/qs/quality_gate.py --quick tests/qs` |
| `ui/*.j2` templates, `ui/resources/**` JS/CSS | `python scripts/qs/quality_gate.py --quick tests/test_dashboard_rendering.py` |

(A failed git probe does **not** take the early exit — it fails closed
and the full pipeline runs.)

*Cold baselines do not take the exit.* The exit requires a **warm**
`.testmondata`, because "testmon could never select a test" is only true
while it has a baseline: against a cold or purged one testmon
select-alls, and those over-selected tests can fail. Skipping them would
turn a real failure into a PASS, so a cold baseline falls through to the
full run instead. On a warm baseline the deferred re-sync is a pure cost
shift: the next `.py` run selects more tests than usual, once. One gap
remains (#341): after a package install/upgrade/removal or a Python
micro bump, testmon resets its environment fingerprint and would
select-all, but the baseline still looks warm — so a non-`.py` run can
be optimistic there until the next `.py` change set.

**Raw-`pytest` grammar rule.** Allowed: `pytest <path>::<nodeid> [-v]`
— the positional argument MUST contain `::`. Forbidden as a habitual
command: any `pytest` invocation whose positional argument lacks
`::`, e.g., `pytest tests/`, `pytest tests/ha_tests`,
`pytest tests/test_foo.py`. Use `--quick` on the enclosing file or
directory instead.

**Carve-out — `--seed-testmon` (QS-276).** The one sanctioned
whole-suite-ish `pytest --testmon` invocation is routed through
`quality_gate.py --seed-testmon`, never run as a raw `pytest`. It keeps
`quality_gate.py` the single pytest owner: it only refreshes
`.testmondata` (no coverage, no pass/fail verdict) and is invoked
detached/best-effort by `finish-task` to rebuild the main baseline
after a merge. A bare `pytest --testmon` remains forbidden. QS-286: its
companion `--seed-testmon-status` is the sanctioned **read-only** query
subcommand (no pytest/coverage/testmon import) — it reads the
`.testmondata.seed-status` marker the detached run writes and reports
whether it is safe to close the terminal. QS-299: a second companion,
`--seed-testmon-follow --seed-token <T>`, is the sanctioned **read-only**
streaming follower (also no pytest/coverage/testmon import) — it tails the
same marker plus the `.testmondata.seed.log` and prints one progress line
per poll until a terminal verdict, so finish-task can surface completion
inline in the same session; its exit code (0 ok / 5 superseded / 4 still
running / 1 rerun / 3 unreadable) is a **completion signal, not a gate**.
The detached seed accepts `--detached` (own process group) and
`--seed-token <T>` (per-run identity); a newer seed automatically preempts
an earlier still-running one (last-wins), and a stale baseline is always
safe (new worktrees just over-select tests).

**Carve-out — CI test workflows (QS-292).**
`.github/workflows/pr-quality.yml` and
`.github/workflows/release.yml` invoke `pytest` and `coverage`
directly, by design. The raw-`pytest` grammar rule and the
"`quality_gate.py` is the single test entry point" rule govern
*interactive and agent* commands, not CI. Where the PR suite is
sharded, the job providing the required status check on `main` ends in
a single authoritative fail-under-100 coverage verdict over the whole
of `custom_components/quiet_solar/` — spelled
`pytest --cov-fail-under=100` or `coverage report --fail-under=100`.

**UI-only fast path.** When only `custom_components/quiet_solar/ui/*.j2`
templates and `custom_components/quiet_solar/ui/resources/**` assets
change (optionally mixed with dev-only paths), the gate runs only
`tests/test_dashboard_rendering.py` plus any changed test files —
skipping ruff, mypy, translations, and full coverage. Use `--full` to
force the full suite.

## Architecture constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. `ha_model/` bridges both.
- **Solver step size**: `SOLVER_STEP_S = 900` in `const.py` — don't touch.
- **All config keys in `const.py`** — never hardcode strings.
- **Async rules**: no blocking calls in async code, use `hass.async_add_executor_job()`.
- **Logging**: lazy `%s`, no f-strings in log calls, no periods at end.
- **Translations**: NEVER edit `translations/en.json` — edit `strings.json`, run `bash scripts/generate-translations.sh`.

### Doc maintenance

The agent-facing documentation hierarchy lives under
[../agents/](../agents/) — short, addressable files anchored to source
via `covers:` frontmatter. The drift checker
`scripts/qs/check_doc_drift.py` validates that every `covers:` path
exists and flags docs whose source was modified without a
co-modification. The four orchestrator agents
([qs-create-plan](../../.claude/agents/qs-create-plan.md),
[qs-implement-task](../../.claude/agents/qs-implement-task.md),
[qs-implement-setup-task](../../.claude/agents/qs-implement-setup-task.md),
[qs-review-task](../../.claude/agents/qs-review-task.md)) wire the
checker into their phase protocol. Taxonomy: **concept** (one source
file), **principle** (cross-cutting rule), **use-case** (end-to-end
scenario), **persona** (user archetype).

### Harness sync

Agent files live in three harness directories: `.claude/agents/`,
`.cursor/agents/`, `.opencode/agents/`. Each agent's core protocol
(TDD steps, quality gate, hard rules) must stay aligned across all
three directories. The YAML frontmatter (between the `---`
delimiters) and harness-specific sections (session-spawn logic,
handoff commands) legitimately differ — Claude uses
`claude --agent`, OpenCode uses `spawn_session.py`, Cursor uses the
in-session agent picker.

The drift checker `scripts/qs/check_doc_drift.py` enforces
**co-modification**: when any `.<harness>/agents/*.md` file appears
in the modified set, it verifies that the corresponding files in the
other two harness directories were also modified. Violation exits 1.

**When editing agent files:** always edit all three copies. The
canonical workflow is to make the functional change in all three
harnesses, adapting harness-specific sections (handoff, session
spawn) as needed for each.

## Workflow routing

Each phase runs as an interactive `claude --agent qs-<phase>` session
(preferred — open a fresh terminal) or as a `/<phase>` slash command
(fallback — degraded one-shot UX; the GUI can instead run the phase
agent directly, see [harness.md](harness.md)). Do NOT ask
which phase to use — infer from context.

| You say                                                      | Preferred launcher                       | Fallback           |
| ------------------------------------------------------------ | ---------------------------------------- | ------------------ |
| "Setup task 3.2" / describe feature / "work on issue #42"    | `claude --agent qs-setup-task` on main   | `/setup-task`      |
| "Create plan" (inside worktree)                              | `claude --agent qs-create-plan`          | `/create-plan`     |
| "Implement task" (inside worktree)                           | `claude --agent qs-implement-task`       | `/implement-task`  |
| "Review PR #5" or "review task"                              | `claude --agent qs-review-task`          | `/review-task`     |
| "Merge PR #5" or "finish task"                               | `claude --agent qs-finish-task`          | `/finish-task`     |
| "Create a release"                                           | `claude --agent qs-release` on main      | `/release`         |
| Bug fix / small fix                                          | `claude --agent qs-setup-task` on main   | `/setup-task`      |

See [overview.md](overview.md) section "Orchestrators are interactive
sessions; sub-agents are parallel fan-out" for the rationale.

Each command delegates to a static agent under `.claude/agents/` (or
`.cursor/agents/`). Agents discover task context at runtime from
`git branch --show-current` — there is no per-task agent rendering.

### Lanes & axes (QS-332)

Every task is born in exactly one of **6 lanes** — {bug, feature, epic}
× {product, factory} — declared at setup as GitHub labels: exactly one
`target:*` (`product`/`factory`), plus `scale:task` with exactly one
`kind:*` (`bug`/`feature`) for tasks, or `scale:epic` with **no kind**
for epics. `scale:task` is the implicit CLI default but is always
applied explicitly as a label. The lane protocol files live in
`docs/workflow/lanes/<lane>.md`; `python scripts/qs/context.py` exposes
`labels`, `kind`, `target`, `scale`, `lane`, `parent_epic`. The domain
module is `scripts/qs/targets.py` (path classification, declaration
truth table, parent-epic parsing) — one machine-readable definition,
consumed by `fetch_issue.py`, `setup_task.py`, and the quality gate.

The **lane check** runs in `--impacted`, the full gate, and the CI
`--lane-check` job: a **missing declaration fails**; a **cross-target
diff warns loudly but never fails** — purpose, not path, is the
classifier (a product-declared fix that must also enhance a test tool
stays a product task). The warning lists the crossing files and always
prints the split recommendation; splitting is at human discretion.
`create_pr.py` auto-appends `Refs #<epic>` toward a declared parent
epic and machine-injects a `## Lane note` when the diff crosses the
declared target.

**Commit authorization**: agents are authorized to commit and push as
part of their defined workflow steps (e.g., the implement-task agent
auto-commits and opens a PR after the quality gate passes). Outside of
agent-driven phases, always ask the user before committing.

## Code rules reference

Before implementing code, read [project-context.md](project-context.md)
for the full 42-rule set covering naming, async, logging, error
handling, and testing patterns.
