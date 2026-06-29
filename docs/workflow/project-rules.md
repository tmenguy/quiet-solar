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
# Impacted-tests inner loop (QS-276) — the implement-phase default
# before commit/PR. Runs only the testmon-selected tests under
# --cov=<package>, then diff-cover --fail-under=100 on the CHANGED
# lines. Guarantees the lines YOU changed are 100% covered in ~seconds.
# QS-278: coverage ACCUMULATES across runs (--cov-append), so a no-op
# re-run (testmon selects 0 tests) or a single-file edit (small subset)
# still has every changed-vs-origin line covered — no spurious FAIL, and
# the run stays fast. The accumulated data is reset only on a fresh
# select-all baseline (missing/rebuilt .testmondata).
python scripts/qs/quality_gate.py --impacted

# Full quality gate (pytest 100% cov + ruff + mypy + translations).
# Authoritative whole-repo gate — enforced in CI on every PR; run
# locally on explicit request or when you suspect coverage lost in
# UNCHANGED code (which --impacted cannot see).
python scripts/qs/quality_gate.py

# Caching for repeated FULL-gate runs — skips gates when git state
# matches a previous pass on a clean tree.
python scripts/qs/quality_gate.py --cache

# Auto-fix formatting and lint.
python scripts/qs/quality_gate.py --fix

# JSON output for scripts.
python scripts/qs/quality_gate.py --json

# Fast iteration on one or more EXPLICIT test paths (files or dirs).
# Uses xdist + sysmon, skips coverage / ruff / mypy / translations.
# The canonical TDD red/green/refactor command while you iterate on a
# known test target; --impacted is the pre-commit gate that finds the
# impacted tests for you.
python scripts/qs/quality_gate.py --quick tests/test_solver.py
python scripts/qs/quality_gate.py --quick tests/ha_tests
python scripts/qs/quality_gate.py --quick tests/test_solver.py tests/test_constraints.py

# Refresh the testmon baseline (no coverage, no verdict). Sanctioned
# non-gate subcommand — used by finish-task after a merge.
python scripts/qs/quality_gate.py --seed-testmon

# Ad-hoc single-node pytest — debugging only.
source venv/bin/activate && pytest tests/test_solver.py::test_function_name -v
```

**Local-vs-CI coverage invariant (QS-276).** Local commits run
`--impacted` (the lines you changed are 100% covered); the **whole-repo
100% coverage** requirement is enforced in **CI on every PR** and is
what actually guarantees full coverage. The three iteration commands
relate as: `--impacted` is the pre-commit default (finds + runs the
impacted tests, checks changed-line coverage); `--quick` is for hammering
an explicit test path you already know; `--cache` accelerates repeated
*full*-gate runs. `--impacted` is mutually exclusive with
`--quick`/`--cache`/`--no-cache`/`--full`/`--fix`.

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
after a merge. A bare `pytest --testmon` remains forbidden.

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
(fallback — degraded one-shot UX kept for Claude Desktop). Do NOT ask
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

**Commit authorization**: agents are authorized to commit and push as
part of their defined workflow steps (e.g., the implement-task agent
auto-commits and opens a PR after the quality gate passes). Outside of
agent-driven phases, always ask the user before committing.

## Code rules reference

Before implementing code, read [project-context.md](project-context.md)
for the full 42-rule set covering naming, async, logging, error
handling, and testing patterns.
