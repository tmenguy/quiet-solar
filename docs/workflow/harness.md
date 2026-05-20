# Harness abstraction

The pipeline runs across four harnesses with different mechanics:

| Harness         | Agent dir          | Slash commands     | Session spawn                 | Tool allowlist        |
| --------------- | ------------------ | ------------------ | ----------------------------- | --------------------- |
| Claude Code     | `.claude/agents/`  | `.claude/commands/`| `claude` CLI on worktree      | `tools:` frontmatter  |
| Cursor 2.4+     | `.cursor/agents/`  | `/<name>`          | New Cursor workspace          | `readonly:` boolean   |
| OpenCode        | `.opencode/agents/`| UI agent picker    | HTTP API: POST /session + POST /session/<id>/prompt_async (no reload) | `permission:` block   |
| Codex (future)  | `.codex/agents/`   | TBD                | TBD                           | TBD                   |

The agent **bodies** are identical across harnesses (same protocol, same
hard rules). Only the **frontmatter** differs and the **handoff
mechanics** are isolated in Python.

## Detection — `scripts/qs/harness.py`

`harness.detect()` returns one of `claude-code` / `cursor` / `opencode` /
`codex` / `unknown`. Order of resolution:

1. `QS_HARNESS` env var (explicit override).
2. `CLAUDECODE=1` → `claude-code`.
3. `OPENCODE_SERVER_PORT` set → `opencode`.
4. `CURSOR_TRACE_ID` set → `cursor`.
5. `CODEX_AGENT_*` env vars set → `codex`.
6. Default: `claude-code`.

## Launcher dispatch — `scripts/qs/launchers/`

When a phase finishes and the user is about to start the next phase
(typically in a fresh interactive session), the agent calls
`python scripts/qs/next_step.py --next-cmd <phase> --work-dir <wd>
--issue <N> --title <t>`. The `--next-cmd` arg accepts either the bare
phase name (`create-plan`) or the slash form (`/create-plan`) for
back-compat; unknown phases raise `ValueError` and produce a JSON
error payload (no silent fallback). The phase-name → agent-name
mapping lives in `scripts/qs/launchers/phases.py` as a static dict —
no filesystem scan, so this works from any CWD.

`next_step.py` delegates to the harness-specific launcher:

- **`launchers/claude.py`** — emits a `sh /tmp/qs_launch_<N>.sh` one-liner
  whose generated script runs `claude --agent qs-<phase>` in the
  worktree. The `--agent` flag is what makes the new session
  interactive — Claude Code loads the agent body as the system prompt
  and the user can converse with the phase persona mid-flight (QS-175).
- **`launchers/cursor.py`** — emits a `cursor-agent --workspace <wd>
  --agent qs-<phase>` invocation (the `cli_context`) when
  `cursor-agent` is on PATH. When the binary is missing, falls back to
  the legacy prompt-positional form (the user opens Cursor manually
  and types `/<phase>` in chat). The IDE launcher (`new_context`)
  invokes `cursor <wd>` directly — Cursor doesn't expose a `--agent`
  flag for the IDE path, so the user types the slash command in chat
  once the IDE opens.
- **`launchers/opencode.py`** — under `caller='next_step'`
  (intermediate phases), POSTs to the OpenCode HTTP API via
  `scripts/qs/spawn_session.py` to create a new session in the same
  OpenCode instance with the next phase's agent already activated and
  a kickoff prompt sent. Under `caller='setup_task'` (Phase 1 →
  create-plan, cross-workspace), emits a CLI-form
  `opencode <worktree> --agent <name>` invocation instead, because
  the new worktree is a different OpenCode workspace. Falls back to
  the CLI form when the OpenCode server is unreachable
  (`shutil.which('opencode')` probe required). **Closed limitation**
  (QS-177 AC #12, closed by QS-190):
  spawn_session.py performs a pre-flight check on
  `<work_dir>/.opencode/agents/<agent>.md` before the HTTP API call;
  a missing agent file now produces a clean `agent_file_missing`
  exit shape instead of silently landing on the default agent.
- **`launchers/codex.py`** — stub.

All launchers return a dict with at minimum:

- `tool` (string, e.g. `"claude-code"`)
- `same_context` (string, slash-form fallback command)
- `new_context` (string, shell command to spawn a fresh session)

The **Claude** and **Cursor** launchers additionally emit `agent` (the
resolved `qs-<phase>` name). The **Codex** and **OpenCode** launchers
do NOT emit `agent` because they accept free-form `--next-cmd` values
that may not map to a static phase — see
`tests/qs/launchers/test_next_step_cli.py::test_codex_passes_known_phase_through_unchanged`
for the grep-able contract pin.

**Whitespace in `--next-cmd`** (review-fix #03 NTH7): codex/opencode
treat `--next-cmd` as a free-form string, so trailing or leading
whitespace inside an otherwise-non-empty value is preserved verbatim
(`--next-cmd "create-plan "` → `same_context: "create-plan "`). This is
intentional — explicit free-form is a feature, not a bug. The
empty / whitespace-only case is rejected for all harnesses by
`next_step.main()` after `parse_args()` returns.

PyCharm convenience commands (`pycharm_context`,
`pycharm_applescript_context`) are added when PyCharm is detected on
macOS and the work dir is a worktree.

## Why not synchronize agent files via a script?

Two approaches were considered:

- **Generate `.cursor/agents/` from `.claude/agents/` at build time**
  — saves duplicate writes but adds a sync step and breaks if anyone
  edits cursor agents directly.
- **Hand-maintain both directories** — duplicates content but keeps
  each harness's agents directly editable.

We chose hand-maintained. Agent bodies are stable; the marginal cost of
two copies is low; the cost of a missed sync is high. A diff lint script
(`scripts/qs/lint_agents.py`, future) can verify the bodies stay
aligned.

## Adding a new harness

1. Add a detection branch to `scripts/qs/harness.py::detect()`.
2. Add `scripts/qs/launchers/<harness>.py` with at least a
   `build_payload(work_dir, issue, title, next_cmd, ...) -> dict`
   function returning `{tool, same_context, new_context, ...}`.
3. Create the harness's agent directory (e.g., `.codex/agents/`) and
   copy the bodies from `.claude/agents/`, adjusting frontmatter to the
   harness's format.
4. Add the harness's slash-command equivalents if it has them.
5. Update this table.
