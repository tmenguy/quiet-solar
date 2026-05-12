# Harness abstraction

The pipeline runs across four harnesses with different mechanics:

| Harness         | Agent dir          | Slash commands     | Session spawn                 | Tool allowlist        |
| --------------- | ------------------ | ------------------ | ----------------------------- | --------------------- |
| Claude Code     | `.claude/agents/`  | `.claude/commands/`| `claude` CLI on worktree      | `tools:` frontmatter  |
| Cursor 2.4+     | `.cursor/agents/`  | `/<name>`          | New Cursor workspace          | `readonly:` boolean   |
| OpenCode (legacy)| `.opencode/agents/`| `.opencode/commands/`| HTTP API + `/instance/reload` | `permission:` block   |
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

When a phase finishes and the next phase runs in a new session
(e.g., `setup-task` → `create-plan` crosses workspaces), the agent
calls `python scripts/qs/next_step.py --phase <p> --next-cmd
"/create-plan"`. That delegates to the harness-specific launcher:

- **`launchers/claude.py`** — emits a `sh /tmp/qs_launch_<N>.sh` one-liner
  that opens a terminal in the worktree and runs `claude` with launch
  options.
- **`launchers/cursor.py`** — emits instructions to open the worktree as
  a new Cursor workspace and type the next slash command. Cursor doesn't
  have a CLI launcher equivalent today.
- **`launchers/opencode.py`** — emits the same OpenCode HTTP-API spawn
  approach as the legacy pipeline (delegates to `scripts/qs_opencode/`).
- **`launchers/codex.py`** — stub.

All launchers return a dict with at minimum `tool`, `same_context`,
`new_context`. PyCharm convenience commands (`pycharm_context`,
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
