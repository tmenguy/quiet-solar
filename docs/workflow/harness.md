# Harness abstraction

The pipeline runs across four harnesses with different mechanics:

| Harness         | Agent dir          | Slash commands     | Session spawn                 | Tool allowlist        |
| --------------- | ------------------ | ------------------ | ----------------------------- | --------------------- |
| Claude Code     | `.claude/agents/`  | `.claude/commands/`| `claude` CLI on worktree      | `tools:` frontmatter  |
| Cursor 2.4+     | `.cursor/agents/`  | `/<name>`          | New Cursor workspace          | `readonly:` boolean   |
| OpenCode        | `.opencode/agents/`| UI agent picker    | HTTP API: POST /session + POST /session/<id>/prompt_async (no reload) | `permission:` block   |
| Codex (future)  | `.codex/agents/`   | TBD                | TBD                           | TBD                   |

Each agent's **core protocol** (phase steps, quality-gate rules, hard
rules) MUST stay aligned across harnesses. The **frontmatter** and the
**declared harness-specific sections** legitimately differ, per
[project-rules.md](project-rules.md) § "Harness sync": handoff /
session-spawn blocks (Claude prints launchers for the user; OpenCode
runs `spawn_session.py` in-band), the per-harness "Code intelligence
(LSP)" sections, slash-form vs bare phase references, and tool-guard
prose matching each harness's permission model. The mechanical handoff
logic itself is isolated in Python (`scripts/qs/launchers/`).

## Code intelligence (LSP)

The Claude harness wires a native Python language server. The built-in
`LSP` tool (backed by the official `pyright-lsp` plugin, enabled in
`.claude/settings.json`) gives agents pyright **diagnostics** — type
errors and missing imports surfaced in-turn, before the quality gate —
**and** code **navigation** (definitions, references, hover types,
symbols). Because the qs `tools:` allowlists are closed, `LSP` is granted
only to the 8 code-navigating agents; the blind reviewers and
merge/release agents deliberately omit it. The plugin shells out to the
`pyright-langserver` binary, a per-machine prerequisite installed with
`npm install -g pyright` (machine-level, **not** in the venv or
`requirements*.txt` — the product type-checker stays mypy); it degrades
gracefully to grep when the binary is absent.

This is **Claude-only** for now. Per the multi-harness contract, the
other harnesses provide their own code intelligence rather than a shared
layer: Cursor (2.4+) has ambient editor-native LSP (no agent tool to
enable), and OpenCode bundles pyright but surfaces it as diagnostics-only
(no navigation), so it is intentionally not enabled there. Full rationale,
the per-harness capability matrix, and the rebuttal of the old
jedi-via-MCP plan live in
[../agents/lsp-evaluation.md](../agents/lsp-evaluation.md).

## Detection — `scripts/qs/harness.py`

`harness.detect()` returns one of `claude-code` / `cursor` / `opencode` /
`codex` (there is no `unknown` — detection falls back to `claude-code`).
Order of resolution:

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
  (QS-177 AC #12, closed by QS-190 — best-effort):
  spawn_session.py performs a pre-flight check on
  `<work_dir>/.opencode/agents/<agent>.md` (existence + readability +
  non-empty) AND on the worktree directory itself before the HTTP API
  call; missing / unreadable / empty agent files and an invalid
  worktree produce clean `agent_file_missing` / `agent_file_unreadable` /
  `agent_file_empty` / `worktree_invalid` exit shapes instead of
  silently landing on the default agent. A TOCTOU window remains
  between the pre-flight and the HTTP request.
- **`launchers/codex.py`** — stub.

All launchers return a dict with at minimum:

- `tool` (string, e.g. `"claude-code"`)
- `same_context` (string, slash-form fallback command)
- `new_context` (string, shell command to spawn a fresh session)

The **Claude**, **Cursor**, and **OpenCode** launchers additionally
emit `agent` (the resolved `qs-<phase>` name — all three resolve
`--next-cmd` strictly via `PHASE_TO_AGENT`). Only **Codex** payloads
carry no `agent` key: the codex launcher accepts free-form `--next-cmd`
values that may not map to a static phase — see
`tests/qs/launchers/test_next_step_cli.py::test_codex_passes_known_phase_through_unchanged`
for the grep-able contract pin. (The pre-QS-177 pipeline treated
opencode as free-form too; the static-agent pipeline made it strict.)

**Whitespace in `--next-cmd`** (review-fix #03 NTH7): codex treats
`--next-cmd` as a free-form string, so trailing or leading whitespace
inside an otherwise-non-empty value is preserved verbatim
(`--next-cmd "create-plan "` → `same_context: "create-plan "`). This is
intentional — explicit free-form is a feature, not a bug. Claude,
cursor, and opencode resolve strictly and reject unknown values. The
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
- **Hand-maintain all harness directories** — duplicates content but
  keeps each harness's agents directly editable.

We chose hand-maintained. Agent bodies are stable; the marginal cost of
three copies is low; the cost of a missed sync is high.
`check_doc_drift.py` enforces co-modification only; a content-level
sync checker (`scripts/qs/lint_agents.py` — not yet built; folded into
follow-up [#289](https://github.com/tmenguy/quiet-solar/issues/289))
could verify the aligned sections stay aligned.

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
