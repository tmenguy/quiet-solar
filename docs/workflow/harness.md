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

The Claude launcher also has one **filesystem side effect**: it pins the
resolved phase agent into the worktree's local settings, for the benefit
of the GUI launch surface documented next.

## GUI launch surface (Claude Code Desktop)

The Claude harness has **two launch surfaces**: the CLI
(`claude --agent qs-<phase>` — what the launcher emits) and the **Claude
Code GUI** (`Claude.app`). The GUI is a *surface*, not a harness: it
shares `.claude/` wholesale, so it has no agent directory of its own,
and `scripts/qs/harness.py` detects it as `claude-code`. GUI sessions
keep using `--harness claude-code`, and no launcher output is
conditional on the surface — a GUI user reads the same payload a CLI
user does, plus the `[Claude Code GUI]` block each orchestrator prints
at handoff.

### The mechanism — the `agent` settings key

The GUI exposes no `--agent` flag and no way to launch a session
programmatically (no URL scheme creates a session on a directory, and
`open -a "Claude" <dir>` ignores the folder). What it does honour is an
`agent` key in the settings file:

```json
{ "agent": "qs-implement-task" }
```

A session started **without** `--agent` in a directory whose
`.claude/settings.local.json` carries that key runs its main thread as
the named agent. This is documented upstream
(<https://code.claude.com/docs/en/settings.md> — "Run the main thread as
a named subagent…", scopes User / Project / Local).

`launchers/claude.py::_write_phase_agent` writes that key at **every**
handoff, so the worktree is always pinned to the phase the pipeline just
handed off to, and reports the outcome as the payload's
`phase_agent_pinned` — the handoff blocks say "should now be pinned"
because two guards, a race, and any `OSError` can each skip the write.

Two guards keep it narrow:

1. the destination must already contain `.claude/agents/<agent>.md`
   (a pin naming an absent agent silently falls back to the default one);
2. the destination must be a **linked worktree** — its `.git` is a
   *file* holding a `gitdir:` pointer. Note that `utils.is_worktree`
   alone does **not** establish this: it is a single inequality against
   `get_main_worktree()`, i.e. "is not the main checkout", and it answers
   `True` for a throwaway directory, a second clone, or even this repo's
   main checkout when the launcher runs from a cwd inside another repo.
   The `.git`-is-a-file check is what makes the two guards independent.

The file is **local and never committed** — `.gitignore` carries
`.claude/settings.local.json*` (that path plus the writer's temp sibling;
the rest of `.claude/` is tracked).

It is **not** purely machine-written: Claude Code persists the user's own
`permissions` decisions, `model`, and `env` there. The writer is therefore
deliberately timid — it will decline to pin rather than touch bytes it does
not fully understand:

- **A file it can parse as a JSON object** is shallow-merged: `agent` is
  replaced, every other top-level key is kept.
- **Anything else is left exactly as it is, and the pin is skipped** — an
  unreadable file, one that does not parse, or one that parses to something
  other than an object (`null`, `[1, 2]`, `"x"`, empty, NUL-filled). Always
  with a warning on stderr. There is **no backup and no rebuild**: earlier
  revisions rebuilt the file and kept a `.bak`, and that safety net produced
  three consecutive must-fix findings of its own, so the destructive path
  was removed rather than patched again.
- **A symlink is refused**, not followed — at the settings file *or* at
  `.claude` itself. Following either would move the write outside the
  worktree: into `~/.claude/settings.json` (user scope, affecting every
  project), or into the main checkout this section promises is never
  pinned. Checking only the file was not enough, because a symlinked
  `.claude` leaves the file itself looking perfectly ordinary.
- **Permissions**: a fresh pin file is created `0o600`; an existing file's
  mode is preserved, so your `chmod` survives. The temp sibling is created
  private and widened only once populated, so a merged `env` token is never
  briefly world-readable.

The write is atomic (temp sibling + `os.replace`) and best-effort: it never
breaks a handoff, and every skip above reports `phase_agent_pinned: false`.

**A CLI session that passes `--agent` is unaffected: the flag overrides
the key.** A launcher one-liner always lands on the agent it names,
whatever the pin says. A CLI session that does *not* pass the flag —
a bare `claude`, `claude -p …`, an SDK run — is pinned like any other
(see Traps).
*(Evidence: scratch-directory test, two agents, the flag won — QS-311
finding F2, verified 2026-07-30 on `claude` 2.1.220. Marked because this
claim is load-bearing for every "use the Preferred line above" recovery
instruction, and it was previously the only unmarked claim in the
section.)*

### The GUI loop

One GUI session per phase, exactly as on the CLI:

1. **New session** — this is mandatory, not stylistic. The GUI reopens
   the previous session by default, and a restored session keeps the
   agent it was created with (the key is read at *session* start).
2. Select the worktree directory.
3. Name it something like `QS_<N> implement-task`.
4. Work the phase; at the handoff, repeat from step 1 for the next one.

`/setup-task` seeds the loop: it creates the worktree, pins
`qs-create-plan` into it (unless `--no-worktree`, see Traps), and prints
the directory to open.

### Traps

- **A bad pin fails silently.** An unknown agent name falls back to the
  default agent with no error, and the GUI displays the active agent
  *nowhere* (the CLI header does show it). That is why the writer
  refuses to pin an agent whose file is absent from the destination.
- **Stale pins.** The pin reflects the *last handoff*, not necessarily
  the phase you intend to work. Combined with session restore and the
  invisible agent name, a reopened GUI session can silently be the wrong
  orchestrator — and orchestrators commit and push. Confirm the phase
  before working in a reopened GUI session; when in doubt, re-run the
  previous handoff to refresh the pin, or use the CLI *passing*
  `--agent`, which always wins.
- **The main-checkout gap.** `setup-task` and `release` run on the main
  checkout, which is never pinned (by design — guard 2). Reach them in
  the GUI via the slash form `/setup-task` / `/release`, which is what
  the slash commands are still for. Same for `setup-task --no-worktree`:
  the work dir *is* the main checkout, so that run reports
  `phase_agent_pinned: false` and there is no GUI pin to open.
- **GUI self-isolation drops the pin.** The pin file is gitignored *by
  design*, so it does not exist in a git worktree the GUI creates for
  itself when isolation is enabled: the tracked `.claude/agents/*.md`
  come along with `HEAD`, the untracked pin does not, and the sub-tree
  boots as the default agent — invisibly, like every other bad-pin case.
  The good news is that the sub-tree *does* inherit `HEAD`, so you still
  land on `QS_<N>` with your work in place; it is only the persona that is
  lost. Either disable isolation for pipeline worktrees, or work the phase
  from the CLI, passing `--agent`.
- **Headless and bare CLI runs are pinned too.** The mechanism is "a
  session started *without* `--agent` runs its main thread as the named
  agent" — and nothing about that is GUI-specific. `claude -p …`, an
  Agent-SDK run, or any script whose `cwd` is inside a pinned worktree
  inherits the pin, with no interactive header to reveal it. Worst case:
  after the review-task → finish-task handoff, a bare invocation in that
  worktree boots as the orchestrator whose job is to merge the PR and
  remove the worktree. **Pass `--agent` explicitly in any automation.**
- **A corrupt or symlinked pin file produces no pin — and stays that way.**
  Those are the states the writer refuses (above), and like every other
  bad-pin case the GUI shows you nothing. The signals are
  `phase_agent_pinned: false` in the handoff payload (the orchestrator is
  instructed to stop claiming the pin when it sees that) and a stderr
  warning naming the file. **The skip is terminal, not transient:** nothing
  repairs the file, so every later handoff in that worktree refuses it too.
  The remedy is `rm .claude/settings.local.json` — the next handoff
  recreates it at `0600` — or, for a symlink, remove the link. `--agent`
  works regardless and needs no repair.
- **A failed write leaves the *previous* phase's pin in place.** The
  writer is best-effort, so an `OSError` on the publish means the worktree
  stays pinned to the phase before this one — strictly worse than being
  unpinned, because it looks intentional. `phase_agent_pinned: false`
  cannot distinguish "no pin" from "stale pin"; when you see it, check the
  file or just pass `--agent`.
- **The pin races the app.** The writer does a read-modify-write on a
  file Claude Code also owns, with no lock. A handoff normally runs from
  *inside* a live session on that same worktree, so a permission the user
  approves at just the wrong moment can be dropped — or can drop the
  `agent` key, un-pinning the worktree the handoff text just described.
  A re-read immediately before the publish shrinks the window; it does
  not close it. Harmless in practice (re-run the handoff, or use the
  CLI), recorded because the failure is silent.

### Hybrid: `/desktop`

From a running CLI session, `/desktop` (alias `/app`) transfers the
session to the GUI on the same directory and branch, and the agent
persona survives the transfer — the smoothest route into the GUI when
you are already on the CLI. Caveats:

- It **fails on an empty session** with `transcript_missing`, which
  reads like data loss but only means "complete one exchange first".
- It **terminates the originating CLI session**, so there is no
  dual-agent hazard.
- It is **undocumented upstream and feature-flagged**, so it may vanish;
  the loop above does not depend on it.
- Persona survival was **observed once (n=1)** under controlled
  conditions — treat it as evidence, not as an established contract.

Verified 2026-07-31 against `claude` 2.1.220 and `Claude.app`
(`com.anthropic.claudefordesktop`) 1.24012.9.

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

**Byte-identical blocks must be edited in lockstep.** Some agent
passages are intentionally mirrored byte-for-byte across all three
harness files and guarded by a test. The clearest example is the QS-299
post-merge **seed/follow launch block** in `qs-finish-task.md` (the
`--seed-testmon --detached --seed-token …` launch plus the empty-token
guard), pinned by
`tests/test_quality_gate.py::TestFinishTaskRefreshesBaseline::test_seed_launch_block_byte_identical_across_harnesses`.
Any edit to such a block must be applied identically to `.claude`,
`.cursor`, and `.opencode` in the same change, or that test fails. The
surrounding per-harness prose (e.g. the background+monitor mechanism)
deliberately differs and is not part of the pinned slice.

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
