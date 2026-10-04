# The Control Plane

The Control Plane is the second pipeline's runtime core (epic
[#369](../epics/QS-369.md), child 5: [#399](https://github.com/tmenguy/quiet-solar/issues/399)).
It is Python with no LLM, and there is one of it for all runs. It holds:

- the runtime state, in `harness_state.db`;
- the run queues;
- the commands that LLM sessions call;
- the tools layer;
- the hooks.

Code: [`scripts/qs/control_plane/`](../../scripts/qs/control_plane/), behind the
entry shim [`scripts/qs/cp.py`](../../scripts/qs/cp.py). Design and decisions:
[`docs/stories/QS-399.story.md`](../stories/QS-399.story.md).

## Invocation

Every session, worktree sessions included, calls **the main checkout's copy**:

```bash
<MAIN>/venv/bin/python <MAIN>/scripts/qs/cp.py <command> …
```

Never call a worktree's copy. Only the main checkout's code may open the live DB,
`<MAIN>/harness_state.db`. Any other copy is refused with `PATH_GUARD` (exit 7).
Set `QS_CP_DB` to point at a temporary DB instead.

Every command prints **one JSON object** on stdout:

- `{"ok": true, …}` on success;
- `{"ok": false, "error": "<CODE>", "detail": "…"}` on failure.

The hooks print the Claude Code hook protocol instead, or nothing.

| exit | codes | meaning |
|---|---|---|
| 0 | — | ok. For a hook: allow, or its JSON decision |
| 1 | `INTERNAL`, `TOOL_FAILED` | unexpected error, or a tool step failed (recorded) |
| 2 | `USAGE` | bad arguments, or no session id |
| 3 | `STALE_TOKEN` | the token is superseded. The payload carries `superseded_by`, `at` and the self-end `instructions` |
| 4 | `STOPPED` | the caller's node is stopped |
| 5 | `SCHEMA_TOO_NEW`, `SCHEMA_PENDING` | the code is older than the DB, or a migration did not finish in time (`restart_wait` from `wait`) |
| 6 | `BUSY` | a lock, a cap, an in-flight call of the same tool and key, the node cap, or an unknown liveness |
| 7 | `PATH_GUARD` | this copy of the code may not open this DB |
| 8 | `NOT_FOUND`, `CONFLICT`, `INVALID_STATE` | unknown id; uniqueness, argument or receipt mismatch, or a token used out of bounds; forbidden transition |
| 9 | `POLICY_REFUSED` | merge policy, a foreign or `core.hooksPath` hook, a write into the main checkout, a live DB outside `main`, … |

**Tokens.** A token is `run:R<n>.<epoch>.<nonce>` or `node:N<n>.<generation>.<nonce>`.

- Every effectful command takes `--token`. The CLI never reads the token from the environment.
- Tokens appear only in the responses of `run open`, `run claim`, `tool spawn` and `tool resume`.
- No read command ever outputs a nonce.

**Session ids.** `--session-id` defaults to `$CLAUDE_CODE_SESSION_ID`. If neither is set, the command fails with `USAGE`.

## Commands

Kinds:

- **exempt:** never waits and never starts the daemon;
- **read:** checks the schema without waiting, never starts the daemon, and answers empty on a missing DB;
- **write:** waits for the self-migration, starting the daemon if needed.

| command | arguments | token | kind |
|---|---|---|---|
| `version` | — | — | exempt |
| `daemon` | — | — | exempt |
| `ensure` | — | — | exempt |
| `hook stop` / `hook pre-tool-use` | stdin: hook JSON | — | exempt |
| `hook pre-push` | git's args and stdin | (`QS_CP_TOKEN`) | exempt |
| `hooks-settings` | `--role node\|orchestrator` | — | exempt |
| `session status` | `[--session-id]` | — | read |
| `snapshot` | `[--run R]` | — | read |
| `task show` | `--task T` | — | read |
| `export-summary` | `--task T --out-worktree WT` | — | read |
| `run open` | `--name SLUG --title … [--session-id] [--session-name] [--permission-mode] [--full-grant]` | returns run | write |
| `run claim` | `RUN [--session-id] [--session-name] [--permission-mode] [--full-grant] [--takeover]` | returns run | write |
| `run bind-name` | `RUN` | run | write |
| `run set-mode` | `--permission-mode M [--full-grant\|--no-full-grant]` | run | write |
| `run set-plan` | `--run R --file F` | run | write |
| `run close` | — | run | write |
| `task add` | `[--run R] --title --kind epic\|feature\|bug [--target] [--parent T] [--issue N] [--lane L] [--deliverable] [--item-of T]` | run | write |
| `task set` | `--task T [--issue] [--worktree] [--branch] [--pr-number --pr-url] [--ci-state --ci-sha]` | run | write |
| `task state` | `--task T --to STATE\|unblock [--note]` | run, or node (own task, node range) | write |
| `task dep` / `task root` / `task work-list` | `add\|remove …` | run | write |
| `criteria set` / `validate` / `state` | `--task T …` | run | write |
| `question open` | `--task T --text-file F [--blocking]` | run, or node (own task) | write |
| `question ask` / `question answer` | `Q-n …` | run | write |
| `decision add` | `--text --reason --source [--task]` | run | write |
| `report post` | `--task --phase --round --status --summary --fields-file` | run, or node (own task) | write |
| `digest put` | `--task --file` | run, or node (own task) | write |
| `msg post` | `--run R --to orchestrator\|node:T --kind K --payload-file F [--dedupe-key D]` | run (any recipient), node (`orchestrator` only) | write |
| `msg pop` / `msg ack` | `--run R --as … [--visibility S]` / `ID --receipt R` | the recipient's (a stopped node may still pop and ack; the receipt of a dead-lettered message's last delivery still acks it) | write |
| `wait` | `--run R [--timeout S] [--poll S]` | run | write |
| `node stop` / `node take-over` | `--task T` (`stop` on a stopped node is a noop) | run | write |
| `node hand-back` | `--task T --summary-file F` | run, or node (own task) | write |
| `lock acquire` / `lock release` | `--name integration:<branch> [--purpose] [--session-id] [--timeout S]` | run, or node (own deliverable, own session) | write |
| `tool <name>` | `--task T --key K [--args-file F]` | per tool | write |

Time flags (`--visibility`, `--poll`, `wait --timeout`) must be finite and above 0; `lock acquire --timeout` may be 0 (try once). Anything else is `USAGE`.

`run open` and `run claim` start the daemon after the run is opened or claimed. A daemon start error is reported as `"daemon": "error"` with `daemon_error`; the run token is still printed.

### Exempt commands

- **The hooks.** On a schema mismatch or an internal error they fail open at once. The exceptions are DB-free: `pre-push`'s refusal of work-item refs, and `PreToolUse`'s DB-access rule, which denies even while the DB is missing, migrating or contended.
- **`hooks-settings` and `version`.** No DB at all.
- **`daemon`.** It migrates.
- **`ensure`.** It reads the daemon lease defensively.
- **The read commands** (`session status`, `snapshot`, `task show`, `export-summary`):
  - a missing DB is an empty answer;
  - an older DB is `SCHEMA_PENDING` at once;
  - a newer DB is `SCHEMA_TOO_NEW`.

## Schema and self-migration

### Tables

`harness_state.db` holds, at schema v1:

| group | tables |
|---|---|
| runs | `runs`, `run_leases`, `run_sessions` |
| task tree | `tasks` (with the work-item fields `deliverable_id`, `item_k`, `next_item_k`), `run_roots`, `work_list`, `task_deps`, `criteria`, `task_history` |
| nodes | `nodes`: one row per generation |
| queues and records | `messages`, `questions`, `decisions`, `digests`, `reports`, `integrations` |
| coordination | `locks`, `cap_slots`, `tool_calls`, `waiters`, `daemon_lease`, `hook_events` |
| bookkeeping | `meta`, `counters` |

The schema version is `PRAGMA user_version`.

### Version checks

| DB vs code | write commands | read commands and hooks |
|---|---|---|
| equal | proceed | proceed |
| newer | refuse with `SCHEMA_TOO_NEW` | refuse with `SCHEMA_TOO_NEW` |
| older | wait for the daemon's migration, up to `MIGRATE_WAIT_S`, then `SCHEMA_PENDING` | `SCHEMA_PENDING` at once |

Every write transaction re-checks the version, so a long-lived process running old code never writes to a migrated DB.

### Who migrates

Only the daemon migrates the live DB, and only when **both** hold:

- it runs the main checkout's code;
- `main` is checked out there.

A DB already at the current schema needs no migration, so `migrate` is a noop without that check: the daemon also starts while the main checkout is on another branch or a detached HEAD. A test DB, which is not a live-DB path, may be migrated by anyone.

Lock contention during a migration is `BUSY`. The daemon retries it (`MIGRATE_BUSY_RETRIES` times, one tick apart) and never records it in the backoff sidecar.

Before migrating an existing DB, the daemon backs it up to `QS_CP_BACKUP_DIR`, which is never inside a checkout.

### Migration recipe for later children (#375 and others)

1. **Append** `Migration(n + 1, "<name>", (<single statements>…))` to `MIGRATIONS` in `migrations.py`.
   - Use single statements only, never `executescript`.
   - A test pins the versions as `1..N`.
2. **Land it on `main`.** The next `ensure` sees a lower-schema daemon lease, stops the old daemon (SIGTERM while it is alive, fresh or hung), and starts one that backs up and migrates. If the old daemon is still alive after `DAEMON_RESTART_WAIT_S`, `ensure` answers `restart_pending` and starts nothing; `wait` then exits 5 with `restart_wait`.
3. **If the migration fails,** the daemon writes `<db>.migrate-error.json`. `ensure` does not respawn the daemon for `MIGRATE_BACKOFF_S`, and waiting commands get `SCHEMA_PENDING` with the error.

## Locks and caps

| name | holder | purpose |
|---|---|---|
| `integration:<branch>` | a tool's process group, **or** a session across commands (`lock acquire`) | integration into one deliverable branch (#400); child 7's merge of that branch |
| `main-merge` | a tool's process group | serialises merges into `main` (child 7) |
| `main-checkout` | a tool's process group | fetch, fast-forward, worktree setup and cleanup in `<MAIN>` |

### `LOCK_ORDER`

1. `integration:*`, in lexical order among themselves;
2. `main-merge`;
3. `main-checkout`;
4. a gate slot, always last.

`hold()` asserts the order of the names it is given. Across commands, a name that sorts before a session-held `integration:*` lock of the same token subject is refused with `CONFLICT`.

### When a holder is dead

- **A process holder** is dead when its pid (checked with its start time) **and** its process group are gone.
- **Liveness is tri-state.** The start time comes from `ps -o lstart=` run with `TZ=UTC0` and `LC_ALL=C`. When `ps` itself fails (a timeout, an exec failure, unparseable output) the answer is *unknown*, and unknown always keeps the holder: a failed probe is never evidence of absence.
- **A session holder** is dead when any of these holds:
  - its token is superseded;
  - its session is absent from a successful `claude agents --json` listing;
  - or, only if the listing failed, its recorded pid is proven dead;
  - its run is closed;
  - every deliverable on its branch is terminal.
- A session-held lock stays held while a **co-holder** (one of the holder's own tools) is alive.

### Caps

- **Gate cap:** `QS_CP_MAX_GATES` slots (default 2), for `tool gate` and #400's integration finish. The cap counts the live slot rows against the caller's own limit, so a process started with a smaller limit never exceeds it.
- **Node cap:** `QS_CP_MAX_NODES` (default 4), counted across all runs. A node counts when it is:
  - `spawning`, while its spawn call's holder is alive or its launch has not settled;
  - `running`, `idle` or `taken_over`, when the listing shows it, or the listing failed, or it launched less than `LAUNCH_SETTLE_S` ago.

## Tools

Every tool runs through `run_recorded`:

1. Verify the token. A call already finished under the key is replayed at once, before its steps are built: a deleted argument file or a task column cleared meanwhile cannot break the replay.
2. Claim `tool_calls(tool, key)` before any effect.
3. Probe: the probe is authoritative for each step it reports, done or not done.
4. Run the missing steps under the locks or the cap, re-checking the token and lock ownership before each step.
5. Record the call `succeeded` or `failed`.

**Keys** are chosen by the caller: `msg:<id>` for a popped message, or `task:<id>:<purpose>` otherwise. Replaying the same key never repeats an effect. The key's `args_hash` covers the arguments and the content of every `*_file` argument: the same key with an edited file is a `CONFLICT`.

A probe that cannot tell (a failed or unparseable `gh` listing) answers `BUSY`, "replay the same key later", never "not done".

| tool | token | runs in | locks / cap | probe |
|---|---|---|---|---|
| `worktree-create` (`phase`) | run | `<MAIN>` | `main-checkout` | none; `setup_task.py` is idempotent |
| `worktree-cleanup` | run | `<MAIN>` | `main-checkout` | directory absent and unregistered. Refuses the main checkout and a directory whose `.git` is not a file (`POLICY_REFUSED`); a path another non-terminal task shares only clears this task's column (`shared_with`) |
| `gate` (`mode`, `paths`) | run, or node (own task) | worktree | `gates` slot | none; safe to re-run |
| `spawn` (`agent`, `model?`, `permission_mode`, `prompt_file`, `replace?`) | run | worktree | node cap (`--replace` frees the replaced node's place first) | the reserved row through `spawn_tool_key`, and the listing by name. A launch never listed within `LAUNCH_SETTLE_S` is reaped and relaunched |
| `resume` (`message_file`) | run | worktree | node cap | the row through `spawn_tool_key`, and the listing by name and `startedAt` |
| `issue-create` (`title`, `body_file`, `labels`) | run | `<MAIN>` | — | a `<!-- qs-cp-key: … -->` marker in the issue bodies, found with `gh issue list --search` |
| `pr-create` (`title`, `summary_file`) | run, or node (own task) | worktree | — | the marker in the PR bodies of the branch |
| `push` | run, or node (own task) | worktree | — | `git ls-remote` equals `HEAD` |
| `merge` | run | `<MAIN>` | `integration:<branch>`, `main-merge` | `gh pr view` shows `MERGED` |

Notes:

- Every built-in tool except `worktree-cleanup` refuses a terminal task with `INVALID_STATE`, recorded `failed`.
- `merge` also requires the task to be `ready_to_merge`, unless the PR is already merged. Once the PR is merged, `merge_sha` is always recorded; if the task left `ready_to_merge` meanwhile, the call still succeeds, with `state_conflict: {"expected": "ready_to_merge", "actual": …}`, for the orchestrator or the maintainer to reconcile.
- `merge` is refused by the default merge policy until child 7 installs one (`merge_policy.install`).

**Outcomes.**

| what happened | the call | the caller sees |
|---|---|---|
| a step failed, or the state drifted | recorded `failed`; the key is spent | `TOOL_FAILED` (exit 1), with `result.error` |
| `BUSY`, `POLICY_REFUSED`, `STALE_TOKEN` or `STOPPED` before any effect | claim released, so the same key can be retried | that code |
| the same codes after an effect | left `started`; a same-key replay takes it over once this process has exited | that code |
| another process took the claim over mid-call | left to that process | `STALE_TOKEN` |

### The `ToolSpec` API (frozen for #400 and later children)

```python
@dataclass(frozen=True)
class StepCtx:
    task: sqlite3.Row; args: Mapping[str, Any]; outputs: Mapping[str, Any]
    runner: Runner; claude: ClaudeCli; main: Path
    tool: str; key: str; clock: Clock          # additive: markers and timeouts
    def write(self) -> ContextManager[sqlite3.Connection]: ...
    def record_output(self, value: Any) -> None: ...   # inside write(): the DB-only step's output, atomically

@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[StepCtx], Any]   # returns the step's JSON-able output; raise StepFailed on its own failure
    inject_token: bool = False      # the step's runner adds QS_CP_TOKEN (steps that may git push)
    detach: bool = False            # start_new_session (claude --bg launches)

@dataclass(frozen=True)
class ToolSpec:
    name: str
    steps: Callable[[Row, Mapping[str, Any]], Sequence[Step]]   # validate args here: before the claim
    locks: Callable[[Row, Mapping[str, Any]], Sequence[str]] = no locks   # sorted by LOCK_ORDER
    cap: str | None = None                                       # "gates"
    probe: Callable[[StepCtx], dict[str, Any | None]] | None = None
    guard: Callable[[StepCtx], None] | None = None               # raise INVALID_STATE on drift
    on_success: Callable[[sqlite3.Connection, StepCtx], dict[str, Any]] | None = None
    refuse_when_stopped: bool = True
    token_kinds: frozenset[str] = frozenset({"run"})

def register(spec: ToolSpec) -> None   # CONFLICT if the name exists
def invoke(name, *, key, task_id, args, token, actor, ctx=None) -> dict
```

- `argv_step(...)`, `task_state_guard(...)` and `compose(...)` are helpers. They are not frozen.
- Every registered tool is reachable as `cp.py tool <name> --task T --key K [--args-file F] --token …`.

## Hooks

| hook | wired | input → output | failure |
|---|---|---|---|
| `Stop` (orchestrator) | child 9 writes `hooks-settings --role orchestrator` into main's pin; **6a must not ship before** | stdin `session_id` → `{"decision": "block", "reason": …}` when a message waits (pop it), or once when no `wait` runs; a loop guard allows and records a `hook_events` `alert` | fail open |
| `PreToolUse` (every registered session; matcher `Bash\|Edit\|Write\|SendMessage`) | nodes: `claude --bg --settings '<hooks-settings --role node>'` at spawn | `tool_name`, `tool_input` → `hookSpecificOutput.permissionDecision: "deny"` for DB access outside `cp.py` (any session), `gh pr merge` (registered sessions), `SendMessage` from a superseded or stopped session | fail open |
| `pre-push` (git) | a common-dir shim installed by `tool worktree-create` (marker `# qs-control-plane pre-push shim v2`) | git's ref lines → exit 1 refuses `QS_<N>_<k>` refs everywhere, and, in a registered worktree, a stopped node, a foreign ref, or a missing or stale `QS_CP_TOKEN` | fail closed only for a proven-registered worktree |

Whether `--settings` hooks survive a bare `claude --bg --resume` is unverified, an open point for 6b. `pre-push` is git-level, so it applies regardless.

The shim and the settings commands run `<MAIN>/venv/bin/python`. Without it they fall back to `python3` only if it is 3.14 or newer (the package's syntax); otherwise they print one warning line and allow.

## Environment variables

| variable | default | for |
|---|---|---|
| `QS_CP_DB` | `<MAIN>/harness_state.db` | a temporary DB (tests, experiments). Never a live DB from another checkout |
| `QS_CP_BACKUP_DIR` | `~/.local/state/quiet-solar/cp-backups/` | migration backups. Never inside a checkout |
| `QS_CP_MAX_GATES` | 2 | gate slots |
| `QS_CP_MAX_NODES` | 4 | node sessions across all runs |
| `QS_CP_TOKEN` | — | **git only**: injected by the tools into the steps that may push, so `pre-push` accepts them. Every `claude` launch removes it |

## Module constants

Each is overridable by a function argument.

| constant | value | module |
|---|---|---|
| `MAX_ATTEMPTS` | 5 | `messages` |
| `VISIBILITY_S` | 900 | `messages` |
| `TICK_S` | 5 | `daemon` |
| `IDLE_EXIT_S` | 1800 | `daemon` |
| `STALE_AFTER_S` | 30 | `daemon` |
| `DAEMON_RESTART_WAIT_S` | 15 | `daemon` |
| `MIGRATE_WAIT_S` | 60 | `db` |
| `MIGRATE_BACKOFF_S` | 300 | `daemon` |
| `MIGRATE_BUSY_RETRIES` | 6 | `daemon` |
| `LAUNCH_SETTLE_S` | 60 | `nodes` |
| `GATE_WAIT_S` | 1800 | `locks` |
| `LOCK_WAIT_S` | 600 | `locks` |
| `DIGEST_MAX_BYTES` | 16 KiB | `reports` |

## Seams

| seam | real | in tests |
|---|---|---|
| `Clock` | `SystemClock` | `FakeClock`: `sleep` advances time |
| `Runner` | `subprocess.run` with an environment delta | `FakeRunner`: records argv, cwd, environment delta and `detach` |
| `ProcessProbe` | `kill(pid, 0)` plus `ps -o lstart=` (UTC, C locale; a failed `ps` is unknown), and `killpg` | `FakeProbe`: a fresh pid per call, reaped when the CLI call returns |
| `ClaudeCli` | `claude agents --json`, `claude --bg …` | `FakeClaude`: a scripted listing |
| `ProcessSetup` | `setpgid`, `signal.signal` | a recorder: pytest's own process never changes group or gets a handler |
| `Popen` | `subprocess.Popen`, for `ensure` | a recorder |
| `faults.hit(name)` | no-op | `faults.arm(name, exc, skip=n)` |
| `merge_policy` | refuses | `install(fn)` |
| `export.LEDGER_SECTIONS` | empty: "No ledger recorded." | #375 fills it |
| the daemon's `tick_hooks` | none | child 14's active loop |

## Conventions: what no hook enforces

- `git push --no-verify` bypasses `pre-push`.
- A superseded session claiming its run back with `--takeover`: takeover is the maintainer's act, and the self-end payload says so.
- Passing another session's token.
- **`run claim --session-id` trusts its caller.** The id names the session claiming the lease, and nothing checks that the caller is that session. The threat model covers agent mistakes, not adversarial sessions: forging another session's id is a deliberate act.
- `gh` issue and PR edits made outside the tools.
- `SendMessage` from an unregistered session.
- DB access that slips past the Bash segment rule, for example through an allowlisted program, or code that reads the path from a variable. Known gaps: `find … -delete` and `sed -i` behind an allowlisted program name, `$(…)` command substitution, `git rm`. The hook guards against accidents; it is not a sandbox.
- **`gh pr merge` through a wrapper.** The merge rule matches a segment that starts with `gh pr merge`; a wrapper (`env gh pr merge`, `command gh …`, an alias, a script, `gh api`) is not caught.
- **The fencing window on external effects.** GitHub never sees the token. A superseded session's already-started step can still land; its next step is refused. `merge` narrows the window with `--match-head-commit`.
- **The session-lock fallback limit.**
  - When the liveness listing fails, a session-held lock falls back to the pid stamped at the session's last use. After an app restart that pid can be stale, so the lock can be freed while the session is alive.
  - A node that is not listed again, such as a reaped `--bg` node not yet resumed, also loses its lock.
  - The branch move is a ref compare-and-swap, so the cost is redoing an integration step, never corruption.
- **The launch-settle limit.** A `claude --bg` session that only appears in the listing more than `LAUNCH_SETTLE_S` after its launch may be launched twice: a same-key replay reaps the unlisted launch and relaunches it.
- **The hook `session_id` assumption.** The hook's stdin `session_id`, `$CLAUDE_CODE_SESSION_ID` and the `sessionId` of `claude agents --json` are assumed to be the same identifier. The last two were verified equal on 2026-10-04; the hook field is unverified.
- **`--settings` hooks after a bare resume:** unverified.
- **The orchestrator's hooks are unwired until child 9.**
