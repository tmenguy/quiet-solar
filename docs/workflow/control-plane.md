# The Control Plane

The Control Plane is the second pipeline's runtime core (epic
[#369](../epics/QS-369.md), child 5: [#399](https://github.com/tmenguy/quiet-solar/issues/399)).
It is Python with no LLM, and there is one of it for all runs. It holds:

- the runtime state, in `harness_state.db`;
- the run queues;
- the commands that LLM sessions call;
- the tools layer;
- the hooks;
- the **active loop** (child 14, [#406](https://github.com/tmenguy/quiet-solar/issues/406)):
  the daemon's tick hooks, which watch every run and alert, but never decide;
- the **review sessions** (child 15, [#407](https://github.com/tmenguy/quiet-solar/issues/407)):
  the global review of a run and the cross-run coherence review, one-shot LLM
  sessions it launches and records ([Review sessions](#review-sessions-407)).

Code: [`scripts/qs/control_plane/`](../../scripts/qs/control_plane/), behind the
entry shim [`scripts/qs/cp.py`](../../scripts/qs/cp.py). Design and decisions:
[`docs/stories/QS-399.story.md`](../stories/QS-399.story.md) and, for the active
loop, [`docs/stories/QS-406.story.md`](../stories/QS-406.story.md); for the review sessions,
[`docs/stories/QS-407.story.md`](../stories/QS-407.story.md).

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
| 3 | `STALE_TOKEN` | the token is superseded. The payload carries `superseded_by`, `at` and the self-end `instructions` (a review token: `review_id`, `state` and the review's own `instructions`) |
| 4 | `STOPPED` | the caller's node is stopped |
| 5 | `SCHEMA_TOO_NEW`, `SCHEMA_PENDING` | the code is older than the DB, or a migration did not finish in time (`restart_wait` from `wait`) |
| 6 | `BUSY` | a lock, a cap, an in-flight call of the same tool and key, the node cap, or an unknown liveness |
| 7 | `PATH_GUARD` | this copy of the code may not open this DB |
| 8 | `NOT_FOUND`, `CONFLICT`, `INVALID_STATE` | unknown id; uniqueness, argument or receipt mismatch, a token used out of bounds (a task of another run included), or a tool call's claim taken over by another process (`claim_taken_over`); forbidden transition |
| 9 | `POLICY_REFUSED` | merge policy, a foreign or `core.hooksPath` hook, a write into the main checkout, a live DB outside `main`, … |

**Tokens.** A token is `run:R<n>.<epoch>.<nonce>`, `node:N<n>.<generation>.<nonce>`, or a review token `review:V<n>.<0|1>.<nonce>` (#407: `1` the review session's token, `0` the review pilot's launch grant; see [Review sessions](#review-sessions-407)).

- Every effectful command takes `--token`. The CLI never reads the token from the environment.
- Tokens appear only in the responses of `run open`, `run claim`, `tool spawn` and `tool resume`. A review token appears in no response: the session token is in the review's launch prompt, the grant in the pilot's launcher argv.
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
| `halt clear` | `--reason TEXT` | — (the maintainer, through the `ask` prompt) | exempt |
| `restore` | `--confirm` | — (the maintainer, through the `ask` prompt) | exempt |
| `hook stop` / `hook pre-tool-use` | stdin: hook JSON | — | exempt |
| `hook pre-push` | git's args and stdin | (`QS_CP_TOKEN`) | exempt |
| `hooks-settings` | `--role node\|orchestrator` | — | exempt |
| `review read` | `diff --base SHA --head SHA [--path P]` / `show --rev SHA --path P` / `grep --rev SHA --pattern X [--path P]` / `ls-tree --rev SHA [--path P]` / `files --base SHA --head SHA [--path P]` (#407: a review's only code read, a fixed git argv on hex shas) | — | exempt |
| `session status` | `[--session-id]` | — | read |
| `snapshot` | `[--run R]` | — | read |
| `task show` | `--task T` (the task in full, its ledger and `convergence` included) | — | read |
| `ledger show` | `--task T [--phase plan\|build] [--family] [--state S[,S…]]` (see [The ledger](#the-ledger-375)) | — | read |
| `export-summary` | `--task T --out-worktree WT` | — | read |
| `run open` | `--name SLUG --title … [--session-id] [--session-name] [--permission-mode] [--full-grant]` | returns run | write |
| `run claim` | `RUN [--session-id] [--session-name] [--permission-mode] [--full-grant] [--takeover]` | returns run | write |
| `run bind-name` | `RUN` | run | write |
| `run set-mode` | `--permission-mode M [--full-grant\|--no-full-grant]` | run | write |
| `run set-plan` | `--run R --file F` | run | write |
| `run close` | — | run | write |
| `task add` | `[--run R] --title --kind epic\|feature\|bug [--target] [--parent T] [--issue N] [--lane L] [--deliverable] [--item-of T]` (`--parent` / `--item-of` must be a task of the caller's run, or of none: `CONFLICT` otherwise; `--item-of` with `--deliverable` or `--issue` is `USAGE` — a work item lands in its deliverable's PR, a deliverable is its own issue `QS_<M>`, #400) | run | write |
| `task set` | `--task T [--issue] [--worktree] [--branch] [--pr-number --pr-url] [--ci-state --ci-sha]` (`--issue` / `--pr-number` / `--pr-url` on a work item is `INVALID_STATE`, #400; the `ci_state` vocabulary `pending\|green\|red` is child 14's contract, not checked by the schema) | run | write |
| `task state` | `--task T --to STATE\|unblock [--note]` | run, or node (own task, node range) | write |
| `task dep` / `task root` / `task work-list` | `add\|remove …` (both tasks of a `task dep` must be of the caller's run, or of none: `CONFLICT` otherwise) | run | write |
| `criteria set` / `validate` / `state` | `--task T …` | run | write |
| `question open` | `--task T --text-file F [--blocking]` | run, or node (own task) | write |
| `question ask` / `question answer` | `Q-n …` | run | write |
| `decision add` | `--text --reason --source [--task]` | run | write |
| `report post` | `--task --phase plan\|build --round --status converged\|continuing\|blocked --summary --fields-file` (`--fields-file`: a JSON object on the closed `fields` schema; see [Node reports, digests and worktree files](#node-reports-digests-and-worktree-files-424)) | run, or node (own task) | write |
| `digest put` | `--task --file` (`--file`: a JSON object with exactly the `DIGEST_KEYS`, string values, #424) | run, or node (own task) | write |
| `msg post` | `--run R --to orchestrator\|node:T --kind K --payload-file F [--dedupe-key D]` | run (any recipient), node (`orchestrator` only) | write |
| `msg pop` / `msg ack` | `--run R --as … [--visibility S]` / `ID --receipt R` | the recipient's (a stopped node may still pop and ack; the receipt of a dead-lettered message's last delivery still acks it) | write |
| `wait` | `--run R [--timeout S] [--poll S]` | run | write |
| `node stop` / `node take-over` | `--task T` (`stop` on a stopped node is a noop) | run | write |
| `node hand-back` | `--task T --summary-file F` | run, or node (own task) | write |
| `lock acquire` / `lock release` | `--name integration:<branch> [--purpose] [--session-id] [--timeout S]` | run, or node (own deliverable, own session) | write |
| `round start` | `--task T --phase plan\|build [--head SHA] [--base SHA]` (`--head` required for `build`) | run, or node (own task) | write |
| `finding open` | `--task T --phase P --source S [--round R] --input F\|-` | run, or node (own task), or review session (a task of its scope, its own source) | write |
| `finding classify` | `ID --class must_fix\|should_fix\|nice_to_have\|out_of_scope [--note]` | run, or node (own task) | write |
| `finding state` | `ID --to open\|resolved\|rejected\|deferred\|settled [--reason] [--commit] [--cause ID]` | run, or node (own task) | write |
| `blast-radius set` | `--task T --value ok\|doubt\|too_large --head-sha S --review R [--reason]` | run | write |
| `review finish` | `--input F` (#407: the review session's hand-back, once) | review session | write |
| `tool <name>` | `[--task T] --key K [--args-file F]` (`--task` required by a task-bound tool, refused by a task-less one such as `review`: `USAGE`) | per tool | write |

Time flags (`--visibility`, `--poll`, `wait --timeout`) must be finite, above 0 and at most 86400 seconds; `lock acquire --timeout` may be 0 (try once). `wait --poll` must also be at least 0.05 and, when both are given, no more than `--timeout`. Anything else is `USAGE`.

`run open` and `run claim` start the daemon after the run is opened or claimed. A daemon start error is reported as `"daemon": "error"` with `daemon_error`; the run token is still printed.

### Exempt commands

- **The hooks.** On a schema mismatch or an internal error they fail open at once. The exceptions are DB-free: `pre-push`'s refusal of work-item refs, and `PreToolUse`'s DB-access rule, which denies even while the DB is missing, migrating or contended.
- **`hooks-settings` and `version`.** No DB at all.
- **`review read`** (#407). No DB, no token: one fixed `git` invocation in the main checkout (see [Reading code](#reading-code-review-read)).
- **`daemon`.** It migrates.
- **`ensure`.** It reads the daemon lease defensively.
- **`halt clear` and `restore`** (#406). The maintainer's commands: no token, since the DB may have no open run, or be lost. A newer DB is `SCHEMA_TOO_NEW` before anything else; see [The merge gate](#the-merge-gate-and-the-self-check) and [Backups and restore](#backups-and-restore).
- **The read commands** (`session status`, `snapshot`, `task show`, `ledger show`, `export-summary`):
  - a missing DB is an empty answer;
  - an older DB is `SCHEMA_PENDING` at once;
  - a newer DB is `SCHEMA_TOO_NEW`.

## Schema and self-migration

### Tables

`harness_state.db` holds, at schema v4 (v1: the tables below except the ledger, `alerts` and the review tables; v2: the finding ledger of #375; v3: the active loop's `alerts`, #406; v4: the review sessions, #407):

| group | tables |
|---|---|
| runs | `runs`, `run_leases`, `run_sessions` |
| task tree | `tasks` (with the work-item fields `deliverable_id`, `item_k`, `next_item_k`), `run_roots`, `work_list`, `task_deps`, `criteria`, `task_history` |
| nodes | `nodes`: one row per generation |
| queues and records | `messages`, `questions`, `decisions`, `digests`, `reports`, `integrations` |
| coordination | `locks`, `cap_slots`, `tool_calls`, `waiters`, `daemon_lease`, `hook_events` |
| active loop (v3, #406) | `alerts`: one row per (run, occurrence) |
| bookkeeping | `meta`, `counters` |
| the ledger (v2) | `rounds`, `findings`, `finding_events`, `blast_radius` (see [The ledger](#the-ledger-375)) |
| review sessions (v4, #407) | `reviews`, `review_tasks`, `review_triggers` (see [Review sessions](#review-sessions-407)) |

Migration v3 (#406) adds `alerts` and seeds `meta.hook_events_cursor` past the existing `hook_events` (so history is never replayed). Migration v4 (#407) adds the review tables and seeds `meta.review_pilot_cursor` past the existing `task_history`. The active loop also keeps these `meta` keys, written at runtime: `selfcheck`, `selfcheck_override`, `selfcheck_pending`, `last_backup_at`, `last_backup_error`.

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

Contention anywhere in a migration (the migrate lock, opening the DB, reading `user_version`, the backup, a step, `COMMIT`: any SQLite `BUSY` or `LOCKED` code, extended codes included) is `BUSY`. The daemon retries it (`MIGRATE_BUSY_RETRIES` times, one tick apart) and never records it in the backoff sidecar.

Before migrating an existing DB, the daemon backs it up to `backups.db_dir(db_path)` (a per-DB subdirectory of `QS_CP_BACKUP_DIR`, never inside a checkout; see [Backups and restore](#backups-and-restore)). A refused backup directory fails the migration, through the `migrate-error` sidecar.

### Migration recipe for later children (#375 and others)

1. **Append** `Migration(n + 1, "<name>", (<single statements>…))` to `MIGRATIONS` in `migrations.py` (v2 is #375's `Migration(2, "the finding ledger (#375)", schema_ledger.STATEMENTS)`).
   - Use single statements only, never `executescript`.
   - Keep the statements in a module named by what it holds (`schema_ledger.py`), so a renumbering at a catch-up leaves no wrong name.
   - A test pins the versions as `1..N`. The tests use `conftest.CUR` / `NEXT` (the current and a newer version), never a literal, so a renumbering needs no test edit.
2. **Land it on `main`.** The next `ensure` sees a lower-schema daemon lease, stops the old daemon (or SIGKILLs a hung one, see below; details in [The daemon and `ensure`](#the-daemon-and-ensure)), and starts one that backs up and migrates. If the old daemon cannot be proven gone after `DAEMON_RESTART_WAIT_S`, `ensure` answers `restart_pending` and starts nothing; `wait` then exits 5 with `restart_wait`.
3. **If the migration fails,** the daemon writes `<db>.migrate-error.json`. `ensure` does not respawn the daemon for `MIGRATE_BACKOFF_S`, and waiting commands get `SCHEMA_PENDING` with the error.

### The daemon and `ensure`

- **Its lease.** `daemon` takes a singleton `flock`, then reads its own start time. It retries `PID_START_RETRIES` times; if `ps` keeps failing it logs and exits (`"exit": "no_pid_start"`) rather than write a lease with a NULL `pid_start` that nobody could verify. Its final lease clear never masks the loop's result: a failure there is logged.
- **Its heartbeat.** The daemon beats its lease at the start of every tick and again after every tick hook, so a slow tick is never mistaken for a hung daemon. A tick hook must return within `STALE_AFTER_S`, or call `daemon.beat(conn, clock)` itself while it works: that contract is what keeps a healthy daemon from being SIGKILLed, since `ensure` kills only a daemon whose heartbeat is at least `STALE_AFTER_S` old.
- **A `BUSY` beat is skipped.** A heartbeat that hits contention is logged and skipped, and the loop goes on. `wait` does the same with a `BUSY` poll.
- **`ensure`'s answers.** `already_running` (a fresh lease at this schema), `started`, `restarted`, `migrate_failed` (inside `MIGRATE_BACKOFF_S` of a recorded migration error), `restart_pending`, `stale_alive`, and `newer_running` (a stale lease of a newer schema whose pid is not proven dead: nothing is spawned, since the new daemon would only exit on the held singleton; commands get `SCHEMA_TOO_NEW`).
- **Stopping the previous daemon.** `ensure` stops a daemon whose lease is older than this code's schema and that is not proven dead, and a daemon of this very schema whose lease is stale while its pid is proven alive with a matching start time (a hung daemon: a new one would exit at once on the singleton lock). A daemon of a **newer** schema is never signalled: `ensure` answers `newer_running` and spawns nothing while its pid is not proven dead, and commands get `SCHEMA_TOO_NEW`. A stale same-schema lease whose liveness is unknown is not stopped either: a new daemon is simply started.
  - A stale lease with no `pid_start` cannot be verified (its pid may be reused): it is never signalled or waited on, and a new daemon simply starts (it exits if the singleton is still held).
  - **The singleton `flock` is the authority on "gone".** The kernel drops it when the daemon exits, before its parent reaps it, so `ensure` checks it before any signal and inside both waits: free means the old daemon is gone (it exited, crashed, or is a zombie that `kill(pid, 0)` and `ps` still report alive), and the new daemon starts at once.
  - A daemon proven dead is not waited for.
  - SIGTERM goes only to a daemon proven alive whose identity is checkable (a recorded start time, or a fresh lease). A pid already gone at the signal is fine.
  - The wait ends as soon as the lease is cleared or another daemon's pid holds it, the singleton is free, or the pid is proven dead.
  - When `DAEMON_RESTART_WAIT_S` passes, `ensure` sends SIGKILL only if the old pid is still proven alive, has the same start time, and its heartbeat has not moved since the SIGTERM and is at least `STALE_AFTER_S` old; it then waits up to `KILL_WAIT_S` (same exits) and starts the new daemon.
  - Otherwise (liveness unknown with the singleton held, a heartbeat still moving, or a SIGKILL that did not take) it answers `restart_pending` for an older schema or `stale_alive` for the same one, and starts nothing. `wait` treats both as `restart_wait` (exit 5).
- **`restoring`** (#406). While a `restore` holds the DB (its `<db>.restoring` marker names a live pid), `ensure` answers `restoring` and spawns nothing, and a starting daemon exits at once (`"exit": "restoring"`). `wait` keeps polling. A marker whose pid is proven dead (a killed `restore`) is unlinked by the next `ensure`.
- **The spawned environment** (#406, D16). `ensure` starts the daemon with `strip_env(os.environ)`: every `CLAUDE_CODE_MESSAGING_*` name, `CLAUDE_CODE_SESSION_ID`, `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT` and the run token `QS_CP_TOKEN` are dropped, so neither the daemon nor anything it spawns carries the spawning session's messaging secret, identity or run token. A daemon started by hand keeps its own environment.

## The active loop (#406)

The daemon is the active loop: one Python process for every run. Each tick (`TICK_S`) runs the registered **tick hooks**, in this order:

| hook | cadence | does |
|---|---|---|
| `code_version` | every tick | restart on new Control Plane code ([Restart on new code](#restart-on-new-code)) |
| `selfcheck` | every tick, deciding from `meta.selfcheck` | the post-merge self-check that opens the merge gate |
| `detectors` | `DETECT_EVERY_S`; overlap every `OVERLAP_EVERY_S` | overlap, stalled nodes, rounds, state anomalies, cycles, duplicates ([Detectors](#detectors)) |
| `review_pilot` | `PILOT_EVERY_S` | closes dead reviews; launches the cross-run review (#407, [The review pilot](#the-review-pilot)) |
| `hook_route` | every tick | routes `hook_events` alerts to their run's orchestrator queue |
| `ci_watch` | its own schedule | `tasks.ci_state` / `ci_sha` for every open PR; `ci_red` |
| `liveness_watchdog` | `LIVENESS_EVERY_S` | `node_dead`, `orchestrator_dead`, and the wake-up ladder |
| `backup` | `BACKUP_EVERY_S`, from `meta` | periodic backups and `backup_failed` |

### Registering a tick hook

`ticks.register(name, fn)` (`ValueError` on a duplicate name); `fn(conn, clock)` gets the daemon's single long-lived connection. The built-in hooks are listed in `activeloop.register_builtin()` and `activeloop.BUILTIN_NAMES`; `cli` calls `register_builtin()` at import and in `main`, so a later child adds its hook to that list and to `BUILTIN_NAMES` (child 15 added `review_pilot`), and the self-check sees a missing one. A hook takes its side effects from `activeloop.seams()` (`Seams`: `runner`, `probe`, `claude`, `main`, `github`, and `popen`, the review pilot's detached launcher, `None` when absent), built once per daemon. **The contract:**

- never let `STALE_AFTER_S` pass without a beat;
- give every subprocess a timeout of at most `HOOK_SUBPROCESS_S`, and call `daemon.beat(conn, clock)` after it;
- run subprocesses and beats outside any `db.write`: read, release, call, beat, write;
- bound the work per tick, never launch a long job in-process, and throttle with a `ticks.Throttle` or a DB timestamp.

A hook that raises is logged and skipped. A SIGTERM is honoured after the current hook; a restart request at the end of the tick.

### Alerts

An alert is raised **once per occurrence**: `alerts.sync_locked(conn, clock, kinds=…, active=[Condition(kind, subject, run_ids, payload)])` inserts one `alerts` row per (open run, kind, subject) and posts one orchestrator message whose `dedupe_key` is the fingerprint `<kind>:<sha256(subject)[:16]>:<n>`. While the condition holds, later syncs refresh the row (`last_seen`, `payload`) and post nothing; when it stops holding, the row is cleared, silently; a recurrence is a new occurrence (`n + 1`). A condition raised again per value carries the value in its subject (a CI sha, a round, a lock's `acquired_at`). A hook syncs only the kinds whose input it knows this tick, so an unknown input clears nothing; **cold in-memory state counts as unknown**, so a daemon restart is never a recurrence. `alerts.event_locked` records a one-shot alert (cleared at once, posted once).

Kinds (`alerts.KINDS`, each 1–32 characters; `CROSS_RUN_KINDS` for child 15): `overlap` / `overlap_cross_run`, `node_stalled`, `too_many_rounds`, `task_without_node`, `dependency_violated`, `lock_held_long`, `gate_slot_held_long`, `leftover_item`, `leftover_scratch`, `dependency_cycle` / `dependency_cycle_cross_run`, `duplicate_task` / `duplicate_task_cross_run`, `ci_red` (severity `must-fix`; every other kind is `alert`), `node_dead`, `orchestrator_dead`, `orchestrator_not_listening`, `selfcheck_failed`, `backup_failed`, `review_failed` (#407), and the routed hook kinds (`queue_not_draining`, `idle_without_wait`, `merge_state_conflict`, any other 1–32-character kind that is not daemon-owned, else `hook_alert`). `ROUNDS_ALERT` is 5.

**The read path.** `snapshot["alerts"]` lists every open alert row, then the `ALERTS_LIMIT` newest cleared ones (after the `--run` filter). The raw `hook_events` alerts are in `snapshot["hook_alerts"]`, so an unroutable one stays visible. `hook_alerts` is global: `--run` does not filter it, because `hook_events` has no run column and a raw event may name no run at all.

**Hook routing.** `meta.hook_events_cursor` (seeded by migration v3, so history is never replayed) marks the last event read. Each tick routes the next `alert` events to their run (`detail.run_id`, else the run of `detail.task_id`) as one-shot alerts with subject `hook:<id>`, and moves the cursor, routable or not. Each event is routed inside its own SAVEPOINT: one that raises is rolled back alone, logged, and skipped, so it never blocks the cursor. A transient DB failure (`sqlite3.OperationalError` such as a full disk or an I/O error, or `BUSY`) is not the event's fault: it rolls the whole batch back, the cursor stays, and the next tick retries. A cursor that is not an integer, or is out of range (negative, or beyond SQLite's 64-bit integers; a hand edit), is reseeded the way migration v3 seeds it, past the existing `hook_events` (logged); routing goes on from the next new event. An event whose `kind` is daemon-owned (`alerts.KINDS - ROUTED_HOOK_KINDS`, e.g. `ci_red` or `selfcheck_failed`) is routed as `hook_alert`, so a hook can never pose as the daemon or claim `must-fix`.

### Restart on new code

`cli._daemon` first reads `codever.loaded_version()`: the sha256 of `scripts/qs/control_plane/**/*.py`, `cp.py`, `models.py` and `targets.py` in the main checkout, or `unverified:<hash>` when a hashed file changed since the package was imported (a future mtime is logged once and ignored). The `code_version` hook recomputes the disk version every tick; a change seen on **two consecutive ticks**, with no git operation in progress in the main checkout (`index.lock`, `MERGE_HEAD`, `rebase-merge`, `rebase-apply`), asks the daemon to restart. An `index.lock` older than `GIT_LOCK_STALE_S` is a crashed git's leftover: it is ignored (logged once), so it never freezes the restart or the overlap detector; `MERGE_HEAD` and a rebase stay "in progress" however old (the merge gate's `stuck` state names them, and a stale lock too). It finishes its tick, exits `new_code` (lease cleared, `flock` released), and `cli._daemon` calls `ensure`, which starts the new code (`"respawn"` in the result; an `ensure` error is `"respawn": "error"` and the next `wait` retries). Commands and hooks load new code at their next call; a running `wait` keeps its old code until a schema change stops it.

### The merge gate and the self-check

**Automatic merges open only after a self-check pass for the code on disk.** `tool merge`'s `policy` step asks `mergegate.merge_allowed` before the merge policy (after the probe, before `gh pr view --json headRefOid` and `gh pr merge`):

| gate state | `tool merge` |
|---|---|
| `open`: a pass, or a maintainer override, for this code version | the merge policy, as before |
| `pending`: not checked yet | `BUSY`, "replay the same key shortly" (routine for about a minute after a Control Plane merge) |
| `retrying`: failed fewer than `SELFCHECK_ESCALATE_AFTER` times | `BUSY`, with `next_retry_at` |
| `failed`: `SELFCHECK_ESCALATE_AFTER` failures | `POLICY_REFUSED`, "ask the maintainer" |
| `stuck`: pending for more than `PENDING_ESCALATE_S` (the daemon may not start) | `POLICY_REFUSED`, naming a leftover git lock file when there is one, "ask the maintainer" |

The `selfcheck` hook runs while the daemon's loaded code is the disk code: `schema` (the version and `PRAGMA quick_check`), `entry` (`<MAIN>/venv/bin/python cp.py version`, which must list every built-in tool and tick hook) and `registry` (the same, in-process, plus `cli.build_parser()`). Any exception is a recorded failure. A failure is retried every `SELFCHECK_RETRY_S`, forever (the throttle is `meta.selfcheck.at`, so a new daemon does not retry early), and `selfcheck_failed` reaches every open run from the first failure; a pass, an override or a version change clears it.

**Clearing a halt.** The maintainer asks the orchestrator, which runs `cp.py halt clear --reason TEXT`. The `PreToolUse` hook answers `ask`, so the app shows the maintainer an approval prompt and nothing runs before he approves (see [Hooks](#hooks)). The override applies to the current code version only (a revert to that exact version gets it back: same code, same decision); it also clears `selfcheck_failed`. The refusal text never names the command, so an orchestrator escalates instead of clearing on its own.

**The import-failure limit.** A version that cannot start records no pass: merges are `BUSY`, then `stuck` after `PENDING_ESCALATE_S`. If that version also carries a migration, every write command (and `halt clear`) answers `SCHEMA_PENDING`, with the migrate-error sidecar, before reaching the gate. The fix is to revert the code.

### Detectors

Every `DETECT_EVERY_S`, for every run and across runs. Times are compared as ISO strings against `now - threshold`, so a non-ISO stamp is never "older".

- **Overlap:** non-terminal tasks with a resolvable local branch; a deliverable against the newer of `main` / `origin/main` (`origin/main` when they diverge), a work item against `QS_<N>`; `git diff --name-only base...tip` per pair (never an item with its own deliverable), and the conflicting files from `git merge-tree --write-tree --name-only --no-messages` (**git ≥ 2.38**; it writes loose tree objects into the main repository, which `git gc` collects). At most `OVERLAP_MAX_CALLS` git calls per tick: a longer walk continues on the next runs, re-emitting the last complete result meanwhile; skipped while a git operation is in progress. A failed `git diff` for one branch (no merge base: an orphan branch, a shallow clone) skips that branch, logged once per failure episode, and keeps its last overlap alerts as they were (unknown, not cleared) while every task they name is still a live branch; a failed `merge-tree` reports `conflicts: null` for that pair. Either failure is not retried for the rest of the walk (a walk resumed on the next run skips it). A failed `rev-parse` of one task's branch (or of a work item's deliverable branch) is that task's unknown the same way, logged once per task and failure episode, and is not retried for the rest of the walk either. `main` / `origin/main` are resolved first, before any task, so a repo-wide failure (every git call exits 128: dubious ownership, a corrupt ref store) stops the walk on its first call instead of marking each task unknown. A git timeout or a missing git (the runner's exit 124 / 127) stops the whole walk instead, like a failed `rev-parse` of `main` / `origin/main` or a failed ancestry check: both kinds are left out on that tick, the last complete result (if any) is re-emitted on the ticks in between, and the next walk waits `OVERLAP_EVERY_S`. Any other error escaping the walk (a failed lease beat) resets the walk's state, so the next walk retries every key, and also waits `OVERLAP_EVERY_S`, so a persistent failure is not re-walked every tick. A complete walk prunes the diff and merge-tree caches to the keys it used. A fresh daemon takes its "last alerts" from the open overlap alerts, so a restart with an unknown branch is never a recurrence. A detector that raises is logged once per exception type until it succeeds again.
- **`node_stalled`:** a `running` node with no tool call in flight and no activity (launch, report, message, tool call) for `STALL_S`. A `started` tool call whose holder is dead (a SIGKILLed `cp.py`) is not in flight.
- **`too_many_rounds`:** a phase past `ROUNDS_ALERT` rounds; each new round is a new occurrence.
- **State anomalies:** `task_without_node` (`ORPHAN_GRACE_S`); `dependency_violated` (a started, non-terminal task whose dependency is neither `merged` nor `validated`: a dependent starts only after its dependency merged); `lock_held_long` (`LOCK_HELD_*`, routed to the holder's run); `gate_slot_held_long` (`GATE_SLOT_HELD_S`); `leftover_item` and `leftover_scratch` (a finished deliverable's item worktree or `QS_<N>_<k>_integration` scratch; a failed `git worktree list` leaves `leftover_scratch` out, logged once per failure episode).
- **`dependency_cycle`** (Tarjan over every `task_deps` row, across runs) and **`duplicate_task`** (the same normalised title).

Each detector runs and syncs in its own `try`: one that raises is logged, its kinds are left out (nothing cleared), and the others still run.

### The CI watcher

Every deliverable with a PR, not terminal, in an open run, through one `gh api graphql` call per `CI_BATCH` PRs. A poll writes `tasks.ci_state` (`pending | green | red`) and `ci_sha` on a change only, and only when every batch returned data; a `MERGED` or `CLOSED` PR gets `NULL`. `ci_red` (must-fix) is derived from the stored rows on every tick: once per red sha. The schedule is `CI_FAST_S` while a check is pending or a head moved, else `CI_SLOW_S`; failures (a missing `gh` included) back off from `CI_BACKOFF_MIN_S` to `CI_BACKOFF_MAX_S`; below `CI_RATE_FLOOR` remaining calls it waits for GitHub's `resetAt`, believed at most `CI_RESET_CAP_S` ahead (a stamp without an offset is UTC; one that cannot be read waits `CI_SLOW_S`). A new daemon polls at once. An origin that is not a GitHub remote is logged once. A `null` check context is skipped, and a PR whose answer cannot be parsed is unknown for that poll (no write, logged once until it parses again) instead of failing the batch. A red row with no `ci_sha` gets the subject `<task>@-`.

### Liveness and the watchdog

`claude agents --json` (it lists desktop sessions too, with their idle status) feeds `nodes.refresh`, `node_dead` (a task's current node `reaped` on `LIVENESS_CONFIRM` listings in a row) and `orchestrator_dead` (a lease session absent from `LIVENESS_CONFIRM` listings). A failed listing changes nothing.

**What a listed `done` means** (#424, live on 2026-10-18 with Claude Code 2.1.302, throwaway haiku `--bg` sessions):

| # | check | observed |
|---|---|---|
| L1 | a `--bg` session idle between turns (process alive) | listed: `status: idle`, `state: done`, `pid` set |
| L2 | the same session during a turn | `status: busy`, `state: working` |
| L3 | a session that ended (`claude stop`) | **absent** from `claude agents --json`; only `--all` lists it (`status: null`, `state: done`, `pid: null`) |
| L4 | a bare `claude --bg --resume <full id> "<msg>"` of an ended session | the same session id, listed `busy/working` then `idle/done` ("woke session … with its saved options") |
| L5 | a bare `--resume` of a session still listed | Claude Code **forks a copy** (a new session id, the same name); `tool resume` already refuses a listed node (`INVALID_STATE … is live`) |
| L6 | `effortLevel` through `--settings` on a `--bg` session | it **reaches** the session (the banner shows the effort) |
| L7 | what a bare resume of an ended session keeps | every launch option: `-n`, `--agent`, `--permission-mode`, `--settings` (its hooks fire again), `--add-dir`, `--model`; the model and the effort held |
| L8 | `--add-dir <dir>` followed directly by the prompt | **the prompt is lost**: `--add-dir <directories...>` takes several values and swallows it; the session starts idle with no prompt. `tool spawn` puts `--add-dir` before `--permission-mode` (#424 D2) |
| L9 | `kill -9` of a listed session's pid | absent from the default listing within 5 s, as in L3 |

So a listed `done` means **the turn finished**, whatever the kind of session; the default listing (what `ClaudeCli.agents` reads, no `--all`) never lists an ended or killed session. For a node, `done` is not "ended", and the five sites stay as they were: `nodes.refresh` (the only one that reads `status` / `state`) maps `idle` in either field to `idle` and anything else listed to `running` (an unexpected pair, `busy/done` or `null/null`, reads live: the fail-safe side); `tool resume`'s live check, `locks.admit_node` and `tool spawn`'s late-launch adoption test presence only; `locks.session_holder_dead` tests presence by session id. For a single-turn review, "turn finished" is "review over" (`reviewpilot._session_move`, #421). A finished review's process stays alive, though: stopping it is #423. An ended node is resumed by its full id (`tool resume`), never respawned. Trust inheritance from a parent folder was seen live by #421 (fresh `cp-reviews/<db>/V<n>` folders under a once-trusted `~/.local/state/quiet-solar`); the post-merge run of #424 (AC9) confirms it for a new `QS_<N>` under `quiet-solar-worktrees`.

**Waking a stalled orchestrator.** A run is stalled when its head message has waited `WATCHDOG_S`, no `wait` is live, and its lease session is listed idle. Then:

1. **the messenger:** one `claude --bg -n qs-wake-<run>-m<msg> --model <models' fast class> --permission-mode auto --allowedTools=SendMessage "<prompt>"`, in `QS_CP_MESSENGER_DIR` (0700, outside every checkout), which sends the session named after the run a wake text (never a token). It is recorded as a `tool_calls` row `('watchdog-messenger', 'msg:<id>')`, so it is launched once per head, and capped at one launch per `MESSENGER_TTL_S`: with N runs stalled at once, the last waits about (N - 1) × 5 min. **This is the one documented exception to "only a task's node is spawned".**
2. **`orchestrator_not_listening`:** when the run is still stalled `WAKE_RETRY_S` after the launch, at once when the launch failed, or when the run name is not bound to its lease session (`run bind-name`).

A launch never leaves its row `started`: an unusable `QS_CP_MESSENGER_DIR`, or messenger arguments that cannot be built (the `models` import, the `fast` class), fails the launch before the claim, and any failure after the claim closes the row as `failed` (the error text in `result`), while a spawn that ran is recorded as such even if the beat after it fails (the lease is beaten after every spawn attempt, failed or not); a `started` row older than `MESSENGER_TTL_S`, or whose holder is dead, is failed too. The messenger's environment never carries the session identity or the messaging variables (`daemon.STRIPPED_ENV`, `CLAUDE_CODE_MESSAGING_*`) nor the run token, even when the daemon was started by hand inside a session. Its wake text names `<main>/venv/bin/python`, or the daemon's own interpreter when that venv is missing, both paths shell-quoted.

Verified on 2026-10-09 (QS-406 T1): the messenger wakes an idle desktop session. The orchestrator must be addressable by the run's name: `/rename <run name>` sets it (the app's sidebar rename does not). The inbox socket (`$CLAUDE_CODE_MESSAGING_SOCKET`) is **not** used: its frame format is undocumented and the frames tried were not delivered; that rung is a follow-up issue. The full spike results (folder trust in the messenger directory, what a `SessionStart` hook inherits, the `!` pane's missing variables) are in the T1 table of [QS-406's story](../stories/QS-406.story.md#t1--the-inbox-spike-run-2026-10-09-with-the-maintainer-desktop-app-claude-code-21285).

### Backups and restore

**Periodic backups.** While a run is open, the `backup` hook copies the DB every `BACKUP_EVERY_S` to `backups.db_dir(db_path)` = `QS_CP_BACKUP_DIR/<sha256(db path)[:8]>/harness_state.periodic.<stamp>.db`. Every copy (periodic, migration `v<N>`, a restore's `replaced`) is written to a `.partial`, fsynced and renamed; the directory is then fsynced too, so the rename survives a crash (best-effort: a filesystem that refuses a directory fsync, such as FUSE, 9p or SMB, is logged once and the copy still counts), 0600 in 0700 directories (a user-owned 0755 `cp-backups` left by #399 is tightened; a foreign one is refused), with `journal_mode = DELETE`. Rotation keeps every periodic copy of the last 24 h, then the newest per UTC day for 7 days (a copy of the last 24 h counts as its day's newest), and deletes a `.partial` whose stamp is older than `BACKUP_EVERY_S` (a crashed copy's leftover, a restore's staged copy included). A file whose stamp is not a real date is ignored by rotation and by `restore`. A failed rotation is logged only: the copy was taken. `v<N>` and `replaced` copies are never deleted by code: delete old ones by hand. A failed backup is `backup_failed` for every open run until the next success, retried every `BACKUP_EVERY_S` (a `meta.last_backup_error` that is not an object is retried at once). #399's backups at the top of `cp-backups` are left alone; move one into the DB's subdirectory to make it restorable.

**Restoring.** The maintainer asks the orchestrator, which runs `cp.py restore --confirm` behind the same `ask` prompt as `halt clear`. It restores the newest `periodic` or `v<N>` copy **in place** (SQLite's backup API: the file and its WAL stay, open connections see the restored content), after:

- checking the live DB (newer → `SCHEMA_TOO_NEW`; unreadable → `CONFLICT`, "move it aside, then retry"), the backup (none → `NOT_FOUND`; failing `quick_check`, newer, another page size, or missing a `waiters`, `daemon_lease` or `meta` table → `CONFLICT`) and the tool calls (one in flight → `BUSY`) — a refusal here touches nothing;
- writing the `restoring` marker, stopping the daemon (`daemon.stop`: `not_running`, `stopped`, or `signalled` while it finishes a tick) and taking its `flock` and the migrate lock, within one `RESTORE_LOCK_WAIT_S` deadline (`BUSY` otherwise, "the daemon may be finishing a tick; retry");
- keeping the live DB as `harness_state.replaced.<stamp>.db` (the undo): any non-empty file at the DB path is kept, migrated or not; only a missing or empty one is not. A copy that cannot be written is a `CONFLICT` refusal, nothing restored and the live DB untouched; its hint follows the error: "free space" for an `OSError` or SQLite's `SQLITE_FULL` / `SQLITE_CANTOPEN` ("database or disk is full"), "retry in a moment" for `SQLITE_BUSY` / `SQLITE_LOCKED`, and a plain "could not be copied" (no free-space claim) for any other SQLite error; only a live DB SQLite reports as corrupt gets the "move it aside" hint.

The live `waiters` rows and the `selfcheck` / `selfcheck_override` / `selfcheck_pending` keys are carried across (they describe the processes and the code on disk, not the data), the daemon lease is cleared, `last_backup_error` is dropped (neither the backup's nor the live one is kept: a restore starts a fresh backup cycle, so a copy taken right after a failed attempt does not bring `backup_failed` back), and every exit path removes the marker, releases the locks and calls `ensure`. That fix-up runs on a **private staged copy** of the backup (`harness_state.restoring.<stamp>.db.partial`, 0600, removed on every exit), in one transaction; the finished copy is then written into the live DB in **one** step, the restore's only write to it. A failure staging the backup (a full disk is the likeliest: a restore needs about twice the DB's size free) or of the fix-up is an `INTERNAL` refusal with the live DB untouched, never a "move it aside" hint (a policy refusal while staging, such as a backup directory owned by another user, keeps its own code, `POLICY_REFUSED`, and carries `replaced` too; its own hint, if any, comes first, then "nothing was restored" with the `replaced` path); a failure of that final write is an `INTERNAL` error saying the live DB may hold part of the backup. `replaced` (in the error, as in every error raised after the live DB was kept) points at the pre-restore copy: that is where the pre-restore data is, whatever a retry does. **Everything recorded after the backup is lost; run tokens issued since then are stale (each orchestrator must `run claim` again);** the next ticks re-sync CI and liveness, and an older-schema backup is migrated by the daemon that starts. To restore an **older** backup, move the newer files aside first. While a restore holds the migrate lock, read commands and hooks see `SCHEMA_PENDING`, or fail open. **An accepted limit:** ordinary `cp.py` tool writes do not check the `restoring` marker, so one accepted after the `replaced` copy is written and before the final write lands in neither that copy nor the restored DB; restore is a rare, maintainer-confirmed operation, so this window is not closed.


## The ledger (#375)

The finding ledger is the shared, durable record of the review findings of both loops (plan and build) of the second pipeline's nodes (6b). It **recognises, records and flags by code**; the node judges. It refuses only malformed input (`USAGE`), a token used out of its scope (`CONFLICT`), and a red-CI finding off the task's red sha (`INVALID_STATE`). Nothing is refused to force convergence. Code: [`ledger.py`](../../scripts/qs/control_plane/ledger.py), schema [`schema_ledger.py`](../../scripts/qs/control_plane/schema_ledger.py); design: [`docs/stories/QS-375.story.md`](../stories/QS-375.story.md).

### The tables

| table | one row per | notes |
|---|---|---|
| `rounds` | review pass of one phase of one task: `(task_id, phase, round)` | `base_sha..head_sha` is the round's diff. It shares the natural key `(task_id, phase, round)` with its `reports` rows, with no foreign key |
| `findings` | finding | its `source`, `reviewer`, `severity`, `classification`, `category`, `title`, `body`, `file` / `symbol` / lines, `fingerprint`, `replay_key`, `matched_id`, `flags` (JSON), `unchanged_lines`, `state`, `decided_seq`, `reason`, `resolved_sha`, `ci_sha`, `integration_id`, `actor` |
| `finding_events` | `classify` or `state` change | `from_value` / `to_value`, `reason`, `commit_sha`, `cause_id`, the phase's latest `round` at the time, `actor`. A finding's opening is its row, so there is no `open` event |
| `blast_radius` | rating, append-only | `value` (`ok\|doubt\|too_large`), `head_sha`, `review` (the id of the review it came from), `reason`, `actor` |

The vocabularies live in `schema_ledger`: `PHASES` (`plan`, `build`), `SOURCES` (reviewer-type `reviewer`, `coderabbit`, `global_review`, `cross_run_review`; authority `ci`, `gate`, `detector`, `orchestrator`, `maintainer`), `SEVERITIES`, `CLASSES` (the severities plus `out_of_scope`), `STATES` (`open`, `resolved`, `rejected`, `deferred`, `settled`) and `CATEGORIES` (`correctness`, `edge-case`, `test`, `security`, `performance`, `design`, `scope`, `docs`, `style`, `ci`, `gate`, `other`). `CATEGORIES` is checked in Python, not by SQL, so a later child extends it without a migration.

### Rounds

`round start` numbers rounds `1, 2, …` per `(task, phase)` and answers `{"ok": true, "task_id", "phase", "round", "base_sha", "head_sha"}`.

- **The base.** Round 1 takes the node's `--base` (NULL allowed): `git merge-base origin/main QS_<N>` for a deliverable, the fork point on the local `QS_<N>` for a work item. A later round defaults to the previous round's head; after a catch-up with `main`, the node passes `--base <the catch-up merge commit>`, so `main`'s changes stay out of the delta.
- **`--head`** is required for `build` rounds, optional for `plan`.
- **What is not a round:** a catch-up's conflict-resolution delta. Every `round start` call is a new round (no idempotency key); a restarted node, or one that lost `round start`'s answer, reads `latest_round` and its head through `ledger show` before calling it again.
- `--head` and `--base` are stripped; a blank one is absent.

### Opening findings

`finding open` reads one JSON object, or a non-empty list, from a file or from stdin (`--input -`), and opens it in one transaction. It answers **only the ids**, one per item in order: `{"ok": true, "ids": [...]}`. `--round` defaults to the phase's latest round, or `0` before any round; a given `--round` must be `>= 1` and exist (`NOT_FOUND`).

| key | type | required |
|---|---|---|
| `severity` | `must_fix\|should_fix\|nice_to_have` | yes, except `ci` / `gate` (overridden to `must_fix`) |
| `category` | one of `CATEGORIES` | yes |
| `title` | string holding at least one letter or digit | yes |
| `body` | string | yes |
| `file`, `symbol` | string; `file` repo-relative and normalised (NFC, each segment stripped, `a//b/ ./c` → `a/b/c`; absolute, a drive letter, `..`, a backslash or a `:<line>` suffix is `USAGE`: lines go in `line_start` / `line_end`), `symbol` trimmed and NFC | no |
| `line_start`, `line_end` | int `>= 1`; `line_end` defaults to `line_start` | no |
| `reviewer` | string, trimmed and NFC (the lens, e.g. `qs-review-blind-hunter`) | no |
| `unchanged_lines` | bool, recorded as given (`1` / `0` / NULL) | no |
| `relates_to` | list of finding ids, de-duplicated (`NOT_FOUND` if one is unknown, or of another run) | no |
| `sha` | string | `ci` only, required there |
| `integration_id` | int | `gate` only, required there |

An unknown key, a wrong type, a value off its vocabulary, `line_start > line_end`, `line_end` alone, a bad path, an integer beyond SQLite's range, a lone surrogate in any string, the separator U+001F in `title`, `file`, `symbol`, `reviewer` or `sha`, non-JSON or non-UTF-8 input (file or stdin) and an empty object or list are all `USAGE`. The optional key fields (`file`, `symbol`, `reviewer`, `sha`) are stripped, and a blank one is absent; `title` and `body` are stored as given.

- **`ci` findings** need `sha` equal to `tasks.ci_sha` while `tasks.ci_state = 'red'`, else `INVALID_STATE`. They are stored with `ci_sha`, `must_fix` / `must_fix`.
- **`gate` findings** need `integration_id`, an `integrations` row whose item or deliverable is the task (`NOT_FOUND` / `CONFLICT`), and are stored `must_fix` / `must_fix`.
- **Exact replay.** `replay_key` is a `sha256` over task, phase, round, source, `reviewer`, the fingerprint, the normalised title, the lines, `ci_sha` and `integration_id` (the round is left out for `ci` and `gate`). A replayed item returns its existing id, even after a new round started or the CI turned green; its other fields (`severity`, `body`) are ignored: **the first write wins**. Two findings on one symbol, a second reviewer and a red CI on a new sha are never merged.

### Matching, flags and the born state

- **The fingerprint** is `sha256(file␟symbol␟category)` (`␟` is U+001F, NULL encoded as ""), plus `␟norm(title)` when both `file` and `symbol` are empty. `norm` applies NFKC, case-folds, and folds every run of non-alphanumerics into one space. Task and lines are left out.
- **The family:** the task's deliverable (or the task itself) and every work item of that deliverable. Matching looks across the family and across phases; inside a batch, item 2 can match item 1.
- **Flags** are computed at open and stored on the row: `{"kind", "id", "state", "reason", "commit", "source", "strong", "cross_task", "escalated"}`. They record what was true at open; `ledger show` adds each flagged row's `current_state`.
  - `matches`: an earlier family row with the same fingerprint. `strong` when the normalised title is equal too; `escalated` on a strong match when the new item is `must_fix` and the flagged row's effective class (`classification`, else `severity`) is not.
  - `overlaps_fix`: an earlier family row, `resolved` or `settled`, with the same `file` and the same non-empty `symbol` but another fingerprint: the code-detected flip-flop candidates.
  - `relates_to`: the rows the item names. This is how a reviewer holding the ledger excerpt, or the orchestrator opening a CI finding, points at a settled or resolved row that has no shared symbol.
  - `cross_task` is true on any kind whose row is on another task; `strong` and `escalated` are false off `matches`.
- **"Latest"** is the most recent state decision: the highest `decided_seq`, a monotonic number drawn from `counters` (kind `finding_decision`, `db.next_seq`) at birth and at every state change, never on a state-preserving classify, a noop or a replay.
- **`matched_id`** is the first hit of: the latest same-task strong match, the latest family strong match, the latest weak match.
- **Born closed.** A reviewer-type item whose latest **same-task** strong match is `rejected` or `settled` is born in that state, whatever its severity, with the reason `matches #<id>: <reason>` (a reason already starting with `matches #` is reused verbatim, so it names the root and never nests). It stays closed until the node reopens it.
- **Born open:** everything else, with its `reason` NULL and the earlier rationale in its flag: a weak match, a cross-task strong match with no same-task one (the node re-assesses it in the deliverable's wider context; once decided, that decision is the task's own), a same-task strong match that is `open`, `deferred` or `resolved`, an overlap or a relation alone, and **every** authority source (`ci`, `gate`, `detector`, `orchestrator`, `maintainer`). A red CI check reopens a settled decision, never the reverse.

### States and classification

- **`finding state`**: free transitions between `open`, `resolved`, `rejected`, `deferred` and `settled`. `resolved` needs `--commit`; `rejected` and `settled` need `--reason`; a reopen (`open`) and `deferred` take an optional `--reason` (blank is none); `--commit` on any other target is `USAGE`. `--cause <id>` (any target, `NOT_FOUND` if unknown or of another run, `USAGE` if it is the finding itself) records why, e.g. the red CI finding that reopened a settled row. The same state is a noop (`"changed": false`, no event), checked after the arguments and `--cause`. The row holds only the latest state (`reason` the new `--reason` or NULL, `resolved_sha` set on `resolved` only); the history is in `finding_events`. Answers `{"ok": true, "finding_id", "state", "changed"}`.
- **`finding classify`** records the class in any state, with a `classify` event (its reason is `--note`), and drives open ⇄ deferred: an open row classed `nice_to_have` or `out_of_scope` moves to `deferred`; a deferred row classed `must_fix` or `should_fix` moves back to `open` (a reopening, so it blocks again). A move also writes a `state` event and sets the row's `reason`, both `classified <class>`. The same class is a noop. Answers `{"ok": true, "finding_id", "state", "classification", "changed"}`. At birth `classification` is NULL, except `ci` / `gate` (`must_fix`).
- **Deferred rows** stay in `findings`, with their class: `ledger show --state deferred`, `task show`, `snapshot` and the exported summary show them, and the run's final review (child 13) reads them.

### Who writes

`ledger.writer(conn, token, task_id, *, kinds)` is the one check, the point child 15 extends: a node token writes on its own task, a run token on any task of its run (another run's task is `CONFLICT`). The arguments are validated first (`USAGE`); then the ID-taking commands check the token, then the finding (`NOT_FOUND`), then its task's scope, so a stale token is `STALE_TOKEN`, never `NOT_FOUND`. `relates_to` and `--cause` read only findings of the writer's run (`NOT_FOUND` otherwise). The CLI flags (`--head`, `--base`, `--commit`, `--reason`, `--note`, `--head-sha`, `--review`) are stripped; a blank optional one is absent, but a blank `--commit` on another target than `resolved` is still `USAGE`. A task with no run is written with a run token and records the principal's `run_id`. Every row and event records its `actor` (`orchestrator` for a run token, `node:<task>` for a node token); `source` is recorded as given. `blast-radius set` takes a run token only: a node never rates its own blast radius, which feeds the merge's second human gate. A **review session token** (#407) writes `finding open` only, on a task of its scope in any run, with its own source (`global_review` / `cross_run_review`, another is `CONFLICT`); each item's `reviewer` is the review's id (a given one is replaced), the actor `review:V<n>`, and `relates_to` stays in the task's run. Its blast-radius ratings come through `review finish` (`ledger.append_blast_radius_locked`, shared with `blast-radius set`).

### Convergence

`ledger.convergence(conn, task_id, phase)` → `{converged, consistent, blocking, latest_round, counts}`, shown under `convergence: {plan, build}` by `task show` and `ledger show`.

- **`converged`**: the node has declared it. `latest_round >= 1` and the latest `reports` row (highest id) of `(task, phase, latest_round)` has `status = 'converged'`; a report posted with the run token counts too. A round still in review never reads as converged, a later non-converged report of the same round withdraws the declaration, and a report counts only when its `phase` is exactly `plan` or `build` — since #424 `report post` refuses any other phase (`USAGE`), so a misspelled one is never stored.
- **`blocking`**: the task's rows of that phase that are `open` with an effective class of `must_fix` or `should_fix` (round-0 rows and unclassified should-fixes included; an open nice-to-have or `out_of_scope` row never blocks).
- **`consistent`**: `blocking == []`. A node that declares convergence with blocking rows still open is converged and **not** consistent; nothing overrides it, so a flip-flop left open (for example while it waits for another perspective) never stalls the loop by code. The orchestrator, child 7's merge conditions and child 17 read the discrepancy.
- No hard round stop here: `ROUNDS_ALERT` stays child 14's.

`ledger.convergence` on an unknown task is `NOT_FOUND`.

**`blast-radius set`** appends a rating and answers `{"ok": true, "id", "task_id", "value", "head_sha"}`. `ledger.blast_radius(conn, task_id, head_sha=None)` is child 7's reader: the latest rating (`head_sha` omitted), or the latest for that sha, else `None`; a given but blank `head_sha` is `None`, never another head's rating.

### Reading the ledger

`ledger show --task T [--phase P] [--family] [--state S[,S…]]` answers `{"ok": true, "task_id", "rounds", "findings", "finding_events", "blast_radius", "convergence": {"plan", "build"}}`:

- `findings` are the rows (flags parsed), each flag with the flagged row's `current_state`; `finding_events` are those rows' events;
- `--phase` filters `rounds` and `findings`; `--state` filters `findings` (an unknown or empty state list is `USAGE`); `--family` widens `findings` and their events to the family;
- `rounds` and `blast_radius` (every rating, oldest first) stay per task; no DB is `NOT_FOUND`, as for `task show`.

`task show` carries the same ledger keys (`rounds`, `findings`, `finding_events`, `blast_radius`, `convergence`), unfiltered, next to its `task`.

### The focused-reviewer contract

**What a reviewer receives** in its prompt from the node:

- **the diff:** `git diff <base>..<head>` from `round start`'s answer (round 1 is the full integrated diff; later rounds are the delta);
- **the acceptance criteria:** `task show` → `criteria`;
- **the ledger excerpt:** for example `ledger show --task T --phase P --family --state open,rejected,settled,deferred` (6b may tune the filter), so it does not raise them again. `--family` widens only `findings` and their events to the family; `rounds`, `blast_radius` and `convergence` stay per task. A re-raise of a work item's closed row on the deliverable is born `open`, `cross_task`, and blocks until the node triages it;
- **the write coordinates:** `--task`, `--phase`, `--round` and the node's `--token` (the reviewer runs inside the node's session, so a superseded node's reviewers get `STALE_TOKEN` too);
- **its instructions:** apply a relevance-to-this-task test; suggest no refactor outside the scope; in a later round raise no finding on lines the round did not change (a red CI is the exception); set `unchanged_lines` from the diff; put the ids of the excerpt rows a finding touches in `relates_to`. These are instructions only: the ledger records and refuses nothing for them.

**What it writes:** one heredoc call, `finding open --source reviewer --input -`, with the items above. **What it returns** to the node: only the ids. It calls no other ledger command (the token would allow `finding state` / `finding classify` on its task: an accepted risk, forbidden by this contract and by 6b's reviewer templates, with every write's actor recorded; a reviewer-scoped token is 6b's / 15's).

**Then the node:**

1. reads the rows and flags with `ledger show`, the `escalated`, `cross_task` and weak-match flags first;
2. triages them (`finding classify`, `finding state`). On a flip-flop it may ask the orchestrator for another perspective (`question open` or `msg post`) instead of deciding alone; that decision lands as `finding state --to settled` with its rationale, written with the run token (actor `orchestrator`);
3. reads `convergence.blocking`;
4. posts its report with the round number, `converged` when it judges the loop done, with its `blocking` ids in the report's fields. The orchestrator does the same when it declares with the run token.

**A correction from the orchestrator:** `finding open` (`ci`, `gate`, `detector`, `orchestrator` or `maintainer`, with `relates_to` when it touches a settled row), then `msg post --kind fix` with `{"finding_id": <id>, …}` in the payload (a convention; `messages` does not check it).

### What child 17 reads

From the tables alone: the rounds; the findings per round with `source` and `reviewer`; `unchanged_lines`; the classification and its changes (`classify` events); the reopenings (`state` events to `open`, with `cause_id`); the convergence declarations (`consistent` is computed live, so the node's `converged` report carries its `blocking` ids); the flip-flops (`overlaps_fix` / `relates_to` flags, `matches` on a resolved row, `settled` rows and their rationale).

## Review sessions (#407)

Two **one-shot LLM sessions**, launched and recorded by the Control Plane, replace a permanent coherence checker (maintainer, 2026-10-02): the **global review of a run** (every task of one run read together: does each fit the global plan, do they fit together, each deliverable's blast radius) and the **cross-run coherence review** (tasks of several runs that collide: the same code, the same task twice). The per-task review is not one of them: it is the node's own build loop (6b). Code: [`reviews.py`](../../scripts/qs/control_plane/reviews.py), [`reviewpilot.py`](../../scripts/qs/control_plane/reviewpilot.py), [`gitread.py`](../../scripts/qs/control_plane/gitread.py), schema [`schema_reviews.py`](../../scripts/qs/control_plane/schema_reviews.py); agents `qs-global-review` and `qs-cross-run-review` (templates in [`scripts/qs/agent_templates/`](../../scripts/qs/agent_templates/)).

**No ceiling** (maintainer, 2026-10-13: the only ceilings are CodeRabbit's, 6b's): no cap on review sessions, no per-run budget, and a review never counts against the node cap (`locks.admit_node` counts only `nodes` rows). Coalescing, deferral and the pilot's one review per event are the only bounds. **No global review for a run with fewer than 2 tasks.**

### The tables

| table | holds |
|---|---|
| `reviews` | one row per review: `id` (`V<n>`), `kind` (`global` / `cross_run`, checked in Python: `schema_reviews.KINDS`), `coalesce_key`, `home_run_id` (`NULL` cross-run), `state` (`requested → launching → running → finished`, or `failed` / `abandoned`), the agent, model, effort and permission mode, `nonce` / `grant_nonce`, the launch `name`, `session_id` / `short_id`, `dir`, `tool_key` (`review/<call key>`), `requested_by`, `summary`, `error`, `heads_at` (the capture marker), the times. A partial unique index allows one in-flight review (`requested`, `launching`, `running`) per `coalesce_key` |
| `review_tasks` | its scope: one row per task, with its `run_id`, the `base_sha` / `head_sha` / `head_state` it reviewed, and its `outcome` (`pass` / `fail`), `reason`, `blast_radius_id`, `recorded_at` |
| `review_triggers` | every trigger: `review_id` (`NULL` while queued or skipped), `actor`, `trigger` (JSON), the raw call `tool_key`, `outcome` (`queued`, `created`, `joined`, `skipped`). One row per pilot attempt (`UNIQUE (tool_key) WHERE tool_key LIKE 'pilot:%'`): the pilot's queue |

`snapshot` shows the three tables; with `--run R`, `review_tasks` of R and the `reviews` / `review_triggers` of a review with a task in R (a queued pilot row, with no review yet, only unfiltered). Neither `nonce` nor `grant_nonce` ever leaves the DB. A cross-run launcher in flight has `tool_calls.run_id NULL`, so a run-filtered `tool_calls_in_flight` does not show it; its `reviews` row does (an accepted limit).

### The two token forms

`review:V<n>.<e>.<nonce>`, `V[1-9][0-9]*`, `e` in `{0, 1}`, 32 hex digits (a full match: anything else, a trailing newline included, is `USAGE`). `tokens.require` maps `e = 1` to the effective kind `review` and `e = 0` to `review_grant`, and checks it against the command's `kinds`; every command but three has neither (`CONFLICT`). Expiry is **by state**, never by clock: the pilot closes dead and expired reviews.

| form | minted | valid while | accepted by |
|---|---|---|---|
| session (`e = 1`, `reviews.nonce`) | when the row becomes `launching` (the run-token path inserts it so; the grant path's `requested → launching`) | `launching` or `running` | `finding open` (a task of its scope, any run, its own source) and `review finish` |
| launch grant (`e = 0`, `reviews.grant_nonce`) | by the pilot's request | not `failed` / `abandoned` | `tool review`, for its own review under the key it was minted for (`reviews.tool_key == "review/" + key`) |

Outside its valid states a token is `STALE_TOKEN`, with `instructions` "This review is closed or superseded: stop now, take no further action." A `NULL` or empty stored nonce is `STALE_TOKEN` before any comparison; nonces are compared with `hmac.compare_digest`. A review's `Principal` carries `review_id`, `review_kind` and its `scope`; its `run_id` is the home run, or `""` cross-run; its actor is `review:V<n>`. With `task_id`, the task must be in the scope (`CONFLICT` otherwise), whatever its run. The reads (`snapshot`, `task show`, `ledger show`, `review read`) take no token.

### `tool review`: one review session

`cp.py tool review --key K --args-file F --token T` (task-less: no `--task`):

| token | args | answer |
|---|---|---|
| run of R | `{"kind": "global", "run": "R", "trigger"?: {…}}` (`run` by id or name) | launch, join, `BUSY` (deferred) or skip |
| run of another run, or `{"kind": "cross_run"}`, or `{"review": …}` | | `CONFLICT` |
| grant of V7, key `K` with `reviews.tool_key == "review/K"` | `{"review": "V7"}` | launch |
| grant of V7 with another key or other args; a session or node token | | `CONFLICT` |

A missing or unknown `kind`, a missing `run`, an unknown argument or a non-object `trigger` is `USAGE`, and so is a run token's key starting with `pilot:` in any case (`schema_reviews.PILOT_PREFIX`: reserved for the review pilot's queue). Every refusal comes from `authorize`, before the claim.

**The scope** (`reviews.request_locked`, shared with the pilot):

- **global:** every task of the run that is a deliverable or a work item and is not terminal (`dropped`, `merged`, `validated`). Fewer than 2 (work items count: a deliverable split into 3 items is 4 tasks) → `{"skipped": "single_task", "tasks": […]}`, a success with no review row and no trigger row; a replayed key answers the same.
- **cross-run** (the pilot only): the event's tasks, re-filtered at every request: tasks of closed runs (or of no run) are dropped, except a merge event's merged task. Skipped when nothing left is non-terminal (`no_open_task`), when an alert event spans fewer than 2 open runs (`single_run`), or when a merge event has no open run other than the merged task's (`no_other_run`).

**Coalescing and deferral** (Q6, maintainer 2026-10-16). The coalesce key is `global:<run>` or `cross_run:<sorted kept task ids joined by "|">`. When the key has a review in flight:

- its heads are not captured yet (`heads_at IS NULL`) and, for a global review, the scope is the same → the request **joins** it: a `joined` trigger row, no launch, `{"coalesced_into": "V<n>"}`;
- otherwise the request is **deferred**, since it would be absorbed by a review that never sees the new commits: `tool review` answers `BUSY` with `review_id`, "replay this key after it closes" (released: nothing written, the same key works later); the pilot keeps the event queued.

So after a running review closes, **one** new review of the latest heads starts, and the triggers of its first seconds join it. A key whose call ended (a launch, a join, a failure) is spent: after a `review_failed` for its own review or one it joined, a caller uses a new key.

**The steps.** Each re-reads the review and checks its own precondition.

1. `reserve` (one transaction). Run token: `request_locked` (skip, join, `BUSY`, or a new row `launching` with its session nonce). Grant: `requested → launching`, minting the session nonce; a row already `launching` under this key is a takeover and re-mints nothing.
2. `prepare`: stamps `heads_at` first (guarded; from that instant a trigger defers), then captures each pending scope task's base and head **once**, in `<MAIN>` (see below), resolves `(model, effort)` through `Ctx.resolve_model(<agent>, None)` (the agent's own policy class: a review spans lanes), creates the private directory and copies `<MAIN>/.claude/agents/<agent>.md` into it (missing → "render the agents on main").
3. `launch`: stamps `launch_at`, then `claude --bg --agent <agent> -n qs-review-<db tag>-V<n> --model M <permission args> --settings '<settings>' "<prompt>"`, its cwd the review directory, `QS_CP_TOKEN` and the session identity removed. A non-zero exit other than 124 fails the step with `claude --bg exited N: <output tail>` (stderr's, else stdout's), plus `; <trust hint>` when Claude Code refused the folder (see **Folder trust** below) — redacted before the cut, at most `reviews.ERROR_MAX` (500) characters, the prefix and the hint always kept (an over-long review root is shown by its end, after `…`; `liveness.bg_failure`, #421).
4. `identify`: the listing by name, until `launch_at + LAUNCH_SETTLE_S`, re-reading the review on every poll (a read transaction): a review already `finished` (a quick session that exited before it was ever listed), or `running` once the pilot adopted it, ends the step successfully; `failed` / `abandoned` fails it.

**The unknown-listing rule** (one rule, for `identify` and the pilot's closing of dead reviews): **a launched review (`launch_at` set) fails early only when a known listing does not show it; under an unknown listing (`try_agents()` is `None`) only `REQUEST_TTL_S` fails it.** The latest poll's listing decides, not an earlier one: so an `identify` whose listing is unknown at its deadline fails the call (`TOOL_FAILED`, `reviews.Unidentified`, its output `{review_id, state: "launching", listing: "unknown"}`) but leaves the review `launching`, its session token valid, for the pilot to adopt by name on a later known listing (or to fail at `REQUEST_TTL_S`); and a `claude --bg` that timed out (the runner's exit 124: the session may have forked) is not a launch failure: `launch` answers `{launched, timeout}` and `identify` decides by name, its deadline then at least one `LAUNCH_SETTLE_S` past its first poll (a genuine exit 124 costs the same one settle window).

Steps 2–4 are noops after a skip or a join. **Any other failure** of `prepare`, `launch` or `identify` (never listed under a known listing included) first moves the review to `failed` (a guarded `UPDATE … WHERE state IN ('requested', 'launching')`, with `error`) and, when the row moved, posts `review_failed` to each open run of its scope; the call is then `TOOL_FAILED`: the key is spent, the coalesce key free, the session token void. A replayed key never launches again: the probe reports the launch done whenever `launch_at` is set, and a takeover of a dead holder only finishes `identify` against the original deadline. `on_success` moves `launching → running` and records the session; on a row already `finished` (the session finished while the launcher was identifying) it records only the session id and the answer carries that state. The answer: `{review_id, kind, name, session_id, state, tasks: [{task_id, run_id, base_sha, head_sha, head_state}]}`, never a token.

**The base and head reviewed**, per task, the first row that applies (a git timeout or exec failure, exit 124 / 127, fails the step; any other non-zero exit means absent; only 0 means found):

| the task | `head_sha` | `base_sha` | `head_state` |
|---|---|---|---|
| `merge_sha` set (hex, lower-cased first) and present locally | `merge_sha` | its first parent | `merged` |
| a branch that resolves | its tip | a work item: the merge base with its deliverable's branch, when that resolves; else the merge base with `<main base>` | `branch` |
| a branch that does not resolve | `NULL` | `NULL` | `unresolved` |
| no branch | `NULL` | `NULL` | `plan` (the review covers its plan) |

`<main base>` is the overlap detector's rule: the newer of `refs/heads/main` and `refs/remotes/origin/main`, `origin/main` when they diverged. `base_sha..head_sha` is the task's own change. Outcomes on a `NULL` head are advisory.

### Where a review runs

In a **private directory outside every checkout**: `QS_CP_REVIEW_DIR` (default `~/.local/state/quiet-solar/cp-reviews/`), then `<sha256(<DB path>)[:8]>/V<n>/`, each level 0700 (`reviews.review_dir`; a directory inside a checkout or a linked worktree is `POLICY_REFUSED`). It holds `in/` (the session's own files, the only place it reads or writes), `.claude/agents/<agent>.md` (the copied agent), `args.json` and `launcher.log` (the pilot's). The session gets no `--add-dir`: it reads code only through `review read`, and state through `snapshot`, `task show` and `ledger show`.

**Permissions** (`reviews.REVIEW_PERMISSION_MODE`, `dontAsk`; `auto` is the fallback, both built by `reviews.launch_args`):

- `dontAsk`: `--permission-mode dontAsk --allowedTools=<list>` (one word), the list being `Read(//<dir>/in/**)`, `Edit(//<dir>/in/**)`, `<dir>` the review directory without its leading `/` (the absolute-path form: `in` = `/x/V1/in` gives `Read(//x/V1/in/**)`; Claude Code matches only `Edit(path)` rules for file writes, and an `Edit` rule allows the Write tool — a `Write(…)` rule matches nothing, #421) and `Bash(<python> <MAIN>/scripts/qs/cp.py <cmd>:*)` for exactly `snapshot`, `task show`, `ledger show`, `review read`, `finding open`, `review finish` (`<python>` is `<MAIN>/venv/bin/python`, else the launcher's interpreter). A path with whitespace, a control character, a `,` (the rule separator) or a shell metacharacter fails the review (`POLICY_REFUSED`): the unquoted pinned form would stop matching. Any other character, non-ASCII included, is kept verbatim.
- `auto`: `--permission-mode auto --disallowedTools=Edit,NotebookEdit`. By the same matching rule, whether `--disallowedTools=Edit` also blocks the Write tool (and so the session's `in/` files) is unverified: the T13 re-run of #421 probes it.

**Folder trust** (#421). Observed with Claude Code 2.1.299: `claude --bg` refuses a folder that is not trusted (`Workspace not trusted. Run claude in … once and accept the trust prompt`, exit 1). Trust is inherited from a parent folder, so run `claude` once in `~/.local/state/quiet-solar` (the default parent of `cp-reviews/` and `cp-messenger/`) and accept the prompt; with `QS_CP_REVIEW_DIR` / `QS_CP_MESSENGER_DIR` overridden, trust each folder or a shared parent. A refused review launch says so in `reviews.error` (and so in the `review_failed` alert, and in a global review's `tool review` answer) with a hint naming the review root — the narrowest folder to trust; the shared parent covers the messenger too. A refused messenger launch logs the same hint, naming its directory, once per daemon run in `harness_state.daemon.log` (`messenger launch refused: …`). Nothing writes Claude Code's trust store for you. **Node worktrees** (#424): trust `quiet-solar-worktrees` (the root of every `QS_<N>` and `QS_<N>_<k>` worktree `tool worktree-create` / `tool item-create` make) once, by running `claude` there and accepting the prompt. A refused `tool spawn` / `tool resume` launch names the worktree's parent folder in the same hint.

`--settings` is the node hook fragment (its `PreToolUse` DB-access deny applies to any session), the policy's `effortLevel`, and `"env": {"QS_CP_DB": "<the launcher's DB, resolved>"}`, so the session's `cp.py` calls open the same DB as the launcher. The prompt carries the facts: review id and kind, the scope (task, run, title, branch, `base_sha`, `head_sha`, `head_state`), the exact `cp.py` prefix, the `in/` directory, the session token, the deadline (`launch_at + REVIEW_TTL_S`); a task with no run shows `-` as its run. The agents pin the command form: one command per Bash call, no pipe, `&&`, `;`, redirect, `$(…)`, environment prefix or heredoc; `--token` only on the two writes.

### Reading code: `review read`

Exempt (no DB, no token), one fixed `git` invocation in the main checkout, on **hex shas only** (lower case, 7 to 64 characters, `gitread.SHA_RE`, the same check as the capture's: a ref or a leading `-` is `USAGE`), with `-c core.pager=cat -c diff.external=`, `GIT_CONFIG_NOSYSTEM=1`, `GIT_PAGER=cat`, `GIT_EXTERNAL_DIFF` and the inherited repository location (`GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`, `GIT_COMMON_DIR`; `gitread.ENV_REMOVED`) and every other inherited `GIT_*` variable but those set here (`gitread.env_removed()`, applied to the capture's git too) unset, `-c core.quotePath=false` (non-ASCII bytes are not quoted, so an ordinary name printed by `files` / `ls-tree` is a valid `--path`; a name holding `"`, `\`, a tab or a newline is still C-quoted, and `grep` prints `<rev>:<path>:<line>:`, not a bare path) and a timeout:

| op | arguments | runs |
|---|---|---|
| `diff` | `--base SHA --head SHA [--path P]` | `git diff --no-ext-diff --no-textconv --no-color <base> <head> [-- P]` |
| `show` | `--rev SHA --path P` | `git cat-file blob <rev>:<P>` |
| `grep` | `--rev SHA --pattern X [--path P]` | `git -c grep.patternType=basic grep -n --no-color -e <X> <rev> [-- P]` (no match: empty): `<X>` is a basic regex (BRE) whatever the user's git config (#421); the agents pass it single-quoted (`--pattern 'f(x)'`), since it goes through the Bash tool's shell, and a `--pattern` or `--path` value starting with `-` in the `=` form (`--pattern='--bg'`, `--path=-x`: with a space, argparse reads it as an option, `USAGE`) |
| `ls-tree` | `--rev SHA [--path P]` | `git ls-tree -r --name-only <rev> [-- P]` |
| `files` | `--base SHA --head SHA [--path P]` | `git diff --no-ext-diff --no-textconv --no-color --name-status <base> <head> [-- P]` (the changed files, when a `diff` is `truncated`; a truncated `files` is narrowed with `--path <dir>`) |

A missing or extra argument for the op is `USAGE`; `P` goes through the ledger's path normalisation. No other git option can be passed. The answer is `{"ok": true, "output", "truncated", "lossy"}`: the output is decoded with `surrogateescape` (`Runner.run(decode_errors=…)`), cut at `READ_MAX_BYTES` (on bytes, backing off to a UTF-8 boundary: a character the cut splits is dropped), and printed with `errors="replace"`; `lossy` when a byte kept was not UTF-8. A failed git is `NOT_FOUND` (exit 128: a bad object or path, or a `grep` pattern git cannot compile) or `INTERNAL`.

### `review finish`: the hand-back

`cp.py review finish --input F --token <session>`, once:

```json
{"summary": "…",
 "tasks": {"T3": {"outcome": "pass", "blast_radius": "ok", "reason": "…"},
           "T7": {"outcome": "fail", "reason": "…"}}}
```

The file is read and parsed first (non-JSON or a duplicate key is `USAGE`), then the token (`CONFLICT` / `STALE_TOKEN`), then the shape: every scope task exactly once (a task outside the scope is `CONFLICT`, a missing one `USAGE`), `outcome` `pass` or `fail`, a `reason` for `fail`; `blast_radius` (`ok`, `doubt`, `too_large`) only in a global review, required for every deliverable with a head and refused on a `NULL` head. Nothing cross-checks the outcome against the findings. In one transaction: the review `finished` (from `launching` or `running`), each task's outcome, the blast-radius rows (`head_sha` the reviewed head, `review` `V<n>`, actor `review:V<n>`), and one **`review_done`** message per **open** run of the scope (recipient `orchestrator`, sender `review:V<n>`, `dedupe_key` `review:V<n>`), its payload `{review_id, kind, summary, tasks: [{task_id, base_sha, head_sha, head_state, outcome, reason, blast_radius, finding_ids}], other_runs}` for that run's tasks. Afterwards the session token is void: a second call is `STALE_TOKEN`.

**The reader** (child 7): `reviews.outcome(conn, task_id, *, kind, head_sha)` → the latest **finished** outcome of that task, kind and head (`{review_id, outcome, reason, base_sha, head_sha, finished_at}`), else `None`; a `None` or blank `head_sha` is `None` (an outcome on no head is advisory). `ledger.blast_radius(conn, task, head_sha)` reads the global review's rating (`review = V<n>`). Both readers take the **full** `head_sha` (the capture stores the resolved full sha), matched stripped and case-insensitively; an abbreviated sha matches nothing.

### The review pilot

The tick hook `review_pilot` (after `detectors`, before `hook_route`; every `PILOT_EVERY_S`). The daemon holds no token: the **launch grant** is the launcher's only authority.

1. **Reap** the launcher processes that finished (`poll()`).
2. **Close dead reviews** (read, probe, beat, write). Each move is a guarded `UPDATE … WHERE id = ? AND state = <the state read> AND <its condition, re-evaluated in SQL>`, and posts one `review_failed` one-shot alert (subject `review:V<n>:<state>`, payload `{review_id, kind, state, error, finding_ids}`) to each open run of the scope, only when the row moved. With `since = COALESCE(launch_at, updated_at)`:
   - `requested` / `launching` whose launcher's call is `failed`, or `started` by a dead holder with `since` older than `LAUNCH_SETTLE_S`, or `requested` with no call row `PILOT_RETRY_S` after `created_at` (the launcher never started), or `since` older than `REQUEST_TTL_S` (a hung launcher) → `failed`. **Adoption:** a `launching` review with `launch_at` set (its `claude --bg` ran) whose session the listing shows by name is not failed but moved to `running` with its `session_id` / `short_id` (a guarded `UPDATE … WHERE state = 'launching' AND launch_at IS NOT NULL`, no alert): the launcher died after the launch, the session lives. The listing is taken for such a review too; under a failed listing neither a failed call nor a dead launcher fails it early, only `REQUEST_TTL_S` does (the unknown-listing rule, above);
   - `running` with `launch_at` older than `REVIEW_TTL_S`, or whose session is absent **or listed `done`** (its `status` or `state`; a review session is single-turn, so `done` means it ended, #421) on `LIVENESS_CONFIRM` successful listings in a row → `abandoned`, with error "the review session ended without finishing" (listed `done`) or "the review session is gone" (absent), from the latest listing; a listed live session resets the count. The listing is taken only while a review runs; a failed listing changes nothing; the count lives in memory and starts cold.

   No `claude stop`: a dead or late session's token is void.
3. **The queue and the rules.** Every new event is written at first sight as a `review_triggers` row `pilot:<event>:1`, `queued`:
   - **alerts** (`PILOT_KINDS`: `overlap_cross_run`, `duplicate_task_cross_run` — not `dependency_cycle_cross_run`, already alerted to every run): the open rows grouped by `(kind, subject)`, scope the payload's `tasks`. One occurrence, one event: a group whose alert ids meet the ids stored by an earlier row of that `(kind, subject)` is that event (newly raised ids join it); a group with none is a new event `alert:<kind>:<sha256(subject)[:16]>:<lowest id>`;
   - **a merge that touches another run:** a task that reached `merged` in `task_history` after `meta.review_pilot_cursor`, named by any `overlap_cross_run` row (open or cleared) with a task still non-terminal in another open run; scope the merged task plus those; event `merge:<task>:<history id>`. The rows and the cursor move in one transaction (at most `MERGES_PER_TICK` merges per tick).

   Then each queued row is re-requested through `request_locked` (the same scope, coalescing and deferral as `tool review`), which marks it `created`, `joined` or `skipped`, or leaves it `queued` when deferred; a queued alert event none of whose alerts is still open is `skipped`. A `skipped` row leaves the queue for good. On `created`, outside the transaction, the pilot writes `args.json` and starts, detached and never waited on:

   ```text
   <MAIN>/venv/bin/python <MAIN>/scripts/qs/cp.py tool review --key pilot:<event>:<n> --args-file <dir>/args.json --token review:V<n>.0.<grant>
   ```

   (`start_new_session`, stdin closed, output to `launcher.log`, cwd `<MAIN>`, the environment through `daemon.strip_env`, with `QS_CP_DB` set to the daemon's DB, resolved: a relative path would name another file from `<MAIN>`). Any failure there moves the review to `failed`. With no `popen` seam the pilot requests nothing (logged once).

**The retry** (Q7, maintainer 2026-10-16): an event whose latest attempt's review ended **`failed`** (the launch failed: nothing ran) gets attempt `n + 1`, a new row, while `n < PILOT_ATTEMPTS`; merge events too, whatever the cursor. A review that ended **`abandoned`** (it ran, then died or expired) is not retried: its `review_failed` alert tells the orchestrators and the maintainer, who decide.

**The merge rule's limit:** an overlap the detector never saw is not caught; the node's catch-up with `main` still meets it.

## Node reports, digests and worktree files (#424)

`report post` and `digest put` take a closed, exact contract (`reports.py`). Every check is of type and shape only, never text parsing, and runs before the transaction: a refusal writes no row and posts no message. **Shape is `USAGE`, size is `CONFLICT`**, for both.

**A report.** `--phase` is exactly one of `schema_ledger.PHASES` (`plan`, `build`); `--status` one of `reports.STATUSES` (`converged`, `continuing`, `blocked`); `--round` an int, as before. `--fields-file` holds a JSON object (non-JSON is `USAGE`) on this schema, `reports.FIELDS` = `("blocking", "questions", "discovered")`, every key optional (`{}` is valid):

| key | type | meaning |
|---|---|---|
| `blocking` | a list of positive ints (a `bool`, `0`, a negative or a float is refused) | the finding ids of `convergence.blocking` the node saw when it posted (child 17's evidence) |
| `questions` | a list of strings, each a full match of `reports.QUESTION_ID_RE` (`Q-[1-9][0-9]*`) | the open questions (`question open`) this round raised or still waits on |
| `discovered` | a list of objects with keys out of `reports.DISCOVERED_KEYS` = `("title", "why", "shrinks")` | tasks found outside the task: `title` and `why` required, non-empty strings (after `.strip()`); `shrinks`, optional, a real `bool` (`0`, `1`, `null` refused). `shrinks: true` means the discovery shrinks this task, the one case where the node waits for the answer |

`reports.validate_fields` names the first violation, in this order: not an object; an unknown key; then each key in `FIELDS` order (its type, then its items in order; for a `discovered` item: an object, no unknown key, `title` and `why` present, both non-empty strings, `shrinks` a `bool`). The canonical form (`messages.canonical`: compact, sorted, `ensure_ascii=False`, so accented text costs its UTF-8 size) over `FIELDS_MAX_BYTES` (16 KiB) is `CONFLICT`. Strings are stored as given, never stripped. Nothing cross-checks ids against the DB, and duplicates are harmless. The `report` message payload is unchanged (no `fields`): the orchestrator reads them through `task show` / `snapshot`.

**A digest** is the task's short memory, replaced on every put: read by the global and cross-run review sessions, the run's final review, a fresh node resuming a rotated task, and the exported summary. No reader is code that parses it, so its values are plain text (free Markdown), but its shape is exact. `digest put --file F` reads F as JSON: an object with exactly `reports.DIGEST_KEYS`, each a string (`""`: nothing to say), in any order:

| key | holds |
|---|---|
| `goal` | the goal and the validated acceptance criteria |
| `why` | the use case, or the root cause for a bug |
| `plan` | approach, modules, interfaces, assumptions |
| `done` | what was done |
| `problems` | major problems |
| `decisions` | decisions and rejected findings, with their reasons |
| `questions` | open questions |
| `discovered` | discovered tasks |

Its canonical JSON is stored in `digests.body` (no migration), capped at `DIGEST_MAX_BYTES` (`CONFLICT`, checked after the shape); the answer is `{task_id, bytes}`, `bytes` the canonical UTF-8 length. **Reading a stored body** follows one rule, `reports.parse_digest`: the object when the body parses as JSON and is a digest, else `None`. `task show` returns `digest.body` as that object, or as the raw text of a body stored before #424; `snapshot` carries no body. The export renders a digest under `## Digest` as one `### <Key>` block per key in `DIGEST_KEYS` order (an empty value reads `_None._`). A value is rendered so it cannot break the summary's outline (`export._value_lines`): it is a **blockquote** (every line `> `), and the blank line after it ends the quote, which closes any fence, list or HTML block the value left open, however nested. Inside it, leading blank lines are dropped; a Markdown heading (1 to 6 `#` then a space, a tab or the end), nested in a quote or a list item or not, is pushed three levels down, capped at 6, its prefix kept, so a value's `## Decisions` cannot add a second one to the summary, while `#424` or `#tag` is kept; only the lines of a **bare** code fence (```` ``` ```` / `~~~` with no list or quote marker, at most three columns in, not inside an HTML block) are kept as they are — CommonMark ends such a fence only with a bare closer or, when it is indented (so perhaps inside a list item), at a less-indented line — and every other `#` line errs on demotion, which is cosmetic; a setext underline (`---` / `===` under a text line) gets a blank line before it, so it is no heading; a footnote definition's `[^` is escaped, so its text stays in place. The marker patterns are possessive: no line backtracks exponentially. Whitespace and line ends are CommonMark's (spaces and tabs; `\r\n`, `\r`, `\n`). A body off the contract is rendered raw, as before. A string holding a lone surrogate (`json.loads` accepts `\ud800`; argv's `surrogateescape` makes one of a non-UTF-8 byte) is `USAGE`, in a digest, in `fields` and in a report's `--summary`; so is JSON nested too deep for the decoder (`messages.parse_payload`). A `blocking` id must fit a SQLite INTEGER (`reports.SQL_INT_MAX`).

**A node's worktree files.** A node keeps its own files in `.qs-node/` at its worktree's top level, gitignored (`/.qs-node/`): `plan.md` (the plan, the contract once the plan loop converges), `digest.json` (its digest as last put: `digest put --file .qs-node/digest.json`), `binding.json` and `round.json`. The node writes them (6b-2; the suggested layouts, which 6b-2 may adjust, are `binding.json` = `{"run", "task", "node", "generation"}`, **never the token**, and `round.json` = `{"phase", "round"}`); child 18 defines what it reads. Nothing in the Control Plane reads or writes them, and `tool worktree-cleanup` removes them with the worktree (`git worktree remove --force` deletes ignored files too).

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
- A session-held lock stays held while a **co-holder** (one of the holder's own tools) is alive. A co-hold re-stamps the holder's pid from the listing only when that pid's start time is known; an unknown start, or a pid gone since the listing, keeps the recorded pid (never a NULL start).
- A listing pid outside the `pid_t` range (2³¹ or more) is dropped, and a pid that overflows `kill` is dead.

### Caps

- **Gate cap:** `QS_CP_MAX_GATES` slots (default 2), for `tool gate` and #400's integration finish. The cap counts the live slot rows against the caller's own limit, so a process started with a smaller limit never exceeds it.
- **Node cap:** `QS_CP_MAX_NODES` (default 4), counted across all runs. A node counts when it is:
  - `spawning`, while its spawn call's holder is alive or its launch has not settled;
  - `running`, `idle` or `taken_over`, when the listing shows it, or the listing failed, or it launched less than `LAUNCH_SETTLE_S` ago.

## Tools

Every tool runs through `run_recorded`:

1. Every `*_file` argument is read **once**, when the call starts (before the token check), into a cache. Then verify the token. A call already finished under the key is replayed at once, before its steps are built: a deleted argument file or a task column cleared meanwhile cannot break the replay. Otherwise a non-regular file (a FIFO, `/dev/stdin`) or a file that is not UTF-8 is `USAGE`, and a fresh call with a missing file is `USAGE`, before the claim and before any effect. A call whose in-flight holder released the key meanwhile becomes a fresh call at the claim, and needs every file too.
2. Claim `tool_calls(tool, key)` before any effect.
3. Probe: the probe is authoritative for each step it reports, done or not done.
4. Run the missing steps under the locks or the cap, re-checking the token and lock ownership before each step. A step reads its file from the cache only when it runs, so a takeover whose effect step is already recorded (or found by the probe) never needs the file again. `spawn` and `resume` read their file before they stamp `launch_at`, so a takeover with the file gone is `USAGE` with nothing launched, and the same key works once the file is back. A `spawn` with no caller `model` also resolves its model policy again there, before `launch_at` (QS-405): a refusal at that point fails the call (`TOOL_FAILED`, nothing launched).
5. Record the call `succeeded` or `failed`.

**Keys** are chosen by the caller: `msg:<id>` for a popped message, or `task:<id>:<purpose>` otherwise. A key is 1 to 200 characters out of `[A-Za-z0-9._:/-]` (`USAGE` otherwise), so it can never break a `gh` search query. Replaying the same key never repeats an effect. The key's `args_hash` covers the arguments and the content of every `*_file` argument: the same key with an edited file is a `CONFLICT`.

A probe that cannot tell (a failed or unparseable `gh` listing or `gh pr view`) answers `BUSY`, "replay the same key later", never "not done".

| tool | token | runs in | locks / cap | probe |
|---|---|---|---|---|
| `worktree-create` (`phase`) | run | `<MAIN>` | `main-checkout` | none; `setup_task.py` is idempotent |
| `worktree-cleanup` | run | `<MAIN>` | `main-checkout` | directory absent and unregistered. Refuses the main checkout and a directory whose `.git` is not a file (`POLICY_REFUSED`); a path another non-terminal task shares only clears this task's column (`shared_with`) |
| `gate` (`mode`, `paths`) | run, or node (own task) | worktree | `gates` slot | none; safe to re-run |
| `spawn` (`agent`, `model?`, `permission_mode`, `prompt_file`, `replace?`) | run | worktree | node cap (`--replace` frees the replaced node's place first) | **The model (QS-405).** A caller `model?` is an override: passed as `--model` verbatim, with no `effortLevel` sent. Otherwise the policy (`models.spawn_policy`, of the running tree — `<MAIN>`'s) gives the `--model` and the `--settings` `effortLevel` for the agent under the task's lane: the deliverable's `lane`, else derived from its `kind` and `target` (`nodes.lane_of`). It is checked first in `reserve` (an unknown agent or an invalid lane is `USAGE`, released, and any other failure is re-raised unchanged (an import failure is `INTERNAL`), both raised before a re-take or a new row; adopting a late launch launches nothing and needs no policy) and resolved again in `launch`. With no `effortLevel` sent, the session inherits the worktree's pinned effort (`.claude/settings.local.json`) or the user's: the spawn cannot clear a pinned effort. A stem whose row exists only on an unmerged branch is `USAGE`. **The probe:** the reserved row through `spawn_tool_key`, and the listing by name. A launch never listed within `LAUNCH_SETTLE_S` is reaped. If the replay that re-takes the reaped row sees its launch listed after all, it adopts it (same name, same nonce). Otherwise it relaunches: the relaunch rotates the row's nonce (the first launch's token is `STALE_TOKEN`) and its name (`<name>-r<n>`), so a late first launch is never adopted, and a listing that shows it later gets it a best-effort `claude stop <id>` (an adopted launch too stops its listed earlier launches). Only a listed session whose `cwd` resolves to the task's worktree is stopped: a same-named session of another checkout or DB, or one listed without a `cwd`, is left alone. **The argv** (#424 D2): `--agent A -n <name> --model M --add-dir <MAIN> --permission-mode P --settings <json> "<prompt>"`, the prompt right after `--settings`: `--add-dir` takes several values and would swallow a prompt that follows it (L8). **A failed launch** (any non-zero exit, a timeout's 124 included) fails `TOOL_FAILED` with the `liveness.bg_failure` text as the recorded `result.detail`: `claude --bg exited N: <tail>[; <trust hint>]`, at most `liveness.BG_ERROR_MAX` characters, the output redacted before the cut (the node's own nonce too, whitespace allowed between its characters, so an echo wrapped inside the token leaks nothing), the hint naming the worktree's parent folder when Claude Code refused it as untrusted; the answer's `detail` is that text after `spawn: `. An `agent`, `model` or `permission_mode` that starts with `-` or holds a NUL is `USAGE` before any effect. A prompt that starts with `-` or holds a NUL fails the call before any launch (`TOOL_FAILED`, `launch_at` unwritten): `claude` would read the last positional as an option, and no argv carries a NUL |
| `resume` (`message_file`) | run | worktree | node cap | the row through `spawn_tool_key`, and the listing by name and `startedAt`. The launch is a bare `claude --bg --resume <session id> "<msg>"` (no `--add-dir`: a bare resume keeps every launch option, L7); a failed one is recorded like `spawn`'s, after `resume: `, and a message that starts with `-`, holds a NUL or is empty is refused the same way |
| `issue-create` (`title`, `body_file`, `labels`) | run | `<MAIN>` | — | the exact `<!-- qs-cp-key: … -->` marker in the issue bodies: first a consistent listing of the 100 newest issues (the search index lags), then, on a miss, `gh issue list --search` (`--limit 100`) |
| `pr-create` (`title`, `summary_file`) | run, or node (own task) | worktree | — | the marker in the PR bodies of the branch |
| `push` | run, or node (own task) | worktree | — | `git ls-remote` equals `HEAD` |
| `merge` | run | `<MAIN>` | `integration:<branch>`, `main-merge` | `gh pr view` shows `MERGED` (a failed or unparseable view is `BUSY`) |
| `item-create` (#400) | run | `<MAIN>` | `main-checkout` | none; `setup_task.py <N> --item <k>` is idempotent. Sets the item's `worktree` and `branch` |
| `item-cleanup` (`delete_branch`, `discard_unintegrated`, `force`) (#400) | run | `<MAIN>` | `integration:QS_<N>` (process-held; omitted for a NULL `branch`), `main-checkout` | none: `cleanup_worktree.py --item`, then `integrate_item.py drop` (a refused cleanup never drops a live scratch); any state. Clears `worktree` only — a cleaned item keeps its `branch` |
| `integrate-start` (#400) | run, or node (own task) | `<MAIN>` | `integration:QS_<N>` (co-held: the caller's **session** must hold it), `main-checkout` | the session check only (`POLICY_REFUSED` otherwise, before any wait) |
| `integrate-finish` (#400) | run, or node (own task) | `<MAIN>` | `integration:QS_<N>` (co-held), `gates` slot | the session check only |
| `integrate-drop` (#400) | run, or node (own task) | `<MAIN>` | `integration:QS_<N>` (co-held), `main-checkout` | the session check only; any state |
| `review` (`kind`, `run`, `trigger?` / `review`) (#407) | run (its own run's global review), or a review's launch grant | the review's private directory | none: no lock, no slot, no node cap | `launch_at` set: the launch is never repeated, a takeover only finishes `identify`. Task-less (no `--task`); see [Review sessions](#review-sessions-407) |

Notes:

- Every built-in tool except `worktree-cleanup` refuses a terminal task with `INVALID_STATE`, recorded `failed`.
- `merge` also requires the task to be `ready_to_merge`, unless the PR is already merged. Once the PR is merged, `merge_sha` is recorded when known (a missing `mergeCommit` is read once more, after `MERGE_SHA_RETRY_S`; an unknown sha never overwrites a recorded one). A task already `merged` is a noop success (`noop: true`). If the task left `ready_to_merge` for any other state meanwhile, the call still succeeds, with `state_conflict: {"expected": "ready_to_merge", "actual": …}`, for the orchestrator or the maintainer to reconcile; it is also recorded as a `hook_events` `alert` (hook `tool:merge`), so `snapshot` shows it.
- `merge` is refused by the default merge policy until child 7 installs one (`merge_policy.install`).
- `merge` checks the **merge gate** first (#406): `BUSY` while the self-check for the code on disk is pending or retrying, `POLICY_REFUSED` once it failed or is stuck; see [The merge gate and the self-check](#the-merge-gate-and-the-self-check). Both release the claim.

### Work items and integration (#400)

A deliverable `QS_<N>` (one GitHub issue, one PR) is split into work items
`QS_<N>_<k>` (`k` from `task add --item-of`), each in its own worktree
`<repo>-worktrees/QS_<N>_<k>`, cut from the **local** `QS_<N>` and **never
pushed** (no upstream; `pre-push` refuses the ref). The git mechanics are
plain scripts outside this package — `setup_task.py --item`,
`cleanup_worktree.py --item` and `integrate_item.py` — and the five tools
above add the locks, the fencing, the gate slot and the `integrations` rows.
Full design: `docs/stories/QS-400.story.md` §7–§9.

**An item is never a deliverable.** A piece of a split that ships on its
own is a new deliverable: its own issue `M`, branch `QS_<M>` and PR, added
without `--item-of` (as a child, with a dependency, when it must land
first). `QS_<N>_<k>` only ever names a work item that lands in `QS_<N>`'s
PR. Enforced at every write path: `task add` (`USAGE`), `tasks.update_fields`
— the one writer of `issue_number` / `pr_number` / `pr_url`, behind `task set`
and every tool's `on_success` (`INVALID_STATE`) — and the deliverable-only
tools `worktree-create`, `issue-create`, `pr-create`, `push` and `merge`, which
refuse a work item when their steps are built (a bare `INVALID_STATE`: no
probe, no `tool_calls` row). The item tools' guard also refuses a malformed
row, except `item-cleanup` and `integrate-drop`, so a leftover is never
stranded.

The integration flow (the caller is the item's own node, or the
orchestrator; child 6b decides when):

```text
cp.py lock acquire --name integration:QS_<N> --purpose "integrate item <k>" --token T
cp.py tool integrate-start  --task <item> --key item:<id>:start:<tip>.1    --token T
  outcome conflicts → resolve in <repo>-worktrees/QS_<N>_<k>_integration, git add, git commit
cp.py tool integrate-finish --task <item> --key item:<id>:finish:<head>.1  --token T
  outcome gate_red → fix in the scratch and commit (new head), or retry a flaky gate (.2)
cp.py tool integrate-drop   --task <item> --key item:<id>:drop:1           --token T
cp.py lock release --name integration:QS_<N> --token T
```

- `integrate-start` merges the item tip into a per-item scratch worktree
  `QS_<N>_<k>_integration`, detached at `QS_<N>` (`--no-ff`, so the item tip
  stays an ancestor of `QS_<N>`). `.result.outcome` is `merged`,
  `conflicts` (+ `files`, `scratch`) or `already-integrated`.
- `integrate-finish` checks the scratch (no merge in progress, clean, no
  conflict markers left in the conflicted files, `QS_<N>` unmoved), runs
  `quality_gate.py --impacted` on the exact scratch `HEAD`, and only if green
  moves the local `QS_<N>` by compare-and-swap (`update-ref <new> <old>`, or
  `merge --ff-only` in the worktree holding `QS_<N>`). **No push** — the
  deliverable's node pushes through `tool push` when it wants CI.
  `.result.outcome` is `ok` (+ `merge_commit`) or `gate_red` (+ `tail`). A
  successful call means "integrated" only when `outcome == "ok"`.
- `integrate-drop` removes the scratch and reports the undo points
  (`dropped_head`, and `discarded_snapshot` — a commit of the whole tree —
  when it was dirty).
- Records: a `conflict` / `noop` row per answered start, a `gate_red` row per
  red finish, and one `ok` row per (item, `merge_commit`), deduplicated in
  the same transaction. `ok` is the only row consumers read as "integrated".

**Keys — the one rule.** Every step is content-idempotent (the git state and
the DB decide, not the key), and `integrate_item.py` refuses a concurrent
duplicate with a per-item file lock (`scratch-busy`). So:

- the **same key** only when the call got **no answer** (a crash, a kill, a
  timeout of the caller's shell, or `BUSY` before any effect);
- a **new key** after any other answer (`succeeded`, `failed`, or a code
  raised after an effect). A new key is always safe; a caller that lost its
  counters picks a new suffix.

`stale-scratch`, `deliverable-moved` or `deliverable-worktree-busy`:
`integrate-drop`, then `integrate-start` again. `scratch-busy`: a gate is
still running (its watchdog holds the item's file lock until the gate ends
or is killed at its deadline) — wait up to the reported `max_wait_s`, then
retry with a new key. A terminal deliverable's session lock is dead, so the
leftovers are cleaned by `item-cleanup`, item by item.

**Dependencies for child 6b / 6c:**

- the integrating session writes in the sibling directory
  `QS_<N>_<k>_integration`, so it needs that path allowed (e.g. `--add-dir`
  at spawn);
- run `integrate-finish` in the background and wait for its answer — the
  gate can take up to an hour, past a shell tool's timeout (Claude Code's
  Bash caps at 10 min);
- while `integration:QS_<N>` is held, the deliverable's node must not commit
  on `QS_<N>` and must keep its worktree clean when `move` runs. The lock
  does not enforce this.

**Outcomes.**

| what happened | the call | the caller sees |
|---|---|---|
| a step failed, or the state drifted | recorded `failed`; the key is spent | `TOOL_FAILED` (exit 1), with `result.error` |
| `BUSY`, `POLICY_REFUSED`, `STALE_TOKEN`, `STOPPED` or `USAGE` (an argument file gone on a takeover; the model policy refuses the agent or lane — `tool spawn` with no caller model) before any effect | claim released, so the same key can be retried | that code |
| the same codes after an effect | left `started`; a same-key replay takes it over once this process has exited | that code |
| another process took the claim over mid-call | left to that process | `CONFLICT` (exit 8) with `claim_taken_over: true` — never `STALE_TOKEN`, whose exit 3 tells a session to end |

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
    token_kinds: frozenset[str] = frozenset({"run"})            # effective kinds: run, node, review, review_grant
    task_bound: bool = True                                      # #407: False → no --task, StepCtx.task is None
    authorize: Callable[[Connection, Principal, Mapping, str], None] | None = None   # #407: after the token, before the claim

def register(spec: ToolSpec) -> None   # CONFLICT if the name exists
def invoke(name, *, key, task_id, args, token, actor, ctx=None) -> dict
```

- `invoke(ctx=None)` builds `default_ctx()`, which has **no model-policy resolver** (QS-405): a `spawn` through it with no caller `model` is `USAGE`. Pass a `Ctx` with `resolve_model` set (the CLI passes `Deps.resolve_model`).

- `argv_step(...)`, `task_state_guard(...)` and `compose(...)` are helpers. They are not frozen.
- Every registered tool is reachable as `cp.py tool <name> --task T --key K [--args-file F] --token …` (no `--task` for a task-less tool).
- **A task-less tool** (`task_bound=False`, #407) runs with `task_id` and, for a cross-run caller, `run_id` `NULL` in `tool_calls`; it replays by its key like any other. Its `authorize(conn, principal, args, key)` runs in the first transaction, right after the token check and before the replay: a refusal there (`CONFLICT`, `USAGE`) claims nothing. `locks.hold` re-checks the token against the spec's `token_kinds` (`Held.kinds`) before every step.
- **Recorded errors** never hold a review or node token's nonce: `tools.redact` rewrites `review:V<n>.<e>.<nonce>` as `review:V<n>.<e>.<redacted>` and (#424) `node:N<n>.<g>.<nonce>` as `node:N<n>.<g>.<redacted>` in a failed call's `result` and detail, in `reviews.error`, and in a failed `claude --bg` launch's output (before its tail is cut; a failed node launch's output is also scrubbed of the node's own nonce, even wrapped, as `<redacted>`). Run tokens are not redacted: the run token never reaches a launch. Only error paths redact: a success or a replay answers with the full node token.

## Hooks

| hook | wired | input → output | failure |
|---|---|---|---|
| `Stop` (orchestrator) | child 9 writes `hooks-settings --role orchestrator` into main's pin; **6a must not ship before** | stdin `session_id` → `{"decision": "block", "reason": …}` when a message waits (pop it), or once when no `wait` runs; a loop guard allows and records a `hook_events` `alert` | fail open |
| `PreToolUse` (every registered session; matcher `Bash\|Edit\|Write\|SendMessage`) | nodes: `claude --bg --model <m> --settings '<hooks-settings --role node, plus the policy's effortLevel>'` at spawn | `tool_name`, `tool_input` → `hookSpecificOutput.permissionDecision: "deny"` for DB access outside `cp.py` (any session), `gh pr merge` (registered sessions), `SendMessage` from a superseded or stopped session; **`"ask"`** for `cp.py halt clear` / `cp.py restore` (#406) | fail open (the `ask` survives a DB failure) |
| `pre-push` (git) | a common-dir shim installed by `tool worktree-create` (marker `# qs-control-plane pre-push shim v2`) | git's ref lines → exit 1 refuses `QS_<N>_<k>` refs everywhere, and, in a registered worktree, a stopped node, a foreign ref, or a missing or stale `QS_CP_TOKEN`. On a path several tasks registered, it judges a non-terminal task first, newest first | fail closed only for a proven-registered worktree |

**The DB-access rule** (`PreToolUse`, Bash): a segment that names `harness_state.db` is denied unless it starts with a read-only program (`grep`, `rg`, `git`, `ls`, `sed -n`, `cat`, `head`, `tail`, `wc`, `find`, `echo`) or is a `cp.py` call. Even then:

- a redirect onto the DB file itself (`harness_state.db`, `-wal`, `-shm`) is denied, inside a `cp.py` segment too; a sibling such as `> harness_state.db.json`, `.bak`, `-wal.bak` or `-backup.sql` is allowed;
- `find` stays read-only only without `-delete`, `-exec`, `-execdir`, `-ok`, `-okdir`, `-fprint`, `-fprint0`, `-fprintf` or `-fls`, quoted or not (`'-delete'` is `-delete`).

**The maintainer's approval** (`PreToolUse`, Bash, #406): a segment whose unquoted words contain `cp.py` (or a path ending in `/cp.py`) followed by `halt clear` or `restore` gets `permissionDecision: "ask"`, so the app shows the maintainer an approval prompt and nothing runs until he approves. It holds with no DB and under a newer DB, and a deny elsewhere in the command wins (`… && gh pr merge 5` from a registered session is denied). It answers `ask` only in the permission modes of `hooks.ASK_MODES` — `default`, `acceptEdits`, `auto`, `bypassPermissions`: verified on 2026-10-09 that a hook's `ask` shows a real prompt even in `bypassPermissions` — and `deny` elsewhere (`plan`, `dontAsk`), with the hint "switch this session to default mode in the app, then ask again". A payload with no `permission_mode` at all (or an empty or non-string one) is denied too (the safe default for a maintainer-only command), with the hint to run the command in a terminal instead. It is recorded in `hook_events` as an `allow` of kind `maintainer_ask`. A payload that mentions `cp.py` but cannot be parsed (so its mode is unknown too) is denied the same way, with the hint to run the command in a terminal, and recorded as a `deny` of kind `error`; any other parse failure fails open. Fallback: the maintainer can run the command himself by typing `!` in the app (a pasted `!…` is sent as a message).

Resume and effort, checked live on 2026-10-18 (#424, Claude Code 2.1.302; see the L-table in [Liveness and the watchdog](#liveness-and-the-watchdog)): `effortLevel` through `--settings` reaches a `--bg` session (L6), and a bare `claude --bg --resume` of an ended session keeps every option it was launched with — its `--settings` hooks fire again, and its model and effort hold (L7; `resume` passes no flag, since any flag forks a copy). Still open until the post-merge run of #424 (AC9): the hooks of the same node session opened from a terminal inside its worktree (`claude --resume <id>`), and after `/desktop`. `pre-push` is git-level, so it applies regardless.

The shim and the settings commands run `<MAIN>/venv/bin/python`. Without it they fall back to `python3` only if it is 3.14 or newer (the package's syntax); otherwise they print one warning line and allow.

## Environment variables

| variable | default | for |
|---|---|---|
| `QS_CP_DB` | `<MAIN>/harness_state.db` | a temporary DB (tests, experiments). Never a live DB from another checkout |
| `QS_CP_BACKUP_DIR` | `~/.local/state/quiet-solar/cp-backups/` | every backup, in a per-DB subdirectory. Never inside a checkout or a linked worktree (`POLICY_REFUSED`, naming the variable) |
| `QS_CP_MESSENGER_DIR` | `~/.local/state/quiet-solar/cp-messenger/` | the watchdog messenger's working directory (0700). Never inside a checkout |
| `QS_CP_REVIEW_DIR` | `~/.local/state/quiet-solar/cp-reviews/` | where review sessions run, one 0700 directory per DB and review (#407). Never inside a checkout or a linked worktree |
| `QS_CP_MAX_GATES` | 2 | gate slots |
| `QS_CP_MAX_NODES` | 4 | node sessions across all runs |
| `QS_CP_TOKEN` | — | **git only**: injected by the tools into the steps that may push, so `pre-push` accepts them. Every `claude` launch removes it |

## Module constants

Most are overridable by a function argument. These are module-level only (tests patch the module): `MIGRATE_BUSY_RETRIES`, `PID_START_RETRIES`, `PID_START_RETRY_S`, `KILL_WAIT_S`, `MAX_TIME_S`, `MIN_POLL_S`, `IDENTIFY_POLL_S` and `MERGE_SHA_RETRY_S`.

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
| `PID_START_RETRIES` | 3 | `daemon` |
| `PID_START_RETRY_S` | 1 | `daemon` |
| `KILL_WAIT_S` | 2 | `daemon` |
| `MAX_TIME_S` / `MIN_POLL_S` | 86400 / 0.05 | `cli` |
| `LAUNCH_SETTLE_S` | 60 | `nodes` |
| `IDENTIFY_POLL_S` / `MERGE_SHA_RETRY_S` | 2 / 2 | `tools` |
| `GATE_WAIT_S` | 1800 | `locks` |
| `LOCK_WAIT_S` | 600 | `locks` |
| `DIGEST_MAX_BYTES` / `FIELDS_MAX_BYTES` | 16 KiB / 16 KiB | `reports` |
| `BG_ERROR_MAX` | 500 (`reviews.ERROR_MAX` is an alias) | `liveness` |
| `HOOK_SUBPROCESS_S` | 20 | `ticks` |
| `ALERTS_LIMIT` / `ROUNDS_ALERT` | 50 / 5 | `alerts` |
| `SELFCHECK_RETRY_S` / `SELFCHECK_ESCALATE_AFTER` / `PENDING_ESCALATE_S` | 600 / 3 / 600 | `mergegate` |
| `DETECT_EVERY_S` / `OVERLAP_EVERY_S` / `OVERLAP_MAX_CALLS` | 30 / 120 / 20 | `detectors` |
| `STALL_S` / `ORPHAN_GRACE_S` / `GATE_SLOT_HELD_S` | 3600 / 600 / 4500 | `detectors` |
| `LOCK_HELD_PROCESS_S` (process `main-checkout`, `main-merge`, `integration:*`) / `LOCK_HELD_SESSION_INTEGRATION_S` (session `integration:*`) / `LOCK_HELD_DEFAULT_S` | 1200 / 10800 / 3600 | `detectors` |
| `CI_FAST_S` / `CI_SLOW_S` | 30 / 300 | `ciwatch` |
| `CI_BACKOFF_MIN_S` / `CI_BACKOFF_MAX_S` / `CI_BATCH` / `CI_RATE_FLOOR` | 60 / 900 / 25 / 200 | `ciwatch` |
| `CI_RESET_CAP_S` | 3600 | `ciwatch` |
| `GIT_LOCK_STALE_S` | 600 | `codever` |
| `LIVENESS_EVERY_S` / `LIVENESS_CONFIRM` | 30 / 2 | `watchdog` |
| `WATCHDOG_S` / `WAKE_RETRY_S` / `MESSENGER_TTL_S` | 120 / 300 / 300 | `watchdog` |
| `BACKUP_EVERY_S` / `BACKUP_KEEP_ALL_S` / `BACKUP_KEEP_DAYS` | 900 / 86400 / 7 | `backups` |
| `RESTORE_LOCK_WAIT_S` | 60 | `restore` |
| `REVIEW_TTL_S` / `REQUEST_TTL_S` | 3600 / 600 | `schema_reviews` |
| `REVIEW_PERMISSION_MODE` / `GIT_TIMEOUT_S` | `"dontAsk"` (`"auto"` the fallback) / 60 | `reviews` |
| `READ_MAX_BYTES` / `READ_TIMEOUT_S` | 256 KiB / 60 | `gitread` |
| `PILOT_EVERY_S` / `PILOT_RETRY_S` / `PILOT_ATTEMPTS` / `LIVENESS_CONFIRM` / `MERGES_PER_TICK` | 30 / 60 / 2 / 2 / 100 | `reviewpilot` |
| `PILOT_KINDS` | `overlap_cross_run`, `duplicate_task_cross_run` | `reviewpilot` |

## Seams

| seam | real | in tests |
|---|---|---|
| `Clock` | `SystemClock` | `FakeClock`: `sleep` advances time |
| `Runner` | `subprocess.run` with an environment delta, its output decoded with `decode_errors` (`strict` by default; `review read` passes `surrogateescape`) | `FakeRunner`: records argv, cwd, environment delta, `detach` and `decode_errors` |
| `ProcessProbe` | `kill(pid, 0)` plus `ps -o lstart=` (UTC, C locale; a failed `ps` is unknown), and `killpg` | `FakeProbe`: a fresh pid per call, reaped when the CLI call returns |
| `ClaudeCli` | `claude agents --json`, `claude --bg …`, `claude stop <id>` | `FakeClaude`: a scripted listing |
| `ProcessSetup` | `setpgid`, `signal.signal` | a recorder: pytest's own process never changes group or gets a handler |
| `Popen` | `subprocess.Popen`, for `ensure`, and `Seams.popen` for the review pilot's launcher (`make_seams` passes it at call time; `None` launches nothing) | a recorder returning a `FakeProc` (a scriptable `poll()` / `returncode`, a `pid`) |
| `faults.hit(name)` | no-op | `faults.arm(name, exc, skip=n)` |
| `merge_policy` | refuses | `install(fn)` |
| `Deps.resolve_model` → `Ctx.resolve_model` (QS-405) | `cli._policy_resolver`: `models.spawn_policy`, imported at call time — the Control Plane's one import outside the standard library (a test pins it) | the same resolver, or a stand-in; child 15 reuses the seam |
| `export.LEDGER_SECTIONS` | the ledger's three sections (`Rounds`, `Findings`, `Blast radius`), registered by `cli` at import (`ledger.register_export()`); an empty registry renders "No ledger recorded." | the conftest teardown clears it and registers it again |
| the daemon's `tick_hooks` | `ticks.hooks()`: the built-in hooks of [The active loop](#the-active-loop-406) | `ticks.hooks` returns `[]` (a hook-less daemon); the `active_loop` fixture restores it. A tick hook must return within `STALE_AFTER_S` or beat the lease itself; `ensure` SIGKILLs only a daemon whose heartbeat is at least `STALE_AFTER_S` old |
| `activeloop.make_seams` | `Seams(Runner, ProcessProbe, ClaudeCli, paths.main(), ciwatch.GitHub, subprocess.Popen)` | the test `Deps` (its `FakePopen` too), the fake main checkout and a `FakeGitHub` |
| `codever.loaded_version` | the hash at `cli._daemon` entry | the fake main checkout's `code_version`; the `real_loaded_version` fixture restores it |
| `mergegate.merge_allowed` | the gate | `open`; the `real_gate` fixture restores it |

## Conventions: what no hook enforces

- `git push --no-verify` bypasses `pre-push`.
- A superseded session claiming its run back with `--takeover`: takeover is the maintainer's act, and the self-end payload says so.
- Passing another session's token.
- **`run claim --session-id` trusts its caller.** The id names the session claiming the lease, and nothing checks that the caller is that session. The threat model covers agent mistakes, not adversarial sessions: forging another session's id is a deliberate act.
- `gh` issue and PR edits made outside the tools.
- `SendMessage` from an unregistered session.
- DB access that slips past the Bash segment rule, for example through an allowlisted program, or code that reads the path from a variable. Known gaps: `sed -i` behind an allowlisted program name, `$(…)` command substitution, `git rm`, and a `find` whose pattern does not name `harness_state.db` literally (`find . -name 'harness_state*' -delete`). The hook guards against accidents; it is not a sandbox.
- **`gh pr merge` through a wrapper.** The merge rule matches a segment that starts with `gh pr merge`; a wrapper (`env gh pr merge`, `command gh …`, an alias, a script, `gh api`) is not caught.
- **The fencing window on external effects.** GitHub never sees the token. A superseded session's already-started step can still land; its next step is refused. `merge` narrows the window with `--match-head-commit`.
- **The session-lock fallback limit.**
  - When the liveness listing fails, a session-held lock falls back to the pid stamped at the session's last use. After an app restart that pid can be stale, so the lock can be freed while the session is alive.
  - A node that is not listed again, such as a reaped `--bg` node not yet resumed, also loses its lock.
  - The branch move is a ref compare-and-swap, so the cost is redoing an integration step, never corruption.
- **The launch-settle limit.** A `claude --bg` session that only appears in the listing more than `LAUNCH_SETTLE_S` after its launch may be launched twice: a same-key replay reaps the unlisted launch and relaunches it, unless the re-take sees the late launch listed, which it then adopts. The relaunch rotates the node's nonce and name, so the late first session holds a `STALE_TOKEN` and is never adopted; when a later listing shows it, the tool sends it a best-effort `claude stop <id>` (only sessions whose `cwd` is the task's worktree), and otherwise it ends on its first `cp.py` call (`STALE_TOKEN`, exit 3). The `resume` equivalent is weaker: a late duplicate `--resume` keeps a valid token, because a resumed node's nonce cannot rotate (the session keeps its id and its prompt).
- **The singleton probe takes the real lock.** `ensure` checks "gone" by taking the daemon's singleton `flock` and releasing it at once. A daemon that starts during that instant exits with `held_elsewhere`, and the next `ensure` heals it.
- **A permanently unknown `ps`.** When `ps` keeps failing, an older daemon's liveness stays unknown. If nothing holds the singleton `flock`, `ensure` knows it is gone and starts the new daemon; while the `flock` is held, it never signals it and answers `restart_pending` (or starts a new daemon for a stale same-schema lease) until `ps` answers again.
- **The hook `session_id` assumption.** The hook's stdin `session_id`, `$CLAUDE_CODE_SESSION_ID` and the `sessionId` of `claude agents --json` are assumed to be the same identifier. The last two were verified equal on 2026-10-04; the hook field is unverified.
- **`--settings` hooks after a bare resume:** they fire again (#424 L7). **Unverified:** the hooks of a node session opened from a terminal inside its worktree, and after `/desktop` (#424 AC9, after merge).
- **The model and effort of a resumed node, and `effortLevel` through `--settings` on a `--bg` session:** verified (#424): the effort reaches a `--bg` session (L6), and a bare resume keeps the model and the effort (L7).
- **An idle node holds its cap slot.** A node idle between turns is alive (listed `idle/done`, L1) and keeps its place until `node stop` or its task ends; stopping idle nodes is the orchestrator's and the merge's job (6a, 7).
- **The orchestrator's hooks are unwired until child 9.**
- **The watchdog messenger** is the one exception to "only a task's node is spawned" (#406). `claude --bg` runs in Claude Code's background service with **the CLI's own login**: with an expired CLI login the launch still exits 0 and the messenger then stops ("Login expired"), so the run ends up `orchestrator_not_listening`. Keep the CLI logged in (run `claude`, then `/login`). Its folder must be trusted too: see **Folder trust** in "Where a review runs".
- **The `ask` guard is matched per Bash segment.** A wrapper (`bash -c '…'`, `env`, a script) is not caught; the guard covers agent mistakes, like the other matchers. In `plan` / `dontAsk` mode the command is denied instead.
- **The inbox socket is unused.** The messaging socket and its token (`CLAUDE_CODE_MESSAGING_*`) are visible to Bash-tool commands and `SessionStart` hooks; the Control Plane never stores or reads them, and `ensure` strips them (with the session identity) from the daemon's environment, as the watchdog does from the messenger's.
- **The self-check and a broken version.** A Control Plane version that cannot start the daemon keeps merges closed (`stuck` after `PENDING_ESCALATE_S`); if it also carries a migration, write commands answer `SCHEMA_PENDING`. Revert the code.
- **A review's token is visible** (#407). The Control Plane writes no review token to a file of its own, but the session token is in the `claude --bg` argv (visible in `ps`, as a node's is) and in Claude Code's own session transcripts, and the launch grant is in the launcher's argv. The threat model is unchanged (agent mistakes, not adversarial sessions); a recorded error is redacted, `launcher.log` never holds a nonce, and `snapshot` strips both nonces.
- **The `dontAsk` allowlist is a guard, not a sandbox.** It pins the six `cp.py` commands and `Read` / `Edit` rules on `in/` (an `Edit` rule also allows the Write tool, #421); the bypass forms (`> file`, `;`, `$(…)`, backticks) are meant to be denied without a prompt, which the post-merge live check of #407 (T13) verifies. If `dontAsk` fails there, `REVIEW_PERMISSION_MODE = "auto"` is the built-in fallback — but whether `auto` (its `--disallowedTools=Edit`) can write `in/` is unverified: the T13 re-run of #421 probes it. The agents' own rule ("use only these six commands …") holds in both modes.
- **The allowlist's path limit.** The main checkout, the interpreter and the review directory are pinned unquoted: a path holding whitespace, a `,` (it separates the `--allowedTools` rules) or a shell metacharacter (a quote, `\`, `$`, a backtick, or one of `;&|<>(){}[]*?!#~`) cannot be pinned, and every review fails `POLICY_REFUSED` at launch. Move the checkout, or set `QS_CP_REVIEW_DIR` to a plain path. Non-ASCII letters are fine.
- **The DB a review talks to** is the launcher's: `--settings` carries `env.QS_CP_DB`, and the pilot's launcher gets the daemon's DB as an absolute `QS_CP_DB`.
- **The review pilot's merge rule** sees only overlaps the detector saw (an `overlap_cross_run` row, open or cleared).
- **Review directories are kept** (they are small): delete old ones by hand, like the `v<N>` backups.
