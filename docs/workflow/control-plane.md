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
| 8 | `NOT_FOUND`, `CONFLICT`, `INVALID_STATE` | unknown id; uniqueness, argument or receipt mismatch, a token used out of bounds (a task of another run included), or a tool call's claim taken over by another process (`claim_taken_over`); forbidden transition |
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
| `task add` | `[--run R] --title --kind epic\|feature\|bug [--target] [--parent T] [--issue N] [--lane L] [--deliverable] [--item-of T]` (`--parent` / `--item-of` must be a task of the caller's run, or of none: `CONFLICT` otherwise; `--item-of` with `--deliverable` or `--issue` is `USAGE` — a work item lands in its deliverable's PR, a deliverable is its own issue `QS_<M>`, #400) | run | write |
| `task set` | `--task T [--issue] [--worktree] [--branch] [--pr-number --pr-url] [--ci-state --ci-sha]` (`--issue` / `--pr-number` / `--pr-url` on a work item is `INVALID_STATE`, #400) | run | write |
| `task state` | `--task T --to STATE\|unblock [--note]` | run, or node (own task, node range) | write |
| `task dep` / `task root` / `task work-list` | `add\|remove …` (both tasks of a `task dep` must be of the caller's run, or of none: `CONFLICT` otherwise) | run | write |
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

Time flags (`--visibility`, `--poll`, `wait --timeout`) must be finite, above 0 and at most 86400 seconds; `lock acquire --timeout` may be 0 (try once). `wait --poll` must also be at least 0.05 and, when both are given, no more than `--timeout`. Anything else is `USAGE`.

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

Contention anywhere in a migration (the migrate lock, opening the DB, reading `user_version`, the backup, a step, `COMMIT`: any SQLite `BUSY` or `LOCKED` code, extended codes included) is `BUSY`. The daemon retries it (`MIGRATE_BUSY_RETRIES` times, one tick apart) and never records it in the backoff sidecar.

Before migrating an existing DB, the daemon backs it up to `QS_CP_BACKUP_DIR`, which is never inside a checkout.

### Migration recipe for later children (#375 and others)

1. **Append** `Migration(n + 1, "<name>", (<single statements>…))` to `MIGRATIONS` in `migrations.py`.
   - Use single statements only, never `executescript`.
   - A test pins the versions as `1..N`.
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
| `spawn` (`agent`, `model?`, `permission_mode`, `prompt_file`, `replace?`) | run | worktree | node cap (`--replace` frees the replaced node's place first) | **The model (QS-405).** A caller `model?` is an override: passed as `--model` verbatim, with no `effortLevel` sent. Otherwise the policy (`models.spawn_policy`, of the running tree — `<MAIN>`'s) gives the `--model` and the `--settings` `effortLevel` for the agent under the task's lane: the deliverable's `lane`, else derived from its `kind` and `target` (`nodes.lane_of`). It is checked first in `reserve` (an unknown agent or an invalid lane is `USAGE`, released, and any other failure is `INTERNAL`, both raised before a re-take or a new row; adopting a late launch launches nothing and needs no policy) and resolved again in `launch`. With no `effortLevel` sent, the session inherits the worktree's pinned effort (`.claude/settings.local.json`) or the user's: the spawn cannot clear a pinned effort. A stem whose row exists only on an unmerged branch is `USAGE`. **The probe:** the reserved row through `spawn_tool_key`, and the listing by name. A launch never listed within `LAUNCH_SETTLE_S` is reaped. If the replay that re-takes the reaped row sees its launch listed after all, it adopts it (same name, same nonce). Otherwise it relaunches: the relaunch rotates the row's nonce (the first launch's token is `STALE_TOKEN`) and its name (`<name>-r<n>`), so a late first launch is never adopted, and a listing that shows it later gets it a best-effort `claude stop <id>` (an adopted launch too stops its listed earlier launches). Only a listed session whose `cwd` resolves to the task's worktree is stopped: a same-named session of another checkout or DB, or one listed without a `cwd`, is left alone |
| `resume` (`message_file`) | run | worktree | node cap | the row through `spawn_tool_key`, and the listing by name and `startedAt` |
| `issue-create` (`title`, `body_file`, `labels`) | run | `<MAIN>` | — | the exact `<!-- qs-cp-key: … -->` marker in the issue bodies: first a consistent listing of the 100 newest issues (the search index lags), then, on a miss, `gh issue list --search` (`--limit 100`) |
| `pr-create` (`title`, `summary_file`) | run, or node (own task) | worktree | — | the marker in the PR bodies of the branch |
| `push` | run, or node (own task) | worktree | — | `git ls-remote` equals `HEAD` |
| `merge` | run | `<MAIN>` | `integration:<branch>`, `main-merge` | `gh pr view` shows `MERGED` (a failed or unparseable view is `BUSY`) |
| `item-create` (#400) | run | `<MAIN>` | `main-checkout` | none; `setup_task.py <N> --item <k>` is idempotent. Sets the item's `worktree` and `branch` |
| `item-cleanup` (`delete_branch`, `discard_unintegrated`, `force`) (#400) | run | `<MAIN>` | `integration:QS_<N>` (process-held; omitted for a NULL `branch`), `main-checkout` | none: `cleanup_worktree.py --item`, then `integrate_item.py drop` (a refused cleanup never drops a live scratch); any state. Clears `worktree` only — a cleaned item keeps its `branch` |
| `integrate-start` (#400) | run, or node (own task) | `<MAIN>` | `integration:QS_<N>` (co-held: the caller's **session** must hold it), `main-checkout` | the session check only (`POLICY_REFUSED` otherwise, before any wait) |
| `integrate-finish` (#400) | run, or node (own task) | `<MAIN>` | `integration:QS_<N>` (co-held), `gates` slot | the session check only |
| `integrate-drop` (#400) | run, or node (own task) | `<MAIN>` | `integration:QS_<N>` (co-held), `main-checkout` | the session check only; any state |

Notes:

- Every built-in tool except `worktree-cleanup` refuses a terminal task with `INVALID_STATE`, recorded `failed`.
- `merge` also requires the task to be `ready_to_merge`, unless the PR is already merged. Once the PR is merged, `merge_sha` is recorded when known (a missing `mergeCommit` is read once more, after `MERGE_SHA_RETRY_S`; an unknown sha never overwrites a recorded one). A task already `merged` is a noop success (`noop: true`). If the task left `ready_to_merge` for any other state meanwhile, the call still succeeds, with `state_conflict: {"expected": "ready_to_merge", "actual": …}`, for the orchestrator or the maintainer to reconcile; it is also recorded as a `hook_events` `alert` (hook `tool:merge`), so `snapshot` shows it.
- `merge` is refused by the default merge policy until child 7 installs one (`merge_policy.install`).

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
    token_kinds: frozenset[str] = frozenset({"run"})

def register(spec: ToolSpec) -> None   # CONFLICT if the name exists
def invoke(name, *, key, task_id, args, token, actor, ctx=None) -> dict
```

- `invoke(ctx=None)` builds `default_ctx()`, which has **no model-policy resolver** (QS-405): a `spawn` through it with no caller `model` is `USAGE`. Pass a `Ctx` with `resolve_model` set (the CLI passes `Deps.resolve_model`).

- `argv_step(...)`, `task_state_guard(...)` and `compose(...)` are helpers. They are not frozen.
- Every registered tool is reachable as `cp.py tool <name> --task T --key K [--args-file F] --token …`.

## Hooks

| hook | wired | input → output | failure |
|---|---|---|---|
| `Stop` (orchestrator) | child 9 writes `hooks-settings --role orchestrator` into main's pin; **6a must not ship before** | stdin `session_id` → `{"decision": "block", "reason": …}` when a message waits (pop it), or once when no `wait` runs; a loop guard allows and records a `hook_events` `alert` | fail open |
| `PreToolUse` (every registered session; matcher `Bash\|Edit\|Write\|SendMessage`) | nodes: `claude --bg --model <m> --settings '<hooks-settings --role node, plus the policy's effortLevel>'` at spawn | `tool_name`, `tool_input` → `hookSpecificOutput.permissionDecision: "deny"` for DB access outside `cp.py` (any session), `gh pr merge` (registered sessions), `SendMessage` from a superseded or stopped session | fail open |
| `pre-push` (git) | a common-dir shim installed by `tool worktree-create` (marker `# qs-control-plane pre-push shim v2`) | git's ref lines → exit 1 refuses `QS_<N>_<k>` refs everywhere, and, in a registered worktree, a stopped node, a foreign ref, or a missing or stale `QS_CP_TOKEN`. On a path several tasks registered, it judges a non-terminal task first, newest first | fail closed only for a proven-registered worktree |

**The DB-access rule** (`PreToolUse`, Bash): a segment that names `harness_state.db` is denied unless it starts with a read-only program (`grep`, `rg`, `git`, `ls`, `sed -n`, `cat`, `head`, `tail`, `wc`, `find`, `echo`) or is a `cp.py` call. Even then:

- a redirect onto the DB file itself (`harness_state.db`, `-wal`, `-shm`) is denied, inside a `cp.py` segment too; a sibling such as `> harness_state.db.json`, `.bak`, `-wal.bak` or `-backup.sql` is allowed;
- `find` stays read-only only without `-delete`, `-exec`, `-execdir`, `-ok`, `-okdir`, `-fprint`, `-fprint0`, `-fprintf` or `-fls`, quoted or not (`'-delete'` is `-delete`).

Whether `--settings` hooks survive a bare `claude --bg --resume` is unverified, an open point for 6b. So are two model points (QS-405), which 6b answers and records here: whether a resumed node keeps the model and effort it was spawned with (`resume` passes no flag — any flag forks a copy), and whether `effortLevel` through `--settings` takes effect on a `--bg` session at all. `pre-push` is git-level, so it applies regardless.

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
| `DIGEST_MAX_BYTES` | 16 KiB | `reports` |

## Seams

| seam | real | in tests |
|---|---|---|
| `Clock` | `SystemClock` | `FakeClock`: `sleep` advances time |
| `Runner` | `subprocess.run` with an environment delta | `FakeRunner`: records argv, cwd, environment delta and `detach` |
| `ProcessProbe` | `kill(pid, 0)` plus `ps -o lstart=` (UTC, C locale; a failed `ps` is unknown), and `killpg` | `FakeProbe`: a fresh pid per call, reaped when the CLI call returns |
| `ClaudeCli` | `claude agents --json`, `claude --bg …`, `claude stop <id>` | `FakeClaude`: a scripted listing |
| `ProcessSetup` | `setpgid`, `signal.signal` | a recorder: pytest's own process never changes group or gets a handler |
| `Popen` | `subprocess.Popen`, for `ensure` | a recorder |
| `faults.hit(name)` | no-op | `faults.arm(name, exc, skip=n)` |
| `merge_policy` | refuses | `install(fn)` |
| `Deps.resolve_model` → `Ctx.resolve_model` (QS-405) | `cli._policy_resolver`: `models.spawn_policy`, imported at call time — the Control Plane's one import outside the standard library (a test pins it) | the same resolver, or a stand-in; child 15 reuses the seam |
| `export.LEDGER_SECTIONS` | empty: "No ledger recorded." | #375 fills it |
| the daemon's `tick_hooks` | none | child 14's active loop. A tick hook must return within `STALE_AFTER_S` or beat the lease itself (`daemon.beat(conn, clock)`); the daemon beats before and after every hook. `ensure` SIGKILLs only a daemon whose heartbeat is at least `STALE_AFTER_S` old, so a hook that keeps this contract is never killed |

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
- **`--settings` hooks after a bare resume:** unverified.
- **The model and effort of a resumed node, and `effortLevel` through `--settings` on a `--bg` session:** unverified (QS-405; 6b's resume check records the result).
- **The orchestrator's hooks are unwired until child 9.**
