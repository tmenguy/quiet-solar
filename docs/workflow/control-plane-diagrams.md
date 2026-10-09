# The Control Plane — schema, APIs and sequence diagrams

Companion to [`control-plane.md`](control-plane.md), which is the
normative reference: every table, flag, exit code and constant is defined
there. This page is the visual overview for reviewing
[PR #401](https://github.com/tmenguy/quiet-solar/pull/401) (QS-399, child 5
of epic #369). Diagrams are Mermaid.

---

## 1. Context: who talks to the Control Plane

The Control Plane is plain Python with no LLM: one SQLite file, one CLI,
three hooks and a background daemon. Every LLM session runs it as a
subprocess: the orchestrator (one per run) and its nodes (one
`claude --bg` session per task). No session ever talks to another session
directly; they only talk through the DB, using queues, reports, questions
and tokens.

```mermaid
flowchart LR
    subgraph Humans
        M[Maintainer]
    end

    subgraph Sessions["LLM sessions (Claude Code)"]
        O["Orchestrator session<br/>(interactive, 1 per run)"]
        N1["Node session T1<br/>claude --bg"]
        N2["Node session T2<br/>claude --bg"]
    end

    subgraph CP["Control Plane (main checkout only)"]
        CLI["cp.py CLI<br/>(one JSON object per call)"]
        HK["Hooks<br/>Stop · PreToolUse · pre-push"]
        D["Daemon<br/>(singleton, ticks, migrates)"]
        DB[("harness_state.db<br/>SQLite WAL, schema v1")]
    end

    subgraph Ext["Effects outside the DB"]
        GIT["git / worktrees<br/>(main checkout + linked)"]
        GH["GitHub (gh)<br/>issues, PRs, merges"]
        CC["claude CLI<br/>agents --json, --bg, stop"]
        GATE["quality_gate.py"]
    end

    M -- "talks to" --> O
    O -- "cp.py run/task/msg/wait/tool …<br/>--token run:…" --> CLI
    N1 -- "cp.py report/msg/digest/tool …<br/>--token node:…" --> CLI
    N2 -- "cp.py …" --> CLI
    O -. "Stop / PreToolUse" .-> HK
    N1 -. "PreToolUse" .-> HK
    GIT -. "pre-push (QS_CP_TOKEN)" .-> HK
    CLI --> DB
    HK --> DB
    D --> DB
    CLI -- "ensure (Popen, detached)" --> D
    CLI -- "tool layer" --> GIT
    CLI -- "tool layer" --> GH
    CLI -- "tool spawn/resume" --> CC
    CLI -- "tool gate (cap)" --> GATE
```

The design rests on these invariants:

- **One copy of the code touches the live DB.** Only
  `<MAIN>/scripts/qs/cp.py` may open `<MAIN>/harness_state.db`; any other
  copy gets `PATH_GUARD` (exit 7). Only the daemon, running on `main`,
  migrates it, after a backup.
- **Fencing tokens.** `run:R<n>.<epoch>.<nonce>` / `node:N<n>.<gen>.<nonce>`
  is passed on every effectful command. A superseded token gets
  `STALE_TOKEN` (exit 3) together with self-end instructions.
- **Idempotent tools.** A tool call is keyed by `(tool, key)`: replaying
  the key never repeats an effect.
- **Liveness is tri-state.** A probe answers alive, dead or *unknown*, and
  unknown never frees a holder.

---

## 2. Package architecture

`scripts/qs/control_plane/` has 42 modules (~9.9k lines) behind
`scripts/qs/cp.py`, the active loop (#406) included. Arrows point from a module to the modules it imports;
the infrastructure layer is imported by almost everything.

```mermaid
flowchart TB
    cp["cp.py (entry shim)"] --> cli

    subgraph L4["Entry layer"]
        cli["cli.py<br/>COMMANDS table, arg parsing, exit codes"]
        hooks["hooks.py<br/>Stop · PreToolUse · pre-push · installer · settings"]
    end

    subgraph L3["Effects layer"]
        tools["tools.py<br/>run_recorded · ToolSpec API · 8 built-in tools"]
        wait["wait.py"]
        daemon["daemon.py<br/>run loop · ensure · stop path"]
        export["export.py"]
        snapshot["snapshot.py"]
        merge_policy["merge_policy.py (seam)"]
    end

    subgraph L3b["Active loop (#406): the daemon's tick hooks"]
        activeloop["activeloop.py<br/>Seams · register_builtin"]
        ticks["ticks.py<br/>registry · Throttle"]
        selfcheck["selfcheck.py<br/>code_version · selfcheck hooks"]
        detectors["detectors.py<br/>overlap · stalls · rounds · anomalies · cycles · duplicates"]
        hookroute["hookroute.py<br/>hook_events cursor"]
        ciwatch["ciwatch.py<br/>GitHub · CiWatcher"]
        watchdog["watchdog.py<br/>liveness · wake-up ladder"]
        restore["restore.py<br/>backup hook · restore"]
        alerts["alerts.py<br/>kinds · occurrence engine"]
    end

    subgraph L2["Domain layer"]
        runs["runs.py<br/>leases, takeover"]
        tasks["tasks.py<br/>tree, transitions"]
        nodes["nodes.py<br/>generations, transitions"]
        messages["messages.py<br/>queues"]
        questions["questions.py"]
        reports["reports.py"]
        decisions["decisions.py"]
        locks["locks.py<br/>locks, caps, session locks"]
        tokens["tokens.py<br/>require / fencing"]
    end

    subgraph L1["Infrastructure layer"]
        db["db.py<br/>connect, write/read txns, file_lock, schema check"]
        migrations["migrations.py + schema_v1.py + schema_v2.py"]
        codever["codever.py (leaf)<br/>code version"]
        mergegate["mergegate.py (leaf)<br/>merge gate"]
        backups["backups.py (leaf)<br/>write_copy · rotate"]
        paths["paths.py<br/>path guard, main checkout"]
        liveness["liveness.py<br/>ProcessProbe, ClaudeCli"]
        runner["runner.py (Runner seam)"]
        clock["clock.py (Clock seam)"]
        errors["errors.py (CpError → exit code)"]
        faults["faults.py (test faults)"]
        procsetup["procsetup.py (setpgid, signals)"]
    end

    cli --> tools & wait & daemon & export & snapshot & hooks
    cli --> activeloop & restore & codever & mergegate & alerts
    activeloop --> ticks
    activeloop -.->|"lazy: register_builtin()"| selfcheck & detectors & hookroute & ciwatch & watchdog & restore
    selfcheck & detectors & hookroute & ciwatch & watchdog & restore --> activeloop & alerts
    selfcheck --> codever & mergegate
    restore --> daemon & backups
    tools --> mergegate & codever
    migrations --> backups
    snapshot --> alerts
    alerts --> messages
    cli --> runs & tasks & nodes & messages & questions & reports & decisions & locks
    hooks --> runs & tasks & messages & tokens & wait & liveness
    tools --> locks & nodes & tasks & tokens & hooks & merge_policy & liveness & runner
    wait --> daemon & messages & tokens
    daemon --> migrations & liveness & procsetup
    snapshot --> daemon & tasks
    export --> tasks
    locks --> nodes & tokens & liveness
    nodes --> messages & tasks & tokens
    questions --> messages & tasks & tokens
    reports --> messages & tasks & tokens
    decisions --> tasks & tokens
    messages --> runs & tokens
    tasks --> runs & tokens
    runs --> tokens & liveness
    L2 --> db
    db --> migrations & paths
```

**Seams.** Every external dependency goes through an injectable seam:
`Clock`, `Runner`, `ProcessProbe`, `ClaudeCli`, `ProcessSetup`, `Popen`,
`faults`, `merge_policy`, `export.LEDGER_SECTIONS`, the daemon's
`tick_hooks`, and the active loop's `Seams` (with a `GitHub` for the CI
watcher). The tests run with no real `claude`, `gh`, `git push`, signals,
subprocess or socket under the daemon (a guard test bans them).

---

## 3. Data model (`harness_state.db`, schema v2)

There are 25 tables (`alerts` is v2's, #406); the schema version is `PRAGMA user_version`.
Text ids are `R<n>` (run), `T<n>` (task), `N<n>` (node) and `Q<n>`
(question).

```mermaid
erDiagram
    runs ||--|| run_leases : "current lease (epoch, nonce, session)"
    runs ||--o{ run_sessions : "lease history (superseded_by)"
    runs ||--o{ run_roots : ""
    runs ||--o{ work_list : ""
    runs ||--o{ tasks : "run_id (nullable)"
    tasks ||--o{ tasks : "parent_id (tree)"
    tasks ||--o{ tasks : "deliverable_id + item_k (work items, #400)"
    tasks ||--o{ task_deps : "depends_on"
    tasks ||--o{ criteria : "acceptance criteria"
    tasks ||--o{ task_history : "every transition"
    tasks ||--o{ nodes : "one row per generation"
    runs ||--o{ nodes : ""
    runs ||--o{ messages : "queue per recipient"
    tasks ||--o{ questions : ""
    messages |o--o| questions : "message_id"
    tasks ||--o| digests : "16 KiB max"
    tasks ||--o{ reports : "phase, round, status"
    nodes |o--o{ reports : "node_id"
    tasks ||--o{ integrations : "item_task_id / deliverable_id"
    runs ||--o{ decisions : ""
    runs ||--o{ alerts : "one row per occurrence (v2)"
    messages |o--o| alerts : "message_id"

    runs {
        text id PK
        text name UK
        text state "open|closed"
        text global_plan
    }
    run_leases {
        text run_id PK
        int epoch
        text nonce
        text session_id
        text pending_bind_session_id
    }
    tasks {
        text id PK
        text run_id FK
        text parent_id FK
        text state "proposed…validated|blocked|dropped"
        text blocked_from
        int is_deliverable
        text deliverable_id FK
        int item_k
        int next_item_k
        text worktree
        text branch
        int pr_number
        text merge_sha
    }
    nodes {
        text id PK
        text task_id FK
        int generation
        text name UK "run-T-gN[-rK]"
        text nonce
        text session_id
        text state "spawning…superseded"
        text spawn_tool_key
        text launch_at
    }
    alerts {
        int id PK
        text run_id FK
        text kind "1-32 chars"
        text subject
        text fingerprint UK "kind:sha16:n, per run"
        text payload
        int message_id FK
        text first_seen
        text last_seen
        text cleared_at "NULL while open"
    }
    messages {
        int id PK
        text recipient "orchestrator|node:T"
        text kind
        text state "queued|in_flight|acked|dead"
        text visible_at
        int attempts
        text receipt
        text dedupe_key
    }
```

The coordination tables have no foreign keys. They are keyed by name,
key or singleton id, and their holders are checked for liveness rather
than referenced:

```mermaid
erDiagram
    locks {
        text name PK "integration:B | main-merge | main-checkout"
        text holder_kind "process|session"
        int holder_pid
        text holder_pid_start
        int holder_pgid
        text holder_session_id
        int cohold_pid "a tool of the session holder"
        text holder_actor
        text token_subject
    }
    cap_slots {
        text cap PK "gates"
        int slot PK
        int holder_pid
        text holder_pid_start
        int holder_pgid
    }
    tool_calls {
        text tool PK
        text key PK
        text args_hash "args digest : files digest"
        text state "started|succeeded|failed"
        int holder_pid
        text outputs "JSON, per step"
        text result
    }
    waiters {
        int id PK
        text run_id
        int pid
        text heartbeat_at
    }
    daemon_lease {
        int id PK "always 1"
        int pid
        text pid_start
        int schema_version
        text heartbeat_at
    }
    hook_events {
        int id PK
        text hook
        text decision "allow|block|deny|alert"
        text detail
    }
```

### State machines

```mermaid
stateDiagram-v2
    direction LR
    state "Task" as T {
        [*] --> proposed
        proposed --> ready
        ready --> planning
        planning --> contracted
        contracted --> building
        building --> ready_to_merge
        ready_to_merge --> merged
        merged --> validated
        validated --> [*]
        note right of building
            any non-terminal → blocked (remembers blocked_from)
            blocked → unblock → blocked_from
            any non-terminal → dropped
            node tokens may move only within
            planning … ready_to_merge (+ blocked)
        end note
    }
```

```mermaid
stateDiagram-v2
    direction LR
    [*] --> spawning : tool spawn (reserve)
    spawning --> running : identify OK
    spawning --> reaped : never listed / holder dead
    reaped --> spawning : same-key replay (adopt if listed,<br/>else new nonce + name -r<n>)
    running --> idle
    idle --> running : tool resume
    running --> taken_over : node take-over
    idle --> taken_over
    taken_over --> running : hand-back
    running --> reaped
    idle --> reaped
    spawning --> stopped
    running --> stopped : node stop
    idle --> stopped
    taken_over --> stopped
    reaped --> stopped
    spawning --> superseded : spawn --replace
    running --> superseded
    idle --> superseded
    taken_over --> superseded
    reaped --> superseded
    stopped --> [*]
    superseded --> [*]
```

```mermaid
stateDiagram-v2
    direction LR
    state "Message" as MSG {
        [*] --> queued : msg post (dedupe_key)
        queued --> in_flight : msg pop (receipt, visibility timeout)
        in_flight --> queued : visibility expired (redelivered first)
        in_flight --> acked : msg ack (receipt)
        in_flight --> dead : attempts > MAX_ATTEMPTS (5)
        dead --> acked : ack with last receipt
    }
```

```mermaid
stateDiagram-v2
    direction LR
    state "tool_calls(tool,key)" as TC {
        [*] --> started : claim (before any effect)
        started --> succeeded : all steps + on_success
        started --> failed : step failed / state drift (key spent)
        started --> [*] : BUSY/POLICY/STALE/STOPPED/USAGE<br/>before any effect (claim released)
        started --> started : holder dead → same-key replay takes over
        succeeded --> succeeded : replay returns recorded result
        failed --> failed : replay returns recorded failure
    }
```

---

## 4. API surface

### 4.1 CLI: `<MAIN>/venv/bin/python <MAIN>/scripts/qs/cp.py <command> …`

Every command prints exactly one JSON object: `{"ok": true, …}` or
`{"ok": false, "error": CODE, "detail": …}`.

```mermaid
flowchart LR
    subgraph exempt["exempt: no wait, no daemon start"]
        e1[version]
        e2[daemon]
        e3[ensure]
        e4["hook stop · hook pre-tool-use · hook pre-push"]
        e5["hooks-settings --role node|orchestrator"]
    end
    subgraph read["read: schema check, never waits"]
        r1[session status]
        r2["snapshot [--run]"]
        r3["task show --task"]
        r4["export-summary --task --out-worktree"]
    end
    subgraph write["write: waits for self-migration, takes --token"]
        w1["run open · claim · bind-name · set-mode · set-plan · close"]
        w2["task add · set · state · dep · root · work-list"]
        w3["criteria set · validate · state"]
        w4["question open · ask · answer · decision add"]
        w5["report post · digest put"]
        w6["msg post · pop · ack · wait"]
        w7["node stop · take-over · hand-back"]
        w8["lock acquire · release (integration:*)"]
        w9["tool &lt;name&gt; --task --key [--args-file]"]
    end
```

| exit | codes | meaning |
|---|---|---|
| 0 | | ok |
| 1 | `INTERNAL`, `TOOL_FAILED` | unexpected error, or a tool step failed (recorded) |
| 2 | `USAGE` | bad args, no session id, bad key, bad or non-UTF-8 file |
| 3 | `STALE_TOKEN` | the token is superseded: **end yourself** (payload has instructions) |
| 4 | `STOPPED` | the caller's node is stopped |
| 5 | `SCHEMA_TOO_NEW`, `SCHEMA_PENDING` | code/DB version mismatch (`restart_wait`) |
| 6 | `BUSY` | lock, cap, in-flight call, node cap, unknown liveness: retry |
| 7 | `PATH_GUARD` | this copy of the code may not open this DB |
| 8 | `NOT_FOUND`, `CONFLICT`, `INVALID_STATE` | bad id, mismatch (incl. `claim_taken_over`, cross-run), forbidden transition |
| 9 | `POLICY_REFUSED` | merge policy, foreign hook, write into main, … |

### 4.2 Who may call what (token kinds)

```mermaid
flowchart LR
    RT["run token<br/>(orchestrator)"] --> A1["everything on its own run:<br/>tasks, criteria, questions, decisions,<br/>msg to anyone, wait, node stop/take-over,<br/>all 8 tools"]
    NT["node token<br/>(node of task T)"] --> A2["own task only:<br/>task state (node range), question open,<br/>report post, digest put, msg → orchestrator,<br/>node hand-back, lock acquire (own deliverable),<br/>tool gate · push · pr-create"]
    NT -. "stopped node" .-> A3["still: reads, msg pop/ack<br/>refused: gate, push, pr-create, state writes (exit 4)"]
```

### 4.3 Tools (`tool <name>`)

| tool | token | runs in | locks / cap | probe (what makes a replay safe) |
|---|---|---|---|---|
| `worktree-create` | run | `<MAIN>` | `main-checkout` | `setup_task.py` is idempotent |
| `worktree-cleanup` | run | `<MAIN>` | `main-checkout` | dir absent + unregistered; refuses main / non-linked; shared path only clears the column |
| `gate` | run / node | worktree | `gates` slot | safe to re-run |
| `spawn` | run | worktree | node cap | node row by `spawn_tool_key` + listing by name; adopt or rotate on relaunch |
| `resume` | run | worktree | node cap | node row + listing by name and `startedAt` |
| `issue-create` | run | `<MAIN>` | | `<!-- qs-cp-key -->` marker: recent listing, then search |
| `pr-create` | run / node | worktree | | marker in the branch's PR bodies |
| `push` | run / node | worktree | | `git ls-remote` == `HEAD` |
| `merge` | run | `<MAIN>` | `integration:<b>`, `main-merge` | `gh pr view` == `MERGED`; refused by the default `merge_policy` |

`LOCK_ORDER`: `integration:*` (lexical) → `main-merge` → `main-checkout`
→ gate slot.

### 4.4 Python API frozen for later children (#400, child 7, child 14)

```mermaid
classDiagram
    class ToolSpec {
        +name: str
        +steps(task, args) Sequence~Step~
        +locks(task, args) Sequence~str~
        +cap: str | None
        +probe(StepCtx) dict
        +guard(StepCtx)
        +on_success(conn, StepCtx) dict
        +refuse_when_stopped: bool
        +token_kinds: frozenset
    }
    class Step {
        +name: str
        +run(StepCtx) Any
        +inject_token: bool
        +detach: bool
    }
    class StepCtx {
        +task: Row
        +args: Mapping
        +outputs: Mapping
        +runner: Runner
        +claude: ClaudeCli
        +main: Path
        +tool: str
        +key: str
        +clock: Clock
        +write() ContextManager~Connection~
        +record_output(value)
    }
    class tools {
        +register(ToolSpec)
        +invoke(name, key, task_id, args, token, actor) dict
    }
    class locks {
        +hold(names, cap) ContextManager
        +acquire_session(name, session)
        +release_session(name, session)
    }
    class tasks {
        +allocate_item_k(deliverable)
        +record_integration(...)
    }
    class daemon {
        +tick_hooks: Sequence~TickHook~
        +beat(conn, clock)
    }
    tools --> ToolSpec
    ToolSpec --> Step
    Step --> StepCtx
```

---

## 5. Sequence diagrams

### 5.1 Opening a run and the orchestrator's event loop

The orchestrator never polls. It runs `wait` in the background, and the
**Stop hook** keeps the session from going idle while messages are queued
or while nothing is waiting.

```mermaid
sequenceDiagram
    autonumber
    actor M as Maintainer
    participant O as Orchestrator session
    participant CLI as cp.py
    participant DB as harness_state.db
    participant D as Daemon
    participant SH as Stop hook

    M->>O: "start run demo"
    O->>CLI: run open --name demo --title …
    CLI->>DB: BEGIN IMMEDIATE · insert runs, run_leases(epoch 1, nonce)
    CLI->>D: ensure → Popen(cp.py daemon, detached)
    D->>DB: flock singleton · migrate (noop) · daemon_lease
    CLI-->>O: {"run":"R1","token":"run:R1.1.ab12…","daemon":"started"}
    O->>CLI: wait --run R1 --token … (background)
    CLI->>DB: tokens.require · insert waiters row
    loop every --poll (1 s)
        CLI->>DB: heartbeat waiter · re-check token · visible_count(orchestrator)
    end
    Note over O: the session finishes its turn
    O->>SH: Stop {session_id}
    SH->>DB: lease for session? queue head? live waiter?
    SH-->>O: allow (a waiter is alive, the queue is empty)
    Note over DB: later a node posts a message (see 5.3)
    CLI-->>O: wait returns {"pending":1}
    O->>CLI: msg pop --run R1 --as orchestrator --token …
    CLI->>DB: queued → in_flight (receipt, visible_at = now + 900 s)
    CLI-->>O: {id, kind, payload, receipt}
    O->>O: handle the message
    O->>CLI: msg ack ID --receipt …
    CLI->>DB: in_flight → acked
    O->>CLI: wait … (again)
    alt the session tries to stop with messages queued
        O->>SH: Stop
        SH-->>O: {"decision":"block","reason":"N message(s) wait: pop …"}
    else no live waiter
        SH-->>O: block once: "start wait in the background" (then allow + alert)
    end
```

### 5.2 Spawning a node (`tool spawn`): an idempotent, recorded tool

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator
    participant CLI as cp.py tool spawn
    participant DB as harness_state.db
    participant CC as claude CLI
    participant N as Node session (bg)

    O->>CLI: tool spawn --task T1 --key task:T1:spawn --args-file a.json --token run:…
    CLI->>CLI: key charset check · read *_file once (regular, UTF-8) → digest
    CLI->>DB: require(token) · replay(tool,key)?
    alt key already finished
        DB-->>CLI: recorded result
        CLI-->>O: {"replayed":true, …}
    end
    CLI->>DB: claim tool_calls(spawn,key) = started (holder pid/pgid)
    CLI->>DB: probe: own node row? launch pending? (authoritative per step)
    CLI->>CC: claude agents --json (prelude listing)
    CLI->>DB: step reserve: admit_node (node cap) · insert nodes(N1, gen 1, nonce, spawning)
    CLI->>DB: step launch: stamp launch_at
    CLI->>CC: claude --bg --agent … -n demo-T1-g1 --settings <node hooks><br/>prompt + "token node:N1.1.<nonce>" (QS_CP_TOKEN stripped from env)
    CC-->>N: background session starts
    loop identify, until LAUNCH_SETTLE_S
        CLI->>CC: claude agents --json → find name
    end
    CLI->>DB: on_success: nodes N1 spawning → running (session_id) · tool_calls succeeded
    CLI-->>O: {"node_id":"N1","name":"demo-T1-g1","token":"node:N1.1.…"}
```

### 5.3 A node working: reports, questions, the gate, push and PR

```mermaid
sequenceDiagram
    autonumber
    participant N as Node session (T1)
    participant PT as PreToolUse hook
    participant CLI as cp.py
    participant DB as harness_state.db
    participant G as quality_gate.py
    participant GIT as git + pre-push hook
    participant GH as GitHub

    N->>PT: Bash "sqlite3 harness_state.db …"
    PT-->>N: deny: DB access only through cp.py (static rule, holds even with the DB busy)
    N->>CLI: task state --task T1 --to building --token node:…
    CLI->>DB: tokens.require(node, own task, node range) · transition + history
    N->>CLI: tool gate --task T1 --key task:T1:gate1 --token node:…
    CLI->>DB: claim · take gates cap slot (≤ QS_CP_MAX_GATES)
    CLI->>G: quality_gate.py --quick …
    CLI->>DB: release slot · succeeded
    N->>CLI: tool push --task T1 --key task:T1:push1 --token node:…
    CLI->>GIT: git push (env QS_CP_TOKEN=node:… injected for this step only)
    GIT->>CLI: pre-push hook: registered worktree? node stopped? own branch? token current?
    CLI-->>GIT: exit 0 (allow)
    GIT->>GH: push QS_<N>
    N->>CLI: tool pr-create --task T1 … (marker in body)
    CLI->>GH: gh pr create
    N->>CLI: report post --task T1 --phase build --round 1 --status converged …
    CLI->>DB: insert reports · msg → orchestrator (kind report)
    Note over DB: the orchestrator's wait returns {"pending":1} (5.1)
    N->>CLI: question open --task T1 --text-file q.md --blocking
    CLI->>DB: insert questions · msg → orchestrator
```

### 5.4 Crash and replay: why a tool never repeats an effect

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator
    participant C1 as cp.py (attempt 1)
    participant C2 as cp.py (attempt 2)
    participant DB as harness_state.db
    participant GH as GitHub

    O->>C1: tool issue-create --key task:T2:issue --token …
    C1->>DB: claim (tool,key) started · holder = C1 pid/pgid
    C1->>GH: create_issue.py (body has <!-- qs-cp-key: issue-create/task:T2:issue -->)
    GH-->>C1: #512 created
    Note over C1: 💥 crash before persist_outputs
    O->>C2: same command, same key
    C2->>DB: replay? state = started, not finished
    C2->>DB: claim: holder C1 dead (pid + start time + pgid) → take over
    C2->>GH: probe: recent issue listing (consistent), then marker search
    GH-->>C2: #512 carries the marker → step "create" done
    C2->>DB: on_success: tasks.issue_number = 512 · succeeded
    C2-->>O: {"result":{"issue":512}, "replayed":false}
    Note over C2,GH: probe failed / unparseable → BUSY ("replay later"), never "not done"
```

### 5.5 Run takeover and fencing (a superseded session ends itself)

```mermaid
sequenceDiagram
    autonumber
    actor M as Maintainer
    participant O1 as Old orchestrator (S1)
    participant O2 as New orchestrator (S2)
    participant CLI as cp.py
    participant DB as harness_state.db

    O2->>CLI: run claim R1 (session S2)
    CLI->>DB: listing: S1 alive?
    CLI-->>O2: CONFLICT: held by a live session, ask the maintainer, retry with --takeover
    M->>O2: "take it over"
    O2->>CLI: run claim R1 --takeover
    CLI->>DB: CAS on (epoch, session) · epoch 1 → 2 · new nonce · run_sessions S1 superseded_by S2
    CLI-->>O2: token run:R1.2.<new nonce>
    O1->>CLI: msg pop … --token run:R1.1.<old>
    CLI-->>O1: exit 3 STALE_TOKEN {superseded_by:S2, at, instructions:"stop working, end the session"}
    O1->>O1: stops (and the Stop hook allows a superseded session)
    Note over O1,DB: locks held by S1's tokens, its waiter and its SendMessage are fenced too
```

### 5.6 Self-migration: new schema code lands on `main`

```mermaid
sequenceDiagram
    autonumber
    participant X as Any write command (new code)
    participant E as ensure
    participant Dold as Old daemon (schema v1)
    participant Dnew as New daemon (schema v2)
    participant DB as harness_state.db
    participant BK as QS_CP_BACKUP_DIR

    X->>DB: user_version 1 < code 2 → wait for migration (≤ MIGRATE_WAIT_S)
    X->>E: ensure
    E->>DB: daemon_lease: schema 1 (older), pid alive?
    E->>E: singleton flock free? (zombie / exited → skip to start)
    E->>Dold: SIGTERM (only if proven alive with a recorded pid_start)
    Dold->>DB: finishes its tick, clears lease, releases flock
    E->>E: _wait_gone: lease cleared / flock free / pid dead
    alt still alive, same pid_start, heartbeat frozen ≥ STALE_AFTER_S
        E->>Dold: SIGKILL
    else cannot prove gone
        E-->>X: restart_pending → wait exits 5 (restart_wait)
    end
    E->>Dnew: Popen(cp.py daemon)
    Dnew->>Dnew: flock · main checked out on main? (only when migrating)
    Dnew->>BK: backup harness_state.db
    Dnew->>DB: BEGIN IMMEDIATE · migration v2 statements · user_version = 2 · COMMIT
    Dnew->>DB: daemon_lease(schema 2)
    X->>DB: user_version == 2 → proceed
    Note over Dnew,DB: on failure: <db>.migrate-error.json → ensure backs off 300 s,<br/>writers get SCHEMA_PENDING · BUSY is retried, never recorded
```

`ensure` can answer with any of these statuses:

| status | when | what `wait` does |
|---|---|---|
| `already_running` | the lease is fresh, at this schema or newer | keeps polling |
| `started` | there is no live daemon, so a new one is started | keeps polling |
| `restarted` | an older daemon was stopped, or found dead, and replaced | keeps polling |
| `restart_pending` | an older daemon cannot be proven gone | exit 5, `restart_wait` |
| `stale_alive` | a stale daemon of this schema is alive and won't stop | exit 5, `restart_wait` |
| `newer_running` | a stale daemon of a newer schema is not proven dead; nothing is started | commands get `SCHEMA_TOO_NEW` |
| `migrate_failed` | a migration error was recorded less than 300 s ago | `SCHEMA_PENDING` |

The daemon writes its heartbeat before and after every tick hook. A tick
hook must return within `STALE_AFTER_S` (30 s) or call `daemon.beat`
itself. A daemon that does so is never sent SIGKILL.

### 5.7 Merging (`tool merge`) under locks and the merge policy

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator
    participant CLI as cp.py tool merge
    participant DB as harness_state.db
    participant MP as merge_policy (child 7)
    participant GH as GitHub

    O->>CLI: tool merge --task T1 --key task:T1:merge --token run:…
    CLI->>DB: claim · guard: task ready_to_merge
    CLI->>DB: hold [integration:QS_7, main-merge] (LOCK_ORDER, process group)
    CLI->>MP: step policy(task) → refuses by default until child 7 installs one
    CLI->>GH: step head: PR head sha
    CLI->>GH: step merge: gh pr merge --match-head-commit <sha>
    CLI->>GH: gh pr view → mergeCommit (retry once after 2 s)
    CLI->>DB: on_success: merge_sha · ready_to_merge → merged
    alt task left ready_to_merge meanwhile
        CLI->>DB: still succeeded, with state_conflict + hook_events alert
    end
    CLI->>DB: release locks
    CLI-->>O: {"merge_sha":…}
```

### 5.8 Session-held integration lock (contract with #400)

```mermaid
sequenceDiagram
    autonumber
    participant N as Node / orchestrator session
    participant CLI as cp.py
    participant DB as harness_state.db
    participant CC as claude agents --json

    N->>CLI: lock acquire --name integration:QS_7 --session-id S --token …
    CLI->>DB: existing holder?
    alt held by another session
        CLI->>CC: listing (before the txn)
        CLI->>DB: holder dead if: token superseded · absent from a successful listing ·<br/>(listing failed) pid proven dead · run closed · every deliverable terminal
        CLI-->>N: BUSY (holder alive or unknown) / take it over
    end
    CLI->>DB: insert locks(holder_kind=session, session, pid stamp)
    loop integration steps (#400 tools)
        N->>CLI: tool … (co-holder: its process group keeps the lock alive)
        CLI->>DB: hold() inside: a name sorting before a held integration:* → CONFLICT
    end
    N->>CLI: lock release --name integration:QS_7
```

---

## 6. Review history (PR #401)

| round | must-fix | should-fix | outcome |
|---|---|---|---|
| 1 | 3 | 15 | fix plan #01: 24 fixes (text-id ordering, PreToolUse fail-open, spawn `--replace` at cap, tri-state liveness, replay before steps, …) |
| 2 | 0 | 9 | fix plan #02: 21 fixes (listing-then-search probe, relaunch nonce/name rotation, restart path, files read once, run scoping, …) |
| 3 | 0 | 4 | fix plan #03: 11 fixes (UTF-8 before `launch_at`, daemon stop path with the flock as authority, `edit_dep` scoping, …) |
| 4 | 0 | 1 (story text) | fix plan #04: final polish (superseded-launch stop scoped by worktree cwd, SIGKILL only on a heartbeat ≥ `STALE_AFTER_S` old, `newer_running`) |
| 5 | 0 | 1 (story text) | converged on code: 918 tests, 100% coverage; fix plan #05 = story wording only |

Accepted limits, recorded in `control-plane.md` "Conventions":
- `--no-verify` bypasses the pre-push hook.
- `run claim --session-id` trusts its caller.
- The DB-access hook guards against accidents; it is not a sandbox.
- `gh pr merge` run through a wrapper is not caught.
- External effects have a fencing window.
- A launch can come up late (the launch-settle limit).
- `--settings` hooks after a bare resume are unverified.
