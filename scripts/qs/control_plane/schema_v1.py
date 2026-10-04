"""Schema v1 (§4): one tuple of single SQL statements, run one by one with ``execute``."""

from __future__ import annotations

TASK_STATES = (
    "proposed",
    "ready",
    "planning",
    "contracted",
    "building",
    "ready_to_merge",
    "merged",
    "validated",
    "blocked",
    "dropped",
)
NODE_STATES = ("spawning", "running", "idle", "taken_over", "reaped", "stopped", "superseded")


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


STATEMENTS: tuple[str, ...] = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)",
    "CREATE TABLE counters (kind TEXT PRIMARY KEY, next INTEGER NOT NULL)",
    """CREATE TABLE runs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('open', 'closed')),
        global_plan TEXT,
        created_at TEXT NOT NULL,
        closed_at TEXT
    )""",
    """CREATE TABLE run_leases (
        run_id TEXT PRIMARY KEY REFERENCES runs(id),
        epoch INTEGER NOT NULL,
        nonce TEXT NOT NULL,
        session_id TEXT NOT NULL,
        short_id TEXT,
        session_name TEXT,
        permission_mode TEXT,
        full_grant INTEGER NOT NULL DEFAULT 0,
        claimed_at TEXT NOT NULL,
        name_bound_session_id TEXT,
        pending_bind_session_id TEXT
    )""",
    """CREATE TABLE run_sessions (
        run_id TEXT NOT NULL REFERENCES runs(id),
        epoch INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        claimed_at TEXT NOT NULL,
        superseded_at TEXT,
        superseded_by TEXT,
        PRIMARY KEY (run_id, epoch)
    )""",
    f"""CREATE TABLE tasks (
        id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES runs(id),
        parent_id TEXT REFERENCES tasks(id),
        issue_number INTEGER,
        title TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('epic', 'feature', 'bug')),
        target TEXT,
        is_deliverable INTEGER NOT NULL DEFAULT 0,
        deliverable_id TEXT REFERENCES tasks(id),
        item_k INTEGER,
        next_item_k INTEGER NOT NULL DEFAULT 1,
        state TEXT NOT NULL CHECK (state IN {_in(TASK_STATES)}),
        blocked_from TEXT,
        lane TEXT,
        worktree TEXT,
        branch TEXT,
        pr_number INTEGER,
        pr_url TEXT,
        ci_state TEXT,
        ci_sha TEXT,
        merge_sha TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (deliverable_id, item_k)
    )""",
    """CREATE TABLE run_roots (
        run_id TEXT NOT NULL REFERENCES runs(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        PRIMARY KEY (run_id, task_id)
    )""",
    """CREATE TABLE work_list (
        run_id TEXT NOT NULL REFERENCES runs(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        PRIMARY KEY (run_id, task_id)
    )""",
    """CREATE TABLE task_deps (
        task_id TEXT NOT NULL REFERENCES tasks(id),
        depends_on TEXT NOT NULL REFERENCES tasks(id),
        PRIMARY KEY (task_id, depends_on),
        CHECK (task_id != depends_on)
    )""",
    """CREATE TABLE criteria (
        task_id TEXT NOT NULL REFERENCES tasks(id),
        idx INTEGER NOT NULL,
        text TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'met', 'waived')),
        validated_at TEXT,
        PRIMARY KEY (task_id, idx)
    )""",
    """CREATE TABLE task_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        at TEXT NOT NULL,
        actor TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        note TEXT
    )""",
    f"""CREATE TABLE nodes (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        generation INTEGER NOT NULL,
        name TEXT NOT NULL UNIQUE,
        nonce TEXT NOT NULL,
        session_id TEXT,
        short_id TEXT,
        permission_mode TEXT,
        state TEXT NOT NULL CHECK (state IN {_in(NODE_STATES)}),
        spawn_tool_key TEXT,
        launch_at TEXT,
        spawned_at TEXT NOT NULL,
        taken_over_at TEXT,
        handed_back_at TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE (task_id, generation)
    )""",
    """CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(id),
        recipient TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (length(kind) BETWEEN 1 AND 32),
        sender TEXT NOT NULL,
        payload TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('queued', 'in_flight', 'acked', 'dead')),
        visible_at TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        receipt TEXT,
        popped_by TEXT,
        dedupe_key TEXT,
        created_at TEXT NOT NULL,
        acked_at TEXT,
        UNIQUE (run_id, dedupe_key)
    )""",
    "CREATE INDEX messages_queue ON messages (run_id, recipient, state, id)",
    """CREATE TABLE questions (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES runs(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        message_id INTEGER REFERENCES messages(id),
        text TEXT NOT NULL,
        blocking INTEGER NOT NULL DEFAULT 0,
        state TEXT NOT NULL CHECK (state IN ('open', 'asked', 'answered')),
        answer TEXT,
        reason TEXT,
        created_at TEXT NOT NULL,
        answered_at TEXT
    )""",
    """CREATE TABLE decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(id),
        task_id TEXT REFERENCES tasks(id),
        text TEXT NOT NULL,
        reason TEXT NOT NULL,
        source TEXT NOT NULL,
        at TEXT NOT NULL
    )""",
    """CREATE TABLE digests (
        task_id TEXT PRIMARY KEY REFERENCES tasks(id),
        body TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        node_id TEXT REFERENCES nodes(id),
        phase TEXT NOT NULL,
        round INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('converged', 'continuing', 'blocked')),
        summary TEXT NOT NULL,
        fields TEXT NOT NULL,
        at TEXT NOT NULL
    )""",
    """CREATE TABLE integrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_task_id TEXT NOT NULL REFERENCES tasks(id),
        deliverable_id TEXT NOT NULL REFERENCES tasks(id),
        item_tip TEXT NOT NULL,
        merge_commit TEXT,
        result TEXT NOT NULL CHECK (result IN ('ok', 'gate_red', 'conflict', 'noop', 'error')),
        tool_call_key TEXT,
        at TEXT NOT NULL
    )""",
    """CREATE TABLE locks (
        name TEXT PRIMARY KEY,
        holder_kind TEXT NOT NULL CHECK (holder_kind IN ('process', 'session')),
        holder_pid INTEGER,
        holder_pid_start TEXT,
        holder_pgid INTEGER,
        holder_session_id TEXT,
        holder_epoch INTEGER,
        cohold_pid INTEGER,
        cohold_pid_start TEXT,
        cohold_pgid INTEGER,
        holder_actor TEXT NOT NULL,
        token_subject TEXT NOT NULL,
        purpose TEXT,
        acquired_at TEXT NOT NULL
    )""",
    """CREATE TABLE cap_slots (
        cap TEXT NOT NULL,
        slot INTEGER NOT NULL,
        holder_pid INTEGER NOT NULL,
        holder_pid_start TEXT,
        holder_pgid INTEGER,
        holder_actor TEXT NOT NULL,
        acquired_at TEXT NOT NULL,
        PRIMARY KEY (cap, slot)
    )""",
    """CREATE TABLE tool_calls (
        tool TEXT NOT NULL,
        key TEXT NOT NULL,
        args_hash TEXT NOT NULL,
        run_id TEXT,
        task_id TEXT,
        actor TEXT NOT NULL,
        args TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('started', 'succeeded', 'failed')),
        holder_pid INTEGER,
        holder_pid_start TEXT,
        holder_pgid INTEGER,
        outputs TEXT NOT NULL DEFAULT '{}',
        result TEXT,
        exit_code INTEGER,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        PRIMARY KEY (tool, key)
    )""",
    """CREATE TABLE waiters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        pid INTEGER NOT NULL,
        pid_start TEXT,
        started_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL
    )""",
    """CREATE TABLE daemon_lease (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        pid INTEGER,
        pid_start TEXT,
        schema_version INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        heartbeat_at TEXT
    )""",
    """CREATE TABLE hook_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hook TEXT NOT NULL,
        session_id TEXT,
        decision TEXT NOT NULL CHECK (decision IN ('allow', 'block', 'deny', 'alert')),
        detail TEXT NOT NULL,
        at TEXT NOT NULL
    )""",
)

TABLES: tuple[str, ...] = (
    "meta",
    "counters",
    "runs",
    "run_leases",
    "run_sessions",
    "tasks",
    "run_roots",
    "work_list",
    "task_deps",
    "criteria",
    "task_history",
    "nodes",
    "messages",
    "questions",
    "decisions",
    "digests",
    "reports",
    "integrations",
    "locks",
    "cap_slots",
    "tool_calls",
    "waiters",
    "daemon_lease",
    "hook_events",
)
