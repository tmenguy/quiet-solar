"""Schema v4 (#407): review sessions — ``reviews``, ``review_tasks``, ``review_triggers``.

Named by what it holds, like ``schema_ledger``; it imports nothing from the package, so ``tokens`` can read
its vocabularies without importing ``reviews``. ``kind`` is checked in Python (a later child adds a kind with
no migration); ``state`` and the trigger ``outcome`` are checked by SQL.

The partial unique index on ``reviews(coalesce_key)`` makes two in-flight reviews of one key impossible; the
one on ``review_triggers(tool_key)`` makes one pilot attempt one row. The last statement seeds the review
pilot's ``task_history`` cursor past the existing history (v3's precedent).
"""

from __future__ import annotations

KINDS = ("global", "cross_run")
STATES = ("requested", "launching", "running", "finished", "failed", "abandoned")
IN_FLIGHT = ("requested", "launching", "running")
CLOSED = ("finished", "failed", "abandoned")
OUTCOMES = ("pass", "fail")
HEAD_STATES = ("merged", "branch", "unresolved", "plan")
TRIGGER_OUTCOMES = ("queued", "created", "joined", "skipped")
SOURCES = {"global": "global_review", "cross_run": "cross_run_review"}  # the ledger source each kind writes
AGENTS = {"global": "qs-global-review", "cross_run": "qs-cross-run-review"}  # the agent stem each kind launches
REVIEW_TTL_S = 3600.0  # a launching / running review older than this (from launch_at) is abandoned
REQUEST_TTL_S = 600.0  # a requested / launching review older than this (from launch_at or updated_at) failed
PILOT_PREFIX = "pilot:"
CURSOR_KEY = "review_pilot_cursor"


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


STATEMENTS: tuple[str, ...] = (
    f"""CREATE TABLE reviews (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        coalesce_key TEXT NOT NULL,
        home_run_id TEXT REFERENCES runs(id),
        state TEXT NOT NULL CHECK (state IN {_in(STATES)}),
        agent TEXT NOT NULL,
        model TEXT,
        effort TEXT,
        permission_mode TEXT,
        nonce TEXT,
        grant_nonce TEXT,
        name TEXT NOT NULL,
        session_id TEXT,
        short_id TEXT,
        dir TEXT,
        tool_key TEXT,
        requested_by TEXT NOT NULL,
        summary TEXT,
        error TEXT,
        heads_at TEXT,
        created_at TEXT NOT NULL,
        launch_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL
    )""",
    f"CREATE UNIQUE INDEX reviews_in_flight ON reviews (coalesce_key) WHERE state IN {_in(IN_FLIGHT)}",
    "CREATE INDEX reviews_state ON reviews (state)",
    f"""CREATE TABLE review_tasks (
        review_id TEXT NOT NULL REFERENCES reviews(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        run_id TEXT REFERENCES runs(id),
        base_sha TEXT,
        head_sha TEXT,
        head_state TEXT CHECK (head_state IS NULL OR head_state IN {_in(HEAD_STATES)}),
        outcome TEXT CHECK (outcome IS NULL OR outcome IN {_in(OUTCOMES)}),
        reason TEXT,
        blast_radius_id INTEGER REFERENCES blast_radius(id),
        recorded_at TEXT,
        PRIMARY KEY (review_id, task_id)
    )""",
    "CREATE INDEX review_tasks_task ON review_tasks (task_id, head_sha)",
    f"""CREATE TABLE review_triggers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_id TEXT REFERENCES reviews(id),
        actor TEXT NOT NULL,
        "trigger" TEXT NOT NULL,
        tool_key TEXT,
        outcome TEXT NOT NULL CHECK (outcome IN {_in(TRIGGER_OUTCOMES)}),
        at TEXT NOT NULL
    )""",
    f"CREATE UNIQUE INDEX review_triggers_pilot ON review_triggers (tool_key) WHERE tool_key LIKE '{PILOT_PREFIX}%'",
    # Seeded past the existing history, so the review pilot never replays an old merge (v3's precedent).
    f"INSERT OR REPLACE INTO meta (key, value) SELECT '{CURSOR_KEY}', CAST(COALESCE(max(id), 0) AS TEXT)"
    " FROM task_history",
)

TABLES: tuple[str, ...] = ("reviews", "review_tasks", "review_triggers")
