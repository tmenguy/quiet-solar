"""The finding ledger's schema (#375): four tables and their vocabularies.

Named by what it holds, not by its version, so a renumbering at a catch-up
leaves no wrong name. Like ``schema_v1`` it imports nothing from the package
(``ledger`` imports ``db``, which imports ``migrations``, which imports this).
"""

from __future__ import annotations

PHASES = ("plan", "build")
REVIEWER_SOURCES = ("reviewer", "coderabbit", "global_review", "cross_run_review")
AUTHORITY_SOURCES = ("ci", "gate", "detector", "orchestrator", "maintainer")
SOURCES = REVIEWER_SOURCES + AUTHORITY_SOURCES
SEVERITIES = ("must_fix", "should_fix", "nice_to_have")
CLASSES = (*SEVERITIES, "out_of_scope")
STATES = ("open", "resolved", "rejected", "deferred", "settled")
# Checked in Python, not by an SQL CHECK: a later child extends it without a migration.
CATEGORIES = (
    "correctness",
    "edge-case",
    "test",
    "security",
    "performance",
    "design",
    "scope",
    "docs",
    "style",
    "ci",
    "gate",
    "other",
)
BLAST_VALUES = ("ok", "doubt", "too_large")
EVENT_KINDS = ("classify", "state")


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


STATEMENTS: tuple[str, ...] = (
    f"""CREATE TABLE rounds (
        task_id TEXT NOT NULL REFERENCES tasks(id),
        phase TEXT NOT NULL CHECK (phase IN {_in(PHASES)}),
        round INTEGER NOT NULL CHECK (round >= 1),
        base_sha TEXT,
        head_sha TEXT,
        actor TEXT NOT NULL,
        started_at TEXT NOT NULL,
        PRIMARY KEY (task_id, phase, round)
    )""",
    f"""CREATE TABLE findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT REFERENCES runs(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        phase TEXT NOT NULL CHECK (phase IN {_in(PHASES)}),
        round INTEGER NOT NULL CHECK (round >= 0),
        source TEXT NOT NULL CHECK (source IN {_in(SOURCES)}),
        reviewer TEXT,
        severity TEXT NOT NULL CHECK (severity IN {_in(SEVERITIES)}),
        classification TEXT CHECK (classification IN {_in(CLASSES)}),
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        body TEXT NOT NULL,
        file TEXT,
        symbol TEXT,
        line_start INTEGER,
        line_end INTEGER,
        fingerprint TEXT NOT NULL,
        title_norm TEXT NOT NULL,
        replay_key TEXT NOT NULL UNIQUE,
        matched_id INTEGER REFERENCES findings(id),
        flags TEXT NOT NULL DEFAULT '[]',
        unchanged_lines INTEGER CHECK (unchanged_lines IN (0, 1)),
        state TEXT NOT NULL CHECK (state IN {_in(STATES)}),
        decided_seq INTEGER NOT NULL,
        reason TEXT,
        resolved_sha TEXT,
        ci_sha TEXT,
        integration_id INTEGER REFERENCES integrations(id),
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX findings_fingerprint ON findings (fingerprint)",
    "CREATE INDEX findings_task ON findings (task_id, phase, state)",
    f"""CREATE TABLE finding_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id INTEGER NOT NULL REFERENCES findings(id),
        at TEXT NOT NULL,
        actor TEXT NOT NULL,
        round INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN {_in(EVENT_KINDS)}),
        from_value TEXT,
        to_value TEXT,
        reason TEXT,
        commit_sha TEXT,
        cause_id INTEGER REFERENCES findings(id)
    )""",
    f"""CREATE TABLE blast_radius (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        value TEXT NOT NULL CHECK (value IN {_in(BLAST_VALUES)}),
        head_sha TEXT NOT NULL,
        review TEXT NOT NULL,
        reason TEXT,
        actor TEXT NOT NULL,
        at TEXT NOT NULL
    )""",
)

TABLES: tuple[str, ...] = ("rounds", "findings", "finding_events", "blast_radius")
