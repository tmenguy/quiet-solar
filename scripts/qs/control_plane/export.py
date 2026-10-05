"""The per-merge export (§11): a task summary written into the task's worktree.

``docs/stories/QS-<issue>.summary.md`` lands with the PR (child 7 calls it
before the merge). Nothing tracked is ever written into the main checkout.
``LEDGER_SECTIONS`` is the seam #375 fills.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from . import errors, tasks

LedgerSection = tuple[str, Callable[[sqlite3.Connection, str], str]]
LEDGER_SECTIONS: list[LedgerSection] = []


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def render_summary(conn: sqlite3.Connection, task_id: str) -> str:
    task = tasks.get(conn, task_id)
    issue = task["issue_number"]
    lines = [f"# QS-{issue} — {task['title']}", "", f"Task {task['id']} · issue #{issue} · state `{task['state']}`", ""]

    lines += ["## Digest", ""]
    digest = conn.execute("SELECT body FROM digests WHERE task_id = ?", (task_id,)).fetchone()
    lines += [digest["body"].rstrip() if digest is not None else "_No digest recorded._", ""]

    lines += ["## Acceptance criteria", ""]
    criteria = conn.execute("SELECT * FROM criteria WHERE task_id = ? ORDER BY idx", (task_id,)).fetchall()
    if criteria:
        lines += ["| # | criterion | state | validated |", "|---|---|---|---|"]
        lines += [f"| {c['idx']} | {_cell(c['text'])} | {c['state']} | {c['validated_at'] or 'no'} |" for c in criteria]
    else:
        lines.append("_No criteria recorded._")
    lines.append("")

    lines += ["## Decisions", ""]
    decisions = conn.execute("SELECT * FROM decisions WHERE task_id = ? ORDER BY id", (task_id,)).fetchall()
    lines += [f"- {d['text']} — {d['reason']} ({d['source']}, {d['at']})" for d in decisions] or [
        "_No decisions recorded._"
    ]
    lines.append("")

    lines += ["## PR and merge", ""]
    pr = f"PR #{task['pr_number']} ({task['pr_url']})" if task["pr_number"] is not None else "No PR"
    lines += [f"{pr} · merge {task['merge_sha'] or 'pending'}", ""]

    lines += ["## Ledger", ""]
    if LEDGER_SECTIONS:
        for title, render in LEDGER_SECTIONS:
            lines += [f"### {title}", "", render(conn, task_id).rstrip(), ""]
    else:
        lines += ["No ledger recorded.", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_summary(conn: sqlite3.Connection, task_id: str, worktree: Path) -> Path:
    """Write the summary into a **linked** worktree; the main checkout is refused."""
    wt = Path(worktree).resolve()
    if not (wt / ".git").is_file():
        raise errors.CpError(
            "POLICY_REFUSED", f"{wt} is not a linked worktree: nothing is written in the main checkout"
        )
    issue = tasks.get(conn, task_id)["issue_number"]
    if issue is None:
        raise errors.CpError("INVALID_STATE", f"task {task_id} has no issue number")
    out = wt / "docs" / "stories" / f"QS-{issue}.summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_summary(conn, task_id))
    return out
