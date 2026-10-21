"""The per-merge export (§11): a task summary written into the task's worktree.

``docs/stories/QS-<issue>.summary.md`` lands with the PR (child 7 calls it
before the merge). Nothing tracked is ever written into the main checkout.
``LEDGER_SECTIONS`` is the seam #375 fills.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from . import errors, reports, tasks

LedgerSection = tuple[str, Callable[[sqlite3.Connection, str], str]]
LEDGER_SECTIONS: list[LedgerSection] = []


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


_ATX_RE = re.compile(r" {0,3}(#{1,6})(?=[ \t]|$)(.*)")  # a CommonMark ATX heading
_FENCE_RE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")  # a code fence's opener or closer
_SETEXT_RE = re.compile(r" {0,3}(?:=+|-+)[ \t]*")  # a setext heading's underline, when it follows a text line


def _value_lines(value: str) -> list[str]:
    """A digest value's lines, rendered so it cannot break the summary's own outline (#424 D5, fix #01 F1).

    Leading blank lines are dropped. A heading is pushed three levels down (capped at 6); a line inside a code fence
    is kept as it is, and a fence still open at the value's end is closed; a setext underline gets a blank line before
    it (a `---` becomes a thematic break, a `===` plain text). Everything else is kept verbatim.
    """
    lines = value.rstrip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    out: list[str] = []
    fence: str | None = None  # the open fence's marker
    for line in lines:
        match = _FENCE_RE.fullmatch(line)
        if fence is not None:
            out.append(line)
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            continue
        if match and not (match[1][0] == "`" and "`" in match[2]):  # a backtick fence's info has no backtick
            fence = match[1]
            out.append(line)
            continue
        heading = _ATX_RE.fullmatch(line)
        if heading:
            out.append("#" * min(6, len(heading[1]) + 3) + heading[2])
            continue
        if _SETEXT_RE.fullmatch(line) and out and out[-1].strip():
            out.append("")
        out.append(line)
    if fence is not None:
        out.append(fence)
    return out


def _digest_lines(body: str) -> list[str]:
    """A digest as ``### <Key>`` blocks in ``DIGEST_KEYS`` order (#424 D5); a body off the contract, raw."""
    digest = reports.parse_digest(body)
    if digest is None:
        return [body.rstrip(), ""]
    lines: list[str] = []
    for key in reports.DIGEST_KEYS:
        lines += [f"### {key.capitalize()}", "", *(_value_lines(digest[key]) or ["_None._"]), ""]
    return lines


def render_summary(conn: sqlite3.Connection, task_id: str) -> str:
    task = tasks.get(conn, task_id)
    issue = task["issue_number"]
    lines = [f"# QS-{issue} — {task['title']}", "", f"Task {task['id']} · issue #{issue} · state `{task['state']}`", ""]

    lines += ["## Digest", ""]
    digest = conn.execute("SELECT body FROM digests WHERE task_id = ?", (task_id,)).fetchone()
    lines += _digest_lines(digest["body"]) if digest is not None else ["_No digest recorded._", ""]

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
