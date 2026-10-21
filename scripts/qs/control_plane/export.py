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


# Possessive (`*+`): a line of nested markers never backtracks exponentially (fix #03 H1).
_PREFIX = r"(?:[ \t]*>[ \t]?|[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+)*+"  # blockquote and list-item markers
_ATX_RE = re.compile(rf"({_PREFIX}[ \t]*)(#{{1,6}})(?=[ \t]|$)(.*)")  # a heading, nested or not
_SETEXT_RE = re.compile(r"(?:[ \t]*>[ \t]?)*+[ \t]*(?:=+|-+)[ \t]*")  # a setext underline, under a text line
_DEFINITION_RE = re.compile(rf"({_PREFIX}[ \t]*)\[(?=[^\]]*\]:)")  # a footnote / link-reference definition
_LINE_END_RE = re.compile(r"\r\n|\r|\n")  # CommonMark's line ends (not `str.splitlines`'s)


def _value_lines(value: str) -> list[str]:
    """A digest value as a blockquote that cannot break the summary's own outline (#424 D5, fixes #01 to #04).

    Every line is quoted (``> ``): the blank line after the value ends the quote, which closes any fence, list or
    HTML block left open inside it. Inside, line by line, with no Markdown state at all (a code-fence tracker
    falls out of step with the renderer — review rounds 1 to 4 found a new way each time):

    * leading blank lines are dropped;
    * a heading, nested in a quote or a list item or not, at any indentation, is pushed three levels down (capped
      at 6, its prefix kept);
    * a setext underline under a text line gets a blank line before it (a ``---`` becomes a thematic break, a
      ``===`` plain text);
    * a footnote or link-reference definition (``[^n]:`` / ``[x]:`` at the line's start) is escaped (``\\[``):
      it would otherwise act on the whole summary.

    Inside a code block these rewrites are cosmetic (a ``# comment`` reads ``#### comment``): the safe side.
    Whitespace is CommonMark's: spaces and tabs only.
    """
    lines = _LINE_END_RE.split(value.rstrip(" \t\r\n"))
    while lines and not lines[0].strip(" \t"):
        lines.pop(0)
    out: list[str] = []
    for line in lines:
        heading = _ATX_RE.fullmatch(line)
        if heading:
            out.append(heading[1] + "#" * min(6, len(heading[2]) + 3) + heading[3])
            continue
        if _SETEXT_RE.fullmatch(line) and out and out[-1].strip(" \t>"):
            out.append("")
        definition = _DEFINITION_RE.match(line)
        out.append(line if definition is None else f"{definition[1]}\\{line[definition.end(1) :]}")
    return [f"> {line}" if line else ">" for line in out]


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
