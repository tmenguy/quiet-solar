"""The CodeRabbit tools (#425 D7): ``tool coderabbit-trigger`` asks for a head's review, ``tool coderabbit`` ingests it.

In this repo CodeRabbit reviews only on request. Until 2026-10-26 every pushed head got the commit status
``CodeRabbit`` = "Review skipped: manual review required…"; a ``@coderabbitai review`` comment moved it to
``pending`` "Review in progress", then "Review completed". Since 2026-10-27 (PR #433) CodeRabbit reports through a
**check run** named ``CodeRabbit`` of the ``coderabbitai`` app instead: ``completed`` / ``skipped`` "Review
skipped" on a push, ``in_progress`` once requested, then ``completed`` / ``success`` "Finished". Both signals are
read, the newer wins (``head_state``). The review object (``commit_id`` = the head) is posted before the
"completed" signal, and some completed reviews post none at all (nothing to say). This module assumes CodeRabbit
stays on manual review here: with auto-review on, the trigger answers ``already`` and ``CR_MAX_REVIEWS`` bounds
nothing.

- ``tool coderabbit-trigger`` (args ``{"head"}``, one step ``trigger``): the task's ``triggered`` results reaching
  ``CR_MAX_REVIEWS`` → ``ceiling`` (no ``gh`` call); a head whose CodeRabbit state is ``completed`` or
  ``in_progress`` → ``already`` (no comment, not counted); else ``gh pr comment <n> --body "@coderabbitai review"``
  → ``triggered``. No probe: a crash between the comment and the record can post it twice on a takeover
  (accepted: a duplicate request is harmless).
- ``tool coderabbit`` (args ``{"head"}``, one step ``ingest``): the head's latest build round (none → ``BUSY``);
  the bot's reviews of the head → their inline comments (thread replies dropped, de-duplicated by id) → one
  ``source: coderabbit`` finding each, at most ``CR_MAX_FINDINGS``, most severe first, on that round. No review
  object: a ``completed`` head state at least ``CR_SETTLE_S`` old is a review with nothing to say; anything else
  is ``BUSY`` ("not reviewed yet"). Review-body comments (Nitpick, Outside diff range, Duplicate) are not
  ingested; ``body_comments`` says whether there were some.

A failed or unparseable ``gh`` answer is ``BUSY``: released, nothing recorded, the same key replays. Both tools
take a run or a node token (a node: its own task) and refuse — before the claim, so no key is spent — a terminal
task, a work item, a task with no PR and a ``head`` that is not 40 hex characters.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from . import errors, ledger, tools
from .node_policy import CR_MAX_FINDINGS, CR_MAX_REVIEWS, CR_SETTLE_S

TRIGGER = "coderabbit-trigger"
INGEST = "coderabbit"
BOT = "coderabbitai[bot]"
REVIEWER = "coderabbitai"  # the `reviewer` of every ingested finding
STATUS_CONTEXT = "CodeRabbit"
CHECK_NAME = "CodeRabbit"  # the check run CodeRabbit reports through since 2026-10-27 (M1)
CHECK_APP = "coderabbitai"
CHECK_IN_PROGRESS = frozenset({"queued", "in_progress", "waiting", "requested", "pending"})
_NOT_A_REVIEW_RE = re.compile(r"rate limited|skipped", re.IGNORECASE)
_EPOCH = datetime.min.replace(tzinfo=UTC)
GH_TIMEOUT_S = 120
TITLE_MAX = 120
_HEAD_RE = re.compile(r"[0-9a-fA-F]{40}")

SEVERITY_LABELS = {"Critical": "must_fix", "Major": "should_fix", "Minor": "nice_to_have", "Trivial": "nice_to_have"}
CATEGORY_LABELS = {  # a new CodeRabbit label is a one-line edit here
    "Functional Correctness": "correctness",
    "Data Integrity & Integration": "correctness",
    "Potential issue": "correctness",
    "Stability & Availability": "edge-case",
    "Security & Privacy": "security",
    "Performance & Scalability": "performance",
    "Maintainability & Code Quality": "design",
}
_RANK = {"must_fix": 0, "should_fix": 1, "nice_to_have": 2}
# One header segment: `**<label>**` or `_<label>_`; segments joined by ` | ` (2026-10-21 and before) or ` · `
# (2026-10-26 on, the label then carrying a `<picture>` icon).
_SEGMENT_RE = re.compile(r"\*\*(.+?)\*\*|_(.+?)_")
_SEPARATOR_RE = re.compile(r"\s+[|·]\s+")
_TAG_RE = re.compile(r"<[^>]*>")
_LEAD_RE = re.compile(r"^[\W_]+")
_BOLD_LINE_RE = re.compile(r"\*\*((?:(?!\*\*).)+)\*\*")
_BODY_SECTION_RE = re.compile(r"<summary>[^\n]*?(?:Nitpick|Outside diff range|Duplicate) comments", re.IGNORECASE)


# --------------------------------------------------------------------------- parsing a comment


def _label(segment: str) -> str:
    """A segment's label: its HTML (a `<picture>` icon) and its leading emoji / punctuation removed."""
    return _LEAD_RE.sub("", _TAG_RE.sub("", segment)).strip()


def _header_labels(line: str) -> list[str] | None:
    """The labels of ``line`` when it is a header: every segment wrapped, one of them a known label."""
    parts = _SEPARATOR_RE.split(line.strip())
    labels = []
    for part in parts:
        m = _SEGMENT_RE.fullmatch(part)
        if m is None:
            return None
        labels.append(_label(m.group(1) or m.group(2)))
    if any(lbl in SEVERITY_LABELS or lbl in CATEGORY_LABELS for lbl in labels):
        return labels
    return None


def _find_header(body: str) -> tuple[int | None, list[str]]:
    for i, line in enumerate(body.splitlines()):
        labels = _header_labels(line)
        if labels is not None:
            return i, labels
    return None, []


def header(body: str) -> tuple[str, str]:
    """``(severity, category)`` from the comment's header line; ``should_fix`` / ``other`` when it has none."""
    _, labels = _find_header(body)
    severity = next((SEVERITY_LABELS[lbl] for lbl in labels if lbl in SEVERITY_LABELS), "should_fix")
    category = next((CATEGORY_LABELS[lbl] for lbl in labels if lbl in CATEGORY_LABELS), "other")
    return severity, category


def _title(body: str) -> str | None:
    """The first line wholly in bold that is not the header and sits outside every ``<details>``."""
    at, _ = _find_header(body)
    depth = 0
    for i, line in enumerate(body.splitlines()):
        opened, closed = line.count("<details"), line.count("</details>")
        if depth == 0 and opened == 0 and i != at:
            m = _BOLD_LINE_RE.fullmatch(line.strip())
            if m is not None:
                return m.group(1)
        depth = max(0, depth + opened - closed)
    return None


def _lines(comment: Mapping[str, Any]) -> tuple[Any, Any]:
    if comment.get("line") is not None:
        start, end = comment.get("start_line") or comment["line"], comment["line"]
    elif comment.get("original_line") is not None:
        start, end = comment.get("original_start_line") or comment["original_line"], comment["original_line"]
    else:
        return None, None
    if isinstance(start, int) and isinstance(end, int) and start > end:
        start = end  # an inverted pair collapses to its end
    return start, end


def to_item(comment: Mapping[str, Any]) -> dict[str, Any]:
    """One inline comment → one ledger item (``source: coderabbit``), not yet validated."""
    body = str(comment.get("body") or "")
    severity, category = header(body)
    path = comment.get("path")
    start, end = _lines(comment)
    title = (_title(body) or "").replace(ledger.SEP, "").strip()[:TITLE_MAX]
    if not ledger.norm(title):
        where = f"{path}:{end}" if end is not None else str(path)
        title = f"CodeRabbit: {where}"[:TITLE_MAX]
    return {
        "severity": severity,
        "category": category,
        "title": title,
        "body": body,
        "file": path,
        "line_start": start,
        "line_end": end,
        "reviewer": REVIEWER,
    }


def items(comments: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The valid items, most severe first and at most ``CR_MAX_FINDINGS``, and how many were dropped."""
    valid: list[dict[str, Any]] = []
    dropped = 0
    for comment in comments:
        try:
            valid += ledger.validate_items([to_item(comment)], "coderabbit")
        except errors.CpError:
            dropped += 1  # one bad comment never fails the batch
    valid.sort(key=lambda item: _RANK[item["severity"]])
    keep = valid[:CR_MAX_FINDINGS]
    return keep, dropped + len(valid) - len(keep)


# --------------------------------------------------------------------------- GitHub


def _busy(what: str, detail: str) -> errors.CpError:
    return errors.CpError("BUSY", f"{what}: {detail}; replay the same key later")


def _slurp(ctx: tools.StepCtx, path: str) -> list[Any]:
    """``gh api --paginate --slurp <path>``: the list of pages; ``BUSY`` on a failed or unparseable answer."""
    res = ctx.runner.run(["gh", "api", "--paginate", "--slurp", path], cwd=ctx.main, timeout=GH_TIMEOUT_S)
    if not res.ok:
        raise _busy(f"gh api {path}", f"exited {res.returncode}")
    try:
        pages = json.loads(res.stdout)
    except ValueError as exc:
        raise _busy(f"gh api {path}", "unparseable answer") from exc
    if not isinstance(pages, list):
        raise _busy(f"gh api {path}", "unexpected answer")
    return pages


def _pages(ctx: tools.StepCtx, path: str) -> list[Any]:
    """Every page's list flattened (a list endpoint); ``BUSY`` on a failed or odd answer."""
    pages = _slurp(ctx, path)
    if not all(isinstance(page, list) for page in pages):
        raise _busy(f"gh api {path}", "unexpected answer")
    return [entry for page in pages for entry in page]


def _check_runs(ctx: tools.StepCtx, head: str) -> list[dict[str, Any]]:
    """The head's ``CodeRabbit`` check runs of the ``coderabbitai`` app. The endpoint answers an object per page
    (``{total_count, check_runs}``), not a list; ``BUSY`` on a failed or odd answer."""
    path = f"repos/{{owner}}/{{repo}}/commits/{head}/check-runs?check_name={CHECK_NAME}"
    pages = _slurp(ctx, path)
    if not all(isinstance(page, dict) and isinstance(page.get("check_runs"), list) for page in pages):
        raise _busy(f"gh api {path}", "unexpected answer")
    return [
        run
        for page in pages
        for run in page["check_runs"]
        if isinstance(run, dict)
        and run.get("name") == CHECK_NAME
        and isinstance(run.get("app"), dict)
        and run["app"].get("slug") == CHECK_APP
    ]


def _stamp(value: Any) -> datetime | None:
    try:
        at = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def _status_state(status: Mapping[str, Any]) -> tuple[str, datetime | None]:
    at = _stamp(status.get("created_at"))
    if status.get("state") == "pending":
        return "in_progress", at
    if "Review completed" in str(status.get("description")):
        return "completed", at
    return "not_requested", at


def _check_state(run: Mapping[str, Any]) -> tuple[str, datetime | None]:
    if run.get("status") in CHECK_IN_PROGRESS:
        return "in_progress", _stamp(run.get("started_at"))
    at = _stamp(run.get("completed_at"))
    output = run.get("output")
    if run.get("status") != "completed" or run.get("conclusion") != "success" or not isinstance(output, dict):
        return "not_requested", at
    said = f"{output.get('title') or ''}\n{str(output.get('summary') or '').partition(chr(10))[0]}"
    if _NOT_A_REVIEW_RE.search(said):  # the title or the summary's first line: never a step of its table
        return "not_requested", at
    return "completed", at


def _order(entry: tuple[str, datetime | None, int, Any]) -> tuple[bool, datetime, int, int]:
    """The latest entry: by its time (an unparseable one first), a check run over a status, then by id."""
    _, at, kind, ident = entry
    return at is not None, at or _EPOCH, kind, ident if isinstance(ident, int) else -1


def head_state(ctx: tools.StepCtx, head: str) -> tuple[str, datetime | None]:
    """The head's CodeRabbit state from the newer of its ``CodeRabbit`` statuses and check runs (M1: CodeRabbit
    reports through a check run since 2026-10-27), and that entry's time (``None`` when unparseable).

    - ``in_progress``: a ``pending`` status; a ``queued`` / ``in_progress`` check run (at its ``started_at``).
    - ``completed``: a "Review completed" status; a ``completed`` / ``success`` check run (at its
      ``completed_at``) whose title or summary's first line does not say "rate limited" or "skipped".
    - ``not_requested``: neither exists, "Review skipped…", "Review rate limited", any other conclusion.

    The latest entry is chosen deterministically: by ``(time, kind, id)``. A failed or odd ``gh`` answer of
    either endpoint is ``BUSY``."""
    entries: list[tuple[str, datetime | None, int, Any]] = [
        (*_status_state(s), 0, s.get("id"))
        for s in _pages(ctx, f"repos/{{owner}}/{{repo}}/commits/{head}/statuses")
        if isinstance(s, dict) and s.get("context") == STATUS_CONTEXT
    ]
    entries += [(*_check_state(run), 1, run.get("id")) for run in _check_runs(ctx, head)]
    if not entries:
        return "not_requested", None
    state, at, _, _ = max(entries, key=_order)
    return state, at


# --------------------------------------------------------------------------- the steps


def _head(args: Mapping[str, Any]) -> str:
    return str(args["head"]).lower()


def _trigger(ctx: tools.StepCtx) -> None:
    task_id = ctx.task["id"]
    with ctx.write() as conn:
        triggered = conn.execute(
            "SELECT count(*) FROM tool_calls WHERE tool = ? AND task_id = ? AND state = 'succeeded'"
            " AND json_extract(result, '$.status') = 'triggered'",
            (TRIGGER, task_id),
        ).fetchone()[0]
        if triggered >= CR_MAX_REVIEWS:
            ctx.record_output({"status": "ceiling"})
            return
    state, _ = head_state(ctx, _head(ctx.args))  # outside any transaction (a network call)
    if state in ("completed", "in_progress"):
        with ctx.write():
            ctx.record_output({"status": "already"})
        return
    pr = str(ctx.task["pr_number"])
    res = ctx.runner.run(
        ["gh", "pr", "comment", pr, "--body", "@coderabbitai review"], cwd=ctx.main, timeout=GH_TIMEOUT_S
    )
    if not res.ok:
        raise _busy(f"gh pr comment {pr}", f"exited {res.returncode}")
    with ctx.write():
        ctx.record_output({"status": "triggered"})


def _ingest(ctx: tools.StepCtx) -> None:
    task_id, head, pr = ctx.task["id"], _head(ctx.args), ctx.task["pr_number"]
    with ctx.write() as conn:
        found = conn.execute(
            "SELECT max(round) FROM rounds WHERE task_id = ? AND phase = 'build' AND lower(head_sha) = ?",
            (task_id, head),
        ).fetchone()[0]
    if found is None:
        raise errors.CpError("BUSY", f"no build round for this head yet ({head}); replay the same key later")
    reviews = [
        r
        for r in _pages(ctx, f"repos/{{owner}}/{{repo}}/pulls/{pr}/reviews")
        if isinstance(r, dict)
        and type(r.get("id")) is int  # an entry without an integer id (a `bool` is none) is skipped (S5)
        and isinstance(r.get("user"), dict)
        and r["user"].get("login") == BOT
        and str(r.get("commit_id")).lower() == head
    ]
    out: dict[str, Any] = {"status": "reviewed", "review_ids": [r["id"] for r in reviews], "ids": [], "dropped": 0}
    out["body_comments"] = any(_BODY_SECTION_RE.search(str(r.get("body") or "")) for r in reviews)
    if not reviews:
        state, at = head_state(ctx, head)
        settled = at is not None and (ctx.clock.now() - at).total_seconds() >= CR_SETTLE_S
        if state != "completed" or not settled:
            raise errors.CpError("BUSY", f"{head} is not reviewed yet ({state}); replay the same key later")
        with ctx.write():  # a completed review with nothing to say
            ctx.record_output(out)
        return
    wanted = set(out["review_ids"])
    seen: set[int] = set()
    comments = []
    for comment in _pages(ctx, f"repos/{{owner}}/{{repo}}/pulls/{pr}/comments"):
        if not isinstance(comment, dict):
            continue
        review_id = comment.get("pull_request_review_id")
        if not (type(review_id) is int and review_id in wanted):  # type first: unhashable, or `True == 1` (R3)
            continue
        cid = comment.get("id")
        if comment.get("in_reply_to_id") is not None or (type(cid) is int and cid in seen):
            continue  # a thread reply, or the same comment twice
        if type(cid) is int:  # only an integer id de-duplicates: an id-less comment never swallows the next
            seen.add(cid)
        comments.append(comment)
    batch, out["dropped"] = items(comments)
    with ctx.write() as conn:
        if batch:
            w = ledger.writer(conn, ctx._call.token, task_id)
            out["ids"] = ledger.open_findings_locked(
                conn, ctx.clock, w, task_id=task_id, phase="build", source="coderabbit", round_=int(found), batch=batch
            )["ids"]
        ctx.record_output(out)


def _steps(step: tools.Step) -> Any:
    def build(task: tools.TaskRow, args: Mapping[str, Any]) -> Sequence[tools.Step]:
        """Refusals when the steps are built: before the claim, so a refused call spends no key."""
        if task["state"] not in tools.NON_TERMINAL_STATES:
            raise errors.CpError("INVALID_STATE", f"task {task['id']} is {task['state']}")
        tools._not_an_item(task)
        tools._need(task, "pr_number")
        head = args.get("head")
        if not isinstance(head, str) or not _HEAD_RE.fullmatch(head):
            raise errors.CpError("USAGE", "args.head must be a commit sha (40 hex characters)")
        return (step,)

    return build


def _success(step: str) -> Any:
    def on_success(conn: sqlite3.Connection, ctx: tools.StepCtx) -> dict[str, Any]:
        return dict(ctx.outputs[step])

    return on_success


SPECS: tuple[tools.ToolSpec, ...] = (
    tools.ToolSpec(
        TRIGGER,
        _steps(tools.Step("trigger", _trigger)),
        guard=tools.non_terminal,
        on_success=_success("trigger"),
        token_kinds=frozenset({"run", "node"}),
    ),
    tools.ToolSpec(
        INGEST,
        _steps(tools.Step("ingest", _ingest)),
        guard=tools.non_terminal,
        on_success=_success("ingest"),
        token_kinds=frozenset({"run", "node"}),
    ),
)
NAMES = frozenset(spec.name for spec in SPECS)


def register_coderabbit_tools() -> None:
    """Register both tools; idempotent (``cli`` calls it at import and in ``main``, the tests after a reset)."""
    for spec in SPECS:
        if spec.name not in tools.REGISTRY:
            tools.register(spec)
