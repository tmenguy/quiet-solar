#!/usr/bin/env python3
"""Discover task context for the current worktree.

Static agents call this once on startup to learn everything they need
about the current task: issue, title, branch, story file, PR number,
worktree path, and harness.

Usage::

    python scripts/qs/context.py            # JSON to stdout
    python scripts/qs/context.py --issue 42 # force a specific issue

Source of truth:

- ``issue``: parsed from ``git branch --show-current`` (``QS_<N>``)
- ``title``, ``labels``, ``kind``/``target``/``scale``/``lane``,
  ``parent_epic``: from one ``gh issue view <N>`` call (QS-332 — the
  axis parsing lives in :mod:`targets`)
- ``story_file``: ``docs/stories/QS-<N>.story.md`` (if it exists)
- ``pr_number``: from ``gh pr list --head <branch>`` (if open)
- ``worktree``: current working directory
- ``harness``: from :mod:`scripts.qs.harness`

The two ``gh`` calls behind the issue fields and ``pr_number`` are the
only network work here and dominate startup latency, so they are fetched
**concurrently** — do not re-serialize them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import targets  # type: ignore[import-not-found]

from harness import detect as detect_harness  # type: ignore[import-not-found]

from utils import (  # type: ignore[import-not-found]
    find_latest_review_fix,
    find_pr_for_branch,
    find_story_file,
    get_current_branch,
    get_issue_from_branch,
    get_repo_root,
    output_json,
    run_gh,
)


def _empty_issue_fields() -> dict:
    """The degraded issue-fields shape — also the ``_settle`` default at
    both settle call sites (QS-332 review I1). A fresh dict every call so
    no caller can mutate a shared default.
    """
    return {"title": "", "labels": [], "body": ""}


def _issue_fields(issue: int) -> dict:
    """Fetch ``title``, ``labels`` (names) and ``body`` in ONE ``gh`` call.

    Returns :func:`_empty_issue_fields` on any failure (non-zero exit,
    unparseable JSON) — degraded fields, never an exception, matching the
    old ``_issue_title`` contract.
    """
    result = run_gh(
        ["issue", "view", str(issue), "--json", "title,labels,body"],
        check=False,
        # Concurrent with the PR lookup — see ``utils.run``'s ``stdin``.
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return _empty_issue_fields()
    try:
        data = json.loads(result.stdout)
        return {
            "title": data.get("title", ""),
            "labels": [lb["name"] for lb in data.get("labels", [])],
            "body": data.get("body", ""),
        }
    except (json.JSONDecodeError, TypeError, KeyError):
        return _empty_issue_fields()


def _settle(future: Future | None, default: Any) -> tuple[Any, BaseException | None]:
    """Retrieve a future's outcome as ``(value, exception)`` without raising.

    Draining *both* futures before anything propagates is the point:
    ``concurrent.futures`` does not log an unretrieved future exception, so
    a sibling failure would otherwise be discarded without a trace
    (review fix #01 S1). Prior art:
    ``quality_gate.py::_run_cheap_gates_parallel`` — which catches
    ``Exception``; this catches ``BaseException`` deliberately, because a
    worker's ``BaseException`` would otherwise skip the drain and reopen
    that exact hole on a narrow path (review fix #02 R1). The caller
    re-raises, so nothing is swallowed.

    The ``done()``/``exception()`` guard separates the two things a
    ``BaseException`` here can mean: the *worker's* outcome (settle it) or
    an interrupt landing in *our own* ``result()`` wait (propagate it). The
    latter must not be settled — doing so swallowed the interrupt,
    discarded the worker's real failure, and mislabelled the interrupt as a
    concurrent ``gh`` failure (review fix #03 A). ``exception(timeout=0)``
    does not block on a done future and returns ``None`` for a successful
    one.
    """
    if future is None:
        return default, None
    try:
        return future.result(), None
    except BaseException as exc:
        if not future.done() or future.exception(timeout=0) is not exc:
            raise
        return default, exc


def _note_sibling(exc: BaseException, sibling: BaseException | None) -> None:
    """Attach ``sibling`` to ``exc`` so a concurrent failure stays visible."""
    if sibling is not None:
        exc.add_note(f"concurrent gh call also failed: {sibling!r}")


def build_context(issue_override: int | None = None) -> dict:
    """Assemble the context dictionary for the current task.

    The ``title`` and ``pr_number`` lookups are issued concurrently; both
    outcomes are always retrieved before either can propagate, and a hard
    failure surfaces deterministically (title first) carrying any sibling
    failure as a note.
    """
    branch = get_current_branch()
    issue = issue_override if issue_override is not None else get_issue_from_branch(branch)
    # ``0`` is reported faithfully but is not a usable issue number, so
    # guard explicitly instead of relying on truthiness — which conflates
    # ``QS_0`` with "no issue" and silently discarded ``--issue 0``
    # (review fix #01 N1; ``main()`` rejects non-positive overrides).
    has_issue = issue is not None and issue > 0

    # The two ``gh`` calls are independent (one needs only ``issue``, the
    # other only ``branch``) and dominate startup, so they overlap.
    # ``subprocess.run`` blocks in ``os.waitpid`` with the GIL released, so
    # the two children genuinely run at once. ``shutdown(wait=True)`` on
    # ``with``-exit is deliberate: it joins the other future rather than
    # abandoning a live subprocess — do not "fix" it with
    # ``cancel_futures=True``.
    #
    # Thread-pool exhaustion (low ``ulimit -u`` / pids cgroup) is an
    # accepted risk: ``submit`` queues the work item *before* the thread
    # start that raises, so an "inline fallback" cannot un-queue it and
    # double-executes the call (review fix #02 M1). Do not add one.
    with ThreadPoolExecutor(max_workers=2) as pool:
        title_future = pool.submit(_issue_fields, issue) if has_issue else None
        pr_future: Future | None = None
        try:
            # Inside the ``try`` so the drain below covers it: if this
            # ``submit`` raises (thread exhaustion), ``title_future`` is
            # already in flight and would otherwise never be settled
            # (review fix #03 C). The window *before* the ``try`` — the
            # title submit itself — is not covered.
            pr_future = (
                pool.submit(find_pr_for_branch, branch, stdin=subprocess.DEVNULL)
                if branch
                else None
            )
            # Local git work runs while both gh calls are in flight.
            story_path: Path | None = find_story_file(issue) if has_issue else None
            review_fix_path: Path | None = find_latest_review_fix(issue) if has_issue else None
        except BaseException as local_exc:
            # The local work failed with both children still running: drain
            # them so neither exception is discarded, then let the local
            # error propagate as the primary one (review fix #01 S1).
            for _value, exc in (
                _settle(title_future, _empty_issue_fields()),
                _settle(pr_future, None),
            ):
                _note_sibling(local_exc, exc)
            raise
        fields, title_exc = _settle(title_future, _empty_issue_fields())
        pr_info, pr_exc = _settle(pr_future, None)

    # Raised outside the ``with`` so the pool has already joined both
    # children. A hard ``gh`` failure still propagates unchanged in type
    # and exit code (behavioural delta 1).
    if title_exc is not None:
        _note_sibling(title_exc, pr_exc)
        raise title_exc
    if pr_exc is not None:
        raise pr_exc

    # QS-332: the lane axes are APPENDED after the existing keys, in this
    # order — labels, kind, target, scale, lane, parent_epic. That order is
    # the emission contract `tests/qs/test_context.py`'s byte-identity pin
    # is written against; do not reorder.
    axes = targets.parse_axes(fields["labels"])
    return {
        "harness": detect_harness(),
        "branch": branch,
        "issue": issue,
        "title": fields["title"],
        "story_file": str(story_path) if story_path else "",
        "story_exists": bool(story_path),
        "latest_review_fix": str(review_fix_path) if review_fix_path else "",
        "pr_number": pr_info["pr_number"] if pr_info else None,
        "pr_url": pr_info["url"] if pr_info else "",
        "worktree": str(get_repo_root()),
        "labels": fields["labels"],
        "kind": axes["kind"],
        "target": axes["target"],
        "scale": axes["scale"],
        "lane": axes["lane"],
        "parent_epic": targets.parse_parent_epic(fields["body"]),
    }


def _issue_number(raw: str) -> int:
    """argparse ``type`` for ``--issue``: reject non-positive numbers.

    ``--issue 0`` used to be silently discarded in favour of the
    branch-derived issue, and ``--issue -1`` was truthy enough to issue a
    doomed ``gh issue view -1`` and then degrade to an empty title with
    exit 0. Failing at the boundary beats a useless context
    (review fix #01 N1).
    """
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"issue number must be positive, got {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover the current task's context.")
    parser.add_argument(
        "--issue", type=_issue_number, default=None, help="Force a specific issue number."
    )
    args = parser.parse_args()

    ctx = build_context(issue_override=args.issue)
    output_json(ctx)
    sys.exit(0)


if __name__ == "__main__":
    main()
