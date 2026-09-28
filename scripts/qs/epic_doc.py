#!/usr/bin/env python3
"""Epic documents: session state, the landing rule, the epic issue body (QS-340).

The epic × factory lane (``docs/workflow/lanes/epic-factory.md``) has no
PR: its rationale document ``docs/epics/QS-<N>.md`` reaches ``main`` by
one direct commit, and the epic issue is the live scoreboard of its
children. The three destructive-or-subtle steps of that lane are code,
not prose:

- ``status --issue N [--sync]`` — read-only session state for
  ``qs-decompose-epic`` (mode selection) and ``qs-finish-task`` (is the
  worktree safe to discard?). ``--sync`` fast-forwards a reused worktree
  that is behind ``origin/main`` and holds nothing local.
- ``land --issue N --message MSG [--merged P=<blob>]... [--dry-run]`` —
  the **only** way an epic document reaches ``main``: it lands **only the
  epic's own** ``docs/epics/QS-<N>.md``; any other changed path is refused
  (``out-of-scope``). It also refuses a stale ``docs/agents/`` doc, and a
  path ``main`` changed since the worktree's base (unless merged against
  the current ``main`` blob). The landing commit is built on ``origin/main`` with
  plumbing through a temporary index, so the working tree is **never
  modified before the push is verified** — a failure cannot lose the
  draft.
- ``sync-issue --issue N [--rewrite-from FILE]`` — writes the epic issue
  body. By default additive and marker-free: it prepends the
  rationale-document link and adds the missing ``- [ ] #n`` child lines,
  and regenerates the ``- (not filed) <child>`` lines it owns under a
  ``## Children`` heading; it never deletes or rewrites any other line.
  ``--rewrite-from FILE`` (QS-383) is the destructive form used when an
  epic is redesigned: FILE's text replaces the body wholesale, then the
  same additive pass enforces the link and child lines on it. ``-`` reads
  the text from stdin (no scratch file). A child ticked in the current
  body stays ticked; a blank or unreadable FILE is refused.

Contract: JSON on stdout (:func:`utils.output_json`), exit 0 on success
(``ok`` / ``landed`` / ``already-landed`` / ``ok-dry-run`` / ``synced``
/ ``unchanged``), exit 1 on any refusal. Every git / gh call goes through
``utils.run_git`` / ``utils.run_gh`` with ``check=False``; a failure is a
JSON refusal, never a traceback. No side effects at import.

Usage::

    python scripts/qs/epic_doc.py status --issue 369 --sync
    python scripts/qs/epic_doc.py land --issue 369 --message "..." --dry-run
    python scripts/qs/epic_doc.py sync-issue --issue 369
    python scripts/qs/epic_doc.py sync-issue --issue 369 --rewrite-from - <<'QS_EPIC_BODY'
    ...the full new body...
    QS_EPIC_BODY
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import check_doc_drift  # type: ignore[import-not-found]
import targets  # type: ignore[import-not-found]

import utils  # type: ignore[import-not-found]

DOC_PREFIX = "docs/epics/"

_OK_STATUSES = frozenset({"ok", "landed", "already-landed", "ok-dry-run", "synced", "unchanged"})


class Refusal(Exception):  # noqa: N818 — a refusal is a normal outcome, not an error
    """A JSON-reported refusal: ``status`` plus extra payload fields."""

    def __init__(self, status: str, **fields: Any) -> None:
        super().__init__(status)
        self.status = status
        self.fields = fields


def doc_path(issue: int) -> str:
    """Repo-relative path of the epic document for ``issue``."""
    return f"{DOC_PREFIX}QS-{issue}.md"


# ---------------------------------------------------------------------------
# git helpers — every call is check=False and runs from the repo root
# ---------------------------------------------------------------------------


def _git(root: str, args: list[str], **kwargs: Any):
    return utils.run_git(args, check=False, cwd=root, **kwargs)


def _git_ok(root: str, args: list[str], **kwargs: Any) -> str:
    """Run git; return stdout, or raise ``git-error`` on a non-zero exit."""
    result = _git(root, args, **kwargs)
    if result.returncode != 0:
        raise Refusal(
            "git-error",
            command="git " + " ".join(args),
            detail=(result.stderr or result.stdout or "").strip(),
        )
    return result.stdout


def _git_flag(root: str, args: list[str]) -> bool:
    """Run a git predicate: exit 0 → True, 1 → False, anything else → ``git-error``."""
    result = _git(root, args)
    if result.returncode in (0, 1):
        return result.returncode == 0
    raise Refusal(
        "git-error",
        command="git " + " ".join(args),
        detail=(result.stderr or result.stdout or "").strip(),
    )


def _toplevel() -> str:
    result = utils.run_git(["rev-parse", "--show-toplevel"], check=False)
    if result.returncode != 0:
        raise Refusal("git-error", command="git rev-parse --show-toplevel", detail=(result.stderr or "").strip())
    return result.stdout.strip()


def _fetch_main(root: str) -> None:
    _git_ok(root, ["fetch", "origin", "main"])


def _changed_paths(root: str) -> tuple[str, list[str]]:
    """Return ``(base, changed)`` — every path changed since ``origin/main``'s merge-base.

    Committed (``--no-renames``, so a rename contributes both sides) plus
    staged, unstaged and untracked (a porcelain rename entry contributes
    both paths).
    """
    base = _git_ok(root, ["merge-base", "HEAD", "origin/main"]).strip()
    changed: set[str] = set()
    committed = _git_ok(root, ["diff", "--name-only", "-z", "--no-renames", base, "HEAD"])
    changed.update(p for p in committed.split("\0") if p)
    porcelain = _git_ok(
        root,
        ["-c", "core.quotepath=false", "status", "--porcelain", "-z", "--untracked-files=all"],
    )
    fields = porcelain.split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        changed.add(entry[3:])
        if entry[0] in "RC" or entry[1] in "RC":
            # -z porcelain v1: the rename/copy SOURCE is the next field.
            if i < len(fields) and fields[i]:
                changed.add(fields[i])
            i += 1
    return base, sorted(changed)


def _main_blob(root: str, path: str) -> str | None:
    """The ``origin/main:<path>`` blob sha, or ``None`` when absent."""
    out = _git_ok(root, ["ls-tree", "-z", "origin/main", "--", path])
    for entry in out.split("\0"):
        meta, _, name = entry.partition("\t")
        parts = meta.split()
        if name == path and len(parts) == 3 and parts[1] == "blob":
            return parts[2]
    return None


def _worktree_blob(root: str, path: str) -> str | None:
    """The blob sha of the working-tree file (not written), or ``None`` when absent."""
    if not (Path(root) / path).is_file():
        return None
    return _git_ok(root, ["hash-object", "--", path]).strip()


def _blob_content(root: str, sha: str | None) -> str | None:
    if sha is None:
        return None
    try:
        return _git_ok(root, ["cat-file", "blob", sha])
    except UnicodeDecodeError:  # N2: a non-UTF-8 blob is a refusal, not a traceback
        raise Refusal("undecodable", detail=sha) from None


def _read_doc(root: str, path: str) -> str:
    """Read a worktree file as UTF-8; a non-UTF-8 file is a refusal (N2)."""
    try:
        return (Path(root) / path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise Refusal("undecodable", detail=path) from None


def _reset_to_main(root: str) -> None:
    _git_ok(root, ["reset", "--hard", "origin/main"])


# ---------------------------------------------------------------------------
# gh helpers
# ---------------------------------------------------------------------------


def _issue_info(issue: int) -> tuple[list[str], str, str]:
    """``(labels, state, body)`` from one ``gh issue view`` call.

    A lookup failure is ``lookup-failed`` — never "not an epic".
    """
    result = utils.run_gh(["issue", "view", str(issue), "--json", "labels,state,body"], check=False)
    if result.returncode != 0:
        raise Refusal("lookup-failed", issue=issue, detail=(result.stderr or "").strip())
    try:
        data = json.loads(result.stdout)
        labels = [lb["name"] for lb in data.get("labels") or []]
        state = data.get("state") or ""
        body = data.get("body") or ""
    except json.JSONDecodeError, TypeError, KeyError, AttributeError:
        raise Refusal("lookup-failed", issue=issue, detail="invalid JSON from gh") from None
    return labels, state, body


def _require_epic(issue: int, labels: list[str]) -> None:
    if targets.parse_axes(labels)["scale"] != "epic":
        raise Refusal(
            "not-an-epic",
            issue=issue,
            detail=f"issue #{issue} is not scale:epic — epic_doc.py only lands epic documents",
        )


def _repo_url() -> str:
    result = utils.run_gh(["repo", "view", "--json", "url"], check=False)
    if result.returncode != 0:
        raise Refusal("lookup-failed", detail=(result.stderr or "").strip())
    try:
        url = json.loads(result.stdout)["url"]
    except json.JSONDecodeError, TypeError, KeyError:
        raise Refusal("lookup-failed", detail="invalid JSON from gh repo view") from None
    return str(url).rstrip("/")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _all_blobs_on_main(
    changed: list[str], main_blobs: dict[str, str | None], local_blobs: dict[str, str | None]
) -> bool:
    """S4: every changed path's worktree blob already equals its origin/main blob.

    The single predicate ``cmd_land`` uses for its already-landed shortcut and
    ``cmd_status`` uses to report ``landed_not_reset`` — a land whose local
    ``reset --hard`` never ran leaves the worktree byte-equal to main.

    N2: requires at least one changed path with a non-``None`` local blob. A
    doc committed on the branch and then deleted from the working tree is
    absent both locally and on main (``None == None``); without this guard
    that reads as "already landed" and ``--sync`` would reset-drop the only
    commit that still holds the doc.
    """
    return (
        bool(changed)
        and any(local_blobs[p] is not None for p in changed)
        and all(local_blobs[p] == main_blobs[p] for p in changed)
    )


def cmd_status(issue: int, *, sync: bool) -> dict:
    root = _toplevel()
    _fetch_main(root)
    doc = doc_path(issue)
    _base, changed = _changed_paths(root)
    local_modified = doc in changed
    # ``doc_differs`` is the HEAD-vs-origin/main delta on the doc alone (either
    # direction); ``behind`` is whether HEAD trails origin/main on ANY path, so
    # ``--sync`` fast-forwards a clean worktree stale only on lane files /
    # docs/agents/ / templates (S4).
    doc_differs = not _git_flag(root, ["diff", "--quiet", "HEAD", "origin/main", "--", doc])
    behind = int(_git_ok(root, ["rev-list", "--count", "HEAD..origin/main"]).strip()) > 0
    # S3: local commits not on origin/main — so a false ``safe_to_discard`` can be
    # explained even when the delta is not the epic doc (a lane edit, a note).
    unpushed_commits = int(_git_ok(root, ["rev-list", "--count", "origin/main..HEAD"]).strip())
    ancestor = _git_flag(root, ["merge-base", "--is-ancestor", "HEAD", "origin/main"])
    main_blobs = {p: _main_blob(root, p) for p in changed}
    local_blobs = {p: _worktree_blob(root, p) for p in changed}
    landed_not_reset = _all_blobs_on_main(changed, main_blobs, local_blobs)
    safe_to_discard = (not changed and ancestor) or landed_not_reset
    synced = False
    if sync and behind and safe_to_discard:
        if changed:
            # landed-not-reset: the bytes are already on main, so reset instead
            # of a ff-merge that would refuse to clobber the untracked doc.
            _reset_to_main(root)
        else:
            _git_ok(root, ["merge", "--ff-only", "origin/main"])
        synced = True
        behind = False
    main_sha = _main_blob(root, doc)
    on_main = main_sha is not None
    local = (Path(root) / doc).is_file()
    if synced:
        # N1: the sync mutated the worktree to match origin/main exactly, so the
        # snapshot fields computed above are stale — report the post-sync truth.
        local_modified = doc_differs = landed_not_reset = False
        safe_to_discard = True
        changed = []
        unpushed_commits = 0
    diff = ""
    if local_modified:
        before = _blob_content(root, main_sha) or ""
        after = _read_doc(root, doc) if local else ""
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"origin/main:{doc}",
                tofile=doc,
            )
        )
    return {
        "status": "ok",
        "issue": issue,
        "doc": doc,
        "local": local,
        "on_main": on_main,
        "local_modified": local_modified,
        "behind": behind,
        "doc_differs": doc_differs,
        "landed_not_reset": landed_not_reset,
        "safe_to_discard": safe_to_discard,
        "synced": synced,
        "changed": changed,
        "unpushed_commits": unpushed_commits,
        "mode": "RESUME" if on_main else "DECOMPOSE",
        "diff": diff,
    }


# ---------------------------------------------------------------------------
# land
# ---------------------------------------------------------------------------


def _parse_merged(values: list[str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for value in values:
        path, sep, blob = value.rpartition("=")
        if not sep or not path or not blob:
            raise Refusal("bad-arguments", detail=f"--merged expects PATH=<blob>, got {value!r}")
        merged[path] = blob
    return merged


def _drift(root: str, changed: list[str]) -> tuple[dict, list[str]]:
    """Run the doc-drift checker in-process; decide on the parsed report."""
    buf = io.StringIO()
    # S5: the module contract is "JSON refusal, never a traceback". argparse
    # raises SystemExit (a BaseException, not caught by ``except Exception``),
    # and check_doc_drift.main could raise anything — turn both into a refusal.
    try:
        with contextlib.redirect_stdout(buf):
            check_doc_drift.main(["--repo-root", root, "--json", "--paths", *changed])
    except SystemExit as exc:
        raise Refusal("drift", detail=f"check_doc_drift.py exited: {exc}", output=buf.getvalue()) from None
    except Exception as exc:  # noqa: BLE001 — any failure becomes a JSON refusal
        raise Refusal("drift", detail=str(exc), output=buf.getvalue()) from None
    try:
        report = json.loads(buf.getvalue())
    except json.JSONDecodeError:
        raise Refusal("drift", detail="check_doc_drift.py produced no JSON report", output=buf.getvalue()) from None
    warnings = [f"malformed frontmatter: {p}" for p in report.get("malformed_frontmatter") or []]
    warnings += [f"missing covers path: {p}" for p in report.get("missing_covers") or []]
    return report, warnings


def _build_commit(root: str, changed: list[str], message: str) -> str | None:
    """Build the landing commit on ``origin/main`` via a temporary index.

    Returns the commit sha, or ``None`` when the resulting tree equals
    ``origin/main``'s. The working tree and the real index are untouched.
    """
    tmpdir = tempfile.mkdtemp(prefix="qs_epic_index_")
    # A not-yet-existing path: git refuses a pre-created empty index file.
    env = {"GIT_INDEX_FILE": os.path.join(tmpdir, "index")}
    try:
        _git_ok(root, ["read-tree", "origin/main"], env=env)
        for path in changed:
            if (Path(root) / path).is_file():
                blob = _git_ok(root, ["hash-object", "-w", "--", path]).strip()
                _git_ok(
                    root,
                    ["update-index", "--add", "--cacheinfo", f"100644,{blob},{path}"],
                    env=env,
                )
            else:
                _git_ok(root, ["update-index", "--force-remove", "--", path], env=env)
        tree = _git_ok(root, ["write-tree"], env=env).strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    main_tree = _git_ok(root, ["rev-parse", "origin/main^{tree}"]).strip()
    if tree == main_tree:
        return None
    return _git_ok(root, ["commit-tree", tree, "-p", "origin/main", "-m", message]).strip()


def _validate_decomposition(root: str, doc: str) -> None:
    """S2: refuse before pushing when the issue's own doc has no readable Decomposition.

    ``sync-issue`` reads the ``## Decomposition`` table right after a land; a
    doc with no section, a numbered heading (``## 5. Decomposition``) or a
    malformed cell (``TBD``, ``#371 (merged)``) would pass an unchecked land
    and only surface as sync-issue's ``unparseable-decomposition`` after the
    push — leaving FINALIZE half-done. So the issue's own doc is always parsed,
    a missing section included.
    """
    try:
        parse_decomposition(_read_doc(root, doc))
    except UnparseableDecomposition as exc:
        raise Refusal("unparseable-decomposition", doc=doc, detail=str(exc)) from None


def cmd_land(  # noqa: C901 — the nine steps read best as one sequence
    issue: int, *, message: str, merged: list[str], dry_run: bool
) -> dict:
    merged_map = _parse_merged(merged)
    if not message.strip():
        # S8: commit-tree would happily write an empty-message commit to main.
        raise Refusal("bad-arguments", detail="--message must not be empty or whitespace-only")
    root = _toplevel()
    labels, _state, _body = _issue_info(issue)
    _require_epic(issue, labels)
    _fetch_main(root)
    doc = doc_path(issue)
    base, changed = _changed_paths(root)

    # N6: an epic session lands its OWN document and nothing else — not another
    # epic's docs/epics/QS-M.md, not swap files, not *.orig/*.bak backups. The
    # single landable path is exactly ``docs/epics/QS-<issue>.md``.
    offenders = [p for p in changed if p != doc]
    if offenders:
        raise Refusal(
            "out-of-scope",
            offenders=offenders,
            detail=(
                f"an epic document lands alone: the only landable path is {doc} "
                "(no other epic's doc, swap files, *.orig/*.bak or backups)"
            ),
        )

    if not changed:
        if _main_blob(root, doc) is not None:
            return {"status": "already-landed", "paths": [], "reset": False}
        raise Refusal("missing-doc", doc=doc, detail=f"{doc} is neither local nor on main")
    if not (Path(root) / doc).is_file():
        raise Refusal("missing-doc", doc=doc, detail=f"{doc} does not exist in the worktree")

    main_blobs = {p: _main_blob(root, p) for p in changed}
    local_blobs = {p: _worktree_blob(root, p) for p in changed}
    if _all_blobs_on_main(changed, main_blobs, local_blobs):
        # The bytes are already on main (e.g. a push that landed while the
        # local reset never ran): the reset loses nothing, and the doc on main
        # was already validated when it first landed — no need to re-check.
        if dry_run:
            # S1: dry-run stops before building the commit — it never resets.
            return {"status": "ok-dry-run", "already_landed": True, "paths": changed, "reset": False}
        _reset_to_main(root)
        return {"status": "already-landed", "paths": changed, "reset": True}

    # S2: validate the Decomposition table before any plumbing, dry-run and
    # real alike (but after the already-landed shortcut, so cleaning up a
    # landed worktree never re-validates).
    _validate_decomposition(root, doc)

    report, warnings = _drift(root, changed)
    if report.get("stale_docs"):
        raise Refusal(
            "drift", drift=report, warnings=warnings, detail="stale docs/agents/ documents — update them first"
        )

    conflicts: list[dict] = []
    for path in changed:
        if _git_flag(root, ["diff", "--quiet", base, "origin/main", "--", path]):
            continue  # main did not touch this path since the worktree's base
        if main_blobs[path] is not None and merged_map.get(path) == main_blobs[path]:
            continue
        conflicts.append(
            {
                "path": path,
                "main_blob": main_blobs[path],
                "main_content": _blob_content(root, main_blobs[path]),
            }
        )
    if conflicts:
        raise Refusal(
            "conflict",
            conflicts=conflicts,
            detail=(
                "main changed these paths since this worktree's base: merge the "
                "main_content shown here into the local file, then re-run with "
                "--merged <path>=<main_blob>"
            ),
        )

    if dry_run:
        return {"status": "ok-dry-run", "paths": changed, "drift": report, "warnings": warnings}

    # N4: under the single-path N6 scope, ``_build_commit`` can never return
    # ``None`` here — the only changed path is the doc, and the already-landed
    # shortcut above already returned when its bytes equalled main, so the built
    # tree always differs from ``origin/main``'s.
    sha = _build_commit(root, changed, message)
    if sha is None:
        # N6: defensive — the already-landed shortcut above already returned when
        # the built tree equalled main, so ``None`` here can only mean an internal
        # invariant broke. Refuse loudly rather than push a ``None`` ref to main.
        raise Refusal(
            "git-error",
            detail="internal: tree equals main after the already-landed shortcut",
        )
    push = _git(root, ["push", "origin", f"{sha}:refs/heads/main"])
    if push.returncode != 0:
        detail = (push.stderr or push.stdout or "").strip()
        low = detail.lower()
        if re.search(r"gh006|gh013|protected branch|rule violation", low):
            # N3: a protected branch or a repository ruleset rejects every
            # re-run — direct the user to the documented hand-opened-PR fallback.
            hint = (
                "main is protected (branch protection or a repository ruleset): "
                "land cannot push directly. Open the documented hand-opened PR "
                f"carrying `Refs #{issue}` (never create_pr.py, whose `Fixes #{issue}` "
                "would close the epic) — see the epic-factory lane doc."
            )
        elif re.search(r"non-fast-forward|fetch first", low):
            # N3: only a genuine non-fast-forward (main moved) is fixed by a re-run.
            hint = "main moved — re-run to rebuild on the new origin/main"
        else:
            # N3/N7: any other failure (auth, network, …) is not "main moved" —
            # don't send the user in a re-run loop; surface the raw detail.
            hint = "push failed for another reason — inspect `detail` and resolve it"
        raise Refusal("push-rejected", sha=sha, detail=detail, hint=hint)

    _fetch_main(root)
    if not _git_flag(root, ["merge-base", "--is-ancestor", sha, "origin/main"]):
        raise Refusal("verify-failed", sha=sha, detail="the pushed commit is not an ancestor of origin/main")
    _reset_to_main(root)
    return {"status": "landed", "sha": sha, "paths": changed, "drift": report, "warnings": warnings}


# ---------------------------------------------------------------------------
# sync-issue — Decomposition table parsing + additive body update
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One Decomposition row: the child's text and its issue (``None`` = not filed)."""

    child: str
    issue: int | None


_DECOMPOSITION_RE = re.compile(r"^##\s+Decomposition\s*$", re.IGNORECASE)
_H2_RE = re.compile(r"^##\s")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_SEPARATOR_RE = re.compile(r"^\s*:?-+:?\s*$")
_ISSUE_CELL_RES = (re.compile(r"^#(\d+)$"), re.compile(r"^\[#(\d+)\]\([^)]*\)$"))
_NOT_FILED_RE = re.compile(r"^not filed$", re.IGNORECASE)


class UnparseableDecomposition(ValueError):
    """The Decomposition section cannot be read — nothing may be written."""


def _cells(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    return [c.strip() for c in _CELL_SPLIT_RE.split(text)]


def _issue_cell(cell: str) -> int | None:
    for regex in _ISSUE_CELL_RES:
        match = regex.match(cell)
        if match:
            number = int(match.group(1))
            if number < 1:  # N1: #0 (and below) is never a real issue number
                raise UnparseableDecomposition(f"issue number must be >= 1, got {cell!r}")
            return number
    if _NOT_FILED_RE.match(cell):
        return None
    raise UnparseableDecomposition(f"issue cell {cell!r} is not #N, [#N](url) or 'not filed'")


def parse_decomposition(text: str) -> list[Row]:
    """Every row of every ``child`` + ``issue`` table under ``## Decomposition``.

    ``###`` subsections are included; the section ends at the next H2.
    Tables without both columns are ignored. Raises
    :class:`UnparseableDecomposition` on a missing section, a ragged row
    or an unreadable issue cell.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _DECOMPOSITION_RE.match(ln)), None)
    if start is None:
        raise UnparseableDecomposition("no '## Decomposition' section")
    section: list[str] = []
    for line in lines[start + 1 :]:
        if _H2_RE.match(line):
            break
        section.append(line)

    rows: list[Row] = []
    i = 0
    while i < len(section):
        if not section[i].lstrip().startswith("|"):
            i += 1
            continue
        table: list[str] = []
        while i < len(section) and section[i].lstrip().startswith("|"):
            table.append(section[i])
            i += 1
        header = [c.lower() for c in _cells(table[0])]
        if "child" not in header or "issue" not in header:
            continue
        if len(table) < 2 or not all(_SEPARATOR_RE.match(c) for c in _cells(table[1])):
            raise UnparseableDecomposition("decomposition table has no separator row")
        child_col, issue_col = header.index("child"), header.index("issue")
        for line in table[2:]:
            cells = _cells(line)
            if len(cells) != len(header):
                raise UnparseableDecomposition(f"ragged decomposition row: {line!r}")
            child = cells[child_col]
            if not child:  # N1: an empty child cell is not a real child
                raise UnparseableDecomposition(f"empty child cell: {line!r}")
            rows.append(Row(child=child, issue=_issue_cell(cells[issue_col])))
    return rows


_TASK_RE = re.compile(r"^\s*[-*] \[[ xX]\] #(\d+)\b")
_BLOCK_LINE_RE = re.compile(r"^\s*[-*] (?:\[[ xX]\] |\(not filed\) )")
_OWNED_RE = re.compile(r"^\s*[-*] \(not filed\) ")
_CHILDREN_RE = re.compile(r"^##\s+Children\s*$", re.IGNORECASE)
_SECTION_END_RE = re.compile(r"^#{1,2}\s")


def _blocks(content: list[str]) -> list[tuple[int, int]]:
    """``(start, end)`` (end exclusive) of every maximal run of task-list / owned lines."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(content):
        if _BLOCK_LINE_RE.match(content[i]):
            j = i
            while j < len(content) and _BLOCK_LINE_RE.match(content[j]):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


Line = tuple[str, str]  # (content, line ending: "\n" | "\r\n" | "" for the unterminated last line)


def _split_lines(body: str) -> list[Line]:
    """Split on ``\n`` only; ``"".join(c + e ...)`` gives ``body`` back byte-for-byte."""
    pieces = body.split("\n")
    lines: list[Line] = []
    for i, piece in enumerate(pieces):
        if i == len(pieces) - 1:
            if piece:
                lines.append((piece, ""))
        elif piece.endswith("\r"):
            lines.append((piece[:-1], "\r\n"))
        else:
            lines.append((piece, "\n"))
    return lines


def _splice(lines: list[Line], start: int, end: int, new: list[Line]) -> None:
    """Replace ``lines[start:end]`` by ``new``, keeping the body's final terminator.

    Only the line that ends up last in the whole body may be unterminated.
    """
    at_end = end == len(lines)
    final_eol = lines[-1][1] if at_end and lines else None
    lines[start:end] = new
    if at_end and new and final_eol is not None:
        for k in range(start, len(lines) - 1):
            content, eol = lines[k]
            if not eol:
                lines[k] = (content, "\n")
        lines[-1] = (lines[-1][0], final_eol)


def _insert_block(lines: list[Line], at: int, texts: list[str], nl: str) -> None:
    """Insert ``texts`` at ``at`` with one blank line of separation on each side."""
    before = at > 0 and bool(lines[at - 1][0].strip())
    after = at < len(lines) and bool(lines[at][0].strip())
    new = [(t, nl) for t in ([""] if before else []) + texts + ([""] if after else [])]
    if at == len(lines) and lines and not lines[-1][1]:
        lines[-1] = (lines[-1][0], nl)
        new[-1] = (new[-1][0], "")
    lines[at:at] = new


def _owned_by_heading(content: list[str], start: int) -> bool:
    """Whether the block at ``start`` sits directly under a ``## Children`` heading."""
    k = start - 1
    while k >= 0 and not content[k].strip():
        k -= 1
    return k >= 0 and bool(_CHILDREN_RE.match(content[k]))


def _last_text_line(content: list[str], lo: int, hi: int) -> int | None:
    return next((k for k in range(hi - 1, lo - 1, -1) if content[k].strip()), None)


def sync_body(body: str, rows: list[Row], *, link: str | None) -> tuple[str, list[int], list[str]]:
    """Return ``(new_body, added_issue_numbers, owned_children)``.

    ``link`` (the rationale-document line) is prepended when given. Every
    line the function does not own keeps its exact bytes (``\r\n``
    included); a second run over its own output is a no-op.
    """
    nl = "\r\n" if "\r\n" in body else "\n"
    lines = _split_lines(body)
    content = [c for c, _ in lines]

    table_numbers = {r.issue for r in rows if r.issue is not None}
    present = {int(m.group(1)) for c in content if (m := _TASK_RE.match(c))}
    added: list[int] = []
    task_lines: list[str] = []
    for row in rows:
        if row.issue is not None and row.issue not in present and row.issue not in added:
            added.append(row.issue)
            task_lines.append(f"- [ ] #{row.issue} — {row.child}")
    unfiled = [r.child for r in rows if r.issue is None]

    spans = _blocks(content)
    heading = next((k for k, c in enumerate(content) if _CHILDREN_RE.match(c)), None)
    section_end = len(content)
    if heading is not None:
        section_end = next(
            (k for k in range(heading + 1, len(content)) if _SECTION_END_RE.match(content[k])),
            len(content),
        )

    def _mentions_table_number(span: tuple[int, int]) -> bool:
        s, e = span
        return any((m := _TASK_RE.match(content[k])) and int(m.group(1)) in table_numbers for k in range(s, e))

    section_blocks = [(s, e) for s, e in spans if heading is not None and heading < s < section_end]
    if heading is not None:
        # S7(a): the ## Children section owns its anchor — prefer a block inside
        # the section over an earlier body block (e.g. a "Decomposition sketch"
        # list) that merely mentions a filed number and would otherwise capture
        # the new children while ## Children goes stale.
        anchor = next((sp for sp in section_blocks if _mentions_table_number(sp)), None)
        if anchor is None:
            anchor = next(iter(section_blocks), None)
    else:
        anchor = next((sp for sp in spans if _mentions_table_number(sp)), None)

    owned: list[str] = []
    if anchor is not None:
        start, end = anchor
        if anchor in section_blocks:
            # S1: ownership is a property of the whole ## Children section, not
            # of the anchor block alone. When a ``(not filed)`` block sits first
            # and the task-line block (the anchor) follows after a blank line,
            # ``_owned_by_heading(anchor)`` is False even though the section is
            # owned — leaving the unfiled lines unstripped and re-listing a
            # just-filed child both as filed and as ``(not filed)``. A prose
            # line (a wave label) between ``## Children`` and the list blocks
            # makes ``_owned_by_heading`` False for every block, so a section
            # that already carries ``(not filed)`` lines is also owned — else
            # filing a child there would list it both ways forever.
            is_owned = any(
                _owned_by_heading(content, s)
                or any(_OWNED_RE.match(content[k]) for k in range(s, e))
                for s, e in section_blocks
            )
        else:
            is_owned = _owned_by_heading(content, start)
        owned = unfiled if is_owned else []
        owned_texts = [f"- (not filed) {c}" for c in owned]
        extra = task_lines + owned_texts
        if is_owned and anchor in section_blocks:
            # S7(b)/S2: owned (not filed) lines can be spread across several
            # blocks of the section (a list split by a blank line). Strip them
            # from every block, then regenerate: the newly filed task lines go
            # to the anchor, while the regenerated (not filed) lines are
            # re-appended to the block that ALREADY held them (S2 — so an
            # unchanged table never relocates them, and an unchanged table yields
            # an unchanged body), falling back to the anchor when no block held
            # any. Splices run back-to-front so an earlier edit can't shift a
            # later block's indices.
            owned_blocks = [
                (s, e)
                for s, e in section_blocks
                if any(_OWNED_RE.match(content[k]) for k in range(s, e))
            ]
            owned_target = owned_blocks[0] if owned_blocks else anchor
            edits: list[tuple[int, int, list[Line]]] = []
            for s, e in section_blocks:
                kept = [ln for ln in lines[s:e] if not _OWNED_RE.match(ln[0])]
                additions: list[Line] = []
                if (s, e) == anchor:
                    additions += [(t, nl) for t in task_lines]
                if (s, e) == owned_target:
                    additions += [(t, nl) for t in owned_texts]
                edits.append((s, e, kept + additions))
            for s, e, new_lines in sorted(edits, key=lambda x: x[0], reverse=True):
                _splice(lines, s, e, new_lines)
        else:
            kept = [ln for ln in lines[start:end] if not (is_owned and _OWNED_RE.match(ln[0]))]
            _splice(lines, start, end, kept + [(t, nl) for t in extra])
    elif heading is not None:
        last = _last_text_line(content, heading + 1, section_end)
        if last is None:  # an empty ``## Children`` section: the block goes directly under it
            owned = unfiled
            at = heading + 1
        else:
            at = last + 1
        block = task_lines + [f"- (not filed) {c}" for c in owned]
        if block:
            _insert_block(lines, at, block, nl)
    elif task_lines or unfiled:
        owned = unfiled
        block = ["## Children", "", *task_lines, *(f"- (not filed) {c}" for c in owned)]
        last = _last_text_line(content, 0, len(content))
        _insert_block(lines, 0 if last is None else last + 1, block, nl)

    new_body = "".join(c + e for c, e in lines)
    if link is not None:
        new_body = f"{link}{nl}{nl}{new_body}"
    return new_body, added, owned


def _edit_body(issue: int, body: str) -> None:
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 — closed before gh reads it
        "w", encoding="utf-8", suffix=".md", prefix="qs_epic_body_", delete=False
    )
    try:
        with handle:
            handle.write(body)
        result = utils.run_gh(["issue", "edit", str(issue), "--body-file", handle.name], check=False)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
    if result.returncode != 0:
        raise Refusal("edit-failed", issue=issue, detail=(result.stderr or "").strip())


def _read_body_file(path: str) -> str:
    """The replacement body for ``--rewrite-from`` (``-`` = stdin); blank or unreadable is a refusal.

    Both sources are read as bytes and decoded as UTF-8 (a leading BOM
    dropped, ``\r\n`` kept). A closed or interactive stdin is refused
    rather than read — it would crash or hang.
    """
    try:
        if path == "-":
            if sys.stdin is None or sys.stdin.isatty():
                raise Refusal("missing-body-file", path=path, detail="stdin is closed or a terminal — pipe the body in")
            text = sys.stdin.buffer.read().decode("utf-8-sig")
        else:
            text = Path(path).read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError, ValueError, AttributeError) as exc:
        raise Refusal("missing-body-file", path=path, detail=str(exc)) from None
    if not text.strip():
        raise Refusal("empty-body-file", path=path, detail="refusing to blank the epic issue body")
    return text


_TICKED_RE = re.compile(r"^\s*[-*] \[[xX]\] #(\d+)\b")
_UNTICKED_RE = re.compile(r"^(\s*[-*] )\[ \]( #(\d+)\b.*)$")


def carry_ticks(current: str, new: str) -> str:
    """Keep a child ticked across a rewrite.

    A ``- [ ] #n`` line in ``new`` becomes ``- [x] #n`` when ``#n`` is ticked
    in ``current`` — a rewrite never silently un-ticks a finished child.
    Every other line keeps its exact bytes.
    """
    ticked = {int(m.group(1)) for line in current.splitlines() if (m := _TICKED_RE.match(line))}
    lines = _split_lines(new)
    out = []
    for content, end in lines:
        m = _UNTICKED_RE.match(content)
        if m and int(m.group(3)) in ticked:
            content = f"{m.group(1)}[x]{m.group(2)}"
        out.append(content + end)
    return "".join(out)


def _normalised(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip()


def cmd_sync_issue(issue: int, *, rewrite_from: str | None = None) -> dict:
    root = _toplevel()
    labels, state, body = _issue_info(issue)
    _require_epic(issue, labels)
    doc = doc_path(issue)
    doc_file = Path(root) / doc
    if not doc_file.is_file():
        raise Refusal("missing-doc", doc=doc, detail=f"{doc} does not exist in the worktree")
    try:
        rows = parse_decomposition(_read_doc(root, doc))
    except UnparseableDecomposition as exc:
        raise Refusal("unparseable-decomposition", doc=doc, detail=str(exc)) from None

    # --rewrite-from: the text replaces the body as the base; the additive
    # pass below still enforces the link and child lines, and the ticks are
    # carried over its output (a child line it re-adds stays ticked too).
    base = body if rewrite_from is None else _read_body_file(rewrite_from)
    extra = {} if rewrite_from is None else {"rewrite_from": rewrite_from}
    link = None
    # A rewrite text is agent-written: a prose mention of the path is not the
    # link, so only a markdown link to the doc counts there.
    has_link = doc in base if rewrite_from is None else f"[{doc}](" in base
    if not has_link:
        link = f"**Rationale document:** [{doc}]({_repo_url()}/blob/main/{doc})"
    new_body, added, owned = sync_body(base, rows, link=link)
    if rewrite_from is not None:
        new_body = carry_ticks(body, new_body)
        # Line endings (a web-UI edit stores \r\n, a heredoc \n) and the
        # trailing newline a heredoc / `gh -q` adds are not a change.
        if _normalised(new_body) == _normalised(body):
            new_body = body
    if new_body == body:
        return {"status": "unchanged", "state": state, "added": [], "owned": owned, "link_added": False, **extra}
    _edit_body(issue, new_body)
    return {
        "status": "synced",
        "state": state,
        "added": added,
        "owned": owned,
        "link_added": link is not None,
        **extra,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _issue_number(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"issue number must be positive, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="epic_doc.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="session state of the epic worktree")
    status.add_argument("--issue", type=_issue_number, required=True)
    status.add_argument("--sync", action="store_true", help="fast-forward when behind origin/main and safe to discard")

    land = sub.add_parser(
        "land",
        help="land only the epic's own docs/epics/QS-<N>.md on main by direct commit",
    )
    land.add_argument("--issue", type=_issue_number, required=True)
    land.add_argument("--message", required=True, help="commit message, used verbatim")
    land.add_argument(
        "--merged",
        action="append",
        default=[],
        metavar="PATH=BLOB",
        help="accept a conflict on PATH merged against main blob BLOB",
    )
    land.add_argument("--dry-run", action="store_true", help="stop before building the commit")

    sync = sub.add_parser(
        "sync-issue",
        help="sync the epic issue body: additive by default, or rewritten from a file (--rewrite-from)",
    )
    sync.add_argument("--issue", type=_issue_number, required=True)
    sync.add_argument(
        "--rewrite-from",
        metavar="FILE",
        help="replace the body with FILE's text ('-' = stdin), then add the link and child lines",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            payload = cmd_status(args.issue, sync=args.sync)
        elif args.command == "land":
            payload = cmd_land(args.issue, message=args.message, merged=args.merged, dry_run=args.dry_run)
        else:
            payload = cmd_sync_issue(args.issue, rewrite_from=args.rewrite_from)
    except Refusal as refusal:
        payload = {"status": refusal.status, **refusal.fields}
    utils.output_json(payload)
    return 0 if payload["status"] in _OK_STATUSES else 1


if __name__ == "__main__":
    sys.exit(main())
