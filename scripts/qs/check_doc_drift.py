#!/usr/bin/env python3
"""Check that ``docs/agents/`` docs stay in sync with their ``covers:`` source.

A documentation entry under ``docs/agents/`` declares a ``covers:``
list in its YAML frontmatter — the source files it summarizes. This
script:

1. Walks every ``docs/agents/**/*.md``, parses the frontmatter, and
   builds the inverse index ``source -> [docs]``.
2. Validates that every ``covers:`` path actually exists on disk
   (a renamed / deleted source produces exit 2 because the doc still
   claims to describe it).
3. Reads a list of "modified paths" — either from the staged diff
   (``git diff --cached --diff-filter=ACMRD --name-only`` — default)
   or from ``--paths`` (override; what the create-plan / implement
   agents pass).
4. Flags every doc that covers a modified source but was not itself
   co-modified as drift.

Exit codes (AC-7):
    0 — clean
    1 — drift detected (covered source changed, covering doc did not)
    2 — malformed frontmatter, missing ``covers:`` path, or
        non-existent ``--repo-root``

Exit-code precedence (review-fix #01 N4): when both stale docs and
malformed/missing covers are present, **exit 2 wins**. Drift (exit 1)
is reported only when the doc tree is internally consistent.

Output (non-JSON) is human-readable lines on stdout. ``--json``
switches to the machine-readable schema:

.. code-block:: json

    {
      "stale_docs": [
        {"doc": "docs/agents/concepts/X.md",
         "stale_sources": ["custom_components/.../X.py"],
         "last_verified": "YYYY-MM-DD"}
      ],
      "missing_covers": ["docs/agents/concepts/Y.md::custom_components/.../Z.py"],
      "malformed_frontmatter": ["docs/agents/concepts/W.md"]
    }

``covers:`` entries outside ``custom_components/quiet_solar/`` are
ignored (warning on stderr only) — see story QS-185 for the
self-reference paradox rationale.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Resolve paths relative to repo root (parent of scripts/qs/).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent.parent

# Drift is only tracked for source files under this prefix. Off-tree
# ``covers:`` entries get a warning, not an error, so a doc can
# legitimately reference an out-of-scope path (e.g., a workflow doc)
# without tripping exit 2.
TRACKED_PREFIX = "custom_components/quiet_solar/"

# Subprocess hard timeout. Long enough that a healthy ``git diff
# --cached`` returns; short enough that an index-lock-contention hang
# can't stall the implement/review agents indefinitely. Review-fix
# #01 S4.
GIT_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# Frontmatter parsing


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    """Return the YAML frontmatter of ``path`` as a dict.

    Raises ``ValueError`` (or ``yaml.YAMLError``) if the frontmatter
    can't be parsed — main()'s scan loop turns those into the
    ``malformed_frontmatter`` bucket.

    Robust to:
    - UTF-8 BOM (``utf-8-sig`` — review-fix #01 S6).
    - CRLF and bare ``\\r`` (legacy macOS / corrupted) line endings
      (review-fix #01 S5).
    - Closing ``---`` at end of file with no trailing newline
      (review-fix #01 S5).
    """
    # ``utf-8-sig`` transparently strips a UTF-8 BOM if present.
    text = path.read_text(encoding="utf-8-sig")
    # Normalize all common line-ending variants. Replace CRLF before
    # bare CR so we don't double-replace the LF.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing opening '---' delimiter")
    rest = text[len("---\n") :]

    end = rest.find("\n---\n")
    if end == -1:
        # Fall back to ``\n---`` at end-of-file (no trailing newline).
        stripped = rest.rstrip()
        if stripped.endswith("\n---"):
            end = len(stripped) - len("\n---")
        else:
            raise ValueError(f"{path}: missing closing '---' delimiter")

    data = yaml.safe_load(rest[:end])
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    return data


def _stringify_last_verified(value: Any) -> str:
    """Coerce a frontmatter ``last_verified`` value to an ISO string.

    PyYAML auto-parses ``YYYY-MM-DD`` as ``datetime.date``; we want
    the JSON output to be the original string form.
    """
    if isinstance(value, _dt.date):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Path normalization


def _normalize_path(raw: str, repo_root: Path) -> str:
    """Normalize a path string for comparison with ``source_to_docs`` keys.

    Review-fix #01 M2: covers a list of malformations that agents and
    automation realistically produce — Windows backslashes, leading
    ``./``, absolute paths from ``$PWD`` expansion, trailing slashes,
    stray whitespace.

    - Backslashes → forward slashes.
    - Strip leading ``./`` (any number of times).
    - If absolute, attempt to relativize against ``repo_root``; if the
      path lives outside the repo, leave it absolute (the caller will
      then see no match in ``source_to_docs``, which is correct —
      drift detection only fires for in-tree paths).
    - Strip trailing ``/``.
    """
    s = raw.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]

    # Detect absolute paths. POSIX: leading ``/``. Windows-style drive
    # letters (``C:/...``) are not used in this repo but we tolerate
    # them for forward compatibility.
    is_absolute = s.startswith("/") or (len(s) >= 2 and s[1] == ":")
    if is_absolute:
        try:
            s = Path(s).resolve().relative_to(repo_root).as_posix()
        except (ValueError, OSError):
            # Outside ``repo_root`` (or unresolvable): leave the
            # absolute string alone. No source under
            # ``custom_components/quiet_solar/`` could ever match.
            pass

    while s.endswith("/") and len(s) > 1:
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# Doc-tree scan


def _scan_docs(
    repo_root: Path,
) -> tuple[
    dict[str, list[Path]],  # source-path -> [doc paths]
    list[Path],  # malformed docs
    list[tuple[Path, str]],  # missing covers: (doc, source_path)
    dict[Path, str],  # doc -> last_verified
]:
    """Walk ``docs/agents/`` and return the index + malformed/missing buckets."""
    docs_root = repo_root / "docs" / "agents"
    source_to_docs: dict[str, list[Path]] = {}
    malformed: list[Path] = []
    missing: list[tuple[Path, str]] = []
    last_verified: dict[Path, str] = {}

    if not docs_root.exists():
        return source_to_docs, malformed, missing, last_verified

    for doc_path in sorted(docs_root.rglob("*.md")):
        # Skip symlinks. ``rglob`` follows them by default; a self-
        # referential symlink would hang the scan, and a symlink into
        # another tree would duplicate reports (review-fix #01 N9).
        if doc_path.is_symlink():
            continue
        try:
            fm = _parse_frontmatter(doc_path)
        except (ValueError, yaml.YAMLError) as exc:
            print(f"warning: {doc_path}: {exc}", file=sys.stderr)
            malformed.append(doc_path)
            continue

        last_verified[doc_path] = _stringify_last_verified(fm.get("last_verified"))

        if "covers" not in fm:
            # Index / glossary / persona docs that don't anchor to a
            # specific source file. They live happily inside the
            # ``docs/agents/`` tree without being drift-tracked.
            continue

        covers = fm["covers"]
        # Review-fix #01 N10: surface a clearer message when the
        # ``covers:`` key parses to ``None`` (the author wrote
        # ``covers:`` with no value).
        if covers is None:
            print(
                f"warning: {doc_path}: 'covers:' has no value — either "
                "omit the key or write 'covers: []'.",
                file=sys.stderr,
            )
            malformed.append(doc_path)
            continue
        if not isinstance(covers, list):
            print(
                f"warning: {doc_path}: 'covers' must be a list, got {type(covers).__name__}",
                file=sys.stderr,
            )
            malformed.append(doc_path)
            continue

        # De-duplicate while preserving first-occurrence order
        # (review-fix #01 S7). A typo / paste error in a ``covers:``
        # list shouldn't double-report drift in the JSON output.
        deduped: list[Any] = list(dict.fromkeys(covers))
        for entry in deduped:
            if not isinstance(entry, str):
                print(
                    f"warning: {doc_path}: 'covers' entry must be a string, got {type(entry).__name__}",
                    file=sys.stderr,
                )
                malformed.append(doc_path)
                break

            # Review-fix #01 N7: strip stray whitespace so a YAML
            # quoted-string indent doesn't silently demote a
            # ``covers:`` entry to "off-tree".
            entry = entry.strip()

            if not entry.startswith(TRACKED_PREFIX):
                # Off-tree entry — warn but don't flag as error or
                # add to the drift index.
                print(
                    f"warning: {doc_path}: covers entry {entry!r} is outside "
                    f"'{TRACKED_PREFIX}'; ignored (not drift-tracked)",
                    file=sys.stderr,
                )
                continue

            full = repo_root / entry
            if not full.exists():
                missing.append((doc_path, entry))
                continue

            # Review-fix #01 N8: directory entries don't cascade.
            # Reject them explicitly so the author writes a file
            # path (or a glob, when we support that).
            if full.is_dir():
                print(
                    f"warning: {doc_path}: covers entry {entry!r} is a directory; "
                    "use a file path (directory entries don't cascade).",
                    file=sys.stderr,
                )
                malformed.append(doc_path)
                break

            source_to_docs.setdefault(entry, []).append(doc_path)

    return source_to_docs, malformed, missing, last_verified


# ---------------------------------------------------------------------------
# Modified-path source (staged diff or --paths override)


def _git_staged_paths(repo_root: Path) -> list[str]:
    """Return the staged file paths via ``git diff --cached``.

    On non-zero exit (e.g., not a git repo), missing ``git`` binary
    (review-fix #01 M1), or hung index lock (review-fix #01 S4), log
    to stderr and return an empty list. The caller treats "no
    modifications" as a clean gate, which is the right behaviour for
    a non-git environment.

    ``-c core.quotepath=false`` (review-fix #01 S8) disables git's
    default C-style escaping of non-ASCII paths so the returned
    strings match ``source_to_docs`` keys directly.
    """
    cmd = [
        "git",
        "-c",
        "core.quotepath=false",
        "diff",
        "--cached",
        "--diff-filter=ACMRD",
        "--name-only",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
            timeout=GIT_TIMEOUT_S,
        )
    except FileNotFoundError:
        print(
            "warning: git binary not found on PATH; treating as no staged changes",
            file=sys.stderr,
        )
        return []
    except subprocess.TimeoutExpired:
        print(
            f"warning: git diff timeout exceeded ({GIT_TIMEOUT_S}s); "
            "treating as no staged changes",
            file=sys.stderr,
        )
        return []

    if result.returncode != 0:
        print(
            f"warning: git diff failed (exit {result.returncode}): "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Cross-harness agent sync

HARNESS_DIRS = (".claude", ".cursor", ".opencode")



def _check_harness_sync(
    modified: set[str],
    repo_root: Path,
) -> list[dict[str, str | list[str]]]:
    """Check co-modification of agent files across harness directories.

    Agent bodies legitimately differ across harnesses — each has its
    own session-spawn/handoff logic (Claude uses ``claude --agent``,
    OpenCode uses ``spawn_session.py``, Cursor uses the agent picker).
    A byte-identical body check would reject valid harness-specific
    adaptations.

    Instead, this is a **co-modification check**: when an agent file
    in one harness is modified, the counterpart files in the other
    harness directories should also appear in the modified set. This
    catches the common case ("edited .claude/agents/foo.md but forgot
    .cursor/ and .opencode/") without rejecting legitimate differences.

    Only flags drift when the counterpart file actually exists on disk.
    Harness-specific agents (present in only one harness) are exempt.

    Returns a list of drift entries:
    ``{"agent": basename, "out_of_sync": [harness_dirs]}``.
    """
    # Collect which agent basenames were modified and from which harnesses
    agent_harnesses: dict[str, set[str]] = {}
    for path_str in modified:
        for harness in HARNESS_DIRS:
            prefix = f"{harness}/agents/"
            if path_str.startswith(prefix) and path_str.endswith(".md"):
                basename = path_str[len(prefix):]
                agent_harnesses.setdefault(basename, set()).add(harness)

    if not agent_harnesses:
        return []

    drift: list[dict[str, str | list[str]]] = []

    for basename, modified_harnesses in agent_harnesses.items():
        out_of_sync: list[str] = []
        for harness in HARNESS_DIRS:
            if harness in modified_harnesses:
                continue  # This harness was co-modified — OK
            # Only flag drift when the counterpart file actually exists.
            # Harness-specific agents (no counterpart) are exempt.
            agent_path = repo_root / harness / "agents" / basename
            if agent_path.is_file():
                out_of_sync.append(harness)

        if out_of_sync:
            drift.append({
                "agent": basename,
                "out_of_sync": sorted(out_of_sync),
            })

    return drift


# ---------------------------------------------------------------------------
# Drift detection


def _detect_drift(
    source_to_docs: dict[str, list[Path]],
    modified: set[str],
    repo_root: Path,
) -> dict[Path, list[str]]:
    """Return ``doc -> [stale sources]`` for every uncovered modification."""
    # Resolve modified doc paths against ``repo_root`` so we can
    # compare with the absolute paths returned by ``_scan_docs``.
    modified_docs: set[Path] = set()
    for entry in modified:
        if not entry.endswith(".md"):
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        modified_docs.add(candidate.resolve())

    # Pre-resolve every doc once so the inner loop is a hash lookup
    # rather than an O(N) resolve per source (review-fix #01 S9).
    # On Windows / network filesystems, ``Path.resolve()`` is non-
    # trivial; hoisting the call here turns O(N×M) into O(N+M).
    all_docs = {doc for docs in source_to_docs.values() for doc in docs}
    resolved: dict[Path, Path] = {doc: doc.resolve() for doc in all_docs}

    stale: dict[Path, list[str]] = {}
    for source, docs in source_to_docs.items():
        if source not in modified:
            continue
        for doc in docs:
            if resolved[doc] in modified_docs:
                continue
            stale.setdefault(doc, []).append(source)
    return stale


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_doc_drift.py",
        description="Verify docs/agents/ docs are in sync with their covers: source files.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Repository root (default: this script's grand-parent directory).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help=(
            "Override the staged-diff source. ``--paths`` with no "
            "arguments explicitly means 'no modifications' (clean "
            "gate). Omit the flag entirely to read modifications from "
            "``git diff --cached``. Path forms are normalized: "
            "backslashes → forward slashes, leading ``./`` stripped, "
            "absolute paths inside ``--repo-root`` relativized."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of text.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()

    # Review-fix #01 N11: a typo in ``--repo-root`` previously
    # produced a silent green gate (the scan returns an empty index).
    # Fail loudly instead so misconfiguration is visible.
    if not repo_root.is_dir():
        print(
            f"error: --repo-root does not exist or is not a directory: {repo_root}",
            file=sys.stderr,
        )
        return 2

    source_to_docs, malformed, missing, last_verified = _scan_docs(repo_root)

    if args.paths is not None:
        modified = {_normalize_path(p, repo_root) for p in args.paths}
    else:
        modified = {_normalize_path(p, repo_root) for p in _git_staged_paths(repo_root)}

    stale = _detect_drift(source_to_docs, modified, repo_root)
    harness_drift = _check_harness_sync(modified, repo_root)

    def _rel(p: Path) -> str:
        # ``_scan_docs`` walks ``repo_root/docs/agents/``, so every
        # path in the report is guaranteed to be a descendant of
        # ``repo_root``. No fallback branch needed.
        return str(p.relative_to(repo_root))

    report: dict[str, Any] = {
        "stale_docs": [
            {
                "doc": _rel(doc),
                # Dedup sources defensively for the JSON output —
                # the dedup at parse-time handles the common case
                # but two ``covers:`` entries could still differ
                # by whitespace and resolve to the same canonical
                # form (review-fix #01 S7).
                "stale_sources": sorted(set(sources)),
                "last_verified": last_verified.get(doc, ""),
            }
            for doc, sources in sorted(stale.items(), key=lambda kv: _rel(kv[0]))
        ],
        "missing_covers": sorted(f"{_rel(doc)}::{src}" for doc, src in missing),
        "malformed_frontmatter": sorted(_rel(p) for p in malformed),
        "harness_drift": [
            {
                "agent": entry["agent"],
                "out_of_sync": entry["out_of_sync"],
            }
            for entry in sorted(harness_drift, key=lambda e: e["agent"])
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for path in report["malformed_frontmatter"]:
            print(f"malformed: {path}")
        for entry in report["missing_covers"]:
            print(f"missing covers path: {entry}")
        for item in report["stale_docs"]:
            print(
                f"stale: {item['doc']} "
                f"(sources: {', '.join(item['stale_sources'])}; "
                f"last_verified={item['last_verified']})"
            )
        for item in report["harness_drift"]:
            print(
                f"harness drift: {item['agent']} "
                f"(out of sync: {', '.join(item['out_of_sync'])})"
            )

    if report["malformed_frontmatter"] or report["missing_covers"]:
        return 2
    if report["stale_docs"] or report["harness_drift"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
