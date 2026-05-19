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
    2 — malformed frontmatter or missing ``covers:`` path

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


# ---------------------------------------------------------------------------
# Frontmatter parsing


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    """Return the YAML frontmatter of ``path`` as a dict.

    Raises ``ValueError`` (or ``yaml.YAMLError``) if the frontmatter
    can't be parsed — main()'s scan loop turns those into the
    ``malformed_frontmatter`` bucket.
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing opening '---' delimiter")
    rest = text[len("---\n") :]
    end = rest.find("\n---\n")
    if end == -1:
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
        if not isinstance(covers, list):
            print(
                f"warning: {doc_path}: 'covers' must be a list, got {type(covers).__name__}",
                file=sys.stderr,
            )
            malformed.append(doc_path)
            continue

        doc_is_malformed = False
        for entry in covers:
            if not isinstance(entry, str):
                print(
                    f"warning: {doc_path}: 'covers' entry must be a string, got {type(entry).__name__}",
                    file=sys.stderr,
                )
                malformed.append(doc_path)
                doc_is_malformed = True
                break

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

            source_to_docs.setdefault(entry, []).append(doc_path)

        # If a covers item was malformed (not a string), break already
        # ran and skipped further entries; nothing else to do here.
        del doc_is_malformed

    return source_to_docs, malformed, missing, last_verified


# ---------------------------------------------------------------------------
# Modified-path source (staged diff or --paths override)


def _git_staged_paths(repo_root: Path) -> list[str]:
    """Return the staged file paths via ``git diff --cached``.

    On non-zero exit (e.g., not a git repo), log to stderr and return
    an empty list. The caller treats "no modifications" as a clean
    gate, which is the right behaviour for a non-git environment.
    """
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--diff-filter=ACMRD",
            "--name-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    if result.returncode != 0:
        print(
            f"warning: git diff failed (exit {result.returncode}): "
            f"{result.stderr.strip()}",
            file=sys.stderr,
        )
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


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

    stale: dict[Path, list[str]] = {}
    for source, docs in source_to_docs.items():
        if source not in modified:
            continue
        for doc in docs:
            if doc.resolve() in modified_docs:
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
        help="Override the staged-diff source. Any number of paths "
        "(0 = 'no modifications'). When omitted, the script runs "
        "git diff --cached.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of text.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()

    source_to_docs, malformed, missing, last_verified = _scan_docs(repo_root)

    if args.paths is not None:
        modified = set(args.paths)
    else:
        modified = set(_git_staged_paths(repo_root))

    stale = _detect_drift(source_to_docs, modified, repo_root)

    def _rel(p: Path) -> str:
        # ``_scan_docs`` walks ``repo_root/docs/agents/``, so every
        # path in the report is guaranteed to be a descendant of
        # ``repo_root``. No fallback branch needed.
        return str(p.relative_to(repo_root))

    report: dict[str, Any] = {
        "stale_docs": [
            {
                "doc": _rel(doc),
                "stale_sources": sorted(sources),
                "last_verified": last_verified.get(doc, ""),
            }
            for doc, sources in sorted(stale.items(), key=lambda kv: _rel(kv[0]))
        ],
        "missing_covers": sorted(f"{_rel(doc)}::{src}" for doc, src in missing),
        "malformed_frontmatter": sorted(_rel(p) for p in malformed),
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

    if report["malformed_frontmatter"] or report["missing_covers"]:
        return 2
    if report["stale_docs"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
