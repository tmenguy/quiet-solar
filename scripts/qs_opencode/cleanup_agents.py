#!/usr/bin/env python3
"""Remove per-task OpenCode agent files for a given issue.

Called by ``qs-finish-task-QS-<N>`` just before the worktree is deleted.
Removes every ``qs-*-QS-<issue>.md`` under ``<work_dir>/.opencode/agents/``.

Usage::

    python scripts/qs_opencode/cleanup_agents.py \\
        --work-dir /path/to/worktree \\
        --issue 42

Never deletes files outside the given worktree's ``.opencode/agents/`` dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import output_json  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove per-task OpenCode agent files for an issue",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be removed without deleting",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    agent_dir = work_dir / ".opencode" / "agents"
    if not agent_dir.is_dir():
        output_json({
            "work_dir": str(work_dir),
            "issue": args.issue,
            "removed": [],
            "summary": f"No .opencode/agents/ directory under {work_dir}",
        })
        return

    pattern = f"qs-*-QS-{args.issue}.md"
    matches = sorted(agent_dir.glob(pattern))

    removed: list[str] = []
    for path in matches:
        rel = path.relative_to(work_dir)
        if args.dry_run:
            removed.append(str(rel))
            continue
        try:
            path.unlink()
            removed.append(str(rel))
        except OSError as exc:
            # Don't let a single failure abort the whole cleanup; report it.
            removed.append(f"FAILED {rel}: {exc}")

    output_json({
        "work_dir": str(work_dir),
        "issue": args.issue,
        "pattern": pattern,
        "removed": removed,
        "dry_run": args.dry_run,
        "summary": (
            f"{'Would remove' if args.dry_run else 'Removed'} "
            f"{len(removed)} agent file(s) for QS-{args.issue}"
        ),
    })


if __name__ == "__main__":  # pragma: no cover
    main()
