#!/usr/bin/env python3
"""Create a release: determine tag, bump version, commit, tag, push.

Usage:
    python scripts/qs/release.py [--dry-run]

Output: JSON with tag, version, and status.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from utils import get_current_branch, get_main_worktree, output_json, run_git


def determine_next_tag() -> str:
    """Find the next release tag for today using local git tags."""
    today = datetime.now().strftime("%Y.%m.%d")

    result = run_git(["tag", "--list", f"v{today}.*"], check=False)

    existing = []
    for line in result.stdout.strip().split("\n"):
        tag = line.strip()
        if tag.startswith(f"v{today}."):
            idx_str = tag.replace(f"v{today}.", "")
            try:
                existing.append(int(idx_str))
            except ValueError:
                pass

    if not existing:
        return f"v{today}.0"
    return f"v{today}.{max(existing) + 1}"


def bump_manifest(version: str) -> Path:
    """Update manifest.json version field."""
    main_dir = get_main_worktree()
    manifest = main_dir / "custom_components" / "quiet_solar" / "manifest.json"
    content = manifest.read_text()
    new_content = re.sub(r'"version":\s*"[^"]*"', f'"version": "{version}"', content)
    manifest.write_text(new_content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create release")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    # Must be on main
    branch = get_current_branch()
    if branch != "main":
        output_json({"error": f"Must be on main branch, currently on {branch}"})
        sys.exit(1)

    # Pull latest
    run_git(["pull"])

    tag = determine_next_tag()
    version = tag.lstrip("v")

    if args.dry_run:
        output_json({"dry_run": True, "tag": tag, "version": version})
        return

    # Bump manifest
    manifest_path = bump_manifest(version)

    # Commit and push (skip commit if manifest was already at this version)
    run_git(["add", str(manifest_path)])
    diff_check = run_git(["diff", "--cached", "--quiet"], check=False)
    if diff_check.returncode != 0:
        run_git(["commit", "-m", f"bump version to {version}"])
        run_git(["push", "origin", "main"])

    # Tag and push tag
    run_git(["tag", tag])
    run_git(["push", "origin", tag])

    output_json({
        "tag": tag,
        "version": version,
        "status": "released",
        "detail": "GitHub Actions will run the release pipeline. Check the Actions tab for progress.",
    })


if __name__ == "__main__":
    main()
