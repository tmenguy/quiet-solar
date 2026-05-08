"""Migrate implementation artifacts from _bmad-output/ to _qsprocess_opencode/stories/.

Copies markdown files verbatim (no content scrubbing) with naming convention:
- Files containing Github-#N → QS-N.story.md
- Files without Github-#N → QS-legacy-{slug}.story.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _slugify(name: str, *, max_len: int = 80) -> str:
    """Convert a filename to a lowercase hyphenated slug."""
    # Strip .md extension
    slug = re.sub(r"\.md$", "", name, flags=re.IGNORECASE)
    # Lowercase
    slug = slug.lower()
    # Replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    # Truncate then strip again (truncation may expose a trailing hyphen)
    return slug[:max_len].rstrip("-")


def _compute_target_name(source_name: str) -> str:
    """Map a source filename to the target QS naming convention."""
    match = re.search(r"Github-#(\d+)", source_name)
    if match:
        all_matches = re.findall(r"Github-#(\d+)", source_name)
        if len(all_matches) > 1:
            print(f"WARNING: multiple Github-# refs in {source_name}, using #{all_matches[0]}")
        issue_num = match.group(1)
        return f"QS-{issue_num}.story.md"
    slug = _slugify(source_name)
    if not slug:
        slug = "unnamed"
    return f"QS-legacy-{slug}.story.md"


def migrate(
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool = False,
) -> int:
    """Run the migration. Return 0 on success."""
    sources = sorted(src_dir.glob("*.md"))
    if not sources:
        print(f"No .md files found in {src_dir}")
        return 0

    # Build mapping, detect collisions
    target_map: dict[str, list[Path]] = {}
    for src in sources:
        target_name = _compute_target_name(src.name)
        target_map.setdefault(target_name, []).append(src)

    migrated = 0
    skipped = 0
    collisions = 0

    for target_name, src_files in sorted(target_map.items()):
        for idx, src_file in enumerate(src_files):
            if idx == 0:
                actual_target = target_name
            else:
                # Collision: add -duplicate-N suffix
                base = target_name.replace(".story.md", "")
                actual_target = f"{base}-duplicate-{idx}.story.md"
                print(f"WARNING: collision for {target_name} — {src_file.name} → {actual_target}")

            dst_path = dst_dir / actual_target
            if dst_path.exists():
                print(f"Skipped: {src_file.name} → {actual_target} (already exists)")
                skipped += 1
                continue

            # Count collision only for non-skipped duplicates
            if idx > 0:
                collisions += 1

            if dry_run:
                print(f"DRY RUN: {src_file.name} → {actual_target}")
                migrated += 1
            else:
                dst_path.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"Migrated: {src_file.name} → {actual_target}")
                migrated += 1

    total = len(sources)
    if total != migrated + skipped:
        print(
            f"Count mismatch: {total} sources != {migrated} migrated + {skipped} skipped",
            file=sys.stderr,
        )
        return 1
    print(f"\nSummary: {total} source files, {migrated} migrated, {skipped} skipped, {collisions} collisions")
    return 0


def main() -> None:
    """Entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Migrate implementation artifacts to QS naming convention")
    parser.add_argument("src_dir", type=Path, help="Source directory")
    parser.add_argument("dst_dir", type=Path, help="Destination directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without writing files",
    )
    args = parser.parse_args()

    if not args.src_dir.is_dir():
        print(f"Error: source directory not found: {args.src_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.dst_dir.is_dir():
        print(f"Error: destination directory not found: {args.dst_dir}", file=sys.stderr)
        sys.exit(1)

    sys.exit(migrate(args.src_dir, args.dst_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
