"""Reconcile sharded CI test counts against the collected total (QS-292).

Layer-2 guard of the sharded ``pr-quality.yml`` test matrix. Given the
downloaded shard artifacts it asserts that:

1. exactly ``--splits`` JUnit shard reports arrived, and their shard
   indices are precisely ``1..splits`` (a right-count/wrong-index set,
   e.g. a missing shard 3 plus a stray shard 99, is caught here);
2. no ``(classname, name)`` test case appears in more than one shard —
   identity, not arithmetic, so an *offsetting* duplicate-plus-drop
   (test A run twice, test B never) cannot balance the books; and
3. the number of executed test cases equals the total recorded by
   shard 1's collect-only step.

Together these catch a shard silently not running, a
``--splits``/matrix-length mismatch, and a split that is not a true
partition. (The one-off partition-identity proof over *collected node
IDs* is a local planning step, not this script's job.)

Stdlib only by design: the aggregation job installs nothing but
``coverage``.

Usage::

    python3 scripts/qs/ci_reconcile_shards.py <dir> --splits N

Success prints ``executed=<n> collected=<n>`` and exits 0. Every
failure mode prints exactly one ``::error::`` line (a GitHub Actions
annotation) and exits 1.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree

SHARD_FILE_RE = re.compile(r"junit-shard-(\d+)\.xml")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shard_dir",
        type=Path,
        help="directory holding junit-shard-*.xml and total-collected.txt",
    )
    parser.add_argument(
        "--splits",
        type=int,
        required=True,
        help="expected number of JUnit shard files",
    )
    args = parser.parse_args(argv)
    shard_dir: Path = args.shard_dir
    splits: int = args.splits

    junit_files = sorted(shard_dir.glob("junit-shard-*.xml"))
    if len(junit_files) != splits:
        # Fails safe, but is NOT clearable by re-running this job alone
        # once the shard artifacts have expired, so name the remedy.
        print(
            f"::error::expected {splits} JUnit shard files in {shard_dir}, "
            f"found {len(junit_files)} — shard artifacts missing or expired; "
            f"re-run ALL jobs, not just this one"
        )
        return 1

    indices: set[int] = set()
    for junit_file in junit_files:
        match = SHARD_FILE_RE.fullmatch(junit_file.name)
        if match is None:
            print(f"::error::unexpected shard artifact name: {junit_file.name}")
            return 1
        indices.add(int(match.group(1)))

    expected_indices = set(range(1, splits + 1))
    if indices != expected_indices:
        print(
            f"::error::shard indices {sorted(indices)} != expected "
            f"{sorted(expected_indices)} — a shard is missing or misnamed"
        )
        return 1

    total_file = shard_dir / "total-collected.txt"
    if not total_file.is_file():
        print(f"::error::missing {total_file} (shard 1's collect-only step)")
        return 1
    try:
        collected = int(total_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as err:
        print(f"::error::unreadable collected count in {total_file}: {err}")
        return 1

    executed = 0
    seen: set[tuple[str, str]] = set()
    for junit_file in junit_files:
        try:
            root = ElementTree.parse(junit_file).getroot()
        except ElementTree.ParseError:
            print(f"::error::unparseable JUnit XML: {junit_file}")
            return 1
        for case in root.findall(".//testcase"):
            executed += 1
            seen.add((case.get("classname", ""), case.get("name", "")))

    if len(seen) != executed:
        print(
            f"::error::{executed - len(seen)} duplicate test case(s) ran in more "
            f"than one shard — the split is not a partition"
        )
        return 1

    if executed != collected:
        print(f"::error::executed test count {executed} != collected {collected} — a shard silently dropped tests")
        return 1

    print(f"executed={executed} collected={collected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
