"""Reconcile sharded CI test counts against the collected total (QS-292).

Layer-2 guard of the sharded ``pr-quality.yml`` test matrix: asserts that
exactly ``--splits`` JUnit shard reports arrived and that the number of
executed test cases across them equals the total recorded by shard 1's
collect-only step. Catches a shard silently not running or a
``--splits``/matrix-length mismatch. (The one-off partition-identity
proof — union == collected, pairwise disjoint — is a local planning
step, not this script's job.)

Stdlib only by design: the aggregation job installs nothing but
``coverage``.

Usage::

    python3 scripts/qs/ci_reconcile_shards.py <dir> --splits N

Success prints ``executed=<n> collected=<n>`` and exits 0. Each failure
mode prints one ``::error::`` line (GitHub Actions annotation) and
exits 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree


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
        print(f"::error::expected {splits} JUnit shard files in {shard_dir}, found {len(junit_files)}")
        return 1

    total_file = shard_dir / "total-collected.txt"
    if not total_file.is_file():
        print(f"::error::missing {total_file} (shard 1's collect-only step)")
        return 1
    try:
        collected = int(total_file.read_text(encoding="utf-8").strip())
    except ValueError:
        print(f"::error::unparseable collected count in {total_file}")
        return 1

    executed = 0
    for junit_file in junit_files:
        try:
            root = ElementTree.parse(junit_file).getroot()
        except ElementTree.ParseError:
            print(f"::error::unparseable JUnit XML: {junit_file}")
            return 1
        executed += len(root.findall(".//testcase"))

    if executed != collected:
        print(
            f"::error::executed test count {executed} != collected "
            f"{collected} — a shard silently dropped or double-ran tests"
        )
        return 1

    print(f"executed={executed} collected={collected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
