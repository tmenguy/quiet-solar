#!/usr/bin/env python3
"""Live-check probe for the qs-node pipeline (throwaway, #437 / #442).

A deliberately tiny script: it exists so the node's first live run has a
real tracked change to plan, review, implement, gate and commit. With no
argument it prints ``live-probe ok`` and exits 0; with ``--version`` it
prints ``live-probe 1`` and exits 0; any other argument list exits 2.

Usage::

    python scripts/qs/live_probe.py [--version]
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    """Run the probe. ``argv`` is the argument list without the program name."""
    if not argv:
        print("live-probe ok")
        return 0
    if argv == ["--version"]:
        print("live-probe 1")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
