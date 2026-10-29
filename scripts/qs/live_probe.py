#!/usr/bin/env python3
"""Live-check probe for the qs-node pipeline (throwaway, #437 / #442).

A deliberately tiny script: it exists so the node's first live run has a
real tracked change to plan, review, implement, gate and commit. It prints
``live-probe ok`` and exits 0; ``argv`` is accepted for the ``main(argv)``
signature and ignored (a probe takes no options).

Usage::

    python scripts/qs/live_probe.py
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    """Run the probe. ``argv`` is the argument list without the program name; it is ignored."""
    print("live-probe ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
