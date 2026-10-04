"""Control Plane entry shim (QS-399) — no logic.

Always run as ``<MAIN>/venv/bin/python <MAIN>/scripts/qs/cp.py <command> …``;
see ``docs/workflow/control-plane.md``. Running the script puts
``scripts/qs`` on ``sys.path``, so the package imports as ``control_plane``.
"""

from control_plane.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
