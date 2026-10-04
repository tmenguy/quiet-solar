"""The Control Plane (QS-399): runtime state, run queues, commands, tools and hooks.

Python, no LLM, one for all runs. Every session calls
``<MAIN>/venv/bin/python <MAIN>/scripts/qs/cp.py <command> …`` — see
``docs/workflow/control-plane.md``.
"""

PACKAGE = "control_plane"
