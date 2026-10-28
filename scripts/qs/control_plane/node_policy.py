"""The node's ceilings and constants (#425 D7): a constants-only leaf module.

Read by the node's template (through ``render_agents._make_env``'s env
globals ``cr_wait_s``, ``rotate_rounds``, ``ci_wait_max_s``), by the
CodeRabbit tools (``coderabbit.py``) and by the catch-up (``catchup.py``,
#440: ``MAX_CATCHUPS`` is read there only, never rendered into the template). Values confirmed by the maintainer at
#425's FINALIZE (2026-10-27); a retune is a one-line edit here.
"""

from __future__ import annotations

CR_MAX_REVIEWS = 5  # CodeRabbit review requests per task (one per build round; catch-up heads excluded)
CR_WAIT_S = 900  # wait for a requested review before the round goes on without it (observed 5-10 min)
CR_MAX_FINDINGS = 30  # findings ingested per review, most severe first
CR_SETTLE_S = 120  # how old a "Review completed" status must be before "no review object" means "nothing to say"
ROTATE_ROUNDS = 4  # rounds (both loops) between rotation requests
CI_WAIT_MAX_S = 2700  # CI still pending after a push before the node blocks
MAX_CATCHUPS = 3  # catch-ups with main since the last done (#440: a 4th blocks before any merge)
