"""Shared timing helpers for the QS-304 command-lifecycle tests.

Everything here is **derived** from the production constants, so a change to the
ladder cannot silently invalidate a test's hardcoded arithmetic (nice-to-have 15/16
of review round #03).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from custom_components.quiet_solar.home_model.load import (
    COMMAND_RELAUNCH_BASE_DELAY_S,
    NUM_MAX_COMMAND_RELAUNCH,
)

# The observed quiet-solar load-management cycle period.
CYCLE_S = 7

# The announce-branch ERROR needle — the one log line only `_escalate_or_recover`'s
# announce emits, which is what distinguishes §2's path from the QS-307 give-up
# (which also latches and also pushes nothing). Shared here (review fix QS-319#01/5)
# so the pure-domain and HA test modules cannot drift apart on a reworded line.
LOST_CONTROL_LOG = "Lost control of load"

# Cumulative wall time of the growing part of the ladder: 50 + 100 + ... + 300.
LADDER_TOTAL_S = sum(COMMAND_RELAUNCH_BASE_DELAY_S * (n + 1) for n in range(NUM_MAX_COMMAND_RELAUNCH))

# Each rung fires on the first load-management cycle AFTER its deadline, so the
# observed wall time runs up to one cycle per rung longer — 17 m 52 s in the incident
# log against a nominal 17 m 30 s.
LADDER_WALL_S = LADDER_TOTAL_S + NUM_MAX_COMMAND_RELAUNCH * CYCLE_S

# The saturated relaunch cadence, i.e. the interval between service calls once the
# backoff has stopped growing.
SATURATED_INTERVAL_S = COMMAND_RELAUNCH_BASE_DELAY_S * NUM_MAX_COMMAND_RELAUNCH


def expected_relaunches(duration_s: float, tolerance: int = 2) -> tuple[int, int]:
    """Return inclusive (low, high) bounds for relaunches over `duration_s`.

    Derived from the saturated cadence rather than hardcoded, so the bounds move with
    the constants. `tolerance` absorbs cycle quantisation at both ends.
    """
    nominal = int(duration_s // SATURATED_INTERVAL_S)
    return max(nominal - tolerance, 0), nominal + tolerance


async def drive(load, start: datetime, duration_s: float, step_s: float = CYCLE_S) -> datetime:
    """Run the load-management cycle over `duration_s`, returning the next cycle time."""
    time = start
    end = start + timedelta(seconds=duration_s)
    while time <= end:
        await load.check_and_relaunch_command(time)
        time = time + timedelta(seconds=step_s)
    return time


def count_log(caplog, needle: str) -> int:
    """Count log records whose formatted message contains `needle`."""
    return len([r for r in caplog.records if needle in r.getMessage()])
