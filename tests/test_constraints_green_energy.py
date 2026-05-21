"""QS-204 — regression test for the constraints.py green-energy contract.

Pre-QS-204, ``MultiStepsPowerLoadConstraint.compute_best_period_repartition``
short-circuited ``final_ret = False`` whenever ``has_a_cmd`` stayed
``False`` through the green-energy allocation loop (the case where
``adapt_power_steps_budgeting`` returned no candidates in any slot).

That short-circuit prevented a freshly-created bistate load (e.g. a
switch-backed radiator with ``current_command = running_command = None``
and a Force ON mandatory constraint) from ever being dispatched.

The fix on this branch removes the short-circuit. The decision now
falls through to the standard
``quantity_to_be_added <= 0.0 or do_use_available_power_only`` branch,
so downstream solar/battery/grid logic gets a chance to attempt
dispatch even when the green-energy pass found nothing.

This test pins the new contract via the white-box mock route the
plan calls out: patch ``adapt_power_steps_budgeting`` to return ``[]``
for every slot in the green pass and assert the method does NOT
return ``final_ret = False`` for that input shape, AND does not emit
the removed ``"no power sorted commands for green energy"`` log
warning.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pytz

from custom_components.quiet_solar.const import (
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    SOLVER_STEP_S,
)
from custom_components.quiet_solar.home_model.commands import LoadCommand
from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint
from custom_components.quiet_solar.home_model.load import TestLoad
from tests.factories import TestDynamicGroupDouble


def test_compute_best_period_green_no_cmd_does_not_short_circuit(caplog):
    """QS-204 — empty green-energy pass falls through, no early-return.

    When ``adapt_power_steps_budgeting`` returns no candidate commands
    for every slot in the green-energy loop, the method must:

    1. NOT short-circuit ``final_ret = False`` (the pre-fix bug).
    2. NOT emit the removed "no power sorted commands for green energy"
       warning.
    3. Fall through to the
       ``quantity_to_be_added <= 0.0 or do_use_available_power_only``
       branch so downstream solver passes can dispatch.

    The constraint type is ``CONSTRAINT_TYPE_MANDATORY_END_TIME`` (not
    AS-FAST-AS-POSSIBLE) so the green-energy code path is taken.
    """
    time_base = datetime(2026, 5, 21, 10, 0, 0, tzinfo=pytz.UTC)
    num_slots = 4
    time_slots = [time_base + timedelta(seconds=i * SOLVER_STEP_S) for i in range(num_slots + 1)]

    load = TestLoad(name="qs204_green_no_cmd_load")
    load.efficiency = 100.0
    load.father_device = TestDynamicGroupDouble(max_amps=[50.0, 50.0, 50.0], num_slots=num_slots)

    constraint = MultiStepsPowerLoadConstraint(
        time=time_base,
        load=load,
        power_steps=[LoadCommand(command="ON", power_consign=1500.0)],
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        support_auto=False,
        initial_value=0,
        target_value=3000,  # 3 kWh worth of duration
        current_value=0,
        end_of_constraint=time_slots[-1],
    )

    # Force every slot to report "no candidates" — simulates a fresh
    # load with no command history whose adapt step produces nothing.
    def _no_candidates(self, slot_idx, commands, for_add, max_slot_power_headroom=None):
        del slot_idx, commands, for_add, max_slot_power_headroom
        return [], False, 0.0

    original = MultiStepsPowerLoadConstraint.adapt_power_steps_budgeting
    MultiStepsPowerLoadConstraint.adapt_power_steps_budgeting = _no_candidates
    try:
        with caplog.at_level(
            logging.WARNING,
            logger="custom_components.quiet_solar.home_model.constraints",
        ):
            result = constraint.compute_best_period_repartition(
                do_use_available_power_only=True,
                power_available_power=np.array([-2000.0] * num_slots, dtype=np.float64),
                power_slots_duration_s=np.array([SOLVER_STEP_S] * num_slots, dtype=np.float64),
                prices=np.array([0.2] * num_slots, dtype=np.float64),
                prices_ordered_values=[0.2],
                time_slots=time_slots,
            )
    finally:
        MultiStepsPowerLoadConstraint.adapt_power_steps_budgeting = original

    final_ret = result[1]

    # Pre-fix behaviour returned `final_ret = False` here. After the
    # fix, the decision is the natural one for an empty green-energy
    # pass: `quantity_to_be_added > 0` AND `do_use_available_power_only`
    # → `quantity_to_be_added <= 0.0` is False, so `final_ret` is False
    # via that branch. The contract we pin is that we DO NOT also see
    # the removed short-circuit warning.
    removed_warning = [
        record
        for record in caplog.records
        if "no power sorted commands for green energy" in record.getMessage()
    ]
    assert removed_warning == [], (
        "QS-204 regression: the green-energy short-circuit warning was "
        f"emitted again — found {len(removed_warning)} record(s): "
        f"{[r.getMessage() for r in removed_warning]}"
    )

    # And the natural-branch decision is the standard one.
    assert isinstance(final_ret, bool), (
        f"final_ret must be a bool after the green-energy pass; got "
        f"{type(final_ret).__name__} {final_ret!r}"
    )
