"""QS-362 regression test.

lbbrhzn/ocpp v0.12.0 registers `sensor.<cpid>_current_offered` for every charge point.
An OCPP device that binds this entity must run a full load-management cycle without
raising anything from the charger module (issue #362: a per-cycle `KeyError` from an
unregistered probe entity).

The test goes through the public cycle entry point on purpose: it names no
implementation symbol, so it guards the generic contract "an OCPP device exposing the
v0.12.0 extra entities runs a cycle cleanly".
"""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from tests.test_charger_coverage_deep import (
    _create_ocpp_charger,
    _init_charger_states,
    _make_hass,
    _make_home,
)


@pytest.mark.asyncio
async def test_ocpp_cycle_with_v012_current_offered_entity_runs_without_error(caplog):
    """QS-362: an OCPP device exposing lbbrhzn/ocpp v0.12.0's
    `sensor.<cpid>_current_offered` must run a full load-management cycle
    without any ERROR from the charger module."""
    hass, home = _make_hass(), _make_home()
    charger = _create_ocpp_charger(
        hass, home, extra_entity_ids=("sensor.ocppcharger_current_offered",)
    )
    _init_charger_states(charger)
    charger.is_charger_unavailable = MagicMock(return_value=False)
    charger.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger.is_not_plugged = MagicMock(return_value=False)
    charger.is_plugged = MagicMock(return_value=True)
    charger.set_charging_num_phases = AsyncMock(return_value=False)
    charger.set_max_charging_current = AsyncMock(return_value=True)
    charger.reboot = AsyncMock()
    now = datetime.now(pytz.UTC)

    with caplog.at_level(logging.ERROR, logger="custom_components.quiet_solar.ha_model.charger"):
        await charger.check_load_activity_and_constraints(now)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [
        r.getMessage() + (f" / {r.exc_info[1]!r}" if r.exc_info else "") for r in errors
    ]
