"""QS-204 AC3 — Force ON dispatches ``switch.turn_on`` end-to-end.

A switch-backed radiator with no command history (fresh setup,
``current_command`` and ``running_command`` both ``None``) used to
ignore a "Force ON" select-change because the constraints.py
green-energy short-circuit refused to allocate any command when
``has_a_cmd`` stayed ``False`` throughout the slot loop. The user
applied the constraints.py fix on this branch; this test pins the
end-to-end behaviour so a future regression cannot silently break
"Force ON" again.

Test shape:

1. Set up home + a switch-backed radiator. Pre-set the backing
   switch entity state to ``"off"``.
2. Register an async-mock handler on ``switch.turn_on`` and
   ``switch.turn_off`` to capture every service call.
3. Read the radiator's ``bistate_mode`` select entity id from the
   entity registry.
4. Call ``select.select_option`` with the namespaced
   ``"radiator_mode_on"`` option (matches the T4 rename).
5. Advance time by ``_load_update_scan_interval + 1`` seconds so the
   load-management cycle fires and the solver re-evaluates.
6. Assert the ``switch.turn_on`` mock recorded exactly one call
   against the configured backing switch entity.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_SWITCH,
    CONF_TYPE_NAME_QSRadiator,
    DATA_HANDLER,
    DEVICE_TYPE,
    DOMAIN,
)

pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_radiator_force_on_dispatches_switch_turn_on(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """AC3 — selecting Force ON makes the solver dispatch ``switch.turn_on``.

    The radiator is configured with no calendar, no override, and a
    backing switch currently reported as ``"off"``. After the user
    selects the namespaced ``radiator_mode_on`` option, the next
    load-management cycle must allocate an ON command and dispatch
    ``switch.turn_on`` to the backing switch exactly once.

    The forecast-history initialiser is mocked out — it loops on
    empty sensor history in the test harness, and the dispatch logic
    we're testing does not depend on it.
    """
    backing_switch = "switch.qs204_force_on_target"
    hass.states.async_set(backing_switch, "off", {})

    # The forecast-history initialiser is unrelated to the dispatch
    # we're testing and loops on an empty HA recorder in this harness.
    # Short-circuit it so the test stays bounded. The dispatch path
    # we're exercising does not read from the forecast history.
    forecast_init_patch = patch(
        "custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals.init",
        new=AsyncMock(return_value=(None, None)),
    )

    forecast_init_patch.start()
    try:
        await _run_force_on_test(hass, home_config_entry, entity_registry, backing_switch)
    finally:
        forecast_init_patch.stop()


async def _run_force_on_test(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
    backing_switch: str,
) -> None:
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    radiator_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_NAME: "QS-204 Force ON Radiator",
            DEVICE_TYPE: CONF_TYPE_NAME_QSRadiator,
            CONF_SWITCH: backing_switch,
            CONF_POWER: 1500,
        },
        entry_id="qs204_force_on_radiator_entry",
        title="QS-204 Force ON Radiator",
        unique_id="qs204_force_on_radiator_entry_unique",
    )
    radiator_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(radiator_entry.entry_id)
    await hass.async_block_till_done()

    # Enable the home so the load-management cycle actually executes
    # the solver (default is `home_mode_sensors_only` which short-
    # circuits update_loads_constraints early).
    home_mode_entity = next(
        (
            e.entity_id
            for e in er.async_entries_for_config_entry(entity_registry, home_config_entry.entry_id)
            if e.domain == "select" and "home_mode" in (e.unique_id or "")
        ),
        None,
    )
    assert home_mode_entity is not None, "home_mode select missing from home entry"
    await hass.services.async_call(
        "select",
        "select_option",
        {"entity_id": home_mode_entity, "option": "home_mode_on"},
        blocking=True,
    )
    await hass.async_block_till_done()

    # Spy on switch.turn_on / switch.turn_off so we can assert the
    # dispatch happened exactly once.
    turn_on_calls = async_mock_service(hass, "switch", "turn_on")
    turn_off_calls = async_mock_service(hass, "switch", "turn_off")

    # Locate the radiator's bistate_mode select via the entity registry.
    entity_entries = er.async_entries_for_config_entry(entity_registry, radiator_entry.entry_id)
    select_entries = [
        e for e in entity_entries
        if e.domain == "select" and e.translation_key == "radiator_mode"
    ]
    assert len(select_entries) == 1, (
        f"Expected exactly one radiator_mode select on the radiator entry, "
        f"got {select_entries}"
    )
    select_entity_id = select_entries[0].entity_id

    # Sanity — the radiator instance is registered with the home.
    # NOTE: by the time the home is moved to `home_mode_on` and the
    # entity setup settles, the load-management cycle may have already
    # dispatched a `CMD_IDLE` to the radiator. That's the expected
    # baseline — the test exercises the Force ON transition FROM the
    # idle/no-command state TO the on state.
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    radiators = [d for d in data_handler.home._all_devices if d.name == "QS-204 Force ON Radiator"]
    assert len(radiators) == 1

    # Trigger Force ON via the namespaced literal.
    await hass.services.async_call(
        "select",
        "select_option",
        {"entity_id": select_entity_id, "option": "radiator_mode_on"},
        blocking=True,
    )
    await hass.async_block_till_done()

    # Run one load-management cycle directly. Going through the
    # registered time-interval (async_fire_time_changed) ends up
    # blocking on long-running forecast-probe initialisation in this
    # test harness — calling the cycle directly is faster and equally
    # exercises the path that dispatches commands.
    now = dt_util.utcnow() + datetime.timedelta(
        seconds=data_handler._load_update_scan_interval + 1
    )
    await data_handler.async_update_loads(now)
    await hass.async_block_till_done()

    # AC3 — exactly one switch.turn_on dispatched to the backing entity.
    matching_on = [c for c in turn_on_calls if c.data.get("entity_id") == backing_switch]
    assert len(matching_on) == 1, (
        f"AC3 fail: expected exactly one switch.turn_on dispatch to "
        f"{backing_switch}, got {len(matching_on)} (all turn_on calls: {turn_on_calls})"
    )
    # And no spurious turn_off on the same entity.
    matching_off = [c for c in turn_off_calls if c.data.get("entity_id") == backing_switch]
    assert matching_off == [], (
        f"AC3 fail: unexpected switch.turn_off dispatched after Force ON: {matching_off}"
    )
