"""QS-204 AC1 — options-flow save no longer orphans the saved entry.

A switch-backed radiator (and any other QS device) disappeared from
the dashboard after editing its options flow because
``async_reload_quiet_solar`` excluded the edited entry from the
post-wipe reload pass. The HA update-listener registered in
``async_setup_entry`` was supposed to pick the entry up, but it
raced against ``hass.data[DOMAIN] = {}`` so the entry's reload
attached to nothing — the device never re-registered with the
freshly-built ``QSHome._all_devices``.

The fix appends an explicit reload of ``except_for_entry_id`` after
the main reload loop. This test fails on the parent commit (pre-T2)
and passes after T2.

The test also pins the class-wide nature of the fix by exercising it
against a ``QSOnOffDuration`` entry alongside the radiator (AC1's
non-radiator probe).
"""

from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_SWITCH,
    CONF_TYPE_NAME_QSOnOffDuration,
    CONF_TYPE_NAME_QSRadiator,
    DATA_HANDLER,
    DEVICE_TYPE,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.on_off_duration import QSOnOffDuration
from custom_components.quiet_solar.ha_model.radiator import QSRadiator

pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_options_flow_save_preserves_radiator_and_on_off_duration_in_home(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """AC1 — after an options-flow save, the edited entry is still in the home.

    The flow uses two non-home entries: a switch-backed radiator and a
    QSOnOffDuration probe. The radiator's options flow is the one being
    driven (the user-reported bug), but the QSOnOffDuration probe pins
    that the fix is in ``async_reload_quiet_solar`` rather than in the
    radiator specifically — every QS entry class hits the same code
    path.
    """
    # Pre-seed entity states so the device-mixin doesn't refuse setup.
    hass.states.async_set("switch.qs204_radiator_switch", "off", {})
    hass.states.async_set("switch.qs204_radiator_switch_new", "off", {})
    hass.states.async_set("switch.qs204_on_off_switch", "off", {})

    # Home first so the data handler builds a real QSHome.
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    radiator_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_NAME: "QS-204 Radiator",
            DEVICE_TYPE: CONF_TYPE_NAME_QSRadiator,
            CONF_SWITCH: "switch.qs204_radiator_switch",
            CONF_POWER: 1500,
        },
        entry_id="qs204_radiator_entry",
        title="QS-204 Radiator",
        unique_id="qs204_radiator_entry_unique",
    )
    radiator_entry.add_to_hass(hass)

    on_off_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_NAME: "QS-204 On Off Probe",
            DEVICE_TYPE: CONF_TYPE_NAME_QSOnOffDuration,
            CONF_SWITCH: "switch.qs204_on_off_switch",
            CONF_POWER: 800,
        },
        entry_id="qs204_on_off_entry",
        title="QS-204 On Off Probe",
        unique_id="qs204_on_off_entry_unique",
    )
    on_off_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(radiator_entry.entry_id)
    await hass.config_entries.async_setup(on_off_entry.entry_id)
    await hass.async_block_till_done()

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    # Sanity: both devices are present BEFORE the options-flow save.
    pre_radiators = [d for d in home._all_devices if isinstance(d, QSRadiator)]
    pre_on_offs = [d for d in home._all_devices if isinstance(d, QSOnOffDuration)]
    assert len(pre_radiators) == 1, (
        f"Pre-condition failure: expected exactly one QSRadiator, got {pre_radiators}"
    )
    assert len(pre_on_offs) == 1, (
        f"Pre-condition failure: expected exactly one QSOnOffDuration, got {pre_on_offs}"
    )

    # Drive the radiator options flow: change the switch entity.
    result = await hass.config_entries.options.async_init(radiator_entry.entry_id)
    flow_id = result["flow_id"]

    result = await hass.config_entries.options.async_configure(
        flow_id,
        user_input={
            CONF_NAME: "QS-204 Radiator",
            CONF_SWITCH: "switch.qs204_radiator_switch_new",
            CONF_POWER: 1500,
        },
    )
    await hass.async_block_till_done()

    # AC1 — the radiator re-registered with the freshly-built home,
    # now wired to the new switch entity.
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    home = data_handler.home

    radiators = [d for d in home._all_devices if isinstance(d, QSRadiator)]
    assert len(radiators) == 1, (
        f"AC1 fail (radiator orphaned): expected exactly one QSRadiator after "
        f"options-flow save, got {radiators}"
    )
    radiator = radiators[0]
    assert radiator.switch_entity == "switch.qs204_radiator_switch_new"

    # AC1 — the radiator still shows up in the radiators dashboard
    # section so the dashboard re-renders pick it up.
    dashboard_radiators = home.get_devices_for_dashboard_section("radiators")
    assert radiator in dashboard_radiators

    # AC1 non-radiator probe — the QSOnOffDuration entry that was NOT
    # the options-flow target must also be in the rebuilt home. This
    # proves the fix is class-wide (in async_reload_quiet_solar, not
    # in the radiator code path).
    on_offs = [d for d in home._all_devices if isinstance(d, QSOnOffDuration)]
    assert len(on_offs) == 1, (
        f"AC1 class-wide fail (non-radiator orphaned): expected exactly one "
        f"QSOnOffDuration after the options-flow save, got {on_offs}"
    )
    assert on_offs[0].name == "QS-204 On Off Probe"
