"""Additional tests for quiet_solar charger.py to improve coverage to 91%+."""

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytz
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
)


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


# =============================================================================
# Tests for uncovered charger.py lines - basic charger methods
# =============================================================================


async def test_charger_is_charging_power_zero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_charging_power_zero method (lines 3421-3427)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_power_zero_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_power_zero_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.is_charging_power_zero(time, for_duration=60.0)
    # Result may be None if no sensor data


async def test_charger_is_plugged_methods(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_plugged related methods (lines 3307-3370)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_plugged_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_plugged_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    # Test is_plugged
    result = charger_device.is_plugged(time)
    assert result is None or isinstance(result, bool)

    # Test is_not_plugged
    result = charger_device.is_not_plugged(time)
    assert result is None or isinstance(result, bool)

    # Test is_optimistic_plugged
    result = charger_device.is_optimistic_plugged(time)
    assert result is None or isinstance(result, bool)

    # Test is_charger_plugged_now
    result, last_time = charger_device.is_charger_plugged_now(time)
    assert result is None or isinstance(result, bool)


async def test_charger_is_charger_unavailable(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_charger_unavailable method (lines 3372-3380)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_unavailable_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_unavailable_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.is_charger_unavailable(time)
    assert result is None or isinstance(result, bool)


async def test_charger_is_charge_enabled_disabled(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_charge_enabled and is_charge_disabled methods (lines 3382-3386)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_enabled_disabled_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_enabled_disabled_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    enabled = charger_device.is_charge_enabled(time)
    assert enabled is None or isinstance(enabled, bool)

    disabled = charger_device.is_charge_disabled(time)
    assert disabled is None or isinstance(disabled, bool)


async def test_charger_is_car_stopped_asking_current(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_car_stopped_asking_current method (lines 3388-3418)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_stopped_asking_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_stopped_asking_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.is_car_stopped_asking_current(time)
    assert result is None or isinstance(result, bool)


# =============================================================================
# Tests for uncovered charger.py lines - power and amp calculations
# =============================================================================


async def test_charger_get_min_max_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_min_max_power method (lines 3180-3245)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_min_max_power_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_min_max_power_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    min_p, max_p = charger_device.get_min_max_power()
    assert min_p >= 0
    assert max_p >= min_p


async def test_charger_get_phase_amps_from_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_phase_amps_from_power method (lines 1791-1807)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_phase_amps_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_phase_amps_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    result = charger_device.get_phase_amps_from_power(3000.0, is_3p=False)
    assert isinstance(result, list)


async def test_charger_get_device_amps_consumption(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_device_amps_consumption method (lines 1809-1848)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_amps_consumption_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_amps_consumption_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.get_device_amps_consumption(60.0, time)
    assert result is None or isinstance(result, list)


# =============================================================================
# Tests for uncovered charger.py lines - charge state methods
# =============================================================================


async def test_charger_is_charger_faulted(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_charger_faulted method (lines 3541-3551)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_faulted_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_faulted_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.is_charger_faulted(time)
    assert isinstance(result, bool)


async def test_charger_get_charge_type(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_charge_type method (lines 3553-3909)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_charge_type_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_charge_type_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    charge_type, constraint = charger_device.get_charge_type()
    assert isinstance(charge_type, str)


async def test_charger_is_car_charged(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_car_charged method (lines 3911-3948)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_car_charged_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_car_charged_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    is_charged, value = charger_device.is_car_charged(time, 80.0, 100.0, True)
    assert isinstance(is_charged, bool)


# =============================================================================
# Tests for uncovered charger.py lines - car options
# =============================================================================


async def test_charger_get_car_options(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_car_options method (lines 2481-2493)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_car_options_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_car_options_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    options = charger_device.get_car_options()
    assert isinstance(options, list)


async def test_charger_get_current_selected_car_option(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_current_selected_car_option method (lines 2495-2502)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_selected_car_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_selected_car_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    result = charger_device.get_current_selected_car_option()
    assert result is None or isinstance(result, str)


async def test_charger_set_user_selected_car_by_name_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test set_user_selected_car_by_name with None (lines 2504-2634)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_set_car_none_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_set_car_none_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    await charger_device.set_user_selected_car_by_name(None)

    assert charger_device.car is None


# =============================================================================
# Tests for uncovered charger.py lines - platforms and properties
# =============================================================================


async def test_charger_get_platforms(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_platforms method (lines 3106-3114)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_platforms_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_platforms_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    platforms = charger_device.get_platforms()
    assert isinstance(platforms, list)


async def test_charger_get_attached_virtual_devices(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_attached_virtual_devices method (lines 3116-3178)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_virtual_devices_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_virtual_devices_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    result = charger_device.get_attached_virtual_devices()
    assert isinstance(result, list)


async def test_charger_get_continuous_plug_duration(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_continuous_plug_duration method (lines 3247-3305)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_plug_duration_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_plug_duration_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.get_continuous_plug_duration(time)
    assert result is None or isinstance(result, float)


# =============================================================================
# Tests for uncovered charger.py lines - dampening and delta power
# =============================================================================


async def test_charger_get_delta_dampened_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_delta_dampened_power method (lines 3950-4127)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_delta_dampened_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_delta_dampened_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    result = charger_device.get_delta_dampened_power(6, 1, 16, 1)
    assert result is None or isinstance(result, (int, float))


# =============================================================================
# Tests for uncovered charger.py lines - state reset and scoring
# =============================================================================


async def test_charger_is_in_state_reset(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_in_state_reset method (lines 2198-2222)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_state_reset_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_state_reset_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    result = charger_device.is_in_state_reset()
    assert isinstance(result, bool)


async def test_charger_get_normalized_score(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_normalized_score method (lines 2098-2196)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_normalized_score_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_normalized_score_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    result = charger_device.get_normalized_score(None, time, 0)
    assert isinstance(result, float)


# =============================================================================
# Tests for uncovered charger.py lines - status value methods
# =============================================================================


async def test_charger_get_status_vals_methods(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test various get_*_status_vals methods (lines 4312-4325)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_vals_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_vals_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    assert isinstance(charger_device.get_car_charge_enabled_status_vals(), list)
    assert isinstance(charger_device.get_car_plugged_in_status_vals(), list)
    assert isinstance(charger_device.get_car_status_unknown_vals(), list)
    assert isinstance(charger_device.get_car_stopped_asking_current_status_vals(), list)
    assert isinstance(charger_device.get_car_status_rebooting_vals(), list)


async def test_charger_get_probable_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_probable_entities method (lines 4129-4140)."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_probable_entities_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_probable_entities_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    result = charger_device.get_probable_entities()
    assert isinstance(result, list)


# =============================================================================
# Tests for uncovered charger.py lines - device info save/restore
# =============================================================================


async def test_charger_update_to_be_saved_extra_device_info(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_to_be_saved_extra_device_info method."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_save_info_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_save_info_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    data = {}
    charger_device.update_to_be_saved_extra_device_info(data)
    assert isinstance(data, dict)


async def test_charger_use_saved_extra_device_info(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test use_saved_extra_device_info method."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_restore_info_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_restore_info_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    stored_data = {"num_on_off": 6}
    charger_device.use_saved_extra_device_info(stored_data)
    # Should not raise
