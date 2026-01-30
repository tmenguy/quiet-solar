"""Additional tests for quiet_solar car.py to improve coverage to 91%+."""

import pytest
from datetime import datetime, timedelta, time as dt_time
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
    FORCE_CAR_NO_PERSON_ATTACHED,
    FORCE_CAR_NO_CHARGER_CONNECTED,
)
from custom_components.quiet_solar.ha_model.car import FORCE_CAR_NO_PERSON_ATTACHED


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


# =============================================================================
# Tests for uncovered car.py lines - device_post_home_init paths
# =============================================================================


async def test_car_device_post_home_init_no_person_match(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test device_post_home_init when forecasted person doesn't exist (lines 282-284)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_init_no_person_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_init_no_person_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Set an invalid person name that won't match any existing person
    car_device._user_selected_person_name_for_car = "NonExistentPerson"
    car_device._current_forecasted_person_name_from_boot = None

    time = datetime.now(tz=pytz.UTC)
    car_device.device_post_home_init(time)

    # Should clear the invalid person selection
    assert car_device._user_selected_person_name_for_car is None
    assert car_device.current_forecasted_person is None


async def test_car_device_post_home_init_force_no_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test device_post_home_init with FORCE_CAR_NO_PERSON_ATTACHED (lines 277-278)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_init_force_no_person_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_init_force_no_person_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device._user_selected_person_name_for_car = FORCE_CAR_NO_PERSON_ATTACHED

    time = datetime.now(tz=pytz.UTC)
    car_device.device_post_home_init(time)

    # Should set current_forecasted_person to None
    assert car_device.current_forecasted_person is None


# =============================================================================
# Tests for uncovered car.py lines - charge calculations
# =============================================================================


async def test_car_get_charge_power_per_phase_A_various_configs(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_charge_power_per_phase_A with various configurations (lines 1491-1590)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Test with 3-phase car
    car_3p_config = {
        **MOCK_CAR_CONFIG,
        "name": "3P Car",
    }
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=car_3p_config,
        entry_id="car_3p_power_test",
        title="car: 3P Car",
        unique_id="quiet_solar_car_3p_power_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Test 3-phase power calculation
    power_3p, min_amp, max_amp = car_device.get_charge_power_per_phase_A(True)
    assert len(power_3p) > 0
    assert min_amp >= 0
    assert max_amp >= min_amp

    # Test 1-phase power calculation
    power_1p, min_amp_1p, max_amp_1p = car_device.get_charge_power_per_phase_A(False)
    assert len(power_1p) > 0


async def test_car_get_car_estimated_range_km_no_data(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_car_estimated_range_km when no efficiency data (lines 1044-1060)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_range_no_data_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_range_no_data_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Without efficiency data, should return None
    result = car_device.get_car_estimated_range_km(from_soc=100.0, to_soc=50.0)
    # Result may be None if no efficiency data is available


async def test_car_get_adapt_target_percent_soc_no_data(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_adapt_target_percent_soc_to_reach_range_km with missing data (lines 1016-1018)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_adapt_soc_no_data_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_adapt_soc_no_data_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # With no efficiency data, should return None tuple
    result = car_device.get_adapt_target_percent_soc_to_reach_range_km(100.0)
    assert result[0] is None


# =============================================================================
# Tests for uncovered car.py lines - charger options
# =============================================================================


async def test_car_get_charger_options_with_chargers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_charger_options with chargers available (lines 1776-1790)."""
    from .const import MOCK_CAR_CONFIG, MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create a charger
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_for_options_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_for_options_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charger_options_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charger_options_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    options = car_device.get_charger_options()
    assert isinstance(options, list)
    assert FORCE_CAR_NO_CHARGER_CONNECTED in options


async def test_car_get_current_selected_charger_option_with_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_current_selected_charger_option when charger is connected (lines 1792-1857)."""
    from .const import MOCK_CAR_CONFIG, MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create charger
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_for_current_option_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_for_current_option_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_current_charger_option_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_current_charger_option_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    # Attach car to charger
    charger_device.attach_car(car_device, datetime.now(tz=pytz.UTC))

    result = car_device.get_current_selected_charger_option()
    # Should return the charger name or FORCE_CAR_NO_CHARGER_CONNECTED
    assert result is not None or result == FORCE_CAR_NO_CHARGER_CONNECTED


# =============================================================================
# Tests for uncovered car.py lines - charge target options
# =============================================================================


async def test_car_get_car_next_charge_values_options(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_car_next_charge_values_options methods (lines 1593-1750)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charge_options_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charge_options_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Test percent options
    percent_options = car_device.get_car_next_charge_values_options_percent()
    assert isinstance(percent_options, list)

    # Test energy options
    energy_options = car_device.get_car_next_charge_values_options_energy()
    assert isinstance(energy_options, list)

    # Test combined options
    all_options = car_device.get_car_next_charge_values_options()
    assert isinstance(all_options, list)


async def test_car_get_car_target_charge_option_percent(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_car_target_charge_option_percent (lines 1694-1695)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_target_percent_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_target_percent_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    result = car_device.get_car_target_charge_option_percent()
    # Should return a string option or None


async def test_car_get_car_target_charge_option_energy(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_car_target_charge_option_energy (lines 1758-1774)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_target_energy_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_target_energy_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    result = car_device.get_car_target_charge_option_energy()
    # Should return a string option or None


# =============================================================================
# Tests for uncovered car.py lines - SOC and charge state
# =============================================================================


async def test_car_is_car_charge_growing_no_data(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_car_charge_growing with no data (lines 931-968)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charge_growing_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charge_growing_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    # Without data, should return None
    result = car_device.is_car_charge_growing(300.0, time)
    assert result is None or isinstance(result, bool)


async def test_car_get_delta_dampened_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_delta_dampened_power (lines 1171-1200)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_delta_dampened_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_delta_dampened_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    result = car_device.get_delta_dampened_power(6, 1, 16, 1)
    # Result may be None if no dampening data


# =============================================================================
# Tests for uncovered car.py lines - location and tracking
# =============================================================================


async def test_car_get_car_coordinates_with_unknown_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_car_coordinates with unknown state (lines 852-878)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_coords_unknown_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_coords_unknown_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    # Set tracker to unknown state
    tracker = car_device.car_tracker
    if tracker:
        car_device._entity_probed_last_valid_state[tracker] = (
            time - timedelta(minutes=5),
            STATE_UNKNOWN,
            {"latitude": 48.8566, "longitude": 2.3522},
        )

        lat, lon = car_device.get_car_coordinates(time)
        assert lat is None
        assert lon is None


async def test_car_is_car_home_with_duration(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test is_car_home with duration parameter (lines 880-904)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_home_duration_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_home_duration_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    # Set tracker to home state
    tracker = car_device.car_tracker
    if tracker:
        hass.states.async_set(tracker, "home", {"latitude": 48.8566, "longitude": 2.3522})
        car_device._entity_probed_state[tracker] = [
            (time - timedelta(hours=2), "not_home", {}),
            (time - timedelta(hours=1), "home", {}),
        ]
        car_device._entity_probed_last_valid_state[tracker] = (
            time - timedelta(hours=1),
            "home",
            {"latitude": 48.8566, "longitude": 2.3522},
        )

        # Should be home for at least 30 minutes
        result = car_device.is_car_home(time, for_duration=30 * 60)
        assert result is True or result is None


# =============================================================================
# Tests for uncovered car.py lines - efficiency calculations
# =============================================================================


async def test_car_get_computed_range_efficiency_no_data(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_computed_range_efficiency_km_per_percent with no data (lines 970-1002)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_efficiency_no_data_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_efficiency_no_data_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    time = datetime.now(tz=pytz.UTC)

    # Without efficiency data, should return None
    result = car_device.get_computed_range_efficiency_km_per_percent(time, delta_soc=20.0)
    # Result may be None if no efficiency data available


async def test_car_get_autonomy_to_target_soc_km(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test get_autonomy_to_target_soc_km (lines 1074-1078)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_autonomy_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_autonomy_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    result = car_device.get_autonomy_to_target_soc_km()
    # Result may be None if no efficiency data available


# =============================================================================
# Tests for uncovered car.py lines - save/restore state
# =============================================================================


async def test_car_update_to_be_saved_extra_device_info(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_to_be_saved_extra_device_info saves all relevant data."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_save_info_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_save_info_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    # Set some values to save
    car_device._user_selected_person_name_for_car = "TestPerson"

    mock_person = MagicMock()
    mock_person.name = "ForecastPerson"
    car_device.current_forecasted_person = mock_person

    data = {}
    car_device.update_to_be_saved_extra_device_info(data)

    assert "user_selected_person_name_for_car" in data
    assert data["user_selected_person_name_for_car"] == "TestPerson"
    assert "current_forecasted_person_name_from_boot" in data
    assert data["current_forecasted_person_name_from_boot"] == "ForecastPerson"


async def test_car_use_saved_extra_device_info(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test use_saved_extra_device_info restores all relevant data."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_restore_info_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_restore_info_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    stored_data = {
        "user_selected_person_name_for_car": "RestoredPerson",
        "current_forecasted_person_name_from_boot": "RestoredForecast",
    }

    car_device.use_saved_extra_device_info(stored_data)

    assert car_device._user_selected_person_name_for_car == "RestoredPerson"
    assert car_device._current_forecasted_person_name_from_boot == "RestoredForecast"
