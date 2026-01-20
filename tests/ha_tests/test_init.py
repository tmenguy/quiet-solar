"""Tests for quiet_solar integration setup and unload."""

from unittest.mock import patch

import pytest

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant

from custom_components.quiet_solar.const import DOMAIN, DATA_HANDLER


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_setup_home_entry(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test setting up the home config entry."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED
    assert DOMAIN in hass.data
    assert DATA_HANDLER in hass.data[DOMAIN]


async def test_setup_unload_home_entry(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test setting up and unloading the home config entry."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED

    await hass.config_entries.async_unload(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.NOT_LOADED


async def test_setup_car_entry_with_home(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test setting up car entry after home is set up."""
    # First set up home
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED

    # Create a new car config entry (not from fixture to avoid auto-setup)
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from .const import MOCK_CAR_CONFIG

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_entry_new_123",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_new_123",
    )
    car_entry.add_to_hass(hass)

    # Then set up car
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert car_entry.state is ConfigEntryState.LOADED


async def test_data_handler_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test that data handler is created on setup."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert DATA_HANDLER in hass.data[DOMAIN]
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    assert data_handler is not None
    assert data_handler.home is not None


@pytest.mark.skip(reason="Charger requires more complex mocking of Wallbox entities - to be implemented")
async def test_setup_charger_entry_with_home(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test setting up charger entry after home is set up.

    Note: This test is skipped because the Wallbox charger type requires
    specific entity IDs (number.*_maximum_charging_current, switch.*_pause_resume, etc.)
    that need to be mocked. A future test should set up these mock entities.
    """
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED

    # Create charger config entry
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from .const import MOCK_CHARGER_CONFIG

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_entry_new_123",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_new_123",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED


async def test_setup_person_entry_with_dependencies(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test setting up person entry with all dependencies."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from .const import MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG

    # Set up home
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Set up car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_entry_person_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_person_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Set up person
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_PERSON_CONFIG,
        entry_id="person_entry_test",
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id="quiet_solar_person_test",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    assert person_entry.state is ConfigEntryState.LOADED


async def test_reload_entry(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test reloading a config entry."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED

    # Reload the entry
    await hass.config_entries.async_reload(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert home_config_entry.state is ConfigEntryState.LOADED
