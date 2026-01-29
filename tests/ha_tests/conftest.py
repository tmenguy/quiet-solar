"""Fixtures for quiet_solar HA tests.

This module provides pytest fixtures following Home Assistant's testing patterns,
using pytest-homeassistant-custom-component for a real HA test harness.
"""

from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.config_entries import ConfigEntry, SOURCE_USER
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN

from .const import (
    MOCK_HOME_CONFIG,
    MOCK_CAR_CONFIG,
    MOCK_CHARGER_CONFIG,
    MOCK_PERSON_CONFIG,
    MOCK_BATTERY_CONFIG,
    MOCK_SOLAR_CONFIG,
    MOCK_DYNAMIC_GROUP_CONFIG,
    MOCK_HEAT_PUMP_CONFIG,
    MOCK_HOME_ENTRY_ID,
    MOCK_CAR_ENTRY_ID,
    MOCK_CHARGER_ENTRY_ID,
    MOCK_PERSON_ENTRY_ID,
    MOCK_BATTERY_ENTRY_ID,
    MOCK_SOLAR_ENTRY_ID,
    MOCK_DYNAMIC_GROUP_ENTRY_ID,
    MOCK_HEAT_PUMP_ENTRY_ID,
    MOCK_SENSOR_STATES,
)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable loading custom integrations in all tests.

    This fixture is from pytest-homeassistant-custom-component and allows
    the test to discover and load integrations from the custom_components directory.
    """
    yield


@pytest.fixture
def mock_setup_entry() -> Generator[AsyncMock]:
    """Override async_setup_entry to prevent full setup during config flow tests."""
    with patch(
        "custom_components.quiet_solar.async_setup_entry",
        return_value=True,
    ) as mock_setup:
        yield mock_setup


@pytest.fixture
async def mock_sensor_states(hass: HomeAssistant) -> None:
    """Set up mock sensor states in Home Assistant."""
    for entity_id, data in MOCK_SENSOR_STATES.items():
        hass.states.async_set(entity_id, data["state"], data.get("attributes", {}))


@pytest.fixture
def home_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for home device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_HOME_CONFIG,
        entry_id=MOCK_HOME_ENTRY_ID,
        title=f"home: {MOCK_HOME_CONFIG['name']}",
        unique_id=f"quiet_solar_home_{MOCK_HOME_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def car_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for car device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_CAR_CONFIG,
        entry_id=MOCK_CAR_ENTRY_ID,
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id=f"quiet_solar_car_{MOCK_CAR_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def charger_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for charger device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_CHARGER_CONFIG,
        entry_id=MOCK_CHARGER_ENTRY_ID,
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id=f"quiet_solar_charger_{MOCK_CHARGER_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def person_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for person device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_PERSON_CONFIG,
        entry_id=MOCK_PERSON_ENTRY_ID,
        title=f"person: {MOCK_PERSON_CONFIG['name']}",
        unique_id=f"quiet_solar_person_{MOCK_PERSON_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def battery_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for battery device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_BATTERY_CONFIG,
        entry_id=MOCK_BATTERY_ENTRY_ID,
        title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
        unique_id=f"quiet_solar_battery_{MOCK_BATTERY_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def solar_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for solar device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_SOLAR_CONFIG,
        entry_id=MOCK_SOLAR_ENTRY_ID,
        title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
        unique_id=f"quiet_solar_solar_{MOCK_SOLAR_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def dynamic_group_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for dynamic group device."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_DYNAMIC_GROUP_CONFIG,
        entry_id=MOCK_DYNAMIC_GROUP_ENTRY_ID,
        title=f"dynamic_group: {MOCK_DYNAMIC_GROUP_CONFIG['name']}",
        unique_id=f"quiet_solar_dynamic_group_{MOCK_DYNAMIC_GROUP_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
def heat_pump_config_entry(hass: HomeAssistant) -> ConfigEntry:
    """Create and register mock config entry for heat pump device (PilotedDevice)."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        source=SOURCE_USER,
        data=MOCK_HEAT_PUMP_CONFIG,
        entry_id=MOCK_HEAT_PUMP_ENTRY_ID,
        title=f"heat_pump: {MOCK_HEAT_PUMP_CONFIG['name']}",
        unique_id=f"quiet_solar_heat_pump_{MOCK_HEAT_PUMP_ENTRY_ID}",
    )
    config_entry.add_to_hass(hass)
    return config_entry


@pytest.fixture
async def setup_home_entry(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    mock_sensor_states: None,
) -> ConfigEntry:
    """Set up the home config entry."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()
    return home_config_entry


@pytest.fixture
async def setup_car_entry(
    hass: HomeAssistant,
    car_config_entry: ConfigEntry,
    setup_home_entry: ConfigEntry,  # Car requires home to be set up first
) -> ConfigEntry:
    """Set up the car config entry (requires home)."""
    await hass.config_entries.async_setup(car_config_entry.entry_id)
    await hass.async_block_till_done()
    return car_config_entry


@pytest.fixture
async def setup_charger_entry(
    hass: HomeAssistant,
    charger_config_entry: ConfigEntry,
    setup_home_entry: ConfigEntry,  # Charger requires home to be set up first
) -> ConfigEntry:
    """Set up the charger config entry (requires home)."""
    await hass.config_entries.async_setup(charger_config_entry.entry_id)
    await hass.async_block_till_done()
    return charger_config_entry


@pytest.fixture
async def setup_person_entry(
    hass: HomeAssistant,
    person_config_entry: ConfigEntry,
    setup_home_entry: ConfigEntry,  # Person requires home
    setup_car_entry: ConfigEntry,   # Person needs cars to be set up
) -> ConfigEntry:
    """Set up the person config entry (requires home and car)."""
    await hass.config_entries.async_setup(person_config_entry.entry_id)
    await hass.async_block_till_done()
    return person_config_entry


@pytest.fixture
def platforms() -> list[Platform]:
    """Return platforms to test.

    Override this fixture in test modules to limit which platforms are loaded.
    """
    return [Platform.SENSOR, Platform.SWITCH, Platform.BUTTON, Platform.NUMBER, Platform.SELECT, Platform.TIME]


@pytest.fixture
def override_platforms(platforms: list[Platform]) -> Generator[None]:
    """Override the platforms loaded by quiet_solar."""
    # Note: quiet_solar uses dynamic platform discovery per device,
    # so we may need to patch at the device level instead
    yield


# =============================================================================
# Real Object Factory Fixtures
# =============================================================================
# These fixtures provide factory functions for creating real quiet-solar objects
# using the real hass fixture from pytest_homeassistant_custom_component.


@pytest.fixture
def constraint_factory():
    """Factory fixture for creating real constraint objects.

    Constraints don't require HA, so this can be used in any test.

    Example:
        def test_constraint(constraint_factory):
            constraint = constraint_factory(
                constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                target_value=80.0,
            )
            assert constraint.target_value == 80.0
    """
    from tests.factories import create_constraint

    def factory(**kwargs):
        return create_constraint(**kwargs)

    return factory


@pytest.fixture
def charge_percent_constraint_factory():
    """Factory fixture for creating charge percent constraints.

    Example:
        def test_car_charging(charge_percent_constraint_factory):
            constraint = charge_percent_constraint_factory(
                initial_value=50.0,
                target_value=80.0,
                total_capacity_wh=60000.0,
            )
    """
    from tests.factories import create_charge_percent_constraint

    def factory(**kwargs):
        return create_charge_percent_constraint(**kwargs)

    return factory


@pytest.fixture
def load_command_factory():
    """Factory fixture for creating LoadCommand objects.

    Example:
        def test_commands(load_command_factory):
            cmd = load_command_factory(power_consign=7000.0)
            assert cmd.power_consign == 7000.0
    """
    from tests.factories import create_load_command

    def factory(**kwargs):
        return create_load_command(**kwargs)

    return factory


@pytest.fixture
def state_cmd_factory():
    """Factory fixture for creating QSStateCmd objects.

    Example:
        def test_charger_state(state_cmd_factory):
            state = state_cmd_factory(initial_value=True)
            assert state.value is True
    """
    from tests.factories import create_state_cmd

    def factory(**kwargs):
        return create_state_cmd(**kwargs)

    return factory


@pytest.fixture
def minimal_home_factory():
    """Factory fixture for creating minimal home models for unit tests.

    This creates a mock home suitable for pure logic tests that don't
    need full HA integration.

    Example:
        def test_device_logic(minimal_home_factory):
            home = minimal_home_factory(name="My Home", is_3p=True)
    """
    from tests.factories import create_minimal_home_model

    def factory(**kwargs):
        return create_minimal_home_model(**kwargs)

    return factory


@pytest.fixture
def charger_group_factory():
    """Factory fixture for creating QSChargerGroup objects.

    Example:
        def test_charger_group(charger_group_factory, minimal_home_factory):
            home = minimal_home_factory()
            group = charger_group_factory(home=home, max_amps=[32.0, 32.0, 32.0])
    """
    from tests.factories import create_charger_group

    def factory(**kwargs):
        return create_charger_group(**kwargs)

    return factory


@pytest.fixture
def minimal_load_factory():
    """Factory fixture for creating MinimalTestLoad objects.

    Useful for testing constraint and solver logic without full device setup.

    Example:
        def test_solver(minimal_load_factory):
            load = minimal_load_factory(name="TestLoad", power=2000.0)
    """
    from tests.factories import MinimalTestLoad

    def factory(**kwargs):
        return MinimalTestLoad(**kwargs)

    return factory


# =============================================================================
# HA Integration Factory Fixtures
# =============================================================================
# These fixtures create real quiet-solar device objects using the real hass
# fixture. Use these for integration tests that need full HA behavior.


@pytest.fixture
def car_factory(hass: HomeAssistant):
    """Factory fixture for creating real QSCar instances.

    This factory creates real QSCar objects using the actual class with
    the real hass fixture from pytest_homeassistant_custom_component.

    Example:
        async def test_car(hass, car_factory, setup_home_entry):
            home = hass.data[DOMAIN].get(setup_home_entry.entry_id)
            config_entry = MockConfigEntry(domain=DOMAIN, data={...})
            config_entry.add_to_hass(hass)
            car = await car_factory(config_entry=config_entry, home=home)
    """
    from tests.factories import create_real_car

    async def factory(config_entry, home, **kwargs):
        return await create_real_car(
            hass=hass,
            config_entry=config_entry,
            home=home,
            **kwargs,
        )

    return factory


@pytest.fixture
def charger_factory(hass: HomeAssistant):
    """Factory fixture for creating real QSChargerGeneric instances.

    Example:
        async def test_charger(hass, charger_factory, setup_home_entry):
            home = hass.data[DOMAIN].get(setup_home_entry.entry_id)
            config_entry = MockConfigEntry(domain=DOMAIN, data={...})
            config_entry.add_to_hass(hass)
            charger = await charger_factory(config_entry=config_entry, home=home)
    """
    from tests.factories import create_real_charger

    async def factory(config_entry, home, **kwargs):
        return await create_real_charger(
            hass=hass,
            config_entry=config_entry,
            home=home,
            **kwargs,
        )

    return factory


# =============================================================================
# Convenience Fixtures for Common Test Setups
# =============================================================================


@pytest.fixture
async def setup_home_and_car(
    hass: HomeAssistant,
    setup_home_entry: ConfigEntry,
    setup_car_entry: ConfigEntry,
) -> tuple[ConfigEntry, ConfigEntry]:
    """Set up home and car together.

    Returns tuple of (home_entry, car_entry) with both devices initialized.
    """
    return setup_home_entry, setup_car_entry


@pytest.fixture
async def setup_home_and_charger(
    hass: HomeAssistant,
    setup_home_entry: ConfigEntry,
    setup_charger_entry: ConfigEntry,
) -> tuple[ConfigEntry, ConfigEntry]:
    """Set up home and charger together.

    Returns tuple of (home_entry, charger_entry) with both devices initialized.
    """
    return setup_home_entry, setup_charger_entry


@pytest.fixture
async def setup_full_charging(
    hass: HomeAssistant,
    setup_home_entry: ConfigEntry,
    setup_car_entry: ConfigEntry,
    setup_charger_entry: ConfigEntry,
) -> tuple[ConfigEntry, ConfigEntry, ConfigEntry]:
    """Set up home, car, and charger for charging tests.

    Returns tuple of (home_entry, car_entry, charger_entry).
    """
    return setup_home_entry, setup_car_entry, setup_charger_entry


@pytest.fixture
def get_device_from_entry(hass: HomeAssistant):
    """Helper fixture to get a device from its config entry.

    Example:
        def test_device(hass, setup_car_entry, get_device_from_entry):
            car = get_device_from_entry(setup_car_entry)
            assert car.name == "Test Car"
    """

    def getter(entry: ConfigEntry):
        return hass.data[DOMAIN].get(entry.entry_id)

    return getter
