"""Extended tests for QSClimateDuration in ha_model/climate_controller.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from homeassistant.const import CONF_NAME
from homeassistant.components import climate
from homeassistant.components.climate import HVACMode
from homeassistant.core import HomeAssistant
import pytz

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.ha_model.climate_controller import (
    QSClimateDuration,
    get_hvac_modes,
)
from custom_components.quiet_solar.home_model.commands import CMD_ON, CMD_OFF, CMD_IDLE
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_CLIMATE_HVAC_MODE_OFF,
    SENSOR_CONSTRAINT_SENSOR_CLIMATE,
    CONF_SWITCH,
)

from tests.factories import create_minimal_home_model


@pytest.fixture
def climate_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Mock config entry for climate tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_climate_entry",
        data={CONF_NAME: "Test Climate"},
        title="Test Climate",
    )


@pytest.fixture
def climate_home(hass: HomeAssistant):
    """Home and data handler for climate tests."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    data_handler = MagicMock()
    data_handler.home = home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = data_handler
    return home


def test_init_with_default_hvac_modes(
    hass: HomeAssistant, climate_config_entry, climate_home
):
    """Test initialization with default HVAC modes."""
    device = QSClimateDuration(
        hass=hass,
        config_entry=climate_config_entry,
        home=climate_home,
            **{
                CONF_NAME: "Living Room Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.living_room",
            }
        )

    assert device.name == "Living Room Climate"
    assert device.climate_entity == "climate.living_room"
    assert device._state_off == str(HVACMode.OFF.value)
    assert device._state_on == str(HVACMode.AUTO.value)
    assert device.is_load_time_sensitive is True


def test_init_with_custom_hvac_modes(
    hass: HomeAssistant, climate_config_entry, climate_home
):
    """Test initialization with custom HVAC modes."""
    device = QSClimateDuration(
        hass=hass,
        config_entry=climate_config_entry,
        home=climate_home,
            **{
                CONF_NAME: "Bedroom Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.bedroom",
                CONF_CLIMATE_HVAC_MODE_ON: "heat",
                CONF_CLIMATE_HVAC_MODE_OFF: "fan_only",
            }
        )

    assert device._state_on == "heat"
    assert device._state_off == "fan_only"
    assert device._bistate_mode_on == "heat"
    assert device._bistate_mode_off == "fan_only"


def test_init_bistate_entity_equals_climate_entity(
    hass: HomeAssistant, climate_config_entry, climate_home
):
    """Test that bistate_entity is set to climate_entity."""
    device = QSClimateDuration(
        hass=hass,
        config_entry=climate_config_entry,
        home=climate_home,
            **{
                CONF_NAME: "Test Climate",
                CONF_SWITCH: "switch.climate_helper",
                CONF_CLIMATE: "climate.test",
            }
        )

    assert device.bistate_entity == "climate.test"


@pytest.fixture
def climate_device(hass: HomeAssistant, climate_config_entry, climate_home):
    """QSClimateDuration instance for property/translation/execute tests."""
    return QSClimateDuration(
        hass=hass,
        config_entry=climate_config_entry,
        home=climate_home,
        **{
            CONF_NAME: "Test Climate",
            CONF_SWITCH: "switch.climate_helper",
            CONF_CLIMATE: "climate.test",
        }
    )


def test_climate_state_on_getter(climate_device):
    """Test climate_state_on getter."""
    assert climate_device.climate_state_on == str(HVACMode.AUTO.value)


def test_climate_state_on_setter(climate_device):
    """Test climate_state_on setter."""
    climate_device.climate_state_on = "heat"
    assert climate_device.climate_state_on == "heat"
    assert climate_device._state_on == "heat"


def test_climate_state_off_getter(climate_device):
    """Test climate_state_off getter."""
    assert climate_device.climate_state_off == str(HVACMode.OFF.value)


def test_climate_state_off_setter(climate_device):
    """Test climate_state_off setter."""
    climate_device.climate_state_off = "fan_only"
    assert climate_device.climate_state_off == "fan_only"
    assert climate_device._state_off == "fan_only"


def test_get_virtual_current_constraint_translation_key(climate_device):
    """Test get_virtual_current_constraint_translation_key returns correct key."""
    result = climate_device.get_virtual_current_constraint_translation_key()
    assert result == SENSOR_CONSTRAINT_SENSOR_CLIMATE


def test_get_select_translation_key(climate_device):
    """Test get_select_translation_key returns correct key."""
    result = climate_device.get_select_translation_key()
    assert result == "climate_mode"


@pytest.fixture
def climate_device_execute(hass: HomeAssistant, climate_config_entry, climate_home):
    """QSClimateDuration instance for execute_command tests (custom HVAC modes)."""
    return QSClimateDuration(
        hass=hass,
        config_entry=climate_config_entry,
        home=climate_home,
        **{
            CONF_NAME: "Test Climate",
            CONF_SWITCH: "switch.climate_helper",
            CONF_CLIMATE: "climate.test",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
        }
    )


@pytest.fixture
def recorded_service_calls(hass: HomeAssistant):
    """Record service calls (domain, service, service_data) for assertions.
    Does not call the real service (climate domain may not be loaded in test).
    """
    from homeassistant.core import ServiceRegistry

    recorded = []

    async def record_only(self, domain, service, service_data=None, **kwargs):
        if self is hass.services:
            recorded.append((domain, service, service_data or {}))
        return None

    with patch.object(ServiceRegistry, "async_call", record_only):
        yield recorded


@pytest.mark.asyncio
async def test_execute_command_turn_on(
    climate_device_execute, recorded_service_calls
):
    """Test execute_command_system with CMD_ON sets HVAC to heat."""
    time = datetime.datetime.now(pytz.UTC)

    result = await climate_device_execute.execute_command_system(
        time, CMD_ON, state=None
    )

    assert result is False
    climate_calls = [c for c in recorded_service_calls if c[0] == climate.DOMAIN]
    assert len(climate_calls) >= 1
    assert climate_calls[0][1] == climate.SERVICE_SET_HVAC_MODE


@pytest.mark.asyncio
async def test_execute_command_turn_off(
    climate_device_execute, recorded_service_calls
):
    """Test execute_command_system with CMD_OFF sets HVAC to off."""
    time = datetime.datetime.now(pytz.UTC)

    result = await climate_device_execute.execute_command_system(
        time, CMD_OFF, state=None
    )

    assert result is False
    climate_calls = [c for c in recorded_service_calls if c[0] == climate.DOMAIN]
    assert len(climate_calls) >= 1


@pytest.mark.asyncio
async def test_execute_command_idle(
    climate_device_execute, recorded_service_calls
):
    """Test execute_command_system with CMD_IDLE sets HVAC to off."""
    time = datetime.datetime.now(pytz.UTC)

    result = await climate_device_execute.execute_command_system(
        time, CMD_IDLE, state=None
    )

    assert result is False
    climate_calls = [c for c in recorded_service_calls if c[0] == climate.DOMAIN]
    assert len(climate_calls) >= 1


@pytest.mark.asyncio
async def test_execute_command_with_override_state(
    climate_device_execute, recorded_service_calls
):
    """Test execute_command_system with explicit state override."""
    time = datetime.datetime.now(pytz.UTC)

    result = await climate_device_execute.execute_command_system(
        time, CMD_ON, state="cool"
    )

    assert result is False
    climate_calls = [c for c in recorded_service_calls if c[0] == climate.DOMAIN]
    assert len(climate_calls) >= 1
    last_call = climate_calls[-1]
    assert last_call[2].get(climate.ATTR_HVAC_MODE) == "cool"


@pytest.mark.asyncio
async def test_execute_command_invalid_raises(climate_device_execute):
    """Test execute_command_system with invalid command raises ValueError."""
    from custom_components.quiet_solar.home_model.commands import LoadCommand

    time = datetime.datetime.now(pytz.UTC)
    invalid_cmd = LoadCommand(command="invalid", power_consign=0)

    with pytest.raises(ValueError, match="Invalid command"):
        await climate_device_execute.execute_command_system(
            time, invalid_cmd, state=None
        )


def test_get_possibles_modes(
    hass: HomeAssistant, climate_config_entry, climate_home
):
    """Test get_possibles_modes returns modes from registry."""
    device = QSClimateDuration(
        hass=hass,
        config_entry=climate_config_entry,
        home=climate_home,
        **{
            CONF_NAME: "Test Climate",
            CONF_SWITCH: "switch.climate_helper",
            CONF_CLIMATE: "climate.test",
        }
    )
    mock_entry = MagicMock()
    mock_entry.capabilities = {"hvac_modes": ["off", "heat", "cool", "auto"]}
    mock_registry = MagicMock()
    mock_registry.async_get.return_value = mock_entry

    with patch(
        "custom_components.quiet_solar.ha_model.climate_controller.er.async_get",
        return_value=mock_registry,
    ):
        modes = device.get_possibles_modes()

    assert modes == ["off", "heat", "cool", "auto"]
