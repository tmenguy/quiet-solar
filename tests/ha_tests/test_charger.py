"""Tests for quiet_solar charger.py functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

import pytz
from types import SimpleNamespace

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH,
    CONF_IS_3P,
    CHARGER_NO_CAR_CONNECTED,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_SCHEDULE,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
    CAR_CHARGE_TYPE_TARGET_MET,
    CAR_CHARGE_TYPE_NOT_PLUGGED,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_FROM_CONSIGN,
    CMD_AUTO_GREEN_CAP,
    CMD_AUTO_GREEN_CONSIGN,
    CMD_AUTO_PRICE,
    CMD_IDLE,
    CMD_ON,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MAX_UTC,
    MultiStepsPowerLoadConstraint,
    MultiStepsPowerLoadConstraintChargePercent,
)
from custom_components.quiet_solar.ha_model.charger import (
    CHARGER_ADAPTATION_WINDOW_S,
    QSChargerGroup,
    QSChargerStatus,
    QSStateCmd,
)


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_charger_initialization(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger device initialization."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_init_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_init_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED

    # Verify charger device was created
    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None
    assert charger_device.name == MOCK_CHARGER_CONFIG['name']


async def test_charger_amp_limits(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger amperage limits configuration."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_amp_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_amp_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_min_charge == MOCK_CHARGER_CONFIG[CONF_CHARGER_MIN_CHARGE]
    assert charger_device.charger_max_charge == MOCK_CHARGER_CONFIG[CONF_CHARGER_MAX_CHARGE]


async def test_charger_status_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger status sensor reading."""
    from .const import MOCK_CHARGER_CONFIG

    # Set up mock status sensor
    hass.states.async_set("sensor.test_charger_status", "Charging")

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_status_sensor == "sensor.test_charger_status"


async def test_charger_pause_resume_switch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger pause/resume switch configuration."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_switch_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_switch_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_pause_resume_switch == "switch.test_charger_pause_resume"


async def test_charger_max_current_number(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger max current number entity configuration."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_current_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_current_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.charger_max_charging_current_number == "number.test_charger_max_current"


async def test_charger_plugged_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger plugged sensor configuration."""
    from .const import MOCK_CHARGER_CONFIG

    # Set up mock plugged sensor
    hass.states.async_set("binary_sensor.test_charger_plugged", "on")

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
    assert charger_device.charger_plugged == "binary_sensor.test_charger_plugged"


async def test_charger_select_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger select entities are created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_select_entity_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_select_entity_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Check select entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )

    select_entries = [e for e in entity_entries if e.domain == "select"]

    # Should have at least connected car selection
    assert len(select_entries) >= 1, f"Expected at least 1 select entity, got {len(select_entries)}"


async def test_charger_button_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger button entities are created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_button_entity_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_button_entity_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Check button entities
    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )

    button_entries = [e for e in entity_entries if e.domain == "button"]

    # Should have buttons: reset, clean constraints
    assert len(button_entries) >= 2, f"Expected at least 2 button entities, got {len(button_entries)}"


async def test_charger_generic_car_creation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test that charger creates a generic car for unknown vehicles."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_generic_car_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_generic_car_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    # Charger should have a default generic car
    assert hasattr(charger_device, '_default_generic_car')
    assert charger_device._default_generic_car is not None


async def test_charger_with_car_connected(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger with a car connected."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create charger
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_connected_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_connected_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    # Create car
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_connected_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_connected_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED
    assert car_entry.state is ConfigEntryState.LOADED


async def test_multiple_chargers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test multiple chargers can be created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create first charger
    charger1_config = {
        **MOCK_CHARGER_CONFIG,
        "name": "Charger 1",
        "charger_max_charging_current_number": "number.charger1_max_current",
        "charger_pause_resume_switch": "switch.charger1_pause_resume",
        "charger_status_sensor": "sensor.charger1_status",
        "charger_plugged": "binary_sensor.charger1_plugged",
    }

    # Set up mock entities for charger 1
    hass.states.async_set("number.charger1_max_current", "32", {"min": 6, "max": 32})
    hass.states.async_set("switch.charger1_pause_resume", "on")
    hass.states.async_set("sensor.charger1_status", "Ready")
    hass.states.async_set("binary_sensor.charger1_plugged", "off")

    charger1_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger1_config,
        entry_id="charger1_multi_test",
        title="charger: Charger 1",
        unique_id="quiet_solar_charger1_multi_test",
    )
    charger1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger1_entry.entry_id)
    await hass.async_block_till_done()

    # Create second charger
    charger2_config = {
        **MOCK_CHARGER_CONFIG,
        "name": "Charger 2",
        "charger_max_charging_current_number": "number.charger2_max_current",
        "charger_pause_resume_switch": "switch.charger2_pause_resume",
        "charger_status_sensor": "sensor.charger2_status",
        "charger_plugged": "binary_sensor.charger2_plugged",
    }

    # Set up mock entities for charger 2
    hass.states.async_set("number.charger2_max_current", "32", {"min": 6, "max": 32})
    hass.states.async_set("switch.charger2_pause_resume", "on")
    hass.states.async_set("sensor.charger2_status", "Ready")
    hass.states.async_set("binary_sensor.charger2_plugged", "off")

    charger2_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger2_config,
        entry_id="charger2_multi_test",
        title="charger: Charger 2",
        unique_id="quiet_solar_charger2_multi_test",
    )
    charger2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger2_entry.entry_id)
    await hass.async_block_till_done()

    assert charger1_entry.state is ConfigEntryState.LOADED
    assert charger2_entry.state is ConfigEntryState.LOADED


async def test_charger_single_phase(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger configured for single phase."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,  # Already configured for single phase
        entry_id="charger_1p_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_1p_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None
    # Single phase config
    assert MOCK_CHARGER_CONFIG.get("device_is_3p") is False


async def test_charger_three_phase(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger configured for three phase."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Create 3-phase charger config
    charger_3p_config = {
        **MOCK_CHARGER_CONFIG,
        "name": "Charger 3P",
        "device_is_3p": True,
        "charger_max_charging_current_number": "number.charger_3p_max_current",
        "charger_pause_resume_switch": "switch.charger_3p_pause_resume",
        "charger_status_sensor": "sensor.charger_3p_status",
        "charger_plugged": "binary_sensor.charger_3p_plugged",
    }

    # Set up mock entities
    hass.states.async_set("number.charger_3p_max_current", "32", {"min": 6, "max": 32})
    hass.states.async_set("switch.charger_3p_pause_resume", "on")
    hass.states.async_set("sensor.charger_3p_status", "Ready")
    hass.states.async_set("binary_sensor.charger_3p_plugged", "off")

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger_3p_config,
        entry_id="charger_3p_test",
        title="charger: Charger 3P",
        unique_id="quiet_solar_charger_3p_test",
    )
    charger_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device is not None


async def test_charger_get_platforms(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger returns correct platforms."""
    from .const import MOCK_CHARGER_CONFIG
    from homeassistant.const import Platform

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

    assert Platform.SENSOR in platforms
    assert Platform.SELECT in platforms
    assert Platform.BUTTON in platforms


async def test_charger_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger unload."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_unload_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_unload_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.LOADED

    # Unload
    await hass.config_entries.async_unload(charger_entry.entry_id)
    await hass.async_block_till_done()

    assert charger_entry.state is ConfigEntryState.NOT_LOADED


async def test_charger_default_car_options(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger default car options."""
    from .const import MOCK_CHARGER_CONFIG
    from custom_components.quiet_solar.const import CHARGER_NO_CAR_CONNECTED

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

    # Get car options
    options = charger_device.get_car_options()

    # Should have at least the "no car connected" option and generic car
    assert CHARGER_NO_CAR_CONNECTED in options


async def test_charger_charge_state_initial(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger initial charge state."""
    from .const import MOCK_CHARGER_CONFIG
    from homeassistant.const import STATE_UNKNOWN

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_charge_state_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_charge_state_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    # Initial charge state should be unknown
    assert charger_device.charge_state == STATE_UNKNOWN


async def test_charger_dashboard_sort_string(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger dashboard sort string."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_sort_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_sort_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    # Charger should have a sort string
    sort_string = charger_device.dashboard_sort_string_in_type
    assert sort_string is not None


async def test_charger_conf_type_name(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger conf_type_name class attribute."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
    from custom_components.quiet_solar.const import CONF_TYPE_NAME_QSChargerGeneric

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    assert QSChargerGeneric.conf_type_name == CONF_TYPE_NAME_QSChargerGeneric


async def test_charger_voltage(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger voltage from home."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_HOME_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_voltage_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_voltage_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    # Charger should inherit voltage from home
    assert charger_device.voltage == MOCK_HOME_CONFIG.get("home_voltage", 230)


async def test_charger_is_plugged_property(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger is_plugged property."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_plugged_prop_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_plugged_prop_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    # Charger should have is_plugged property
    assert hasattr(charger_device, 'is_plugged')


async def test_charger_connected_car_property(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger car connection property."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_car_prop_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_car_prop_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    # Charger should have _default_generic_car property
    assert hasattr(charger_device, '_default_generic_car')


async def test_charger_current_power_w(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger charge state property."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_power_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_power_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    # Charger should have charge_state property
    assert hasattr(charger_device, 'charge_state')


async def test_charger_get_charge_type(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger get_charge_type method."""
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
    # Charger should have get_charge_type method
    charge_type = charger_device.get_charge_type()
    assert isinstance(charge_type, tuple)


async def test_charger_power_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger power steps calculation."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_power_steps_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_power_steps_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    # Charger should have power steps
    assert hasattr(charger_device, 'get_best_car')


async def test_charger_enabled_property(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger name property."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_enabled_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_enabled_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    # Charger should have name property
    assert charger_device.name == MOCK_CHARGER_CONFIG['name']


async def test_charger_time_entity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger time entity is created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_time_entity_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_time_entity_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    time_entries = [e for e in entity_entries if e.domain == "time"]
    # Charger should have default charge time entity
    assert len(time_entries) >= 1


async def test_charger_sensor_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger sensor entities are created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_sensor_entities_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_sensor_entities_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    sensor_entries = [e for e in entity_entries if e.domain == "sensor"]
    # Charger should have multiple sensors
    assert len(sensor_entries) >= 1


async def test_charger_current_num_phases_switch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test current_num_phases with phase switch."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    switch_entity = "switch.charger_phase"
    hass.states.async_set(switch_entity, "on")

    charger_config = {
        **MOCK_CHARGER_CONFIG,
        CONF_IS_3P: True,
        CONF_CHARGER_THREE_TO_ONE_PHASE_SWITCH: switch_entity,
    }
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=charger_config,
        entry_id="charger_phase_switch_test",
        title="charger: Charger Phase Switch",
        unique_id="quiet_solar_charger_phase_switch_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    assert charger_device.current_num_phases == 1

    hass.states.async_set(switch_entity, "off")
    assert charger_device.current_num_phases == 3


async def test_charger_phase_amps_from_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test power to phase amps conversions."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_power_amps_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_power_amps_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_charger_power_amps",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_charger_power_amps",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.attach_car(car_device, datetime.now(tz=pytz.UTC))

    power_steps, min_charge, max_charge = car_device.get_charge_power_per_phase_A(True)
    target_power = power_steps[min_charge]

    amps = charger_device.get_phase_amps_from_power(target_power, is_3p=True)
    assert amps == [min_charge, min_charge, min_charge]

    assert charger_device.get_phase_amps_from_power(0.0, is_3p=True) == [0, 0, 0]

    too_high = power_steps[max_charge] + 100000
    assert charger_device._get_amps_from_power_steps(power_steps, too_high, safe_border=False) is None


async def test_charger_set_user_selected_car_detaches(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test selecting another car detaches current one."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_user_select_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_user_select_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_charger_user_select",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_charger_user_select",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.attach_car(car_device, datetime.now(tz=pytz.UTC))
    charger_device.update_charger_for_user_change = AsyncMock()

    await charger_device.set_user_selected_car_by_name("Other Car")

    assert charger_device.car is None
    assert charger_device.user_attached_car_name == "Other Car"
    charger_device.update_charger_for_user_change.assert_awaited_once()


async def test_charger_car_options_plugged_unplugged(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test car options in plugged/unplugged states."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_options_state_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_options_state_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_charger_options",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_charger_options",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.is_optimistic_plugged = MagicMock(return_value=True)

    options = charger_device.get_car_options()
    assert MOCK_CAR_CONFIG["name"] in options
    assert CHARGER_NO_CAR_CONNECTED in options

    charger_device.is_optimistic_plugged = MagicMock(return_value=False)
    assert charger_device.get_car_options() == [CHARGER_NO_CAR_CONNECTED]


async def test_charger_stable_dynamic_status_from_consign(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test stable status calculation for consign commands."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_consign_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_consign_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_status_consign",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_status_consign",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.attach_car(car_device, datetime.now(tz=pytz.UTC))

    charger_device.qs_enable_device = True
    charger_device.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=1500.0)
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
    charger_device.get_median_sensor = MagicMock(return_value=1200.0)
    charger_device._inner_expected_charge_state = SimpleNamespace(
        value=True, is_ok_to_set=MagicMock(return_value=True)
    )
    charger_device._inner_amperage = SimpleNamespace(value=charger_device.min_charge)
    charger_device._inner_num_active_phases = SimpleNamespace(
        value=3, is_ok_to_set=MagicMock(return_value=True)
    )

    status = charger_device.get_stable_dynamic_charge_status(datetime.now(tz=pytz.UTC))
    assert status is not None
    assert status.possible_amps is not None
    assert status.command.is_like(CMD_AUTO_FROM_CONSIGN)


async def test_charger_stable_dynamic_status_green_cap(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test stable status for green cap and phase switching."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_green_cap_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_green_cap_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_status_green_cap",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_status_green_cap",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.attach_car(car_device, datetime.now(tz=pytz.UTC))

    charger_device.qs_enable_device = True
    charger_device.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0.0)
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
    charger_device.get_median_sensor = MagicMock(return_value=0.0)
    charger_device._inner_expected_charge_state = SimpleNamespace(
        value=False, is_ok_to_set=MagicMock(return_value=True)
    )
    charger_device._inner_amperage = SimpleNamespace(value=0)
    charger_device._inner_num_active_phases = SimpleNamespace(
        value=3, is_ok_to_set=MagicMock(return_value=True)
    )

    car_device.qs_bump_solar_charge_priority = True
    charger_device.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0.0)

    status = charger_device.get_stable_dynamic_charge_status(datetime.now(tz=pytz.UTC))
    assert status is not None
    assert status.possible_amps is not None


async def test_charger_check_load_activity_unplugged(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test unplugged path resets car and charger."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_unplugged_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_unplugged_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_unplugged_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_unplugged_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    car_device._user_selected_person_name_for_car = "Person A"

    charger_device.is_not_plugged = MagicMock(return_value=True)
    charger_device.is_plugged = MagicMock(return_value=False)
    charger_device.set_charging_num_phases = AsyncMock(return_value=True)
    charger_device.reset = MagicMock()
    charger_device.home.get_best_persons_cars_allocations = AsyncMock(return_value={})

    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    do_force = await charger_device.check_load_activity_and_constraints(time)

    assert do_force is True
    assert car_device.user_selected_person_name_for_car is None
    assert charger_device.user_attached_car_name is None
    charger_device.reset.assert_called_once()


async def test_charger_check_load_activity_plugged_no_car(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test plugged path with no car selected."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_plugged_no_car_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_plugged_no_car_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_plugged = MagicMock(return_value=True)
    charger_device.get_best_car = MagicMock(return_value=None)
    charger_device.reset = MagicMock()

    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    do_force = await charger_device.check_load_activity_and_constraints(time)

    assert do_force is True
    charger_device.reset.assert_called_once()


async def test_charger_check_load_activity_force_constraint(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test force constraint creation for plugged car."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_force_constraint_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_force_constraint_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_force_constraint_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_force_constraint_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device

    class DummyConstraint:
        def __init__(self, **kwargs):
            self.name = "dummy"
            self.type = kwargs.get("type")
            self.load_param = kwargs.get("load_param")
            self.as_fast_as_possible = True
            self.end_of_constraint = DATETIME_MAX_UTC

        def is_constraint_active_for_time_period(self, time):
            return True

        def reset_load_param(self, value):
            self.load_param = value

    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_plugged = MagicMock(return_value=True)
    charger_device.get_best_car = MagicMock(return_value=car_device)
    charger_device.clean_constraints_for_load_param_and_info = MagicMock()
    charger_device.push_live_constraint = MagicMock(return_value=True)
    charger_device.command_and_constraint_reset = MagicMock()
    car_device.can_use_charge_percent_constraints = MagicMock(return_value=True)
    car_device.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)
    car_device.get_car_charge_percent = MagicMock(return_value=30.0)
    car_device.get_best_person_next_need = AsyncMock(
        return_value=(None, None, None, None)
    )
    car_device.do_force_next_charge = True
    car_device.do_next_charge_time = None

    with patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraintChargePercent",
        DummyConstraint,
    ):
        time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        do_force = await charger_device.check_load_activity_and_constraints(time)

    assert do_force is True
    charger_device.push_live_constraint.assert_called()


async def test_charger_get_charge_type_variants(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charge type determination with constraints."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

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

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_for_charge_type_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_for_charge_type_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device.is_charger_faulted = MagicMock(return_value=False)
    charger_device.possible_charge_error_start_time = None

    fast_ct = SimpleNamespace(
        as_fast_as_possible=True,
        end_of_constraint=DATETIME_MAX_UTC,
        load_info=None,
        is_constraint_active_for_time_period=MagicMock(return_value=True),
        is_constraint_met=MagicMock(return_value=False),
    )
    charger_device._constraints = [fast_ct]
    charge_type, ct = charger_device.get_charge_type()
    assert charge_type == CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE
    assert ct is fast_ct

    sched_ct = SimpleNamespace(
        as_fast_as_possible=False,
        end_of_constraint=datetime(2026, 1, 16, 8, 0, tzinfo=pytz.UTC),
        load_info={"person": "Person A"},
        is_constraint_active_for_time_period=MagicMock(return_value=True),
        is_constraint_met=MagicMock(return_value=False),
    )
    charger_device._constraints = [sched_ct]
    charge_type, ct = charger_device.get_charge_type()
    assert charge_type in (CAR_CHARGE_TYPE_SCHEDULE, CAR_CHARGE_TYPE_PERSON_AUTOMATED)
    assert ct is sched_ct

    met_ct = SimpleNamespace(
        as_fast_as_possible=False,
        end_of_constraint=DATETIME_MAX_UTC,
        load_info=None,
        is_constraint_active_for_time_period=MagicMock(return_value=False),
        is_constraint_met=MagicMock(return_value=True),
    )
    charger_device._constraints = [met_ct]
    charge_type, ct = charger_device.get_charge_type()
    assert charge_type == CAR_CHARGE_TYPE_TARGET_MET
    assert ct is met_ct

    charger_device.car = None
    charge_type, ct = charger_device.get_charge_type()
    assert charge_type == CAR_CHARGE_TYPE_NOT_PLUGGED


async def test_charger_update_power_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test power step updates when car attached."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_power_steps_update_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_power_steps_update_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_power_steps_update_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_power_steps_update_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device.can_do_3_to_1_phase_switch = MagicMock(return_value=True)
    charger_device._device_is_3p_conf = True
    charger_device.charger_min_charge = 1
    charger_device.charger_max_charge = 3
    car_device.car_charger_min_charge = 1
    car_device.car_charger_max_charge = 3
    car_device.get_charge_power_per_phase_A = MagicMock(
        side_effect=[
            ([0, 1000, 2000, 3000], 1, 3),
            ([0, 500, 1500, 2500], 1, 3),
        ]
    )

    charger_device.update_power_steps()
    assert charger_device._power_steps


async def test_charger_ensure_correct_state_unavailable(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test ensure_correct_state returns unavailable."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_ensure_unavailable_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_ensure_unavailable_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device._do_update_charger_state = AsyncMock()
    charger_device.is_charger_unavailable = MagicMock(return_value=True)

    res = await charger_device.ensure_correct_state(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    )
    assert res[0] is None


async def test_charger_check_load_activity_with_timed_constraints(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test check_load_activity creates timed constraints."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_timed_constraints_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_timed_constraints_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_timed_constraints_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_timed_constraints_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device._constraints = []
    charger_device._power_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=1000)]
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_plugged = MagicMock(return_value=True)
    charger_device.get_best_car = MagicMock(return_value=car_device)
    charger_device.clean_constraints_for_load_param_and_info = MagicMock()
    charger_device.push_live_constraint = MagicMock(return_value=True)
    charger_device.home.force_next_solve = MagicMock()

    person = SimpleNamespace(
        name="Person A",
        notify_of_forecast_if_needed=AsyncMock(),
    )

    car_device.can_use_charge_percent_constraints = MagicMock(return_value=True)
    car_device.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)
    car_device.get_car_charge_percent = MagicMock(return_value=20.0)
    car_device.get_best_person_next_need = AsyncMock(
        return_value=(False, datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC), 60.0, person)
    )
    car_device.do_force_next_charge = False
    car_device.do_next_charge_time = datetime(2026, 1, 15, 11, 0, tzinfo=pytz.UTC)

    class DummyConstraint:
        def __init__(self, **kwargs):
            self.name = "dummy"
            self.load_param = kwargs.get("load_param")
            self.end_of_constraint = kwargs.get("end_of_constraint", DATETIME_MAX_UTC)
            self.as_fast_as_possible = kwargs.get("type") == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
            self.is_before_battery = kwargs.get("is_before_battery", False)
            self.target_value = kwargs.get("target_value")
            self.load_info = kwargs.get("load_info")

        def is_constraint_active_for_time_period(self, time):
            return True

        def is_constraint_met(self, time):
            return False

        def reset_load_param(self, value):
            self.load_param = value

    with patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraintChargePercent",
        DummyConstraint,
    ), patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraint",
        DummyConstraint,
    ):
        do_force = await charger_device.check_load_activity_and_constraints(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )

    assert do_force is True
    person.notify_of_forecast_if_needed.assert_awaited()


async def test_charger_scores_and_before_battery(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test score and before-battery computation."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_score_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_score_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_score_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_score_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device.qs_bump_solar_charge_priority = True
    charger_device.current_command = CMD_AUTO_GREEN_CAP

    car_device.get_car_charge_percent = MagicMock(return_value=50.0)
    car_device.car_battery_capacity = 60000

    ct = SimpleNamespace(
        is_before_battery=True,
        end_of_constraint=datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC),
        as_fast_as_possible=False,
        is_mandatory=True,
    )

    is_before = charger_device.compute_is_before_battery(ct, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    score = charger_device.get_normalized_score(ct, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))

    assert isinstance(is_before, bool)
    assert score >= 0.0


async def test_charger_stable_status_cmd_on(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test stable status handling for CMD_ON."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG
    from custom_components.quiet_solar.home_model.commands import CMD_ON

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_cmd_on_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_cmd_on_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_status_cmd_on_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_status_cmd_on_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device.current_command = CMD_ON
    charger_device._expected_amperage.value = 6
    charger_device._expected_num_active_phases.value = 1
    charger_device._expected_charge_state.value = True
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
    charger_device.get_median_sensor = MagicMock(return_value=0)
    charger_device._device_is_3p_conf = True
    charger_device.charger_min_charge = 1
    charger_device.charger_max_charge = 2
    car_device.car_charger_min_charge = 1
    car_device.car_charger_max_charge = 2
    car_device.get_charge_power_per_phase_A = MagicMock(
        return_value=([0, 1000, 2000], 1, 2)
    )

    status = charger_device.get_stable_dynamic_charge_status(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    assert status is not None


async def test_charger_stable_status_green_cap_zero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test stable status handling for green cap zero."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_status_green_cap_zero_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_status_green_cap_zero_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_status_green_cap_zero_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_status_green_cap_zero_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device.current_command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0)
    charger_device._expected_amperage.value = 0
    charger_device._expected_num_active_phases.value = 1
    charger_device._expected_charge_state.value = False
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=False)
    charger_device.get_median_sensor = MagicMock(return_value=0)

    status = charger_device.get_stable_dynamic_charge_status(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    assert status is not None
    assert status.possible_amps is not None


async def test_charger_stable_status_early_returns(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test early return conditions for stable status."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_stable_early_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_stable_early_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)

    charger_device.qs_enable_device = False
    assert (
        charger_device.get_stable_dynamic_charge_status(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )
        is None
    )

    charger_device.qs_enable_device = True
    charger_device.car = None
    charger_device.is_not_plugged = MagicMock(return_value=True)
    assert (
        charger_device.get_stable_dynamic_charge_status(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )
        is None
    )

    charger_device.car = MagicMock()
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_charger_unavailable = MagicMock(return_value=True)
    assert (
        charger_device.get_stable_dynamic_charge_status(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )
        is None
    )

    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device._probe_and_enforce_stopped_charge_command_state = MagicMock(return_value=True)
    assert (
        charger_device.get_stable_dynamic_charge_status(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )
        is None
    )


async def test_charger_ensure_correct_state_phase_change(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test ensure_correct_state with phase correction."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_ensure_phase_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_ensure_phase_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_ensure_phase_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_ensure_phase_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.car = hass.data[DOMAIN].get(car_entry.entry_id)

    charger_device.is_in_state_reset = MagicMock(return_value=False)
    charger_device._asked_for_reboot_at_time = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    charger_device.check_if_reboot_happened = MagicMock(return_value=True)
    charger_device.update_data_request = AsyncMock()
    charger_device.set_charging_num_phases = AsyncMock()
    charger_device.is_charge_enabled = MagicMock(return_value=True)
    charger_device.is_charge_disabled = MagicMock(return_value=False)
    charger_device.get_charging_current = MagicMock(return_value=6)
    charger_device.set_charging_current = AsyncMock()

    charger_device._expected_num_active_phases.value = 1
    charger_device._expected_amperage.value = 6
    charger_device._expected_charge_state.value = True

    with patch.object(
        type(charger_device), "current_num_phases", new_callable=PropertyMock, return_value=3
    ):
        result = await charger_device._ensure_correct_state(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), probe_only=False
        )

    assert result is False
    charger_device.set_charging_num_phases.assert_awaited()


async def test_charger_ensure_correct_state_amperage_mismatch(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test ensure_correct_state triggers amperage set."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_ensure_amp_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_ensure_amp_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_ensure_amp_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_ensure_amp_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.car = hass.data[DOMAIN].get(car_entry.entry_id)

    charger_device.is_in_state_reset = MagicMock(return_value=False)
    charger_device.update_data_request = AsyncMock()
    charger_device.is_charge_enabled = MagicMock(return_value=True)
    charger_device.is_charge_disabled = MagicMock(return_value=False)
    charger_device.get_charging_current = MagicMock(return_value=2)
    charger_device.set_charging_current = AsyncMock()

    charger_device._expected_num_active_phases.value = 1
    charger_device._expected_amperage.value = 6
    charger_device._expected_charge_state.value = True

    with patch.object(
        type(charger_device), "current_num_phases", new_callable=PropertyMock, return_value=1
    ):
        result = await charger_device._ensure_correct_state(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), probe_only=False
        )

    assert result is False
    charger_device.set_charging_current.assert_awaited()


async def test_charger_check_load_activity_agenda_person_constraints(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test agenda and person constraint creation."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_agenda_person_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_agenda_person_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_agenda_person_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_agenda_person_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_plugged = MagicMock(return_value=True)
    charger_device.get_best_car = MagicMock(return_value=car_device)
    charger_device.clean_constraints_for_load_param_and_info = MagicMock()
    charger_device.attach_car = MagicMock(side_effect=lambda car, time: setattr(charger_device, "car", car))
    charger_device.push_live_constraint = MagicMock(return_value=True)
    charger_device.push_unique_and_current_end_of_constraint_from_agenda = MagicMock(return_value=True)
    charger_device.set_live_constraints = MagicMock()
    charger_device.command_and_constraint_reset = MagicMock()
    charger_device.is_car_charged = MagicMock(return_value=(False, None))
    charger_device.is_off_grid = MagicMock(return_value=False)
    charger_device.qs_bump_solar_charge_priority = False
    charger_device._auto_constraints_cleaned_at_user_reset = []
    charger_device._constraints = []
    charger_device._power_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=1000)]

    person = SimpleNamespace(name="Person A", notify_of_forecast_if_needed=AsyncMock())

    car_device.can_use_charge_percent_constraints = MagicMock(return_value=True)
    car_device.setup_car_charge_target_if_needed = AsyncMock(return_value=80.0)
    car_device.get_car_charge_percent = MagicMock(return_value=20.0)
    car_device.do_force_next_charge = False
    car_device.do_next_charge_time = None
    car_device.get_next_scheduled_event = AsyncMock(
        return_value=(datetime(2026, 1, 15, 18, 0, tzinfo=pytz.UTC), None)
    )
    car_device.get_best_person_next_need = AsyncMock(
        return_value=(False, datetime(2026, 1, 15, 16, 0, tzinfo=pytz.UTC), 50.0, person)
    )
    car_device.set_next_charge_target_percent = AsyncMock()
    car_device.get_car_target_SOC = MagicMock(return_value=80.0)
    car_device.get_car_minimum_ok_SOC = MagicMock(return_value=30.0)
    car_device.car_default_charge = 60.0
    car_device.car_battery_capacity = 60000

    class DummyConstraint:
        def __init__(self, **kwargs):
            self.name = "dummy"
            self.type = kwargs.get("type")
            self.load_param = kwargs.get("load_param")
            self.load_info = kwargs.get("load_info")
            self.from_user = kwargs.get("from_user", False)
            self.target_value = kwargs.get("target_value")
            self.initial_value = kwargs.get("initial_value")
            self.end_of_constraint = kwargs.get("end_of_constraint", DATETIME_MAX_UTC)
            self.as_fast_as_possible = self.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
            self.is_before_battery = False
            self.is_mandatory = self.type == CONSTRAINT_TYPE_MANDATORY_END_TIME

        def is_constraint_active_for_time_period(self, time):
            return True

        def is_constraint_met(self, time):
            return False

        def reset_load_param(self, value):
            self.load_param = value

    with patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraintChargePercent",
        DummyConstraint,
    ), patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraint",
        DummyConstraint,
    ):
        do_force = await charger_device.check_load_activity_and_constraints(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )

    assert do_force is True
    person.notify_of_forecast_if_needed.assert_awaited()


async def test_charger_check_load_activity_energy_constraints(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test energy-based constraints and forced charge."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_energy_constraints_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_energy_constraints_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_energy_constraints_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_energy_constraints_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)

    charger_device.is_charger_unavailable = MagicMock(return_value=False)
    charger_device.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.is_plugged = MagicMock(return_value=True)
    charger_device.get_best_car = MagicMock(return_value=car_device)
    charger_device.clean_constraints_for_load_param_and_info = MagicMock()
    charger_device.attach_car = MagicMock(side_effect=lambda car, time: setattr(charger_device, "car", car))
    charger_device.push_live_constraint = MagicMock(return_value=True)
    charger_device.set_live_constraints = MagicMock()
    charger_device.command_and_constraint_reset = MagicMock()
    charger_device._constraints = []
    charger_device._power_steps = [copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=1000)]

    car_device.can_use_charge_percent_constraints = MagicMock(return_value=False)
    car_device.get_car_target_charge_energy = MagicMock(return_value=15000.0)
    car_device.do_force_next_charge = True
    car_device.do_next_charge_time = None
    car_device.car_battery_capacity = 60000
    car_device.car_default_charge = 60.0

    class DummyConstraint:
        def __init__(self, **kwargs):
            self.name = "dummy"
            self.type = kwargs.get("type")
            self.load_param = kwargs.get("load_param")
            self.load_info = kwargs.get("load_info")
            self.from_user = kwargs.get("from_user", False)
            self.target_value = kwargs.get("target_value")
            self.initial_value = kwargs.get("initial_value")
            self.end_of_constraint = kwargs.get("end_of_constraint", DATETIME_MAX_UTC)
            self.as_fast_as_possible = self.type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
            self.is_before_battery = False
            self.is_mandatory = self.type == CONSTRAINT_TYPE_MANDATORY_END_TIME

        def is_constraint_active_for_time_period(self, time):
            return True

        def is_constraint_met(self, time):
            return False

        def reset_load_param(self, value):
            self.load_param = value

    with patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraintChargePercent",
        DummyConstraint,
    ), patch(
        "custom_components.quiet_solar.ha_model.charger.MultiStepsPowerLoadConstraint",
        DummyConstraint,
    ):
        do_force = await charger_device.check_load_activity_and_constraints(
            datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
        )

    assert do_force is True


def test_charger_status_helpers() -> None:
    """Test QSChargerStatus helper methods."""
    dummy_car = SimpleNamespace(
        name="Car A",
        get_charge_power_per_phase_A=MagicMock(return_value=([0, 1000, 2000], 1, 2)),
    )
    dummy_charger = SimpleNamespace(
        name="Charger A",
        mono_phase_index=1,
        car=dummy_car,
        update_amps_with_delta=MagicMock(return_value=[0, 5, 0]),
        get_delta_dampened_power=MagicMock(return_value=100.0),
        _get_amps_from_power_steps=MagicMock(return_value=1),
        can_do_3_to_1_phase_switch=MagicMock(return_value=True),
    )

    status = QSChargerStatus(dummy_charger)
    status.current_real_max_charging_amp = 5
    status.current_active_phase_number = 1
    status.budgeted_amp = 6
    status.budgeted_num_phases = 3
    status.possible_amps = [0, 6, 8]
    status.possible_num_phases = [1, 3]

    assert status.name == "Charger A/Car A"
    assert status.get_current_charging_amps() == [0.0, 5, 0.0]
    assert status.get_budget_amps() == [6, 6, 6]
    assert status.update_amps_with_delta([0, 5, 0], 1, 1) == [0, 5, 0]
    assert status.get_diff_power(5, 1, 6, 1) == 100.0

    next_amp, next_phases = status.can_change_budget(
        allow_state_change=True, allow_phase_change=True, increase=True
    )
    assert next_amp is not None
    assert next_phases in [1, 3]


def test_charger_status_consign_phase_switch() -> None:
    """Test QSChargerStatus consign amps with phase switch."""
    dummy_car = SimpleNamespace(
        name="Car B",
        get_charge_power_per_phase_A=MagicMock(
            side_effect=[
                ([0, 1000, 2000], 1, 2),
                ([0, 3000, 6000], 1, 2),
            ]
        ),
    )
    dummy_charger = SimpleNamespace(
        name="Charger B",
        mono_phase_index=0,
        car=dummy_car,
        _get_amps_from_power_steps=MagicMock(side_effect=[None, 2]),
        can_do_3_to_1_phase_switch=MagicMock(return_value=True),
        min_charge=1,
        max_charge=3,
    )

    status = QSChargerStatus(dummy_charger)
    status.command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=3000)
    status.current_active_phase_number = 1
    status.possible_amps = [0, 1, 2, 3]
    status.possible_num_phases = [1, 3]

    possible_phases, consign_amp = status.get_consign_amps_values(
        consign_is_minimum=True, add_tolerance=0.0
    )
    assert possible_phases is not None
    assert consign_amp is not None


def test_state_cmd_transitions() -> None:
    """Test QSStateCmd transitions and launch logic."""
    cmd = QSStateCmd()
    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    assert cmd.is_ok_to_set(time, min_change_time=0) is True
    cmd.set(1, time)
    assert cmd.is_ok_to_launch(1, time) is True
    cmd.register_launch(1, time)
    assert cmd.is_ok_to_launch(1, time) is False


async def test_charger_execute_command_and_probe(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test execute_command and probe_if_command_set paths."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_execute_cmd_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_execute_cmd_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_execute_cmd_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_execute_cmd_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.car = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.is_optimistic_plugged = MagicMock(return_value=True)
    charger_device.is_in_state_reset = MagicMock(return_value=False)
    charger_device._do_update_charger_state = AsyncMock()
    charger_device._ensure_correct_state = AsyncMock(return_value=True)
    charger_device._reset_state_machine = MagicMock()
    charger_device._probe_and_enforce_stopped_charge_command_state = MagicMock()

    time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    cmd = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=1000)
    res = await charger_device.execute_command(time, cmd)
    assert res is True
    charger_device._reset_state_machine.assert_called_once()

    charger_device.current_command = copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=500)
    res = await charger_device.execute_command(time, cmd)
    assert res is True

    charger_device.is_optimistic_plugged = MagicMock(return_value=False)
    res = await charger_device.execute_command(time, CMD_IDLE)
    assert res is True

    charger_device.is_optimistic_plugged = MagicMock(return_value=True)
    charger_device._ensure_correct_state = AsyncMock(return_value=False)
    result = await charger_device.probe_if_command_set(time, cmd)
    assert result is False

    charger_device.is_optimistic_plugged = MagicMock(return_value=False)
    result = await charger_device.probe_if_command_set(time, CMD_IDLE)
    assert result is True


async def test_charger_probe_stopped_state_handling(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test _probe_and_enforce_stopped_charge_command_state branches."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_probe_stopped_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_probe_stopped_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.car = None
    handled = charger_device._probe_and_enforce_stopped_charge_command_state(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), command=None, probe_only=False
    )
    assert handled is True

    charger_device.car = SimpleNamespace(
        name="Car A",
        car_charger_min_charge=6,
        car_charger_max_charge=32,
    )
    charger_device.is_car_stopped_asking_current = MagicMock(return_value=True)
    handled = charger_device._probe_and_enforce_stopped_charge_command_state(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), command=CMD_IDLE, probe_only=False
    )
    assert handled is True
    assert charger_device._expected_charge_state.value is False


def test_charger_power_helpers() -> None:
    """Test charger power helper methods."""
    dummy_car = SimpleNamespace(
        get_charge_power_per_phase_A=MagicMock(return_value=([0, 1000, 2000], 1, 2))
    )
    dummy_charger = SimpleNamespace(
        car=dummy_car,
        mono_phase_index=0,
        get_charge_power_per_phase_A=MagicMock(return_value=([0, 1000, 2000], 1, 2)),
    )

    charger_status = QSChargerStatus(dummy_charger)
    charger_status.current_real_max_charging_amp = 0
    charger_status.current_active_phase_number = 3
    assert charger_status.get_current_charging_amps() == [0.0, 0.0, 0.0]


async def test_charger_group_budget_handling() -> None:
    """Test QSChargerGroup budget application."""
    dynamic_group = SimpleNamespace(
        home=SimpleNamespace(),
        name="Group A",
        _childrens=[],
        is_current_acceptable=MagicMock(return_value=True),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger = DummyCharger()
    charger.name = "Charger A"
    charger.mono_phase_index = 0
    charger.min_charge = 6
    charger.max_charge = 32
    charger.charger_default_idle_charge = 6
    charger._expected_charge_state = QSStateCmd()
    charger._expected_amperage = QSStateCmd()
    charger._expected_num_active_phases = QSStateCmd()
    charger.num_on_off = 0
    charger._ensure_correct_state = AsyncMock()
    charger.car = SimpleNamespace(name="Car A")
    cs = QSChargerStatus(charger)
    cs.current_real_max_charging_amp = 10
    cs.current_active_phase_number = 1
    cs.budgeted_amp = 2
    cs.budgeted_num_phases = 1

    await group.apply_budgets([cs], [cs], datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), check_charger_state=True)
    assert charger._expected_charge_state.value is False
    charger._ensure_correct_state.assert_awaited()


async def test_charger_group_ensure_state_and_dyn_handle() -> None:
    """Test QSChargerGroup ensure_correct_state and dyn_handle."""
    dynamic_group = SimpleNamespace(
        home=SimpleNamespace(),
        name="Group B",
        _childrens=[],
        is_current_acceptable=MagicMock(return_value=True),
    )
    group = QSChargerGroup(dynamic_group)

    charger = SimpleNamespace(
        name="Charger B",
        qs_enable_device=True,
        ensure_correct_state=AsyncMock(return_value=(True, False, datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC))),
        get_stable_dynamic_charge_status=MagicMock(
            return_value=SimpleNamespace(charge_score=10, get_current_charging_amps=MagicMock(return_value=[0, 0, 0]),
                                         get_budget_amps=MagicMock(return_value=[0, 0, 0]),
                                         get_diff_power=MagicMock(return_value=0),
                                         current_real_max_charging_amp=0,
                                         current_active_phase_number=1,
                                         budgeted_amp=0,
                                         budgeted_num_phases=1,
                                         name="cs")
        ),
    )
    group._chargers = [charger]

    actionable, verified_time = await group.ensure_correct_state(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    assert actionable
    assert verified_time is not None

    group.remaining_budget_to_apply = [actionable[0]]
    group.apply_budgets = AsyncMock()
    await group.dyn_handle(datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    group.apply_budgets.assert_awaited()


async def test_charger_saved_info_and_user_updates(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test saved info handling and user actions."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_saved_info_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_saved_info_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.home.force_next_solve = MagicMock()
    charger_device.do_run_check_load_activity_and_constraints = AsyncMock(return_value=True)

    data = {}
    charger_device._auto_constraints_cleaned_at_user_reset = [
        SimpleNamespace(to_dict=MagicMock(return_value={"key": "value"}))
    ]
    charger_device.update_to_be_saved_extra_device_info(data)
    assert "auto_constraints_cleaned_at_user_reset" in data

    with patch(
        "custom_components.quiet_solar.ha_model.charger.LoadConstraint.new_from_saved_dict",
        side_effect=[SimpleNamespace(), None],
    ):
        charger_device.use_saved_extra_device_info(
            {"auto_constraints_cleaned_at_user_reset": [{"a": 1}, {"b": 2}]}
        )

    await charger_device.update_charger_for_user_change()
    charger_device.home.force_next_solve.assert_called_once()


async def test_charger_user_clean_constraints_and_group_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test user constraint cleanup and charger group power helpers."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_clean_constraints_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_clean_constraints_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.home.force_next_solve = MagicMock()
    charger_device.update_charger_for_user_change = AsyncMock()

    ct = SimpleNamespace(
        is_constraint_active_for_time_period=MagicMock(return_value=True),
        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
        load_param="car",
        load_info={"person": "Person A"},
        from_user=False,
    )
    charger_device._constraints = [ct]

    with patch(
        "custom_components.quiet_solar.ha_model.charger.AbstractLoad.user_clean_constraints",
        new=AsyncMock(),
    ):
        await charger_device.user_clean_constraints()

    assert charger_device._auto_constraints_cleaned_at_user_reset

    charger_device.father_device.get_average_power = MagicMock(return_value=50.0)
    assert charger_device.is_charger_group_power_zero(
        datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), for_duration=60
    )


async def test_charger_group_property_and_bump_solar(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger group property and bump solar setter."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_group_property_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_group_property_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.father_device = SimpleNamespace(
        home=charger_device.home,
        name="Parent",
        _childrens=[charger_device],
        charger_group=None,
        get_average_power=MagicMock(return_value=0.0),
    )

    group = charger_device.charger_group
    assert isinstance(group, QSChargerGroup)
    assert charger_device.father_device.charger_group is group

    ct = SimpleNamespace(is_before_battery=False)
    charger_device.get_current_active_constraint = MagicMock(return_value=ct)
    charger_device.car = SimpleNamespace(qs_bump_solar_charge_priority=False)
    charger_device.qs_bump_solar_charge_priority = True
    assert ct.is_before_battery is True


async def test_charger_group_budget_strategy() -> None:
    """Test QSChargerGroup apply_budget_strategy splitting."""
    dynamic_group = SimpleNamespace(
        home=SimpleNamespace(),
        name="Group C",
        _childrens=[],
        is_current_acceptable=MagicMock(return_value=False),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger = DummyCharger()
    charger.name = "Charger C"
    charger.mono_phase_index = 0
    charger.car = SimpleNamespace(name="Car C")

    cs_decrease = QSChargerStatus(charger)
    cs_decrease.current_real_max_charging_amp = 10
    cs_decrease.current_active_phase_number = 1
    cs_decrease.budgeted_amp = 6
    cs_decrease.budgeted_num_phases = 1

    cs_increase = QSChargerStatus(charger)
    cs_increase.current_real_max_charging_amp = 0
    cs_increase.current_active_phase_number = 1
    cs_increase.budgeted_amp = 6
    cs_increase.budgeted_num_phases = 1

    cs_phase = QSChargerStatus(charger)
    cs_phase.current_real_max_charging_amp = 16
    cs_phase.current_active_phase_number = 3
    cs_phase.budgeted_amp = 10
    cs_phase.budgeted_num_phases = 1

    actionable = [cs_decrease, cs_increase, cs_phase]
    group.apply_budgets = AsyncMock()

    await group.apply_budget_strategy(actionable, current_real_cars_power=1000.0, time=datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    group.apply_budgets.assert_awaited()


async def test_charger_group_update_and_shave_helpers() -> None:
    """Test group shaving helpers."""
    dynamic_group = SimpleNamespace(
        home=SimpleNamespace(),
        name="Group D",
        _childrens=[],
        dyn_group_max_phase_current=32,
        is_current_acceptable=MagicMock(return_value=False),
        is_current_acceptable_and_diff=MagicMock(return_value=(True, [2, 2, 2])),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger = DummyCharger()
    charger.name = "Charger D"
    charger.mono_phase_index = 0
    charger.min_charge = 6
    charger.max_charge = 32
    charger.update_amps_with_delta = MagicMock(return_value=[6, 0, 0])
    charger.get_delta_dampened_power = MagicMock(return_value=100.0)
    charger.car = SimpleNamespace(name="Car D")

    cs = QSChargerStatus(charger)
    cs.command = copy_command(CMD_AUTO_GREEN_CAP, power_consign=0)
    cs.possible_amps = [6, 10]
    cs.possible_num_phases = [1, 3]
    cs.budgeted_amp = 10
    cs.budgeted_num_phases = 1
    cs.current_real_max_charging_amp = 10
    cs.current_active_phase_number = 1
    cs.can_be_started_and_stopped = True
    cs.charge_score = 1

    await group._shave_mandatory_budgets([cs], [0, 0, 0], [10, 0, 0], datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    await group._shave_current_budgets([cs], datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
    group._update_and_prob_for_amps_reduction([0, 0, 0], [6, 0, 0], [0, 0, 0], datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))


async def test_charger_group_prepare_budgets_for_algo() -> None:
    """Test prepare budgets for algorithm."""
    dynamic_group = SimpleNamespace(
        home=SimpleNamespace(),
        name="Group E",
        _childrens=[],
        is_current_acceptable=MagicMock(return_value=True),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger = DummyCharger()
    charger.name = "Charger E"
    charger.mono_phase_index = 0
    charger.update_amps_with_delta = MagicMock(return_value=[0, 0, 0])
    charger.get_delta_dampened_power = MagicMock(return_value=0.0)
    charger.car = SimpleNamespace(name="Car E")

    cs = QSChargerStatus(charger)
    cs.possible_amps = [6, 10]
    cs.possible_num_phases = [1, 3]
    cs.current_real_max_charging_amp = 5
    cs.current_active_phase_number = 1

    await group._do_prepare_budgets_for_algo([cs], do_reset_allocation=False)
    await group._do_prepare_budgets_for_algo([cs], do_reset_allocation=True)


async def test_charger_group_prepare_and_shave_budgets() -> None:
    """Test _do_prepare_and_shave_budgets flow."""
    dynamic_group = SimpleNamespace(
        home=SimpleNamespace(),
        name="Group F",
        _childrens=[],
        is_current_acceptable=MagicMock(return_value=True),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger = DummyCharger()
    charger.name = "Charger F"
    charger.mono_phase_index = 0
    charger.update_amps_with_delta = MagicMock(return_value=[0, 0, 0])
    charger.get_delta_dampened_power = MagicMock(return_value=0.0)
    charger.car = SimpleNamespace(name="Car F")

    cs = QSChargerStatus(charger)
    cs.possible_amps = [6, 10]
    cs.possible_num_phases = [1]
    cs.current_real_max_charging_amp = 6
    cs.current_active_phase_number = 1

    group._shave_mandatory_budgets = AsyncMock(return_value=[0, 0, 0])
    group._shave_current_budgets = AsyncMock(return_value=([cs], True))

    await group._do_prepare_and_shave_budgets([cs], do_reset_allocation=False, time=datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))


async def test_charger_group_budgeting_algorithm_minimize_diffs() -> None:
    """Test budgeting algorithm decision flow."""
    home = SimpleNamespace(
        battery=None,
        battery_can_discharge=MagicMock(return_value=False),
        get_best_tariff=MagicMock(return_value=0.25),
        get_tariff=MagicMock(return_value=0.05),
    )
    dynamic_group = SimpleNamespace(
        home=home,
        name="Group G",
        _childrens=[],
        is_current_acceptable=MagicMock(return_value=True),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger_best = DummyCharger()
    charger_best.name = "Charger Best"
    charger_best.mono_phase_index = 0
    charger_best.min_charge = 6
    charger_best.max_charge = 32
    charger_best.qs_bump_solar_charge_priority = False
    charger_best._expected_charge_state = QSStateCmd()
    charger_best.get_delta_dampened_power = MagicMock(return_value=200.0)
    charger_best.car = SimpleNamespace(name="Car Best")

    charger_other = DummyCharger()
    charger_other.name = "Charger Other"
    charger_other.mono_phase_index = 0
    charger_other.min_charge = 6
    charger_other.max_charge = 32
    charger_other.qs_bump_solar_charge_priority = False
    charger_other._expected_charge_state = QSStateCmd()
    charger_other.get_delta_dampened_power = MagicMock(return_value=100.0)
    charger_other.car = SimpleNamespace(name="Car Other")

    cs_best = QSChargerStatus(charger_best)
    cs_best.command = copy_command(CMD_AUTO_GREEN_CONSIGN, power_consign=1500)
    cs_best.possible_amps = [0, 6]
    cs_best.possible_num_phases = [1]
    cs_best.current_real_max_charging_amp = 0
    cs_best.current_active_phase_number = 1
    cs_best.budgeted_amp = 0
    cs_best.budgeted_num_phases = 1
    cs_best.charge_score = 10
    cs_best.can_be_started_and_stopped = True
    cs_best.is_before_battery = True

    cs_other = QSChargerStatus(charger_other)
    cs_other.command = copy_command(CMD_AUTO_PRICE, power_consign=1000)
    cs_other.possible_amps = [6, 7]
    cs_other.possible_num_phases = [1]
    cs_other.current_real_max_charging_amp = 6
    cs_other.current_active_phase_number = 1
    cs_other.budgeted_amp = 6
    cs_other.budgeted_num_phases = 1
    cs_other.charge_score = 1
    cs_other.can_be_started_and_stopped = True
    cs_other.is_before_battery = False

    group._do_prepare_and_shave_budgets = AsyncMock(return_value=([cs_best, cs_other], True, False))

    success, should_reset, did_reset = await group.budgeting_algorithm_minimize_diffs(
        [cs_best, cs_other],
        full_available_home_power=2000.0,
        grid_available_home_power=1500.0,
        allow_budget_reset=True,
        time=datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC),
    )

    assert success is True
    assert should_reset is True
    assert did_reset is True


async def test_charger_group_dyn_handle_flow() -> None:
    """Test QSChargerGroup dyn_handle main flow."""
    now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    home = SimpleNamespace(
        battery=None,
        get_available_power_values=MagicMock(
            return_value=[
                (now - timedelta(seconds=30), 500.0),
                (now - timedelta(seconds=10), 500.0),
            ]
        ),
        get_grid_consumption_power_values=MagicMock(
            return_value=[
                (now - timedelta(seconds=30), 300.0),
                (now - timedelta(seconds=10), 300.0),
            ]
        ),
    )
    dynamic_group = SimpleNamespace(
        home=home,
        name="Group H",
        _childrens=[],
        accurate_power_sensor="sensor.accurate",
        get_median_sensor=MagicMock(return_value=800.0),
        is_current_acceptable=MagicMock(return_value=True),
    )
    group = QSChargerGroup(dynamic_group)

    class DummyCharger:
        pass

    charger = DummyCharger()
    charger.name = "Charger H"
    charger.min_charge = 6
    charger.qs_enable_device = True
    charger.car = SimpleNamespace(name="Car H")
    charger._expected_charge_state = QSStateCmd()
    charger._expected_charge_state.value = True
    charger.update_car_dampening_value = MagicMock()
    charger.is_charging_power_zero = MagicMock(return_value=False)
    charger.dampening_power_value_for_car_consumption = MagicMock(return_value=1.0)

    cs = QSChargerStatus(charger)
    cs.accurate_current_power = 600.0
    cs.current_real_max_charging_amp = 6
    cs.current_active_phase_number = 1
    cs.charge_score = 5
    cs.is_before_battery = True

    group._chargers = [charger]
    group.ensure_correct_state = AsyncMock(
        return_value=([cs], now - timedelta(seconds=CHARGER_ADAPTATION_WINDOW_S * 2))
    )
    group.budgeting_algorithm_minimize_diffs = AsyncMock(return_value=(True, False, False))
    group.apply_budget_strategy = AsyncMock()

    await group.dyn_handle(now)
    group.apply_budget_strategy.assert_awaited()


async def test_charger_device_power_helpers(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test QSChargerGeneric power helper methods."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_power_helpers_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_power_helpers_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_power_helpers_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_power_helpers_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device.charger_min_charge = 1
    charger_device.charger_max_charge = 3
    car_device.car_charger_min_charge = 1
    car_device.car_charger_max_charge = 3

    car_device.get_charge_power_per_phase_A = MagicMock(return_value=([0, 1000, 2000, 3000], 1, 3))

    assert charger_device.get_phase_amps_from_power(0, is_3p=False) == [0, 0, 0]
    assert charger_device.get_phase_amps_from_power(2000, is_3p=True) == [2, 2, 2]
    assert charger_device.get_phase_amps_from_power(1000, is_3p=False)[charger_device.mono_phase_index] == 1

    charger_device.is_charge_enabled = MagicMock(return_value=True)
    charger_device.get_charging_current = MagicMock(return_value=6)
    with patch.object(type(charger_device), "current_3p", new_callable=PropertyMock, return_value=False):
        amps = charger_device.get_device_amps_consumption(None, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
        assert amps[charger_device.mono_phase_index] == 6

    with patch.object(type(charger_device), "current_3p", new_callable=PropertyMock, return_value=True):
        amps = charger_device.get_device_amps_consumption(None, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC))
        assert amps == [6, 6, 6]

    steps = [0, 1000, 2000, 3000, 4000]
    assert charger_device._get_amps_from_power_steps(steps, 0, safe_border=False) == 0.0
    assert charger_device._get_amps_from_power_steps(steps, 1000, safe_border=True) == charger_device.min_charge

    charger_device.charger_consumption_W = 70
    assert charger_device.dampening_power_value_for_car_consumption(30.0) == 0.0
    assert charger_device.dampening_power_value_for_car_consumption(200.0) == 200.0

    now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    charger_device.is_charger_plugged_now = MagicMock(return_value=(True, now))
    state_time, state, _ = charger_device.is_plugged_state_getter("dummy", now)
    assert state_time == now
    assert state is not None

    charger_device.is_charger_plugged_now = MagicMock(return_value=(None, None))
    _, state, _ = charger_device.is_plugged_state_getter("dummy", now)
    assert state is None

    DummyPercent = type(MultiStepsPowerLoadConstraintChargePercent.__name__, (), {})
    DummyEnergy = type(MultiStepsPowerLoadConstraint.__name__, (), {})
    assert charger_device.get_update_value_callback_for_constraint_class(DummyPercent()) is not None
    assert charger_device.get_update_value_callback_for_constraint_class(DummyEnergy()) is not None


async def test_charger_device_charge_checks(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test charger charge state helpers."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_charge_checks_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_charge_checks_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_charge_checks_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_charge_checks_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device

    car_device.get_delta_dampened_power = MagicMock(return_value=50.0)
    assert charger_device.get_delta_dampened_power(6, 1, 10, 1) == 50.0

    now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    charger_device.is_car_stopped_asking_current = MagicMock(return_value=True)
    is_charged, result = charger_device.is_car_charged(
        now, current_charge=20.0, target_charge=50.0, is_target_percent=True
    )
    assert is_charged is True
    assert result == 50.0

    charger_device.is_car_stopped_asking_current = MagicMock(return_value=False)
    is_charged, result = charger_device.is_car_charged(
        now, current_charge=49.0, target_charge=50.0, is_target_percent=True, accept_bigger_tolerance=True
    )
    assert is_charged is True
    assert result == 50.0

    is_charged, result = charger_device.is_car_charged(
        now, current_charge=99.0, target_charge=100.0, is_target_percent=True, accept_bigger_tolerance=False
    )
    assert is_charged is False
    assert result == 99.0


async def test_charger_constraint_update_value_callback(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test constraint_update_value_callback_soc path."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_constraint_update_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_constraint_update_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_constraint_update_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_constraint_update_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device

    now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    charger_device._do_update_charger_state = AsyncMock()
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.current_command = copy_command(CMD_ON)
    charger_device._compute_added_charge_update = MagicMock(return_value=2.0)
    charger_device.is_car_stopped_asking_current = MagicMock(return_value=False)
    charger_device.is_charging_power_zero = MagicMock(return_value=True)
    charger_device.is_charger_group_power_zero = MagicMock(return_value=True)
    charger_device.on_device_state_change = AsyncMock()
    charger_device.possible_charge_error_start_time = None
    charger_device._expected_charge_state.value = True
    charger_device._expected_charge_state.last_ping_time_success = now - timedelta(seconds=3600)

    car_device.car_charge_percent_sensor = "sensor.car_soc"
    car_device.get_car_charge_percent = MagicMock(return_value=None)
    car_device.is_car_charge_growing = MagicMock(return_value=False)
    car_device.setup_car_charge_target_if_needed = AsyncMock()

    charger_device.father_device.charger_group = SimpleNamespace(dyn_handle=AsyncMock())

    class DummyConstraint:
        def __init__(self) -> None:
            self.current_value = 10.0
            self.target_value = 80.0
            self.first_value_update = now - timedelta(hours=1)
            self.last_value_update = now
            self.last_value_change_update = now - timedelta(minutes=10)

        def is_constraint_met(self, time: datetime, current_value: float) -> bool:
            return False

    ct = DummyConstraint()
    result, do_continue = await charger_device.constraint_update_value_callback_soc(
        ct, now, is_target_percent=True
    )
    assert result is not None
    assert do_continue is True
    charger_device.on_device_state_change.assert_awaited()


async def test_charger_constraint_update_value_callback_unplugged(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test constraint callback early returns."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_constraint_unplugged_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_constraint_unplugged_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    charger_device.car = None
    charger_device._do_update_charger_state = AsyncMock()

    class DummyConstraint:
        def __init__(self) -> None:
            self.current_value = 10.0
            self.target_value = 80.0
            self.first_value_update = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
            self.last_value_update = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
            self.last_value_change_update = datetime(2026, 1, 15, 8, 50, tzinfo=pytz.UTC)

        def is_constraint_met(self, time: datetime, current_value: float) -> bool:
            return False

    ct = DummyConstraint()
    result, cont = await charger_device.constraint_update_value_callback_soc(
        ct, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), is_target_percent=True
    )
    assert result is None
    assert cont is False

    charger_device.car = SimpleNamespace(
        name="Car X",
        get_car_charge_percent=MagicMock(return_value=20.0),
        is_car_charge_growing=MagicMock(return_value=True),
        setup_car_charge_target_if_needed=AsyncMock(),
        car_charge_percent_sensor="sensor.car",
    )
    charger_device.is_not_plugged = MagicMock(side_effect=[False, True])
    charger_device.current_command = copy_command(CMD_ON)
    result, cont = await charger_device.constraint_update_value_callback_soc(
        ct, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), is_target_percent=True
    )
    assert result is None
    assert cont is True


async def test_charger_constraint_update_value_callback_sensor_path(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test constraint callback sensor-based path."""
    from .const import MOCK_CHARGER_CONFIG, MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_constraint_sensor_test",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_constraint_sensor_test",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_constraint_sensor_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_constraint_sensor_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    charger_device = hass.data[DOMAIN].get(charger_entry.entry_id)
    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    charger_device.car = car_device
    charger_device._do_update_charger_state = AsyncMock()
    charger_device.is_not_plugged = MagicMock(return_value=False)
    charger_device.current_command = copy_command(CMD_ON)
    charger_device._compute_added_charge_update = MagicMock(return_value=5.0)

    car_device.get_car_charge_percent = MagicMock(return_value=90.0)
    car_device.is_car_charge_growing = MagicMock(return_value=None)
    car_device.setup_car_charge_target_if_needed = AsyncMock()
    car_device.car_charge_percent_sensor = "sensor.car"

    class DummyConstraint:
        def __init__(self) -> None:
            self.current_value = 80.0
            self.target_value = 91.0
            self.first_value_update = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
            self.last_value_update = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
            self.last_value_change_update = datetime(2026, 1, 15, 8, 30, tzinfo=pytz.UTC)

        def is_constraint_met(self, time: datetime, current_value: float) -> bool:
            return False

    ct = DummyConstraint()
    result, cont = await charger_device.constraint_update_value_callback_soc(
        ct, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC), is_target_percent=True
    )
    assert result is not None
    assert cont is True
