"""Tests for quiet_solar config flow."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from homeassistant import config_entries
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    UnitOfElectricCurrent,
    UnitOfPower,
)
from homeassistant.components.notify import DOMAIN as NOTIFY_DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.config_flow import (
    QSFlowHandler,
    QSOptionsFlowHandler,
    LOAD_TYPES_MENU,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DEVICE_TYPE,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_GRID_POWER_SENSOR,
    DATA_HANDLER,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS,
    CONF_CAR_ODOMETER_SENSOR,
    CONF_CAR_ESTIMATED_RANGE_SENSOR,
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES,
    CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P,
    CONF_DEVICE_DASHBOARD_SECTION,
    CONF_DEVICE_DYNAMIC_GROUP_NAME,
    CONF_MONO_PHASE,
    CONF_LOAD_IS_BOOST_ONLY,
    CONF_NUM_MAX_ON_OFF,
    CONF_POWER,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_ACCURATE_POWER_SENSOR,
    CONF_PHASE_1_AMPS_SENSOR,
    CONF_CALENDAR,
    CONF_PERSON_PERSON_ENTITY,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    CONF_CLIMATE,
    CONF_CLIMATE_HVAC_MODE_OFF,
    CONF_CLIMATE_HVAC_MODE_ON,
    CONF_DEVICE_TO_PILOT_NAME,
    CONF_PERSON_NOTIFICATION_TIME,
    CONF_TYPE_NAME_QSHeatPump,
    CONF_TYPE_NAME_QSClimateDuration,
    CONF_OFF_GRID_ENTITY,
    CONF_OFF_GRID_STATE_VALUE,
    CONF_OFF_GRID_INVERTED,
)
from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.battery import QSBattery
from custom_components.quiet_solar.ha_model.solar import QSSolar
from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.ha_model.heat_pump import QSHeatPump

from tests.factories import create_minimal_home_model


def _schema_has_key(schema, key: str) -> bool:
    return any(getattr(item, "schema", None) == key for item in schema.schema)


def _init_options_flow(hass: HomeAssistant, config_entry: MockConfigEntry) -> QSOptionsFlowHandler:
    flow = QSOptionsFlowHandler(config_entry)
    flow.hass = hass
    flow.handler = config_entry.entry_id
    hass.data.setdefault(DOMAIN, {})[config_entry.entry_id] = config_entry
    return flow


@pytest.fixture
def mock_data_handler(hass: HomeAssistant):
    """Provide mock data handler for config flow tests using real hass."""
    handler = MagicMock()
    handler.hass = hass
    handler.home = None
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = handler
    return handler


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Provide MockConfigEntry for config flow tests, added to hass."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_entry_123",
        data={CONF_NAME: "Test Device"},
        title="Test Device",
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_get_common_schema_includes_optional_fields(hass: HomeAssistant):
    """Test get_common_schema includes optional selectors."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="entry_1",
        data={CONF_NAME: "Device", CONF_DEVICE_DASHBOARD_SECTION: "Home"},
    )
    config_entry.add_to_hass(hass)
    flow = _init_options_flow(hass, config_entry)
    hass.config_entries.async_get_entry = MagicMock(return_value=config_entry)

    fake_home = create_minimal_home_model()
    fake_home.dashboard_sections = ["Home", "Garage"]
    root_group = QSHome.__new__(QSHome)
    root_group.name = "Root"
    other_group = MagicMock()
    other_group.name = "Group1"
    fake_home._all_dynamic_groups = [root_group, other_group]
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = MagicMock(home=fake_home)
    hass.services.async_register(NOTIFY_DOMAIN, "mobile_app", MagicMock())

    hass.states.async_set(
        "sensor.power",
        "10",
        {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT},
    )
    hass.states.async_set(
        "sensor.amps",
        "1",
        {ATTR_UNIT_OF_MEASUREMENT: UnitOfElectricCurrent.AMPERE},
    )
    hass.states.async_set("calendar.test", "off", {})

    with patch(
        "custom_components.quiet_solar.config_flow._filter_quiet_solar_entities",
        side_effect=lambda _h, entities: entities,
    ):
        schema = flow.get_common_schema(
            type=QSDynamicGroup.conf_type_name,
            add_power_value_selector=1000,
            add_load_power_sensor=True,
            add_load_power_sensor_mandatory=True,
            add_calendar=True,
            add_boost_only=True,
            add_mobile_app=True,
            add_efficiency_selector=True,
            add_is_3p=True,
            add_max_phase_amps_selector=16,
            add_power_group_selector=True,
            add_max_on_off=True,
            add_amps_sensors=True,
            add_phase_number=True,
        )

    key_names = {key.schema for key in schema}
    assert CONF_NAME in key_names
    assert CONF_DEVICE_DYNAMIC_GROUP_NAME in key_names
    assert CONF_IS_3P in key_names
    assert CONF_MONO_PHASE in key_names
    assert CONF_LOAD_IS_BOOST_ONLY in key_names
    assert CONF_NUM_MAX_ON_OFF in key_names
    assert CONF_POWER in key_names
    assert CONF_DYN_GROUP_MAX_PHASE_AMPS in key_names
    assert CONF_ACCURATE_POWER_SENSOR in key_names
    assert CONF_PHASE_1_AMPS_SENSOR in key_names
    assert CONF_CALENDAR in key_names


@pytest.mark.asyncio
async def test_flow_user_init_no_home(hass, mock_data_handler):
    """Test user flow when no home exists - should only show home option."""
    mock_data_handler.home = None
    
    flow = QSFlowHandler()
    flow.hass = hass
    
    result = await flow.async_step_user()
    
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "user"
    assert QSHome.conf_type_name in result["menu_options"]
    assert len(result["menu_options"]) == 1


@pytest.mark.asyncio
async def test_flow_user_init_with_home(hass, mock_data_handler):
    """Test user flow when home exists - should show all device types except home."""
    mock_home = create_minimal_home_model()
    mock_home._battery = None
    mock_home._solar_plant = None
    mock_data_handler.home = mock_home

    flow = QSFlowHandler()
    flow.hass = hass
    
    result = await flow.async_step_user()
    
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "user"
    assert QSHome.conf_type_name not in result["menu_options"]
    assert "charger" in result["menu_options"]  # Charger is a submenu
    assert QSCar.conf_type_name in result["menu_options"]


@pytest.mark.asyncio
async def test_flow_user_with_battery_installed(hass, mock_data_handler):
    """Test that battery option is hidden when battery already exists."""
    mock_home = create_minimal_home_model()
    mock_home._battery = MagicMock()  # Battery installed
    mock_home._solar_plant = None
    mock_data_handler.home = mock_home

    flow = QSFlowHandler()
    flow.hass = hass
    
    result = await flow.async_step_user()
    
    assert QSBattery.conf_type_name not in result["menu_options"]


@pytest.mark.asyncio
async def test_flow_user_with_solar_installed(hass, mock_data_handler):
    """Test that solar option is hidden when solar already exists."""
    mock_home = create_minimal_home_model()
    mock_home._battery = None
    mock_home._solar_plant = MagicMock()  # Solar installed
    mock_data_handler.home = mock_home

    flow = QSFlowHandler()
    flow.hass = hass
    
    result = await flow.async_step_user()
    
    assert QSSolar.conf_type_name not in result["menu_options"]


@pytest.mark.asyncio
async def test_flow_home_step_shows_form(hass):
    """Test home configuration step shows form."""
    flow = QSFlowHandler()
    flow.hass = hass
    
    result = await flow.async_step_home()
    
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == QSHome.conf_type_name


@pytest.mark.asyncio
async def test_flow_home_step_creates_entry(hass):
    """Test home configuration creates entry with correct data."""
    flow = QSFlowHandler()
    flow.hass = hass
    
    user_input = {
        CONF_NAME: "My Home",
        CONF_HOME_VOLTAGE: 230,
        CONF_IS_3P: True,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_home(user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "home: My Home"
    assert result["data"][CONF_NAME] == "My Home"
    assert result["data"][CONF_HOME_VOLTAGE] == 230
    assert result["data"][DEVICE_TYPE] == QSHome.conf_type_name


@pytest.mark.asyncio
async def test_flow_home_step_schema_includes_off_grid_fields(hass):
    """Test home configuration schema includes off-grid detection fields."""
    flow = QSFlowHandler()
    flow.hass = hass

    result = await flow.async_step_home()

    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_OFF_GRID_ENTITY)
    assert _schema_has_key(schema, CONF_OFF_GRID_STATE_VALUE)
    assert _schema_has_key(schema, CONF_OFF_GRID_INVERTED)


@pytest.mark.asyncio
async def test_flow_home_step_creates_entry_with_off_grid_entity(hass):
    """Test home configuration stores off-grid entity data."""
    flow = QSFlowHandler()
    flow.hass = hass

    user_input = {
        CONF_NAME: "My Home",
        CONF_HOME_VOLTAGE: 230,
        CONF_IS_3P: True,
        CONF_OFF_GRID_ENTITY: "binary_sensor.grid_relay",
        CONF_OFF_GRID_STATE_VALUE: "",
        CONF_OFF_GRID_INVERTED: True,
    }

    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_home(user_input)

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_OFF_GRID_ENTITY] == "binary_sensor.grid_relay"
    assert result["data"][CONF_OFF_GRID_INVERTED] is True


@pytest.mark.asyncio
async def test_flow_home_step_creates_entry_without_off_grid_entity(hass):
    """Test home configuration works without off-grid entity (optional)."""
    flow = QSFlowHandler()
    flow.hass = hass

    user_input = {
        CONF_NAME: "My Home",
        CONF_HOME_VOLTAGE: 230,
        CONF_IS_3P: True,
    }

    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_home(user_input)

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert CONF_OFF_GRID_ENTITY not in result["data"]


@pytest.mark.asyncio
async def test_flow_charger_menu(hass):
    """Test charger submenu shows charger types."""
    flow = QSFlowHandler()
    flow.hass = hass
    
    result = await flow.async_step_charger()
    
    assert result["type"] == FlowResultType.MENU
    assert result["step_id"] == "charger"
    assert len(result["menu_options"]) > 0


@pytest.mark.asyncio
async def test_flow_charger_generic_creates_entry(hass, mock_data_handler):
    """Test generic charger configuration creates entry."""
    mock_data_handler.home = create_minimal_home_model()
    mock_data_handler.home._all_dynamic_groups = [MagicMock()]

    flow = QSFlowHandler()
    flow.hass = hass
    
    user_input = {
        CONF_NAME: "Test Charger",
        CONF_CHARGER_MIN_CHARGE: 6,
        CONF_CHARGER_MAX_CHARGE: 16,
        CONF_IS_3P: False,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_charger_generic(user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_NAME] == "Test Charger"
    assert result["data"][CONF_CHARGER_MIN_CHARGE] == 6


@pytest.mark.asyncio
async def test_flow_car_creates_entry(hass, mock_data_handler):
    """Test car configuration creates entry."""
    mock_data_handler.home = create_minimal_home_model()

    flow = QSFlowHandler()
    flow.hass = hass
    
    user_input = {
        CONF_NAME: "Test Car",
        CONF_CAR_BATTERY_CAPACITY: 50000,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock):
        result = await flow.async_step_car(user_input)
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_NAME] == "Test Car"
    assert result["data"][CONF_CAR_BATTERY_CAPACITY] == 50000


@pytest.mark.asyncio
async def test_flow_cleans_none_values(hass):
    """Test that None values are cleaned from data."""
    flow = QSFlowHandler()
    flow.hass = hass
    
    data = {
        CONF_NAME: "Test",
        CONF_GRID_POWER_SENSOR: None,
        CONF_IS_3P: True,
    }
    
    flow.clean_data(data)
    
    assert CONF_NAME in data
    assert CONF_IS_3P in data
    assert CONF_GRID_POWER_SENSOR not in data


@pytest.mark.asyncio
async def test_options_flow_is_creation_flow(hass, mock_config_entry):
    """Test options flow is_creation_flow returns False."""
    from awesomeversion import AwesomeVersion
    from homeassistant.const import __version__ as HAVERSION
    from custom_components.quiet_solar.config_flow import HA_OPTIONS_FLOW_VERSION_THRESHOLD
    
    # Set up the handler property for newer HA versions
    flow = QSOptionsFlowHandler(mock_config_entry)
    
    # For newer HA versions, we need to set the handler manually
    if AwesomeVersion(HAVERSION) >= HA_OPTIONS_FLOW_VERSION_THRESHOLD:
        flow.handler = mock_config_entry.entry_id
        # Register the entry in hass so async_get_entry can find it
        hass.data.setdefault(DOMAIN, {})[mock_config_entry.entry_id] = mock_config_entry
    
    flow.hass = hass
    
    # Test the method directly
    assert flow.is_creation_flow() is False


@pytest.mark.asyncio
async def test_get_entry_title_formats_correctly(hass):
    """Test entry title formatting."""
    flow = QSFlowHandler()
    flow.hass = hass
    
    data = {
        CONF_NAME: "My Device",
        DEVICE_TYPE: QSHome.conf_type_name
    }
    
    title = flow.get_entry_title(data)
    
    assert "My Device" in title
    assert "home" in title.lower()


@pytest.mark.asyncio
async def test_flow_unique_id_set(hass):
    """Test that unique ID is set for new entries."""
    flow = QSFlowHandler()
    flow.hass = hass
    
    user_input = {
        CONF_NAME: "Unique Device",
        CONF_IS_3P: True,
    }
    
    with patch.object(flow, 'async_set_unique_id', new_callable=AsyncMock) as mock_set_unique:
        with patch.object(flow, 'async_create_entry', return_value={"type": FlowResultType.CREATE_ENTRY}):
            await flow._async_entry_next({**user_input, DEVICE_TYPE: QSHome.conf_type_name})
    
    mock_set_unique.assert_called_once()
    call_arg = mock_set_unique.call_args[0][0]
    assert "Unique Device" in call_arg
    assert QSHome.conf_type_name in call_arg


@pytest.mark.asyncio
async def test_options_flow_car_includes_percent_and_length_selectors(
    hass: HomeAssistant,
):
    """Test car options include percent and length selectors when available."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_car_percent_123",
        data={
            CONF_NAME: "Test Car",
            DEVICE_TYPE: QSCar.conf_type_name,
            CONF_CAR_CHARGER_MIN_CHARGE: 6,
            CONF_CAR_CHARGER_MAX_CHARGE: 16,
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: None,
        },
        title="car: Test Car",
    )
    config_entry.add_to_hass(hass)

    hass.states.async_set(
        "sensor.test_percent",
        "50",
        {"unit_of_measurement": "%"},
    )
    hass.states.async_set(
        "number.test_percent",
        "80",
        {"unit_of_measurement": "%"},
    )
    hass.states.async_set(
        "sensor.test_length",
        "120",
        {"unit_of_measurement": "km"},
    )

    flow = _init_options_flow(hass, config_entry)

    with patch(
        "custom_components.quiet_solar.config_flow._filter_quiet_solar_entities",
        side_effect=lambda _hass, entities: entities,
    ):
        result = await flow.async_step_car()
    assert result["type"] == FlowResultType.FORM

    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_CAR_CHARGE_PERCENT_SENSOR)
    assert _schema_has_key(schema, CONF_CAR_CHARGE_PERCENT_MAX_NUMBER)
    assert _schema_has_key(schema, CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS)
    assert _schema_has_key(schema, CONF_CAR_ODOMETER_SENSOR)
    assert _schema_has_key(schema, CONF_CAR_ESTIMATED_RANGE_SENSOR)


@pytest.mark.asyncio
async def test_options_flow_car_steps_schema(hass: HomeAssistant):
    """Test car options include steps when configured."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_car_steps_123",
        data={
            CONF_NAME: "Test Car",
            DEVICE_TYPE: QSCar.conf_type_name,
            CONF_CAR_CHARGER_MIN_CHARGE: 6,
            CONF_CAR_CHARGER_MAX_CHARGE: 16,
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "5",
        },
        title="car: Test Car",
    )
    config_entry.add_to_hass(hass)

    hass.states.async_set(
        "number.test_percent",
        "80",
        {"unit_of_measurement": "%"},
    )

    flow = _init_options_flow(hass, config_entry)

    with patch(
        "custom_components.quiet_solar.config_flow._filter_quiet_solar_entities",
        side_effect=lambda _hass, entities: entities,
    ):
        result = await flow.async_step_car()

    assert result["type"] == FlowResultType.FORM
    assert _schema_has_key(result["data_schema"], CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS)


@pytest.mark.asyncio
async def test_options_flow_car_force_dampening_fields(hass: HomeAssistant):
    """Test car options add dampening fields when enabled."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_car_damp_123",
        data={
            CONF_NAME: "Test Car",
            DEVICE_TYPE: QSCar.conf_type_name,
            CONF_CAR_CHARGER_MIN_CHARGE: 6,
            CONF_CAR_CHARGER_MAX_CHARGE: 8,
            CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: False,
        },
        title="car: Test Car",
    )
    config_entry.add_to_hass(hass)

    flow = _init_options_flow(hass, config_entry)

    with patch(
        "custom_components.quiet_solar.config_flow._filter_quiet_solar_entities",
        side_effect=lambda _hass, entities: entities,
    ):
        result = await flow.async_step_car(
            {
                CONF_CAR_CHARGER_MIN_CHARGE: 6,
                CONF_CAR_CHARGER_MAX_CHARGE: 8,
                CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert _schema_has_key(result["data_schema"], CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P)
    assert _schema_has_key(result["data_schema"], "charge_6")
    assert _schema_has_key(result["data_schema"], "charge_8")


@pytest.mark.asyncio
async def test_options_flow_person_aborts_without_person_entities(
    hass: HomeAssistant,
):
    """Test person options aborts when no person entities exist."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_person_abort_123",
        data={
            CONF_NAME: "Test Person",
            DEVICE_TYPE: QSPerson.conf_type_name,
        },
        title="person: Test Person",
    )
    config_entry.add_to_hass(hass)

    flow = _init_options_flow(hass, config_entry)

    result = await flow.async_step_person()
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "no_person_entities"


@pytest.mark.asyncio
async def test_options_flow_person_includes_authorized_cars(
    hass: HomeAssistant,
    mock_data_handler,
):
    """Test person options include authorized cars selector."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_person_cars_123",
        data={
            CONF_NAME: "Test Person",
            DEVICE_TYPE: QSPerson.conf_type_name,
            CONF_PERSON_PREFERRED_CAR: "Car A",
        },
        title="person: Test Person",
    )
    config_entry.add_to_hass(hass)

    hass.states.async_set("person.test_person", "home", {})

    car_regular = MagicMock()
    car_regular.name = "Car A"
    car_regular.car_is_invited = False
    car_invited = MagicMock()
    car_invited.name = "Car B"
    car_invited.car_is_invited = True

    mock_home = create_minimal_home_model()
    mock_home._cars = [car_regular, car_invited]
    mock_data_handler.home = mock_home

    flow = _init_options_flow(hass, config_entry)

    result = await flow.async_step_person()
    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_PERSON_PERSON_ENTITY)
    assert _schema_has_key(schema, CONF_PERSON_AUTHORIZED_CARS)


# =============================================================================
# Heat Pump Config Flow Tests
# =============================================================================


@pytest.mark.asyncio
async def test_flow_heat_pump_step_shows_form(hass: HomeAssistant, mock_data_handler):
    """Test heat pump configuration step shows form with expected fields."""
    mock_data_handler.home = create_minimal_home_model()
    mock_data_handler.home._all_dynamic_groups = [MagicMock()]

    flow = QSFlowHandler()
    flow.hass = hass

    result = await flow.async_step_heat_pump()

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == QSHeatPump.conf_type_name
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_NAME)
    assert _schema_has_key(schema, CONF_POWER)
    assert _schema_has_key(schema, CONF_IS_3P)
    assert _schema_has_key(schema, CONF_NUM_MAX_ON_OFF)


@pytest.mark.asyncio
async def test_flow_heat_pump_step_creates_entry(hass: HomeAssistant, mock_data_handler):
    """Test heat pump configuration creates entry with correct data."""
    mock_data_handler.home = create_minimal_home_model()
    mock_data_handler.home._all_dynamic_groups = [MagicMock()]

    flow = QSFlowHandler()
    flow.hass = hass

    user_input = {
        CONF_NAME: "My Heat Pump",
        CONF_POWER: 2000,
        CONF_IS_3P: True,
    }

    with patch.object(flow, "async_set_unique_id", new_callable=AsyncMock):
        result = await flow.async_step_heat_pump(user_input)

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_NAME] == "My Heat Pump"
    assert result["data"][CONF_POWER] == 2000
    assert result["data"][DEVICE_TYPE] == QSHeatPump.conf_type_name


@pytest.mark.asyncio
async def test_options_flow_heat_pump_shows_form(hass: HomeAssistant, mock_data_handler):
    """Test heat pump options flow shows form with existing data."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_heat_pump_opt_123",
        data={
            CONF_NAME: "Test Heat Pump",
            DEVICE_TYPE: CONF_TYPE_NAME_QSHeatPump,
            CONF_POWER: 3000,
            CONF_IS_3P: False,
        },
        title="heat_pump: Test Heat Pump",
    )
    config_entry.add_to_hass(hass)

    mock_data_handler.home = create_minimal_home_model()
    mock_data_handler.home._all_dynamic_groups = [MagicMock()]

    flow = _init_options_flow(hass, config_entry)

    result = await flow.async_step_heat_pump()

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == CONF_TYPE_NAME_QSHeatPump
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_NAME)
    assert _schema_has_key(schema, CONF_POWER)


@pytest.mark.asyncio
async def test_options_flow_heat_pump_updates_entry(hass: HomeAssistant, mock_data_handler):
    """Test heat pump options flow updates entry with new data."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_heat_pump_upd_123",
        data={
            CONF_NAME: "Test Heat Pump",
            DEVICE_TYPE: CONF_TYPE_NAME_QSHeatPump,
            CONF_POWER: 2000,
        },
        title="heat_pump: Test Heat Pump",
    )
    config_entry.add_to_hass(hass)

    mock_data_handler.home = create_minimal_home_model()
    mock_data_handler.home._all_dynamic_groups = [MagicMock()]

    flow = _init_options_flow(hass, config_entry)

    user_input = {
        CONF_NAME: "Updated Heat Pump",
        CONF_POWER: 4000,
        CONF_IS_3P: True,
    }

    with patch(
        "custom_components.quiet_solar.config_flow.async_reload_quiet_solar",
        new_callable=AsyncMock,
    ):
        result = await flow.async_step_heat_pump(user_input)

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert config_entry.data[CONF_NAME] == "Updated Heat Pump"
    assert config_entry.data[CONF_POWER] == 4000


# =============================================================================
# Climate with Heat Pump Options Tests
# =============================================================================


@pytest.mark.asyncio
async def test_climate_step_with_heat_pumps_and_default(
    hass: HomeAssistant, mock_data_handler
):
    """Test climate step shows heat pump selector with default value."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_climate_hp_default_123",
        data={
            CONF_NAME: "Test Climate",
            DEVICE_TYPE: CONF_TYPE_NAME_QSClimateDuration,
            CONF_POWER: 1000,
            CONF_CLIMATE: "climate.living_room",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
            CONF_DEVICE_TO_PILOT_NAME: "My Heat Pump",
        },
        title="climate: Test Climate",
    )
    config_entry.add_to_hass(hass)

    mock_hp = MagicMock()
    mock_hp.name = "My Heat Pump"
    mock_home = create_minimal_home_model()
    mock_home._heat_pumps = [mock_hp]
    mock_data_handler.home = mock_home

    flow = _init_options_flow(hass, config_entry)

    with patch(
        "custom_components.quiet_solar.config_flow.get_hvac_modes",
        return_value=["off", "heat"],
    ):
        result = await flow.async_step_climate()

    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_DEVICE_TO_PILOT_NAME)


@pytest.mark.asyncio
async def test_climate_step_with_heat_pumps_no_default(
    hass: HomeAssistant, mock_data_handler
):
    """Test climate step shows heat pump selector without default value."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_climate_hp_nodef_123",
        data={
            CONF_NAME: "Test Climate",
            DEVICE_TYPE: CONF_TYPE_NAME_QSClimateDuration,
            CONF_POWER: 1000,
            CONF_CLIMATE: "climate.living_room",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
        },
        title="climate: Test Climate",
    )
    config_entry.add_to_hass(hass)

    mock_hp1 = MagicMock()
    mock_hp1.name = "Heat Pump A"
    mock_hp2 = MagicMock()
    mock_hp2.name = "Heat Pump B"
    mock_home = create_minimal_home_model()
    mock_home._heat_pumps = [mock_hp1, mock_hp2]
    mock_data_handler.home = mock_home

    flow = _init_options_flow(hass, config_entry)

    with patch(
        "custom_components.quiet_solar.config_flow.get_hvac_modes",
        return_value=["off", "heat"],
    ):
        result = await flow.async_step_climate()

    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_DEVICE_TO_PILOT_NAME)


@pytest.mark.asyncio
async def test_climate_step_without_heat_pumps(
    hass: HomeAssistant, mock_data_handler
):
    """Test climate step works when no heat pumps are available."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_climate_no_hp_123",
        data={
            CONF_NAME: "Test Climate",
            DEVICE_TYPE: CONF_TYPE_NAME_QSClimateDuration,
            CONF_POWER: 1000,
            CONF_CLIMATE: "climate.living_room",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
        },
        title="climate: Test Climate",
    )
    config_entry.add_to_hass(hass)

    mock_home = create_minimal_home_model()
    mock_home._heat_pumps = []
    mock_data_handler.home = mock_home

    flow = _init_options_flow(hass, config_entry)

    with patch(
        "custom_components.quiet_solar.config_flow.get_hvac_modes",
        return_value=["off", "heat"],
    ):
        result = await flow.async_step_climate()

    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"]
    # No heat pump selector should be present
    assert not _schema_has_key(schema, CONF_DEVICE_TO_PILOT_NAME)


@pytest.mark.asyncio
async def test_climate_step_with_heat_pump_user_input(
    hass: HomeAssistant, mock_data_handler
):
    """Test climate step creates entry with heat pump selection."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_climate_hp_input_123",
        data={
            CONF_NAME: "Test Climate",
            DEVICE_TYPE: CONF_TYPE_NAME_QSClimateDuration,
            CONF_POWER: 1000,
            CONF_CLIMATE: "climate.living_room",
            CONF_CLIMATE_HVAC_MODE_OFF: "off",
            CONF_CLIMATE_HVAC_MODE_ON: "heat",
        },
        title="climate: Test Climate",
    )
    config_entry.add_to_hass(hass)

    mock_hp = MagicMock()
    mock_hp.name = "My Heat Pump"
    mock_home = create_minimal_home_model()
    mock_home._heat_pumps = [mock_hp]
    mock_data_handler.home = mock_home

    flow = _init_options_flow(hass, config_entry)

    user_input = {
        CONF_NAME: "Test Climate",
        CONF_POWER: 1000,
        CONF_CLIMATE: "climate.living_room",
        CONF_CLIMATE_HVAC_MODE_OFF: "off",
        CONF_CLIMATE_HVAC_MODE_ON: "heat",
        CONF_DEVICE_TO_PILOT_NAME: "My Heat Pump",
    }

    with patch(
        "custom_components.quiet_solar.config_flow.async_reload_quiet_solar",
        new_callable=AsyncMock,
    ):
        result = await flow.async_step_climate(user_input)

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert config_entry.data[CONF_DEVICE_TO_PILOT_NAME] == "My Heat Pump"


# =============================================================================
# Person notification time with existing default
# =============================================================================


@pytest.mark.asyncio
async def test_options_flow_person_notification_time_with_default(
    hass: HomeAssistant,
    mock_data_handler,
):
    """Test person options show notification time with existing default."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_person_notif_123",
        data={
            CONF_NAME: "Test Person",
            DEVICE_TYPE: QSPerson.conf_type_name,
            CONF_PERSON_NOTIFICATION_TIME: "08:30:00",
        },
        title="person: Test Person",
    )
    config_entry.add_to_hass(hass)

    hass.states.async_set("person.test_person", "home", {})

    mock_home = create_minimal_home_model()
    mock_home._cars = []
    mock_data_handler.home = mock_home

    flow = _init_options_flow(hass, config_entry)

    result = await flow.async_step_person()
    assert result["type"] == FlowResultType.FORM
    schema = result["data_schema"]
    assert _schema_has_key(schema, CONF_PERSON_NOTIFICATION_TIME)


# =============================================================================
# QSFlowHandler.is_creation_flow
# =============================================================================


@pytest.mark.asyncio
async def test_flow_handler_is_creation_flow(hass: HomeAssistant):
    """Test QSFlowHandler.is_creation_flow returns True."""
    flow = QSFlowHandler()
    flow.hass = hass
    assert flow.is_creation_flow() is True
