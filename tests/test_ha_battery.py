"""Tests for QSBattery class in ha_model/battery.py."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    SERVICE_TURN_ON,
    SERVICE_TURN_OFF,
    ATTR_ENTITY_ID,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.ha_model.battery import QSBattery
from custom_components.quiet_solar.home_model.commands import (
    LoadCommand,
    CMD_ON,
    CMD_IDLE,
    CMD_AUTO_GREEN_ONLY,
    CMD_GREEN_CHARGE_AND_DISCHARGE,
    CMD_GREEN_CHARGE_ONLY,
    CMD_FORCE_CHARGE,
    copy_command,
)

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR,
    CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER,
    CONF_BATTERY_MAX_CHARGE_POWER_NUMBER,
    CONF_BATTERY_CHARGE_PERCENT_SENSOR,
    CONF_BATTERY_CHARGE_FROM_GRID_SWITCH,
    CONF_BATTERY_IS_DC_COUPLED,
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    DOMAIN,
    DATA_HANDLER,
)

from tests.factories import create_minimal_home_model


async def _async_set_state(hass, entity_id: str, state: str, attributes: dict | None = None):
    """Set state; await only if async_set returns a coroutine (real HA)."""
    result = hass.states.async_set(entity_id, state, attributes)
    if result is not None:
        await result


@pytest.fixture
def battery_config_entry() -> MockConfigEntry:
    """Config entry for battery tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_battery_entry",
        data={CONF_NAME: "Test Battery"},
        title="battery: Test Battery",
    )


@pytest.fixture
def battery_home():
    """Home for battery tests; add get_current_over_clamp_production_power where needed."""
    home = create_minimal_home_model()
    home.get_current_over_clamp_production_power = MagicMock(return_value=0.0)
    return home


@pytest.fixture
def battery_data_handler(battery_home):
    """Data handler for battery tests."""
    handler = MagicMock()
    handler.home = battery_home
    return handler


@pytest.fixture
def battery_hass_data(hass: HomeAssistant, battery_data_handler):
    """Set hass.data[DOMAIN][DATA_HANDLER] for battery tests."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = battery_data_handler


@pytest.fixture
def recorded_service_calls(hass: HomeAssistant):
    """Record service calls (domain, service, service_data) for assertions."""
    from homeassistant.core import ServiceRegistry

    recorded = []

    async def record_only(self, domain, service, service_data=None, **kwargs):
        recorded.append((domain, service, service_data or {}))

    with patch.object(ServiceRegistry, "async_call", record_only):
        yield recorded


class TestQSBatteryInit:
    """Test QSBattery initialization."""

    def test_init_with_all_sensors(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test initialization with all sensor configurations."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_PERCENT_SENSOR: "sensor.battery_soc",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

        assert battery.charge_discharge_sensor == "sensor.battery_power"
        assert battery.max_discharge_number == "number.max_discharge"
        assert battery.max_charge_number == "number.max_charge"
        assert battery.charge_percent_sensor == "sensor.battery_soc"
        assert battery.charge_from_grid_switch == "switch.charge_from_grid"
        assert battery.is_dc_coupled is True
        assert battery.capacity == 10000

    def test_init_with_minimal_sensors(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test initialization with minimal sensor configurations."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_CAPACITY: 7000,
            },
        )

        assert battery.charge_discharge_sensor == "sensor.battery_power"
        assert battery.max_discharge_number is None
        assert battery.max_charge_number is None
        assert battery.charge_percent_sensor is None
        assert battery.charge_from_grid_switch is None
        assert battery.is_dc_coupled is False

    def test_init_without_optional_sensors(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test initialization without any optional sensors."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
            },
        )

        assert battery.charge_discharge_sensor is None
        assert battery.max_discharge_number is None
        assert battery.max_charge_number is None


class TestQSBatteryCurrentCharge:
    """Test QSBattery current_charge property."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance with charge percent sensor and capacity."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_PERCENT_SENSOR: "sensor.battery_soc",
                CONF_BATTERY_CAPACITY: 10000,
            },
        )

    def test_current_charge_normal(self, battery):
        """Test current_charge with normal value."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)

        result = battery.current_charge

        assert result == 5000.0

    def test_current_charge_none(self, battery):
        """Test current_charge when sensor returns None."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        result = battery.current_charge

        assert result is None

    def test_current_charge_zero_percent(self, battery):
        """Test current_charge at 0%."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=0.0)

        result = battery.current_charge

        assert result == 0.0

    def test_current_charge_full(self, battery):
        """Test current_charge at 100%."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=100.0)

        result = battery.current_charge

        assert result == 10000.0


class TestQSBatteryCommandToValues:
    """Test QSBattery _command_to_values method."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance with number/switch entities."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

    def test_command_to_values_cmd_on(self, battery):
        """Test _command_to_values with CMD_ON."""
        result = battery._command_to_values(CMD_ON)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_idle(self, battery):
        """Test _command_to_values with CMD_IDLE."""
        result = battery._command_to_values(CMD_IDLE)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_auto_green_only(self, battery):
        """Test _command_to_values with CMD_AUTO_GREEN_ONLY."""
        result = battery._command_to_values(CMD_AUTO_GREEN_ONLY)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_green_charge_and_discharge(self, battery):
        """Test _command_to_values with CMD_GREEN_CHARGE_AND_DISCHARGE."""
        result = battery._command_to_values(CMD_GREEN_CHARGE_AND_DISCHARGE)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_green_charge_only(self, battery):
        """Test _command_to_values with CMD_GREEN_CHARGE_ONLY."""
        result = battery._command_to_values(CMD_GREEN_CHARGE_ONLY)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 0
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_force_charge(self, battery):
        """Test _command_to_values with CMD_FORCE_CHARGE."""
        command = copy_command(CMD_FORCE_CHARGE, power_consign=3000)
        result = battery._command_to_values(command)

        assert result["charge_from_grid"] is True
        assert result["max_discharging_power"] == 0
        assert result["max_charging_power"] == 3000

    def test_command_to_values_invalid_command(self, battery):
        """Test _command_to_values with invalid command."""
        invalid_command = LoadCommand(command="invalid", power_consign=0.0)

        with pytest.raises(ValueError, match="Invalid command"):
            battery._command_to_values(invalid_command)

    def test_command_to_values_without_optional_entities(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test _command_to_values when optional entities are None."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery Minimal",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

        result = battery._command_to_values(CMD_ON)

        assert result["charge_from_grid"] is None
        assert result["max_discharging_power"] is None
        assert result["max_charging_power"] is None


class TestQSBatteryExecuteCommand:
    """Test QSBattery execute_command method."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance for execute_command tests."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

    @pytest.mark.asyncio
    async def test_execute_command_success(self, hass, battery):
        """Test successful command execution."""
        await _async_set_state(hass,"number.max_discharge", "5000")
        await _async_set_state(hass,"number.max_charge", "5000")
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        time = datetime.now(pytz.UTC)

        result = await battery.execute_command(time, CMD_ON)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_command_green_charge_only(
        self, hass, battery, recorded_service_calls
    ):
        """Test execute_command with green charge only."""
        await _async_set_state(hass,"number.max_discharge", "5000")
        await _async_set_state(hass,"number.max_charge", "5000")
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        time = datetime.now(pytz.UTC)

        result = await battery.execute_command(time, CMD_GREEN_CHARGE_ONLY)

        assert result is False
        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert any(
            c[2].get("value") == 0
            for c in calls
            if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        )

    @pytest.mark.asyncio
    async def test_execute_command_force_charge(
        self, hass, battery, recorded_service_calls
    ):
        """Test execute_command with force charge."""
        await _async_set_state(hass,"number.max_discharge", "5000")
        await _async_set_state(hass,"number.max_charge", "5000")
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        time = datetime.now(pytz.UTC)
        command = copy_command(CMD_FORCE_CHARGE, power_consign=3000)

        result = await battery.execute_command(time, command)

        assert result is False
        calls = [c for c in recorded_service_calls if c[1] == SERVICE_TURN_ON]
        assert len(calls) >= 1


class TestQSBatteryProbeIfCommandSet:
    """Test QSBattery probe_if_command_set method."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance for probe tests."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

    @pytest.mark.asyncio
    async def test_probe_command_matches(self, hass, battery):
        """Test probe_if_command_set when command matches current state."""
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        await _async_set_state(hass,"number.max_discharge", "5000")
        await _async_set_state(hass,"number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is True

    @pytest.mark.asyncio
    async def test_probe_command_does_not_match(self, hass, battery):
        """Test probe_if_command_set when command doesn't match."""
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        await _async_set_state(hass,"number.max_discharge", "5000")
        await _async_set_state(hass,"number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_GREEN_CHARGE_ONLY)

        assert result is False

    @pytest.mark.asyncio
    async def test_probe_switch_unavailable(self, hass, battery):
        """Test probe_if_command_set when switch is unavailable."""
        await _async_set_state(hass,"switch.charge_from_grid", STATE_UNAVAILABLE)
        await _async_set_state(hass,"number.max_discharge", "5000")
        await _async_set_state(hass,"number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_discharge_number_unavailable(self, hass, battery):
        """Test probe_if_command_set when discharge number is unavailable."""
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        await _async_set_state(hass,"number.max_discharge", STATE_UNKNOWN)
        await _async_set_state(hass,"number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is None


class TestQSBatteryGridCharging:
    """Test QSBattery grid charging methods."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance with charge_from_grid switch."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
            },
        )

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_enable(
        self, hass, battery, recorded_service_calls
    ):
        """Test enabling grid charging."""
        await _async_set_state(hass,"switch.charge_from_grid", "off")
        battery.is_charge_from_grid_current = False

        await battery.set_charge_from_grid(True)

        calls = [c for c in recorded_service_calls if c[1] == SERVICE_TURN_ON]
        assert len(calls) == 1
        assert calls[0][0] == Platform.SWITCH

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_disable(
        self, hass, battery, recorded_service_calls
    ):
        """Test disabling grid charging."""
        await _async_set_state(hass,"switch.charge_from_grid", "on")
        battery.is_charge_from_grid_current = True

        await battery.set_charge_from_grid(False)

        calls = [c for c in recorded_service_calls if c[1] == SERVICE_TURN_OFF]
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_no_change(self, battery, recorded_service_calls):
        """Test set_charge_from_grid when value already set."""
        battery.is_charge_from_grid_current = True

        await battery.set_charge_from_grid(True)

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_none_switch(self, battery, recorded_service_calls):
        """Test set_charge_from_grid when switch is None."""
        battery.charge_from_grid_switch = None

        await battery.set_charge_from_grid(True)

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_enabled(self, hass, battery):
        """Test is_charge_from_grid returns True when switch is on."""
        await _async_set_state(hass,"switch.charge_from_grid", "on")

        result = await battery.is_charge_from_grid()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_disabled(self, hass, battery):
        """Test is_charge_from_grid returns False when switch is off."""
        await _async_set_state(hass,"switch.charge_from_grid", "off")

        result = await battery.is_charge_from_grid()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_unavailable(self, hass, battery):
        """Test is_charge_from_grid returns None when switch is unavailable."""
        await _async_set_state(hass,"switch.charge_from_grid", STATE_UNAVAILABLE)

        result = await battery.is_charge_from_grid()

        assert result is None

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_no_switch(self, battery):
        """Test is_charge_from_grid returns None when no switch configured."""
        battery.charge_from_grid_switch = None

        result = await battery.is_charge_from_grid()

        assert result is None


class TestQSBatteryPowerManagement:
    """Test QSBattery power management methods."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance with max discharge/charge numbers."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_valid(self, hass, battery):
        """Test get_max_discharging_power with valid numeric state."""
        await _async_set_state(hass,"number.max_discharge", "3000")

        result = battery.get_max_discharging_power()

        assert result == 3000

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_unavailable(self, hass, battery):
        """Test get_max_discharging_power when unavailable."""
        await _async_set_state(hass,"number.max_discharge", STATE_UNAVAILABLE)

        result = battery.get_max_discharging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_unknown(self, hass, battery):
        """Test get_max_discharging_power when unknown."""
        await _async_set_state(hass,"number.max_discharge", STATE_UNKNOWN)

        result = battery.get_max_discharging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_invalid_string(self, hass, battery):
        """Test get_max_discharging_power with invalid string."""
        await _async_set_state(hass,"number.max_discharge", "not_a_number")

        result = battery.get_max_discharging_power()

        assert result is None

    def test_get_max_discharging_power_no_entity(self, battery):
        """Test get_max_discharging_power when entity not configured."""
        battery.max_discharge_number = None

        result = battery.get_max_discharging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_charging_power_valid(self, hass, battery):
        """Test get_max_charging_power with valid numeric state."""
        await _async_set_state(hass,"number.max_charge", "4000")

        result = battery.get_max_charging_power()

        assert result == 4000

    @pytest.mark.asyncio
    async def test_get_max_charging_power_unavailable(self, hass, battery):
        """Test get_max_charging_power when unavailable."""
        await _async_set_state(hass,"number.max_charge", STATE_UNAVAILABLE)

        result = battery.get_max_charging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_charging_power_invalid(self, hass, battery):
        """Test get_max_charging_power with invalid value."""
        await _async_set_state(hass,"number.max_charge", "invalid")

        result = battery.get_max_charging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_set_max_discharging_power(
        self, hass, battery, recorded_service_calls
    ):
        """Test setting max discharging power."""
        await _async_set_state(hass,"number.max_discharge", "5000")

        await battery.set_max_discharging_power(3000)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 3000

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_no_change(
        self, hass, battery, recorded_service_calls
    ):
        """Test set_max_discharging_power when value already set."""
        await _async_set_state(hass,"number.max_discharge", "3000")

        await battery.set_max_discharging_power(3000)

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_clamped_max(
        self, hass, battery, recorded_service_calls
    ):
        """Test set_max_discharging_power clamped to max."""
        await _async_set_state(hass,"number.max_discharge", "0")

        await battery.set_max_discharging_power(10000)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 5000

    @pytest.mark.asyncio
    async def test_set_max_charging_power(
        self, hass, battery, recorded_service_calls
    ):
        """Test setting max charging power."""
        await _async_set_state(hass,"number.max_charge", "5000")

        await battery.set_max_charging_power(4000)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 4000

    @pytest.mark.asyncio
    async def test_set_max_power_none_entity(self, battery, recorded_service_calls):
        """Test set_max_*_power with None entity."""
        battery.max_discharge_number = None

        await battery.set_max_discharging_power(3000)

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_power_none_value(self, battery, recorded_service_calls):
        """Test set_max_*_power with None value."""
        await battery.set_max_discharging_power(None)

        assert len(recorded_service_calls) == 0


class TestQSBatteryDCCoupled:
    """Test QSBattery DC coupled behavior."""

    def test_dc_coupled_no_command(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test get_current_battery_asked_change_for_outside_production_system with no command."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            },
        )
        battery.current_command = None

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 0.0

    def test_dc_coupled_zero_consign(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test get_current_battery_asked_change_for_outside_production_system with zero consign."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            },
        )
        battery.current_command = LoadCommand(command="on", power_consign=0.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 0.0

    def test_not_dc_coupled(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test get_current_battery_asked_change_for_outside_production_system when not DC coupled."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: False,
                CONF_BATTERY_CAPACITY: 10000,
            },
        )
        battery.current_command = LoadCommand(command="on", power_consign=3000.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 3000.0

    def test_dc_coupled_with_clamp(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test DC coupled with inverter clamp."""
        battery_home.get_current_over_clamp_production_power = MagicMock(
            return_value=500.0
        )
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            },
        )
        battery.current_command = LoadCommand(command="on", power_consign=3000.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 2500.0

    def test_dc_coupled_negative_consign(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test DC coupled with negative power consign (discharge)."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            },
        )
        battery.current_command = LoadCommand(command="on", power_consign=-2000.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == -2000.0


class TestQSBatteryDischarge:
    """Test QSBattery discharge capability methods."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery instance for discharge tests."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_CHARGE_PERCENT_SENSOR: "sensor.battery_soc",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
            },
        )

    @pytest.mark.asyncio
    async def test_battery_can_discharge_true(self, hass, battery):
        """Test battery_can_discharge returns True when discharge possible."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        await _async_set_state(hass,"number.max_discharge", "5000")

        result = battery.battery_can_discharge()

        assert result is True

    @pytest.mark.asyncio
    async def test_battery_can_discharge_false_zero_power(self, hass, battery):
        """Test battery_can_discharge returns False when max discharge is 0."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        await _async_set_state(hass,"number.max_discharge", "0")

        result = battery.battery_can_discharge()

        assert result is False

    @pytest.mark.asyncio
    async def test_battery_can_discharge_empty(self, hass, battery):
        """Test battery_can_discharge when battery is empty."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=0.0)
        await _async_set_state(hass,"number.max_discharge", "5000")

        result = battery.battery_can_discharge()

        assert result is False

    @pytest.mark.asyncio
    async def test_battery_get_current_possible_max_discharge_power(
        self, hass, battery
    ):
        """Test battery_get_current_possible_max_discharge_power."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        await _async_set_state(hass,"number.max_discharge", "3000")

        result = battery.battery_get_current_possible_max_discharge_power()

        assert result == 3000

    def test_battery_get_current_possible_max_discharge_unknown_charge(self, battery):
        """Test battery_get_current_possible_max_discharge_power with unknown charge."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        result = battery.battery_get_current_possible_max_discharge_power()

        assert result == 5000


class TestQSBatteryPlatforms:
    """Test QSBattery get_platforms method."""

    def test_get_platforms(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Test get_platforms returns expected platforms."""
        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CAPACITY: 10000,
            },
        )

        platforms = battery.get_platforms()

        assert Platform.SENSOR in platforms
