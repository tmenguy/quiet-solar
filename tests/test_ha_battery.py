"""Tests for QSBattery class in ha_model/battery.py."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
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

from tests.conftest import FakeHass, FakeConfigEntry


class TestQSBatteryInit:
    """Test QSBattery initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
            title="battery: Test Battery"
        )

        # Mock home and data handler
        self.home = MagicMock()
        self.home.get_current_over_clamp_production_power = MagicMock(return_value=0.0)

        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_with_all_sensors(self):
        """Test initialization with all sensor configurations."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
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
            }
        )

        assert battery.charge_discharge_sensor == "sensor.battery_power"
        assert battery.max_discharge_number == "number.max_discharge"
        assert battery.max_charge_number == "number.max_charge"
        assert battery.charge_percent_sensor == "sensor.battery_soc"
        assert battery.charge_from_grid_switch == "switch.charge_from_grid"
        assert battery.is_dc_coupled is True
        assert battery.capacity == 10000

    def test_init_with_minimal_sensors(self):
        """Test initialization with minimal sensor configurations."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_CAPACITY: 7000,
            }
        )

        assert battery.charge_discharge_sensor == "sensor.battery_power"
        assert battery.max_discharge_number is None
        assert battery.max_charge_number is None
        assert battery.charge_percent_sensor is None
        assert battery.charge_from_grid_switch is None
        assert battery.is_dc_coupled is False

    def test_init_without_optional_sensors(self):
        """Test initialization without any optional sensors."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
            }
        )

        assert battery.charge_discharge_sensor is None
        assert battery.max_discharge_number is None
        assert battery.max_charge_number is None


class TestQSBatteryCurrentCharge:
    """Test QSBattery current_charge property."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_PERCENT_SENSOR: "sensor.battery_soc",
                CONF_BATTERY_CAPACITY: 10000,
            }
        )

    def test_current_charge_normal(self):
        """Test current_charge with normal value."""
        # Mock get_sensor_latest_possible_valid_value to return 50%
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)

        result = self.battery.current_charge

        # 50% of 10000 Wh = 5000 Wh
        assert result == 5000.0

    def test_current_charge_none(self):
        """Test current_charge when sensor returns None."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        result = self.battery.current_charge

        assert result is None

    def test_current_charge_zero_percent(self):
        """Test current_charge at 0%."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=0.0)

        result = self.battery.current_charge

        assert result == 0.0

    def test_current_charge_full(self):
        """Test current_charge at 100%."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=100.0)

        result = self.battery.current_charge

        assert result == 10000.0


class TestQSBatteryCommandToValues:
    """Test QSBattery _command_to_values method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            }
        )

    def test_command_to_values_cmd_on(self):
        """Test _command_to_values with CMD_ON."""
        result = self.battery._command_to_values(CMD_ON)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_idle(self):
        """Test _command_to_values with CMD_IDLE."""
        result = self.battery._command_to_values(CMD_IDLE)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_auto_green_only(self):
        """Test _command_to_values with CMD_AUTO_GREEN_ONLY."""
        result = self.battery._command_to_values(CMD_AUTO_GREEN_ONLY)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_green_charge_and_discharge(self):
        """Test _command_to_values with CMD_GREEN_CHARGE_AND_DISCHARGE."""
        result = self.battery._command_to_values(CMD_GREEN_CHARGE_AND_DISCHARGE)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 5000
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_green_charge_only(self):
        """Test _command_to_values with CMD_GREEN_CHARGE_ONLY."""
        result = self.battery._command_to_values(CMD_GREEN_CHARGE_ONLY)

        assert result["charge_from_grid"] is False
        assert result["max_discharging_power"] == 0  # No discharge allowed
        assert result["max_charging_power"] == 5000

    def test_command_to_values_cmd_force_charge(self):
        """Test _command_to_values with CMD_FORCE_CHARGE."""
        command = copy_command(CMD_FORCE_CHARGE, power_consign=3000)
        result = self.battery._command_to_values(command)

        assert result["charge_from_grid"] is True
        assert result["max_discharging_power"] == 0
        assert result["max_charging_power"] == 3000  # Power consign

    def test_command_to_values_invalid_command(self):
        """Test _command_to_values with invalid command."""
        invalid_command = LoadCommand(command="invalid", power_consign=0.0)

        with pytest.raises(ValueError, match="Invalid command"):
            self.battery._command_to_values(invalid_command)

    def test_command_to_values_without_optional_entities(self):
        """Test _command_to_values when optional entities are None."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery Minimal",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            }
        )

        result = battery._command_to_values(CMD_ON)

        # Values should be None when entities are not configured
        assert result["charge_from_grid"] is None
        assert result["max_discharging_power"] is None
        assert result["max_charging_power"] is None


class TestQSBatteryExecuteCommand:
    """Test QSBattery execute_command method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            }
        )

        # Set up state for current values
        self.hass.states.set("number.max_discharge", "5000")
        self.hass.states.set("number.max_charge", "5000")
        self.hass.states.set("switch.charge_from_grid", "off")

    @pytest.mark.asyncio
    async def test_execute_command_success(self):
        """Test successful command execution."""
        time = datetime.now(pytz.UTC)

        result = await self.battery.execute_command(time, CMD_ON)

        assert result is False  # execute_command returns False

    @pytest.mark.asyncio
    async def test_execute_command_green_charge_only(self):
        """Test execute_command with green charge only."""
        time = datetime.now(pytz.UTC)

        # Should set max_discharging_power to 0
        result = await self.battery.execute_command(time, CMD_GREEN_CHARGE_ONLY)

        assert result is False
        # Verify service was called for discharge power
        calls = [c for c in self.hass.services.calls if c[1] == "set_value"]
        # Check that the discharge power was set to 0
        assert any(
            c[2].get("value") == 0
            for c in calls
            if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        )

    @pytest.mark.asyncio
    async def test_execute_command_force_charge(self):
        """Test execute_command with force charge."""
        time = datetime.now(pytz.UTC)
        command = copy_command(CMD_FORCE_CHARGE, power_consign=3000)

        result = await self.battery.execute_command(time, command)

        assert result is False
        # Verify switch was turned on for grid charging
        calls = [c for c in self.hass.services.calls if c[1] == SERVICE_TURN_ON]
        assert len(calls) >= 1


class TestQSBatteryProbeIfCommandSet:
    """Test QSBattery probe_if_command_set method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_DISCHARGE_SENSOR: "sensor.battery_power",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            }
        )

    @pytest.mark.asyncio
    async def test_probe_command_matches(self):
        """Test probe_if_command_set when command matches current state."""
        time = datetime.now(pytz.UTC)

        # Set states to match CMD_ON values
        self.hass.states.set("switch.charge_from_grid", "off")
        self.hass.states.set("number.max_discharge", "5000")
        self.hass.states.set("number.max_charge", "5000")

        result = await self.battery.probe_if_command_set(time, CMD_ON)

        assert result is True

    @pytest.mark.asyncio
    async def test_probe_command_does_not_match(self):
        """Test probe_if_command_set when command doesn't match."""
        time = datetime.now(pytz.UTC)

        # Set states that don't match CMD_GREEN_CHARGE_ONLY (expects max_discharge = 0)
        self.hass.states.set("switch.charge_from_grid", "off")
        self.hass.states.set("number.max_discharge", "5000")  # Should be 0
        self.hass.states.set("number.max_charge", "5000")

        result = await self.battery.probe_if_command_set(time, CMD_GREEN_CHARGE_ONLY)

        assert result is False

    @pytest.mark.asyncio
    async def test_probe_switch_unavailable(self):
        """Test probe_if_command_set when switch is unavailable."""
        time = datetime.now(pytz.UTC)

        self.hass.states.set("switch.charge_from_grid", STATE_UNAVAILABLE)
        self.hass.states.set("number.max_discharge", "5000")
        self.hass.states.set("number.max_charge", "5000")

        result = await self.battery.probe_if_command_set(time, CMD_ON)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_discharge_number_unavailable(self):
        """Test probe_if_command_set when discharge number is unavailable."""
        time = datetime.now(pytz.UTC)

        self.hass.states.set("switch.charge_from_grid", "off")
        self.hass.states.set("number.max_discharge", STATE_UNKNOWN)
        self.hass.states.set("number.max_charge", "5000")

        result = await self.battery.probe_if_command_set(time, CMD_ON)

        assert result is None


class TestQSBatteryGridCharging:
    """Test QSBattery grid charging methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CHARGE_FROM_GRID_SWITCH: "switch.charge_from_grid",
                CONF_BATTERY_CAPACITY: 10000,
            }
        )

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_enable(self):
        """Test enabling grid charging."""
        self.hass.states.set("switch.charge_from_grid", "off")
        self.battery.is_charge_from_grid_current = False

        await self.battery.set_charge_from_grid(True)

        # Verify turn_on service was called
        calls = [c for c in self.hass.services.calls if c[1] == SERVICE_TURN_ON]
        assert len(calls) == 1
        assert calls[0][2]["entity_id"] == "switch.charge_from_grid"

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_disable(self):
        """Test disabling grid charging."""
        self.hass.states.set("switch.charge_from_grid", "on")
        self.battery.is_charge_from_grid_current = True

        await self.battery.set_charge_from_grid(False)

        # Verify turn_off service was called
        calls = [c for c in self.hass.services.calls if c[1] == SERVICE_TURN_OFF]
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_no_change(self):
        """Test set_charge_from_grid when value already set."""
        self.battery.is_charge_from_grid_current = True

        await self.battery.set_charge_from_grid(True)

        # No service calls should be made
        assert len(self.hass.services.calls) == 0

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_none_switch(self):
        """Test set_charge_from_grid when switch is None."""
        self.battery.charge_from_grid_switch = None

        await self.battery.set_charge_from_grid(True)

        # No service calls should be made
        assert len(self.hass.services.calls) == 0

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_enabled(self):
        """Test is_charge_from_grid returns True when switch is on."""
        self.hass.states.set("switch.charge_from_grid", "on")

        result = await self.battery.is_charge_from_grid()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_disabled(self):
        """Test is_charge_from_grid returns False when switch is off."""
        self.hass.states.set("switch.charge_from_grid", "off")

        result = await self.battery.is_charge_from_grid()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_unavailable(self):
        """Test is_charge_from_grid returns None when switch is unavailable."""
        self.hass.states.set("switch.charge_from_grid", STATE_UNAVAILABLE)

        result = await self.battery.is_charge_from_grid()

        assert result is None

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_no_switch(self):
        """Test is_charge_from_grid returns None when no switch configured."""
        self.battery.charge_from_grid_switch = None

        result = await self.battery.is_charge_from_grid()

        assert result is None


class TestQSBatteryPowerManagement:
    """Test QSBattery power management methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            }
        )

    def test_get_max_discharging_power_valid(self):
        """Test get_max_discharging_power with valid numeric state."""
        self.hass.states.set("number.max_discharge", "3000")

        result = self.battery.get_max_discharging_power()

        assert result == 3000

    def test_get_max_discharging_power_unavailable(self):
        """Test get_max_discharging_power when unavailable."""
        self.hass.states.set("number.max_discharge", STATE_UNAVAILABLE)

        result = self.battery.get_max_discharging_power()

        assert result is None

    def test_get_max_discharging_power_unknown(self):
        """Test get_max_discharging_power when unknown."""
        self.hass.states.set("number.max_discharge", STATE_UNKNOWN)

        result = self.battery.get_max_discharging_power()

        assert result is None

    def test_get_max_discharging_power_invalid_string(self):
        """Test get_max_discharging_power with invalid string."""
        self.hass.states.set("number.max_discharge", "not_a_number")

        result = self.battery.get_max_discharging_power()

        assert result is None

    def test_get_max_discharging_power_no_entity(self):
        """Test get_max_discharging_power when entity not configured."""
        self.battery.max_discharge_number = None

        result = self.battery.get_max_discharging_power()

        assert result is None

    def test_get_max_charging_power_valid(self):
        """Test get_max_charging_power with valid numeric state."""
        self.hass.states.set("number.max_charge", "4000")

        result = self.battery.get_max_charging_power()

        assert result == 4000

    def test_get_max_charging_power_unavailable(self):
        """Test get_max_charging_power when unavailable."""
        self.hass.states.set("number.max_charge", STATE_UNAVAILABLE)

        result = self.battery.get_max_charging_power()

        assert result is None

    def test_get_max_charging_power_invalid(self):
        """Test get_max_charging_power with invalid value."""
        self.hass.states.set("number.max_charge", "invalid")

        result = self.battery.get_max_charging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_set_max_discharging_power(self):
        """Test setting max discharging power."""
        self.hass.states.set("number.max_discharge", "5000")

        await self.battery.set_max_discharging_power(3000)

        # Verify service call
        calls = [c for c in self.hass.services.calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 3000

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_no_change(self):
        """Test set_max_discharging_power when value already set."""
        self.hass.states.set("number.max_discharge", "3000")

        await self.battery.set_max_discharging_power(3000)

        # No service calls should be made
        assert len(self.hass.services.calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_clamped_max(self):
        """Test set_max_discharging_power clamped to max."""
        self.hass.states.set("number.max_discharge", "0")

        await self.battery.set_max_discharging_power(10000)  # Above max

        # Should be clamped to 5000 (max_discharging_power)
        calls = [c for c in self.hass.services.calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 5000

    @pytest.mark.asyncio
    async def test_set_max_charging_power(self):
        """Test setting max charging power."""
        self.hass.states.set("number.max_charge", "5000")

        await self.battery.set_max_charging_power(4000)

        # Verify service call
        calls = [c for c in self.hass.services.calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 4000

    @pytest.mark.asyncio
    async def test_set_max_power_none_entity(self):
        """Test set_max_*_power with None entity."""
        self.battery.max_discharge_number = None

        await self.battery.set_max_discharging_power(3000)

        # No service calls should be made
        assert len(self.hass.services.calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_power_none_value(self):
        """Test set_max_*_power with None value."""
        await self.battery.set_max_discharging_power(None)

        # No service calls should be made
        assert len(self.hass.services.calls) == 0


class TestQSBatteryDCCoupled:
    """Test QSBattery DC coupled behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.home.get_current_over_clamp_production_power = MagicMock(return_value=0.0)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_dc_coupled_no_command(self):
        """Test get_current_battery_asked_change_for_outside_production_system with no command."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            }
        )
        battery.current_command = None

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 0.0

    def test_dc_coupled_zero_consign(self):
        """Test get_current_battery_asked_change_for_outside_production_system with zero consign."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            }
        )
        battery.current_command = LoadCommand(command="on", power_consign=0.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 0.0

    def test_not_dc_coupled(self):
        """Test get_current_battery_asked_change_for_outside_production_system when not DC coupled."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: False,
                CONF_BATTERY_CAPACITY: 10000,
            }
        )
        battery.current_command = LoadCommand(command="on", power_consign=3000.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == 3000.0

    def test_dc_coupled_with_clamp(self):
        """Test DC coupled with inverter clamp."""
        self.home.get_current_over_clamp_production_power = MagicMock(return_value=500.0)

        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            }
        )
        battery.current_command = LoadCommand(command="on", power_consign=3000.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        # Should be reduced by clamp: 3000 - 500 = 2500
        assert result == 2500.0

    def test_dc_coupled_negative_consign(self):
        """Test DC coupled with negative power consign (discharge)."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_IS_DC_COUPLED: True,
                CONF_BATTERY_CAPACITY: 10000,
            }
        )
        battery.current_command = LoadCommand(command="on", power_consign=-2000.0)

        result = battery.get_current_battery_asked_change_for_outside_production_system()

        assert result == -2000.0


class TestQSBatteryDischarge:
    """Test QSBattery discharge capability methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_CHARGE_PERCENT_SENSOR: "sensor.battery_soc",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
            }
        )

    def test_battery_can_discharge_true(self):
        """Test battery_can_discharge returns True when discharge possible."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        self.hass.states.set("number.max_discharge", "5000")

        result = self.battery.battery_can_discharge()

        assert result is True

    def test_battery_can_discharge_false_zero_power(self):
        """Test battery_can_discharge returns False when max discharge is 0."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        self.hass.states.set("number.max_discharge", "0")

        result = self.battery.battery_can_discharge()

        assert result is False

    def test_battery_can_discharge_empty(self):
        """Test battery_can_discharge when battery is empty."""
        # Return 0 charge which should trigger is_value_empty
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=0.0)
        self.hass.states.set("number.max_discharge", "5000")

        result = self.battery.battery_can_discharge()

        assert result is False

    def test_battery_get_current_possible_max_discharge_power(self):
        """Test battery_get_current_possible_max_discharge_power."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        self.hass.states.set("number.max_discharge", "3000")

        result = self.battery.battery_get_current_possible_max_discharge_power()

        assert result == 3000

    def test_battery_get_current_possible_max_discharge_unknown_charge(self):
        """Test battery_get_current_possible_max_discharge_power with unknown charge."""
        self.battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        result = self.battery.battery_get_current_possible_max_discharge_power()

        # Should return max discharging power when charge is unknown
        assert result == 5000


class TestQSBatteryPlatforms:
    """Test QSBattery get_platforms method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_battery_entry",
            data={CONF_NAME: "Test Battery"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_get_platforms(self):
        """Test get_platforms returns expected platforms."""
        battery = QSBattery(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_CAPACITY: 10000,
            }
        )

        platforms = battery.get_platforms()

        assert Platform.SENSOR in platforms
