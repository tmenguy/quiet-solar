"""Tests for QSBattery class in ha_model/battery.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import pytz
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_NAME,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    Platform,
)
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR,
    CONF_BATTERY_CHARGE_FROM_GRID_SWITCH,
    CONF_BATTERY_CHARGE_PERCENT_SENSOR,
    CONF_BATTERY_IS_DC_COUPLED,
    CONF_BATTERY_MAX_CHARGE_POWER_NUMBER,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.battery import QSBattery
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_GREEN_ONLY,
    CMD_FORCE_CHARGE,
    CMD_GREEN_CHARGE_AND_DISCHARGE,
    CMD_GREEN_CHARGE_ONLY,
    CMD_IDLE,
    CMD_ON,
    LoadCommand,
    copy_command,
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


class TestQSBatteryDischargeFloor:
    """Test the outage safety floor emission (min_discharging_power)."""

    def _floored_battery(self, hass, battery_config_entry, battery_home, floor, max_dis=5000, max_cha=5000):
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
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: max_dis,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: max_cha,
                CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: floor,
            },
        )

    def test_green_charge_only_emits_floor(self, hass, battery_config_entry, battery_home, battery_hass_data):
        """AC 1: CMD_GREEN_CHARGE_ONLY emits the floor, never 0."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        result = battery._command_to_values(CMD_GREEN_CHARGE_ONLY)
        assert result["max_discharging_power"] == 300

    def test_force_charge_emits_floor(self, hass, battery_config_entry, battery_home, battery_hass_data):
        """AC 2: CMD_FORCE_CHARGE emits the floor, never 0."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        command = copy_command(CMD_FORCE_CHARGE, power_consign=3000)
        result = battery._command_to_values(command)
        assert result["max_discharging_power"] == 300

    @pytest.mark.asyncio
    async def test_execute_green_charge_only_writes_floor(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """AC 1: executing CMD_GREEN_CHARGE_ONLY writes set_value(number, 300)."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert any(c[2].get("value") == 300 for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge")

    @pytest.mark.asyncio
    async def test_probe_confirms_non_integer_floor(
        self, hass, battery_config_entry, battery_home, battery_hass_data
    ):
        """AC 4: a non-integer floor (300.7 -> 301) confirms; no eternal-retry."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300.7)
        # init rounds 300.7 -> 301.0
        assert battery.min_discharging_power == 301.0
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        await _async_set_state(hass, "number.max_discharge", "301")
        await _async_set_state(hass, "number.max_charge", "5000")

        result = await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        assert result is True

    @pytest.mark.asyncio
    async def test_kw_entity_write_converts_and_probe_confirms(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """S2: a kW-denominated number entity gets the unit-converted floor and confirms."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        await _async_set_state(hass, "number.max_discharge", "5", {"unit_of_measurement": "kW"})
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        # 300 W written as 0.3 kW (not a raw 300 that would land as 300 kW)
        assert any(
            c[2].get("value") == pytest.approx(0.3)
            for c in calls
            if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        )

        # entity now reads the converted value; probe reads it back as 300 W and confirms
        await _async_set_state(hass, "number.max_discharge", "0.3", {"unit_of_measurement": "kW"})
        result = await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)
        assert result is True

    @pytest.mark.asyncio
    async def test_stepped_entity_snaps_floor_and_probe_confirms(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """S2: a min/max/step-constrained entity snaps the floor; write and probe agree."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        attrs = {"unit_of_measurement": "W", "min": 0, "max": 10000, "step": 40}
        await _async_set_state(hass, "number.max_discharge", "5000", attrs)
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        # 300 W snaps UP to the next multiple of the 40 W step: 320 W (a floor
        # is a safety minimum and must never be lowered by snapping)
        assert any(
            c[2].get("value") == pytest.approx(320.0)
            for c in calls
            if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        )

        await _async_set_state(hass, "number.max_discharge", "320", attrs)
        result = await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)
        assert result is True

    @pytest.mark.asyncio
    async def test_kw_entity_step_one_snaps_floor_up_never_zero(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """R1: a kW entity with step 1 snaps the floor UP to 1 kW, never down to 0."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        attrs = {"unit_of_measurement": "kW", "step": 1}
        await _async_set_state(hass, "number.max_discharge", "5", attrs)
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        # 300 W -> 0.3 kW -> ceil to 1 kW (>= floor, NEVER the hazardous 0)
        discharge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert discharge_writes
        assert all(v == pytest.approx(1.0) for v in discharge_writes)
        assert all(v > 0 for v in discharge_writes)

        await _async_set_state(hass, "number.max_discharge", "1", attrs)
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY) is True

    @pytest.mark.asyncio
    async def test_stepped_entity_snap_up_stays_in_range(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """R1: min/step entity — snap-up stays in range, no out-of-range write, probe confirms."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 100)
        attrs = {"unit_of_measurement": "W", "min": 100, "max": 5000, "step": 250}
        await _async_set_state(hass, "number.max_discharge", "5000", attrs)
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        # floor 100 -> clamp to min 100 -> ceil(100/250)*250 = 250 (in range, >= floor)
        writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert writes and all(v == pytest.approx(250.0) for v in writes)

        await _async_set_state(hass, "number.max_discharge", "250", attrs)
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY) is True

    @pytest.mark.asyncio
    async def test_kw_charge_entity_write_converts_and_probe_confirms(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """R5: a kW-denominated max_charge_number gets the converted value and confirms."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        await _async_set_state(hass, "number.max_discharge", "0", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "number.max_charge", "1", {"unit_of_measurement": "kW"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        command = copy_command(CMD_FORCE_CHARGE, power_consign=2300)

        await battery.execute_command(datetime.now(pytz.UTC), command)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        # 2300 W charge -> 2.3 kW written (not a raw 2300 landing as 2300 kW)
        charge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_charge"]
        assert charge_writes and all(v == pytest.approx(2.3) for v in charge_writes)

        # simulate the landed states (force-charge turns the grid switch on)
        await _async_set_state(hass, "number.max_charge", "2.3", {"unit_of_measurement": "kW"})
        await _async_set_state(hass, "switch.charge_from_grid", "on")
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), command) is True

    @pytest.mark.asyncio
    async def test_stepped_charge_entity_snaps_down(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """R5/T3: a charge limit is a maximum — it snaps DOWN (never raised past its cap)."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        await _async_set_state(hass, "number.max_discharge", "0", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "number.max_charge", "0", {"unit_of_measurement": "W", "step": 100})
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        command = copy_command(CMD_FORCE_CHARGE, power_consign=2340)

        await battery.execute_command(datetime.now(pytz.UTC), command)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        charge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_charge"]
        # 2340 -> floor to the 100 W step -> 2300 (a limit is never raised)
        assert charge_writes and all(v == pytest.approx(2300.0) for v in charge_writes)

    @pytest.mark.asyncio
    async def test_charge_consign_above_max_probe_confirms_clamped(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """T2: a consign above max_charging_power is clamped identically by write and probe."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        await _async_set_state(hass, "number.max_discharge", "0", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "number.max_charge", "0", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        command = copy_command(CMD_FORCE_CHARGE, power_consign=6000)

        await battery.execute_command(datetime.now(pytz.UTC), command)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        charge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_charge"]
        assert charge_writes and all(v == pytest.approx(5000.0) for v in charge_writes)

        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "on")
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), command) is True

    @pytest.mark.asyncio
    async def test_charge_max_not_divisible_by_step_snaps_down_in_range(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """T1(a): charge max not a step multiple → snap DOWN, in range, probe confirms."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        attrs = {"unit_of_measurement": "W", "max": 5000, "step": 300}
        await _async_set_state(hass, "number.max_discharge", "0", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "number.max_charge", "0", attrs)
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        command = copy_command(CMD_FORCE_CHARGE, power_consign=5000)

        await battery.execute_command(datetime.now(pytz.UTC), command)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        charge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_charge"]
        # floor(5000/300)*300 = 4800 (in range, never the out-of-range 5100)
        assert charge_writes and all(v == pytest.approx(4800.0) for v in charge_writes)

        await _async_set_state(hass, "number.max_charge", "4800", attrs)
        await _async_set_state(hass, "switch.charge_from_grid", "on")
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), command) is True

    @pytest.mark.asyncio
    async def test_charge_min_step_force_charge_stays_in_range(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """T1(b): min/step charge entity — a tiny consign stays at the entity min, in range."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        attrs = {"unit_of_measurement": "W", "min": 100, "step": 250}
        await _async_set_state(hass, "number.max_discharge", "0", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "number.max_charge", "5000", attrs)
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        command = copy_command(CMD_FORCE_CHARGE, power_consign=100)

        await battery.execute_command(datetime.now(pytz.UTC), command)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        charge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_charge"]
        # never a below-min 0 write (the old regression); clamps up to the entity min
        assert charge_writes and all(v == pytest.approx(100.0) for v in charge_writes)

    @pytest.mark.asyncio
    async def test_discharge_restore_stepped_never_exceeds_configured_max(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """T3: the max-discharge restore snaps DOWN, never above the configured max."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        # entity range allows 5100, but the configured max is 5000 — must not overshoot
        attrs = {"unit_of_measurement": "W", "max": 6000, "step": 300}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_AND_DISCHARGE)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        discharge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        # floor(5000/300)*300 = 4800 <= 5000 (never the 5100 overshoot)
        assert discharge_writes and all(v == pytest.approx(4800.0) and v <= 5000.0 for v in discharge_writes)

    @pytest.mark.asyncio
    async def test_floor_exact_step_multiple_stays_exact(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """T5: a floor that is an exact step multiple does not overshoot by a whole step."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 400)
        attrs = {"unit_of_measurement": "W", "step": 100}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        discharge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        # 400 is already a multiple of 100 -> stays 400 (not 500)
        assert discharge_writes and all(v == pytest.approx(400.0) for v in discharge_writes)

    @pytest.mark.asyncio
    async def test_non_numeric_step_treated_as_absent(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """T7: a non-numeric step attribute is treated as absent — no crash, raw write."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        attrs = {"unit_of_measurement": "W", "step": "unknown"}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        discharge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert discharge_writes and all(v == pytest.approx(300.0) for v in discharge_writes)

        await _async_set_state(hass, "number.max_discharge", "300", attrs)
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY) is True

    @pytest.mark.asyncio
    async def test_entity_min_forcing_value_above_request_logs_warning(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls, caplog
    ):
        """T8/U5: the entity min forces the landed value above the request — warn ONCE."""
        import logging

        battery = self._floored_battery(hass, battery_config_entry, battery_home, 0)
        attrs = {"unit_of_measurement": "W", "min": 1000}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        with caplog.at_level(logging.WARNING):
            # two probes with the same divergence must warn only once (U5 dedupe)
            await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)
            await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        discharge_writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert discharge_writes and all(v == pytest.approx(1000.0) for v in discharge_writes)
        assert caplog.text.count("above the requested") == 1

    @pytest.mark.asyncio
    async def test_restore_snap_down_respects_safety_floor(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """U1: a snapped-down restore never drops below the safety floor F."""
        # F=1100, configured max=1200, step 500: floor(1200/500)*500 = 1000 < F
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 1100, max_dis=1200)
        attrs = {"unit_of_measurement": "W", "max": 6000, "step": 500}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_AND_DISCHARGE)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        # snapped-down 1000 would drop below F; the floor wins upward -> 1100
        assert writes and all(v == pytest.approx(1100.0) for v in writes)

    @pytest.mark.asyncio
    async def test_floor_equals_max_restore_and_green_only_agree(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """U1: floor == max on a stepped entity — restore and green-only land the same >= F."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 5000, max_dis=5000)
        attrs = {"unit_of_measurement": "W", "max": 6000, "step": 300}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_AND_DISCHARGE)
        restore = [
            c[2].get("value")
            for c in recorded_service_calls
            if c[1] == "set_value" and c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        ]
        recorded_service_calls.clear()
        # move the entity off the target so the green-only write is not short-circuited
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)
        green_only = [
            c[2].get("value")
            for c in recorded_service_calls
            if c[1] == "set_value" and c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        ]
        assert restore and green_only
        assert restore[-1] == pytest.approx(5000.0)
        assert green_only[-1] == pytest.approx(5000.0)

    @pytest.mark.asyncio
    async def test_floor_ceil_capped_at_configured_max(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """U2: the floor's ceil snap is capped at the configured max, not the looser entity max."""
        # floor 4900, configured max 5000, step 300, entity max 6000 -> ceil 5100 must cap at 5000
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 4900, max_dis=5000)
        attrs = {"unit_of_measurement": "W", "max": 6000, "step": 300}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert writes and all(v == pytest.approx(5000.0) and v <= 5000.0 for v in writes)

    @pytest.mark.asyncio
    async def test_entity_max_below_floor_underdelivers_and_warns(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls, caplog
    ):
        """U3: entity max below the requested floor forces under-delivery — landed < request, warn."""
        import logging

        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        attrs = {"unit_of_measurement": "W", "max": 100}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        with caplog.at_level(logging.WARNING):
            await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert writes and all(v == pytest.approx(100.0) for v in writes)
        assert "below the requested" in caplog.text

    @pytest.mark.asyncio
    async def test_step_larger_than_entity_max_not_zeroed(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """U7: step > entity max must not zero the floor (lands at the raw entity max)."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        # kW entity: max 0.05 kW, step 0.1 kW -> a step-aligned cap would be 0
        attrs = {"unit_of_measurement": "kW", "max": 0.05, "step": 0.1}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        assert writes and all(v == pytest.approx(0.05) and v > 0 for v in writes)

    @pytest.mark.asyncio
    async def test_inconsistent_min_max_attributes_treated_as_absent(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """U6: min > max (corrupt attributes) are ignored, avoiding an out-of-range eternal retry."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        attrs = {"unit_of_measurement": "W", "min": 500, "max": 100}
        await _async_set_state(hass, "number.max_discharge", "0", attrs)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        writes = [c[2].get("value") for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"]
        # inconsistent bounds ignored -> the floor 300 lands unclamped
        assert writes and all(v == pytest.approx(300.0) for v in writes)
        await _async_set_state(hass, "number.max_discharge", "300", attrs)
        assert await battery.probe_if_command_set(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY) is True

    @pytest.mark.asyncio
    async def test_unavailable_entity_skips_write(
        self, hass, battery_config_entry, battery_home, battery_hass_data, recorded_service_calls
    ):
        """U4: no write is issued while the number entity is unavailable (stale attributes)."""
        battery = self._floored_battery(hass, battery_config_entry, battery_home, 300)
        await _async_set_state(hass, "number.max_discharge", STATE_UNAVAILABLE)
        await _async_set_state(hass, "number.max_charge", "5000", {"unit_of_measurement": "W"})
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        await battery.execute_command(datetime.now(pytz.UTC), CMD_GREEN_CHARGE_ONLY)

        discharge_writes = [
            c for c in recorded_service_calls if c[1] == "set_value" and c[2].get(ATTR_ENTITY_ID) == "number.max_discharge"
        ]
        assert discharge_writes == []


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
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        time = datetime.now(pytz.UTC)

        result = await battery.execute_command(time, CMD_ON)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_command_green_charge_only(self, hass, battery, recorded_service_calls):
        """Test execute_command with green charge only."""
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        time = datetime.now(pytz.UTC)

        result = await battery.execute_command(time, CMD_GREEN_CHARGE_ONLY)

        assert result is False
        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert any(c[2].get("value") == 0 for c in calls if c[2].get(ATTR_ENTITY_ID) == "number.max_discharge")

    @pytest.mark.asyncio
    async def test_execute_command_force_charge(self, hass, battery, recorded_service_calls):
        """Test execute_command with force charge."""
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        await _async_set_state(hass, "switch.charge_from_grid", "off")
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
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is True

    @pytest.mark.asyncio
    async def test_probe_command_does_not_match(self, hass, battery):
        """Test probe_if_command_set when command doesn't match."""
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_GREEN_CHARGE_ONLY)

        assert result is False

    @pytest.mark.asyncio
    async def test_probe_switch_unavailable(self, hass, battery):
        """Test probe_if_command_set when switch is unavailable."""
        await _async_set_state(hass, "switch.charge_from_grid", STATE_UNAVAILABLE)
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", "5000")
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_discharge_number_unavailable(self, hass, battery):
        """Test probe_if_command_set when discharge number is unavailable."""
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        await _async_set_state(hass, "number.max_discharge", STATE_UNKNOWN)
        await _async_set_state(hass, "number.max_charge", "5000")
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
    async def test_set_charge_from_grid_enable(self, hass, battery, recorded_service_calls):
        """Test enabling grid charging."""
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        battery.is_charge_from_grid_current = False

        await battery.set_charge_from_grid(True)

        calls = [c for c in recorded_service_calls if c[1] == SERVICE_TURN_ON]
        assert len(calls) == 1
        assert calls[0][0] == Platform.SWITCH

    @pytest.mark.asyncio
    async def test_set_charge_from_grid_disable(self, hass, battery, recorded_service_calls):
        """Test disabling grid charging."""
        await _async_set_state(hass, "switch.charge_from_grid", "on")
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
        await _async_set_state(hass, "switch.charge_from_grid", "on")

        result = await battery.is_charge_from_grid()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_disabled(self, hass, battery):
        """Test is_charge_from_grid returns False when switch is off."""
        await _async_set_state(hass, "switch.charge_from_grid", "off")

        result = await battery.is_charge_from_grid()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_charge_from_grid_unavailable(self, hass, battery):
        """Test is_charge_from_grid returns None when switch is unavailable."""
        await _async_set_state(hass, "switch.charge_from_grid", STATE_UNAVAILABLE)

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
        await _async_set_state(hass, "number.max_discharge", "3000")

        result = battery.get_max_discharging_power()

        assert result == 3000

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_unavailable(self, hass, battery):
        """Test get_max_discharging_power when unavailable."""
        await _async_set_state(hass, "number.max_discharge", STATE_UNAVAILABLE)

        result = battery.get_max_discharging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_unknown(self, hass, battery):
        """Test get_max_discharging_power when unknown."""
        await _async_set_state(hass, "number.max_discharge", STATE_UNKNOWN)

        result = battery.get_max_discharging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_discharging_power_invalid_string(self, hass, battery):
        """Test get_max_discharging_power with invalid string."""
        await _async_set_state(hass, "number.max_discharge", "not_a_number")

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
        await _async_set_state(hass, "number.max_charge", "4000")

        result = battery.get_max_charging_power()

        assert result == 4000

    @pytest.mark.asyncio
    async def test_get_max_charging_power_unavailable(self, hass, battery):
        """Test get_max_charging_power when unavailable."""
        await _async_set_state(hass, "number.max_charge", STATE_UNAVAILABLE)

        result = battery.get_max_charging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_max_charging_power_invalid(self, hass, battery):
        """Test get_max_charging_power with invalid value."""
        await _async_set_state(hass, "number.max_charge", "invalid")

        result = battery.get_max_charging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_set_max_discharging_power(self, hass, battery, recorded_service_calls):
        """Test setting max discharging power."""
        await _async_set_state(hass, "number.max_discharge", "5000")

        await battery.set_max_discharging_power(3000)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 3000

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_no_change(self, hass, battery, recorded_service_calls):
        """Test set_max_discharging_power when value already set."""
        await _async_set_state(hass, "number.max_discharge", "3000")

        await battery.set_max_discharging_power(3000)

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_clamped_max(self, hass, battery, recorded_service_calls):
        """Test set_max_discharging_power clamped to max."""
        await _async_set_state(hass, "number.max_discharge", "0")

        await battery.set_max_discharging_power(10000)

        calls = [c for c in recorded_service_calls if c[1] == "set_value"]
        assert len(calls) == 1
        assert calls[0][2]["value"] == 5000

    @pytest.mark.asyncio
    async def test_set_max_charging_power(self, hass, battery, recorded_service_calls):
        """Test setting max charging power."""
        await _async_set_state(hass, "number.max_charge", "5000")

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
        battery_home.get_current_over_clamp_production_power = MagicMock(return_value=500.0)
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
        await _async_set_state(hass, "number.max_discharge", "5000")

        result = battery.battery_can_discharge()

        assert result is True

    @pytest.mark.asyncio
    async def test_battery_can_discharge_false_zero_power(self, hass, battery):
        """Test battery_can_discharge returns False when max discharge is 0."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        await _async_set_state(hass, "number.max_discharge", "0")

        result = battery.battery_can_discharge()

        assert result is False

    @pytest.mark.asyncio
    async def test_battery_can_discharge_empty(self, hass, battery):
        """Test battery_can_discharge when battery is empty."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=0.0)
        await _async_set_state(hass, "number.max_discharge", "5000")

        result = battery.battery_can_discharge()

        assert result is False

    @pytest.mark.asyncio
    async def test_battery_get_current_possible_max_discharge_power(self, hass, battery):
        """Test battery_get_current_possible_max_discharge_power."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=50.0)
        await _async_set_state(hass, "number.max_discharge", "3000")

        result = battery.battery_get_current_possible_max_discharge_power()

        assert result == 3000

    def test_battery_get_current_possible_max_discharge_unknown_charge(self, battery):
        """Test battery_get_current_possible_max_discharge_power with unknown charge."""
        battery.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        result = battery.battery_get_current_possible_max_discharge_power()

        assert result == 5000


class TestQSBatteryProbeChargeNumberNone:
    """Test probe_if_command_set returns None when get_max_charging_power() is None."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery with all entities configured."""
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
    async def test_probe_returns_none_when_charge_number_unavailable(self, hass, battery):
        """Lines 101-103: probe returns None when max_charging_power state is unavailable."""
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", STATE_UNAVAILABLE)
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is None

    @pytest.mark.asyncio
    async def test_probe_returns_none_when_charge_number_unknown(self, hass, battery):
        """Lines 101-103: probe returns None when max_charging_power state is unknown."""
        await _async_set_state(hass, "switch.charge_from_grid", "off")
        await _async_set_state(hass, "number.max_discharge", "5000")
        await _async_set_state(hass, "number.max_charge", STATE_UNKNOWN)
        time = datetime.now(pytz.UTC)

        result = await battery.probe_if_command_set(time, CMD_ON)

        assert result is None


class TestQSBatterySetMaxChargingPowerEdge:
    """Test set_max_charging_power no-op and exception paths."""

    @pytest.fixture
    def battery(
        self,
        hass,
        battery_config_entry,
        battery_home,
        battery_hass_data,
    ):
        """Battery with max charge number configured."""
        return QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_MAX_CHARGE_POWER_NUMBER: "number.max_charge",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )

    @pytest.mark.asyncio
    async def test_set_max_charging_power_no_op_same_value(self, hass, battery, recorded_service_calls):
        """Line 231/241: set_max_charging_power is a no-op when value equals current."""
        await _async_set_state(hass, "number.max_charge", "4000")

        await battery.set_max_charging_power(4000)

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_charging_power_no_arg_returns_early(self, hass, battery, recorded_service_calls):
        """Line 231: set_max_charging_power() with no argument (power=None) returns early."""
        await _async_set_state(hass, "number.max_charge", "4000")

        await battery.set_max_charging_power()

        assert len(recorded_service_calls) == 0

    @pytest.mark.asyncio
    async def test_set_max_charging_power_exception_caught(self, hass, battery):
        """Lines 253-254: set_max_charging_power catches exception from service call."""
        from homeassistant.core import ServiceRegistry

        await _async_set_state(hass, "number.max_charge", "5000")

        async def raise_error(*args, **kwargs):
            raise RuntimeError("Service fail")

        with patch.object(ServiceRegistry, "async_call", raise_error):
            await battery.set_max_charging_power(3000)

    @pytest.mark.asyncio
    async def test_get_max_charging_power_bad_float(self, hass, battery):
        """Lines 220-223: get_max_charging_power returns None on unparseable state."""
        await _async_set_state(hass, "number.max_charge", "not_a_number")

        result = battery.get_max_charging_power()

        assert result is None

    @pytest.mark.asyncio
    async def test_set_max_discharging_power_exception_caught(
        self, hass, battery_config_entry, battery_home, battery_hass_data
    ):
        """Lines 171-172: set_max_discharging_power catches exception from service call."""
        from homeassistant.core import ServiceRegistry

        battery = QSBattery(
            hass=hass,
            config_entry=battery_config_entry,
            home=battery_home,
            **{
                CONF_NAME: "Test Battery",
                CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER: "number.max_discharge",
                CONF_BATTERY_CAPACITY: 10000,
                CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
                CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            },
        )
        await _async_set_state(hass, "number.max_discharge", "5000")

        async def raise_error(*args, **kwargs):
            raise RuntimeError("Service fail")

        with patch.object(ServiceRegistry, "async_call", raise_error):
            await battery.set_max_discharging_power(3000)


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
