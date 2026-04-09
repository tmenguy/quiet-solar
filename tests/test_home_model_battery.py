"""Tests for home_model/battery.py."""

from __future__ import annotations

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    MAX_POWER_INFINITE,
)
from custom_components.quiet_solar.home_model.battery import Battery


def test_charge_from_grid_base_property():
    """Base Battery.charge_from_grid always returns False."""
    battery = Battery(name="bat", **{
        CONF_BATTERY_CAPACITY: 10000,
        CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
        CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
    })
    assert battery.charge_from_grid is False


def test_get_charger_power_charge_from_excess_solar_and_soc_clamp():
    """Test charging from excess solar; SOC clamp limits near-full battery."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 3000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 3000,
        },
    )
    battery._current_charge_value = 9000

    # available_power=-1000 means 1000 W excess solar; only 1000 Wh room left
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=-1000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.current_charge,
    )

    assert charging_power == 1000.0
    assert ac_flow == 1000.0  # all AC in, no AC out
    assert possible_discharge == 3000.0  # plenty of energy to discharge

    # Full battery: no more charging possible
    battery._current_charge_value = battery.get_value_full()
    charging_power, ac_flow, _ = battery.get_charger_power(
        available_power=-2000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.current_charge,
    )
    assert charging_power == 0.0
    assert ac_flow == 0.0


def test_get_charger_power_discharge_clamps_to_inverter():
    """Test discharge power clamped by inverter AC limit."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 4000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 4000,
        },
    )
    battery._current_charge_value = 8000

    # available_power=3000 means 3000 W deficit; inverter caps AC output at 2000
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=3000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=2000.0,
        duration_s=3600,
        current_charge=battery.current_charge,
    )

    assert charging_power == -2000.0  # negative = discharging
    assert ac_flow == -2000.0  # AC out
    assert possible_discharge == 4000.0


def test_battery_current_possible_max_discharge_power():
    """Test max discharge power with missing and empty charge values."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 4000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 4000,
        },
    )

    battery._current_charge_value = None
    assert battery.battery_get_current_possible_max_discharge_power() == 4000

    battery._current_charge_value = battery.get_value_empty()
    assert battery.battery_get_current_possible_max_discharge_power() == 0.0


def test_get_charger_power_inverter_limit_and_dc_clamp():
    """Test inverter AC limit caps charging; DC-coupled path adds clamped power."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 3000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 3000,
        },
    )

    # Inverter limit caps AC charging to 1200 W
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=-5000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=1200.0,
        duration_s=3600,
        current_charge=None,  # defaults to 0.0
    )
    assert charging_power == 1200.0
    assert ac_flow == 1200.0
    assert possible_discharge == 0.0  # empty battery can't discharge

    # DC-coupled path: clamped_over_dc_power adds direct DC charging
    battery._current_charge_value = 5000.0
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=0.0,  # balanced load
        clamped_over_dc_power=2000.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.current_charge,
    )
    assert charging_power == 2000.0  # all from DC path
    assert ac_flow == 0.0  # no AC flow
    assert possible_discharge == 3000.0  # mid-charge battery

    # None max_inverter uses float("inf") — no AC capping
    charging_power_uncapped, _, _ = battery.get_charger_power(
        available_power=-5000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.current_charge,
    )
    assert charging_power_uncapped == 3000.0  # limited by max_charging_power, not inverter


def test_charge_discharge_helpers_and_availability():
    """Test helper methods for full/empty and available energy."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 3000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 3000,
        },
    )

    assert battery.is_value_full(None) is False
    assert battery.is_value_empty(None) is True

    battery._current_charge_value = None
    assert battery.get_available_energy() == 0.0

    battery._current_charge_value = battery.get_value_empty() + 100.0
    assert battery.get_available_energy() == 100.0


def test_get_charger_power_discharge_soc_clamp_and_unknown_charge():
    """Test discharge: empty battery blocked, unknown charge defaults to 0, full battery discharges."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 4000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 4000,
        },
    )

    # Unknown charge (None) defaults to 0.0 => empty => can't discharge
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=2000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=None,
    )
    assert charging_power == 0.0
    assert ac_flow == 0.0
    assert possible_discharge == 0.0

    # Full battery can discharge
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=2000.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.get_value_full(),
    )
    assert charging_power == -2000.0  # negative = discharging
    assert ac_flow == -2000.0
    assert possible_discharge == 4000.0  # full battery, max discharge available


def test_battery_max_discharge_infinite():
    """Test max discharge power when unlimited."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 4000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: MAX_POWER_INFINITE,
        },
    )
    battery._current_charge_value = battery.get_value_full()
    assert battery.get_max_discharging_power() is None
    assert battery.battery_get_current_possible_max_discharge_power() == MAX_POWER_INFINITE
