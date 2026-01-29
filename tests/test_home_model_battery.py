"""Tests for home_model/battery.py."""
from __future__ import annotations

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    MAX_POWER_INFINITE,
)
from custom_components.quiet_solar.home_model.battery import Battery


def test_get_best_charge_power_clamps_to_solar_and_capacity():
    """Test charge power respects solar and remaining capacity."""
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

    power = battery.get_best_charge_power(
        power_in=5000,
        solar_production=1000,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.current_charge,
    )

    assert power == 1000


def test_get_best_discharge_power_clamps_to_inverter():
    """Test discharge power respects inverter limit."""
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

    power = battery.get_best_discharge_power(
        power_out=3000,
        solar_production=2000,
        max_inverter_dc_to_ac_power=4000,
        duration_s=3600,
        current_charge=battery.current_charge,
    )

    assert power == 2000


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


def test_get_best_charge_power_handles_min_and_inverter():
    """Test charge power respects min and inverter limits."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 3000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 3000,
        },
    )
    battery.min_charging_power = 100.0

    assert (
        battery.get_best_charge_power(
            power_in=50,
            solar_production=1000,
            max_inverter_dc_to_ac_power=None,
            duration_s=3600,
            current_charge=None,
        )
        == 0.0
    )

    power = battery.get_best_charge_power(
        power_in=5000,
        solar_production=5000,
        max_inverter_dc_to_ac_power=1200,
        duration_s=3600,
        current_charge=None,
    )
    assert power == 1200


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


def test_get_best_discharge_power_handles_min_and_unknown_charge():
    """Test discharge power with min threshold and unknown charge."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 4000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 4000,
        },
    )
    battery.min_discharging_power = 100.0

    assert (
        battery.get_best_discharge_power(
            power_out=50,
            solar_production=0.0,
            max_inverter_dc_to_ac_power=None,
            duration_s=3600,
            current_charge=None,
        )
        == 0.0
    )

    power = battery.get_best_discharge_power(
        power_out=2000,
        solar_production=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=None,
    )
    assert power == 0.0

    power = battery.get_best_discharge_power(
        power_out=2000,
        solar_production=0.0,
        max_inverter_dc_to_ac_power=None,
        duration_s=3600,
        current_charge=battery.get_value_full(),
    )
    assert power == 2000


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
