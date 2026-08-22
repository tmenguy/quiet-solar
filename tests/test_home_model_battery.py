"""Tests for home_model/battery.py."""

from __future__ import annotations

import pytest

from custom_components.quiet_solar.const import (
    CONF_BATTERY_CAPACITY,
    CONF_BATTERY_IS_DC_COUPLED,
    CONF_BATTERY_MAX_CHARGE_PERCENT,
    CONF_BATTERY_MAX_CHARGE_POWER_VALUE,
    CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE,
    CONF_BATTERY_MIN_CHARGE_PERCENT,
    CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE,
    MAX_POWER_INFINITE,
)
from custom_components.quiet_solar.home_model.battery import Battery, coerce_finite_float


def _make_battery(**overrides):
    """Build a Battery with sensible defaults, allowing per-test overrides."""
    kwargs = {
        CONF_BATTERY_CAPACITY: 10000,
        CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
        CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
    }
    kwargs.update(overrides)
    return Battery(name="bat", **kwargs)


def test_min_discharging_power_defaults_to_zero_when_unset():
    """A battery configured without the floor key keeps floor = 0 (opt-in)."""
    battery = _make_battery()
    assert battery.min_discharging_power == 0.0


def test_min_discharging_power_from_config_key():
    """The configured floor reaches Battery.min_discharging_power."""
    battery = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 300})
    assert battery.min_discharging_power == 300.0


def test_min_discharging_power_clamped_above_max():
    """A floor above max_discharging_power is capped to the max at init."""
    battery = _make_battery(
        **{
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 2000,
            CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 3000,
        }
    )
    assert battery.min_discharging_power == 2000.0


def test_min_discharging_power_negative_clamped_to_zero():
    """A negative floor (YAML/import path) is clamped to 0."""
    battery = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: -500})
    assert battery.min_discharging_power == 0.0


def test_min_discharging_power_integer_normalized_rounds():
    """A non-integer floor is rounded to an integer at init (probe consistency)."""
    battery = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 300.7})
    assert battery.min_discharging_power == 301.0
    battery_down = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 299.4})
    assert battery_down.min_discharging_power == 299.0


def test_min_discharging_power_none_tolerated():
    """A null floor (corrupt/hand-edited entry) is treated as 0, not a crash."""
    battery = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: None})
    assert battery.min_discharging_power == 0.0


def test_min_discharging_power_string_tolerated():
    """R6: a string floor (corrupt entry) is coerced to float, not a crash."""
    battery = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: "300"})
    assert battery.min_discharging_power == 300.0
    garbage = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: "not-a-number"})
    assert garbage.min_discharging_power == 0.0


def test_max_discharging_power_null_tolerated():
    """R6: a null max discharge power (corrupt entry) falls back to the default."""
    battery = _make_battery(**{CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: None})
    assert battery.max_discharging_power == 1500.0
    # the floor still clamps against the fallback max
    floored = _make_battery(
        **{CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: None, CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 5000}
    )
    assert floored.min_discharging_power == 1500.0


def test_negative_max_discharge_does_not_produce_negative_floor():
    """T6(a): a corrupt negative max discharge is clamped to 0, so the floor is not negative."""
    battery = _make_battery(**{CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: -100})
    assert battery.max_discharging_power == 0.0
    assert battery.min_discharging_power == 0.0


def test_nan_floor_rejected():
    """T6(b): a 'nan' floor coerces but is non-finite → rejected to the default 0 (no NaN poison)."""
    battery = _make_battery(**{CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: "nan"})
    assert battery.min_discharging_power == 0.0
    inf_max = _make_battery(**{CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: "inf"})
    assert inf_max.max_discharging_power == 1500.0


def test_corrupt_charge_max_coerced():
    """T6(c): a null / non-numeric max charging power falls back to the default, no crash."""
    battery = _make_battery(**{CONF_BATTERY_MAX_CHARGE_POWER_VALUE: None})
    assert battery.max_charging_power == 1500.0
    garbage = _make_battery(**{CONF_BATTERY_MAX_CHARGE_POWER_VALUE: "oops"})
    assert garbage.max_charging_power == 1500.0


def test_boolean_entry_rejected():
    """X7: a boolean is float()-able (True == 1.0) but is corrupt config — it
    must fall back to the default, not silently become 1 W / 1 %."""
    assert coerce_finite_float(True, 5.0) == 5.0
    assert coerce_finite_float(False, None) is None
    battery = _make_battery(**{CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: True})
    assert battery.max_discharging_power == 1500.0


def test_corrupt_capacity_coerced():
    """U8: a null / 'nan' capacity falls back to the default (no TypeError / NaN poison)."""
    null_cap = _make_battery(**{CONF_BATTERY_CAPACITY: None})
    assert null_cap.capacity == 7000.0
    assert null_cap.get_value_full() == 7000.0  # no crash downstream
    nan_cap = _make_battery(**{CONF_BATTERY_CAPACITY: "nan"})
    assert nan_cap.capacity == 7000.0


def test_corrupt_soc_percents_coerced():
    """U8: null / non-numeric SOC percents fall back to their defaults."""
    battery = _make_battery(
        **{CONF_BATTERY_MIN_CHARGE_PERCENT: None, CONF_BATTERY_MAX_CHARGE_PERCENT: "bad"}
    )
    assert battery.min_charge_SOC_percent == 0.0
    assert battery.max_charge_SOC_percent == 100.0
    assert battery.min_soc == 0.0
    assert battery.max_soc == 1.0


def test_negative_capacity_clamped_to_zero():
    """V3: a finite-but-negative capacity is clamped to 0 (no inverted trajectories)."""
    battery = _make_battery(**{CONF_BATTERY_CAPACITY: -7000})
    assert battery.capacity == 0.0
    assert battery.get_value_full() == 0.0
    assert battery.get_value_empty() == 0.0


def test_out_of_range_soc_percents_clamped():
    """V3: SOC percents are clamped to [0, 100] and min <= max is enforced."""
    over = _make_battery(**{CONF_BATTERY_MAX_CHARGE_PERCENT: 150})
    assert over.max_charge_SOC_percent == 100.0
    inverted = _make_battery(
        **{CONF_BATTERY_MIN_CHARGE_PERCENT: 80, CONF_BATTERY_MAX_CHARGE_PERCENT: 40}
    )
    assert inverted.min_charge_SOC_percent <= inverted.max_charge_SOC_percent
    assert inverted.min_soc <= inverted.max_soc


def test_floor_reclamped_after_round_with_fractional_max():
    """V4: rounding a floor==max never pushes min_discharging_power above a fractional max."""
    battery = _make_battery(
        **{CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 1499.5, CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE: 2000}
    )
    # floor clamped to 1499.5, round -> 1500, re-clamped back to 1499.5 (<= max)
    assert battery.min_discharging_power <= battery.max_discharging_power
    assert battery.min_discharging_power == pytest.approx(1499.5)


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


def test_get_charger_power_dc_coupled_discharge_clamped_by_pv():
    """DC-coupled: battery discharge limited by remaining inverter capacity after PV."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
            CONF_BATTERY_IS_DC_COUPLED: True,
        },
    )
    battery._current_charge_value = 8000

    # PV=5900, inverter=6000 → only 100W inverter headroom for battery discharge
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=1100.0,  # ua=7000, pv=5900 → 1100 deficit
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=6000.0,
        duration_s=3600,
        current_charge=battery.current_charge,
        solar_production=5900.0,
    )
    # discharge_inverter_limit = max(0, 6000 - 5900) = 100
    # battery_ac_out = min(5000, min(1100, 100)) = 100
    assert ac_flow == -100.0
    assert charging_power == -100.0

    # AC-coupled battery: same scenario but no PV-inverter sharing
    battery_ac = Battery(
        name="Battery_AC",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
            CONF_BATTERY_IS_DC_COUPLED: False,
        },
    )
    battery_ac._current_charge_value = 8000

    _, ac_flow_ac, _ = battery_ac.get_charger_power(
        available_power=1100.0,
        clamped_over_dc_power=0.0,
        max_inverter_dc_to_ac_power=6000.0,
        duration_s=3600,
        current_charge=battery_ac.current_charge,
        solar_production=5900.0,
    )
    # AC-coupled: discharge_inverter_limit = inverter_ac_limit = 6000 (no PV sharing)
    assert ac_flow_ac == -1100.0  # full 1100W discharge (limited by demand, not inverter)


def test_get_charger_power_dc_coupled_pv_saturates_inverter():
    """DC-coupled: PV exceeds inverter → zero discharge headroom."""
    battery = Battery(
        name="Battery",
        device_type="battery",
        **{
            CONF_BATTERY_CAPACITY: 10000,
            CONF_BATTERY_MAX_CHARGE_POWER_VALUE: 5000,
            CONF_BATTERY_MAX_DISCHARGE_POWER_VALUE: 5000,
            CONF_BATTERY_IS_DC_COUPLED: True,
        },
    )
    battery._current_charge_value = 8000

    # PV=15000 > inverter=12000 → discharge_inverter_limit = 0
    charging_power, ac_flow, possible_discharge = battery.get_charger_power(
        available_power=500.0,
        clamped_over_dc_power=3000.0,  # excess PV beyond inverter
        max_inverter_dc_to_ac_power=12000.0,
        duration_s=3600,
        current_charge=battery.current_charge,
        solar_production=15000.0,
    )
    # discharge_inverter_limit = max(0, 12000 - 15000) = 0
    # battery_ac_out = min(5000, min(500, 0)) = 0
    assert ac_flow >= 0.0  # no discharge, only charging from clamped DC
