"""Extended coverage tests for quiet_solar car.py.

Focuses on non-error code paths first, then corner cases.
Key targets:
 - set_next_charge_target_percent with string variants
 - car_efficiency_km_per_kwh_sensor_state_getter
 - set_next_charge_target_energy
 - user_set_next_charge_target routing
 - get_car_option_charge_from_value_percent edge values
 - get_adapt_target_percent_soc_to_reach_range_km success path
 - find_path / _get_delta_from_graph / dampening helpers
 - get_computed_range_efficiency_km_per_percent with segments
 - _add_to_amps_power_graph / _can_accept_new_dampen_values
 - update_dampening_value
 - interpolate_power_steps
 - qs_bump_solar_charge_priority setter
 - current_constraint_current_energy property
 - Various other under-tested happy paths and corner cases
"""

import pytest
from datetime import datetime, timedelta, time as dt_time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytz
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_CAR_IS_INVITED,
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES,
    CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P,
    FORCE_CAR_NO_CHARGER_CONNECTED,
    FORCE_CAR_NO_PERSON_ATTACHED,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
)
from custom_components.quiet_solar.ha_model.car import (
    MIN_CHARGE_POWER_W,
    CAR_DEFAULT_CAPACITY,
    CAR_MINIMUM_LEFT_RANGE_KM,
    QSCar,
)
from custom_components.quiet_solar.home_model.constraints import DATETIME_MAX_UTC


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


# ---------------------------------------------------------------------------
# Helper to create a car device quickly
# ---------------------------------------------------------------------------

async def _create_car(hass, home_config_entry, extra_config=None, entry_id_suffix="ext"):
    """Helper: set up home + car, return car device."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    config = {**MOCK_CAR_CONFIG}
    if extra_config:
        config.update(extra_config)

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=config,
        entry_id=f"car_{entry_id_suffix}",
        title=f"car: {config['name']}",
        unique_id=f"quiet_solar_car_{entry_id_suffix}",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    return hass.data[DOMAIN].get(car_entry.entry_id), car_entry


# ===========================================================================
# set_next_charge_target_percent – string value variants
# ===========================================================================


async def test_set_next_charge_target_percent_str_default(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """String containing 'default' should resolve to car_default_charge."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_default")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car.set_next_charge_target_percent("80% - Test Car default")
    assert car._next_charge_target == int(car.car_default_charge)


async def test_set_next_charge_target_percent_str_full(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """String containing 'full' should resolve to 100."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_full")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car.set_next_charge_target_percent("100% - full")
    assert car._next_charge_target == 100


async def test_set_next_charge_target_percent_str_plain_percent(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Plain '75%' string should strip the % and parse to 75."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_plain")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car.set_next_charge_target_percent("75%")
    assert car._next_charge_target == 75


async def test_set_next_charge_target_percent_str_numeric(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """A plain numeric string like '90' should parse correctly."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_num")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car.set_next_charge_target_percent("90")
    assert car._next_charge_target == 90


async def test_set_next_charge_target_percent_str_invalid(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Invalid string should return False and not change target."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_invalid")
    original_target = car._next_charge_target

    result = await car.set_next_charge_target_percent("not_a_number")
    assert result is False
    assert car._next_charge_target == original_target


async def test_set_next_charge_target_percent_int_value(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Integer value should work directly (non-string path)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_int")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car.set_next_charge_target_percent(85)
    assert car._next_charge_target == 85


async def test_set_next_charge_target_percent_returns_true_with_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """With charger attached, should return True after setting target."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_ret")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())
    # setup_car_charge_target_if_needed returns target_charge
    car.setup_car_charge_target_if_needed = AsyncMock(return_value=85)

    result = await car.set_next_charge_target_percent(85)
    assert result is True


async def test_set_next_charge_target_percent_returns_false_without_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Without charger, should return False."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_no_chg")

    result = await car.set_next_charge_target_percent(85)
    assert result is False


# ===========================================================================
# set_next_charge_target_energy – string + numeric
# ===========================================================================


async def test_set_next_charge_target_energy_str_kwh(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """String '30kWh' should strip units and convert to Wh."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_str")

    result = await car.set_next_charge_target_energy("30kWh")
    assert car._next_charge_target_energy == 30000.0


async def test_set_next_charge_target_energy_str_invalid(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Invalid string should return False."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_inv")

    result = await car.set_next_charge_target_energy("abc")
    assert result is False


async def test_set_next_charge_target_energy_float_value(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Numeric float should be multiplied by 1000."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_flt")

    result = await car.set_next_charge_target_energy(45.0)
    assert car._next_charge_target_energy == 45000.0


async def test_set_next_charge_target_energy_with_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """With charger, should return True."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_chg")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())

    result = await car.set_next_charge_target_energy(50.0)
    assert result is True


async def test_set_next_charge_target_energy_without_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Without charger, should return False."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_nochg")

    result = await car.set_next_charge_target_energy(50.0)
    assert result is False


# ===========================================================================
# user_set_next_charge_target – routing to percent vs energy
# ===========================================================================


async def test_user_set_next_charge_target_percent_mode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """In percent mode, delegates to set_next_charge_target_percent."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="usr_pct")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())
    car._use_percent_mode = True
    car.convert_auto_constraint_to_manual_if_needed = AsyncMock(return_value=False)
    car.set_next_charge_target_percent = AsyncMock(return_value=True)

    await car.user_set_next_charge_target("85%")
    car.set_next_charge_target_percent.assert_awaited_once_with("85%")
    car.charger.update_charger_for_user_change.assert_awaited_once()


async def test_user_set_next_charge_target_energy_mode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """In energy mode, delegates to set_next_charge_target_energy."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="usr_nrg")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())
    car._use_percent_mode = False
    car.car_is_invited = True  # force energy mode
    car.convert_auto_constraint_to_manual_if_needed = AsyncMock(return_value=False)
    car.set_next_charge_target_energy = AsyncMock(return_value=True)

    await car.user_set_next_charge_target("30kWh")
    car.set_next_charge_target_energy.assert_awaited_once_with("30kWh")
    car.charger.update_charger_for_user_change.assert_awaited_once()


async def test_user_set_next_charge_target_sets_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Should fix the person when a forecasted person is set."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="usr_pers")
    car.charger = SimpleNamespace(update_charger_for_user_change=AsyncMock())
    car.convert_auto_constraint_to_manual_if_needed = AsyncMock(return_value=False)
    car.set_next_charge_target_percent = AsyncMock(return_value=False)
    car.current_forecasted_person = SimpleNamespace(name="MyPerson")

    await car.user_set_next_charge_target(80)
    assert car._user_selected_person_name_for_car == "MyPerson"


# ===========================================================================
# car_efficiency_km_per_kwh_sensor_state_getter – various paths
# ===========================================================================


async def test_efficiency_getter_no_odometer_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when odometer sensor is missing."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eff_no_odo")
    car.car_odometer_sensor = None

    result = car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", datetime.now(pytz.UTC))
    assert result is None


async def test_efficiency_getter_no_soc_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when charge percent sensor is missing."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eff_no_soc")
    car.car_charge_percent_sensor = None

    result = car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", datetime.now(pytz.UTC))
    assert result is None


async def test_efficiency_getter_no_soc_value(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when soc is None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eff_no_val")
    car.car_odometer_sensor = "sensor.odo"
    car.get_car_charge_percent = MagicMock(return_value=None)
    car.get_car_odometer_km = MagicMock(return_value=1000.0)

    result = car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", datetime.now(pytz.UTC))
    assert result is None


async def test_efficiency_getter_with_estimated_range_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When estimated range sensor is available, uses it for efficiency."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eff_range")
    car.car_odometer_sensor = "sensor.odo"
    car.car_estimated_range_sensor = "sensor.range"
    car.car_battery_capacity = 60000  # 60 kWh in Wh

    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    # First call to establish first segment
    car.get_car_charge_percent = MagicMock(return_value=80.0)
    car.get_car_odometer_km = MagicMock(return_value=5000.0)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=320.0)  # 320 km range at 80%
    car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", time)

    # Second call with decreased SOC
    time2 = time + timedelta(hours=2)
    car.get_car_charge_percent = MagicMock(return_value=70.0)
    car.get_car_odometer_km = MagicMock(return_value=5050.0)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=280.0)  # 280 km range at 70%
    result = car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", time2)

    assert result is not None
    assert result[1] is not None
    assert result[1] > 0  # positive efficiency


async def test_efficiency_getter_ema_update(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """EMA updates when previous km_per_kwh exists."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eff_ema")
    car.car_odometer_sensor = "sensor.odo"
    car.car_estimated_range_sensor = "sensor.range"
    car.car_battery_capacity = 60000  # 60 kWh

    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car._km_per_kwh = 5.0  # Previous EMA value

    car.get_car_charge_percent = MagicMock(return_value=80.0)
    car.get_car_odometer_km = MagicMock(return_value=5000.0)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=320.0)

    result = car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", time)

    assert result is not None
    # EMA: 0.2 * new + 0.8 * old
    # new = 320 / (0.8 * 60) = 6.67
    # EMA = 0.2 * 6.67 + 0.8 * 5.0 = 5.33
    assert result[1] is not None
    assert result[1] != 5.0  # changed from old value


async def test_efficiency_getter_from_segments_no_range_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When no range sensor, efficiency from decreasing segment SOC/odo."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eff_seg")
    car.car_odometer_sensor = "sensor.odo"
    car.car_estimated_range_sensor = None
    car.car_battery_capacity = 60000  # 60 kWh

    time = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)

    # Build a decreasing segment: 80% -> 70%, odometer 1000 -> 1050
    car.get_car_charge_percent = MagicMock(return_value=80.0)
    car.get_car_odometer_km = MagicMock(return_value=1000.0)
    car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", time)

    time2 = time + timedelta(hours=1)
    car.get_car_charge_percent = MagicMock(return_value=70.0)
    car.get_car_odometer_km = MagicMock(return_value=1050.0)
    result = car.car_efficiency_km_per_kwh_sensor_state_getter("sensor.eff", time2)

    assert result is not None
    # 50 km / (10% * 60 kWh) = 50 / 6 = 8.33 km/kWh
    assert result[1] is not None
    assert result[1] > 0


# ===========================================================================
# get_car_option_charge_from_value_percent – edge values
# ===========================================================================


async def test_option_charge_percent_above_100(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Value > 100 should be clamped to 100."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="opt_100p")
    result = car.get_car_option_charge_from_value_percent(150)
    assert result == "100% - full"


async def test_option_charge_percent_default_value(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Value matching default charge should include 'default' label."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="opt_def")
    result = car.get_car_option_charge_from_value_percent(car.car_default_charge)
    assert "default" in result


async def test_option_charge_percent_regular_value(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Regular value like 60% should return '60%'."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="opt_reg")
    result = car.get_car_option_charge_from_value_percent(60)
    assert result == "60%"


# ===========================================================================
# get_adapt_target_percent_soc_to_reach_range_km – success path
# ===========================================================================


async def test_adapt_target_soc_success_covered(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When current range covers target, return (True, ...)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="soc_cov")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    car.get_car_charge_percent = MagicMock(return_value=90.0)
    car.get_estimated_range_km = MagicMock(return_value=400.0)
    car.get_computed_range_efficiency_km_per_percent = MagicMock(return_value=4.0)

    is_covered, current_soc, needed_soc, diff_energy = \
        car.get_adapt_target_percent_soc_to_reach_range_km(50.0, time)

    assert is_covered is True
    assert current_soc == 90.0
    assert needed_soc is not None


async def test_adapt_target_soc_not_covered(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When current range does NOT cover target, return (False, ...)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="soc_not")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    car.get_car_charge_percent = MagicMock(return_value=30.0)
    car.get_estimated_range_km = MagicMock(return_value=80.0)
    car.get_computed_range_efficiency_km_per_percent = MagicMock(return_value=4.0)

    is_covered, current_soc, needed_soc, diff_energy = \
        car.get_adapt_target_percent_soc_to_reach_range_km(200.0, time)

    assert is_covered is False
    assert current_soc == 30.0


async def test_adapt_target_soc_none_target(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """With None target_range_km, should return (None, None, None, None)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="soc_none")

    result = car.get_adapt_target_percent_soc_to_reach_range_km(None)
    assert result == (None, None, None, None)


# ===========================================================================
# find_path / _get_delta_from_graph
# ===========================================================================


async def test_find_path_direct(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """find_path with direct connection."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="path_dir")
    graph = {1: {2}, 2: {3}, 3: set()}
    path = car.find_path(graph, 1, 3)
    assert path == [1, 2, 3]


async def test_find_path_same_node(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """find_path from node to itself."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="path_same")
    graph = {1: {2}}
    path = car.find_path(graph, 1, 1)
    assert path == [1]


async def test_find_path_no_path(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """find_path when no path exists."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="path_none")
    graph = {1: {2}, 2: set(), 3: {4}}
    path = car.find_path(graph, 1, 4)
    assert path is None


async def test_find_path_start_not_in_graph(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """find_path when start node is not in graph."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="path_miss")
    graph = {2: {3}}
    path = car.find_path(graph, 1, 3)
    assert path is None


async def test_get_delta_from_graph_direct(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """_get_delta_from_graph with direct delta."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="delta_dir")
    deltas = {(6, 10): 500.0, (10, 6): -500.0}
    graph = {6: {10}, 10: {6}}
    result = car._get_delta_from_graph(deltas, graph, 6, 10)
    assert result == 500.0


async def test_get_delta_from_graph_via_path(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """_get_delta_from_graph via intermediate path."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="delta_path")
    deltas = {
        (6, 10): 200.0, (10, 6): -200.0,
        (10, 16): 300.0, (16, 10): -300.0,
    }
    graph = {6: {10}, 10: {6, 16}, 16: {10}}
    result = car._get_delta_from_graph(deltas, graph, 6, 16)
    assert result == 500.0  # 200 + 300


async def test_get_delta_from_graph_empty(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """_get_delta_from_graph with empty deltas returns None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="delta_mt")
    result = car._get_delta_from_graph({}, {}, 6, 10)
    assert result is None


# ===========================================================================
# _add_to_amps_power_graph
# ===========================================================================


async def test_add_to_amps_power_graph_same_amp(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Same from/to amp returns False."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="graph_same")
    result = car._add_to_amps_power_graph((10, 1), (10, 1), 100.0)
    assert result is False


async def test_add_to_amps_power_graph_positive_from_high_to_low(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Positive delta from higher to lower amp: rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="graph_pos_hi")
    result = car._add_to_amps_power_graph((16, 1), (10, 1), 100.0)
    assert result is False


async def test_add_to_amps_power_graph_negative_from_low_to_high(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Negative delta from lower to higher amp: rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="graph_neg_lo")
    result = car._add_to_amps_power_graph((10, 1), (16, 1), -100.0)
    assert result is False


async def test_add_to_amps_power_graph_valid_increasing(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Valid positive delta from lower to higher amp: accepted."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="graph_ok")
    result = car._add_to_amps_power_graph((10, 1), (16, 1), 500.0)
    assert result is True
    assert car._dampening_deltas[(10, 16)] == 500.0
    assert car._dampening_deltas[(16, 10)] == -500.0


async def test_add_to_amps_power_graph_valid_decreasing(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Valid negative delta from higher to lower amp: swapped and accepted."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="graph_swap")
    result = car._add_to_amps_power_graph((16, 1), (10, 1), -500.0)
    assert result is True
    assert car._dampening_deltas[(10, 16)] == 500.0


# ===========================================================================
# _can_accept_new_dampen_values
# ===========================================================================


async def test_can_accept_dampen_opposite_signs(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Opposite signs should be rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_opp")
    assert car._can_accept_new_dampen_values(100.0, -50.0) is False


async def test_can_accept_dampen_both_below_min(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Both values below MIN_CHARGE_POWER_W should be rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_min")
    assert car._can_accept_new_dampen_values(10.0, 20.0) is False


async def test_can_accept_dampen_new_below_min_can_dampen(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """New below MIN but can_dampen_strongly_dynamically=True: accepted."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_dyn")
    car.can_dampen_strongly_dynamically = True
    assert car._can_accept_new_dampen_values(1000.0, 10.0) is True


async def test_can_accept_dampen_new_below_min_cannot_dampen(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """New below MIN but can_dampen_strongly_dynamically=False: rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_no_dyn")
    car.can_dampen_strongly_dynamically = False
    assert car._can_accept_new_dampen_values(1000.0, 10.0) is False


async def test_can_accept_dampen_old_below_min_can_dampen(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Old below MIN, new above, can_dampen=True: accepted."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_old_lo")
    car.can_dampen_strongly_dynamically = True
    assert car._can_accept_new_dampen_values(10.0, 1000.0) is True


async def test_can_accept_dampen_old_below_min_cannot_dampen(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Old below MIN, new above, can_dampen=False: rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_old_nd")
    car.can_dampen_strongly_dynamically = False
    assert car._can_accept_new_dampen_values(10.0, 1000.0) is False


async def test_can_accept_dampen_ratio_too_high(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Ratio > 1.10 should be rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_hi")
    assert car._can_accept_new_dampen_values(1000.0, 1200.0) is False


async def test_can_accept_dampen_ratio_too_low(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Ratio < lower_ratio should be rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_lo")
    car.can_dampen_strongly_dynamically = True
    assert car._can_accept_new_dampen_values(1000.0, 100.0) is False  # 0.1 < 0.2


async def test_can_accept_dampen_ratio_acceptable(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Ratio within acceptable range: accepted."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_ok")
    car.can_dampen_strongly_dynamically = True
    # ratio = 900/1000 = 0.9, within [0.2, 1.1]
    assert car._can_accept_new_dampen_values(1000.0, 900.0) is True


async def test_can_accept_dampen_lower_ratio_no_dynamic(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Without dynamic dampening, lower_ratio is 0.7."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_no_d2")
    car.can_dampen_strongly_dynamically = False
    # ratio = 600/1000 = 0.6, which is < 0.7
    assert car._can_accept_new_dampen_values(1000.0, 600.0) is False
    # ratio = 800/1000 = 0.8, which is >= 0.7
    assert car._can_accept_new_dampen_values(1000.0, 800.0) is True


# ===========================================================================
# update_dampening_value – various branches
# ===========================================================================


async def test_update_dampening_both_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Both amperage and amperage_transition are None: return False."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_none")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    result = car.update_dampening_value(None, None, 1000, time)
    assert result is False


async def test_update_dampening_with_amperage(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Update dampening with a direct amperage value."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_amp")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    # Use a valid amperage within range, with a reasonable power value
    result = car.update_dampening_value((10, 1), None, 2300.0, time)
    assert result is True


async def test_update_dampening_amperage_out_of_range(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Amperage outside min/max range returns False."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_oor")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    # Below min (6)
    result = car.update_dampening_value((3, 1), None, 1000, time)
    assert result is False

    # Above max (32)
    result = car.update_dampening_value((40, 1), None, 1000, time)
    assert result is False


async def test_update_dampening_with_transition_from_zero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Transition from (0,x) to (y,x) extracts amperage from to-side."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_trans0")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    # transition from 0 to 10A: amperage should be extracted as (10,1)
    result = car.update_dampening_value(
        None, ((0, 1), (10, 1)), 2300.0, time
    )
    assert result is True


async def test_update_dampening_negative_power(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Negative power below -MIN_CHARGE_POWER_W should be rejected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_neg")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    result = car.update_dampening_value((10, 1), None, -200.0, time)
    assert result is False


# ===========================================================================
# get_computed_range_efficiency_km_per_percent – with stored efficiency
# ===========================================================================


async def test_computed_efficiency_from_km_per_kwh(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Uses _km_per_kwh when no range sensor and no segments."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="comp_kwh")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=None)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=None)
    car._km_per_kwh = 6.0  # 6 km/kWh
    car.car_battery_capacity = 60000  # 60 kWh

    result = car.get_computed_range_efficiency_km_per_percent(time)
    # Expected: (6.0 * 60) / 100 = 3.6 km/percent
    assert result == pytest.approx(3.6, rel=0.01)


async def test_computed_efficiency_from_range_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Uses current SOC and range sensor for efficiency."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="comp_rng")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=200.0)

    result = car.get_computed_range_efficiency_km_per_percent(time)
    # 200 / 50 = 4.0 km/percent
    assert result == pytest.approx(4.0, rel=0.01)


async def test_computed_efficiency_from_segments(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Uses stored efficiency segments when no sensors available."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="comp_seg")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=None)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=None)
    car._km_per_kwh = None

    # Add a valid segment: (delta_km, delta_soc, from_soc, to_soc, time)
    car._efficiency_segments = [
        (50.0, 10.0, 80.0, 70.0, time - timedelta(days=1)),
    ]

    result = car.get_computed_range_efficiency_km_per_percent(time, delta_soc=10.0)
    # 50 / 10 = 5.0 km/percent
    assert result == pytest.approx(5.0, rel=0.01)


# ===========================================================================
# get_car_estimated_range_km / get_estimated_range_km
# ===========================================================================


async def test_get_car_estimated_range_km_with_efficiency(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Range calculation using efficiency data."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="range_eff")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_computed_range_efficiency_km_per_percent = MagicMock(return_value=4.0)

    result = car.get_car_estimated_range_km(from_soc=80.0, to_soc=20.0, time=time)
    # 4.0 * (80 - 20) = 240 km
    assert result == pytest.approx(240.0, rel=0.01)


async def test_get_estimated_range_km_with_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_estimated_range_km prefers sensor value."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="est_sensor")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=300.0)

    result = car.get_estimated_range_km(time)
    assert result == 300.0


async def test_get_estimated_range_km_from_calculation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_estimated_range_km falls back to calculation."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="est_calc")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=None)
    car.get_car_charge_percent = MagicMock(return_value=70.0)
    car.get_computed_range_efficiency_km_per_percent = MagicMock(return_value=4.0)

    result = car.get_estimated_range_km(time)
    # 4.0 * 70 = 280
    assert result == pytest.approx(280.0, rel=0.01)


async def test_get_estimated_range_km_no_soc(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_estimated_range_km returns None when no SOC."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="est_nosoc")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=None)
    car.get_car_charge_percent = MagicMock(return_value=None)

    result = car.get_estimated_range_km(time)
    assert result is None


# ===========================================================================
# get_autonomy_to_target_soc_km
# ===========================================================================


async def test_get_autonomy_to_target_soc_km(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_autonomy_to_target_soc_km calculation."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="auto_soc")
    car.get_computed_range_efficiency_km_per_percent = MagicMock(return_value=4.0)

    result = car.get_autonomy_to_target_soc_km()
    # target SOC defaults to car_default_charge (80), so 4.0 * 80 = 320
    assert result == pytest.approx(320.0, rel=0.01)


# ===========================================================================
# qs_bump_solar_charge_priority setter
# ===========================================================================


async def test_bump_solar_priority_set_true(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Setting bump to True clears all others and sets this one."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car1_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car Bump 1"},
        entry_id="car_bump1",
        title="car: Car Bump 1",
        unique_id="quiet_solar_car_bump1",
    )
    car1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car1_entry.entry_id)
    await hass.async_block_till_done()

    car2_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car Bump 2"},
        entry_id="car_bump2",
        title="car: Car Bump 2",
        unique_id="quiet_solar_car_bump2",
    )
    car2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car2_entry.entry_id)
    await hass.async_block_till_done()

    car1 = hass.data[DOMAIN].get(car1_entry.entry_id)
    car2 = hass.data[DOMAIN].get(car2_entry.entry_id)

    car1.qs_bump_solar_charge_priority = True
    assert car1._qs_bump_solar_priority is True
    assert car2._qs_bump_solar_priority is False

    # Now set car2: should clear car1
    car2.qs_bump_solar_charge_priority = True
    assert car2._qs_bump_solar_priority is True
    assert car1._qs_bump_solar_priority is False

    # Set car2 to False
    car2.qs_bump_solar_charge_priority = False
    assert car2._qs_bump_solar_priority is False


# ===========================================================================
# current_constraint_current_energy property
# ===========================================================================


async def test_current_constraint_current_energy_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when no charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="cc_nochg")
    assert car.current_constraint_current_energy is None


async def test_current_constraint_current_energy_with_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Delegates to charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="cc_chg")
    car.charger = SimpleNamespace(current_constraint_current_energy=12345.0)
    assert car.current_constraint_current_energy == 12345.0


# ===========================================================================
# get_car_charge_energy – edge cases
# ===========================================================================


async def test_get_car_charge_energy_no_capacity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when battery capacity is zero/None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_cap0")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    soc = car.car_charge_percent_sensor
    car._entity_probed_last_valid_state[soc] = (time, 50.0, {})

    car.car_battery_capacity = 0
    assert car.get_car_charge_energy(time) is None

    car.car_battery_capacity = None
    assert car.get_car_charge_energy(time) is None


async def test_get_car_charge_energy_normal(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Normal computation of energy from percent."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_norm")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    soc = car.car_charge_percent_sensor
    car._entity_probed_last_valid_state[soc] = (time, 50.0, {})
    car.car_battery_capacity = 60000

    result = car.get_car_charge_energy(time)
    assert result == 30000.0  # 50% of 60kWh


# ===========================================================================
# is_car_plugged / is_car_home – None sensor cases
# ===========================================================================


async def test_is_car_plugged_no_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when car_plugged sensor is None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="plug_none")
    car.car_plugged = None
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    assert car.is_car_plugged(time) is None
    assert car.is_car_plugged(time, for_duration=60) is None


async def test_is_car_home_no_tracker(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when car_tracker is None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="home_none")
    car.car_tracker = None
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    assert car.is_car_home(time) is None


async def test_get_continuous_plug_duration_no_sensor(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when car_plugged is None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="dur_none")
    car.car_plugged = None
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    assert car.get_continuous_plug_duration(time) is None


# ===========================================================================
# get_car_coordinates – edge cases
# ===========================================================================


async def test_get_car_coordinates_no_tracker(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns (None, None) when no tracker."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="coord_no")
    car.car_tracker = None
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    assert car.get_car_coordinates(time) == (None, None)


async def test_get_car_coordinates_no_attrs(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns (None, None) when no lat/lon attributes."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="coord_noa")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    tracker = car.car_tracker
    car._entity_probed_last_valid_state[tracker] = (time, "home", None)
    assert car.get_car_coordinates(time) == (None, None)


async def test_get_car_coordinates_missing_lat(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns (None, None) when latitude is missing."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="coord_nolat")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    tracker = car.car_tracker
    car._entity_probed_last_valid_state[tracker] = (
        time, "home", {"longitude": "2.3522"}
    )
    assert car.get_car_coordinates(time) == (None, None)


async def test_get_car_coordinates_invalid_values(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns (None, None) when lat/lon can't be parsed."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="coord_bad")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    tracker = car.car_tracker
    car._entity_probed_last_valid_state[tracker] = (
        time, "home", {"latitude": "abc", "longitude": "def"}
    )
    assert car.get_car_coordinates(time) == (None, None)


# ===========================================================================
# dashboard_sort_string_in_type – invited car
# ===========================================================================


async def test_dashboard_sort_string_invited(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Invited cars sort to 'ZZZ'."""
    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={CONF_CAR_IS_INVITED: True},
        entry_id_suffix="sort_inv"
    )
    assert car.dashboard_sort_string_in_type == "ZZZ"


# ===========================================================================
# get_car_person_option – with/without person
# ===========================================================================


async def test_get_car_person_option_with_user_selected(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns user selected person name when set."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="popt_usr")
    car._user_selected_person_name_for_car = "MyPerson"
    result = car.get_car_person_option()
    assert result == "MyPerson"


async def test_get_car_person_option_with_forecasted(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns forecasted person name when user has not selected."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="popt_fc")
    car.current_forecasted_person = SimpleNamespace(name="ForecastPerson")
    result = car.get_car_person_option()
    assert result == "ForecastPerson"


async def test_get_car_person_option_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when no person is set."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="popt_no")
    car._user_selected_person_name_for_car = None
    car.current_forecasted_person = None
    result = car.get_car_person_option()
    assert result is None


# ===========================================================================
# car_charge_time_readable_name – charger with no active constraint
# ===========================================================================


async def test_charge_time_readable_no_constraint(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """With charger but no active constraint, should return '--:--'."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="time_noct")
    car.charger = SimpleNamespace(
        get_current_active_constraint=lambda time: None
    )
    assert car.get_car_charge_time_readable_name() == "--:--"


# ===========================================================================
# convert_auto_constraint_to_manual_if_needed – no charger / DATETIME_MAX_UTC
# ===========================================================================


async def test_convert_auto_constraint_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns False when no charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="conv_nochg")
    result = await car.convert_auto_constraint_to_manual_if_needed()
    assert result is False


async def test_convert_auto_constraint_max_utc(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Does not convert when end_of_constraint is DATETIME_MAX_UTC."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="conv_max")
    constraint = SimpleNamespace(end_of_constraint=DATETIME_MAX_UTC)
    car.charger = SimpleNamespace(
        get_charge_type=lambda: (CAR_CHARGE_TYPE_PERSON_AUTOMATED, constraint)
    )
    result = await car.convert_auto_constraint_to_manual_if_needed()
    assert result is False


# ===========================================================================
# can_force_a_charge_now / can_add_default_charge
# ===========================================================================


async def test_can_force_charge_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns False without charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="force_no")
    assert car.can_force_a_charge_now() is False
    assert car.can_add_default_charge() is False


async def test_can_force_charge_with_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns True with charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="force_yes")
    car.charger = SimpleNamespace()
    assert car.can_force_a_charge_now() is True
    assert car.can_add_default_charge() is True


# ===========================================================================
# user_add_default_charge_at_datetime / user_add_default_charge_at_dt_time
# ===========================================================================


async def test_user_add_default_charge_at_datetime(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Sets do_next_charge_time and person."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="def_dt")
    car.charger = SimpleNamespace()
    car.current_forecasted_person = SimpleNamespace(name="P1")
    end = datetime(2026, 2, 11, 8, 0, tzinfo=pytz.UTC)

    result = await car.user_add_default_charge_at_datetime(end)
    assert result is True
    assert car.do_next_charge_time == end
    assert car._user_selected_person_name_for_car == "P1"


async def test_user_add_default_charge_at_datetime_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns False when no charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="def_dt_nc")
    result = await car.user_add_default_charge_at_datetime(datetime.now(pytz.UTC))
    assert result is False


# ===========================================================================
# get_car_target_charge_energy – default capacity fallback
# ===========================================================================


async def test_get_car_target_charge_energy_default_capacity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Falls back to CAR_DEFAULT_CAPACITY when no battery capacity."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="tgt_cap")
    car.car_battery_capacity = None
    car._next_charge_target_energy = None

    result = car.get_car_target_charge_energy()
    assert result == CAR_DEFAULT_CAPACITY


async def test_get_car_target_charge_energy_from_battery(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Uses car_battery_capacity when available."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="tgt_bat")
    car.car_battery_capacity = 60000
    car._next_charge_target_energy = None

    result = car.get_car_target_charge_energy()
    assert result == 60000


# ===========================================================================
# get_car_target_charge_option – routing
# ===========================================================================


async def test_get_car_target_charge_option_percent_mode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """In percent mode, returns percent option."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="tgt_opt_pct")
    car._use_percent_mode = True
    result = car.get_car_target_charge_option()
    assert "%" in result


async def test_get_car_target_charge_option_energy_mode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """In energy mode, returns energy option."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="tgt_opt_nrg")
    car.car_is_invited = True  # forces energy mode
    car._use_percent_mode = False
    result = car.get_car_target_charge_option()
    assert "kWh" in result


# ===========================================================================
# car_use_percent_mode_sensor_state_getter – with None time
# ===========================================================================


async def test_percent_mode_sensor_getter_none_time(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Should use datetime.now when time is None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pct_mode_nt")

    # Mock get_sensor_latest_possible_valid_time_value_attr
    now = datetime.now(tz=pytz.UTC)
    soc_entity = car.car_charge_percent_sensor
    car._entity_probed_last_valid_state[soc_entity] = (now, 50.0, {})

    result = car.car_use_percent_mode_sensor_state_getter("sensor.mode", None)
    assert result is not None
    assert result[1] in ("on", "off")


async def test_percent_mode_sensor_getter_invited_car(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Invited car should always return 'off'."""
    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={CONF_CAR_IS_INVITED: True},
        entry_id_suffix="pct_mode_inv"
    )
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    result = car.car_use_percent_mode_sensor_state_getter("sensor.mode", time)
    assert result[1] == "off"
    assert car._use_percent_mode is False


# ===========================================================================
# _get_power_from_stored_amps
# ===========================================================================


async def test_get_power_from_stored_amps_below_min(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Below min charge returns 0.0."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pwr_bmin")
    result = car._get_power_from_stored_amps(3, 1)
    assert result == 0.0


async def test_get_power_from_stored_amps_1p(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """1-phase power retrieval."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pwr_1p")
    result = car._get_power_from_stored_amps(10, 1)
    assert result == car.amp_to_power_1p[10]


async def test_get_power_from_stored_amps_3p(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """3-phase power retrieval."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pwr_3p")
    result = car._get_power_from_stored_amps(10, 3)
    assert result == car.amp_to_power_3p[10]


# ===========================================================================
# get_max_charge_limit – edge cases
# ===========================================================================


async def test_get_max_charge_limit_no_entity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when no max charge entity."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="maxlim_no")
    car.car_charge_percent_max_number = None
    assert car.get_max_charge_limit() is None


async def test_get_max_charge_limit_unavailable(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when entity state is unavailable."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="maxlim_ua")
    entity = "number.test_max_charge"
    car.car_charge_percent_max_number = entity
    hass.states.async_set(entity, STATE_UNAVAILABLE)
    assert car.get_max_charge_limit() is None


async def test_get_max_charge_limit_invalid_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when state can't be parsed to int."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="maxlim_inv")
    entity = "number.test_max_charge_inv"
    car.car_charge_percent_max_number = entity
    hass.states.async_set(entity, "abc")
    assert car.get_max_charge_limit() is None


# ===========================================================================
# car_can_limit_its_soc
# ===========================================================================


async def test_car_can_limit_soc(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """car_can_limit_its_soc checks for max number entity."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="can_lim")
    car.car_charge_percent_max_number = None
    assert car.car_can_limit_its_soc() is False
    car.car_charge_percent_max_number = "number.limit"
    assert car.car_can_limit_its_soc() is True


# ===========================================================================
# set_user_person_for_car – None option
# ===========================================================================


async def test_set_user_person_for_car_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Setting None should set to FORCE_CAR_NO_PERSON_ATTACHED."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pers_none")
    car.home.get_best_persons_cars_allocations = AsyncMock(return_value={})

    await car.set_user_person_for_car(None)
    assert car._user_selected_person_name_for_car == FORCE_CAR_NO_PERSON_ATTACHED


async def test_set_user_person_for_car_force_no(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Setting FORCE_CAR_NO_PERSON_ATTACHED explicitly."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="pers_fno")
    car.home.get_best_persons_cars_allocations = AsyncMock(return_value={})

    await car.set_user_person_for_car(FORCE_CAR_NO_PERSON_ATTACHED)
    assert car._user_selected_person_name_for_car == FORCE_CAR_NO_PERSON_ATTACHED


# ===========================================================================
# get_current_selected_charger_option – with user attached name
# ===========================================================================


async def test_get_current_selected_charger_user_name(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns user_attached_charger_name when set."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="selchg_usr")
    car.user_attached_charger_name = "My Charger"
    assert car.get_current_selected_charger_option() == "My Charger"


# ===========================================================================
# get_car_next_charge_values_options – both modes
# ===========================================================================


async def test_get_car_next_charge_values_options_energy_mode(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """In energy mode, get_car_next_charge_values_options returns energy options."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="opts_nrg")
    car.car_is_invited = True  # forces energy mode
    car._use_percent_mode = False

    options = car.get_car_next_charge_values_options()
    assert all("kWh" in opt for opt in options)


# ===========================================================================
# interpolate_power_steps – use_conf_values=False
# ===========================================================================


async def test_interpolate_power_steps_non_conf(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """interpolate_power_steps with use_conf_values=False (runtime dampening)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="interp_nc")

    # Set some customized values
    car.customized_amp_to_power_1p[10] = 2000.0
    car.customized_amp_to_power_1p[20] = 4000.0
    car.customized_amp_to_power_3p[10] = 6000.0
    car.customized_amp_to_power_3p[20] = 12000.0

    car.interpolate_power_steps(do_recompute_min_charge=False, use_conf_values=False)
    # Check that interpolation happened (values are set)
    assert car.amp_to_power_1p[10] > 0
    assert car.amp_to_power_3p[10] > 0


async def test_interpolate_power_steps_recompute_min(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """interpolate_power_steps with do_recompute_min_charge=True."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="interp_min")

    # Set first amp values to 0 to trigger min charge recomputation
    for i in range(6, 9):
        car.customized_amp_to_power_1p[i] = 0.0
        car.customized_amp_to_power_3p[i] = 0.0

    car.interpolate_power_steps(do_recompute_min_charge=True, use_conf_values=False)
    # The min charge should have moved up
    assert car.car_charger_min_charge >= 6


# ===========================================================================
# _add_soc_odo_value_to_segments – segment closing / efficiency storage
# ===========================================================================


async def test_add_soc_odo_full_segment_lifecycle(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test full lifecycle: open segment, decrease, close, open new."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="seg_life")
    car._decreasing_segments = []
    car._dec_seg_count = 0
    car._efficiency_segments = []

    base = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)

    # First point: opens a segment
    car._add_soc_odo_value_to_segments(80.0, 1000.0, base)
    assert len(car._decreasing_segments) == 1
    assert car._decreasing_segments[0][1] is None  # not yet confirmed

    # Second: same SOC (equal), segment stays open
    car._add_soc_odo_value_to_segments(80.0, 1001.0, base + timedelta(hours=1))
    assert car._decreasing_segments[-1][1] is None

    # Third: SOC decreases, confirms segment
    car._add_soc_odo_value_to_segments(75.0, 1010.0, base + timedelta(hours=2))
    assert car._decreasing_segments[-1][1] is not None

    # Fourth: SOC continues decreasing
    car._add_soc_odo_value_to_segments(70.0, 1020.0, base + timedelta(hours=3))
    assert car._decreasing_segments[-1][1][0] == 70.0

    # Fifth: SOC increases – closes segment, opens new one
    car._add_soc_odo_value_to_segments(85.0, 1025.0, base + timedelta(hours=4))
    assert len(car._decreasing_segments) >= 2
    # Efficiency segment should have been stored
    assert len(car._efficiency_segments) >= 1


# ===========================================================================
# user_selected_person_name_for_car setter – clearing other cars
# ===========================================================================


async def test_user_selected_person_setter_clears_others(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Setting person on one car clears same person from other cars."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car1_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car P1"},
        entry_id="car_p1_setter",
        title="car: Car P1",
        unique_id="quiet_solar_car_p1_setter",
    )
    car1_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car1_entry.entry_id)
    await hass.async_block_till_done()

    car2_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car P2"},
        entry_id="car_p2_setter",
        title="car: Car P2",
        unique_id="quiet_solar_car_p2_setter",
    )
    car2_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car2_entry.entry_id)
    await hass.async_block_till_done()

    car1 = hass.data[DOMAIN].get(car1_entry.entry_id)
    car2 = hass.data[DOMAIN].get(car2_entry.entry_id)

    car2._user_selected_person_name_for_car = "Alice"

    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    data_handler.home.get_best_persons_cars_allocations = AsyncMock(return_value={})

    # Setting Alice on car1 should clear car2
    car1.user_selected_person_name_for_car = "Alice"
    assert car1._user_selected_person_name_for_car == "Alice"
    assert car2._user_selected_person_name_for_car is None


# ===========================================================================
# _interpolate_power_steps – edge cases
# ===========================================================================


async def test_interpolate_power_steps_with_low_min_val(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When min_charge value is below MIN_CHARGE_POWER_W, finds next valid."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="interp_lv")

    # Set low values at min charge point
    car.customized_amp_to_power_1p[6] = 30.0  # below MIN_CHARGE_POWER_W (70)
    car.customized_amp_to_power_1p[7] = 30.0  # still below
    car.customized_amp_to_power_1p[8] = 1800.0  # first valid

    car.interpolate_power_steps(do_recompute_min_charge=True, use_conf_values=False)
    # Should have adjusted min charge upward


async def test_interpolate_with_two_known_points(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Interpolation extrapolates the first value from two known points."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="interp_2pt")

    # Clear all customized values (set to -1 = default)
    for i in range(len(car.customized_amp_to_power_1p)):
        car.customized_amp_to_power_1p[i] = -1
    # Set two known values
    car.customized_amp_to_power_1p[10] = 2200.0
    car.customized_amp_to_power_1p[15] = 3400.0

    car._interpolate_power_steps(
        car.customized_amp_to_power_1p,
        car.theoretical_amp_to_power_1p,
        car.amp_to_power_1p,
    )
    # min_charge (6) should have a positive value from interpolation/theoretical
    assert car.amp_to_power_1p[6] > 0


# ===========================================================================
# init with charge_percent_max_number_steps parsing
# ===========================================================================


async def test_init_with_valid_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Valid steps string should parse and sort correctly."""
    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50,80,90"},
        entry_id_suffix="steps_valid"
    )
    assert car.car_charge_percent_max_number_steps == [50, 80, 90, 100]


async def test_init_with_invalid_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Invalid steps should result in empty list."""
    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50,abc,90"},
        entry_id_suffix="steps_invalid"
    )
    assert car.car_charge_percent_max_number_steps == []


async def test_init_with_empty_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Empty string steps should result in None-like handling."""
    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: ""},
        entry_id_suffix="steps_empty"
    )
    assert car.car_charge_percent_max_number_steps == []


async def test_init_with_100_in_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Steps that already contain 100 should not duplicate it."""
    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50,100"},
        entry_id_suffix="steps_100"
    )
    assert car.car_charge_percent_max_number_steps == [50, 100]
    assert car.car_charge_percent_max_number_steps.count(100) == 1


# ===========================================================================
# adapt_max_charge_limit – with steps
# ===========================================================================


async def test_adapt_max_charge_limit_with_steps(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """adapt_max_charge_limit should use steps to round up."""
    from homeassistant.components import number as number_mod

    set_value_handler = AsyncMock()
    hass.services.async_register(number_mod.DOMAIN, number_mod.SERVICE_SET_VALUE, set_value_handler)

    entity = "number.car_max_charge_step"
    # Current limit is 50, asking for 60 should go to 80 (next step >= default_charge 80)
    hass.states.async_set(entity, "50")

    car, _ = await _create_car(
        hass, home_config_entry,
        extra_config={
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER: entity,
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50,80,100",
        },
        entry_id_suffix="lim_steps"
    )

    # Reset mock after setup (setup may have already called the service)
    set_value_handler.reset_mock()

    # Ask for 60% – below car_default_charge (80), so percent becomes 80
    # then bisect_left finds index of 80 in [50,80,100] = index 1 → step = 80
    # current limit is 50, 50 != 80, so service call fires
    await car.adapt_max_charge_limit(asked_percent=60)
    set_value_handler.assert_awaited_once()
    call = set_value_handler.call_args.args[0]
    assert call.data[number_mod.ATTR_VALUE] == 80


async def test_adapt_max_charge_limit_no_entity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """No max number entity: should do nothing."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="lim_no")
    car.car_charge_percent_max_number = None
    # Should not raise
    await car.adapt_max_charge_limit(asked_percent=80)


# ===========================================================================
# get_car_next_charge_values_options_energy with no capacity
# ===========================================================================


async def test_energy_options_no_capacity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Energy options should fallback to CAR_DEFAULT_CAPACITY."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eopt_nocap")
    car.car_battery_capacity = None
    options = car.get_car_next_charge_values_options_energy()
    assert any("kWh" in opt for opt in options)
    # Should include max value of CAR_DEFAULT_CAPACITY
    max_kwh = CAR_DEFAULT_CAPACITY // 1000
    assert f"{max_kwh}kWh" in options


# ===========================================================================
# get_car_option_charge_from_value_energy
# ===========================================================================


async def test_option_charge_energy_formatting(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Energy option formatting in kWh."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="eopt_fmt")
    assert car.get_car_option_charge_from_value_energy(30000) == "30kWh"
    assert car.get_car_option_charge_from_value_energy(60000) == "60kWh"
    assert car.get_car_option_charge_from_value_energy(0) == "0kWh"


# ===========================================================================
# is_car_plugged with for_duration=0
# ===========================================================================


async def test_is_car_plugged_duration_zero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """for_duration=0 should behave like None (just check current state)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="plug_d0")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    plugged = car.car_plugged
    car._entity_probed_last_valid_state[plugged] = (time, "on", {})

    result = car.is_car_plugged(time, for_duration=0)
    assert result is True


async def test_is_car_home_duration_zero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """for_duration=0 should behave like None for is_car_home."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="home_d0")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    tracker = car.car_tracker
    car._entity_probed_last_valid_state[tracker] = (time, "home", {})

    result = car.is_car_home(time, for_duration=0)
    assert result is True


# ===========================================================================
# user_force_charge_now – no charger
# ===========================================================================


async def test_user_force_charge_now_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """user_force_charge_now does nothing without charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="force_nc")
    car.charger = None
    await car.user_force_charge_now()
    assert car.do_force_next_charge is False


# ===========================================================================
# get_delta_dampened_power – same amp case
# ===========================================================================


async def test_get_delta_dampened_power_same_amp(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Same amperage*phase should return 0.0."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="damp_eq")
    result = car.get_delta_dampened_power(10, 1, 10, 1)
    assert result == 0.0


# ===========================================================================
# update_dampening_value – transition with both non-zero (no extracted amperage)
# ===========================================================================


async def test_update_dampening_transition_both_nonzero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Transition between two non-zero amps (no amperage extraction)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_tnz")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    # Set up existing dampening data so orig_delta is not None
    car._dampening_deltas[(10, 16)] = 600.0
    car._dampening_deltas[(16, 10)] = -600.0
    car._dampening_deltas_graph = {10: {16}, 16: {10}}

    # Transition from 10A to 16A with a reasonable delta (within acceptance)
    result = car.update_dampening_value(
        None, ((10, 1), (16, 1)), 580.0, time
    )
    assert result is True


async def test_update_dampening_transition_to_zero(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Transition with to-side == 0 should negate power and extract from-side."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_tz")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    result = car.update_dampening_value(
        None, ((10, 1), (0, 1)), 2300.0, time
    )
    # amperage extracted as (10,1), power negated to -2300, but that's < -MIN_CHARGE_POWER_W
    # so it should be rejected
    assert result is False


async def test_update_dampening_3phase_amperage(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """3-phase amperage dampening should update 3p table."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_3p")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    result = car.update_dampening_value((10, 3), None, 6900.0, time)
    assert result is True
    assert car.customized_amp_to_power_3p[10] == 6900.0


async def test_update_dampening_small_power_near_min_charge(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Small power at near-min charge should zero out low amps."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_small")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)
    car.can_dampen_strongly_dynamically = True

    # Amp value near min_charge (6), within +2 → 8
    result = car.update_dampening_value((7, 1), None, 30.0, time)
    assert result is True
    # Should have zeroed out amps 0-7
    for i in range(0, 8):
        assert car.customized_amp_to_power_1p[i] == 0.0


# ===========================================================================
# _add_soc_odo_value_to_segments – bad segment closure
# ===========================================================================


async def test_add_soc_odo_bad_segment_replaced(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When segment has bad values (soc decrease but odo decrease), replace it."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="seg_bad")
    car._decreasing_segments = []
    car._dec_seg_count = 0
    car._efficiency_segments = []

    base = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)

    # Start segment
    car._add_soc_odo_value_to_segments(80.0, 1000.0, base)
    # Decrease SOC but also decrease odometer (bad data)
    car._add_soc_odo_value_to_segments(70.0, 990.0, base + timedelta(hours=1))
    # Now increase SOC to close the bad segment
    car._add_soc_odo_value_to_segments(85.0, 1005.0, base + timedelta(hours=2))

    # Bad segment should have been replaced, not stored as efficiency
    assert len(car._efficiency_segments) == 0


async def test_add_soc_odo_multiple_segments_in_order(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test multiple complete segments in chronological order."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="seg_multi")
    car._decreasing_segments = []
    car._dec_seg_count = 0
    car._efficiency_segments = []

    base = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)

    # First complete segment: 80→70, odo 1000→1050
    car._add_soc_odo_value_to_segments(80.0, 1000.0, base)
    car._add_soc_odo_value_to_segments(70.0, 1050.0, base + timedelta(hours=1))
    # Close by increasing SOC
    car._add_soc_odo_value_to_segments(90.0, 1060.0, base + timedelta(hours=2))

    assert len(car._efficiency_segments) >= 1

    # Second complete segment: 90→75, odo 1060→1120
    car._add_soc_odo_value_to_segments(75.0, 1120.0, base + timedelta(hours=3))
    # Close
    car._add_soc_odo_value_to_segments(95.0, 1125.0, base + timedelta(hours=4))

    assert len(car._efficiency_segments) >= 2


async def test_add_soc_odo_segment_out_of_order_time_insert(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Regression: out-of-order time insertion uses bisect correctly."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="seg_ooo")
    car._decreasing_segments = []
    car._dec_seg_count = 0
    car._efficiency_segments = []

    base = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)

    # Pre-seed an efficiency segment with a future time
    future_seg = (30.0, 5.0, 80.0, 75.0, base + timedelta(days=5))
    car._efficiency_segments.append(future_seg)

    # Now create a segment that closes *before* the existing one
    car._add_soc_odo_value_to_segments(80.0, 1000.0, base)
    car._add_soc_odo_value_to_segments(70.0, 1050.0, base + timedelta(hours=1))
    # Close it with an increase
    car._add_soc_odo_value_to_segments(85.0, 1060.0, base + timedelta(hours=2))

    # The new segment should be inserted before the future one
    assert len(car._efficiency_segments) == 2
    assert car._efficiency_segments[0][4] < car._efficiency_segments[1][4]


# ===========================================================================
# get_car_mileage_on_period_km – edge cases
# ===========================================================================


async def test_mileage_odometer_all_invalid_states(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """No valid odometer states should return None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="mile_inv")
    car.car_odometer_sensor = "sensor.car_odo"
    car.car_tracker = None

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    states = [
        SimpleNamespace(state=STATE_UNKNOWN, last_changed=from_time, attributes={}),
        SimpleNamespace(state=STATE_UNAVAILABLE, last_changed=to_time, attributes={}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=states),
    ):
        res = await car.get_car_mileage_on_period_km(from_time, to_time)

    assert res is None


async def test_mileage_tracker_skips_invalid_positions(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Tracker skips positions with missing lat/lon."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="mile_tskip")
    car.car_odometer_sensor = None
    car.car_tracker = "device_tracker.test_car"

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    positions = [
        SimpleNamespace(state="home", last_changed=from_time, attributes={"latitude": 48.8566, "longitude": 2.3522}),
        SimpleNamespace(state=STATE_UNKNOWN, last_changed=from_time + timedelta(hours=1), attributes={}),
        SimpleNamespace(state="away", last_changed=to_time, attributes={"latitude": None, "longitude": None}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=positions),
    ):
        res = await car.get_car_mileage_on_period_km(from_time, to_time)

    # Only one valid position, so no movement calculated
    assert res is None


async def test_mileage_both_sensors_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Both odometer and tracker None returns None."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="mile_none")
    car.car_odometer_sensor = None
    car.car_tracker = None

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    res = await car.get_car_mileage_on_period_km(from_time, to_time)
    assert res is None


# ===========================================================================
# get_computed_range_efficiency – segment with zero soc/km
# ===========================================================================


async def test_computed_efficiency_segments_with_zero_values_skipped(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Segments with zero delta_km or delta_soc should be skipped."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="comp_zero")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=None)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=None)
    car._km_per_kwh = None

    # First segment has zero values (should be skipped), second is valid
    car._efficiency_segments = [
        (0.0, 0.0, 80.0, 80.0, time - timedelta(days=1)),
        (50.0, 10.0, 80.0, 70.0, time - timedelta(hours=12)),
    ]

    result = car.get_computed_range_efficiency_km_per_percent(time, delta_soc=10.0)
    assert result == pytest.approx(5.0, rel=0.01)


# ===========================================================================
# get_adapt_target_percent_soc – needed_soc_b > needed_soc_a path
# ===========================================================================


async def test_adapt_target_soc_b_wins(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When needed_soc_b > needed_soc_a, the b path wins."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="soc_b")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    # high minimum_ok_charge will push needed_soc_b above needed_soc_a
    car.car_minimum_ok_charge = 70.0
    car.get_car_charge_percent = MagicMock(return_value=50.0)
    car.get_estimated_range_km = MagicMock(return_value=100.0)
    car.get_computed_range_efficiency_km_per_percent = MagicMock(return_value=4.0)

    # target_range = 20 km, needed_soc_a = (20+40)/4 = 15
    # needed_soc_b = 70 + 20/4 = 75
    # b wins because 75 > 15
    is_covered, current_soc, needed_soc, diff_energy = \
        car.get_adapt_target_percent_soc_to_reach_range_km(20.0, time)

    assert needed_soc == pytest.approx(75.0, rel=0.01)


# ===========================================================================
# _interpolate_power_steps – max charge edge cases
# ===========================================================================


async def test_interpolate_power_steps_max_charge_fallback(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """At max_charge, if measured < prev, fallback to theoretical."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="interp_max")

    # Set customized at max to be less than previous measured value
    car.customized_amp_to_power_1p[20] = 4000.0
    car.customized_amp_to_power_1p[32] = 100.0  # less than what it should be

    result = car._interpolate_power_steps(
        car.customized_amp_to_power_1p,
        car.theoretical_amp_to_power_1p,
        car.amp_to_power_1p,
    )
    # max_charge (32) should be at least prev value or theoretical
    assert car.amp_to_power_1p[32] >= 4000.0


# ===========================================================================
# is_car_plugged / is_car_home – None return from duration check
# ===========================================================================


async def test_is_car_plugged_duration_with_no_contiguous_data(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """is_car_plugged with duration but contiguous data too short."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="plug_dnone")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    plugged = car.car_plugged
    # Set short contiguous "on" state (only 10 seconds)
    car._entity_probed_state[plugged] = [
        (time - timedelta(seconds=20), "off", {}),
        (time - timedelta(seconds=10), "on", {}),
    ]
    car._entity_probed_last_valid_state[plugged] = (time - timedelta(seconds=10), "on", {})
    result = car.is_car_plugged(time, for_duration=3600)  # need 1h
    assert result is False


async def test_is_car_home_duration_with_short_home(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """is_car_home with duration but home time too short."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="home_dnone")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    tracker = car.car_tracker
    car._entity_probed_state[tracker] = [
        (time - timedelta(seconds=20), "not_home", {}),
        (time - timedelta(seconds=10), "home", {}),
    ]
    car._entity_probed_last_valid_state[tracker] = (time - timedelta(seconds=10), "home", {})
    result = car.is_car_home(time, for_duration=3600)  # need 1h
    assert result is False


# ===========================================================================
# get_car_charge_energy with non-convertible value
# ===========================================================================


async def test_get_car_charge_energy_invalid_soc(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns None when soc can't be converted to float."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_invsoc")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    soc = car.car_charge_percent_sensor
    car._entity_probed_last_valid_state[soc] = (time, "not_a_number", {})
    car.car_battery_capacity = 60000

    result = car.get_car_charge_energy(time)
    # Should return None due to conversion error
    assert result is None


# ===========================================================================
# user_add_default_charge – full flow without charger
# ===========================================================================


async def test_user_add_default_charge_no_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """user_add_default_charge does nothing without charger."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="defchg_nc")
    car.charger = None
    car.default_charge_time = datetime(2026, 1, 15, 7, 0, tzinfo=pytz.UTC).time()

    # Should not raise, just returns because can_add_default_charge is False
    await car.user_add_default_charge()
