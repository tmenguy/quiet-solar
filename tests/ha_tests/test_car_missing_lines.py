"""Tests targeting specific uncovered lines in ha_model/car.py.

Covers:
 - Custom 3p power charge values (line 144)
 - Mileage computation with odometer and GPS tracker (lines 398-399, 405, 416-417, 442)
 - get_car_charge_type with charger (line 568)
 - Efficiency segment bisect: SOC increases over open segment (lines 632-634)
 - _async_bootstrap_efficiency_from_history branches (lines 688, 692, 705-710, 717, 720-721, 729, 735, 737, 756, 764, 767, 772, 780)
 - reset with home=None (line 800)
 - is_car_plugged / is_car_home for_duration returning None (lines 840, 845, 894, 899)
 - attach_charger (lines 804-805)
 - get_car_charge_energy exception (line 894... actually the exception branch)
 - _get_delta_from_graph broken path and bad-path log (lines 954-956, 960)
 - get_computed_range_efficiency_km_per_percent old segment break (line 983)
 - get_autonomy_to_target_soc_km None target (line 1075)
 - adapt_max_charge_limit steps overflow (line 1091)
 - adapt_max_charge_limit service exception (lines 1109-1110)
 - update_dampening_value transition graph return False (line 1299)
 - update_dampening_value amperage acceptance reject (lines 1326-1328)
 - update_dampening_value graph add fail for amperage (line 1335)
 - update_dampening_value 3p cross-update to 1p (line 1351)
 - user_add_default_charge_at_dt_time can_add=False (line 1509)
 - set_user_person_for_car with charger update (line 379)
"""

from datetime import datetime, timedelta
from datetime import time as dt_time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CAR_CHARGE_TYPE_NOT_PLUGGED,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES,
    CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    DATA_HANDLER,
    DOMAIN,
)

pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def _create_car(hass, home_config_entry, extra_config=None, entry_id_suffix="ml"):
    """Helper: set up home + car, return car device."""
    from homeassistant.config_entries import ConfigEntryState

    from .const import MOCK_CAR_CONFIG

    if home_config_entry.state is not ConfigEntryState.LOADED:
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
# Custom 3p power charge values (line 144)
# ===========================================================================


async def test_custom_power_charge_values_3p(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Car with CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P=True sets 3p values and cross-maps to 1p."""
    extra = {
        CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
        CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P: True,
        CONF_CAR_CHARGER_MIN_CHARGE: 6,
        CONF_CAR_CHARGER_MAX_CHARGE: 32,
        "charge_10": 6900.0,
    }
    car, _ = await _create_car(hass, home_config_entry, extra_config=extra, entry_id_suffix="3p_custom")

    assert car.car_is_custom_power_charge_values_3p is True
    assert car.can_dampen_strongly_dynamically is False
    assert car.conf_customized_amp_to_power_3p[10] == 6900.0
    # 3p amp 10 -> 1p amp 30 cross-mapping (10*3=30, within [6,32])
    assert car.conf_customized_amp_to_power_1p[30] == 6900.0


async def test_custom_power_charge_values_1p_cross_to_3p(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Car with custom 1p values cross-maps to 3p when amp % 3 == 0. Covers line 144."""
    extra = {
        CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
        CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P: False,
        CONF_CAR_CHARGER_MIN_CHARGE: 6,
        CONF_CAR_CHARGER_MAX_CHARGE: 32,
        "charge_9": 2070.0,
    }
    car, _ = await _create_car(hass, home_config_entry, extra_config=extra, entry_id_suffix="1p_cross3p")

    assert car.conf_customized_amp_to_power_1p[9] == 2070.0
    # 9 % 3 == 0, 9//3 = 3 < min_charge(6), so no cross-map to 3p
    assert car.conf_customized_amp_to_power_3p[3] == -1

    # Use amp 18 which is % 3 == 0 and 18//3 = 6 >= min_charge
    extra2 = {
        CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
        CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P: False,
        CONF_CAR_CHARGER_MIN_CHARGE: 6,
        CONF_CAR_CHARGER_MAX_CHARGE: 32,
        "charge_18": 4140.0,
    }
    car2, _ = await _create_car(hass, home_config_entry, extra_config=extra2, entry_id_suffix="1p_cross3p2")
    assert car2.conf_customized_amp_to_power_1p[18] == 4140.0
    # 18 % 3 == 0, 18//3 = 6 >= min_charge(6) and <= max_charge(32)
    assert car2.conf_customized_amp_to_power_3p[6] == 4140.0


# ===========================================================================
# Mileage computation (lines 398-399, 405, 416-417, 442)
# ===========================================================================


async def test_mileage_odometer_type_error_in_from_states(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Odometer states with non-numeric values trigger TypeError/ValueError continue (lines 398-399, 416-417)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="mile_typerr")
    car.car_odometer_sensor = "sensor.car_odo"
    car.car_tracker = None

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    # All states are non-numeric → both from_state and to_state remain None
    states = [
        SimpleNamespace(
            state="not_a_number", last_changed=from_time - timedelta(hours=1), attributes={"unit_of_measurement": "km"}
        ),
        SimpleNamespace(
            state="also_bad", last_changed=from_time + timedelta(minutes=1), attributes={"unit_of_measurement": "km"}
        ),
        SimpleNamespace(state="still_bad", last_changed=to_time, attributes={"unit_of_measurement": "km"}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=states),
    ):
        res = await car.get_car_mileage_on_period_km(from_time, to_time)

    assert res is None


async def test_mileage_odometer_first_state_after_from_time(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When no prev_state before from_time, uses first state as from_state (line 405)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="mile_first")
    car.car_odometer_sensor = "sensor.car_odo"
    car.car_tracker = None

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    states = [
        SimpleNamespace(
            state="100", last_changed=from_time + timedelta(minutes=1), attributes={"unit_of_measurement": "km"}
        ),
        SimpleNamespace(state="150", last_changed=to_time, attributes={"unit_of_measurement": "km"}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=states),
    ):
        res = await car.get_car_mileage_on_period_km(from_time, to_time)

    assert res == pytest.approx(50.0, rel=0.01)


async def test_mileage_tracker_none_attributes(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """GPS tracker positions with None attributes are skipped (line 442)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="mile_noneattr")
    car.car_odometer_sensor = None
    car.car_tracker = "device_tracker.test_car"

    from_time = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    to_time = datetime(2026, 1, 15, 12, 0, tzinfo=pytz.UTC)

    positions = [
        SimpleNamespace(state="home", last_changed=from_time, attributes=None),
        SimpleNamespace(
            state="home",
            last_changed=from_time + timedelta(hours=1),
            attributes={"latitude": 48.8566, "longitude": 2.3522},
        ),
        SimpleNamespace(state="away", last_changed=to_time, attributes={"latitude": 48.8600, "longitude": 2.3600}),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.car.load_from_history",
        new=AsyncMock(return_value=positions),
    ):
        res = await car.get_car_mileage_on_period_km(from_time, to_time)

    assert res is not None
    assert res > 0.0


# ===========================================================================
# get_car_charge_type with charger (line 568)
# ===========================================================================


async def test_get_car_charge_type_with_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When charger is attached, delegates to charger.get_charge_type."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="chtype_chg")
    car.charger = SimpleNamespace(get_charge_type=lambda: ("some_type", None))
    assert car.get_car_charge_type() == "some_type"


async def test_get_car_charge_type_without_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Without charger returns CAR_CHARGE_TYPE_NOT_PLUGGED."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="chtype_nochg")
    assert car.get_car_charge_type() == CAR_CHARGE_TYPE_NOT_PLUGGED


# ===========================================================================
# _add_soc_odo_value_to_segments: SOC increases over open segment (lines 632-634)
# ===========================================================================


async def test_segment_soc_increase_over_open_replaces(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When SOC increases above open segment start, replace the segment (lines 632-634)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="seg_inc")
    car._decreasing_segments = []
    car._dec_seg_count = 0
    car._efficiency_segments = []

    base = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)

    car._add_soc_odo_value_to_segments(80.0, 1000.0, base)
    assert len(car._decreasing_segments) == 1
    old_count = car._dec_seg_count

    # SOC increases above 80 while segment is still open (segment[1] is None)
    car._add_soc_odo_value_to_segments(85.0, 1005.0, base + timedelta(hours=1))
    # Segment should be replaced (same index), dec_seg_count incremented
    assert len(car._decreasing_segments) == 1
    assert car._decreasing_segments[0][0][0] == 85.0
    assert car._dec_seg_count == old_count + 1


# ===========================================================================
# _async_bootstrap_efficiency_from_history (lines 688-780)
# ===========================================================================


async def test_bootstrap_efficiency_no_hass(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns early when hass is None (line 688)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_nohass")
    car.hass = None
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    await car._async_bootstrap_efficiency_from_history(time)
    assert car._km_per_kwh is None


async def test_bootstrap_efficiency_no_odometer(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns early when odometer sensor is None (line 688-692 path)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_noodo")
    car.car_odometer_sensor = None
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    await car._async_bootstrap_efficiency_from_history(time)


async def test_bootstrap_efficiency_no_capacity(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns early when battery capacity is zero (line 692)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_nocap")
    car.car_battery_capacity = 0
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    await car._async_bootstrap_efficiency_from_history(time)


async def test_bootstrap_efficiency_exception_loading(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Exception during history loading is caught (lines 705-707)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_exc")
    car.car_odometer_sensor = "sensor.odo"
    car.car_charge_percent_sensor = "sensor.soc"
    car.car_battery_capacity = 60000
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    with patch(
        "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
        side_effect=RuntimeError("test error"),
    ):
        await car._async_bootstrap_efficiency_from_history(time)
    assert car._km_per_kwh is None


async def test_bootstrap_efficiency_empty_odos(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Returns early when odos list is empty (line 710)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_empty")
    car.car_odometer_sensor = "sensor.odo"
    car.car_charge_percent_sensor = "sensor.soc"
    car.car_battery_capacity = 60000
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = AsyncMock(return_value=[])

    with patch(
        "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
        return_value=mock_recorder,
    ):
        await car._async_bootstrap_efficiency_from_history(time)


async def test_bootstrap_efficiency_full_success(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Full bootstrap with valid decreasing SOC segments computes efficiency (covers lines 717-780)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_ok")
    car.car_odometer_sensor = "sensor.odo"
    car.car_charge_percent_sensor = "sensor.soc"
    car.car_battery_capacity = 60000
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    base = time - timedelta(days=5)

    odo_states = [
        SimpleNamespace(state="1000", last_changed=base),
        SimpleNamespace(state="1050", last_changed=base + timedelta(hours=2)),
        SimpleNamespace(state="1100", last_changed=base + timedelta(hours=4)),
        SimpleNamespace(state="1100", last_changed=base + timedelta(hours=6)),
    ]
    soc_states = [
        SimpleNamespace(state="80", last_changed=base),
        SimpleNamespace(state="70", last_changed=base + timedelta(hours=2)),
        SimpleNamespace(state="60", last_changed=base + timedelta(hours=4)),
        SimpleNamespace(state="90", last_changed=base + timedelta(hours=6)),
    ]

    call_count = 0

    async def mock_executor_job(fn, entity_id):
        nonlocal call_count
        call_count += 1
        if "odo" in entity_id:
            return odo_states
        return soc_states

    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = mock_executor_job

    with (
        patch(
            "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "custom_components.quiet_solar.ha_model.car.state_changes_during_period",
            side_effect=lambda hass, start, end, eid, **kw: {eid: odo_states if "odo" in eid else soc_states},
        ),
    ):
        await car._async_bootstrap_efficiency_from_history(time)

    assert car._km_per_kwh is not None
    assert car._km_per_kwh > 0


async def test_bootstrap_efficiency_invalid_states_filtered(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Bootstrap filters invalid/unknown states (lines 716-721)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_filter")
    car.car_odometer_sensor = "sensor.odo"
    car.car_charge_percent_sensor = "sensor.soc"
    car.car_battery_capacity = 60000
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    base = time - timedelta(days=5)

    odo_states = [
        SimpleNamespace(state=STATE_UNKNOWN, last_changed=base),
        SimpleNamespace(state="bad_val", last_changed=base + timedelta(hours=1)),
        SimpleNamespace(state="1000", last_changed=base + timedelta(hours=2)),
        SimpleNamespace(state="1060", last_changed=base + timedelta(hours=4)),
    ]
    soc_states = [
        SimpleNamespace(state=None, last_changed=base),
        SimpleNamespace(state="unavailable", last_changed=base + timedelta(hours=1)),
        SimpleNamespace(state="80", last_changed=base + timedelta(hours=2)),
        SimpleNamespace(state="70", last_changed=base + timedelta(hours=4)),
    ]

    async def mock_executor_job(fn, entity_id):
        if "odo" in entity_id:
            return odo_states
        return soc_states

    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = mock_executor_job

    with (
        patch(
            "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "custom_components.quiet_solar.ha_model.car.state_changes_during_period",
            side_effect=lambda hass, start, end, eid, **kw: {eid: odo_states if "odo" in eid else soc_states},
        ),
    ):
        await car._async_bootstrap_efficiency_from_history(time)

    # Only 2 valid points each, which is enough for series
    # but only 1 decreasing pair, so may or may not get efficiency


async def test_bootstrap_efficiency_too_few_points(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Bootstrap with < 2 valid points returns early (line 729)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_few")
    car.car_odometer_sensor = "sensor.odo"
    car.car_charge_percent_sensor = "sensor.soc"
    car.car_battery_capacity = 60000
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    odo_states = [SimpleNamespace(state="1000", last_changed=time - timedelta(hours=1))]
    soc_states = [SimpleNamespace(state="80", last_changed=time - timedelta(hours=1))]

    async def mock_executor_job(fn, entity_id):
        if "odo" in entity_id:
            return odo_states
        return soc_states

    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = mock_executor_job

    with (
        patch(
            "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "custom_components.quiet_solar.ha_model.car.state_changes_during_period",
            side_effect=lambda hass, start, end, eid, **kw: {eid: odo_states if "odo" in eid else soc_states},
        ),
    ):
        await car._async_bootstrap_efficiency_from_history(time)

    assert car._km_per_kwh is None


async def test_bootstrap_efficiency_no_valid_segments(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Bootstrap with no decreasing SOC segments gives no efficiency (line 780)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="boot_noseg")
    car.car_odometer_sensor = "sensor.odo"
    car.car_charge_percent_sensor = "sensor.soc"
    car.car_battery_capacity = 60000
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    base = time - timedelta(days=5)

    odo_states = [
        SimpleNamespace(state="1000", last_changed=base),
        SimpleNamespace(state="1050", last_changed=base + timedelta(hours=2)),
    ]
    # SOC increasing — no decreasing segments
    soc_states = [
        SimpleNamespace(state="50", last_changed=base),
        SimpleNamespace(state="80", last_changed=base + timedelta(hours=2)),
    ]

    async def mock_executor_job(fn, entity_id):
        if "odo" in entity_id:
            return odo_states
        return soc_states

    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = mock_executor_job

    with (
        patch(
            "custom_components.quiet_solar.ha_model.car.recorder_get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "custom_components.quiet_solar.ha_model.car.state_changes_during_period",
            side_effect=lambda hass, start, end, eid, **kw: {eid: odo_states if "odo" in eid else soc_states},
        ),
    ):
        await car._async_bootstrap_efficiency_from_history(time)

    assert car._km_per_kwh is None


# ===========================================================================
# reset with home=None (line 800)
# ===========================================================================


async def test_reset_with_no_home(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """reset() when home is None no longer modifies current_forecasted_person."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="reset_nohome")
    car.current_forecasted_person = SimpleNamespace(name="P1")
    car.home = None
    car.reset()
    assert car.current_forecasted_person is not None
    assert car.charger is None


# ===========================================================================
# attach_charger (lines 804-805)
# ===========================================================================


async def test_attach_charger(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """attach_charger calls charger.attach_car (lines 804-805)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="attach_chg")
    mock_charger = MagicMock()
    mock_charger.name = "TestCharger"
    mock_charger.attach_car = MagicMock()

    car.attach_charger(mock_charger)
    mock_charger.attach_car.assert_called_once()


# ===========================================================================
# is_car_plugged for_duration returns None (line 840)
# ===========================================================================


async def test_is_car_plugged_for_duration_returns_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """is_car_plugged with for_duration when get_last_state_value_duration returns None (line 840)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="plug_fdnone")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_last_state_value_duration = MagicMock(return_value=(None,))
    result = car.is_car_plugged(time, for_duration=300)
    assert result is None


async def test_is_car_plugged_latest_state_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """is_car_plugged without for_duration when latest state is None (line 845)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="plug_lsnone")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)
    result = car.is_car_plugged(time)
    assert result is None


# ===========================================================================
# is_car_home for_duration returns None (line 894)
# ===========================================================================


async def test_is_car_home_for_duration_returns_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """is_car_home with for_duration when contiguous returns None (line 894)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="home_fdnone")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_last_state_value_duration = MagicMock(return_value=(None,))
    result = car.is_car_home(time, for_duration=300)
    assert result is None


async def test_is_car_home_latest_state_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """is_car_home without for_duration when latest state is None (line 899)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="home_lsnone")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)
    result = car.is_car_home(time)
    assert result is None


# ===========================================================================
# _get_delta_from_graph: broken path and bad path log (lines 954-956, 960)
# ===========================================================================


async def test_get_delta_from_graph_path_with_missing_delta(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Path found but delta missing for a hop causes None return (lines 954-956)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="delta_miss")
    # Graph says 6->10->16 but delta (10,16) missing
    deltas = {(6, 10): 200.0, (10, 6): -200.0, (6, 16): None}
    graph = {6: {10}, 10: {6, 16}, 16: {10}}
    # deltas has >2 entries, path 6->10->16 exists, but (10,16) not in deltas
    result = car._get_delta_from_graph(deltas, graph, 6, 16)
    assert result is None


async def test_get_delta_from_graph_path_doesnt_match_endpoints(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Path returned doesn't start/end as expected (line 960)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="delta_badpath")
    deltas = {(6, 10): 200.0, (10, 6): -200.0, (10, 16): 300.0}
    graph = {6: {10}, 10: {6, 16}, 16: {10}}

    # find_path returns a bad path (not matching endpoints)
    with patch.object(car, "find_path", return_value=[6, 10]):
        result = car._get_delta_from_graph(deltas, graph, 6, 16)
    # Path exists but doesn't end at 16, so logs error and returns None
    assert result is None


# ===========================================================================
# get_computed_range_efficiency: old segment break (line 983)
# ===========================================================================


async def test_computed_efficiency_old_segment_breaks_loop(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Segment older than CAR_MAX_EFFICIENCY_HISTORY_S breaks the loop (line 983)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="comp_old")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=None)
    car.get_car_estimated_range_km_from_sensor = MagicMock(return_value=None)
    car._km_per_kwh = None

    # All segments are too old (>31 days)
    car._efficiency_segments = [
        (50.0, 10.0, 80.0, 70.0, time - timedelta(days=60)),
        (40.0, 8.0, 70.0, 62.0, time - timedelta(days=40)),
    ]

    result = car.get_computed_range_efficiency_km_per_percent(time, delta_soc=10.0)
    assert result is None


# ===========================================================================
# get_autonomy_to_target_soc_km None target (line 1075)
# ===========================================================================


async def test_autonomy_to_target_soc_none(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_autonomy_to_target_soc_km returns None when target SOC is None (line 1075)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="auto_none")
    car.get_car_target_SOC = MagicMock(return_value=None)
    result = car.get_autonomy_to_target_soc_km()
    assert result is None


# ===========================================================================
# adapt_max_charge_limit: steps overflow (line 1091) and service exception (lines 1109-1110)
# ===========================================================================


async def test_adapt_max_charge_limit_steps_overflow(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When bisect_left returns index >= len(steps), uses last step (line 1091)."""
    from homeassistant.components import number as number_mod

    set_value_handler = AsyncMock()
    hass.services.async_register(number_mod.DOMAIN, number_mod.SERVICE_SET_VALUE, set_value_handler)

    entity = "number.car_max_charge_overflow"
    hass.states.async_set(entity, "50")

    car, _ = await _create_car(
        hass,
        home_config_entry,
        extra_config={
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER: entity,
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50,80",
            CONF_DEFAULT_CAR_CHARGE: 80.0,
        },
        entry_id_suffix="lim_overflow",
    )
    set_value_handler.reset_mock()

    # Ask for 95% — steps are [50,80,100], bisect_left(100) -> past steps [50,80,100]
    # Actually steps will be [50, 80, 100] (100 auto-appended)
    # bisect_left([50,80,100], 95) = 2, which is index 2, so step = 100
    # But if steps only had [50, 80] (no 100 auto-append): bisect would overflow
    # Let's use steps="50,70" -> [50,70,100]. default_charge=80 -> percent=80
    car2, _ = await _create_car(
        hass,
        home_config_entry,
        extra_config={
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER: entity,
            CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS: "50,70",
            CONF_DEFAULT_CAR_CHARGE: 80.0,
        },
        entry_id_suffix="lim_overflow2",
    )
    set_value_handler.reset_mock()

    # Ask for 90 → percent becomes max(90, default_charge=80)=90
    # steps = [50,70,100], bisect_left(90) = 2, step=100
    await car2.adapt_max_charge_limit(asked_percent=90)
    set_value_handler.assert_awaited_once()
    call = set_value_handler.call_args.args[0]
    assert call.data[number_mod.ATTR_VALUE] == 100


async def test_adapt_max_charge_limit_service_exception(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Exception calling service is caught and logged (lines 1109-1110)."""
    entity = "number.car_max_charge_exc"
    hass.states.async_set(entity, "50")

    car, _ = await _create_car(
        hass,
        home_config_entry,
        extra_config={CONF_CAR_CHARGE_PERCENT_MAX_NUMBER: entity},
        entry_id_suffix="lim_exc",
    )
    with patch(
        "homeassistant.core.ServiceRegistry.async_call",
        side_effect=RuntimeError("service fail"),
    ):
        # Should not raise
        await car.adapt_max_charge_limit(asked_percent=90)


# ===========================================================================
# update_dampening_value: transition graph return False (line 1299)
# ===========================================================================


async def test_update_dampening_transition_graph_add_fails(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Transition between two non-zero amps: _add_to_amps_power_graph returns False (line 1299)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_graphfail")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    # Set up so get_delta_dampened_power returns non-None
    car._dampening_deltas[(10, 16)] = 600.0
    car._dampening_deltas[(16, 10)] = -600.0
    car._dampening_deltas_graph = {10: {16}, 16: {10}}

    # _can_accept_new_dampen_values returns True, but _add_to_amps_power_graph returns False
    with patch.object(car, "_add_to_amps_power_graph", return_value=False):
        result = car.update_dampening_value(None, ((10, 1), (16, 1)), 580.0, time)
    assert result is False


# ===========================================================================
# update_dampening_value: amperage acceptance reject (lines 1326-1328)
# ===========================================================================


async def test_update_dampening_amperage_reject_by_acceptance(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When _can_accept_new_dampen_values rejects the amperage value (lines 1326-1328)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_ampreject")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)

    # First set a reference value for amp 10 (high value)
    car.customized_amp_to_power_1p[10] = 2300.0
    car.amp_to_power_1p[10] = 2300.0
    car.interpolate_power_steps(do_recompute_min_charge=False, use_conf_values=False)

    # Try to set a much higher value (ratio > 1.10 → rejected)
    result = car.update_dampening_value((10, 1), None, 5000.0, time)
    assert result is False


# ===========================================================================
# update_dampening_value: graph add fail for direct amperage (line 1335)
# ===========================================================================


async def test_update_dampening_amperage_graph_add_fails(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """When _add_to_amps_power_graph fails for direct amperage path (line 1335)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_ampgfail")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    with patch.object(car, "_add_to_amps_power_graph", return_value=False):
        result = car.update_dampening_value((10, 1), None, 2300.0, time)
    assert result is False


# ===========================================================================
# update_dampening_value: 3p cross-update to 1p (line 1351)
# ===========================================================================


async def test_update_dampening_1p_cross_to_3p(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """1p dampening with amp divisible by 3 cross-updates 3p table (line 1351)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="upd_1p_3p")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.get_car_charge_percent = MagicMock(return_value=50.0)

    # amp=18, 1-phase: 18 % 3 == 0, 18//3=6 which is >= min_charge(6)
    result = car.update_dampening_value((18, 1), None, 4140.0, time)
    assert result is True
    assert car.customized_amp_to_power_1p[18] == 4140.0
    assert car.customized_amp_to_power_3p[6] == 4140.0


# ===========================================================================
# user_add_default_charge_at_dt_time: can_add=False (line 1509)
# ===========================================================================


async def test_user_add_default_charge_at_dt_time_cannot_add(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """user_add_default_charge_at_dt_time returns False when can_add_default_charge is False (line 1509)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="def_dtnochg")
    car.charger = None  # no charger → can_add_default_charge() returns False

    result = await car.user_add_default_charge_at_dt_time(dt_time(8, 0))
    assert result is False


# ===========================================================================
# set_user_person_for_car with charger update (line 379)
# ===========================================================================


async def test_set_user_person_triggers_charger_update(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """set_user_person_for_car triggers charger update when car has charger (line 379)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Car Charger Person"},
        entry_id="car_chg_pers",
        title="car: Car Charger Person",
        unique_id="quiet_solar_car_chg_pers",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car = hass.data[DOMAIN].get(car_entry.entry_id)
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    data_handler.home.compute_and_set_best_persons_cars_allocations = AsyncMock(return_value={})

    charger_mock = MagicMock()
    charger_mock.update_charger_for_user_change = AsyncMock()
    car.charger = charger_mock

    mock_person = MagicMock()
    mock_person.name = "NewPerson"
    mock_person.authorized_cars = [car.name]
    data_handler.home._persons = [mock_person]
    data_handler.home.get_person_by_name = MagicMock(return_value=mock_person)

    await car.set_user_person_for_car("NewPerson")
    assert car.user_selected_person_name_for_car == "NewPerson"
    # Charger update now happens inside compute_and_set_best_persons_cars_allocations
    data_handler.home.compute_and_set_best_persons_cars_allocations.assert_awaited_once()


# ===========================================================================
# get_car_charge_energy: exception path (implied by non-float SOC)
# ===========================================================================


async def test_get_car_charge_energy_exception_in_float_conversion(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_car_charge_energy catches exception on float conversion of non-numeric SOC."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="nrg_exc")
    time = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
    car.car_battery_capacity = 60000

    # Return an object that causes float() to raise
    car.get_car_charge_percent = MagicMock(return_value=object())
    result = car.get_car_charge_energy(time)
    assert result is None


# ===========================================================================
# get_max_charge_limit: invalid state (exception path)
# ===========================================================================


async def test_get_max_charge_limit_float_state(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """get_max_charge_limit with float-like string that int() can't parse."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="maxlim_flt")
    entity = "number.test_max_charge_flt"
    car.car_charge_percent_max_number = entity
    hass.states.async_set(entity, "85.5")
    # int("85.5") raises ValueError → exception caught, returns None
    result = car.get_max_charge_limit()
    assert result is None


# ===========================================================================
# Authorization check coverage: _is_person_authorized_for_car, _fix_user_selected_person_from_forecast,
# device_post_home_init unauthorized branch, set_user_person_for_car unauthorized branch
# ===========================================================================


async def test_is_person_authorized_for_car_no_home(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """_is_person_authorized_for_car returns True when home is None (line 275)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="auth_nohome")
    car.home = None
    assert car._is_person_authorized_for_car("AnyPerson") is True


async def test_fix_user_selected_person_from_forecast_unauthorized(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """_fix_user_selected_person_from_forecast skips assignment for unauthorized person (line 288)."""
    car, _ = await _create_car(hass, home_config_entry, entry_id_suffix="fix_unauth")
    data_handler = hass.data[DOMAIN][DATA_HANDLER]

    mock_person = MagicMock()
    mock_person.name = "UnauthorizedPerson"
    mock_person.authorized_cars = []  # car not authorized
    data_handler.home._persons = [mock_person]

    car.current_forecasted_person = mock_person
    car.user_selected_person_name_for_car = None

    car._fix_user_selected_person_from_forecast()
    assert car.user_selected_person_name_for_car is None


async def test_device_post_home_init_clears_unauthorized_person(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """device_post_home_init clears user_selected_person when person is not authorized (lines 256-261)."""
    from .const import MOCK_CAR_CONFIG, MOCK_PERSON_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_unauth_init",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_unauth_init",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    # Create a person that does NOT authorize this car
    person_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            **MOCK_PERSON_CONFIG,
            "name": "Unauthorized Person",
            CONF_PERSON_AUTHORIZED_CARS: [],
            CONF_PERSON_PREFERRED_CAR: "",
        },
        entry_id="person_unauth_init",
        title="person: Unauthorized Person",
        unique_id="quiet_solar_person_unauth_init",
    )
    person_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(person_entry.entry_id)
    await hass.async_block_till_done()

    car_device = hass.data[DOMAIN].get(car_entry.entry_id)
    car_device.use_saved_extra_device_info(
        {
            "user_selected_person_name_for_car": "Unauthorized Person",
            "current_forecasted_person_name_from_boot": "Unauthorized Person",
        }
    )

    car_device.device_post_home_init(datetime.now(tz=pytz.UTC))

    # Person exists but is not authorized for this car -> cleared
    assert car_device.user_selected_person_name_for_car is None
    assert car_device.current_forecasted_person is None


async def test_set_user_person_for_car_rejects_unauthorized(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """set_user_person_for_car rejects an unauthorized person (lines 347-351)."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG, "name": "Auth Test Car"},
        entry_id="car_set_unauth",
        title="car: Auth Test Car",
        unique_id="quiet_solar_car_set_unauth",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    car = hass.data[DOMAIN].get(car_entry.entry_id)
    data_handler = hass.data[DOMAIN][DATA_HANDLER]
    data_handler.home.compute_and_set_best_persons_cars_allocations = AsyncMock(return_value={})

    mock_person = MagicMock()
    mock_person.name = "BadPerson"
    mock_person.authorized_cars = []  # car not in authorized list
    data_handler.home.get_person_by_name = MagicMock(return_value=mock_person)

    car.user_selected_person_name_for_car = None
    await car.set_user_person_for_car("BadPerson")

    # Should be rejected, value unchanged
    assert car.user_selected_person_name_for_car is None
