"""Tests for quiet_solar home history values."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import pytz
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant

from custom_components.quiet_solar.const import DATA_HANDLER, DOMAIN
from custom_components.quiet_solar.ha_model.device import HADeviceMixin
from custom_components.quiet_solar.ha_model.home import (
    BEGINING_OF_TIME,
    BUFFER_SIZE_IN_INTERVALS,
    FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
    INTERVALS_MN,
    QSHomeConsumptionHistoryAndForecast,
    QSHomeMode,
    QSSolarHistoryVals,
)
from custom_components.quiet_solar.home_model.load import AbstractLoad
from custom_components.quiet_solar.home_model.home_utils import get_average_time_series


@pytest.mark.asyncio
async def test_history_vals_save_and_read(tmp_path) -> None:
    """Test saving and reading history values."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)
    values[0][0] = 10.0
    values[1][0] = 1.0
    vals.values = values

    await vals.save_values()

    loaded = vals.read_value()
    assert np.allclose(loaded, values)

    loaded_async = await vals.read_values_async()
    assert np.allclose(loaded_async, values)


@pytest.mark.asyncio
async def test_history_vals_init_loads_existing_values(tmp_path) -> None:
    """Test init loads values and keeps history metadata."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    time_now = BEGINING_OF_TIME + timedelta(days=3, minutes=10)
    idx, days = vals.get_index_from_time(time_now)

    values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)
    values[0][idx - 1] = 20.0
    values[1][idx - 1] = days
    np.save(vals.file_path, values)

    history_start, history_end = await vals.init(time_now)

    assert history_start is None
    assert history_end is None
    assert np.allclose(vals.values, values)


@pytest.mark.asyncio
async def test_history_vals_init_bad_shape_resets(tmp_path) -> None:
    """Test init resets invalid stored shapes."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    np.save(vals.file_path, np.zeros((1, 10)))

    time_now = BEGINING_OF_TIME + timedelta(days=2)
    history_start, history_end = await vals.init(time_now)

    assert history_start is None
    assert history_end is None
    assert vals.values.shape == (2, BUFFER_SIZE_IN_INTERVALS)
    assert float(vals.values[1].sum()) == 0.0


def test_history_vals_current_interval_cache(tmp_path) -> None:
    """Test caching and flushing current interval values."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

    time_start = BEGINING_OF_TIME + timedelta(days=1, minutes=5)
    vals.add_value(time_start, 10.0)
    vals.add_value(time_start + timedelta(minutes=5), 20.0)

    assert vals.is_time_in_current_interval(time_start + timedelta(minutes=9)) is True

    from_time, interval_val = vals.get_current_interval_value()
    assert from_time is not None

    expected = get_average_time_series(
        vals._current_values,
        first_timing=from_time,
        last_timing=from_time + timedelta(minutes=INTERVALS_MN),
    )
    assert interval_val == expected

    assert vals.store_and_flush_current_vals() is True

    idx, days = vals.get_index_from_time(time_start)
    assert vals.values[1][idx] == days
    assert vals.values[0][idx] == expected

    time_out, closest_val = vals.get_closest_stored_value(time_start + timedelta(minutes=2))
    assert time_out is not None
    assert closest_val == expected


def test_history_vals_update_current_forecast_if_needed(tmp_path) -> None:
    """Test forecast update threshold logic."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    time_now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    vals._last_forecast_update_time = time_now

    assert vals.update_current_forecast_if_needed(time_now) is False
    assert vals.update_current_forecast_if_needed(
        time_now + timedelta(minutes=INTERVALS_MN - 1)
    ) is False
    assert vals.update_current_forecast_if_needed(
        time_now + timedelta(minutes=INTERVALS_MN)
    ) is True


def test_history_vals_compute_now_forecast(tmp_path) -> None:
    """Test compute_now_forecast uses current values and sets forecast."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    time_start = BEGINING_OF_TIME + timedelta(days=5, minutes=0)
    time_now = time_start + timedelta(minutes=10)
    _, now_days = vals.get_index_from_time(time_now)

    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)
    vals.values[0][:] = 50.0
    vals.values[1][:] = now_days

    vals.add_value(time_start, 42.0)

    forecast_vals = vals.compute_now_forecast(
        time_now=time_now,
        history_in_hours=2,
        future_needed_in_hours=2,
        set_as_current=True,
    )

    assert forecast_vals
    assert vals._current_forecast == forecast_vals
    assert any(val == 42.0 for _, val in forecast_vals)


def test_history_vals_add_value_extends_gaps(tmp_path) -> None:
    """Test add_value fills gaps when intervals are skipped."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

    time_start = BEGINING_OF_TIME + timedelta(days=2, minutes=0)
    vals.add_value(time_start, 10.0)

    gap_time = time_start + timedelta(minutes=INTERVALS_MN * 3)
    vals.add_value(gap_time, 10.0)

    idx_start, days = vals.get_index_from_time(time_start)
    next_idx = (idx_start + 1) % BUFFER_SIZE_IN_INTERVALS
    assert vals.values[1][idx_start] == days
    assert vals.values[0][idx_start] == 10.0
    assert vals.values[0][next_idx] == 10.0


def test_history_vals_get_closest_value_with_current_interval(tmp_path) -> None:
    """Test closest value uses current interval cache."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

    time_start = BEGINING_OF_TIME + timedelta(days=1, minutes=0)
    idx, days = vals.get_index_from_time(time_start)
    vals._current_idx = idx
    vals._current_days = days
    vals._current_values = [
        (time_start, 30.0),
        (time_start + timedelta(minutes=5), 30.0),
    ]

    query_time = time_start + timedelta(minutes=INTERVALS_MN + 1)
    time_out, value = vals.get_closest_stored_value(query_time)

    assert time_out is not None
    assert value == 30.0


def test_history_vals_compute_now_forecast_no_values(tmp_path) -> None:
    """Test compute_now_forecast returns empty without values."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    time_now = BEGINING_OF_TIME + timedelta(days=4, minutes=10)
    forecast_vals = vals.compute_now_forecast(
        time_now=time_now,
        history_in_hours=2,
        future_needed_in_hours=1,
        set_as_current=False,
    )

    assert forecast_vals == []


@pytest.mark.asyncio
async def test_history_vals_init_with_states(tmp_path) -> None:
    """Test init loads states and flushes on invalid data."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    class DummyStates:
        def get(self, entity_id):
            return SimpleNamespace(attributes={})

    class DummyHass:
        def __init__(self) -> None:
            self.states = DummyStates()

        async def async_add_executor_job(self, func, *args):
            return func(*args)

    vals.hass = DummyHass()

    time_start = BEGINING_OF_TIME + timedelta(days=2, minutes=0)
    states = [
        SimpleNamespace(last_changed=time_start, state="10"),
        SimpleNamespace(last_changed=time_start + timedelta(minutes=INTERVALS_MN * 2), state=STATE_UNKNOWN),
        SimpleNamespace(last_changed=time_start + timedelta(minutes=INTERVALS_MN * 3), state="on"),
    ]

    switch_device = SimpleNamespace(get_power_from_switch_state=lambda state: 100.0)

    with patch(
        "custom_components.quiet_solar.ha_model.home.load_from_history",
        return_value=states,
    ):
        await vals.init(time_start + timedelta(minutes=INTERVALS_MN * 3), reset_for_switch_device=switch_device)

@pytest.mark.asyncio
async def test_forecast_combine_and_light_reset(tmp_path) -> None:
    """Test combine helper and light reset path."""
    forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path=str(tmp_path))

    val1 = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 0.0, 1.0]])
    val2 = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.0, 1.0]])

    combined_add = forecast._combine_stored_forecast_values(val1, val2, do_add=True)
    assert combined_add[0][0] == 11.0
    assert combined_add[0][3] == 44.0
    assert combined_add[1][0] == 1.0
    assert combined_add[1][3] == 1.0

    combined_sub = forecast._combine_stored_forecast_values(val1, val2, do_add=False)
    assert combined_sub[0][0] == 9.0
    assert combined_sub[0][3] == 36.0

    time_now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    result = await forecast.reset_forecasts(time_now, light_reset=True)
    assert result is True

    expected_file = tmp_path / f"{FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER}.npy"
    assert expected_file.exists()


@pytest.mark.asyncio
async def test_forecast_reset_full_path(
    hass: HomeAssistant, home_config_entry, tmp_path
) -> None:
    """Test full reset_forecasts path with real history values."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    home = hass.data[DOMAIN][DATA_HANDLER].home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value

    home.physical_battery = SimpleNamespace(charge_discharge_sensor="sensor.battery_power")
    home.physical_solar_plant = SimpleNamespace(
        solar_inverter_active_power="sensor.solar_power",
        solar_inverter_input_active_power=None,
    )
    home.grid_active_power_sensor = "sensor.grid_power"
    home.grid_active_power_sensor_inverted = False
    home._childrens = []

    class DummyLoad(HADeviceMixin, AbstractLoad):
        def __init__(self, hass_ref: HomeAssistant) -> None:
            self.hass = hass_ref
            self.name = "Dummy load"
            self._enabled = True
            self.devices_to_pilot = []
            self.load_is_auto_to_be_boosted = False
            self.switch_entity = None
            self.dynamic_group_name = None
            self.piloted_device_name = None

        def get_best_power_HA_entity(self):
            return "sensor.load_power"

        def register_all_on_change_states(self) -> None:
            return None

        def get_attached_virtual_devices(self):
            return []

    for entity_id in (
        "sensor.battery_power",
        "sensor.solar_power",
        "sensor.grid_power",
        "sensor.load_power",
        FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
    ):
        hass.states.async_set(entity_id, "100", {"unit_of_measurement": "W"})

    time_now = datetime(2026, 1, 15, 8, 0, tzinfo=pytz.UTC)
    time_start = time_now - timedelta(hours=2)
    states = [
        SimpleNamespace(last_changed=time_start, state="100"),
        SimpleNamespace(last_changed=time_start + timedelta(minutes=30), state="110"),
    ]

    dummy_load = DummyLoad(hass)
    home._childrens = [dummy_load]

    forecast = QSHomeConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
    forecast._all_loads = [dummy_load]

    with patch(
        "custom_components.quiet_solar.ha_model.home.load_from_history",
        return_value=states,
    ):
        result = await forecast.reset_forecasts(time_now, light_reset=False)

    assert result is True


@pytest.mark.asyncio
async def test_forecast_reset_with_input_active_power(
    hass: HomeAssistant, home_config_entry, tmp_path
) -> None:
    """Test reset_forecasts using solar input active power."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    home = hass.data[DOMAIN][DATA_HANDLER].home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value

    home.physical_battery = SimpleNamespace(charge_discharge_sensor="sensor.battery_power", is_dc_coupled=True)
    home.physical_solar_plant = SimpleNamespace(
        solar_inverter_active_power=None,
        solar_inverter_input_active_power="sensor.solar_input_power",
        solar_max_output_power_value=1000.0,
    )
    home.grid_active_power_sensor = "sensor.grid_power"
    home.grid_active_power_sensor_inverted = False
    home._childrens = []

    for entity_id in (
        "sensor.battery_power",
        "sensor.solar_input_power",
        "sensor.grid_power",
        FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
    ):
        hass.states.async_set(entity_id, "120", {"unit_of_measurement": "W"})

    time_now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)
    time_start = time_now - timedelta(hours=2)
    states = [
        SimpleNamespace(last_changed=time_start, state="120"),
        SimpleNamespace(last_changed=time_start + timedelta(minutes=30), state="130"),
    ]

    forecast = QSHomeConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
    forecast._all_loads = []

    with patch(
        "custom_components.quiet_solar.ha_model.home.load_from_history",
        return_value=states,
    ):
        result = await forecast.reset_forecasts(time_now, light_reset=False)

    assert result is True


@pytest.mark.asyncio
async def test_history_vals_init_with_invalid_states(
    hass: HomeAssistant, tmp_path
) -> None:
    """Test init handles invalid and out-of-range states."""
    forecast = SimpleNamespace(home=SimpleNamespace(hass=hass), storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")

    time_now = BEGINING_OF_TIME + timedelta(days=2, hours=2)
    early_time = time_now - timedelta(days=BUFFER_SIZE_IN_INTERVALS)
    future_time = time_now + timedelta(minutes=30)
    valid_time = time_now - timedelta(hours=1)

    states = [
        SimpleNamespace(last_changed=valid_time, state="10"),
        None,
        SimpleNamespace(last_changed=early_time, state="5"),
        SimpleNamespace(last_changed=future_time, state="5"),
        SimpleNamespace(last_changed=valid_time + timedelta(minutes=INTERVALS_MN), state=STATE_UNKNOWN),
        SimpleNamespace(last_changed=valid_time + timedelta(minutes=INTERVALS_MN * 2), state="bad"),
    ]

    with patch(
        "custom_components.quiet_solar.ha_model.home.load_from_history",
        return_value=states,
    ):
        await vals.init(time_now)


def test_get_time_from_state_edge_cases() -> None:
    """Test get_time_from_state returns None for invalid inputs."""
    from custom_components.quiet_solar.ha_model.home import get_time_from_state

    assert get_time_from_state(None) is None

    state_unknown_attr = SimpleNamespace(
        last_updated=datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC),
        attributes={"last_updated": "unknown"},
    )
    assert get_time_from_state(state_unknown_attr) == state_unknown_attr.last_updated

    state_invalid_string = SimpleNamespace(
        last_updated="not-a-date",
        attributes=None,
    )
    assert get_time_from_state(state_invalid_string) is None


def test_history_vals_compute_now_forecast_no_prev_val(tmp_path) -> None:
    """Test compute_now_forecast returns empty without previous values."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

    time_now = BEGINING_OF_TIME + timedelta(days=3, minutes=1)

    with (
        patch.object(vals, "_get_possible_now_val_for_forcast", return_value=None),
        patch.object(vals, "_get_possible_past_consumption_for_forecast", return_value=[1]),
        patch.object(
            vals,
            "_get_predicted_data",
            return_value=(np.array([1.0, 2.0]), np.array([0, 0])),
        ),
        patch.object(vals, "get_utc_time_from_index", return_value=time_now + timedelta(minutes=1)),
    ):
        result = vals.compute_now_forecast(
            time_now=time_now,
            history_in_hours=1,
            future_needed_in_hours=1,
            set_as_current=False,
        )

    assert result == []


def test_history_vals_compute_now_forecast_prev_value_fill(tmp_path) -> None:
    """Test compute_now_forecast fills missing previous value."""
    forecast = SimpleNamespace(home=None, hass=None, storage_path=str(tmp_path))
    vals = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_power")
    vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

    time_now = BEGINING_OF_TIME + timedelta(days=3, minutes=10)
    now_idx, now_days = vals.get_index_from_time(time_now)

    forecast_values = np.array([5.0, 6.0])
    past_days = np.array([now_days, now_days])

    with (
        patch.object(vals, "_get_possible_now_val_for_forcast", return_value=None),
        patch.object(vals, "_get_possible_past_consumption_for_forecast", return_value=[1]),
        patch.object(vals, "_get_predicted_data", return_value=(forecast_values, past_days)),
        patch.object(vals, "get_utc_time_from_index", return_value=time_now),
        patch.object(vals, "get_index_from_time", return_value=(now_idx, now_days)),
    ):
        result = vals.compute_now_forecast(
            time_now=time_now,
            history_in_hours=1,
            future_needed_in_hours=2,
            set_as_current=False,
        )

    assert result
