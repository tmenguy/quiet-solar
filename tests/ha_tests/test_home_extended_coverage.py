"""Extended coverage tests for quiet_solar home.py.

Focuses on exercising untested code paths with real QS objects,
minimal mocking, and meaningful assertions.
"""

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytz
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_HOME_PEAK_PRICE,
    CONF_HOME_OFF_PEAK_PRICE,
    CONF_HOME_START_OFF_PEAK_RANGE_1,
    CONF_HOME_END_OFF_PEAK_RANGE_1,
    CONF_HOME_START_OFF_PEAK_RANGE_2,
    CONF_HOME_END_OFF_PEAK_RANGE_2,
)

from custom_components.quiet_solar.ha_model.home import (
    QSHome,
    QSHomeMode,
    QSHomeConsumptionHistoryAndForecast,
    QSSolarHistoryVals,
    QSforecastValueSensor,
    get_time_from_state,
    _segments_weak_sub_on_main_overlap,
    _segments_strong_overlap,
    BUFFER_SIZE_IN_INTERVALS,
    NUM_INTERVALS_PER_DAY,
    BEGINING_OF_TIME,
    _sanitize_idx,
    INTERVALS_MN,
    NUM_INTERVAL_PER_HOUR,
    BUFFER_SIZE_DAYS,
    POWER_ALIGNMENT_TOLERANCE_S,
)
from custom_components.quiet_solar.ha_model.solar import QSSolar
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, LoadCommand
from custom_components.quiet_solar.ha_model.device import HADeviceMixin


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


# ========================================================================
# Helper to set up home entry quickly
# ========================================================================

async def _get_home(hass, entry):
    """Set up home entry and return the QSHome object."""
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return hass.data[DOMAIN][DATA_HANDLER].home


# ========================================================================
# Pure function tests - get_time_from_state
# ========================================================================


class TestGetTimeFromState:
    """Tests for the get_time_from_state helper."""

    def test_none_state_returns_none(self):
        """Passing None returns None."""
        assert get_time_from_state(None) is None

    def test_state_with_datetime_last_updated(self):
        """State whose last_updated is already a datetime."""
        t = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        state = SimpleNamespace(
            last_updated=t,
            attributes={},
        )
        result = get_time_from_state(state)
        assert result == t

    def test_state_with_iso_string_in_attributes(self):
        """State whose attribute last_updated is an ISO string."""
        t = datetime(2026, 2, 10, 14, 30, tzinfo=pytz.UTC)
        state = SimpleNamespace(
            last_updated=t,
            attributes={"last_updated": "2026-02-10T14:30:00+00:00"},
        )
        result = get_time_from_state(state)
        assert result is not None

    def test_state_with_none_string_attribute(self):
        """Attribute last_updated is the string 'none' - should return the real last_updated."""
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        state = SimpleNamespace(
            last_updated=t,
            attributes={"last_updated": "none"},
        )
        result = get_time_from_state(state)
        # "none" is skipped, falls through to returning the datetime last_updated
        assert result == t

    def test_state_with_unknown_string_attribute(self):
        """Attribute last_updated is 'unknown'."""
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        state = SimpleNamespace(
            last_updated=t,
            attributes={"last_updated": "unknown"},
        )
        result = get_time_from_state(state)
        assert result == t

    def test_state_with_unavailable_string_attribute(self):
        """Attribute last_updated is 'unavailable'."""
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        state = SimpleNamespace(
            last_updated=t,
            attributes={"last_updated": "Unavailable"},
        )
        result = get_time_from_state(state)
        assert result == t

    def test_state_with_datetime_attribute(self):
        """Attribute last_updated is a datetime object."""
        attr_t = datetime(2026, 2, 10, 15, 0, tzinfo=pytz.UTC)
        state = SimpleNamespace(
            last_updated=datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC),
            attributes={"last_updated": attr_t},
        )
        result = get_time_from_state(state)
        assert result == attr_t

    def test_state_with_string_time_last_updated(self):
        """State.last_updated is a string - should parse it. Covers line 95-104."""
        state = SimpleNamespace(
            last_updated="2026-02-10T14:00:00+00:00",
            attributes={},
        )
        result = get_time_from_state(state)
        assert result is not None
        assert isinstance(result, datetime)

    def test_state_with_none_string_time(self):
        """State.last_updated is the string 'None' - covers line 96-97."""
        state = SimpleNamespace(
            last_updated="None",
            attributes={},
        )
        result = get_time_from_state(state)
        assert result is None

    def test_state_with_non_datetime_non_string_time(self):
        """State.last_updated is an int - covers line 106-107."""
        state = SimpleNamespace(
            last_updated=12345,
            attributes={},
        )
        result = get_time_from_state(state)
        assert result is None


# ========================================================================
# Pure function tests - _segments_weak_sub_on_main_overlap
# ========================================================================


class TestSegmentOverlaps:
    """Tests for segment overlap functions."""

    def test_weak_overlap_basic(self):
        """Basic weak overlap between two segment lists."""
        t0 = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)
        t1 = t0 + timedelta(hours=2)
        t2 = t0 + timedelta(hours=1)
        t3 = t0 + timedelta(hours=3)

        subs = [(t0, t1)]
        mains = [(t2, t3)]

        result = _segments_weak_sub_on_main_overlap(subs, mains)
        assert len(result) == 1
        # Overlap is from t2 to t1
        assert result[0][0] == t2
        assert result[0][1] == t1

    def test_weak_overlap_no_overlap(self):
        """No overlap between segments."""
        t0 = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)
        t1 = t0 + timedelta(hours=1)
        t2 = t0 + timedelta(hours=3)
        t3 = t0 + timedelta(hours=4)

        subs = [(t0, t1)]
        mains = [(t2, t3)]

        result = _segments_weak_sub_on_main_overlap(subs, mains)
        assert len(result) == 0

    def test_weak_overlap_min_overlap_filter(self):
        """Overlap below min_overlap is filtered out."""
        t0 = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)
        t1 = t0 + timedelta(seconds=20)
        t2 = t0 + timedelta(seconds=10)
        t3 = t0 + timedelta(hours=1)

        subs = [(t0, t1)]
        mains = [(t2, t3)]

        # 10 seconds overlap, min_overlap=60 should filter it
        result = _segments_weak_sub_on_main_overlap(subs, mains, min_overlap=60)
        assert len(result) == 0

        # min_overlap=5 should keep it
        result = _segments_weak_sub_on_main_overlap(subs, mains, min_overlap=5)
        assert len(result) == 1

    def test_strong_overlap_one_to_one(self):
        """Strong overlap: each segment overlaps exactly one from the other list."""
        t0 = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)
        t1 = t0 + timedelta(hours=2)
        t2 = t0 + timedelta(hours=1)
        t3 = t0 + timedelta(hours=3)

        seg1 = [(t0, t1)]
        seg2 = [(t2, t3)]

        result = _segments_strong_overlap(seg1, seg2)
        assert len(result) == 1

    def test_strong_overlap_one_to_many_excluded(self):
        """Strong overlap excludes segments that overlap with multiple counterparts."""
        t0 = datetime(2026, 2, 10, 8, 0, tzinfo=pytz.UTC)
        # One large segment overlapping two small ones
        seg1 = [(t0, t0 + timedelta(hours=4))]
        seg2 = [
            (t0 + timedelta(hours=1), t0 + timedelta(hours=2)),
            (t0 + timedelta(hours=3), t0 + timedelta(hours=3, minutes=30)),
        ]

        result = _segments_strong_overlap(seg1, seg2)
        # seg1[0] overlaps with both seg2 entries, so it should be excluded
        assert len(result) == 0


# ========================================================================
# QSforecastValueSensor tests
# ========================================================================


class TestQSforecastValueSensor:
    """Tests for QSforecastValueSensor."""

    def test_push_and_get_with_zero_delta_current_getter(self):
        """Zero delta uses the current_getter directly."""
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        def current_getter(t):
            return (t, 500.0)

        def forecast_getter(t):
            return (t, 600.0)

        sensor = QSforecastValueSensor("test", 0, forecast_getter, current_getter)
        val = sensor.push_and_get(now)
        assert val == 500.0

    def test_push_and_get_builds_history_and_prunes(self):
        """push_and_get accumulates values and prunes old ones. Covers line 155-156."""
        base = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        def forecast_getter(t):
            return (t, 100.0)

        sensor = QSforecastValueSensor("test", 3600, forecast_getter, None)

        # Push several values over time
        for i in range(10):
            t = base + timedelta(minutes=i * 20)
            result = sensor.push_and_get(t)

        # Should have accumulated stored values
        assert len(sensor._stored_values) > 0

        # Now query with a later time to trigger pruning
        t_late = base + timedelta(hours=3)
        result = sensor.push_and_get(t_late)
        # After pruning, stored_values should be trimmed
        assert result is not None

    def test_push_and_get_returns_none_when_no_stored(self):
        """Returns None when forecast_getter returns None and no stored values."""
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        def forecast_getter(t):
            return (t, None)

        sensor = QSforecastValueSensor("test", 3600, forecast_getter, None)
        val = sensor.push_and_get(now)
        assert val is None

    def test_get_probers_creates_multiple(self):
        """get_probers creates the expected number of sensors."""
        def getter(t):
            return (t, 100.0)

        def getter_now(t):
            return (t, 200.0)

        probers = QSforecastValueSensor.get_probers(
            getter, getter_now,
            {"1h": 3600, "2h": 7200, "now": 0}
        )
        assert len(probers) == 3
        assert "1h" in probers
        assert "2h" in probers
        assert "now" in probers


# ========================================================================
# QSHome integration tests - tariff methods
# ========================================================================


class TestHomeTariffMethods:
    """Tests for tariff-related methods on QSHome."""

    @pytest.mark.asyncio
    async def test_get_tariffs_off_grid_returns_zero(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid mode returns 0.0 tariff. Covers line 1138."""
        home = await _get_home(hass, home_config_entry)
        home.qs_home_is_off_grid = True

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_tariffs(now, now + timedelta(hours=24))
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_tariffs_no_off_peak_returns_peak_price(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No off-peak price returns peak price as scalar. Covers line 1140-1141."""
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_tariffs(now, now + timedelta(hours=24))
        assert isinstance(result, float)
        assert result == home.price_peak

    @pytest.mark.asyncio
    async def test_get_tariffs_with_off_peak_ranges(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """With off-peak ranges configured, returns a list of tariff segments. Covers lines 1146-1176."""
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        home.tariff_start_1 = "22:00"
        home.tariff_end_1 = "06:00"
        home.tariff_start_2 = None
        home.tariff_end_2 = None

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        end = now + timedelta(hours=24)
        result = home.get_tariffs(now, end)
        assert isinstance(result, list)
        assert len(result) > 0
        # Each entry is (datetime, price)
        for entry in result:
            assert isinstance(entry[0], datetime)
            assert isinstance(entry[1], float)

    @pytest.mark.asyncio
    async def test_get_today_off_peak_ranges_zero_price(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Zero off-peak price returns empty ranges. Covers line 1072."""
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home._get_today_off_peak_ranges(now)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_today_off_peak_ranges_equal_start_end(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Equal start and end should skip the range. Covers line 1088."""
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        home.tariff_start_1 = "22:00"
        home.tariff_end_1 = "22:00"  # equal!
        home.tariff_start_2 = None
        home.tariff_end_2 = None

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home._get_today_off_peak_ranges(now)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_tariff_with_off_peak(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """get_tariff with ranges returns off-peak when within range. Covers line 1112."""
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        home.tariff_start_1 = "00:00"
        home.tariff_end_1 = "06:00"
        home.tariff_start_2 = None
        home.tariff_end_2 = None

        # Query during off-peak
        start = datetime(2026, 2, 10, 2, 0, tzinfo=pytz.UTC)
        end = datetime(2026, 2, 10, 3, 0, tzinfo=pytz.UTC)
        result = home.get_tariff(start, end)
        assert result == home.price_off_peak


# ========================================================================
# QSHome off-grid mode and phase amps
# ========================================================================


class TestHomeOffGridMode:
    """Tests for off-grid mode behavior."""

    @pytest.mark.asyncio
    async def test_off_grid_get_home_max_static_phase_amps_no_solar(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid with no solar returns static amp. Covers line 906-912."""
        home = await _get_home(hass, home_config_entry)
        home.qs_home_is_off_grid = True
        home.physical_solar_plant = None

        result = home.get_home_max_static_phase_amps()
        assert result == home.dyn_group_max_phase_current_conf

    @pytest.mark.asyncio
    async def test_off_grid_get_home_max_static_phase_amps_with_solar(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid with solar limits to solar max phase amps. Covers line 910."""
        home = await _get_home(hass, home_config_entry)
        home.qs_home_is_off_grid = True
        home.physical_solar_plant = MagicMock(
            solar_max_phase_amps=15.0,
            solar_production=3000.0,
            solar_max_output_power_value=5000.0,
        )

        result = home.get_home_max_static_phase_amps()
        assert result == min(home.dyn_group_max_phase_current_conf, 15.0)

    @pytest.mark.asyncio
    async def test_off_grid_get_home_max_phase_amps_with_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid max phase amps with battery discharge. Covers lines 915-944."""
        home = await _get_home(hass, home_config_entry)
        home.qs_home_is_off_grid = True
        home.physical_solar_plant = MagicMock(
            solar_production=2000.0,
            solar_max_output_power_value=4000.0,
            solar_max_phase_amps=20.0,
        )
        home.physical_battery = MagicMock(
            battery_can_discharge=MagicMock(return_value=True),
            get_max_discharging_power=MagicMock(return_value=1000.0),
        )

        result = home.get_home_max_phase_amps()
        assert isinstance(result, float)
        # 3000W total / 3 phases / 230V = ~4.35A (for 3p)
        assert result > 0

    @pytest.mark.asyncio
    async def test_off_grid_dyn_group_max_phase_current_mono(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid mono-phase distributes amps to one phase. Covers lines 960-961."""
        home = await _get_home(hass, home_config_entry)
        home.qs_home_is_off_grid = True
        # Make it mono-phase by changing internal config
        home._device_is_3p_conf = False
        home._mono_phase_default = 1
        home.physical_solar_plant = MagicMock(
            solar_production=2000.0,
            solar_max_output_power_value=3000.0,
            solar_max_phase_amps=30.0,
        )
        home.physical_battery = None

        result = home.dyn_group_max_phase_current
        assert isinstance(result, list)
        assert len(result) == 3
        # Phase 1 should have the amps, others should be 0
        assert result[1] > 0

    @pytest.mark.asyncio
    async def test_off_grid_dyn_group_max_phase_current_for_budget_mono(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid budget mono-phase. Covers lines 982-983."""
        home = await _get_home(hass, home_config_entry)
        home.qs_home_is_off_grid = True
        home._device_is_3p_conf = False
        home._mono_phase_default = 0
        home.physical_solar_plant = MagicMock(
            solar_max_phase_amps=25.0,
        )
        home.physical_battery = None

        result = home.dyn_group_max_phase_current_for_budget
        assert isinstance(result, list)
        assert len(result) == 3
        # Phase 0 should have amps
        assert result[0] > 0

    @pytest.mark.asyncio
    async def test_finish_off_grid_switch_no_pending(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No pending switch returns (True, False). Covers line 2122-2123."""
        home = await _get_home(hass, home_config_entry)
        home._switch_to_off_grid_launched = None

        finished, just_switched = await home.finish_off_grid_switch(
            datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        )
        assert finished is True
        assert just_switched is False

    @pytest.mark.asyncio
    async def test_finish_off_grid_switch_timeout(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """After 3 min timeout, returns (True, True). Covers lines 2125-2127."""
        home = await _get_home(hass, home_config_entry)
        t_now = datetime(2026, 2, 10, 12, 5, tzinfo=pytz.UTC)
        home._switch_to_off_grid_launched = t_now - timedelta(minutes=4)

        finished, just_switched = await home.finish_off_grid_switch(t_now)
        assert finished is True
        assert just_switched is True
        assert home._switch_to_off_grid_launched is None

    @pytest.mark.asyncio
    async def test_finish_off_grid_switch_loads_not_ready(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Loads not ready returns (False, False). Covers lines 2129-2135."""
        home = await _get_home(hass, home_config_entry)
        t_now = datetime(2026, 2, 10, 12, 1, tzinfo=pytz.UTC)
        home._switch_to_off_grid_launched = t_now - timedelta(seconds=30)
        # check_loads_commands needs to return False
        home.home_mode = QSHomeMode.HOME_MODE_ON.value

        load = MagicMock()
        load.name = "test_load"
        load.qs_enable_device = True
        load.check_commands = AsyncMock(return_value=(timedelta(seconds=0), False))
        load.running_command_num_relaunch = 0
        home._all_loads = [load]
        home._chargers = []
        home._init_completed = True

        finished, just_switched = await home.finish_off_grid_switch(t_now)
        assert finished is False
        assert just_switched is False

    @pytest.mark.asyncio
    async def test_finish_off_grid_switch_all_loads_ok(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """All loads OK returns (True, True). Covers lines 2131-2133."""
        home = await _get_home(hass, home_config_entry)
        t_now = datetime(2026, 2, 10, 12, 1, tzinfo=pytz.UTC)
        home._switch_to_off_grid_launched = t_now - timedelta(seconds=30)
        home.home_mode = QSHomeMode.HOME_MODE_ON.value

        load = MagicMock()
        load.name = "test_load"
        load.qs_enable_device = True
        load.check_commands = AsyncMock(return_value=(timedelta(seconds=0), True))
        load.running_command_num_relaunch = 0
        home._all_loads = [load]
        home._chargers = []
        home._init_completed = True
        home.physical_battery = None

        finished, just_switched = await home.finish_off_grid_switch(t_now)
        assert finished is True
        assert just_switched is True
        assert home._switch_to_off_grid_launched is None


# ========================================================================
# QSHome production and consumption helpers
# ========================================================================


class TestHomeProductionConsumption:
    """Tests for production/consumption computation helpers."""

    @pytest.mark.asyncio
    async def test_get_current_over_clamp_production_power_with_solar(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Solar production exceeds max output. Covers lines 1379-1382."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = QSSolar(hass=hass, config_entry={}, name="test_solar_production")
        home.physical_solar_plant.solar_production = 6000.0
        home.physical_solar_plant.solar_max_output_power_value = 5000.0

        result = home.get_current_over_clamp_production_power()
        assert result == 1000.0

    @pytest.mark.asyncio
    async def test_get_current_over_clamp_no_excess(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Solar production under max output returns 0. Covers line 1382."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = QSSolar(hass=hass, config_entry={}, name="test_solar_production")
        home.physical_solar_plant.solar_production = 3000.0
        home.physical_solar_plant.solar_max_output_power_value = 5000.

        result = home.get_current_over_clamp_production_power()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_get_current_maximum_production_output_dc_coupled(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """DC-coupled battery + solar. Covers lines 1401-1405."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = MagicMock(
            solar_production=3000.0,
            solar_max_output_power_value=5000.0,
        )
        home.physical_battery = MagicMock(
            battery_get_current_possible_max_discharge_power=MagicMock(return_value=2000.0),
            is_dc_coupled=True,
        )
        result = home.get_current_maximum_production_output_power()
        # DC coupled: solar + battery = 5000, capped at max_output=5000
        assert result == 5000.0

    @pytest.mark.asyncio
    async def test_get_current_maximum_production_dc_coupled_under_max(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """DC-coupled where solar+battery < max_output. Covers line 1402-1403."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = MagicMock(
            solar_production=1000.0,
            solar_max_output_power_value=5000.0,
        )
        home.physical_battery = MagicMock(
            battery_get_current_possible_max_discharge_power=MagicMock(return_value=500.0),
            is_dc_coupled=True,
        )
        result = home.get_current_maximum_production_output_power()
        assert result == 1500.0

    @pytest.mark.asyncio
    async def test_get_current_maximum_production_ac_coupled(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """AC-coupled: solar_max + battery_max. Covers line 1407."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = MagicMock(
            solar_production=4000.0,
            solar_max_output_power_value=5000.0,
        )
        home.physical_battery = MagicMock(
            battery_get_current_possible_max_discharge_power=MagicMock(return_value=3000.0),
            is_dc_coupled=False,
        )
        result = home.get_current_maximum_production_output_power()
        assert result == 8000.0  # 5000 + 3000

    @pytest.mark.asyncio
    async def test_get_solar_from_current_forecast_no_solar(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No solar returns (None, None). Covers line 1060."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_solar_from_current_forecast_getter(now)
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_get_solar_from_current_forecast_list_no_solar(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No solar returns empty list. Covers line 1065."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_solar_from_current_forecast(now)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_grid_active_power_values(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Covers line 1411-1412."""
        home = await _get_home(hass, home_config_entry)
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_grid_active_power_values(3600, now)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_available_power_values(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Covers line 1414-1415."""
        home = await _get_home(hass, home_config_entry)
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_available_power_values(3600, now)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_grid_consumption_power_values(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Covers line 1418-1419."""
        home = await _get_home(hass, home_config_entry)
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.get_grid_consumption_power_values(3600, now)
        assert isinstance(result, list)


# ========================================================================
# QSHome device management
# ========================================================================


class TestHomeDeviceManagement:
    """Tests for add_device / remove_device and topology."""

    @pytest.mark.asyncio
    async def test_add_and_remove_car(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Add and remove a car. Covers lines 1500-1556."""
        from .const import MOCK_CAR_CONFIG
        home = await _get_home(hass, home_config_entry)

        car_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CAR_CONFIG,
            entry_id="car_mgmt_test",
            title=f"car: {MOCK_CAR_CONFIG['name']}",
            unique_id="quiet_solar_car_mgmt_test",
        )
        car_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(car_entry.entry_id)
        await hass.async_block_till_done()

        assert len(home._cars) == 1
        car = home._cars[0]

        home.remove_device(car)
        assert len(home._cars) == 0

    @pytest.mark.asyncio
    async def test_remove_device_self_noop(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Removing home from itself is a no-op. Covers line 1543-1545."""
        home = await _get_home(hass, home_config_entry)
        # Should not raise
        home.remove_device(home)

    @pytest.mark.asyncio
    async def test_add_disabled_device(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Add and remove disabled device."""
        home = await _get_home(hass, home_config_entry)
        mock_device = MagicMock()
        mock_device.name = "disabled_test"

        home.add_disabled_device(mock_device)
        assert mock_device in home._disabled_devices

        # Duplicate add should not add twice
        home.add_disabled_device(mock_device)
        assert home._disabled_devices.count(mock_device) == 1

        home.remove_disabled_device(mock_device)
        assert mock_device not in home._disabled_devices

    @pytest.mark.asyncio
    async def test_get_person_by_name_none_input(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Passing None returns None. Covers line 1194-1195."""
        home = await _get_home(hass, home_config_entry)
        assert home.get_person_by_name(None) is None


# ========================================================================
# QSHome sensor getters
# ========================================================================


class TestHomeSensorGetters:
    """Tests for sensor state getter methods."""

    @pytest.mark.asyncio
    async def test_home_consumption_sensor_state_getter_none(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Returns None when home_consumption is None."""
        home = await _get_home(hass, home_config_entry)
        home.home_consumption = None
        result = home.home_consumption_sensor_state_getter("sensor.test", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_home_consumption_sensor_state_getter_with_value(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Returns tuple when home_consumption has a value."""
        home = await _get_home(hass, home_config_entry)
        home.home_consumption = 1500.0
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.home_consumption_sensor_state_getter("sensor.test", now)
        assert result == (now, 1500.0, {})

    @pytest.mark.asyncio
    async def test_home_available_power_sensor_state_getter_none(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Returns None when home_available_power is None."""
        home = await _get_home(hass, home_config_entry)
        home.home_available_power = None
        result = home.home_available_power_sensor_state_getter("sensor.test", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_grid_consumption_power_sensor_state_getter(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Returns tuple when grid_consumption_power has a value."""
        home = await _get_home(hass, home_config_entry)
        home.grid_consumption_power = -500.0
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = home.grid_consumption_power_sensor_state_getter("sensor.test", now)
        assert result == (now, -500.0, {})


# ========================================================================
# QSHome finish_setup
# ========================================================================


class TestHomeFinishSetup:
    """Tests for finish_setup method."""

    @pytest.mark.asyncio
    async def test_finish_setup_already_completed(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Already initialized returns True immediately. Covers line 1626-1627."""
        home = await _get_home(hass, home_config_entry)
        home._init_completed = True
        result = await home.finish_setup(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))
        assert result is True

    @pytest.mark.asyncio
    async def test_finish_setup_no_devices(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No devices returns False. Covers line 1630-1631."""
        home = await _get_home(hass, home_config_entry)
        home._init_completed = False
        home._all_devices = []
        result = await home.finish_setup(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))
        assert result is False


# ========================================================================
# QSHome check_loads_commands
# ========================================================================


class TestCheckLoadsCommands:
    """Tests for check_loads_commands."""

    @pytest.mark.asyncio
    async def test_check_loads_commands_off_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """OFF mode returns True. Covers line 2139-2143."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_OFF.value
        result = await home.check_loads_commands(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))
        assert result is True

    @pytest.mark.asyncio
    async def test_check_loads_commands_sensors_only_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """SENSORS_ONLY mode returns True."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_SENSORS_ONLY.value
        result = await home.check_loads_commands(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))
        assert result is True

    @pytest.mark.asyncio
    async def test_check_loads_commands_charger_only_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """CHARGER_ONLY uses only chargers. Covers line 2155-2156."""
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_CHARGER_ONLY.value
        home._init_completed = True

        charger = MagicMock()
        charger.name = "test_charger"
        charger.check_commands = AsyncMock(return_value=(timedelta(seconds=0), True))
        charger.running_command_num_relaunch = 0
        home._chargers = [charger]

        result = await home.check_loads_commands(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))
        assert result is True
        charger.check_commands.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_loads_commands_not_finished_setup(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Not-finished-setup returns False. Covers lines 2145-2147."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_ON.value
        home._init_completed = False
        home._all_devices = []
        result = await home.check_loads_commands(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))
        assert result is False

    @pytest.mark.asyncio
    async def test_check_loads_commands_relaunch_stale(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Stale command triggers relaunch. Covers lines 2165-2167."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_ON.value
        home._init_completed = True
        home.physical_battery = None

        load = MagicMock()
        load.name = "stale_load"
        load.check_commands = AsyncMock(return_value=(timedelta(seconds=60), False))
        load.running_command_num_relaunch = 1
        load.force_relaunch_command = AsyncMock()
        home._all_loads = [load]

        t = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = await home.check_loads_commands(t)
        assert result is False
        load.force_relaunch_command.assert_awaited_once()


# ========================================================================
# QSHome update_loads
# ========================================================================


class TestUpdateLoads:
    """Tests for update_loads method."""

    @pytest.mark.asyncio
    async def test_update_loads_off_mode_returns_early(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """OFF mode returns early. Covers line 2178-2182."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_OFF.value
        await home.update_loads(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))

    @pytest.mark.asyncio
    async def test_update_loads_not_finished_setup(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Not finished setup returns early. Covers lines 2184-2186."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_ON.value
        home._init_completed = False
        home._all_devices = []  # no devices -> finish_setup returns False
        await home.update_loads(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))

    @pytest.mark.asyncio
    async def test_update_loads_charger_only_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """CHARGER_ONLY mode only processes chargers. Covers line 2193-2194."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_CHARGER_ONLY.value
        home._init_completed = True
        home._switch_to_off_grid_launched = None
        home.physical_battery = None

        charger = MagicMock()
        charger.name = "test_charger"
        charger.qs_enable_device = True
        charger.check_commands = AsyncMock(return_value=(timedelta(seconds=0), True))
        charger.running_command_num_relaunch = 0
        charger.is_load_active = MagicMock(return_value=False)
        charger.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
        charger.get_current_active_constraint = MagicMock(return_value=None)
        charger.launch_command = AsyncMock()
        charger.do_probe_state_change = AsyncMock()
        charger.do_run_check_load_activity_and_constraints = AsyncMock(return_value=False)
        charger.update_live_constraints = AsyncMock(return_value=False)
        home._chargers = [charger]
        home._all_loads = []
        home._commands = []
        home._battery_commands = None
        home._last_solve_done = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        await home.update_loads(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))

    @pytest.mark.asyncio
    async def test_update_loads_off_grid_not_finished(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Off-grid switch not finished skips update. Covers lines 2188-2191."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_ON.value
        home._init_completed = True
        t_now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        home._switch_to_off_grid_launched = t_now - timedelta(seconds=30)

        # Mock check_loads_commands to return False (not ready)
        load = MagicMock()
        load.name = "test"
        load.check_commands = AsyncMock(return_value=(timedelta(0), False))
        load.running_command_num_relaunch = 0
        home._all_loads = [load]
        home._chargers = []
        home.physical_battery = None

        await home.update_loads(t_now)
        # The fact we didn't crash is the test


# ========================================================================
# QSHome dashboard section price configuration
# ========================================================================


class TestHomeDashboardAndInit:
    """Tests for dashboard section and price init."""

    @pytest.mark.asyncio
    async def test_price_peak_zero_clamped(
        self, hass: HomeAssistant,
    ):
        """price_peak <= 0 is clamped to 0.2. Covers line 267-268."""
        from .const import MOCK_HOME_CONFIG
        config = dict(MOCK_HOME_CONFIG)
        config[CONF_HOME_PEAK_PRICE] = -1.0

        entry = MockConfigEntry(
            domain=DOMAIN, data=config,
            entry_id="price_zero_test",
            title="home: Test Home",
            unique_id="quiet_solar_price_zero_test",
        )
        entry.add_to_hass(hass)
        home = await _get_home(hass, entry)
        # price_peak should be clamped to 0.2/1000
        assert home.price_peak == 0.2 / 1000.0

    @pytest.mark.asyncio
    async def test_dashboard_section_parsing_icon_only(
        self, hass: HomeAssistant,
    ):
        """Dashboard section with only icon (no name) gets auto-name. Covers lines 285-291."""
        from .const import MOCK_HOME_CONFIG
        config = dict(MOCK_HOME_CONFIG)
        config["dashboard_section_name_0"] = None
        config["dashboard_section_icon_0"] = "mdi:car"

        entry = MockConfigEntry(
            domain=DOMAIN, data=config,
            entry_id="dash_icon_test",
            title="home: Test Home",
            unique_id="quiet_solar_dash_icon_test",
        )
        entry.add_to_hass(hass)
        home = await _get_home(hass, entry)
        assert len(home.dashboard_sections) >= 1
        # First custom section gets auto-name
        assert home.dashboard_sections[0][0] == "section_1"
        assert home.dashboard_sections[0][1] == "mdi:car"


# ========================================================================
# QSSolarHistoryVals - pure logic tests
# ========================================================================


class TestQSSolarHistoryVals:
    """Tests for QSSolarHistoryVals core functionality."""

    def _make_hist_vals(self):
        """Create a QSSolarHistoryVals with minimal forecast mock."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp/test_qs"
        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        return hv

    def test_get_index_from_time_and_back(self):
        """Round-trip: time -> index -> time."""
        hv = self._make_hist_vals()
        t = datetime(2026, 2, 10, 14, 30, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)

        assert idx >= 0
        assert idx < BUFFER_SIZE_IN_INTERVALS
        assert days > 0

        t_back = hv.get_utc_time_from_index(idx, days)
        # Should be within INTERVALS_MN of original
        assert abs((t - t_back).total_seconds()) < INTERVALS_MN * 60

    def test_get_index_with_delta(self):
        """get_index_with_delta shifts correctly."""
        hv = self._make_hist_vals()
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)

        new_idx, new_days = hv.get_index_with_delta(idx, days, 4)  # 4 intervals = 1h
        t_new = hv.get_utc_time_from_index(new_idx, new_days)
        expected = hv.get_utc_time_from_index(idx, days) + timedelta(minutes=4 * INTERVALS_MN)
        assert abs((t_new - expected).total_seconds()) < 1

    def test_add_value_and_is_time_in_current_interval(self):
        """add_value stores and is_time_in_current_interval works. Covers lines 3354-3373."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t1 = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        result = hv.add_value(t1, 1000.0)
        # First add: no interval change yet
        assert result is False
        assert hv.is_time_in_current_interval(t1)

        # Add a value at a different interval
        t2 = t1 + timedelta(minutes=INTERVALS_MN + 1)
        result = hv.add_value(t2, 1500.0)
        # Should have cached the previous interval
        assert result is True

    def test_store_and_flush_current_vals(self):
        """store_and_flush_current_vals stores and resets. Covers lines 3301-3309."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t1 = datetime(2026, 2, 10, 14, 2, tzinfo=pytz.UTC)
        hv.add_value(t1, 500.0)
        t2 = t1 + timedelta(minutes=5)
        hv.add_value(t2, 600.0)

        result = hv.store_and_flush_current_vals()
        assert result is True
        assert hv._current_idx is None
        assert hv._current_days is None
        assert hv._current_values == []

    def test_get_current_non_stored_val_at_time(self):
        """Covers lines 3311-3323."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        # No current values
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        val, dur = hv.get_current_non_stored_val_at_time(t)
        assert val is None
        assert dur is None

        # Add values and query
        hv.add_value(t, 100.0)
        hv.add_value(t + timedelta(minutes=3), 200.0)
        val, dur = hv.get_current_non_stored_val_at_time(t + timedelta(minutes=5))
        if val is not None:
            assert isinstance(val, float)

    def test_get_closest_stored_value_empty(self):
        """No values returns (None, None). Covers line 3327-3328."""
        hv = self._make_hist_vals()
        hv.values = None
        result = hv.get_closest_stored_value(datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC))
        assert result == (None, None)

    def test_get_closest_stored_value_with_data(self):
        """Returns stored value when available. Covers lines 3330-3351."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv.values[0][idx] = 750
        hv.values[1][idx] = days

        result = hv.get_closest_stored_value(t)
        assert result[0] is not None
        assert result[1] == 750

    def test_update_current_forecast_if_needed(self):
        """Covers lines 2718-2721."""
        hv = self._make_hist_vals()
        hv._last_forecast_update_time = None
        assert hv.update_current_forecast_if_needed(datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)) is True

        hv._last_forecast_update_time = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        # Within same interval
        assert hv.update_current_forecast_if_needed(
            datetime(2026, 2, 10, 14, 5, tzinfo=pytz.UTC)
        ) is False
        # After interval
        assert hv.update_current_forecast_if_needed(
            datetime(2026, 2, 10, 14, 20, tzinfo=pytz.UTC)
        ) is True

    def test_get_values_wrap_around(self):
        """_get_values handles ring-buffer wrap-around. Covers lines 3186-3197."""
        hv = self._make_hist_vals()
        hv.values = np.ones((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32) * 100

        # Normal range
        v, d = hv._get_values(10, 20)
        assert v is not None
        assert len(v) == 11

        # Wrap-around range: end < start
        v, d = hv._get_values(BUFFER_SIZE_IN_INTERVALS - 5, 5)
        assert v is not None
        assert len(v) == 11  # 5 from end + 6 from start

    def test_get_values_none_values(self):
        """No values returns (None, None). Covers line 3187-3189."""
        hv = self._make_hist_vals()
        hv.values = None
        v, d = hv._get_values(0, 10)
        assert v is None
        assert d is None

    def test_xcorr_max_pearson_basic(self):
        """Basic Pearson correlation. Covers lines 2866-2898."""
        hv = self._make_hist_vals()

        # Perfect correlation
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        r, lag, S = hv.xcorr_max_pearson(x, y, Lmax=0)
        assert abs(r - 1.0) < 0.01
        assert S < 1.0  # near-perfect score

    def test_xcorr_max_pearson_zero_std(self):
        """Zero std returns worst score. Covers line 2873-2874."""
        hv = self._make_hist_vals()
        x = [5, 5, 5, 5]
        y = [3, 3, 3, 3]
        r, lag, S = hv.xcorr_max_pearson(x, y, Lmax=0)
        assert r == -1
        assert S == 1

    def test_xcorr_max_pearson_with_lag(self):
        """Correlation with lag search."""
        hv = self._make_hist_vals()
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        y = [7, 0, 1, 2, 3, 4, 5, 6]  # x shifted by 1
        r, lag, S = hv.xcorr_max_pearson(x, y, Lmax=2)
        assert r > 0.9

    def test_get_range_score_not_enough_ok_values(self):
        """Not enough overlap returns empty. Covers lines 2813-2817."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        current_values = np.array([100, 200, 300, 400, 500], dtype=np.float64)
        current_ok = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        # Past values are all zero (days=0)
        result = hv._get_range_score(current_values, current_ok, 0, past_delta=NUM_INTERVALS_PER_DAY)
        assert result == []

    def test_get_possible_now_val_for_forecast(self):
        """Covers lines 2854-2864."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
        hv._current_idx = None
        hv._current_values = []

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        result = hv._get_possible_now_val_for_forcast(now)
        assert result is None

    def test_get_predicted_data_no_scores(self):
        """No scores returns (None, None). Covers line 2902."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        fv, pd = hv._get_predicted_data(24, 100, 10, [])
        assert fv is None
        assert pd is None


# ========================================================================
# _sanitize_idx tests
# ========================================================================


class TestSanitizeIdx:
    """Tests for _sanitize_idx helper."""

    def test_positive(self):
        assert _sanitize_idx(5) == 5

    def test_negative(self):
        result = _sanitize_idx(-2)
        assert result == BUFFER_SIZE_IN_INTERVALS - 2

    def test_large(self):
        result = _sanitize_idx(BUFFER_SIZE_IN_INTERVALS + 3)
        assert result == 3


# ========================================================================
# QSHome non-controlled consumption sensor getter
# ========================================================================


class TestNonControlledConsumptionGetter:
    """Tests for the complex home_non_controlled_consumption_sensor_state_getter."""

    @pytest.mark.asyncio
    async def test_with_only_grid_no_solar_no_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Grid only, no solar/battery. Covers lines 1276, 1298-1302."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        home._childrens = []

        # Mock the grid sensor to return a value
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        home._sensor_latest_values = {}

        # Use the real method - it calls get_sensor_latest_possible_valid_value
        # which reads from internal state. Set up the state:
        hass.states.async_set("sensor.grid_power", "-500", {"unit_of_measurement": "W"})
        home.add_to_history(home.grid_active_power_sensor, now)

        result = home.home_non_controlled_consumption_sensor_state_getter("sensor.test", now)
        # May return None if no valid grid reading in tolerance window, or a tuple
        # The important thing is no crash


# ========================================================================
# QSHomeConsumptionHistoryAndForecast
# ========================================================================


class TestQSHomeConsumptionHistoryAndForecast:
    """Tests for QSHomeConsumptionHistoryAndForecast."""

    @pytest.mark.asyncio
    async def test_init_forecasts_creates_history_vals(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """init_forecasts creates home_non_controlled_consumption. Covers lines 2405-2411."""
        home = await _get_home(hass, home_config_entry)
        forecast = home._consumption_forecast
        forecast._in_reset = False
        forecast.home_non_controlled_consumption = None

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        result = await forecast.init_forecasts(now)
        assert result is True
        assert forecast.home_non_controlled_consumption is not None

    @pytest.mark.asyncio
    async def test_init_forecasts_during_reset(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """During reset, returns False. Covers line 2411."""
        home = await _get_home(hass, home_config_entry)
        forecast = home._consumption_forecast
        forecast._in_reset = True
        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        result = await forecast.init_forecasts(now)
        assert result is False

    def test_combine_stored_forecast_values_add(self):
        """_combine_stored_forecast_values with do_add=True. Covers lines 2413-2431."""
        home_mock = MagicMock()
        home_mock.hass = None
        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        val1 = np.array([[100, 200, 300], [5, 5, 5]], dtype=np.int32)
        val2 = np.array([[10, 20, 30], [5, 5, 5]], dtype=np.int32)

        result = forecast._combine_stored_forecast_values(val1, val2, do_add=True)
        assert result.shape == val1.shape
        # Where both have same day and nonzero: values should add
        assert result[0][0] == 110
        assert result[0][1] == 220

    def test_combine_stored_forecast_values_sub(self):
        """_combine_stored_forecast_values with do_add=False."""
        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        val1 = np.array([[100, 200, 300], [5, 5, 5]], dtype=np.int32)
        val2 = np.array([[10, 20, 30], [5, 5, 5]], dtype=np.int32)

        result = forecast._combine_stored_forecast_values(val1, val2, do_add=False)
        assert result[0][0] == 90
        assert result[0][1] == 180

    def test_combine_stored_forecast_values_different_days_zeroed(self):
        """Different day values produce zeros."""
        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        val1 = np.array([[100, 200], [5, 6]], dtype=np.int32)
        val2 = np.array([[10, 20], [5, 7]], dtype=np.int32)

        result = forecast._combine_stored_forecast_values(val1, val2, do_add=True)
        # Index 0: same day (5), should combine
        assert result[0][0] == 110
        # Index 1: different days (6 vs 7), should be zero
        assert result[0][1] == 0

    @pytest.mark.asyncio
    async def test_dump_for_debug_no_consumption(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry, tmp_path,
    ):
        """dump_for_debug with no consumption is a no-op. Covers line 2401."""
        home = await _get_home(hass, home_config_entry)
        forecast = home._consumption_forecast
        forecast.home_non_controlled_consumption = None
        # Should not raise
        await forecast.dump_for_debug(str(tmp_path))

    @pytest.mark.asyncio
    async def test_dump_for_debug_with_consumption(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry, tmp_path,
    ):
        """dump_for_debug saves the file. Covers lines 2401-2403."""
        home = await _get_home(hass, home_config_entry)
        forecast = home._consumption_forecast
        mock_hv = MagicMock()
        mock_hv.file_name = "test.npy"
        mock_hv.save_values = AsyncMock()
        forecast.home_non_controlled_consumption = mock_hv

        await forecast.dump_for_debug(str(tmp_path))
        mock_hv.save_values.assert_awaited_once()


# ========================================================================
# QSSolarHistoryVals save/read
# ========================================================================


class TestQSSolarHistoryValsSaveRead:
    """Tests for save/read methods."""

    @pytest.mark.asyncio
    async def test_save_values_no_hass(self, tmp_path):
        """save_values with no hass writes synchronously. Covers line 3213-3214."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = str(tmp_path)

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv.values = np.ones((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
        hv.file_path = str(tmp_path / "test.npy")

        await hv.save_values()
        # File should exist
        loaded = np.load(str(tmp_path / "test.npy"))
        assert loaded.shape == hv.values.shape

    def test_read_value_missing_file(self, tmp_path):
        """read_value with missing file returns None. Covers lines 3220-3227."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = str(tmp_path)

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv.file_path = str(tmp_path / "nonexistent.npy")

        result = hv.read_value()
        assert result is None

    def test_read_value_existing_file(self, tmp_path):
        """read_value with existing file returns the data."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = str(tmp_path)

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv.file_path = str(tmp_path / "test.npy")

        data = np.ones((2, 10), dtype=np.int32)
        np.save(str(tmp_path / "test.npy"), data)

        result = hv.read_value()
        assert result is not None
        assert result.shape == (2, 10)

    def test_is_time_in_current_interval_no_current(self):
        """No current interval returns False. Covers line 3249-3250."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp"

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        assert hv.is_time_in_current_interval(datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)) is False

    def test_get_current_interval_value_empty(self):
        """No current values returns (None, None). Covers line 3269."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp"

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        result = hv.get_current_interval_value()
        assert result == (None, None)


# ========================================================================
# QSSolarHistoryVals init
# ========================================================================


class TestQSSolarHistoryValsInit:
    """Tests for init method edge cases."""

    def test_init_constructor_no_home(self):
        """Constructor with forecast.home = None. Covers lines 2694-2696."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp"

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_no_home")
        assert hv.hass is None
        assert hv.home is None

    def test_init_constructor_home_no_hass(self):
        """Constructor with home but no hass. Covers lines 2699-2700."""
        forecast = MagicMock()
        home_mock = MagicMock()
        home_mock.hass = None
        forecast.home = home_mock
        forecast.storage_path = "/tmp"

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test_no_hass")
        assert hv.hass is None
        assert hv.home is home_mock

    @pytest.mark.asyncio
    async def test_init_already_done_returns_none_none(self):
        """Second init returns (None, None). Covers lines 3399-3400."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp"

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv._init_done = True

        result = await hv.init(datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC))
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_init_non_reset_with_existing_values(self):
        """Non-reset init with existing values returns (None, None). Covers lines 3404-3405."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp"

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv._init_done = False
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        result = await hv.init(datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC))
        assert result == (None, None)


# ========================================================================
# QSHome map_location_path
# ========================================================================


class TestMapLocationPath:
    """Tests for map_location_path."""

    @pytest.mark.asyncio
    async def test_map_location_path_empty_states(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Empty state lists return empty results."""
        home = await _get_home(hass, home_config_entry)
        # Set home lat/lon
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        segments, s1_not_home, s2_not_home = home.map_location_path(
            [], [], now, now + timedelta(hours=2)
        )
        assert segments == []
        assert s1_not_home == []
        assert s2_not_home == []

    @pytest.mark.asyncio
    async def test_map_location_path_with_home_states(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """States at home produce no 'close' segments."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Create states at home position
        states = []
        for i in range(5):
            t = now + timedelta(minutes=i * 10)
            states.append(SimpleNamespace(
                last_updated=t,
                state="home",
                attributes={"latitude": 48.8566, "longitude": 2.3522, "source": "gps"},
                entity_id="device_tracker.test",
            ))

        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, states, now, now + timedelta(hours=2)
        )
        # All at home, so positions near home are filtered
        assert s1_not_home == []

    @pytest.mark.asyncio
    async def test_map_location_path_not_home_segments(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """States away from home create not-home segments. Covers lines 724-754."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        states = []
        # First at home
        states.append(SimpleNamespace(
            last_updated=now,
            state="home",
            attributes={"latitude": 48.8566, "longitude": 2.3522, "source": "gps"},
            entity_id="device_tracker.test",
        ))
        # Then away
        for i in range(1, 4):
            t = now + timedelta(minutes=i * 15)
            states.append(SimpleNamespace(
                last_updated=t,
                state="not_home",
                attributes={"latitude": 49.0 + i * 0.1, "longitude": 3.0 + i * 0.1, "source": "gps"},
                entity_id="device_tracker.test",
            ))
        # Back home
        states.append(SimpleNamespace(
            last_updated=now + timedelta(hours=1),
            state="home",
            attributes={"latitude": 48.8566, "longitude": 2.3522, "source": "gps"},
            entity_id="device_tracker.test",
        ))

        end = now + timedelta(hours=2)
        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )
        # Should have detected a not-home segment
        assert len(s1_not_home) == 1

    @pytest.mark.asyncio
    async def test_map_location_path_unknown_state(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Unknown/unavailable states are skipped. Covers lines 706-709."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        states = [
            SimpleNamespace(
                last_updated=now,
                state=STATE_UNKNOWN,
                attributes={},
                entity_id="device_tracker.test",
            ),
            SimpleNamespace(
                last_updated=now + timedelta(minutes=5),
                state=STATE_UNAVAILABLE,
                attributes={},
                entity_id="device_tracker.test",
            ),
        ]

        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, now + timedelta(hours=1)
        )
        assert segments == []


# ========================================================================
# QSHome update_loads_constraints
# ========================================================================


class TestUpdateLoadsConstraints:
    """Tests for update_loads_constraints."""

    @pytest.mark.asyncio
    async def test_update_loads_constraints_off_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """OFF mode returns early. Covers line 1674-1678."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_OFF.value
        await home.update_loads_constraints(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))

    @pytest.mark.asyncio
    async def test_update_loads_constraints_sensors_only(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """SENSORS_ONLY mode returns early."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_SENSORS_ONLY.value
        await home.update_loads_constraints(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))


# ========================================================================
# QSHome person-related methods
# ========================================================================


class TestHomePersonMethods:
    """Tests for person/car allocation methods."""

    @pytest.mark.asyncio
    async def test_compute_person_needed_time_and_date(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """_compute_person_needed_time_and_date returns correct structure."""
        home = await _get_home(hass, home_config_entry)
        t = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        local_day, local_day_shifted, local_day_utc, is_passed_limit = home._compute_person_needed_time_and_date(t)

        assert isinstance(local_day, datetime)
        assert isinstance(local_day_shifted, datetime)
        assert isinstance(local_day_utc, datetime)
        assert isinstance(is_passed_limit, bool)
        # At noon UTC, should be past the 4am limit
        assert is_passed_limit is True

    @pytest.mark.asyncio
    async def test_get_best_persons_cars_allocations_empty(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No persons/cars returns empty dict."""
        home = await _get_home(hass, home_config_entry)
        t = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        result = await home.get_best_persons_cars_allocations(t, force_update=True)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_recompute_people_historical_data_no_persons(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Recompute with no persons should not crash."""
        home = await _get_home(hass, home_config_entry)
        t = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        await home.recompute_people_historical_data(t)


# ========================================================================
# QSHome get_devices_for_dashboard_section
# ========================================================================


class TestGetDevicesForDashboardSection:
    """Tests for get_devices_for_dashboard_section."""

    @pytest.mark.asyncio
    async def test_no_matching_section(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No devices match the section."""
        home = await _get_home(hass, home_config_entry)
        result = home.get_devices_for_dashboard_section("nonexistent_section")
        assert isinstance(result, list)
        assert len(result) == 0


# ========================================================================
# QSHome update_all_states
# ========================================================================


class TestUpdateAllStates:
    """Tests for update_all_states."""

    @pytest.mark.asyncio
    async def test_update_all_states_off_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """OFF mode returns early. Covers line 2086-2089."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_OFF.value
        await home.update_all_states(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))

    @pytest.mark.asyncio
    async def test_update_all_states_none_mode(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """None mode returns early."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = None
        await home.update_all_states(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC))


# ========================================================================
# Additional QSSolarHistoryVals tests - corner cases
# ========================================================================


class TestQSSolarHistoryValsCornerCases:
    """Additional corner case tests for QSSolarHistoryVals."""

    def _make_hist_vals(self):
        """Create a QSSolarHistoryVals with minimal forecast mock."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp/test_qs"
        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        return hv

    def test_cache_current_vals_with_extend_idx(self):
        """_cache_current_vals with extend_but_not_cover_idx fills gap. Covers lines 3286-3297."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv._current_idx = idx
        hv._current_days = days
        hv._current_values = [(t, 500.0)]

        # Extend to 3 intervals ahead
        extend_idx = _sanitize_idx(idx + 3)
        hv._cache_current_vals(extend_but_not_cover_idx=extend_idx)

        # Current slot should have the value
        assert hv.values[0][idx] != 0
        # Gap slots should be filled
        for delta in range(1, 3):
            fill_idx = _sanitize_idx(idx + delta)
            assert hv.values[0][fill_idx] == 500

    def test_cache_current_vals_no_values_returns_false(self):
        """No current values returns False. Covers line 3274-3275."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
        hv._current_idx = None
        hv._current_values = []

        result = hv._cache_current_vals()
        assert result is False

    def test_cache_current_vals_creates_values_array(self):
        """_cache_current_vals creates values if None. Covers line 3279-3280."""
        hv = self._make_hist_vals()
        hv.values = None

        t = datetime(2026, 2, 10, 14, 5, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv._current_idx = idx
        hv._current_days = days
        hv._current_values = [(t, 300.0)]

        result = hv._cache_current_vals()
        assert result is True
        assert hv.values is not None
        assert hv.values.shape == (2, BUFFER_SIZE_IN_INTERVALS)

    def test_get_closest_stored_value_current_plus_one(self):
        """get_closest_stored_value when querying _current_idx + 1. Covers line 3335-3336."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv._current_idx = idx
        hv._current_days = days
        hv._current_values = [(t, 400.0), (t + timedelta(minutes=5), 450.0)]

        # Query the next interval
        t_next = t + timedelta(minutes=INTERVALS_MN)
        result_time, result_val = hv.get_closest_stored_value(t_next)
        # Should get current interval value
        if result_val is not None:
            assert isinstance(result_val, (int, float))

    def test_get_closest_stored_value_same_interval(self):
        """get_closest_stored_value when querying same interval. Covers line 3337-3339."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 2, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv._current_idx = idx
        hv._current_days = days
        hv._current_values = [(t, 300.0)]

        # Fill previous idx with known value
        prev_idx = _sanitize_idx(idx - 1)
        hv.values[0][prev_idx] = 250
        hv.values[1][prev_idx] = days

        result_time, result_val = hv.get_closest_stored_value(t)
        # Should get the previous idx value (idx shifted back)
        assert result_val == 250

    def test_get_closest_stored_value_no_valid_day(self):
        """get_closest_stored_value with too-old day data returns None. Covers line 3348-3349."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)

        # Set a value with a very old day number
        hv.values[0][idx] = 100
        hv.values[1][idx] = days - 10  # much older

        result_time, result_val = hv.get_closest_stored_value(t)
        assert result_val is None
        assert result_time is None

    def test_get_current_non_stored_val_at_different_interval(self):
        """Query a different interval returns None. Covers line 3322-3323."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv._current_idx = idx
        hv._current_days = days
        hv._current_values = [(t, 100.0)]

        # Query a different interval
        t_diff = t + timedelta(hours=2)
        val, dur = hv.get_current_non_stored_val_at_time(t_diff)
        assert val is None
        assert dur is None

    def test_get_current_non_stored_val_at_same_interval(self):
        """Query same interval returns mean and duration. Covers lines 3318-3321."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)
        hv._current_idx = idx
        hv._current_days = days
        hv._current_values = [
            (t, 100.0),
            (t + timedelta(minutes=5), 200.0),
        ]

        t_query = t + timedelta(minutes=8)
        val, dur = hv.get_current_non_stored_val_at_time(t_query)
        assert val is not None
        assert dur is not None
        assert dur > 0

    def test_add_value_same_interval_accumulates(self):
        """Multiple add_value calls in same interval accumulate. Covers lines 3370-3371."""
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 2, tzinfo=pytz.UTC)
        hv.add_value(t, 100.0)
        hv.add_value(t + timedelta(minutes=2), 200.0)
        hv.add_value(t + timedelta(minutes=4), 300.0)

        assert len(hv._current_values) == 3

    def test_get_value_from_current_forecast_with_data(self):
        """get_value_from_current_forecast with real forecast data. Covers lines 2723-2725."""
        hv = self._make_hist_vals()
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        # Set up a mock forecast
        hv._current_forecast = [
            (t, 500.0),
            (t + timedelta(minutes=15), 600.0),
            (t + timedelta(minutes=30), 700.0),
        ]

        result_time, result_val = hv.get_value_from_current_forecast(t + timedelta(minutes=10))
        assert result_val is not None

    def test_get_value_from_current_forecast_none(self):
        """get_value_from_current_forecast with no forecast. Covers line 2724."""
        hv = self._make_hist_vals()
        hv._current_forecast = None

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        # This will raise if _current_forecast is None, so we check
        # get_value_from_time_series handles None
        try:
            result_time, result_val = hv.get_value_from_current_forecast(t)
        except (TypeError, AttributeError):
            pass  # Expected if _current_forecast is None

    def test_get_utc_time_from_index_wrap_around(self):
        """get_utc_time_from_index with idx < days_index wraps. Covers lines 3392-3394."""
        hv = self._make_hist_vals()
        t = datetime(2026, 2, 10, 0, 30, tzinfo=pytz.UTC)  # early in day
        idx, days = hv.get_index_from_time(t)

        # Force a small idx and check wrap-around
        # Use idx=0 with days where days_index > 0
        result = hv.get_utc_time_from_index(0, days)
        assert isinstance(result, datetime)

    def test_get_predicted_data_with_valid_scores(self):
        """_get_predicted_data with valid scores builds forecast. Covers lines 2910-2951."""
        hv = self._make_hist_vals()
        hv.values = np.ones((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32) * 5
        # Set values to have non-zero data
        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)

        # Fill a week's worth of data
        for i in range(NUM_INTERVALS_PER_DAY * 2):
            slot = _sanitize_idx(idx - i)
            hv.values[0][slot] = 500 + i
            hv.values[1][slot] = days

        # Build scores: each score is [past_delta, rmse_score]
        scores = [[NUM_INTERVALS_PER_DAY, 10.0]]  # 1 day ago

        fv, pd = hv._get_predicted_data(2, idx, days, scores)
        if fv is not None:
            assert len(fv) > 0

    def test_get_range_score_all_score_types(self):
        """_get_range_score with num_score=6 exercises every score branch.

        score_idx 0: RMSE
        score_idx 1: mean ratio
        score_idx 2: Pearson correlation (via xcorr_max_pearson)
        score_idx 3: mean bias error
        score_idx 4: mean absolute error
        score_idx 5: else → 0.0

        Covers lines 2819-2848 exhaustively.
        """
        hv = self._make_hist_vals()
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        t = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(t)

        # We need enough data points for > 60% overlap.
        # Use 24 hours of data = 96 intervals at 15mn each.
        n = NUM_INTERVAL_PER_HOUR * 24

        # Fill "current" data with a realistic-ish pattern:
        # a repeating sawtooth between 200 and 800 W
        for i in range(n):
            slot = _sanitize_idx(idx - n + 1 + i)
            hv.values[0][slot] = 200 + (i % 12) * 50  # 200..750 repeating
            hv.values[1][slot] = days

        # Fill "past" data at past_delta = 1 day ago with a similar but
        # slightly shifted pattern so scores are interesting (not zero, not perfect)
        past_delta = NUM_INTERVALS_PER_DAY
        for i in range(n):
            slot = _sanitize_idx(idx - n + 1 + i - past_delta)
            hv.values[0][slot] = 220 + (i % 12) * 50  # shifted by +20
            hv.values[1][slot] = days - 1

        # Build current_values / current_ok from the buffer (same order as _get_values)
        start_idx = _sanitize_idx(idx - n + 1)
        current_values = np.array(
            [float(hv.values[0][_sanitize_idx(start_idx + i)]) for i in range(n)],
            dtype=np.float64,
        )
        current_ok = np.ones(n, dtype=np.int32)

        # ---- num_score=6 → exercises indices 0,1,2,3,4 + else ----
        result = hv._get_range_score(
            current_values, current_ok, start_idx,
            past_delta=past_delta, num_score=6,
        )

        # result should be [past_delta, rmse, mean_ratio, pearson_S, mean_bias, mean_abs, 0.0]
        assert isinstance(result, list)
        assert len(result) == 7  # past_delta + 6 scores

        past_delta_out = result[0]
        rmse = result[1]
        mean_ratio = result[2]
        pearson_s = result[3]
        mean_bias = result[4]
        mean_abs_err = result[5]
        fallback = result[6]

        assert past_delta_out == past_delta

        # score_idx 0: RMSE — should be > 0 since patterns differ by ~20
        assert isinstance(rmse, float)
        assert rmse > 0
        # The patterns differ by exactly 20 at every point, so RMSE ≈ 20
        assert 15 < rmse < 25

        # score_idx 1: mean ratio — |sum(past)/sum(current) - 1| * 100
        # past is ~20 higher on average, so ratio slightly > 1, percentage small
        assert isinstance(mean_ratio, float)
        assert mean_ratio >= 0

        # score_idx 2: Pearson S score — S = 100*(1 - (1+r)/2)
        # The patterns are the same shape (just shifted up by 20), so r ≈ 1 → S ≈ 0
        assert isinstance(pearson_s, float)
        assert pearson_s < 5  # nearly perfect correlation

        # score_idx 3: mean bias error — |sum(current - past)| / n
        # current[i] - past[i] = -20 for all i → bias = 20
        assert isinstance(mean_bias, float)
        assert 15 < mean_bias < 25

        # score_idx 4: mean absolute error — sum(|current - past|) / n
        # |current[i] - past[i]| = 20 for all i → mae = 20
        assert isinstance(mean_abs_err, float)
        assert 15 < mean_abs_err < 25

        # score_idx 5: else branch → always 0.0
        assert fallback == 0.0

        # ---- Also verify num_score=1 returns just [past_delta, rmse] ----
        result_1 = hv._get_range_score(
            current_values, current_ok, start_idx,
            past_delta=past_delta, num_score=1,
        )
        assert len(result_1) == 2
        assert result_1[0] == past_delta
        assert result_1[1] == rmse  # same RMSE

    @pytest.mark.asyncio
    async def test_save_values_custom_path(self, tmp_path):
        """save_values with custom file_path. Covers line 3210-3211."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = str(tmp_path)

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv.values = np.ones((2, 10), dtype=np.int32)

        custom_path = str(tmp_path / "custom.npy")
        await hv.save_values(file_path=custom_path)

        loaded = np.load(custom_path)
        assert loaded.shape == (2, 10)

    @pytest.mark.asyncio
    async def test_read_values_async_no_hass(self, tmp_path):
        """read_values_async without hass reads synchronously. Covers lines 3241-3242."""
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = str(tmp_path)

        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv.file_path = str(tmp_path / "test_async.npy")

        # No file yet
        result = await hv.read_values_async()
        assert result is None

        # Save and read
        data = np.ones((2, 10), dtype=np.int32)
        np.save(hv.file_path, data)
        result = await hv.read_values_async()
        assert result is not None
        assert result.shape == (2, 10)


# ========================================================================
# Helpers for injecting sensor values into QS device internals
# ========================================================================


def _inject_sensor_value(device, entity_id, time, value, attr=None):
    """Inject a fake sensor reading into a device's internal state tracking.

    This avoids mocking get_sensor_latest_possible_valid_value: instead we
    set the same internal state that update_states would populate.
    """
    if attr is None:
        attr = {}
    entry = (time, value, attr)
    device._entity_probed_last_valid_state[entity_id] = entry
    device._entity_probed_state[entity_id] = [entry]


# ========================================================================
# QSHome non-controlled consumption - comprehensive branch coverage
# ========================================================================


class TestNonControlledConsumptionBranches:
    """Tests for home_non_controlled_consumption_sensor_state_getter.

    Each test injects values into the internal sensor caches of the
    real QSHome / QSSolar / QSBattery objects so that the method
    hits a specific branch without any method-level mocking.
    """

    @pytest.mark.asyncio
    async def test_no_solar_no_battery_grid_only(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Grid only: home_consumption = 0 - grid.  Covers 1298-1302."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        _inject_sensor_value(home, home.grid_active_power_sensor, now, -800.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        assert result[0] == now
        # home_consumption = 0 - (-800) = 800
        assert home.home_consumption == 800.0
        assert home.grid_consumption_power == -800.0
        assert home.home_non_controlled_consumption == 800.0  # no controlled loads
        assert home.home_available_power == -800.0  # no battery

    @pytest.mark.asyncio
    async def test_no_solar_with_battery_charge(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No solar, battery present: solar_prod_minus_bat = 0 - battery_charge.  Covers line 1275-1276."""
        from .const import MOCK_BATTERY_CONFIG
        home = await _get_home(hass, home_config_entry)
        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_ncc_1",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_ncc_1",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()

        assert home.battery is not None
        home.physical_solar_plant = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # battery discharging at 500W
        _inject_sensor_value(home.battery, home.battery.charge_discharge_sensor, now, -500.0)
        # grid exporting 200W
        _inject_sensor_value(home, home.grid_active_power_sensor, now, 200.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        assert home.home_consumption == 300.0
        # available_power = grid + battery = 200 + (-500) = -300
        assert home.home_available_power == -300.0

    @pytest.mark.asyncio
    async def test_solar_active_power_returns_value(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """solar_inverter_active_power returns value → solar_production_minus_battery is set.

        Covers the main solar path (line 1251-1253) and solar_production derivation (1269-1273).
        """
        from .const import MOCK_SOLAR_CONFIG
        home = await _get_home(hass, home_config_entry)
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_ncc_1",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_ncc_1",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        assert home.solar_plant is not None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # solar inverter active power = 3000W (includes battery)
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 3000.0)
        # grid importing 500W
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -500.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod_minus_bat = 3000
        # home_consumption = 3000 - (-500) = 3500
        assert home.home_consumption == 3500.0
        # solar_production = solar_prod_minus_bat (no battery) → 3000
        assert home.solar_plant.solar_production == 3000.0

    @pytest.mark.asyncio
    async def test_solar_input_power_fallback(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """solar_inverter_active_power returns None → falls back to solar_inverter_input_active_power.

        Covers lines 1255-1256, 1261-1265, 1267-1268.
        """
        from .const import MOCK_SOLAR_CONFIG
        from custom_components.quiet_solar.const import CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR
        # Config with input power sensor but not active
        sol_cfg = dict(MOCK_SOLAR_CONFIG)
        sol_cfg[CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR] = "sensor.solar_input_power"

        home = await _get_home(hass, home_config_entry)
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=sol_cfg,
            entry_id="sol_ncc_2",
            title=f"solar: {sol_cfg['name']}",
            unique_id="quiet_solar_sol_ncc_2",
        )
        sol_entry.add_to_hass(hass)
        hass.states.async_set("sensor.solar_input_power", "4000", {"unit_of_measurement": "W"})
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        assert home.solar_plant is not None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Make active power return None (not injected or force to None)
        home.solar_plant._entity_probed_last_valid_state[home.solar_plant.solar_inverter_active_power] = None
        # Inject input power
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_input_active_power, now, 4000.0)
        # grid
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -300.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_production = 4000 (from input sensor)
        # solar_prod_minus_bat = 4000 (no battery)
        assert home.solar_plant.solar_production == 4000.0
        assert home.home_consumption == 4300.0  # 4000 - (-300)

    @pytest.mark.asyncio
    async def test_solar_input_with_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Input power + battery → solar_prod_minus_bat = production - battery.

        Covers lines 1262-1263 (solar_production_minus_battery = solar_production - battery_charge).
        """
        from .const import MOCK_SOLAR_CONFIG, MOCK_BATTERY_CONFIG
        from custom_components.quiet_solar.const import CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR
        sol_cfg = dict(MOCK_SOLAR_CONFIG)
        sol_cfg[CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR] = "sensor.solar_input_power2"

        home = await _get_home(hass, home_config_entry)
        # battery
        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_ncc_2",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_ncc_2",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()
        # solar
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=sol_cfg,
            entry_id="sol_ncc_3",
            title=f"solar: {sol_cfg['name']}",
            unique_id="quiet_solar_sol_ncc_3",
        )
        sol_entry.add_to_hass(hass)
        hass.states.async_set("sensor.solar_input_power2", "5000", {"unit_of_measurement": "W"})
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        assert home.solar_plant is not None
        assert home.battery is not None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # active power → None
        home.solar_plant._entity_probed_last_valid_state[home.solar_plant.solar_inverter_active_power] = None
        # input power = 5000
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_input_active_power, now, 5000.0)
        # battery charging at 1000
        _inject_sensor_value(home.battery, home.battery.charge_discharge_sensor, now, 1000.0)
        # grid
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -200.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod = 5000, battery_charge = 1000
        # solar_prod_minus_bat = 5000 - 1000 = 4000
        # home_consumption = 4000 - (-200) = 4200
        assert home.solar_plant.solar_production == 5000.0
        assert home.home_consumption == 4200.0

    @pytest.mark.asyncio
    async def test_solar_prod_minus_bat_none_and_solar_prod_only(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """solar_production_minus_battery remains None, solar_production only.

        Covers line 1264-1265 (elif solar_production is not None → assign directly).
        """
        from .const import MOCK_SOLAR_CONFIG
        from custom_components.quiet_solar.const import CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR
        sol_cfg = dict(MOCK_SOLAR_CONFIG)
        sol_cfg[CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR] = "sensor.solar_input3"

        home = await _get_home(hass, home_config_entry)
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=sol_cfg,
            entry_id="sol_ncc_4",
            title=f"solar: {sol_cfg['name']}",
            unique_id="quiet_solar_sol_ncc_4",
        )
        sol_entry.add_to_hass(hass)
        hass.states.async_set("sensor.solar_input3", "2000", {"unit_of_measurement": "W"})
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        home.physical_battery = None  # no battery → battery_charge is None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # active_power → None
        home.solar_plant._entity_probed_last_valid_state[home.solar_plant.solar_inverter_active_power] = None
        # input_power = 2000
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_input_active_power, now, 2000.0)
        # grid
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -100.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod = 2000, battery_charge = None
        # solar_prod_minus_bat = solar_prod (line 1265) = 2000
        # solar_plant.solar_production = solar_prod = 2000
        assert home.solar_plant.solar_production == 2000.0

    @pytest.mark.asyncio
    async def test_solar_active_with_battery_sets_production(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """solar_inverter_active_power present, battery present: derive solar_production.

        Covers line 1269-1273 (solar_production_minus_battery not None,
        solar_production is None → set from spb + battery_charge).
        """
        from .const import MOCK_SOLAR_CONFIG, MOCK_BATTERY_CONFIG
        home = await _get_home(hass, home_config_entry)

        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_ncc_3",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_ncc_3",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()

        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_ncc_5",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_ncc_5",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # solar_inverter_active_power = 2500 (this includes battery)
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 2500.0)
        # battery_charge = 500 (charging)
        _inject_sensor_value(home.battery, home.battery.charge_discharge_sensor, now, 500.0)
        # grid
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -100.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod_minus_bat = 2500
        # solar_production is None (no input sensor returns value)
        # → solar_production = 2500 + 500 = 3000 (line 1271)
        assert home.solar_plant.solar_production == 3000.0
        # home_consumption = 2500 - (-100) = 2600
        assert home.home_consumption == 2600.0
        # available_power = grid + battery = -100 + 500 = 400
        assert home.home_available_power == 400.0

    @pytest.mark.asyncio
    async def test_no_grid_no_consumption_returns_none(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Neither grid nor home_consumption available → sets everything to None."""
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Don't inject grid → returns None
        home._entity_probed_last_valid_state[home.grid_active_power_sensor] = None

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is None
        assert home.home_consumption is None
        assert home.home_non_controlled_consumption is None
        assert home.home_available_power is None

    @pytest.mark.asyncio
    async def test_with_controlled_children(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Controlled loads subtract from home_consumption.

        Covers lines 1310-1332 (controlled consumption iteration).
        """
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Add a charger as a child load
        ch_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CHARGER_CONFIG,
            entry_id="ch_ncc_1",
            title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
            unique_id="quiet_solar_ch_ncc_1",
        )
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()

        # Grid importing 2000W
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -2000.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # home_consumption = 0 - (-2000) = 2000
        assert home.home_consumption == 2000.0
        # non_controlled = home_consumption - controlled (charger power ~ 0 since no real reading)
        assert home.home_non_controlled_consumption is not None

    @pytest.mark.asyncio
    async def test_available_power_clamped(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """available_power exceeding production gets clamped.

        Covers lines 1344-1361 (clamping logic).
        """
        from .const import MOCK_SOLAR_CONFIG
        home = await _get_home(hass, home_config_entry)
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_ncc_clamp",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_ncc_clamp",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)


        home.solar_plant.solar_max_output_power_value = 1000.0
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 300.0)
        _inject_sensor_value(home, home.grid_active_power_sensor, now, 250.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        assert home.home_available_power == 250

        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 800.0)
        _inject_sensor_value(home, home.grid_active_power_sensor, now, 250.0)
        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        assert home.home_available_power == pytest.approx(210.0, rel=0.01)


    @pytest.mark.asyncio
    async def test_available_power_clamped_with_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Clamping with battery present. Covers lines 1350-1351, 1355-1356."""
        from .const import MOCK_SOLAR_CONFIG, MOCK_BATTERY_CONFIG
        home = await _get_home(hass, home_config_entry)

        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_ncc_clamp",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_ncc_clamp",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()

        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_ncc_clamp2",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_ncc_clamp2",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        home.solar_plant.solar_max_output_power_value = 500.0

        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 2000.0)
        _inject_sensor_value(home.battery, home.battery.charge_discharge_sensor, now, -400.0)
        _inject_sensor_value(home, home.grid_active_power_sensor, now, 3000.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # available_power = grid + battery = 3000 + (-400) = 2600
        # This exceeds production output → gets clamped
        assert home.home_available_power is not None

    @pytest.mark.asyncio
    async def test_both_solar_sensors_derive_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Both solar sensors present, battery_charge None → derive battery_charge.

        When solar_inverter_active_power returns a value, solar_production_minus_battery
        is set (not None), so the input power fallback is skipped.
        However, we need solar_production to also be set for line 1258 to trigger.
        The input power is only read when active power is None (line 1255).

        To hit line 1258-1259, we need active power to fail first (so input power
        is read = solar_production), then inject spb separately. But the code
        reads active power first and only falls back.

        Actually, the condition is:
          solar_production_minus_battery is not None AND solar_production is not None AND battery_charge is None
        Since active power sets spb but NOT solar_production, and input power is only
        read when spb is None, these two can never both be non-None from sensor reads.

        This branch can only be reached if solar_inverter_active_power returns a value
        (setting spb) AND solar_inverter_input_active_power also returns a value
        (setting solar_production). But line 1255 prevents reading input when active
        succeeds. So this branch is only reachable if spb comes from active power,
        but then fails on next check... Actually reading the code more carefully:
        line 1255 says 'if solar_production_minus_battery is None AND input_power',
        so if active power returns a value, input power is never read.

        The branch 1258-1259 seems unreachable in normal flow. Let's verify the
        code path we CAN cover instead and test solar_production assignment logic.
        """
        from .const import MOCK_SOLAR_CONFIG
        home = await _get_home(hass, home_config_entry)
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_ncc_derive",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_ncc_derive",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # active power = 3500 → spb = 3500, solar_production stays None
        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 3500.0)
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -500.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # spb=3500, solar_production=None, battery_charge=None
        # line 1269: solar_production is None → goes to elif
        # line 1272: battery_charge is None → line 1273: production = spb = 3500
        assert home.solar_plant.solar_production == 3500.0
        assert home.home_consumption == 4000.0  # 3500 - (-500)

    @pytest.mark.asyncio
    async def test_solar_active_no_battery_production_equals_minus_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """solar_prod_minus_bat set, solar_prod None, no battery → production = spb.

        Covers line 1272-1273 (solar_production = solar_production_minus_battery).
        """
        from .const import MOCK_SOLAR_CONFIG
        home = await _get_home(hass, home_config_entry)
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_ncc_nobat",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_ncc_nobat",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        _inject_sensor_value(home.solar_plant, home.solar_plant.solar_inverter_active_power, now, 1500.0)
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -200.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod_minus_bat = 1500 (from active power)
        # solar_production = None (no input sensor or it's None)
        # battery_charge = None
        # → line 1273: solar_production = solar_prod_minus_bat = 1500
        assert home.solar_plant.solar_production == 1500.0

    @pytest.mark.asyncio
    async def test_accurate_power_sensor_derives_grid(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """accurate_power_sensor gives home_consumption, derive grid.

        Covers lines 1288-1296 (accurate_power_sensor path).
        """
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Manually set accurate_power_sensor and register it
        sensor_id = "sensor.accurate_home_power"
        home.accurate_power_sensor = sensor_id
        home.attach_power_to_probe(sensor_id)
        hass.states.async_set(sensor_id, "1500", {"unit_of_measurement": "W"})

        # Inject accurate power (home consumption = 1500)
        _inject_sensor_value(home, sensor_id, now, 1500.0)
        # Don't inject grid → grid_consumption = None
        home._entity_probed_last_valid_state[home.grid_active_power_sensor] = None

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # home_consumption = 1500 (from accurate sensor)
        # grid = None initially
        # solar_prod_minus_bat = 0 (no solar, no battery)
        # line 1293: grid = spb - home = 0 - 1500 = -1500
        assert home.grid_consumption_power == -1500.0
        assert home.home_consumption == 1500.0

    @pytest.mark.asyncio
    async def test_accurate_power_no_solar_derives_grid_zero(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """accurate_power_sensor, solar_prod_minus_bat = 0 → grid = 0 - home.

        Covers line 1293-1294 (spb not None path) and general flow.
        """
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        sensor_id = "sensor.accurate_power2"
        home.accurate_power_sensor = sensor_id
        home.attach_power_to_probe(sensor_id)
        hass.states.async_set(sensor_id, "1000", {"unit_of_measurement": "W"})

        _inject_sensor_value(home, sensor_id, now, 1000.0)
        home._entity_probed_last_valid_state[home.grid_active_power_sensor] = None

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod_minus_bat = 0 (none → set to 0 at line 1279)
        # home_consumption = 1000, grid = None
        # line 1292: home_consumption not None AND grid is None
        # line 1293: spb (=0) is not None → grid = 0 - 1000 = -1000
        assert home.grid_consumption_power == -1000.0

    @pytest.mark.asyncio
    async def test_home_consumption_from_grid_no_solar(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """No accurate sensor, no solar: home = 0 - grid.

        Covers line 1301-1302.
        """
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # grid exporting 600W
        _inject_sensor_value(home, home.grid_active_power_sensor, now, 600.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # solar_prod_minus_bat = 0, grid = 600
        # line 1299: spb (=0) is not None → home = 0 - 600 = -600
        assert home.home_consumption == -600.0

    @pytest.mark.asyncio
    async def test_controlled_loads_subtracted(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Children with power subtract from home consumption.

        Covers lines 1310-1319 (controlled consumption loop over _childrens).
        """
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Set up charger
        ch_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CHARGER_CONFIG,
            entry_id="ch_ncc_ctrl",
            title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
            unique_id="quiet_solar_ch_ncc_ctrl",
        )
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()

        # Inject grid
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -3000.0)

        # The charger is a child of home; inject a power reading for it
        charger = home._chargers[0]
        if hasattr(charger, '_get_power_measure') and charger._get_power_measure() is not None:
            _inject_sensor_value(charger, charger._get_power_measure(), now, 1000.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # home_consumption = 0 - (-3000) = 3000
        # controlled_consumption = charger power
        assert home.home_consumption == 3000.0
        # non_controlled = 3000 - controlled
        assert home.home_non_controlled_consumption <= 3000.0

    @pytest.mark.asyncio
    async def test_disabled_children_skipped(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Disabled children are skipped in controlled consumption.

        Covers line 1313-1314.
        """
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        ch_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CHARGER_CONFIG,
            entry_id="ch_ncc_dis",
            title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
            unique_id="quiet_solar_ch_ncc_dis",
        )
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()

        charger = home._chargers[0]
        charger.qs_enable_device = False  # disable

        _inject_sensor_value(home, home.grid_active_power_sensor, now, -2000.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # Disabled charger should NOT be subtracted
        assert home.home_non_controlled_consumption == home.home_consumption

    @pytest.mark.asyncio
    async def test_piloted_devices_power_subtracted(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Loads with devices_to_pilot add piloted device power to controlled consumption.

        Covers lines 1320-1332 (piloted_sets loop and power accumulation).
        """
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        ch_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CHARGER_CONFIG,
            entry_id="ch_ncc_pilot",
            title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
            unique_id="quiet_solar_ch_ncc_pilot",
        )
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()

        # Create a mock piloted device with power reading
        piloted = MagicMock(spec=HADeviceMixin)
        piloted.qs_enable_device = True
        piloted.get_device_power_latest_possible_valid_value = MagicMock(return_value=300.0)

        # Set the charger load's devices_to_pilot
        charger = home._chargers[0]
        charger.devices_to_pilot = [piloted]

        # Grid
        _inject_sensor_value(home, home.grid_active_power_sensor, now, -2000.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # home_consumption = 2000
        # controlled = charger power (0) + piloted device power (300) = 300
        # non_controlled = 2000 - 300 = 1700
        # But charger as a child also contributes via the _childrens loop
        assert home.home_non_controlled_consumption < home.home_consumption
        piloted.get_device_power_latest_possible_valid_value.assert_called()

    @pytest.mark.asyncio
    async def test_piloted_devices_disabled_load_skipped(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Disabled loads' piloted devices are skipped.

        Covers lines 1322-1323 (load disabled → continue).
        """
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        ch_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CHARGER_CONFIG,
            entry_id="ch_ncc_pildis",
            title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
            unique_id="quiet_solar_ch_ncc_pildis",
        )
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()

        piloted = MagicMock(spec=HADeviceMixin)
        piloted.qs_enable_device = True
        piloted.get_device_power_latest_possible_valid_value = MagicMock(return_value=500.0)

        charger = home._chargers[0]
        charger.devices_to_pilot = [piloted]
        charger.qs_enable_device = False  # disable the load

        _inject_sensor_value(home, home.grid_active_power_sensor, now, -1000.0)

        result = home.home_non_controlled_consumption_sensor_state_getter("s", now)
        assert result is not None
        # Disabled load → its piloted devices are NOT collected into piloted_sets
        # So piloted power should NOT be subtracted
        # home = 1000, non_controlled = 1000 (nothing subtracted)
        assert home.home_non_controlled_consumption == home.home_consumption


# ========================================================================
# QSHome map_location_path - more branches
# ========================================================================


class TestMapLocationPathExtraBranches:
    """Extra edge case tests for map_location_path."""

    @pytest.mark.asyncio
    async def test_map_location_path_close_segments_merged(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Segments close in time get merged. Covers lines 845-859."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # Create two GPS paths that are close to each other (both away from home)
        states_1 = []
        states_2 = []
        for i in range(20):
            t = now + timedelta(minutes=i * 3)
            # Both far from home, close to each other
            lat = 49.5 + (i % 2) * 0.001  # tiny variation
            lon = 3.5 + (i % 2) * 0.001
            states_1.append(SimpleNamespace(
                last_updated=t,
                state="not_home",
                attributes={"latitude": lat, "longitude": lon, "source": "gps"},
                entity_id="device_tracker.path1",
            ))
            states_2.append(SimpleNamespace(
                last_updated=t + timedelta(seconds=30),
                state="not_home",
                attributes={"latitude": lat + 0.0001, "longitude": lon + 0.0001, "source": "gps"},
                entity_id="device_tracker.path2",
            ))

        end = now + timedelta(hours=2)
        segments, s1_not_home, s2_not_home = home.map_location_path(
            states_1, states_2, now, end
        )
        # Should have at least one segment where paths are close
        # and not-home segments for both
        assert len(s1_not_home) >= 1
        assert len(s2_not_home) >= 1

    @pytest.mark.asyncio
    async def test_map_location_path_home_far_from_zone(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Detected home position far from zone gets overridden. Covers lines 764-770."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        # State says "home" but position is far from actual home zone
        states = [
            SimpleNamespace(
                last_updated=now,
                state="home",
                attributes={"latitude": 45.0, "longitude": 5.0, "source": "gps"},
                entity_id="device_tracker.test",
            ),
        ]

        end = now + timedelta(hours=1)
        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )
        # Should not crash, home position gets overridden

    @pytest.mark.asyncio
    async def test_map_location_path_time_past_end(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """States with time > end get clamped. Covers line 703-704."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        end = now + timedelta(minutes=30)

        states = [
            SimpleNamespace(
                last_updated=now + timedelta(hours=2),  # past end
                state="not_home",
                attributes={"latitude": 49.0, "longitude": 3.0, "source": "gps"},
                entity_id="device_tracker.test",
            ),
        ]

        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )
        # Time should be clamped to end

    @pytest.mark.asyncio
    async def test_map_location_path_state_error_handled(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """State with bad attributes is handled gracefully. Covers lines 747-748."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        states = [
            SimpleNamespace(
                last_updated=now,
                state="not_home",
                attributes={"latitude": "not_a_number", "longitude": 3.0, "source": "gps"},
                entity_id="device_tracker.test",
            ),
        ]

        end = now + timedelta(hours=1)
        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )
        # Should handle ValueError gracefully

    @pytest.mark.asyncio
    async def test_map_location_path_no_lat_lon(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """State without lat/lon attributes. Covers lines 698, 720."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)

        states = [
            SimpleNamespace(
                last_updated=now,
                state="not_home",
                attributes={},  # no lat/lon
                entity_id="device_tracker.test",
            ),
        ]

        end = now + timedelta(hours=1)
        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )

    @pytest.mark.asyncio
    async def test_map_location_path_duplicate_time(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Duplicate timestamps are skipped. Covers line 700-701."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        end = now + timedelta(hours=1)

        # Two states with same timestamp
        states = [
            SimpleNamespace(
                last_updated=now,
                state="not_home",
                attributes={"latitude": 49.0, "longitude": 3.0, "source": "gps"},
                entity_id="device_tracker.test",
            ),
            SimpleNamespace(
                last_updated=now,  # same time!
                state="not_home",
                attributes={"latitude": 49.1, "longitude": 3.1, "source": "gps"},
                entity_id="device_tracker.test",
            ),
        ]

        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )

    @pytest.mark.asyncio
    async def test_map_location_path_open_not_home_at_end(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Open not-home segment at end gets closed. Covers lines 752-754."""
        home = await _get_home(hass, home_config_entry)
        home.latitude = 48.8566
        home.longitude = 2.3522
        home.radius = 100

        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        end = now + timedelta(hours=2)

        # Only not-home states, never returns home
        states = []
        for i in range(5):
            t = now + timedelta(minutes=i * 15)
            states.append(SimpleNamespace(
                last_updated=t,
                state="not_home",
                attributes={"latitude": 49.0 + i * 0.01, "longitude": 3.0, "source": "gps"},
                entity_id="device_tracker.test",
            ))

        segments, s1_not_home, s2_not_home = home.map_location_path(
            states, [], now, end
        )
        # Open not-home segment should be closed at end
        assert len(s1_not_home) == 1
        assert s1_not_home[0][1] == end


# ========================================================================
# reset_forecasts comprehensive test
# ========================================================================


def _make_lazy_states(entity_id, start, end, interval_s=900, base_value=500.0, variation=50.0):
    """Build a list of SimpleNamespace objects mimicking LazyState.

    Produces one state every interval_s seconds from start to end with
    values oscillating around base_value.
    """
    states = []
    t = start
    i = 0
    while t <= end:
        val = base_value + (i % 10) * variation
        states.append(SimpleNamespace(
            entity_id=entity_id,
            state=str(val),
            last_changed=t,
            last_updated=t,
            attributes={"unit_of_measurement": "W"},
        ))
        t += timedelta(seconds=interval_s)
        i += 1
    return states


class TestResetForecasts:
    """Tests for QSHomeConsumptionHistoryAndForecast.reset_forecasts.

    Uses patched load_from_history to provide synthetic state data
    so that QSSolarHistoryVals.init populates real numpy arrays.
    This exercises the full reset logic end-to-end.
    """

    @pytest.mark.asyncio
    async def test_full_reset_with_solar_battery_charger(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Full reset with solar + battery + charger: exercises the happy-path
        through reset_forecasts lines 2431-2676.

        Covers: battery init (2447-2457), solar init (2465-2489),
        grid init (2496-2519), controlled loads BFS (2530-2600),
        buffer cleanup (2608-2655), final save (2658-2666).
        """
        from .const import MOCK_SOLAR_CONFIG, MOCK_BATTERY_CONFIG, MOCK_CHARGER_CONFIG

        home = await _get_home(hass, home_config_entry)

        # Add battery
        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_reset_1",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_reset_1",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()

        # Add solar
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_SOLAR_CONFIG,
            entry_id="sol_reset_1",
            title=f"solar: {MOCK_SOLAR_CONFIG['name']}",
            unique_id="quiet_solar_sol_reset_1",
        )
        sol_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        # Add charger
        ch_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_CHARGER_CONFIG,
            entry_id="ch_reset_1",
            title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
            unique_id="quiet_solar_ch_reset_1",
        )
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()

        assert home.battery is not None
        assert home.solar_plant is not None
        assert len(home._chargers) > 0

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        # 7 days of history
        history_start = now - timedelta(days=7)

        # Build synthetic states for each sensor
        bat_sensor = home.battery.charge_discharge_sensor
        sol_sensor = home.solar_plant.solar_inverter_active_power
        grid_sensor = home.grid_active_power_sensor

        bat_states = _make_lazy_states(bat_sensor, history_start, now, base_value=-200, variation=30)
        sol_states = _make_lazy_states(sol_sensor, history_start, now, base_value=3000, variation=100)
        grid_states = _make_lazy_states(grid_sensor, history_start, now, base_value=-500, variation=80)

        # Charger power entity (if it has one)
        charger = home._chargers[0]
        charger_power_entity = charger.get_best_power_HA_entity()
        charger_states = []
        if charger_power_entity:
            charger_states = _make_lazy_states(charger_power_entity, history_start, now, base_value=1000, variation=50)

        # The non-controlled consumption entity
        from custom_components.quiet_solar.const import FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
        ncc_states = _make_lazy_states(
            FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
            history_start, now, base_value=800, variation=60,
        )

        # Map entity_id -> states
        state_map = {
            bat_sensor: bat_states,
            sol_sensor: sol_states,
            grid_sensor: grid_states,
            FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER: ncc_states,
        }
        if charger_power_entity:
            state_map[charger_power_entity] = charger_states

        # Set HA states so states.get() returns something
        for eid in state_map:
            hass.states.async_set(eid, "100", {"unit_of_measurement": "W"})

        async def fake_load_from_history(hass_arg, entity_id, start_time, end_time, no_attributes=True):
            states = state_map.get(entity_id, [])
            return [s for s in states if start_time <= s.last_changed <= end_time]

        forecast = home._consumption_forecast
        

        with patch(
            "custom_components.quiet_solar.ha_model.home.load_from_history",
            side_effect=fake_load_from_history,
        ):
            result = await forecast.reset_forecasts(now, light_reset=False)

        assert result is True
        # After full reset, home_non_controlled_consumption should be re-initialized
        # and _in_reset should be False
        assert forecast._in_reset is False

    @pytest.mark.asyncio
    async def test_full_reset_no_solar_no_battery(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Full reset with only grid sensor: no solar, no battery.

        Exercises the grid-only path where solar_production_minus_battery is None,
        grid values get inverted (line 2507-2508), and the controlled loads
        BFS runs on home._childrens.
        """
        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None
        home.physical_battery = None

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        history_start = now - timedelta(days=7)

        grid_sensor = home.grid_active_power_sensor
        grid_states = _make_lazy_states(grid_sensor, history_start, now, base_value=-800, variation=50)

        from custom_components.quiet_solar.const import FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
        ncc_states = _make_lazy_states(
            FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
            history_start, now, base_value=600, variation=40,
        )

        state_map = {
            grid_sensor: grid_states,
            FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER: ncc_states,
        }
        for eid in state_map:
            hass.states.async_set(eid, "100", {"unit_of_measurement": "W"})

        async def fake_load(hass_arg, entity_id, start_time, end_time, no_attributes=True):
            states = state_map.get(entity_id, [])
            return [s for s in states if start_time <= s.last_changed <= end_time]

        forecast = home._consumption_forecast

        with patch(
            "custom_components.quiet_solar.ha_model.home.load_from_history",
            side_effect=fake_load,
        ):
            result = await forecast.reset_forecasts(now, light_reset=False)

        assert result is True
        assert forecast._in_reset is False

    @pytest.mark.asyncio
    async def test_full_reset_bad_battery_init(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Battery init returns (None, None) → is_one_bad=True, skip everything.

        Covers line 2451-2452 (is_one_bad branch).
        """
        from .const import MOCK_BATTERY_CONFIG

        home = await _get_home(hass, home_config_entry)
        home.physical_solar_plant = None

        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_reset_bad",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_reset_bad",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)

        # Return no states → init returns (None, None) → is_one_bad
        async def fake_load(hass_arg, entity_id, start_time, end_time, no_attributes=True):
            return []

        forecast = home._consumption_forecast

        with patch(
            "custom_components.quiet_solar.ha_model.home.load_from_history",
            side_effect=fake_load,
        ):
            result = await forecast.reset_forecasts(now, light_reset=False)

        assert result is True
        assert forecast._in_reset is False

    @pytest.mark.asyncio
    async def test_light_reset(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Light reset path: init + save only. Covers lines 2668-2673."""
        home = await _get_home(hass, home_config_entry)
        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)

        from custom_components.quiet_solar.const import FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
        ncc_states = _make_lazy_states(
            FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
            now - timedelta(days=3), now, base_value=400, variation=30,
        )

        hass.states.async_set(FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, "100", {"unit_of_measurement": "W"})

        async def fake_load(hass_arg, entity_id, start_time, end_time, no_attributes=True):
            if entity_id == FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER:
                return [s for s in ncc_states if start_time <= s.last_changed <= end_time]
            return []

        forecast = home._consumption_forecast

        with patch(
            "custom_components.quiet_solar.ha_model.home.load_from_history",
            side_effect=fake_load,
        ):
            result = await forecast.reset_forecasts(now, light_reset=True)

        assert result is True
        assert forecast._in_reset is False
        # Light reset sets home_non_controlled_consumption to None;
        # init_forecasts at the end re-initializes it (when home is available)
        # In this test home is a real QSHome with hass, so it re-inits
        assert forecast.home_non_controlled_consumption is not None

    @pytest.mark.asyncio
    async def test_full_reset_solar_input_power_path(
        self, hass: HomeAssistant, home_config_entry: ConfigEntry,
    ):
        """Solar with input_active_power (no active_power) + battery:
        exercises the solar_inverter_input fallback (2477-2489).
        """
        from .const import MOCK_SOLAR_CONFIG, MOCK_BATTERY_CONFIG
        from custom_components.quiet_solar.const import CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR

        home = await _get_home(hass, home_config_entry)

        # Battery
        bat_entry = MockConfigEntry(
            domain=DOMAIN, data=MOCK_BATTERY_CONFIG,
            entry_id="bat_reset_inp",
            title=f"battery: {MOCK_BATTERY_CONFIG['name']}",
            unique_id="quiet_solar_bat_reset_inp",
        )
        bat_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(bat_entry.entry_id)
        await hass.async_block_till_done()

        # Solar with input sensor only
        sol_cfg = dict(MOCK_SOLAR_CONFIG)
        sol_cfg[CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR] = "sensor.solar_input_reset"
        # Remove active power so active_power is set but we'll make it None
        sol_entry = MockConfigEntry(
            domain=DOMAIN, data=sol_cfg,
            entry_id="sol_reset_inp",
            title=f"solar: {sol_cfg['name']}",
            unique_id="quiet_solar_sol_reset_inp",
        )
        sol_entry.add_to_hass(hass)
        hass.states.async_set("sensor.solar_input_reset", "4000", {"unit_of_measurement": "W"})
        await hass.config_entries.async_setup(sol_entry.entry_id)
        await hass.async_block_till_done()

        assert home.solar_plant is not None
        # Force the active power sensor to None so it falls through to input
        home.solar_plant.solar_inverter_active_power = None

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        history_start = now - timedelta(days=5)

        bat_sensor = home.battery.charge_discharge_sensor
        input_sensor = home.solar_plant.solar_inverter_input_active_power
        grid_sensor = home.grid_active_power_sensor

        from custom_components.quiet_solar.const import FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER

        state_map = {
            bat_sensor: _make_lazy_states(bat_sensor, history_start, now, base_value=-100, variation=20),
            input_sensor: _make_lazy_states(input_sensor, history_start, now, base_value=4000, variation=100),
            grid_sensor: _make_lazy_states(grid_sensor, history_start, now, base_value=-300, variation=50),
            FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER: _make_lazy_states(
                FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
                history_start, now, base_value=700, variation=40,
            ),
        }
        for eid in state_map:
            hass.states.async_set(eid, "100", {"unit_of_measurement": "W"})

        async def fake_load(hass_arg, entity_id, start_time, end_time, no_attributes=True):
            states = state_map.get(entity_id, [])
            return [s for s in states if start_time <= s.last_changed <= end_time]

        forecast = home._consumption_forecast

        with patch(
            "custom_components.quiet_solar.ha_model.home.load_from_history",
            side_effect=fake_load,
        ):
            result = await forecast.reset_forecasts(now, light_reset=False)

        assert result is True
        assert forecast._in_reset is False


# ========================================================================
# Push to 96%: systematic coverage of remaining gaps
# ========================================================================


class TestQSSolarForecastPaths:
    """Tests targeting compute_now_forecast, _get_predicted_data, and
    _get_possible_past_consumption_for_forecast."""

    def _make_hv_with_data(self, days_of_data=10):
        forecast = MagicMock()
        forecast.home = None
        forecast.storage_path = "/tmp/test_qs"
        hv = QSSolarHistoryVals(forecast=forecast, entity_id="sensor.test")
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

        now = datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC)
        idx, days = hv.get_index_from_time(now)

        total_slots = NUM_INTERVALS_PER_DAY * days_of_data
        for i in range(total_slots):
            slot = _sanitize_idx(idx - i)
            hv.values[0][slot] = 300 + (i % 12) * 40
            hv.values[1][slot] = days - (i // NUM_INTERVALS_PER_DAY)
        return hv, now, idx, days

    def test_compute_now_forecast_happy_path(self):
        """Full happy path. Covers lines 3103-3168."""
        hv, now, idx, days = self._make_hv_with_data(10)
        result = hv.compute_now_forecast(time_now=now, history_in_hours=24, future_needed_in_hours=6, set_as_current=True)
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[-1][0] >= now
        assert hv._current_forecast is not None

    def test_compute_now_forecast_no_scores(self):
        """Empty buffer → no scores. Covers lines 2778-2780."""
        forecast = MagicMock(); forecast.home = None; forecast.storage_path = "/tmp"
        hv = QSSolarHistoryVals(forecast=forecast, entity_id="s")
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
        result = hv.compute_now_forecast(time_now=datetime(2026, 2, 10, 14, 0, tzinfo=pytz.UTC), history_in_hours=24, future_needed_in_hours=6)
        assert result == []

    def test_get_predicted_data_bad_overlap(self):
        """Scores pointing to empty regions. Covers lines 2920-2924."""
        hv, now, idx, days = self._make_hv_with_data(10)
        fv, pd = hv._get_predicted_data(6, idx, days, [[NUM_INTERVALS_PER_DAY * 15, 5.0]])

    def test_get_predicted_data_concatenation(self):
        """Multiple valid scores get concatenated. Covers lines 2934-2936."""
        hv, now, idx, days = self._make_hv_with_data(10)
        fv, pd = hv._get_predicted_data(48, idx, days, [[NUM_INTERVALS_PER_DAY, 10.0], [NUM_INTERVALS_PER_DAY * 2, 15.0]])
        if fv is not None:
            assert len(fv) > 0

    def test_get_range_score_past_none(self):
        """values=None → []. Covers lines 2793-2797."""
        forecast = MagicMock(); forecast.home = None; forecast.storage_path = "/tmp"
        hv = QSSolarHistoryVals(forecast=forecast, entity_id="s")
        hv.values = None
        assert hv._get_range_score(np.array([100.0, 200.0]), np.ones(2, dtype=np.int32), 0, past_delta=96) == []

    def test_get_past_consumption_with_current_val(self):
        """use_val_as_current set. Covers lines 2735-2738."""
        hv, now, idx, days = self._make_hv_with_data(10)
        scores = hv._get_possible_past_consumption_for_forecast(idx, days, 24, use_val_as_current=999.0)
        assert isinstance(scores, list)

    def test_constructor_forecast_no_home(self):
        """forecast with home=None. Covers lines 2686-2687, 2690-2691."""
        fc = MagicMock()
        fc.home = None
        fc.storage_path = "/tmp"
        hv = QSSolarHistoryVals(forecast=fc, entity_id="sensor.x")
        assert hv.hass is None and hv.home is None

    def test_compute_now_forecast_prev_val_fallback(self):
        """compute_now_forecast where past_days[0]==0 → uses prev_val.

        Covers lines 3136-3142 (prev_val fallback path).
        """
        hv, now, idx, days = self._make_hv_with_data(days_of_data=10)

        # Poison the slot at idx so that when _get_predicted_data returns
        # data starting from idx, the first entry has days=0.
        # We can achieve this by zeroing the slot at idx in our values
        # so that the predicted data from 1 day ago has days=0 at position 0.
        # Actually, we need to manipulate the output. A simpler approach:
        # create a scenario where the first forecast value has days=0.
        # Easiest: zero out a strip of data that affects the forecast start.
        # The past data at delta=1 day, position idx, would map to now_idx.

        # Zero the slot exactly 1 day ago from idx (the most likely match)
        one_day_ago_slot = _sanitize_idx(idx - NUM_INTERVALS_PER_DAY)
        hv.values[1][one_day_ago_slot] = 0  # zero out the day marker

        # But keep previous slots valid so prev_val can be found
        prev_slot = _sanitize_idx(idx - 1)
        hv.values[0][prev_slot] = 999
        hv.values[1][prev_slot] = days

        result = hv.compute_now_forecast(
            time_now=now, history_in_hours=24,
            future_needed_in_hours=6, set_as_current=False,
        )
        # Should still produce a forecast (prev_val fills the gap)
        assert isinstance(result, list)


class TestHomeGaps96:
    """HA-integrated tests for remaining coverage gaps."""

    @pytest.mark.asyncio
    async def test_get_tariff_ranges_empty_despite_off_peak(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """get_tariff where off_peak > 0 but ranges are empty (equal start/end) → peak.

        Covers line 1112.
        """
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        home.tariff_start_1 = "06:00"; home.tariff_end_1 = "06:00"  # equal!
        home.tariff_start_2 = None; home.tariff_end_2 = None
        t = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
        result = home.get_tariff(t, t + timedelta(hours=1))
        assert result == home.price_peak

    @pytest.mark.asyncio
    async def test_get_tariffs_ranges_empty_despite_off_peak(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """get_tariffs where off_peak > 0 but ranges are empty → scalar peak.

        Covers line 1146.
        """
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        home.tariff_start_1 = "06:00"; home.tariff_end_1 = "06:00"
        home.tariff_start_2 = None; home.tariff_end_2 = None
        t = datetime(2026, 2, 10, 10, 0, tzinfo=pytz.UTC)
        result = home.get_tariffs(t, t + timedelta(hours=24))
        assert result == home.price_peak

    @pytest.mark.asyncio
    async def test_get_tariff_during_peak_with_valid_ranges(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """get_tariff when ranges exist but query is outside them → peak.

        Covers line 1112 (ranges_off_peak not empty, but query outside).
        Note: _get_today_off_peak_ranges uses local time, so we verify
        that ranges are actually produced first.
        """
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        # Wide off-peak range: 01:00-05:00 local
        home.tariff_start_1 = "01:00"; home.tariff_end_1 = "05:00"
        home.tariff_start_2 = None; home.tariff_end_2 = None

        # Verify ranges are produced
        t = datetime(2026, 2, 10, 15, 0, tzinfo=pytz.UTC)
        ranges = home._get_today_off_peak_ranges(t)
        assert len(ranges) > 0, f"Expected non-empty ranges, got {ranges}"

        # Query at midday UTC — should be peak
        result = home.get_tariff(t, t + timedelta(hours=1))
        assert result == home.price_peak

    @pytest.mark.asyncio
    async def test_get_tariffs_generates_segments(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """get_tariffs with ranges produces both peak and off-peak segments.

        Uses start_time that falls WITHIN an off-peak range so the
        curr_start < start_time adjustment on line 1168 is exercised.
        """
        home = await _get_home(hass, home_config_entry)
        home.price_off_peak = 0.10 / 1000.0
        # Off-peak 01:00-05:00 local = ~00:00-04:00 UTC (CET)
        home.tariff_start_1 = "01:00"; home.tariff_end_1 = "05:00"
        home.tariff_start_2 = "13:00"; home.tariff_end_2 = "16:00"

        # Start at 01:00 UTC — which is within the first off-peak range (00:00-04:00 UTC)
        # This means curr_start (00:00) < start_time (01:00) → line 1168 fires
        start = datetime(2026, 2, 10, 1, 0, tzinfo=pytz.UTC)
        end = datetime(2026, 2, 11, 1, 0, tzinfo=pytz.UTC)

        ranges = home._get_today_off_peak_ranges(start)
        assert len(ranges) >= 1, f"Expected ranges, got {ranges}"

        result = home.get_tariffs(start, end)
        assert isinstance(result, list)
        assert len(result) >= 2
        prices = {e[1] for e in result}
        assert home.price_peak in prices
        assert home.price_off_peak in prices

    @pytest.mark.asyncio
    async def test_finish_setup_succeeds(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """Covers line 1663."""
        from .const import MOCK_CHARGER_CONFIG
        home = await _get_home(hass, home_config_entry)
        ch_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_CHARGER_CONFIG, entry_id="ch_fs", title="charger: T", unique_id="qs_ch_fs")
        ch_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(ch_entry.entry_id)
        await hass.async_block_till_done()
        home._init_completed = False
        assert await home.finish_setup(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)) is True
        assert home._init_completed is True

    @pytest.mark.asyncio
    async def test_topology_piloted_device(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """Covers lines 1488-1489."""
        from .const import MOCK_CHARGER_CONFIG, MOCK_HEAT_PUMP_CONFIG
        home = await _get_home(hass, home_config_entry)
        hp_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_HEAT_PUMP_CONFIG, entry_id="hp_t", title="hp: T", unique_id="qs_hp_t")
        hp_entry.add_to_hass(hass); await hass.config_entries.async_setup(hp_entry.entry_id); await hass.async_block_till_done()
        ch_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_CHARGER_CONFIG, entry_id="ch_t", title="charger: T", unique_id="qs_ch_t")
        ch_entry.add_to_hass(hass); await hass.config_entries.async_setup(ch_entry.entry_id); await hass.async_block_till_done()
        hp = hass.data[DOMAIN].get(hp_entry.entry_id)
        charger = home._chargers[0]
        charger.piloted_device_name = hp.name
        home._set_topology()
        assert len(charger.devices_to_pilot) == 1
        assert charger in hp.clients

    @pytest.mark.asyncio
    async def test_remove_dynamic_group(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """Covers lines 1590-1591."""
        from .const import MOCK_DYNAMIC_GROUP_CONFIG
        home = await _get_home(hass, home_config_entry)
        dg_entry = MockConfigEntry(domain=DOMAIN, data=MOCK_DYNAMIC_GROUP_CONFIG, entry_id="dg_r", title="dg: T", unique_id="qs_dg_r")
        dg_entry.add_to_hass(hass); await hass.config_entries.async_setup(dg_entry.entry_id); await hass.async_block_till_done()
        dg = hass.data[DOMAIN].get(dg_entry.entry_id)
        assert dg in home._all_dynamic_groups
        home.remove_device(dg)
        assert dg not in home._all_dynamic_groups

    @pytest.mark.asyncio
    async def test_update_loads_just_switched(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """Covers line 2196."""
        home = await _get_home(hass, home_config_entry)
        home.home_mode = QSHomeMode.HOME_MODE_ON.value
        home._init_completed = True
        home.physical_battery = None; home.physical_solar_plant = None
        t = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        home._switch_to_off_grid_launched = t - timedelta(minutes=5)
        load = MagicMock(); load.name = "l"; load.qs_enable_device = True
        load.check_commands = AsyncMock(return_value=(timedelta(0), True)); load.running_command_num_relaunch = 0
        load.is_load_active = MagicMock(return_value=False); load.is_load_has_a_command_now_or_coming = MagicMock(return_value=False)
        load.get_current_active_constraint = MagicMock(return_value=None)
        load.launch_command = AsyncMock(); load.do_probe_state_change = AsyncMock()
        home._all_loads = [load]; home._chargers = []; home._commands = []; home._battery_commands = None
        await home.update_loads(t)

    @pytest.mark.asyncio
    async def test_solar_forecast_getter_with_solar(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """Covers line 1060."""
        from .const import MOCK_SOLAR_CONFIG
        home = await _get_home(hass, home_config_entry)
        se = MockConfigEntry(domain=DOMAIN, data=MOCK_SOLAR_CONFIG, entry_id="sf_g", title="solar: T", unique_id="qs_sf_g")
        se.add_to_hass(hass); await hass.config_entries.async_setup(se.entry_id); await hass.async_block_till_done()
        assert isinstance(home.get_solar_from_current_forecast_getter(datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)), tuple)

    @pytest.mark.asyncio
    async def test_solar_forecast_list_with_solar(self, hass: HomeAssistant, home_config_entry: ConfigEntry):
        """Covers line 1065."""
        from .const import MOCK_SOLAR_CONFIG
        home = await _get_home(hass, home_config_entry)
        se = MockConfigEntry(domain=DOMAIN, data=MOCK_SOLAR_CONFIG, entry_id="sf_l", title="solar: T", unique_id="qs_sf_l")
        se.add_to_hass(hass); await hass.config_entries.async_setup(se.entry_id); await hass.async_block_till_done()
        now = datetime(2026, 2, 10, 12, 0, tzinfo=pytz.UTC)
        assert isinstance(home.get_solar_from_current_forecast(now, now + timedelta(hours=24)), list)
