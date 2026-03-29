"""Tests for Story 3.14: Solar forecast scoring and history tracking.

Covers:
- get_historical_data() on QSSolarHistoryVals
- compute_score() new implementation on QSSolarProvider
- update_forecast_probers() on QSSolar
- solar_forecast_set_and_reset()
- update_consumption_and_forecast_history()
- End-to-end scoring with synthetic daily data
"""

from __future__ import annotations

import datetime
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    SOLCAST_SOLAR_DOMAIN,
    QSForecastSolarSensors,
)
from custom_components.quiet_solar.ha_model.home import (
    BUFFER_SIZE_IN_INTERVALS,
    INTERVALS_MN,
    NUM_INTERVAL_PER_HOUR,
    QSHomeSolarAndConsumptionHistoryAndForecast,
    QSSolarHistoryVals,
)
from custom_components.quiet_solar.ha_model.solar import (
    QSSolar,
    QSSolarProvider,
)
from tests.conftest import FakeConfigEntry

# ============================================================================
# Helpers
# ============================================================================


class _TestProvider(QSSolarProvider):
    """Concrete provider for testing abstract base class."""

    async def get_power_series_from_orchestrator(self, orchestrator, start_time=None, end_time=None):
        return getattr(orchestrator, "power_series", [])


def _make_solar(fake_hass, providers_config=None, **kwargs):
    """Create a QSSolar instance for testing."""
    from homeassistant.const import CONF_NAME

    config_entry = FakeConfigEntry(entry_id="solar_scoring_test", data={CONF_NAME: "Test Solar"})
    home = MagicMock()
    home.physical_3p = False
    home.voltage = 230
    home.hass = fake_hass
    home.force_update_all = AsyncMock()
    home.solar_and_consumption_forecast = None

    config = {
        CONF_NAME: "Test Solar",
        "hass": fake_hass,
        "config_entry": config_entry,
        "home": home,
    }
    if providers_config is not None:
        config[CONF_SOLAR_FORECAST_PROVIDERS] = providers_config
    config.update(kwargs)
    return QSSolar(**config)


def _make_history_vals(time_now, hours=24, value_fn=None, storage_path="/tmp"):
    """Create a QSSolarHistoryVals with synthetic data filled in.

    value_fn(hour_offset) -> power_value, called for each 15-min interval.
    Default: a bell curve peaking at solar noon.
    """
    forecast = MagicMock()
    forecast.home = MagicMock()
    forecast.home.hass = None
    forecast.storage_path = storage_path

    hv = QSSolarHistoryVals(entity_id="sensor.test_power", forecast=forecast)
    hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

    if value_fn is None:
        # Default: bell-curve solar pattern
        def value_fn(h):
            if 6 <= h <= 18:
                return max(0.0, 1000.0 * np.sin(np.pi * (h - 6) / 12))
            return 0.0

    num_intervals = hours * NUM_INTERVAL_PER_HOUR
    now_idx, now_days = hv.get_index_from_time(time_now)

    for i in range(num_intervals):
        idx = (now_idx - num_intervals + 1 + i) % BUFFER_SIZE_IN_INTERVALS
        hour_offset = (i - num_intervals) / NUM_INTERVAL_PER_HOUR
        # Hours since midnight for the bell-curve
        interval_time = time_now + timedelta(hours=hour_offset)
        local_hour = interval_time.hour + interval_time.minute / 60
        hv.values[0, idx] = value_fn(local_hour)
        hv.values[1, idx] = now_days  # Mark as having data

    return hv


# ============================================================================
# get_historical_data tests
# ============================================================================


class TestGetHistoricalData:
    """Test QSSolarHistoryVals.get_historical_data()."""

    def test_returns_empty_when_values_none(self):
        """get_historical_data returns [] when values is None."""
        forecast = MagicMock()
        forecast.storage_path = "/tmp"
        hv = QSSolarHistoryVals(entity_id="sensor.test", forecast=forecast)
        hv.values = None
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert hv.get_historical_data(t, past_hours=24) == []

    def test_returns_empty_when_insufficient_valid_data(self):
        """get_historical_data returns [] when < 60% valid entries."""
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        hv = _make_history_vals(t, hours=24, value_fn=lambda h: 100.0)
        # Zero out the day markers to simulate missing data
        hv.values[1, :] = 0
        assert hv.get_historical_data(t, past_hours=24) == []

    def test_returns_correct_time_series(self):
        """get_historical_data returns proper (time, value) pairs."""
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        hv = _make_history_vals(t, hours=2, value_fn=lambda h: 500.0)

        result = hv.get_historical_data(t, past_hours=2)
        # Should return approximately 2 hours worth of intervals
        assert len(result) > 0
        # All non-zero values should be 500.0
        for ts, val in result:
            assert isinstance(ts, datetime.datetime)
            if val != 0.0:
                assert val == 500.0

    def test_timestamps_are_ordered(self):
        """Timestamps in result are chronologically ordered."""
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        hv = _make_history_vals(t, hours=4, value_fn=lambda h: 100.0)

        result = hv.get_historical_data(t, past_hours=4)
        for i in range(1, len(result)):
            assert result[i][0] > result[i - 1][0]

    def test_zero_day_entries_produce_zero_values(self):
        """Entries where day_vals == 0 should produce value 0.0."""
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        hv = _make_history_vals(t, hours=1, value_fn=lambda h: 999.0)

        now_idx, now_days = hv.get_index_from_time(t)
        # Clear one entry's day marker
        clear_idx = (now_idx - 2) % BUFFER_SIZE_IN_INTERVALS
        hv.values[1, clear_idx] = 0

        result = hv.get_historical_data(t, past_hours=1)
        zero_vals = [v for _, v in result if v == 0.0]
        assert len(zero_vals) >= 1


# ============================================================================
# compute_score tests
# ============================================================================


class TestComputeScore:
    """Test QSSolarProvider.compute_score() with the new history-based API."""

    def _setup_provider_with_histories(self, actuals_fn, forecast_fn, t0, hours=24, sensor_name="qs_solar_forecast_8h"):
        """Wire up a provider with mock histories returning synthetic data."""
        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        # Production actuals
        actual_data = []
        for i in range(hours * NUM_INTERVAL_PER_HOUR):
            ts = t0 - timedelta(hours=hours) + timedelta(minutes=i * INTERVALS_MN)
            actual_data.append((ts, actuals_fn(ts.hour + ts.minute / 60)))

        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = actual_data
        forecast_handler.solar_production_history = mock_prod

        # Forecast history
        fc_data = []
        for i in range(hours * NUM_INTERVAL_PER_HOUR):
            ts = t0 - timedelta(hours=hours) + timedelta(minutes=i * INTERVALS_MN)
            fc_data.append((ts, forecast_fn(ts.hour + ts.minute / 60)))

        mock_fc = MagicMock(spec=QSSolarHistoryVals)
        mock_fc.get_historical_data.return_value = fc_data
        forecast_handler.solar_forecast_history_per_provider = {"prov": {sensor_name: mock_fc}}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        return provider

    def test_perfect_forecast_gives_zero_mae(self):
        """Score is 0 when forecast matches actuals exactly."""
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        fn = lambda h: max(0, 1000 * np.sin(np.pi * max(0, h - 6) / 12)) if 6 <= h <= 18 else 0
        provider = self._setup_provider_with_histories(fn, fn, t0)
        provider.compute_score(t0)
        assert provider.score is not None
        assert provider.score == 0.0

    def test_constant_offset_gives_known_mae(self):
        """Score equals the constant offset between forecast and actuals."""
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider = self._setup_provider_with_histories(
            actuals_fn=lambda h: 800.0,
            forecast_fn=lambda h: 1000.0,
            t0=t0,
        )
        provider.compute_score(t0)
        assert provider.score is not None
        assert abs(provider.score - 200.0) < 0.01

    def test_nighttime_only_gives_none(self):
        """Score is None when all values are 0 (nighttime)."""
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider = self._setup_provider_with_histories(
            actuals_fn=lambda h: 0.0,
            forecast_fn=lambda h: 0.0,
            t0=t0,
        )
        provider.compute_score(t0)
        assert provider.score is None

    def test_prefers_8h_lookahead_over_lower(self):
        """compute_score picks >= 8h sensor first, not lower ones."""
        solar_mock = MagicMock()
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        data_12 = [(t0 + timedelta(hours=h), 1000.0) for h in range(12)]
        data_4 = [(t0 + timedelta(hours=h), 2000.0) for h in range(12)]

        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = data_12

        mock_8h = MagicMock(spec=QSSolarHistoryVals)
        mock_8h.get_historical_data.return_value = data_12
        mock_4h = MagicMock(spec=QSSolarHistoryVals)
        mock_4h.get_historical_data.return_value = data_4

        handler.solar_production_history = mock_prod
        handler.solar_forecast_history_per_provider = {
            "prov": {
                "qs_solar_forecast_4h": mock_4h,
                "qs_solar_forecast_8h": mock_8h,
            }
        }
        solar_mock.home.solar_and_consumption_forecast = handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(t0)
        # 8h sensor returns 1000 = actuals, so MAE = 0
        assert provider.score == 0.0

    def test_fallback_to_largest_available_sensor(self):
        """When no >= 8h sensor exists, falls back to largest available."""
        solar_mock = MagicMock()
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        data = [(t0 + timedelta(hours=h), 500.0) for h in range(12)]
        fc_data = [(t0 + timedelta(hours=h), 700.0) for h in range(12)]

        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = data
        mock_4h = MagicMock(spec=QSSolarHistoryVals)
        mock_4h.get_historical_data.return_value = fc_data

        handler.solar_production_history = mock_prod
        handler.solar_forecast_history_per_provider = {"prov": {"qs_solar_forecast_4h": mock_4h}}
        solar_mock.home.solar_and_consumption_forecast = handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(t0)
        assert provider.score is not None
        assert abs(provider.score - 200.0) < 0.01

    def test_solar_none_returns_early(self):
        """compute_score returns when solar is None."""
        provider = _TestProvider(solar=None, domain="test", provider_name="prov")
        provider.compute_score(datetime.datetime.now(pytz.UTC))
        assert provider.score is None

    def test_no_production_history_returns_early(self):
        """compute_score returns when solar_production_history is None."""
        solar_mock = MagicMock()
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        handler.solar_production_history = None
        solar_mock.home.solar_and_consumption_forecast = handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(datetime.datetime.now(pytz.UTC))
        assert provider.score is None


# ============================================================================
# update_forecast_probers tests
# ============================================================================


class TestUpdateForecastProbers:
    """Test QSSolar.update_forecast_probers()."""

    @pytest.mark.asyncio
    async def test_populates_sensor_values(self, fake_hass):
        """update_forecast_probers fills solar_forecast_sensor_values."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Solcast"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)

        # Set a forecast on the active provider so probers can read it
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider = solar.solar_forecast_providers["Solcast"]
        provider.solar_forecast = [(t0 + timedelta(hours=h), 500.0 + h * 10) for h in range(25)]

        # Also set the aggregate forecast on QSSolar
        solar.solar_forecast = provider.solar_forecast

        await solar.update_forecast_probers(t0)

        # Should have entries for each QSForecastSolarSensors key
        assert len(solar.solar_forecast_sensor_values) > 0

    @pytest.mark.asyncio
    async def test_populates_per_provider_values(self, fake_hass):
        """update_forecast_probers fills per-provider sensor values."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Solcast"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider = solar.solar_forecast_providers["Solcast"]
        provider.solar_forecast = [(t0 + timedelta(hours=h), 500.0 + h * 10) for h in range(25)]
        solar.solar_forecast = provider.solar_forecast

        await solar.update_forecast_probers(t0)

        # Should have entries with provider_name prefix
        for key in solar.solar_forecast_sensor_values_per_provider:
            assert key.startswith("Solcast_")


# ============================================================================
# solar_forecast_set_and_reset tests
# ============================================================================


class TestSolarForecastSetAndReset:
    """Test QSHomeSolarAndConsumptionHistoryAndForecast.solar_forecast_set_and_reset()."""

    @pytest.mark.asyncio
    async def test_init_creates_solar_production_history(self, tmp_path):
        """solar_forecast_set_and_reset creates solar_production_history when solar plant exists."""
        from types import SimpleNamespace

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_inverter_active_power=None,
                solar_inverter_input_active_power="sensor.solar_input",
                solar_forecast_providers={},
            ),
            ha_entities={},
        )
        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        forecast._in_reset = False

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # Patch QSSolarHistoryVals to avoid real file I/O
        mock_hv = MagicMock(spec=QSSolarHistoryVals)
        mock_hv.init = AsyncMock()

        from unittest.mock import patch

        with patch("custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals", return_value=mock_hv):
            await forecast.solar_forecast_set_and_reset(t, for_reset=False)

        assert forecast.solar_production_history is mock_hv

    @pytest.mark.asyncio
    async def test_reset_clears_histories(self, tmp_path):
        """solar_forecast_set_and_reset with for_reset=True clears histories."""
        from types import SimpleNamespace

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_inverter_active_power=None,
                solar_inverter_input_active_power="sensor.solar_input",
                solar_forecast_providers={},
            ),
            ha_entities={},
        )
        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        forecast.solar_production_history = MagicMock()
        forecast.solar_forecast_history = {"a": MagicMock()}
        forecast.solar_forecast_history_per_provider = {"p": {"a": MagicMock()}}

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        mock_hv = MagicMock(spec=QSSolarHistoryVals)
        mock_hv.init = AsyncMock()
        mock_hv.save_values = AsyncMock()

        from unittest.mock import patch

        with patch("custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals", return_value=mock_hv):
            await forecast.solar_forecast_set_and_reset(t, for_reset=True)

        assert forecast.solar_production_history is None
        assert forecast.solar_forecast_history is None
        assert forecast.solar_forecast_history_per_provider is None


# ============================================================================
# update_consumption_and_forecast_history tests
# ============================================================================


class TestUpdateConsumptionAndForecastHistory:
    """Test update_consumption_and_forecast_history()."""

    @pytest.mark.asyncio
    async def test_records_solar_production(self, tmp_path):
        """Records solar production value when history is available."""
        from types import SimpleNamespace

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_production=500.0,
                solar_forecast_sensor_values={"qs_solar_forecast_8h": 600.0},
                solar_forecast_sensor_values_per_provider={},
            ),
            home_non_controlled_consumption=100.0,
        )

        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        forecast._in_reset = False

        mock_consumption = MagicMock()
        mock_consumption.add_value.return_value = False
        mock_consumption.update_current_forecast_if_needed.return_value = False
        forecast.home_non_controlled_consumption = mock_consumption

        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.add_value.return_value = True
        mock_prod.save_values = AsyncMock()
        forecast.solar_production_history = mock_prod

        forecast.solar_forecast_history = None
        forecast.solar_forecast_history_per_provider = None

        forecast.init_forecasts = AsyncMock(return_value=True)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        await forecast.update_consumption_and_forecast_history(t)

        mock_prod.add_value.assert_called_once_with(t, 500.0)
        mock_prod.save_values.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_forecast_sensor_values(self, tmp_path):
        """Records forecast sensor values when history is available."""
        from types import SimpleNamespace

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_production=None,
                solar_forecast_sensor_values={"qs_solar_forecast_8h": 600.0},
                solar_forecast_sensor_values_per_provider={"Solcast_qs_solar_forecast_8h": 650.0},
            ),
            home_non_controlled_consumption=100.0,
        )

        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        forecast._in_reset = False

        mock_consumption = MagicMock()
        mock_consumption.add_value.return_value = False
        mock_consumption.update_current_forecast_if_needed.return_value = False
        forecast.home_non_controlled_consumption = mock_consumption

        forecast.solar_production_history = None

        # Set up forecast history
        mock_fc = MagicMock(spec=QSSolarHistoryVals)
        mock_fc.add_value.return_value = True
        mock_fc.save_values = AsyncMock()
        forecast.solar_forecast_history = {"qs_solar_forecast_8h": mock_fc}

        # Set up per-provider history
        mock_pfc = MagicMock(spec=QSSolarHistoryVals)
        mock_pfc.add_value.return_value = True
        mock_pfc.save_values = AsyncMock()
        forecast.solar_forecast_history_per_provider = {"Solcast": {"qs_solar_forecast_8h": mock_pfc}}

        forecast.init_forecasts = AsyncMock(return_value=True)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        await forecast.update_consumption_and_forecast_history(t)

        mock_fc.add_value.assert_called_once_with(t, 600.0)
        mock_pfc.add_value.assert_called_once_with(t, 650.0)


# ============================================================================
# End-to-end scoring simulation with synthetic daily data
# ============================================================================


class TestEndToEndScoring:
    """Simulate a full day of solar data and verify scoring works correctly."""

    def test_scoring_with_synthetic_daily_data(self):
        """End-to-end: fill histories with synthetic data, run scoring, verify MAE."""
        t_noon = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # Actual production: bell curve, peak 1000W at noon
        def actual_fn(h):
            if 6 <= h <= 18:
                return max(0, 1000.0 * np.sin(np.pi * (h - 6) / 12))
            return 0.0

        # Forecast: consistently 100W over actual
        def forecast_fn(h):
            return actual_fn(h) + 100.0 if actual_fn(h) > 0 else 0.0

        # Build actual histories
        actuals = []
        forecasts = []
        for i in range(24 * NUM_INTERVAL_PER_HOUR):
            ts = t_noon - timedelta(hours=24) + timedelta(minutes=i * INTERVALS_MN)
            h = ts.hour + ts.minute / 60
            actuals.append((ts, actual_fn(h)))
            forecasts.append((ts, forecast_fn(h)))

        solar_mock = MagicMock()
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = actuals
        handler.solar_production_history = mock_prod

        mock_fc = MagicMock(spec=QSSolarHistoryVals)
        mock_fc.get_historical_data.return_value = forecasts
        handler.solar_forecast_history_per_provider = {"solcast": {"qs_solar_forecast_8h": mock_fc}}

        solar_mock.home.solar_and_consumption_forecast = handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="solcast")

        provider.compute_score(t_noon)

        assert provider.score is not None
        # The MAE should be ~100W (the constant offset) for daytime slots
        assert abs(provider.score - 100.0) < 1.0

    def test_scoring_with_two_providers_auto_select(self, fake_hass):
        """End-to-end: two providers, different accuracy, auto-select picks best."""
        t_noon = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Good"},
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Bad"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)

        # Set different scores
        solar.solar_forecast_providers["Good"].score = 50.0
        solar.solar_forecast_providers["Bad"].score = 300.0

        # Both providers are non-stale
        for p in solar.solar_forecast_providers.values():
            p._latest_successful_forecast_time = t_noon

        # Auto-select should pick the one with lower score
        solar.auto_select_best_provider()
        assert solar.active_provider_name == "Good"

    def test_scoring_cycle_runs_and_skips(self, fake_hass):
        """Scoring cycle runs once per half-day (local time), skips duplicates."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.compute_score = MagicMock()

        # First call runs the cycle
        t1 = datetime.datetime(2024, 6, 15, 8, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(t1)
        assert provider.compute_score.call_count == 1

        # Same half-day should be skipped
        provider.compute_score.reset_mock()
        solar._run_scoring_cycle(t1 + timedelta(hours=1))
        assert provider.compute_score.call_count == 0

        # Much later (next day) should run again
        provider.compute_score.reset_mock()
        t_next_day = t1 + timedelta(days=1)
        solar._run_scoring_cycle(t_next_day)
        assert provider.compute_score.call_count == 1


# ============================================================================
# get_forecast_histories_for_provider tests
# ============================================================================


class TestGetForecastHistoriesForProvider:
    """Test get_forecast_histories_for_provider()."""

    def test_returns_histories_for_known_provider(self):
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_hv = MagicMock(spec=QSSolarHistoryVals)
        handler.solar_forecast_history_per_provider = {"prov_a": {"8h": mock_hv}}

        result = handler.get_forecast_histories_for_provider("prov_a")
        assert result == {"8h": mock_hv}

    def test_returns_empty_dict_for_unknown_provider(self):
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        handler.solar_forecast_history_per_provider = {"prov_a": {"8h": MagicMock()}}

        result = handler.get_forecast_histories_for_provider("unknown")
        assert result == {}

    def test_returns_empty_when_none(self):
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        handler.solar_forecast_history_per_provider = None

        result = handler.get_forecast_histories_for_provider("any")
        assert result == {}


# ============================================================================
# _compute_mae_from_aligned edge cases
# ============================================================================


class TestComputeMaeFromAligned:
    """Test _compute_mae_from_aligned static method."""

    def test_mixed_daytime_nighttime(self):
        """Only daytime slots (at least one > 0) contribute to MAE."""
        t = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        # 3 night (both 0), 2 day (offset 100)
        fc = [
            (t, 0.0),
            (t + timedelta(hours=1), 0.0),
            (t + timedelta(hours=2), 0.0),
            (t + timedelta(hours=6), 500.0),
            (t + timedelta(hours=7), 600.0),
        ]
        ac = [
            (t, 0.0),
            (t + timedelta(hours=1), 0.0),
            (t + timedelta(hours=2), 0.0),
            (t + timedelta(hours=6), 400.0),
            (t + timedelta(hours=7), 500.0),
        ]
        result = QSSolarProvider._compute_mae_from_aligned(fc, ac)
        assert result is not None
        assert abs(result - 100.0) < 0.01

    def test_different_lengths_uses_min(self):
        """When lists differ in length, uses min(len(fc), len(ac))."""
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        fc = [(t, 100.0), (t + timedelta(hours=1), 200.0)]
        ac = [(t, 100.0)]
        result = QSSolarProvider._compute_mae_from_aligned(fc, ac)
        assert result is not None
        assert result == 0.0


# ============================================================================
# compute_non_controlled_forecast_intl test
# ============================================================================


class TestComputeNonControlledForecastIntl:
    """Test compute_non_controlled_forecast_intl on the forecast class."""

    def test_calls_compute_now_forecast(self):
        """The method delegates to home_non_controlled_consumption.compute_now_forecast."""
        from types import SimpleNamespace

        mock_consumption = MagicMock()
        mock_consumption.compute_now_forecast.return_value = [(None, 100.0)]

        mock_home = SimpleNamespace(_period=timedelta(seconds=86400), hass=None)
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=mock_home, storage_path="/tmp")
        handler.home_non_controlled_consumption = mock_consumption

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        result = handler.compute_non_controlled_forecast_intl(t)

        assert result == [(None, 100.0)]
        mock_consumption.compute_now_forecast.assert_called_once()


# ============================================================================
# Additional coverage: dump_for_debug, consumption save, forecast set with ha_entities
# ============================================================================


class TestDumpForDebugSolarHistories:
    """Test dump_for_debug covers solar history branches."""

    @pytest.mark.asyncio
    async def test_dumps_all_solar_histories(self, tmp_path):
        from types import SimpleNamespace

        handler = QSHomeSolarAndConsumptionHistoryAndForecast(
            home=SimpleNamespace(hass=None), storage_path=str(tmp_path)
        )

        mock_consumption = MagicMock()
        mock_consumption.file_name = "consumption.npy"
        mock_consumption.save_values = AsyncMock()
        handler.home_non_controlled_consumption = mock_consumption

        mock_prod = MagicMock()
        mock_prod.file_name = "production.npy"
        mock_prod.save_values = AsyncMock()
        handler.solar_production_history = mock_prod

        mock_fc = MagicMock()
        mock_fc.file_name = "forecast_8h.npy"
        mock_fc.save_values = AsyncMock()
        handler.solar_forecast_history = {"qs_solar_forecast_8h": mock_fc}

        mock_pfc = MagicMock()
        mock_pfc.file_name = "prov_fc_8h.npy"
        mock_pfc.save_values = AsyncMock()
        handler.solar_forecast_history_per_provider = {"Solcast": {"qs_solar_forecast_8h": mock_pfc}}

        await handler.dump_for_debug(str(tmp_path))

        mock_consumption.save_values.assert_called_once()
        mock_prod.save_values.assert_called_once()
        mock_fc.save_values.assert_called_once()
        mock_pfc.save_values.assert_called_once()


class TestUpdateConsumptionSaveAndCompute:
    """Cover the consumption save and compute forecast paths."""

    @pytest.mark.asyncio
    async def test_consumption_save_on_add_value_true(self, tmp_path):
        """When add_value returns True, save_values is called."""
        from types import SimpleNamespace

        home = SimpleNamespace(
            hass=None,
            solar_plant=None,
            home_non_controlled_consumption=100.0,
        )
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        handler._in_reset = False

        mock_consumption = MagicMock()
        mock_consumption.add_value.return_value = True
        mock_consumption.save_values = AsyncMock()
        mock_consumption.update_current_forecast_if_needed.return_value = True
        handler.home_non_controlled_consumption = mock_consumption
        handler.compute_non_controlled_forecast_intl = MagicMock()

        handler.solar_production_history = None
        handler.solar_forecast_history = None
        handler.solar_forecast_history_per_provider = None
        handler.init_forecasts = AsyncMock(return_value=True)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        await handler.update_consumption_and_forecast_history(t)

        mock_consumption.save_values.assert_called_once()
        handler.compute_non_controlled_forecast_intl.assert_called_once_with(t)


class TestSolarForecastSetWithHaEntities:
    """Cover solar_forecast_set_and_reset when ha_entities are populated."""

    @pytest.mark.asyncio
    async def test_creates_forecast_history_when_ha_entities_present(self, tmp_path):
        """When ha_entities contains forecast sensor names, histories are created."""
        from types import SimpleNamespace
        from unittest.mock import patch

        ha_entities = {}
        for name in QSForecastSolarSensors:
            ha_entities[name] = f"sensor.{name}"

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_inverter_active_power=None,
                solar_inverter_input_active_power=None,
                solar_forecast_providers={},
            ),
            ha_entities=ha_entities,
        )
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        handler._in_reset = False
        handler.solar_production_history = MagicMock()  # Already initialized
        handler.solar_forecast_history = None
        handler.solar_forecast_history_per_provider = None

        mock_hv = MagicMock(spec=QSSolarHistoryVals)
        mock_hv.init = AsyncMock()

        with patch("custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals", return_value=mock_hv):
            await handler.solar_forecast_set_and_reset(time=datetime.datetime.now(pytz.UTC), for_reset=False)

        assert handler.solar_forecast_history is not None
        assert len(handler.solar_forecast_history) == len(QSForecastSolarSensors)

    @pytest.mark.asyncio
    async def test_creates_per_provider_forecast_history(self, tmp_path):
        """When ha_entities and providers exist, per-provider histories are created."""
        from types import SimpleNamespace
        from unittest.mock import patch

        ha_entities = {}
        for name in QSForecastSolarSensors:
            ha_entities[f"Solcast_{name}"] = f"sensor.solcast_{name}"

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_inverter_active_power=None,
                solar_inverter_input_active_power=None,
                solar_forecast_providers={"Solcast": MagicMock()},
            ),
            ha_entities=ha_entities,
        )
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))
        handler._in_reset = False
        handler.solar_production_history = MagicMock()
        handler.solar_forecast_history = MagicMock()
        handler.solar_forecast_history_per_provider = None

        mock_hv = MagicMock(spec=QSSolarHistoryVals)
        mock_hv.init = AsyncMock()

        with patch("custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals", return_value=mock_hv):
            await handler.solar_forecast_set_and_reset(time=datetime.datetime.now(pytz.UTC), for_reset=False)

        assert handler.solar_forecast_history_per_provider is not None
        assert "Solcast" in handler.solar_forecast_history_per_provider
        assert len(handler.solar_forecast_history_per_provider["Solcast"]) == len(QSForecastSolarSensors)

    @pytest.mark.asyncio
    async def test_reset_with_ha_entities_clears_after_save(self, tmp_path):
        """for_reset=True saves and then clears all histories."""
        from types import SimpleNamespace
        from unittest.mock import patch

        ha_entities = {}
        for name in QSForecastSolarSensors:
            ha_entities[name] = f"sensor.{name}"
            ha_entities[f"Solcast_{name}"] = f"sensor.solcast_{name}"

        home = SimpleNamespace(
            hass=None,
            solar_plant=SimpleNamespace(
                solar_inverter_active_power=None,
                solar_inverter_input_active_power="sensor.solar_input",
                solar_forecast_providers={"Solcast": MagicMock()},
            ),
            ha_entities=ha_entities,
        )
        handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=home, storage_path=str(tmp_path))

        mock_hv = MagicMock(spec=QSSolarHistoryVals)
        mock_hv.init = AsyncMock()
        mock_hv.save_values = AsyncMock()

        with patch("custom_components.quiet_solar.ha_model.home.QSSolarHistoryVals", return_value=mock_hv):
            await handler.solar_forecast_set_and_reset(time=datetime.datetime.now(pytz.UTC), for_reset=True)

        assert handler.solar_production_history is None
        assert handler.solar_forecast_history is None
        assert handler.solar_forecast_history_per_provider is None
        # save_values should have been called for reset
        assert mock_hv.save_values.call_count > 0


# ============================================================================
# get_historical_data defensive guard (line 3497)
# ============================================================================


class TestGetHistoricalDataDefensiveGuard:
    """Test that get_historical_data returns [] when _get_values returns None."""

    def test_returns_empty_when_get_values_returns_none(self):
        """get_historical_data returns [] when _get_values returns (None, None).

        This is a defensive guard — normally unreachable because values is checked
        first, but protects against concurrent mutation.
        """
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        hv = _make_history_vals(t, hours=24, value_fn=lambda h: 100.0)
        # values is not None, but we patch _get_values to return None
        assert hv.values is not None
        with patch.object(hv, "_get_values", return_value=(None, None)):
            result = hv.get_historical_data(t, past_hours=24)
        assert result == []


# ============================================================================
# add_value duplicate timestamp replacement (lines 4186-4187)
# ============================================================================


class TestAddValueDuplicateTimestamp:
    """Test that add_value replaces existing value when same timestamp is used."""

    def test_duplicate_timestamp_replaces_value(self):
        """Calling add_value twice with the same timestamp replaces the first value."""
        forecast = MagicMock()
        forecast.storage_path = "/tmp"
        forecast.home = MagicMock()
        forecast.home.hass = None
        hv = QSSolarHistoryVals(entity_id="sensor.test", forecast=forecast)
        hv.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float64)

        t = datetime.datetime(2024, 6, 15, 12, 3, tzinfo=pytz.UTC)  # within a 15-min interval

        # First add
        hv.add_value(t, 100.0)
        assert len(hv._current_values) == 1
        assert hv._current_values[0] == (t, 100.0)

        # Second add with same timestamp — should replace, not append
        hv.add_value(t, 200.0)
        assert len(hv._current_values) == 1
        assert hv._current_values[0] == (t, 200.0)
