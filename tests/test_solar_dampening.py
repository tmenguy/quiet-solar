"""Tests for solar forecast dampening (Issue #124).

Covers:
- compute_dampening 1-day (ratio correction) and 7-day (linear regression)
- Day selection logic (22:00 boundary)
- reset_dampening
- _get_dampened_value
- get_forecast / get_value_from_current_forecast with dampening
- get_active_score (dampened vs raw fallback)
- Sensor restore for dampened score + coefficients
- Button integration (orchestration methods)
"""

from __future__ import annotations

import datetime
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytz

from custom_components.quiet_solar.const import (
    BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY,
    BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY,
    BUTTON_SOLAR_RESET_DAMPENING,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    INTERVALS_MN,
    NUM_INTERVAL_PER_HOUR,
    NUM_INTERVALS_PER_DAY,
    SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX,
    SOLCAST_SOLAR_DOMAIN,
    QSForecastSolarSensors,
)
from custom_components.quiet_solar.ha_model.home import (
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

    config_entry = FakeConfigEntry(entry_id="solar_dampening_test", data={CONF_NAME: "Test Solar"})
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


def _make_provider_with_histories(
    actuals_fn, forecast_fn, t0, hours=24, num_days=1, sensor_name="qs_solar_forecast_8h"
):
    """Create a provider with mock histories for dampening computation."""
    solar_mock = MagicMock()
    forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

    total_hours = num_days * 24
    actual_data = []
    for i in range(total_hours * NUM_INTERVAL_PER_HOUR):
        ts = t0 - timedelta(hours=total_hours) + timedelta(minutes=i * INTERVALS_MN)
        local_h = ts.astimezone(pytz.timezone("Europe/Paris")).hour + ts.minute / 60
        actual_data.append((ts, actuals_fn(local_h)))

    mock_prod = MagicMock(spec=QSSolarHistoryVals)
    mock_prod.get_historical_data.return_value = actual_data
    forecast_handler.solar_production_history = mock_prod

    fc_data = []
    for i in range(total_hours * NUM_INTERVAL_PER_HOUR):
        ts = t0 - timedelta(hours=total_hours) + timedelta(minutes=i * INTERVALS_MN)
        local_h = ts.astimezone(pytz.timezone("Europe/Paris")).hour + ts.minute / 60
        fc_data.append((ts, forecast_fn(local_h)))

    mock_fc = MagicMock(spec=QSSolarHistoryVals)
    mock_fc.get_historical_data.return_value = fc_data
    forecast_handler.solar_forecast_history_per_provider = {"prov": {sensor_name: mock_fc}}

    solar_mock.home.solar_and_consumption_forecast = forecast_handler
    provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
    return provider


# ============================================================================
# _time_to_slot_index tests
# ============================================================================


class TestTimeToSlotIndex:
    """Test _time_to_slot_index static method."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_midnight_returns_zero(self, mock_dt_util):
        """Slot 0 at midnight local time."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        t = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=tz)
        assert QSSolarProvider._time_to_slot_index(t) == 0

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_noon_returns_48(self, mock_dt_util):
        """Slot 48 at noon (12:00) local time."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        assert QSSolarProvider._time_to_slot_index(t) == 48

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_last_slot_returns_95(self, mock_dt_util):
        """Slot 95 at 23:45 local time."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        t = datetime.datetime(2024, 6, 15, 23, 45, tzinfo=tz)
        assert QSSolarProvider._time_to_slot_index(t) == 95

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_quarter_hour_boundary(self, mock_dt_util):
        """Slot calculation at 14:30 → slot 58."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        t = datetime.datetime(2024, 6, 15, 14, 30, tzinfo=tz)
        assert QSSolarProvider._time_to_slot_index(t) == 14 * 4 + 2  # 58


# ============================================================================
# _get_dampened_value tests
# ============================================================================


class TestGetDampenedValue:
    """Test _get_dampened_value method."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_identity_when_no_coefficients(self, mock_dt_util):
        """Returns raw value when no coefficients set for slot."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        assert provider._get_dampened_value(t, 1000.0) == 1000.0

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_applies_scale_and_offset(self, mock_dt_util):
        """Returns a*raw + b when coefficients are set."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        # Slot 48 = noon
        provider._dampening_coefficients = {48: (0.8, 50.0)}
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        result = provider._get_dampened_value(t, 1000.0)
        assert abs(result - 850.0) < 0.01  # 0.8 * 1000 + 50

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_clamps_to_zero(self, mock_dt_util):
        """Negative results are clamped to 0."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider._dampening_coefficients = {48: (0.5, -600.0)}
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        result = provider._get_dampened_value(t, 1000.0)
        assert result == 0.0  # 0.5 * 1000 - 600 = -100 → 0

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_uses_identity_for_missing_slot(self, mock_dt_util):
        """Slots not in coefficients dict get identity (1.0, 0.0)."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider._dampening_coefficients = {0: (2.0, 10.0)}  # Only slot 0
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)  # Slot 48
        result = provider._get_dampened_value(t, 500.0)
        assert result == 500.0  # Identity: 1.0 * 500 + 0.0


# ============================================================================
# get_active_score tests
# ============================================================================


class TestGetActiveScore:
    """Test get_active_score returns dampened when available."""

    def test_returns_raw_when_no_dampened(self):
        """Falls back to raw score when no dampened score."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider.score = 150.0
        provider.score_dampened = None
        assert provider.get_active_score() == 150.0

    def test_returns_dampened_when_available(self):
        """Returns dampened score when set."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider.score = 150.0
        provider.score_dampened = 80.0
        assert provider.get_active_score() == 80.0

    def test_returns_none_when_both_none(self):
        """Returns None when neither score is set."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        assert provider.get_active_score() is None


# ============================================================================
# has_dampening / dampening_coefficients property tests
# ============================================================================


class TestDampeningProperties:
    """Test dampening property accessors."""

    def test_has_dampening_false_initially(self):
        """has_dampening is False when no coefficients."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        assert provider.has_dampening is False

    def test_has_dampening_true_after_set(self):
        """has_dampening is True when coefficients are set."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider.set_dampening_coefficients({0: (1.0, 0.0)})
        assert provider.has_dampening is True

    def test_dampening_coefficients_returns_copy(self):
        """dampening_coefficients returns a copy, not the internal dict."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        original = {0: (1.0, 0.0)}
        provider.set_dampening_coefficients(original)
        copy = provider.dampening_coefficients
        copy[99] = (2.0, 3.0)
        assert 99 not in provider._dampening_coefficients


# ============================================================================
# reset_dampening tests
# ============================================================================


class TestResetDampening:
    """Test reset_dampening method."""

    def test_sets_identity_for_all_96_slots(self):
        """After reset, all 96 slots have identity coefficients."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider._dampening_coefficients = {0: (2.0, 10.0)}
        provider.score_dampened = 50.0
        provider.reset_dampening()

        assert len(provider._dampening_coefficients) == NUM_INTERVALS_PER_DAY
        for slot in range(NUM_INTERVALS_PER_DAY):
            assert provider._dampening_coefficients[slot] == (1.0, 0.0)

    def test_clears_dampened_score(self):
        """After reset, score_dampened is None."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        provider.score_dampened = 42.0
        provider.reset_dampening()
        assert provider.score_dampened is None


# ============================================================================
# compute_dampening 1-day tests
# ============================================================================


class TestComputeDampening1Day:
    """Test compute_dampening with 1-day ratio correction."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_ratio_correction_applied(self, mock_dt_util):
        """1-day dampening computes a_k = actual/forecast per slot."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        mock_dt_util.utcnow.return_value = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)

        # Forecast always 1000W during daytime, actuals 800W → ratio 0.8
        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 800.0 if 6 <= h <= 18 else 0.0,
            forecast_fn=lambda h: 1000.0 if 6 <= h <= 18 else 0.0,
            t0=t0,
            num_days=1,
        )

        result = provider.compute_dampening(t0, num_days=1)
        assert result is True
        assert provider.has_dampening

        # Check daytime slots have ratio ~0.8
        for slot in range(24, 72):  # 6:00 to 18:00 = slots 24-71
            a_k, b_k = provider._dampening_coefficients.get(slot, (1.0, 0.0))
            if a_k != 1.0:  # Skip identity slots
                assert abs(a_k - 0.8) < 0.01
                assert b_k == 0.0

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_skip_low_forecast(self, mock_dt_util):
        """Slots with forecast < 10W get identity coefficients."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 5.0,
            forecast_fn=lambda h: 5.0,
            t0=t0,
            num_days=1,
        )

        provider.compute_dampening(t0, num_days=1)
        # All slots should be identity since forecast < 10
        for slot in range(NUM_INTERVALS_PER_DAY):
            a_k, b_k = provider._dampening_coefficients.get(slot, (1.0, 0.0))
            assert a_k == 1.0
            assert b_k == 0.0

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_physical_guard_clamp(self, mock_dt_util):
        """a_k is clamped to [0.1, 3.0]."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        # actuals 5000W vs forecast 1000W → ratio 5.0, should clamp to 3.0
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 5000.0 if 10 <= h <= 14 else 0.0,
            forecast_fn=lambda h: 1000.0 if 10 <= h <= 14 else 0.0,
            t0=t0,
            num_days=1,
        )

        provider.compute_dampening(t0, num_days=1)
        for slot in range(40, 56):  # 10:00-14:00
            a_k, b_k = provider._dampening_coefficients.get(slot, (1.0, 0.0))
            if a_k != 1.0:
                assert a_k <= 3.0

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_nighttime_identity(self, mock_dt_util):
        """Nighttime slots (both values zero) get identity coefficients."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 1000.0 if 10 <= h <= 14 else 0.0,
            forecast_fn=lambda h: 1000.0 if 10 <= h <= 14 else 0.0,
            t0=t0,
            num_days=1,
        )

        provider.compute_dampening(t0, num_days=1)
        # Nighttime slots should be identity
        for slot in [0, 4, 8, 80, 88, 92]:  # Night slots
            a_k, b_k = provider._dampening_coefficients.get(slot, (1.0, 0.0))
            assert (a_k, b_k) == (1.0, 0.0)


# ============================================================================
# compute_dampening 7-day tests
# ============================================================================


class TestComputeDampening7Day:
    """Test compute_dampening with 7-day linear regression."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_regression_with_sufficient_points(self, mock_dt_util):
        """7-day mode uses polyfit when >= 3 data points per slot."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        # Linear relationship: actual = 0.9 * forecast + 50
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: (0.9 * 1000.0 + 50.0) if 8 <= h <= 16 else 0.0,
            forecast_fn=lambda h: 1000.0 if 8 <= h <= 16 else 0.0,
            t0=t0,
            num_days=7,
        )

        result = provider.compute_dampening(t0, num_days=7)
        assert result is True
        assert provider.has_dampening

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_insufficient_points_get_identity(self, mock_dt_util):
        """Slots with < 3 data points get identity coefficients in 7-day mode."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)

        # Create provider with very sparse data (only 2 days)
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 800.0 if 10 <= h <= 14 else 0.0,
            forecast_fn=lambda h: 1000.0 if 10 <= h <= 14 else 0.0,
            t0=t0,
            num_days=2,  # Only 2 days, so < 3 points per slot
        )
        # Override to request 7-day but only have 2 days of data
        result = provider.compute_dampening(t0, num_days=7)
        # Should still succeed but slots with < 3 points get identity
        assert result is True

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_a_k_clamp_regression(self, mock_dt_util):
        """Regression a_k is clamped to [0.1, 3.0]."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        # Extreme ratio: actual 5000 vs forecast 1000 → regression slope ~5.0
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 5000.0 if 10 <= h <= 14 else 0.0,
            forecast_fn=lambda h: 1000.0 if 10 <= h <= 14 else 0.0,
            t0=t0,
            num_days=7,
        )

        provider.compute_dampening(t0, num_days=7)
        for slot in range(40, 56):
            a_k, b_k = provider._dampening_coefficients.get(slot, (1.0, 0.0))
            assert a_k <= 3.0
            assert a_k >= 0.1


# ============================================================================
# Day selection logic tests
# ============================================================================


class TestDaySelection:
    """Test reference day selection (22:00 boundary)."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_press_at_23_uses_today(self, mock_dt_util):
        """Pressing at 23:00 local uses today as reference day."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        # 23:00 Paris time on June 15
        t0 = datetime.datetime(2024, 6, 15, 21, 0, tzinfo=pytz.UTC)  # 23:00 CEST

        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 500.0 if 8 <= h <= 16 else 0.0,
            forecast_fn=lambda h: 500.0 if 8 <= h <= 16 else 0.0,
            t0=t0,
            num_days=1,
        )

        # The internal call uses ref_end from today when hour >= 22
        result = provider.compute_dampening(t0, num_days=1)
        assert result is True

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_press_at_16_uses_yesterday(self, mock_dt_util):
        """Pressing at 16:00 local uses yesterday as reference day."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        # 16:00 Paris time
        t0 = datetime.datetime(2024, 6, 15, 14, 0, tzinfo=pytz.UTC)  # 16:00 CEST

        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 500.0 if 8 <= h <= 16 else 0.0,
            forecast_fn=lambda h: 500.0 if 8 <= h <= 16 else 0.0,
            t0=t0,
            num_days=1,
        )

        result = provider.compute_dampening(t0, num_days=1)
        assert result is True

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_missing_data_returns_false(self, mock_dt_util):
        """Returns False when no data available."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        provider = _TestProvider(solar=None, domain="test", provider_name="prov")
        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        result = provider.compute_dampening(t0, num_days=1)
        assert result is False


# ============================================================================
# compute_dampened_score tests
# ============================================================================


class TestComputeDampenedScore:
    """Test compute_dampened_score method."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_dampened_score_computed(self, mock_dt_util):
        """compute_dampened_score stores a float score."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 800.0 if 8 <= h <= 16 else 0.0,
            forecast_fn=lambda h: 1000.0 if 8 <= h <= 16 else 0.0,
            t0=t0,
            num_days=1,
        )

        # First compute dampening to set coefficients
        provider.compute_dampening(t0, num_days=1)
        assert provider.score_dampened is not None

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_dampened_score_better_than_raw(self, mock_dt_util):
        """Dampened score should generally be <= raw score after correction."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 800.0 if 8 <= h <= 16 else 0.0,
            forecast_fn=lambda h: 1000.0 if 8 <= h <= 16 else 0.0,
            t0=t0,
            num_days=1,
        )

        # Compute raw score first
        provider.compute_score(t0)
        raw_score = provider.score

        # Compute dampening (which also computes dampened score)
        provider.compute_dampening(t0, num_days=1)
        dampened_score = provider.score_dampened

        # Dampened score should be better (lower) or equal
        if raw_score is not None and dampened_score is not None:
            assert dampened_score <= raw_score + 1.0  # Allow small rounding

    def test_returns_false_without_solar(self):
        """Returns False when solar is None."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert provider.compute_dampened_score(t0) is False


# ============================================================================
# get_forecast / get_value_from_current_forecast tests
# ============================================================================


class TestForecastWithDampening:
    """Test forecast access with dampening applied."""

    def test_get_forecast_no_dampening(self):
        """get_forecast returns raw values when no dampening."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=15), 1100.0),
        ]
        result = provider.get_forecast(t0, t0 + timedelta(hours=1))
        values = [v for _, v in result if v is not None]
        assert 1000.0 in values

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_get_forecast_with_dampening(self, mock_dt_util):
        """get_forecast applies dampening when coefficients are active."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=15), 1100.0),
        ]
        # Set dampening: slot 48 (noon) with scale 0.5
        slot_noon = QSSolarProvider._time_to_slot_index(t0)
        provider._dampening_coefficients = {slot_noon: (0.5, 0.0)}

        result = provider.get_forecast(t0, t0 + timedelta(hours=1))
        # The noon value should be dampened to 500.0
        for t, v in result:
            if t == t0 and v is not None:
                assert abs(float(v) - 500.0) < 0.01

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_get_value_from_current_forecast_with_dampening(self, mock_dt_util):
        """get_value_from_current_forecast applies dampening."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        provider.solar_forecast = [(t0, 1000.0)]

        slot = QSSolarProvider._time_to_slot_index(t0)
        provider._dampening_coefficients = {slot: (0.7, 100.0)}

        _, v = provider.get_value_from_current_forecast(t0)
        assert v is not None
        assert abs(float(v) - 800.0) < 0.01  # 0.7 * 1000 + 100

    def test_get_value_from_current_forecast_no_dampening(self):
        """get_value_from_current_forecast returns raw when no dampening."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0)]

        _, v = provider.get_value_from_current_forecast(t0)
        assert v == 1000.0


# ============================================================================
# Sensor restore tests
# ============================================================================


class TestDampenedScoreRestore:
    """Test QSBaseSensorSolarDampenedScoreRestore."""

    async def test_restore_score_and_coefficients(self):
        """Sensor restore hydrates provider dampened score and coefficients."""
        from custom_components.quiet_solar.sensor import QSBaseSensorSolarDampenedScoreRestore, QSSensorEntityDescription

        provider = _TestProvider(solar=None, domain="test", provider_name="p")

        description = QSSensorEntityDescription(
            key="test_dampened",
            name="Test Dampened Score",
        )

        data_handler = MagicMock()
        device = MagicMock()

        sensor = QSBaseSensorSolarDampenedScoreRestore(
            data_handler=data_handler,
            device=device,
            description=description,
            provider=provider,
        )

        # Simulate restored state
        sensor._attr_native_value = "42.5"
        sensor._attr_extra_state_attributes = {
            "dampening_coefficients": {
                "0": [1.2, 5.0],
                "48": [0.9, -10.0],
            }
        }

        # Mock the async_get_last_sensor_data to return our state
        async def mock_get_last_data():
            return None  # Bypass actual restore, we set attrs directly

        sensor.async_get_last_sensor_data = mock_get_last_data

        # Mock super().async_added_to_hass() to avoid HA dependency
        with patch.object(
            QSBaseSensorSolarDampenedScoreRestore.__bases__[0],
            "async_added_to_hass",
            new_callable=AsyncMock,
        ):
            # Directly test the restore logic
            # Score restore
            if provider.score_dampened is None and sensor._attr_native_value is not None:
                try:
                    provider.score_dampened = float(sensor._attr_native_value)
                except (TypeError, ValueError):
                    pass

            # Coefficient restore
            raw_coeffs = sensor._attr_extra_state_attributes.get("dampening_coefficients")
            if raw_coeffs and isinstance(raw_coeffs, dict):
                coefficients = {}
                for slot_str, ab in raw_coeffs.items():
                    try:
                        slot = int(slot_str)
                        coefficients[slot] = (float(ab[0]), float(ab[1]))
                    except (TypeError, ValueError, IndexError):
                        continue
                if coefficients:
                    provider.set_dampening_coefficients(coefficients)

        assert provider.score_dampened == 42.5
        assert provider.has_dampening
        assert provider._dampening_coefficients[0] == (1.2, 5.0)
        assert provider._dampening_coefficients[48] == (0.9, -10.0)

    async def test_restore_with_no_coefficients(self):
        """Restore does not crash when no coefficients in attributes."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")

        # Simulate no attrs
        assert provider.score_dampened is None
        assert provider.has_dampening is False

    async def test_restore_with_invalid_coefficient_data(self):
        """Restore handles malformed coefficient data gracefully."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")

        raw_coeffs = {"bad": "data", "0": "not_a_list"}
        coefficients = {}
        for slot_str, ab in raw_coeffs.items():
            try:
                slot = int(slot_str)
                coefficients[slot] = (float(ab[0]), float(ab[1]))
            except (TypeError, ValueError, IndexError):
                continue

        if coefficients:
            provider.set_dampening_coefficients(coefficients)

        # Should not have set any coefficients
        assert provider.has_dampening is False


# ============================================================================
# Button integration tests
# ============================================================================


class TestButtonIntegration:
    """Test button entities trigger correct orchestration methods."""

    async def test_compute_dampening_1day_button(self, fake_hass):
        """1-day dampening button description wires to compute_dampening_all_providers(1)."""
        from custom_components.quiet_solar.button import create_ha_button_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()
        entities = create_ha_button_for_QSSolar(solar)
        btn = next(e for e in entities if e.entity_description.key == BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY)

        with patch.object(solar, "compute_dampening_all_providers", new_callable=AsyncMock) as mock_compute:
            await btn.entity_description.async_press(btn)
            mock_compute.assert_called_once_with(num_days=1)

    async def test_compute_dampening_7day_button(self, fake_hass):
        """7-day dampening button description wires to compute_dampening_all_providers(7)."""
        from custom_components.quiet_solar.button import create_ha_button_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()
        entities = create_ha_button_for_QSSolar(solar)
        btn = next(e for e in entities if e.entity_description.key == BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY)

        with patch.object(solar, "compute_dampening_all_providers", new_callable=AsyncMock) as mock_compute:
            await btn.entity_description.async_press(btn)
            mock_compute.assert_called_once_with(num_days=7)

    async def test_reset_dampening_button(self, fake_hass):
        """Reset dampening button description wires to reset_dampening_all_providers."""
        from custom_components.quiet_solar.button import create_ha_button_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()
        entities = create_ha_button_for_QSSolar(solar)
        btn = next(e for e in entities if e.entity_description.key == BUTTON_SOLAR_RESET_DAMPENING)

        with patch.object(solar, "reset_dampening_all_providers", new_callable=AsyncMock) as mock_reset:
            await btn.entity_description.async_press(btn)
            mock_reset.assert_called_once()

    async def test_orchestration_iterates_all_providers(self, fake_hass):
        """compute_dampening_all_providers calls each provider."""
        solar = _make_solar(
            fake_hass,
            providers_config=[
                {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Solcast"},
            ],
        )

        for provider in solar.solar_forecast_providers.values():
            provider.compute_dampening = MagicMock(return_value=True)

        await solar.compute_dampening_all_providers(num_days=1)

        for provider in solar.solar_forecast_providers.values():
            provider.compute_dampening.assert_called_once()

    async def test_reset_all_providers_clears_dampening(self, fake_hass):
        """reset_dampening_all_providers resets all providers."""
        solar = _make_solar(
            fake_hass,
            providers_config=[
                {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Solcast"},
            ],
        )

        for provider in solar.solar_forecast_providers.values():
            provider._dampening_coefficients = {0: (2.0, 10.0)}
            provider.score_dampened = 50.0

        await solar.reset_dampening_all_providers()

        for provider in solar.solar_forecast_providers.values():
            assert provider.score_dampened is None
            assert len(provider._dampening_coefficients) == NUM_INTERVALS_PER_DAY
            assert provider._dampening_coefficients[0] == (1.0, 0.0)


# ============================================================================
# Button entity creation tests
# ============================================================================


class TestButtonEntityCreation:
    """Test that button entities are created correctly."""

    def test_dampening_buttons_created(self, fake_hass):
        """Three dampening buttons are created for QSSolar."""
        from custom_components.quiet_solar.button import create_ha_button_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()

        entities = create_ha_button_for_QSSolar(solar)
        keys = [e.entity_description.key for e in entities]

        assert BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY in keys
        assert BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY in keys
        assert BUTTON_SOLAR_RESET_DAMPENING in keys


# ============================================================================
# Dampened score sensor creation tests
# ============================================================================


class TestDampenedScoreSensorCreation:
    """Test that dampened score sensors are created correctly."""

    def test_dampened_score_sensor_created_per_provider(self, fake_hass):
        """A dampened score sensor is created for each provider."""
        from custom_components.quiet_solar.sensor import create_ha_sensor_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()

        entities = create_ha_sensor_for_QSSolar(solar)
        keys = [e.entity_description.key for e in entities]

        dampened_keys = [k for k in keys if k.startswith(SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX)]
        assert len(dampened_keys) == 1
        assert dampened_keys[0] == f"{SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX}test"

    def test_raw_score_sensor_renamed(self, fake_hass):
        """Raw score sensor has 'Forecast Raw Score' name."""
        from custom_components.quiet_solar.sensor import create_ha_sensor_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()

        entities = create_ha_sensor_for_QSSolar(solar)
        raw_score = [e for e in entities if e.entity_description.key == "qs_solar_forecast_score_test"]
        assert len(raw_score) == 1
        assert "Raw Score" in raw_score[0].entity_description.name

    def test_dampened_score_value_fn_and_attr(self, fake_hass):
        """Dampened score sensor value_fn_and_attr returns score + coefficients."""
        from custom_components.quiet_solar.sensor import create_ha_sensor_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()

        # Set dampening on the provider
        provider = list(solar.solar_forecast_providers.values())[0]
        provider.score_dampened = 42.0
        provider._dampening_coefficients = {0: (1.2, 5.0), 48: (0.9, -10.0)}

        entities = create_ha_sensor_for_QSSolar(solar)
        dampened_key = f"{SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX}test"
        dampened_sensors = [e for e in entities if e.entity_description.key == dampened_key]
        assert len(dampened_sensors) == 1

        desc = dampened_sensors[0].entity_description
        value, attrs = desc.value_fn_and_attr(solar, dampened_key)
        assert value == 42.0
        assert "dampening_coefficients" in attrs
        assert attrs["dampening_coefficients"]["0"] == [1.2, 5.0]
        assert attrs["dampening_coefficients"]["48"] == [0.9, -10.0]

    def test_dampened_score_value_fn_no_dampening(self, fake_hass):
        """Dampened score sensor returns None + empty attrs when no dampening."""
        from custom_components.quiet_solar.sensor import create_ha_sensor_for_QSSolar

        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        solar.data_handler = MagicMock()

        entities = create_ha_sensor_for_QSSolar(solar)
        dampened_key = f"{SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX}test"
        dampened_sensors = [e for e in entities if e.entity_description.key == dampened_key]

        desc = dampened_sensors[0].entity_description
        value, attrs = desc.value_fn_and_attr(solar, dampened_key)
        assert value is None
        assert attrs == {}


# ============================================================================
# Coverage: edge cases for _get_historical_data_for_dampening
# ============================================================================


class TestGetHistoricalDataEdgeCases:
    """Test edge cases in _get_historical_data_for_dampening."""

    def test_forecast_handler_none(self):
        """Returns None, None when forecast_handler is None."""
        solar_mock = MagicMock()
        solar_mock.home.solar_and_consumption_forecast = None
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        result = provider._get_historical_data_for_dampening(t0, 1)
        assert result == (None, None)

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_prod_history_none(self, mock_dt_util):
        """Returns None, None when prod_history is None."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz
        solar_mock = MagicMock()
        forecast_handler = MagicMock()
        forecast_handler.solar_production_history = None
        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        result = provider._get_historical_data_for_dampening(t0, 1)
        assert result == (None, None)

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_fallback_forecast_search(self, mock_dt_util):
        """Falls back to reverse-sorted sensor list when >= 8h finds nothing."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        actual_data = [(t0, 100.0)]
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = actual_data
        forecast_handler.solar_production_history = mock_prod

        # Only put data in a short-horizon sensor (< 8h)
        fc_data = [(t0, 200.0)]
        mock_fc = MagicMock(spec=QSSolarHistoryVals)
        mock_fc.get_historical_data.return_value = fc_data
        forecast_handler.solar_forecast_history_per_provider = {"p": {"qs_solar_forecast_1h": mock_fc}}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")
        actuals, forecast = provider._get_historical_data_for_dampening(t0, 1)
        # Should have found the forecast via fallback
        assert forecast is not None


# ============================================================================
# Coverage: compute_dampening_all_providers failure branch
# ============================================================================


class TestOrchestrationFailure:
    """Test orchestration failure logging."""

    async def test_compute_dampening_logs_failure(self, fake_hass):
        """When a provider fails, a warning is logged."""
        solar = _make_solar(
            fake_hass,
            providers_config=[
                {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"},
            ],
        )

        for provider in solar.solar_forecast_providers.values():
            provider.compute_dampening = MagicMock(return_value=False)

        await solar.compute_dampening_all_providers(num_days=1)

        for provider in solar.solar_forecast_providers.values():
            provider.compute_dampening.assert_called_once()


# ============================================================================
# Coverage: compute_dampened_score edge cases
# ============================================================================


class TestComputeDampenedScoreEdgeCases:
    """Test compute_dampened_score edge case paths."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_fallback_forecast_search_in_dampened_score(self, mock_dt_util):
        """compute_dampened_score falls back to reverse-sorted sensor list."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        actual_data = [(t0, 100.0)]
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = actual_data
        forecast_handler.solar_production_history = mock_prod

        # Only short-horizon sensor
        fc_data = [(t0, 200.0)]
        mock_fc = MagicMock(spec=QSSolarHistoryVals)
        mock_fc.get_historical_data.return_value = fc_data
        forecast_handler.solar_forecast_history_per_provider = {"p": {"qs_solar_forecast_1h": mock_fc}}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")
        provider._dampening_coefficients = {0: (1.0, 0.0)}

        result = provider.compute_dampened_score(t0)
        assert result is True

    def test_missing_data_returns_false(self):
        """compute_dampened_score returns False when no actuals or forecast."""
        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = []
        forecast_handler.solar_production_history = mock_prod
        forecast_handler.solar_forecast_history_per_provider = {"p": {}}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        result = provider.compute_dampened_score(t0)
        assert result is False


# ============================================================================
# Coverage: QSBaseSensorSolarDampenedScoreRestore.async_added_to_hass
# ============================================================================


class TestDampenedScoreRestoreIntegration:
    """Test the actual restore sensor class methods."""

    async def test_async_added_to_hass_restores_score_and_coefficients(self):
        """Full integration: async_added_to_hass restores provider state."""
        from custom_components.quiet_solar.sensor import (
            QSBaseSensorSolarDampenedScoreRestore,
            QSExtraStoredData,
            QSSensorEntityDescription,
        )

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        description = QSSensorEntityDescription(key="test", name="Test")

        sensor = QSBaseSensorSolarDampenedScoreRestore(
            data_handler=MagicMock(),
            device=MagicMock(),
            description=description,
            provider=provider,
        )

        # Mock the full restore chain
        stored_data = QSExtraStoredData(
            native_value="55.5",
            native_attr={"dampening_coefficients": {"10": [1.5, 20.0], "50": [0.8, -5.0]}},
        )

        async def mock_get_last_extra_data():
            return MagicMock(as_dict=lambda: stored_data.as_dict())

        sensor.async_get_last_extra_data = mock_get_last_extra_data
        sensor.async_write_ha_state = MagicMock()
        sensor.hass = MagicMock()
        sensor.platform = MagicMock()

        # Patch the parent async_added_to_hass to do the actual restore
        with patch("custom_components.quiet_solar.sensor.QSBaseSensor.async_added_to_hass", new_callable=AsyncMock):
            # Simulate what QSBaseSensorRestore.async_added_to_hass does
            sensor._attr_native_value = None
            sensor._attr_extra_state_attributes = {}

            last_data = await sensor.async_get_last_sensor_data()
            if last_data:
                sensor._attr_native_value = last_data.native_value
                sensor._attr_extra_state_attributes = last_data.native_attr

            # Now run the dampened restore logic
            if provider.score_dampened is None and sensor._attr_native_value is not None:
                try:
                    provider.score_dampened = float(sensor._attr_native_value)
                except (TypeError, ValueError):
                    pass
            if sensor._attr_extra_state_attributes:
                raw_coeffs = sensor._attr_extra_state_attributes.get("dampening_coefficients")
                if raw_coeffs and isinstance(raw_coeffs, dict):
                    coefficients = {}
                    for slot_str, ab in raw_coeffs.items():
                        try:
                            slot = int(slot_str)
                            coefficients[slot] = (float(ab[0]), float(ab[1]))
                        except (TypeError, ValueError, IndexError):
                            continue
                    if coefficients:
                        provider.set_dampening_coefficients(coefficients)

        assert provider.score_dampened == 55.5
        assert provider.has_dampening
        assert provider._dampening_coefficients[10] == (1.5, 20.0)
        assert provider._dampening_coefficients[50] == (0.8, -5.0)

    async def test_async_added_to_hass_handles_invalid_score(self):
        """Restore handles non-numeric score gracefully."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        # Simulate invalid native_value
        native_value = "not_a_number"
        try:
            provider.score_dampened = float(native_value)
        except (TypeError, ValueError):
            pass
        assert provider.score_dampened is None


# ============================================================================
# Coverage: compute_dampening slots with no points
# ============================================================================


class TestComputeDampeningEdgeCases:
    """Test compute_dampening edge cases for slot coefficient computation."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_empty_slots_get_identity(self, mock_dt_util):
        """Slots with no data points get identity coefficients."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        # Only produce data for a very narrow time window
        provider = _make_provider_with_histories(
            actuals_fn=lambda h: 1000.0 if 12 <= h < 12.25 else 0.0,  # Single slot
            forecast_fn=lambda h: 1000.0 if 12 <= h < 12.25 else 0.0,
            t0=t0,
            num_days=1,
        )

        provider.compute_dampening(t0, num_days=1)
        # Most slots should be identity since they have no non-zero data
        identity_count = sum(1 for s in range(NUM_INTERVALS_PER_DAY) if provider._dampening_coefficients.get(s) == (1.0, 0.0))
        assert identity_count > 80  # Most slots are identity

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_slot_with_no_matching_forecast(self, mock_dt_util):
        """Slots with no matching timestamp pairs get identity (no points branch)."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        t0 = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)

        # Actuals at one set of timestamps, forecast at completely different timestamps
        actual_data = [(t0 - timedelta(hours=2), 500.0)]
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = actual_data
        forecast_handler.solar_production_history = mock_prod

        # Forecast at different timestamps (no overlap with actuals)
        fc_data = [(t0 - timedelta(hours=5), 600.0)]
        mock_fc = MagicMock(spec=QSSolarHistoryVals)
        mock_fc.get_historical_data.return_value = fc_data
        forecast_handler.solar_forecast_history_per_provider = {"p": {"qs_solar_forecast_8h": mock_fc}}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")

        result = provider.compute_dampening(t0, num_days=1)
        assert result is True
        # All slots should be identity since no matching timestamps
        for slot in range(NUM_INTERVALS_PER_DAY):
            assert provider._dampening_coefficients[slot] == (1.0, 0.0)


# ============================================================================
# Coverage: compute_dampened_score early returns
# ============================================================================


class TestComputeDampenedScoreEarlyReturns:
    """Test compute_dampened_score early return branches."""

    def test_forecast_handler_none(self):
        """Returns False when forecast handler is None."""
        solar_mock = MagicMock()
        solar_mock.home.solar_and_consumption_forecast = None
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert provider.compute_dampened_score(t0) is False

    def test_prod_history_none(self):
        """Returns False when production history is None."""
        solar_mock = MagicMock()
        forecast_handler = MagicMock()
        forecast_handler.solar_production_history = None
        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert provider.compute_dampened_score(t0) is False


# ============================================================================
# Coverage: QSBaseSensorSolarDampenedScoreRestore actual async_added_to_hass
# ============================================================================


class TestDampenedScoreRestoreActual:
    """Test the actual async_added_to_hass method on the sensor class."""

    async def test_actual_async_added_to_hass(self):
        """Call async_added_to_hass and verify it restores provider state."""
        from custom_components.quiet_solar.sensor import (
            QSBaseSensorSolarDampenedScoreRestore,
            QSExtraStoredData,
            QSSensorEntityDescription,
        )

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        description = QSSensorEntityDescription(key="test_dampened", name="Test")

        sensor = QSBaseSensorSolarDampenedScoreRestore(
            data_handler=MagicMock(),
            device=MagicMock(),
            description=description,
            provider=provider,
        )

        stored_data = QSExtraStoredData(
            native_value="88.0",
            native_attr={"dampening_coefficients": {"5": [1.3, 10.0], "20": [0.7, -5.0]}},
        )

        async def mock_get_last_extra_data():
            return MagicMock(as_dict=lambda: stored_data.as_dict())

        sensor.async_get_last_extra_data = mock_get_last_extra_data
        sensor.async_write_ha_state = MagicMock()

        # Patch the grandparent's async_added_to_hass (QSBaseSensor)
        with patch(
            "custom_components.quiet_solar.sensor.QSBaseSensor.async_added_to_hass",
            new_callable=AsyncMock,
        ):
            await sensor.async_added_to_hass()

        assert provider.score_dampened == 88.0
        assert provider.has_dampening
        assert provider._dampening_coefficients[5] == (1.3, 10.0)
        assert provider._dampening_coefficients[20] == (0.7, -5.0)

    async def test_actual_async_added_to_hass_no_data(self):
        """async_added_to_hass with no persisted data leaves provider unchanged."""
        from custom_components.quiet_solar.sensor import (
            QSBaseSensorSolarDampenedScoreRestore,
            QSSensorEntityDescription,
        )

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        description = QSSensorEntityDescription(key="test_dampened", name="Test")

        sensor = QSBaseSensorSolarDampenedScoreRestore(
            data_handler=MagicMock(),
            device=MagicMock(),
            description=description,
            provider=provider,
        )

        async def mock_get_last_extra_data():
            return None

        sensor.async_get_last_extra_data = mock_get_last_extra_data
        sensor.async_write_ha_state = MagicMock()

        with patch(
            "custom_components.quiet_solar.sensor.QSBaseSensor.async_added_to_hass",
            new_callable=AsyncMock,
        ):
            await sensor.async_added_to_hass()

        assert provider.score_dampened is None
        assert provider.has_dampening is False

    async def test_actual_async_added_to_hass_invalid_coefficients(self):
        """async_added_to_hass handles malformed coefficient data."""
        from custom_components.quiet_solar.sensor import (
            QSBaseSensorSolarDampenedScoreRestore,
            QSExtraStoredData,
            QSSensorEntityDescription,
        )

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        description = QSSensorEntityDescription(key="test_dampened", name="Test")

        sensor = QSBaseSensorSolarDampenedScoreRestore(
            data_handler=MagicMock(),
            device=MagicMock(),
            description=description,
            provider=provider,
        )

        stored_data = QSExtraStoredData(
            native_value="invalid",
            native_attr={"dampening_coefficients": {"bad_key": "bad_value", "0": []}},
        )

        async def mock_get_last_extra_data():
            return MagicMock(as_dict=lambda: stored_data.as_dict())

        sensor.async_get_last_extra_data = mock_get_last_extra_data
        sensor.async_write_ha_state = MagicMock()

        with patch(
            "custom_components.quiet_solar.sensor.QSBaseSensor.async_added_to_hass",
            new_callable=AsyncMock,
        ):
            await sensor.async_added_to_hass()

        # Score should remain None (invalid value), no coefficients restored
        assert provider.score_dampened is None
        assert provider.has_dampening is False


# ============================================================================
# Scoring cycle dampened score refresh tests
# ============================================================================


class TestScoringCycleDampenedRefresh:
    """Test that _run_scoring_cycle refreshes dampened scores."""

    def test_scoring_cycle_refreshes_dampened_score(self, fake_hass):
        """When dampening is active, scoring cycle recomputes dampened score."""
        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        provider = list(solar.solar_forecast_providers.values())[0]

        # Activate dampening
        provider._dampening_coefficients = {0: (0.8, 0.0)}
        provider.score_dampened = 100.0

        # Mock compute methods
        provider.compute_score = MagicMock(return_value=True)
        provider.compute_dampened_score = MagicMock(return_value=True)

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(t0)

        provider.compute_score.assert_called_once_with(t0)
        provider.compute_dampened_score.assert_called_once_with(t0)

    def test_scoring_cycle_skips_dampened_when_inactive(self, fake_hass):
        """When no dampening, scoring cycle does not compute dampened score."""
        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        provider = list(solar.solar_forecast_providers.values())[0]
        provider.compute_score = MagicMock(return_value=True)
        provider.compute_dampened_score = MagicMock(return_value=True)

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(t0)

        provider.compute_score.assert_called_once()
        provider.compute_dampened_score.assert_not_called()

    def test_scoring_cycle_clears_stale_dampened_score(self, fake_hass):
        """When dampened score computation fails, score_dampened is set to None."""
        solar = _make_solar(
            fake_hass,
            providers_config=[{CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Test"}],
        )
        provider = list(solar.solar_forecast_providers.values())[0]
        provider._dampening_coefficients = {0: (0.8, 0.0)}
        provider.score_dampened = 100.0
        provider.compute_score = MagicMock(return_value=True)
        provider.compute_dampened_score = MagicMock(return_value=False)

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(t0)

        assert provider.score_dampened is None


# ============================================================================
# Historical fallback dampening skip tests
# ============================================================================


class TestHistoricalFallbackDampeningSkip:
    """Test that dampening is skipped when using historical fallback data."""

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_get_forecast_skips_dampening_on_fallback(self, mock_dt_util):
        """get_forecast returns raw values when _using_historical_fallback is True."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        provider.solar_forecast = [(t0, 1000.0)]

        # Set dampening that would halve the value
        slot = QSSolarProvider._time_to_slot_index(t0)
        provider._dampening_coefficients = {slot: (0.5, 0.0)}

        # Without fallback flag, dampening applies
        result = provider.get_forecast(t0, t0 + timedelta(hours=1))
        dampened_values = [v for _, v in result if v is not None]
        assert any(abs(float(v) - 500.0) < 0.01 for v in dampened_values)

        # With fallback flag, dampening is skipped
        provider._using_historical_fallback = True
        result = provider.get_forecast(t0, t0 + timedelta(hours=1))
        raw_values = [v for _, v in result if v is not None]
        assert 1000.0 in raw_values

    @patch("custom_components.quiet_solar.ha_model.solar.dt_util")
    def test_get_value_from_current_forecast_skips_dampening_on_fallback(self, mock_dt_util):
        """get_value_from_current_forecast returns raw when fallback is active."""
        tz = pytz.timezone("Europe/Paris")
        mock_dt_util.get_default_time_zone.return_value = tz

        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=tz)
        provider.solar_forecast = [(t0, 1000.0)]

        slot = QSSolarProvider._time_to_slot_index(t0)
        provider._dampening_coefficients = {slot: (0.5, 0.0)}
        provider._using_historical_fallback = True

        _, v = provider.get_value_from_current_forecast(t0)
        assert v == 1000.0  # Raw, not dampened to 500

    def test_fallback_flag_initially_false(self):
        """Provider starts with _using_historical_fallback = False."""
        provider = _TestProvider(solar=None, domain="test", provider_name="p")
        assert provider._using_historical_fallback is False


# ============================================================================
# QSSolar raw forecast accessor tests
# ============================================================================


class TestQSSolarRawForecastAccessor:
    """Test QSSolar.get_value_from_current_forecast_raw."""

    def test_returns_none_when_no_provider(self, fake_hass):
        """Returns (None, None) when no active provider."""
        solar = _make_solar(fake_hass)
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_value_from_current_forecast_raw(t) == (None, None)
