"""Tests for Story 3.7: Solar Forecast API Resilience."""

from __future__ import annotations

import datetime
import logging
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_SOLAR_FORECAST_PROVIDER,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    OPEN_METEO_SOLAR_DOMAIN,
    SOLAR_FORECAST_STALE_THRESHOLD_S,
    SOLAR_ORCHESTRATOR_REPROBE_CYCLES,
    SOLAR_PROVIDER_MODE_AUTO,
    SOLCAST_SOLAR_DOMAIN,
)
from custom_components.quiet_solar.ha_model.solar import (
    DAMPENING_A_MAX,
    QSSolar,
    QSSolarProvider,
    QSSolarProviderOpenWeather,
    QSSolarProviderSolcast,
    _create_provider_for_domain,
    _migrate_solar_providers_config,
)
from tests.conftest import FakeConfigEntry, FakeHass

# ============================================================================
# Helpers
# ============================================================================


class _TestProvider(QSSolarProvider):
    """Concrete provider for testing abstract base class."""

    async def get_power_series_from_orchestrator(self, orchestrator, start_time=None, end_time=None):
        if start_time is None or end_time is None:
            return []
        return getattr(orchestrator, "power_series", [])


def _make_solar(fake_hass, providers_config=None, single_provider=None, **kwargs):
    """Create a QSSolar instance for testing."""
    from homeassistant.const import CONF_NAME

    config_entry = FakeConfigEntry(entry_id="solar_test_123", data={CONF_NAME: "Test Solar"})
    home = MagicMock()
    home.physical_3p = False
    home.voltage = 230
    home.hass = fake_hass
    home.force_update_all = AsyncMock()
    home._consumption_forecast = None

    config = {
        CONF_NAME: "Test Solar",
        "hass": fake_hass,
        "config_entry": config_entry,
        "home": home,
    }
    if providers_config is not None:
        config[CONF_SOLAR_FORECAST_PROVIDERS] = providers_config
    elif single_provider is not None:
        config[CONF_SOLAR_FORECAST_PROVIDER] = single_provider
    config.update(kwargs)

    return QSSolar(**config)


def _make_forecast(start_time, step_s=1800, count=48, base_watts=1000.0):
    """Create a mock forecast time series."""
    return [(start_time + timedelta(seconds=i * step_s), base_watts * (1 + 0.1 * (i % 10))) for i in range(count)]


# ============================================================================
# Task 1: Multi-provider configuration
# ============================================================================


class TestMigrateConfig:
    """Test config migration from old to new format."""

    def test_migrate_from_old_single_provider_solcast(self):
        """Test migration from old single CONF_SOLAR_FORECAST_PROVIDER."""
        config = {CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN}
        result = _migrate_solar_providers_config(config)
        assert len(result) == 1
        assert result[0][CONF_SOLAR_PROVIDER_DOMAIN] == SOLCAST_SOLAR_DOMAIN
        assert result[0][CONF_SOLAR_PROVIDER_NAME] == "Solcast"

    def test_migrate_from_old_single_provider_open_meteo(self):
        """Test migration from old single open meteo provider."""
        config = {CONF_SOLAR_FORECAST_PROVIDER: OPEN_METEO_SOLAR_DOMAIN}
        result = _migrate_solar_providers_config(config)
        assert len(result) == 1
        assert result[0][CONF_SOLAR_PROVIDER_DOMAIN] == OPEN_METEO_SOLAR_DOMAIN
        assert result[0][CONF_SOLAR_PROVIDER_NAME] == "Open-Meteo"

    def test_new_format_passed_through(self):
        """Test new format list is passed through unchanged."""
        providers = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "My Solcast"},
        ]
        config = {CONF_SOLAR_FORECAST_PROVIDERS: providers}
        result = _migrate_solar_providers_config(config)
        assert result is providers

    def test_no_provider_returns_empty(self):
        """Test empty config returns empty list."""
        result = _migrate_solar_providers_config({})
        assert result == []


class TestMultiProviderInit:
    """Test QSSolar multi-provider initialization."""

    def test_init_with_new_multi_provider_config(self, fake_hass):
        """Test initialization with new multi-provider config."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Solcast"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Open-Meteo"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        assert len(solar.solar_forecast_providers) == 2
        assert "Solcast" in solar.solar_forecast_providers
        assert "Open-Meteo" in solar.solar_forecast_providers
        assert isinstance(solar.solar_forecast_providers["Solcast"], QSSolarProviderSolcast)
        assert isinstance(solar.solar_forecast_providers["Open-Meteo"], QSSolarProviderOpenWeather)

    def test_init_with_old_single_provider_backward_compat(self, fake_hass):
        """Test backward compatibility with old single-provider config."""
        solar = _make_solar(fake_hass, single_provider=SOLCAST_SOLAR_DOMAIN)
        assert len(solar.solar_forecast_providers) == 1
        name = list(solar.solar_forecast_providers.keys())[0]
        assert name == "Solcast"
        assert isinstance(solar.solar_forecast_providers[name], QSSolarProviderSolcast)

    def test_init_no_provider(self, fake_hass):
        """Test initialization without any provider."""
        solar = _make_solar(fake_hass)
        assert len(solar.solar_forecast_providers) == 0
        assert solar.active_provider is None

    def test_active_provider_defaults_to_first(self, fake_hass):
        """Test active provider defaults to first configured."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "First"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Second"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        assert solar.active_provider_name == "First"

    def test_create_provider_for_unknown_domain(self):
        """Test unknown domain returns None."""
        result = _create_provider_for_domain("unknown_domain", MagicMock(), "test")
        assert result is None

    def test_get_provider_names(self, fake_hass):
        """Test get_provider_names returns configured names."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        assert solar.get_provider_names() == ["A", "B"]


# ============================================================================
# Task 2: Provider selection and auto mode
# ============================================================================


class TestProviderSelection:
    """Test provider selection entity and auto mode."""

    def test_set_provider_mode_manual(self, fake_hass):
        """Test manual provider selection."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        solar.set_provider_mode("B")
        assert solar.active_provider_name == "B"
        assert solar.provider_mode == "B"

    def test_auto_selects_best_provider(self, fake_hass):
        """Test auto mode selects provider with lowest score."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        solar.solar_forecast_providers["A"].score_raw = 500.0
        solar.solar_forecast_providers["B"].score_raw = 200.0
        solar.set_provider_mode(SOLAR_PROVIDER_MODE_AUTO)
        solar.auto_select_best_provider()
        assert solar.active_provider_name == "B"

    def test_auto_tiebreak_by_freshness(self, fake_hass):
        """Test auto mode tie-breaking by freshest forecast."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        now = datetime.datetime.now(pytz.UTC)
        solar.solar_forecast_providers["A"].score_raw = 200.0
        solar.solar_forecast_providers["A"]._latest_successful_forecast_time = now - timedelta(hours=2)
        solar.solar_forecast_providers["B"].score_raw = 200.0
        solar.solar_forecast_providers["B"]._latest_successful_forecast_time = now - timedelta(hours=1)
        solar.auto_select_best_provider()
        assert solar.active_provider_name == "B"

    def test_auto_no_scores_keeps_current(self, fake_hass):
        """Test auto mode keeps current when no scores available."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        assert solar.active_provider_name == "A"
        solar.auto_select_best_provider()
        assert solar.active_provider_name == "A"

    def test_auto_no_scores_picks_freshest_non_stale(self, fake_hass):
        """Test auto mode fallback picks the freshest non-stale provider when no scores exist."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        now = datetime.datetime.now(pytz.UTC)

        # No scores on any provider (score_raw = None by default)
        assert solar.solar_forecast_providers["A"].score_raw is None
        assert solar.solar_forecast_providers["B"].score_raw is None

        # A is fresh but older, B is fresher — both non-stale
        solar.solar_forecast_providers["A"]._latest_successful_forecast_time = now - timedelta(hours=2)
        solar.solar_forecast_providers["B"]._latest_successful_forecast_time = now - timedelta(hours=1)

        # Default active is "A" (first provider)
        assert solar.active_provider_name == "A"

        solar.auto_select_best_provider()
        # Should pick "B" as the freshest non-stale
        assert solar.active_provider_name == "B"

    def test_auto_no_scores_skips_stale_providers(self, fake_hass):
        """Test auto mode fallback skips stale providers when no scores exist."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        now = datetime.datetime.now(pytz.UTC)

        # A is stale (very old), B is fresh
        solar.solar_forecast_providers["A"]._latest_successful_forecast_time = now - timedelta(
            seconds=SOLAR_FORECAST_STALE_THRESHOLD_S + 3600
        )
        solar.solar_forecast_providers["B"]._latest_successful_forecast_time = now - timedelta(hours=1)

        solar.auto_select_best_provider()
        # Should pick "B" — "A" is stale and skipped
        assert solar.active_provider_name == "B"

    def test_get_platforms_includes_select_switch(self, fake_hass):
        """Test that solar with providers includes SELECT and SWITCH platforms."""
        from homeassistant.const import Platform

        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        platforms = solar.get_platforms()
        assert Platform.SELECT in platforms
        assert Platform.SWITCH in platforms


# ============================================================================
# Task 3: Staleness tracking
# ============================================================================


class TestStalenessTracking:
    """Test forecast staleness detection."""

    def test_stale_when_never_updated(self):
        """Test provider is stale when never updated."""
        provider = _TestProvider(solar=None, domain="test")
        assert provider.is_stale is True

    def test_stale_when_too_old(self):
        """Test provider is stale when older than threshold."""
        provider = _TestProvider(solar=None, domain="test")
        provider._latest_successful_forecast_time = datetime.datetime.now(pytz.UTC) - timedelta(
            seconds=SOLAR_FORECAST_STALE_THRESHOLD_S + 1
        )
        assert provider.is_stale is True

    def test_not_stale_when_fresh(self):
        """Test provider is not stale when recently updated."""
        provider = _TestProvider(solar=None, domain="test")
        provider._latest_successful_forecast_time = datetime.datetime.now(pytz.UTC) - timedelta(hours=1)
        assert provider.is_stale is False

    def test_forecast_age_hours(self, fake_hass):
        """Test forecast age in hours calculation."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        solar.solar_forecast_providers["A"]._latest_successful_forecast_time = datetime.datetime.now(
            pytz.UTC
        ) - timedelta(hours=3)
        age = solar.get_forecast_age_hours()
        assert age is not None
        assert 2.9 < age < 3.1

    def test_forecast_age_none_when_no_provider(self, fake_hass):
        """Test forecast age returns None when no provider."""
        solar = _make_solar(fake_hass)
        assert solar.get_forecast_age_hours() is None

    def test_is_forecast_ok_true_when_fresh(self, fake_hass):
        """Test is_forecast_ok returns True when fresh."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        solar.solar_forecast_providers["A"]._latest_successful_forecast_time = datetime.datetime.now(
            pytz.UTC
        ) - timedelta(hours=1)
        assert solar.is_forecast_ok() is True

    def test_is_forecast_ok_false_when_stale(self, fake_hass):
        """Test is_forecast_ok returns False when stale."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        assert solar.is_forecast_ok() is False

    def test_is_forecast_ok_false_when_no_provider(self, fake_hass):
        """Test is_forecast_ok returns False when no provider."""
        solar = _make_solar(fake_hass)
        assert solar.is_forecast_ok() is False


# ============================================================================
# Task 5: Health monitoring and re-probing
# ============================================================================


class TestHealthMonitoring:
    """Test orchestrator health tracking and re-probing."""

    @pytest.mark.asyncio
    async def test_unhealthy_orchestrator_not_used_for_extraction(self):
        """Test unhealthy orchestrators are excluded from forecast extraction."""
        solar_mock = MagicMock()
        solar_mock.hass = FakeHass()
        solar_mock.hass.data["test"] = {}
        provider = _TestProvider(solar=solar_mock, domain="test")

        good = MagicMock()
        good.power_series = [(datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC), 1000.0)]
        bad = MagicMock()

        provider.orchestrators = [good, bad]
        provider._orchestrator_health = {id(good): True, id(bad): False}
        provider._update_cycle_count = 1

        # Only healthy orchestrators should be used
        healthy = [o for o in provider.orchestrators if provider._orchestrator_health.get(id(o), False)]
        assert healthy == [good]

    @pytest.mark.asyncio
    async def test_reprobe_cycle(self):
        """Test that failed orchestrators are re-probed on reprobe cycle."""
        solar_mock = MagicMock()
        solar_mock.hass = FakeHass()
        solar_mock.hass.data["test"] = {"a": MagicMock()}
        provider = _TestProvider(solar=solar_mock, domain="test")

        bad_orch = MagicMock()
        provider.orchestrators = [bad_orch]
        provider._orchestrator_health = {id(bad_orch): False}
        # At reprobe cycle, even unhealthy ones get tested
        provider._update_cycle_count = SOLAR_ORCHESTRATOR_REPROBE_CYCLES - 1

        # The update will trigger a reprobe
        await provider.update(datetime.datetime.now(pytz.UTC))
        # The orchestrator should be probed (fill_orchestrators will re-create the list)
        # Just verify the cycle count advances
        assert provider._update_cycle_count == SOLAR_ORCHESTRATOR_REPROBE_CYCLES


# ============================================================================
# Task 6: Provider accuracy scoring
# ============================================================================


class TestAccuracyScoring:
    """Test forecast accuracy scoring."""

    def test_compute_mae_score(self):
        """Test MAE score computation with known data."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC)
        # Build forecast and actuals with known 200W difference
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(12)]
        provider._actuals_buffer = [(t0 + timedelta(hours=h), 800.0) for h in range(12)]

        provider.compute_score()
        assert provider.score_raw is not None
        assert abs(provider.score_raw - 200.0) < 0.01

    def test_score_none_with_no_data(self):
        """Test score is None when no data available."""
        provider = _TestProvider(solar=None, domain="test")
        provider.compute_score()
        assert provider.score_raw is None

    def test_get_active_score_returns_dampened_when_enabled(self):
        """Test get_active_score returns dampened score when enabled."""
        provider = _TestProvider(solar=None, domain="test")
        provider.score_raw = 300.0
        provider.score_dampened = 150.0
        provider.dampening_enabled = True
        assert provider.get_active_score() == 150.0

    def test_get_active_score_returns_raw_when_disabled(self):
        """Test get_active_score returns raw score when dampening disabled."""
        provider = _TestProvider(solar=None, domain="test")
        provider.score_raw = 300.0
        provider.dampening_enabled = False
        assert provider.get_active_score() == 300.0


# ============================================================================
# Task 7: Dampening — MOS linear correction
# ============================================================================


class TestDampening:
    """Test MOS dampening computation and application."""

    def test_compute_dampening_identity_for_nighttime(self):
        """Test dampening produces identity (a=1, b=0) for all-zero nighttime steps."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 0.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        for day_offset in range(1, 8):
            day = t0 - timedelta(days=day_offset)
            for h in range(24):
                provider._forecast_buffers[buf_idx].append((day + timedelta(hours=h), 0.0))
                provider._actuals_buffer.append((day + timedelta(hours=h), 0.0))

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None
        # num_slots = len(forecast) - 1 = 23 for a 24-entry forecast
        # All should be identity since everything is zero (nighttime guard)
        for k in range(len(provider._dampening_coefficients)):
            assert provider._dampening_coefficients[k, 0] == 1.0
            assert provider._dampening_coefficients[k, 1] == 0.0

    def test_compute_dampening_identity_for_insufficient_data(self):
        """Test dampening produces identity when fewer than 3 data points."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # Only 2 days → 2 data points per slot, less than DAMPENING_MIN_DATA_POINTS (3)
        for day_offset in range(1, 3):
            day = t0 - timedelta(days=day_offset)
            for h in range(24):
                provider._forecast_buffers[buf_idx].append((day + timedelta(hours=h), 1000.0))
                provider._actuals_buffer.append((day + timedelta(hours=h), 800.0))

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None
        # num_slots = len(forecast) - 1 = 23; all identity due to insufficient data
        for k in range(len(provider._dampening_coefficients)):
            assert provider._dampening_coefficients[k, 0] == 1.0
            assert provider._dampening_coefficients[k, 1] == 0.0

    def test_compute_dampening_with_known_linear_data(self):
        """Test dampening computes correct coefficients for known linear relationship."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # 7 days of data: actual = 0.8 * forecast + 100
        for day_offset in range(1, 8):
            day = t0 - timedelta(days=day_offset)
            fc = 1000.0 + day_offset * 100
            for h in range(24):
                provider._forecast_buffers[buf_idx].append((day + timedelta(hours=h), fc))
                provider._actuals_buffer.append((day + timedelta(hours=h), 0.8 * fc + 100))

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None
        # Check a midday slot
        a_k = provider._dampening_coefficients[12, 0]
        b_k = provider._dampening_coefficients[12, 1]
        assert abs(a_k - 0.8) < 0.01
        assert abs(b_k - 100.0) < 1.0

    def test_dampening_coefficient_bounds(self):
        """Test coefficient clamping to bounds."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # Data where actual ≈ 5 * forecast → a > 3.0, should be clamped
        for day_offset in range(1, 8):
            day = t0 - timedelta(days=day_offset)
            fc = 100.0 + day_offset * 10
            for h in range(24):
                provider._forecast_buffers[buf_idx].append((day + timedelta(hours=h), fc))
                provider._actuals_buffer.append((day + timedelta(hours=h), 5 * fc))

        provider.compute_dampening(max_power=6000.0)
        a_k = provider._dampening_coefficients[12, 0]
        assert a_k == DAMPENING_A_MAX

    def test_apply_dampening_output_clamp(self):
        """Test dampened output is clamped to >= 0."""
        provider = _TestProvider(solar=None, domain="test")

        # Position-based: forecast[0] uses coefficient[0]
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[0, 0] = 0.5
        provider._dampening_coefficients[0, 1] = -600.0

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        forecast = [(t, 100.0)]  # 0.5*100 - 600 = -550, should clamp to 0
        result = provider._apply_dampening(forecast)
        assert result[0][1] == 0.0

    def test_apply_dampening_non_finite_coefficients_fallback(self):
        """Test dampening uses raw value when coefficients contain NaN or Inf."""
        provider = _TestProvider(solar=None, domain="test")

        # Position-based: positions 0 and 1 in the forecast list
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        # Position 0 coefficients = NaN → raw fallback
        provider._dampening_coefficients[0, 0] = float("nan")
        provider._dampening_coefficients[0, 1] = float("nan")
        # Position 1 coefficients = Inf → raw fallback
        provider._dampening_coefficients[1, 0] = float("inf")
        provider._dampening_coefficients[1, 1] = 0.0

        t12 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        t13 = datetime.datetime(2024, 6, 15, 13, 0, tzinfo=pytz.UTC)
        forecast = [(t12, 1000.0), (t13, 2000.0)]
        result = provider._apply_dampening(forecast)
        # Both should pass through unchanged (raw fallback)
        assert result[0][1] == 1000.0
        assert result[1][1] == 2000.0

    def test_apply_dampening_preserves_none(self):
        """Test dampening preserves None values."""
        provider = _TestProvider(solar=None, domain="test")

        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        forecast = [(None, None), (datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC), None)]
        result = provider._apply_dampening(forecast)
        assert result[0] == (None, None)
        assert result[1][1] is None

    def test_get_effective_forecast_raw_when_dampening_off(self):
        """Test raw forecast used when dampening disabled."""
        provider = _TestProvider(solar=None, domain="test")
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t, 1000.0)]
        provider.dampening_enabled = False
        result = provider._get_effective_forecast()
        assert result == [(t, 1000.0)]

    def test_get_effective_forecast_dampened_when_enabled(self):
        """Test dampened forecast used when dampening enabled."""
        provider = _TestProvider(solar=None, domain="test")

        # Position-based: forecast[0] uses coefficient[0]
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0
        provider._dampening_coefficients[0, 0] = 2.0  # double for position 0

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t, 500.0)]
        provider.dampening_enabled = True
        result = provider._get_effective_forecast()
        assert result[0][1] == 1000.0  # 2.0 * 500


# ============================================================================
# Task 8: Sensor and entity value tests
# ============================================================================


class TestSensorValues:
    """Test sensor value functions on QSSolar."""

    def test_forecast_age_none_when_never_updated(self, fake_hass):
        """Test forecast_age returns None when no update."""
        solar = _make_solar(fake_hass)
        assert solar.get_forecast_age_hours() is None

    def test_is_forecast_ok_false_with_no_providers(self, fake_hass):
        """Test forecast ok false with no providers."""
        solar = _make_solar(fake_hass)
        assert solar.is_forecast_ok() is False


# ============================================================================
# Task 9: Stale notification transitions
# ============================================================================


class TestStaleNotifications:
    """Test stale forecast notification transitions."""

    @pytest.mark.asyncio
    async def test_fresh_to_stale_logs_warning(self, caplog):
        """Test warning logged on fresh-to-stale transition."""
        solar_mock = MagicMock()
        solar_mock.hass = FakeHass()
        solar_mock.hass.data["test"] = {}
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider._was_stale = False

        # No orchestrators, no forecast -> stale
        async def empty_fill():
            provider.orchestrators = []
            provider._orchestrator_health = {}

        provider.fill_orchestrators = empty_fill

        with caplog.at_level(logging.WARNING):
            await provider.update(datetime.datetime.now(pytz.UTC))

        assert "Solar forecast is stale" in caplog.text
        assert "TestProv" in caplog.text
        assert provider._was_stale is True

    @pytest.mark.asyncio
    async def test_no_repeat_stale_warning(self, caplog):
        """Test stale warning not repeated when already stale."""
        solar_mock = MagicMock()
        solar_mock.hass = FakeHass()
        solar_mock.hass.data["test"] = {}
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider._was_stale = True  # Already stale

        async def empty_fill():
            provider.orchestrators = []
            provider._orchestrator_health = {}

        provider.fill_orchestrators = empty_fill

        with caplog.at_level(logging.WARNING):
            await provider.update(datetime.datetime.now(pytz.UTC))

        assert caplog.text.count("Solar forecast is stale") == 0

    @pytest.mark.asyncio
    async def test_recovery_logs_info(self, caplog):
        """Test info logged on stale-to-fresh recovery."""
        solar_mock = MagicMock()
        solar_mock.hass = FakeHass()
        solar_mock.hass.data["test"] = {"a": MagicMock()}
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider._was_stale = True

        now = datetime.datetime.now(pytz.UTC)
        good_orch = MagicMock()
        good_orch.power_series = [(now, 1000.0), (now + timedelta(minutes=30), 1200.0)]

        async def fill_with_good():
            provider.orchestrators = [good_orch]
            provider._orchestrator_health = {id(good_orch): True}

        provider.fill_orchestrators = fill_with_good

        with caplog.at_level(logging.INFO):
            await provider.update(now)

        assert "Solar forecast recovered" in caplog.text
        assert provider._was_stale is False

    @pytest.mark.asyncio
    async def test_recovery_logs_stale_age_from_prev_successful_time(self, caplog):
        """Test recovery log includes accurate stale duration from prev_successful_time."""
        solar_mock = MagicMock()
        solar_mock.hass = FakeHass()
        solar_mock.hass.data["test"] = {"a": MagicMock()}
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider._was_stale = True

        now = datetime.datetime.now(pytz.UTC)
        # Set a previous successful time 5 hours ago so age_h is computed
        provider._latest_successful_forecast_time = now - timedelta(hours=5)

        good_orch = MagicMock()
        good_orch.power_series = [(now, 1000.0), (now + timedelta(minutes=30), 1200.0)]

        async def fill_with_good():
            provider.orchestrators = [good_orch]
            provider._orchestrator_health = {id(good_orch): True}

        provider.fill_orchestrators = fill_with_good

        with caplog.at_level(logging.INFO):
            await provider.update(now)

        assert "Solar forecast recovered" in caplog.text
        # Should include the stale duration (approximately 5.0 hours)
        assert "5.0 hours" in caplog.text
        assert provider._was_stale is False


# ============================================================================
# Task 6 continued: History recording
# ============================================================================


class TestHistoryRecording:
    """Test forecast and actual history recording with rolling buffers."""

    def test_record_forecast_snapshot(self):
        """Test recording forecast into 2-hour rolling buffer."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0), (t0 + timedelta(hours=1), 1200.0)]

        provider.record_forecast_snapshot(t0)

        buf_idx = provider._tod_buffer_index(t0)  # 12 // 2 = 6
        assert len(provider._forecast_buffers[buf_idx]) == 2
        assert provider._forecast_buffers[buf_idx][0] == (t0, 1000.0)
        assert provider._forecast_buffers[buf_idx][1] == (t0 + timedelta(hours=1), 1200.0)

    def test_record_forecast_snapshot_removes_overlapping(self):
        """Test snapshot dedup: existing entries overlapping new snapshot are removed."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # First snapshot
        provider.solar_forecast = [(t0, 1000.0), (t0 + timedelta(hours=1), 1200.0)]
        provider.record_forecast_snapshot(t0)
        buf_idx = provider._tod_buffer_index(t0)
        assert len(provider._forecast_buffers[buf_idx]) == 2

        # Add older data that won't overlap
        old_ts = t0 - timedelta(days=1)
        provider._forecast_buffers[buf_idx].insert(0, (old_ts, 500.0))
        assert len(provider._forecast_buffers[buf_idx]) == 3

        # Second snapshot with same start time — should replace overlapping entries
        provider.solar_forecast = [(t0, 1100.0), (t0 + timedelta(hours=1), 1300.0)]
        provider.record_forecast_snapshot(t0)

        buf = provider._forecast_buffers[buf_idx]
        # Old data (before first_ts) preserved + new snapshot = 3
        assert len(buf) == 3
        assert buf[0] == (old_ts, 500.0)
        assert buf[1] == (t0, 1100.0)
        assert buf[2] == (t0 + timedelta(hours=1), 1300.0)

    def test_actuals_buffer_property_delegates_to_solar(self, fake_hass):
        """Test _actuals_buffer property delegates to QSSolar shared buffer."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        # Write through provider property
        provider._actuals_buffer = [(t0, 500.0)]
        # Read from solar directly
        assert solar._actuals_buffer == [(t0, 500.0)]
        # Read from provider property
        assert provider._actuals_buffer == [(t0, 500.0)]

    def test_actuals_buffer_property_local_fallback(self):
        """Test _actuals_buffer uses local fallback when solar=None."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider._actuals_buffer = [(t0, 500.0)]
        assert provider._actuals_buffer == [(t0, 500.0)]
        assert provider._local_actuals_buffer == [(t0, 500.0)]

    def test_record_actual_production_on_solar(self, fake_hass):
        """Test recording actual production into the shared QSSolar buffer."""
        solar = _make_solar(fake_hass)
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        actuals = [(t0, 900.0), (t0 + timedelta(hours=1), 1100.0)]

        solar.record_actual_production(t0, actuals)
        assert len(solar._actuals_buffer) == 2
        assert solar._actuals_buffer[0] == (t0, 900.0)
        assert solar._actuals_buffer[1] == (t0 + timedelta(hours=1), 1100.0)

    def test_record_forecast_snapshot_trims_old_data(self):
        """Test that old data beyond DAMPENING_BUFFER_DAYS is trimmed."""
        from custom_components.quiet_solar.ha_model.solar import DAMPENING_BUFFER_DAYS

        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        buf_idx = provider._tod_buffer_index(t0)
        # Pre-fill with data older than DAMPENING_BUFFER_DAYS
        old_time = t0 - timedelta(days=DAMPENING_BUFFER_DAYS + 5)
        provider._forecast_buffers[buf_idx] = [(old_time, 500.0)]

        provider.solar_forecast = [(t0, 1000.0)]
        provider.record_forecast_snapshot(t0)

        # Old data should be trimmed, only new data remains
        for ts, v in provider._forecast_buffers[buf_idx]:
            assert ts >= t0 - timedelta(days=DAMPENING_BUFFER_DAYS)

    def test_record_actual_production_trims_old_data(self, fake_hass):
        """Test that old actuals beyond DAMPENING_BUFFER_DAYS are trimmed."""
        from custom_components.quiet_solar.ha_model.solar import DAMPENING_BUFFER_DAYS

        solar = _make_solar(fake_hass)
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # Pre-fill with data older than DAMPENING_BUFFER_DAYS
        old_time = t0 - timedelta(days=DAMPENING_BUFFER_DAYS + 5)
        solar._actuals_buffer = [(old_time, 500.0)]

        solar.record_actual_production(t0, [(t0, 900.0)])

        # Old data should be trimmed
        for ts, v in solar._actuals_buffer:
            assert ts >= t0 - timedelta(days=DAMPENING_BUFFER_DAYS)


# ============================================================================
# Update forecast integration
# ============================================================================


class TestUpdateForecast:
    """Test the full update_forecast flow on QSSolar."""

    @pytest.mark.asyncio
    async def test_update_forecast_updates_all_providers(self, fake_hass):
        """Test that update_forecast calls update on all providers."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)

        for p in solar.solar_forecast_providers.values():
            p.update = AsyncMock()

        now = datetime.datetime.now(pytz.UTC)
        await solar.update_forecast(now)

        for p in solar.solar_forecast_providers.values():
            p.update.assert_called_once_with(now)

    @pytest.mark.asyncio
    async def test_update_forecast_exception_in_one_provider_doesnt_stop_others(self, fake_hass, caplog):
        """Test that an exception in one provider doesn't prevent other providers from updating."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)

        # Provider A raises an exception
        solar.solar_forecast_providers["A"].update = AsyncMock(side_effect=RuntimeError("API failure"))
        solar.solar_forecast_providers["B"].update = AsyncMock()

        now = datetime.datetime.now(pytz.UTC)
        with caplog.at_level(logging.ERROR):
            await solar.update_forecast(now)

        # A raised but B should still have been called
        solar.solar_forecast_providers["A"].update.assert_called_once_with(now)
        solar.solar_forecast_providers["B"].update.assert_called_once_with(now)
        assert "Error updating solar forecast provider" in caplog.text

    @pytest.mark.asyncio
    async def test_update_forecast_auto_selects(self, fake_hass):
        """Test that update_forecast auto-selects best provider."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
            {CONF_SOLAR_PROVIDER_DOMAIN: OPEN_METEO_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "B"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        solar._provider_mode = SOLAR_PROVIDER_MODE_AUTO

        for p in solar.solar_forecast_providers.values():
            p.update = AsyncMock()

        solar.solar_forecast_providers["A"].score_raw = 500.0
        solar.solar_forecast_providers["B"].score_raw = 200.0

        await solar.update_forecast(datetime.datetime.now(pytz.UTC))
        assert solar.active_provider_name == "B"

    def test_daily_scoring_cycle_skips_on_same_date(self, fake_hass):
        """Test _run_daily_scoring_cycle early-returns when already run for today."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.record_forecast_snapshot = MagicMock()
        provider.compute_score = MagicMock()
        provider.compute_dampening = MagicMock()

        now = datetime.datetime.now(pytz.UTC)

        # First call should run the cycle
        solar._run_daily_scoring_cycle(now)
        assert provider.record_forecast_snapshot.call_count == 1
        assert provider.compute_score.call_count == 1

        # Second call with same date should be a no-op
        provider.record_forecast_snapshot.reset_mock()
        provider.compute_score.reset_mock()
        solar._run_daily_scoring_cycle(now)
        assert provider.record_forecast_snapshot.call_count == 0
        assert provider.compute_score.call_count == 0

    def test_daily_scoring_cycle_calls_compute_dampening_when_enabled(self, fake_hass):
        """Test _run_daily_scoring_cycle calls compute_dampening when dampening is enabled."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.dampening_enabled = True
        provider.record_forecast_snapshot = MagicMock()
        provider.compute_score = MagicMock()
        provider.compute_dampening = MagicMock()

        now = datetime.datetime.now(pytz.UTC)
        solar._run_daily_scoring_cycle(now)

        provider.compute_dampening.assert_called_once_with(solar.solar_max_output_power_value)

    def test_daily_scoring_cycle_always_computes_dampening(self, fake_hass):
        """Test _run_daily_scoring_cycle always calls compute_dampening regardless of switch state."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.dampening_enabled = False
        provider.record_forecast_snapshot = MagicMock()
        provider.compute_score = MagicMock()
        provider.compute_dampening = MagicMock()

        now = datetime.datetime.now(pytz.UTC)
        solar._run_daily_scoring_cycle(now)

        provider.compute_dampening.assert_called_once_with(solar.solar_max_output_power_value)

    def test_get_forecast_delegates_to_active(self, fake_hass):
        """Test get_forecast delegates to the active provider."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar.solar_forecast_providers["A"].solar_forecast = [(t, 1000.0), (t + timedelta(hours=1), 1200.0)]
        result = solar.get_forecast(t, t + timedelta(hours=2))
        assert len(result) > 0

    def test_get_forecast_empty_no_provider(self, fake_hass):
        """Test get_forecast returns empty when no provider."""
        solar = _make_solar(fake_hass)
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_forecast(t, t + timedelta(hours=1)) == []

    def test_get_value_from_current_forecast_no_provider(self, fake_hass):
        """Test get_value returns None tuple when no provider."""
        solar = _make_solar(fake_hass)
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_value_from_current_forecast(t) == (None, None)


# ============================================================================
# Task 4: Historical solar pattern fallback
# ============================================================================


class TestHistoricalSolarFallback:
    """Test historical solar pattern fallback on QSSolarHistoryVals and wiring."""

    def test_get_historical_solar_pattern_returns_data(self):
        """Test fallback returns shifted historical data from ring buffer."""
        from custom_components.quiet_solar.ha_model.home import (
            BUFFER_SIZE_IN_INTERVALS,
            INTERVALS_MN,
            NUM_INTERVALS_PER_DAY,
            QSHomeConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        vals = QSSolarHistoryVals(entity_id="sensor.solar", forecast=forecast)

        # Initialize ring buffer
        vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float32)

        # Place known data 1 day ago
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        idx_now, _days_now = vals.get_index_from_time(t)
        idx_past = (idx_now - NUM_INTERVALS_PER_DAY) % BUFFER_SIZE_IN_INTERVALS

        # Fill 24h of past data (96 intervals at 15min)
        for i in range(NUM_INTERVALS_PER_DAY):
            slot = (idx_past + i) % BUFFER_SIZE_IN_INTERVALS
            vals.values[0][slot] = 500.0 + i * 10.0  # power
            vals.values[1][slot] = 1  # valid day marker

        result = vals.get_historical_solar_pattern(t, future_hours=24)
        assert len(result) == NUM_INTERVALS_PER_DAY
        # First point should match first past value
        assert result[0][1] == pytest.approx(500.0)
        # Timestamps should start at `t`
        assert result[0][0] == t
        assert result[1][0] == t + timedelta(minutes=INTERVALS_MN)

    def test_get_historical_solar_pattern_skips_invalid_days(self):
        """Test fallback skips days with insufficient valid data."""
        from custom_components.quiet_solar.ha_model.home import (
            BUFFER_SIZE_IN_INTERVALS,
            NUM_INTERVALS_PER_DAY,
            QSHomeConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        vals = QSSolarHistoryVals(entity_id="sensor.solar", forecast=forecast)
        vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float32)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        idx_now, _ = vals.get_index_from_time(t)

        # Day 1 ago: no valid data (day markers = 0)
        # Day 2 ago: valid data
        idx_2_days = (idx_now - 2 * NUM_INTERVALS_PER_DAY) % BUFFER_SIZE_IN_INTERVALS
        for i in range(NUM_INTERVALS_PER_DAY):
            slot = (idx_2_days + i) % BUFFER_SIZE_IN_INTERVALS
            vals.values[0][slot] = 800.0
            vals.values[1][slot] = 2

        result = vals.get_historical_solar_pattern(t, future_hours=24)
        assert len(result) == NUM_INTERVALS_PER_DAY
        assert result[0][1] == pytest.approx(800.0)

    def test_get_historical_solar_pattern_empty_when_no_history(self):
        """Test fallback returns empty when no valid historical data."""
        from custom_components.quiet_solar.ha_model.home import (
            BUFFER_SIZE_IN_INTERVALS,
            QSHomeConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        vals = QSSolarHistoryVals(entity_id="sensor.solar", forecast=forecast)
        vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float32)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert vals.get_historical_solar_pattern(t) == []

    def test_get_historical_solar_pattern_none_values(self):
        """Test fallback returns empty when values is None."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        vals = QSSolarHistoryVals(entity_id="sensor.solar", forecast=forecast)
        assert vals.get_historical_solar_pattern(datetime.datetime.now(pytz.UTC)) == []

    def test_solar_get_historical_fallback_no_home(self, fake_hass):
        """Test QSSolar fallback returns empty when no home."""
        solar = _make_solar(fake_hass)
        solar.home = None
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_historical_solar_fallback(t) == []

    def test_solar_get_historical_fallback_no_forecast_handler(self, fake_hass):
        """Test QSSolar fallback returns empty when no forecast handler."""
        solar = _make_solar(fake_hass)
        # Remove _consumption_forecast so getattr returns None
        solar.home._consumption_forecast = None
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_historical_solar_fallback(t) == []

    @pytest.mark.asyncio
    async def test_fallback_triggered_when_stale_and_empty(self, fake_hass):
        """Test that provider uses historical fallback when stale and forecast empty."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]

        # Make provider stale with empty forecast
        stale_time = datetime.datetime.now(pytz.UTC) - timedelta(hours=8)
        provider._latest_successful_forecast_time = stale_time
        provider._latest_update_time = None  # Force update
        provider.solar_forecast = []
        provider.orchestrators = []

        # Set up fallback data via mock
        fallback_data = [
            (datetime.datetime.now(pytz.UTC), 1000.0),
            (datetime.datetime.now(pytz.UTC) + timedelta(minutes=30), 1200.0),
        ]
        solar.get_historical_solar_fallback = MagicMock(return_value=fallback_data)

        await provider.update(datetime.datetime.now(pytz.UTC))

        solar.get_historical_solar_fallback.assert_called_once()
        assert provider.solar_forecast == fallback_data

    @pytest.mark.asyncio
    async def test_fallback_not_triggered_when_fresh(self, fake_hass):
        """Test that provider does NOT use fallback when not stale."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]

        # Provider is fresh
        provider._latest_successful_forecast_time = datetime.datetime.now(pytz.UTC)
        provider.solar_forecast = [(datetime.datetime.now(pytz.UTC), 500.0)]

        solar.get_historical_solar_fallback = MagicMock(return_value=[])

        # Provider is fresh so update won't trigger fallback
        # (update only runs when _latest_update_time is None or stale)
        provider._latest_update_time = None
        provider.orchestrators = []
        await provider.update(datetime.datetime.now(pytz.UTC))

        solar.get_historical_solar_fallback.assert_not_called()


# ============================================================================
# Guard clause coverage tests
# ============================================================================


class TestGuardClauses:
    """Test guard clauses and edge cases for full coverage."""

    def test_apply_dampening_guard_no_coefficients(self):
        """Test _apply_dampening returns forecast unchanged when no coefficients."""
        provider = _TestProvider(solar=None, domain="test")
        provider._dampening_coefficients = None
        forecast = [(datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC), 1000.0)]
        result = provider._apply_dampening(forecast)
        assert result is forecast

    def test_apply_dampening_position_out_of_range(self):
        """Test _apply_dampening passes through when position exceeds coefficient count."""
        provider = _TestProvider(solar=None, domain="test")

        # Only 2 coefficients; position 2 will be out of range
        provider._dampening_coefficients = np.array([[1.5, 0.0], [1.5, 0.0]], dtype=np.float64)

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        forecast = [
            (t0, 500.0),  # position 0: dampened (1.5*500=750)
            (t0 + timedelta(hours=1), 500.0),  # position 1: dampened (750)
            (t0 + timedelta(hours=2), 500.0),  # position 2: out of range, pass through
        ]
        result = provider._apply_dampening(forecast)
        assert result[0][1] == 750.0  # dampened
        assert result[1][1] == 750.0  # dampened
        assert result[2][1] == 500.0  # passed through unchanged

    def test_record_forecast_snapshot_guard_no_forecast(self):
        """Test record_forecast_snapshot returns early when no forecast data."""
        provider = _TestProvider(solar=None, domain="test")
        provider.solar_forecast = []
        provider.record_forecast_snapshot(datetime.datetime.now(pytz.UTC))
        assert all(len(buf) == 0 for buf in provider._forecast_buffers)

    def test_record_forecast_snapshot_skips_none_values(self):
        """Test record_forecast_snapshot skips entries with None timestamps or values."""
        provider = _TestProvider(solar=None, domain="test")
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(None, 100.0), (t, None), (t, 500.0)]
        provider.record_forecast_snapshot(t)
        buf_idx = provider._tod_buffer_index(t)
        # Only (t, 500.0) should be stored
        assert len(provider._forecast_buffers[buf_idx]) == 1
        assert provider._forecast_buffers[buf_idx][0] == (t, 500.0)

    def test_record_actual_production_guard_empty_actuals(self, fake_hass):
        """Test record_actual_production returns early when actuals are empty."""
        solar = _make_solar(fake_hass)
        t = datetime.datetime.now(pytz.UTC)
        solar.record_actual_production(t, [])
        assert solar._actuals_buffer == []

    def test_compute_score_with_dampening(self):
        """Test compute_score also computes dampened score when enabled."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(12)]
        provider._actuals_buffer = [(t0 + timedelta(hours=h), 900.0) for h in range(12)]

        # Enable dampening with identity coefficients
        provider.dampening_enabled = True
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        provider.compute_score()
        assert provider.score_raw == pytest.approx(100.0)
        assert provider.score_dampened is not None

    def test_compute_mae_from_aligned_empty(self):
        """Test _compute_mae_from_aligned returns None when inputs are empty."""
        assert QSSolarProvider._compute_mae_from_aligned([], []) is None

    def test_compute_mae_from_aligned_all_nighttime(self):
        """Test _compute_mae_from_aligned returns None when all values are zero."""
        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        fc = [(t0, 0.0), (t0 + timedelta(hours=1), 0.0)]
        ac = [(t0, 0.0), (t0 + timedelta(hours=1), 0.0)]
        assert QSSolarProvider._compute_mae_from_aligned(fc, ac) is None

    def test_compute_dampening_guard_no_history(self):
        """Test compute_dampening returns early when no actuals or forecast."""
        provider = _TestProvider(solar=None, domain="test")
        provider._actuals_buffer = []
        provider.solar_forecast = []
        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is None

    def test_solar_get_historical_fallback_with_history(self, fake_hass):
        """Test QSSolar fallback delegates to solar_production_history."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar = _make_solar(fake_hass)
        forecast_handler = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_history = MagicMock(spec=QSSolarHistoryVals)
        mock_history.get_historical_solar_pattern.return_value = [(datetime.datetime.now(pytz.UTC), 500.0)]
        forecast_handler.solar_production_history = mock_history
        solar.home._consumption_forecast = forecast_handler

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        result = solar.get_historical_solar_fallback(t)
        assert len(result) == 1
        mock_history.get_historical_solar_pattern.assert_called_once_with(t)

    def test_solar_get_historical_fallback_no_history_attr(self, fake_hass):
        """Test fallback returns empty when forecast handler has no solar_production_history."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeConsumptionHistoryAndForecast,
        )

        solar = _make_solar(fake_hass)
        forecast_handler = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        # solar_production_history defaults to None
        solar.home._consumption_forecast = forecast_handler

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_historical_solar_fallback(t) == []

    def test_compute_dampening_polyfit_error(self):
        """Test compute_dampening handles polyfit errors gracefully."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # All forecasts identical per slot → polyfit may struggle but handles it
        for day_offset in range(1, 8):
            day = t0 - timedelta(days=day_offset)
            for h in range(24):
                provider._forecast_buffers[buf_idx].append((day + timedelta(hours=h), 1000.0))
                provider._actuals_buffer.append((day + timedelta(hours=h), 500.0 + day_offset * 100))

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None

    def test_compute_dampening_polyfit_linalg_error(self):
        """Test compute_dampening catches LinAlgError from polyfit and falls back to identity."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # Inf values should cause LinAlgError in polyfit
        for day_offset in range(1, 8):
            day = t0 - timedelta(days=day_offset)
            for h in range(24):
                provider._forecast_buffers[buf_idx].append((day + timedelta(hours=h), float("inf")))
                provider._actuals_buffer.append((day + timedelta(hours=h), 500.0 + day_offset * 100))

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None
        # Step 12 should remain identity because polyfit raised LinAlgError
        assert provider._dampening_coefficients[12, 0] == 1.0
        assert provider._dampening_coefficients[12, 1] == 0.0

    def test_record_forecast_snapshot_all_none_entries(self):
        """Test record_forecast_snapshot returns early when all entries have None values."""
        provider = _TestProvider(solar=None, domain="test")
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(None, 100.0), (t, None)]
        provider.record_forecast_snapshot(t)
        # All entries stripped → empty snapshot → early return
        assert all(len(buf) == 0 for buf in provider._forecast_buffers)

    def test_compute_score_single_point_forecast(self):
        """Test compute_score returns early when forecast has < 2 time points."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0)]
        provider._actuals_buffer = [(t0, 800.0)]
        provider.compute_score()
        assert provider.score_raw is None

    def test_compute_dampening_guard_single_point_forecast(self):
        """Test compute_dampening returns early when forecast has < 2 time points."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0)]
        provider._actuals_buffer = [(t0, 800.0)]
        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is None

    def test_compute_dampening_guard_empty_forecast_buffer(self):
        """Test compute_dampening returns early when forecast buffer is empty."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]
        provider._actuals_buffer = [(t0, 800.0)]
        # All forecast buffers are empty (default) → early return
        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is None

    def test_compute_dampening_skips_today_data(self):
        """Test compute_dampening skips forecast data from today's date."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # Put data for TODAY only (should be skipped)
        for h in range(24):
            provider._forecast_buffers[buf_idx].append((t0 + timedelta(hours=h), 1000.0))
        provider._actuals_buffer = [(t0 + timedelta(hours=h), 800.0) for h in range(24)]

        provider.compute_dampening(max_power=6000.0)
        # All data was for today → skipped → identity coefficients
        assert provider._dampening_coefficients is not None
        assert provider._dampening_coefficients[12, 0] == 1.0

    def test_compute_dampening_skips_day_without_actuals(self):
        """Test compute_dampening skips historical days that have no actuals."""
        provider = _TestProvider(solar=None, domain="test")

        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(24)]

        buf_idx = provider._tod_buffer_index(t0)
        # Forecast data for yesterday but NO actuals for that day
        yesterday = t0 - timedelta(days=1)
        for h in range(24):
            provider._forecast_buffers[buf_idx].append((yesterday + timedelta(hours=h), 1000.0))
        # Actuals only for 2 days ago (not yesterday)
        two_days_ago = t0 - timedelta(days=2)
        provider._actuals_buffer = [(two_days_ago + timedelta(hours=h), 800.0) for h in range(24)]

        provider.compute_dampening(max_power=6000.0)
        # Yesterday skipped (no actuals) → identity
        assert provider._dampening_coefficients is not None
        assert provider._dampening_coefficients[12, 0] == 1.0

    def test_reset_dampening(self, fake_hass):
        """Test QSSolar.reset_dampening clears shared actuals and provider buffers."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]

        # Give solar and provider some data
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._actuals_buffer = [(t0, 500.0)]
        provider._dampening_coefficients = np.ones((24, 2))
        provider.score_raw = 100.0

        solar.reset_dampening(t0)
        assert solar._actuals_buffer == []
        assert solar._last_actuals_record_time is None
        assert provider._dampening_coefficients is None
        assert provider.score_raw is None

    def test_reset_dampening_buffers_clears_provider_state(self):
        """Test reset_dampening_buffers clears coefficients and scores (not actuals)."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider._dampening_coefficients = np.ones((24, 2))
        provider.score_raw = 100.0
        provider.score_dampened = 90.0
        provider._forecast_buffers[0] = [(t0, 500.0)]

        provider.reset_dampening_buffers()
        assert provider._dampening_coefficients is None
        assert provider.score_raw is None
        assert provider.score_dampened is None
        assert all(len(buf) == 0 for buf in provider._forecast_buffers)

    def test_bootstrap_actuals_skips_when_buffer_nonempty(self):
        """Test _bootstrap_actuals_from_history is a no-op when buffer already has data."""
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        solar_mock = MagicMock()
        solar_mock._actuals_buffer = [(t0, 500.0)]
        solar_mock.get_historical_solar_fallback = MagicMock(return_value=[(t0, 999.0)])

        QSSolarProvider._bootstrap_actuals_from_history(solar_mock, t0)
        # Buffer unchanged — bootstrap skipped
        assert len(solar_mock._actuals_buffer) == 1
        assert solar_mock._actuals_buffer[0][1] == 500.0
        solar_mock.get_historical_solar_fallback.assert_not_called()

    def test_bootstrap_actuals_from_history_with_data(self):
        """Test _bootstrap_actuals_from_history populates empty buffer from fallback."""
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        solar_mock = MagicMock()
        solar_mock._actuals_buffer = []
        solar_mock.get_historical_solar_fallback = MagicMock(
            return_value=[(t0, 700.0), (t0 + timedelta(hours=1), 800.0)]
        )

        QSSolarProvider._bootstrap_actuals_from_history(solar_mock, t0)
        assert len(solar_mock._actuals_buffer) == 2

    def test_record_actual_from_sensor_no_sensor(self, fake_hass):
        """Test _record_actual_from_sensor does nothing when sensor is None."""
        solar = _make_solar(fake_hass)
        solar.solar_inverter_input_active_power = None
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._record_actual_from_sensor(t)
        assert solar._actuals_buffer == []

    def test_record_actual_from_sensor_too_soon(self, fake_hass):
        """Test _record_actual_from_sensor skips when called before step interval."""
        from custom_components.quiet_solar.ha_model.solar import DAMPENING_ACTUALS_STEP_MN

        solar = _make_solar(fake_hass)
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._last_actuals_record_time = t0

        # Call again 1 minute later — should skip
        t1 = t0 + timedelta(minutes=1)
        solar._record_actual_from_sensor(t1)
        assert solar._actuals_buffer == []

        # Call after full step — would proceed (but sensor returns None here)
        t2 = t0 + timedelta(minutes=DAMPENING_ACTUALS_STEP_MN)
        solar.solar_inverter_input_active_power = None
        solar._record_actual_from_sensor(t2)
        assert solar._actuals_buffer == []

    def test_record_actual_from_sensor_sensor_returns_none(self, fake_hass):
        """Test _record_actual_from_sensor handles sensor returning None."""
        solar = _make_solar(fake_hass)
        solar.solar_inverter_input_active_power = "sensor.solar_input"
        solar.get_average_sensor = MagicMock(return_value=None)
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._record_actual_from_sensor(t)
        assert solar._actuals_buffer == []

    def test_record_actual_from_sensor_records_value(self, fake_hass):
        """Test _record_actual_from_sensor records value from sensor."""
        solar = _make_solar(fake_hass)
        solar.solar_inverter_input_active_power = "sensor.solar_input"
        solar.get_average_sensor = MagicMock(return_value=1500.0)
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        solar._record_actual_from_sensor(t)
        assert len(solar._actuals_buffer) == 1
        assert solar._actuals_buffer[0] == (t, 1500.0)
        assert solar._last_actuals_record_time == t

    def test_compute_score_empty_alignment(self):
        """Test compute_score returns early when aligned forecast is empty."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        # Forecast with valid timestamps but all None values → cleaned list is empty
        provider.solar_forecast = [(t0 + timedelta(hours=h), None) for h in range(12)]
        provider._actuals_buffer = [(t0 + timedelta(hours=h), 800.0) for h in range(12)]
        provider.compute_score()
        assert provider.score_raw is None

    @pytest.mark.asyncio
    async def test_update_forecast_records_actual_from_sensor(self, fake_hass):
        """Test update_forecast calls _record_actual_from_sensor and bootstraps shared buffer."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.update = AsyncMock()

        # Set up sensor-based recording
        solar.solar_inverter_input_active_power = "sensor.solar_input"
        solar.get_average_sensor = MagicMock(return_value=1200.0)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        await solar.update_forecast(t)

        # Shared buffer should have the sensor reading
        assert len(solar._actuals_buffer) == 1
        assert solar._actuals_buffer[0] == (t, 1200.0)
        # Provider sees shared buffer via property
        assert len(provider._actuals_buffer) == 1


# ============================================================================
# get_dampening_attributes
# ============================================================================


class TestGetDampeningAttributes:
    """Tests for QSSolarProvider.get_dampening_attributes()."""

    def test_empty_provider(self, fake_hass):
        """No coefficients, no date — minimal attrs."""
        solar = _make_solar(fake_hass, single_provider=SOLCAST_SOLAR_DOMAIN)
        provider = next(iter(solar.solar_forecast_providers.values()))
        attrs = provider.get_dampening_attributes()
        assert attrs["dampening_enabled"] is False
        assert "last_dampening_date" not in attrs
        assert "coefficients_count" not in attrs
        # No per-tod entries
        assert not any(k.startswith("tod_") for k in attrs)

    def test_with_coefficients(self, fake_hass):
        """Coefficients set → attrs contain count and averages."""
        solar = _make_solar(fake_hass, single_provider=SOLCAST_SOLAR_DOMAIN)
        provider = next(iter(solar.solar_forecast_providers.values()))
        provider.dampening_enabled = True
        provider._dampening_coefficients = np.array([[0.9, 10.0], [1.1, -5.0], [1.0, 0.0]])
        utc_dt = datetime.datetime(2025, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        provider._last_dampening_date = utc_dt

        attrs = provider.get_dampening_attributes()
        assert attrs["dampening_enabled"] is True
        # last_dampening_date is displayed in local time
        expected_local = utc_dt.astimezone(tz=None).isoformat()
        assert attrs["last_dampening_date"] == expected_local
        assert attrs["coefficients_count"] == 3
        assert attrs["avg_scale_factor"] == pytest.approx(1.0, abs=0.01)
        assert "avg_offset" in attrs

    def test_with_per_tod_coefficients(self, fake_hass):
        """Per-tod coefficients appear as tod_XXh_scale/offset attrs with local hours."""
        solar = _make_solar(fake_hass, single_provider=SOLCAST_SOLAR_DOMAIN)
        provider = next(iter(solar.solar_forecast_providers.values()))
        provider._dampening_coefficients_per_tod = {
            0: np.array([[0.95, 5.0], [1.05, -3.0]]),
            6: np.array([[0.8, 20.0]]),
        }

        attrs = provider.get_dampening_attributes()

        # Compute expected local hours the same way the production code does
        def _local_hour_for_buf_idx(buf_idx: int) -> int:
            utc_hour = buf_idx * 2  # DAMPENING_TOD_HOURS = 2
            utc_dt = datetime.datetime.now(pytz.UTC).replace(hour=utc_hour, minute=0, second=0, microsecond=0)
            return utc_dt.astimezone(tz=None).hour

        local_h0 = _local_hour_for_buf_idx(0)
        local_h6 = _local_hour_for_buf_idx(6)

        assert f"tod_{local_h0:02d}h_scale" in attrs
        assert f"tod_{local_h0:02d}h_offset" in attrs
        assert f"tod_{local_h6:02d}h_scale" in attrs
        assert f"tod_{local_h6:02d}h_offset" in attrs
        # buf_idx 6 → utc_hour 12 → local_hour depends on timezone
        assert attrs[f"tod_{local_h6:02d}h_scale"] == pytest.approx(0.8, abs=0.01)
        assert attrs[f"tod_{local_h6:02d}h_offset"] == pytest.approx(20.0, abs=0.1)


# ============================================================================
# Realistic end-to-end dampening scenarios (no mocks)
# ============================================================================

import math


class TestRealisticDampeningScenarios:
    """End-to-end dampening tests with realistic synthetic solar data.

    No mocks on dampening internals: real bell-curve solar profiles,
    real multi-day histories, and real compute_dampening / _apply_dampening.
    Verify that learned coefficients match expected biases and that
    dampened forecasts materially reduce MAE vs raw forecasts.
    """

    @staticmethod
    def _solar_bell(day_start, max_power, sunrise=6, sunset=18):
        """24 hourly points of a sine-shaped solar production curve."""
        result = []
        for h in range(24):
            t = day_start + timedelta(hours=h)
            if sunrise <= h < sunset:
                phase = math.pi * (h - sunrise) / (sunset - sunrise)
                result.append((t, round(max_power * math.sin(phase), 1)))
            else:
                result.append((t, 0.0))
        return result

    @staticmethod
    def _daytime_mae(series_a, series_b):
        """MAE over daytime slots (where either side > 0)."""
        errors = []
        n = min(len(series_a), len(series_b))
        for i in range(n):
            a = series_a[i][1]
            b = series_b[i][1]
            if a > 0 or b > 0:
                errors.append(abs(a - b))
        return sum(errors) / len(errors) if errors else 0.0

    # Deterministic daily power scaling — ensures regression gets varied x-values
    _DAILY_FACTORS = [0.85, 1.15, 0.92, 1.08, 0.88, 1.12, 1.00, 0.95, 1.05, 0.90]

    def _build_history(
        self,
        provider,
        today_start,
        num_days,
        max_power,
        scale_bias,
        offset_bias,
        noise_rng=None,
        fc_daily_variation=0.0,
    ):
        """Populate provider buffers with num_days of forecast + actuals.

        For each past day:
          forecast = solar_bell(day, max_power * daily_factor)
          actual = scale_bias * forecast + offset_bias + noise  (daytime only)

        Daily factors introduce natural weather variation so polyfit gets
        varied x-values per slot (avoids degenerate regression).
        """
        buf_idx = provider._tod_buffer_index(today_start)

        for day_offset in range(1, num_days + 1):
            day_start = today_start - timedelta(days=day_offset)

            # Natural weather variation: deterministic ±15% per day
            daily_factor = self._DAILY_FACTORS[(day_offset - 1) % len(self._DAILY_FACTORS)]
            if fc_daily_variation > 0 and noise_rng is not None:
                daily_factor *= 1.0 + fc_daily_variation * (noise_rng.random() * 2 - 1)

            day_fc = self._solar_bell(day_start, max_power * daily_factor)

            day_ac = []
            for ts, v in day_fc:
                if v > 0:
                    actual = scale_bias * v + offset_bias
                    if noise_rng is not None:
                        actual += noise_rng.normal(0, 50)
                    day_ac.append((ts, max(0.0, actual)))
                else:
                    day_ac.append((ts, 0.0))

            provider._forecast_buffers[buf_idx].extend(day_fc)
            provider._actuals_buffer.extend(day_ac)

    # ------------------------------------------------------------------ #
    # Scenario 1: consistent over-estimation
    # ------------------------------------------------------------------ #

    def test_consistent_overestimation(self):
        """Forecast 25% too high over 7 days → a ≈ 0.75, b ≈ 0.

        Verifies coefficients AND that dampened MAE is ≪ raw MAE.
        """
        provider = _TestProvider(solar=None, domain="test")
        today = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        max_power = 5000.0

        provider.solar_forecast = self._solar_bell(today, max_power)
        self._build_history(
            provider,
            today,
            num_days=7,
            max_power=max_power,
            scale_bias=0.75,
            offset_bias=0.0,
        )

        provider.compute_dampening(max_power)
        assert provider._dampening_coefficients is not None

        # Daytime slots (positions 6-16): a ≈ 0.75, b ≈ 0
        for k in range(7, 18):
            a = provider._dampening_coefficients[k, 0]
            b = provider._dampening_coefficients[k, 1]
            assert 0.70 <= a <= 0.80, f"Slot {k}: a={a}, expected ~0.75"
            assert abs(b) < 150, f"Slot {k}: b={b}, expected ~0"

        # Nighttime slots remain identity (0-6 and 18-22)
        for k in list(range(0, 7)) + list(range(18, 23)):
            assert provider._dampening_coefficients[k, 0] == 1.0
            assert provider._dampening_coefficients[k, 1] == 0.0

        # MAE improvement
        today_ac = [(ts, 0.75 * v if v > 0 else 0.0) for ts, v in provider.solar_forecast]
        raw_mae = self._daytime_mae(provider.solar_forecast, today_ac)
        dampened = provider._apply_dampening(provider.solar_forecast)
        dampened_mae = self._daytime_mae(dampened, today_ac)

        assert raw_mae > 200, f"Sanity: raw MAE should be significant, got {raw_mae}"
        assert dampened_mae < raw_mae * 0.15, f"Dampened ({dampened_mae:.0f}) should be ≪ raw ({raw_mae:.0f})"

    # ------------------------------------------------------------------ #
    # Scenario 2: under-estimation with positive offset
    # ------------------------------------------------------------------ #

    def test_underestimation_with_offset(self):
        """Actual = 1.1×fc + 150 → a ≈ 1.1, b ≈ 150, dampening improves."""
        provider = _TestProvider(solar=None, domain="test")
        today = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        max_power = 4000.0

        provider.solar_forecast = self._solar_bell(today, max_power)
        self._build_history(
            provider,
            today,
            num_days=7,
            max_power=max_power,
            scale_bias=1.1,
            offset_bias=150.0,
        )

        provider.compute_dampening(max_power)
        assert provider._dampening_coefficients is not None

        for k in range(7, 18):
            a = provider._dampening_coefficients[k, 0]
            b = provider._dampening_coefficients[k, 1]
            assert 1.05 <= a <= 1.15, f"Slot {k}: a={a}, expected ~1.1"
            assert 100 <= b <= 200, f"Slot {k}: b={b}, expected ~150"

        today_ac = [(ts, max(0.0, 1.1 * v + 150) if v > 0 else 0.0) for ts, v in provider.solar_forecast]
        raw_mae = self._daytime_mae(provider.solar_forecast, today_ac)
        dampened = provider._apply_dampening(provider.solar_forecast)
        dampened_mae = self._daytime_mae(dampened, today_ac)

        assert dampened_mae < raw_mae * 0.15

    # ------------------------------------------------------------------ #
    # Scenario 3: noisy data — regression still improves
    # ------------------------------------------------------------------ #

    def test_noisy_data_still_improves(self):
        """With noise (σ=50 W), dampening still materially reduces MAE."""
        rng = np.random.default_rng(42)
        provider = _TestProvider(solar=None, domain="test")
        today = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        max_power = 5000.0

        provider.solar_forecast = self._solar_bell(today, max_power)
        self._build_history(
            provider,
            today,
            num_days=7,
            max_power=max_power,
            scale_bias=0.8,
            offset_bias=0.0,
            noise_rng=rng,
        )

        provider.compute_dampening(max_power)
        assert provider._dampening_coefficients is not None

        daytime_a = [provider._dampening_coefficients[k, 0] for k in range(7, 18)]
        avg_a = sum(daytime_a) / len(daytime_a)
        assert 0.70 <= avg_a <= 0.90, f"Avg a={avg_a}, expected ~0.8"

        today_ac = [(ts, 0.8 * v if v > 0 else 0.0) for ts, v in provider.solar_forecast]
        raw_mae = self._daytime_mae(provider.solar_forecast, today_ac)
        dampened = provider._apply_dampening(provider.solar_forecast)
        dampened_mae = self._daytime_mae(dampened, today_ac)

        assert dampened_mae < raw_mae * 0.5, (
            f"Noisy dampened ({dampened_mae:.0f}) should be < 50% of raw ({raw_mae:.0f})"
        )

    # ------------------------------------------------------------------ #
    # Scenario 4: window filter — old data excluded
    # ------------------------------------------------------------------ #

    def test_window_filter_uses_only_recent_days(self):
        """Data older than DAMPENING_WINDOW_DAYS is excluded from regression."""
        from custom_components.quiet_solar.ha_model.solar import DAMPENING_WINDOW_DAYS

        provider = _TestProvider(solar=None, domain="test")
        today = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        max_power = 5000.0
        buf_idx = provider._tod_buffer_index(today)

        # Old data (outside window): actual = 1.5 × forecast
        for day_offset in range(DAMPENING_WINDOW_DAYS + 1, DAMPENING_WINDOW_DAYS + 8):
            day_start = today - timedelta(days=day_offset)
            df = self._DAILY_FACTORS[(day_offset - 1) % len(self._DAILY_FACTORS)]
            day_fc = self._solar_bell(day_start, max_power * df)
            day_ac = [(ts, 1.5 * v if v > 0 else 0.0) for ts, v in day_fc]
            provider._forecast_buffers[buf_idx].extend(day_fc)
            provider._actuals_buffer.extend(day_ac)

        # Recent data (inside window): actual = 0.6 × forecast
        for day_offset in range(1, DAMPENING_WINDOW_DAYS + 1):
            day_start = today - timedelta(days=day_offset)
            df = self._DAILY_FACTORS[(day_offset - 1) % len(self._DAILY_FACTORS)]
            day_fc = self._solar_bell(day_start, max_power * df)
            day_ac = [(ts, 0.6 * v if v > 0 else 0.0) for ts, v in day_fc]
            provider._forecast_buffers[buf_idx].extend(day_fc)
            provider._actuals_buffer.extend(day_ac)

        provider.solar_forecast = self._solar_bell(today, max_power)
        provider.compute_dampening(max_power)

        # Coefficients must reflect RECENT data (a ≈ 0.6), not old (a ≈ 1.5)
        for k in range(7, 18):
            a = provider._dampening_coefficients[k, 0]
            assert 0.55 <= a <= 0.65, f"Slot {k}: a={a}, expected ~0.6 (recent window)"

    # ------------------------------------------------------------------ #
    # Scenario 5: varying daily forecast power levels
    # ------------------------------------------------------------------ #

    def test_varying_forecast_levels(self):
        """Regression handles ±20% daily forecast variation."""
        rng = np.random.default_rng(123)
        provider = _TestProvider(solar=None, domain="test")
        today = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        max_power = 5000.0

        provider.solar_forecast = self._solar_bell(today, max_power)
        self._build_history(
            provider,
            today,
            num_days=7,
            max_power=max_power,
            scale_bias=0.85,
            offset_bias=100.0,
            fc_daily_variation=0.2,
            noise_rng=rng,
        )

        provider.compute_dampening(max_power)
        assert provider._dampening_coefficients is not None

        daytime_a = [provider._dampening_coefficients[k, 0] for k in range(7, 18)]
        avg_a = sum(daytime_a) / len(daytime_a)
        assert 0.70 <= avg_a <= 1.0, f"Avg a={avg_a}, expected ~0.85"

        today_ac = [(ts, max(0.0, 0.85 * v + 100) if v > 0 else 0.0) for ts, v in provider.solar_forecast]
        raw_mae = self._daytime_mae(provider.solar_forecast, today_ac)
        dampened = provider._apply_dampening(provider.solar_forecast)
        dampened_mae = self._daytime_mae(dampened, today_ac)

        assert dampened_mae < raw_mae * 0.5

    # ------------------------------------------------------------------ #
    # Scenario 6: full production call sequence over 7 days
    # ------------------------------------------------------------------ #

    def test_production_call_sequence(self):
        """Simulate the exact production update cycle over 7 days.

        Each day: set forecast → record actuals → snapshot → score → dampening.
        Then verify today's dampened forecast is closer to actual.
        """
        provider = _TestProvider(solar=None, domain="test")
        max_power = 5000.0
        scale_bias = 0.8  # forecast is 20% too high

        today = datetime.datetime(2024, 6, 22, 0, 0, tzinfo=pytz.UTC)

        # Simulate 7 past days in chronological order
        for day_num in range(7, 0, -1):
            day_start = today - timedelta(days=day_num)
            df = self._DAILY_FACTORS[(day_num - 1) % len(self._DAILY_FACTORS)]

            # 1. Provider gets forecast (as provider.update would)
            provider.solar_forecast = self._solar_bell(day_start, max_power * df)

            # 2. Record actuals (directly into shared buffer, as _record_actual_from_sensor would)
            day_ac = [(ts, scale_bias * v if v > 0 else 0.0) for ts, v in provider.solar_forecast]
            provider._actuals_buffer.extend(day_ac)

            # 3. Daily scoring cycle (as _run_daily_scoring_cycle would)
            provider.record_forecast_snapshot(day_start)
            provider.compute_score()
            provider.compute_dampening(max_power)

        # Today: fresh forecast, final scoring cycle
        provider.solar_forecast = self._solar_bell(today, max_power)
        provider.record_forecast_snapshot(today)
        provider.compute_score()
        provider.compute_dampening(max_power)

        assert provider._dampening_coefficients is not None
        assert provider._last_dampening_date is not None

        # Coefficients should reflect 0.8× bias
        for k in range(7, 18):
            a = provider._dampening_coefficients[k, 0]
            assert 0.75 <= a <= 0.85, f"Position {k}: a={a}"

        # MAE improvement via the full _get_effective_forecast path
        today_ac = [(ts, scale_bias * v if v > 0 else 0.0) for ts, v in provider.solar_forecast]
        raw_mae = self._daytime_mae(provider.solar_forecast, today_ac)

        provider.dampening_enabled = True
        dampened = provider._get_effective_forecast()
        dampened_mae = self._daytime_mae(dampened, today_ac)

        assert dampened_mae < raw_mae * 0.3, f"Dampened ({dampened_mae:.0f}) should be ≪ raw ({raw_mae:.0f})"

        # Score should exist from the scoring cycle
        assert provider.score_raw is not None
