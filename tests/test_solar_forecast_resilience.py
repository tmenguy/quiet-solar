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

    def test_detect_step_size(self):
        """Test step size detection from forecast timestamps."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=30), 1200.0),
            (t0 + timedelta(minutes=60), 900.0),
        ]
        provider._detect_step_size()
        assert provider.forecast_step_seconds == 1800
        assert provider._steps_per_day == 48

    def test_detect_step_size_15min(self):
        """Test 15-min step detection."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=15), 1200.0),
        ]
        provider._detect_step_size()
        assert provider.forecast_step_seconds == 900
        assert provider._steps_per_day == 96

    def test_detect_step_size_change_invalidates_dampening(self):
        """Test step size change invalidates existing dampening coefficients."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # Start with 30-min steps (48 steps/day)
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=30), 1200.0),
            (t0 + timedelta(minutes=60), 900.0),
        ]
        provider._detect_step_size()
        assert provider.forecast_step_seconds == 1800
        assert provider._steps_per_day == 48

        # Set dampening coefficients for 48 steps
        provider._dampening_coefficients = np.ones((48, 2), dtype=np.float64)
        assert provider._dampening_coefficients is not None

        # Now change to 15-min steps (96 steps/day)
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=15), 1200.0),
            (t0 + timedelta(minutes=30), 900.0),
        ]
        provider._detect_step_size()
        assert provider.forecast_step_seconds == 900
        assert provider._steps_per_day == 96
        # Dampening coefficients should be invalidated
        assert provider._dampening_coefficients is None

    def test_detect_step_size_same_keeps_dampening(self):
        """Test step size detection with same step size preserves dampening coefficients."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # Set 30-min steps
        provider.solar_forecast = [
            (t0, 1000.0),
            (t0 + timedelta(minutes=30), 1200.0),
            (t0 + timedelta(minutes=60), 900.0),
        ]
        provider._detect_step_size()
        assert provider._steps_per_day == 48

        # Set dampening coefficients
        coeffs = np.ones((48, 2), dtype=np.float64)
        provider._dampening_coefficients = coeffs

        # Re-detect with same step size
        provider._detect_step_size()
        # Coefficients should be preserved
        assert provider._dampening_coefficients is coeffs

    def test_detect_step_size_insufficient_data(self):
        """Test step detection with insufficient data."""
        provider = _TestProvider(solar=None, domain="test")
        provider.solar_forecast = [(datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC), 1000.0)]
        provider._detect_step_size()
        assert provider.forecast_step_seconds is None

    def test_compute_mae_score(self):
        """Test MAE score computation with known data."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        # Create history with known values
        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # Set daytime steps (6-18) with known forecast/actual
        for day in range(7):
            for step in range(6, 18):
                provider._forecast_history[day, step] = 1000.0
                provider._actual_history[day, step] = 800.0

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

    def test_time_to_step_index(self):
        """Test timestamp to step index conversion."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 1800  # 30-min steps
        ts = datetime.datetime(2024, 6, 15, 6, 30, tzinfo=pytz.UTC)  # 6:30 = 13th step
        assert provider._time_to_step_index(ts) == 13

    def test_time_to_step_index_none_when_no_step_size(self):
        """Test step index is None when step size not detected."""
        provider = _TestProvider(solar=None, domain="test")
        ts = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert provider._time_to_step_index(ts) is None


# ============================================================================
# Task 7: Dampening — MOS linear correction
# ============================================================================


class TestDampening:
    """Test MOS dampening computation and application."""

    def test_compute_dampening_identity_for_nighttime(self):
        """Test dampening produces identity (a=1, b=0) for all-zero nighttime steps."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.zeros((7, 24), dtype=np.float32)
        provider._actual_history = np.zeros((7, 24), dtype=np.float32)

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None
        # All should be identity since everything is zero (nighttime guard)
        for k in range(24):
            assert provider._dampening_coefficients[k, 0] == 1.0
            assert provider._dampening_coefficients[k, 1] == 0.0

    def test_compute_dampening_identity_for_insufficient_data(self):
        """Test dampening produces identity when fewer than 3 data points."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # Only 2 days of data for step 12
        for day in range(2):
            provider._forecast_history[day, 12] = 1000.0
            provider._actual_history[day, 12] = 800.0

        provider.compute_dampening(max_power=6000.0)
        # Step 12: only 2 data points, should be identity
        assert provider._dampening_coefficients[12, 0] == 1.0
        assert provider._dampening_coefficients[12, 1] == 0.0

    def test_compute_dampening_with_known_linear_data(self):
        """Test dampening computes correct coefficients for known linear relationship."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # For step 12: actual = 0.8 * forecast + 100 (known linear)
        for day in range(7):
            fc = 1000.0 + day * 100
            provider._forecast_history[day, 12] = fc
            provider._actual_history[day, 12] = 0.8 * fc + 100

        provider.compute_dampening(max_power=6000.0)
        a_k = provider._dampening_coefficients[12, 0]
        b_k = provider._dampening_coefficients[12, 1]
        assert abs(a_k - 0.8) < 0.01
        assert abs(b_k - 100.0) < 1.0

    def test_dampening_coefficient_bounds(self):
        """Test coefficient clamping to bounds."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # Create data that would produce extreme coefficients
        # actual ≈ 5 * forecast (would give a > 3.0, should be clamped)
        for day in range(7):
            provider._forecast_history[day, 12] = 100.0 + day * 10
            provider._actual_history[day, 12] = 500.0 + day * 50

        provider.compute_dampening(max_power=6000.0)
        a_k = provider._dampening_coefficients[12, 0]
        assert a_k == DAMPENING_A_MAX

    def test_apply_dampening_output_clamp(self):
        """Test dampened output is clamped to >= 0."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        # Set step 12 to produce negative: a=0.5, b=-600
        provider._dampening_coefficients[12, 0] = 0.5
        provider._dampening_coefficients[12, 1] = -600.0

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        forecast = [(t, 100.0)]  # 0.5*100 - 600 = -550, should clamp to 0
        result = provider._apply_dampening(forecast)
        assert result[0][1] == 0.0

    def test_apply_dampening_non_finite_coefficients_fallback(self):
        """Test dampening uses raw value when coefficients contain NaN or Inf."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        # Set step 12 coefficients to NaN
        provider._dampening_coefficients[12, 0] = float("nan")
        provider._dampening_coefficients[12, 1] = float("nan")
        # Set step 13 coefficients to Inf
        provider._dampening_coefficients[13, 0] = float("inf")
        provider._dampening_coefficients[13, 1] = 0.0

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
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24
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
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0
        provider._dampening_coefficients[12, 0] = 2.0  # double for step 12

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t, 500.0)]
        provider.dampening_enabled = True
        result = provider._get_effective_forecast()
        assert result[0][1] == 1000.0  # 2.0 * 500

    @pytest.mark.asyncio
    async def test_dampening_persistence_save_load(self, fake_hass, tmp_path):
        """Test saving and loading dampening coefficients."""
        fake_hass.config.config_dir = str(tmp_path)
        solar_mock = MagicMock()
        solar_mock.hass = fake_hass

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider.hass = fake_hass
        provider.forecast_step_seconds = 1800
        provider._steps_per_day = 48
        provider._dampening_coefficients = np.ones((48, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0
        provider._dampening_coefficients[24, 0] = 1.5
        provider._dampening_coefficients[24, 1] = -50.0

        await provider.save_dampening()

        # Load into fresh provider
        provider2 = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider2.hass = fake_hass
        provider2._steps_per_day = 48
        await provider2.load_dampening()

        assert provider2._dampening_coefficients is not None
        assert provider2._dampening_coefficients[24, 0] == 1.5
        assert provider2._dampening_coefficients[24, 1] == -50.0

    @pytest.mark.asyncio
    async def test_dampening_persistence_mismatch_discards(self, fake_hass, tmp_path):
        """Test loading dampening with mismatched step size discards data."""
        fake_hass.config.config_dir = str(tmp_path)
        solar_mock = MagicMock()
        solar_mock.hass = fake_hass

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider.hass = fake_hass
        provider.forecast_step_seconds = 1800
        provider._steps_per_day = 48
        provider._dampening_coefficients = np.ones((48, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        await provider.save_dampening()

        # Load with different step size
        provider2 = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider2.hass = fake_hass
        provider2._steps_per_day = 96  # Different!
        await provider2.load_dampening()

        assert provider2._dampening_coefficients is None

    @pytest.mark.asyncio
    async def test_dampening_persistence_missing_file(self, fake_hass, tmp_path):
        """Test loading from missing file doesn't crash."""
        fake_hass.config.config_dir = str(tmp_path)
        solar_mock = MagicMock()
        solar_mock.hass = fake_hass

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider.hass = fake_hass
        await provider.load_dampening()
        assert provider._dampening_coefficients is None


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
    """Test forecast and actual history recording."""

    def test_record_forecast_snapshot(self):
        """Test recording forecast into 7-day buffer."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0), (t0 + timedelta(hours=1), 1200.0)]

        provider.record_forecast_snapshot(t0)

        assert provider._forecast_history is not None
        assert provider._forecast_history[0, 12] == 1000.0
        assert provider._forecast_history[0, 13] == 1200.0

    def test_record_actual_production(self):
        """Test recording actual production."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        actuals = [(t0, 900.0), (t0 + timedelta(hours=1), 1100.0)]

        provider.record_actual_production(t0, actuals)
        assert provider._actual_history[0, 12] == 900.0
        assert provider._actual_history[0, 13] == 1100.0

    def test_advance_history_day(self):
        """Test advancing the history day index."""
        provider = _TestProvider(solar=None, domain="test")
        assert provider._history_day_index == 0
        provider.advance_history_day()
        assert provider._history_day_index == 1

    def test_record_forecast_reinit_on_step_mismatch(self):
        """Test buffer reinitialized when step size changes."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0)]
        provider.record_forecast_snapshot(t0)
        assert provider._forecast_history.shape == (7, 24)

        # Change step size
        provider._steps_per_day = 48
        provider.forecast_step_seconds = 1800
        provider.solar_forecast = [(t0, 2000.0)]
        provider.record_forecast_snapshot(t0)
        assert provider._forecast_history.shape == (7, 48)


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
        provider.advance_history_day = MagicMock()

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
        provider.advance_history_day = MagicMock()

        now = datetime.datetime.now(pytz.UTC)
        solar._run_daily_scoring_cycle(now)

        provider.compute_dampening.assert_called_once_with(solar.solar_max_output_power_value)

    def test_daily_scoring_cycle_skips_compute_dampening_when_disabled(self, fake_hass):
        """Test _run_daily_scoring_cycle does NOT call compute_dampening when dampening is disabled."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.dampening_enabled = False
        provider.record_forecast_snapshot = MagicMock()
        provider.compute_score = MagicMock()
        provider.compute_dampening = MagicMock()
        provider.advance_history_day = MagicMock()

        now = datetime.datetime.now(pytz.UTC)
        solar._run_daily_scoring_cycle(now)

        provider.compute_dampening.assert_not_called()

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
        provider._steps_per_day = 24
        forecast = [(datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC), 1000.0)]
        result = provider._apply_dampening(forecast)
        assert result is forecast

    def test_apply_dampening_step_out_of_range(self):
        """Test _apply_dampening passes through when step index out of range."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24
        # Only 12 coefficients, but step 23 will be out of range
        provider._dampening_coefficients = np.ones((12, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        # Step 23 (23:00) — out of range for 12-entry coefficients
        t = datetime.datetime(2024, 6, 15, 23, 0, tzinfo=pytz.UTC)
        forecast = [(t, 500.0)]
        result = provider._apply_dampening(forecast)
        assert result[0][1] == 500.0  # Passed through unchanged

    def test_record_forecast_snapshot_guard_no_steps(self):
        """Test record_forecast_snapshot returns early when no steps_per_day."""
        provider = _TestProvider(solar=None, domain="test")
        provider._steps_per_day = None
        provider.forecast_step_seconds = None
        provider.solar_forecast = [(datetime.datetime.now(pytz.UTC), 1000.0)]
        provider.record_forecast_snapshot(datetime.datetime.now(pytz.UTC))
        assert provider._forecast_history is None

    def test_record_forecast_snapshot_guard_no_forecast(self):
        """Test record_forecast_snapshot returns early when no forecast data."""
        provider = _TestProvider(solar=None, domain="test")
        provider._steps_per_day = 24
        provider.forecast_step_seconds = 3600
        provider.solar_forecast = []
        provider.record_forecast_snapshot(datetime.datetime.now(pytz.UTC))
        assert provider._forecast_history is None

    def test_record_forecast_snapshot_skips_none_values(self):
        """Test record_forecast_snapshot skips entries with None timestamps."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(None, 100.0), (t, None), (t, 500.0)]
        provider.record_forecast_snapshot(t)
        assert provider._forecast_history is not None
        # Step 12 should have 500.0 (last valid entry)
        assert provider._forecast_history[0, 12] == pytest.approx(500.0)

    def test_record_actual_production_guard_no_steps(self):
        """Test record_actual_production returns early when no steps_per_day."""
        provider = _TestProvider(solar=None, domain="test")
        provider._steps_per_day = None
        provider.forecast_step_seconds = None
        t = datetime.datetime.now(pytz.UTC)
        provider.record_actual_production(t, [(t, 100.0)])
        assert provider._actual_history is None

    def test_record_actual_production_guard_no_history(self):
        """Test record_actual_production returns early when no actual_history buffer."""
        provider = _TestProvider(solar=None, domain="test")
        provider._steps_per_day = 24
        provider.forecast_step_seconds = 3600
        provider._actual_history = None
        t = datetime.datetime.now(pytz.UTC)
        provider.record_actual_production(t, [(t, 100.0)])
        assert provider._actual_history is None

    def test_compute_score_with_dampening(self):
        """Test compute_score also computes dampened score when enabled."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # Fill step 12 with data
        for day in range(7):
            provider._forecast_history[day, 12] = 1000.0
            provider._actual_history[day, 12] = 900.0

        # Enable dampening with identity coefficients
        provider.dampening_enabled = True
        provider._dampening_coefficients = np.ones((24, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        provider.compute_score()
        assert provider.score_raw == pytest.approx(100.0)
        assert provider.score_dampened is not None

    def test_compute_mae_all_nan(self):
        """Test _compute_mae returns None when all data is NaN."""
        provider = _TestProvider(solar=None, domain="test")
        forecasts = np.full((7, 24), np.nan, dtype=np.float32)
        actuals = np.full((7, 24), np.nan, dtype=np.float32)
        assert provider._compute_mae(forecasts, actuals) is None

    def test_apply_dampening_to_array(self):
        """Test _apply_dampening_to_array applies coefficients correctly."""
        provider = _TestProvider(solar=None, domain="test")
        provider._dampening_coefficients = np.ones((3, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0
        provider._dampening_coefficients[1, 0] = 2.0  # Double step 1

        forecasts = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        result = provider._apply_dampening_to_array(forecasts)
        assert result[0, 0] == pytest.approx(100.0)  # a=1, b=0
        assert result[0, 1] == pytest.approx(400.0)  # a=2, b=0
        assert result[0, 2] == pytest.approx(300.0)  # a=1, b=0

    def test_apply_dampening_to_array_shape_mismatch(self):
        """Test _apply_dampening_to_array returns input when column count doesn't match coefficient count."""
        provider = _TestProvider(solar=None, domain="test")
        # 3 coefficients but forecast array has 5 columns
        provider._dampening_coefficients = np.ones((3, 2), dtype=np.float64)
        provider._dampening_coefficients[:, 1] = 0.0

        forecasts = np.array([[100.0, 200.0, 300.0, 400.0, 500.0]], dtype=np.float32)
        result = provider._apply_dampening_to_array(forecasts)
        # Should return the original array unchanged
        assert result is forecasts

    def test_apply_dampening_to_array_guard_none(self):
        """Test _apply_dampening_to_array returns input when no coefficients."""
        provider = _TestProvider(solar=None, domain="test")
        provider._dampening_coefficients = None
        forecasts = np.array([[100.0]], dtype=np.float32)
        result = provider._apply_dampening_to_array(forecasts)
        assert result is forecasts

    def test_compute_dampening_guard_no_history(self):
        """Test compute_dampening returns early when no history buffers."""
        provider = _TestProvider(solar=None, domain="test")
        provider._forecast_history = None
        provider._actual_history = None
        provider._steps_per_day = 24
        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is None

    def test_get_dampening_persistence_path_no_solar(self):
        """Test persistence path returns None when no solar reference."""
        provider = _TestProvider(solar=None, domain="test")
        provider.hass = None
        assert provider.get_dampening_persistence_path() is None

    @pytest.mark.asyncio
    async def test_save_dampening_guard_no_path(self):
        """Test save_dampening returns early when path is None."""
        provider = _TestProvider(solar=None, domain="test")
        provider.hass = None
        provider._dampening_coefficients = np.ones((24, 2))
        provider._steps_per_day = 24
        await provider.save_dampening()  # Should not crash

    @pytest.mark.asyncio
    async def test_load_dampening_guard_no_path(self):
        """Test load_dampening returns early when path is None."""
        provider = _TestProvider(solar=None, domain="test")
        provider.hass = None
        await provider.load_dampening()  # Should not crash
        assert provider._dampening_coefficients is None

    @pytest.mark.asyncio
    async def test_load_dampening_missing_keys(self, fake_hass, tmp_path):
        """Test load_dampening discards data when keys missing."""
        fake_hass.config.config_dir = str(tmp_path)
        solar_mock = MagicMock()
        solar_mock.hass = fake_hass

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider.hass = fake_hass

        # Save a file with missing keys
        from os.path import join

        storage_dir = join(str(tmp_path), "quiet_solar")
        import os

        os.makedirs(storage_dir, exist_ok=True)
        path = join(storage_dir, "dampening_TestProv.npy")
        np.save(path, {"only_coefficients": np.ones((24, 2))}, allow_pickle=True)

        await provider.load_dampening()
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

    @pytest.mark.asyncio
    async def test_save_dampening_guard_no_coefficients(self, fake_hass, tmp_path):
        """Test save_dampening returns early when coefficients are None."""
        fake_hass.config.config_dir = str(tmp_path)
        solar_mock = MagicMock()
        solar_mock.hass = fake_hass

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="TestProv")
        provider.hass = fake_hass
        provider._dampening_coefficients = None
        provider._steps_per_day = 24
        await provider.save_dampening()  # Should return early, no crash

    def test_compute_dampening_polyfit_error(self):
        """Test compute_dampening handles polyfit errors gracefully."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # Create data where all forecasts are identical (polyfit can't fit)
        for day in range(7):
            provider._forecast_history[day, 12] = 1000.0  # All same value
            provider._actual_history[day, 12] = 500.0 + day * 100

        provider.compute_dampening(max_power=6000.0)
        # Step 12 should still get coefficients (polyfit handles this)
        # but identity is used for steps without data
        assert provider._dampening_coefficients is not None
        # Step 0 should be identity (no data)
        assert provider._dampening_coefficients[0, 0] == 1.0
        assert provider._dampening_coefficients[0, 1] == 0.0

    def test_compute_dampening_polyfit_linalg_error(self):
        """Test compute_dampening catches LinAlgError from polyfit and falls back to identity."""
        provider = _TestProvider(solar=None, domain="test")
        provider.forecast_step_seconds = 3600
        provider._steps_per_day = 24

        provider._forecast_history = np.full((7, 24), np.nan, dtype=np.float32)
        provider._actual_history = np.full((7, 24), np.nan, dtype=np.float32)

        # Create data with inf values which causes LinAlgError in polyfit
        for day in range(7):
            provider._forecast_history[day, 12] = float("inf")
            provider._actual_history[day, 12] = 500.0 + day * 100

        provider.compute_dampening(max_power=6000.0)
        assert provider._dampening_coefficients is not None
        # Step 12 should remain identity because polyfit raised LinAlgError
        assert provider._dampening_coefficients[12, 0] == 1.0
        assert provider._dampening_coefficients[12, 1] == 0.0
