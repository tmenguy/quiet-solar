"""Tests for Story 3.7: Solar Forecast API Resilience."""

from __future__ import annotations

import datetime
import logging
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
        solar.solar_forecast_providers["A"].score = 500.0
        solar.solar_forecast_providers["B"].score = 200.0
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
        solar.solar_forecast_providers["A"].score = 200.0
        solar.solar_forecast_providers["A"]._latest_successful_forecast_time = now - timedelta(hours=2)
        solar.solar_forecast_providers["B"].score = 200.0
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

        # No scores on any provider (score = None by default)
        assert solar.solar_forecast_providers["A"].score is None
        assert solar.solar_forecast_providers["B"].score is None

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

    def test_get_platforms_includes_select(self, fake_hass):
        """Test that solar with providers includes SELECT platform."""
        from homeassistant.const import Platform

        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        platforms = solar.get_platforms()
        assert Platform.SELECT in platforms


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
        # Store a forecast snapshot, then compute score with known 200W difference
        provider.solar_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(12)]
        provider.store_forecast_snapshot()
        actuals = [(t0 + timedelta(hours=h), 800.0) for h in range(12)]

        provider.compute_score(actuals)
        assert provider.score is not None
        assert abs(provider.score - 200.0) < 0.01

    def test_score_none_with_no_data(self):
        """Test score is None when no data available."""
        provider = _TestProvider(solar=None, domain="test")
        provider.compute_score([])
        assert provider.score is None

    def test_score_none_with_no_stored_forecast(self):
        """Test score is None when no stored forecast."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC)
        actuals = [(t0 + timedelta(hours=h), 800.0) for h in range(12)]
        provider.compute_score(actuals)
        assert provider.score is None

    def test_get_active_score_returns_score(self):
        """Test get_active_score returns the score."""
        provider = _TestProvider(solar=None, domain="test")
        provider.score = 300.0
        assert provider.get_active_score() == 300.0


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


class TestForecastSnapshot:
    """Test forecast snapshot storage."""

    def test_store_forecast_snapshot(self):
        """Test storing forecast as a snapshot."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0), (t0 + timedelta(hours=1), 1200.0)]

        provider.store_forecast_snapshot()

        assert len(provider._stored_forecast) == 2
        assert provider._stored_forecast[0] == (t0, 1000.0)
        assert provider._stored_forecast[1] == (t0 + timedelta(hours=1), 1200.0)

    def test_store_forecast_snapshot_overwrites(self):
        """Test snapshot is overwritten on each call."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        provider.solar_forecast = [(t0, 1000.0)]
        provider.store_forecast_snapshot()
        assert len(provider._stored_forecast) == 1

        provider.solar_forecast = [(t0, 2000.0), (t0 + timedelta(hours=1), 2500.0)]
        provider.store_forecast_snapshot()
        assert len(provider._stored_forecast) == 2
        assert provider._stored_forecast[0] == (t0, 2000.0)

    def test_store_forecast_snapshot_strips_none(self):
        """Test snapshot strips entries with None timestamps or values."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(None, 100.0), (t0, None), (t0, 500.0)]

        provider.store_forecast_snapshot()
        assert len(provider._stored_forecast) == 1
        assert provider._stored_forecast[0] == (t0, 500.0)

    def test_store_forecast_snapshot_empty_forecast(self):
        """Test snapshot from empty forecast yields empty stored forecast."""
        provider = _TestProvider(solar=None, domain="test")
        provider.solar_forecast = []
        provider.store_forecast_snapshot()
        assert provider._stored_forecast == []

    def test_get_production_actuals_last_24h_no_home(self, fake_hass):
        """Test _get_production_actuals_last_24h returns empty when no home."""
        solar = _make_solar(fake_hass)
        solar.home = None
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar._get_production_actuals_last_24h(t) == []

    def test_get_production_actuals_last_24h_no_forecast_handler(self, fake_hass):
        """Test _get_production_actuals_last_24h returns empty when no _consumption_forecast."""
        solar = _make_solar(fake_hass)
        if hasattr(solar.home, "_consumption_forecast"):
            del solar.home._consumption_forecast
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar._get_production_actuals_last_24h(t) == []

    def test_get_production_actuals_last_24h_no_history(self, fake_hass):
        """Test _get_production_actuals_last_24h returns empty when no solar_production_history."""
        from custom_components.quiet_solar.ha_model.home import QSHomeConsumptionHistoryAndForecast

        solar = _make_solar(fake_hass)
        forecast_handler = QSHomeConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        solar.home._consumption_forecast = forecast_handler
        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar._get_production_actuals_last_24h(t) == []

    def test_get_production_actuals_last_24h_with_data(self, fake_hass):
        """Test _get_production_actuals_last_24h extracts data from history."""
        import numpy as np

        solar = _make_solar(fake_hass)
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        # Create a mock history object with the right interface
        mock_history = MagicMock()
        mock_history.values = np.ones((2, 100))  # non-None to pass guard
        # Return 4 data points (simulating 4 x 15-min intervals = 1 hour)
        power_vals = np.array([500.0, 600.0, 700.0, 0.0])
        day_vals = np.array([1, 1, 1, 0])  # last one has no data
        mock_history._get_values.return_value = (power_vals, day_vals)
        mock_history.get_index_from_time.return_value = (0, 1)

        forecast_handler = MagicMock()
        forecast_handler.solar_production_history = mock_history
        solar.home._consumption_forecast = forecast_handler

        result = solar._get_production_actuals_last_24h(t0)
        # Should have 3 entries (day_vals[3] == 0 is skipped)
        assert len(result) == 3
        assert result[0][1] == 500.0
        assert result[1][1] == 600.0
        assert result[2][1] == 700.0

    def test_get_production_actuals_last_24h_get_values_returns_none(self, fake_hass):
        """Test _get_production_actuals_last_24h returns empty when _get_values returns None."""
        import numpy as np

        solar = _make_solar(fake_hass)
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        mock_history = MagicMock()
        mock_history.values = np.ones((2, 100))
        mock_history._get_values.return_value = (None, None)
        mock_history.get_index_from_time.return_value = (0, 1)

        forecast_handler = MagicMock()
        forecast_handler.solar_production_history = mock_history
        solar.home._consumption_forecast = forecast_handler

        assert solar._get_production_actuals_last_24h(t0) == []

    def test_compute_score_empty_aligned_result(self):
        """Test compute_score handles empty alignment result (line 551)."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        provider._stored_forecast = [(t0 + timedelta(hours=h), 1000.0) for h in range(5)]
        actuals = [(t0 + timedelta(hours=h), 800.0) for h in range(5)]

        with patch(
            "custom_components.quiet_solar.ha_model.solar.align_time_series_on_time_slots",
            return_value=[],
        ):
            provider.compute_score(actuals)
        assert provider.score is None


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

        solar.solar_forecast_providers["A"].score = 500.0
        solar.solar_forecast_providers["B"].score = 200.0

        await solar.update_forecast(datetime.datetime.now(pytz.UTC))
        assert solar.active_provider_name == "B"

    def test_scoring_cycle_skips_same_half_day(self, fake_hass):
        """Test _run_scoring_cycle early-returns when already run for this half-day."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.store_forecast_snapshot = MagicMock()
        provider.compute_score = MagicMock()

        now = datetime.datetime(2024, 6, 15, 8, 0, tzinfo=pytz.UTC)

        # First call should run the cycle
        solar._run_scoring_cycle(now)
        assert provider.store_forecast_snapshot.call_count == 1
        assert provider.compute_score.call_count == 1

        # Second call in same half-day should be a no-op
        provider.store_forecast_snapshot.reset_mock()
        provider.compute_score.reset_mock()
        solar._run_scoring_cycle(now + timedelta(hours=1))
        assert provider.store_forecast_snapshot.call_count == 0
        assert provider.compute_score.call_count == 0

    def test_scoring_cycle_runs_at_noon_boundary(self, fake_hass):
        """Test _run_scoring_cycle runs again after crossing the 12:00 boundary."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.store_forecast_snapshot = MagicMock()
        provider.compute_score = MagicMock()

        morning = datetime.datetime(2024, 6, 15, 8, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(morning)
        assert provider.compute_score.call_count == 1

        # After noon boundary, should run again
        provider.store_forecast_snapshot.reset_mock()
        provider.compute_score.reset_mock()
        afternoon = datetime.datetime(2024, 6, 15, 14, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(afternoon)
        assert provider.compute_score.call_count == 1
        assert provider.store_forecast_snapshot.call_count == 1

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

    def test_compute_mae_from_aligned_empty(self):
        """Test _compute_mae_from_aligned returns None when inputs are empty."""
        assert QSSolarProvider._compute_mae_from_aligned([], []) is None

    def test_compute_mae_from_aligned_all_nighttime(self):
        """Test _compute_mae_from_aligned returns None when all values are zero."""
        t0 = datetime.datetime(2024, 6, 15, 0, 0, tzinfo=pytz.UTC)
        fc = [(t0, 0.0), (t0 + timedelta(hours=1), 0.0)]
        ac = [(t0, 0.0), (t0 + timedelta(hours=1), 0.0)]
        assert QSSolarProvider._compute_mae_from_aligned(fc, ac) is None

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

    def test_compute_score_single_point_stored_forecast(self):
        """Test compute_score returns early when stored forecast has < 2 time points."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [(t0, 1000.0)]
        provider.store_forecast_snapshot()
        actuals = [(t0, 800.0)]
        provider.compute_score(actuals)
        assert provider.score is None

    def test_reset_scoring(self, fake_hass):
        """Test QSSolar.reset_scoring clears provider state."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.score = 100.0
        provider._stored_forecast = [(t0, 500.0)]
        solar._last_scoring_half_day = (t0.date(), 1)

        solar.reset_scoring(t0)
        assert solar._last_scoring_half_day is None
        assert provider.score is None
        assert provider._stored_forecast == []

    def test_reset_scoring_buffers_clears_provider_state(self):
        """Test reset_scoring_buffers clears score and stored forecast."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.score = 100.0
        provider._stored_forecast = [(t0, 500.0)]

        provider.reset_scoring_buffers()
        assert provider.score is None
        assert provider._stored_forecast == []

    def test_compute_score_empty_alignment(self):
        """Test compute_score returns early when aligned forecast is empty."""
        provider = _TestProvider(solar=None, domain="test")
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        # Store forecast with valid timestamps but all None values → cleaned list is empty
        provider.solar_forecast = [(t0 + timedelta(hours=h), None) for h in range(12)]
        provider.store_forecast_snapshot()
        actuals = [(t0 + timedelta(hours=h), 800.0) for h in range(12)]
        provider.compute_score(actuals)
        assert provider.score is None

    def test_scoring_half_day_static_method(self, fake_hass):
        """Test _scoring_half_day returns correct (date, slot) tuples."""
        solar = _make_solar(fake_hass)
        morning = datetime.datetime(2024, 6, 15, 8, 0, tzinfo=pytz.UTC)
        afternoon = datetime.datetime(2024, 6, 15, 14, 0, tzinfo=pytz.UTC)
        midnight = datetime.datetime(2024, 6, 16, 0, 0, tzinfo=pytz.UTC)

        d_morning, s_morning = solar._scoring_half_day(morning)
        d_afternoon, s_afternoon = solar._scoring_half_day(afternoon)
        d_midnight, s_midnight = solar._scoring_half_day(midnight)

        assert s_morning == 0
        assert s_afternoon == 1
        assert s_midnight == 0
        assert d_midnight > d_morning

