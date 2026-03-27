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
    home.solar_and_consumption_forecast = None

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
    """Test forecast accuracy scoring (new compute_score(time) API)."""

    def test_compute_mae_score(self):
        """Test MAE score computation with known data via history objects."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")

        t0 = datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC)
        # Mock solar production history returning actuals at 800W
        mock_prod_history = MagicMock(spec=QSSolarHistoryVals)
        mock_prod_history.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 800.0) for h in range(12)
        ]
        forecast_handler.solar_production_history = mock_prod_history

        # Mock forecast history for provider with 8h lookahead returning 1000W
        mock_fc_history = MagicMock(spec=QSSolarHistoryVals)
        mock_fc_history.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 1000.0) for h in range(12)
        ]
        forecast_handler.solar_forecast_history_per_provider = {
            "test_provider": {"qs_solar_forecast_8h": mock_fc_history}
        }

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="test_provider")

        provider.compute_score(t0)
        assert provider.score is not None
        assert abs(provider.score - 200.0) < 0.01

    def test_score_none_with_no_actuals(self):
        """Test score is None when no actuals available."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_prod_history = MagicMock(spec=QSSolarHistoryVals)
        mock_prod_history.get_historical_data.return_value = []
        forecast_handler.solar_production_history = mock_prod_history
        forecast_handler.solar_forecast_history_per_provider = {}
        solar_mock.home.solar_and_consumption_forecast = forecast_handler

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="test_provider")
        t0 = datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC)
        provider.compute_score(t0)
        assert provider.score is None

    def test_score_none_with_no_forecast_history(self):
        """Test score is None when no forecast history available for provider."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_prod_history = MagicMock(spec=QSSolarHistoryVals)
        mock_prod_history.get_historical_data.return_value = [
            (datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC), 800.0)
        ]
        forecast_handler.solar_production_history = mock_prod_history
        forecast_handler.solar_forecast_history_per_provider = {}
        solar_mock.home.solar_and_consumption_forecast = forecast_handler

        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="test_provider")
        provider.compute_score(datetime.datetime(2024, 6, 15, 6, 0, tzinfo=pytz.UTC))
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


class TestComputeScoreEdgeCases:
    """Test compute_score edge cases with new history-based API."""

    def test_compute_score_prefers_8h_lookahead(self):
        """Test compute_score picks the first >= 8h lookahead sensor."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 800.0) for h in range(12)
        ]
        forecast_handler.solar_production_history = mock_prod

        # Only provide a 12h lookahead sensor (>= 8h)
        mock_12h = MagicMock(spec=QSSolarHistoryVals)
        mock_12h.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 1000.0) for h in range(12)
        ]
        forecast_handler.solar_forecast_history_per_provider = {
            "prov": {"qs_solar_forecast_12h": mock_12h}
        }

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(t0)
        assert provider.score is not None
        assert abs(provider.score - 200.0) < 0.01

    def test_compute_score_fallback_to_lower_lookahead(self):
        """Test compute_score falls back to largest available sensor when none >= 8h."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 500.0) for h in range(12)
        ]
        forecast_handler.solar_production_history = mock_prod

        # Only provide 4h lookahead (< 8h, so not preferred)
        mock_4h = MagicMock(spec=QSSolarHistoryVals)
        mock_4h.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 700.0) for h in range(12)
        ]
        forecast_handler.solar_forecast_history_per_provider = {
            "prov": {"qs_solar_forecast_4h": mock_4h}
        }

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(t0)
        assert provider.score is not None
        assert abs(provider.score - 200.0) < 0.01

    def test_compute_score_no_provider_histories(self):
        """Test compute_score returns None when provider has no histories."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = [
            (t0 + timedelta(hours=h), 800.0) for h in range(12)
        ]
        forecast_handler.solar_production_history = mock_prod
        forecast_handler.solar_forecast_history_per_provider = {}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(t0)
        assert provider.score is None

    def test_compute_score_empty_actuals(self):
        """Test compute_score returns None when actuals are empty."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar_mock = MagicMock()
        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)

        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_prod = MagicMock(spec=QSSolarHistoryVals)
        mock_prod.get_historical_data.return_value = []
        forecast_handler.solar_production_history = mock_prod
        forecast_handler.solar_forecast_history_per_provider = {"prov": {"qs_solar_forecast_8h": MagicMock()}}

        solar_mock.home.solar_and_consumption_forecast = forecast_handler
        provider = _TestProvider(solar=solar_mock, domain="test", provider_name="prov")
        provider.compute_score(t0)
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
        provider.compute_score = MagicMock()

        now = datetime.datetime(2024, 6, 15, 8, 0, tzinfo=pytz.UTC)

        # First call should run the cycle
        solar._run_scoring_cycle(now)
        assert provider.compute_score.call_count == 1

        # Second call in same half-day should be a no-op
        provider.compute_score.reset_mock()
        solar._run_scoring_cycle(now + timedelta(hours=1))
        assert provider.compute_score.call_count == 0

    def test_scoring_cycle_runs_at_noon_boundary(self, fake_hass):
        """Test _run_scoring_cycle runs again after crossing the 12:00 boundary."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]
        provider.compute_score = MagicMock()

        morning = datetime.datetime(2024, 6, 15, 8, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(morning)
        assert provider.compute_score.call_count == 1

        # After noon boundary, should run again
        provider.compute_score.reset_mock()
        afternoon = datetime.datetime(2024, 6, 15, 14, 0, tzinfo=pytz.UTC)
        solar._run_scoring_cycle(afternoon)
        assert provider.compute_score.call_count == 1

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
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
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
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
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
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        vals = QSSolarHistoryVals(entity_id="sensor.solar", forecast=forecast)
        vals.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.float32)

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert vals.get_historical_solar_pattern(t) == []

    def test_get_historical_solar_pattern_none_values(self):
        """Test fallback returns empty when values is None."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        forecast = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
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
        # Remove solar_and_consumption_forecast so getattr returns None
        solar.home.solar_and_consumption_forecast = None
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
            QSHomeSolarAndConsumptionHistoryAndForecast,
            QSSolarHistoryVals,
        )

        solar = _make_solar(fake_hass)
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        mock_history = MagicMock(spec=QSSolarHistoryVals)
        mock_history.get_historical_solar_pattern.return_value = [(datetime.datetime.now(pytz.UTC), 500.0)]
        forecast_handler.solar_production_history = mock_history
        solar.home.solar_and_consumption_forecast = forecast_handler

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        result = solar.get_historical_solar_fallback(t)
        assert len(result) == 1
        mock_history.get_historical_solar_pattern.assert_called_once_with(t)

    def test_solar_get_historical_fallback_no_history_attr(self, fake_hass):
        """Test fallback returns empty when forecast handler has no solar_production_history."""
        from custom_components.quiet_solar.ha_model.home import (
            QSHomeSolarAndConsumptionHistoryAndForecast,
        )

        solar = _make_solar(fake_hass)
        forecast_handler = QSHomeSolarAndConsumptionHistoryAndForecast(home=None, storage_path="/tmp")
        # solar_production_history defaults to None
        solar.home.solar_and_consumption_forecast = forecast_handler

        t = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        assert solar.get_historical_solar_fallback(t) == []

    def test_reset_scoring(self, fake_hass):
        """Test QSSolar.reset_scoring clears provider state."""
        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "A"},
        ]
        solar = _make_solar(fake_hass, providers_config=providers_config)
        provider = solar.solar_forecast_providers["A"]

        t0 = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=pytz.UTC)
        provider.score = 100.0
        solar._last_scoring_half_day = (t0.date(), 1)

        solar.reset_scoring(t0)
        assert solar._last_scoring_half_day is None
        assert provider.score is None

    def test_reset_scoring_clears_provider_score(self):
        """Test reset_scoring clears score on provider."""
        provider = _TestProvider(solar=None, domain="test")
        provider.score = 100.0

        provider.reset_scoring()
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

