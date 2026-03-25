"""Tests for the Forecast.Solar provider (Story 3.13)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytz

from custom_components.quiet_solar.const import (
    CONF_SOLAR_FORECAST_PROVIDER,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    FORECAST_SOLAR_DOMAIN,
    OPEN_METEO_SOLAR_DOMAIN,
    SOLCAST_SOLAR_DOMAIN,
)
from custom_components.quiet_solar.ha_model.solar import (
    QSSolarProviderForecastSolar,
    _create_provider_for_domain,
    _migrate_solar_providers_config,
)
from tests.conftest import FakeConfigEntry, FakeHass


def _make_solar_mock(hass: FakeHass | None = None) -> MagicMock:
    """Create a mock QSSolar with a FakeHass."""
    solar = MagicMock()
    solar.hass = hass or FakeHass()
    return solar


def _make_coordinator(watts: dict[datetime, int] | None = None) -> MagicMock:
    """Create a mock coordinator whose .data.watts returns the given dict."""
    coordinator = MagicMock()
    coordinator.data.watts = watts
    return coordinator


def _register_forecast_solar_entry(
    hass: FakeHass, coordinator: MagicMock, entry_id: str = "fs_entry_1"
) -> FakeConfigEntry:
    """Register a FakeConfigEntry for forecast_solar with runtime_data set."""
    entry = FakeConfigEntry(entry_id=entry_id)
    entry.runtime_data = coordinator
    if FORECAST_SOLAR_DOMAIN not in hass.data:
        hass.data[FORECAST_SOLAR_DOMAIN] = {}
    hass.data[FORECAST_SOLAR_DOMAIN][entry_id] = entry
    return entry


# ============================================================================
# Provider creation
# ============================================================================


class TestForecastSolarProviderCreation:
    """Tests for provider instantiation."""

    def test_provider_domain(self):
        """Test provider has correct domain."""
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        assert provider.domain == FORECAST_SOLAR_DOMAIN

    def test_provider_name(self):
        """Test provider stores the given name."""
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        assert provider.provider_name == "Forecast.Solar"

    def test_provider_empty_name(self):
        """Test provider with empty name."""
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar)
        assert provider.provider_name == ""


# ============================================================================
# Orchestrator discovery (fill_orchestrators)
# ============================================================================


class TestForecastSolarOrchestratorDiscovery:
    """Tests for fill_orchestrators using config_entries pattern."""

    async def test_fill_finds_entry_runtime_data(self):
        """Test orchestrator discovered via entry.runtime_data (direct)."""
        hass = FakeHass()
        coordinator = _make_coordinator()
        _register_forecast_solar_entry(hass, coordinator)

        solar = _make_solar_mock(hass)
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        await provider.fill_orchestrators()

        assert len(provider.orchestrators) == 1
        assert provider.orchestrators[0] is coordinator

    async def test_fill_multiple_entries(self):
        """Test multiple forecast_solar config entries yield multiple orchestrators."""
        hass = FakeHass()
        coord1 = _make_coordinator()
        coord2 = _make_coordinator()
        _register_forecast_solar_entry(hass, coord1, entry_id="fs_1")
        _register_forecast_solar_entry(hass, coord2, entry_id="fs_2")

        solar = _make_solar_mock(hass)
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        await provider.fill_orchestrators()

        assert len(provider.orchestrators) == 2

    async def test_fill_no_entries(self):
        """Test no orchestrators when no config entries exist."""
        hass = FakeHass()
        solar = _make_solar_mock(hass)
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        await provider.fill_orchestrators()

        assert len(provider.orchestrators) == 0

    async def test_fill_skips_entry_without_runtime_data(self):
        """Test entries without runtime_data attribute are skipped."""
        hass = FakeHass()
        entry = FakeConfigEntry(entry_id="broken_entry")
        # Do NOT set entry.runtime_data — it should trigger AttributeError
        hass.data[FORECAST_SOLAR_DOMAIN] = {"broken_entry": entry}

        solar = _make_solar_mock(hass)
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        await provider.fill_orchestrators()

        assert len(provider.orchestrators) == 0

    async def test_fill_health_tracking(self):
        """Test orchestrator health is initialized to True."""
        hass = FakeHass()
        coordinator = _make_coordinator()
        _register_forecast_solar_entry(hass, coordinator)

        solar = _make_solar_mock(hass)
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")
        await provider.fill_orchestrators()

        assert provider._orchestrator_health[id(coordinator)] is True


# ============================================================================
# Data extraction (get_power_series_from_orchestrator)
# ============================================================================


class TestForecastSolarDataExtraction:
    """Tests for get_power_series_from_orchestrator."""

    async def test_extract_watts_dict(self):
        """Test extraction from Estimate.watts dict[datetime, int]."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 11, 0, tzinfo=utc)
        t3 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        watts = {t1: 500, t2: 1200, t3: 800}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, t1, t3)

        assert len(result) == 3
        # Values should be float
        assert all(isinstance(v, float) for _, v in result)
        assert result[0] == (t1, 500.0)
        assert result[1] == (t2, 1200.0)
        assert result[2] == (t3, 800.0)

    async def test_extract_returns_utc(self):
        """Test that returned datetimes are in UTC."""
        eastern = pytz.timezone("US/Eastern")
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=eastern)
        t2 = datetime(2026, 3, 24, 12, 0, tzinfo=eastern)
        watts = {t1: 100, t2: 200}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, t1, t2)

        for dt, _ in result:
            assert dt.tzinfo == pytz.UTC

    async def test_extract_empty_dict(self):
        """Test extraction from empty watts dict."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        coordinator = _make_coordinator({})
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, t1, t2)

        assert result == []

    async def test_extract_none_coordinator_data(self):
        """Test extraction when orchestrator.data is None."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        coordinator = MagicMock()
        coordinator.data = None
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, t1, t2)

        assert result == []

    async def test_extract_none_data(self):
        """Test extraction when watts is None."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        coordinator = _make_coordinator(None)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, t1, t2)

        assert result == []

    async def test_extract_none_start_time(self):
        """Test extraction returns empty when start_time is None."""
        coordinator = _make_coordinator({})
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, None, datetime.now(tz=pytz.UTC))

        assert result == []

    async def test_extract_none_end_time(self):
        """Test extraction returns empty when end_time is None."""
        coordinator = _make_coordinator({})
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, datetime.now(tz=pytz.UTC), None)

        assert result == []

    async def test_extract_time_window_subset(self):
        """Test extraction returns only data within the time window."""
        utc = pytz.UTC
        t0 = datetime(2026, 3, 24, 9, 0, tzinfo=utc)
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 11, 0, tzinfo=utc)
        t3 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        t4 = datetime(2026, 3, 24, 13, 0, tzinfo=utc)
        watts = {t0: 100, t1: 500, t2: 1200, t3: 800, t4: 300}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        # Query window from t1 to t3
        result = await provider.get_power_series_from_orchestrator(coordinator, t1, t3)

        values = [v for _, v in result]
        assert 500.0 in values
        assert 1200.0 in values
        assert 800.0 in values
        # Out-of-range values must be excluded
        assert 100.0 not in values
        assert 300.0 not in values

    async def test_extract_start_beyond_data(self):
        """Test extraction returns empty when start is beyond all data."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        watts = {t1: 500}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        far_future = datetime(2026, 12, 31, 0, 0, tzinfo=utc)
        result = await provider.get_power_series_from_orchestrator(
            coordinator, far_future, far_future + timedelta(hours=1)
        )

        assert result == []

    async def test_extract_start_between_data_points(self):
        """Test start_time falls between data points — includes preceding point."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 11, 0, tzinfo=utc)
        t3 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        watts = {t1: 500, t2: 1200, t3: 800}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        # Start at 10:30 — between t1 and t2, should include t1
        mid = datetime(2026, 3, 24, 10, 30, tzinfo=utc)
        result = await provider.get_power_series_from_orchestrator(coordinator, mid, t3)

        values = [v for _, v in result]
        assert 500.0 in values  # t1 included (preceding point)
        assert 1200.0 in values
        assert 800.0 in values

    async def test_extract_end_beyond_last_data_point(self):
        """Test end_time beyond last data point — clamps to last index."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 11, 0, tzinfo=utc)
        watts = {t1: 500, t2: 1200}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        far_end = datetime(2026, 3, 24, 23, 0, tzinfo=utc)
        result = await provider.get_power_series_from_orchestrator(coordinator, t1, far_end)

        assert len(result) == 2
        assert result[0] == (t1, 500.0)
        assert result[1] == (t2, 1200.0)

    async def test_extract_end_before_start_in_data(self):
        """Test extraction returns empty when end_time is before first data point."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 11, 0, tzinfo=utc)
        watts = {t1: 500, t2: 1200}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        early = datetime(2026, 3, 24, 8, 0, tzinfo=utc)
        result = await provider.get_power_series_from_orchestrator(
            coordinator, early, datetime(2026, 3, 24, 9, 0, tzinfo=utc)
        )

        assert result == []

    async def test_extract_unsorted_dict_becomes_sorted(self):
        """Test that an unordered watts dict is sorted by time."""
        utc = pytz.UTC
        t1 = datetime(2026, 3, 24, 12, 0, tzinfo=utc)
        t2 = datetime(2026, 3, 24, 10, 0, tzinfo=utc)
        t3 = datetime(2026, 3, 24, 11, 0, tzinfo=utc)
        watts = {t1: 800, t2: 500, t3: 1200}

        coordinator = _make_coordinator(watts)
        solar = _make_solar_mock()
        provider = QSSolarProviderForecastSolar(solar=solar, provider_name="Forecast.Solar")

        result = await provider.get_power_series_from_orchestrator(coordinator, t2, t1)

        # Should be sorted by time
        times = [dt for dt, _ in result]
        assert times == sorted(times)


# ============================================================================
# Factory registration (_create_provider_for_domain)
# ============================================================================


class TestFactoryRegistration:
    """Tests for Forecast.Solar in the provider factory."""

    def test_factory_creates_forecast_solar(self):
        """Test _create_provider_for_domain returns ForecastSolar provider."""
        solar = _make_solar_mock()
        provider = _create_provider_for_domain(FORECAST_SOLAR_DOMAIN, solar, "Forecast.Solar")
        assert isinstance(provider, QSSolarProviderForecastSolar)
        assert provider.provider_name == "Forecast.Solar"

    def test_factory_still_creates_solcast(self):
        """Test factory still creates Solcast provider (no regression)."""
        from custom_components.quiet_solar.ha_model.solar import QSSolarProviderSolcast

        solar = _make_solar_mock()
        provider = _create_provider_for_domain(SOLCAST_SOLAR_DOMAIN, solar, "Solcast")
        assert isinstance(provider, QSSolarProviderSolcast)

    def test_factory_still_creates_open_meteo(self):
        """Test factory still creates Open-Meteo provider (no regression)."""
        from custom_components.quiet_solar.ha_model.solar import QSSolarProviderOpenWeather

        solar = _make_solar_mock()
        provider = _create_provider_for_domain(OPEN_METEO_SOLAR_DOMAIN, solar, "Open-Meteo")
        assert isinstance(provider, QSSolarProviderOpenWeather)

    def test_factory_unknown_domain_returns_none(self):
        """Test factory returns None for unknown domain."""
        solar = _make_solar_mock()
        provider = _create_provider_for_domain("unknown_domain", solar, "Unknown")
        assert provider is None


# ============================================================================
# Migration function
# ============================================================================


class TestMigration:
    """Tests for _migrate_solar_providers_config with Forecast.Solar."""

    def test_migrate_old_forecast_solar_single_provider(self):
        """Test migration of old single-provider config for forecast_solar."""
        config = {CONF_SOLAR_FORECAST_PROVIDER: FORECAST_SOLAR_DOMAIN}
        result = _migrate_solar_providers_config(config)
        assert len(result) == 1
        assert result[0][CONF_SOLAR_PROVIDER_DOMAIN] == FORECAST_SOLAR_DOMAIN
        assert result[0][CONF_SOLAR_PROVIDER_NAME] == "Forecast.Solar"

    def test_migrate_old_solcast_still_works(self):
        """Test migration of old Solcast config still works."""
        config = {CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN}
        result = _migrate_solar_providers_config(config)
        assert result[0][CONF_SOLAR_PROVIDER_NAME] == "Solcast"

    def test_migrate_new_format_passthrough(self):
        """Test new multi-provider format passes through unchanged."""
        providers = [{CONF_SOLAR_PROVIDER_DOMAIN: FORECAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Forecast.Solar"}]
        config = {CONF_SOLAR_FORECAST_PROVIDERS: providers}
        result = _migrate_solar_providers_config(config)
        assert result is providers

    def test_migrate_unknown_domain_uses_raw_value(self):
        """Test migration of unknown domain uses the raw domain string as label."""
        config = {CONF_SOLAR_FORECAST_PROVIDER: "some_unknown_domain"}
        result = _migrate_solar_providers_config(config)
        assert len(result) == 1
        assert result[0][CONF_SOLAR_PROVIDER_NAME] == "some_unknown_domain"


# ============================================================================
# Multi-provider integration (AC3 — scoring, dampening, auto-selection)
# ============================================================================


class TestForecastSolarMultiProviderIntegration:
    """Integration tests verifying Forecast.Solar participates in Story 3.7 features."""

    def _make_multi_provider_solar(self, hass: FakeHass) -> MagicMock:
        """Create a QSSolar with Forecast.Solar + Solcast providers via factory."""
        from custom_components.quiet_solar.ha_model.solar import QSSolar

        providers_config = [
            {CONF_SOLAR_PROVIDER_DOMAIN: FORECAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Forecast.Solar"},
            {CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_PROVIDER_NAME: "Solcast"},
        ]
        config = {
            CONF_SOLAR_FORECAST_PROVIDERS: providers_config,
            "name": "test_solar",
        }
        solar = QSSolar.__new__(QSSolar)
        solar.hass = hass
        solar._name = "test_solar"
        solar.solar_forecast_providers = {}
        for p_cfg in providers_config:
            domain = p_cfg[CONF_SOLAR_PROVIDER_DOMAIN]
            name = p_cfg[CONF_SOLAR_PROVIDER_NAME]
            provider = _create_provider_for_domain(domain, solar, name)
            if provider is not None:
                solar.solar_forecast_providers[name] = provider
        return solar

    def test_forecast_solar_alongside_solcast(self):
        """Test Forecast.Solar coexists with Solcast in multi-provider dict."""
        hass = FakeHass()
        solar = self._make_multi_provider_solar(hass)

        assert len(solar.solar_forecast_providers) == 2
        assert "Forecast.Solar" in solar.solar_forecast_providers
        assert "Solcast" in solar.solar_forecast_providers
        assert isinstance(solar.solar_forecast_providers["Forecast.Solar"], QSSolarProviderForecastSolar)

    def test_forecast_solar_has_score_attributes(self):
        """Test Forecast.Solar provider inherits score tracking from base."""
        hass = FakeHass()
        solar = self._make_multi_provider_solar(hass)
        fs_provider = solar.solar_forecast_providers["Forecast.Solar"]

        # Score attributes from base class should exist
        assert hasattr(fs_provider, "score_raw")
        assert hasattr(fs_provider, "score_dampened")
        assert hasattr(fs_provider, "_dampening_coefficients")

    def test_forecast_solar_staleness_detection(self):
        """Test Forecast.Solar provider inherits staleness detection."""
        hass = FakeHass()
        solar = self._make_multi_provider_solar(hass)
        fs_provider = solar.solar_forecast_providers["Forecast.Solar"]

        # Provider starts with no forecast time — should be stale
        assert fs_provider.is_stale is True

        # Set a recent forecast time — should not be stale
        fs_provider._latest_successful_forecast_time = datetime.now(tz=UTC)
        assert fs_provider.is_stale is False

    def test_forecast_solar_health_monitoring(self):
        """Test Forecast.Solar provider inherits health monitoring."""
        hass = FakeHass()
        solar = self._make_multi_provider_solar(hass)
        fs_provider = solar.solar_forecast_providers["Forecast.Solar"]

        # Health tracking attributes from base class
        assert hasattr(fs_provider, "_orchestrator_health")
        assert hasattr(fs_provider, "_update_cycle_count")
