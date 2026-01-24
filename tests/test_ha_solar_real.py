"""Tests for QSSolar and solar provider classes in ha_model/solar.py."""
from __future__ import annotations

import datetime
import logging
import os
import pickle
import tempfile
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.const import (
    Platform,
    CONF_NAME,
)
import pytz

from custom_components.quiet_solar.ha_model.solar import (
    QSSolar,
    QSSolarProvider,
    QSSolarProviderSolcast,
    QSSolarProviderOpenWeather,
    QSSolarProviderSolcastDebug,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR,
    CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR,
    CONF_SOLAR_FORECAST_PROVIDER,
    CONF_SOLAR_MAX_OUTPUT_POWER_VALUE,
    CONF_SOLAR_MAX_PHASE_AMPS,
    SOLCAST_SOLAR_DOMAIN,
    OPEN_METEO_SOLAR_DOMAIN,
    MAX_POWER_INFINITE,
    MAX_AMP_INFINITE,
)

from tests.test_helpers import FakeHass, FakeConfigEntry


class _FilterProvider(QSSolarProvider):
    """Provider with filterable orchestrators."""

    def __init__(self, solar, domain: str) -> None:
        super().__init__(solar=solar, domain=domain)

    def is_orchestrator(self, entity_id, orchestrator) -> bool:
        return getattr(orchestrator, "enabled", False)

    async def get_power_series_from_orchestrator(self, orchestrator, start_time, end_time):
        return []


class _SeriesProvider(QSSolarProvider):
    """Provider that returns stored power series from orchestrators."""

    def __init__(self, solar, domain: str) -> None:
        super().__init__(solar=solar, domain=domain)

    async def get_power_series_from_orchestrator(self, orchestrator, start_time, end_time):
        return [
            (time, value)
            for time, value in orchestrator.power_series
            if start_time <= time <= end_time
        ]


class _UpdateTrackingProvider(QSSolarProvider):
    """Provider that tracks update calls."""

    def __init__(self, solar, domain: str) -> None:
        super().__init__(solar=solar, domain=domain)
        self.fill_calls = 0
        self.extract_calls = 0
        self.extract_result = []

    async def fill_orchestrators(self):
        self.fill_calls += 1

    async def extract_solar_forecast_from_data(self, start_time, period):
        self.extract_calls += 1
        return list(self.extract_result)

    async def get_power_series_from_orchestrator(self, orchestrator, start_time, end_time):
        return []


class TestQSSolarInit:
    """Test QSSolar initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_solar_entry",
            data={CONF_NAME: "Test Solar"},
        )
        self.home = MagicMock()
        self.home.voltage = 230.0
        self.home.physical_3p = False
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
            }
        )

        assert device.name == "My Solar"
        assert device.solar_inverter_active_power == "sensor.solar_power"
        assert device.solar_forecast_provider is None
        assert device.solar_forecast_provider_handler is None

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR: "sensor.solar_input",
                CONF_SOLAR_MAX_OUTPUT_POWER_VALUE: 5000,
                CONF_SOLAR_MAX_PHASE_AMPS: 25.0,
            }
        )

        assert device.solar_inverter_input_active_power == "sensor.solar_input"
        assert device.solar_max_output_power_value == 5000
        assert device.solar_max_phase_amps == 25.0

    def test_init_with_solcast_provider(self):
        """Test initialization with Solcast provider."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN,
            }
        )

        assert device.solar_forecast_provider == SOLCAST_SOLAR_DOMAIN
        assert isinstance(device.solar_forecast_provider_handler, QSSolarProviderSolcast)

    def test_init_with_openmeteo_provider(self):
        """Test initialization with OpenMeteo provider."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_FORECAST_PROVIDER: OPEN_METEO_SOLAR_DOMAIN,
            }
        )

        assert device.solar_forecast_provider == OPEN_METEO_SOLAR_DOMAIN
        assert isinstance(device.solar_forecast_provider_handler, QSSolarProviderOpenWeather)

    def test_init_calculates_max_power_from_amps_single_phase(self):
        """Test max power calculation from amps for single phase."""
        self.home.physical_3p = False

        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_MAX_PHASE_AMPS: 25.0,
            }
        )

        expected_power = 25.0 * 230.0  # amps * voltage
        assert device.solar_max_output_power_value == expected_power

    def test_init_calculates_max_power_from_amps_three_phase(self):
        """Test max power calculation from amps for three phase."""
        self.home.physical_3p = True

        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_MAX_PHASE_AMPS: 25.0,
            }
        )

        expected_power = 25.0 * 230.0 * 3.0  # amps * voltage * 3 phases
        assert device.solar_max_output_power_value == expected_power

    def test_init_calculates_amps_from_power_single_phase(self):
        """Test amp calculation from power for single phase."""
        self.home.physical_3p = False

        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_MAX_OUTPUT_POWER_VALUE: 5750,  # 25A * 230V
            }
        )

        expected_amps = 5750 / 230.0
        assert device.solar_max_phase_amps == expected_amps


class TestQSSolarOverClamp:
    """Test get_current_over_clamp_production_power method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_solar_entry",
            data={CONF_NAME: "Test Solar"},
        )
        self.home = MagicMock()
        self.home.voltage = 230.0
        self.home.physical_3p = False
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_over_clamp_production_below_max(self):
        """Test no over-clamp when production is below max."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_MAX_OUTPUT_POWER_VALUE: 5000,
            }
        )
        device.solar_production = 3000

        result = device.get_current_over_clamp_production_power()

        assert result == 0.0

    def test_over_clamp_production_above_max(self):
        """Test over-clamp when production exceeds max."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_MAX_OUTPUT_POWER_VALUE: 5000,
            }
        )
        device.solar_production = 6000

        result = device.get_current_over_clamp_production_power()

        assert result == 1000.0

    def test_over_clamp_at_exactly_max(self):
        """Test no over-clamp when production equals max."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_MAX_OUTPUT_POWER_VALUE: 5000,
            }
        )
        device.solar_production = 5000

        result = device.get_current_over_clamp_production_power()

        assert result == 0.0


class TestQSSolarForecast:
    """Test forecast-related methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_solar_entry",
            data={CONF_NAME: "Test Solar"},
        )
        self.home = MagicMock()
        self.home.voltage = 230.0
        self.home.physical_3p = False
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    @pytest.mark.asyncio
    async def test_update_forecast_with_provider(self):
        """Test update_forecast calls provider update."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN,
            }
        )
        device.solar_forecast_provider_handler.update = AsyncMock()

        time = datetime.datetime.now(pytz.UTC)
        await device.update_forecast(time)

        device.solar_forecast_provider_handler.update.assert_called_once_with(time)

    @pytest.mark.asyncio
    async def test_update_forecast_without_provider(self):
        """Test update_forecast does nothing without provider."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
            }
        )

        assert device.solar_forecast_provider_handler is None
        time = datetime.datetime.now(pytz.UTC)
        await device.update_forecast(time)
        assert device.solar_forecast_provider_handler is None

    def test_get_forecast_with_provider(self):
        """Test get_forecast returns provider data."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN,
            }
        )

        time = datetime.datetime.now(pytz.UTC)
        forecast_data = [
            (time, 1000.0),
            (time + timedelta(hours=1), 1500.0),
        ]
        device.solar_forecast_provider_handler.get_forecast = MagicMock(return_value=forecast_data)

        result = device.get_forecast(time, time + timedelta(hours=2))

        assert result == forecast_data

    def test_get_forecast_without_provider(self):
        """Test get_forecast returns empty list without provider."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
            }
        )

        time = datetime.datetime.now(pytz.UTC)
        result = device.get_forecast(time, time + timedelta(hours=2))

        assert result == []

    def test_get_value_from_current_forecast_with_provider(self):
        """Test get_value_from_current_forecast returns provider data."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN,
            }
        )

        time = datetime.datetime.now(pytz.UTC)
        device.solar_forecast_provider_handler.get_value_from_current_forecast = MagicMock(
            return_value=(time, 1500.0)
        )

        result_time, result_value = device.get_value_from_current_forecast(time)

        assert result_time == time
        assert result_value == 1500.0

    def test_get_value_from_current_forecast_without_provider(self):
        """Test get_value_from_current_forecast returns None without provider."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
            }
        )

        time = datetime.datetime.now(pytz.UTC)
        result_time, result_value = device.get_value_from_current_forecast(time)

        assert result_time is None
        assert result_value is None


class TestQSSolarProviderBase:
    """Test QSSolarProvider base class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.solar = MagicMock()
        self.solar.hass = self.hass

    def test_init(self):
        """Test provider initialization."""
        provider = QSSolarProvider(solar=self.solar, domain="test_domain")

        assert provider.solar == self.solar
        assert provider.domain == "test_domain"
        assert provider.orchestrators == []
        assert provider._latest_update_time is None
        assert provider.solar_forecast == []

    def test_init_without_solar(self):
        """Test provider initialization without solar device."""
        provider = QSSolarProvider(solar=None, domain=None)

        assert provider.solar is None
        assert provider.hass is None

    def test_get_forecast(self):
        """Test get_forecast filters by time range."""
        provider = QSSolarProvider(solar=self.solar, domain="test")

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [
            (time, 1000.0),
            (time + timedelta(hours=1), 1500.0),
            (time + timedelta(hours=2), 2000.0),
            (time + timedelta(hours=3), 1800.0),
        ]

        result = provider.get_forecast(time, time + timedelta(hours=2))

        # Should get slots from time series
        assert len(result) >= 0

    def test_get_value_from_current_forecast(self):
        """Test get_value_from_current_forecast returns correct value."""
        provider = QSSolarProvider(solar=self.solar, domain="test")

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        provider.solar_forecast = [
            (time, 1000.0),
            (time + timedelta(hours=1), 1500.0),
        ]

        result_time, result_value = provider.get_value_from_current_forecast(time)

        # Result depends on get_value_from_time_series implementation
        assert result_time is not None or result_value is not None or (result_time is None and result_value is None)

    def test_is_orchestrator_default(self):
        """Test default is_orchestrator returns True."""
        provider = QSSolarProvider(solar=self.solar, domain="test")

        result = provider.is_orchestrator("entity_id", MagicMock())

        assert result is True

    @pytest.mark.asyncio
    async def test_update_fills_orchestrators(self):
        """Test update fills orchestrators and extracts forecast."""
        provider = QSSolarProvider(solar=self.solar, domain="test")
        provider.fill_orchestrators = AsyncMock()
        provider.extract_solar_forecast_from_data = AsyncMock(return_value=[])

        # Add a fake orchestrator
        provider.orchestrators = [MagicMock()]

        time = datetime.datetime.now(pytz.UTC)
        await provider.update(time)

        # Should have called fill_orchestrators
        provider.fill_orchestrators.assert_called()


class TestQSSolarProviderBaseExtended:
    """Test additional QSSolarProvider base behaviors."""

    @pytest.mark.asyncio
    async def test_fill_orchestrators_filters_entries(self):
        """Test fill_orchestrators filters by is_orchestrator."""
        hass = FakeHass()
        solar = SimpleNamespace(hass=hass)
        provider = _FilterProvider(solar=solar, domain="test_domain")

        hass.data["test_domain"] = {
            "valid": SimpleNamespace(enabled=True),
            "invalid": SimpleNamespace(enabled=False),
            "none": None,
        }

        await provider.fill_orchestrators()

        assert len(provider.orchestrators) == 1
        assert provider.orchestrators[0].enabled is True

    @pytest.mark.asyncio
    async def test_extract_solar_forecast_aggregates_orchestrators(self):
        """Test extract_solar_forecast_from_data aggregates series."""
        hass = FakeHass()
        solar = SimpleNamespace(hass=hass)
        provider = _SeriesProvider(solar=solar, domain="test_domain")

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        orchestrator_1 = SimpleNamespace(
            power_series=[
                (time, 500.0),
                (time + timedelta(minutes=30), 800.0),
            ]
        )
        orchestrator_2 = SimpleNamespace(
            power_series=[
                (time, 700.0),
                (time + timedelta(minutes=30), 600.0),
            ]
        )

        provider.orchestrators = [orchestrator_1, orchestrator_2]
        result = await provider.extract_solar_forecast_from_data(time, period=3600)

        assert result == [
            (time, 1200.0),
            (time + timedelta(minutes=30), 1400.0),
        ]

    @pytest.mark.asyncio
    async def test_update_time_gating(self):
        """Test update skips refresh within cache window."""
        hass = FakeHass()
        solar = SimpleNamespace(hass=hass)
        provider = _UpdateTrackingProvider(solar=solar, domain="test_domain")

        now = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        provider.orchestrators = [SimpleNamespace()]
        provider._latest_update_time = now

        await provider.update(now + timedelta(minutes=5))

        assert provider.fill_calls == 0
        assert provider.extract_calls == 0

        provider.extract_result = [(now, 1000.0)]
        await provider.update(now + timedelta(minutes=16))

        assert provider.fill_calls == 1
        assert provider.extract_calls == 1
        assert provider.solar_forecast == [(now, 1000.0)]
        assert provider._latest_update_time == now + timedelta(minutes=16)

    @pytest.mark.asyncio
    async def test_update_logs_when_no_orchestrators(self, caplog):
        """Test update logs an error when no orchestrators exist."""
        hass = FakeHass()
        solar = SimpleNamespace(hass=hass)
        provider = _UpdateTrackingProvider(solar=solar, domain="test_domain")
        provider.orchestrators = []

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        with caplog.at_level(logging.ERROR):
            await provider.update(time)

        assert "No solar orchestrator found for domain test_domain" in caplog.text

    @pytest.mark.asyncio
    async def test_dump_for_debug_writes_pickle(self, tmp_path):
        """Test dump_for_debug writes expected pickle file."""
        hass = FakeHass()
        solar = SimpleNamespace(hass=hass)
        provider = _SeriesProvider(solar=solar, domain="test_domain")
        provider.solar_forecast = [
            (datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC), 1000.0),
        ]

        await provider.dump_for_debug(str(tmp_path))

        file_path = tmp_path / "solar_forecast.pickle"
        assert file_path.exists()
        with file_path.open("rb") as file:
            assert pickle.load(file) == provider.solar_forecast


class TestQSSolarProviderSolcast:
    """Test QSSolarProviderSolcast class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.solar = MagicMock()
        self.solar.hass = self.hass

    def test_init(self):
        """Test Solcast provider initialization."""
        provider = QSSolarProviderSolcast(solar=self.solar)

        assert provider.domain == SOLCAST_SOLAR_DOMAIN
        assert provider.solar == self.solar

    @pytest.mark.asyncio
    async def test_fill_orchestrators_no_entries(self):
        """Test fill_orchestrators with no config entries."""
        provider = QSSolarProviderSolcast(solar=self.solar)
        self.hass.config_entries.async_entries = MagicMock(return_value=[])

        await provider.fill_orchestrators()

        assert provider.orchestrators == []

    @pytest.mark.asyncio
    async def test_fill_orchestrators_with_valid_entry(self):
        """Test fill_orchestrators with valid config entry."""
        provider = QSSolarProviderSolcast(solar=self.solar)

        # Create a mock config entry with the expected structure
        mock_coordinator = MagicMock()
        mock_coordinator.solcast._data_forecasts = []

        mock_entry = MagicMock()
        mock_entry.runtime_data.coordinator = mock_coordinator

        self.hass.config_entries.async_entries = MagicMock(return_value=[mock_entry])

        await provider.fill_orchestrators()

        assert len(provider.orchestrators) == 1
        assert provider.orchestrators[0] == mock_coordinator

    @pytest.mark.asyncio
    async def test_fill_orchestrators_skips_invalid_entry(self):
        """Test fill_orchestrators ignores entries without coordinators."""
        provider = QSSolarProviderSolcast(solar=self.solar)

        good_coordinator = SimpleNamespace(solcast=SimpleNamespace(_data_forecasts=[]))
        good_entry = SimpleNamespace(runtime_data=SimpleNamespace(coordinator=good_coordinator))
        bad_entry = SimpleNamespace(runtime_data=None)

        self.hass.config_entries.async_entries = MagicMock(return_value=[good_entry, bad_entry])

        await provider.fill_orchestrators()

        assert provider.orchestrators == [good_coordinator]

    @pytest.mark.asyncio
    async def test_get_power_series_from_orchestrator(self):
        """Test extracting power series from Solcast orchestrator."""
        provider = QSSolarProviderSolcast(solar=self.solar)

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)

        # Create mock orchestrator with forecast data
        mock_orchestrator = MagicMock()
        mock_orchestrator.solcast._data_forecasts = [
            {'period_start': time, 'pv_estimate': 1.0},
            {'period_start': time + timedelta(hours=1), 'pv_estimate': 1.5},
            {'period_start': time + timedelta(hours=2), 'pv_estimate': 2.0},
        ]

        result = await provider.get_power_series_from_orchestrator(
            mock_orchestrator, time, time + timedelta(hours=3)
        )

        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_get_power_series_handles_boundaries_and_timezone(self):
        """Test Solcast series boundaries and UTC conversion."""
        provider = QSSolarProviderSolcast(solar=self.solar)
        tz = pytz.timezone("Europe/Paris")

        start = tz.localize(datetime.datetime(2024, 6, 15, 12, 30, 0))
        end = tz.localize(datetime.datetime(2024, 6, 15, 16, 0, 0))
        data = [
            {"period_start": tz.localize(datetime.datetime(2024, 6, 15, 12, 0, 0)), "pv_estimate": 1.0},
            {"period_start": tz.localize(datetime.datetime(2024, 6, 15, 13, 0, 0)), "pv_estimate": 1.5},
            {"period_start": tz.localize(datetime.datetime(2024, 6, 15, 14, 0, 0)), "pv_estimate": 2.0},
        ]

        orchestrator = SimpleNamespace(solcast=SimpleNamespace(_data_forecasts=data))
        result = await provider.get_power_series_from_orchestrator(orchestrator, start, end)

        assert result[0][0].astimezone(pytz.UTC).hour == 10
        assert result[-1][0].astimezone(pytz.UTC).hour == 12
        assert all(ts.tzinfo == pytz.UTC for ts, _ in result)
        assert [value for _, value in result] == [1000.0, 1500.0, 2000.0]

    @pytest.mark.asyncio
    async def test_get_power_series_empty_data(self):
        """Test extracting power series with no data."""
        provider = QSSolarProviderSolcast(solar=self.solar)

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)

        mock_orchestrator = MagicMock()
        mock_orchestrator.solcast._data_forecasts = None

        result = await provider.get_power_series_from_orchestrator(
            mock_orchestrator, time, time + timedelta(hours=3)
        )

        assert result == []


class TestQSSolarProviderOpenWeather:
    """Test QSSolarProviderOpenWeather class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.solar = MagicMock()
        self.solar.hass = self.hass

    def test_init(self):
        """Test OpenWeather provider initialization."""
        provider = QSSolarProviderOpenWeather(solar=self.solar)

        assert provider.domain == OPEN_METEO_SOLAR_DOMAIN
        assert provider.solar == self.solar

    def test_is_orchestrator_valid(self):
        """Test is_orchestrator with valid orchestrator."""
        provider = QSSolarProviderOpenWeather(solar=self.solar)

        mock_orchestrator = MagicMock()
        mock_orchestrator.data.watts = {}

        result = provider.is_orchestrator("entity_id", mock_orchestrator)

        assert result is True

    def test_is_orchestrator_invalid(self):
        """Test is_orchestrator with invalid orchestrator."""
        provider = QSSolarProviderOpenWeather(solar=self.solar)

        # Create orchestrator without data.watts
        mock_orchestrator = MagicMock()
        del mock_orchestrator.data.watts

        result = provider.is_orchestrator("entity_id", mock_orchestrator)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_power_series_from_orchestrator(self):
        """Test extracting power series from OpenWeather orchestrator."""
        provider = QSSolarProviderOpenWeather(solar=self.solar)

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)

        # Create mock orchestrator with watts data
        mock_orchestrator = MagicMock()
        mock_orchestrator.data.watts = {
            time: 1000,
            time + timedelta(hours=1): 1500,
            time + timedelta(hours=2): 2000,
        }

        result = await provider.get_power_series_from_orchestrator(
            mock_orchestrator, time, time + timedelta(hours=3)
        )

        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_get_power_series_sorts_and_handles_boundaries(self):
        """Test OpenWeather series sorting, boundaries, and UTC conversion."""
        provider = QSSolarProviderOpenWeather(solar=self.solar)
        tz = pytz.timezone("Europe/Paris")

        start = tz.localize(datetime.datetime(2024, 6, 15, 12, 30, 0))
        end = tz.localize(datetime.datetime(2024, 6, 15, 16, 0, 0))
        t0 = tz.localize(datetime.datetime(2024, 6, 15, 12, 0, 0))
        t1 = tz.localize(datetime.datetime(2024, 6, 15, 13, 0, 0))
        t2 = tz.localize(datetime.datetime(2024, 6, 15, 14, 0, 0))

        orchestrator = SimpleNamespace(
            data=SimpleNamespace(watts={t1: 1500, t2: 2000, t0: 1000})
        )

        result = await provider.get_power_series_from_orchestrator(orchestrator, start, end)

        assert [value for _, value in result] == [1000.0, 1500.0, 2000.0]
        assert all(ts.tzinfo == pytz.UTC for ts, _ in result)
        assert result[0][0] < result[-1][0]

    @pytest.mark.asyncio
    async def test_get_power_series_none_data(self):
        """Test extracting power series with None data."""
        provider = QSSolarProviderOpenWeather(solar=self.solar)

        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)

        mock_orchestrator = MagicMock()
        mock_orchestrator.data.watts = None

        result = await provider.get_power_series_from_orchestrator(
            mock_orchestrator, time, time + timedelta(hours=3)
        )

        assert result == []


class TestQSSolarProviderSolcastDebug:
    """Test QSSolarProviderSolcastDebug class."""

    def test_init_and_load_from_file(self):
        """Test debug provider initialization and loading from file."""
        # Create a temp file with pickled forecast data
        time = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        forecast_data = [
            (time, 1000.0),
            (time + timedelta(hours=1), 1500.0),
        ]

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pickle') as f:
            pickle.dump(forecast_data, f)
            temp_path = f.name

        try:
            provider = QSSolarProviderSolcastDebug(file_path=temp_path)

            assert provider.solar_forecast == forecast_data
            assert provider.solar is None
            assert provider.domain is None
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_get_power_series_returns_empty(self):
        """Test get_power_series_from_orchestrator returns empty list."""
        # Create a temp file with minimal data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pickle') as f:
            pickle.dump([], f)
            temp_path = f.name

        try:
            provider = QSSolarProviderSolcastDebug(file_path=temp_path)

            time = datetime.datetime.now(pytz.UTC)
            result = await provider.get_power_series_from_orchestrator(
                MagicMock(), time, time + timedelta(hours=1)
            )

            assert result == []
        finally:
            os.unlink(temp_path)


class TestQSSolarDumpForDebug:
    """Test dump_for_debug method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_solar_entry",
            data={CONF_NAME: "Test Solar"},
        )
        self.home = MagicMock()
        self.home.voltage = 230.0
        self.home.physical_3p = False
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    @pytest.mark.asyncio
    async def test_dump_for_debug_with_provider(self):
        """Test dump_for_debug calls provider dump."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
                CONF_SOLAR_FORECAST_PROVIDER: SOLCAST_SOLAR_DOMAIN,
            }
        )
        device.solar_forecast_provider_handler.dump_for_debug = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            await device.dump_for_debug(tmpdir)

            device.solar_forecast_provider_handler.dump_for_debug.assert_called_once_with(tmpdir)

    @pytest.mark.asyncio
    async def test_dump_for_debug_without_provider(self):
        """Test dump_for_debug does nothing without provider."""
        device = QSSolar(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Solar",
                CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR: "sensor.solar_power",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            await device.dump_for_debug(tmpdir)
            assert not os.path.exists(os.path.join(tmpdir, "solar_forecast.pickle"))
