from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from operator import itemgetter
from os.path import join

import aiofiles.os
import numpy as np
import pytz
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_ACCURATE_POWER_SENSOR,
    CONF_IS_3P,
    CONF_SOLAR_FORECAST_PROVIDER,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR,
    CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR,
    CONF_SOLAR_MAX_OUTPUT_POWER_VALUE,
    CONF_SOLAR_MAX_PHASE_AMPS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    FLOATING_PERIOD_S,
    FORECAST_SOLAR_DOMAIN,
    MAX_AMP_INFINITE,
    MAX_POWER_INFINITE,
    OPEN_METEO_SOLAR_DOMAIN,
    SOLAR_FORECAST_STALE_THRESHOLD_S,
    SOLAR_ORCHESTRATOR_REPROBE_CYCLES,
    SOLAR_PROVIDER_MODE_AUTO,
    SOLCAST_SOLAR_DOMAIN,
    CONF_TYPE_NAME_QSSolar,
    QSForecastSolarSensors,
)
from ..ha_model.device import HADeviceMixin
from ..home_model.home_utils import (
    align_time_series_and_values,
    get_slots_from_time_series,
    get_value_from_time_series,
)
from ..home_model.load import AbstractDevice

_LOGGER = logging.getLogger(__name__)

SOLAR_AND_PROVIDER_UPDATE_PERIOD_UPDATE_S = 15 * 60


def _migrate_solar_providers_config(config_data: dict) -> list[dict]:
    """Migrate old single-provider config to new multi-provider list format."""
    providers = config_data.get(CONF_SOLAR_FORECAST_PROVIDERS)
    if providers is not None:
        return providers

    old_provider = config_data.get(CONF_SOLAR_FORECAST_PROVIDER)
    if old_provider is None:
        return []

    domain_labels = {
        SOLCAST_SOLAR_DOMAIN: "Solcast",
        FORECAST_SOLAR_DOMAIN: "Forecast.Solar",
        OPEN_METEO_SOLAR_DOMAIN: "Open-Meteo",
    }
    label = domain_labels.get(old_provider, old_provider)
    return [{CONF_SOLAR_PROVIDER_DOMAIN: old_provider, CONF_SOLAR_PROVIDER_NAME: label}]


def _create_provider_for_domain(domain: str, solar: QSSolar, name: str) -> QSSolarProvider:
    """Create the correct provider subclass for a given domain."""
    if domain == SOLCAST_SOLAR_DOMAIN:
        return QSSolarProviderSolcast(solar=solar, provider_name=name)
    if domain == OPEN_METEO_SOLAR_DOMAIN:
        return QSSolarProviderOpenWeather(solar=solar, provider_name=name)
    if domain == FORECAST_SOLAR_DOMAIN:
        return QSSolarProviderForecastSolar(solar=solar, provider_name=name)
    _LOGGER.warning("Unknown solar forecast domain %s, skipping", domain)
    return None


class QSSolar(HADeviceMixin, AbstractDevice):
    conf_type_name = CONF_TYPE_NAME_QSSolar

    def __init__(self, **kwargs) -> None:
        self.solar_inverter_active_power = kwargs.pop(CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, None)
        self.solar_inverter_input_active_power = kwargs.pop(CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, None)

        # Pop both old and new config keys — migration produces the list
        self.solar_forecast_provider = kwargs.pop(CONF_SOLAR_FORECAST_PROVIDER, None)
        raw_providers = kwargs.pop(CONF_SOLAR_FORECAST_PROVIDERS, None)

        self.solar_max_output_power_value = kwargs.pop(CONF_SOLAR_MAX_OUTPUT_POWER_VALUE, MAX_POWER_INFINITE)
        self.solar_max_phase_amps = kwargs.pop(CONF_SOLAR_MAX_PHASE_AMPS, MAX_AMP_INFINITE)

        # Legacy single-provider reference (kept for backward compat)
        self.solar_forecast_provider_handler: QSSolarProvider | None = None

        # Multi-provider infrastructure
        self.solar_forecast_providers: dict[str, QSSolarProvider] = {}
        self._provider_mode: str = SOLAR_PROVIDER_MODE_AUTO
        self._active_provider_name: str | None = None

        kwargs[CONF_ACCURATE_POWER_SENSOR] = self.solar_inverter_active_power

        home = kwargs.get("home", None)
        if home:
            kwargs[CONF_IS_3P] = home.physical_3p

        super().__init__(**kwargs)

        if (
            self.solar_max_output_power_value == MAX_POWER_INFINITE
            and self.solar_max_phase_amps != MAX_AMP_INFINITE
            and self.home is not None
        ):
            if self.physical_3p:
                self.solar_max_output_power_value = self.solar_max_phase_amps * self.home.voltage * 3.0
            else:
                self.solar_max_output_power_value = self.solar_max_phase_amps * self.home.voltage
        elif (
            self.solar_max_output_power_value != MAX_POWER_INFINITE
            and self.solar_max_phase_amps == MAX_AMP_INFINITE
            and self.home is not None
        ):
            if self.physical_3p:
                self.solar_max_phase_amps = self.solar_max_output_power_value / (self.home.voltage * 3.0)
            else:
                self.solar_max_phase_amps = self.solar_max_output_power_value / self.home.voltage

        self.attach_power_to_probe(self.solar_inverter_active_power)
        self.attach_power_to_probe(self.solar_inverter_input_active_power)

        # Build provider config from either new list or old single key
        config_data = {}
        if raw_providers is not None:
            config_data[CONF_SOLAR_FORECAST_PROVIDERS] = raw_providers
        elif self.solar_forecast_provider is not None:
            config_data[CONF_SOLAR_FORECAST_PROVIDER] = self.solar_forecast_provider

        provider_configs = _migrate_solar_providers_config(config_data)
        for pc in provider_configs:
            domain = pc[CONF_SOLAR_PROVIDER_DOMAIN]
            name = pc[CONF_SOLAR_PROVIDER_NAME]
            provider = _create_provider_for_domain(domain, self, name)
            if provider is not None:
                self.solar_forecast_providers[name] = provider
                _LOGGER.info("Created solar forecast provider %s (%s)", name, domain)

        # Set active provider: first one by default
        if self.solar_forecast_providers:
            self._active_provider_name = next(iter(self.solar_forecast_providers))
            self.solar_forecast_provider_handler = self.solar_forecast_providers[self._active_provider_name]

        self._last_scoring_half_day: tuple | None = None
        self.solar_production = 0
        self.inverter_output_power = 0

        from .home import QSforecastValueSensor  # deferred to avoid circular import

        self.solar_forecast_sensor_values_probers = QSforecastValueSensor.get_probers(
            self.get_value_from_current_forecast, None, QSForecastSolarSensors
        )

        self.solar_forecast_sensor_values_per_provider_probers = {}
        for name, provider in self.solar_forecast_providers.items():
            self.solar_forecast_sensor_values_per_provider_probers[name] = QSforecastValueSensor.get_probers(
                provider.get_value_from_current_forecast, None, QSForecastSolarSensors
            )

        self.solar_forecast_sensor_values = {}
        self.solar_forecast_sensor_values_per_provider = {}

    async def update_forecast_probers(self, time: datetime):
        for name, prober in self.solar_forecast_sensor_values_probers.items():
            self.solar_forecast_sensor_values[name] = prober.push_and_get(time)

        for provider_name, probers in self.solar_forecast_sensor_values_per_provider_probers.items():
            for name, prober in probers.items():
                full_name = f"{provider_name}_{name}"
                self.solar_forecast_sensor_values_per_provider[full_name] = prober.push_and_get(time)

    @property
    def provider_mode(self) -> str:
        """Return current provider selection mode."""
        return self._provider_mode

    @property
    def active_provider_name(self) -> str | None:
        """Return name of the currently active provider."""
        return self._active_provider_name

    @property
    def active_provider(self) -> QSSolarProvider | None:
        """Return the currently active provider."""
        if self._active_provider_name is None:
            return None
        return self.solar_forecast_providers.get(self._active_provider_name)

    def get_provider_names(self) -> list[str]:
        """Return list of configured provider names."""
        return list(self.solar_forecast_providers.keys())

    def set_provider_mode(self, mode: str | None) -> None:
        """Set provider selection mode ('auto' or a provider name)."""
        # Treat None or unknown provider names as auto
        if mode is None or (mode != SOLAR_PROVIDER_MODE_AUTO and mode not in self.solar_forecast_providers):
            mode = SOLAR_PROVIDER_MODE_AUTO
        self._provider_mode = mode
        if mode != SOLAR_PROVIDER_MODE_AUTO:
            self._set_active_provider(mode)

    def _set_active_provider(self, name: str) -> None:
        """Switch the active provider to the named one."""
        if name in self.solar_forecast_providers:
            self._active_provider_name = name
            self.solar_forecast_provider_handler = self.solar_forecast_providers[name]

    def auto_select_best_provider(self) -> None:
        """In auto mode, select provider with lowest score (MAE), falling back to freshness."""
        if not self.solar_forecast_providers:
            return
        best_name = None
        best_score = float("inf")
        best_freshness: datetime | None = None
        for name, provider in self.solar_forecast_providers.items():
            score = provider.get_active_score()
            if score is None:
                continue
            freshness = provider.latest_successful_forecast_time
            if score < best_score or (
                score == best_score and freshness is not None and (best_freshness is None or freshness > best_freshness)
            ):
                best_score = score
                best_name = name
                best_freshness = freshness
        if best_name is not None:
            self._set_active_provider(best_name)
            return
        # Fallback: no scores yet — pick the freshest non-stale provider
        freshest_name = None
        freshest_time: datetime | None = None
        for name, provider in self.solar_forecast_providers.items():
            if provider.is_stale:
                continue
            ft = provider.latest_successful_forecast_time
            if ft is not None and (freshest_time is None or ft > freshest_time):
                freshest_time = ft
                freshest_name = name
        if freshest_name is not None:
            self._set_active_provider(freshest_name)

    def get_historical_solar_fallback(self, time: datetime) -> list[tuple[datetime, float]]:
        """Return historical solar production as a fallback forecast when APIs are stale."""
        if self.home is None:
            return []
        forecast_handler = getattr(self.home, "solar_and_consumption_forecast", None)
        if forecast_handler is None:
            return []
        history = getattr(forecast_handler, "solar_production_history", None)
        if history is None:
            return []
        return history.get_historical_solar_pattern(time)

    def get_current_over_clamp_production_power(self) -> float:
        if self.solar_production > self.solar_max_output_power_value:
            return self.solar_production - self.solar_max_output_power_value
        return 0.0

    async def update_forecast(self, time: datetime) -> None:
        """Update all providers and select active one."""
        for provider in self.solar_forecast_providers.values():
            try:
                await provider.update(time)
            except Exception:
                _LOGGER.exception("Error updating solar forecast provider %s", provider.provider_name)

        # Scoring cycle at 00:00 and 12:00 local time
        self._run_scoring_cycle(time)

        if self._provider_mode == SOLAR_PROVIDER_MODE_AUTO:
            self.auto_select_best_provider()

        # Keep legacy handler in sync
        self.solar_forecast_provider_handler = self.active_provider

    def reset_scoring(self, time: datetime) -> None:
        """Clear all scoring state for all providers."""
        self._last_scoring_half_day = None
        for provider in self.solar_forecast_providers.values():
            provider.reset_scoring()

    async def force_scoring_cycle(self, time: datetime | None = None) -> None:
        """Force a scoring recomputation, bypassing the half-day throttle."""
        if time is None:
            time = dt_util.utcnow()
        self._last_scoring_half_day = None
        self._run_scoring_cycle(time)

        if self._provider_mode == SOLAR_PROVIDER_MODE_AUTO:
            self.auto_select_best_provider()

        self.solar_forecast_provider_handler = self.active_provider

    @staticmethod
    def _scoring_half_day(time: datetime) -> tuple:
        """Return (local_date, half_day_slot) — 0 for 00:00-11:59, 1 for 12:00-23:59."""
        local_now = time.astimezone(tz=None) if time.tzinfo else time
        return (local_now.date(), 0 if local_now.hour < 12 else 1)

    def _run_scoring_cycle(self, time: datetime) -> None:
        """Run scoring at 00:00 and 12:00 local time boundaries."""
        current_half = self._scoring_half_day(time)

        if self._last_scoring_half_day is not None and self._last_scoring_half_day >= current_half:
            return

        self._last_scoring_half_day = current_half

        for name, provider in self.solar_forecast_providers.items():
            # Score using previously stored forecast vs last 24h actuals
            is_score_computed = provider.compute_score(time)

            _LOGGER.info(
                "Scoring cycle for provider %s: score=%.2f is_score_computed=%s",
                name,
                provider.score if provider.score is not None else float("nan"),
                is_score_computed,
            )

    def get_forecast(
        self, start_time: datetime, end_time: datetime | None
    ) -> list[tuple[datetime | None, str | float | None]]:
        provider = self.active_provider
        if provider is not None:
            return provider.get_forecast(start_time, end_time)
        return []

    def get_value_from_current_forecast(self, time: datetime) -> tuple[datetime | None, str | float | None]:
        provider = self.active_provider
        if provider is not None:
            return provider.get_value_from_current_forecast(time)
        return (None, None)

    async def dump_for_debug(self, debug_path: str) -> None:
        provider = self.active_provider
        if provider is not None:
            await provider.dump_for_debug(debug_path)

    def get_forecast_age_hours(self) -> float | None:
        """Return hours since last successful forecast update from active provider."""
        provider = self.active_provider
        if provider is None or provider.latest_successful_forecast_time is None:
            return None
        now = datetime.now(pytz.UTC)
        delta = (now - provider.latest_successful_forecast_time).total_seconds()
        return delta / 3600.0

    def is_forecast_ok(self) -> bool:
        """Return True if active forecast is fresh (< stale threshold)."""
        provider = self.active_provider
        if provider is None:
            return False
        return not provider.is_stale

    def get_platforms(self):
        """Return platforms including select for solar provider selection."""
        from homeassistant.const import Platform

        platforms = super().get_platforms()
        platforms_set = set(platforms)
        if self.solar_forecast_providers:
            platforms_set.add(Platform.SELECT)
        return list(platforms_set)


class QSSolarProvider:
    def __init__(self, solar: QSSolar | None, domain: str | None, provider_name: str = "", **kwargs) -> None:
        self.solar = solar
        if solar is not None:
            self.hass = solar.hass
        else:
            self.hass = None
        self.orchestrators: list = []
        self.domain = domain
        self.provider_name = provider_name
        self._latest_update_time: datetime | None = None
        self.solar_forecast: list[tuple[datetime | None, float | None]] = []

        # Staleness tracking (Task 3)
        self._latest_successful_forecast_time: datetime | None = None

        # Health tracking (Task 5)
        self._orchestrator_health: dict[int, bool] = {}
        self._update_cycle_count: int = 0

        # Scoring
        self.score: float | None = None

        # Stale notification state (Task 9)
        self._was_stale: bool = False

    @property
    def latest_successful_forecast_time(self) -> datetime | None:
        """Return time of last successful forecast update."""
        return self._latest_successful_forecast_time

    @property
    def is_stale(self) -> bool:
        """Return True if forecast is stale (>threshold or never received)."""
        if self._latest_successful_forecast_time is None:
            return True
        now = datetime.now(pytz.UTC)
        age_s = (now - self._latest_successful_forecast_time).total_seconds()
        return age_s > SOLAR_FORECAST_STALE_THRESHOLD_S

    def get_active_score(self) -> float | None:
        """Return the provider's forecast accuracy score (MAE vs actuals)."""
        return self.score

    def get_forecast(
        self, start_time: datetime, end_time: datetime | None
    ) -> list[tuple[datetime | None, str | float | None]]:
        return get_slots_from_time_series(self.solar_forecast, start_time, end_time)

    def get_value_from_current_forecast(self, time: datetime) -> tuple[datetime | None, str | float | None]:
        res = get_value_from_time_series(self.solar_forecast, time)
        return res[0], res[1]

    async def fill_orchestrators(self):
        """Return the orchestrators for the domain."""
        self.orchestrators = []
        self._orchestrator_health = {}
        for entry_id, orchestrator in self.solar.hass.data.get(self.domain, {}).items():
            if orchestrator is not None:
                self.orchestrators.append(orchestrator)
                self._orchestrator_health[id(orchestrator)] = True

    async def update(self, time: datetime) -> None:
        if (
            len(self.orchestrators) == 0
            or self._latest_update_time is None
            or (time - self._latest_update_time).total_seconds() > SOLAR_AND_PROVIDER_UPDATE_PERIOD_UPDATE_S
        ):
            await self.fill_orchestrators()

            self._update_cycle_count += 1

            # Health-based validation (Task 5)
            for orchestrator in self.orchestrators:
                orch_id = id(orchestrator)
                is_healthy = self._orchestrator_health.get(orch_id, True)

                if is_healthy or (self._update_cycle_count % SOLAR_ORCHESTRATOR_REPROBE_CYCLES == 0):
                    try:
                        await self.get_power_series_from_orchestrator(orchestrator, None, None)
                        self._orchestrator_health[orch_id] = True
                    except Exception:
                        if is_healthy:
                            _LOGGER.warning("Orchestrator %s for domain %s marked unhealthy", orchestrator, self.domain)
                        self._orchestrator_health[orch_id] = False

            healthy_orchestrators = [o for o in self.orchestrators if self._orchestrator_health.get(id(o), False)]

            if len(healthy_orchestrators) > 0:
                # Temporarily swap in only healthy orchestrators for extraction
                original_orchestrators = self.orchestrators
                self.orchestrators = healthy_orchestrators
                try:
                    new_forecast = await self.extract_solar_forecast_from_data(time, period=FLOATING_PERIOD_S)
                finally:
                    self.orchestrators = original_orchestrators

                if len(new_forecast) > 0:
                    self.solar_forecast = new_forecast
                    prev_successful_time = self._latest_successful_forecast_time
                    self._latest_update_time = time
                    self._latest_successful_forecast_time = time

                    # Stale transition notification (Task 9)
                    if self._was_stale:
                        age_h = 0.0
                        if prev_successful_time is not None:
                            age_h = (time - prev_successful_time).total_seconds() / 3600.0
                        _LOGGER.info(
                            "Solar forecast recovered for provider %s (was stale for %.1f hours)",
                            self.provider_name,
                            age_h,
                        )
                        self._was_stale = False
            else:
                _LOGGER.error(
                    "No healthy solar orchestrator found for provider %s (%s)", self.provider_name, self.domain
                )

            # Check stale transition (Task 9)
            if self.is_stale and not self._was_stale:
                _LOGGER.warning(
                    "Solar forecast is stale for provider %s (last successful update: %s), falling back to historical patterns",
                    self.provider_name,
                    self._latest_successful_forecast_time,
                )
                self._was_stale = True

            # Historical fallback (Task 4): when stale and no forecast data
            if self.is_stale and len(self.solar_forecast) == 0 and self.solar is not None:
                fallback = self.solar.get_historical_solar_fallback(time)
                if fallback:
                    self.solar_forecast = fallback
                    _LOGGER.warning(
                        "Using historical solar pattern as fallback for provider %s (%d points)",
                        self.provider_name,
                        len(fallback),
                    )

    async def extract_solar_forecast_from_data(
        self, start_time: datetime, period: float
    ) -> list[tuple[datetime | None, float | None]]:
        """Extract forecast from all orchestrators (legacy API)."""
        end_time = start_time + timedelta(seconds=period)
        vals = []
        for orchestrator in self.orchestrators:
            s = await self.get_power_series_from_orchestrator(orchestrator, start_time, end_time)
            if s:
                vals.append(s)
        if len(vals) == 0:
            return []
        v_aggregated = vals[0]
        for v in vals[1:]:
            v_aggregated = align_time_series_and_values(v_aggregated, v, operation=lambda x, y: x + y)
        return v_aggregated

    def reset_scoring(self) -> None:
        """Clear scoring state (stored forecast + score)."""
        self.score = None

    def compute_score(self, time: datetime) -> bool:
        """Compute MAE score comparing stored forecast vs provided actuals."""
        if self.solar is None or self.solar.home is None:
            return False
        forecast_handler = self.solar.home.solar_and_consumption_forecast
        if forecast_handler is None:
            return False
        prod_history = forecast_handler.solar_production_history
        if prod_history is None:
            return False

        actual_24_hours = prod_history.get_historical_data(time, past_hours=24)

        histories = forecast_handler.get_forecast_histories_for_provider(self.provider_name)

        past_forecast = None
        # in histories we end up having QSForecastSolarSensors with the timing, 8h can be a good one

        name_list = [(v, n) for n, v in QSForecastSolarSensors.items()]
        # sort it by time
        name_list.sort(key=lambda x: x[0])

        for value_s, name in name_list:
            if value_s >= 8 * 3600:
                if name in histories:
                    past_forecast = histories[name].get_historical_data(time, past_hours=24)
                    if past_forecast:
                        break

        if past_forecast is None:
            name_list.sort(key=lambda x: x[0], reverse=True)
            for value_s, name in name_list:
                if name in histories:
                    past_forecast = histories[name].get_historical_data(time, past_hours=24)
                    if past_forecast:
                        break

        aligned_actuals = actual_24_hours
        aligned_forecast = past_forecast

        if not aligned_actuals or not aligned_forecast:
            _LOGGER.warning(
                "compute_score: missing data for provider %s, actuals=%d points, forecast=%d points",
                self.provider_name,
                len(aligned_actuals) if aligned_actuals else 0,
                len(aligned_forecast) if aligned_forecast else 0,
            )
            return False

        self.score = self._compute_mae_from_aligned(aligned_forecast, aligned_actuals)
        return True

    @staticmethod
    def _compute_mae_from_aligned(
        forecast_aligned: list[tuple[datetime, float]],
        actuals_aligned: list[tuple[datetime, float]],
    ) -> float | None:
        """Compute MAE from two aligned time series (same slot boundaries)."""
        if len(forecast_aligned) != len(actuals_aligned):
            _LOGGER.warning(
                "MAE length mismatch: forecast=%s actuals=%s, truncating to min",
                len(forecast_aligned),
                len(actuals_aligned),
            )
        n = min(len(forecast_aligned), len(actuals_aligned))
        if n == 0:
            return None

        errors = []
        for i in range(n):
            fv = forecast_aligned[i][1]
            av = actuals_aligned[i][1]
            # Only score daytime slots (at least one > 0)
            if fv > 0 or av > 0:
                errors.append(abs(fv - av))

        if not errors:
            return None
        return float(np.mean(errors))

    async def dump_for_debug(self, debug_path: str) -> None:
        storage_path = debug_path
        await aiofiles.os.makedirs(storage_path, exist_ok=True)
        file_path = join(storage_path, "solar_forecast.pickle")

        def _pickle_save(file_path, obj):
            with open(file_path, "wb") as file:
                pickle.dump(obj, file)

        await self.hass.async_add_executor_job(_pickle_save, file_path, self.solar_forecast)

    @abstractmethod
    async def get_power_series_from_orchestrator(
        self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime | None, str | float | None]]:
        """Return the power series from the orchestrator.

        When called with start_time=None and end_time=None, this acts as
        a validation probe: it must access the orchestrator's data source
        and raise if the structure is invalid, or return [] if valid.
        """


class QSSolarProviderSolcastDebug(QSSolarProvider):
    def __init__(self, file_path: str, **kwargs) -> None:
        super().__init__(solar=None, domain=None, **kwargs)
        self.read_from_debug_dump(file_path)

    def read_from_debug_dump(self, file_path: str) -> None:
        def _pickle_load(file_path):
            with open(file_path, "rb") as file:
                return pickle.load(file)

        self.solar_forecast = _pickle_load(file_path)

    async def get_power_series_from_orchestrator(
        self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime | None, str | float | None]]:
        return []


class QSSolarProviderSolcast(QSSolarProvider):
    def __init__(self, solar: QSSolar, provider_name: str = "", **kwargs) -> None:
        super().__init__(solar=solar, domain=SOLCAST_SOLAR_DOMAIN, provider_name=provider_name, **kwargs)

    async def fill_orchestrators(self):
        """Return the orchestrators for the domain."""
        self.orchestrators = []
        self._orchestrator_health = {}
        entries = self.hass.config_entries.async_entries(self.domain)
        for entry in entries:
            try:
                orch = entry.runtime_data.coordinator
                self.orchestrators.append(orch)
                self._orchestrator_health[id(orch)] = True
            except AttributeError, TypeError:
                _LOGGER.debug("Skipping config entry %s for %s: runtime_data not ready", entry.entry_id, self.domain)

    async def get_power_series_from_orchestrator(
        self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime | None, str | float | None]]:
        solcast_api = orchestrator.solcast
        if hasattr(solcast_api, "data_forecasts"):
            data = solcast_api.data_forecasts
        elif hasattr(solcast_api, "_data_forecasts"):
            data = solcast_api._data_forecasts
        else:
            raise AttributeError(
                f"Solcast API object {solcast_api!r} exposes neither 'data_forecasts' nor '_data_forecasts'"
            )
        if start_time is None or end_time is None:
            return []
        if data is None:
            return []

        start_idx = bisect_left(data, start_time, key=itemgetter("period_start"))
        if start_idx >= len(data):
            return []

        if start_idx > 0:
            if data[start_idx]["period_start"] != start_time:
                start_idx -= 1

        end_idx = bisect_right(data, end_time, key=itemgetter("period_start")) - 1
        if end_idx < start_idx:
            return []

        return [
            (d["period_start"].astimezone(tz=pytz.UTC), 1000.0 * d["pv_estimate"])
            for d in data[start_idx : end_idx + 1]
            if d["pv_estimate"] is not None
        ]


class QSSolarProviderOpenWeather(QSSolarProvider):
    def __init__(self, solar: QSSolar, provider_name: str = "", **kwargs) -> None:
        super().__init__(solar=solar, domain=OPEN_METEO_SOLAR_DOMAIN, provider_name=provider_name, **kwargs)

    async def get_power_series_from_orchestrator(
        self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime | None, str | float | None]]:
        if orchestrator.data is None:
            return []
        data = orchestrator.data.watts
        if start_time is None or end_time is None:
            return []
        if data is None:
            return []

        data = [(t, p) for t, p in data.items()]
        data.sort(key=itemgetter(0))

        start_idx = bisect_left(data, start_time, key=itemgetter(0))
        if start_idx >= len(data):
            return []

        if start_idx > 0:
            if data[start_idx][0] != start_time:
                start_idx -= 1

        end_idx = bisect_right(data, end_time, key=itemgetter(0)) - 1
        if end_idx < start_idx:
            return []

        return [(d[0].astimezone(tz=pytz.UTC), float(d[1])) for d in data[start_idx : end_idx + 1] if d[1] is not None]


class QSSolarProviderForecastSolar(QSSolarProvider):
    def __init__(self, solar: QSSolar, provider_name: str = "", **kwargs) -> None:
        super().__init__(solar=solar, domain=FORECAST_SOLAR_DOMAIN, provider_name=provider_name, **kwargs)

    async def fill_orchestrators(self):
        """Return the orchestrators for the domain."""
        self.orchestrators = []
        self._orchestrator_health = {}
        entries = self.hass.config_entries.async_entries(self.domain)
        for entry in entries:
            try:
                orch = entry.runtime_data
                self.orchestrators.append(orch)
                self._orchestrator_health[id(orch)] = True
            except AttributeError, TypeError:
                _LOGGER.debug("Skipping config entry %s for %s: runtime_data not ready", entry.entry_id, self.domain)

    async def get_power_series_from_orchestrator(
        self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime | None, str | float | None]]:
        if orchestrator.data is None:
            return []
        data = orchestrator.data.watts
        if start_time is None or end_time is None:
            return []
        if data is None:
            return []

        data = [(t, p) for t, p in data.items()]
        data.sort(key=itemgetter(0))

        start_idx = bisect_left(data, start_time, key=itemgetter(0))
        if start_idx >= len(data):
            return []

        if start_idx > 0:
            if data[start_idx][0] != start_time:
                start_idx -= 1

        end_idx = bisect_right(data, end_time, key=itemgetter(0)) - 1
        if end_idx < start_idx:
            return []

        return [(d[0].astimezone(tz=pytz.UTC), float(d[1])) for d in data[start_idx : end_idx + 1] if d[1] is not None]
