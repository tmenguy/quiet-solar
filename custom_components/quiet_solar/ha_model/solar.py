from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from bisect import bisect_left
from datetime import datetime, timedelta
from operator import itemgetter
from os.path import join

import aiofiles.os
import numpy as np
import pytz

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
    DOMAIN,
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
)
from ..ha_model.device import HADeviceMixin
from ..home_model.load import (
    AbstractDevice,
    align_time_series_and_values,
    get_slots_from_time_series,
    get_value_from_time_series,
)

_LOGGER = logging.getLogger(__name__)

DAMPENING_A_MIN = 0.1
DAMPENING_A_MAX = 3.0
DAMPENING_MIN_DATA_POINTS = 3
DAMPENING_HISTORY_DAYS = 7


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

        self._last_daily_scoring_date: datetime | None = None
        self.solar_production = 0
        self.inverter_output_power = 0

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
        forecast_handler = getattr(self.home, "_consumption_forecast", None)
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

        # Daily scoring/dampening cycle at day boundary
        self._run_daily_scoring_cycle(time)

        if self._provider_mode == SOLAR_PROVIDER_MODE_AUTO:
            self.auto_select_best_provider()

        # Keep legacy handler in sync
        self.solar_forecast_provider_handler = self.active_provider

    def _run_daily_scoring_cycle(self, time: datetime) -> None:
        """Run scoring/dampening pipeline once per day (at midnight boundary)."""
        local_now = time.astimezone(tz=None) if time.tzinfo else time
        local_date = local_now.date()

        if self._last_daily_scoring_date is not None and self._last_daily_scoring_date >= local_date:
            return

        self._last_daily_scoring_date = local_date

        for name, provider in self.solar_forecast_providers.items():
            # Record forecast snapshot for the ending day
            provider.record_forecast_snapshot(time)
            # Compute scores
            provider.compute_score()
            # Compute dampening if enabled
            if provider.dampening_enabled:
                provider.compute_dampening(self.solar_max_output_power_value)
            # Advance to next day slot
            provider.advance_history_day()
            _LOGGER.debug(
                "Daily scoring cycle for provider %s: raw=%.2f, dampened=%s",
                name,
                provider.score_raw if provider.score_raw is not None else float("nan"),
                provider.score_dampened,
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
        """Return platforms including select and switch for solar."""
        from homeassistant.const import Platform

        platforms = super().get_platforms()
        platforms_set = set(platforms)
        if self.solar_forecast_providers:
            platforms_set.add(Platform.SELECT)
            platforms_set.add(Platform.SWITCH)
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

        # Scoring (Task 6)
        self.forecast_step_seconds: int | None = None
        self._steps_per_day: int | None = None
        self._forecast_history: np.ndarray | None = None  # shape (7, steps_per_day)
        self._actual_history: np.ndarray | None = None  # shape (7, steps_per_day)
        self._history_day_index: int = 0
        self._last_score_date: datetime | None = None
        self.score_raw: float | None = None
        self.score_dampened: float | None = None

        # Dampening (Task 7)
        self.dampening_enabled: bool = False
        self._dampening_coefficients: np.ndarray | None = None  # shape (steps_per_day, 2)
        self._last_dampening_date: datetime | None = None

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
        """Return the active score (dampened if enabled, raw otherwise)."""
        if self.dampening_enabled and self.score_dampened is not None:
            return self.score_dampened
        return self.score_raw

    def get_forecast(
        self, start_time: datetime, end_time: datetime | None
    ) -> list[tuple[datetime | None, str | float | None]]:
        forecast = self._get_effective_forecast()
        return get_slots_from_time_series(forecast, start_time, end_time)

    def get_value_from_current_forecast(self, time: datetime) -> tuple[datetime | None, str | float | None]:
        forecast = self._get_effective_forecast()
        res = get_value_from_time_series(forecast, time)
        return res[0], res[1]

    def _get_effective_forecast(self) -> list[tuple[datetime | None, float | None]]:
        """Return dampened forecast if enabled, otherwise raw."""
        if self.dampening_enabled and self._dampening_coefficients is not None and self._steps_per_day is not None:
            return self._apply_dampening(self.solar_forecast)
        return self.solar_forecast

    def _apply_dampening(
        self, forecast: list[tuple[datetime | None, float | None]]
    ) -> list[tuple[datetime | None, float | None]]:
        """Apply MOS dampening to a forecast time series."""
        if not forecast or self._dampening_coefficients is None or self._steps_per_day is None:
            return forecast

        result = []
        for ts, value in forecast:
            if ts is None or value is None:
                result.append((ts, value))
                continue
            step_idx = self._time_to_step_index(ts)
            if step_idx is not None and 0 <= step_idx < len(self._dampening_coefficients):
                a_k, b_k = self._dampening_coefficients[step_idx]
                if np.isfinite(a_k) and np.isfinite(b_k):
                    dampened = max(0.0, a_k * value + b_k)
                    result.append((ts, dampened))
                else:
                    result.append((ts, value))
            else:
                result.append((ts, value))
        return result

    def _time_to_step_index(self, ts: datetime) -> int | None:
        """Convert a timestamp to a step index within the day."""
        if self.forecast_step_seconds is None or self.forecast_step_seconds <= 0:
            return None
        seconds_since_midnight = ts.hour * 3600 + ts.minute * 60 + ts.second
        return seconds_since_midnight // self.forecast_step_seconds

    def _detect_step_size(self) -> None:
        """Detect the forecast temporal resolution using median of consecutive gaps."""
        if len(self.solar_forecast) < 2:
            return
        gaps = []
        for i in range(min(len(self.solar_forecast) - 1, 10)):
            ts0 = self.solar_forecast[i][0]
            ts1 = self.solar_forecast[i + 1][0]
            if ts0 is not None and ts1 is not None:
                gap_s = int((ts1 - ts0).total_seconds())
                if gap_s > 0:
                    gaps.append(gap_s)
        if gaps:
            step_s = int(np.median(gaps))
            if step_s > 0:
                old_steps = self._steps_per_day
                self.forecast_step_seconds = step_s
                self._steps_per_day = 86400 // step_s
                # Invalidate dampening coefficients if step size changed
                if (
                    old_steps is not None
                    and old_steps != self._steps_per_day
                    and self._dampening_coefficients is not None
                ):
                    _LOGGER.info(
                        "Step size changed for %s (%s -> %s steps/day), invalidating dampening coefficients",
                        self.provider_name,
                        old_steps,
                        self._steps_per_day,
                    )
                    self._dampening_coefficients = None

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
            or (time - self._latest_update_time).total_seconds() > 15 * 60
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

                    # Detect step size on first successful forecast
                    if self.forecast_step_seconds is None:
                        self._detect_step_size()

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
                    _LOGGER.info(
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

    def record_forecast_snapshot(self, time: datetime) -> None:
        """Record current forecast into the 7-day history buffer for scoring/dampening."""
        if self._steps_per_day is None or self.forecast_step_seconds is None:
            return
        if not self.solar_forecast:
            return

        # Initialize buffers on first call
        if self._forecast_history is None:
            self._forecast_history = np.full((DAMPENING_HISTORY_DAYS, self._steps_per_day), np.nan, dtype=np.float32)
            self._actual_history = np.full((DAMPENING_HISTORY_DAYS, self._steps_per_day), np.nan, dtype=np.float32)

        # Check buffer shape matches current step size
        if self._forecast_history.shape[1] != self._steps_per_day:
            self._forecast_history = np.full((DAMPENING_HISTORY_DAYS, self._steps_per_day), np.nan, dtype=np.float32)
            self._actual_history = np.full((DAMPENING_HISTORY_DAYS, self._steps_per_day), np.nan, dtype=np.float32)
            self._history_day_index = 0

        day_slot = self._history_day_index % DAMPENING_HISTORY_DAYS

        # Clear the slot
        self._forecast_history[day_slot, :] = np.nan

        # Fill from current forecast
        for ts, value in self.solar_forecast:
            if ts is None or value is None:
                continue
            step_idx = self._time_to_step_index(ts)
            if step_idx is not None and 0 <= step_idx < self._steps_per_day:
                self._forecast_history[day_slot, step_idx] = value

    def record_actual_production(self, time: datetime, actual_values: list[tuple[datetime, float]]) -> None:
        """Record actual solar production into the history buffer."""
        if self._steps_per_day is None or self.forecast_step_seconds is None:
            return
        if self._actual_history is None:
            return

        day_slot = self._history_day_index % DAMPENING_HISTORY_DAYS
        self._actual_history[day_slot, :] = np.nan

        for ts, value in actual_values:
            step_idx = self._time_to_step_index(ts)
            if step_idx is not None and 0 <= step_idx < self._steps_per_day:
                self._actual_history[day_slot, step_idx] = value

    def advance_history_day(self) -> None:
        """Move to next day slot in the rolling buffer (call at midnight)."""
        self._history_day_index += 1

    def compute_score(self) -> None:
        """Compute MAE score over the 7-day window."""
        if self._forecast_history is None or self._actual_history is None:
            return

        # Raw score
        self.score_raw = self._compute_mae(self._forecast_history, self._actual_history)

        # Dampened score
        if self.dampening_enabled and self._dampening_coefficients is not None:
            dampened_forecasts = self._apply_dampening_to_array(self._forecast_history)
            self.score_dampened = self._compute_mae(dampened_forecasts, self._actual_history)

    def _compute_mae(self, forecasts: np.ndarray, actuals: np.ndarray) -> float | None:
        """Compute MAE over daytime steps, excluding all-NaN steps."""
        # Mask: valid where both have values and at least one is > 0
        valid = ~np.isnan(forecasts) & ~np.isnan(actuals)
        daytime = (forecasts > 0) | (actuals > 0)
        mask = valid & daytime

        if not np.any(mask):
            return None

        errors = np.abs(forecasts[mask] - actuals[mask])
        return float(np.mean(errors))

    def _apply_dampening_to_array(self, forecasts: np.ndarray) -> np.ndarray:
        """Apply dampening coefficients to a forecast array."""
        if self._dampening_coefficients is None:
            return forecasts
        if self._dampening_coefficients.shape[0] != forecasts.shape[1]:
            return forecasts
        result = np.copy(forecasts)
        for k in range(result.shape[1]):
            a_k, b_k = self._dampening_coefficients[k]
            col = result[:, k]
            valid = ~np.isnan(col)
            col[valid] = np.maximum(0.0, a_k * col[valid] + b_k)
        return result

    def compute_dampening(self, max_power: float) -> None:
        """Compute MOS dampening coefficients from 7-day history."""
        if self._forecast_history is None or self._actual_history is None or self._steps_per_day is None:
            return

        coeffs = np.ones((self._steps_per_day, 2), dtype=np.float64)
        coeffs[:, 1] = 0.0  # identity: a=1, b=0

        b_bound = max_power * 0.3

        for k in range(self._steps_per_day):
            fc_col = self._forecast_history[:, k]
            ac_col = self._actual_history[:, k]

            valid = ~np.isnan(fc_col) & ~np.isnan(ac_col)
            fc_valid = fc_col[valid]
            ac_valid = ac_col[valid]

            # Nighttime guard
            if len(fc_valid) == 0 or (np.all(fc_valid == 0) and np.all(ac_valid == 0)):
                continue

            # Minimum data guard: need points where forecast > 0 or actual > 0
            daytime_mask = (fc_valid > 0) | (ac_valid > 0)
            if np.sum(daytime_mask) < DAMPENING_MIN_DATA_POINTS:
                continue

            fc_fit = fc_valid[daytime_mask]
            ac_fit = ac_valid[daytime_mask]

            try:
                a_k, b_k = np.polyfit(fc_fit, ac_fit, deg=1)
            except np.linalg.LinAlgError, ValueError:
                continue

            # Coefficient bounds
            a_k = float(np.clip(a_k, DAMPENING_A_MIN, DAMPENING_A_MAX))
            b_k = float(np.clip(b_k, -b_bound, b_bound))

            coeffs[k, 0] = a_k
            coeffs[k, 1] = b_k

        self._dampening_coefficients = coeffs

    def get_dampening_persistence_path(self) -> str | None:
        """Return path for persisting dampening data."""
        if self.solar is None or self.hass is None:
            return None
        storage_dir = join(self.hass.config.path(), DOMAIN)
        safe_name = self.provider_name.replace(" ", "_").replace("/", "_")
        return join(storage_dir, f"dampening_{safe_name}.npy")

    async def save_dampening(self) -> None:
        """Persist dampening coefficients and steps_per_day to disk."""
        path = self.get_dampening_persistence_path()
        if path is None or self._dampening_coefficients is None or self._steps_per_day is None:
            return

        storage_dir = join(self.hass.config.path(), DOMAIN)
        await aiofiles.os.makedirs(storage_dir, exist_ok=True)

        data = {
            "coefficients": self._dampening_coefficients,
            "steps_per_day": self._steps_per_day,
        }

        def _save(p, d):
            np.save(p, d, allow_pickle=True)

        await self.hass.async_add_executor_job(_save, path, data)

    async def load_dampening(self) -> None:
        """Load dampening coefficients from disk."""
        path = self.get_dampening_persistence_path()
        if path is None:
            return

        def _load(p):
            return np.load(p, allow_pickle=True).item()

        try:
            data = await self.hass.async_add_executor_job(_load, path)
        except OSError, ValueError, EOFError:
            return

        stored_steps = data.get("steps_per_day")
        coeffs = data.get("coefficients")
        if stored_steps is None or coeffs is None:
            return

        # Discard if step size doesn't match current provider resolution
        if self._steps_per_day is not None and stored_steps != self._steps_per_day:
            _LOGGER.info(
                "Dampening data for %s has mismatched step size (%s vs %s), reinitializing",
                self.provider_name,
                stored_steps,
                self._steps_per_day,
            )
            return

        self._dampening_coefficients = coeffs
        self._steps_per_day = stored_steps

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

        end_idx = bisect_left(data, end_time, key=itemgetter("period_start"))
        if end_idx >= len(data):
            end_idx = len(data) - 1

        return [
            (d["period_start"].astimezone(tz=pytz.UTC), 1000.0 * d["pv_estimate"])
            for d in data[start_idx : end_idx + 1]
        ]


class QSSolarProviderOpenWeather(QSSolarProvider):
    def __init__(self, solar: QSSolar, provider_name: str = "", **kwargs) -> None:
        super().__init__(solar=solar, domain=OPEN_METEO_SOLAR_DOMAIN, provider_name=provider_name, **kwargs)

    async def get_power_series_from_orchestrator(
        self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime | None, str | float | None]]:
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

        end_idx = bisect_left(data, end_time, key=itemgetter(0))
        if end_idx >= len(data):
            end_idx = len(data) - 1

        return [(d[0].astimezone(tz=pytz.UTC), float(d[1])) for d in data[start_idx : end_idx + 1]]


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

        end_idx = bisect_left(data, end_time, key=itemgetter(0))
        if end_idx >= len(data):
            end_idx = len(data) - 1

        return [(d[0].astimezone(tz=pytz.UTC), float(d[1])) for d in data[start_idx : end_idx + 1]]
