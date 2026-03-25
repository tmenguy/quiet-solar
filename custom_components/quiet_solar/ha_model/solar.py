from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from operator import itemgetter
from os.path import join
from typing import Any

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
from ..home_model.home_utils import (
    align_time_series_and_values,
    align_time_series_on_time_slots,
    get_slots_from_time_series,
    get_value_from_time_series,
)
from ..home_model.load import AbstractDevice

_LOGGER = logging.getLogger(__name__)

DAMPENING_A_MIN = 0.1
DAMPENING_A_MAX = 3.0
DAMPENING_MIN_DATA_POINTS = 3
DAMPENING_BUFFER_DAYS = 365  # How long to retain raw time-series (actuals + forecast snapshots)
DAMPENING_WINDOW_DAYS = 7  # Sliding window for dampening regression computation
DAMPENING_NUM_TOD_BUFFERS = 12
DAMPENING_TOD_HOURS = 2
DAMPENING_ACTUALS_STEP_MN = 5  # Record actual production every N minutes from sensor


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

        # Shared actuals buffer (single buffer for all providers)
        self._actuals_buffer: list[tuple[datetime, float]] = []
        self._last_actuals_record_time: datetime | None = None

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

    def record_actual_production(self, time: datetime, actual_values: list[tuple[datetime, float]]) -> None:
        """Record actual solar production into the shared continuous rolling buffer.

        New values replace existing entries at the same timestamp (merge-by-time).
        """
        if not actual_values:
            return

        cutoff = time - timedelta(days=DAMPENING_BUFFER_DAYS)
        merged: dict[datetime, float] = {ts: v for ts, v in self._actuals_buffer if ts >= cutoff}
        for ts, v in actual_values:
            if ts >= cutoff:
                merged[ts] = v
        self._actuals_buffer = sorted(merged.items(), key=itemgetter(0))

    def _record_actual_from_sensor(self, time: datetime) -> None:
        """Record actual production from the inverter input power sensor.

        Called during update_forecast. Reads get_average_sensor over
        the last DAMPENING_ACTUALS_STEP_MN minutes and pushes to the
        shared actuals buffer. Skips if called too soon after last recording.
        """
        step_seconds = DAMPENING_ACTUALS_STEP_MN * 60

        if self._last_actuals_record_time is not None:
            elapsed = (time - self._last_actuals_record_time).total_seconds()
            if elapsed < step_seconds:
                return

        if self.solar_inverter_input_active_power is None:
            return

        value = self.get_average_sensor(
            self.solar_inverter_input_active_power,
            step_seconds,
            time,
            min_val=0.0,
        )
        if value is None:
            return

        self._last_actuals_record_time = time
        self.record_actual_production(time, [(time, value)])

    async def update_forecast(self, time: datetime) -> None:
        """Update all providers and select active one."""
        for provider in self.solar_forecast_providers.values():
            try:
                await provider.update(time)
            except Exception:
                _LOGGER.exception("Error updating solar forecast provider %s", provider.provider_name)

        # Bootstrap shared actuals from history if empty (first startup only)
        QSSolarProvider._bootstrap_actuals_from_history(self, time)

        # Record actual production from sensor at regular intervals
        self._record_actual_from_sensor(time)

        # Daily scoring/dampening cycle at day boundary
        self._run_daily_scoring_cycle(time)

        if self._provider_mode == SOLAR_PROVIDER_MODE_AUTO:
            self.auto_select_best_provider()

        # Keep legacy handler in sync
        self.solar_forecast_provider_handler = self.active_provider

    def reset_dampening(self, time: datetime) -> None:
        """Clear all dampening buffers for all providers and re-bootstrap actuals."""
        # Clear shared actuals buffer and re-bootstrap from history
        self._actuals_buffer = []
        fallback = self.get_historical_solar_fallback(time)
        if fallback:
            self._actuals_buffer = list(fallback)
            _LOGGER.info(
                "Re-bootstrapped shared actuals buffer from historical data (%d points)",
                len(self._actuals_buffer),
            )
        self._last_actuals_record_time = None

        for provider in self.solar_forecast_providers.values():
            provider.reset_dampening_buffers()

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
            # Compute scores — always, regardless of dampening switch state
            provider.compute_score()
            # Compute dampening — always, regardless of dampening switch state
            provider.compute_dampening(self.solar_max_output_power_value)
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

        # Scoring
        self._last_score_date: datetime | None = None
        self.score_raw: float | None = None
        self.score_dampened: float | None = None

        # Dampening — resolution-independent rolling buffers
        self.dampening_enabled: bool = False
        self._dampening_coefficients: np.ndarray | None = None
        self._dampening_coefficients_per_tod: dict[int, np.ndarray | None] = {}
        self._last_dampening_date: datetime | None = None

        # Local actuals fallback only used in tests where solar=None
        self._local_actuals_buffer: list[tuple[datetime, float]] = []

        # Forecast snapshots: 12 rolling buffers (one per 2-hour time-of-day)
        self._forecast_buffers: list[list[tuple[datetime, float]]] = [[] for _ in range(DAMPENING_NUM_TOD_BUFFERS)]

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

    @property
    def _actuals_buffer(self) -> list[tuple[datetime, float]]:
        """Return the shared actuals buffer from the parent QSSolar.

        Falls back to a local buffer for test scenarios where solar=None.
        """
        if self.solar is not None:
            return self.solar._actuals_buffer
        return self._local_actuals_buffer

    @_actuals_buffer.setter
    def _actuals_buffer(self, value: list[tuple[datetime, float]]) -> None:
        if self.solar is not None:
            self.solar._actuals_buffer = value
        else:
            self._local_actuals_buffer = value

    def get_active_score(self) -> float | None:
        """Return the active score (dampened if enabled, raw otherwise)."""
        if self.dampening_enabled and self.score_dampened is not None:
            return self.score_dampened
        return self.score_raw

    def get_dampening_attributes(self) -> dict[str, Any]:
        """Return dampening info as a dict for HA sensor attributes.

        All internal data is stored in UTC.  User-facing hour labels and
        timestamps are converted to local time for display.
        """
        attrs: dict[str, Any] = {"dampening_enabled": self.dampening_enabled}
        if self._last_dampening_date is not None:
            # Display in local time for the user
            local_dt = self._last_dampening_date.astimezone(tz=None)
            attrs["last_dampening_date"] = local_dt.isoformat()
        if self._dampening_coefficients is not None and len(self._dampening_coefficients) > 0:
            attrs["coefficients_count"] = len(self._dampening_coefficients)
            attrs["avg_scale_factor"] = round(float(np.mean(self._dampening_coefficients[:, 0])), 4)
            attrs["avg_offset"] = round(float(np.mean(self._dampening_coefficients[:, 1])), 2)
        for buf_idx in range(DAMPENING_NUM_TOD_BUFFERS):
            # Internal buffer index is UTC-based; convert to local hour for display
            utc_hour = buf_idx * DAMPENING_TOD_HOURS
            utc_dt = datetime.now(pytz.UTC).replace(hour=utc_hour, minute=0, second=0, microsecond=0)
            local_hour = utc_dt.astimezone(tz=None).hour
            coeffs = self._dampening_coefficients_per_tod.get(buf_idx)
            if coeffs is not None and len(coeffs) > 0:
                attrs[f"tod_{local_hour:02d}h_scale"] = round(float(np.mean(coeffs[:, 0])), 4)
                attrs[f"tod_{local_hour:02d}h_offset"] = round(float(np.mean(coeffs[:, 1])), 2)
        return attrs

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
        if self.dampening_enabled and self._dampening_coefficients is not None:
            return self._apply_dampening(self.solar_forecast)
        return self.solar_forecast

    def _apply_dampening(
        self, forecast: list[tuple[datetime | None, float | None]]
    ) -> list[tuple[datetime | None, float | None]]:
        """Apply MOS dampening coefficients to a forecast time series.

        Coefficients are indexed by forecast position (matching the slot
        boundaries used during computation via align_time_series_on_time_slots).
        """
        if not forecast or self._dampening_coefficients is None:
            return forecast

        result = []
        for i, (ts, value) in enumerate(forecast):
            if ts is None or value is None:
                result.append((ts, value))
                continue
            if i < len(self._dampening_coefficients):
                a_k, b_k = self._dampening_coefficients[i]
                if np.isfinite(a_k) and np.isfinite(b_k):
                    dampened = max(0.0, a_k * value + b_k)
                    result.append((ts, dampened))
                else:
                    result.append((ts, value))
            else:
                result.append((ts, value))
        return result

    @staticmethod
    def _tod_buffer_index(time: datetime) -> int:
        """Return the 2-hour time-of-day buffer index (0..11) for a given time.

        Uses UTC hour to avoid DST discontinuities in stored data.
        """
        return time.hour // DAMPENING_TOD_HOURS

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
        """Record current forecast into the matching 2-hour rolling buffer.

        Before appending, removes any existing buffer entries whose timestamp
        falls within the new snapshot's range, preventing stale overlapping data.
        """
        if not self.solar_forecast:
            return

        buf_idx = self._tod_buffer_index(time)
        cutoff = time - timedelta(days=DAMPENING_BUFFER_DAYS)

        # Copy current forecast (strip None values)
        snapshot: list[tuple[datetime, float]] = [
            (ts, v) for ts, v in self.solar_forecast if ts is not None and v is not None
        ]
        if not snapshot:
            return

        # Remove existing entries that overlap with the new snapshot range
        first_ts = snapshot[0][0]
        buf = self._forecast_buffers[buf_idx]
        buf = [(ts, v) for ts, v in buf if ts < first_ts and ts >= cutoff]
        buf.extend(snapshot)
        self._forecast_buffers[buf_idx] = buf

    def reset_dampening_buffers(self) -> None:
        """Clear dampening buffers (forecasts + coefficients). Actuals buffer is shared on QSSolar."""
        self._forecast_buffers = [[] for _ in range(DAMPENING_NUM_TOD_BUFFERS)]
        self._dampening_coefficients = None
        self._dampening_coefficients_per_tod = {}
        self.score_raw = None
        self.score_dampened = None

    @staticmethod
    def _bootstrap_actuals_from_history(solar: QSSolar, time: datetime) -> None:
        """On first startup with empty shared buffer, bootstrap from HA sensor history."""
        if solar._actuals_buffer:
            return
        fallback = solar.get_historical_solar_fallback(time)
        if fallback:
            solar._actuals_buffer = list(fallback)
            _LOGGER.info(
                "Bootstrapped shared actuals buffer from historical data (%d points)",
                len(solar._actuals_buffer),
            )

    def compute_score(self) -> None:
        """Compute MAE scores using the current forecast and actuals buffer."""
        if not self._actuals_buffer or not self.solar_forecast:
            return

        # Build forecast slot boundaries from current forecast
        fc_times = [ts for ts, _ in self.solar_forecast if ts is not None]
        if len(fc_times) < 2:
            return

        # Align actuals onto forecast slots
        aligned_actuals = align_time_series_on_time_slots(self._actuals_buffer, fc_times)
        aligned_forecast = align_time_series_on_time_slots(
            [(ts, v) for ts, v in self.solar_forecast if ts is not None and v is not None],
            fc_times,
        )

        if not aligned_actuals or not aligned_forecast:
            return

        # Compute raw MAE (undampened forecast vs actuals)
        self.score_raw = self._compute_mae_from_aligned(aligned_forecast, aligned_actuals)

        # Compute dampened MAE (dampened forecast vs actuals)
        if self._dampening_coefficients is not None:
            dampened_fc = self._apply_dampening(self.solar_forecast)
            dampened_fc_clean = [(ts, v) for ts, v in dampened_fc if ts is not None and v is not None]
            aligned_dampened = align_time_series_on_time_slots(dampened_fc_clean, fc_times)
            if aligned_dampened:
                self.score_dampened = self._compute_mae_from_aligned(aligned_dampened, aligned_actuals)
        else:
            self.score_dampened = self.score_raw

    @staticmethod
    def _compute_mae_from_aligned(
        forecast_aligned: list[tuple[datetime, float]],
        actuals_aligned: list[tuple[datetime, float]],
    ) -> float | None:
        """Compute MAE from two aligned time series (same slot boundaries)."""
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

    def compute_dampening(self, max_power: float) -> None:
        """Compute MOS dampening coefficients from rolling history buffers.

        Selects the forecast buffer closest to the current time-of-day,
        shifts historical timestamps to align with today's forecast,
        and computes per-slot linear regression coefficients.
        """
        if not self.solar_forecast or not self._actuals_buffer:
            return

        fc_times = [ts for ts, _ in self.solar_forecast if ts is not None]
        if len(fc_times) < 2:
            return

        # All timestamps are UTC — consistent for storage and DST-safe
        now = fc_times[0]
        buf_idx = self._tod_buffer_index(now)
        forecast_buf = self._forecast_buffers[buf_idx]

        if not forecast_buf:
            return

        # Number of aligned slots = number of consecutive time-boundary pairs
        num_slots = len(fc_times) - 1
        coeffs = np.ones((num_slots, 2), dtype=np.float64)
        coeffs[:, 1] = 0.0  # identity: a=1, b=0

        b_bound = max_power * 0.3

        # Group historical forecast data by day, shift to today, and align
        today_date = now.date()
        window_start = today_date - timedelta(days=DAMPENING_WINDOW_DAYS)
        hist_days = self._group_by_day(forecast_buf)

        fc_values_per_slot: list[list[float]] = [[] for _ in range(num_slots)]
        ac_values_per_slot: list[list[float]] = [[] for _ in range(num_slots)]

        for day_date, day_fc in hist_days.items():
            if day_date == today_date or day_date < window_start:
                continue
            # Shift timestamps to today
            day_offset = today_date - day_date
            shifted_fc = [(ts + day_offset, v) for ts, v in day_fc]

            # Align both onto current forecast time slots
            aligned_fc = align_time_series_on_time_slots(shifted_fc, fc_times)
            # For actuals, shift actuals from that day similarly
            day_start = datetime.combine(day_date, datetime.min.time(), tzinfo=now.tzinfo)
            day_end = day_start + timedelta(days=1)
            day_actuals = [(ts, v) for ts, v in self._actuals_buffer if day_start <= ts < day_end]
            if not day_actuals:
                continue
            shifted_actuals = [(ts + day_offset, v) for ts, v in day_actuals]
            aligned_ac = align_time_series_on_time_slots(shifted_actuals, fc_times)

            n = min(len(aligned_fc), len(aligned_ac), num_slots)
            for k in range(n):
                fc_values_per_slot[k].append(aligned_fc[k][1])
                ac_values_per_slot[k].append(aligned_ac[k][1])

        for k in range(num_slots):
            fc_arr = np.array(fc_values_per_slot[k])
            ac_arr = np.array(ac_values_per_slot[k])

            if len(fc_arr) == 0:
                continue

            # Nighttime guard
            if np.all(fc_arr == 0) and np.all(ac_arr == 0):
                continue

            # Daytime filter
            daytime_mask = (fc_arr > 0) | (ac_arr > 0)
            if np.sum(daytime_mask) < DAMPENING_MIN_DATA_POINTS:
                continue

            fc_fit = fc_arr[daytime_mask]
            ac_fit = ac_arr[daytime_mask]

            try:
                a_k, b_k = np.polyfit(fc_fit, ac_fit, deg=1)
            except (np.linalg.LinAlgError, ValueError):
                continue

            a_k = float(np.clip(a_k, DAMPENING_A_MIN, DAMPENING_A_MAX))
            b_k = float(np.clip(b_k, -b_bound, b_bound))

            coeffs[k, 0] = a_k
            coeffs[k, 1] = b_k

        self._dampening_coefficients = coeffs
        self._dampening_coefficients_per_tod[buf_idx] = coeffs
        self._last_dampening_date = datetime.now(pytz.UTC)

    @staticmethod
    def _group_by_day(
        time_series: list[tuple[datetime, float]],
    ) -> dict[Any, list[tuple[datetime, float]]]:
        """Group a time series by date."""
        groups: dict[Any, list[tuple[datetime, float]]] = {}
        for ts, v in time_series:
            d = ts.date()
            if d not in groups:
                groups[d] = []
            groups[d].append((ts, v))
        return groups

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
            except (AttributeError, TypeError):
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
            except (AttributeError, TypeError):
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
