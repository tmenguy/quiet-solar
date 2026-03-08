import logging
import pickle
from abc import abstractmethod
from bisect import bisect_left
from datetime import datetime, timedelta
from operator import itemgetter
from os.path import join

import aiofiles.os
import pytz

from ..const import CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, \
    SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_FORECAST_PROVIDER, OPEN_METEO_SOLAR_DOMAIN, DOMAIN, FLOATING_PERIOD_S, \
    CONF_TYPE_NAME_QSSolar, CONF_SOLAR_MAX_OUTPUT_POWER_VALUE, CONF_SOLAR_MAX_PHASE_AMPS, CONF_ACCURATE_POWER_SENSOR, \
    MAX_POWER_INFINITE, MAX_AMP_INFINITE, CONF_IS_3P
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice, align_time_series_and_values, get_slots_from_time_series, \
    get_value_from_time_series

_LOGGER = logging.getLogger(__name__)

class QSSolar(HADeviceMixin, AbstractDevice):

    conf_type_name = CONF_TYPE_NAME_QSSolar

    def __init__(self, **kwargs) -> None:
        self.solar_inverter_active_power = kwargs.pop(CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, None)
        self.solar_inverter_input_active_power = kwargs.pop(CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, None)
        self.solar_forecast_provider = kwargs.pop(CONF_SOLAR_FORECAST_PROVIDER, None)


        self.solar_max_output_power_value = kwargs.pop(CONF_SOLAR_MAX_OUTPUT_POWER_VALUE, MAX_POWER_INFINITE)
        self.solar_max_phase_amps = kwargs.pop(CONF_SOLAR_MAX_PHASE_AMPS, MAX_AMP_INFINITE)

        self.solar_forecast_provider_handler: QSSolarProvider | None = None
        kwargs[CONF_ACCURATE_POWER_SENSOR] = self.solar_inverter_active_power # to allow proper measurement

        home = kwargs.get("home", None)
        if home:
            kwargs[CONF_IS_3P] = home.physical_3p

        super().__init__(**kwargs)

        if self.solar_max_output_power_value == MAX_POWER_INFINITE and self.solar_max_phase_amps != MAX_AMP_INFINITE and self.home is not None:
            if self.physical_3p:
                self.solar_max_output_power_value = self.solar_max_phase_amps * self.home.voltage * 3.0
            else:
                self.solar_max_output_power_value = self.solar_max_phase_amps * self.home.voltage
        elif self.solar_max_output_power_value != MAX_POWER_INFINITE and self.solar_max_phase_amps == MAX_AMP_INFINITE and self.home is not None:
            if self.physical_3p:
                self.solar_max_phase_amps = self.solar_max_output_power_value / (self.home.voltage * 3.0)
            else:
                self.solar_max_phase_amps = self.solar_max_output_power_value / self.home.voltage


        self.attach_power_to_probe(self.solar_inverter_active_power)
        self.attach_power_to_probe(self.solar_inverter_input_active_power)

        if self.solar_forecast_provider is not None:
            _LOGGER.info(f"Creating solar forecast provider handler for {self.solar_forecast_provider}")
            if self.solar_forecast_provider == SOLCAST_SOLAR_DOMAIN:
                self.solar_forecast_provider_handler = QSSolarProviderSolcast(self)
            elif self.solar_forecast_provider == OPEN_METEO_SOLAR_DOMAIN:
                self.solar_forecast_provider_handler = QSSolarProviderOpenWeather(self)

        self.solar_production = 0
        self.inverter_output_power = 0

    def get_current_over_clamp_production_power(self) -> float:

        if self.solar_production > self.solar_max_output_power_value:
            return self.solar_production - self.solar_max_output_power_value

        return 0.0

    async def update_forecast(self, time: datetime) -> None:
        if self.solar_forecast_provider_handler is not None:
            await self.solar_forecast_provider_handler.update(time)

    def get_forecast(self, start_time: datetime, end_time: datetime | None) -> list[tuple[datetime | None, str | float | None]]:
        if self.solar_forecast_provider_handler is not None:
            return self.solar_forecast_provider_handler.get_forecast(start_time, end_time)
        return []

    def get_value_from_current_forecast(self, time: datetime) -> tuple[datetime | None, str | float | None]:
        if self.solar_forecast_provider_handler is not None:
            return self.solar_forecast_provider_handler.get_value_from_current_forecast(time)
        return (None, None)


    async def dump_for_debug(self, debug_path: str) -> None:

        if self.solar_forecast_provider_handler is not None:
            await self.solar_forecast_provider_handler.dump_for_debug(debug_path)





class QSSolarProvider:

    def __init__(self, solar: QSSolar | None, domain:str | None, **kwargs) -> None:
        self.solar = solar
        if solar is not None:
            self.hass = solar.hass
        else:
            self.hass = None
        self.orchestrators = []
        self.domain = domain
        self._latest_update_time : datetime | None = None
        self.solar_forecast : list[tuple[datetime | None, float | None]] = []

    def get_forecast(self, start_time: datetime, end_time: datetime | None) -> list[tuple[datetime | None, str | float | None]]:
        return get_slots_from_time_series(self.solar_forecast, start_time, end_time)

    def get_value_from_current_forecast(self, time: datetime) -> tuple[datetime | None, str | float | None]:
        res = get_value_from_time_series(self.solar_forecast, time)
        return res[0], res[1]


    async def fill_orchestrators(self):
        """Return the orchestrators for the domain."""
        self.orchestrators = []
        for entry_id, orchestrator in self.solar.hass.data.get(self.domain, {}).items():
            if orchestrator is not None:
                self.orchestrators.append(orchestrator)


    async def update(self, time: datetime) -> None:

        if len(self.orchestrators) == 0 or self._latest_update_time is None or (time - self._latest_update_time).total_seconds() > 15*60:

            await self.fill_orchestrators()

            validated = []
            for orchestrator in self.orchestrators:
                try:
                    await self.get_power_series_from_orchestrator(orchestrator, None, None)
                    validated.append(orchestrator)
                except Exception:
                    _LOGGER.warning("Invalid orchestrator %s for domain %s, skipping", orchestrator, self.domain)
            self.orchestrators = validated

            if len(self.orchestrators) > 0:
                self.solar_forecast: list[tuple[datetime | None, float | None]] = []
                self.solar_forecast = await self.extract_solar_forecast_from_data(time, period=FLOATING_PERIOD_S)

                if len(self.solar_forecast) > 0:
                    self._latest_update_time = time
            else:
                _LOGGER.error("No solar orchestrator found for domain %s", self.domain)


    async def extract_solar_forecast_from_data(self, start_time: datetime, period: float) -> list[
        tuple[datetime | None, float | None]]:

        # the period may be : FLOATING_PERIOD of course

        end_time = start_time + timedelta(seconds=period)

        vals = []

        for orchestrator in self.orchestrators:
            s = await self.get_power_series_from_orchestrator(orchestrator, start_time, end_time)
            if s:
                vals.append(s)

        if len(vals) == 0:
            return []

        # merge the data
        v_aggregated = vals[0]

        for v in vals[1:]:
            v_aggregated = align_time_series_and_values(v_aggregated, v, operation=lambda x, y: x + y)

        # _LOGGER.info(f"extract_solar_forecast_from_data for {self.domain} from {start_time} to {end_time} : {len(v_aggregated)}")

        return v_aggregated

    async def dump_for_debug(self, debug_path: str) -> None:

        storage_path =  debug_path
        await aiofiles.os.makedirs(storage_path, exist_ok=True)
        file_path = join(storage_path, "solar_forecast.pickle")

        def _pickle_save(file_path, obj):
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)

        await self.hass.async_add_executor_job(
            _pickle_save, file_path, self.solar_forecast
        )

    @abstractmethod
    async def get_power_series_from_orchestrator(self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None) -> list[
        tuple[datetime | None, str | float | None]]:
        """Return the power series from the orchestrator.

        When called with start_time=None and end_time=None, this acts as
        a validation probe: it must access the orchestrator's data source
        and raise if the structure is invalid, or return [] if valid.
        """


class QSSolarProviderSolcastDebug(QSSolarProvider):

    def __init__(self,  file_path:str, **kwargs) -> None:
        super().__init__(solar=None, domain=None, **kwargs)
        self.read_from_debug_dump(file_path)

    def read_from_debug_dump(self, file_path: str) -> None:

        def _pickle_load(file_path):
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        self.solar_forecast = _pickle_load(file_path)

    async def get_power_series_from_orchestrator(self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None) -> list[
        tuple[datetime | None, str | float | None]]:
        return []


class QSSolarProviderSolcast(QSSolarProvider):

    def __init__(self, solar: QSSolar, **kwargs) -> None:
        super().__init__(solar=solar, domain=SOLCAST_SOLAR_DOMAIN, **kwargs)

    async def fill_orchestrators(self):
        """Return the orchestrators for the domain."""
        self.orchestrators = []
        entries = self.hass.config_entries.async_entries(self.domain)
        for entry in entries:
            try:
                self.orchestrators.append(entry.runtime_data.coordinator)
            except (AttributeError, TypeError):
                pass

    async def get_power_series_from_orchestrator(self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None) -> list[
        tuple[datetime | None, str | float | None]]:
        # Support both the new public attribute (data_forecasts, solcast_solar >= v5)
        # and the old private one (_data_forecasts, solcast_solar < v5) for
        # backward compatibility. Raise AttributeError if neither exists so the
        # base-class validation pass can detect an invalid orchestrator.
        solcast_api = orchestrator.solcast
        if hasattr(solcast_api, "data_forecasts"):
            data = solcast_api.data_forecasts
        elif hasattr(solcast_api, "_data_forecasts"):
            data = solcast_api._data_forecasts
        else:
            raise AttributeError(
                f"Solcast API object {solcast_api!r} exposes neither "
                "'data_forecasts' nor '_data_forecasts'"
            )
        if start_time is None or end_time is None:
            return []
        if data is None:
            return []

        start_idx = bisect_left(data, start_time, key=itemgetter('period_start'))
        if start_idx >= len(data):
            return []

        if start_idx > 0:
            if data[start_idx]['period_start'] != start_time:
                start_idx -= 1

        end_idx = bisect_left(data, end_time, key=itemgetter('period_start'))
        if end_idx >= len(data):
            end_idx = len(data) - 1

        return [(d['period_start'].astimezone(tz=pytz.UTC), 1000.0 * d["pv_estimate"]) for d in data[start_idx:end_idx + 1]]


class QSSolarProviderOpenWeather(QSSolarProvider):

    def __init__(self, solar: QSSolar, **kwargs) -> None:
        super().__init__(solar=solar, domain=OPEN_METEO_SOLAR_DOMAIN, **kwargs)

    async def get_power_series_from_orchestrator(self, orchestrator, start_time: datetime | None = None, end_time: datetime | None = None) -> list[
        tuple[datetime | None, str | float | None]]:
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

        return [(d[0].astimezone(tz=pytz.UTC), float(d[1])) for d in data[start_idx:end_idx + 1]]

