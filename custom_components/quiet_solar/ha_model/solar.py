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
    SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_FORECAST_PROVIDER, OPEN_METEO_SOLAR_DOMAIN, DOMAIN, FLOATING_PERIOD_S
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice, align_time_series_and_values, get_slots_from_time_serie

_LOGGER = logging.getLogger(__name__)

class QSSolar(HADeviceMixin, AbstractDevice):

    def __init__(self, **kwargs) -> None:
        self.solar_inverter_active_power = kwargs.pop(CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, None)
        self.solar_inverter_input_active_power = kwargs.pop(CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, None)
        self.solar_forecast_provider = kwargs.pop(CONF_SOLAR_FORECAST_PROVIDER, None)
        self.solar_forecast_provider_handler: QSSolarProvider | None = None
        super().__init__(**kwargs)

        self.attach_power_to_probe(self.solar_inverter_active_power)
        self.attach_power_to_probe(self.solar_inverter_input_active_power)

        if self.solar_forecast_provider is not None:
            _LOGGER.info(f"Creating solar forecast provider handler for {self.solar_forecast_provider}")
            if self.solar_forecast_provider == SOLCAST_SOLAR_DOMAIN:
                self.solar_forecast_provider_handler = QSSolarProviderSolcast(self)
            elif self.solar_forecast_provider == OPEN_METEO_SOLAR_DOMAIN:
                self.solar_forecast_provider_handler = QSSolarProviderOpenWeather(self)

    async def update_forecast(self, time: datetime) -> None:
        if self.solar_forecast_provider_handler is not None:
            await self.solar_forecast_provider_handler.update(time)

    def get_forecast(self, start_time: datetime, end_time: datetime | None) -> list[tuple[datetime | None, str | float | None]]:
        if self.solar_forecast_provider_handler is not None:
            return self.solar_forecast_provider_handler.get_forecast(start_time, end_time)
        return []

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
        return get_slots_from_time_serie(self.solar_forecast, start_time, end_time)


    def is_orchestrator(self, entity_id, orchestrator) -> bool:
        return True

    async def fill_orchestrators(self):
        """ Returns the orchestrators for the domain """
        self.orchestrators = []
        for entry_id, orchestrator in self.solar.hass.data.get(self.domain, {}).items():
            # _LOGGER.info(f"Adding orchestrator {orchestrator} for {self.domain}")
            if orchestrator is not None and self.is_orchestrator(entry_id, orchestrator):
                self.orchestrators.append(orchestrator)
                    # _LOGGER.info(f"YES Adding solar orchestrator {orchestrator} for {self.domain} and key {entry_id}")
            # else:
                    # _LOGGER.info(f"NOT Adding solar orchestrator {orchestrator} for {self.domain} and key {entry_id}")


    async def update(self, time: datetime) -> None:

        if len(self.orchestrators) == 0 or self._latest_update_time is None or (time - self._latest_update_time).total_seconds() > 15*60:

            await self.fill_orchestrators()

            if len(self.orchestrators) > 0:
                self.solar_forecast: list[tuple[datetime | None, float | None]] = []
                self.solar_forecast = await self.extract_solar_forecast_from_data(time, period=FLOATING_PERIOD_S)

                if len(self.solar_forecast) > 0:

                    # the value are "on" the timing, and we need more the value between the timing and the next one (slot)
                    prev_value = self.solar_forecast[0][1]
                    for i in range(1, len(self.solar_forecast)):
                        self.solar_forecast[i-1] = (self.solar_forecast[i-1][0], (self.solar_forecast[i][1] + prev_value)/2.0)
                        prev_value = self.solar_forecast[i][1]

                    self._latest_update_time = time
            else:
                _LOGGER.error(f"No solar orchestrator found for domain {self.domain}")


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
    async def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None]]:
         """ Returns the power series from the orchestrator"""


class QSSolarProviderSolcastDebug(QSSolarProvider):

    def __init__(self,  file_path:str, **kwargs) -> None:
        super().__init__(solar=None, domain=None, **kwargs)
        self.read_from_debug_dump(file_path)

    def read_from_debug_dump(self, file_path: str) -> None:

        def _pickle_load(file_path):
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        self.solar_forecast = _pickle_load(file_path)

    async def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None]]:
        return []


class QSSolarProviderSolcast(QSSolarProvider):

    def __init__(self, solar: QSSolar, **kwargs) -> None:
        super().__init__(solar=solar, domain=SOLCAST_SOLAR_DOMAIN, **kwargs)


    async def fill_orchestrators(self):
        """ Returns the orchestrators for the domain """
        self.orchestrators = []

        entries = self.hass.config_entries.async_entries(self.domain)

        for entry in entries:
            try:
                orchestrator = entry.runtime_data.coordinator
                # just to check we have what we need
                data = orchestrator.solcast._data_forecasts
                self.orchestrators.append(orchestrator)
            except:
                pass


    async def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None]]:
        data = orchestrator.solcast._data_forecasts
        if data is not None:

            start_idx = bisect_left(data, start_time, key=itemgetter('period_start'))
            if start_idx >= len(data):
                return []

            if start_idx > 0:
                if data[start_idx]['period_start'] != start_time:
                    start_idx -= 1

            end_idx = bisect_left(data, end_time, key=itemgetter('period_start'))
            if end_idx >= len(data):
                end_idx = len(data) - 1

            return [ (d['period_start'].astimezone(tz=pytz.UTC), 1000.0*d["pv_estimate"]) for d in data[start_idx:end_idx+1]]
        return []


class QSSolarProviderOpenWeather(QSSolarProvider):

    def __init__(self, solar: QSSolar, **kwargs) -> None:
        super().__init__(solar=solar, domain=OPEN_METEO_SOLAR_DOMAIN, **kwargs)

    def is_orchestrator(self, entity_id, orchestrator) -> bool:
        try:
            data = orchestrator.data.watts
        except:
            return False
        return True

    async def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None]]:
        data = orchestrator.data.watts

        if data is not None:

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

            return [ (d[0].astimezone(tz=pytz.UTC), float(d[1])) for d in data[start_idx:end_idx+1]]
        return []

