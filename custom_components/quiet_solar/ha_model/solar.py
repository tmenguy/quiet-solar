import logging
from abc import abstractmethod
from bisect import bisect_left
from datetime import datetime, timedelta
from operator import itemgetter

import pytz
from homeassistant.core import callback, Event, EventStateChangedData
from homeassistant.helpers.event import async_track_state_change_event, async_track_utc_time_change

from ..const import CONF_SOLAR_INVERTER_ACTIVE_POWER_SENSOR, CONF_SOLAR_INVERTER_INPUT_POWER_SENSOR, \
    SOLCAST_SOLAR_DOMAIN, CONF_SOLAR_FORECAST_PROVIDER, OPEN_METEO_SOLAR_DOMAIN
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice, align_time_series_and_values, FLOATING_PERIOD

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



class QSSolarProvider:

    def __init__(self, solar: QSSolar, domain:str, **kwargs) -> None:
        self.solar = solar
        self.orchestrators = []
        self.domain = domain
        self._unsub = None

        for _, orchestrator in self.solar.hass.data.get(SOLCAST_SOLAR_DOMAIN, {}).items():
            _LOGGER.info(f"Adding orchestrator {orchestrator} for {self.domain}")
            self.orchestrators.append(orchestrator)

        self.solar_forecast: list[tuple[datetime | None, str | float | None, dict | None]] = []

        self.solar_forecast = self.extract_solar_forecast_from_data(datetime.now(tz=pytz.UTC), period=FLOATING_PERIOD)

        self.auto_update()

    def extract_solar_forecast_from_data(self, start_time: datetime, period: float) -> list[
        tuple[datetime | None, str | float | None, dict | None]]:

        # the period may be : FLOATING_PERIOD of course

        end_time = start_time + timedelta(seconds=period)

        vals = []

        for orchestrator in self.orchestrators:
            s = self.get_power_series_from_orchestrator(orchestrator, start_time, end_time)
            if s:
                vals.append(s)

        if len(vals) == 0:
            return []

        # merge the data
        v_aggregated = vals[0]

        for v in vals[1:]:
            v_aggregated = align_time_series_and_values(v_aggregated, v, operation=lambda x, y: x + y)

        return v_aggregated


    async def  _update_callback(self, time: datetime) -> None:
            self.solar_forecast = self.extract_solar_forecast_from_data(time, period=FLOATING_PERIOD)
            _LOGGER.info(f"Update solar forecast for {self.domain} num items {len(self.solar_forecast)} at {time}")

    def auto_update(self):


        async_track_utc_time_change(
            self.solar.hass,
            self._update_callback,
            minute=15,
        )

    @abstractmethod
    def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None, dict | None]]:
         """ Returns the power series from the orchestrator"""




class QSSolarProviderSolcast(QSSolarProvider):

    def __init__(self, solar: QSSolar, **kwargs) -> None:
        super().__init__(solar=solar, domain=SOLCAST_SOLAR_DOMAIN, **kwargs)

    def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None, dict | None]]:
        data = orchestrator.solcast._data_forecasts
        if data is not None:

            start_idx = bisect_left(data, start_time, key=itemgetter('period_start'))
            if start_idx > 0:
                if data[start_idx]['period_start'] != start_time:
                    start_idx -= 1

            end_idx = bisect_left(data, end_time, key=itemgetter('period_start'))
            if end_idx >= len(data):
                end_idx = len(data) - 1

            return [ (d['period_start'], d["pv_estimate"], {}) for d in data[start_idx:end_idx+1]]
        return []



class QSSolarProviderOpenWeather(QSSolarProvider):

    def __init__(self, solar: QSSolar, **kwargs) -> None:
        super().__init__(solar=solar, domain=OPEN_METEO_SOLAR_DOMAIN, **kwargs)

    def get_power_series_from_orchestrator(self, orchestrator, start_time:datetime, end_time:datetime) -> list[
        tuple[datetime | None, str | float | None, dict | None]]:
        data = orchestrator.forecast.data.watts

        if data is not None:

            data = [(t, p) for t, p in data.items()]
            data.sort(key=itemgetter(0))

            start_idx = bisect_left(data, start_time, key=itemgetter(0))
            if start_idx > 0:
                if data[start_idx][0] != start_time:
                    start_idx -= 1

            end_idx = bisect_left(data, end_time, key=itemgetter(0))
            if end_idx >= len(data):
                end_idx = len(data) - 1

            return [ (d[0], float(d[1]), {}) for d in data[start_idx:end_idx+1]]
        return []



