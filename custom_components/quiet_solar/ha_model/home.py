import logging
import pickle
from enum import StrEnum
from operator import itemgetter
from os.path import join
from typing import Mapping, Any
from datetime import datetime, timedelta

import aiofiles.os
import pytz
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder.models import LazyState
from homeassistant.components.recorder import get_instance as recorder_get_instance
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE

from ..const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    HOME_CONSUMPTION_SENSOR, HOME_NON_CONTROLLED_CONSUMPTION_SENSOR, HOME_AVAILABLE_POWER_SENSOR, DOMAIN, \
    SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, FLOATING_PERIOD_S, CONF_HOME_START_OFF_PEAK_RANGE_1, \
    CONF_HOME_END_OFF_PEAK_RANGE_1, CONF_HOME_START_OFF_PEAK_RANGE_2, CONF_HOME_END_OFF_PEAK_RANGE_2, \
    CONF_HOME_PEAK_PRICE, CONF_HOME_OFF_PEAK_PRICE, QSForecastHomeNonControlledSensors, QSForecastSolarSensors, \
    FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
from ..ha_model.battery import QSBattery
from ..ha_model.car import QSCar
from ..ha_model.charger import QSChargerGeneric

from ..ha_model.device import HADeviceMixin, get_average_sensor, convert_power_to_w
from ..ha_model.solar import QSSolar
from ..home_model.commands import LoadCommand, CMD_IDLE
from ..home_model.load import AbstractLoad, AbstractDevice, get_slots_from_time_serie
from ..home_model.solver import PeriodSolver

import numpy as np
import numpy.typing as npt

class QSHomeMode(StrEnum):
    HOME_MODE_OFF = "home_mode_off"
    HOME_MODE_SENSORS_ONLY = "home_mode_sensors_only"
    HOME_MODE_CHARGER_ONLY = "home_mode_charger_only"
    HOME_MODE_ON = "home_mode_on"


_LOGGER = logging.getLogger(__name__)




MAX_SENSOR_HISTORY_S = 60*60*24*7

POWER_ALIGNEMENT_TOLERANCE_S = 120

class QSforecastValueSensor:

    _stored_values: list[tuple[datetime, float]] = []

    @classmethod
    def get_probers(cls, getter, names_and_duration):

        probers = {}

        for name, duration_s in names_and_duration.items():
            probers[name] = cls(name, duration_s, getter)

        return probers




    def __init__(self, name, duration_s, forecast_getter):
        self._stored_values = []
        self._getter = forecast_getter
        self._delta = timedelta(seconds=duration_s)
        self.name = name


    def push_and_get(self, time: datetime) -> float | None:
        future_time = time + self._delta
        future_val = self._getter(future_time)
        if future_val:
            #sorted by nature
            self._stored_values.append((future_val[0][0], future_val[0][1]))

        if not self._stored_values:
            return None

        value = None
        # find the last value before time
        num_trash = 0
        for t, v in self._stored_values:
            if t <= time:
                value = v
                num_trash += 1
            else:
                break

        if num_trash > 1:
            self._stored_values = self._stored_values[num_trash - 1:]


        return value



class QSHome(HADeviceMixin, AbstractDevice):

    _battery: QSBattery = None
    voltage: int = 230

    _chargers : list[QSChargerGeneric] = []
    _cars: list[QSCar] = []

    _all_devices : list[HADeviceMixin] = []
    _solar_plant: QSSolar | None = None
    _all_loads : list[AbstractLoad] = []

    _period : timedelta = timedelta(seconds=FLOATING_PERIOD_S)
    _commands : list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]] = []
    _solver_step_s : timedelta = timedelta(seconds=900)
    _update_step_s : timedelta = timedelta(seconds=5)

    grid_active_power_sensor: str = None
    grid_active_power_sensor_inverted: bool = False

    def __init__(self, **kwargs) -> None:

        self.voltage = kwargs.pop(CONF_HOME_VOLTAGE, 230)
        self.grid_active_power_sensor = kwargs.pop(CONF_GRID_POWER_SENSOR, None)
        self.grid_active_power_sensor_inverted = kwargs.pop(CONF_GRID_POWER_SENSOR_INVERTED, False)

        self.tariff_start_1 = kwargs.pop(CONF_HOME_START_OFF_PEAK_RANGE_1, None)
        self.tariff_end_1  = kwargs.pop(CONF_HOME_END_OFF_PEAK_RANGE_1, None)
        self.tariff_start_2 = kwargs.pop(CONF_HOME_START_OFF_PEAK_RANGE_2, None)
        self.tariff_end_2 = kwargs.pop(CONF_HOME_END_OFF_PEAK_RANGE_2, None)
        self.price_peak = kwargs.pop(CONF_HOME_PEAK_PRICE, 0.2)
        if self.price_peak <= 0:
            self.price_peak = 0.2
        self.price_peak = self.price_peak / 1000.0
        self.price_off_peak = kwargs.pop(CONF_HOME_OFF_PEAK_PRICE, 0.0) / 1000.0

        self.home_non_controlled_consumption_sensor = HOME_NON_CONTROLLED_CONSUMPTION_SENSOR
        self.home_available_power_sensor = HOME_AVAILABLE_POWER_SENSOR
        self.home_consumption_sensor = HOME_CONSUMPTION_SENSOR

        self.home_non_controlled_power_forecast_sensor_values = {}
        self.home_solar_forecast_sensor_values = {}

        self.home_non_controlled_power_forecast_sensor_values_providers = QSforecastValueSensor.get_probers(
            self.get_non_controlled_consumption_from_current_forecast,
            QSForecastHomeNonControlledSensors)

        self.home_solar_forecast_sensor_values_providers = QSforecastValueSensor.get_probers(
            self.get_solar_from_current_forecast,
            QSForecastSolarSensors)

        kwargs["home"] = self
        self.home = self
        super().__init__(**kwargs)
        self.home = self

        self._all_devices.append(self)

        self.home_non_controlled_consumption = None
        self.home_consumption = None
        self.home_available_power = None
        self.home_mode = None

        self._last_active_load_time = None
        if self.grid_active_power_sensor_inverted:
            self.attach_power_to_probe(self.grid_active_power_sensor,
                                          transform_fn=lambda x, a: -x)
        else:
            self.attach_power_to_probe(self.grid_active_power_sensor)

        self.attach_power_to_probe(self.home_consumption_sensor,
                                      non_ha_entity_get_state=self.home_consumption_sensor_state_getter)

        self.attach_power_to_probe(self.home_available_power_sensor,
                                      non_ha_entity_get_state=self.home_available_power_sensor_state_getter)

        self.attach_power_to_probe(self.home_non_controlled_consumption_sensor,
                                      non_ha_entity_get_state=self.home_non_controlled_consumption_sensor_state_getter)


        self._consumption_forecast = QSHomeConsumptionHistoryAndForecast(self)
        self.register_all_on_change_states()
        self._last_solve_done  : datetime | None = None


    def force_next_solve(self):
        self._last_solve_done = None

    def get_non_controlled_consumption_from_current_forecast(self, start_time:datetime, end_time:datetime | None = None) -> list[tuple[datetime | None, float | None]]:
        if self._consumption_forecast:
            if self._consumption_forecast.home_non_controlled_consumption:
                return self._consumption_forecast.home_non_controlled_consumption.get_from_current_forecast(start_time, end_time)
        return []

    async def _compute_non_controlled_forecast_intl(self, time: datetime) -> list[tuple[datetime | None, float | None]]:

        return  await self._consumption_forecast.home_non_controlled_consumption.get_forecast_and_set_as_current(
                time_now=time,
                history_in_hours=24, futur_needed_in_hours=int(self._period.total_seconds() // 3600) + 1)

    async def compute_non_controlled_forecast(self, time: datetime) -> list[tuple[datetime | None, float | None]]:

        unavoidable_consumption_forecast = []
        if await self._consumption_forecast.init_forecasts(time):
            unavoidable_consumption_forecast = await self._compute_non_controlled_forecast_intl(time)

        return unavoidable_consumption_forecast


    def get_solar_from_current_forecast(self, start_time:datetime, end_time:datetime | None = None) -> list[tuple[datetime | None, float | None]]:
        if self._solar_plant:
                return self._solar_plant.get_forecast(start_time, end_time)
        return []

    def get_tariffs(self, start_time:datetime, end_time:datetime) -> list[tuple[datetime, float]] | float:

        if self.price_off_peak == 0:
            return self.price_peak

        start_day = start_time.replace(hour=0, minute=0, second=0, microsecond=0)  # beware it is utc time

        to_prob = [
            (self.tariff_start_1, self.tariff_end_1, self.price_off_peak),
            (self.tariff_start_2, self.tariff_end_2, self.price_off_peak)
        ]

        ranges_off_peak = []

        for start, end, price in to_prob:
            if start is None or end is None or price is None or price == 0:
                continue

            if start == end:
                continue

            start_time_local = start_time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            start_utc = datetime.fromisoformat(f"{start_time_local.strftime("%Y-%m-%d")} {start}").replace(tzinfo=None).astimezone(tz=pytz.UTC)
            end_utc = datetime.fromisoformat(f"{start_time_local.strftime("%Y-%m-%d")} {end}").replace(tzinfo=None).astimezone(tz=pytz.UTC)

            if end_utc < start_utc:
                end_utc += timedelta(days=1)

            ranges_off_peak.append((start_utc, end_utc))

        if not ranges_off_peak:
            return self.price_peak

        ranges_off_peak = sorted(ranges_off_peak, key=lambda x: x[0])

        span = end_time - start_time
        num_day = span.days + 1
        start_day = start_time.replace(hour=0, minute=0, second=0, microsecond=0) # beware it is utc time
        tariffs = []
        curr_start = start_day
        for d in range(num_day):
            for first_start, first_end in ranges_off_peak:

                curr_end = first_start + timedelta(days=d)

                if curr_end > curr_start and curr_end >= start_time:
                    if curr_start < start_time:
                        curr_start = start_time
                    tariffs.append((curr_start, self.price_peak))

                curr_start = curr_end
                curr_end = first_end + timedelta(days=d)

                if curr_end > curr_start and curr_end >= start_time:
                    if curr_start < start_time:
                        curr_start = start_time
                    tariffs.append((curr_start, self.price_off_peak))

                curr_start = curr_end

                if curr_end >= end_time:
                    break

        return tariffs

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([ Platform.SENSOR, Platform.SELECT, Platform.BUTTON ])
        return list(parent)

    def get_car_by_name(self, name: str) -> QSCar | None:
        for car in self._cars:
            if car.name == name:
                return car
        return None

    def home_consumption_sensor_state_getter(self, entity_id: str,  time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if self.home_consumption is None:
            return None

        return (time, self.home_consumption, {})

    def home_available_power_sensor_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if self.home_available_power is None:
            return None

        return (time, self.home_available_power, {})



    def home_non_controlled_consumption_sensor_state_getter(self,
                                                            entity_id: str,
                                                            time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        # Home real consumption : Solar Production - grid_active_power_sensor - battery_charge_discharge_sensor
        # Available Power : grid_active_power_sensor + battery_charge_discharge_sensor
        # controlled consumption: sum of all controlled loads
        # Non controlled consumption : Home real consumption - controlled consumption

        # get solar production
        solar_production_minus_battery = None

        battery_charge = None
        if self._battery is not None:
            battery_charge = self._battery.get_sensor_latest_possible_valid_value(self._battery.charge_discharge_sensor, tolerance_seconds=POWER_ALIGNEMENT_TOLERANCE_S, time=time)

        if self._solar_plant is not None:
            solar_production = None
            if self._solar_plant.solar_inverter_active_power:
                # this one has the battery inside!
                solar_production_minus_battery = self._solar_plant.get_sensor_latest_possible_valid_value(self._solar_plant.solar_inverter_active_power, tolerance_seconds=POWER_ALIGNEMENT_TOLERANCE_S, time=time)

            if solar_production_minus_battery is None and self._solar_plant.solar_inverter_input_active_power:
                solar_production = self._solar_plant.get_sensor_latest_possible_valid_value(self._solar_plant.solar_inverter_input_active_power, tolerance_seconds=POWER_ALIGNEMENT_TOLERANCE_S, time=time)

            if solar_production_minus_battery is None:
                if solar_production is not None and battery_charge is not None:
                    solar_production_minus_battery = solar_production - battery_charge
                elif solar_production is not None:
                    solar_production_minus_battery = solar_production

        elif battery_charge is not None:
            solar_production_minus_battery = 0 - battery_charge


        if solar_production_minus_battery is None:
            solar_production_minus_battery = 0

        # get grid consumption
        grid_consumption = self.get_sensor_latest_possible_valid_value(self.grid_active_power_sensor, tolerance_seconds=POWER_ALIGNEMENT_TOLERANCE_S, time=time)

        if grid_consumption is None:
            self.home_non_controlled_consumption = None
            self.home_consumption = None
            self.home_available_power = None
        else:
            if solar_production_minus_battery is not None:
                home_consumption = solar_production_minus_battery - grid_consumption
            else:
                home_consumption = 0 - grid_consumption

            controlled_consumption = 0

            for load in self._all_loads:

                if not isinstance(load, AbstractLoad):
                    continue

                if load.load_is_auto_to_be_boosted:
                    continue

                if not isinstance(load, HADeviceMixin):
                    continue

                v = load.get_device_power_latest_possible_valid_value(tolerance_seconds=POWER_ALIGNEMENT_TOLERANCE_S, time=time)

                if v is not None:
                    controlled_consumption += v

            val = home_consumption - controlled_consumption

            self.home_non_controlled_consumption = val
            self.home_consumption = home_consumption
            if battery_charge is not None:
                self.home_available_power = grid_consumption + battery_charge
            else:
                self.home_available_power = grid_consumption

        val = self.home_non_controlled_consumption

        # slight hack to push the value to history:
        self.add_to_history(self.home_available_power_sensor, time)

        if val is None:
            return None

        return (time, val, {})

    def get_grid_active_power_values(self, duration_before_s: float, time: datetime)-> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self.grid_active_power_sensor, duration_before_s, time)

    def get_available_power_values(self, duration_before_s: float, time: datetime)-> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self.home_available_power_sensor, duration_before_s, time)

    def get_battery_charge_values(self, duration_before_s: float, time: datetime) -> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        if self._battery is None:
            return []
        return self._battery.get_state_history_data(self._battery.charge_discharge_sensor, duration_before_s, time)


    async def set_max_discharging_power(self, power: float | None = None, blocking: bool = False):
        if self._battery is not None:
            await self._battery.set_max_discharging_power(power, blocking)

    async def set_max_charging_power(self, power: float | None, blocking: bool = False):
        if self._battery is not None:
            await self._battery.set_max_charging_power(power, blocking)


    def add_device(self, device):

        device.home = self

        if isinstance(device, QSBattery):
            self._battery = device
        elif isinstance(device, QSCar):
            self._cars.append(device)
        elif isinstance(device, QSChargerGeneric):
            self._chargers.append(device)
        elif isinstance(device, QSSolar):
            self._solar_plant = device

        if isinstance(device, AbstractLoad):
            self._all_loads.append(device)

        if isinstance(device, HADeviceMixin):
            device.register_all_on_change_states()
            self._all_devices.append(device)


    async def update_loads_constraints(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            QSHomeMode.HOME_MODE_SENSORS_ONLY.value
            ]:
            return

        # check for active loads
        # for now we just check if a charger is plugged in
        for load in self._all_loads:
            if await load.do_run_check_load_activity_and_constraints(time):
                self.force_next_solve()


    async def update_forecast_probers(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            ]:
            return

        for name, prober in self.home_non_controlled_power_forecast_sensor_values_providers.items():
            self.home_non_controlled_power_forecast_sensor_values[name] = prober.push_and_get(time)

        for name, prober in self.home_solar_forecast_sensor_values_providers.items():
            self.home_solar_forecast_sensor_values[name] = prober.push_and_get(time)


    async def update_all_states(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            ]:
            return

        if self._solar_plant:
            await self._solar_plant.update_forecast(time)

        for device in self._all_devices:
            await device.update_states(time)

        if await self._consumption_forecast.init_forecasts(time):
            if self.home_non_controlled_consumption is not None:
                await self._consumption_forecast.home_non_controlled_consumption.add_value(time,
                                                                                     self.home_non_controlled_consumption,
                                                                                     do_save=True)
            if self._consumption_forecast.home_non_controlled_consumption.update_current_forecast_if_needed(time):
                await self._compute_non_controlled_forecast_intl(time)



    async def update_loads(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            QSHomeMode.HOME_MODE_SENSORS_ONLY.value
            ]:
            return

        await self.update_loads_constraints(time)

        all_loads = self._all_loads
        if self.home_mode == QSHomeMode.HOME_MODE_CHARGER_ONLY.value:
            all_loads = self._chargers

        for load in all_loads:

            wait_time = await load.check_commands(time=time)
            if wait_time > timedelta(seconds=60):
                if load.running_command_num_relaunch < 3:
                    await load.force_relaunch_command(time)
                else:
                    # we have an issue with this command ....
                    pass

        do_force_solve = False
        loads_to_reset = []
        for load in all_loads:

            if load.is_load_active(time) is False:
                continue

            if load.is_load_command_set(time) is False:
                continue


            if (await load.update_live_constraints(time, self._period)) :
                do_force_solve = True

            if load.is_load_active(time) is False:
                loads_to_reset.append(load)

        for load in loads_to_reset:
            # set them back to a kind of "idle" state, many times will be "OFF" CMD
            await load.launch_command(time=time, command = CMD_IDLE)


        active_loads = []
        for load in all_loads:

            if load.is_load_active(time) is False:
                continue

            active_loads.append(load)


        if self._last_solve_done is None or (time - self._last_solve_done) > timedelta(seconds=5*60):
            do_force_solve = True

        # we may also want to force solve ... if we have less energy than what was expected too ....imply force every 5mn

        if do_force_solve and active_loads:

            _LOGGER.info("DO SOLVE")

            self._last_solve_done = time

            unavoidable_consumption_forecast = await self.compute_non_controlled_forecast(time)
            pv_forecast = self.get_solar_from_current_forecast(time, time + self._period)

            end_time = time + self._period
            solver = PeriodSolver(
                start_time = time,
                end_time = end_time,
                tariffs = self.get_tariffs(time, end_time),
                actionable_loads = active_loads,
                battery = self._battery,
                pv_forecast  = pv_forecast,
                unavoidable_consumption_forecast = unavoidable_consumption_forecast,
                step_s = self._solver_step_s
            )

            # need to tweak a bit if there is some available power now for ex (or not) vs what is forecasted here.
            # use the available power virtual sensor to modify the begining of the PeriodSolver available power
            # computation based on forecasts

            self._commands, _ = solver.solve()


        for load, commands in self._commands:
            while len(commands) > 0 and commands[0][0] < time + self._update_step_s:
                cmd_time, command = commands.pop(0)
                _LOGGER.info("Launch command %s at %s for load %s", command.command, cmd_time, load.name)
                await load.launch_command(time, command)
                # only launch one at a time for a given load
                break


    async def reset_forecasts(self, time: datetime = None):
        if time is None:
            time = datetime.now(pytz.UTC)
        if self._consumption_forecast:
            await self._consumption_forecast.reset_forecasts(time)


    async def dump_for_debug(self):
        storage_path: str = join(self.hass.config.path(), DOMAIN, "debug")
        await aiofiles.os.makedirs(storage_path, exist_ok=True)

        if self._solar_plant:
            await self._solar_plant.dump_for_debug(storage_path)

        time = datetime.now(pytz.UTC)

        if self._consumption_forecast:
            await self._consumption_forecast.dump_for_debug(storage_path)

        debugs = {"now":time}


        file_path = join(storage_path, "debug_conf.pickle")

        def _pickle_save(file_path, obj):
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)

        await self.hass.async_add_executor_job(
            _pickle_save, file_path, debugs
        )


# to be able to easily fell on the same week boundaries, it has to be a multiple of 7, take 80 to go more than 1.5 year
BUFFER_SIZE_DAYS = 80*7
# interval in minutes between 2 measures
NUM_INTERVAL_PER_HOUR = 4
INTERVALS_MN = 60//NUM_INTERVAL_PER_HOUR # has to be a multiple of 60
NUM_INTERVALS_PER_DAY = (24*NUM_INTERVAL_PER_HOUR)
BEGINING_OF_TIME = datetime(2022, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
BUFFER_SIZE_IN_INTERVALS = BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY
class QSHomeConsumptionHistoryAndForecast:

    def __init__(self, home: QSHome | None, storage_path:str=None) -> None:
        self.home = home
        if home is None:
            self.hass = None
        else:
            self.hass = home.hass

        self.home_non_controlled_consumption : QSSolarHistoryVals | None = None
        self._in_reset = False
        if storage_path is None:
            if self.hass is None:
                self.storage_path = None
            else:
                self.storage_path = join(self.hass.config.path(), DOMAIN)
        else:
            self.storage_path = storage_path

        # ok now go through the various spots

    async def dump_for_debug(self, path:str):
        if self.home_non_controlled_consumption is not None:
            file_path = join(path, self.home_non_controlled_consumption.file_name)
            await self.home_non_controlled_consumption.save_values(file_path)

    async def load_from_history(self, entity_id:str, start_time: datetime, end_time: datetime) -> list[LazyState]:

        if self.hass is None:
            return []

        def load_history_from_db(start_time: datetime, end_time: datetime) -> list:
            """Load history from the database."""
            return state_changes_during_period(
                self.hass,
                start_time,
                end_time,
                entity_id,
                include_start_time_state=True,
                no_attributes=True,
            ).get(entity_id, [])

        states : list[LazyState] = await recorder_get_instance(self.hass).async_add_executor_job(load_history_from_db, start_time, end_time)
        # states : list[LazyState] = await self.hass.async_add_executor_job(load_history_from_db, start_time, end_time)
        return states

    async def init_forecasts(self, time: datetime):

        if self._in_reset is False and self.home_non_controlled_consumption is None:
            self.home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
            await self.home_non_controlled_consumption.init(time)

        return not self._in_reset

    async def reset_forecasts(self, time: datetime):

        self._in_reset = True

        self.home_non_controlled_consumption = None

        # compute from history all the values that are not yet computed ... from "true HA" sensors, not QS computed ones

        # start with home consumption

        battery_charge = None

        strt = BEGINING_OF_TIME
        end = time + timedelta(days=1)

        is_one_bad = False

        battery_charge = None
        if self.home._battery is not None:
            if self.home._battery.charge_discharge_sensor:
                battery_charge = QSSolarHistoryVals(entity_id=self.home._battery.charge_discharge_sensor, forecast=self)
                s, e = await battery_charge.init(time, for_reset=True)
                if s is None or e is None:
                    is_one_bad = True
                else:
                    if s > strt:
                        strt = s
                    if e < end:
                        end = e


        solar_production_minus_battery = None

        if is_one_bad is False:
            if self.home._solar_plant is not None:
                solar_production = None
                if self.home._solar_plant.solar_inverter_active_power:
                    # this one has the battery inside!
                    solar_production_minus_battery =  QSSolarHistoryVals(entity_id=self.home._solar_plant.solar_inverter_active_power, forecast=self)
                    s, e = await solar_production_minus_battery.init(time, for_reset=True)
                    if s is None or e is None:
                        is_one_bad = True
                    else:
                        if s > strt:
                            strt = s
                        if e < end:
                            end = e
                elif self.home._solar_plant.solar_inverter_input_active_power:
                    solar_production_minus_battery = QSSolarHistoryVals(entity_id=self.home._solar_plant.solar_inverter_input_active_power, forecast=self)
                    s, e = await solar_production_minus_battery.init(time, for_reset=True)
                    if s is None or e is None:
                        is_one_bad = True
                    else:
                        if s > strt:
                            strt = s
                        if e < end:
                            end = e

                    if is_one_bad is False and battery_charge is not None:
                        solar_production_minus_battery.values[0] = solar_production_minus_battery.values[0] - battery_charge.values[0]

        home_consumption = None
        if is_one_bad is False:
            if self.home.grid_active_power_sensor:
                home_consumption = QSSolarHistoryVals(entity_id=self.home.grid_active_power_sensor, forecast=self)
                s, e = await home_consumption.init(time, for_reset=True)
                if s is None or e is None:
                    is_one_bad = True
                else:
                    if s > strt:
                        strt = s
                    if e < end:
                        end = e
                    if solar_production_minus_battery is None:
                        if self.home.grid_active_power_sensor_inverted is False:
                            home_consumption.values[0] = (-1)*home_consumption.values[0]
                    else:
                        if self.home.grid_active_power_sensor_inverted is False:
                            home_consumption.values[0] = solar_production_minus_battery.values[0] - home_consumption.values[0]
                        else:
                            home_consumption.values[0] = solar_production_minus_battery.values[0] + home_consumption.values[0]

        if is_one_bad is False and home_consumption is not None:

            #ok we do have the computed home consumption ... now time for the controlled loads

            controlled_power_values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
            added_controlled = False

            for load in self.home._all_loads:

                if not isinstance(load, AbstractLoad):
                    continue

                if load.load_is_auto_to_be_boosted:
                    continue

                if not isinstance(load, HADeviceMixin):
                    continue

                ha_best_entity_id = load.get_best_power_HA_entity()

                if ha_best_entity_id is not None:
                    load_sensor = QSSolarHistoryVals(entity_id=ha_best_entity_id, forecast=self)
                    s, e = await load_sensor.init(time, for_reset=True)
                    if s is None or e is None:
                        is_one_bad = True
                        break
                    else:
                        if s > strt:
                            strt = s
                        if e < end:
                            end = e
                    controlled_power_values[0] += load_sensor.values[0]
                    added_controlled = True
                else:
                    if isinstance(load, AbstractLoad):
                        if load.switch_entity:
                            end_time = time
                            start_time = time - timedelta(days=BUFFER_SIZE_DAYS-1)
                            load_sensor = QSSolarHistoryVals(entity_id=load.switch_entity, forecast=self)

                            states : list[LazyState] = await self.load_from_history(load.switch_entity, start_time, end_time)

                            if states:
                                s = states[0].last_changed
                                e = states[-1].last_changed
                                load_sensor._current_idx = None
                                now_idx, now_days = load_sensor.get_index_from_time(time)
                                num_add = 0
                                for s in states:
                                    if s is  None or s.state is  None or s.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                                        continue
                                    value = load.get_power_from_switch_state(s.state)
                                    if value is None:
                                        continue
                                    num_add += 1
                                    await load_sensor.add_value(s.last_changed, float(value), do_save=False)

                                if num_add > 0:
                                    if load_sensor._current_idx is not None and load_sensor._current_idx != now_idx:
                                        await load_sensor._store_current_vals(time=None, do_save=False)

                                    if s > strt:
                                        strt = s
                                    if e < end:
                                        end = e

                                    controlled_power_values[0] += load_sensor.values[0]
                                    added_controlled = True

            if is_one_bad is False and added_controlled:
                home_consumption.values[0] = home_consumption.values[0] - controlled_power_values[0]



        if is_one_bad is False and home_consumption is not None:
            # ok we do have now a pretty good idea of the non-controllable house consumption:
            # let's first clean it with the proper start and end
            strt_idx, strt_days = home_consumption.get_index_from_time(strt)
            end_idx, end_days = home_consumption.get_index_from_time(end)
            #rolling buffer cleaning
            if strt_idx < end_idx:
                if strt_idx > 0:
                    home_consumption.values[0][0:strt_idx-1] = 0
                    home_consumption.values[1][0:strt_idx-1] = 0

                home_consumption.values[0][end_idx:] = 0
                home_consumption.values[1][end_idx:] = 0
            elif strt_idx > end_idx:
                home_consumption.values[0][end_idx+1:strt_idx] = 0
                home_consumption.values[1][end_idx+1:strt_idx] = 0
            else:
                # equal
                is_one_bad = True

        if is_one_bad is False:
            # now we do have something to save!
            home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
            home_non_controlled_consumption.values = home_consumption.values
            await home_non_controlled_consumption.save_values()
            self.home_non_controlled_consumption = None

        self._in_reset = False
        return True



class QSSolarHistoryVals:

    def __init__(self, forecast: QSHomeConsumptionHistoryAndForecast, entity_id:str) -> None:

        self.forecast = forecast
        if forecast is None:
            self.hass = None
            self.home = None
        else:
            self.home = forecast.home
            if self.home is None:
                self.hass = None
            else:
                self.hass =  self.home.hass

        self.entity_id:str = entity_id
        self.values : npt.NDArray | None = None
        self.storage_path : str = forecast.storage_path
        self.file_name = f"{self.entity_id}.npy"
        self.file_path = join(self.storage_path, self.file_name)

        self._current_values : list[tuple[datetime, float|None]] = []
        self._current_idx = None
        self._current_days = None
        self._init_done = False
        self._current_forecast = None
        self._last_forecast_update_time = None


    def update_current_forecast_if_needed(self, time: datetime) -> bool:
        if self._last_forecast_update_time is None or (time - self._last_forecast_update_time) >= timedelta(minutes=INTERVALS_MN):
            return True
        return False

    def get_from_current_forecast(self, start_time: datetime, end_time: datetime | None) -> list[tuple[datetime | None, str | float | None]]:
        return get_slots_from_time_serie(self._current_forecast, start_time, end_time)


    async def get_forecast_and_set_as_current(self, time_now: datetime, history_in_hours: int, futur_needed_in_hours: int) -> list[tuple[datetime, float]]:

        _LOGGER.debug("get_forecast_and_set_as_current called")

        self._last_forecast_update_time = time_now

        now_idx, now_days = self.get_index_from_time(time_now)

        time_now_from_idx = self.get_utc_time_from_index(now_idx, now_days)

        now_idx -= 1
        end_idx = self._sanitize_idx(now_idx)
        start_idx = self._sanitize_idx(end_idx - (history_in_hours*NUM_INTERVAL_PER_HOUR))

        current_values, current_days = self._get_values(start_idx, end_idx)
        current_ok_vales = np.asarray(current_days!=0, dtype=np.int32)

        best_past_probes_in_days = [
            1,
            3,
            6,
            7,
            7*4,
            7*8,
            7*12,
            7*51,
            7*52,
            7*53,
        ]

        scores = []

        for probe_days in best_past_probes_in_days:

            if probe_days*24 < futur_needed_in_hours:
                continue


            past_start_idx = self._sanitize_idx(start_idx - probe_days*NUM_INTERVALS_PER_DAY)
            past_end_idx = self._sanitize_idx(end_idx - probe_days*NUM_INTERVALS_PER_DAY)

            past_values, past_days = self._get_values(past_start_idx, past_end_idx)
            past_ok_values = np.asarray(past_days!=0, dtype=np.int32)

            check_vals = past_ok_values*current_ok_vales
            num_ok_vals = np.sum(check_vals)

            if num_ok_vals < 0.6*past_days.shape[0]:
                # bad history
                _LOGGER.debug(
                    f"get_forecast_and_set_as_current trash a past match for bad values {num_ok_vals} - {past_days.shape[0]}")
                continue

            score = float(np.sqrt(np.sum(np.square(current_values - past_values)*check_vals)))/float(num_ok_vals)

            scores.append((score, probe_days))


        if not scores:
            _LOGGER.debug("get_forecast_and_set_as_current no scores !!!")
            return []


        scores = sorted(scores, key=itemgetter(0))

        for score, probe_days in scores:

            # now we have the best past, let's forecast the future
            past_start_idx = self._sanitize_idx(now_idx - probe_days * NUM_INTERVALS_PER_DAY)
            past_end_idx = self._sanitize_idx(past_start_idx + ((futur_needed_in_hours + 1) * NUM_INTERVAL_PER_HOUR)) # add one hour to cover the futur

            forecast_values, past_days = self._get_values(past_start_idx, past_end_idx)
            past_ok_values = np.asarray(past_days != 0, dtype=np.int32)
            num_ok_vals = np.sum(past_ok_values)

            if num_ok_vals < 0.6*past_days.shape[0]:
                # bad forecast
                _LOGGER.debug(f"get_forecast_and_set_as_current trash a forecast for bad values {num_ok_vals} - {past_days.shape[0]}")
                continue

            forecast = []

            prob_last_idx = end_idx

            # add back true value at begining if needed
            while True:
                if self.values[1][prob_last_idx] != 0:
                    time_first = self.get_utc_time_from_index(prob_last_idx, self.values[1][prob_last_idx])  + timedelta(minutes=INTERVALS_MN//2)
                    if time_first < time_now_from_idx:
                        forecast.insert(0, ( time_first , self.values[0][prob_last_idx]) )
                    if time_first <= time_now:
                        break

                prob_last_idx -= 1
                prob_last_idx = self._sanitize_idx(prob_last_idx)



            for i in range(past_days.shape[0]):
                if past_ok_values[i] == 0:
                    continue
                # adding INTERVALS_MN//2 has we have a mean on INTERVALS_MN
                forecast.append((time_now_from_idx+timedelta(minutes=i*INTERVALS_MN + INTERVALS_MN//2), forecast_values[i]))

            # complement with the future if there is not enough data at the end
            if forecast[-1][0] < time_now + timedelta(hours=futur_needed_in_hours):
                forecast.append((time_now + timedelta(hours=futur_needed_in_hours), forecast[-1][1]))

            self._current_forecast = forecast
            _LOGGER.debug(f"get_forecast_and_set_as_current A GOOD ONE STORED  {num_ok_vals} - {past_days.shape[0]}")
            return forecast

        _LOGGER.debug("get_forecast_and_set_as_current nothing works!")
        return []


    def _sanitize_idx(self, idx):
        # it works for negative values too -2 * 10 => 8
        return idx % BUFFER_SIZE_IN_INTERVALS


    def _get_values(self, start_idx, end_idx):
        start_idx = self._sanitize_idx(start_idx)
        end_idx = self._sanitize_idx(end_idx)

        if end_idx > start_idx:
            return self.values[0][start_idx:end_idx+1], self.values[1][start_idx:end_idx+1]
        else:
            return np.concatenate((self.values[0][start_idx:], self.values[0][:end_idx+1])), np.concatenate((self.values[1][start_idx:], self.values[1][:end_idx+1]))


    async def save_values(self, file_path: str = None) -> None:
            def _save_values_to_file(path: str, values) -> None:
                """Write numpy."""
                try:
                    np.save(path, values)
                except:
                    pass

            if file_path is None:
                file_path = self.file_path

            if self.hass is None:
                _save_values_to_file(file_path, self.values)
            else:
                await self.hass.async_add_executor_job(
                    _save_values_to_file, file_path, self.values
                )
    async def read_values(self):

            def _load_values_from_file(path: str) -> Any | None:
                """Read numpy."""
                ret = None
                try:
                    ret =  np.load(path)
                except:
                    ret = None

                return ret

            if self.hass is None:
                return _load_values_from_file(self.file_path)
            else:
                return await self.hass.async_add_executor_job(
                    _load_values_from_file, self.file_path)

    async def _store_current_vals(self, time: datetime | None = None, do_save: bool = False, extend_but_not_cover_idx=None) -> None:

        if self._current_values:
            if self.values is None:
                self.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)

            nv = get_average_sensor(self._current_values, last_timing=time)
            self.values[0][self._current_idx] = nv
            self.values[1][self._current_idx] = self._current_days

            if extend_but_not_cover_idx is not None:
                if extend_but_not_cover_idx <= self._current_idx:
                    # we have circled around the ring buffer
                    extend_but_not_cover_idx = extend_but_not_cover_idx + BUFFER_SIZE_IN_INTERVALS

                for i in range(self._current_idx+1, extend_but_not_cover_idx):
                    new_idx = self._sanitize_idx(i)
                    self.values[0][new_idx] = nv
                    self.values[1][new_idx] = self._current_days

            if do_save:
                await self.save_values()



    async def add_value(self, time: datetime, value: float, do_save: bool = False):

        idx, days = self.get_index_from_time(time)

        if self._current_idx is None:
            self._current_idx = idx
            self._current_days = days
            self._current_values = []

        if self._current_idx != idx:
            await self._store_current_vals(time, do_save=do_save, extend_but_not_cover_idx=idx)
            self._current_idx = idx
            self._current_days = days
            self._current_values = [(time, value)]
        else:
            self._current_values.append((time, value))


    def get_index_from_time(self, time: datetime):
        days = (time - BEGINING_OF_TIME).days
        idx = int((time-BEGINING_OF_TIME).total_seconds()//(60*INTERVALS_MN))%(BUFFER_SIZE_IN_INTERVALS)
        return idx, days

    def get_utc_time_from_index(self, idx:int, days:int) -> datetime:
        days_index = int((days*24*60)//INTERVALS_MN)%(BUFFER_SIZE_IN_INTERVALS)
        if idx >= days_index:
            num_intervals = int(idx) - days_index
            return BEGINING_OF_TIME + timedelta(days=int(days)) + timedelta(minutes=int(num_intervals * INTERVALS_MN))
        else:
            num_intervals = (BUFFER_SIZE_IN_INTERVALS - days_index + idx)
            return BEGINING_OF_TIME + timedelta(days=int(days-1)) + timedelta(minutes=int(num_intervals * INTERVALS_MN))



    async def init(self, time:datetime, for_reset:bool = False) -> tuple[datetime | None, datetime | None]:

        if self._init_done:
            return None, None

        self._init_done = True

        if for_reset is False and self.values is not None:
            return None, None

        now_idx, now_days = self.get_index_from_time(time)

        if for_reset:
            self.values = None
        else:
            await aiofiles.os.makedirs(self.storage_path, exist_ok=True)
            self.values = await self.read_values()

        last_bad_idx = None
        last_bad_days = None
        do_save = False
        if self.values is not None:

            if self.values.shape[0] != 2 or self.values.shape[1] != BUFFER_SIZE_IN_INTERVALS:
                self.values = None

        if self.values is not None:

            last_bad_idx = now_idx
            last_bad_days = now_days
            # find a value before that may be good and filled already
            # if self.values[1][last_good_idx] is 0 : not filled
            # self.values[1][last_good_idx] is more than BUFFER_SIZE_DAYS difference ...a year: not valid
            num_slots_before_now_idx = 0

            while True:

                try_prev_idx = last_bad_idx - 1
                if try_prev_idx < 0:
                    try_prev_idx = self.values.shape[1] - 1

                if self.values[1][try_prev_idx] == 0 or now_days - self.values[1][try_prev_idx] >= BUFFER_SIZE_DAYS:
                    last_bad_idx = try_prev_idx
                    last_bad_days = self.values[1][try_prev_idx]
                    num_slots_before_now_idx += 1
                else:
                    break

            # not sure why we would need to reset all in case we have a 1 week gap
            # if num_slots_before_now_idx*60*INTERVALS_MN > MAX_SENSOR_HISTORY_S:
            #    last_bad_idx = None
        else:
            do_save = True
            self.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)


        end_time = time
        if last_bad_idx is not None:
            start_time = self.get_utc_time_from_index(last_bad_idx, last_bad_days)
        else:
            start_time = time - timedelta(days=BUFFER_SIZE_DAYS-1)

        states : list[LazyState] = await self.forecast.load_from_history(self.entity_id, start_time, end_time)

        # states : ordered LazySates
        # we will fill INTERVALS_IN_MN slots with the mean of the values in the state
        # the last remaining for the current one will be put as current value
        self._current_idx = None
        history_start = None
        history_end = None

        if states:

            # in states we have "LazyState" objects ... with no attribute
            real_state = self.hass.states.get(self.entity_id)
            if real_state is None:
                state_attr = {}
            else:
                state_attr = real_state.attributes

            history_start = states[0].last_changed
            history_end = states[-1].last_changed
            for s in states:
                if s is  None or s.state is  None or s.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                    continue
                if s.last_changed < start_time:
                    continue
                if s.last_changed > end_time:
                    continue

                power_value = convert_power_to_w(float(s.state), state_attr)

                await self.add_value(s.last_changed, power_value, do_save=False)
                # need to save if one value added, only do  it once at the end
                do_save = True

            if self._current_idx is not None and self._current_idx != now_idx:
                await self._store_current_vals(time=None, do_save=False)

            self._current_idx = None

            if for_reset is False and do_save:
                await self.save_values()

        return history_start, history_end




































