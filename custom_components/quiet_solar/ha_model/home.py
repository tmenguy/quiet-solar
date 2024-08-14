import logging
from enum import StrEnum
import os
from operator import itemgetter
from os.path import join
from typing import Mapping, Any
from datetime import datetime, timedelta

import aiofiles.os
import homeassistant
import pytz
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder.models import LazyState
from homeassistant.components.recorder import get_instance as recorder_get_instance
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE

from ..const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    HOME_CONSUMPTION_SENSOR, HOME_NON_CONTROLLED_CONSUMPTION_SENSOR, HOME_AVAILABLE_POWER_SENSOR, DOMAIN, \
    SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
from ..ha_model.battery import QSBattery
from ..ha_model.car import QSCar
from ..ha_model.charger import QSChargerGeneric

from ..ha_model.device import HADeviceMixin, get_average_sensor
from ..ha_model.solar import QSSolar
from ..home_model.commands import LoadCommand
from ..home_model.load import AbstractLoad, AbstractDevice
from ..home_model.solver import PeriodSolver

import numpy as np
import numpy.typing as npt

class QSHomeMode(StrEnum):
    HOME_MODE_OFF = "home_mode_off"
    HOME_MODE_SENSORS_ONLY = "home_mode_sensors_only"
    HOME_MODE_CHARGER_ONLY = "home_mode_charger_only"
    HOME_MODE_ON = "home_mode_on"
    HOME_MODE_RESET_SENSORS = "home_mode_reset_sensors"


_LOGGER = logging.getLogger(__name__)




MAX_SENSOR_HISTORY_S = 60*60*24*7

POWER_ALIGNEMENT_TOLERANCE_S = 120
class QSHome(HADeviceMixin, AbstractDevice):

    _battery: QSBattery = None
    voltage: int = 230

    _chargers : list[QSChargerGeneric] = []
    _cars: list[QSCar] = []

    _all_devices : list[HADeviceMixin] = []
    _solar_plant: QSSolar | None = None
    _all_loads : list[AbstractLoad] = []

    _period : timedelta = timedelta(hours=36)
    _commands : list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]] = []
    _solver_step_s : timedelta = timedelta(seconds=900)
    _update_step_s : timedelta = timedelta(seconds=5)

    grid_active_power_sensor: str = None
    grid_active_power_sensor_inverted: bool = False

    def __init__(self, **kwargs) -> None:

        self.voltage = kwargs.pop(CONF_HOME_VOLTAGE, 230)
        self.grid_active_power_sensor = kwargs.pop(CONF_GRID_POWER_SENSOR, None)
        self.grid_active_power_sensor_inverted = kwargs.pop(CONF_GRID_POWER_SENSOR_INVERTED, False)

        self.home_non_controlled_consumption_sensor = HOME_NON_CONTROLLED_CONSUMPTION_SENSOR
        self.home_available_power_sensor = HOME_AVAILABLE_POWER_SENSOR
        self.home_consumption_sensor = HOME_CONSUMPTION_SENSOR

        kwargs["home"] = self
        self.home = self
        super().__init__(**kwargs)
        self.home = self

        self._all_devices.append(self)

        self.home_non_controlled_consumption = None
        self.home_consumption = None
        self.home_available_power = None
        self.home_mode = QSHomeMode.HOME_MODE_SENSORS_ONLY.value

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

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]


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

        if self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            QSHomeMode.HOME_MODE_SENSORS_ONLY.value
            ]:
            return

        # check for active loads
        # for now we just check if a charger is plugged in
        for load in self._all_loads:
            await load.check_load_activity_and_constraints(time)

    async def update_all_states(self, time: datetime):

        if self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            ]:
            return

        if QSHomeMode.HOME_MODE_RESET_SENSORS.value == self.home_mode:
            await self._consumption_forecast.reset_forecasts(time)
        else:
            if self._solar_plant:
                await self._solar_plant.update_forecast(time)

            for device in self._all_devices:
                await device.update_states(time)

            if self.home_non_controlled_consumption is not None:
                if await self._consumption_forecast.init_forecasts(time):
                    await self._consumption_forecast.home_non_controlled_consumption.add_value(time,
                                                                                         self.home_non_controlled_consumption,
                                                                                         do_save=True)

    async def update(self, time: datetime):

        if self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            QSHomeMode.HOME_MODE_SENSORS_ONLY.value
            ]:
            return

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
            load.launch_command(CMD_IDLE, time)

        if self.home_mode == QSHomeMode.HOME_MODE_ON.value:

            # we may also want to force solve ... if we have less energy than what was expected too ....

            if do_force_solve:

                active_loads = []
                for load in all_loads:

                    if load.is_load_active(time) is False:
                        continue

                    active_loads.append(load)

                unavoidable_consumption_forecast = None
                if await self._consumption_forecast.init_forecasts(time):
                    unavoidable_consumption_forecast = await self._consumption_forecast.home_non_controlled_consumption.get_forecast(time_now=time,
                                                                                                  history_in_hours=24,
                                                                                                  futur_needed_in_hours=int(self._period.total_seconds()//3600))
                pv_forecast = None

                if self._solar_plant:
                    pv_forecast = self._solar_plant.get_forecast(time, time + self._period)



                solver = PeriodSolver(
                    start_time = time,
                    end_time = time + self._period,
                    tariffs = None,
                    actionable_loads = active_loads,
                    battery = self._battery,
                    pv_forecast  = pv_forecast,
                    unavoidable_consumption_forecast = unavoidable_consumption_forecast,
                    step_s = self._solver_step_s
                )

                # need to tweak a bit if there is some available power now for ex (or not) vs what is forecasted here.
                # use teh available power virtual sensor to modify the begining of the PeriodSolver available power
                # computation based on forecasts

                self._commands = solver.solve()

            for load, commands in self._commands:
                while len(commands) > 0 and commands[0][0] < time + self._update_step_s:
                    cmd_time, command = commands.pop(0)
                    await load.launch_command(time, command)
                    # only launch one at a time for a given load
                    break





# to be able to easily fell on the same week boundaries, it has to be a multiple of 7, take 80 to go more than 1.5 year
BUFFER_SIZE_DAYS = 80*7
# interval in minutes between 2 measures
NUM_INTERVAL_PER_HOUR = 4
INTERVALS_MN = 60//NUM_INTERVAL_PER_HOUR # has to be a multiple of 60
NUM_INTERVALS_PER_DAY = (24*NUM_INTERVAL_PER_HOUR)
BEGINING_OF_TIME = datetime(2022, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)

class QSHomeConsumptionHistoryAndForecast:

    def __init__(self, home: QSHome) -> None:
        self.home = home
        self.hass = home.hass
        self.home_non_controlled_consumption : QSSolarHistoryVals | None = None
        self._is_reset_operated = False
        self._in_reset = False

        # ok now go through the various spots

    async def load_from_history(self, entity_id:str, start_time: datetime, end_time: datetime) -> list[LazyState]:

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
            self.home_non_controlled_consumption = QSSolarHistoryVals(entity_id=SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
            await self.home_non_controlled_consumption.init(time)

        return not self._in_reset

    async def reset_forecasts(self, time: datetime):

        # only do it once when asked in the UI
        if self._is_reset_operated is False:
            self._is_reset_operated = True
        else:
            return False

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

            controlled_power_values = np.zeros((2, BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY), dtype=np.int32)
            added_controlled = False

            for load in self.home._all_loads:

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
            home_non_controlled_consumption = QSSolarHistoryVals(entity_id=SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
            home_non_controlled_consumption.values = home_consumption.values
            await home_non_controlled_consumption.save_values()
            self.home_non_controlled_consumption = None

        self._in_reset = False
        return True



class QSSolarHistoryVals:

    def __init__(self, forecast: QSHomeConsumptionHistoryAndForecast, entity_id:str) -> None:

        self.forecast = forecast
        self.home = forecast.home
        self.hass =  self.home.hass
        self.entity_id:str = entity_id
        self.values : npt.NDArray | None = None
        self.storage_path : str = join(self.hass.config.path(), DOMAIN)
        self.file_path = join(self.storage_path, f"{self.entity_id}.npy")

        self._current_values : list[tuple[datetime, float|None]] = []
        self._current_idx = None
        self._current_days = None

        self._init_done = False



    async def get_forecast(self, time_now: datetime, history_in_hours: int, futur_needed_in_hours: int) -> list[tuple[datetime, float]]:

        now_idx, now_days = self.get_index_from_time(time_now)

        now_idx -= 1
        end_idx = self._sanitize_idx(now_idx)
        start_idx = self._sanitize_idx(end_idx - (history_in_hours*NUM_INTERVAL_PER_HOUR))

        current_values, current_days = self._get_values(start_idx, end_idx)
        current_ok_vales = np.asarray(current_days!=0, dtype=np.int32)

        best_past_probes_in_days = [
            1,
            7,
            6,
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
                continue

            score = float(np.sqrt(np.sum(np.square(current_values - past_values))*check_vals))/float(num_ok_vals)

            scores.append((score, probe_days))


        if not scores:
            return []


        scores = sorted(scores, key=itemgetter(0))

        for score, probe_days in scores:

            # now we have the best past, let's forecast the future
            past_start_idx = self._sanitize_idx(now_idx - probe_days * NUM_INTERVALS_PER_DAY)
            past_end_idx = self._sanitize_idx(past_start_idx + (futur_needed_in_hours * NUM_INTERVAL_PER_HOUR))

            forecast_values, past_days = self._get_values(past_start_idx, past_end_idx)
            past_ok_values = np.asarray(past_days != 0, dtype=np.int32)
            num_ok_vals = np.sum(past_ok_values)

            if num_ok_vals < 0.6*past_days.shape[0]:
                # bad forecast
                continue

            forecast = []

            for i in range(past_days.shape[0]):
                if past_ok_values[i] == 0:
                    continue
                forecast.append((time_now+timedelta(minutes=i*INTERVALS_MN), forecast_values[i]))

            return forecast


        return []


    def _sanitize_idx(self, idx):
        if idx < 0:
            idx = BUFFER_SIZE_DAYS * NUM_INTERVALS_PER_DAY - idx

        idx = idx % (BUFFER_SIZE_DAYS * NUM_INTERVALS_PER_DAY)

        return idx


    def _get_values(self, start_idx, end_idx):
        start_idx = self._sanitize_idx(start_idx)
        end_idx = self._sanitize_idx(end_idx)

        if end_idx > start_idx:
            return self.values[0][start_idx:end_idx+1], self.values[1][start_idx:end_idx+1]
        else:
            return np.concatenate((self.values[0][start_idx:], self.values[0][:end_idx+1])), np.concatenate((self.values[1][start_idx:], self.values[1][:end_idx+1]))


    async def save_values(self):
            def _save_values_to_file(path: str, values) -> None:
                """Write numpy."""
                try:
                    np.save(path, values)
                except:
                    pass

            await self.hass.async_add_executor_job(
                _save_values_to_file, self.file_path, self.values
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

            return await self.hass.async_add_executor_job(
                _load_values_from_file, self.file_path)

    async def _store_current_vals(self, time: datetime | None = None, do_save: bool = False, extend_but_not_cover_idx=None) -> None:

        if self._current_values:
            if self.values is None:
                self.values = np.zeros((2, BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY), dtype=np.int32)

            nv = get_average_sensor(self._current_values, last_timing=time)
            self.values[0][self._current_idx] = nv
            self.values[1][self._current_idx] = self._current_days

            if extend_but_not_cover_idx is not None:
                if extend_but_not_cover_idx <= self._current_idx:
                    extend_but_not_cover_idx = extend_but_not_cover_idx + BUFFER_SIZE_DAYS * NUM_INTERVALS_PER_DAY

                for i in range(self._current_idx+1, extend_but_not_cover_idx):
                    new_idx = i % (BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY)
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
        days = (time-BEGINING_OF_TIME).days
        idx = (days % BUFFER_SIZE_DAYS)
        num_intervals = int((int(time.hour*60) + int(time.minute))//INTERVALS_MN)
        return (idx*NUM_INTERVALS_PER_DAY) + num_intervals, days

    def get_utc_time_from_index(self, idx:int, days:int) -> datetime:
        num_intervals = int(idx) % NUM_INTERVALS_PER_DAY
        return BEGINING_OF_TIME + timedelta(days=int(days)) + timedelta(minutes=int(num_intervals*INTERVALS_MN))


    async def init(self, time:datetime, for_reset:bool = False) -> tuple[datetime | None, datetime | None]:

        if self._init_done:
            return None, None

        self._init_done = True

        if for_reset is False and self.values is not None :
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

            if self.values.shape[0] != 2 or self.values.shape[1] != BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY:
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

            if num_slots_before_now_idx*60*INTERVALS_MN > MAX_SENSOR_HISTORY_S:
                last_bad_idx = None
        else:
            do_save = True
            self.values = np.zeros((2, BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY), dtype=np.int32)


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
            history_start = states[0].last_changed
            history_end = states[-1].last_changed
            for s in states:
                if s is  None or s.state is  None or s.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                    continue
                await self.add_value(s.last_changed, float(s.state), do_save=False)

            if self._current_idx is not None and self._current_idx != now_idx:
                await self._store_current_vals(time=None, do_save=False)

            self._current_idx = None

            if for_reset is False and do_save:
                await self.save_values()

        return history_start, history_end




































