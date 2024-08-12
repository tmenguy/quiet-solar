import logging
from enum import StrEnum
import os
from os.path import join
from typing import Mapping, Any
from datetime import datetime, timedelta

import aiofiles.os
import homeassistant
import pytz
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder.models import LazyState
from homeassistant.components.recorder import get_instance as recorder_get_instance
from homeassistant.const import Platform

from ..const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    HOME_CONSUMPTION_SENSOR, HOME_NON_CONTROLLED_CONSUMPTION_SENSOR, HOME_AVAILABLE_POWER_SENSOR, DOMAIN, \
    SENSOR_HOME_CONSUMPTION_POWER, SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
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

    _period : timedelta = timedelta(days=1)
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

        if self._solar_plant:
            await self._solar_plant.update_forecast(time)

        for device in self._all_devices:
            await device.update_states(time)

        if self.home_non_controlled_consumption is not None:
            await self._consumption_forecast.init_forecasts(time)
            await self._consumption_forecast.home_non_controlled_consumption.add_value(time,
                                                                                 self.home_non_controlled_consumption,
                                                                                 do_save=True)
            if self.home_consumption is not None:
                await self._consumption_forecast.home_consumption.add_value(time, self.home_consumption, do_save=True)

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

            if load.is_load_active(time) is False:
                continue

            wait_time = await load.check_commands(time=time)
            if wait_time > timedelta(seconds=60):
                if load.running_command_num_relaunch < 3:
                    await load.force_relaunch_command(time)
                else:
                    # we have an issue with this command ....
                    pass

        do_force_solve = False
        for load in all_loads:

            if load.is_load_active(time) is False:
                continue

            if load.is_load_command_set(time) is False:
                continue


            if (await load.update_live_constraints(time, self._period)) :
                do_force_solve = True


        if False:

            if do_force_solve:
                solver = PeriodSolver(
                    start_time = time,
                    end_time = time + self._period,
                    tariffs = None,
                    actionable_loads = None,
                    battery = self._battery,
                    pv_forecast  = None,
                    unavoidable_consumption_forecast = None,
                    step_s = self._solver_step_s
                )

                self._commands = solver.solve()

            for load, commands in self._commands:
                while len(commands) > 0 and commands[0][0] < time + self._update_step_s:
                    cmd_time, command = commands.pop(0)
                    await load.launch_command(time, command)





# to be able to easily fell on the same week boundaries, it has to be a multiple of 7, take 53 to go more that a year
BUFFER_SIZE_DAYS = 53*7
# interval in minutes between 2 measures
INTERVALS_IN_MN = 15
NUM_INTERVALS_PER_DAY = (24*60)//INTERVALS_IN_MN + 1
BEGINING_OF_TIME = datetime(2022, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
class QSSolarHistoryVals:

    def __init__(self, home: QSHome, entity_id:str) -> None:

        self.home = home
        self.hass = home.hass
        self.entity_id:str = entity_id
        self.values : npt.NDArray | None = None
        self.storage_path : str = join(self.hass.config.path(), DOMAIN)
        self.file_path = join(self.storage_path, f"{self.entity_id}.npy")

        self._current_values : list[tuple[datetime, float|None]] = []
        self._current_idx = None
        self._current_days = None


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
        num_intervals = int((int(time.hour*60) + int(time.minute))//INTERVALS_IN_MN)
        return (idx*NUM_INTERVALS_PER_DAY) + num_intervals, days

    def get_utc_time_from_index(self, idx:int, days:int) -> datetime:
        num_intervals = idx % NUM_INTERVALS_PER_DAY
        return BEGINING_OF_TIME + timedelta(days=days) + timedelta(minutes=num_intervals*INTERVALS_IN_MN)


    async def init(self, time:datetime):

        if self.values is not None:
            return

        now_idx, now_days = self.get_index_from_time(time)

        await aiofiles.os.makedirs(self.storage_path, exist_ok=True)

        self.values = await self.read_values()
        last_bad_idx = None
        last_bad_days = None
        do_save = False
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

            if num_slots_before_now_idx*60*INTERVALS_IN_MN > MAX_SENSOR_HISTORY_S:
                last_bad_idx = None
        else:
            do_save = True
            self.values = np.zeros((2, BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY), dtype=np.uint32)


        end_time = time
        if last_bad_idx is not None:
            start_time = self.get_utc_time_from_index(last_bad_idx, last_bad_days)
        else:
            start_time = time - timedelta(days=BUFFER_SIZE_DAYS)


        def load_history_from_db(start_time: datetime, end_time: datetime) -> list:
            """Load history from the database."""
            return state_changes_during_period(
                self.hass,
                start_time,
                end_time,
                self.entity_id,
                include_start_time_state=True,
                no_attributes=True,
            ).get(self.entity_id, [])

        states : list[LazyState] = await recorder_get_instance(self.hass).async_add_executor_job(load_history_from_db, start_time, end_time)
        #states : list[LazyState] = await self.hass.async_add_executor_job(load_history_from_db, start_time, end_time)

        # states : ordered LazySates
        # we will fill INTERVALS_IN_MN slots with the mean of the values in the state
        # the last remaining for the current one will be put as current value
        self._current_idx = None

        if states:
            for s in states:
                await self.add_value(s.last_changed, float(s.state), do_save=False)

            if self._current_idx is not None and self._current_idx != now_idx:
                await self._store_current_vals(time=None, do_save=False)

            self._current_idx = None

            if do_save:
                await self.save_values()


class QSHomeConsumptionHistoryAndForecast:

    def __init__(self, home: QSHome) -> None:
        self.home = home
        self.hass = home.hass
        self.home_consumption = None
        self.home_non_controlled_consumption = None

    async def init_forecasts(self, time: datetime):

        if self.home_consumption is None:
            self.home_consumption = QSSolarHistoryVals(entity_id=SENSOR_HOME_CONSUMPTION_POWER, home=self.home)
            self.home_non_controlled_consumption = QSSolarHistoryVals(entity_id=SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, home=self.home)

            await self.home_consumption.init(time)
            await self.home_non_controlled_consumption.init(time)


















