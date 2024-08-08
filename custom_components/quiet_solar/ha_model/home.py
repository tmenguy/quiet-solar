from enum import StrEnum
from typing import Mapping, Any
from datetime import datetime, timedelta
from homeassistant.const import Platform
from homeassistant.core import State

from ..const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    HOME_CONSUMPTION_SENSOR, HOME_NON_CONTROLLED_CONSUMPTION_SENSOR, HOME_AVAILABLE_POWER_SENSOR
from ..ha_model.battery import QSBattery
from ..ha_model.car import QSCar
from ..ha_model.charger import QSChargerGeneric

from ..ha_model.device import HADeviceMixin
from ..ha_model.solar import QSSolar
from ..home_model.commands import LoadCommand
from ..home_model.load import AbstractLoad, AbstractDevice
from ..home_model.solver import PeriodSolver


class QSHomeMode(StrEnum):
    HOME_MODE_OFF = "home_mode_off"
    HOME_MODE_SENSORS_ONLY = "home_mode_sensors_only"
    HOME_MODE_CHARGER_ONLY = "home_mode_charger_only"
    HOME_MODE_ON = "home_mode_on"

import logging

_LOGGER = logging.getLogger(__name__)


POWER_ALIGNEMENT_TOLERANCE_S = 60
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

        self.attach_power_to_probe(self.home_available_power_sensor,
                                      non_ha_entity_get_state=self.home_available_power_sensor_state_getter)

        self.attach_power_to_probe(self.home_non_controlled_consumption_sensor,
                                      non_ha_entity_get_state=self.home_non_controlled_consumption_sensor_state_getter)



        self.register_all_on_change_states()

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]


    def home_available_power_sensor_state_getter(self,
                                                            entity_id: str,
                                                            time: datetime | None) -> (
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
        # Non controlled consumption : Home real consumption - controlled consumption
        # controlled consumption: sum of all controlled loads

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

            for device in self._all_devices:
                await device.update_states(time)

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











