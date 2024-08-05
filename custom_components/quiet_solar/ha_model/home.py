from typing import Mapping, Any


from quiet_solar.const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED
from quiet_solar.ha_model.battery import QSBattery
from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.charger import QSChargerGeneric

from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.ha_model.solar import QSSolar
from quiet_solar.home_model.commands import LoadCommand
from quiet_solar.home_model.load import AbstractLoad, AbstractDevice
from datetime import datetime, timedelta
from quiet_solar.home_model.solver import PeriodSolver

import logging

_LOGGER = logging.getLogger(__name__)

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
        kwargs["home"] = self
        self.home = self
        super().__init__(**kwargs)
        self.home = self
        self._all_devices.append(self)

        self._last_active_load_time = None
        if self.grid_active_power_sensor_inverted:
            self.attach_ha_state_to_probe(self.grid_active_power_sensor,
                                          is_numerical=True,
                                          transform_fn=lambda x: -x)
        else:
            self.attach_ha_state_to_probe(self.grid_active_power_sensor,
                                          is_numerical=True)

        self.register_all_on_change_states()


    def get_grid_active_power_values(self, duration_before_s: float, time: datetime)-> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self.grid_active_power_sensor, duration_before_s, time)


    def get_battery_charge_values(self, duration_before_s: float, time: datetime) -> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        if self._battery is None:
            return []
        return self._battery.get_state_history_data(self._battery.charge_discharge_sensor, duration_before_s, time)

    def is_battery_in_auto_mode(self):
        if self._battery is None:
            return False
        else:
            return self._battery.is_battery_in_auto_mode()


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

        # check for active loads

        # for now we just check if a charger is plugged in
        for load in self._all_loads:
            await load.check_load_activity_and_constraints(time)


    async def update(self, time: datetime):


        for load in self._all_loads:

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
        for load in self._all_loads:

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











