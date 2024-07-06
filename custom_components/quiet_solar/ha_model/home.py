from homeassistant.core import callback, Event, EventStateChangedData
from homeassistant.helpers.event import async_track_state_change_event

from quiet_solar.const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED
from quiet_solar.ha_model.battery import QSBattery
from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.charger import QSChargerGeneric
from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.ha_model.solar import QSSolar
from quiet_solar.home_model.battery import Battery
from quiet_solar.home_model.commands import LoadCommand
from quiet_solar.home_model.load import AbstractLoad, AbstractDevice
from datetime import datetime, timedelta
from quiet_solar.home_model.solver import PeriodSolver
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE


class QSHome(HADeviceMixin, AbstractDevice):

    _battery: QSBattery = None

    _chargers : list[QSChargerGeneric] = []
    _cars: list[QSCar] = []


    _devices : list[AbstractDevice] = []
    _solar_plant: QSSolar | None = None
    _all_loads : list[AbstractLoad] = []

    _active_loads: list[AbstractLoad] = []

    _period : timedelta = timedelta(days=1)
    _commands : list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]] = []
    _solver_step_s : timedelta = timedelta(seconds=900)
    _update_step_s : timedelta = timedelta(seconds=5)
    def __init__(self, **kwargs) -> None:
        self.voltage = kwargs.pop(CONF_HOME_VOLTAGE, 230)
        self.grid_active_power_sensor = kwargs.pop(CONF_GRID_POWER_SENSOR, None)
        self.grid_active_power_sensor_inverted = kwargs.pop(CONF_GRID_POWER_SENSOR_INVERTED, False)
        super().__init__(**kwargs)

        self._last_active_load_time = None

        @callback
        def async_threshold_sensor_state_listener(
                event: Event[EventStateChangedData],
        ) -> None:
            """Handle sensor state changes."""
            new_state = event.data["new_state"]
            if new_state is None or new_state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                return

            value = float(new_state.state)
            if self.grid_active_power_sensor_inverted:
                value = -value

            time = new_state.last_updated
            self.add_to_history(new_state.entity_id, time, value)


        if self.grid_active_power_sensor is not None:

                self._unsub = async_track_state_change_event(
                    self.hass,
                    list(self.grid_active_power_sensor,),
                    async_threshold_sensor_state_listener,

            )

    def get_grid_active_power_values(self, duration_before_s: float, time: datetime):
        if self.grid_active_power_sensor is None:
            return []
        return self.get_history_data(self.grid_active_power_sensor, duration_before_s, time)


    def get_battery_charge_values(self, duration_before_s: float, time: datetime):
        if self._battery is None:
            return []
        return self._battery.get_history_data(self._battery.charge_discharge_sensor, duration_before_s, time)

    def is_battery_in_auto_mode(self):
        if self._battery is None:
            return False
        else:
            return self._battery.is_battery_in_auto_mode()


    async def set_max_discharging_power(self, power: float | None, blocking: bool = False):
        if self._battery is not None:
            await self._battery.set_max_discharging_power(power, blocking)

    async def set_max_charging_power(self, power: float | None, blocking: bool = False):
        if self._battery is not None:
            await self._battery.set_max_charging_power(power, blocking)


    def add_device(self, device: AbstractDevice):

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



    async def update(self, time: datetime):




        for load in self._active_loads:
            await load.check_commands(time=time)

        do_force_solve = False
        for load in self._active_loads:
            if (await load.update_live_constraints(time, self._period)) :
                do_force_solve = True


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











