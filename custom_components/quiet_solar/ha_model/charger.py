import bisect
from datetime import datetime
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry
from homeassistant.helpers.entity import Entity

from quiet_solar.const import CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, CONF_CHARGER_PAUSE_RESUME_SWITCH, \
    CONF_CHARGER_PLUGGED, CONF_CHARGER_MAX_CHARGE, CONF_CHARGER_MIN_CHARGE, CONF_CHARGER_IS_3P, \
    CONF_CHARGER_DEVICE_OCPP, CONF_CHARGER_DEVICE_WALLBOX, CONF_POWER_SENSOR
from quiet_solar.sensor import QSSensorEntityDescription, QSBaseSensor, QSBaseSensorRestore
from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.device import HADeviceMixin, align_time_series_and_values, get_average_power
from quiet_solar.home_model.commands import LoadCommand, CMD_AUTO_GREEN_ONLY, CMD_ON, CMD_OFF
from quiet_solar.home_model.load import AbstractLoad
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, \
    ATTR_ENTITY_ID

from homeassistant.components import number, homeassistant


CHARGER_STATE_REFRESH_INTERVAL = 3

CHARGER_ADAPTATION_WINDOW = 15

class QSChargerGeneric(HADeviceMixin, AbstractLoad):

    def __init__(self, **kwargs):
        self.charger_plugged = kwargs.pop(CONF_CHARGER_PLUGGED, None)
        self.charger_max_charging_current_number = kwargs.pop(CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER, None)
        self.charger_pause_resume_switch = kwargs.pop(CONF_CHARGER_PAUSE_RESUME_SWITCH, None)
        self.charger_max_charge = kwargs.pop(CONF_CHARGER_MAX_CHARGE, 32)
        self.charger_min_charge = kwargs.pop(CONF_CHARGER_MIN_CHARGE, 6)
        self.charger_is_3p = kwargs.pop(CONF_CHARGER_IS_3P, False)
        self.car : QSCar | None = None

        self.charge_state = STATE_UNKNOWN

        self._last_command_prob_time = datetime.min
        self._verified_amperage_command_time = None
        super().__init__(**kwargs)

        self._default_generic_car = QSCar(hass=self.hass, home=self.home, config_entry=None, name=f"{self.name}_generic_car")

    def get_platforms(self):
        return [ Platform.SENSOR, Platform.SELECT ]

    def create_ha_entities(self, platform: str) -> list[Entity]:

        entities = []

        if platform == Platform.SENSOR:
            charge_sensor = QSSensorEntityDescription(
                key="charge_state",
                device_class=SensorDeviceClass.ENUM,
                options=[
                    "not_in_charge",
                    "waiting_for_a_planned_charge",
                    "charge_ended",
                    "waiting_for_current_charge",
                    "energy_flap_opened",
                    "charge_in_progress",
                    "charge_error",
                    STATE_UNAVAILABLE,
                    STATE_UNKNOWN,
                ],
            )

            entities.append(QSBaseSensor(data_handler=self.data_handler, qs_device=self, description=charge_sensor))

        entities.extend(self._default_generic_car.create_ha_entities(platform))

        return entities






    def attach_car(self, car):
        self.car = car

    def detach_car(self):
        self.car = None

    def is_plugged(self):
        state = self.hass.states.get(self.charger_plugged)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False
        return state.state == "on"

    async def stop_charge(self):
        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=SERVICE_TURN_OFF,
            target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
            blocking=False
        )

    async def start_charge(self):
        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=SERVICE_TURN_ON,
            target={ATTR_ENTITY_ID: self.charger_pause_resume_switch},
            blocking=False
        )

    async def is_charge_enabled(self):
        result = False
        if self.is_plugged():
            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = False
            else:
                result = state.state == "on"
        return result

    async def is_charge_stopped(self):
        result = True
        if self.is_plugged():
            state = self.hass.states.get(self.charger_pause_resume_switch)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = True
            else:
                result = state.state == "off"
        return result

    async def get_charging_power(self):
        if self.is_plugged() is False:
            return 0.0
        state = self.hass.states.get(self.power_sensor)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return 0.0
        return float(state.state)

    async def set_max_charging_current(self, current, blocking=False):

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.charger_max_charging_current_number}
        range_value = float(current)
        service = number.SERVICE_SET_VALUE
        min_value = float(self.charger_max_charge)
        max_value = float(self.charger_min_charge)
        data[number.ATTR_VALUE] = int(min(max_value, max(min_value, range_value)))
        domain = number.DOMAIN

        await self.hass.services.async_call(
            domain, service, data, blocking=blocking
        )

    async def constraint_update_value_callback_percent_soc(self, ct: AbstractLoad, time: datetime) -> float | None:
        """ Example of a value compute callback for a load constraint. like get a sensor state, compute energy available for a car from battery charge etc
        it could also update the current command of the load if needed
        """


        je dois en plus verifier si le get_charging_power est a 0 ou pas car ca veut dire que la voiture ne demande plus de power non plus!!!


        state = self.hass.states.get(self.car.car_charge_percent_sensor)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            result = None
        else:
            result = float(state.state)

        # check also if the current state of the automation has been properly set
        if self.current_command is None or self.is_plugged() is False:
            self.current_command = None
            self._verified_amperage_command_time = None
        elif self.current_command.command == CMD_AUTO_GREEN_ONLY.command:

            if self.current_command.private == 0:
                #check if we are properly in stop state
                if self.is_charge_stopped():
                    if self._verified_amperage_command_time is None:
                        # we can put back the battery as possibly discharging! as the car won't consume anymore soon ...
                        if self.home.is_battery_in_auto_mode():
                            await self.home.set_max_discharging_power()

                        self._verified_amperage_command_time = time
            else:
                if self.is_charge_enabled():
                    #check we have the expected amperage
                    state = self.hass.states.get(self.charger_max_charging_current_number)

                    if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                        # we don't know
                        # self._verified_amperage_command_time = None #no need of this reset, if it was not known it was verified already
                        pass
                    else:
                        if state.state == self.current_command.private:
                            # ok we have the expected value set we can compute
                            if self._verified_amperage_command_time is None:
                                self._verified_amperage_command_time = time
                        else:
                            # we may have a case where the value has been changed without quiet_solar knwowing! (external order from user for example)
                            # or simply that the amperage command hasn't been set
                            pass

            if self._verified_amperage_command_time is None:
                await self._do_update_charger_state(time)
            else:
                # we will compare now if the current need to be adapted compared to solar production
                if (time - self._verified_amperage_command_time).total_seconds() > CHARGER_ADAPTATION_WINDOW:

                    power_values = self.home.get_grid_active_power_values(CHARGER_ADAPTATION_WINDOW, time)
                    charging_values = None
                    if self.home.is_battery_in_auto_mode():
                        charging_values = self.home.get_battery_charge_values(CHARGER_ADAPTATION_WINDOW, time)

                    available_power = align_time_series_and_values(power_values, charging_values, do_sum=True)

                    # the battery is normally adapting itself to the solar production, so if it is charging .. we will say that this powe is available to the car

                    # do we need a bit of a PID ? (proportional integral derivative? or keep it simple for now) or a convex hul with min / max?
                    # very rough estimation for now:

                    last_p = get_average_power(available_power[-len(available_power)//2:])
                    all_p = get_average_power(available_power)

                    if self.current_command.param == CMD_AUTO_GREEN_ONLY.param:
                        target_delta_power = min(last_p, all_p)
                    else:
                        target_delta_power = max(last_p, all_p)

                    # we will compare now if the current need to be adapted compared to solar production
                    power_steps, min_charge, max_charge = self.car.get_charge_power_per_phase_A(self.charger_is_3p)

                    if self.current_command.private >= min_charge:
                        current_power = power_steps[int(self.current_command.private) - min_charge]
                    else:
                        current_power = 0

                    target_power = current_power + target_delta_power

                    i = bisect.bisect_left(power_steps, target_power)

                    if i == len(power_steps):
                        new_amp = max_charge
                    elif i == 0 and power_steps[0] > target_power:
                        if self.current_command.param == CMD_AUTO_GREEN_ONLY.param:
                            new_amp = 0
                        else:
                            new_amp = min_charge
                    elif power_steps[i] == target_power:
                        new_amp = i + min_charge
                    else:
                        if self.current_command.param == "green_only":
                            new_amp = i + min_charge
                        else:
                            new_amp = min(max_charge, i + min_charge + 1)

                    if new_amp != self.current_command.private:
                        self.current_command.private = new_amp
                        self._verified_amperage_command_time = None

                        if new_amp == 0:
                            await self.stop_charge()
                            # we can put back the battery as possibly discharging! as the car wont consume anymore soon
                            # ... wait and see above for the discharging being ok
                        else:
                            if self.is_charge_stopped():
                                await self.set_max_charging_current(new_amp, blocking=True)
                                await self.start_charge()
                            else:
                                await self.set_max_charging_current(new_amp, blocking=False)

                            # do not allow the battery to discharge!
                            if self.home.is_battery_in_auto_mode():
                                await self.home.set_max_discharging_power(power=0.0)

        elif self.current_command.command == CMD_ON.command:
            self._verified_amperage_command_time = None
        elif self.current_command.command == CMD_OFF.command:
            self._verified_amperage_command_time = None

        return result

    async def execute_command(self, time: datetime, command : LoadCommand):

        # force a homeassistant.update_entity service on the charger entity?
        if command.command == "on":
            await self.start_charge()
        elif command.command == "off" or command.command == "auto":
            await self.stop_charge()

        self._last_command_prob_time = time

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:

        result = False
        if self.is_plugged():
            if command.command == CMD_ON.command:
                result = self.is_charge_enabled()
            elif command.command == CMD_OFF.command:
                result = self.is_charge_stopped()
            elif command.command == CMD_AUTO_GREEN_ONLY.command:
                result =  True

        if result is False:
            await self._do_update_charger_state(time)

        return result

    async def _do_update_charger_state(self, time):
        if (time - self._last_command_prob_time).total_seconds() > CHARGER_STATE_REFRESH_INTERVAL:
            await self.hass.services.async_call(
                homeassistant.DOMAIN,
                homeassistant.SERVICE_UPDATE_ENTITY,
                {ATTR_ENTITY_ID: [self.charger_pause_resume_switch, self.charger_max_charging_current_number]},
                blocking=False
            )
            self._last_command_prob_time = time


class QSChargerOCPP(QSChargerGeneric):

    def __init__(self, **kwargs):
        self.charger_device_ocpp = kwargs.pop(CONF_CHARGER_DEVICE_OCPP, None)

        hass : HomeAssistant | None = kwargs.get("hass", None)

        if self.charger_device_ocpp is not None and hass is not None:
            entity_reg = entity_registry.async_get(hass)
            entries = entity_registry.async_entries_for_device(entity_reg, self.charger_device_ocpp)

            for entry in entries:
                if entry.entity_id.startswith("number.") and entry.entity_id.endswith("_maximum_current"):
                    kwargs[CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER] = entry.entity_id

                if entry.entity_id.startswith("switch.") and entry.entity_id.endswith("_charge_control"):
                    kwargs[CONF_CHARGER_PAUSE_RESUME_SWITCH] = entry.entity_id

                if entry.entity_id.startswith("switch.") and entry.entity_id.endswith("_availability"):
                    kwargs[CONF_CHARGER_PLUGGED] = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_power_active_export"):
                    kwargs[CONF_POWER_SENSOR] = entry.entity_id




        super().__init__(**kwargs)
    def is_plugged(self):
        state = self.hass.states.get(self.charger_plugged)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False

        return state.state == "off"


class QSChargerWallbox(QSChargerGeneric):
    def __init__(self, **kwargs):
        self.charger_device_wallbox = kwargs.pop(CONF_CHARGER_DEVICE_WALLBOX, None)
        hass : HomeAssistant | None = kwargs.get("hass", None)

        if self.charger_device_wallbox is not None and hass is not None:
            entity_reg = entity_registry.async_get(hass)
            entries = entity_registry.async_entries_for_device(entity_reg, self.charger_device_wallbox)

            for entry in entries:
                if entry.entity_id.startswith("number.") and entry.entity_id.endswith("_maximum_charging_current"):
                    kwargs[CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER] = entry.entity_id

                if entry.entity_id.startswith("switch.") and entry.entity_id.endswith("_pause_resume"):
                    kwargs[CONF_CHARGER_PAUSE_RESUME_SWITCH] = entry.entity_id

                if entry.entity_id.startswith("sensor.") and entry.entity_id.endswith("_charging_power"):
                    kwargs[CONF_POWER_SENSOR] = entry.entity_id

        super().__init__(**kwargs)

    def is_plugged(self):
        state = self.hass.states.get(self.charger_pause_resume_switch)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return False

        return True

    async def execute_command(self, time: datetime, command: LoadCommand):
        print(f"Executing command {command}")

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool:
        return True








