import logging
from datetime import datetime
from sys import maxsize, maxunicode
from typing import Any

from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, ATTR_ENTITY_ID, SERVICE_TURN_ON, \
    SERVICE_TURN_OFF
from homeassistant.components import number


from ..const import CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, \
    CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, CONF_BATTERY_CHARGE_PERCENT_SENSOR, CONF_BATTERY_CHARGE_FROM_GRID_SWITCH
from ..ha_model.device import HADeviceMixin
from ..home_model.battery import Battery
from ..home_model.commands import LoadCommand, CMD_ON, CMD_IDLE, CMD_AUTO_GREEN_ONLY, CMD_GREEN_CHARGE_AND_DISCHARGE, \
    CMD_FORCE_CHARGE, CMD_GREEN_CHARGE_ONLY

_LOGGER = logging.getLogger(__name__)

class QSBattery(HADeviceMixin, Battery):


    def __init__(self, **kwargs) -> None:
        self.charge_discharge_sensor = kwargs.pop(CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, None)
        self.max_discharge_number = kwargs.pop(CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, None)
        self.max_charge_number = kwargs.pop(CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, None)
        self.charge_percent_sensor = kwargs.pop(CONF_BATTERY_CHARGE_PERCENT_SENSOR, None)
        self.charge_from_grid_switch = kwargs.pop(CONF_BATTERY_CHARGE_FROM_GRID_SWITCH, None)

        super().__init__(**kwargs)

        self.attach_power_to_probe(self.charge_discharge_sensor)

        self.attach_ha_state_to_probe(self.charge_percent_sensor,
                                      is_numerical=True)

        self.max_discharging_power_current = None
        self.max_charging_power_current = None
        self.is_charge_from_grid_current = None

    @property
    def current_charge(self) -> float | None:
        percent = self.get_sensor_latest_possible_valid_value(entity_id=self.charge_percent_sensor)
        if percent is None:
            return None
        return float(percent * self.capacity)/100.0

    def _command_to_values(self, command: LoadCommand) -> dict[str, Any]:
        if command.is_like_one_of_cmds([CMD_ON, CMD_IDLE, CMD_AUTO_GREEN_ONLY, CMD_GREEN_CHARGE_AND_DISCHARGE]):
            ret = { "charge_from_grid": False, "max_discharging_power": self.max_discharging_power, "max_charging_power": self.max_charging_power}
        elif command.is_like(CMD_GREEN_CHARGE_ONLY):
            ret = { "charge_from_grid": False, "max_discharging_power": 0, "max_charging_power": self.max_charging_power}
        elif command.is_like(CMD_FORCE_CHARGE):
            ret ={ "charge_from_grid": True, "max_discharging_power": 0, "max_charging_power": command.power_consign}
        else:
            raise ValueError("Invalid command")

        if self.charge_from_grid_switch is None:
            ret["charge_from_grid"] = None

        if self.max_discharge_number is None:
            ret["max_discharging_power"] = None

        if self.max_charge_number is None:
            ret["max_charging_power"] = None

        return ret


    async def execute_command(self, time: datetime, command:LoadCommand) -> bool | None:

        if command.is_like(CMD_GREEN_CHARGE_ONLY):
            _LOGGER.info(f"=====> Executing green charge only command on the battery!!!!!!!!!!!!!!!!!!!!!!!!!")

        cmd_to_vals = self._command_to_values(command)
        await self.set_charge_from_grid(cmd_to_vals["charge_from_grid"])
        await self.set_max_discharging_power(cmd_to_vals["max_discharging_power"])
        await self.set_max_charging_power(cmd_to_vals["max_charging_power"])

        return False

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        cmd_to_vals = self._command_to_values(command)

        is_charge_from_grid = await self.is_charge_from_grid()

        if cmd_to_vals["charge_from_grid"] is not None and is_charge_from_grid is  None:
                return None

        max_discharge_power = await self.get_max_discharging_power()

        if cmd_to_vals["max_discharging_power"] is not None and max_discharge_power is None:
            return None

        max_charge_power = await self.get_max_charging_power()

        if cmd_to_vals["max_charging_power"] is not None and max_charge_power is None:
            return None

        return is_charge_from_grid == cmd_to_vals["charge_from_grid"] and max_discharge_power == cmd_to_vals["max_discharging_power"] and max_charge_power == cmd_to_vals["max_charging_power"]


    async def set_charge_from_grid(self, charge_from_grid: bool | None, blocking: bool = False):
        if self.charge_from_grid_switch is None or charge_from_grid is None:
            return

        if self.is_charge_from_grid_current == charge_from_grid:
            return

        if charge_from_grid:
            action = SERVICE_TURN_ON
        else :
            action = SERVICE_TURN_OFF

        _LOGGER.info(f"Executing set_charge_from_grid {action} on {self.charge_from_grid_switch}")
        await self.hass.services.async_call(
            domain=Platform.SWITCH,
            service=action,
            target={"entity_id": self.charge_from_grid_switch}
        )

    async def is_charge_from_grid(self) -> bool | None:
        if self.charge_from_grid_switch is None:
            return None

        state = self.hass.states.get(self.charge_from_grid_switch)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res = None
        else:
            res =  state.state == "on"

        self.is_charge_from_grid_current = res
        return res


    async def set_max_discharging_power(self, power: float | None = None, blocking: bool = False):
        if self.max_discharge_number is None or power is None:
            return

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_discharge_number}
        range_value = float(power)
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_discharging_power)
        max_value = float(self.max_discharging_power)

        val = int(min(max_value, max(min_value, range_value)))

        if val == self.max_discharging_power_current:
            return

        data[number.ATTR_VALUE] = val
        domain = number.DOMAIN


        await self.hass.services.async_call(
            domain, service, data, blocking=blocking
        )

    async def get_max_discharging_power(self):
        if self.max_discharge_number is None:
            return None

        state = self.hass.states.get(self.max_discharge_number)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res = None
        else:
            res =  int(state.state)
        self.max_discharging_power_current = res
        return res

    async def get_max_charging_power(self):
        if self.max_charge_number is None:
            return None

        state = self.hass.states.get(self.max_charge_number)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res = None
        else:
            res =  int(state.state)
        self.max_charging_power_current = res
        return res



    async def set_max_charging_power(self, power: float | None = None, blocking: bool = False):
        if self.max_charge_number is None or power is None:
            return

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_charge_number}
        range_value = float(power)
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_charging_power)
        max_value = float(self.max_charging_power)

        val = int(min(max_value, max(min_value, range_value)))

        if val == self.max_charging_power_current:
            return

        data[number.ATTR_VALUE] = val
        domain = number.DOMAIN

        await self.hass.services.async_call(
            domain, service, data, blocking=blocking
        )




    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([ Platform.SENSOR])
        return list(parent)
