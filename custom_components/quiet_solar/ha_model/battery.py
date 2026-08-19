import logging
import math
from datetime import datetime
from typing import Any

from homeassistant.components import number
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_UNIT_OF_MEASUREMENT,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    Platform,
    UnitOfPower,
)
from homeassistant.util.unit_conversion import PowerConverter

from ..const import (
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR,
    CONF_BATTERY_CHARGE_FROM_GRID_SWITCH,
    CONF_BATTERY_CHARGE_PERCENT_SENSOR,
    CONF_BATTERY_MAX_CHARGE_POWER_NUMBER,
    CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER,
    CONF_TYPE_NAME_QSBattery,
)
from ..ha_model.device import HADeviceMixin, convert_power_to_w
from ..home_model.battery import Battery
from ..home_model.commands import (
    CMD_AUTO_GREEN_ONLY,
    CMD_FORCE_CHARGE,
    CMD_GREEN_CHARGE_AND_DISCHARGE,
    CMD_GREEN_CHARGE_ONLY,
    CMD_IDLE,
    CMD_ON,
    LoadCommand,
)

_LOGGER = logging.getLogger(__name__)


class QSBattery(HADeviceMixin, Battery):
    conf_type_name = CONF_TYPE_NAME_QSBattery

    def __init__(self, **kwargs) -> None:
        self.charge_discharge_sensor = kwargs.pop(CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, None)
        self.max_discharge_number = kwargs.pop(CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, None)
        self.max_charge_number = kwargs.pop(CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, None)
        self.charge_percent_sensor = kwargs.pop(CONF_BATTERY_CHARGE_PERCENT_SENSOR, None)
        self.charge_from_grid_switch = kwargs.pop(CONF_BATTERY_CHARGE_FROM_GRID_SWITCH, None)

        super().__init__(**kwargs)

        self.attach_power_to_probe(self.charge_discharge_sensor)

        self.attach_ha_state_to_probe(self.charge_percent_sensor, is_numerical=True)

        self.is_charge_from_grid_current = None

    @property
    def current_charge(self) -> float | None:
        percent = self.get_sensor_latest_possible_valid_value(entity_id=self.charge_percent_sensor)
        if percent is None:
            return None
        return float(percent * self.capacity) / 100.0

    def _command_to_values(self, command: LoadCommand) -> dict[str, Any]:
        if command.is_like_one_of_cmds([CMD_ON, CMD_IDLE, CMD_AUTO_GREEN_ONLY, CMD_GREEN_CHARGE_AND_DISCHARGE]):
            ret = {
                "charge_from_grid": False,
                "max_discharging_power": self.max_discharging_power,
                "max_charging_power": self.max_charging_power,
            }
        elif command.is_like(CMD_GREEN_CHARGE_ONLY):
            # emit the outage safety floor (default 0) instead of a hard 0
            ret = {
                "charge_from_grid": False,
                "max_discharging_power": self.min_discharging_power,
                "max_charging_power": self.max_charging_power,
            }
        elif command.is_like(CMD_FORCE_CHARGE):
            ret = {
                "charge_from_grid": True,
                "max_discharging_power": self.min_discharging_power,
                "max_charging_power": command.power_consign,
            }
        else:
            raise ValueError("Invalid command")

        if self.charge_from_grid_switch is None:
            ret["charge_from_grid"] = None

        if self.max_discharge_number is None:
            ret["max_discharging_power"] = None

        if self.max_charge_number is None:
            ret["max_charging_power"] = None

        return ret

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:

        if command.is_like(CMD_GREEN_CHARGE_ONLY):
            _LOGGER.info("=====> Executing green charge only command on the battery!!!!!!!!!!!!!!!!!!!!!!!!!")

        cmd_to_vals = self._command_to_values(command)
        await self.set_charge_from_grid(cmd_to_vals["charge_from_grid"])
        await self.set_max_discharging_power(cmd_to_vals["max_discharging_power"])
        await self.set_max_charging_power(cmd_to_vals["max_charging_power"])

        return False

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        cmd_to_vals = self._command_to_values(command)

        is_charge_from_grid = await self.is_charge_from_grid()

        if cmd_to_vals["charge_from_grid"] is not None and is_charge_from_grid is None:
            _LOGGER.debug("probe_if_command_set: battery probe_if_command_set ret None, is_charge_from_grid None")
            return None

        max_discharge_power = self.get_max_discharging_power()

        if cmd_to_vals["max_discharging_power"] is not None and max_discharge_power is None:
            _LOGGER.debug("probe_if_command_set: battery probe_if_command_set ret None, max_discharge_power None")
            return None

        # Compare against the value that actually LANDS on the number entity
        # (unit-converted, min/max-clamped, step-snapped) so a kW-denominated or
        # stepped entity does not make the probe never confirm (eternal retry).
        expected_max_discharge = cmd_to_vals["max_discharging_power"]
        if expected_max_discharge is not None:
            _, expected_max_discharge = self._discharge_number_target(expected_max_discharge)

        max_charge_power = self.get_max_charging_power()

        if cmd_to_vals["max_charging_power"] is not None and max_charge_power is None:
            _LOGGER.debug("probe_if_command_set: battery probe_if_command_set ret None, max_charge_power None")
            return None

        # same landed-value mapping on the charge leg (R5: kW-denominated
        # max_charge_number would otherwise never confirm)
        expected_max_charge = cmd_to_vals["max_charging_power"]
        if expected_max_charge is not None:
            _, expected_max_charge = self._charge_number_target(expected_max_charge)

        return (
            is_charge_from_grid == cmd_to_vals["charge_from_grid"]
            and max_discharge_power == expected_max_discharge
            and max_charge_power == expected_max_charge
        )

    def _number_entity_target(self, entity_id: str | None, power_w: float, snap_up: bool) -> tuple[float, int]:
        """Map a W power target to (value_to_write, expected_landed_w) for a number entity.

        Mirrors the read path (`convert_power_to_w`) so the write, the
        read-back, and the probe comparison all agree even when the entity
        is kW-denominated or carries a min/max/step that would snap the
        value. `snap_up=True` (discharge floor) rounds the step **up** so a
        safety minimum is never lowered — nearest-step rounding could snap a
        floor down to 0 and then confirm the zeroed floor. `snap_up=False`
        (charge limit, not a safety minimum) rounds to the nearest step.
        Falls back to a raw W passthrough when the entity attributes are
        unreadable.
        """
        write_value = float(power_w)
        attributes: dict = {}
        if entity_id is not None:
            state = self.hass.states.get(entity_id)
            if state is not None and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                attributes = state.attributes or {}

        unit = attributes.get(ATTR_UNIT_OF_MEASUREMENT, UnitOfPower.WATT)
        to_entity_unit = unit in UnitOfPower and unit != UnitOfPower.WATT
        if to_entity_unit:
            write_value = PowerConverter.convert(value=write_value, from_unit=UnitOfPower.WATT, to_unit=unit)

        ent_min = attributes.get("min")
        if ent_min is not None:
            write_value = max(write_value, float(ent_min))
        ent_max = attributes.get("max")
        if ent_max is not None:
            write_value = min(write_value, float(ent_max))

        step = attributes.get("step")
        if step:
            step = float(step)
            if snap_up:
                # a discharge floor is a safety minimum — never snap it below the
                # request; then re-cap to the entity max (ceil cannot undershoot min)
                write_value = math.ceil(write_value / step) * step
                if ent_max is not None:
                    write_value = min(write_value, float(ent_max))
            else:
                write_value = round(write_value / step) * step

        landed_w = write_value
        if to_entity_unit:
            landed_w = PowerConverter.convert(value=write_value, from_unit=unit, to_unit=UnitOfPower.WATT)

        return write_value, int(round(landed_w))

    def _discharge_number_target(self, power_w: float) -> tuple[float, int]:
        """Discharge floor target — snaps UP (a safety minimum is never lowered)."""
        return self._number_entity_target(self.max_discharge_number, power_w, snap_up=True)

    def _charge_number_target(self, power_w: float) -> tuple[float, int]:
        """Charge limit target — nearest-step (not a safety minimum)."""
        return self._number_entity_target(self.max_charge_number, power_w, snap_up=False)

    async def set_charge_from_grid(self, charge_from_grid: bool | None, blocking: bool = False):
        if self.charge_from_grid_switch is None or charge_from_grid is None:
            return

        if self.is_charge_from_grid_current == charge_from_grid:
            return

        if charge_from_grid:
            action = SERVICE_TURN_ON
        else:
            action = SERVICE_TURN_OFF

        _LOGGER.info("set_charge_from_grid: battery %s %s %s", charge_from_grid, self.charge_from_grid_switch, action)

        try:
            await self.hass.services.async_call(
                domain=Platform.SWITCH, service=action, target={"entity_id": self.charge_from_grid_switch}
            )
        except Exception as e:
            _LOGGER.error(
                f"set_charge_from_grid: battery error setting charge from grid {e}", exc_info=True, stack_info=True
            )

    async def is_charge_from_grid(self) -> bool | None:
        if self.charge_from_grid_switch is None:
            return None

        state = self.hass.states.get(self.charge_from_grid_switch)
        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            res = None
        else:
            res = state.state == "on"

        _LOGGER.info("is_charge_from_grid: battery %s", res)

        self.is_charge_from_grid_current = res
        return res

    async def set_max_discharging_power(self, power: float | None = None, blocking: bool = False):
        if self.max_discharge_number is None or power is None:
            return

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_discharge_number}
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_discharging_power)
        max_value = float(self.max_discharging_power)

        # domain clamp to [min_discharging_power, max_discharging_power] (W), then
        # map to the entity's unit/step/range so the read-back and probe agree
        clamped_w = min(max_value, max(min_value, float(power)))
        val, expected_w = self._discharge_number_target(clamped_w)

        if expected_w == self.get_max_discharging_power():
            return

        data[number.ATTR_VALUE] = val
        domain = number.DOMAIN

        _LOGGER.info(
            "set_max_discharging_power:battery %s %s %s %s %s", val, self.max_discharge_number, domain, service, data
        )

        try:
            await self.hass.services.async_call(domain, service, data, blocking=blocking)
        except Exception as e:
            _LOGGER.error(
                f"set_max_discharging_power: battery error setting max discharging power {e}",
                exc_info=True,
                stack_info=True,
            )

    def get_max_discharging_power(self):
        res = None
        if self.max_discharge_number is not None:
            state = self.hass.states.get(self.max_discharge_number)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                res = None
            else:
                try:
                    res = float(state.state)
                    res, _ = convert_power_to_w(value=res, attributes=state.attributes)
                    res = int(round(res))
                    _LOGGER.info("get_max_discharging_power: battery %s %s", res, self.max_discharge_number)
                except:
                    res = None
                    _LOGGER.warning("get_max_discharging_power: battery NONE %s", self.max_discharge_number)

        return res

    def clamp_charge_power(self, power: float) -> float:

        if power >= 0:
            max_charge_power = self.get_max_charging_power()
            if max_charge_power is not None:
                return min(power, max_charge_power)
            return power
        else:
            max_discharge_power = self.get_max_discharging_power()
            if max_discharge_power is not None:
                return max(power, -max_discharge_power)
            return power

    def get_max_charging_power(self):

        res = None
        if self.max_charge_number is not None:
            state = self.hass.states.get(self.max_charge_number)
            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                res = None
            else:
                try:
                    res = float(state.state)
                    res, _ = convert_power_to_w(value=res, attributes=state.attributes)
                    res = int(round(res))
                    _LOGGER.info("get_max_charging_power: battery %s  %s", res, self.max_charge_number)
                except:
                    res = None
                    _LOGGER.warning("get_max_charging_power: battery NONE  %s", self.max_charge_number)

        return res

    async def set_max_charging_power(self, power: float | None = None, blocking: bool = False):
        if self.max_charge_number is None or power is None:
            return

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_charge_number}
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_charging_power)
        max_value = float(self.max_charging_power)

        # domain clamp to [min_charging_power, max_charging_power] (W), then map to
        # the entity's unit/step/range so the read-back and probe agree (R5)
        clamped_w = min(max_value, max(min_value, float(power)))
        val, expected_w = self._charge_number_target(clamped_w)

        if expected_w == self.get_max_charging_power():
            return

        data[number.ATTR_VALUE] = val
        domain = number.DOMAIN

        _LOGGER.info(
            "set_max_charging_power: battery %s %s %s %s %s", val, self.max_charge_number, domain, service, data
        )

        try:
            await self.hass.services.async_call(domain, service, data, blocking=blocking)
        except Exception as e:
            _LOGGER.error(
                f"set_max_charging_power: battery error setting max charging power {e}", exc_info=True, stack_info=True
            )

    def get_current_battery_asked_change_for_outside_production_system(self) -> float:

        if self.current_command is None:
            return 0.0

        if self.current_command.power_consign == 0.0:
            return 0.0

        if self.is_dc_coupled is False:
            return self.current_command.power_consign

        if self.current_command.power_consign > 0:
            inverter_clamp = self.home.get_current_over_clamp_production_power()
            if inverter_clamp > 0:
                _LOGGER.warning(
                    f"get_current_battery_asked_change_for_outside_production_system: reduce power command {self.current_command.power_consign:.2f} by {inverter_clamp:.2f} to {self.current_command.power_consign - inverter_clamp}"
                )
            return max(0, self.current_command.power_consign - inverter_clamp)

        return self.current_command.power_consign

    def battery_can_discharge(self):
        return self.battery_get_current_possible_max_discharge_power() > 0.0

    def get_platforms(self):
        parent = super().get_platforms()
        parent = set(parent)
        parent.update([Platform.SENSOR])
        return list(parent)
