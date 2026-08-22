import logging
from datetime import datetime
from typing import Any

from homeassistant.components import number
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    Platform,
)

from ..const import (
    CONF_BATTERY_CHARGE_DISCHARGE_SENSOR,
    CONF_BATTERY_CHARGE_FROM_GRID_SWITCH,
    CONF_BATTERY_CHARGE_PERCENT_SENSOR,
    CONF_BATTERY_MAX_CHARGE_POWER_NUMBER,
    CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER,
    CONF_TYPE_NAME_QSBattery,
)
from ..ha_model.device import HADeviceMixin, convert_power_to_w
from ..ha_model.ha_utils import NumberEntityTargeter
from ..home_model.battery import Battery, coerce_finite_float
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

    # Back-compat alias for the divergence-latch cap, now owned by the shared
    # NumberEntityTargeter (review-fix #10 AA2).
    _NUMBER_DIVERGENCE_LATCH_MAX = NumberEntityTargeter.LATCH_MAX

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
        # The number-entity write/read/snap/divergence machinery lives in the
        # shared NumberEntityTargeter (review-fix #10 AA2); this battery owns one
        # instance (it holds the per-device divergence latch). The battery keeps
        # only thin wrappers below plus its command semantics.
        self._number_helper = NumberEntityTargeter(self.hass)

    @property
    def current_charge(self) -> float | None:
        percent = self.get_sensor_latest_possible_valid_value(entity_id=self.charge_percent_sensor)
        if percent is None:
            return None
        return float(percent * self.capacity) / 100.0

    def _command_to_values(self, command: LoadCommand) -> dict[str, Any]:
        # AA1: `_command_to_values` is the single authority on the discharge
        # value AND its snap semantics — "max_discharging_power_snap_up" says
        # whether that value is the safety floor (snap UP, never lower a minimum)
        # or a restore/max (snap DOWN, never raise past the configured limit).
        # Both the write and the probe read this flag, so they can never derive
        # opposite snap directions.
        if command.is_like_one_of_cmds([CMD_ON, CMD_IDLE, CMD_AUTO_GREEN_ONLY, CMD_GREEN_CHARGE_AND_DISCHARGE]):
            ret = {
                "charge_from_grid": False,
                "max_discharging_power": self.max_discharging_power,
                "max_discharging_power_snap_up": False,
                "max_charging_power": self.max_charging_power,
            }
        elif command.is_like(CMD_GREEN_CHARGE_ONLY):
            # emit the outage safety floor (default 0) instead of a hard 0
            ret = {
                "charge_from_grid": False,
                "max_discharging_power": self.min_discharging_power,
                "max_discharging_power_snap_up": True,
                "max_charging_power": self.max_charging_power,
            }
        elif command.is_like(CMD_FORCE_CHARGE):
            ret = {
                "charge_from_grid": True,
                "max_discharging_power": self.min_discharging_power,
                "max_discharging_power_snap_up": True,
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
        await self.set_max_discharging_power(
            cmd_to_vals["max_discharging_power"], snap_up=cmd_to_vals["max_discharging_power_snap_up"]
        )
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
        # (domain-clamped, unit-converted, min/max-clamped, step-snapped) so a
        # kW-denominated or stepped entity does not make the probe never confirm
        # (eternal retry). Snap direction must match the write's (T3). The probe
        # is a pure read — never emit the divergence warning from here (V6).
        expected_max_discharge = cmd_to_vals["max_discharging_power"]
        expected_discharge_write: float | None = None
        if expected_max_discharge is not None:
            expected_discharge_write, expected_max_discharge = self._discharge_number_target(
                expected_max_discharge, snap_up=cmd_to_vals["max_discharging_power_snap_up"], warn=False
            )

        max_charge_power = self.get_max_charging_power()

        if cmd_to_vals["max_charging_power"] is not None and max_charge_power is None:
            _LOGGER.debug("probe_if_command_set: battery probe_if_command_set ret None, max_charge_power None")
            return None

        # same landed-value mapping on the charge leg (R5: kW-denominated
        # max_charge_number would otherwise never confirm)
        expected_max_charge = cmd_to_vals["max_charging_power"]
        expected_charge_write: float | None = None
        if expected_max_charge is not None:
            expected_charge_write, expected_max_charge = self._charge_number_target(expected_max_charge, warn=False)

        discharge_matches = self._number_reading_matches(
            self.max_discharge_number,
            max_discharge_power,
            expected_max_discharge,
            expected_discharge_write,
            self.max_discharging_power,
        )
        charge_matches = self._number_reading_matches(
            self.max_charge_number,
            max_charge_power,
            expected_max_charge,
            expected_charge_write,
            self.max_charging_power,
        )
        # V5/W3: each confirmed entity clears ONLY its own resolved latch entries
        # (never the whole set), so a divergence that later recurs warns again
        # while a still-current divergence stays latched (no per-cycle re-warn).
        # X3: clear against the EXPECTED landed value, not the reading — an
        # echo-confirm reads a step-neighbour of the landed value, and the latch
        # keys the landed value; clearing with the reading would drop the
        # still-current entry and re-warn every execute/probe cycle.
        if discharge_matches:
            self._clear_number_divergence_latch(self.max_discharge_number, expected_max_discharge)
        if charge_matches:
            self._clear_number_divergence_latch(self.max_charge_number, expected_max_charge)

        return is_charge_from_grid == cmd_to_vals["charge_from_grid"] and discharge_matches and charge_matches

    # ---- thin delegators to the shared NumberEntityTargeter (review-fix #10 AA2) ----
    # The battery keeps these wrappers (and its command semantics above) as its
    # only glue; all the number-entity machinery lives in ha_utils.

    @property
    def _number_divergence_warned(self) -> dict:
        """The shared helper's per-device divergence latch (read-only view)."""
        return self._number_helper._divergence_warned

    def _number_reading_matches(
        self, entity_id: str | None, read_value, expected_value, write_value: float | None, domain_max_w: float
    ) -> bool:
        return self._number_helper.reading_matches(entity_id, read_value, expected_value, write_value, domain_max_w)

    def _entity_step(self, entity_id: str | None, domain_max_w: float) -> tuple[float, float]:
        return self._number_helper.entity_step(entity_id, domain_max_w)

    def _clear_number_divergence_latch(self, entity_id: str | None, confirmed_w: int | None) -> None:
        self._number_helper.clear_latch(entity_id, confirmed_w)

    def _discharge_number_target(self, power_w: float, snap_up: bool, warn: bool = True) -> tuple[float, int]:
        """Discharge target. `snap_up` only for the floor (a safety minimum)."""
        return self._number_helper.target(
            self.max_discharge_number,
            power_w,
            snap_up=snap_up,
            domain_min=self.min_discharging_power,
            domain_max=self.max_discharging_power,
            warn=warn,
        )

    def _charge_number_target(self, power_w: float, warn: bool = True) -> tuple[float, int]:
        """Charge limit target — snaps DOWN (a limit is never raised past its cap)."""
        return self._number_helper.target(
            self.max_charge_number,
            power_w,
            snap_up=False,
            domain_min=self.min_charging_power,
            domain_max=self.max_charging_power,
            warn=warn,
        )

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

    def _number_entity_writable(self, entity_id: str) -> bool:
        """False while the number entity is unknown/unavailable (U4).

        Skip the write then: its unit/min/max/step attributes are missing, so we
        would map with a raw-W fallback (a kW entity would get a 300 -> 300 kW
        write). The next cycle retries once the entity is back with fresh
        attributes, so this is self-healing.
        """
        state = self.hass.states.get(entity_id)
        return state is not None and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)

    async def set_max_discharging_power(
        self, power: float | None = None, blocking: bool = False, *, snap_up: bool = False
    ):
        if self.max_discharge_number is None or power is None:
            return
        if not self._number_entity_writable(self.max_discharge_number):
            # X8/Y5: the grid switch has already flipped by now — surface the
            # half-applied command at info so default installs record it
            _LOGGER.info(
                "set_max_discharging_power: %s unavailable, deferring write of %s W",
                self.max_discharge_number,
                power,
            )
            return

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_discharge_number}
        service = number.SERVICE_SET_VALUE

        # the helper owns the domain clamp + entity unit/step/range mapping, so
        # the read-back and probe agree; snap_up only for the safety floor (T3)
        val, expected_w = self._discharge_number_target(float(power), snap_up=snap_up)

        # W1(c): same step-aware comparison as the probe — a quantized device
        # echo is already the landed value (skip, no write/re-quantize churn),
        # while a genuinely wrong reading re-issues the write.
        if self._number_reading_matches(
            self.max_discharge_number, self.get_max_discharging_power(), expected_w, val, self.max_discharging_power
        ):
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
                except TypeError, ValueError:
                    res = None
                else:
                    # X2: int(round(inf)) raises OverflowError — a non-finite
                    # reading is unparsable, honour the None contract instead
                    res = coerce_finite_float(res, None)
                if res is None:
                    _LOGGER.warning("get_max_discharging_power: battery NONE %s", self.max_discharge_number)
                else:
                    res = int(round(res))
                    _LOGGER.info("get_max_discharging_power: battery %s %s", res, self.max_discharge_number)

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
                except TypeError, ValueError:
                    res = None
                else:
                    # X2: see get_max_discharging_power — non-finite is unparsable
                    res = coerce_finite_float(res, None)
                if res is None:
                    _LOGGER.warning("get_max_charging_power: battery NONE  %s", self.max_charge_number)
                else:
                    res = int(round(res))
                    _LOGGER.info("get_max_charging_power: battery %s  %s", res, self.max_charge_number)

        return res

    async def set_max_charging_power(self, power: float | None = None, blocking: bool = False):
        if self.max_charge_number is None or power is None:
            return
        if not self._number_entity_writable(self.max_charge_number):
            # X8/Y5: see set_max_discharging_power — surface the deferred write at info
            _LOGGER.info(
                "set_max_charging_power: %s unavailable, deferring write of %s W",
                self.max_charge_number,
                power,
            )
            return

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_charge_number}
        service = number.SERVICE_SET_VALUE

        # the helper owns the domain clamp + entity unit/step/range mapping, so
        # the read-back and probe agree even for a consign above max (T2)
        val, expected_w = self._charge_number_target(float(power))

        # W1(c): same step-aware comparison as the probe (see set_max_discharging_power)
        if self._number_reading_matches(
            self.max_charge_number, self.get_max_charging_power(), expected_w, val, self.max_charging_power
        ):
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
