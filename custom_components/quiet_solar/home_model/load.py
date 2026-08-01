import copy
import logging
import random
from bisect import bisect_left
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import pytz
import slugify

from ..const import (
    CHANGE_ON_OFF_STATE_HYSTERESIS_S,
    CONF_DEVICE_DASHBOARD_SECTION,
    CONF_DEVICE_DYNAMIC_GROUP_NAME,
    CONF_DEVICE_EFFICIENCY,
    CONF_DEVICE_TO_PILOT_NAME,
    CONF_IS_3P,
    CONF_LOAD_IS_BOOST_ONLY,
    CONF_MONO_PHASE,
    CONF_NUM_MAX_ON_OFF,
    CONF_POWER,
    CONF_SWITCH,
    CONSTRAINT_ORIGINATOR_AGENDA,
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_USER_OVERRIDE,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    DASHBOARD_NO_SECTION,
    DEVICE_STATUS_CHANGE_CONSTRAINT,
    DEVICE_STATUS_CHANGE_CONSTRAINT_COMPLETED,
    DEVICE_STATUS_CHANGE_ERROR,
    LOAD_TYPE_DASHBOARD_DEFAULT_SECTION,
    NOTIFICATION_TAG_LOST_CONTROL_PREFIX,
    OVERRIDE_STATE_ASKED_FOR_RESET,
    OVERRIDE_STATE_NO_OVERRIDE,
    OVERRIDE_STATE_PREFIX,
    STORAGE_KEY_ASKED_FOR_RESET_FIRST_CMD_RESET_DONE,
    STORAGE_KEY_ASKED_FOR_RESET_TIME,
    STORAGE_KEY_CURRENT_COMMAND,
    STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE,
    STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME,
    STORAGE_KEY_LAST_CHECK_UPDATE,
    STORAGE_KEY_LAST_STATE_CHANGE_TIME,
    STORAGE_KEY_NUM_ON_OFF,
    STORAGE_KEY_USER_ORIGINATED,
)
from .commands import CMD_IDLE, LoadCommand, copy_command
from .constraints import DATETIME_MAX_UTC, DATETIME_MIN_UTC, LoadConstraint

if TYPE_CHECKING:
    import QSDynamicGroup

NUM_MAX_INVALID_PROBES_COMMANDS = 10

# QS-304: the in-flight command relaunch ladder. `NUM_MAX_COMMAND_RELAUNCH` is
# the rung at which the linear backoff STOPS GROWING — it is not a give-up
# count. The cumulative wall time of the growing part is unchanged at
# 50 + 100 + 150 + 200 + 250 + 300 = 1050 s, after which QS declares it has lost
# control of the load and keeps retrying every 300 s, forever.
COMMAND_RELAUNCH_BASE_DELAY_S = 50
NUM_MAX_COMMAND_RELAUNCH = 6

# QS-304: minimum interval between two supersessions of a stale in-flight
# command on a load QS has lost control of. Equal to the saturated ladder delay, so
# a disobedient device sees at most one service call per 300 s **per command
# identity**: the relaunch ladder and the supersede throttle are measured on two
# independent anchors (`running_command_last_launch` and `_last_supersede_time`), so
# a relaunch immediately followed by a superseding command can produce two calls
# inside one nominal window. Bounded to one extra call per episode, and deliberately
# not coupled — sharing one clock would make a supersede delay the ladder, or a
# relaunch delay a newer intent.
SUPERSEDE_MIN_INTERVAL_S = COMMAND_RELAUNCH_BASE_DELAY_S * NUM_MAX_COMMAND_RELAUNCH


class ContactEvidence(StrEnum):
    """What releasing the lost-control clock says about the device (QS-307).

    That question — not the release itself — is what decides whether a lost-control
    EPISODE has ended, and therefore whether the next threshold crossing is worth
    announcing. An `enum` rather than bare strings so a typo is an `AttributeError`
    at the call site instead of a value that silently matches no branch;
    `_clear_unresponsive` takes it as a REQUIRED argument for the same reason, since
    defaulting to the episode-ending value would let a future caller end an incident
    by omission.
    """

    CONFIRMED = "confirmed"  # the device answered: the episode is over
    UNREACHABLE = "unreachable"  # we could not reach it: the episode continues
    UNKNOWN = "unknown"  # we abandoned the command: says nothing either way


# Tolerance for a clearly FUTURE-dated stored timestamp at the RESTORE boundary, and
# nowhere else: `QSBiStateDuration.use_saved_extra_device_info` is the only consumer.
# The runtime override comparisons use plain subtraction on purpose (QS-307), so
# retuning this does not widen any live window.
CLOCK_SKEW_TOLERANCE_S = 60.0

_LOGGER = logging.getLogger(__name__)


def extract_name_and_index_from_dashboard_section_option(section_option: str) -> tuple[str, int | None]:
    name = section_option
    vals = section_option.split(" - ")
    found_index = None
    if len(vals) > 0 and vals[0].startswith("#"):
        try:
            idx = int(vals[0][1:])
            if idx >= 1:
                found_index = idx - 1
        except:
            found_index = None

        if found_index is not None:
            name = section_option[len(vals[0]) + 3 :]

    return name, found_index


def map_section_selected_name_in_section_list(
    section_stored_name: str, section_list: list[tuple[str, str]], compute_options: bool = False
) -> tuple[int | None, list[str] | None]:

    options: list[str] | None = None
    if compute_options:
        options = [DASHBOARD_NO_SECTION]
        for i, sn in enumerate(section_list):
            options.append(f"#{i + 1} - {sn[0]}")

    ret = None
    if section_stored_name != DASHBOARD_NO_SECTION:
        # no need to adapt
        name, found_index = extract_name_and_index_from_dashboard_section_option(section_stored_name)

        found_name_idx = None
        for i, sn in enumerate(section_list):
            if sn[0] == name:
                found_name_idx = i
                break

        # give more weight to a name match than an index match
        if found_name_idx is not None and found_name_idx < len(section_list):
            ret = found_name_idx
        elif found_index is not None and found_index < len(section_list):
            ret = found_index

    return ret, options


class AbstractDevice:
    conf_type_name = "unknown"

    def is_user_overridden(self) -> bool | None:
        """Return whether device is currently user-overridden.

        Returns False by default. Subclasses override for actual logic.
        Return type: True = all overridden, False = none, None = mixed.
        """
        return False

    def __init__(self, name: str, device_type: str | None = None, **kwargs):
        super().__init__()
        self._enabled = True
        self._user_originated: dict[str, Any] = {}
        self._in_user_originated_update: bool = False
        self._power_use_conf = kwargs.pop(CONF_POWER, None)
        self.efficiency = float(min(kwargs.pop(CONF_DEVICE_EFFICIENCY, 100.0), 100.0))
        self._device_is_3p_conf = kwargs.pop(CONF_IS_3P, False)
        self.dynamic_group_name = kwargs.pop(CONF_DEVICE_DYNAMIC_GROUP_NAME, None)
        self.piloted_device_name = kwargs.pop(CONF_DEVICE_TO_PILOT_NAME, None)
        self._mono_phase_conf: str | None = kwargs.pop(CONF_MONO_PHASE, None)
        if self._mono_phase_conf is None:
            # at random allocate phase on 0, 1, or 2
            self._mono_phase_default = random.randint(0, 2)
        else:
            self._mono_phase_default = int(self._mono_phase_conf) - 1

        self._conf_dashboard_section_option = kwargs.pop(CONF_DEVICE_DASHBOARD_SECTION, None)

        self._device_type = device_type

        # device_type is a clever property ... giving back a proper type
        device_type = self.device_type

        if self._conf_dashboard_section_option is None and device_type is not None:
            self._conf_dashboard_section_option = LOAD_TYPE_DASHBOARD_DEFAULT_SECTION.get(device_type)

        if self._conf_dashboard_section_option is None:
            self._conf_dashboard_section_option = DASHBOARD_NO_SECTION

        self.name = name

        self.devices_to_pilot: list[PilotedDevice] = []

        self.device_id = f"qs_{slugify.slugify(name, separator='_')}_{self.device_type}"
        self.home = kwargs.pop("home", None)

        self._dampen_start_transition: datetime | None = None
        self._dampened_computed_power_use: float | None = kwargs.pop(f"measured_{CONF_POWER}", None)

        # QS-304: the "we gave up on this command" clock and the supersede-throttle
        # anchor. Initialised BEFORE the reset below so a `keep_commands=False` wipe
        # can clear them — that wipe destroys the command the clock describes.
        self.unresponsive_since: datetime | None = None
        self._last_supersede_time: datetime | None = None
        # QS-307: the per-EPISODE latch, as against `unresponsive_since`'s
        # per-command clock. Gates the shout, never the clock. See
        # `_clear_unresponsive` and `docs/agents/concepts/load-base.md`.
        self._unresponsive_needs_ack: bool = False

        self.constraint_reset_and_reset_commands_if_needed(keep_commands=False)
        self.last_check_update: datetime | None = None
        self.reset_daily_load_datas()

        # QS-256 (D5): causality anchor — the time of the last REAL command
        # execution (service call). In-memory only, never serialized, no
        # entity/diagnostic exposure. Used by the user-override detection to
        # ignore entity states older than the system's own last action.
        self.last_command_execution_time: datetime | None = None

        self._ack_command(None, None)

        self.num_max_on_off: str | None | int = kwargs.pop(CONF_NUM_MAX_ON_OFF, None)
        if self.num_max_on_off is not None:
            self.num_max_on_off = int(self.num_max_on_off)
            if self.num_max_on_off % 2 == 1:
                self.num_max_on_off += 1

        self.father_device: QSDynamicGroup = self.home

        self._computed_dashboard_section = None

    def set_user_originated(self, key: str, value: Any) -> None:
        self._user_originated[key] = value
        if not self._in_user_originated_update:
            self._in_user_originated_update = True
            try:
                self._on_user_originated_changed(key, value)
            finally:
                self._in_user_originated_update = False

    def _on_user_originated_changed(self, key: str, value: Any) -> None:
        """Override in subclasses to react to user-originated state changes."""

    def get_user_originated(self, key: str, default: Any = None) -> Any:
        return self._user_originated.get(key, default)

    def has_user_originated(self, key: str) -> bool:
        return key in self._user_originated

    def clear_user_originated(self, key: str) -> None:
        self._user_originated.pop(key, None)

    def clear_all_user_originated(self) -> None:
        self._user_originated.clear()

    def is_device_light_on(self) -> bool:
        if self.current_command is None or self.current_command.is_off_or_idle():
            return False
        return True

    @property
    def power_use(self):
        if self._dampened_computed_power_use is not None:
            return float(self._dampened_computed_power_use)
        if self._power_use_conf is None:
            return None
        return float(self._power_use_conf)

    @power_use.setter
    def power_use(self, power: float | None):
        self._dampened_computed_power_use = power

    def get_possible_delta_power_from_piloted_devices_for_budget(self, slot_idx: int | None, add: bool = True) -> float:
        if len(self.devices_to_pilot) == 0:
            return 0.0

        power_delta = 0.0
        for pd in self.devices_to_pilot:
            power_delta += pd.possible_delta_power_for_slot(slot_idx, add)

        return power_delta

    def update_demanding_clients_for_piloted_devices_for_budget(self, slot_idx: int, add: bool) -> int | float:
        if len(self.devices_to_pilot) == 0:
            return 0.0

        power_delta = 0.0
        for pd in self.devices_to_pilot:
            power_delta += pd.update_num_demanding_clients_for_slot(slot_idx, add)

        return power_delta

    def get_phase_amps_from_power_for_piloted_budgeting(self, power: float) -> list[float | int]:
        if len(self.devices_to_pilot) == 0:
            return [0.0, 0.0, 0.0]

        ret = [0.0, 0.0, 0.0]
        for pd in self.devices_to_pilot:
            pd_amps = pd.get_phase_amps_from_power_for_budgeting(power / len(self.devices_to_pilot))
            ret[0] += pd_amps[0]
            ret[1] += pd_amps[1]
            ret[2] += pd_amps[2]

        return ret

    def is_off_grid(self) -> bool:
        if self.home:
            return self.home.is_off_grid()
        return False

    @property
    def dashboard_section(self) -> str | None:
        if (
            self._computed_dashboard_section is None
            and self.home is not None
            and self.home.dashboard_sections is not None
            and len(self.home.dashboard_sections) > 0
        ):
            idx, _ = map_section_selected_name_in_section_list(
                self._conf_dashboard_section_option, self.home.dashboard_sections, compute_options=False
            )
            if idx is not None:
                self._computed_dashboard_section = self.home.dashboard_sections[idx][0]
            else:
                # WF-5 tier 1: surface the resolution failure in logs.
                # This happens when the user has a customised
                # dashboard_sections list that pre-dates a new device
                # type (e.g. water_boiler added in QS-194). Log once
                # per device — the cache `_computed_dashboard_section`
                # prevents repeats on the second property access.
                if self._conf_dashboard_section_option != DASHBOARD_NO_SECTION:
                    _LOGGER.warning(
                        "Device %r requested dashboard section %r but it is "
                        "not present in home.dashboard_sections (%s); device "
                        "will not appear on the dashboard until you add the "
                        "section in home settings",
                        self.name,
                        self._conf_dashboard_section_option,
                        [s[0] for s in self.home.dashboard_sections],
                    )
                self._computed_dashboard_section = DASHBOARD_NO_SECTION

        if self._computed_dashboard_section == DASHBOARD_NO_SECTION:
            return None

        return self._computed_dashboard_section

    @property
    def dashboard_sort_string_in_type(self) -> str:
        return "ZZZ"

    @property
    def dashboard_sort_string(self) -> int:
        ret = ""
        for s in [self.device_type, self.dashboard_sort_string_in_type, self.name]:
            if len(s) >= 255:
                ret += s[:255]
            else:
                ret += s
                ret += " " * (255 - len(s))

        return ret

    @property
    def voltage(self) -> float:
        """Return the voltage of the home."""
        if self.home is not None:
            return self.home.voltage
        return 230.0

    async def async_get_info_from_storage(self, time: datetime, stored_device_info: dict | None):
        if stored_device_info:
            self.use_saved_extra_device_info(stored_device_info)

    def update_available_amps_for_group(self, idx: int, amps: list[float | int], add: bool):
        """Update the available amps for the group based on the device's configuration."""
        if self.father_device is not None:
            return self.father_device.update_available_amps_for_group(idx, amps, add)

    def _has_state_to_reset(self, keep_commands: bool) -> bool:
        """Return True when this reset will actually destroy something.

        QS-306: the reset logs INFO only when it did work. Subclasses that clear
        EXTRA state in their override must extend this predicate, otherwise their
        work is destroyed while the base reports "nothing to reset" at DEBUG.

        Every `getattr` default is required: `AbstractDevice.__init__` calls
        `constraint_reset_and_reset_commands_if_needed` (see the call in
        `__init__`) BEFORE `_constraints` / `current_command` exist, and
        `AbstractLoad` assigns `_last_completed_constraint` only after its own
        `super().__init__()` returns. Overrides must follow the same rule.
        """
        had_constraints = bool(getattr(self, "_constraints", None)) or (
            getattr(self, "_last_completed_constraint", None) is not None
        )
        # `keep_commands=False` also drops `running_command` — an in-flight command
        # awaiting verification — and `_stacked_command`, one queued behind it. All
        # three are loggable work.
        had_commands = not keep_commands and (
            getattr(self, "current_command", None) is not None
            or getattr(self, "running_command", None) is not None
            or getattr(self, "_stacked_command", None) is not None
        )
        return had_constraints or had_commands

    def constraint_reset_and_reset_commands_if_needed(self, keep_commands=True):
        # `AbstractLoad`'s override calls `super()` first, so the predicate still
        # observes the pre-clear values.
        if self._has_state_to_reset(keep_commands):
            _LOGGER.info("Constraint Reset device %s", self.name)
        else:
            _LOGGER.debug("Constraint Reset device %s, nothing to reset", self.name)
        self._constraints: list[LoadConstraint | None] = []
        if keep_commands is False:
            self.current_command: LoadCommand | None = None
            self.prev_command: LoadCommand | None = None
            self.running_command: LoadCommand | None = (
                None  # a command that has been launched but not yet finished, wait for its resolution
            )
            self._stacked_command: LoadCommand | None = (
                None  # a command (keep only the last one) that has been pushed to be executed later when running command is free
            )
            self.running_command_first_launch: datetime | None = None
            self.running_command_last_launch: datetime | None = None
            self.running_command_num_relaunch: int = 0
            self.running_command_num_relaunch_after_invalid: int = 0
            # This wipe destroys BOTH `current_command` and
            # `running_command` — the very command the lost-control clock
            # describes — so the clock goes with it. Leaving it behind made it
            # ownerless: the only re-arm path (`_escalate_or_recover`) is gated
            # on `current_command is not None`, so after the load's reset button
            # or a disable/re-enable the next command launched would flip
            # `is_uncontrollable` True on its very first cycle, with zero
            # relaunches — flashing PROBLEM right after the user's own
            # remediation. Routed through `_clear_unresponsive` to keep the
            # single-writer / one-line-out guarantee.
            # `UNKNOWN`: a wipe is OUR decision, not the device answering, so
            # it must not end a lost-control episode (QS-307).
            self._clear_unresponsive("the command state was reset", contact=ContactEvidence.UNKNOWN)

    # for class overcharging reset
    def reset(self, keep_commands=False):
        _LOGGER.info("Reset device %s", self.name)
        self.constraint_reset_and_reset_commands_if_needed(keep_commands=keep_commands)
        self.reset_daily_load_datas()
        self._dampen_start_transition = None

    async def user_clean_and_reset(self):
        _LOGGER.info("user_clean_and_reset device %s", self.name)
        self.clear_all_user_originated()
        # QS-319: pressing reset is explicit remediation, so it ends any open
        # lost-control episode. Cleared BEFORE `reset()` — ordering is not
        # load-bearing today (the wipe releases with `UNKNOWN`, which never *sets*
        # the latch) but a future widening of `UNKNOWN` must not silently break this.
        self._acknowledge_lost_control("the user reset the device")
        self.reset()

    async def user_clean_constraints(self):
        _LOGGER.info("user_clean_constraints device %s", self.name)
        self.constraint_reset_and_reset_commands_if_needed(keep_commands=True)

    @property
    def qs_enable_device(self) -> bool:
        return self._enabled

    @qs_enable_device.setter
    def qs_enable_device(self, enabled: bool):
        if enabled != self._enabled:
            # QS-319: inside the guard on purpose. It covers BOTH edges (disable and
            # re-enable), and an idempotent re-write cannot reach it — which matters
            # because `switch.py::QSSwitchEntityWithRestore.async_added_to_hass`
            # drives this setter via `async_turn_on/off(for_init=True)` on every HA
            # startup, and a config-entry options re-apply does the same.
            self._acknowledge_lost_control("the user changed the device's enabled state")
            self.reset()
            self._enabled = enabled
            if self.home is not None:
                if enabled is False:
                    self.home.remove_device(self)
                    self.home.add_disabled_device(self)
                    _LOGGER.info("qs_enable_device: %s DISABLE AND REMOVE", self.name)
                else:
                    self.home.add_device(self)
                    self.home.remove_disabled_device(self)
                    _LOGGER.info("qs_enable_device: %s ENABLE AND ADD", self.name)

            if hasattr(self, "_exposed_entities"):
                time = datetime.now(pytz.utc)
                for ha_object in self._exposed_entities:
                    ha_object.async_update_callback(time)

    def prepare_slots_for_amps_budget(
        self,
        time: datetime,
        num_slots: int,
        from_father_budget: list[float | int] | None,
    ):

        _LOGGER.debug("prepare_slots_for_amps_budget for load %s from_father_budget %s", self.name, from_father_budget)

    @property
    def device_type(self):

        if self._device_type is not None:
            return self._device_type

        if hasattr(self.__class__, "conf_type_name"):
            return self.__class__.conf_type_name

        return self.__class__.__name__

    @property
    def physical_num_phases(self) -> int:
        if self._device_is_3p_conf:
            return 3
        return 1

    @property
    def physical_3p(self) -> bool:
        return self.physical_num_phases == 3

    @property
    def current_num_phases(self) -> int:
        return self.physical_num_phases

    @property
    def current_3p(self) -> bool:
        return self.current_num_phases == 3

    def can_do_3_to_1_phase_switch(self):
        return False

    @property
    def mono_phase_index(self) -> int:

        if self._mono_phase_conf is not None:
            return self._mono_phase_default

        if self.father_device is not None and self.father_device != self.home and not self.father_device.physical_3p:
            return self.father_device.mono_phase_index

        return self._mono_phase_default

    def update_amps_with_delta(
        self, from_amps: list[float | int], delta: int | float, is_3p: bool
    ) -> list[float | int]:
        amps = copy.copy(from_amps)
        if is_3p is False:
            amps[self.mono_phase_index] += delta
        else:
            amps[0] += delta
            amps[1] += delta
            amps[2] += delta
        return amps

    def __repr__(self):
        return self.device_id

    # it is a property as it has to be overchargeable (ex: charger for its car)
    # has to be > 1.0
    @property
    def efficiency_factor(self):
        return 100.0 / self.efficiency

    def update_to_be_saved_extra_device_info(self, data_to_update: dict):
        data_to_update[STORAGE_KEY_NUM_ON_OFF] = self.num_on_off
        data_to_update[STORAGE_KEY_CURRENT_COMMAND] = (
            self.current_command.to_dict() if self.current_command is not None else None
        )
        data_to_update[STORAGE_KEY_LAST_STATE_CHANGE_TIME] = (
            self.last_state_change_time.isoformat() if self.last_state_change_time is not None else None
        )
        data_to_update[STORAGE_KEY_LAST_CHECK_UPDATE] = (
            self.last_check_update.isoformat() if self.last_check_update is not None else None
        )
        data_to_update[STORAGE_KEY_USER_ORIGINATED] = self._user_originated

    def use_saved_extra_device_info(self, stored_load_info: dict):
        self._user_originated = stored_load_info.get(STORAGE_KEY_USER_ORIGINATED, {})
        self.num_on_off = stored_load_info.get(STORAGE_KEY_NUM_ON_OFF, 0)

        if self.num_on_off > 0 and self.num_on_off % 2 == 1:
            # because of a reboot we may need a bit more ...
            self.num_on_off -= 1

        if self.num_max_on_off is not None:
            if self.num_max_on_off - self.num_on_off <= 2:
                self.num_on_off = self.num_max_on_off - 2

        cmd_dict = stored_load_info.get(STORAGE_KEY_CURRENT_COMMAND, None)
        if cmd_dict is not None:
            self.current_command = LoadCommand(**cmd_dict)
            # QS-256 (D5): the restored command is the command of record,
            # unconfirmed as of the restore moment — anchor the causality
            # guard at restore time to close the post-restart blind spot
            self.last_command_execution_time = datetime.now(pytz.UTC)

        last_change_str = stored_load_info.get(STORAGE_KEY_LAST_STATE_CHANGE_TIME, None)
        if last_change_str is not None:
            self.last_state_change_time = datetime.fromisoformat(last_change_str)
        else:
            self.last_state_change_time = None

        last_check_update_update_str = stored_load_info.get(STORAGE_KEY_LAST_CHECK_UPDATE, None)
        if last_check_update_update_str is not None:
            self.last_check_update = datetime.fromisoformat(last_check_update_update_str)
        else:
            self.last_check_update = None

    def reset_daily_load_datas(self, time: datetime | None = None):
        self.num_on_off = 0
        self.last_state_change_time: datetime | None = None

    def get_first_unlocked_slot_index(
        self, time_slots: list[datetime], change_state_hysteresis_s: float = CHANGE_ON_OFF_STATE_HYSTERESIS_S
    ) -> int:
        """Return the first slot index where the solver is allowed to change state.
        All slots from 0 to return_value - 1 must keep the load's current command."""
        if self.num_max_on_off is None or self.last_state_change_time is None:
            return 0
        unlock_time = self.last_state_change_time + timedelta(seconds=change_state_hysteresis_s)
        idx = bisect_left(time_slots, unlock_time)
        return min(idx, len(time_slots))

    def get_min_max_power(self) -> tuple[float, float]:
        return 0.0, 0.0

    def get_phase_amps_from_power_for_budgeting(self, power: float) -> list[float | int]:
        return self.get_phase_amps_from_power(power, is_3p=self.physical_3p)

    def get_phase_amps_from_power(self, power: float, is_3p=False) -> list[float | int]:

        if power == 0.0:
            return [0.0, 0.0, 0.0]

        # shouldn't we use sqrt(3) instead of 3 ? according to chatGPT probably .. should check
        if is_3p:
            power = power / 3.0
        p = power / self.voltage
        if is_3p:
            return [p, p, p]
        else:
            ret = [0.0, 0.0, 0.0]
            ret[self.mono_phase_index] = p
            return ret

    def get_current_active_constraint(self, time: datetime | None = None) -> LoadConstraint | None:
        if self.qs_enable_device is False:
            self._constraints = []

        if not self._constraints:
            self._constraints = []

        if time is None:
            time = datetime.now(tz=pytz.UTC)

        for c in self._constraints:
            if c.is_constraint_active_for_time_period(time):
                return c
        return None

    def _ack_command(self, time: datetime | None, command: LoadCommand | None):

        if command is not None:
            _LOGGER.info("ack command %s for load %s", command.command, self.name)
        else:
            _LOGGER.info("ack command None for load %s", self.name)

        self.prev_command = self.current_command
        self.current_command = command
        self.running_command = None
        self.running_command_num_relaunch = 0
        self.running_command_num_relaunch_after_invalid = 0
        self.running_command_first_launch = None
        self.running_command_last_launch = None

        if command is not None:
            # QS-304: a real ack is the only evidence that control came back.
            # `_ack_command(time, None)` is NOT one — it is the give-up on an
            # unavailable probe, i.e. the device failing harder — and
            # `_ack_command(None, None)` is just `__init__` priming the fields.
            # The give-up releases the clock from its own call site instead, so
            # "acked" and "gave up" stay distinguishable (QS-307).
            self._clear_unresponsive("control returned, the command was acked", contact=ContactEvidence.CONFIRMED)

        if command is not None and time is not None and self.prev_command is not None:
            do_count = False
            if command.is_off_or_idle() and not self.prev_command.is_off_or_idle():
                do_count = True
            elif not command.is_off_or_idle() and self.prev_command.is_off_or_idle():
                do_count = True

            if do_count:
                self.num_on_off += 1
                self.last_state_change_time = time
                _LOGGER.info(
                    f"Change load: {self.name} state increment num_on_off:{self.num_on_off} ({command.command})"
                )

    def _anchor_causality_guard_if_executed(self, is_command_set: bool | None, time: datetime) -> None:
        """QS-256 (D5): anchor the causality guard after a REAL execution.

        Shared by `launch_command` and `force_relaunch_command` — set ONLY
        when `execute_command` returned True (never on the probe-already-set
        branch, never in `_ack_command`).
        """
        if is_command_set is True:
            self.last_command_execution_time = time

    def is_command_suppressed_by_override(self, time: datetime, command: LoadCommand) -> bool:
        """Return True when an active user override must swallow this command.

        QS-256 (D1): command swallowing during an active override is BY
        DESIGN (don't fight the user) — but a suppressed command must be
        DROPPED at the `launch_command` drop point, never phantom-acked.
        Default: no override support, nothing suppressed. Subclasses that
        track `external_user_initiated_state` override this.
        """
        return False

    @property
    def effective_command(self) -> LoadCommand | None:
        """Return the best known command, including pending ones not yet acked."""
        return self.current_command or self.running_command or self._stacked_command

    def is_load_has_a_command_now_or_coming(self, time: datetime) -> bool:
        if self.qs_enable_device is False:
            return False

        if self.current_command is not None:
            return True
        if self.running_command is not None:
            return True
        if self._stacked_command is not None:
            return True
        return False

    @property
    def is_uncontrollable(self) -> bool:
        """Return True when a command is in flight AND lost-control history is unresolved.

        Internal state — nothing exposes it to Home Assistant. It decides
        supersede-vs-stack and gates the one-shot escalation.

        The in-flight conjunct is load-bearing: several paths empty the slot with no
        successor, and a clock-only property would stay True forever with no retry
        able to clear it.

        QS-307 (from #308): every path that empties the slot **with no successor**
        releases the clock, including the invalid-probe give-up, which used to keep it
        and hand it to an unrelated next command. A *supersede* is the deliberate
        exception: `abandon_running_command` preserves the clock and
        `launch_command` hands it to the successor on purpose, so the load keeps
        superseding instead of stacking and holds QS-304's saturated 300 s cadence.
        Do NOT "restore the invariant" by releasing it there.

        So `unresponsive_since` tracks the command in flight. "Are we still in an
        unresolved lost-control EPISODE" is a different question, answered by
        `_unresponsive_needs_ack` — do not conflate the two.
        """
        return self.unresponsive_since is not None and self.running_command is not None

    def command_relaunch_delay_s(self) -> float:
        """Return the linear backoff that saturates instead of running out.

        QS-304: 50, 100, 150, 200, 250, 300 s for relaunch counts 0..5, then
        300 s forever. The load stays commandable no matter how long the device
        refuses to converge.
        """
        return float(
            COMMAND_RELAUNCH_BASE_DELAY_S * min(self.running_command_num_relaunch + 1, NUM_MAX_COMMAND_RELAUNCH)
        )

    def _clear_unresponsive(self, reason: str, contact: ContactEvidence) -> None:
        """Release the per-command lost-control clock, logging the exit exactly once.

        The single writer, so the "one line out" guarantee cannot drift between
        its callers. Releasing matters more than the log line: the entry guard is
        `unresponsive_since is None`, so a load that loses control, recovers and
        loses it again must be able to shout twice.

        The message is reason-led rather than saying "regained control", because
        several callers are not recoveries at all — a user override drop means
        control was taken *away* from QS, and a `keep_commands=False` reset simply
        destroys the clock's subject.

        `contact` says what the release tells us about the device, and drives
        `_unresponsive_needs_ack` (the per-EPISODE latch). Three values, because two
        were not enough — a drop we chose ourselves is evidence of nothing either
        way, and forcing it into "recovered" re-opened the push storm the latch
        exists to damp:

        - `CONFIRMED` — the device answered. Ends the episode, and does so
          **unconditionally**: an ack is contact whether or not a clock was live.
        - `UNREACHABLE` — we could not reach it (the invalid-probe give-up).
          Latches the episode, but only if a clock was actually live: latching a
          release that released nothing swallowed the load's FIRST genuine push
          forever, since the give-up fires ~70 s in and the escalation threshold is
          ~1050 s away.
        - `UNKNOWN` — *we* abandoned the command (a deliberate drop, a
          `keep_commands=False` wipe). Says nothing about the device, so it leaves
          the latch exactly as it was.
        """
        # The supersede anchor is cleared unconditionally: it is meaningless
        # without a lost-control episode to throttle, and leaving it behind let a
        # brand-new command inherit a stale anchor and have its first legitimate
        # supersede throttled for up to `SUPERSEDE_MIN_INTERVAL_S`.
        self._last_supersede_time = None

        if contact is ContactEvidence.CONFIRMED:
            # Above the early return on purpose: the device answered, so the episode
            # is over regardless of whether there was a clock left to release.
            self._unresponsive_needs_ack = False

        if self.unresponsive_since is None:
            return

        self.unresponsive_since = None
        if contact is ContactEvidence.UNREACHABLE:
            # Below the early return on purpose (see the docstring): the latch may
            # only suppress a NEXT shout when there was a shout to follow.
            self._unresponsive_needs_ack = True
            _LOGGER.info("Lost-control clock released without contact for load %s: %s", self.name, reason)
        else:
            _LOGGER.info("Lost-control state cleared for load %s: %s", self.name, reason)

    def _acknowledge_lost_control(self, reason: str) -> None:
        """Treat explicit user remediation as acknowledging an open episode.

        QS-319: with the announce-latch in place, every `UNKNOWN` release leaves the
        latch set — including the paths the *user* drives. Without this, disabling a
        broken load, re-enabling it and letting it break again produced no alert at
        all until the next restart. A user who resets or re-enables a device has seen
        the problem and acted, which is precisely what "acknowledged" means.

        The early return is deliberate. Both call sites fire on every reset press and
        every enable/disable transition, the overwhelming majority with no episode
        open; logging unconditionally would claim an acknowledgement that never
        happened.
        """
        if not self._unresponsive_needs_ack:
            return

        self._unresponsive_needs_ack = False
        _LOGGER.info("Lost-control episode acknowledged for load %s: %s", self.name, reason)

    @property
    def has_unacknowledged_lost_control(self) -> bool:
        """Whether an announced lost-control episode is still unacknowledged.

        QS-319: the public, per-EPISODE read of `_unresponsive_needs_ack`, exposed to
        Home Assistant as the `qs_load_lost_control` PROBLEM binary sensor.
        Deliberately NOT `is_uncontrollable`, which is per-COMMAND and flickers False
        every time the slot empties.

        "Unacknowledged", not "unresolved": explicit user remediation acknowledges an
        episode (see `_acknowledge_lost_control`) while the device may well still be
        broken. The next ladder climb then re-announces and this returns True again.

        Readable on every `AbstractDevice`, including ones that will never expose it
        — a battery latches and logs but never pushes, and gets no entity. Do not add
        the sensor to non-loads on the strength of this property alone.
        """
        return self._unresponsive_needs_ack

    def abandon_running_command(self, reason: str) -> None:
        """Drop the stale in-flight command without faking an ack.

        QS-304: `current_command` means "last CONFIRMED command", so it is
        preserved — the device really is still in its old state. `num_on_off`,
        `unresponsive_since` and `_last_supersede_time` are preserved too, and
        so is `running_command_num_relaunch`: the ladder rung is what keeps the
        saturated 300 s cadence across supersession instead of restarting the
        backoff at 50 s and turning the retry into a service-call storm.
        `_stacked_command` is left alone — `launch_command` owns it.

        Preserving the rung is only sound when a successor is launched in the same
        call. Use `_drop_running_command` when the slot is left empty — it resets
        the rung and releases the clock as well.
        """
        _LOGGER.debug(
            "abandon_running_command: dropping %s for load %s, reason: %s", self.running_command, self.name, reason
        )
        self.running_command = None
        self.running_command_num_relaunch_after_invalid = 0
        self.running_command_first_launch = None
        self.running_command_last_launch = None

    async def _notify_unresponsive(self, _time: datetime, _command: LoadCommand) -> None:
        """Push a "QS lost control" notification — no-op for a non-load device.

        QS-304: the relaunch driver also runs against `QSBattery`, which is
        `AbstractDevice`-side and has no notification channel. `AbstractLoad`
        overrides this.
        """
        return

    def _log_stacked_command(self, command: LoadCommand, ctxt: str) -> None:
        """Log a command that could not be executed right away."""
        _LOGGER.info(
            "launch_command: stack command %s for this load %s, ctxt: %s running %s, stacked %s",
            command,
            self.name,
            ctxt,
            self.running_command,
            self._stacked_command,
        )

    def _log_absorbed_command(self, command: LoadCommand, ctxt: str) -> None:
        """Log a command that matched the one already in flight.

        This must not reuse the "stack command" message, which
        claimed the command had been stacked when it was in fact absorbed, and
        reported a misleading "stacked None".
        """
        _LOGGER.info(
            "launch_command: absorb command %s already in flight for this load %s, ctxt: %s, stacked %s",
            command,
            self.name,
            ctxt,
            self._stacked_command,
        )

    @staticmethod
    def _seconds_since(time: datetime, anchor: datetime | None) -> float | None:
        """Return seconds elapsed since `anchor`, or None when it cannot be trusted.

        `None` means "treat as fully elapsed". Two cases produce it: there is no
        anchor, or the anchor lies in the *future* because the clock stepped
        backwards — HA booting without an RTC and NTP later correcting, or a manual
        change.

        Both of this class's clock comparisons go through here on purpose. A
        negative delta is trivially below any threshold, so an unguarded
        comparison freezes whatever it gates for the whole duration of the jump:
        the supersede throttle would stack every command, and the relaunch ladder
        would stop advancing so the rung could never reach the escalation
        threshold — a silently deadlocked command slot with no ERROR and no
        PROBLEM sensor. Sharing one primitive is what stops the two sites drifting
        apart again.
        """
        if anchor is None:
            return None

        elapsed = (time - anchor).total_seconds()
        if elapsed < 0:
            return None

        return elapsed

    def _is_supersede_throttled(self, time: datetime) -> bool:
        """Return True while the supersede window is still closed."""
        elapsed = self._seconds_since(time, self._last_supersede_time)
        return elapsed is not None and elapsed < SUPERSEDE_MIN_INTERVAL_S

    def _drop_running_command(self, reason: str) -> None:
        """Abandon the in-flight command when nothing will succeed it.

        `abandon_running_command` preserves `running_command_num_relaunch` so a
        *supersede* keeps the saturated 300 s cadence — but that is only sound when
        a successor is launched in the same call. When the slot is left empty, the
        rung describes a command that no longer exists and the lost-control clock
        has lost its subject, so both go with it.

        The clock release is deliberately **unconditional**, not gated on
        `current_command is not None`: an emptied slot with no successor has no
        owner for the clock whether or not a confirmed command ever existed. A load
        that has never been acked — a fresh config-entry reload, or a bistate switch
        the user then flips by hand — could otherwise cross the threshold with
        `current_command` still `None`, keep the clock through the drop, and light
        PROBLEM on the very next command with zero relaunches *and* a 50 s rung,
        breaking the one-service-call-per-300 s invariant.

        Doing this here rather than leaving it to `_escalate_or_recover` closes an
        intra-cycle window: `update_loads` calls `launch_command` twice per cycle
        for the same load, and the buttons call it outside the load-management lock.
        """
        self.abandon_running_command(reason=reason)
        self.running_command_num_relaunch = 0
        # `UNKNOWN`: a drop is OUR decision, so it says nothing about the device and
        # must not end a lost-control episode (QS-307).
        self._clear_unresponsive(reason, contact=ContactEvidence.UNKNOWN)

    async def launch_command(self, time: datetime, command: LoadCommand, ctxt="NO CTXT"):
        if self.qs_enable_device is False:
            return

        command = copy_command(command)

        supersede_stale_command = False

        if self.running_command is not None:
            if self.running_command == command:
                # the very same command is already in flight: absorb this one
                self.running_command = command
                # The stack MUST be cleared here: nothing can stack after
                # `launch_command` is entered, so the incoming command is the solver's
                # newest word and anything stacked is obsolete. Keeping it resurrects
                # a command the solver has moved on from (AC13).
                self._stacked_command = None
                self._log_absorbed_command(command, ctxt)
                return

            if not self.is_uncontrollable:
                # another command has been launched, stack this one (we replace the previous stacked one)
                self._stacked_command = command
                self._log_stacked_command(command, ctxt)
                return

            if self._is_supersede_throttled(time):
                # QS-304: we have lost control, but we already superseded
                # recently — stack instead (last one wins) so a jittering
                # power consign cannot turn into a per-cycle service call storm
                self._stacked_command = command
                self._log_stacked_command(command, ctxt)
                return

            # QS-304: QS has lost control of this load, so the newer desired
            # command must SUPERSEDE the stale one instead of starving behind it
            # forever.
            #
            # Only an INTENT here: the abandon and the throttle stamp commit further
            # down, once a successor is really being launched. Committing early left
            # the slot empty with a spent rung whenever a gate below returned, and
            # burnt the window on a supersede that made no service call.
            supersede_stale_command = True

        # there is no running : whatever we will not execute the stacked one but only the last one
        self._stacked_command = None

        # QS-256 (D1): drop point — a command suppressed by an active user
        # override is DROPPED before `running_command` is set: no ack, no
        # counters mutation, nothing for check_commands/force_relaunch_command
        # to resurrect
        if self.is_command_suppressed_by_override(time, command):
            _LOGGER.info(
                "launch_command: command %s suppressed by user override for load %s, ctxt: %s",
                command,
                self.name,
                ctxt,
            )
            if supersede_stale_command:
                # the user has taken control, so the stale command is not merely
                # superseded, it is unwanted: drop it outright
                self._drop_running_command("suppressed by user override")
            return

        if self.current_command is not None and self.current_command == command:
            # We kill the stacked one and keep the current one like the choice above
            self.current_command = command  # needed as command == may have been overcharged to not test everything
            if supersede_stale_command:
                # the device already matches what is now wanted, so the stale
                # in-flight command is simply no longer desired
                self._drop_running_command("the newly requested command was already confirmed")
            return

        if supersede_stale_command:
            # No `await` between here and the launch below, so no other task can
            # observe the momentarily-empty slot.
            self._last_supersede_time = time
            self.abandon_running_command(reason=f"superseded by {command.command}")

        self.running_command = command
        self.running_command_first_launch = time
        self.running_command_last_launch = time

        _LOGGER.info("launch_command: %s for this load %s), ctxt: %s", command, self.name, ctxt)

        try:
            is_command_set = await self.probe_if_command_set(time, self.running_command)
        except Exception as err:
            # A probe that raises cannot tell us anything, so treat it exactly like
            # one returning None and go on to execute. Unguarded, it re-created this
            # story's deadlock on the stack-promotion path — see load-base.md.
            _LOGGER.error(
                "Error while probing command %s for load %s : %s, ctxt: %s",
                command.command,
                self.name,
                err,
                ctxt,
                exc_info=True,
                stack_info=True,
            )
            is_command_set = None

        if is_command_set is True:
            _LOGGER.info("launch_command: Command already set %s for this load %s, ctxt: %s", command, self.name, ctxt)
        else:
            try:
                is_command_set = await self.execute_command(time, command)
            except Exception as err:
                _LOGGER.error(
                    f"Error while executing command {command.command} for load {self.name} : {err}, ctxt: {ctxt}",
                    exc_info=True,
                    stack_info=True,
                )
                is_command_set = None
            self._anchor_causality_guard_if_executed(is_command_set, time)

        if self.running_command is None:
            # QS-307: same guard, same reason as in `force_relaunch_command`. Placed
            # here, at 8 spaces, so it covers BOTH awaits above — the probe and the
            # conditional execute. Do not move it inside the `else:`.
            _LOGGER.info(
                "launch_command: command %s for load %s was dropped while a call to the device was in flight, ctxt: %s",
                command.command,
                self.name,
                ctxt,
            )
            return

        if is_command_set is None:
            # hum we may have an impossibility to launch this command
            _LOGGER.info(
                f"launch_command: Impossible to launch this command {command.command} on this load {self.name}, ctxt: {ctxt}"
            )
        elif is_command_set is True:
            _LOGGER.info("launch_command: ack command %s for this load %s), ctxt: %s", command, self.name, ctxt)
            self._ack_command(time, self.running_command)

        return

    def is_load_command_set(self, time: datetime):
        if self.qs_enable_device is False:
            return False

        return self.running_command is None and self.current_command is not None

    async def check_commands(self, time: datetime) -> tuple[timedelta, bool]:

        res = timedelta(seconds=0)

        if self.qs_enable_device is False:
            return res, True

        command_acked_or_good = True

        if self.running_command is not None:
            _LOGGER.info(
                f"check command {self.running_command.command} for this load {self.name}) (#{self.running_command_num_relaunch_after_invalid})"
            )

            is_command_set = await self.probe_if_command_set(time, self.running_command)
            if is_command_set is None:
                command_acked_or_good = False
                # impossible to run this command for this load ...
                self.running_command_num_relaunch_after_invalid += 1
                _LOGGER.info(
                    f"impossible to check command {self.running_command.command} for this load {self.name}) (#{self.running_command_num_relaunch_after_invalid})"
                )
                if self.running_command_num_relaunch_after_invalid >= NUM_MAX_INVALID_PROBES_COMMANDS:
                    # will kill completely the command ....
                    self._ack_command(time, None)
                    # QS-307 (from #308): this destroys the clock's subject, so the
                    # clock goes with it or the next command inherits it. `UNREACHABLE`
                    # because the device cannot be reached — see `_clear_unresponsive`.
                    self._clear_unresponsive("the probe went unavailable", contact=ContactEvidence.UNREACHABLE)

            if is_command_set is True:
                self._ack_command(time, self.running_command)
                command_acked_or_good = True
            elif self.running_command_last_launch is not None:
                res = time - self.running_command_last_launch
                command_acked_or_good = False

        if self.running_command is None and self._stacked_command is not None:
            await self.launch_command(time, self._stacked_command, ctxt="check_commands, launch stacked command")
            res = timedelta(seconds=0)
            command_acked_or_good = False

        return res, command_acked_or_good

    async def check_and_relaunch_command(self, time: datetime) -> bool:
        """Probe the in-flight command, relaunch it when stale, and escalate.

        The whole retry lifecycle, in the pure layer, so relaunch timing depends only
        on `command_relaunch_delay_s()` and `is_uncontrollable`. Returns
        `command_acked_or_good` for `QSHome.check_loads_commands`' `all_ok`.

        The `finally` is what makes a device whose *probe* raises still climb the
        ladder. The device exception is deliberately not swallowed —
        `QSHome.check_loads_commands` owns the per-load `try/except` — and
        `_finish_command_cycle` never raises, so a housekeeping failure cannot
        replace it.

        NOT protected by a lock: `_update_loads_lock` guards only
        `async_update_loads`, while `button.py` calls into this chain unlocked.

        QS-307 corrected the invariant this used to claim. "Each command-slot
        mutation happens between `await`s" is **no longer true**: the override-expiry
        branch of `QSBiStateDuration.check_load_activity_and_constraints` drops an
        override-aligned command, and three of that method's callers (the bistate
        mode select, the on-duration number, the reset-override button) run outside
        the lock. So the slot CAN be emptied across an await. `launch_command` and
        `force_relaunch_command` both hold the command they launched in a local and
        bail out if the slot is EMPTY, rather than writing counters for, acking, or
        dereferencing a command that is gone.

        Scope of those guards, precisely: they cover the slot being **emptied**, not
        **replaced**. A replaced slot passes an `is None` test, so a button press that
        supersedes the in-flight command mid-await can still stamp its launch time or
        ack it off the previous command's result. That is pre-existing — it behaves
        identically on `main` — and is tracked in #320, deliberately not fixed here.

        The clock is still only cleared alongside the command state it describes.
        """
        try:
            _wait_time, command_acked_or_good = await self.check_commands(time)
        finally:
            await self._finish_command_cycle(time)

        return command_acked_or_good

    async def _finish_command_cycle(self, time: datetime) -> None:
        """Run the relaunch ladder and the escalation housekeeping. Never raises.

        Runs from `check_and_relaunch_command`'s `finally`, so anything escaping
        would replace the device exception being propagated. The broad `except` is
        scoped to one call each: this is the background-cycle boundary, mirroring
        `QSHome.check_loads_commands`' own per-load handler.
        """
        try:
            await self._relaunch_stale_command(time)
        except Exception as err:  # noqa: BLE001 - background cycle boundary, see docstring
            _LOGGER.error(
                "Error relaunching the stale command for load %s: %s", self.name, err, exc_info=True, stack_info=True
            )

        try:
            await self._escalate_or_recover(time)
        except Exception as err:  # noqa: BLE001 - background cycle boundary, see docstring
            _LOGGER.error(
                "Error escalating the command state for load %s: %s", self.name, err, exc_info=True, stack_info=True
            )

    async def _relaunch_stale_command(self, time: datetime) -> None:
        """Relaunch the in-flight command once its backoff delay has elapsed."""
        if self.running_command is None:
            return

        # Checked BEFORE the backoff gate: a load the user disabled should have its
        # stale slot cleaned up immediately, not only once a possibly-300 s rung has
        # elapsed. `force_relaunch_command` owns that cleanup — it clears the slot
        # and never executes anything for a disabled load. Note the branch is
        # defensive either way: the `qs_enable_device` setter calls `reset()`, which
        # empties the slot *before* `_enabled` flips, so it is reachable only when
        # `_enabled` is mutated without the property setter.
        if self.qs_enable_device is False:
            await self.force_relaunch_command(time)
            return

        elapsed = self._seconds_since(time, self.running_command_last_launch)
        if elapsed is not None and elapsed <= self.command_relaunch_delay_s():
            return

        if (
            self._stacked_command is not None
            and self.is_uncontrollable
            and self._stacked_command != self.running_command
            and not self._is_supersede_throttled(time)
        ):
            # Once the supersede window has opened, retry the
            # NEWEST intent rather than the command we already know the device is
            # ignoring. Without this the stacked command starves forever:
            # `check_commands` only promotes the stack when the slot empties, and
            # for an uncontrollable load it never does — so a quiet solver left
            # the stale command being retried every 300 s in perpetuity.
            await self.launch_command(time, self._stacked_command, ctxt="relaunch newest stacked intent")
            return

        await self.force_relaunch_command(time)

    async def _escalate_or_recover(self, time: datetime) -> None:
        """Cross the lost-control threshold once, or re-arm on an emptied slot."""
        if (
            # A load the user disabled must never shout — QS was
            # explicitly told to leave it alone. The guard sits on the escalation
            # branch only, so the housekeeping below still runs for a disabled load.
            self.qs_enable_device is not False
            and self.running_command is not None
            and self.running_command_num_relaunch >= NUM_MAX_COMMAND_RELAUNCH
            and self.unresponsive_since is None
        ):
            # `>=` and not `==`: the counter is resettable, so the threshold has
            # to be a floor. `unresponsive_since is None` is the per-COMMAND guard.
            self.unresponsive_since = time
            if self._unresponsive_needs_ack:
                # QS-307: same episode, no contact since. The clock still has to be
                # re-armed (it drives supersede-vs-stack) but there is nothing new to
                # tell anyone. Logged so the ladder wall stays visible in user logs —
                # and QS-319 makes this the DOMINANT path for a device that stays
                # reachable and simply never obeys, so it is also that device's
                # per-climb diagnostic (a battery, which never pushes, keeps only
                # this).
                _LOGGER.info(
                    "Lost-control clock re-armed for load %s: command %s still unconfirmed after %s relaunches, "
                    "incident already announced, not re-notifying",
                    self.name,
                    self.running_command,
                    self.running_command_num_relaunch,
                )
                unresponsive_command = None
            else:
                _LOGGER.error(
                    # "in this episode": `abandon_running_command` preserves the rung
                    # across a supersede on purpose, so the count can include
                    # relaunches of this command's predecessors.
                    "Lost control of load %s: command %s not confirmed after %s relaunches in this episode",
                    self.name,
                    self.running_command,
                    self.running_command_num_relaunch,
                )
                # QS-319: announcing an episode latches it, so the ladder can climb
                # again — as it does about every 18.5 min for a reachable device that
                # never obeys — without shouting twice. Written BEFORE the await for
                # the same reason `unresponsive_since` is: a notify service that
                # raises must not resurrect the storm. The episode ends on exactly
                # three things — a real ack, explicit user remediation
                # (`_acknowledge_lost_control`), or a process restart / config-entry
                # reload, where nothing restores the latch.
                self._unresponsive_needs_ack = True
                unresponsive_command = self.running_command
        else:
            unresponsive_command = None

        if self.running_command is None:
            # QS-304 invariant: an empty command slot implies rung 0. The rung
            # counts relaunches OF THE IN-FLIGHT COMMAND, so with nothing in
            # flight it describes a command that no longer exists. Leaving it
            # high would make the NEXT command cross the threshold on its very
            # first cycle, with zero relaunches and a false "after N
            # relaunches" message. `abandon_running_command` preserves the rung
            # on purpose — but that is only sound when a successor is launched
            # in the same `launch_command` call, with no cycle in between; the
            # equal-command early-return and the override-suppression drop
            # leave no successor at all.
            self.running_command_num_relaunch = 0
            # `_last_supersede_time` (this line) is cleared OUTSIDE the gate below,
            # and kept that way: paths reach here with `current_command` already
            # nulled, and a stale anchor would throttle the next command's first
            # legitimate supersede.
            self._last_supersede_time = None

            # QS-319: `UNKNOWN`, not `CONFIRMED`. Nobody answered here — *we* emptied
            # the slot, and the previous `CONFIRMED` was a fake ack. It was harmless
            # only while the give-up was the sole latch-setter (it nulls
            # `current_command`, so it skipped the gate below). Now that announcing
            # also latches, claiming contact here would clear the episode on every
            # cycle that runs with an empty slot — which is the PRODUCTION ordering,
            # since `check_loads_commands` runs before the solver's `launch_command`
            # and a supersede-drop therefore leaves the slot empty across the cycle
            # boundary. The latch would never survive, and the once-per-episode rule
            # would do nothing in the field.
            #
            # The gate stays: with `UNKNOWN` it is arguably unnecessary, but removing
            # it would release clocks that are not released today — a behavior change
            # with no evidence behind it.
            if self.current_command is not None:
                self._clear_unresponsive("the command slot emptied with no successor", contact=ContactEvidence.UNKNOWN)

        if unresponsive_command is not None:
            # The push goes LAST: it is the only part that can realistically raise
            # (a subclass may dereference optional state), so nothing that matters is
            # sequenced behind it. `unresponsive_since` is already written, so the
            # once-only guard holds even if it fails.
            await self._notify_unresponsive(time, unresponsive_command)

    async def force_relaunch_command(self, time: datetime):
        if self.qs_enable_device is False:
            # QS-304: route the disabled-device cleanup through the shared drop so
            # the rung and the lost-control clock go with the slot. Leaving the
            # clock behind stranded it with no owner, and the next command was then
            # flagged uncontrollable on its first cycle.
            self._drop_running_command("the load was disabled")

        if self.running_command is not None:
            # review fix QS-256#02: a stale running command suppressed by an
            # active user override must be DROPPED, not retried — retrying
            # could phantom-ack it through the execute_command interception
            # (entity already in the override state) or fight the user
            if self.is_command_suppressed_by_override(time, self.running_command):
                _LOGGER.info(
                    "force_relaunch_command: command %s suppressed by user override for load %s, dropped",
                    self.running_command,
                    self.name,
                )
                # QS-304: this IS an abandon with no successor — drop the stale
                # in-flight command without faking an ack, and let the rung and
                # the lost-control clock go with it (the USER took control, so
                # there is no retry cadence left to protect). `current_command` is
                # intentionally PRESERVED: it is the last acked command of record,
                # and a later non-suppressed launch (e.g. after the override ends)
                # must still be able to compare against it.
                self._drop_running_command("suppressed by user override")
                return

            # Held locally: the slot can empty across the await (see the re-check).
            launched_command = self.running_command
            _LOGGER.info(
                "force launch command %s for this load %s (#%s)",
                launched_command.command,
                self.name,
                self.running_command_num_relaunch,
            )
            self.running_command_num_relaunch += 1
            try:
                is_command_set = await self.execute_command(time, launched_command)
            except Exception as err:
                _LOGGER.error(
                    "Error while executing command %s for load %s : %s",
                    launched_command.command,
                    self.name,
                    err,
                    exc_info=True,
                    stack_info=True,
                )
                is_command_set = None

            # Before the re-check: a service call really landed, so the causality
            # guard must know even if the slot is now empty.
            self._anchor_causality_guard_if_executed(is_command_set, time)

            if self.running_command is None:
                # QS-307: the slot emptied across the await — a user action can reach
                # the override expiry outside `_update_loads_lock`. Everything below
                # describes the command that is now gone, so stop.
                _LOGGER.info(
                    "force_relaunch_command: command %s for load %s was dropped while its service call was in flight",
                    launched_command.command,
                    self.name,
                )
                return

            self.running_command_last_launch = time
            if is_command_set is None:
                _LOGGER.info(
                    "impossible to force command %s for this load %s)", self.running_command.command, self.name
                )
            elif is_command_set is True:
                self._ack_command(time, self.running_command)
            else:
                await self.check_commands(time)

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:
        _LOGGER.info("Executing command unimplemented %s", command)
        return False

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        return True


class PilotedDevice(AbstractDevice):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_demanding_clients: list[int] | None = None
        self.clients: list[AbstractDevice] = []

    @property
    def is_piloted_device_activated(self) -> bool:
        for client in self.clients:
            if client.qs_enable_device:
                if client.current_command is not None and not client.current_command.is_off_or_idle():
                    return True
        return False

    def is_device_light_on(self) -> bool:
        return self.is_piloted_device_activated

    def prepare_slots_for_piloted_device_budget(self, num_slots: int):
        self.num_demanding_clients = [0] * num_slots
        _LOGGER.debug("prepare_slots_for_piloted_device_budget for a piloted device: %s", self.name)

    def possible_delta_power_for_slot(self, slot_idx: int | None, add: bool = True) -> float:
        if self.num_demanding_clients is None or len(self.num_demanding_clients) == 0:
            return 0

        if len(self.clients) == 0:
            return 0

        if add:
            if slot_idx is None or self.num_demanding_clients[slot_idx] == 0:
                # first add!
                return self.power_use
            else:
                return 0

        else:
            if slot_idx is None or self.num_demanding_clients[slot_idx] == 1:
                # last remove!
                return self.power_use

            return 0

    def update_num_demanding_clients_for_slot(self, slot_idx: int, add: bool) -> int | float:
        if self.num_demanding_clients is None or len(self.num_demanding_clients) == 0:
            return 0

        if len(self.clients) == 0:
            return 0

        power_delta = self.possible_delta_power_for_slot(slot_idx, add)

        if add:
            if self.num_demanding_clients[slot_idx] >= len(self.clients):
                _LOGGER.warning(
                    f"update_num_demanding_clients_for_slot for a piloted device: {self.name} too many clients on: {len(self.clients)}"
                )
                self.num_demanding_clients[slot_idx] = len(self.clients) - 1

            self.num_demanding_clients[slot_idx] += 1
        else:
            self.num_demanding_clients[slot_idx] -= 1

            if self.num_demanding_clients[slot_idx] < 0:
                _LOGGER.warning(
                    f"update_num_demanding_clients_for_slot for a piloted device: {self.name} negative num demanding clients fix it"
                )
                self.num_demanding_clients[slot_idx] = 0

        return power_delta


class AbstractLoad(AbstractDevice):
    def __init__(self, **kwargs):
        self.switch_entity = kwargs.pop(CONF_SWITCH, None)
        self.load_is_auto_to_be_boosted = kwargs.pop(CONF_LOAD_IS_BOOST_ONLY, False)
        self.external_user_initiated_state: str | None = None
        self.external_user_initiated_state_time: datetime | None = None
        self.asked_for_reset_user_initiated_state_time: datetime | None = None
        self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done: datetime | None = None

        super().__init__(**kwargs)

        self._last_completed_constraint: LoadConstraint | None = None

        self.current_constraint_current_value: float | None = None
        self.current_constraint_current_energy: float | None = None
        self.current_constraint_current_percent_completion: float | None = None
        self.next_or_current_constraint_start_time: datetime | None = None
        self.next_or_current_constraint_end_time: datetime | None = None

        self.externally_initialized_constraints = False

        self.qs_best_effort_green_only = False

        self._last_hash_state = None

        self.is_load_time_sensitive = False

    def constraint_reset_and_reset_commands_if_needed(self, keep_commands=True):
        super().constraint_reset_and_reset_commands_if_needed(keep_commands=keep_commands)

        self.current_constraint_current_value = None
        self.current_constraint_current_energy = None
        self.current_constraint_current_percent_completion = None
        self.next_or_current_constraint_start_time = None
        self.next_or_current_constraint_end_time = None
        self._last_completed_constraint = None

    def update_to_be_saved_extra_device_info(self, data_to_update: dict):
        super().update_to_be_saved_extra_device_info(data_to_update)
        data_to_update[STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE] = self.external_user_initiated_state
        data_to_update[STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME] = None
        if self.external_user_initiated_state_time is not None:
            data_to_update[STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME] = (
                f"{self.external_user_initiated_state_time}"
            )

        data_to_update[STORAGE_KEY_ASKED_FOR_RESET_TIME] = None
        if self.asked_for_reset_user_initiated_state_time is not None:
            data_to_update[STORAGE_KEY_ASKED_FOR_RESET_TIME] = f"{self.asked_for_reset_user_initiated_state_time}"

        data_to_update[STORAGE_KEY_ASKED_FOR_RESET_FIRST_CMD_RESET_DONE] = None
        if self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is not None:
            data_to_update[STORAGE_KEY_ASKED_FOR_RESET_FIRST_CMD_RESET_DONE] = (
                f"{self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done}"
            )

    @staticmethod
    def _restored_utc_datetime(value: str | None) -> datetime | None:
        """Parse a stored isoformat timestamp, coercing tz-naive values to UTC.

        Review fix QS-256#02: legacy or hand-edited `.storage` entries can
        hold tz-naive strings; downstream datetime arithmetic against
        tz-aware "now" values would raise TypeError.
        """
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=pytz.UTC)
        return parsed

    def use_saved_extra_device_info(self, stored_load_info: dict):
        super().use_saved_extra_device_info(stored_load_info)
        self.external_user_initiated_state = stored_load_info.get(STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE, None)

        self.external_user_initiated_state_time = self._restored_utc_datetime(
            stored_load_info.get(STORAGE_KEY_EXTERNAL_USER_INITIATED_STATE_TIME, None)
        )

        self.asked_for_reset_user_initiated_state_time = self._restored_utc_datetime(
            stored_load_info.get(STORAGE_KEY_ASKED_FOR_RESET_TIME, None)
        )

        self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = self._restored_utc_datetime(
            stored_load_info.get(STORAGE_KEY_ASKED_FOR_RESET_FIRST_CMD_RESET_DONE, None)
        )

    def get_override_state(self):

        overridden_state = self.external_user_initiated_state
        if overridden_state is None:
            ct = self.get_current_active_constraint()
            if ct is not None:
                if (
                    ct.load_info is not None
                    and ct.load_info.get(CONSTRAINT_ORIGINATOR_KEY, None) == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
                ):
                    overridden_state = ct.load_param

        if self.asked_for_reset_user_initiated_state_time is not None:
            return OVERRIDE_STATE_ASKED_FOR_RESET
        if overridden_state is None:
            return OVERRIDE_STATE_NO_OVERRIDE
        return f"{OVERRIDE_STATE_PREFIX}{overridden_state}"

    def is_user_overridden(self) -> bool | None:
        """Return whether load is currently user-overridden.

        Returns True if all controlled activity is user-overridden,
        False if no override is active. Individual loads never return None.
        """
        state = self.get_override_state()
        if state == OVERRIDE_STATE_NO_OVERRIDE:
            return False
        return True

    def is_time_sensitive(self):

        if self.is_best_effort_only_load():
            return False

        return self.is_load_time_sensitive

    def is_best_effort_only_load(self):
        return self.load_is_auto_to_be_boosted or self.qs_best_effort_green_only

    def get_for_solver_constraints(self, start_time: datetime, end_time: datetime) -> list[Any]:
        if self.qs_enable_device is False:
            self._constraints = []

        res = []

        for c in self._constraints:
            if c.is_constraint_active_for_time_period(start_time, end_time):
                res.append(c)

        return res

    def get_normalized_score(self, ct: LoadConstraint, time: datetime, score_span: int = 0) -> float:
        return 0.0

    def get_min_max_power(self) -> (float, float):
        if self.power_use is None:
            return 0.0, 0.0
        return self.power_use, self.power_use

    def support_green_only_switch(self) -> bool:
        return False

    def support_user_override(self) -> bool:
        return False

    # def push_unique_and_current_end_of_constraint_from_agenda(self, time: datetime, new_ct: LoadConstraint):
    #
    #     new_end_time = new_ct.end_of_constraint
    #
    #     if new_end_time == DATETIME_MAX_UTC or new_end_time == DATETIME_MIN_UTC:
    #         return False
    #
    #     if self._last_pushed_end_constraint_from_agenda is None:
    #         self._last_pushed_end_constraint_from_agenda = new_ct
    #     else:
    #         # if the agenda has changed ... we should remove an existing uneeded constraint
    #         if self._last_pushed_end_constraint_from_agenda.end_of_constraint != new_end_time:
    #             for i, ct in enumerate(self._constraints):
    #                 if type(ct) == type(new_ct) \
    #                         and ct.type == new_ct.type \
    #                         and ct.end_of_constraint == self._last_pushed_end_constraint_from_agenda.end_of_constraint:
    #                     self._constraints[i] = None
    #                     break
    #
    #             self._constraints = [c for c in self._constraints if c is not None]
    #
    #     res = self.push_live_constraint(time, new_ct)
    #     self._last_pushed_end_constraint_from_agenda = new_ct
    #
    #     return res

    def push_agenda_constraints(
        self, time: datetime, new_constraints: list[LoadConstraint | None]
    ) -> tuple[bool, list[LoadConstraint]]:
        """Push agenda constraints. Returns (changed, constraints_to_ack)."""
        for new_ct in new_constraints:
            new_ct.add_or_update_load_info(CONSTRAINT_ORIGINATOR_KEY, CONSTRAINT_ORIGINATOR_AGENDA)

        one_c_removed = False
        for i, ct in enumerate(self._constraints):
            if (
                ct
                and ct.load_info is not None
                and ct.load_info.get(CONSTRAINT_ORIGINATOR_KEY, None) == CONSTRAINT_ORIGINATOR_AGENDA
            ):
                # find if we have a agenda one that is matching, if no : we kill it
                found = False
                for new_ct in new_constraints:
                    if ct.eq_no_current(new_ct):
                        found = True
                        break
                if not found:
                    # a not found calendar one : kill it calendar may have changed
                    self._constraints[i] = None
                    one_c_removed = True

        for i, new_ct in enumerate(new_constraints):
            found = False
            for ct in self._constraints:
                if ct is None:
                    continue
                if ct.eq_no_current(new_ct):
                    # it is already in the constraints don't add it back
                    found = True
                    ct.carry_info_from_other_constraint(new_ct)
                    break
            if found:
                new_constraints[i] = None  # mark as found

        new_constraints = [c for c in new_constraints if c is not None]

        res = False
        to_ack: list[LoadConstraint] = []

        if one_c_removed:
            self._constraints = [c for c in self._constraints if c is not None]
            self.set_live_constraints(time, self._constraints)
            res = True

        for new_ct in new_constraints:
            pushed, needs_ack = self.push_live_constraint(time, new_ct)
            if needs_ack:
                to_ack.append(new_ct)
            res = pushed or res

        return res, to_ack

    def get_power_from_switch_state(self, state: str | None) -> float | None:
        if state is None:
            return None
        if state == "on":
            return self.power_use
        else:
            return 0.0

    async def do_run_check_load_activity_and_constraints(self, time: datetime) -> bool:
        if self.qs_enable_device is False:
            return False
        if self.externally_initialized_constraints is False:
            return False
        return await self.check_load_activity_and_constraints(time)

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        return False

    async def async_load_constraints_from_storage(
        self, time: datetime, constraints_dicts: list[dict], stored_executed: dict | None
    ):

        self.constraint_reset_and_reset_commands_if_needed(keep_commands=False)

        for c_dict in constraints_dicts:
            cs_load = LoadConstraint.new_from_saved_dict(time, self, c_dict)
            if cs_load is not None:
                # only restore constraints that can still be active
                if cs_load.is_constraint_active_for_time_period(time):
                    self.push_live_constraint(time, cs_load)

        if stored_executed is not None:
            self._last_completed_constraint = LoadConstraint.new_from_saved_dict(time, self, stored_executed)
        else:
            self._last_completed_constraint = None

        self.externally_initialized_constraints = True

    async def do_probe_state_change(self, time: datetime):

        if self.qs_enable_device is False:
            return

        new_hash = self.get_active_state_hash(time)

        if new_hash is not None:
            # do not notify just after a reset (self._last_hash_state None)
            if self._last_hash_state is not None and self._last_hash_state != new_hash:
                _LOGGER.info("Hash state change for load %s from %s to %s", self.name, self._last_hash_state, new_hash)
                await self.on_device_state_change(time, DEVICE_STATUS_CHANGE_CONSTRAINT)

            self._last_hash_state = new_hash

    async def on_device_state_change(
        self,
        time: datetime,
        device_change_type: str,
        title: str | None = None,
        message: str | None = None,
        *,
        notification_tag: str | None = None,
    ):
        """Announce a device state change — a no-op at the domain layer.

        QS-319: `notification_tag` is keyword-only so no positional caller can
        misbind it onto `title` or `message`. It is a mobile-app payload key,
        meaningful only to `ha_model`, which overrides this.
        """

    async def _notify_unresponsive(self, time: datetime, command: LoadCommand) -> None:
        """Push one notification when QS loses control of this load.

        QS-304: entry only. There is no recovery push — sending an ERROR-status push
        to say things are fine would be wrong, and recovery is exposed as state
        instead (see `has_unacknowledged_lost_control`).

        QS-319 (delivery): the push carries a stable per-load
        `data.tag`, which the mobile-app notify platform treats as a replace-key on
        Android and iOS. The series therefore collapses into ONE notification on the
        phone instead of accumulating. The tag keys on `device_id`, which is
        config-derived and so survives a restart.

        QS-319 (frequency): `_unresponsive_needs_ack` is now set by the announce
        branch itself, so an announced episode is alerted exactly once — including
        for a device that stays *reachable* and simply never obeys, which used to
        re-cross the ladder wall and push about every 18.5 minutes forever. The
        episode ends on a real ack, on explicit user remediation, or on a process
        restart / config-entry reload (where nothing restores the latch).
        """
        await self.on_device_state_change(
            time,
            DEVICE_STATUS_CHANGE_ERROR,
            message=(
                f"Quiet Solar lost control of `{self.name}`: the command "
                f"`{command.command}` was sent repeatedly but the device never confirmed it"
            ),
            notification_tag=f"{NOTIFICATION_TAG_LOST_CONTROL_PREFIX}{self.device_id}",
        )

    def get_update_value_callback_for_constraint_class(
        self, constraint: LoadConstraint
    ) -> Callable[[LoadConstraint, datetime], Awaitable[tuple[float | None, bool]]] | None:
        return None

    def is_load_active(self, time: datetime):
        if self.qs_enable_device is False:
            return False
        if not self._constraints:
            return False
        return True

    def _match_ct(self, ct: LoadConstraint, load_param: str | None, load_info: dict | None = None) -> bool:
        if ct.load_param != load_param:
            return False
        if not load_info:
            return True
        if not ct.load_info:
            return True
        for k, v in load_info.items():
            if k in ct.load_info and ct.load_info[k] != v:
                return False
        return True

    def clean_constraints_for_load_param_and_if_same_key_same_value_info(
        self, time: datetime, load_param: str | None, load_info: dict | None = None, for_full_reset=True
    ) -> bool:
        """Clean constraints that do not match the load_param and load_info the load info matching is loose : ie : it is only NOT matching if their is a common key with a different value"""

        existing_constraints = []
        last_completed_constraint = None
        last_pushed_end_constraint_from_agenda = None

        found_one_bad = False

        if self._last_completed_constraint is not None:
            if self._match_ct(self._last_completed_constraint, load_param, load_info):
                # we have a last completed constraint that is still valid
                if for_full_reset:
                    _LOGGER.info(
                        f"clean_constraints_for_load_param: Found a stored last completed constraint to be kept with {self._last_completed_constraint.load_param}  {self._last_completed_constraint.name}"
                    )
                last_completed_constraint = self._last_completed_constraint
            else:
                found_one_bad = True

        for ct in self._constraints:
            if self._match_ct(ct, load_param, load_info) is False:
                # this constraint is not compatible with the load_param we are looking for
                found_one_bad = True
                continue
            if for_full_reset:
                _LOGGER.info(
                    f"clean_constraints_for_load_param: Found a stored car constraint to be kept with {ct.load_param}  {ct.name}"
                )

            existing_constraints.append(ct)

        if found_one_bad is False and for_full_reset is False:
            # no need to reset, we have all the constraints we need
            _LOGGER.debug(
                "clean_constraints_for_load_param: No bad constraint found for %s, no reset needed", load_param
            )
            return False

        if for_full_reset:
            self.reset(keep_commands=True)
        else:
            self.constraint_reset_and_reset_commands_if_needed(keep_commands=True)

        # if not full reset : do not remember the last completed constraint .... (ex: a plugged car, when plugging in, forget any previously stored constraint)
        if for_full_reset is False:
            self._last_completed_constraint = last_completed_constraint

        for ct in existing_constraints:
            if ct is not None:
                self.push_live_constraint(time, ct)  # ack ignored: restoring existing constraints

        return True

    def reset(self, keep_commands=False):
        _LOGGER.info("Reset load %s", self.name)
        super().reset(keep_commands=keep_commands)

    async def ack_completed_constraint(self, time: datetime, constraint: LoadConstraint | None):
        if self.qs_enable_device is False:
            return

        if (
            constraint is not None
            and constraint.load_info is not None
            and constraint.load_info.get(CONSTRAINT_ORIGINATOR_KEY, None) == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
        ):
            # it is a user override based constraint ... we should reset the override state.
            _LOGGER.info(
                f"Ack completed constraint {constraint.name} for load {self.name} with user override origin, reset override state and set reset ask time"
            )
            self.reset_override_state_and_set_reset_ask_time(time=time)

        self._last_completed_constraint = constraint
        await self.on_device_state_change(time, DEVICE_STATUS_CHANGE_CONSTRAINT_COMPLETED)

    def get_active_readable_name(self, time: datetime | None = None, filter_for_human_notification=False) -> str | None:

        current_constraint = self.get_current_active_constraint(time)

        new_val = None

        if current_constraint is None:
            if filter_for_human_notification is False:
                if self._last_completed_constraint is not None:
                    new_val = "COMPLETED: " + self._last_completed_constraint.get_readable_name_for_load()
                else:
                    new_val = "NOTHING PLANNED"

        else:
            new_val = current_constraint.get_readable_name_for_load()

        return new_val

    def get_active_state_hash(self, time: datetime) -> str:

        current_constraint = self.get_current_active_constraint(time)

        if current_constraint is None:
            if self._last_completed_constraint is not None:
                load_param = "NO"
                if self._last_completed_constraint.load_param is not None:
                    load_param = self._last_completed_constraint.load_param
                new_val = (
                    "COMPLETED:"
                    + self._last_completed_constraint.stable_name
                    + "-"
                    + load_param
                    + "-"
                    + self._last_completed_constraint.end_of_constraint.strftime("%Y-%m-%d %H:%M:%S")
                )
            else:
                new_val = "NOTHING PLANNED"
        else:
            load_param = "NO"
            if current_constraint.load_param is not None:
                load_param = current_constraint.load_param
            if current_constraint.as_fast_as_possible:
                end_str = "ASAP"
            else:
                end_str = current_constraint.end_of_constraint.strftime("%Y-%m-%d %H:%M:%S")
            new_val = "RUNNING:" + current_constraint.stable_name + "-" + load_param + "-" + end_str

        return new_val

    def get_active_constraints(self, time: datetime) -> list[LoadConstraint]:
        if self.qs_enable_device is False:
            self._constraints = []

        if not self._constraints:
            self._constraints = []

        return [c for c in self._constraints if c.is_constraint_active_for_time_period(time)]

    def set_live_constraints(self, time: datetime, constraints: list[LoadConstraint]):

        if not constraints:
            constraints = []

        self._constraints = constraints
        if not constraints:
            return

        self._constraints = [c for c in self._constraints if c is not None]
        self._constraints.sort(key=lambda x: x.end_of_constraint)

        # remove all the infinite constraints but the last one
        if self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
            removed_infinits: list[LoadConstraint] = []
            while self._constraints[-1].end_of_constraint == DATETIME_MAX_UTC:
                removed_infinits.append(self._constraints.pop())
                if len(self._constraints) == 0:
                    break

            # only one infinite is allowed!
            if removed_infinits:
                keep: LoadConstraint = removed_infinits[0]
                for k in removed_infinits:
                    if k.is_constraint_met(time=time):
                        continue
                    if k.score(time) > keep.score(time):
                        keep = k

                self._constraints.append(keep)

        # only one as fast as possible constraint can be active at a time.... and has to be first
        removed_as_fast = [(i, c) for i, c in enumerate(self._constraints) if c.as_fast_as_possible]
        if len(removed_as_fast) == 0 or (len(removed_as_fast) == 1 and removed_as_fast[0][0] == 0):
            # ok if there is a as fast constraint it should be the first one
            pass
        else:
            new_constraints = []
            for i, c in enumerate(self._constraints):
                if i < removed_as_fast[0][0]:
                    continue
                if c.as_fast_as_possible:
                    continue
                new_constraints.append(c)

            keep = removed_as_fast[0][1]
            end_ctr = keep.end_of_constraint
            for _, k in removed_as_fast:
                if k.is_constraint_met(time=time):
                    continue
                if k.score(time) > keep.score(time):
                    keep = k
            keep.end_of_constraint = end_ctr
            self._constraints = [keep]
            self._constraints.extend(new_constraints)

        # check all the constraints that have the same end time, keep the highest score
        current_end = DATETIME_MIN_UTC

        current_cluster: list[tuple[int, LoadConstraint]] = []
        clusters: list[list[tuple[int, LoadConstraint]]] = []

        for i, c in enumerate(self._constraints):
            if c.end_of_constraint == DATETIME_MAX_UTC or c.end_of_constraint == DATETIME_MIN_UTC:
                continue

            if c.end_of_constraint == current_end:
                current_cluster.append((i, c))
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [(i, c)]
                current_end = c.end_of_constraint

        if len(current_cluster) > 1:
            clusters.append(current_cluster)

        if len(clusters) > 0:
            for current_cluster in clusters:
                keep_ic: tuple[int, LoadConstraint] = current_cluster[0]
                for i, c in current_cluster:
                    if c.score(time) > keep_ic[1].score(time):
                        keep_ic = (i, c)

                for i, c in current_cluster:
                    if i == keep_ic[0]:
                        continue
                    else:
                        self._constraints[i] = None

            self._constraints = [c for c in self._constraints if c is not None]

        # and now we may have to recompute the start values of the constraints
        prev_ct = None
        for c in self._constraints:
            if prev_ct is not None:
                c.reset_initial_value_to_follow_prev_if_needed(time, prev_ct)
                if c.is_constraint_met(time=time):
                    # keep the prev energy as it was possibly higher to meet this constraint
                    continue
            prev_ct = c

        self._constraints = [c for c in self._constraints if c.is_constraint_met(time=time) is False]

        # Filter out constraints matching the last completed one (bug #120)
        if self._last_completed_constraint is not None:
            lc = self._last_completed_constraint
            before = len(self._constraints)
            self._constraints = [
                c
                for c in self._constraints
                if not (
                    c.requested_target_value == lc.requested_target_value
                    and (
                        c.end_of_constraint == lc.end_of_constraint
                        or c.end_of_constraint == lc.initial_end_of_constraint
                    )
                )
            ]
            if len(self._constraints) != before:
                _LOGGER.warning(
                    "set_live_constraints: removed %d already-completed "
                    "constraint(s) matching last completed %s for %s",
                    before - len(self._constraints),
                    lc.name,
                    self.name,
                )

        # recompute the constraint start:
        kept = []
        current_start = DATETIME_MIN_UTC
        for c in self._constraints:
            c.current_start_of_constraint = max(current_start, c.start_of_constraint)

            if c.current_start_of_constraint >= c.end_of_constraint:
                # we remove the constraint it is inside another constraint
                continue

            current_start = c.end_of_constraint
            kept.append(c)
            if current_start >= DATETIME_MAX_UTC:
                break

        self._constraints = kept
        if not self._constraints:
            self._constraints = []

    def push_live_constraint(self, time: datetime, constraint: LoadConstraint | None = None) -> tuple[bool, bool]:
        """Push a constraint to the live constraint list.

        Returns (pushed, needs_ack): pushed is True if a change was made,
        needs_ack is True if the constraint was immediately met and the caller
        should await ack_completed_constraint(time, constraint).
        """
        if self.qs_enable_device is False:
            self._constraints = []
            return True, False

        if not self._constraints:
            self._constraints = []

        if constraint is not None:
            # use the requested_target_value instead of teh real target_value, as it could have been updated
            # (by a TimeConstraint for example) and making this test erroneous
            if (
                self._last_completed_constraint is not None
                and self._last_completed_constraint.requested_target_value == constraint.requested_target_value
                and (
                    self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
                    or self._last_completed_constraint.initial_end_of_constraint == constraint.end_of_constraint
                )
            ):
                _LOGGER.debug(
                    "Constraint %s not pushed because same end date (or initial end date) and same target value as last completed one",
                    constraint.name,
                )
                return False, False

            # Carry current_value from completed constraint for same day cycle
            # so that extending a completed target preserves accumulated runtime.
            # Same-cycle carry (Bug #68): end times match.
            if (
                self._last_completed_constraint is not None
                and type(self._last_completed_constraint) == type(constraint)
                and self._last_completed_constraint.current_value > constraint.current_value
                and (
                    self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
                    or self._last_completed_constraint.initial_end_of_constraint == constraint.end_of_constraint
                )
            ):
                constraint.current_value = min(
                    self._last_completed_constraint.current_value,
                    constraint.target_value,
                )

            # If carry-over (from completed or pre-seeded) made constraint
            # immediately met, guardrail _last_completed and signal caller to ack.
            # Return (False, True): no solver-input change was made (bug #120)
            if constraint.is_constraint_met(time=time):
                self._last_completed_constraint = constraint
                return False, True

            for i, c in enumerate(self._constraints):
                if c.eq_no_current(constraint):
                    c.carry_info_from_other_constraint(constraint)
                    return False, False
                if c.end_of_constraint == constraint.end_of_constraint or (
                    c.as_fast_as_possible and constraint.as_fast_as_possible
                ):
                    if c.score(time) == constraint.score(time):
                        _LOGGER.debug(
                            f"Constraint not pushed because same end date as another one, and same score or type old: {c.name} new not added {constraint.name}"
                        )
                        c.carry_info_from_other_constraint(constraint)
                        return False, False
                    else:
                        self._constraints[i] = None
                        _LOGGER.info(
                            f"Constraint {constraint.name} replacing {c.name} one with same end date, different score (last one force replace the new one)"
                        )
                        # the problem here is that we can loose .... the current value
                        if type(c) == type(constraint) and c.current_value > constraint.current_value:
                            constraint.current_value = min(c.current_value, constraint.target_value)
                        # If carry-over made constraint immediately met, signal caller to ack
                        if constraint.is_constraint_met(time=time):
                            self._constraints = [x for x in self._constraints if x is not None]
                            self._last_completed_constraint = constraint
                            return True, True

            self._constraints.append(constraint)
            self.set_live_constraints(time, self._constraints)
            return True, False

        return False, False

    async def update_live_constraints(
        self, time: datetime, period: timedelta, end_constraint_min_tolerancy: timedelta = timedelta(seconds=2)
    ) -> bool:
        # there should be ONLY ONE ACTIVE CONSTRAINT AT A TIME!
        # they are sorted in time order, the first one we find should be executed (could be a constraint with no end date
        # if it is the last and the one before are for the next days)

        if self.qs_enable_device is False:
            self._constraints = []
            return True

        current_constraint = None
        # if self.running_command is not None:
        #    force_solving =  False
        # elif
        # to update any constraint the load must be in a state with the right command working...do not update constraints during its execution
        # well don't like it ... running command will be gracefully handled by launch command
        if not self._constraints:
            self._constraints = []
            force_solving = False
        else:
            force_solving = False

            # be sure we don't forget one ...
            for c in self._constraints:
                c.skip = False

            for i, c in enumerate(self._constraints):
                if c.skip:
                    continue

                do_update_c = False

                if c.is_constraint_met(time=time):
                    c.skip = True
                    force_solving = True
                    await self.ack_completed_constraint(time, c)
                    _LOGGER.info("%s skipped because met", c.name)
                elif c.end_of_constraint <= time and c.is_mandatory is False:
                    _LOGGER.info("%s skipped because not mandatory", c.name)
                    c.skip = True
                    force_solving = True
                elif (
                    c.is_mandatory
                    and c.end_of_constraint < time + end_constraint_min_tolerancy
                    and c.always_end_at_end_of_constraint is False
                ):
                    # c.always_end_at_end_of_constraint : means we will never push it ever
                    # a not met mandatory one! we should expand it or force it
                    duration_s = c.best_duration_extension_to_push_constraint(
                        time, end_constraint_min_tolerancy
                    )  # extend if we continue to push it
                    new_constraint_end = time + duration_s
                    handled_constraint_force = False
                    c.skip = True

                    if i < len(self._constraints) - 1:
                        for j in range(i + 1, len(self._constraints)):
                            nc = self._constraints[j]

                            if nc.skip:
                                continue

                            if nc.end_of_constraint < time:
                                c.skip = True
                                continue

                            if nc.end_of_constraint >= new_constraint_end:
                                break

                            if nc.end_of_constraint < new_constraint_end:
                                if nc.is_constraint_met(time=time):
                                    nc.skip = True
                                else:
                                    force_solving = True
                                    # nc constraint may need to be forced or not
                                    if nc.score(time) > c.score(time):
                                        # we should skip the current one
                                        c.skip = True
                                        handled_constraint_force = True
                                        # make the current constraint the next important one
                                        # to break below after if handled_constraint_force:
                                        c = nc
                                        break
                                    else:
                                        nc.skip = True

                    if handled_constraint_force is False:
                        if c.pushed_count > 4:
                            # TODO: we should send a push notification to the one attached to the constraint!
                            # As it is not met and pushed too many times
                            c.skip = True
                            _LOGGER.info("%s not met and pushed too many times", c.name)
                        else:
                            # unskip the current one
                            c.skip = False
                            c.pushed_count += 1
                            _LOGGER.info(
                                f"{c.name} pushed because mandatory and not met (#pushed {c.pushed_count}) from {c.end_of_constraint} to {new_constraint_end}"
                            )
                            handled_constraint_force = True
                            c.end_of_constraint = new_constraint_end

                    if handled_constraint_force:
                        force_solving = True
                        # ok we have pushed or made a target the next important constraint
                        do_update_c = True
                        c.type = CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE  # force as much as we can....(will only impact off_grid off)
                        _LOGGER.info(
                            f"{c.name} handled_constraint_force is now as fast as possible, end of constraint {c.end_of_constraint} (pushed count {c.pushed_count})"
                        )
                else:
                    do_update_c = True

                if do_update_c and c.is_constraint_active_for_time_period(time, time + period):
                    do_continue_ct = await c.update(time)
                    if do_continue_ct is False:
                        if c.is_constraint_met(time=time):
                            await self.ack_completed_constraint(time, c)
                            _LOGGER.info("%s skipped because met (just after update)", c.name)
                        else:
                            _LOGGER.info("%s stopped by callback (just after update)", c.name)
                        c.skip = True
                    break

            constraints = [c for c in self._constraints if c.skip is False]

            if len(constraints) != len(self._constraints):
                force_solving = True

            self.set_live_constraints(time, constraints)

            current_constraint = self.get_current_active_constraint(time)

        if current_constraint is not None:
            self.current_constraint_current_value = current_constraint.current_value
            self.current_constraint_current_energy = current_constraint.convert_target_value_to_energy(
                current_constraint.current_value
            )
            self.current_constraint_current_percent_completion = current_constraint.get_percent_completion(time)

        else:
            if self._last_completed_constraint is not None:
                self.current_constraint_current_value = self._last_completed_constraint.target_value
                self.current_constraint_current_energy = self._last_completed_constraint.convert_target_value_to_energy(
                    self._last_completed_constraint.target_value
                )
                self.current_constraint_current_percent_completion = 100.0
            else:
                self.current_constraint_current_value = None
                self.current_constraint_current_energy = None
                self.current_constraint_current_percent_completion = None

        self.next_or_current_constraint_start_time = None
        self.next_or_current_constraint_end_time = None
        if self._constraints:
            c = self._constraints[0]
            if c.current_start_of_constraint > DATETIME_MIN_UTC:
                self.next_or_current_constraint_start_time = c.current_start_of_constraint
            if c.end_of_constraint < DATETIME_MAX_UTC:
                self.next_or_current_constraint_end_time = c.end_of_constraint

        return force_solving

    async def mark_current_constraint_has_done(self, time: datetime | None = None):
        if time is None:
            time = datetime.now(tz=pytz.UTC)
        c = self.get_current_active_constraint(time)
        if c:
            # for it has met, will be properly handled in the update constraint for the load
            c.current_value = c.target_value
            await self.update_live_constraints(time, self.home._period)
            if self.is_load_active(time) is False or self.get_current_active_constraint(time) is None:
                await self.launch_command(
                    time=time,
                    command=CMD_IDLE,
                    ctxt=f"mark_current_constraint_has_done constraint {self.get_current_active_constraint(time)} is active {self.is_load_active(time)}",
                )

    async def user_clean_and_reset(self):
        await super().user_clean_and_reset()
        # QS-256 (D7): the reset button must break ANY override loop — clear
        # ALL override fields (a half-cleared ask-time would leave the UI in
        # ASKED_FOR_RESET) and the causality anchor, BEFORE launching CMD_IDLE
        # so the command cannot be suppressed by a stale override
        self.external_user_initiated_state = None
        self.external_user_initiated_state_time = None
        self.asked_for_reset_user_initiated_state_time = None
        self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None
        self.last_command_execution_time = None
        time = datetime.now(tz=pytz.UTC)
        _LOGGER.info("user_clean_and_reset: %s", self.name)
        await self.launch_command(time=time, command=CMD_IDLE, ctxt=f"user_clean_and_reset: {self.name}")

    async def async_reset_override_state(self):
        time = datetime.now(tz=pytz.UTC)
        self.reset_override_state_and_set_reset_ask_time(time=time)
        if await self.do_run_check_load_activity_and_constraints(time):
            self.home.force_next_solve()

    def reset_override_state_and_set_reset_ask_time(self, time: datetime | None = None):
        if self.external_user_initiated_state is None or self.external_user_initiated_state_time is None:
            self.external_user_initiated_state = None
            self.external_user_initiated_state_time = None
            self.asked_for_reset_user_initiated_state_time = None
            return

        self.external_user_initiated_state = None
        self.external_user_initiated_state_time = None
        if self.asked_for_reset_user_initiated_state_time is None:
            self.asked_for_reset_user_initiated_state_time = time
            self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = time


class TestLoad(AbstractLoad):
    __test__ = False

    def __init__(self, min_p=1500, max_p=1500, min_a=7, max_a=7, **kwargs):
        super().__init__(**kwargs)
        self.min_a = min_a
        self.max_a = max_a
        self.min_p = min_p
        self.max_p = max_p

    def get_min_max_power(self) -> tuple[float, float]:
        return self.min_p, self.max_p


# Re-export time series utilities for backward compatibility.
# Canonical home is home_model.home_utils.
from .home_utils import align_time_series_and_values as align_time_series_and_values  # noqa: F401
from .home_utils import get_slots_from_time_series as get_slots_from_time_series  # noqa: F401
from .home_utils import get_value_from_time_series as get_value_from_time_series  # noqa: F401
