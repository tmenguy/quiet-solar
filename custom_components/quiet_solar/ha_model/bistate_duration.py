import logging
from abc import abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import TYPE_CHECKING

import pytz
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, Platform

from ..const import (
    CONSTRAINT_ORIGINATOR_KEY,
    CONSTRAINT_ORIGINATOR_USER_OVERRIDE,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
)
from ..ha_model.device import HADeviceMixin
from ..home_model.commands import CMD_IDLE, LoadCommand
from ..home_model.constraints import DATETIME_MAX_UTC, LoadConstraint, TimeBasedSimplePowerLoadConstraint
from ..home_model.load import AbstractLoad

if TYPE_CHECKING:
    from .bistate_transport import BistateTransport

bistate_modes = [
    "bistate_mode_auto",
    "bistate_mode_exact_calendar",
    "bistate_mode_default",
]

DEFAULT_USER_OVERRIDE_DURATION_S = 4 * 3600
USER_OVERRIDE_STATE_BACK_DURATION_S = 60

ConstraintItemType = namedtuple(
    "ConstraintItem",
    ["start_schedule", "end_schedule", "target_value", "has_user_forced_constraint", "agenda_push", "degraded_type"],
    defaults=(None, None, 0.0, False, False, None),
)

_LOGGER = logging.getLogger(__name__)


class QSBiStateDuration(HADeviceMixin, AbstractLoad):
    """Shared base for bistate-duration loads (on/off, climate, radiator, pool).

    Subclasses carry ONE pair of state attributes with overlapping but
    NOT identical semantics:

    - ``_state_on`` / ``_state_off`` — the underlying HA-state strings
      the transport dispatches to the device (e.g. ``"on"`` / ``"off"``
      for a switch, ``"heat"`` / ``"off"`` for a climate). The B3
      property shadow below routes reads/writes through
      ``self._transport.state_on / state_off``; these are the SERVICE-CALL
      values.
    - ``_bistate_mode_on`` / ``_bistate_mode_off`` — the **bistate-mode
      select** state strings (Force ON / Force OFF entries in the UI).
      These map to translation keys in ``strings.json``.

    Subclasses follow ONE OF TWO conventions for `_bistate_mode_*`:

    1. **`QSOnOffDuration` / `QSPool` / `QSRadiator`**: hard-coded
       namespaced literals like ``"on_off_mode_on"`` /
       ``"radiator_mode_on"`` (one namespace per host class). The
       select shows "Force ON" / "Force OFF" via those keys; the
       `_state_on/off` HA-state value remains the raw service-call
       value ("on"/"off" for switches, "heat"/"off" etc. for
       climate-backed radiators).
    2. **`QSClimateDuration`**: `_bistate_mode_*` mirrors the raw HVAC
       mode (`"heat"`, `"cool"`, `"fan_only"`, …). The `climate_mode`
       translation registers each HVAC mode as a force-mode state key
       so the dropdown renders "Force HVAC Mode HEAT" etc.

    Cross-subclass logic that compares ``_state_on`` against
    ``_bistate_mode_on`` (or ``_state_off`` vs ``_bistate_mode_off``)
    MUST treat the two as decoupled — the raw HVAC mode and the
    namespaced bistate literal share no namespace and may legitimately
    differ for the same logical state. See
    ``docs/agents/concepts/bistate-duration-devices.md`` for the
    full convention.
    """

    # B6 review-fix — class-level default of `None` so
    # `hasattr` / `is None` checks behave consistently before the
    # subclass `__init__` has had a chance to assign the transport.
    _transport: BistateTransport | None = None

    # B3 review-fix — fallback storage for `_state_on/_state_off` when
    # `_transport` is `None`. The property shadows below route reads
    # and writes through the transport when set, otherwise through
    # these private fields. Subclasses inherit the property and never
    # need to re-declare it.
    _state_on_host: str = "on"
    _state_off_host: str = "off"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bistate_mode = "bistate_mode_auto"

        self.default_on_duration: float | None = 1.0
        self.default_on_finish_time: dt_time | None = dt_time(hour=0, minute=0, second=0)

        self.override_duration: float | None = DEFAULT_USER_OVERRIDE_DURATION_S // 3600

        # to be overcharged by the child class. The `_state_on`/`_state_off`
        # writes flow through the property shadow defined below: when
        # `_transport` is set, the transport observes the change; when
        # not, the writes land in `_state_on_host`/`_state_off_host`.
        self._state_on = "on"
        self._state_off = "off"
        self._bistate_mode_on = "bistate_mode_on"
        self._bistate_mode_off = "bistate_mode_off"
        self.bistate_entity = self.switch_entity

        self.is_load_time_sensitive = True

        self.qs_bistate_current_duration_h: float = 0.0
        self.qs_bistate_current_on_h: float = 0.0
        self._previous_bistate_mode: str | None = None

    # B3 review-fix — `_state_on` / `_state_off` are thin views over
    # the transport. Direct writes (`device._state_on = "auto"`) always
    # update the transport so the host and the service-call layer never
    # diverge. When `_transport` is None (e.g. during super().__init__
    # before the subclass has built the transport), the value lands in
    # the per-instance fallback fields.
    @property
    def _state_on(self) -> str:
        if self._transport is not None:
            return self._transport.state_on
        return self._state_on_host

    @_state_on.setter
    def _state_on(self, value: str) -> None:
        if self._transport is not None:
            self._transport.state_on = value
        else:
            self._state_on_host = value

    @property
    def _state_off(self) -> str:
        if self._transport is not None:
            return self._transport.state_off
        return self._state_off_host

    @_state_off.setter
    def _state_off(self, value: str) -> None:
        if self._transport is not None:
            self._transport.state_off = value
        else:
            self._state_off_host = value

    @property
    def hvac_state_on(self) -> str:
        """Public accessor for the transport's ON-state string.

        Exposes the underlying HA state value that maps to "running" —
        for a switch-backed load this is `"on"`, for a climate-backed
        load it's the configured HVAC ON mode (e.g. `"heat"`,
        `"auto"`). Read by the dashboard template so the JS card can
        compare against the live backing-entity state during the
        cold-start grace window (BH10 review-fix). Public because
        HA's Jinja sandbox restricts leading-underscore access.
        """
        return self._state_on

    def _get_today_boundaries(self, time: datetime) -> tuple[datetime, datetime]:
        """Return (start_of_today_utc, start_of_tomorrow_utc) using local midnight."""
        tomorrow_utc = self.get_proper_local_adapted_tomorrow(time)
        today_utc = self.get_proper_local_adapted_today(time)
        return today_utc, tomorrow_utc

    def _is_calendar_based_mode(self, bistate_mode: str) -> bool:
        """Return True if the current mode uses calendar events for daily metrics."""
        return bistate_mode in ("bistate_mode_auto", "bistate_mode_exact_calendar") and self.calendar is not None

    def _overnight_active_constraints(self, time: datetime, tomorrow_utc: datetime) -> list[LoadConstraint]:
        """Return every currently-running active constraint whose finish time falls
        after the today window (overnight finish time), in ``_constraints`` order.

        The today-window loops only count constraints finishing
        ``<= tomorrow_utc``, so an active constraint that started today but ends
        after local midnight (e.g. 06:30 tomorrow) is otherwise dropped from both
        the ring fill and the target — leaving the card empty while the
        constraint_completion sensor rises. The caller adds the returned
        constraints' target/runtime back in.

        A bistate load can hold several live constraints at once
        (``push_live_constraint`` appends; bug #68 / #95), so *all* matching
        constraints are returned, not just the first. A constraint qualifies when
        it (a) finishes after the today window
        (``end_of_constraint > tomorrow_utc``), (b) has already started
        (``current_start_of_constraint <= time``), (c) is active/unmet at ``time``
        (``is_constraint_active_for_time_period``), and (d) carries a real target
        (``target_value is not None`` — a target-less constraint cannot drive a
        ring/target and is reported unmet, so it is skipped defensively). The
        result is empty when no constraint qualifies, so the caller's behaviour is
        unchanged for the common same-day case.
        """
        return [
            ct
            for ct in self._constraints
            if ct.end_of_constraint > tomorrow_utc
            and ct.current_start_of_constraint <= time
            and ct.target_value is not None
            and ct.is_constraint_active_for_time_period(time)
        ]

    async def update_current_metrics(self, time: datetime, end_range: dt_time | None = None):
        """Update bistate UI metrics with today-only values.

        Two distinct paths depending on mode:
        - Calendar path: fetches calendar events and computes today metrics inline
        - Default path: day-filters _constraints and _last_completed_constraint
        """
        duration_s = 0.0
        run_s = 0.0

        # Short-circuit: during a running user override, show override progress only
        for ct in self._constraints:
            if (
                ct.is_constraint_active_for_time_period(time)
                and ct.load_info is not None
                and ct.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
            ):
                self.qs_bistate_current_on_h = ct.current_value / 3600.0
                self.qs_bistate_current_duration_h = ct.target_value / 3600.0
                return

        if self._is_calendar_based_mode(self.bistate_mode):
            # Calendar path: fetch events and compute today target + past actuals
            today_utc, tomorrow_utc = self._get_today_boundaries(time)
            events = await self.get_next_scheduled_events(time=today_utc, give_currently_running_event=True)

            overnight_cts = self._overnight_active_constraints(time, tomorrow_utc)

            target_s = 0.0
            past_actual_s = 0.0
            for ev_start, ev_end in events:
                if ev_end > tomorrow_utc:
                    continue
                ev_duration = (ev_end - ev_start).total_seconds()
                target_s += ev_duration
                if ev_end <= time:
                    past_actual_s += ev_duration

            duration_s = target_s
            run_s = past_actual_s

            # Add active constraint current_value for today
            for ct in self._constraints:
                if any(ct is oct for oct in overnight_cts):
                    continue
                if ct.end_of_constraint > today_utc and ct.end_of_constraint <= tomorrow_utc:
                    run_s += ct.current_value

            # Include the currently-active constraint(s) finishing overnight
            # (after local midnight) — excluded by the in-window loops above.
            for oct in overnight_cts:
                duration_s += oct.target_value
                run_s += oct.current_value

            # AC5: Sanity-check _last_completed_constraint against calendar
            if self._last_completed_constraint is not None:
                lcc = self._last_completed_constraint
                if (
                    lcc.end_of_constraint != DATETIME_MAX_UTC
                    and lcc.end_of_constraint > today_utc
                    and lcc.end_of_constraint <= tomorrow_utc
                    and abs(lcc.current_value - past_actual_s) > 300
                ):
                    _LOGGER.info(
                        "Calendar metrics sync-check: last completed constraint "
                        "current_value=%s differs from inferred past actual=%s "
                        "for %s (expected when multiple calendar events complete "
                        "in one day since only the last constraint is retained)",
                        lcc.current_value,
                        past_actual_s,
                        self.name,
                    )
        else:
            # Default/pool path: day-filter constraints + last-completed
            today_utc, tomorrow_utc = self._get_today_boundaries(time)

            overnight_cts = self._overnight_active_constraints(time, tomorrow_utc)

            for ct in self._constraints:
                if any(ct is oct for oct in overnight_cts):
                    continue
                if ct.end_of_constraint > today_utc and ct.end_of_constraint <= tomorrow_utc:
                    duration_s += ct.target_value
                    run_s += ct.current_value

            # Include the currently-active constraint(s) finishing overnight
            # (after local midnight) — excluded by the in-window loop above.
            for oct in overnight_cts:
                duration_s += oct.target_value
                run_s += oct.current_value

            if self._last_completed_constraint is not None:
                lcc = self._last_completed_constraint
                lcc_end = getattr(lcc, "initial_end_of_constraint", lcc.end_of_constraint)
                # Skip lcc when an active today-constraint of the same type shares
                # its end date (same-day-cycle: push_live_constraint already
                # carried over runtime — mirror its type + end-date checks)
                already_absorbed = any(
                    type(ct) == type(lcc)
                    and (ct.end_of_constraint == lcc.end_of_constraint or ct.end_of_constraint == lcc_end)
                    for ct in self._constraints
                    if ct.end_of_constraint > today_utc and ct.end_of_constraint <= tomorrow_utc
                )
                # Bug #101: skip lcc when it ended exactly at local midnight
                # (previous day's rollover) and a same-type active constraint
                # represents today's cycle — avoids double-counting yesterday.
                # QS-245 fix #01: today's cycle may itself finish overnight
                # (out-of-window), so treat a same-type overnight constraint as
                # today's active cycle too — otherwise the lcc would be added on
                # top of the overnight constraint and yesterday double-counted.
                rollover_from_previous_day = lcc.end_of_constraint == today_utc and (
                    any(
                        type(ct) == type(lcc)
                        for ct in self._constraints
                        if ct.end_of_constraint > today_utc and ct.end_of_constraint <= tomorrow_utc
                    )
                    or any(type(oct) == type(lcc) for oct in overnight_cts)
                )
                if (
                    not already_absorbed
                    and not rollover_from_previous_day
                    and lcc.end_of_constraint != DATETIME_MAX_UTC
                    and lcc.end_of_constraint >= today_utc  # >= includes exact-midnight boundary (DST)
                    and lcc.end_of_constraint <= tomorrow_utc
                ):
                    duration_s += lcc.target_value
                    run_s += lcc.current_value

        self.qs_bistate_current_on_h = run_s / 3600.0
        self.qs_bistate_current_duration_h = duration_s / 3600.0

    async def user_set_default_on_duration(self, float_value: float, for_init: bool = False):
        self.default_on_duration = float_value
        if for_init is False:
            time = datetime.now(pytz.UTC)
            if await self.do_run_check_load_activity_and_constraints(time):
                self.home.force_next_solve()
            await self.home.update_all_states(time)

    async def user_set_bistate_mode(self, option: str, for_init: bool = False):
        if option not in self.get_bistate_modes():
            _LOGGER.error("bistate_mode: %s is not a valid bistate_mode", option)
            return
        self.bistate_mode = option
        if for_init is False:
            time = datetime.now(pytz.UTC)
            if await self.do_run_check_load_activity_and_constraints(time):
                self.home.force_next_solve()
            await self.home.update_all_states(time)

    def get_power_from_switch_state(self, state: str | None) -> float | None:
        if state is None:
            return None
        if state == self._state_on:
            return self.power_use
        else:
            return 0.0

    def get_bistate_modes(self) -> list[str]:
        if not self.support_user_override():
            # do not allow the user to force the bistate mode in any case
            return bistate_modes
        return bistate_modes + [self._bistate_mode_on, self._bistate_mode_off]

    def support_green_only_switch(self) -> bool:
        if self.load_is_auto_to_be_boosted:
            return False
        return True

    def support_user_override(self) -> bool:
        if self.load_is_auto_to_be_boosted:
            return False
        return True

    def get_platforms(self):
        parent = super().get_platforms()
        parent = set(parent)
        parent.update([Platform.SENSOR, Platform.SWITCH, Platform.SELECT, Platform.TIME, Platform.NUMBER])
        return list(parent)

    @abstractmethod
    def get_virtual_current_constraint_translation_key(self) -> str | None:
        """return the translation key for the current constraint"""

    @abstractmethod
    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""

    def expected_state_from_command(self, command: LoadCommand):
        if command is None:
            return None

        if command.is_off_or_idle():
            return self._state_off
        else:
            return self._state_on

    def expected_state_from_command_or_user(self, command: LoadCommand):

        if self.external_user_initiated_state is not None:
            return self.external_user_initiated_state
        return self.expected_state_from_command(command)

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        """check the states of the switch to see if the command is set"""
        state = self.hass.states.get(self.bistate_entity)

        if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE] or state.state is None:
            return None
        else:
            return state.state == self.expected_state_from_command_or_user(command)

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:

        override_state = self.external_user_initiated_state
        ct = self.get_current_active_constraint(time)
        if ct is not None and ct.load_param is not None and self.external_user_initiated_state is not None:
            if not ct.is_mandatory:
                # if the override constraint is no more mandatory
                if command.is_off_or_idle():
                    override_state = None

        if override_state is not None:
            _LOGGER.info(
                f"External state set...intercept execute_command {command} for load {self.name} to stay in state {override_state}"
            )

            state = self.hass.states.get(self.bistate_entity)

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                pass
            elif state.state == override_state:
                return True

        return await self.execute_command_system(time, command, override_state)

    async def execute_command_system(self, time: datetime, command: LoadCommand, state: str | None) -> bool | None:
        """Delegate the service-call to `self._transport`.

        N2 review-fix — `QSOnOffDuration`, `QSClimateDuration`, and
        `QSRadiator` all forwarded to `self._transport.execute(...)`.
        Hoisting the delegation here removes three near-identical
        overrides and keeps the per-backing logic encapsulated in the
        transport strategy. Subclasses may still override if they need
        bespoke pre/post hooks (none do today).

        BH-G — `time` is part of the public method contract (subclasses
        may override and legitimately use it). The base implementation
        doesn't consume it today; we keep the parameter name unchanged
        rather than rename to `_time` so subclasses can call
        `super().execute_command_system(time, ...)` without surprise.
        """
        _ = time  # part of the public contract; not used by the base delegation
        if self._transport is None:  # pragma: no cover — defensive
            raise RuntimeError(f"{type(self).__name__}: _transport not initialised")
        return await self._transport.execute(self.hass, command, state, self._state_on, self._state_off)

    async def _build_mode_constraint_items(
        self, time: datetime, bistate_mode: str, do_push_constraint_after: datetime | None
    ) -> list[ConstraintItemType]:
        """Build constraint items for the current bistate mode.

        Override in subclasses to handle custom modes (e.g. pool auto/winter).
        """
        constraints = []

        if bistate_mode == self._bistate_mode_on:
            end_schedule = self.get_proper_local_adapted_tomorrow(time)
            start_schedule = do_push_constraint_after
            ct = None
            if start_schedule is None or start_schedule < end_schedule:
                ct = ConstraintItemType(
                    start_schedule=start_schedule,
                    end_schedule=end_schedule,
                    target_value=25 * 3600.0,  # 25 hours, more than a day will force the load to be on
                    has_user_forced_constraint=True,
                    agenda_push=False,
                )
                constraints.append(ct)
            _LOGGER.debug(
                f"_build_mode_constraint_items: bistate _bistate_mode_on {self._bistate_mode_on} for load {self.name} {ct}"
            )
        elif bistate_mode == "bistate_mode_default":
            if self.default_on_duration is not None and self.default_on_finish_time is not None:
                end_schedule = self.get_next_time_from_hours(
                    local_hours=self.default_on_finish_time, time_utc_now=time, output_in_utc=True
                )
                start_schedule = do_push_constraint_after
                ct = None
                if start_schedule is None or start_schedule < end_schedule:
                    ct = ConstraintItemType(
                        start_schedule=start_schedule,
                        end_schedule=end_schedule,
                        target_value=self.default_on_duration * 3600.0,
                        has_user_forced_constraint=False,
                        agenda_push=False,
                    )
                    constraints.append(ct)
                _LOGGER.debug(
                    "_build_mode_constraint_items: bistate bistate_mode_default for load %s %s", self.name, ct
                )
        else:
            events = await self.get_next_scheduled_events(time=time, give_currently_running_event=True)

            for ev in events:
                start_schedule, end_schedule = ev
                if start_schedule is not None and end_schedule is not None:
                    if do_push_constraint_after is not None and end_schedule < do_push_constraint_after:
                        continue

                    if end_schedule <= time:
                        continue

                    # start_schedule = max(time, start_schedule) don't do that the constraint has to be stable! for comparison in the push constraints
                    if do_push_constraint_after is not None:
                        start_schedule = max(do_push_constraint_after, start_schedule)
                    if start_schedule >= end_schedule:
                        continue
                    target_value = (end_schedule - start_schedule).total_seconds()
                    if bistate_mode != "bistate_mode_exact_calendar":
                        # bistate mode auto, start should be None or after the overridden time if any
                        start_schedule = do_push_constraint_after

                    ct = ConstraintItemType(
                        start_schedule=start_schedule,
                        end_schedule=end_schedule,
                        target_value=target_value,
                        has_user_forced_constraint=False,
                        agenda_push=True,
                    )
                    constraints.append(ct)
                    _LOGGER.debug(
                        f"_build_mode_constraint_items: bistate calendar {bistate_mode} for load {self.name} {ct}"
                    )

        return constraints

    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        """check the load activity and set proper constraints"""
        do_force_next_solve = False

        bistate_mode = self.bistate_mode
        override_constraint = None
        do_push_constraint_after = None

        # we want to check that the load hasn't been changed externally from the system:
        if self.is_load_command_set(time) and self.support_user_override():
            # we need to know if the state we have is compatible with the current command
            # well more if it has been set ON or any other stuff externally so that we don't want to reset it to OFF
            # because the user wanted to force the state of the load
            # Ex : I have an HVAC that I manually open at 8pm and I don't want the system to close it because he thinks
            # it should be closed because of electricity price or any other stuffs
            # to maintain that:
            # - we detect that the current command is not one that has been set by the system
            # - we store this command and state change time
            # if not done we create a constraint,marked as user, with the proper command detected parameter (ex for an HVAC it could be multiple)
            if self.external_user_initiated_state_time is not None and (
                time - self.external_user_initiated_state_time
            ).total_seconds() > (3600.0 * self.override_duration):
                _LOGGER.info(
                    f"External state time is long, reset from {self.external_user_initiated_state} for load {self.name} "
                )
                # we need to reset the external user initiated state
                self.reset_override_state_and_set_reset_ask_time(time)
                do_force_next_solve = True
                self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = (
                    None  # a proper constraint re-evaluation for the load will be done "normally"
                )
                self.constraint_reset_and_reset_commands_if_needed(
                    keep_commands=True
                )  # remove any constraint if any we will add it back if needed below
            else:
                if self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is not None:
                    # do nothing below, just ask for a proper constraint evaluation to kill the current override:
                    has_a_running_override = False
                    do_force_next_solve = True
                    self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None
                    self.constraint_reset_and_reset_commands_if_needed(
                        keep_commands=True
                    )  # remove any constraint if any we will add it back if needed below

                else:
                    for i, ct in enumerate(self._constraints):
                        if (
                            ct.load_param is not None
                            and ct.load_info is not None
                            and ct.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
                        ):
                            # we do have already a constraint for this override state
                            override_constraint = ct
                            break

                    state = self.hass.states.get(self.bistate_entity)
                    is_command_overridden_state_changed = False
                    expected_state = "UNKNOWN"
                    expected_state_running = "UNKNOWN"
                    current_state = None

                    if (
                        state is not None
                        and state.state not in [STATE_UNKNOWN, STATE_UNAVAILABLE]
                        and state.state is not None
                    ):
                        current_state = state.state

                    if current_state is not None:
                        if (
                            self.external_user_initiated_state is not None
                            and current_state == self.external_user_initiated_state
                        ):
                            # we are still in the same override ... all is good
                            is_command_overridden_state_changed = False
                        elif (
                            self.external_user_initiated_state is not None
                            and current_state != self.external_user_initiated_state
                        ):
                            # hum : we changed the state from the override ... it is like a new override
                            # should we wait a bit to see if the user is changing back the state ?
                            is_command_overridden_state_changed = True
                        else:
                            # no current override ... check if the command is overridden
                            expected_state_running = expected_state = None
                            if self.current_command is None and self.running_command is None:
                                expected_state_running = expected_state = self.expected_state_from_command(CMD_IDLE)
                            else:
                                if self.current_command is not None:
                                    expected_state = self.expected_state_from_command(self.current_command)
                                else:
                                    expected_state = self.expected_state_from_command(CMD_IDLE)

                                if self.running_command is not None:
                                    expected_state_running = self.expected_state_from_command(self.running_command)

                            if (expected_state is not None and current_state == expected_state) or (
                                expected_state_running is not None and current_state == expected_state_running
                            ):
                                is_command_overridden_state_changed = False
                            else:
                                is_command_overridden_state_changed = True

                    if self.asked_for_reset_user_initiated_state_time is not None:
                        if (time - self.asked_for_reset_user_initiated_state_time).total_seconds() < min(
                            float(USER_OVERRIDE_STATE_BACK_DURATION_S), (3600.0 * self.override_duration) / 2.0
                        ):
                            # small time window after asking for reset, do not consider the command overridden
                            # too soon to launch an override again
                            is_command_overridden_state_changed = False
                        else:
                            # long enough ask to check the fact that the override should be finished
                            self.asked_for_reset_user_initiated_state_time = None

                    if is_command_overridden_state_changed:
                        # "back to normal" — user overrides back to base mode state
                        if (
                            self.external_user_initiated_state is not None
                            and bistate_mode in (self._bistate_mode_off, self._bistate_mode_on)
                            and (
                                (
                                    bistate_mode == self._bistate_mode_off
                                    and current_state == self.expected_state_from_command(CMD_IDLE)
                                )
                                or (bistate_mode == self._bistate_mode_on and current_state == self._state_on)
                            )
                        ):
                            _LOGGER.info(
                                "check_load_activity_and_constraints: bistate "
                                "BACK TO NORMAL %s for load %s (mode %s), "
                                "cancelling override from %s",
                                current_state,
                                self.name,
                                bistate_mode,
                                self.external_user_initiated_state,
                            )
                            self.reset_override_state_and_set_reset_ask_time(time)
                            self.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = (
                                None  # no extra cleanup cycle needed, constraints cleared below
                            )
                            self.constraint_reset_and_reset_commands_if_needed(keep_commands=True)
                            override_constraint = None
                            do_force_next_solve = True
                        else:
                            if self.external_user_initiated_state is not None:
                                _LOGGER.info(
                                    "check_load_activity_and_constraints: bistate "
                                    "OVERRIDE BY USER on %s for load %s "
                                    "(previous override: %s, current state: %s)",
                                    self.bistate_entity,
                                    self.name,
                                    self.external_user_initiated_state,
                                    current_state,
                                )
                            else:
                                _LOGGER.info(
                                    "check_load_activity_and_constraints: bistate "
                                    "OVERRIDE BY USER on %s for load %s "
                                    "(current state: %s, expected: %s, running expected: %s)",
                                    self.bistate_entity,
                                    self.name,
                                    current_state,
                                    expected_state,
                                    expected_state_running,
                                )

                            # the user did something different ... just OVERRIDE the automation for a given time
                            self.external_user_initiated_state = current_state
                            self.external_user_initiated_state_time = time

                            # remove any overridden constraint if any
                            self.constraint_reset_and_reset_commands_if_needed(
                                keep_commands=True
                            )  # remove any constraint if any we will add it back if needed below
                            override_constraint = None  # clear stale ref after constraints wiped

                            # we will create a constraint if the asked state is not idle ...
                            if self.expected_state_from_command(CMD_IDLE) == self.external_user_initiated_state:
                                # idle command
                                do_force_next_solve = True
                                # all constraint removed above : command_and_constraint_reset
                            else:
                                end_schedule = time + timedelta(seconds=(3600.0 * self.override_duration))
                                override_constraint = TimeBasedSimplePowerLoadConstraint(
                                    type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                                    degraded_type=CONSTRAINT_TYPE_FILLER_AUTO,
                                    time=time,
                                    load=self,
                                    load_param=self.external_user_initiated_state,
                                    load_info={CONSTRAINT_ORIGINATOR_KEY: CONSTRAINT_ORIGINATOR_USER_OVERRIDE},
                                    from_user=True,
                                    start_of_constraint=time,
                                    end_of_constraint=end_schedule,
                                    power=self.power_use,
                                    initial_value=0,
                                    target_value=3600.0 * self.override_duration,
                                )

                                pushed, needs_ack = self.push_live_constraint(time, override_constraint)
                                if needs_ack:
                                    await self.ack_completed_constraint(time, override_constraint)
                                if pushed:
                                    _LOGGER.info(
                                        "check_load_activity_and_constraints: bistate "
                                        "load %s pushed user override constraint",
                                        self.name,
                                    )
                                    do_force_next_solve = True

                    if (
                        self.external_user_initiated_state is not None
                        and self.expected_state_from_command(CMD_IDLE) == self.external_user_initiated_state
                    ):
                        do_push_constraint_after = self.external_user_initiated_state_time + timedelta(
                            seconds=(3600.0 * self.override_duration)
                        )

                    if override_constraint is not None and override_constraint.end_of_constraint != DATETIME_MAX_UTC:
                        do_push_constraint_after = override_constraint.end_of_constraint + timedelta(seconds=1)

        if bistate_mode == self._bistate_mode_off:
            # remove all constraints if any ... except if we have a running override
            if do_push_constraint_after is not None:
                # keep ONLY the override
                found_override = False
                removed_one = False
                for i, ct in enumerate(self._constraints):
                    if (
                        ct.load_param is not None
                        and ct.load_info is not None
                        and ct.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
                    ):
                        # we do have already a constraint for this override state
                        found_override = True
                    else:
                        # remove this constraint that is not an override
                        self._constraints[i] = None
                        removed_one = True

                if found_override:
                    if removed_one:
                        do_force_next_solve = True
                        self._constraints = [c for c in self._constraints if c is not None]
                        self.set_live_constraints(time, self._constraints)
                else:
                    if override_constraint is not None:
                        do_force_next_solve = True
                        self.set_live_constraints(time, [override_constraint])
                    else:
                        if len(self._constraints) > 0:
                            do_force_next_solve = True
                        self.constraint_reset_and_reset_commands_if_needed(keep_commands=True)
            else:
                if len(self._constraints) > 0:
                    do_force_next_solve = True
                self.constraint_reset_and_reset_commands_if_needed(keep_commands=True)

            _LOGGER.debug(
                f"check_load_activity_and_constraints: bistate _bistate_mode_off {self._bistate_mode_off} for load {self.name}"
            )
        else:
            constraints = await self._build_mode_constraint_items(time, bistate_mode, do_push_constraint_after)

            if len(constraints) > 0:
                # Detect mode change: existing non-override constraint has a
                # different end time than the new constraints → mode switch
                new_ends = {ct.end_schedule for ct in constraints}
                mode_changed = any(
                    c.end_of_constraint not in new_ends
                    for c in self._constraints
                    if not (
                        c.load_info is not None
                        and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
                    )
                )

                # Supplement end-time detection: if the bistate mode string
                # itself changed we definitely switched modes, even when
                # end times happen to coincide.
                if self._previous_bistate_mode is not None and self._previous_bistate_mode != bistate_mode:
                    mode_changed = True

                saved_runtime = 0.0
                if mode_changed:
                    # Save runtime from ALL constraints (override counts toward
                    # daily target just like force-on)
                    for c in self._constraints:
                        if c.current_value > saved_runtime:
                            saved_runtime = c.current_value
                    # Remove old non-override constraints
                    for i, c in enumerate(self._constraints):
                        is_override = (
                            c.load_info is not None
                            and c.load_info.get(CONSTRAINT_ORIGINATOR_KEY, "") == CONSTRAINT_ORIGINATOR_USER_OVERRIDE
                        )
                        if not is_override:
                            self._constraints[i] = None
                    self._constraints = [c for c in self._constraints if c is not None]

                agend_cts = []
                for ct in constraints:
                    type = CONSTRAINT_TYPE_MANDATORY_END_TIME
                    if ct.has_user_forced_constraint is False and self.is_best_effort_only_load():
                        type = CONSTRAINT_TYPE_FILLER_AUTO  # will be after battery filling lowest priority

                    degraded_type = ct.degraded_type if ct.degraded_type is not None else CONSTRAINT_TYPE_FILLER_AUTO

                    initial_cv = None
                    if mode_changed and saved_runtime > 0:
                        initial_cv = min(saved_runtime, ct.target_value)

                    load_mandatory = TimeBasedSimplePowerLoadConstraint(
                        type=type,
                        degraded_type=degraded_type,
                        time=time,
                        load=self,
                        from_user=ct.has_user_forced_constraint,
                        start_of_constraint=ct.start_schedule,
                        end_of_constraint=ct.end_schedule,
                        power=self.power_use,
                        initial_value=0,
                        current_value=initial_cv,
                        target_value=ct.target_value,
                    )

                    if ct.agenda_push:
                        agend_cts.append(load_mandatory)
                    else:
                        push_res, needs_ack = self.push_live_constraint(time, load_mandatory)
                        if needs_ack:
                            await self.ack_completed_constraint(time, load_mandatory)
                        do_force_next_solve = push_res or do_force_next_solve
                        if push_res:
                            _LOGGER.info(
                                f"check_load_activity_and_constraints: bistate load {self.name} pushed non-agenda constraint {load_mandatory}"
                            )

                if len(agend_cts) > 0:
                    push_res, agenda_to_ack = self.push_agenda_constraints(time, agend_cts)
                    for ct_ack in agenda_to_ack:
                        await self.ack_completed_constraint(time, ct_ack)
                    do_force_next_solve = push_res or do_force_next_solve
                    if push_res:
                        _LOGGER.info(
                            f"check_load_activity_and_constraints: bistate load {self.name} pushed agenda constraints {agend_cts}"
                        )

        self._previous_bistate_mode = bistate_mode

        await self.update_current_metrics(time)

        return do_force_next_solve
