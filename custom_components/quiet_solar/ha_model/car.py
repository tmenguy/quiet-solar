import bisect
import logging
import math
from datetime import datetime, timedelta
from datetime import time as dt_time
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import pytz
from haversine import Unit, haversine
from homeassistant.components import number
from homeassistant.components.recorder import get_instance as recorder_get_instance
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder.models import LazyState
from homeassistant.const import ATTR_ENTITY_ID, STATE_UNAVAILABLE, STATE_UNKNOWN, Platform

from ..const import (
    CAR_API_STALE_THRESHOLD_S,
    CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_CALENDAR,
    CAR_CHARGE_TYPE_MANUAL,
    CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE,
    CAR_CHARGE_TYPE_NOT_PLUGGED,
    CAR_CHARGE_TYPE_PERSON_AUTOMATED,
    CAR_EFFICIENCY_KM_PER_KWH,
    CAR_HARD_WIRED_CHARGER,
    CAR_NOT_HOME_AUTO_RESET_S,
    CAR_SOC_STALE_THRESHOLD_S,
    CAR_STALE_MODE_AUTO,
    CAR_STALE_MODE_FORCE_NOT_STALE,
    CAR_STALE_MODE_FORCE_STALE,
    CHARGE_TIME_CONSTRAINTS_CLEARED,
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER,
    CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_CUSTOM_POWER_CHARGE_VALUES,
    CONF_CAR_ESTIMATED_RANGE_SENSOR,
    CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P,
    CONF_CAR_IS_INVITED,
    CONF_CAR_ODOMETER_SENSOR,
    CONF_CAR_PLUGGED,
    CONF_CAR_TRACKER,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_MINIMUM_OK_CAR_CHARGE,
    DEVICE_STATUS_CHANGE_ERROR,
    DEVICE_STATUS_CHANGE_NOTIFY,
    FORCE_CAR_NO_CHARGER_CONNECTED,
    FORCE_CAR_NO_PERSON_ATTACHED,
    MAX_POSSIBLE_AMPERAGE,
    USER_ORIGINATED_CAR_NAME,
    USER_ORIGINATED_CHARGER_NAME,
    CONF_TYPE_NAME_QSCar,
)
from ..ha_model.device import HADeviceMixin
from ..home_model.constraints import DATETIME_MAX_UTC
from ..home_model.load import AbstractDevice
from .device import convert_distance_to_km, load_from_history

if TYPE_CHECKING:
    from ..ha_model.person import QSPerson


_LOGGER = logging.getLogger(__name__)

MIN_CHARGE_POWER_W = 70

CAR_MAX_EFFICIENCY_HISTORY_S = 3600 * 24 * 31

CAR_DEFAULT_CAPACITY = 100000  # 100 kWh

CAR_MINIMUM_LEFT_RANGE_KM = 40.0


def _round_half_up(value: float) -> int:
    """Round half away-from-zero (avoids Python's banker's rounding at `.5`)."""
    return int(math.floor(value + 0.5))


def _finite_soc_or_none(value: Any) -> float | None:
    """Coerce a persisted SOC numeric to a finite float, or `None` (QS-281).

    A legacy/corrupt blob can hold a non-numeric (`str`) or non-finite
    (`nan`/`inf`) value. QS-281's healthy-path accessor reads the base
    directly, so an unsanitized value would crash the new
    `car_best_estimated_soc_percentage` sensor's `value_fn` or emit `nan`/`inf`
    to HA. Mirrors the finite guard in `_capture_last_valid_base_soc`.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


class QSCar(HADeviceMixin, AbstractDevice):
    conf_type_name = CONF_TYPE_NAME_QSCar

    def __init__(self, **kwargs):

        self.car_hard_wired_charger = kwargs.pop(CAR_HARD_WIRED_CHARGER, None)

        self.car_plugged = kwargs.pop(CONF_CAR_PLUGGED, None)
        self.car_tracker = kwargs.pop(CONF_CAR_TRACKER, None)
        self.car_charge_percent_sensor = kwargs.pop(CONF_CAR_CHARGE_PERCENT_SENSOR, None)
        self.car_charge_percent_max_number = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER, None)
        self.car_odometer_sensor = kwargs.pop(CONF_CAR_ODOMETER_SENSOR, None)
        self.car_estimated_range_sensor = kwargs.pop(CONF_CAR_ESTIMATED_RANGE_SENSOR, None)
        self._conf_car_charge_percent_max_number_steps = kwargs.pop(CONF_CAR_CHARGE_PERCENT_MAX_NUMBER_STEPS, None)
        if self._conf_car_charge_percent_max_number_steps == "":
            self._conf_car_charge_percent_max_number_steps = None
        self.car_battery_capacity = kwargs.pop(CONF_CAR_BATTERY_CAPACITY, None)
        self.car_default_charge = kwargs.pop(CONF_DEFAULT_CAR_CHARGE, 100.0)
        self.car_minimum_ok_charge = kwargs.pop(CONF_MINIMUM_OK_CAR_CHARGE, 30.0)

        self.car_efficiency_km_per_kwh_sensor: str = CAR_EFFICIENCY_KM_PER_KWH

        self.car_is_invited = kwargs.pop(CONF_CAR_IS_INVITED, False)

        self.car_charger_min_charge: int = int(max(0, kwargs.pop(CONF_CAR_CHARGER_MIN_CHARGE, 6)))
        self._conf_car_charger_min_charge = self.car_charger_min_charge
        self.car_charger_max_charge: int = min(
            MAX_POSSIBLE_AMPERAGE, int(max(0, kwargs.pop(CONF_CAR_CHARGER_MAX_CHARGE, 32)))
        )
        self._conf_car_charger_max_charge = self.car_charger_max_charge
        self.car_use_custom_power_charge_values = kwargs.pop(CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, False)
        if self.car_use_custom_power_charge_values is False:
            self.car_is_custom_power_charge_values_3p = None
        else:
            self.car_is_custom_power_charge_values_3p = kwargs.pop(CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P, False)

        self.amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self._last_dampening_update = None

        self.theoretical_amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.theoretical_amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)

        self.customized_amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.customized_amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)

        self.conf_customized_amp_to_power_1p = [-1] * (MAX_POSSIBLE_AMPERAGE)
        self.conf_customized_amp_to_power_3p = [-1] * (MAX_POSSIBLE_AMPERAGE)

        self.car_charge_percent_max_number_steps = []
        if self._conf_car_charge_percent_max_number_steps and isinstance(
            self._conf_car_charge_percent_max_number_steps, str
        ):
            vals_str = self._conf_car_charge_percent_max_number_steps.split(",")
            self.car_charge_percent_max_number_steps = []
            for val in vals_str:
                try:
                    v = int(val.strip())
                    if v >= 0 and v <= 100:
                        self.car_charge_percent_max_number_steps.append(v)
                except ValueError:
                    _LOGGER.error("Invalid value %s for car charge percent max number steps, must be an integer", val)
                    self.car_charge_percent_max_number_steps = []
                    break

            if len(self.car_charge_percent_max_number_steps) > 0:
                self.car_charge_percent_max_number_steps.sort()
                if self.car_charge_percent_max_number_steps[-1] != 100:
                    self.car_charge_percent_max_number_steps.append(100)

        super().__init__(**kwargs)

        self._conf_calendar = self.calendar

        for a in range(len(self.theoretical_amp_to_power_1p)):
            val_1p = float(self.voltage * a)
            val_3p = 3 * val_1p

            self.amp_to_power_1p[a] = self.theoretical_amp_to_power_1p[a] = val_1p
            self.amp_to_power_3p[a] = self.theoretical_amp_to_power_3p[a] = val_3p

        self.can_dampen_strongly_dynamically = True
        if self.car_use_custom_power_charge_values:
            self.can_dampen_strongly_dynamically = False

            for a in range(self.car_charger_min_charge, self.car_charger_max_charge + 1):
                val = float(kwargs.pop(f"charge_{a}", -1))
                if val >= 0:
                    if self.car_is_custom_power_charge_values_3p:
                        self.conf_customized_amp_to_power_3p[a] = self.customized_amp_to_power_3p[
                            a
                        ] = self.amp_to_power_3p[a] = val
                        if a * 3 >= self.car_charger_min_charge and a * 3 <= self.car_charger_max_charge:
                            self.conf_customized_amp_to_power_1p[a * 3] = self.customized_amp_to_power_1p[
                                a * 3
                            ] = self.amp_to_power_1p[a * 3] = val

                    else:
                        self.conf_customized_amp_to_power_1p[a] = self.customized_amp_to_power_1p[
                            a
                        ] = self.amp_to_power_1p[a] = val
                        if (
                            a % 3 == 0
                            and a // 3 >= self.car_charger_min_charge
                            and a // 3 <= self.car_charger_max_charge
                        ):
                            self.conf_customized_amp_to_power_3p[a // 3] = self.customized_amp_to_power_3p[
                                a // 3
                            ] = self.amp_to_power_3p[a // 3] = val

        self._salvable_dampening = {}

        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}

        self.charger = None

        self.do_force_next_charge = False
        self.do_next_charge_time: datetime | None = None

        self._reset_charge_targets()

        self.default_charge_time: dt_time | None = None

        self._qs_bump_solar_priority = False

        # Efficiency learning state
        self._km_per_kwh: float | None = None
        self._efficiency_segments: list[
            tuple[float, float, float, float, datetime]
        ] = []  # delta_km, delta_soc, soc_from, soc_to, time of finishing

        # self._efficiency_deltas = {}
        # self._efficiency_deltas_graph = {}

        self._decreasing_segments: list[list[tuple[float, float] | None | int]] = []
        self._dec_seg_count = 0

        self.current_forecasted_person: QSPerson | None = None
        self._current_forecasted_person_name_from_boot: str | None = None

        self.reset()

        # Car API staleness detection (Story 3.9)
        # Critical sensors: must recover for stale exit
        # Supplementary sensors: tracked for entry, not required for exit
        self._car_api_critical_sensors: list[str | None] = [
            self.car_tracker,
            self.car_plugged,
        ]
        self._car_api_supplementary_sensors: list[str | None] = [
            self.car_charge_percent_sensor,
            self.car_odometer_sensor,
            self.car_estimated_range_sensor,
        ]
        self._car_api_all_sensors: list[str] = [
            s for s in self._car_api_critical_sensors + self._car_api_supplementary_sensors if s is not None
        ]
        self._car_api_stale: bool = False
        self._was_car_api_stale: bool = False
        self._car_api_stale_since: datetime | None = None
        self.car_stale_mode_override: str = CAR_STALE_MODE_AUTO
        self.car_api_stale_percent_mode: bool = False
        self._car_api_inferred_home: bool = False
        self._car_api_inferred_plugged: bool = False
        # Dedup key for the manual "trusting manual assignment" WARNING, kept
        # separate from the inferred-flag state so it survives a stale→recovery
        # transition (which clears the inferred flags via `_exit_stale_mode`)
        # and therefore logs exactly once per genuine contradiction episode.
        self._manual_contradiction_logged: bool = False
        self._car_not_home_since: datetime | None = None
        self._departure_auto_reset_done: bool = False

        # ── Estimated-SOC model (Story QS-243) ───────────────────────────
        # Persisted baselines + accumulator that back the effective-SOC
        # accessor when the SOC API is failed, inaccurate, or absent.
        self._user_base_soc_value: float | None = None
        self._last_valid_base_soc_value: float | None = None
        self._computed_added_delta_soc_percent: float | None = None
        self._user_base_soc_entry_sensor_value: float | None = None
        # API stale-percent state captured at the instant the user entered the
        # manual override — drives the 4-case recovery (None for pre-QS-243).
        self._user_base_soc_entry_api_stale: bool | None = None
        # Not persisted — re-anchored on reboot to avoid counting downtime energy.
        self._delta_soc_last_integration_time: datetime | None = None

        self.attach_ha_state_to_probe(self.car_charge_percent_sensor, is_numerical=True, reload_from_history=True)

        self.attach_ha_state_to_probe(self.car_plugged, is_numerical=False, reload_from_history=True)

        self.attach_ha_state_to_probe(self.car_tracker, is_numerical=False, reload_from_history=True)

        self.attach_ha_state_to_probe(self.car_odometer_sensor, conversion_fn=convert_distance_to_km, is_numerical=True)

        self.attach_ha_state_to_probe(
            self.car_estimated_range_sensor,
            conversion_fn=convert_distance_to_km,
            is_numerical=True,
            reload_from_history=True,
        )

        self.attach_ha_state_to_probe(
            self.car_efficiency_km_per_kwh_sensor,
            is_numerical=True,
            non_ha_entity_get_state=self.car_efficiency_km_per_kwh_sensor_state_getter,
        )

    def _car_person_option(self, person_name: str):
        return person_name

    def _on_user_originated_changed(self, key: str, value: Any) -> None:
        """Auto-capture all current car state when any user-originated value changes."""
        super()._on_user_originated_changed(key, value)
        self.set_user_originated("charge_target_percent", self._next_charge_target)
        self.set_user_originated("charge_target_energy", self._next_charge_target_energy)
        self.set_user_originated("bump_solar", self._qs_bump_solar_priority)
        self.set_user_originated("force_charge", self.do_force_next_charge)
        # charge_time: real time supersedes sentinel; None preserves sentinel
        if self.do_next_charge_time is not None:
            self.set_user_originated("charge_time", self.do_next_charge_time.isoformat())
        elif self.get_user_originated("charge_time") != CHARGE_TIME_CONSTRAINTS_CLEARED:
            self.set_user_originated("charge_time", None)
        # person_name: only snapshot forecast if no explicit user selection
        if not self.has_user_originated("person_name"):
            if self.current_forecasted_person is not None:
                if self._is_person_authorized_for_car(self.current_forecasted_person.name):
                    self.set_user_originated("person_name", self.current_forecasted_person.name)

    def update_to_be_saved_extra_device_info(self, data_to_update: dict):
        super().update_to_be_saved_extra_device_info(data_to_update)

        forecasted_name = None
        if self.current_forecasted_person is not None:
            forecasted_name = self.current_forecasted_person.name
        data_to_update["current_forecasted_person_name_from_boot"] = forecasted_name

        # Estimated-SOC model (Story QS-243). The integration cursor is
        # deliberately NOT persisted (re-anchored on reboot).
        data_to_update["user_base_soc_value"] = self._user_base_soc_value
        data_to_update["last_valid_base_soc_value"] = self._last_valid_base_soc_value
        data_to_update["computed_added_delta_soc_percent"] = self._computed_added_delta_soc_percent
        data_to_update["user_base_soc_entry_sensor_value"] = self._user_base_soc_entry_sensor_value
        data_to_update["user_base_soc_entry_api_stale"] = self._user_base_soc_entry_api_stale

    def use_saved_extra_device_info(self, stored_load_info: dict):
        super().use_saved_extra_device_info(stored_load_info)
        self._current_forecasted_person_name_from_boot = stored_load_info.get(
            "current_forecasted_person_name_from_boot", None
        )

        # Estimated-SOC model (Story QS-243). Pre-QS-243 saved blobs lack
        # these keys and default to None (all-None defaults, no exception).
        # QS-281: the numeric SOC fields are sanitized through `_finite_soc_or_none`
        # so a legacy/corrupt blob holding a `str`/`nan`/`inf` can never reach
        # the healthy-path accessor's `base + delta` read (crash / `nan` emit).
        self._user_base_soc_value = _finite_soc_or_none(stored_load_info.get("user_base_soc_value", None))
        self._last_valid_base_soc_value = _finite_soc_or_none(
            stored_load_info.get("last_valid_base_soc_value", None)
        )
        self._computed_added_delta_soc_percent = _finite_soc_or_none(
            stored_load_info.get("computed_added_delta_soc_percent", None)
        )
        self._user_base_soc_entry_sensor_value = _finite_soc_or_none(
            stored_load_info.get("user_base_soc_entry_sensor_value", None)
        )
        self._user_base_soc_entry_api_stale = stored_load_info.get("user_base_soc_entry_api_stale", None)
        # Re-anchor the integration cursor so the next charge cycle does not
        # integrate energy "delivered" during downtime.
        self._delta_soc_last_integration_time = None

    def device_post_home_init(self, time: datetime):
        """Initialize person assignment from persisted state at startup."""
        super().device_post_home_init(time)

        if self._current_forecasted_person_name_from_boot is not None:
            if self.home:
                person = self.home.get_person_by_name(self._current_forecasted_person_name_from_boot)
                if person is not None:
                    self.current_forecasted_person = person

        if self.get_user_originated("person_name") is not None:
            if self.get_user_originated("person_name") == FORCE_CAR_NO_PERSON_ATTACHED:
                self.current_forecasted_person = None
            else:
                if self.home:
                    person = self.home.get_person_by_name(self.get_user_originated("person_name"))
                    if person is None:
                        self.clear_all_user_originated()
                        self.current_forecasted_person = None
                    elif not self._is_person_authorized_for_car(self.get_user_originated("person_name")):
                        _LOGGER.warning(
                            "device_post_home_init: Car:%s clearing stale Person:%s (not authorized)",
                            self.name,
                            self.get_user_originated("person_name"),
                        )
                        self.clear_all_user_originated()
                        self.current_forecasted_person = None
                    else:
                        self.current_forecasted_person = person
        # Try to bootstrap efficiency from history at startup (best-effort, non-blocking)
        try:
            # Asynchronously try to compute an initial km/kWh from HA history
            if self.hass is not None:
                self.hass.async_create_task(self._async_bootstrap_efficiency_from_history(time))
        except Exception as e:
            _LOGGER.error(
                "device_post_home_init: exception for boostrap efficiency device: %s %s",
                self.name,
                e,
                exc_info=True,
                stack_info=True,
            )

    def _is_person_authorized_for_car(self, person_name: str) -> bool:
        """Check if a person is authorized to drive this car."""
        if self.home is None:
            return True  # can't validate without home context
        person = self.home.get_person_by_name(person_name)
        if person is None:
            return True  # person not registered in home, can't validate
        return self.name in person.authorized_cars

    def _fix_user_selected_person_from_forecast(self):
        """Set person_name from current_forecasted_person, with authorization check."""
        if self.has_user_originated("person_name"):
            return
        if self.current_forecasted_person is None:
            return
        if self._is_person_authorized_for_car(self.current_forecasted_person.name):
            self.set_user_originated("person_name", self.current_forecasted_person.name)
        else:
            _LOGGER.warning(
                "Car:%s skipping auto-assignment of Person:%s (not authorized)",
                self.name,
                self.current_forecasted_person.name,
            )

    def get_car_person_readable_forecast_mileage(self, for_small_standalone: bool = True):
        """Person-forecast line ``"<name>: <forecast>"`` for the car card.

        ``for_small_standalone`` (default ``True``) is forwarded to
        ``QSPerson.get_forecast_readable_string`` and selects the leave-time date
        formatting. The default preserves every existing direct caller (e.g. the
        ``qs_car_person_forecast`` sensor) on the compact form; the origin line
        (``get_car_charge_origin_readable_string``) passes ``False`` to get the
        normal ``today HH:MM`` / ``tomorrow HH:MM`` / ``%Y-%m-%d %H:%M`` form.
        """
        person = self.current_forecasted_person
        if person is None:
            return "No forecasted person"
        else:
            # will update the forecast too if needed
            forecast_str = person.get_forecast_readable_string(for_small_standalone=for_small_standalone)
            return f"{person.name}: {forecast_str}"

    def get_car_persons_options(self) -> list[str]:
        options = []
        if self.home:
            # get the possible persons for the house and their needed charge
            for person in self.home._persons:
                if self.name not in person.authorized_cars:
                    continue

                opt_person = self._car_person_option(person.name)
                if opt_person is not None:
                    options.append(opt_person)

        options.append(FORCE_CAR_NO_PERSON_ATTACHED)
        return options

    def get_car_person_option(self) -> str | None:

        p_name = None
        if self.get_user_originated("person_name") is not None:
            p_name = self.get_user_originated("person_name")
        elif self.current_forecasted_person is not None:
            p_name = self.current_forecasted_person.name

        # check the attributed one if any by the home
        if p_name is not None:
            return self._car_person_option(p_name)

        return None

    async def user_set_person_for_car(self, option: str):

        if option is None:
            option = FORCE_CAR_NO_PERSON_ATTACHED

        if option != FORCE_CAR_NO_PERSON_ATTACHED and self.home:
            p_per = self.home.get_person_by_name(option)
            if p_per is None:
                _LOGGER.error("user_set_person_for_car: WRONG PERSON OPTION PASSED %s", option)
                option = FORCE_CAR_NO_PERSON_ATTACHED
            elif not self._is_person_authorized_for_car(option):
                _LOGGER.warning(
                    "user_set_person_for_car: Car:%s Person:%s not authorized, ignoring",
                    self.name,
                    option,
                )
                return

        new_value = option

        if new_value == self.get_user_originated("person_name"):
            return

        self.set_user_originated("person_name", new_value)

        # Check if the manual selection matches what is already forecasted;
        # if so, no reallocation is needed.
        if option == FORCE_CAR_NO_PERSON_ATTACHED and self.current_forecasted_person is None:
            return
        if self.current_forecasted_person is not None and self.current_forecasted_person.name == new_value:
            return

        if self.home:
            if new_value != FORCE_CAR_NO_PERSON_ATTACHED:
                for car in self.home._cars:
                    if car.name != self.name and car.get_user_originated("person_name") == new_value:
                        # the person is being reassigned to self; the other car's
                        # entire snapshot (charge targets, times, etc.) was tied
                        # to that person, so clear everything.
                        car.clear_all_user_originated()

            await self.home.compute_and_set_best_persons_cars_allocations(force_update=True, do_notify=True)

    async def get_car_mileage_on_period_km(self, from_time: datetime, to_time: datetime) -> float | None:

        res = None

        if self.car_odometer_sensor is not None:
            car_odometers: list[LazyState] = await load_from_history(
                self.hass, self.car_odometer_sensor, from_time - timedelta(days=2), to_time, no_attributes=False
            )

            prev_state = None
            from_state = None
            for odo_state in car_odometers:
                if odo_state is None or odo_state.state == STATE_UNKNOWN or odo_state.state == STATE_UNAVAILABLE:
                    continue
                try:
                    v = float(odo_state.state)
                except TypeError, ValueError:
                    continue

                if odo_state.last_changed > from_time:
                    if prev_state is not None:
                        from_state = prev_state
                    else:
                        from_state = odo_state
                    break
                else:
                    prev_state = odo_state

            to_state = None
            for odo_state in reversed(car_odometers):
                if odo_state is None or odo_state.state == STATE_UNKNOWN or odo_state.state == STATE_UNAVAILABLE:
                    continue
                try:
                    v = float(odo_state.state)
                except TypeError, ValueError:
                    continue

                to_state = odo_state
                break

            if from_state is None or to_state is None:
                res = None
            else:
                from_km, _ = convert_distance_to_km(float(from_state.state), from_state.attributes)
                to_km, _ = convert_distance_to_km(float(to_state.state), to_state.attributes)

                return to_km - from_km

        if res is None and self.car_tracker is not None:
            car_positions = await load_from_history(
                self.hass, self.car_tracker, from_time, to_time, no_attributes=False
            )

            prev_pos = None
            for car_position in car_positions:
                if (
                    car_position is None
                    or car_position.state == STATE_UNKNOWN
                    or car_position.state == STATE_UNAVAILABLE
                ):
                    continue

                state_attr: dict[str, Any] = car_position.attributes

                if state_attr is None:
                    continue

                latitude = state_attr.get("latitude", None)
                longitude = state_attr.get("longitude", None)

                if latitude is None or longitude is None:
                    continue

                cur_pos = (float(latitude), float(longitude))

                if prev_pos is not None:
                    res_add = haversine(prev_pos, cur_pos, unit=Unit.KILOMETERS)
                    if res is None:
                        res = 0.0
                    res += res_add

                prev_pos = cur_pos

        return res

    async def get_best_person_next_need(
        self, time: datetime
    ) -> tuple[bool | None, datetime | None, float | None, Any | None]:

        if self.home:
            person = None
            selected = self.get_user_originated("person_name")
            if selected == FORCE_CAR_NO_PERSON_ATTACHED:
                return (None, None, None, None)
            if selected is not None:
                person = self.home.get_person_by_name(selected)
            if person is None:
                person = self.current_forecasted_person

            if person is not None:
                next_usage_time, p_mileage = person.update_person_forecast(time)
                is_person_covered, current_soc, person_min_target_charge, diff_energy = (
                    self.get_adapt_target_percent_soc_to_reach_range_km(p_mileage, time)
                )

                if next_usage_time is not None:
                    # round the next usage time to the prev 30 minutes
                    next_usage_time = next_usage_time.replace(second=0, microsecond=0)
                    minute = (next_usage_time.minute // 30) * 30
                    next_usage_time = next_usage_time.replace(minute=minute)

                return is_person_covered, next_usage_time, person_min_target_charge, person

        return (None, None, None, None)

    def _is_soc_sensor_stale(self, time: datetime) -> bool:
        """Return True if the SOC percent sensor is stale beyond the SOC threshold."""
        if self.car_is_invited or self.car_charge_percent_sensor is None:
            return False
        last_time, last_value, _ = self.get_sensor_latest_possible_valid_time_value_attr(
            self.car_charge_percent_sensor, tolerance_seconds=None, time=time
        )
        if last_time is None or last_value is None:
            return True
        return (time - last_time).total_seconds() > CAR_SOC_STALE_THRESHOLD_S

    def _have_all_api_sensors_reported(self, time: datetime) -> bool:
        """Return True if every tracked API sensor has ever reported a valid value."""
        for sensor_id in self._car_api_all_sensors:
            last_time, last_value, _ = self.get_sensor_latest_possible_valid_time_value_attr(
                sensor_id, tolerance_seconds=None, time=time
            )
            if last_time is None or last_value is None:
                return False
        return True

    def _get_time_for_sensor(self) -> datetime:
        """Return current UTC time for sensor state getters."""
        return datetime.now(tz=pytz.UTC)

    async def update_states(self, time: datetime):
        """Update states and check car API staleness each cycle."""
        await super().update_states(time)
        self._update_car_api_staleness(time)
        self._update_soc_estimation(time)
        self._capture_last_valid_base_soc(time)
        await self._check_departure_auto_reset(time)

    async def _check_departure_auto_reset(self, time: datetime) -> None:
        """Auto-reset car state after confirmed departure (not-home for CAR_NOT_HOME_AUTO_RESET_S).

        This reads the **raw** tracker, so it is also the upper bound on the
        QS-265 manual-trust override: a manually-assigned car whose tracker is
        wrongly "away" is reset (user-originated marker + inferred flags wiped)
        after `CAR_NOT_HOME_AUTO_RESET_S`. See the manual-trust ceiling note in
        `docs/agents/concepts/car-soc-estimation.md`.

        The ceiling is **tracker-only**: it returns early when ``raw_home is
        True``, so a plug-only contradiction (tracker genuinely home, plug
        sensor unplugged) is never time-bounded here — that case relies on the
        charger-side detach (`charger._check_plugged_val`, which only consults
        `is_car_plugged()` when its own plug sensor is inconclusive).
        """
        if self.car_tracker is None:
            return  # No home sensor — cannot detect departure

        raw_home = self._get_raw_is_car_home(time)
        if raw_home is True:
            # Car is home — reset the departure timer and re-arm for next departure
            self._car_not_home_since = None
            self._departure_auto_reset_done = False
            return

        if raw_home is False:
            if self._car_not_home_since is None:
                # Home→not-home transition: start departure timer
                self._car_not_home_since = time
            elif (
                not self._departure_auto_reset_done
                and (time - self._car_not_home_since).total_seconds() >= CAR_NOT_HOME_AUTO_RESET_S
            ):
                # Confirmed departure — perform full reset (once per departure)
                _LOGGER.info(
                    "Car %s confirmed not home for %s minutes — performing departure auto-reset",
                    self.name,
                    int(CAR_NOT_HOME_AUTO_RESET_S / 60),
                )
                self._departure_auto_reset_done = True
                await self.user_clean_and_reset()

    # ── Car API staleness detection (Story 3.9) ──────────────────────────

    def is_car_api_stale(self, time: datetime) -> bool:
        """Return True if ALL tracked API sensors are stale beyond threshold.

        For invited/generic cars (no API sensors configured), always returns False.
        """
        if self.car_is_invited:
            return False

        if not self._car_api_all_sensors:
            return False

        for sensor_id in self._car_api_all_sensors:
            last_time, last_value, _ = self.get_sensor_latest_possible_valid_time_value_attr(
                sensor_id, tolerance_seconds=None, time=time
            )
            if last_time is not None and (time - last_time).total_seconds() <= CAR_API_STALE_THRESHOLD_S:
                return False

        return True

    def is_car_api_ok(self, time: datetime) -> bool:
        """Return True if car API data is fresh (inverse of is_car_api_stale)."""
        return not self.is_car_api_stale(time)

    def is_car_effectively_stale(self, time: datetime) -> bool:
        """Return the effective stale status, combining raw detection with select override.

        This is the main method all UI and behavioral logic reads.
        """
        if self.car_stale_mode_override == CAR_STALE_MODE_FORCE_NOT_STALE:
            return False
        if self.car_stale_mode_override == CAR_STALE_MODE_FORCE_STALE:
            return True
        # auto mode: raw stale detection or stale-percent mode (includes SOC-only)
        return self._car_api_stale or self.car_api_stale_percent_mode

    def _update_car_api_staleness(self, time: datetime) -> None:
        """Check and update car API staleness state. Called each state cycle."""
        raw_stale = self.is_car_api_stale(time)

        # In auto mode, update the raw stale flag
        if self.car_stale_mode_override == CAR_STALE_MODE_AUTO:
            self._car_api_stale = raw_stale

        # Check if recovery from stale-percent mode is possible
        if self.car_api_stale_percent_mode and self.can_exit_stale_percent_mode(time):
            _LOGGER.info(
                "Car %s API data recovered (was stale since %s)",
                self.name,
                self._car_api_stale_since,
            )
            self._schedule_person_notification(
                f"Car {self.name} recovered",
                f"Your {self.name}'s data is available again",
            )
            # For a manual car the inferred override is owned by the per-cycle
            # reconciliation below (tri-state), so preserve it across the exit
            # rather than wiping it here — a None read on this same recovery
            # cycle must hold the override, not drop it (R5-SF1). For non-manual
            # / charger-less cars the reconciliation does not run, so the
            # default unconditional clear still applies.
            self._exit_stale_mode(preserve_inferred=self._charger_assignment_is_user_originated())
            self._was_car_api_stale = False

        # Manual inferred-flag management for attached cars. Runs BEFORE the
        # SOC-only stale entry and regardless of stale-percent mode so that a
        # manually-assigned car keeps the inferred home/plug override even when
        # SOC-stale estimation and a tracker contradiction coincide on the same
        # cycle (SF2), and drops the override as soon as the raw API agrees
        # again (SF1). It is skipped while fully API-stale (all sensors dead),
        # where there is no reliable raw home/plug signal to reconcile against.
        if self.car_stale_mode_override == CAR_STALE_MODE_FORCE_NOT_STALE:
            # Force-Not-Stale means "trust live data" — drop any latched manual
            # inferred override so the raw home/plug truth is honored. The
            # contradiction reconciliation never runs under this override (and
            # a never-stale manual car is not caught by the stale-percent
            # recovery above), so clear the home/plug flags explicitly here
            # (R3-SF1) — only when set, to avoid churning every cycle (R4-NTH2).
            # The WARNING dedup key is deliberately preserved so toggling FNS and
            # back to AUTO during the same ongoing contradiction does not re-log
            # the "trusting manual assignment" WARNING (R4-NTH3). This is also
            # why it intentionally does NOT delegate to `clear_inferred_flags()`
            # (which would reset that dedup key) — the two clearing sites differ
            # on purpose (R5-NTH3).
            if self._car_api_inferred_home or self._car_api_inferred_plugged:
                self._car_api_inferred_home = False
                self._car_api_inferred_plugged = False
        elif self.charger is not None and not self._car_api_stale:
            self.check_charger_assignment_contradiction(
                self.charger.name, time, manual=self._charger_assignment_is_user_originated()
            )

        # SOC-only stale entry: SOC sensor stale but not full API stale
        if (
            not self._car_api_stale
            and not self.car_api_stale_percent_mode
            and self.car_stale_mode_override != CAR_STALE_MODE_FORCE_NOT_STALE
            and self.can_use_charge_percent_constraints()
            and self._is_soc_sensor_stale(time)
        ):
            self._enter_stale_percent_mode(time)

        effectively_stale = self.is_car_effectively_stale(time)

        # Transition: not stale -> stale
        if effectively_stale and not self._was_car_api_stale:
            self._car_api_stale_since = time
            if self._car_api_stale:
                # Full API stale — all sensors older than threshold
                stale_sensors = self._get_stale_sensor_details(time)
                _LOGGER.warning(
                    "Car %s API data is stale (all sensors older than %s hours): %s",
                    self.name,
                    CAR_API_STALE_THRESHOLD_S / 3600,
                    stale_sensors,
                )
            else:
                # SOC-only stale — SOC sensor specifically is stale
                _LOGGER.warning(
                    "Car %s SOC sensor is stale (older than %s hours) — entering stale-percent mode",
                    self.name,
                    CAR_SOC_STALE_THRESHOLD_S / 3600,
                )
            # Notify only for automatic stale detection, not user-initiated Force Stale
            if self.car_stale_mode_override != CAR_STALE_MODE_FORCE_STALE:
                self._schedule_person_notification(
                    f"Warning: {self.name}",
                    f"Warning: {self.name} data is not available, check the car card",
                    error=True,
                )
            # Activate stale-percent mode if car can use percent constraints
            if self.can_use_charge_percent_constraints():
                self._enter_stale_percent_mode(time)

        # Transition: stale -> not stale (from select change or non-percent recovery)
        elif not effectively_stale and self._was_car_api_stale:
            _LOGGER.info(
                "Car %s exiting stale mode (was stale since %s)",
                self.name,
                self._car_api_stale_since,
            )
            self._exit_stale_mode()

        self._was_car_api_stale = effectively_stale

    def _schedule_person_notification(self, title: str, message: str, error: bool = False) -> None:
        """Notify the person attached to this car. Non-blocking."""
        person = self.current_forecasted_person
        if person is None or self.hass is None:
            return
        change_type = DEVICE_STATUS_CHANGE_ERROR if error else DEVICE_STATUS_CHANGE_NOTIFY
        time = datetime.now(tz=pytz.UTC)
        self.hass.async_create_task(
            person.on_device_state_change(time, device_change_type=change_type, title=title, message=message),
            f"qs_stale_notify_{self.name}",
        )

    def _get_stale_sensor_details(self, time: datetime) -> str:
        """Return a string listing each sensor and how long since its last update."""
        parts = []
        for sensor_id in self._car_api_all_sensors:
            last_time, _, _ = self.get_sensor_latest_possible_valid_time_value_attr(
                sensor_id, tolerance_seconds=None, time=time
            )
            if last_time is not None:
                age_h = (time - last_time).total_seconds() / 3600
                parts.append(f"{sensor_id}: {age_h:.1f}h ago")
            else:
                parts.append(f"{sensor_id}: never updated")
        return ", ".join(parts)

    def _exit_stale_mode(self, preserve_inferred: bool = False) -> None:
        """Clear the stale flags and exit stale-percent mode.

        QS-281: this clears ONLY genuinely stale-specific state. The SOC base
        (`_last_valid_base_soc_value`) is no longer stale-specific — it is the
        "last known good raw value" maintained every cycle by the re-anchor
        (`_capture_last_valid_base_soc`), so exiting stale mode must not wipe it
        nor the accumulator/cursor. On a real stale→healthy exit the SOC sensor
        is fresh (that is why we exit), so the same-cycle re-anchor snaps the
        base to the fresh raw and resets the delta only when it genuinely
        differs — a real charge delta now survives stale-exit unless the API
        reports a different value.

        ``preserve_inferred`` keeps the manual inferred home/plug override in
        place so the immediately-following per-cycle reconciliation
        (`check_charger_assignment_contradiction`) can apply its tri-state rule
        instead of this unconditional wipe — otherwise a `None` (unavailable)
        raw read on a recovery cycle would drop a still-valid override that the
        reconciliation could only re-set on an explicit `False` (R5-SF1).
        """
        self._car_api_stale = False
        self.car_api_stale_percent_mode = False
        if not preserve_inferred:
            self._car_api_inferred_home = False
            self._car_api_inferred_plugged = False
        self._car_api_stale_since = None

    def reset_car_api_stale_detection(self) -> None:
        """Reset the detected stale state so the live API gets a fresh chance.

        Never touches `car_stale_mode_override`: an explicit Force stale /
        Force not stale select override is preserved; only the detected
        state is cleared.
        """
        _LOGGER.debug("Resetting detected car API stale state for %s", self.name)
        self._exit_stale_mode()
        self._was_car_api_stale = False

    def _charger_assignment_is_user_originated(self) -> bool:
        """Return True when the current charger assignment originates from the user.

        Checks the durable user-originated markers on both sides of the
        assignment (car-side ``charger_name`` and charger-side ``car_name``);
        automatic ``attach_car()`` touches neither. This detection form and
        the literal ``manual=True`` at the user entry point are equivalent
        by construction: inside the user action ``manual=True`` holds by
        definition, and the markers are the durable trace of that same
        action for the periodic path.
        """
        if self.charger is None:
            return False
        return (
            self.get_user_originated(USER_ORIGINATED_CHARGER_NAME) == self.charger.name
            or self.charger.get_user_originated(USER_ORIGINATED_CAR_NAME) == self.name
        )

    def check_charger_assignment_contradiction(self, charger_name: str, time: datetime, *, manual: bool) -> None:
        """Trust a manual charger assignment over a contradicting tracker (Feature B).

        When this car is assigned to a charger but the API reports not_home or
        not_plugged, the action depends on the assignment origin:

        - ``manual=True`` — the user explicitly assigned this car to the
          charger, so the user's word is ground truth (QS-265). On a
          contradiction the inferred home/plugged flags are set so the car
          keeps being managed and charged, and a single WARNING is logged per
          contradiction episode. The car is **not** marked stale on this path:
          a manually-assigned car's staleness depends only on its SOC sensor
          (plus the all-sensors-dead and force paths). When the raw API agrees
          again the inferred flags are cleared, so the override never outlives
          the contradiction and a later genuine unplug is honored (SF1). The
          user-originated marker is cleared on physical unplug and when another
          charger claims the car, which bounds any damage from a mistaken
          manual pick.
        - ``manual=False`` — the car was auto-attached by plug-time
          correlation. Identity is only a heuristic: the cable proves *a* car
          is present, never *which* car. A contradiction therefore takes no
          action — it neither marks the car stale nor sets the inferred flags.
          Effective staleness still comes from the SOC-only stale entry and the
          all-sensors-dead rule.

        Manual origin detection lives in `_charger_assignment_is_user_originated`.
        Callers must pass the currently assigned charger's name
        (``self.charger.name``) as ``charger_name``.

        The WARNING is deduplicated on `_manual_contradiction_logged`, a flag
        kept separate from the inferred-flag state. It is set on the first log
        of an episode and re-armed only when the raw API genuinely agrees again
        — so a tracker away→home→away cycle logs once per away (one WARNING per
        contradiction episode), and a stale→recovery transition (which clears
        the inferred flags via `_exit_stale_mode` while the contradiction is
        still ongoing) does **not** re-log. The flag covers any contradiction
        kind: a plug-only contradiction (raw_home True, raw_plugged False) sets
        the inferred home flag too, since both flags travel together.

        Tri-state handling: an explicit ``False`` raw read is a contradiction
        (set the override); affirmative ``True`` reads on every available sensor
        clear it; a ``None`` (unavailable/unknown) read is "no new information"
        and **holds** the current override — so a single-cycle sensor flicker to
        unavailable does not drop a still-valid manual override (R4-SF1).
        """
        if self.car_stale_mode_override == CAR_STALE_MODE_FORCE_NOT_STALE:
            return

        if not manual:
            # Auto-attach: identity is only a heuristic — never override the API.
            # This path intentionally does NOT clear pre-existing inferred flags:
            # a manual→non-manual flip while still attached defers clearing to the
            # detach/reassign paths (`clear_inferred_flags`), so a transient
            # auto-mode read does not race-clear a still-valid manual override.
            return

        # Instantaneous reads (for_duration=None): the reconciliation reasons
        # about the current home/plug truth, not a debounced window — the
        # duration-aware path is a different mechanism and never feeds here (R5-NTH4).
        raw_home = self._get_raw_is_car_home(time)
        raw_plugged = self._get_raw_is_car_plugged(time)

        # Contradiction: API affirmatively says not home or not plugged, but the
        # assignment says the car is on the charger. Only an explicit False counts.
        has_contradiction = (raw_home is False) or (raw_plugged is False)
        if not has_contradiction:
            # No contradiction. Clear the override only when the API
            # *affirmatively* agrees on every available sensor — a None
            # (unavailable) read is "no new info", so hold the current override
            # rather than dropping it on a transient flicker (R4-SF1).
            home_ok = raw_home is True if self.car_tracker is not None else True
            plugged_ok = raw_plugged is True if self.car_plugged is not None else True
            if home_ok and plugged_ok:
                # Affirmatively consistent — the inferred override is redundant;
                # clear it (and re-arm the WARNING for the next episode).
                self.clear_inferred_flags()
            return

        # Manual + contradiction: trust the user. Set the inferred flags so the
        # car keeps being managed/charged. Both flags travel together regardless
        # of which sensor contradicts (home-only, plug-only, or both).
        self._car_api_inferred_home = True
        self._car_api_inferred_plugged = True
        # Log one WARNING per contradiction episode (deduped on a dedicated key,
        # so a re-contradiction — including the re-set after a stale recovery
        # with a persistently-away tracker — is an intentional silent no-op).
        if not self._manual_contradiction_logged:
            self._manual_contradiction_logged = True
            _LOGGER.warning(
                "Car %s manually assigned to charger %s but API reports home=%s, plugged=%s "
                "— trusting manual assignment",
                self.name,
                charger_name,
                raw_home,
                raw_plugged,
            )

    def clear_inferred_flags(self) -> None:
        """Clear inferred flags when car is detached from charger.

        Also re-arms the manual-contradiction WARNING so a fresh attachment
        starts a new contradiction episode.
        """
        self._car_api_inferred_home = False
        self._car_api_inferred_plugged = False
        self._manual_contradiction_logged = False

    async def user_set_stale_mode(self, option: str, for_init: bool = False) -> None:
        """Handle stale mode select change. Immediately re-evaluate stale state."""
        self.car_stale_mode_override = option
        if for_init and option == CAR_STALE_MODE_FORCE_STALE:
            # Restore stale-percent mode immediately so get_car_charge_percent()
            # reflects the estimate before the first update_states cycle runs
            if self.can_use_charge_percent_constraints():
                self._enter_stale_percent_mode(None, for_init=True)
        if not for_init:
            time = datetime.now(tz=pytz.UTC)
            self._update_car_api_staleness(time)

    def can_exit_stale_percent_mode(self, time: datetime) -> bool:
        """Context-aware exit from stale-percent mode.

        Common: at least one sensor moved in CAR_API_STALE_THRESHOLD_S, all sensors
        available, SOC fresh.
        Connected path: plug=plugged + home=home.
        Not-connected path: plug=unplugged.
        """
        if not self.car_api_stale_percent_mode:
            return False
        if self.car_stale_mode_override == CAR_STALE_MODE_FORCE_STALE:
            return False
        if self.car_stale_mode_override == CAR_STALE_MODE_FORCE_NOT_STALE:
            return True
        # Common: at least one sensor moved in CAR_API_STALE_THRESHOLD_S. This
        # genuine all-data-dead safety is never bypassed — not even for a manual
        # assignment (a no-SOC-sensor car would otherwise flip-flop here, since
        # `_is_soc_sensor_stale` is vacuously False for it).
        if self.is_car_api_stale(time):
            return False
        # Manual assignment: data is alive, so recover on SOC freshness alone.
        # The raw home/plug *values* may be wrong (that is why the user assigned
        # the car by hand), so they must not gate recovery — only the SOC
        # sensor's freshness does. Guard explicitly on a SOC sensor existing so
        # the branch is self-defending, not merely protected by the preceding
        # all-dead check: for a no-SOC-sensor car `_is_soc_sensor_stale` is
        # vacuously False, which would otherwise mean "always recover" (SF3).
        if self._charger_assignment_is_user_originated() and self.car_charge_percent_sensor is not None:
            return not self._is_soc_sensor_stale(time)
        # Common: all sensors must have valid readings
        if not self._have_all_api_sensors_reported(time):
            return False
        # Common: SOC must be fresh (prevents re-entry flip-flop with 1h threshold)
        if self._is_soc_sensor_stale(time):
            return False

        # Branch on charger connection state
        if self.charger is not None:
            # Connected: plug=plugged AND home=home
            raw_plugged = self._get_raw_is_car_plugged(time)
            raw_home = self._get_raw_is_car_home(time)
            plugged_ok = raw_plugged is True if self.car_plugged is not None else True
            home_ok = raw_home is True if self.car_tracker is not None else True
            return plugged_ok and home_ok
        else:
            # Not connected to a QS charger
            raw_plugged = self._get_raw_is_car_plugged(time)
            if self.car_plugged is None:
                return True  # No plug sensor — other checks sufficient
            if raw_plugged is False:
                return True  # Unplugged — safe to exit
            # Plugged but not attached: allow exit if also home (at our charger)
            if raw_plugged is True:
                raw_home = self._get_raw_is_car_home(time)
                home_ok = raw_home is True if self.car_tracker is not None else True
                return home_ok
            return False

    def car_efficiency_km_per_kwh_sensor_state_getter(
        self, entity_id: str, time: datetime | None
    ) -> tuple[datetime | None, float | str | None, dict | None] | None:

        # Learn efficiency only when car is unplugged and we have both sensors
        if self.car_odometer_sensor is None or self.car_charge_percent_sensor is None:
            return None

        soc = self.get_car_charge_percent_raw_sensor(time)
        odo = self.get_car_odometer_km(time)
        if soc is None or odo is None:
            return None

        # check what was before this sample
        prev_sample = None
        prev_seg_idx = None

        if len(self._decreasing_segments) > 0:
            prev_sample = self._decreasing_segments[-1][1]
            if prev_sample is None:
                prev_sample = self._decreasing_segments[-1][0]
            prev_seg_idx = self._decreasing_segments[-1][2]

        self._add_soc_odo_value_to_segments(soc, odo, time)

        sample_eff = None
        if self.car_estimated_range_sensor is not None:
            current_soc = self.get_car_charge_percent_raw_sensor(time)
            car_estimate = self.get_car_estimated_range_km_from_sensor(time)
            if current_soc is not None and car_estimate is not None and current_soc > 0.0:
                sample_eff = float(car_estimate) / (
                    (float(current_soc) / 100.0) * (float(self.car_battery_capacity) / 1000.0)
                )

        if sample_eff is None:
            if prev_sample is not None and self._decreasing_segments[-1][2] == prev_seg_idx and prev_sample[0] > soc:
                # we have added the new point to the last segment
                prev_soc, prev_odo, prev_time = prev_sample
                if soc < prev_soc and odo > prev_odo:
                    # progressive EMA from segment start to current
                    if self.car_battery_capacity is not None and self.car_battery_capacity > 0:
                        seg_soc0, seg_odo0, seg_t0 = self._decreasing_segments[-1][0]
                        distance_km = float(odo - seg_odo0)
                        delta_soc = float(seg_soc0 - soc)
                        energy_kwh = (delta_soc / 100.0) * (float(self.car_battery_capacity) / 1000.0)
                        if energy_kwh > 0.0 and distance_km > 0.0:
                            sample_eff = distance_km / energy_kwh

        if sample_eff is not None:
            if self._km_per_kwh is None:
                self._km_per_kwh = sample_eff
            else:
                alpha = 0.2
                self._km_per_kwh = alpha * sample_eff + (1.0 - alpha) * self._km_per_kwh

        return (time, self._km_per_kwh, {})

    def get_car_charge_type(self) -> str:
        if self.charger is None:
            return CAR_CHARGE_TYPE_NOT_PLUGGED
        else:
            return self.charger.get_charge_type()[0]

    def get_car_charge_origin_readable_string(self) -> str:
        """Origin-responsive context line for the car card.

        Pure / CPU-only: reads in-memory constraint/load state only. Calls
        ``get_charge_type(return_charge_errors=False)`` so charger-error
        states (Faulted / No Power / Not Plugged) never reach the switch and
        are never surfaced in the origin line.

        Calendar / Manual / as-fast origins return their own dedicated string.
        Every other case — no charger, person-automated, and any other charge
        type (Solar / Solar Priority / Target Met / Not Charging / none) —
        delegates to ``get_car_person_readable_forecast_mileage()`` and surfaces
        the person-forecast line (``"<name>: <forecast>"`` / ``"<name>: No
        forecast"`` / ``"No forecasted person"``).
        """
        if self.charger is not None:
            charge_type, ct = self.charger.get_charge_type(return_charge_errors=False)

            if ct is not None:
                if charge_type == CAR_CHARGE_TYPE_CALENDAR:
                    return f"Calendar · {self._origin_target_single_line(ct)}"

                if charge_type == CAR_CHARGE_TYPE_MANUAL:
                    return f"Manually set to {self._origin_target_single_line(ct)}"

            if charge_type == CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE:
                return "Automatically forced to charge as fast as possible"

            if charge_type == CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE:
                return "Manual as fast as possible charge"

        # No charger, person-automated, or any other type: the person-forecast line.
        # QS-278: render the leave time with the normal date formatting.
        return self.get_car_person_readable_forecast_mileage(for_small_standalone=False)

    @staticmethod
    def _origin_target_single_line(ct) -> str:
        """Target time for the single-line origin row.

        Uses the **normal** ``get_readable_date_string`` formatting
        (``for_small_standalone=False``) so the target renders as
        ``today HH:MM`` / ``tomorrow HH:MM`` / ``%Y-%m-%d %H:%M`` — the more
        legible form wanted on the origin line, rather than the compact
        ``for_small_standalone`` widget form. ``allow_cr=False`` is kept
        explicit to state the single-line intent at the call site; the normal
        formatting branch never emits a newline. Covered by the Calendar/Manual
        tests in ``tests/test_car_charge_origin.py``.
        """
        return ct.get_readable_next_target_date_string(for_small_standalone=False, allow_cr=False)

    async def convert_auto_constraint_to_manual_if_needed(self) -> bool:

        if self.charger is None:
            return False

        type, ct = self.charger.get_charge_type()
        if type == CAR_CHARGE_TYPE_PERSON_AUTOMATED and ct is not None and ct.end_of_constraint != DATETIME_MAX_UTC:
            return await self.user_add_default_charge_at_datetime(ct.end_of_constraint)

        return False

    def get_car_charge_time_readable_name(self):

        if self.charger is None:
            return "--:--"

        # set time as now
        time = datetime.now(pytz.UTC)

        current_constraint = self.charger.get_current_active_constraint(time)

        if current_constraint is None:
            return "--:--"

        return current_constraint.get_readable_next_target_date_string(for_small_standalone=True)

    @property
    def dashboard_sort_string_in_type(self) -> str:
        if self.car_is_invited:
            return "ZZZ"
        return "AAA"

    def get_platforms(self):
        parent = super().get_platforms()
        parent = set(parent)
        parent.update(
            [Platform.SENSOR, Platform.SELECT, Platform.SWITCH, Platform.BUTTON, Platform.TIME, Platform.NUMBER]
        )
        return list(parent)

    def _add_soc_odo_value_to_segments(self, soc: float, odo: float, time: datetime):

        current_vals = (soc, odo, time)

        if len(self._decreasing_segments) > 0:
            current_segment = self._decreasing_segments[-1]
        else:
            current_segment = [current_vals, None, self._dec_seg_count]
            self._decreasing_segments.append(current_segment)
            self._dec_seg_count += 1
            return

        if current_segment[1] is None:
            if soc < current_segment[0][0]:
                # decreasing open really the segment
                current_segment[1] = current_vals
            elif soc == current_segment[0][0]:
                # do nothing keep the segment open
                pass
            else:
                # upper than segment start ... segment no closed, start a new one
                current_segment = [current_vals, None, self._dec_seg_count]
                self._decreasing_segments[-1] = current_segment
                self._dec_seg_count += 1
        else:
            if soc <= current_segment[1][0]:
                # continue decreasing segment
                current_segment[1] = current_vals
            else:
                # close the current segment:
                new_segment = [current_vals, None, self._dec_seg_count]
                self._dec_seg_count += 1
                if current_segment[1][0] < current_segment[0][0] and current_segment[1][1] > current_segment[0][1]:
                    # good soc and odo values, keep it, add the new one
                    self._decreasing_segments.append(new_segment)

                    from_soc = current_segment[0][0]
                    to_soc = current_segment[1][0]

                    # first segment, just add it
                    delta_soc = float(from_soc - to_soc)
                    delta_km = float(current_segment[1][1] - current_segment[0][1])

                    to_be_stored_efficiency_segment = (delta_km, delta_soc, from_soc, to_soc, time)

                    if len(self._efficiency_segments) == 0 or time > self._efficiency_segments[-1][4]:
                        self._efficiency_segments.append(to_be_stored_efficiency_segment)
                    else:
                        # insert it in the time ordered list
                        # bisect key is applied only to list elements, so pass
                        # the bare time value to compare against extracted keys
                        idx = bisect.bisect_left(self._efficiency_segments, time, key=itemgetter(4))
                        # always insert
                        self._efficiency_segments.insert(idx, to_be_stored_efficiency_segment)

                    if (time - self._efficiency_segments[0][4]).total_seconds() > CAR_MAX_EFFICIENCY_HISTORY_S:
                        self._efficiency_segments.pop(0)

                    # to play a bit with graphs with known soc deltas ... not sure it is really useful
                    # self._efficiency_deltas[(from_soc, to_soc)] = delta_km
                    # self._efficiency_deltas[(to_soc, from_soc)] = -delta_km
                    #
                    # fs = self._efficiency_deltas_graph.setdefault(from_soc, set())
                    # fs.add(to_soc)
                    # ts = self._efficiency_deltas_graph.setdefault(to_soc, set())
                    # ts.add(from_soc)

                else:
                    # bad segment, replace it with the new one
                    self._decreasing_segments[-1] = new_segment

    async def _async_bootstrap_efficiency_from_history(self, time: datetime):
        # pull last 14 days; compute efficiency only from segments where SOC decreases
        if self.hass is None:
            return
        if (
            self.car_odometer_sensor is None
            or self.car_charge_percent_sensor is None
            or self.car_battery_capacity is None
            or self.car_battery_capacity <= 0
        ):
            return

        start_time = time - timedelta(days=31)
        end_time = time

        def _load_hist(entity_id: str):
            return state_changes_during_period(
                self.hass, start_time, end_time, entity_id, include_start_time_state=True, no_attributes=True
            ).get(entity_id, [])

        try:
            odos = await recorder_get_instance(self.hass).async_add_executor_job(_load_hist, self.car_odometer_sensor)
            socs = await recorder_get_instance(self.hass).async_add_executor_job(
                _load_hist, self.car_charge_percent_sensor
            )
        except Exception as e:
            _LOGGER.error(
                "_async_bootstrap_efficiency_from_history: exception getting odos and socs %s",
                e,
                exc_info=True,
                stack_info=True,
            )
            return

        if not odos or not socs:
            return

        # Build time series (time, value) with floats, ordered
        def _series(lst):
            res = []
            for s in lst:
                if s is None or s.state in [STATE_UNKNOWN, STATE_UNAVAILABLE, None, "None", "unknown", "unavailable"]:
                    continue
                try:
                    v = float(s.state)
                except TypeError, ValueError:
                    continue
                res.append((s.last_changed, v))
            res.sort(key=lambda x: x[0])
            return res

        odo_series = _series(odos)
        soc_series = _series(socs)
        if len(odo_series) < 2 or len(soc_series) < 2:
            return

        # Helper to get odometer value at or before time (using bisect with key)
        def _odo_at(ts):
            idx = bisect.bisect_right(odo_series, ts, key=itemgetter(0)) - 1
            idx = min(idx, len(odo_series) - 1)
            return odo_series[idx][1]

        total_energy_kwh = 0.0
        total_distance_km = 0.0
        cap_kwh = float(self.car_battery_capacity) / 1000.0

        self._decreasing_segments = []
        self._efficiency_segments = []

        # Iterate SOC segments; only count when SOC decreases

        for t, soc in soc_series:
            odo = _odo_at(t)
            self._add_soc_odo_value_to_segments(soc, odo, t)

        for d_seg in self._decreasing_segments:
            if d_seg[1] is None:
                continue

            soc0, odo0, _ = d_seg[0]
            soc1, odo1, _ = d_seg[1]

            delta_soc = float(soc0 - soc1)

            # we are in a decreasing segment, but maybe with no real SOC decrease, so we check that before counting it
            # we should keep it as we are decreasing
            # if delta_soc < 0.0:
            #    continue

            # impossible to have values at Noe
            # if odo0 is None or odo1 is None:
            #    continue

            delta_km = float(odo1 - odo0)

            if delta_km > 0.0 and delta_soc >= 0:
                energy_kwh = (delta_soc / 100.0) * cap_kwh
                total_energy_kwh += energy_kwh
                total_distance_km += delta_km

        if total_energy_kwh <= 0.0 or total_distance_km <= 1.0:
            return

        eff = total_distance_km / total_energy_kwh
        # no clamping per user request
        self._km_per_kwh = eff

    def reset(self, keep_commands=False):
        super().reset(keep_commands=keep_commands)
        self.interpolate_power_steps(do_recompute_min_charge=True, use_conf_values=True)
        self._dampening_deltas = {}
        self._dampening_deltas_graph = {}
        self.calendar = self._conf_calendar
        self.charger = None
        self.do_force_next_charge = False
        self.do_next_charge_time = None
        self._qs_bump_solar_priority = False

    def attach_charger(self, charger):
        _LOGGER.info("Car %s attaching charger %s", self.name, charger.name)
        charger.attach_car(self, datetime.now(tz=pytz.UTC))

    def detach_charger(self):
        if self.charger is not None:
            _LOGGER.info("Car %s detached charger %s", self.name, self.charger.name)
            self.charger.detach_car()

    def get_continuous_plug_duration(self, time: datetime) -> float | None:

        if self.car_plugged is None:
            return None

        return self.get_last_state_value_duration(
            self.car_plugged, states_vals=["on"], num_seconds_before=None, time=time
        )[0]

    def is_car_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        """Check if car is currently plugged. Returns inferred True if stale-inferred."""
        # Inferred plugged overrides API when stale
        if self._car_api_inferred_plugged:
            return True

        return self._get_raw_is_car_plugged(time, for_duration=for_duration)

    def _get_raw_is_car_plugged(self, time: datetime, for_duration: float | None = None) -> bool | None:
        """Read the actual API plug sensor without inferred overrides."""
        if self.car_plugged is None:
            return None

        if for_duration is None or for_duration == 0:
            for_duration = None

        if for_duration is not None:
            contiguous_status = self.get_last_state_value_duration(
                self.car_plugged, states_vals=["on"], num_seconds_before=8 * for_duration, time=time
            )[0]
            if contiguous_status is not None:
                return contiguous_status >= for_duration and contiguous_status > 0
            else:
                return None
        else:
            latest_state = self.get_sensor_latest_possible_valid_value(entity_id=self.car_plugged, time=time)

            if latest_state is None:
                return None

            return latest_state == "on"

    def get_car_coordinates(self, time: datetime) -> tuple[float, float] | tuple[None, None]:

        if self.car_tracker is None:
            return None, None

        state, state_attr = self.get_sensor_latest_possible_valid_value_and_attr(entity_id=self.car_tracker, time=time)

        if state in [STATE_UNKNOWN, STATE_UNAVAILABLE, None]:
            return None, None

        if state_attr is None:
            return None, None

        latitude: str | None = state_attr.get("latitude", None)
        longitude: str | None = state_attr.get("longitude", None)

        if latitude is None or longitude is None:
            return None, None

        try:
            return float(latitude), float(longitude)
        except ValueError as e:
            _LOGGER.error(
                "get_car_coordinate conversion issue %s %s %s", latitude, longitude, e, exc_info=True, stack_info=True
            )
            return None, None

    def is_car_home(self, time: datetime, for_duration: float | None = None) -> bool | None:
        """Check if car is currently home. Returns inferred True if stale-inferred."""
        # Inferred home overrides API when stale
        if self._car_api_inferred_home:
            return True

        return self._get_raw_is_car_home(time, for_duration=for_duration)

    def _get_raw_is_car_home(self, time: datetime, for_duration: float | None = None) -> bool | None:
        """Read the actual API home tracker without inferred overrides."""
        if self.car_tracker is None:
            return None

        if for_duration is None or for_duration == 0:
            for_duration = None

        if for_duration is not None:
            contiguous_status = self.get_last_state_value_duration(
                self.car_tracker, states_vals=["home"], num_seconds_before=8 * for_duration, time=time
            )[0]
            if contiguous_status is not None:
                return contiguous_status >= for_duration and contiguous_status > 0
            else:
                return None
        else:
            latest_state = self.get_sensor_latest_possible_valid_value(entity_id=self.car_tracker, time=time)

            if latest_state is None:
                return None

            return latest_state == "home"

    def get_car_charge_percent_raw_sensor(
        self, time: datetime | None = None, tolerance_seconds: float | None = None
    ) -> float | None:
        """Return the raw SOC sensor value, ignoring any estimate/override.

        This is the trusted-sensor source used by efficiency learning and the
        charge callback — it must never see the estimated value.
        """
        if self.car_charge_percent_sensor is None:
            return None
        return self.get_sensor_latest_possible_valid_value(
            entity_id=self.car_charge_percent_sensor, time=time, tolerance_seconds=tolerance_seconds
        )

    @property
    def _estimated_soc_percent(self) -> float | None:
        """Absolute estimated SOC (`base + accumulated delta`), clamped, or None.

        Returns None when there is no base (the pure-delta `+XX%` case).
        """
        delta = self._computed_added_delta_soc_percent or 0.0
        if self._user_base_soc_value is not None:
            base = self._user_base_soc_value
        elif self._last_valid_base_soc_value is not None:
            base = self._last_valid_base_soc_value
        else:
            return None
        return max(0.0, min(100.0, base + delta))

    def is_in_soc_estimation_mode(self, time: datetime | None = None) -> bool:
        """Return True when the effective SOC is estimated, not read from the sensor."""
        if self.car_is_invited:
            return False
        if self.car_battery_capacity is None or self.car_battery_capacity <= 0:
            return False
        if self.car_charge_percent_sensor is None:
            return True  # no-sensor car: always estimating
        if self.car_api_stale_percent_mode:
            return True  # API failure / SOC stale / force_stale
        return self._user_base_soc_value is not None  # manual override on a healthy API

    def is_soc_sensor_distrusted(self) -> bool:
        """Return True when the raw SOC sensor cannot be trusted.

        True only when the SOC API is in stale-percent mode or there is no SOC
        sensor at all. A *manual override on a healthy sensor* is NOT distrust:
        the hardware zero-power fault check must still run in that case.
        """
        return self.car_api_stale_percent_mode or self.car_charge_percent_sensor is None

    @property
    def soc_integration_cursor(self) -> datetime | None:
        """Read-only accessor for the SOC accumulator integration cursor."""
        return self._delta_soc_last_integration_time

    def accumulate_soc_delta(self, inc: float | None, time: datetime) -> float:
        """Advance the SOC accumulator and return the value the constraint should use.

        The car owns the running accumulator; the charge callback is its sole
        writer and calls this once per cycle. `inc` is the efficiency-aware
        percent charged since `soc_integration_cursor` (None → first cycle /
        post-reboot / post-base-set: anchor only, do not integrate the gap).
        Returns the absolute estimate when a base exists, otherwise the
        pure-delta accumulator clamped to `[0, 100]`.
        """
        if self._computed_added_delta_soc_percent is None:
            self._computed_added_delta_soc_percent = 0.0
        if self._delta_soc_last_integration_time is None:
            self._delta_soc_last_integration_time = time
        elif inc is not None:
            self._computed_added_delta_soc_percent += inc
            self._delta_soc_last_integration_time = time
        est = self._estimated_soc_percent
        if est is not None:
            return est
        return max(0.0, min(100.0, self._computed_added_delta_soc_percent or 0.0))

    def get_car_charge_percent(
        self, time: datetime | None = None, tolerance_seconds: float | None = None
    ) -> float | None:
        """Get the effective car SOC percent.

        Returns the estimate while estimating (may be None for the pure-delta
        case), otherwise the live raw sensor value.
        """
        if self.is_in_soc_estimation_mode(time):
            return self._estimated_soc_percent
        return self.get_car_charge_percent_raw_sensor(time, tolerance_seconds)

    def get_best_estimated_car_charge_percent(self, time: datetime | None = None) -> float | None:
        """Return the freshest SOC for display (QS-281) — a **pure read**.

        The estimate while estimating or when a system base exists (parity with
        `get_car_charge_percent` for the pure-delta `None` case), otherwise the
        raw sensor. The re-anchor and accumulator updates live elsewhere; see
        `docs/agents/concepts/car-soc-estimation.md` for the full rationale.
        """
        if time is None:
            time = datetime.now(tz=pytz.UTC)
        # Estimating (stale / no-sensor / manual override) OR a healthy sensor
        # that already has a system base → the canonical clamped estimate. The
        # two cases share a return because `_estimated_soc_percent` is a
        # computed-on-read property (`clamp(base + delta)`), so a re-anchor that
        # zeroes the delta is reflected immediately — there is no stored value
        # that could surface a stale one-cycle display. It is `None` only in the
        # pure-delta no-base case. Otherwise the plain raw sensor value, read
        # with the default tolerance (`None`) — identical to the canonical
        # `get_car_charge_percent()` value_fn, which also passes no tolerance.
        if self.is_in_soc_estimation_mode(time) or self._last_valid_base_soc_value is not None:
            return self._estimated_soc_percent
        return self.get_car_charge_percent_raw_sensor(time)

    def _update_soc_estimation(self, time: datetime) -> None:
        """Per-cycle recovery for a user manual override (4-case, keyed on entry state).

        The recovery branches on what the API state was at the moment the user
        entered the override (`_user_base_soc_entry_api_stale`):

        - **Case 1** — entered while in stale API mode: clear once the car has
          exited stale mode and a fresh valid sensor value is available (live
          sensor wins).
        - **Case 2** — entered with API not stale and a valid value: clear only
          when a fresh valid value appears that is *different* from the entry
          value (tolerant compare); if equal, keep both base and delta.
        - **Case 3** — entered with API not stale but no valid value: clear when
          *any* valid value appears.

        System-base recovery (no user override) is handled by `_exit_stale_mode`.
        """
        if self._user_base_soc_value is None:
            return
        if self.car_charge_percent_sensor is None:
            return
        # Every case is triggered by a valid raw reading. Normally we wait for a
        # *fresh* one, but Force-Not-Stale means the user asserts the sensor is
        # trusted even if it is time-stale (S2) — so recovery may proceed then.
        forced_not_stale = self.car_stale_mode_override == CAR_STALE_MODE_FORCE_NOT_STALE
        if self._is_soc_sensor_stale(time) and not forced_not_stale:
            return
        raw = self.get_car_charge_percent_raw_sensor(time)
        if raw is None:
            return

        if self._user_base_soc_entry_api_stale:
            # Case 1 — wait until the car has genuinely left stale-percent mode.
            if not self.car_api_stale_percent_mode:
                self.reset_soc_estimate()
            return

        if self._user_base_soc_entry_sensor_value is None:
            # Case 3 — any valid reading clears.
            self.reset_soc_estimate()
            return

        # Case 2 — clear only on a value that differs from the entry reference
        # (half-up rounding so an exact `.5` reading is not mis-binned — N3).
        if _round_half_up(raw) != _round_half_up(self._user_base_soc_entry_sensor_value):
            self.reset_soc_estimate()

    def _capture_last_valid_base_soc(self, time: datetime) -> None:
        """Re-anchor the SOC base to the last known good raw value (QS-281).

        Runs once per cycle from `update_states`, independent of stale state.
        `_last_valid_base_soc_value` tracks the freshest raw SOC the API has
        reported. When the raw value changes at the **integer** level (mirroring
        the card's `Math.trunc`, so a same-integer heartbeat does NOT re-anchor),
        the accumulated delta is reset to `0.0` and the integration cursor
        re-anchored, so the live estimate snaps back to the truth the API just
        reported. (Sub-integer jitter is absorbed, but a raw oscillating across
        an integer boundary — e.g. `44.9`↔`45.1` — will re-anchor on each flip;
        the integer compare is a coarse de-bounce, not full hysteresis.)

        No-op when the raw value is `None` (no reading yet / no-sensor car) or
        **non-finite** (`nan`/`inf` — a misbehaving template/SOC sensor whose
        literal `"nan"`/`"inf"` survives numeric coercion: `int()` on those
        raises, so the per-cycle `update_states` must not crash). Skipped
        entirely when a manual override is active — that override owns its delta
        lifecycle via `_update_soc_estimation`.
        """
        if self._user_base_soc_value is not None:
            return
        raw = self.get_car_charge_percent_raw_sensor(time)
        # No-op unless `raw` is a finite real number: `None` (no reading), a
        # non-numeric sensor state (the raw read is typed `str | float | None`),
        # or a non-finite `nan`/`inf` must never reach `int()` — it would raise
        # and crash the per-cycle `update_states`.
        if not isinstance(raw, (int, float)) or not math.isfinite(raw):
            return
        base = self._last_valid_base_soc_value
        if not isinstance(base, (int, float)) or not math.isfinite(base) or int(raw) != int(base):
            self._last_valid_base_soc_value = raw
            self._computed_added_delta_soc_percent = 0.0
            self._delta_soc_last_integration_time = None

    def _enter_stale_percent_mode(self, time: datetime | None, for_init: bool = False) -> None:
        """Set stale-percent mode.

        QS-281: the system base is now maintained every cycle by the per-cycle
        re-anchor (`_capture_last_valid_base_soc`), so this no longer captures
        the base on the stale edge — it just flips the mode flag.
        """
        self.car_api_stale_percent_mode = True

    async def user_set_manual_soc_percent(self, value: float, for_init: bool = False) -> None:
        """Set a manual SOC baseline. `for_init` (entity restore) is a no-op."""
        if for_init:
            return
        try:
            value = float(value)
        except TypeError, ValueError:
            return
        if not math.isfinite(value):
            # Reject NaN / ±inf (e.g. a raw `number.set_value` bypassing the card).
            return
        self._user_base_soc_value = float(max(0, min(100, _round_half_up(value))))
        now = datetime.now(tz=pytz.UTC)
        self._user_base_soc_entry_sensor_value = self.get_car_charge_percent_raw_sensor(now)
        # Record the API state at entry so recovery can branch on it (M1).
        self._user_base_soc_entry_api_stale = self.car_api_stale_percent_mode
        self._computed_added_delta_soc_percent = 0.0
        self._delta_soc_last_integration_time = None
        if self.charger:
            await self.charger.update_charger_for_user_change()

    def reset_soc_estimate(self) -> None:
        """Clear every SOC-estimate field and the integration cursor.

        Called on `user_clean_and_reset`, the unplug edge, and a **genuine
        plug-in** (charger.py, the `do_full_reset` branch). It is deliberately
        NOT part of the low-level `reset()`: `attach_car` runs `reset()` on both
        a genuine plug-in AND the boot re-attach, and only the charger can tell
        them apart. Clearing on the genuine-plug branch gives "reset on plug-in"
        while the boot re-attach (`do_full_reset=False`) leaves the persisted
        estimate intact, so it survives an HA reboot.
        """
        self._user_base_soc_value = None
        self._last_valid_base_soc_value = None
        self._computed_added_delta_soc_percent = None
        self._user_base_soc_entry_sensor_value = None
        self._user_base_soc_entry_api_stale = None
        self._delta_soc_last_integration_time = None

    async def user_button_reset_soc_estimate(self) -> None:
        """Reset button action: clear the estimate and re-solve the charger."""
        self.reset_soc_estimate()
        if self.charger:
            await self.charger.update_charger_for_user_change()

    @property
    def qs_car_manual_soc_percent(self) -> float:
        """Current value for the manual-SOC number entity (0 when unset).

        Named to match the number entity's description `key` so the entity's
        restore path (`getattr(device, key)`) reads a numeric default.
        """
        if self._user_base_soc_value is None:
            return 0.0
        return float(self._user_base_soc_value)

    def get_car_charge_energy(self, time: datetime, tolerance_seconds: float | None = None) -> float | None:
        res = self.get_car_charge_percent(time, tolerance_seconds)
        if res is None:
            return None

        if self.car_battery_capacity is None or self.car_battery_capacity == 0:
            return None

        try:
            return float(res) * self.car_battery_capacity / 100.0
        except Exception as e:
            _LOGGER.error(
                "get_car_charge_energy exception: exception for car %s (%s %s) %s",
                self.name,
                res,
                self.car_battery_capacity,
                e,
                exc_info=True,
                stack_info=True,
            )
            return None

    def get_car_odometer_km(self, time: datetime | None = None, tolerance_seconds: float | None = None) -> float | None:
        return self.get_sensor_latest_possible_valid_value(
            entity_id=self.car_odometer_sensor, time=time, tolerance_seconds=tolerance_seconds
        )

    def get_car_estimated_range_km_from_sensor(
        self, time: datetime | None = None, tolerance_seconds: float | None = None
    ) -> float | None:
        return self.get_sensor_latest_possible_valid_value(
            entity_id=self.car_estimated_range_sensor, time=time, tolerance_seconds=tolerance_seconds
        )

    def is_car_charge_growing(self, num_seconds: float, time: datetime) -> bool | None:
        if self.is_in_soc_estimation_mode(time):
            return None
        return self.is_sensor_growing(entity_id=self.car_charge_percent_sensor, num_seconds=num_seconds, time=time)

    def _get_delta_from_graph(self, deltas, deltas_graph, from_v, to_v):

        delta = None

        if len(deltas) > 0:
            delta = deltas.get((from_v, to_v))

            if delta is None and len(deltas) > 2:
                # not direct try a path:
                path = self.find_path(deltas_graph, from_v, to_v)

                if path and len(path) > 1 and path[0] == from_v and path[-1] == to_v:
                    delta = 0
                    for i in range(1, len(path)):
                        d = deltas.get((path[i - 1], path[i]))
                        if d is None:
                            _LOGGER.error(
                                f"_get_delta_from_graph path in error: Car {self.name} deltas {deltas} graph {deltas_graph} from_v {from_v} to_v {to_v} path[i-1] {path[i - 1]} path[i] {path[i]}"
                            )
                            delta = None
                            break

                        delta += d
                elif path:
                    _LOGGER.error(
                        f"_get_delta_from_graph path error: Car {self.name} deltas {deltas} graph {deltas_graph} from_v {from_v} to_v {to_v}"
                    )

        return delta

    def get_computed_range_efficiency_km_per_percent(
        self, time: datetime | None, delta_soc: float = 0.0
    ) -> float | None:

        if time is None:
            time = datetime.now(pytz.UTC)

        current_soc = self.get_car_charge_percent(time)
        car_estimate = self.get_car_estimated_range_km_from_sensor(time)

        if current_soc is not None and car_estimate is not None and current_soc > 0.0:
            return car_estimate / float(current_soc)

        best_segment = None
        # from the more recent to the older
        for i in range(len(self._efficiency_segments) - 1, -1, -1):
            seg = self._efficiency_segments[i]
            if (time - seg[4]).total_seconds() > CAR_MAX_EFFICIENCY_HISTORY_S:
                break

            if seg[0] == 0 or seg[1] == 0:
                continue

            if best_segment is None or (abs(seg[1] - delta_soc) < abs(best_segment[1] - delta_soc)):
                best_segment = seg

        if best_segment is not None:
            return best_segment[0] / best_segment[1]

        if self._km_per_kwh is not None:
            return (self._km_per_kwh * (float(self.car_battery_capacity) / 1000.0)) / 100.0

        _LOGGER.warning("get_computed_range_efficiency_km_per_percent: %s no efficiency data available", self.name)

        return None

    def get_adapt_target_percent_soc_to_reach_range_km(
        self, target_range_km: float | None, time: datetime | None = None
    ) -> tuple[bool | None, float | None, float | None, float | None]:

        km_per_percent = current_soc = current_range_km = None

        minimum_ok_soc = self.get_car_minimum_ok_SOC()

        if target_range_km is not None:
            current_range_km = self.get_estimated_range_km(time)
            current_soc = self.get_car_charge_percent(time)
            km_per_percent = self.get_computed_range_efficiency_km_per_percent(time)

        if km_per_percent is None or target_range_km is None:
            _LOGGER.warning(
                "get_adapt_target_percent_soc_to_reach_range_km: %s error: km_per_percent %s, current_soc %s, current_range_km %s, target_range_km %s",
                self.name,
                km_per_percent,
                current_soc,
                current_range_km,
                target_range_km,
            )
            return None, None, None, None

        target_range_km_a = target_range_km + CAR_MINIMUM_LEFT_RANGE_KM
        needed_soc_a = min(100.0, target_range_km_a / km_per_percent)
        needed_soc_b = 0.0
        target_range_km_b = 0.0
        if minimum_ok_soc is not None:
            # finish at home with the minimal viable soc for the car
            needed_soc_b = min(100.0, minimum_ok_soc + (target_range_km / km_per_percent))
            target_range_km_b = (needed_soc_b) * km_per_percent

        if needed_soc_a > needed_soc_b:
            needed_soc = needed_soc_a
            target_range_km = target_range_km_a
        else:
            needed_soc = needed_soc_b
            target_range_km = target_range_km_b

        # Stale mode: SOC or range unknown — can't determine coverage
        if current_soc is None or current_range_km is None:
            return False, current_soc, needed_soc, None

        diff_energy = (abs(needed_soc - current_soc) * self.car_battery_capacity) / 100.0

        if current_range_km >= target_range_km or current_soc >= needed_soc:
            return True, current_soc, needed_soc, diff_energy
        else:
            return False, current_soc, needed_soc, diff_energy

    def get_car_estimated_range_km(self, from_soc=100.0, to_soc=0.0, time: datetime | None = None) -> float | None:

        # not really useful no?
        # graph_distance = self._get_delta_from_graph(self._efficiency_deltas, self._efficiency_deltas_graph, from_soc, to_soc)

        if time is None:
            time = datetime.now(pytz.UTC)

        delta_soc = abs(from_soc - to_soc)

        eff_perc = self.get_computed_range_efficiency_km_per_percent(time, delta_soc)

        result = None
        if eff_perc is not None:
            result = eff_perc * delta_soc

        return result

    def get_estimated_range_km(self, time: datetime | None = None) -> float | None:

        res = self.get_car_estimated_range_km_from_sensor(time)
        if res is not None:
            return res

        soc = self.get_car_charge_percent(time)
        if soc is None:
            return None
        return self.get_car_estimated_range_km(from_soc=soc, to_soc=0.0, time=time)

    def get_autonomy_to_target_soc_km(self, time: datetime | None = None) -> float | None:

        soc = self.get_car_target_SOC()
        if soc is None:
            return None
        return self.get_car_estimated_range_km(from_soc=soc, to_soc=0.0, time=time)

    async def adapt_max_charge_limit(self, asked_percent):
        # Precondition: this writes the car's native "max charge limit" entity and
        # is only valid while the car is plugged into a QS-managed charger.

        if self.car_charge_percent_max_number is None:
            return

        # Only Quiet Solar should manage the car's native charge limit while the car
        # is plugged into a QS-managed charger. When detached (charging away from
        # home), leave the native limit untouched — the target is preserved and is
        # reapplied automatically on reconnect (see setup_car_charge_target_if_needed
        # callers in charger.py).
        if self.charger is None:
            _LOGGER.debug(
                "Car %s not connected to a managed charger — skipping max charge limit write (target preserved, reapplied on reconnect)",
                self.name,
            )
            return

        percent = asked_percent
        # in fact stop the charge only at the "default charge" of the car else ... continue to charge
        if asked_percent <= self.car_default_charge:
            percent = self.car_default_charge

        if self.car_charge_percent_max_number_steps and len(self.car_charge_percent_max_number_steps) >= 1:
            p_idx = bisect.bisect_left(self.car_charge_percent_max_number_steps, percent)
            p_idx = min(p_idx, len(self.car_charge_percent_max_number_steps) - 1)
            # get the one that is the closest to the percent but bigger or equal
            percent = self.car_charge_percent_max_number_steps[p_idx]

        current_charge_limit = self.get_max_charge_limit()
        if current_charge_limit != percent:
            _LOGGER.info("Car %s set max charge limit from %s%% to %s%%", self.name, current_charge_limit, percent)

            data: dict[str, Any] = {ATTR_ENTITY_ID: self.car_charge_percent_max_number}
            service = number.SERVICE_SET_VALUE
            data[number.ATTR_VALUE] = int(percent)
            domain = number.DOMAIN

            try:
                await self.hass.services.async_call(domain, service, data)
            except Exception as exc:
                _LOGGER.error(
                    f"Car {self.name} failed to set max charge limit to {percent}%: {exc}",
                    exc_info=True,
                    stack_info=True,
                )

    def car_can_limit_its_soc(self):
        if self.car_charge_percent_max_number is None:
            return False
        return True

    def get_max_charge_limit(self):

        result = None
        if self.car_charge_percent_max_number is not None:
            state = self.hass.states.get(self.car_charge_percent_max_number)

            if state is None or state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                result = None
            else:
                try:
                    result = int(state.state)
                except Exception as e:
                    _LOGGER.error(
                        "get_max_charge_limit: exception for car %s (%s) %s",
                        self.name,
                        state.state,
                        e,
                        exc_info=True,
                        stack_info=True,
                    )
                    result = None

        return result

    def find_path(self, graph, start, end, path=None):
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return path

        if start not in graph:
            return None

        for node in graph[start]:
            if node not in path:
                new_path = self.find_path(graph, node, end, path)
                if new_path:
                    return new_path

        return None

    def _get_power_from_stored_amps(self, from_amp: int | float, from_num_phase: int) -> None | float:
        if from_amp < self.car_charger_min_charge:
            from_power = 0.0
        else:
            from_amp = max(min(self.car_charger_max_charge, from_amp), self.car_charger_min_charge)
            if from_num_phase == 1:
                from_power = self.amp_to_power_1p[from_amp]
            else:
                from_power = self.amp_to_power_3p[from_amp]
        return from_power

    def get_delta_dampened_power(
        self, from_amp: int | float, from_num_phase: int, to_amp: int | float, to_num_phase: int
    ) -> float | None:

        if from_amp * from_num_phase == to_amp * to_num_phase:
            return 0.0

        power = self._get_delta_from_graph(
            deltas=self._dampening_deltas,
            deltas_graph=self._dampening_deltas_graph,
            from_v=from_amp * from_num_phase,
            to_v=to_amp * to_num_phase,
        )

        if power is None:
            from_power = self._get_power_from_stored_amps(from_amp, from_num_phase)
            to_power = self._get_power_from_stored_amps(to_amp, to_num_phase)

            if from_power is not None and to_power is not None:
                power = to_power - from_power

        return power

    # def _theoretical_max_power(self, amperage:tuple[float,int] | tuple[int,int], delta_amp:float) -> float:
    #     if amperage[0] == 0:
    #         return 0.0
    #     theoretical_power = float(self.voltage * max(0.0, amperage[0] + delta_amp))
    #     if amperage[1] == 3:
    #         theoretical_power = theoretical_power * 3
    #     return theoretical_power

    def _add_to_amps_power_graph(
        self, from_a: tuple[float, int], to_a: tuple[float, int], power_delta: int | float
    ) -> bool:
        from_amp = int(from_a[0] * from_a[1])
        to_amp = int(to_a[0] * to_a[1])

        if from_amp == to_amp:
            return False

        if from_amp > to_amp:
            if power_delta > 0.0:
                # we do not allow to add a delta that is positive from a higher amperage to a lower amperage
                _LOGGER.warning(
                    f"_add_to_amps_power_graph: {self.name}  from_amp {from_a} > to_amp {to_a} with power_delta {power_delta} - ignoring this value"
                )
                return False

            from_amp, to_amp = to_amp, from_amp
            from_a, to_a = to_a, from_a
            power_delta = -power_delta
        else:
            if power_delta < 0.0:
                # we do not allow to add a delta that is positive from a higher amperage to a lower amperage
                _LOGGER.warning(
                    f"_add_to_amps_power_graph: {self.name}  from_amp {from_a} < to_amp {to_a} with power_delta {power_delta} - ignoring this value"
                )
                return False

        self._dampening_deltas[(from_amp, to_amp)] = power_delta
        self._dampening_deltas[(to_amp, from_amp)] = -power_delta

        fs = self._dampening_deltas_graph.setdefault(from_amp, set())
        fs.add(to_amp)
        ts = self._dampening_deltas_graph.setdefault(to_amp, set())
        ts.add(from_amp)

        return True

    def _can_accept_new_dampen_values(self, old_val: float, new_val: float) -> bool:

        if old_val * new_val < 0:
            # should be same sign
            return False

        old_val = abs(old_val)
        new_val = abs(new_val)

        if old_val < MIN_CHARGE_POWER_W and new_val < MIN_CHARGE_POWER_W:
            return False

        if new_val < MIN_CHARGE_POWER_W:
            # it means we are setting to 0 for the new transition
            if self.can_dampen_strongly_dynamically is False:
                return False
            else:
                return True

        if old_val < MIN_CHARGE_POWER_W:
            # wow it was a 0 something and now it has a value ....
            if self.can_dampen_strongly_dynamically is False:
                return False
            else:
                return True

        ratio = new_val / old_val

        if ratio > 1.10:
            # if growing too much ... we do nothing vs what was there before
            return False

        lower_ratio = 0.2
        if self.can_dampen_strongly_dynamically is False:
            lower_ratio = 0.7

        if ratio < lower_ratio:
            # if going down too much ... we do nothing vs what was there before
            return False

        return True

    def update_dampening_value(
        self,
        amperage: None | tuple[float, int] | tuple[int, int],
        amperage_transition: None | tuple[tuple[int, int] | tuple[float, int], tuple[int, int] | tuple[float, int]],
        power_value_or_delta: int | float,
        time: datetime,
        can_be_saved: bool = False,
    ) -> bool:

        do_update = False

        # if self.can_dampen_strongly_dynamically is False:
        #     _LOGGER.info(f"Car {self.name} cannot dampen dynamically, ignoring amperage {amperage} and amperage_transition {amperage_transition}")
        #     return False

        if amperage_transition is None and amperage is None:
            return False

        if amperage_transition is not None:
            if amperage is None:
                if amperage_transition[0][0] == 0:
                    amperage = amperage_transition[1]
                elif amperage_transition[1][0] == 0:
                    amperage = amperage_transition[0]
                    power_value_or_delta = -power_value_or_delta

            if amperage is None:
                orig_delta = self.get_delta_dampened_power(
                    amperage_transition[0][0],
                    amperage_transition[0][1],
                    amperage_transition[1][0],
                    amperage_transition[1][1],
                )

                if orig_delta is not None:
                    if self._can_accept_new_dampen_values(orig_delta, power_value_or_delta) is False:
                        _LOGGER.info(
                            f"Car {self.name} cannot accept new dampening value for amperage_transition {amperage_transition} with power_value_or_delta {power_value_or_delta} orig_delta {orig_delta} - ignoring this value"
                        )
                        return False

                    if (
                        self._add_to_amps_power_graph(
                            amperage_transition[0], amperage_transition[1], power_value_or_delta
                        )
                        is False
                    ):
                        return False

                    do_update = True

        if amperage is not None:
            if amperage[0] < self.car_charger_min_charge or amperage[0] > self.car_charger_max_charge:
                return False

            if amperage[1] == 3:
                for_3p = True
            else:
                for_3p = False

            amps_val = int(amperage[0])

            old_val = self._get_power_from_stored_amps(amperage[0], amperage[1])

            if power_value_or_delta < -MIN_CHARGE_POWER_W:
                return False

            if abs(power_value_or_delta) < MIN_CHARGE_POWER_W:
                power_value_or_delta = abs(power_value_or_delta)

            if self._can_accept_new_dampen_values(old_val, power_value_or_delta) is False:
                _LOGGER.info(
                    f"Car {self.name} cannot accept new dampening value for amperage {amperage} with power_value_or_delta {power_value_or_delta} orig_delta {old_val} - ignoring this value"
                )
                return False

            if (
                power_value_or_delta >= MIN_CHARGE_POWER_W
                and self._add_to_amps_power_graph((0.0, amperage[1]), (amperage[0], amperage[1]), power_value_or_delta)
                is False
            ):
                return False

            do_recompute_min_charge = False

            car_percent = self.get_car_charge_percent(time)

            is_value_strong_enough = False
            if car_percent is not None and car_percent > 10 and car_percent < 70:
                is_value_strong_enough = True

            if power_value_or_delta >= MIN_CHARGE_POWER_W:
                if for_3p:
                    self.customized_amp_to_power_3p[amps_val] = float(power_value_or_delta)
                    if 3 * amps_val <= self.car_charger_max_charge and 3 * amps_val >= self.car_charger_min_charge:
                        self.customized_amp_to_power_1p[3 * amps_val] = float(power_value_or_delta)
                else:
                    self.customized_amp_to_power_1p[amps_val] = float(power_value_or_delta)
                    if (
                        amps_val % 3 == 0
                        and amps_val // 3 >= self.car_charger_min_charge
                        and amps_val // 3 <= self.car_charger_max_charge
                    ):
                        self.customized_amp_to_power_3p[amps_val // 3] = float(power_value_or_delta)
            elif amps_val <= self._conf_car_charger_min_charge + 2:
                if is_value_strong_enough:
                    power_value_or_delta = 0.0
                    # limit the possibility to have amps 0
                    for i in range(0, amps_val + 1):
                        # no need to do per phase for 0: the car won't take current on amps values only
                        self.customized_amp_to_power_3p[i] = 0.0
                        self.customized_amp_to_power_1p[i] = 0.0

                    if car_percent is None or car_percent < 90.0:
                        do_recompute_min_charge = True

            self.interpolate_power_steps(do_recompute_min_charge=do_recompute_min_charge)
            do_update = True
            if can_be_saved and self.config_entry and is_value_strong_enough:
                if self.car_is_custom_power_charge_values_3p is None:
                    self.car_is_custom_power_charge_values_3p = for_3p

                # only save what was set as conf
                if for_3p == self.car_is_custom_power_charge_values_3p:
                    # for now just store the measured one
                    self._salvable_dampening[f"measured_{CONF_CAR_CUSTOM_POWER_CHARGE_VALUES}"] = (
                        self.car_use_custom_power_charge_values
                    )
                    self._salvable_dampening[f"measured_{CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P}"] = (
                        self.car_is_custom_power_charge_values_3p
                    )
                    self._salvable_dampening[f"measured_charge_{amps_val}"] = power_value_or_delta

                    if (
                        self._last_dampening_update is None
                        or (time - self._last_dampening_update).total_seconds() > 300
                    ):
                        self._last_dampening_update = time
                        data = dict(self.config_entry.data)
                        data.update(self._salvable_dampening)
                        self.save_entry_data_no_reload(data)

        return do_update

    def _interpolate_power_steps(self, customized_amp_to_power, theoretical_amp_to_power, amp_to_power) -> int | float:

        min_charge = self._conf_car_charger_min_charge

        prev_measured_val = customized_amp_to_power[min_charge]

        if prev_measured_val == 0.0 or prev_measured_val > 0 and prev_measured_val < MIN_CHARGE_POWER_W:
            orig_min_charge = min_charge
            for i in range(orig_min_charge, self.car_charger_max_charge):
                prev_measured_val = customized_amp_to_power[i]
                if prev_measured_val == 0.0 or prev_measured_val > 0 and prev_measured_val < MIN_CHARGE_POWER_W:
                    min_charge = i + 1
                    customized_amp_to_power[i] = 0.0
                else:
                    break

        new_vals: list[float] = [0.0] * (len(theoretical_amp_to_power))

        prev_measured_a = min_charge
        prev_measured_val = customized_amp_to_power[prev_measured_a]

        # -1 is by default, so we can use it to detect if no value was set
        if prev_measured_val < 0:
            # compute a best possible first
            prev_measured_val = theoretical_amp_to_power[min_charge]
            first = None
            second = None
            for a in range(min_charge + 1, self.car_charger_max_charge + 1):
                if customized_amp_to_power[a] > 0:
                    if first is None:
                        first = a
                    elif second is None:
                        second = a
                    else:
                        break

            # interpolate the first possible value if we do have some measures, else it will be the theoretical value
            if first is not None and second is not None:
                first_possible_val = (min_charge - first) * (
                    (customized_amp_to_power[second] - customized_amp_to_power[first]) / (second - first)
                ) + customized_amp_to_power[first]
                if first_possible_val > 0:
                    prev_measured_val = min(first_possible_val, prev_measured_val)

        new_vals[min_charge] = prev_measured_val

        for a in range(min_charge + 1, self.car_charger_max_charge + 1):
            measured = customized_amp_to_power[a]

            if a == self.car_charger_max_charge:
                if measured < 0 or (prev_measured_val > 0 and measured > 0 and measured < prev_measured_val):
                    measured = max(prev_measured_val, theoretical_amp_to_power[self.car_charger_max_charge])

            if measured > prev_measured_val or a == self.car_charger_max_charge:
                # only increasing values allowed
                new_vals[a] = measured
                if a > prev_measured_a + 1:
                    for ap in range(prev_measured_a + 1, a):
                        new_vals[ap] = prev_measured_val + (
                            (measured - prev_measured_val) * (ap - prev_measured_a) / (a - prev_measured_a)
                        )
                prev_measured_a = a
                prev_measured_val = measured

        for a in range(0, len(theoretical_amp_to_power)):
            amp_to_power[a] = new_vals[a]

        return min_charge

    def interpolate_power_steps(self, do_recompute_min_charge=False, use_conf_values=False):

        if use_conf_values:
            customized_amp_to_power_3p = self.conf_customized_amp_to_power_3p
            customized_amp_to_power_1p = self.conf_customized_amp_to_power_1p
        else:
            customized_amp_to_power_3p = self.customized_amp_to_power_3p
            customized_amp_to_power_1p = self.customized_amp_to_power_1p

        min_charge_3p = self._interpolate_power_steps(
            customized_amp_to_power_3p, self.theoretical_amp_to_power_3p, self.amp_to_power_3p
        )
        min_charge_1p = self._interpolate_power_steps(
            customized_amp_to_power_1p, self.theoretical_amp_to_power_1p, self.amp_to_power_1p
        )

        if use_conf_values:
            self.car_charger_min_charge = self._conf_car_charger_min_charge

        if do_recompute_min_charge:
            init_car_min_charge = self.car_charger_min_charge
            self.car_charger_min_charge = max(min_charge_3p, min_charge_1p)
            if init_car_min_charge != self.car_charger_min_charge:
                _LOGGER.info(
                    f"interpolate_power_steps: Car {self.name} updated min charge from {init_car_min_charge} to {self.car_charger_min_charge}"
                )

    def get_charge_power_per_phase_A(self, for_3p: bool) -> tuple[list[float], int, int]:
        if for_3p:
            return self.amp_to_power_3p, self.car_charger_min_charge, self.car_charger_max_charge
        else:
            return self.amp_to_power_1p, self.car_charger_min_charge, self.car_charger_max_charge

    async def user_add_default_charge_at_datetime(self, end_charge: datetime) -> bool:

        if self.can_add_default_charge() is False:
            return False

        self.do_next_charge_time = end_charge
        self.set_user_originated("charge_time", end_charge.isoformat())
        # start_time = end_charge
        # end_time = end_charge + timedelta(seconds=60*30)
        # time = datetime.now(pytz.UTC)
        # await self.set_next_scheduled_event(time, start_time, end_time, f"Charge {self.name}")
        return True

    async def user_add_default_charge_at_dt_time(self, default_charge_time: dt_time | None) -> bool:
        if self.can_add_default_charge() is False:
            return False

        if default_charge_time is None:
            _LOGGER.error("Car %s cannot add default charge at None time", self.name)
            return False

        # compute the next occurrence of the default charge time
        next_time = self.get_next_time_from_hours(local_hours=default_charge_time, output_in_utc=True)

        return await self.user_add_default_charge_at_datetime(next_time)

    async def user_add_default_charge(self):

        if self.can_add_default_charge():
            res = await self.user_add_default_charge_at_dt_time(self.default_charge_time)

            if res and self.charger:
                self.do_force_next_charge = False
                self.set_user_originated("force_charge", False)
                await self.charger.update_charger_for_user_change()

    def can_add_default_charge(self) -> bool:
        if self.charger is not None:
            return True
        return False

    def can_force_a_charge_now(self) -> bool:
        if self.charger is not None:
            return True
        return False

    async def user_force_charge_now(self):
        if self.can_force_a_charge_now():
            self.do_force_next_charge = True
            self.do_next_charge_time = None
            self.set_user_originated("force_charge", True)
            if self.charger:
                await self.charger.update_charger_for_user_change()

    def can_use_charge_percent_constraints(self):
        if self.car_is_invited:
            return False
        if self.car_battery_capacity is None or self.car_battery_capacity <= 0:
            return False
        # A SOC sensor is no longer required (Story QS-243): a non-invited car
        # with a true battery capacity estimates SOC from charged energy.
        return True

    def _reset_charge_targets(self):
        self._next_charge_target = None
        self._next_charge_target_energy = None

    async def setup_car_charge_target_if_needed(self, asked_target_charge=None):

        target_charge = asked_target_charge

        if target_charge is None:
            target_charge = self.get_car_target_SOC()

        if target_charge is not None:
            await self.adapt_max_charge_limit(target_charge)

        return target_charge

    def get_car_next_charge_values_options(self):

        if self.can_use_charge_percent_constraints():
            return self.get_car_next_charge_values_options_percent()
        else:
            return self.get_car_next_charge_values_options_energy()

    async def user_set_next_charge_target(self, value: int | float | str):

        do_update = await self.convert_auto_constraint_to_manual_if_needed()

        if self.can_use_charge_percent_constraints():
            do_update = await self.set_next_charge_target_percent(value) or do_update
        else:
            do_update = await self.set_next_charge_target_energy(value) or do_update

        if self.can_use_charge_percent_constraints():
            self.set_user_originated("charge_target_percent", self._next_charge_target)
        else:
            self.set_user_originated("charge_target_energy", self._next_charge_target_energy)

        if do_update and self.charger:
            await self.charger.update_charger_for_user_change()

    def get_car_target_charge_option(self):
        if self.can_use_charge_percent_constraints():
            return self.get_car_target_charge_option_percent()
        else:
            return self.get_car_target_charge_option_energy()

    def get_car_next_charge_values_options_percent(self):
        current_soc = 0

        options = set()
        options.add(100)

        # no need in fact to limit the possible values as the steps are ony for the charge max limit
        # but in fact we will stop charging with the percent of the car target SOC
        # if self.car_charge_percent_max_number_steps and len(self.car_charge_percent_max_number_steps) >= 1:
        #     for v in self.car_charge_percent_max_number_steps:
        #         if v > current_soc:
        #             options.add(v)
        # else:
        if current_soc < 95:
            first = int(current_soc // 5 + 1)
            if first < 20:
                for i in range(first, 20):
                    options.add(i * 5)

        # if current_soc < self.car_default_charge:
        # always add the default
        options.add(int(self.car_default_charge))

        v = int(self.get_car_target_SOC())
        # always add the current set
        options.add(v)

        options = list(options)
        options.sort()

        for i in range(len(options)):
            options[i] = self.get_car_option_charge_from_value_percent(options[i])

        return options

    def get_car_option_charge_from_value_percent(self, value: int | float):
        value = int(float(value))
        if value > 100:
            value = 100

        if value == self.car_default_charge:
            return f"{value}% - {self.name} default"
        elif value == 100:
            return "100% - full"
        else:
            return f"{value}%"

    async def set_next_charge_target_percent(self, value: int | float | str, do_update_charger: bool = True) -> bool:

        if isinstance(value, str):
            if "default" in value:
                value = self.car_default_charge
            elif "full" in value:
                value = 100
            else:
                try:
                    value = value.strip("%")
                    value = float(value)
                except Exception as e:
                    _LOGGER.error(
                        f"Car {self.name} set_next_charge_target: invalid value {value}, must be an integer or 'default' or 'full'{e}",
                        exc_info=True,
                        stack_info=True,
                    )
                    return False

        value = int(value)

        self._next_charge_target = value

        new_target = await self.setup_car_charge_target_if_needed()

        if self.charger and new_target:
            return True
        return False

    def get_car_target_charge_option_percent(self):
        return self.get_car_option_charge_from_value_percent(self.get_car_target_SOC())

    def get_car_target_SOC(self) -> int | float:
        if self._next_charge_target is None:
            self._next_charge_target = self.car_default_charge
        return self._next_charge_target

    def get_car_minimum_ok_SOC(self) -> int | float:
        return self.car_minimum_ok_charge

    def get_car_next_charge_values_options_energy(self):

        max_battery_energy = self.car_battery_capacity
        if max_battery_energy is None or max_battery_energy <= 0:
            max_battery_energy = CAR_DEFAULT_CAPACITY

        max_battery_energy = int(max_battery_energy)

        options = set()
        options.add(max_battery_energy)

        for v in range(0, max_battery_energy, 2000):
            options.add(v)

        options = list(options)
        options.sort()

        for i in range(len(options)):
            options[i] = self.get_car_option_charge_from_value_energy(options[i])

        return options

    def get_car_option_charge_from_value_energy(self, value: int | float):
        value = int(float(value)) // 1000  # kwh
        return f"{value}kWh"

    async def set_next_charge_target_energy(self, value: int | float | str) -> bool:

        if isinstance(value, str):
            try:
                value = value.strip("kWh")
                value = float(value) * 1000.0
            except Exception as e:
                _LOGGER.error(
                    f"Car {self.name} set_next_charge_target_energy: invalid value {value} {e}",
                    exc_info=True,
                    stack_info=True,
                )
                return False
        else:
            value = float(value) * 1000.0

        self._next_charge_target_energy = value

        if self.charger:
            return True
        return False

    def get_car_target_charge_energy(self) -> int | float:
        if self._next_charge_target_energy is None:
            if self.car_battery_capacity is not None and self.car_battery_capacity > 0:
                self._next_charge_target_energy = self.car_battery_capacity
            else:
                self._next_charge_target_energy = CAR_DEFAULT_CAPACITY
        return self._next_charge_target_energy

    def get_car_target_charge_option_energy(self):
        return self.get_car_option_charge_from_value_energy(self.get_car_target_charge_energy())

    @property
    def qs_bump_solar_charge_priority(self) -> bool:
        return self._qs_bump_solar_priority

    @qs_bump_solar_charge_priority.setter
    def qs_bump_solar_charge_priority(self, value: bool):
        if value is False:
            self._qs_bump_solar_priority = False
        else:
            # only one can have a bump — clear others via direct attribute, not property
            for c in self.home._cars:
                c._qs_bump_solar_priority = False
            self._qs_bump_solar_priority = True
        self.set_user_originated("bump_solar", self._qs_bump_solar_priority)

    def get_charger_options(self) -> list[str]:

        time = datetime.now(pytz.UTC)

        options = []
        for charger in self.home._chargers:
            if self.car_hard_wired_charger:
                if charger is not self.car_hard_wired_charger:
                    continue
            if charger.is_optimistic_plugged(time):
                options.append(charger.name)

        options.append(FORCE_CAR_NO_CHARGER_CONNECTED)
        return options

    def get_current_selected_charger_option(self) -> str | None:
        if self.get_user_originated(USER_ORIGINATED_CHARGER_NAME) is not None:
            return self.get_user_originated(USER_ORIGINATED_CHARGER_NAME)

        if self.charger is None:
            return None
        else:
            return self.charger.name

    async def user_set_selected_charger_by_name(self, charger_name: str | None):

        # if the car is already attached to a charger, we detach it
        orig_charger = self.charger

        if self.charger is not None and self.charger.name != charger_name:
            self.detach_charger()

        if charger_name == FORCE_CAR_NO_CHARGER_CONNECTED:
            self.set_user_originated(USER_ORIGINATED_CHARGER_NAME, FORCE_CAR_NO_CHARGER_CONNECTED)
            self.clear_inferred_flags()
            self.reset_car_api_stale_detection()
        elif charger_name is not None:
            charger = None
            for c in self.home._chargers:
                if c.name == charger_name:
                    charger = c
                    break
            if charger is not None:
                self.set_user_originated(USER_ORIGINATED_CHARGER_NAME, charger_name)
                # Feature B: check for contradiction on manual assignment
                self.check_charger_assignment_contradiction(charger_name, datetime.now(tz=pytz.UTC), manual=True)
                await charger.user_set_selected_car_by_name(car_name=self.name)
            else:
                self.clear_user_originated(USER_ORIGINATED_CHARGER_NAME)
        else:
            self.clear_user_originated(USER_ORIGINATED_CHARGER_NAME)
            self.clear_inferred_flags()

        if orig_charger is not None:
            await orig_charger.update_charger_for_user_change()

    async def user_clean_and_reset(self):
        charger = self.charger
        await super().user_clean_and_reset()

        self.current_forecasted_person = None
        self.reset_soc_estimate()
        self.reset_car_api_stale_detection()

        self.reset(keep_commands=True)  # will detach the car

        self._reset_charge_targets()
        await self.setup_car_charge_target_if_needed()

        if charger is not None:
            await charger.user_clean_and_reset()

        if self.home:
            await self.home.compute_and_set_best_persons_cars_allocations(force_update=True, do_notify=True)

    async def user_clean_constraints(self):
        charger = self.charger
        self.set_user_originated("charge_time", CHARGE_TIME_CONSTRAINTS_CLEARED)
        await super().user_clean_constraints()
        self._reset_charge_targets()
        await self.setup_car_charge_target_if_needed()
        if charger is not None:
            await charger.user_clean_constraints()

    @property
    def current_constraint_current_energy(self):
        if self.charger is None:
            return None
        return self.charger.current_constraint_current_energy
