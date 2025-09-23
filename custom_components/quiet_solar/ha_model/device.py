import logging
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from datetime import time as dt_time
from operator import itemgetter
from typing import Mapping, Any, Callable

import pytz
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE, UnitOfPower, ATTR_UNIT_OF_MEASUREMENT, Platform, \
    ATTR_ENTITY_ID, UnitOfElectricCurrent
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State, callback, Event, EventStateChangedData
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util.unit_conversion import PowerConverter
from homeassistant.components import calendar

from ..const import CONF_ACCURATE_POWER_SENSOR, DOMAIN, DATA_HANDLER, COMMAND_BASED_POWER_SENSOR, \
    CONF_CALENDAR, SENSOR_CONSTRAINT_SENSOR, CONF_MOBILE_APP, CONF_MOBILE_APP_NOTHING, CONF_MOBILE_APP_URL, \
    FLOATING_PERIOD_S, DEVICE_CHANGE_CONSTRAINT, DEVICE_CHANGE_CONSTRAINT_COMPLETED, CONF_PHASE_1_AMPS_SENSOR, \
    CONF_PHASE_2_AMPS_SENSOR, CONF_PHASE_3_AMPS_SENSOR, CONF_TYPE_NAME_HADeviceMixin
from ..home_model.home_utils import get_average_time_series
from ..home_model.load import AbstractLoad, AbstractDevice

import numpy as np


from datetime import datetime, timedelta
from homeassistant.core import HomeAssistant
from homeassistant.components.calendar.const import DATA_COMPONENT
from homeassistant.components.calendar import CalendarEntity
from homeassistant.components.calendar.const import CalendarEntityFeature



UNAVAILABLE_STATE_VALUES = [STATE_UNKNOWN, STATE_UNAVAILABLE]

_LOGGER = logging.getLogger(__name__)

CALENDAR_MANAGED_STRING_ID = "[Quiet Solar]"
CALENDAR_MANAGED_STRING = f"{CALENDAR_MANAGED_STRING_ID} Managed by Quiet Solar (do not edit)"


def compute_energy_Wh_rieman_sum(
        power_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
        conservative: bool = False,
        clip_to_zero_under_power: None | float = None) -> tuple[float, float]:
    """Compute energy from power with a rieman sum."""

    energy = 0
    duration_h = 0
    if power_data and len(power_data) > 1:

        # compute a rieman sum, as best as possible , trapezoidal, taking pessimistic asumption
        # as we don't want to artificially go up the previous one
        # (except in rare exceptions like reset, 0 , etc)
        prev_value = power_data[0][1]
        if clip_to_zero_under_power is not None and prev_value is not None and prev_value < clip_to_zero_under_power:
            prev_value = 0.0

        for i in range(len(power_data) - 1):

            next_value = power_data[i+1][1]
            if clip_to_zero_under_power is not None and next_value is not None and next_value < clip_to_zero_under_power:
                next_value = 0.0

            dt_h = float((power_data[i + 1][0] - power_data[i][0]).total_seconds()) / 3600.0
            duration_h += dt_h

            if conservative:
                d_p_w = 0
            else:
                d_p_w = abs(float(next_value - prev_value))

            d_nrj_wh = dt_h * (
                    min(next_value, prev_value) + 0.5 * d_p_w
            )

            energy += d_nrj_wh

            prev_value = next_value

    return energy, duration_h


def convert_power_to_w(value: float, attributes: dict | None = None) -> (float, dict):
    default_unit: str = UnitOfPower.WATT
    new_attr = attributes
    if attributes is None:
        sensor_unit = default_unit
    else:
        sensor_unit = attributes.get(ATTR_UNIT_OF_MEASUREMENT, default_unit)

    if sensor_unit in UnitOfPower and sensor_unit != default_unit:
        value = PowerConverter.convert(value=value, from_unit=sensor_unit, to_unit=default_unit)
        new_attr = {}
        if attributes is not None:
            new_attr = dict(attributes)

        new_attr[ATTR_UNIT_OF_MEASUREMENT] = default_unit

    return value, new_attr


def convert_current_to_amps(value: float, attributes: dict | None = None) -> (float, dict):
    default_unit: str = UnitOfElectricCurrent.AMPERE
    new_attr = attributes
    if attributes is None:
        sensor_unit = default_unit
    else:
        sensor_unit = attributes.get(ATTR_UNIT_OF_MEASUREMENT, default_unit)

    if sensor_unit in UnitOfElectricCurrent and sensor_unit != default_unit:
        value = PowerConverter.convert(value=value, from_unit=sensor_unit, to_unit=default_unit)
        new_attr = {}
        if attributes is not None:
            new_attr = dict(attributes)

        new_attr[ATTR_UNIT_OF_MEASUREMENT] = default_unit

    return value, new_attr

def get_average_power_energy_based(
        power_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]]):
    if len(power_data) == 0:
        return 0
    elif len(power_data) == 1:
        val = power_data[0][1]
        if val is None:
            return 0.0
    else:
        power_data = [(entry[0], entry[1]) for entry in power_data if entry[1] is not None]
        if power_data:
            nrj, dh = compute_energy_Wh_rieman_sum(power_data)
            val = nrj / dh
        else:
            return 0.0

    # do not change units
    return val


def get_median_sensor(sensor_data: list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]] | list[tuple[datetime | None, str | float | None]],
                      last_timing: datetime | None = None):
    if len(sensor_data) == 0:
        return 0
    elif len(sensor_data) == 1:
        val = sensor_data[0][1]
        if val is None:
            val = 0.0
    else:
        vals : list[float] = []
        add_last = 0
        if last_timing is not None:
            add_last = 1
        for i in range(1, len(sensor_data) + add_last):
            value = sensor_data[i - 1][1]
            if value is None:
                continue
            if i == len(sensor_data):
                dt = (last_timing - sensor_data[i - 1][0]).total_seconds()
            else:
                dt = (sensor_data[i][0] - sensor_data[i-1][0]).total_seconds()

            if dt == 0:
                dt = 1

            num_add = int(dt) + 1
            vals.extend([float(value)]*num_add)

        if vals:
            val = np.median(vals)
        else:
            val = 0.0
    # do not change units
    return val

MAX_STATE_HISTORY_S = 7200

class HADeviceMixin:

    conf_type_name = CONF_TYPE_NAME_HADeviceMixin

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, **kwargs):

        self.calendar = kwargs.pop(CONF_CALENDAR, None)
        self.accurate_power_sensor = kwargs.pop(CONF_ACCURATE_POWER_SENSOR, None)

        self.phase_1_amps_sensor = kwargs.pop(CONF_PHASE_1_AMPS_SENSOR, None)
        self.phase_2_amps_sensor = kwargs.pop(CONF_PHASE_2_AMPS_SENSOR, None)
        self.phase_3_amps_sensor = kwargs.pop(CONF_PHASE_3_AMPS_SENSOR, None)


        self.mobile_app = kwargs.pop(CONF_MOBILE_APP, CONF_MOBILE_APP_NOTHING)


        if self.mobile_app is None or self.mobile_app == CONF_MOBILE_APP_NOTHING:
            self.mobile_app = None

        self.mobile_app_url = kwargs.pop(CONF_MOBILE_APP_URL, None)
        if self.mobile_app_url is None or len(self.mobile_app_url) == 0:
            self.mobile_app_url = None
        elif self.mobile_app_url == "/":
            self.mobile_app_url = None
        elif self.mobile_app_url[0] != '/':
            self.mobile_app_url = f"/{self.mobile_app_url}"

        self.secondary_power_sensor = None
        self.best_power_value = None

        self.command_based_power_sensor = COMMAND_BASED_POWER_SENSOR

        super().__init__(**kwargs)
        self.hass = hass
        if hass:
            self.data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
        else:
            self.data_handler = None

        self.config_entry = config_entry

        self._entity_probed_state_is_numerical: dict[str, bool] = {}
        self._entity_probed_state_attached_unfiltered: dict[str, bool] = {}
        self._entity_probed_state_conversion_fn: dict[str, Callable[[float, dict], float] | None] = {}
        self._entity_probed_state_transform_fn: dict[str, Callable[[float, dict], float] | None] = {}
        self._entity_probed_state_non_ha_entity_get_state: dict[str, Callable[[str, datetime | None], tuple[
                                                                                                          datetime | None, float | str | None, dict | None] | None] | None] = {}
        self._entity_probed_state: dict[
            str, list[tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]] = {}
        self._entity_probed_last_valid_state: dict[
            str, tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict] | None] = {}

        self._entity_probed_state_invalid_values: dict[
            str,set[str]] = {}

        self._entity_probed_auto = set()
        self._entity_on_change = set()

        self._exposed_entities = set()
        self.ha_entities = {}

        self.attach_power_to_probe(self.accurate_power_sensor)

        self.attach_amps_to_probe(self.phase_1_amps_sensor)
        self.attach_amps_to_probe(self.phase_2_amps_sensor)
        self.attach_amps_to_probe(self.phase_3_amps_sensor)

        self.attach_power_to_probe(self.command_based_power_sensor,
                                   non_ha_entity_get_state=self.command_power_state_getter)

        self._unsub = None

        self._computed_dashboard_section = None


    async def user_clean_and_reset(self):
        time = datetime.now(tz=pytz.UTC)
        await self.clean_next_qs_scheduled_event(time)

    def get_next_time_from_hours(self, local_hours:dt_time, time_utc_now:datetime | None = None, output_in_utc=True) -> datetime | None:

        if time_utc_now is None:
            time_utc_now = datetime.now(tz=pytz.UTC)

        dt_now = time_utc_now.replace(tzinfo=pytz.UTC).astimezone(tz=None)

        next_time = datetime(year=dt_now.year,
                             month=dt_now.month,
                             day=dt_now.day,
                             hour=local_hours.hour,
                             minute=local_hours.minute,
                             second=local_hours.second)
        next_time = next_time.astimezone(tz=None)
        if next_time < dt_now:
            next_time = next_time + timedelta(days=1)

        if output_in_utc:
            next_time = next_time.replace(tzinfo=None).astimezone(tz=pytz.UTC)

        return next_time


    async def clean_next_qs_scheduled_event(self, time:datetime, start_time_to_check:datetime|None = None, end_time_to_check:datetime|None = None,) -> bool:

        # by construction they can't be in the past, nor more than 24h in the future
        if self.calendar is None:
            return False

        component = self.hass.data[DATA_COMPONENT]
        entity = component.get_entity(self.calendar)

        if not isinstance(entity, CalendarEntity):
            return False

        if not (entity.supported_features & CalendarEntityFeature.DELETE_EVENT):
            return False

        try:
            events = await entity.async_get_events(self.hass, time - timedelta(seconds=10), time + timedelta(days=1, seconds=10))

            remove = []
            for e in events:
                if CALENDAR_MANAGED_STRING_ID in e.description:
                    # remove it
                    remove.append(e)

            kept_one = False
            for e in remove:

                if start_time_to_check is not None and end_time_to_check is not None and e.start == start_time_to_check and e.end == end_time_to_check:
                    # do not remove it if it is the one we want to keep
                    kept_one = True
                    continue

                await entity.async_delete_event(e.uid)

            return kept_one

        except Exception as err:
            _LOGGER.error(f"Error working on calendar in clean_next_qs_scheduled_event {self.calendar} {err}", exc_info=err)
            return False


    async def set_next_scheduled_event(self, time:datetime, start_time:datetime, end_time:datetime, description:str):
        if self.calendar is None:
            return

        # first clean old ones if needed for teh next day ... but keep if it was already created
        found = await self.clean_next_qs_scheduled_event(time, start_time_to_check=start_time, end_time_to_check=end_time)

        if found:
            return

        data = {ATTR_ENTITY_ID: self.calendar}
        service = calendar.CREATE_EVENT_SERVICE
        data[calendar.EVENT_START_DATETIME] = start_time
        data[calendar.EVENT_END_DATETIME] = end_time
        data[calendar.EVENT_SUMMARY] = description
        data[calendar.EVENT_DESCRIPTION] = CALENDAR_MANAGED_STRING
        domain = calendar.DOMAIN

        try:
            await self.hass.services.async_call(
                domain, service, data, blocking=True)
        except Exception as err:
            _LOGGER.error(f"Error setting calendar {self.calendar} {err}", exc_info=err)



    async def get_next_scheduled_event(self, time:datetime, after_end_time:bool=False) -> tuple[datetime|None,datetime|None]:
        if self.calendar is None:
            return None, None

        state = self.hass.states.get(self.calendar)
        state_attr = {}
        if state is None or state.state in UNAVAILABLE_STATE_VALUES:
            state = None

        if state is not None:
            state_attr = state.attributes

        start_time: str | None | datetime = state_attr.get("start_time", None)
        end_time: str | None | datetime = state_attr.get("end_time", None)

        if start_time is not None:
            start_time = datetime.fromisoformat(start_time)
            start_time = start_time.replace(tzinfo=None).astimezone(tz=pytz.UTC)
        if end_time is not None:
            end_time = datetime.fromisoformat(end_time)
            end_time = end_time.replace(tzinfo=None).astimezone(tz=pytz.UTC)


        if after_end_time and start_time is not None and end_time is not None and time >= start_time:
            # time is "during the current event" but if we want the next "start" ask the calendar
            data = {ATTR_ENTITY_ID: self.calendar}
            service = calendar.SERVICE_GET_EVENTS
            data[calendar.EVENT_START_DATETIME] = end_time - timedelta(seconds=1) # -1 to get the "current one" will filter it below in the loop if needed
            data[calendar.EVENT_DURATION] = timedelta(seconds=FLOATING_PERIOD_S+1)
            domain = calendar.DOMAIN

            start_time = None
            end_time = None
            try:
                resp = await self.hass.services.async_call(
                    domain, service, data, blocking=True, return_response=True
                )
                for cals in resp:
                    events = resp[cals].get("events", [])
                    for event in events:
                        # events are sorted by time ... pick the first ok one
                        st_time = datetime.fromisoformat(event["start"])
                        st_time = st_time.astimezone(tz=pytz.UTC)
                        if st_time <= time:
                            continue
                        start_time = st_time
                        end_time = datetime.fromisoformat(event["end"])
                        end_time = end_time.astimezone(tz=pytz.UTC)
                        break
            except Exception as err:
                _LOGGER.error(f"Error reading calendar in get_next_scheduled_event {self.calendar} {err}", exc_info=err)


        return start_time, end_time


    def attach_exposed_has_entity(self, ha_object):
        self._exposed_entities.add(ha_object)
        self.ha_entities[ha_object.entity_description.key] = ha_object

    def command_power_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if not isinstance(self, AbstractLoad):
            return None

        command_value = None

        do_return_None = False
        if self.is_load_command_set(time) is False:
            do_return_None = True
        else:
            if self.current_command is None or self.current_command.is_off_or_idle():
                command_value = 0.0
            else:
                command_value = self.current_command.power_consign
                if command_value is None or command_value == 0:
                    command_value = self.power_use
        value = None
        if self.accurate_power_sensor is not None:
            hist_f = self._entity_probed_state.get(self.accurate_power_sensor, [])
            if hist_f:
                value = hist_f[-1][1]
        elif self.secondary_power_sensor is not None:
            hist_f = self._entity_probed_state.get(self.secondary_power_sensor, [])
            if hist_f:
                value = hist_f[-1][1]
        else:
            value = command_value

        self.best_power_value = value

        if do_return_None:
            return None
        return (time, str(command_value), {})


    def get_virtual_load_HA_power_entity_name(self) -> str | None:
        if not isinstance(self, AbstractLoad):
            return None
        return f"{self.device_id}_power"

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR

    async def on_device_state_change(self, time: datetime, device_change_type:str):
        await self.on_device_state_change_helper(time, device_change_type)

    async def on_device_state_change_helper(self, time: datetime, device_change_type: str, **kwargs):

        load_name = kwargs.get("load_name", self.name)
        mobile_app = kwargs.get("mobile_app", self.mobile_app)
        mobile_app_url = kwargs.get("mobile_app_url", self.mobile_app_url)
        title = kwargs.get("title", f"What will happen for {load_name}?")
        message = kwargs.get("message", None)


        if message is None:
            if device_change_type == DEVICE_CHANGE_CONSTRAINT:
                if isinstance(self, AbstractLoad):
                    message = self.get_active_readable_name(time, filter_for_human_notification=True)
                else:
                    message = "WRONG STATE"
            elif device_change_type == DEVICE_CHANGE_CONSTRAINT_COMPLETED:
                if isinstance(self, AbstractLoad):
                    if self._last_completed_constraint is not None:
                        message = "COMPLETED: " + self._last_completed_constraint.get_readable_name_for_load()


        _LOGGER.info(f"Sending notification for load {load_name} app: {mobile_app} with: {message}")

        if mobile_app is not None and message is not None:

            data={
                "title": title,
                "message": message,
            }
            if mobile_app_url is not None:
                data["data"] = {}
                data["data"]["url"] = mobile_app_url
                data["data"]["clickAction"] = mobile_app_url

            _LOGGER.info(f"Full Sending notification for load {load_name} app: {mobile_app} with: {data}")

            await self.hass.services.async_call(
                domain=Platform.NOTIFY,
                service=mobile_app,
                service_data=data,
            )

    def get_best_power_HA_entity(self):
        if self.accurate_power_sensor is not None:
            return self.accurate_power_sensor
        elif self.secondary_power_sensor is not None:
            return self.secondary_power_sensor
        else:
            return None

    def _get_power_measure(self, fall_back_on_command: bool = True) -> str | None:
        best = self.get_best_power_HA_entity()
        if best is None and fall_back_on_command:
            best = self.command_based_power_sensor
        return best


    def get_sensor_latest_possible_valid_value_and_attr(self, entity_id, time = None) -> tuple[str | float | None, Mapping[str, Any] | None | dict]:

        if entity_id is None:
            return None, None

        last_valid = self._entity_probed_last_valid_state[entity_id]
        if last_valid is None:
            return None, None

        if time is None:
            time = datetime.now(tz=pytz.UTC)

        if time >= last_valid[0]:
            return last_valid[1], last_valid[2]

        return None, None

    def is_sensor_growing(self,
                          entity_id,
                          num_seconds: float | None = None,
                          time = None) -> bool | None:

        if entity_id is None:
            return None

        entity_id_values = self.get_state_history_data(entity_id, num_seconds, time)

        if not entity_id_values or len(entity_id_values) < 2:
            return None

        vals = []
        min_v = None
        max_v = None
        for v in entity_id_values:
            if v[1] is None:
                continue
            vals.append(v[1])
            if min_v is None or v[1] < min_v:
                min_v = v[1]
            if max_v is None or v[1] > max_v:
                max_v = v[1]


        return vals[-1] >= max_v and max_v > min_v and len(vals) > 1


    def get_sensor_latest_possible_valid_value(self,
                                               entity_id,
                                               tolerance_seconds: float | None = None,
                                               time = None) -> str | float | None:
        if entity_id is None:
            return None

        last_valid = self._entity_probed_last_valid_state[entity_id]
        if last_valid is None:
            return None

        if time is None:
            time = datetime.now(tz=pytz.UTC)

        if time >= last_valid[0]:

            if tolerance_seconds is None or tolerance_seconds == 0:
                return last_valid[1]

            hist_f = self._entity_probed_state.get(entity_id, [])

            if not hist_f:
                return last_valid[1]

            if hist_f[-1][0] == last_valid[0] or hist_f[-1][1] is not None:
                # HA update only changed sensor ... if it is the last valid, whatever the time, it is valid
                return last_valid[1]

            if (time - last_valid[0]).total_seconds() > tolerance_seconds:
                return None

            return last_valid[1]
        else:
            vals = self.get_state_history_data(entity_id, tolerance_seconds, time)
            if not vals:
                return None
            return vals[-1][1]

    def get_device_power_latest_possible_valid_value(self,
                                                     tolerance_seconds: float | None,
                                                     time:datetime,
                                                     ignore_auto_load:bool= False) -> float:

        if ignore_auto_load and isinstance(self, AbstractLoad) and self.load_is_auto_to_be_boosted:
            return 0.0

        p = self.get_sensor_latest_possible_valid_value(self._get_power_measure(), tolerance_seconds, time)

        if p is None:
            return 0.0
        return p


    def get_device_amps_consumption(self, tolerance_seconds: float | None, time:datetime) -> list[float|int] | None:

        # first check if we do have an amp sensor for the phases
        p = self.get_device_power_latest_possible_valid_value(tolerance_seconds=tolerance_seconds, time=time)
        pM = None

        is_3p = False
        if isinstance(self, AbstractDevice):
            is_3p = self.current_3p

            if p is not None:
                pM =  self.get_phase_amps_from_power(power=p, is_3p=is_3p)

        return self._get_device_amps_consumption(pM, tolerance_seconds, time, multiplier=1, is_3p=is_3p)

    def _get_device_amps_consumption(self, pM:list[float|int]|None, tolerance_seconds: float | None, time: datetime, multiplier=1, is_3p=False) -> list[float|int] | None:

        ret: list[float | int | None]  = [None, None, None]

        good_mono_p = None
        mono_phase = 0
        if isinstance(self, AbstractDevice):
            mono_phase = self.mono_phase_index

        for i, sensor in enumerate([self.phase_1_amps_sensor, self.phase_2_amps_sensor, self.phase_3_amps_sensor]):
            if sensor is None:
                continue
            p_val = self.get_sensor_latest_possible_valid_value(sensor, tolerance_seconds, time)
            if p_val is not None:
                p_val = p_val*multiplier
                if mono_phase == i:
                    good_mono_p = p_val
                elif good_mono_p is None:
                    # in case of mismatch between sensors and the selected phase
                    good_mono_p = p_val
                ret[i] = p_val

        if is_3p:
            if pM is not None and None in ret:
                ok_phases = [a for a in ret if a is not None]
                p_comp = (sum(pM) - sum(ok_phases)) / (3 - len(ok_phases))
                for i, p_val in enumerate(ret):
                    if p_val is None:
                        ret[i] = p_comp
        else:
            ret = [0.0, 0.0, 0.0]
            if good_mono_p is not None:
                ret[mono_phase] = good_mono_p
            elif pM is not None:
                ret[mono_phase] = sum(pM)
            else:
                ret = [None, None, None]

        if None in ret:
            return None
        else:
            return ret


    def get_median_sensor(self, entity_id: str | None, num_seconds: float | None, time: datetime) -> float | None:
        if entity_id is None:
            return None
        entity_id_values = self.get_state_history_data(entity_id, num_seconds, time)
        if not entity_id_values:
            return None
        return get_median_sensor(entity_id_values, time)

    def get_average_sensor(self, entity_id: str | None, num_seconds: float | None, time: datetime) -> float | None:
        if entity_id is None:
            return None
        entity_id_values = self.get_state_history_data(entity_id, num_seconds, time)
        if not entity_id_values:
            return None
        return get_average_time_series(entity_id_values, last_timing=time)

    def get_median_power(self, num_seconds: float | None, time, use_fallback_command=True) -> float | None:
        return self.get_median_sensor(self._get_power_measure(fall_back_on_command=use_fallback_command), num_seconds, time)



    def get_average_power(self, num_seconds: float | None, time, use_fallback_command=True) -> float | None:
        return self.get_average_sensor(self._get_power_measure(fall_back_on_command=use_fallback_command), num_seconds, time)



    def get_device_power_values(self, duration_before_s: float, time: datetime, use_fallback_command=True) -> list[
        tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self._get_power_measure(fall_back_on_command=use_fallback_command), duration_before_s, time)


    def get_device_real_energy(self, start_time: datetime, end_time:datetime, clip_to_zero_under_power: float | None = None) -> float | None:
        duration_before_s = (end_time - start_time).total_seconds()
        val = self.get_state_history_data(self._get_power_measure(fall_back_on_command=False), duration_before_s, end_time)
        if not val:
            return None
        return compute_energy_Wh_rieman_sum(val, clip_to_zero_under_power=clip_to_zero_under_power)[0]

    def get_device_command_energy(self, start_time: datetime, end_time:datetime) -> float | None:
        duration_before_s = (end_time - start_time).total_seconds()
        val = self.get_state_history_data(self.command_based_power_sensor, duration_before_s, end_time)
        if not val:
            return None
        return compute_energy_Wh_rieman_sum(val)[0]

    def get_last_state_value_duration(self,
                                      entity_id: str,
                                      states_vals: list[str]| set[str],
                                      num_seconds_before: float | None,
                                      time: datetime,
                                      invert_val_probe=False,
                                      allowed_max_holes_s: float | None = None,
                                      count_only_duration=False) -> (float | None, list[tuple[float, datetime, str]]):

        states_vals = set(states_vals)
        if entity_id in self._entity_probed_state:
            # get latest values
            self.add_to_history(entity_id, time)

        from_idx = -1
        if num_seconds_before is None:
            values = self._entity_probed_state.get(entity_id, [])

            if values:
                if time < values[0][0]:
                    from_idx = -1
                elif time == values[0][0]:
                    from_idx = 0
                else:

                    if time >= values[-1][0]:
                        from_idx = len(values) - 1
                    else:
                        from_idx = bisect_right(values, time, key=itemgetter(0))

        else:
            values = self.get_state_history_data(entity_id, num_seconds_before=num_seconds_before, to_ts=time, keep_invalid_states=True)
            if values:
                from_idx = len(values) - 1

        if not values or from_idx < 0:
            return None, []

        if allowed_max_holes_s is None:
            if num_seconds_before is None:
                allowed_max_holes_s = 30.0
            else:
                allowed_max_holes_s = num_seconds_before / 2.0


        state_status_duration = 0

        # check the last states
        current_hole = 0.0

        all_invalid = True

        had_one_good = False

        ok_ranges = []

        for i in range(from_idx, -1, -1):

            ts, state, attr = values[i]

            if i < from_idx:
                next_ts = values[i + 1][0]
            else:
                next_ts = time

            delta_t = (next_ts - ts).total_seconds()

            if state is not None:
                all_invalid = False
                if invert_val_probe:
                    val_prob_ok = state not in states_vals
                else:
                    val_prob_ok = state in states_vals

                if val_prob_ok:
                    ok_ranges.append((delta_t, state, ts))
                    state_status_duration += delta_t
                    # if there was some unavailable between the last good state and this one, we reset the hole
                    # and we add the invalid time as "good" time, only if we are between 2 "good" states
                    if count_only_duration is False and had_one_good:
                        state_status_duration += current_hole

                    current_hole = 0.0

                    had_one_good = True
                else:
                    # it is a bad state: whatever don't count passed it
                    # if we never had a good state, state_status_duration will be 0, this is what we want
                    if count_only_duration is False:
                        break
            else:
                current_hole += delta_t
                if count_only_duration is False and current_hole > allowed_max_holes_s:
                    break

        if all_invalid:
            return None, ok_ranges

        return state_status_duration, ok_ranges

    def register_all_on_change_states(self):

        if len(self._entity_on_change) > 0:
            @callback
            def async_threshold_sensor_state_listener(
                    event: Event[EventStateChangedData],
            ) -> None:
                """Handle sensor state changes."""
                new_state = event.data["new_state"]
                time = new_state.last_updated
                self.add_to_history(new_state.entity_id, time, state=new_state)

            self._unsub = async_track_state_change_event(
                self.hass,
                list(self._entity_on_change),
                async_threshold_sensor_state_listener,
            )

    async def update_states(self, time: datetime):

        for entity_id in self._entity_probed_auto:
            self.add_to_history(entity_id, time)

        for ha_object in self._exposed_entities:
            ha_object.async_update_callback(time)

    def attach_power_to_probe(self, entity_id: str | None, transform_fn: Callable[[float, dict], tuple[float, dict]] | None = None,
                              non_ha_entity_get_state: Callable[[str, datetime | None], tuple[
                                                                                            float | str | None, datetime | None, dict | None] | None] = None):
        self.attach_ha_state_to_probe(entity_id=entity_id, is_numerical=True, transform_fn=transform_fn,
                                      conversion_fn=convert_power_to_w, update_on_change_only=True,
                                      non_ha_entity_get_state=non_ha_entity_get_state)

    def attach_amps_to_probe(self, entity_id: str | None, transform_fn: Callable[[float, dict], tuple[float, dict]] | None = None,
                              non_ha_entity_get_state: Callable[[str, datetime | None], tuple[
                                                                                            float | str | None, datetime | None, dict | None] | None] = None):
        self.attach_ha_state_to_probe(entity_id=entity_id, is_numerical=True, transform_fn=transform_fn,
                                      conversion_fn=convert_current_to_amps, update_on_change_only=True,
                                      non_ha_entity_get_state=non_ha_entity_get_state)

    def get_unfiltered_entity_name(self, entity_id: str | None, strict : bool =False) -> str| None:
        if entity_id is None:
            return None

        if self._entity_probed_state_attached_unfiltered.get(entity_id) is False:
            if strict:
                return None
            else:
                return entity_id

        return f"{entity_id}_no_filters"

    def attach_ha_state_to_probe(self, entity_id: str | None, is_numerical: bool = False,
                                 transform_fn: Callable[[float, dict], tuple[float, dict]] | None = None,
                                 conversion_fn: Callable[[float, dict], tuple[float, dict]] | None = None,
                                 update_on_change_only: bool = True,
                                 non_ha_entity_get_state: str | None | Callable[[str, datetime | None], tuple[
                                                                                               float | str | None, datetime | None, dict | None] | None] = None,
                                 state_invalid_values: set[str]|list[str]|None = None,
                                 attach_unfiltered:bool = False):
        if entity_id is None:
            return

        self._entity_probed_state[entity_id] = []
        self._entity_probed_last_valid_state[entity_id] = None
        self._entity_probed_state_is_numerical[entity_id] = is_numerical
        self._entity_probed_state_transform_fn[entity_id] = transform_fn
        self._entity_probed_state_conversion_fn[entity_id] = conversion_fn
        self._entity_probed_state_non_ha_entity_get_state[entity_id] = non_ha_entity_get_state
        self._entity_probed_state_invalid_values[entity_id] = set()
        self._entity_probed_state_attached_unfiltered[entity_id] = attach_unfiltered

        self._entity_probed_state_invalid_values[entity_id].update(UNAVAILABLE_STATE_VALUES)
        if state_invalid_values:
            self._entity_probed_state_invalid_values[entity_id].update(state_invalid_values)

        if non_ha_entity_get_state is not None:
            self._entity_probed_auto.add(entity_id)
        else:
            if update_on_change_only:
                self._entity_on_change.add(entity_id)
            else:
                self._entity_on_change.add(entity_id)
                self._entity_probed_auto.add(entity_id)


        if attach_unfiltered:
            unfiltered_internal = self.get_unfiltered_entity_name(entity_id, strict=True)
            if unfiltered_internal is not None:
                self.attach_ha_state_to_probe(entity_id=unfiltered_internal,
                                              is_numerical=is_numerical,
                                              transform_fn=transform_fn,
                                              conversion_fn=conversion_fn,
                                              update_on_change_only=update_on_change_only,
                                              non_ha_entity_get_state="FAKE_GETTER",
                                              state_invalid_values=None,
                                              attach_unfiltered=False
                )

            # store a first version
        if self.hass:
            self.add_to_history(entity_id)

    def _clean_times_arrays(self, current_time: datetime, time_array: list[datetime], value_arrays: list[list]) -> list[
        datetime]:
        if len(time_array) == 0:
            return time_array

        while len(time_array) > 0 and (current_time - time_array[0]).total_seconds() > MAX_STATE_HISTORY_S:
            time_array.pop(0)
            for v in value_arrays:
                v.pop(0)

        return time_array

    def add_to_history(self, entity_id: str, time: datetime = None, state: State = None):

        state_getter = self._entity_probed_state_non_ha_entity_get_state[entity_id]

        if isinstance(state_getter, str) and state_getter == "FAKE_GETTER":
            return

        state_time: datetime | None = None

        if state is not None or state_getter is None:
            if state is None:
                state = self.hass.states.get(entity_id)
            state_attr = {}

            unfiltered_internal = self.get_unfiltered_entity_name(entity_id, strict=True)

            if unfiltered_internal is not None:
                to_fill = [(self._entity_probed_state_invalid_values[entity_id], entity_id) , ([] , unfiltered_internal) ]
            else:
                to_fill = [(self._entity_probed_state_invalid_values[entity_id], entity_id)]


            for tf in to_fill:
                invalid_vals, local_entity_id = tf

                if state is None or state.state in invalid_vals:
                    value = None
                    if state is not None:
                        state_time = state.last_updated
                else:
                    value = state.state
                    state_time = state.last_updated

                if state is not None:
                    state_attr = state.attributes

                self._add_state_history(local_entity_id, value, state_time, state, state_attr, time)
        else:
            fake_state = state_getter(entity_id, time)
            if fake_state is not None:
                state_time, value, state_attr = fake_state
            else:
                value = None
                state_attr = {}

            self._add_state_history(entity_id, value, state_time , state, state_attr, time)

    def _add_state_history(self, entity_id, value, state_time, state, state_attr, time):
        if state_attr is None:
            state_attr = {}

        if state_time is None:
            if time is None:
                state_time = datetime.now(tz=pytz.UTC)
            else:
                state_time = time

        if self._entity_probed_state_is_numerical[entity_id]:
            try:
                value = float(value)
            except ValueError:
                value = None
            except TypeError:
                value = None

        if value is not None:
            conversion_fn = self._entity_probed_state_conversion_fn[entity_id]
            if conversion_fn is not None:
                value, state_attr = conversion_fn(value, state_attr)
            transform_fn = self._entity_probed_state_transform_fn[entity_id]
            if transform_fn is not None:
                value, state_attr = transform_fn(value, state_attr)

        val_array = self._entity_probed_state[entity_id]

        if state is None:
            to_add = (state_time, value, None)
        else:
            if state_attr is not None and len(state_attr) == 0:
                state_attr = None
            to_add = (state_time, value, state_attr)

        if value is not None:
            prev_valid = self._entity_probed_last_valid_state[entity_id]
            if prev_valid is None or prev_valid[0] <= state_time:
                self._entity_probed_last_valid_state[entity_id] = to_add

        if not val_array:
            val_array.append(to_add)
        else:
            # small optim to add additional values at the end
            if state_time > val_array[-1][0]:
                val_array.append(to_add)
            elif state_time == val_array[-1][0]:
                val_array[-1] = to_add
            else:
                insert_idx = bisect_left(val_array, state_time, key=itemgetter(0))
                if insert_idx == len(val_array):
                    val_array.append(to_add)
                else:
                    if val_array[insert_idx][0] == state_time:
                        val_array[insert_idx] = to_add
                    else:
                        val_array.insert(insert_idx, to_add)

        while len(val_array) > 1 and (val_array[-1][0] - val_array[0][0]).total_seconds() > MAX_STATE_HISTORY_S:
            val_array.pop(0)

    def get_state_history_data(self, entity_id: str, num_seconds_before: float | None, to_ts: datetime,
                               keep_invalid_states=False) -> list[
        tuple[datetime | None, str | float | None, Mapping[str, Any] | None | dict]]:
        hist_f = self._entity_probed_state.get(entity_id, [])

        if not hist_f:
            return []

        if to_ts is None:
            to_ts = datetime.now(tz=pytz.UTC)

        # the last value is in fact still valid now (by construction : either it was through polling or through a state change)

        if num_seconds_before is None:
            from_ts = hist_f[0][0] - timedelta(seconds=10)
        else:
            from_ts = to_ts - timedelta(seconds=num_seconds_before)

        ret = None

        if from_ts >= hist_f[-1][0]:
            ret = hist_f[-1:]
        elif to_ts < hist_f[0][0]:
            ret = []
        elif to_ts == hist_f[0][0]:
            ret = hist_f[:1]
        else:

            in_s = bisect_left(hist_f, from_ts, key=itemgetter(0))

            # the state is "valid" for its whole duration so pick the one before
            if in_s > 0:
                if hist_f[in_s][0] > from_ts:
                    in_s -= 1

            if to_ts >= hist_f[-1][0]:
                out_s = len(hist_f)
            else:
                out_s = bisect_right(hist_f, to_ts, key=itemgetter(0))

            if in_s == out_s:
                if out_s == len(hist_f):
                    ret = hist_f[-1:]

            if ret is None:
                ret = hist_f[in_s:out_s]

        if ret is None:
            ret = []

        if keep_invalid_states is False and ret:
            return [v for v in ret if v[1] is not None]
        else:
            return ret

    def get_platforms(self) -> list[str]:
        """ returns associated platforms for this device """
        return [Platform.BUTTON, Platform.SENSOR]
