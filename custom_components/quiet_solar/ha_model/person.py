import logging
from bisect import bisect_left
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Any

import pytz
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE
from datetime import time as dt_time

from .car import QSCar
from ..const import CONF_TYPE_NAME_QSPerson, CONF_PERSON_PERSON_ENTITY, CONF_PERSON_AUTHORIZED_CARS, \
    CONF_PERSON_PREFERRED_CAR, CONF_PERSON_NOTIFICATION_TIME
from ..ha_model.device import HADeviceMixin, load_from_history
from ..home_model.constraints import get_readable_date_string
from ..home_model.load import AbstractDevice, get_value_from_time_series



_LOGGER = logging.getLogger(__name__)

MAX_HISTORICAL_DATA_DAYS = 14 # keep last 14 days of data
FORECAST_AUTO_REFRESH_RATE_S = 30*60 # 1 hours

class QSPerson(HADeviceMixin, AbstractDevice):
    """
    Class to track a person's location, home status, and daily driving distance.
    This is used to predict charging needs for the next day.
    """

    conf_type_name = CONF_TYPE_NAME_QSPerson

    def __init__(self, **kwargs):
        self.person_entity_id = kwargs.pop(CONF_PERSON_PERSON_ENTITY, None)

        # Extract authorized cars list, preferred car, and notification time from config
        self.authorized_cars = []
        self.preferred_car = None
        self.notification_time = None

        self.authorized_cars : list[str] = kwargs.pop(CONF_PERSON_AUTHORIZED_CARS, [])
        self.preferred_car  : str | None = kwargs.pop(CONF_PERSON_PREFERRED_CAR, None)
        self.notification_time : str | None = kwargs.pop(CONF_PERSON_NOTIFICATION_TIME, None)

        if self.preferred_car is not None and self.preferred_car not in self.authorized_cars:
            self.authorized_cars.append(self.preferred_car)

        super().__init__(**kwargs)

        self.historical_mileage_data : list[tuple[datetime,float,datetime, int]]= []
        self.serializable_historical_data : list[dict] = []
        self.predicted_mileage : float | None = None
        self.predicted_leave_time : datetime | None = None

        self.has_been_initialized = False
        self._last_request_prediction_time : datetime | None = None

        # Attach state probes
        self.attach_ha_state_to_probe(
            self.person_entity_id,
            is_numerical=False
        )

    def should_recompute_history(self, time:datetime) -> bool:

        if self.authorized_cars is None or len(self.authorized_cars) == 0:
            return False

        return not self.has_been_initialized


    def add_to_mileage_history(self, day:datetime, mileage:float, leave_time:datetime) -> None:

        if leave_time is None:
            _LOGGER.warning("add_to_mileage_history: Leave time not provided for person %s", self.name)
            return

        day = day.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        leave_time = leave_time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        week_day = day.weekday()

        if len(self.historical_mileage_data) == 0:
            self.historical_mileage_data = [(day, mileage, leave_time, week_day)]
        elif day > self.historical_mileage_data[-1][0]:
            self.historical_mileage_data.append( (day, mileage, leave_time, week_day) )
        else:
            insert_idx = bisect_left(self.historical_mileage_data, day, key=itemgetter(0))

            if insert_idx < len(self.historical_mileage_data) and insert_idx >= 0 and self.historical_mileage_data[insert_idx][0] == day:
                # update existing entry
                self.historical_mileage_data[insert_idx] = (day, mileage, leave_time, week_day)
                return

            self.historical_mileage_data.insert(insert_idx, (day, mileage, leave_time, week_day))

        while True:
            if len(self.historical_mileage_data) > MAX_HISTORICAL_DATA_DAYS: # keeps only 2 weeks of data
                self.historical_mileage_data.pop(0)
            else:
                break

        self._last_request_prediction_time = None

        serialized_hist_data = []
        for entry in self.historical_mileage_data:
            serialized_hist_data.append({
                "day": entry[0].isoformat(),
                "mileage": entry[1],
                "leave_time": entry[2].isoformat()
            })
        self.serializable_historical_data = serialized_hist_data

    def _get_best_week_day_guess(self, week_day:int) -> tuple[float | None, dt_time | None]:
        """Get the best guess for mileage and leave time for a given weekday."""
        best_mileage = None
        best_leave_time = None

        # Look for the last two entries for the given weekday
        num_entries = 0
        for entry in reversed(self.historical_mileage_data):
            if entry[3] == week_day:
                num_entries += 1
                if best_mileage is None or entry[1] > best_mileage:
                    best_mileage = entry[1]
                if best_leave_time is None or entry[2].time() < best_leave_time:
                    best_leave_time = entry[2].time()

                if num_entries >= 2:
                    break

        return best_mileage, best_leave_time

    def _compute_person_next_need(self, time:datetime) -> tuple[datetime | None, float | None]:

        local_time = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)

        today_week_day = local_time.weekday()
        tomorrow_week_day = (today_week_day +1) %7

        predicted_mileage_today, predicted_leave_time_today = self._get_best_week_day_guess(today_week_day)
        predicted_mileage_tomorrow, predicted_leave_time_tomorrow = self._get_best_week_day_guess(tomorrow_week_day)

        self.predicted_leave_time = None
        self.predicted_mileage = None

        if predicted_mileage_today is None and predicted_mileage_tomorrow is None:
            if len(self.historical_mileage_data) > 0:
                _LOGGER.warning(f"_compute_person_next_need: EMPTY PREDICTED_TIME for {self.name}")
        else:

            if predicted_leave_time_today is not None:
                today_leave_time = datetime.combine(local_time.date(), predicted_leave_time_today).astimezone(tz=None).replace(tzinfo=None).astimezone(tz=pytz.UTC)
            else:
                today_leave_time = None

            if today_leave_time is not None and today_leave_time > time:
                self.predicted_leave_time = today_leave_time
                self.predicted_mileage = predicted_mileage_today
            elif predicted_leave_time_tomorrow is not None:
                tomorrow_leave_time = datetime.combine(local_time.date() + timedelta(days=1),
                                                       predicted_leave_time_tomorrow).astimezone(tz=None).replace(
                    tzinfo=None).astimezone(tz=pytz.UTC)

                self.predicted_leave_time = tomorrow_leave_time
                self.predicted_mileage = predicted_mileage_tomorrow

            _LOGGER.info(f"_compute_person_next_need: for {self.name} mileage: {self.predicted_mileage}km at {self.predicted_leave_time.time().isoformat()}")

        return self.predicted_leave_time, self.predicted_mileage


    def get_authorized_cars(self) -> list[QSCar]:

        ret = []
        for car_name in self.authorized_cars:
            car = self.home.get_car_by_name(car_name)
            if car is not None and isinstance(car, QSCar):
                ret.append(car)
        return ret

    def get_preferred_car(self) -> QSCar | None:
        """Get the preferred car for this person, if set."""
        if self.preferred_car is None:
            return None

        car = self.home.get_car_by_name(self.preferred_car)
        if car is not None and isinstance(car, QSCar):
            return car

        return None

    def device_post_home_init(self, time: datetime):
        # should happen after the person config entry setup and all
        # so we should have read the person entity id from config and restaured its values from history

        super().device_post_home_init(time)

        sensor_entity = self.ha_entities.get("person_mileage_prediction")
        if sensor_entity is None:
            return

        sensor_entity_id = sensor_entity.entity_id

        current_state = self.hass.states.get(sensor_entity_id)

        if current_state is None or current_state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            current_attributes = None
        else:
            current_attributes = current_state.attributes

        if current_attributes is None:
            self.historical_mileage_data = []
            self.predicted_mileage = None
            self.predicted_leave_time = None
        else:
            self.historical_mileage_data = []

            entries = current_attributes.get("historical_data", [])
            for e in entries:
                try:
                    day_str = e.get("day", None)
                    mileage = e.get("mileage", None)
                    leave_time_str = e.get("leave_time", None)

                    if day_str is None or mileage is None or leave_time_str is None:
                        _LOGGER.warning(
                            f"device_post_home_init: QSPerson {self.name} error in saved data {e}")
                        continue

                    day = datetime.fromisoformat(day_str)
                    leave_time = datetime.fromisoformat(leave_time_str)

                    self.add_to_mileage_history( day, float(mileage), leave_time )
                except Exception as ex:
                    _LOGGER.warning(f"device_post_home_init: QSPerson {self.name} error parsing historical entry {e} : {ex}")


            str_hist_data = ""
            for day, mileage, leave_time, week_day in self.historical_mileage_data:
                if leave_time is None:
                    str_hist_data += f"[{day},{mileage},NONE,{week_day}] "
                else:
                    str_hist_data += f"[{day.date().isoformat()},{mileage},{leave_time.time().isoformat()},{week_day}] "

            local_time = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            today_week_day = local_time.weekday()
            tomorrow_week_day = (today_week_day + 1) % 7
            _LOGGER.info(f"device_post_home_init: QSPerson {self.name} td/tw {today_week_day}/{tomorrow_week_day}: hist_data {str_hist_data}")

            # recompute those, no need to read

            # self.predicted_mileage = current_attributes.get("predicted_mileage", None)
            # self.predicted_leave_time = current_attributes.get("predicted_leave_time", None)
            # if self.predicted_leave_time is not None:
            #    self.predicted_leave_time = datetime.fromisoformat(self.predicted_leave_time)

            self.update_person_forecast(time, force_update=True)

            self.has_been_initialized = current_attributes.get("has_been_initialized", False)
            if len(self.historical_mileage_data) != 0:
                self.has_been_initialized = True

            if self.has_been_initialized is False:
                _LOGGER.warning(f"device_post_home_init: QSPerson {self.name}: no initialization need compute")

    def update_person_forecast(self, time:datetime| None = None, force_update:bool=False) -> tuple[datetime | None, float | None]:
        if time is None:
            time = datetime.now(tz=pytz.UTC)
        if self._last_request_prediction_time is None or \
                (time - self._last_request_prediction_time).total_seconds() > FORECAST_AUTO_REFRESH_RATE_S or \
                (self.predicted_leave_time is not None and self.predicted_leave_time < time) or \
                force_update:
            self._compute_person_next_need(time)
            self._last_request_prediction_time = time

        return self.predicted_leave_time, self.predicted_mileage

    def get_forecast_readable_string(self) -> str:
        """Get a human-readable string of the person's forecast."""
        self.update_person_forecast()

        if self.predicted_mileage is None or self.predicted_leave_time is None:
            return "No forecast"
        else:
            return f"{int(self.predicted_mileage)}km {get_readable_date_string(self.predicted_leave_time, for_small_standalone=True)}"

    def get_person_mileage_serialized_prediction(self) -> tuple[Any | None, dict | None]:
        """Predict the person's mileage for the next day."""
        state_value = self.get_forecast_readable_string()

        serialized_leave_time = None
        if self.predicted_leave_time is not None:
            serialized_leave_time = self.predicted_leave_time.isoformat()

        return    state_value, {
            "historical_data": self.serializable_historical_data,
            "predicted_mileage": self.predicted_mileage,
            "predicted_leave_time": serialized_leave_time,
            "has_been_initialized" : self.has_been_initialized
        }

    def is_person_home(self, time: datetime, for_duration: float | None = None) -> bool | None:
        """Check if the person is home, optionally for a specific duration."""
        if self.person_entity_id is None:
            return None

        if for_duration is not None:
            contiguous_status = self.get_last_state_value_duration(
                self.person_entity_id,
                states_vals=["home"],
                num_seconds_before=8*for_duration,
                time=time
            )[0]

            if contiguous_status is not None:
                return contiguous_status >= for_duration and contiguous_status > 0
            else:
                return None
        else:
            latest_state = self.get_sensor_latest_possible_valid_value(
                entity_id=self.person_entity_id, time=time
            )

            if latest_state is None:
                return None

            return latest_state.lower() == "home"

    def get_platforms(self):
        """Return the platforms that this device supports."""
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([Platform.SENSOR])
        return list(parent)

    @property
    def dashboard_sort_string_in_type(self) -> str:
        """Sort string for dashboard display."""
        return "AAA"

    def reset(self):
        """Reset the device state."""
        super().reset()



