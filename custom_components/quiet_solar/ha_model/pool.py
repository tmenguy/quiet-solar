from datetime import datetime, date, timedelta

import pytz

from ..const import POOL_TEMP_STEPS, CONF_POOL_TEMPERATURE_SENSOR, SENSOR_CONSTRAINT_SENSOR_POOL, \
    CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN, \
    CONF_POOL_WINTER_IDX, CONF_POOL_DEFAULT_IDX
from ..ha_model.on_off_duration import QSOnOffDuration
from ..home_model.constraints import TimeBasedSimplePowerLoadConstraint


class QSPool(QSOnOffDuration):

    def __init__(self, **kwargs):

        self.pool_steps = []
        for min_temp, max_temp, default in POOL_TEMP_STEPS:
            val =  kwargs.pop(f"water_temp_{max_temp}", default)
            self.pool_steps.append([min_temp, max_temp, val])

        self.pool_temperature_sensor = kwargs.pop(CONF_POOL_TEMPERATURE_SENSOR)

        self.qs_pool_force_winter_mode = False

        super().__init__(**kwargs)

        self.attach_ha_state_to_probe(self.pool_temperature_sensor,
                                      is_numerical=True)


    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_POOL
    @property
    def current_water_temperature(self) -> float | None:
        temp = self.get_sensor_latest_possible_valid_value(entity_id=self.pool_temperature_sensor)
        if temp is None:
            return None
        return temp

    def support_green_only_switch(self) -> bool:
        return True

    def get_pool_filter_time_s(self, time: datetime) -> float:

        idx = 0
        if self.qs_pool_force_winter_mode:
            idx = CONF_POOL_WINTER_IDX
        else:
            data = self.get_state_history_data(entity_id=self.pool_temperature_sensor, num_seconds_before= 24*3600, to_ts=time, keep_invalid_states = False)
            if data is None or len(data) == 0:
                idx = CONF_POOL_DEFAULT_IDX
            else:
                temps = [x[1] for x in data]
                temp = min(temps)
                idx = CONF_POOL_DEFAULT_IDX
                for id, t in enumerate(self.pool_steps):
                    min_temp, max_temp, val = t
                    if temp >= min_temp and temp <= max_temp:
                        idx = id
                        break

        return self.pool_steps[idx][2]*3600.0


    async def check_load_activity_and_constraints(self, time: datetime) -> bool:
        # check that we have a connected car, and which one, or that it is completely disconnected
        #  if there is no more car ... just reset

        local_target_date = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
        local_constraint_day = datetime(local_target_date.year, local_target_date.month, local_target_date.day)
        local_tomorrow = local_constraint_day + timedelta(days=1)
        end_schedule = local_tomorrow.replace(tzinfo=None).astimezone(tz=pytz.UTC)

        if end_schedule is not None:

            if self._last_pushed_end_constraint is not None:
                if self._last_pushed_end_constraint == end_schedule:
                    # we already have a constraint for this end time
                    # this is a small optimisation to avoid creating a constraint object just to
                    # let push_live_constraint check that it is already in the list or completed
                    return False

            # schedule the load to be launched
            type = CONSTRAINT_TYPE_MANDATORY_END_TIME
            if self.qs_best_effort_green_only is True:
                type = CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN # will be before battery filling

            load_mandatory = TimeBasedSimplePowerLoadConstraint(
                    type=type,
                    time=time,
                    load=self,
                    from_user=False,
                    end_of_constraint=end_schedule,
                    power=self.power_use,
                    initial_value=0,
                    target_value=self.get_pool_filter_time_s(time),
                    num_max_on_off=6 # so 3 on/off cycle per day here
            )
            # check_end_constraint_exists will check that the constraint is not already in the list
            # or have not been done already after a restart
            res = self.push_live_constraint(time, load_mandatory, check_end_constraint_exists=True)
            self._last_pushed_end_constraint = end_schedule
            return res

        return False



