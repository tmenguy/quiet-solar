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
        if self.bistate_mode != "bistate_mode_auto":
            return await super().check_load_activity_and_constraints(time)
        else:
            end_schedule = self.get_proper_local_adapted_tomorrow(time)
            if end_schedule is not None:

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
                )
                return self.push_unique_and_current_end_of_constraint_from_agenda(time=time, new_ct=load_mandatory)

            return False



