from datetime import datetime, timedelta
from datetime import time as dt_time

from ..const import (
    CONF_POOL_DEFAULT_IDX,
    CONF_POOL_TEMPERATURE_SENSOR,
    CONF_POOL_WINTER_IDX,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_FILLER_AUTO,
    POOL_TEMP_STEPS,
    SENSOR_CONSTRAINT_SENSOR_POOL,
    CONF_TYPE_NAME_QSPool,
)
from ..ha_model.bistate_duration import ConstraintItemType
from ..ha_model.on_off_duration import QSOnOffDuration
from ..home_model.constraints import DATETIME_MIN_UTC


class QSPool(QSOnOffDuration):
    conf_type_name = CONF_TYPE_NAME_QSPool

    def __init__(self, **kwargs):

        self.pool_steps = []
        for min_temp, max_temp, default in POOL_TEMP_STEPS:
            val = kwargs.pop(f"water_temp_{max_temp}", default)
            self.pool_steps.append([min_temp, max_temp, val])

        self.pool_temperature_sensor = kwargs.pop(CONF_POOL_TEMPERATURE_SENSOR)

        super().__init__(**kwargs)

        self.is_load_time_sensitive = False

        self.attach_ha_state_to_probe(self.pool_temperature_sensor, is_numerical=True)

    def get_select_translation_key(self) -> str | None:
        """return the translation key for the select"""
        return "pool_mode"

    def get_virtual_current_constraint_translation_key(self) -> str | None:
        return SENSOR_CONSTRAINT_SENSOR_POOL

    def get_bistate_modes(self) -> list[str]:
        modes = super().get_bistate_modes()
        return modes + ["pool_winter_mode"]

    @property
    def current_water_temperature(self) -> float | None:
        temp = self.get_sensor_latest_possible_valid_value(entity_id=self.pool_temperature_sensor)
        if temp is None:
            return None
        return temp

    def update_current_metrics(self, time: datetime, end_range: dt_time | None = None):

        if end_range is None:
            end_range = self.default_on_finish_time or dt_time(hour=0, minute=0, second=0)

        end_day = self.get_next_time_from_hours(local_hours=end_range, time_utc_now=time, output_in_utc=True)
        duration_s = 0.0
        run_s = 0.0

        ct_to_probe = []
        if self._last_completed_constraint is not None:
            ct_to_probe.append(self._last_completed_constraint)
        if self._constraints:
            ct_to_probe.extend(self._constraints)

        # keep only the one for the current day
        for ct in ct_to_probe:
            if ct.end_of_constraint <= end_day or (
                ct.start_of_constraint != DATETIME_MIN_UTC and ct.start_of_constraint <= end_day
            ):
                if ct.end_of_constraint > end_day - timedelta(hours=24) or (
                    ct.start_of_constraint != DATETIME_MIN_UTC
                    and ct.start_of_constraint > end_day - timedelta(hours=24)
                ):
                    duration_s += ct.target_value
                    run_s += ct.current_value

        self.qs_bistate_current_on_h = run_s / 3600.0
        self.qs_bistate_current_duration_h = duration_s / 3600.0

    def support_green_only_switch(self) -> bool:
        return True

    def get_pool_filter_time_s(self, force_winter: bool, time: datetime) -> float:

        idx = 0
        if force_winter:
            idx = CONF_POOL_WINTER_IDX
        else:
            data = self.get_state_history_data(
                entity_id=self.pool_temperature_sensor,
                num_seconds_before=24 * 3600,
                to_ts=time,
                keep_invalid_states=False,
            )
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

        return self.pool_steps[idx][2] * 3600.0

    async def _build_mode_constraint_items(self, time, bistate_mode, do_push_constraint_after):
        if bistate_mode in ("bistate_mode_auto", "pool_winter_mode"):
            if self.default_on_finish_time is None:
                self.default_on_finish_time = dt_time(hour=0, minute=0, second=0)

            end_schedule = self.get_next_time_from_hours(
                local_hours=self.default_on_finish_time, time_utc_now=time, output_in_utc=True
            )

            force_winter = bistate_mode == "pool_winter_mode"

            if end_schedule is not None:
                degraded_type = CONSTRAINT_TYPE_FILLER
                if self.is_best_effort_only_load():
                    degraded_type = CONSTRAINT_TYPE_FILLER_AUTO

                start_schedule = do_push_constraint_after
                if start_schedule is None or start_schedule < end_schedule:
                    return [
                        ConstraintItemType(
                            start_schedule=start_schedule,
                            end_schedule=end_schedule,
                            target_value=self.get_pool_filter_time_s(force_winter, time),
                            has_user_forced_constraint=False,
                            agenda_push=False,
                            degraded_type=degraded_type,
                        )
                    ]
            return []

        return await super()._build_mode_constraint_items(time, bistate_mode, do_push_constraint_after)
