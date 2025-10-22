import copy
import logging
import pickle
from enum import StrEnum
from operator import itemgetter
from os.path import join
from typing import Mapping, Any
from datetime import datetime, timedelta

import aiofiles.os
import pytz
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder.models import LazyState
from homeassistant.components.recorder import get_instance as recorder_get_instance
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE

from .dynamic_group import QSDynamicGroup
from ..const import CONF_HOME_VOLTAGE, CONF_GRID_POWER_SENSOR, CONF_GRID_POWER_SENSOR_INVERTED, \
    HOME_CONSUMPTION_SENSOR, HOME_NON_CONTROLLED_CONSUMPTION_SENSOR, HOME_AVAILABLE_POWER_SENSOR, DOMAIN, \
    FLOATING_PERIOD_S, CONF_HOME_START_OFF_PEAK_RANGE_1, \
    CONF_HOME_END_OFF_PEAK_RANGE_1, CONF_HOME_START_OFF_PEAK_RANGE_2, CONF_HOME_END_OFF_PEAK_RANGE_2, \
    CONF_HOME_PEAK_PRICE, CONF_HOME_OFF_PEAK_PRICE, QSForecastHomeNonControlledSensors, QSForecastSolarSensors, \
    FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, GRID_CONSUMPTION_SENSOR, DASHBOARD_NUM_SECTION_MAX, \
    CONF_DASHBOARD_SECTION_NAME, CONF_DASHBOARD_SECTION_ICON, DASHBOARD_DEFAULT_SECTIONS, CONF_TYPE_NAME_QSHome, \
    MAX_POWER_INFINITE
from ..ha_model.battery import QSBattery
from ..ha_model.car import QSCar
from ..ha_model.charger import QSChargerGeneric

from ..ha_model.device import HADeviceMixin, convert_power_to_w
from ..ha_model.solar import QSSolar
from ..home_model.commands import LoadCommand, CMD_IDLE
from ..home_model.home_utils import get_average_time_series, add_amps, diff_amps, min_amps
from ..home_model.load import AbstractLoad, AbstractDevice, get_slots_from_time_series, \
    extract_name_and_index_from_dashboard_section_option, get_value_from_time_series
from ..home_model.solver import PeriodSolver

import numpy as np
import numpy.typing as npt

from ..ui.dashboard import generate_dashboard_yaml


class QSHomeMode(StrEnum):
    HOME_MODE_OFF = "home_mode_off"
    HOME_MODE_SENSORS_ONLY = "home_mode_sensors_only"
    HOME_MODE_CHARGER_ONLY = "home_mode_charger_only"
    HOME_MODE_ON = "home_mode_on"


_LOGGER = logging.getLogger(__name__)

MAX_SENSOR_HISTORY_S = 60*60*24*7

POWER_ALIGNMENT_TOLERANCE_S = 120

class QSforecastValueSensor:

    _stored_values: list[tuple[datetime, float]]

    @classmethod
    def get_probers(cls, getter, getter_now, names_and_duration):

        probers = {}

        for name, duration_s in names_and_duration.items():
            probers[name] = cls(name, duration_s, getter, getter_now)

        return probers


    def __init__(self, name, duration_s, forecast_getter, current_getter=None):
        self._stored_values = []
        self._getter = forecast_getter
        self._current_getter = current_getter
        self._delta = timedelta(seconds=duration_s)
        self.name = name


    def push_and_get(self, time: datetime) -> float | None:

        if self._delta == timedelta(seconds=0) and self._current_getter is not None:
            # get the current value
            _, value = self._current_getter(time)
        else:

            future_time = time + self._delta
            future_time, future_val = self._getter(future_time)
            if future_val is not None:
                t,v, found, _ = get_value_from_time_series(self._stored_values, future_time)
                if found is False:
                    self._stored_values.append((future_time, future_val))

            if not self._stored_values:
                return None

            # will give the best value for time
            time_found_idx, value, found, idx = get_value_from_time_series(self._stored_values, time)

            # the list is sorted ... remove all index before idx as we wil never ask again for a time before "time"
            if value is not None and idx > 0 and idx < len(self._stored_values):
                self._stored_values = self._stored_values[idx - 1:]

        return value



class QSHome(QSDynamicGroup):

    conf_type_name = CONF_TYPE_NAME_QSHome

    def __init__(self, **kwargs) -> None:

        self._battery: QSBattery | None = None
        self._voltage: int = 230

        self._chargers: list[QSChargerGeneric] = []
        self._cars: list[QSCar] = []

        self._all_devices: list[HADeviceMixin] = []
        self._disabled_devices: list[HADeviceMixin] = []
        self._solar_plant: QSSolar | None = None
        self._all_loads: list[AbstractLoad] = []
        self._all_dynamic_groups: list[QSDynamicGroup] = []
        self._name_to_groups: dict[str, QSDynamicGroup] = {}

        self._period: timedelta = timedelta(seconds=FLOATING_PERIOD_S)
        self._commands: list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]] = []
        self._battery_commands: list[tuple[datetime, LoadCommand]] = []
        self._solver_step_s: timedelta = timedelta(seconds=900)
        self._update_step_s: timedelta = timedelta(seconds=5)

        self.grid_active_power_sensor: str | None = None
        self.grid_active_power_sensor_inverted: bool = False

        self.qs_home_is_off_grid = False



        self._voltage = kwargs.pop(CONF_HOME_VOLTAGE, 230)
        self.grid_active_power_sensor = kwargs.pop(CONF_GRID_POWER_SENSOR, None)
        self.grid_active_power_sensor_inverted = kwargs.pop(CONF_GRID_POWER_SENSOR_INVERTED, False)

        self.tariff_start_1 = kwargs.pop(CONF_HOME_START_OFF_PEAK_RANGE_1, None)
        self.tariff_end_1  = kwargs.pop(CONF_HOME_END_OFF_PEAK_RANGE_1, None)
        self.tariff_start_2 = kwargs.pop(CONF_HOME_START_OFF_PEAK_RANGE_2, None)
        self.tariff_end_2 = kwargs.pop(CONF_HOME_END_OFF_PEAK_RANGE_2, None)
        self.price_peak = kwargs.pop(CONF_HOME_PEAK_PRICE, 0.2)
        if self.price_peak <= 0:
            self.price_peak = 0.2
        self.price_peak = self.price_peak / 1000.0
        self.price_off_peak = kwargs.pop(CONF_HOME_OFF_PEAK_PRICE, 0.0) / 1000.0


        self.dashboard_sections: list[tuple[str, str]] = []
        num_found_sections = 0
        for i in range(0, DASHBOARD_NUM_SECTION_MAX):
            key_section_name = f"{CONF_DASHBOARD_SECTION_NAME}_{i}"
            key_section_icon = f"{CONF_DASHBOARD_SECTION_ICON}_{i}"

            s_name: str | None = kwargs.pop(key_section_name, None)
            s_icon: str | None = kwargs.pop(key_section_icon, None)

            if s_name is None and s_icon is None:
                continue

            num_found_sections += 1
            if s_name is None:
                s_name = f"section_{num_found_sections}"
            else:
                s_name, found_index = extract_name_and_index_from_dashboard_section_option(s_name)

            self.dashboard_sections.append((s_name, s_icon))

        if num_found_sections == 0:
            self.dashboard_sections = copy.deepcopy(DASHBOARD_DEFAULT_SECTIONS)

        self.home_non_controlled_consumption_sensor = HOME_NON_CONTROLLED_CONSUMPTION_SENSOR
        self.home_available_power_sensor = HOME_AVAILABLE_POWER_SENSOR
        self.home_consumption_sensor = HOME_CONSUMPTION_SENSOR
        self.grid_consumption_power_sensor = GRID_CONSUMPTION_SENSOR

        self.home_non_controlled_power_forecast_sensor_values = {}
        self.home_solar_forecast_sensor_values = {}

        self.home_non_controlled_power_forecast_sensor_values_providers = QSforecastValueSensor.get_probers(
            self.get_non_controlled_consumption_from_current_forecast_getter,
            self.get_non_controlled_consumption_best_stored_value_getter,
            QSForecastHomeNonControlledSensors)

        self.home_solar_forecast_sensor_values_providers = QSforecastValueSensor.get_probers(
            self.get_solar_from_current_forecast_getter,
            None,
            QSForecastSolarSensors)

        kwargs["home"] = self
        self.home = self
        super().__init__(**kwargs)
        self.home = self

        # self._all_devices.append(self)

        self.home_non_controlled_consumption = None
        self.home_consumption = None
        self.home_available_power = None
        self.grid_consumption_power = None
        self.home_mode = None

        self._last_active_load_time = None
        if self.grid_active_power_sensor_inverted:
            self.attach_power_to_probe(self.grid_active_power_sensor,
                                          transform_fn=lambda x, a: (-x, a))
        else:
            self.attach_power_to_probe(self.grid_active_power_sensor)

        self.attach_power_to_probe(self.home_consumption_sensor,
                                      non_ha_entity_get_state=self.home_consumption_sensor_state_getter)

        self.attach_power_to_probe(self.home_available_power_sensor,
                                      non_ha_entity_get_state=self.home_available_power_sensor_state_getter)

        self.attach_power_to_probe(self.grid_consumption_power_sensor,
                                      non_ha_entity_get_state=self.grid_consumption_power_sensor_state_getter)

        self.attach_power_to_probe(self.home_non_controlled_consumption_sensor,
                                      non_ha_entity_get_state=self.home_non_controlled_consumption_sensor_state_getter)


        self._consumption_forecast = QSHomeConsumptionHistoryAndForecast(self)
        # self.register_all_on_change_states()
        self._last_solve_done  : datetime | None = None

        self.add_device(self)

        self._init_completed = False


    async def async_set_off_grid_mode(self, off_grid:bool):



        do_reset = self.qs_home_is_off_grid != off_grid

        self.qs_home_is_off_grid = off_grid

        if do_reset:

            _LOGGER.warning(f"async_set_off_grid_mode: {off_grid}")

            # reset all loads
            for load in self._all_loads:
                if load.qs_enable_device is False:
                    continue
                load.reset()

            # force solve
            self.force_next_solve()

            # force an update of all load will recompute what should be computed, like new constraints, etc
            await self.force_update_all()

    def is_off_grid(self) -> bool:
        return self.qs_home_is_off_grid

    @property
    def voltage(self) -> float:
        """Return the voltage of the home."""
        return self._voltage


    def get_home_max_phase_amp(self) -> float:

        static_amp = self.dyn_group_max_phase_current_conf
        if not self.is_off_grid():
            return static_amp

        # ok we are in off grid mode, we need to limit the current to the max phase current of the home
        available_production_w = 0
        if self._solar_plant is not None:
            available_production_w = self._solar_plant.solar_production

        if self._battery is not None and self._battery.battery_can_discharge():
            available_production_w += self._battery.get_max_discharging_power()  # self._battery.max_discharge_number

        if self._solar_plant and self._solar_plant.solar_max_output_power_value:
            available_production_w = min(available_production_w, self._solar_plant.solar_max_output_power_value)

        if self.physical_3p:
            available_production_amp = (available_production_w / 3.0) / (self.voltage)
        else:
            available_production_amp = available_production_w / (self.voltage)

        available_production_amp = min(self.dyn_group_max_phase_current_conf, available_production_amp)

        if self._solar_plant:
            available_production_amp = min(available_production_amp, self._solar_plant.solar_max_phase_amps)

        return min(static_amp, available_production_amp)


    @property
    def dyn_group_max_phase_current(self) -> list[float|int]:

        static_amps = super().dyn_group_max_phase_current

        if not self.is_off_grid():
            return static_amps

        available_production_amp = self.get_home_max_phase_amp()

        if self.physical_3p:
            self._dyn_group_max_phase_current = [available_production_amp, available_production_amp, available_production_amp]
        else:
            self._dyn_group_max_phase_current = [0, 0, 0]
            self._dyn_group_max_phase_current[self.mono_phase_index] = available_production_amp

        _LOGGER.info(f"dyn_group_max_phase_current: Home in off grid mode, setting max phase current to {available_production_amp}A instead of {self.dyn_group_max_phase_current_conf}A")

        self._dyn_group_max_phase_current = min_amps(self._dyn_group_max_phase_current, static_amps)

        return self._dyn_group_max_phase_current

    def get_devices_for_dashboard_section(self, section_name: str) -> list[HADeviceMixin]:

        found = set()

        for list in [self._all_devices, self._disabled_devices]:
            for d in list:
                if isinstance(d, AbstractDevice):
                    if d.dashboard_section == section_name:
                        found.add(d)

        for list in [self._all_devices]:
            for d in list:
                virtual_devices = d.get_attached_virtual_devices()
                for vd in virtual_devices:
                    # attach the virtual device if the main device is in the section
                    if d in found:
                        found.add(vd)
                    # attach the same type of virtual device is there is one in the section
                    for f in found:
                        if vd.device_type == f.device_type or type(vd) == type(f):
                            found.add(vd)
                            break

        found = sorted(found, key=lambda device: device.dashboard_sort_string)

        return found


    def get_best_tariff(self, time: datetime) -> float:
        if self.price_off_peak == 0:
            return self.price_peak
        else:
            return min(self.price_off_peak, self.price_peak)

    def force_next_solve(self):
        self._last_solve_done = None

    def get_non_controlled_consumption_from_current_forecast_getter(self, start_time:datetime) -> tuple[datetime | None, str | float | None]:
        if self._consumption_forecast:
            if self._consumption_forecast.home_non_controlled_consumption:
                return self._consumption_forecast.home_non_controlled_consumption.get_value_from_current_forecast(start_time)
        return (None, None)

    def get_non_controlled_consumption_best_stored_value_getter(self, start_time:datetime) -> tuple[datetime | None, str | float | None]:
        if self._consumption_forecast:
            if self._consumption_forecast.home_non_controlled_consumption:
                return self._consumption_forecast.home_non_controlled_consumption.get_closest_stored_value(start_time)
        return (None, None)


    def _compute_non_controlled_forecast_intl(self, time: datetime) -> list[tuple[datetime | None, float | None]]:

        forecast = self._consumption_forecast.home_non_controlled_consumption.compute_now_forecast(
                time_now=time,
                history_in_hours=24, future_needed_in_hours=int(self._period.total_seconds() // 3600) + 1,
                set_as_current=True)

        return forecast

    async def compute_non_controlled_forecast(self, time: datetime) -> list[tuple[datetime | None, float | None]]:

        unavoidable_consumption_forecast = []
        if await self._consumption_forecast.init_forecasts(time):
            unavoidable_consumption_forecast = self._compute_non_controlled_forecast_intl(time)

        return unavoidable_consumption_forecast

    def get_solar_from_current_forecast_getter(self, start_time:datetime) -> tuple[datetime | None, str | float | None]:
        if self._solar_plant:
            return self._solar_plant.get_value_from_current_forecast(start_time)
        return (None, None)

    def get_solar_from_current_forecast(self, start_time:datetime, end_time:datetime | None = None) -> list[tuple[datetime | None, float | None]]:
        if self._solar_plant:
            return self._solar_plant.get_forecast(start_time, end_time)
        return []


    def _get_today_off_peak_ranges(self, time:datetime) -> list[tuple[datetime, datetime]]:

        if self.price_off_peak == 0:
            return []

        to_prob = [
            (self.tariff_start_1, self.tariff_end_1, self.price_off_peak),
            (self.tariff_start_2, self.tariff_end_2, self.price_off_peak)
        ]

        ranges_off_peak = []

        start_time_local = time.replace(tzinfo=pytz.UTC).astimezone(tz=None)

        for start, end, price in to_prob:
            if start is None or end is None or price is None or price == 0:
                continue

            if start == end:
                continue


            start_utc = datetime.fromisoformat(f"{start_time_local.strftime("%Y-%m-%d")} {start}").replace(tzinfo=None).astimezone(tz=pytz.UTC)
            end_utc = datetime.fromisoformat(f"{start_time_local.strftime("%Y-%m-%d")} {end}").replace(tzinfo=None).astimezone(tz=pytz.UTC)

            if end_utc < start_utc:
                end_utc += timedelta(days=1)

            ranges_off_peak.append((start_utc, end_utc))

        if len(ranges_off_peak) > 1:
            ranges_off_peak = sorted(ranges_off_peak, key=lambda x: x[0])

        return ranges_off_peak


    def get_tariff(self, start_time: datetime, end_time: datetime) -> float:
        if self.price_off_peak == 0:
            return self.price_peak

        ranges_off_peak = self._get_today_off_peak_ranges(start_time)

        if not ranges_off_peak:
            return self.price_peak

        price_start = self.price_peak
        for range_start, range_end in ranges_off_peak:
            if start_time < range_start:
                price_start = self.price_peak
                break
            if start_time < range_end:
                price_start =  self.price_off_peak
                break

        price_end = self.price_peak
        for range_start, range_end in ranges_off_peak:
            if end_time < range_start:
                price_end = self.price_peak
                break
            if end_time < range_end:
                price_end =  self.price_off_peak
                break

        return max(price_start, price_end)


    def get_tariffs(self, start_time:datetime, end_time:datetime) -> list[tuple[datetime, float]] | float:

        if self.is_off_grid():
            return 0.0

        if self.price_off_peak == 0:
            return self.price_peak

        ranges_off_peak = self._get_today_off_peak_ranges(start_time)

        if not ranges_off_peak:
            return self.price_peak

        span = end_time - start_time
        num_day = span.days + 1
        start_day = start_time.replace(hour=0, minute=0, second=0, microsecond=0) # beware it is utc time
        tariffs = []
        curr_start = start_day
        for d in range(num_day):
            for first_start, first_end in ranges_off_peak:

                curr_end = first_start + timedelta(days=d)

                if curr_end > curr_start and curr_end >= start_time:
                    if curr_start < start_time:
                        curr_start = start_time
                    tariffs.append((curr_start, self.price_peak))

                curr_start = curr_end
                curr_end = first_end + timedelta(days=d)

                if curr_end > curr_start and curr_end >= start_time:
                    if curr_start < start_time:
                        curr_start = start_time
                    tariffs.append((curr_start, self.price_off_peak))

                curr_start = curr_end

                if curr_end >= end_time:
                    break

        return tariffs

    def get_platforms(self):
        parent = super().get_platforms()
        if parent is None:
            parent = set()
        else:
            parent = set(parent)
        parent.update([ Platform.SENSOR, Platform.SELECT, Platform.BUTTON, Platform.SWITCH ])
        return list(parent)

    def get_car_by_name(self, name: str) -> QSCar | None:
        for car in self._cars:
            if car.name == name:
                return car
        return None

    def home_consumption_sensor_state_getter(self, entity_id: str,  time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if self.home_consumption is None:
            return None

        return (time, self.home_consumption, {})

    def home_available_power_sensor_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if self.home_available_power is None:
            return None

        return (time, self.home_available_power, {})

    def grid_consumption_power_sensor_state_getter(self, entity_id: str, time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        if self.grid_consumption_power is None:
            return None

        return (time, self.grid_consumption_power, {})


    def home_non_controlled_consumption_sensor_state_getter(self,
                                                            entity_id: str,
                                                            time: datetime | None) -> (
            tuple[datetime | None, float | str | None, dict | None] | None):

        # Home real consumption : Solar Production - grid_active_power_sensor - battery_charge_discharge_sensor
        # Available Power : grid_active_power_sensor + battery_charge_discharge_sensor
        # controlled consumption: sum of all controlled loads
        # Non controlled consumption : Home real consumption - controlled consumption

        # get solar production
        solar_production_minus_battery = None
        solar_production = None

        battery_charge = None
        if self._battery is not None:
            battery_charge = self._battery.get_sensor_latest_possible_valid_value(self._battery.charge_discharge_sensor, tolerance_seconds=POWER_ALIGNMENT_TOLERANCE_S, time=time)


        if self._solar_plant is not None:
            self._solar_plant.solar_production = 0.0
            solar_production = None
            if self._solar_plant.solar_inverter_active_power:
                # this one has the battery inside!
                solar_production_minus_battery = self._solar_plant.get_sensor_latest_possible_valid_value(self._solar_plant.solar_inverter_active_power, tolerance_seconds=POWER_ALIGNMENT_TOLERANCE_S, time=time)

            if solar_production_minus_battery is None and self._solar_plant.solar_inverter_input_active_power:
                solar_production = self._solar_plant.get_sensor_latest_possible_valid_value(self._solar_plant.solar_inverter_input_active_power, tolerance_seconds=POWER_ALIGNMENT_TOLERANCE_S, time=time)

            if solar_production_minus_battery is None:
                if solar_production is not None and battery_charge is not None:
                    solar_production_minus_battery = solar_production - battery_charge
                elif solar_production is not None:
                    solar_production_minus_battery = solar_production

            if solar_production is not None:
                self._solar_plant.solar_production = solar_production
            elif solar_production_minus_battery is not None:
                if battery_charge is not None:
                    self._solar_plant.solar_production = solar_production_minus_battery + battery_charge
                else:
                    self._solar_plant.solar_production = solar_production_minus_battery

        elif battery_charge is not None:
            solar_production_minus_battery = 0 - battery_charge


        if solar_production_minus_battery is None:
            solar_production_minus_battery = 0

        if self._solar_plant is not None:
            self._solar_plant.solar_production_minus_battery = solar_production_minus_battery

        # get grid consumption
        grid_consumption = self.get_sensor_latest_possible_valid_value(self.grid_active_power_sensor, tolerance_seconds=POWER_ALIGNMENT_TOLERANCE_S, time=time)

        if grid_consumption is None:
            self.home_non_controlled_consumption = None
            self.home_consumption = None
            self.home_available_power = None
            self.grid_consumption_power = None
        else:
            if solar_production_minus_battery is not None:
                home_consumption = solar_production_minus_battery - grid_consumption
            else:
                home_consumption = 0 - grid_consumption

            controlled_consumption = self.get_device_power_latest_possible_valid_value(tolerance_seconds=POWER_ALIGNMENT_TOLERANCE_S, time=time, ignore_auto_load=True)

            val = home_consumption - controlled_consumption

            self.home_non_controlled_consumption = val
            self.home_consumption = home_consumption
            if battery_charge is not None:
                self.home_available_power = grid_consumption + battery_charge
            else:
                self.home_available_power = grid_consumption

            # clamp the available power to what could be really available
            if self.home_available_power > 0:
                # we need to check if what is available will "really" be available to consume by any dynamic load ...
                maximum_production_output = self.get_current_maximum_production_output_power()

                if solar_production_minus_battery + self.home_available_power >= maximum_production_output:
                    if self._battery is not None:
                        max_battery_discharge = self._battery.battery_get_current_possible_max_discharge_power()
                    else:
                        max_battery_discharge = 0

                    _LOGGER.warning("Home available_power CLAMPED: from %.2f to  %.2f, (solar_production_minus_battery:%.2f, maximum_production_output:%.2f) (solar_production:%.2f) (max_battery_discharge:%.2f)", self.home_available_power, max(0.0, maximum_production_output - solar_production_minus_battery), solar_production_minus_battery, maximum_production_output, solar_production, max_battery_discharge )
                    self.home_available_power = max(0.0, maximum_production_output - solar_production_minus_battery)


            self.grid_consumption_power = grid_consumption

        val = self.home_non_controlled_consumption

        # slight hack to push the value to history:
        self.add_to_history(self.home_available_power_sensor, time)
        self.add_to_history(self.grid_consumption_power_sensor, time)

        if val is None:
            return None

        return (time, val, {})

    def get_current_over_clamp_production_power(self) -> float:

        if self._solar_plant is None:
            return 0.0

        if self._solar_plant.solar_production > self._solar_plant.solar_max_output_power_value:
            return self._solar_plant.solar_production - self._solar_plant.solar_max_output_power_value

        return 0.0

    def get_current_maximum_production_output_power(self) -> float:

        maximum_production_output = MAX_POWER_INFINITE

        if self._solar_plant is not None and self._solar_plant.solar_max_output_power_value:
            # we need to check if what is available will "really" be available to consume by any dynamic load ...
            maximum_production_output = self._solar_plant.solar_max_output_power_value

        if self._battery is not None:
            max_battery_discharge = self._battery.battery_get_current_possible_max_discharge_power()
        else:
            max_battery_discharge = 0

        # check what could really be the max production output...because we could not reach the max inverter output
        # with the current production of the solar plant + battery discharge
        if self._solar_plant is not None and self._solar_plant.solar_production + max_battery_discharge < maximum_production_output:
           maximum_production_output = self._solar_plant.solar_production + max_battery_discharge

        return maximum_production_output

    def get_grid_active_power_values(self, duration_before_s: float, time: datetime)-> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self.grid_active_power_sensor, duration_before_s, time)

    def get_available_power_values(self, duration_before_s: float, time: datetime)-> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self.home_available_power_sensor, duration_before_s, time)


    def get_grid_consumption_power_values(self, duration_before_s: float, time: datetime)-> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        return self.get_state_history_data(self.grid_consumption_power_sensor, duration_before_s, time)

    def get_device_amps_consumption(self, tolerance_seconds: float | None, time:datetime) -> list[float|int] | None:

        home_amps = [0, 0, 0]

        if not self.is_off_grid():
            # first check if we do have an amp sensor for the phases
            multiplier = 1
            if self.grid_active_power_sensor_inverted is False:
                # if not inverted it should be -1 as consumption are reversed
                multiplier = -1

            pM = None
            is_3p = False
            if isinstance(self, AbstractDevice):
                is_3p = self.current_3p

            if self.grid_active_power_sensor is not None:
                # this one has been inverted at the moment it was attached to prob
                # so just multiply it by -1
                p = self.get_sensor_latest_possible_valid_value(self.grid_active_power_sensor, tolerance_seconds=tolerance_seconds, time=time)
                if p is not None:
                    p = -1.0*p
                    if p > 0 and isinstance(self, AbstractDevice):
                        pM =  self.get_phase_amps_from_power(power=p, is_3p=is_3p)

            home_amps = self._get_device_amps_consumption(pM, tolerance_seconds, time, multiplier=multiplier, is_3p=is_3p)


        if self._solar_plant:
            solar_amps = self._solar_plant.get_device_amps_consumption(tolerance_seconds, time)
            if solar_amps is not None and home_amps is not None:
                home_amps = add_amps(home_amps, solar_amps)

        return home_amps

    def battery_can_discharge(self):

        if self._battery is None:
            return False

        return self._battery.battery_can_discharge()

    def get_battery_charge_values(self, duration_before_s: float, time: datetime) -> list[tuple[datetime | None, str|float|None, Mapping[str, Any] | None | dict]]:
        if self._battery is None:
            return []
        return self._battery.get_state_history_data(self._battery.charge_discharge_sensor, duration_before_s, time)


    async def set_max_discharging_power(self, power: float | None = None, blocking: bool = False):
        if self._battery is not None:
            await self._battery.set_max_discharging_power(power, blocking)

    async def set_max_charging_power(self, power: float | None, blocking: bool = False):
        if self._battery is not None:
            await self._battery.set_max_charging_power(power, blocking)

    def _set_topology(self):

        self._name_to_groups = {}
        for group in self._all_dynamic_groups:
            self._name_to_groups[group.name] = group
            group._childrens = []

        for group in self._all_dynamic_groups:
            if self != group:
                # it is not home
                father = self._name_to_groups.get(group.dynamic_group_name, self)
                father._childrens.append(group)
                group.father_device = father

        for load in self._all_loads:
            if isinstance(load, QSDynamicGroup):
                continue
            load.father_device = None


        for load in self._all_loads:
            if isinstance(load, QSDynamicGroup):
                continue
            father = self._name_to_groups.get(load.dynamic_group_name, self)
            if load.father_device == father:
                # already in the group
                _LOGGER.warning( f"_set_topology Load {load.name} already in its froup {father.name}")
                continue
            if load.father_device is not None:
                _LOGGER.warning( f"_set_topology Load {load.name} already in added elsewhere in {load.father_device} we wanted {father.name}")
                continue

            father._childrens.append(load)
            load.father_device = father

    def add_device(self, device):

        device.home = self

        if isinstance(device, QSBattery):
            self._battery = device
        elif isinstance(device, QSCar):
            if device not in self._cars:
                self._cars.append(device)
        elif isinstance(device, QSChargerGeneric):
            if device not in self._chargers:
                self._chargers.append(device)
        elif isinstance(device, QSSolar):
            self._solar_plant = device

        if isinstance(device, AbstractLoad):
            if device not in self._all_loads:
                self._all_loads.append(device)

        if isinstance(device, QSDynamicGroup):
            # it can be home....
            if device not in self._all_dynamic_groups:
                self._all_dynamic_groups.append(device)

        if isinstance(device, HADeviceMixin):
            if device not in self._all_devices:
                device.register_all_on_change_states()
                self._all_devices.append(device)

            for attached_device in device.get_attached_virtual_devices():
                if isinstance(attached_device, AbstractDevice):
                    self.add_device(attached_device)

        #will redo the whole topology each time
        self._set_topology()

    def remove_device(self, device):

        if device == self:
            # we can't remove home....
            return 

        device.home = self

        if isinstance(device, QSBattery):
            self._battery = None
        elif isinstance(device, QSCar):
            # remove the car from the list
            try:
                self._cars.remove(device)
            except ValueError:
                _LOGGER.warning(f"Attempted to remove car {device.name} that was not in the list of cars")
        elif isinstance(device, QSChargerGeneric):
            try:
                self._chargers.remove(device)
            except ValueError:
                _LOGGER.warning(f"Attempted to remove charger {device.name} that was not in the list of chargers")
        elif isinstance(device, QSSolar):
            self._solar_plant = None

        if isinstance(device, AbstractLoad):
            try:
                self._all_loads.remove(device)
            except ValueError:
                _LOGGER.warning(f"Attempted to remove load {device.name} that was not in the list of loads")

        if isinstance(device, QSDynamicGroup):
            try:
                self._all_dynamic_groups.remove(device)
            except ValueError:
                _LOGGER.warning(f"Attempted to remove dynamic group {device.name} that was not in the list of dynamic groups")

        if isinstance(device, HADeviceMixin):
            try:
                self._all_devices.remove(device)
            except ValueError:
                _LOGGER.warning(f"Attempted to remove device {device.name} that was not in the list of devices")

        if isinstance(device, HADeviceMixin):
            for attached_device in device.get_attached_virtual_devices():
                self.remove_device(attached_device)

        #will redo the whole topology each time
        self._set_topology()

    def add_disabled_device(self, device):
        if device not in self._disabled_devices:
            self._disabled_devices.append(device)

    def remove_disabled_device(self, device):
        try:
            self._disabled_devices.remove(device)
        except ValueError:
            _LOGGER.warning(f"Attempted to remove device form disabled list {device.name} that was not in the list of disabled devices")

    def finished_setup(self, time: datetime) -> bool:
        """
        Check if the home setup is finished.
        This is used to determine if the home is ready to be used.
        """

        if self._init_completed:
            return True

        # check if we have some devices
        if not self._all_devices:
            return False

        # check if we have a battery
        # if self._battery is None:
        #    return False

        # check if we have a solar plant
        # if self._solar_plant is None:
        #    return False

        # check if we have at least one load ... we may have zero if they are all disabled in fact!
        if not self._all_loads and not self._disabled_devices:
            return False


        for load in self._all_loads:
            if load.externally_initialized_constraints is False:
                return False

        for device in self._all_devices:
            if isinstance(device, AbstractDevice):
                if device.qs_enable_device:
                    device.device_post_home_init(time)

        self._init_completed = True
        return True


    async def update_loads_constraints(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            QSHomeMode.HOME_MODE_SENSORS_ONLY.value
            ]:
            return

        if self.finished_setup(time) is False:
            _LOGGER.info("update_loads_constraints: Home not finished setup, skipping")
            return

        # check for active loads
        for load in self._all_loads:
            if load.qs_enable_device is False:
                continue
            if await load.do_run_check_load_activity_and_constraints(time):
                self.force_next_solve()


    async def update_forecast_probers(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            ]:
            return

        if self.finished_setup(time) is False:
            _LOGGER.info("update_forecast_probers: Home not finished setup, skipping")
            return

        if await self._consumption_forecast.init_forecasts(time) is False:
            _LOGGER.info("update_forecast_probers: _consumption_forecast not ready skipping")
            return

        for name, prober in self.home_non_controlled_power_forecast_sensor_values_providers.items():
            self.home_non_controlled_power_forecast_sensor_values[name] = prober.push_and_get(time)

        for name, prober in self.home_solar_forecast_sensor_values_providers.items():
            self.home_solar_forecast_sensor_values[name] = prober.push_and_get(time)


    async def update_all_states(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            ]:
            return

        if self.finished_setup(time) is False:
            _LOGGER.info("update_all_states: Home not finished setup, skipping")
            return

        if self._solar_plant:
            try:
                await self._solar_plant.update_forecast(time)
            except Exception as err:
                _LOGGER.error(f"Error updating solar forecast {err}", exc_info=err)

        for device in self._all_devices:
            try:
                await device.update_states(time)
            except Exception as err:
                if isinstance(device, AbstractDevice):
                    _LOGGER.error(f"Error updating states for device:{device.name} error: {err}", exc_info=err)


        if await self._consumption_forecast.init_forecasts(time):
            if self._consumption_forecast.home_non_controlled_consumption is not None:
                if self._consumption_forecast.home_non_controlled_consumption.add_value(time, self.home_non_controlled_consumption):
                    await self._consumption_forecast.home_non_controlled_consumption.save_values()
            if self._consumption_forecast.home_non_controlled_consumption.update_current_forecast_if_needed(time):
                self._compute_non_controlled_forecast_intl(time)


    async def update_loads(self, time: datetime):

        if self.home_mode is None or self.home_mode in [
            QSHomeMode.HOME_MODE_OFF.value,
            QSHomeMode.HOME_MODE_SENSORS_ONLY.value
            ]:
            return

        if self.finished_setup(time) is False:
            _LOGGER.info("update_loads: Home not finished setup, skipping")
            return

        try:
            await self.update_loads_constraints(time)
        except Exception as err:
            _LOGGER.error(f"Error updating loads constraints {err}", exc_info=err)

        if self._battery is not None:
            all_loads = self._all_loads
            all_extended_loads = [self._battery]
            all_extended_loads.extend(self._all_loads)
        else:
            all_loads = all_extended_loads = self._all_loads

        if self.home_mode == QSHomeMode.HOME_MODE_CHARGER_ONLY.value:
            all_loads = all_extended_loads = self._chargers

        for load in all_extended_loads:

            try:
                wait_time = await load.check_commands(time=time)
                if wait_time > timedelta(seconds=45):
                    if load.running_command_num_relaunch < 3:
                        await load.force_relaunch_command(time)
                    else:
                        # we have an issue with this command ....
                        pass
            except Exception as err:
                _LOGGER.error(f"Error checking load commands {load.name} {err}", exc_info=err)


        do_force_solve = False
        for load in all_loads:

            if load.is_load_active(time) is False:
                continue

            try:
                # need to add this self._update_step_s to add a tolerancy for the mandatory not met constraint, to not
                # send an unwanted command (see bellow while len(commands) > 0 and commands[0][0] < time + self._update_step_s:
                # to not accidentaly send an idle command on an unfinished command
                if await load.update_live_constraints(time, self._period, 4*self._update_step_s):
                    do_force_solve = True
            except Exception as err:
                _LOGGER.error(f"Error updating live constraints for load {load.name} {err}", exc_info=err)


        active_loads = []
        for load in all_loads:
            if load.is_load_active(time):
                active_loads.append(load)

        if self._last_solve_done is None or (time - self._last_solve_done) > timedelta(seconds=5*60):
            do_force_solve = True

        # we may also want to force solve ... if we have less energy than what was expected too ....imply force every 5mn

        if do_force_solve and (active_loads or self._battery is not None):

            _LOGGER.info("DO SOLVE")

            self._last_solve_done = time

            unavoidable_consumption_forecast = await self.compute_non_controlled_forecast(time)
            pv_forecast = self.get_solar_from_current_forecast(time, time + self._period)

            max_inverter_dc_to_ac_power = None
            if self._solar_plant is not None and self._solar_plant.solar_max_output_power_value is not None:
                max_inverter_dc_to_ac_power = self._solar_plant.solar_max_output_power_value

            end_time = time + self._period
            solver = PeriodSolver(
                start_time = time,
                end_time = end_time,
                tariffs = self.get_tariffs(time, end_time),
                actionable_loads = active_loads,
                battery = self._battery,
                pv_forecast  = pv_forecast,
                unavoidable_consumption_forecast = unavoidable_consumption_forecast,
                step_s = self._solver_step_s,
                max_inverter_dc_to_ac_power=max_inverter_dc_to_ac_power
            )

            # need to tweak a bit if there is some available power now for ex (or not) vs what is forecasted here.
            # use the available power virtual sensor to modify the begining of the PeriodSolver available power
            # computation based on forecasts

            self._commands, self._battery_commands = solver.solve()

        if self._battery and self._battery_commands is not None:

            while len(self._battery_commands) > 0 and self._battery_commands[0][0] < time + self._update_step_s:
                cmd_time, command = self._battery_commands.pop(0)
                await self._battery.launch_command(time, command, ctxt=f"launch command battery true launch at {cmd_time}")
                # only launch one at a time for a given load
                break

        # we should order commands by load that have key constraints ... and the launch command should forbid as a last ressort
        # any command that would go above amps budget (especially true in case of off grid mode)

        delta_amps = [0, 0, 0]

        for load, commands in self._commands:

            prev_cmd = load.current_command
            if prev_cmd is None:
                prev_cmd = CMD_IDLE

            while len(commands) > 0 and commands[0][0] < time + self._update_step_s:

                cmd_time, command = commands.pop(0)

                new_amps = load.get_phase_amps_from_power_for_budgeting(command.power_consign)
                current_amps = load.get_phase_amps_from_power_for_budgeting(prev_cmd.power_consign)
                delta_load_amps = diff_amps(new_amps, current_amps)

                if command.power_consign == 0 or load.father_device.is_delta_current_acceptable(delta_amps=add_amps(delta_amps, delta_load_amps), time=time):
                    # _LOGGER.info(f"---> Set load command {load.name} {command}")
                    await load.launch_command(time, command, ctxt=f"upload_time true launch at {cmd_time}")
                    # only launch one at a time for a given load
                    delta_amps = add_amps(delta_amps, delta_load_amps)
                    break
                else:
                    _LOGGER.warning(f"update_loads: ---> FORBID Set load command {load.name} {command} delta_amps {delta_amps} delta_load_amps {delta_load_amps}")

                prev_cmd = command

        for load in all_loads:
            if load.is_load_has_a_command_now_or_coming(time) is False or load.get_current_active_constraint(time) is None or load.is_load_active(time) is False:
                # set them back to a kind of "idle" state, many times will be "OFF" CMD
                # _LOGGER.info(f"---> Set load idle {load.name} {load.is_load_has_a_command_now_or_coming(time)} {load.get_current_active_constraint(time)} {load.is_load_active(time)}")
                await load.launch_command(time=time, command=CMD_IDLE, ctxt="launch command idle for active, no command loads or not active")

            await load.do_probe_state_change(time)


    async def force_update_all(self, time: datetime = None):
        if time is None:
            time = datetime.now(pytz.UTC)
        await self.update_all_states(time)
        await self.update_loads(time)

    async def reset_forecasts(self, time: datetime = None):
        if time is None:
            time = datetime.now(pytz.UTC)
        if self._consumption_forecast:
            await self._consumption_forecast.reset_forecasts(time)

    async def light_reset_forecasts(self, time: datetime = None):
        if time is None:
            time = datetime.now(pytz.UTC)
        if self._consumption_forecast:
            await self._consumption_forecast.reset_forecasts(time, light_reset=True)


    async def dump_for_debug(self):
        storage_path: str = join(self.hass.config.path(), DOMAIN, "debug")
        await aiofiles.os.makedirs(storage_path, exist_ok=True)

        if self._solar_plant:
            await self._solar_plant.dump_for_debug(storage_path)

        time = datetime.now(pytz.UTC)

        if self._consumption_forecast:
            await self._consumption_forecast.dump_for_debug(storage_path)

        debugs = {"now":time}


        file_path = join(storage_path, "debug_conf.pickle")

        def _pickle_save(file_path, obj):
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)

        await self.hass.async_add_executor_job(
            _pickle_save, file_path, debugs
        )

    async def generate_yaml_for_dashboard(self):
        await generate_dashboard_yaml(self)

# to be able to easily fell on the same week boundaries, it has to be a multiple of 7, take 80 to go more than 1.5 year
BUFFER_SIZE_DAYS = 80*7
# interval in minutes between 2 measures
NUM_INTERVAL_PER_HOUR = 4
INTERVALS_MN = 60//NUM_INTERVAL_PER_HOUR # has to be a multiple of 60
NUM_INTERVALS_PER_DAY = (24*NUM_INTERVAL_PER_HOUR)
BEGINING_OF_TIME = datetime(2022, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
BUFFER_SIZE_IN_INTERVALS = BUFFER_SIZE_DAYS*NUM_INTERVALS_PER_DAY

def _sanitize_idx(idx):
    # it works for negative values too -2 * 10 => 8
    return idx % BUFFER_SIZE_IN_INTERVALS

class QSHomeConsumptionHistoryAndForecast:

    def __init__(self, home: QSHome | None, storage_path:str=None) -> None:
        self.home = home
        if home is None:
            self.hass = None
        else:
            self.hass = home.hass

        self.home_non_controlled_consumption : QSSolarHistoryVals | None = None
        self._in_reset = False
        if storage_path is None:
            if self.hass is None:
                self.storage_path = None
            else:
                self.storage_path = join(self.hass.config.path(), DOMAIN)
        else:
            self.storage_path = storage_path

        # ok now go through the various spots

    async def dump_for_debug(self, path:str):
        if self.home_non_controlled_consumption is not None:
            file_path = join(path, self.home_non_controlled_consumption.file_name)
            await self.home_non_controlled_consumption.save_values(file_path)

    async def load_from_history(self, entity_id:str, start_time: datetime, end_time: datetime) -> list[LazyState]:

        if self.hass is None:
            return []

        def load_history_from_db(start_time: datetime, end_time: datetime) -> list:
            """Load history from the database."""
            return state_changes_during_period(
                self.hass,
                start_time,
                end_time,
                entity_id,
                include_start_time_state=True,
                no_attributes=True,
            ).get(entity_id, [])

        states : list[LazyState] = await recorder_get_instance(self.hass).async_add_executor_job(load_history_from_db, start_time, end_time)
        # states : list[LazyState] = await self.hass.async_add_executor_job(load_history_from_db, start_time, end_time)
        return states

    async def init_forecasts(self, time: datetime):

        if self._in_reset is False and self.home_non_controlled_consumption is None:
            self.home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
            await self.home_non_controlled_consumption.init(time)

        return not self._in_reset

    def _combine_stored_forecast_values(self, val1, val2, do_add = True) :

        val1_ok_values = np.asarray(val1[1] != 0, dtype=np.int32)
        val2_ok_values = np.asarray(val2[1] != 0, dtype=np.int32)

        diff_day = val1[1] - val2[1]
        val_common = np.asarray(diff_day == 0, dtype=np.int32)

        check_vals = val1_ok_values * val2_ok_values * val_common

        ret = np.zeros_like(val1)
        if do_add:
            ret[0] = (val1[0] + val2[0])*check_vals
        else:
            ret[0] = (val1[0] - val2[0])*check_vals

        ret[1] = (val1[1])*check_vals # take days from val1 only, it will be only the common days anyway ...

        return ret


    async def reset_forecasts(self, time: datetime, light_reset=False):

        self._in_reset = True

        if light_reset is False:


            # compute from history all the values that are not yet computed ... from "true HA" sensors, not QS computed ones

            # start with home consumption

            battery_charge = None

            strt = BEGINING_OF_TIME
            end = time + timedelta(days=1)

            is_one_bad = False

            battery_charge = None
            if self.home._battery is not None:
                if self.home._battery.charge_discharge_sensor:
                    battery_charge = QSSolarHistoryVals(entity_id=self.home._battery.charge_discharge_sensor, forecast=self)
                    s, e = await battery_charge.init(time, for_reset=True)
                    if s is None or e is None:
                        is_one_bad = True
                    else:
                        if s > strt:
                            strt = s
                        if e < end:
                            end = e

            values_for_debug = {}


            solar_production_minus_battery = None
            _LOGGER.info(f"Resetting home consumption 1: is_one_bad {is_one_bad}")
            if is_one_bad is False:
                if self.home._solar_plant is not None:
                    if self.home._solar_plant.solar_inverter_active_power:
                        # this one has the battery inside!
                        solar_production_minus_battery =  QSSolarHistoryVals(entity_id=self.home._solar_plant.solar_inverter_active_power, forecast=self)
                        s, e = await solar_production_minus_battery.init(time, for_reset=True)
                        if s is None or e is None:
                            is_one_bad = True
                        else:
                            if s > strt:
                                strt = s
                            if e < end:
                                end = e
                    elif self.home._solar_plant.solar_inverter_input_active_power:
                        solar_production_minus_battery = QSSolarHistoryVals(entity_id=self.home._solar_plant.solar_inverter_input_active_power, forecast=self)
                        s, e = await solar_production_minus_battery.init(time, for_reset=True)
                        if s is None or e is None:
                            is_one_bad = True
                        else:
                            if s > strt:
                                strt = s
                            if e < end:
                                end = e

                        if is_one_bad is False and battery_charge is not None:
                            solar_production_minus_battery.values = self._combine_stored_forecast_values(solar_production_minus_battery.values, battery_charge.values, do_add=False)

            values_for_debug["solar_minus_battery"] = np.copy(solar_production_minus_battery.values)

            home_consumption = None
            _LOGGER.info(f"Resetting home consumption 2: is_one_bad {is_one_bad}")
            if is_one_bad is False:
                if self.home.grid_active_power_sensor:
                    home_consumption = QSSolarHistoryVals(entity_id=self.home.grid_active_power_sensor, forecast=self)
                    s, e = await home_consumption.init(time, for_reset=True)
                    if s is None or e is None:
                        is_one_bad = True
                    else:
                        if s > strt:
                            strt = s
                        if e < end:
                            end = e
                        if solar_production_minus_battery is None:
                            if self.home.grid_active_power_sensor_inverted is False:
                                home_consumption.values[0] = (-1)*home_consumption.values[0]
                            # else do nothing, it is already in the right format
                        else:
                            # if self.home.grid_active_power_sensor_inverted is False:
                            #    home_consumption.values[0] = solar_production_minus_battery.values[0] - home_consumption.values[0]
                            # else:
                            #    home_consumption.values[0] = solar_production_minus_battery.values[0] + home_consumption.values[0]

                            # when not inverted it means grid consumption is negative : when the house consume from teh grid is it negative : so we need to remove it from solar production
                            # if "inverted, simply add it to solar production
                            values_for_debug["grid"] = np.copy(home_consumption.values)
                            home_consumption.values = self._combine_stored_forecast_values(solar_production_minus_battery.values, home_consumption.values, do_add=self.home.grid_active_power_sensor_inverted)


            _LOGGER.info(f"Resetting home consumption 3: is_one_bad {is_one_bad}")
            if is_one_bad is False and home_consumption is not None:

                #ok we do have the computed home consumption ... now time for the controlled loads

                controlled_power_values = None
                added_controlled = False

                bfs_queue = []
                bfs_queue.extend(self.home._childrens)

                while len(bfs_queue) > 0:
                    device = bfs_queue.pop(0)
                    ha_best_entity_id = None
                    if isinstance(device, QSDynamicGroup):
                        if device.accurate_power_sensor is None:
                            # use the children power consumption if there is no group good power sensor
                            bfs_queue.extend(device._childrens)
                            continue
                        else:
                            ha_best_entity_id = device.accurate_power_sensor
                    else:

                        if not isinstance(device, AbstractLoad):
                            continue

                        if device.load_is_auto_to_be_boosted:
                            continue

                        if not isinstance(device, HADeviceMixin):
                            continue

                        ha_best_entity_id = device.get_best_power_HA_entity()

                    switch_device = None
                    if ha_best_entity_id is None:
                        if isinstance(device, AbstractLoad) and device.switch_entity:
                            ha_best_entity_id = device.switch_entity
                            switch_device = device

                    if ha_best_entity_id is not None:
                        load_sensor = QSSolarHistoryVals(entity_id=ha_best_entity_id, forecast=self)
                        s, e = await load_sensor.init(time, for_reset=True, reset_for_switch_device=switch_device)
                        if s is None or e is None:
                            is_one_bad = True
                            break
                        else:
                            if s > strt:
                                strt = s
                            if e < end:
                                end = e

                        values_for_debug[ha_best_entity_id] = np.copy(load_sensor.values)

                        if controlled_power_values is None:
                            controlled_power_values = load_sensor.values
                        else:
                            controlled_power_values = self._combine_stored_forecast_values(controlled_power_values,load_sensor.values, do_add=True)

                        added_controlled = True
                        _LOGGER.info(f"Resetting home consumption 4: is_one_bad {is_one_bad} load {device.name} {ha_best_entity_id}")


                _LOGGER.info(f"Resetting home consumption 6: is_one_bad {is_one_bad} added_controlled {added_controlled}")
                if is_one_bad is False and added_controlled:
                    values_for_debug["controlled_sum"] = np.copy(controlled_power_values)
                    home_consumption.values = self._combine_stored_forecast_values(home_consumption.values, controlled_power_values, do_add=False)

            if is_one_bad is False and home_consumption is not None:
                _LOGGER.info(f"Resetting home consumption 7: is_one_bad {is_one_bad}")
                # ok we do have now a pretty good idea of the non-controllable house consumption:
                # let's first clean it with the proper start and end
                strt_idx, strt_days = home_consumption.get_index_from_time(strt)
                end_idx, end_days = home_consumption.get_index_from_time(end)
                #rolling buffer cleaning
                if strt_idx == end_idx:
                    # equal
                    is_one_bad = True
                else:
                    last_good_reset = end_idx
                    delta_reset = None
                    for i in range(BUFFER_SIZE_IN_INTERVALS):
                        last_good_common = _sanitize_idx(end_idx - i)
                        if home_consumption.values[1][last_good_common] != 0 and delta_reset is None:
                            last_good_reset = last_good_common
                            delta_reset = i
                            break

                    if delta_reset is not None:
                        _LOGGER.info(f"Resetting home consumption 7: start {strt.astimezone()} end {end.astimezone()}")
                        _LOGGER.info(f"Resetting home consumption 7: RESET: found last good idx {last_good_reset} ({delta_reset} from {end_idx}) for time {home_consumption.get_utc_time_from_index(last_good_reset, home_consumption.values[1][last_good_reset]).astimezone()} ")

                        home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
                        await home_non_controlled_consumption.init(time, for_reset=True)

                        for i in range(30*NUM_INTERVAL_PER_HOUR):

                            idx = _sanitize_idx(last_good_reset - i)

                            if home_consumption.values[1][idx] == 0:
                                continue

                            utc_time = home_consumption.get_utc_time_from_index(idx, home_consumption.values[1][idx])
                            days = home_consumption.values[1][idx]
                            # convert utc time to local time
                            loc_time = utc_time.astimezone()
                            val_reset = home_consumption.values[0][idx]
                            val_real = home_non_controlled_consumption.values[0][idx]
                            _LOGGER.info(f"Resetting home consumption 7: {loc_time} ({idx}/{days}): reset {val_reset} real {val_real}")
                            debug_values = "Resetting home consumption 7:"
                            for k, v in values_for_debug.items():
                                debug_values += f" {k}={v[0][idx]} "
                            _LOGGER.debug(debug_values)




            _LOGGER.info(f"Resetting home consumption 8: is_one_bad {is_one_bad}")
            if is_one_bad is False:
                # now we do have something to save!
                home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
                home_non_controlled_consumption.values = home_consumption.values
                await home_non_controlled_consumption.save_values(for_reset=True)
                _LOGGER.info(f"Resetting home consumption 9: is_one_bad {is_one_bad}")
                self.home_non_controlled_consumption = None

            self._in_reset = False

        else:
            home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=self)
            await home_non_controlled_consumption.init(time, for_reset=True)
            await home_non_controlled_consumption.save_values(for_reset=True)
            _LOGGER.info(f"Resetting home consumption LIGHT")
            self.home_non_controlled_consumption = None

        await self.init_forecasts(time)
        return True



class QSSolarHistoryVals:

    def __init__(self, forecast: QSHomeConsumptionHistoryAndForecast, entity_id:str) -> None:

        self.forecast = forecast
        if forecast is None:
            self.hass = None
            self.home = None
        else:
            self.home = forecast.home
            if self.home is None:
                self.hass = None
            else:
                self.hass =  self.home.hass

        self.entity_id:str = entity_id
        self.values : npt.NDArray | None = None
        self.storage_path : str = forecast.storage_path
        self.file_name = f"{self.entity_id}.npy"
        self.file_path = join(self.storage_path, self.file_name)

        self._current_values : list[tuple[datetime, float|None]] = []
        self._current_idx = None
        self._current_days = None
        self._init_done = False
        self._current_forecast = None
        self._last_forecast_update_time = None


    def update_current_forecast_if_needed(self, time: datetime) -> bool:
        if self._last_forecast_update_time is None or (time - self._last_forecast_update_time) >= timedelta(minutes=INTERVALS_MN):
            return True
        return False

    def get_value_from_current_forecast(self, time: datetime) -> tuple[datetime | None, str | float | None]:
        res = get_value_from_time_series(self._current_forecast, time)
        return res[0], res[1]


    def _get_possible_past_consumption_for_forecast(self, now_idx, now_days, history_in_hours: int, use_val_as_current:float = None):

        end_idx = now_idx
        if use_val_as_current is None:
            end_idx -= 1 # step back one INTERVALS_MN as we want to have a full INTERVALS_MN of history for the current values to select
        end_idx = _sanitize_idx(end_idx)

        start_idx = _sanitize_idx(end_idx - (history_in_hours*NUM_INTERVAL_PER_HOUR))

        current_values, current_days = self._get_values(start_idx, end_idx)

        if use_val_as_current is not None:
            #override the last one with the current value
            current_values = np.copy(current_values)
            current_values[-1] = use_val_as_current
            current_days = np.copy(current_days)
            current_days[-1] = now_days

        if current_values is None or current_days is None:
            _LOGGER.debug("_get_possible_past_consumption_for_forecast no current_values !!!")
            return []

        current_ok_vales = np.asarray(current_days!=0, dtype=np.int32)

        best_past_probes_in_days = [
            1,
            3,
            6,
            7,
            7*4,
            7*8,
            7*12,
            7*51,
            7*52,
            7*53,
        ]
        # after a lot for trials with the compute_prediction_score : RMS is best .. and only 7 days is enough
        best_past_probes_in_days = range(1, 7 + 1)

        scores = []

        for probe_days in best_past_probes_in_days:


            for i in range(-2, 3):
                past_delta = probe_days * NUM_INTERVALS_PER_DAY + i

                if past_delta < history_in_hours*NUM_INTERVAL_PER_HOUR:
                    # not enough history for this probe
                    _LOGGER.debug(
                        f"_get_possible_past_consumption_for_forecast (hist: {history_in_hours}) trash a past match {probe_days} days ago + {i} for bad too small history")
                    continue

                score = self._get_range_score(current_values, current_ok_vales, start_idx, past_delta=past_delta)

                if score:
                    scores.append(score)


        if not scores:
            _LOGGER.info(f"_get_possible_past_consumption_for_forecast (hist: {history_in_hours}) no scores !!!")
            return []

        scores = sorted(scores, key=itemgetter(1))

        return scores

    def _get_range_score(self, current_values, current_ok_vales, start_idx, past_delta, num_score=1):
        past_start_idx = _sanitize_idx(start_idx - past_delta)
        past_start_idx = _sanitize_idx(past_start_idx)
        past_end_idx = _sanitize_idx(past_start_idx + current_values.shape[0] - 1)

        past_values, past_days = self._get_values(past_start_idx, past_end_idx)
        past_ok_values = np.asarray(past_days != 0, dtype=np.int32)

        check_vals = past_ok_values * current_ok_vales
        num_ok_vals = np.sum(check_vals)

        if num_ok_vals < 0.6 * current_values.shape[0]:
            # bad history
            # _LOGGER.debug(
            #    f"_get_range_score trash a past match for bad values {num_ok_vals} - {past_days.shape[0]}")
            return []

        # compute various scores
        res = [past_delta]

        for score_idx in range(num_score):
            if score_idx == 0:
                # rmse
                score = float(np.sqrt(np.sum(np.square(current_values - past_values) * check_vals) / float(num_ok_vals)))
            elif score_idx == 1:
                # mean ratio
                score = float(np.abs(np.sum(past_values*check_vals)/np.sum(current_values*check_vals) - 1.0))*100.0
            elif score_idx == 2:
                # pearson correlation no lag
                c_vals = [current_values[i] for i in range(current_values.shape[0]) if check_vals[i] > 0]
                p_vals = [past_values[i] for i in range(past_values.shape[0]) if check_vals[i] > 0]

                if len(c_vals) != num_ok_vals or len(p_vals) != num_ok_vals:
                    _LOGGER.debug(f"_get_range_score trash a past match for bad values after check {len(c_vals)} - {num_ok_vals}")
                    return []

                # score = self.xcorr_max_pearson(c_vals, p_vals, Lmax=4)[2]
                score = self.xcorr_max_pearson(c_vals, p_vals, Lmax=0)[2]
            elif score_idx == 3:
                # mean bias error
                score = float(np.abs(np.sum((current_values - past_values) * check_vals))) / float(num_ok_vals)
            elif score_idx == 4:
                # mean abs error
                score = float(np.sum(np.abs(current_values - past_values) * check_vals)) / float(num_ok_vals)
            else:
                score = 0.0

            res.append(score)

        return res


    def _get_possible_now_val_for_forcast(self, time_now: datetime):

        now_val, now_duration = self.get_current_non_stored_val_at_time(time_now)
        # try to see if the current INTERVALS_MN, still not stored, can be used to improve the forecast
        if now_duration is not None and now_duration > (INTERVALS_MN*60)//2:
            # take it into account, enough data, even if unstored
            pass
        else:
            now_val = None

        return now_val

    def xcorr_max_pearson(self, x, y, Lmax):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(x)
        # z-score
        sx = x.std(ddof=0)
        sy = y.std(ddof=0)
        if sx == 0 or sy == 0:
            return -1, 0, 1
            # no correlation, worst score
        zx = (x - x.mean()) / sx
        zy = (y - y.mean()) / sy

        best_r = -np.inf
        best_lag = 0

        for lag in range(-Lmax, Lmax + 1):
            if lag >= 0:
                a = zx[:n - lag]
                b = zy[lag:]
            else:
                a = zx[-lag:]
                b = zy[:n + lag]
            if len(a) < 2:  # too few points to compute correlation
                continue

            r = np.corrcoef(a, b)[0, 1]  # Pearson on the overlap
            if r > best_r:
                best_r, best_lag = r, lag

        best_r = float(best_r)
        S = 100.0*(1.0 - ((1 + best_r) / 2.0))  # score [0,100] : 0 is best, 1 is worst
        return best_r, best_lag, S



    def _get_predicted_data(self, future_needed_in_hours: int, now_idx, now_days, scores):

        forecast_values = None
        past_days = None

        num_intervals_to_get = (future_needed_in_hours + 1) * NUM_INTERVAL_PER_HOUR  # add one hour to cover the futur
        num_intervals_pushed = 0

        for item_past in scores:

            probe_num_intervals = item_past[0]

            past_start_idx = _sanitize_idx(now_idx + num_intervals_pushed - probe_num_intervals)
            new_to_get = min(probe_num_intervals - num_intervals_pushed - 1, num_intervals_to_get)
            past_end_idx = _sanitize_idx(past_start_idx + new_to_get)

            c_forecast_values, c_past_days = self._get_values(past_start_idx, past_end_idx)

            c_past_ok_values = np.asarray(c_past_days != 0, dtype=np.int32)
            num_ok_vals = np.sum(c_past_ok_values)

            if num_ok_vals < 0.6 * c_past_days.shape[0]:
                # bad forecast
                _LOGGER.debug(
                    f"_get_predicted_data: trash a forecast for bad values {num_ok_vals} - {past_days.shape[0]}")
                continue

            # ok we will keep it
            num_intervals_pushed += new_to_get
            num_intervals_to_get -= new_to_get

            # fill the forecast values
            if forecast_values is None:
                forecast_values = c_forecast_values
                past_days = c_past_days
            else:
                forecast_values = np.concatenate((forecast_values, c_forecast_values))
                past_days = np.concatenate((past_days, c_past_days))

            if num_intervals_to_get <= 0:
                # we are done
                break

        return forecast_values, past_days

    def compute_prediction_score(self, score_number = 1, num_exploration_days = 3):

        #find the "biggest index" of the stored values
        if self.values is None:
            return

        first_max_day = np.argmax(self.values[1]) # will give the first occurence of the max
        now_idx = first_max_day
        now_days = self.values[1][now_idx]
        while now_days == self.values[1][_sanitize_idx(now_idx + 1)]:
            now_idx = _sanitize_idx(now_idx + 1)

        # ok we have the end of the stored values
        forecast_window = 24*NUM_INTERVAL_PER_HOUR
        past_check_window = 24*NUM_INTERVAL_PER_HOUR

        all_res = []

        num_exploration = num_exploration_days*NUM_INTERVALS_PER_DAY

        all_res_numpy = np.zeros((num_exploration, score_number, 2*score_number + 1),
                                 dtype=np.float64)  # +1 for the day offset


        for dpast in range(num_exploration):

            end_idx = _sanitize_idx(now_idx - dpast)
            start_idx = _sanitize_idx(end_idx - forecast_window - 1)
            current_values, current_days = self._get_values(start_idx, end_idx)
            current_ok_vales = np.asarray(current_days != 0, dtype=np.int32)


            check_end_idx = _sanitize_idx(start_idx - 1)
            check_start_idx = _sanitize_idx(check_end_idx - past_check_window - 1)
            check_current_values, check_current_days = self._get_values(check_start_idx, check_end_idx)
            check_current_ok_vales = np.asarray(check_current_days != 0, dtype=np.int32)


            scores = []
            check_scores = []

            best_past_probes_in_days = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            7 * 4,]
            a = [
                7 * 4,
                7 * 8,
                7 * 12,
                7 * 51,
                7 * 52,
                7 * 53,
            ]

            best_past_probes_in_days = range(1, 7+1)


            for past_days in best_past_probes_in_days:

                for i in range(-2, 3):
                    past_delta = past_days*NUM_INTERVALS_PER_DAY + i
                    # score of the forecast prediction vs real values
                    score = self._get_range_score(current_values, current_ok_vales, start_idx, past_delta=past_delta, num_score=score_number)

                    # score of the "check" to get the forecast
                    check_score = self._get_range_score(check_current_values, check_current_ok_vales, check_start_idx, past_delta=past_delta, num_score=score_number)

                    if check_score and score:
                        check_score.extend(score[1:])
                        scores.append(score)
                        check_scores.append(check_score)



            # 1 is rms sum of abs diff
            # 2 is sum_forecast/sum_real
            # 3 is pearson
            # 4 is mean diff
            # 5 is sum of abs diff
            probe_scores = []
            for score_idx in range(score_number):
                probe_scores.append(sorted(check_scores, key=itemgetter(score_idx+1)))

            res = []
            for score_idx in range(score_number):
                probe = probe_scores[score_idx][0]
                probe[0] = probe[0] / (24*NUM_INTERVAL_PER_HOUR)
                res.append(probe)

                all_res_numpy[dpast, score_idx, :] = probe

            all_res.append(res)

            algo_names = {5: " SumAbsDiff ", 1: " RMS        ", 3: " Pearson    ", 4: " MeanDiff   ", 2: " sumF/sumR  "}

            res_table = np.zeros((score_number, 2*score_number+1), dtype=np.float64)
            for probe_idx in range(score_number):
                vals = all_res_numpy[:dpast + 1, probe_idx, 0]
                res_table[probe_idx, 0] = np.mean(vals)
                for res_idx in range(score_number):
                    vals_forecast = all_res_numpy[:dpast+1, probe_idx, score_number + 1 + res_idx]
                    res_table[probe_idx, res_idx+1] = np.mean(vals_forecast)
                    vals_check = all_res_numpy[:dpast + 1, probe_idx, res_idx + 1]
                    res_table[probe_idx, res_idx + 1 + score_number] = np.mean(vals_check)

            # log the result table
            _LOGGER.info(f"Prediction score table (mean over {dpast} explorations):")
            _LOGGER.info("  predictor |                forecast result                    ")

            columns_description = "            | days         |"
            separator =           "------------|--------------|"
            for probe_idx in range(1, score_number+1):
                columns_description += f" {algo_names[probe_idx]} |"
                separator += "--------------|"

            _LOGGER.info(columns_description)
            _LOGGER.info(separator)
            for probe_idx in range(1, score_number+1):
                line = f"{algo_names[probe_idx]}|"
                line += f" {res_table[probe_idx-1, 0]:12.1f} |"
                for res_idx in range(1, score_number+1):
                    line += f"{res_table[probe_idx-1, res_idx]:6.1f}({res_table[probe_idx-1, res_idx+score_number]:6.1f})|"
                _LOGGER.info(line)

    def compute_now_forecast(self, time_now: datetime, history_in_hours: int, future_needed_in_hours: int, set_as_current=False) -> list[tuple[datetime, float]]:

        _LOGGER.debug("compute_now_forecast called")

        if set_as_current:
            self._last_forecast_update_time = time_now

        now_idx, now_days = self.get_index_from_time(time_now)

        now_val = self._get_possible_now_val_for_forcast(time_now)

        scores = self._get_possible_past_consumption_for_forecast(now_idx, now_days, history_in_hours, use_val_as_current=now_val)

        forecast_values, past_days = self._get_predicted_data(future_needed_in_hours, now_idx, now_days, scores)

        if forecast_values is not None and past_days is not None:

            forecast = []

            # add the best possible matches until we have enough future
            time_now_from_idx = self.get_utc_time_from_index(now_idx, now_days)

            if time_now_from_idx > time_now:
                _LOGGER.warning(f"compute_now_forecast: time_now_from_idx > time_now {time_now_from_idx} > {time_now}")

            prev_val = None

            prev_val_idx = now_idx
            tries = NUM_INTERVAL_PER_HOUR * 2
            while tries > 0:
                prev_val_idx = _sanitize_idx(prev_val_idx - 1)
                if self.values[1][prev_val_idx] != 0:
                    prev_val = self.values[0][prev_val_idx]
                    break
                tries -= 1

            if prev_val is None:
                prev_val = now_val

            # the first forecast value represents now_idx, we will replace it with the current value if we have one, now_val
            # by construction idx, so time_now_from_idx < time_now that has been asked
            if now_val is not None:
                forecast_values = np.copy(forecast_values)
                forecast_values[0] = now_val
                past_days = np.copy(past_days)
                past_days[0] = now_days
            else:
                # we want to be sure the first one has a good value
                if past_days[0] == 0:
                    # ok le's get the last good value before
                    if prev_val is not None:
                        forecast_values = np.copy(forecast_values)
                        forecast_values[0] = prev_val
                        past_days = np.copy(past_days)
                        past_days[0] = now_days
                    else:
                        _LOGGER.error("compute_now_forecast no prev_val !!!!")
                        return []

            if time_now < time_now_from_idx + timedelta(minutes=INTERVALS_MN//2):
                # we are before the middle of the current INTERVALS_MN, so we want to have a value before time_now
                # and this value can be real
                if prev_val is None:
                    prev_val = forecast_values[0]
                forecast.append((time_now_from_idx - timedelta(minutes=INTERVALS_MN - INTERVALS_MN//2), prev_val))

            for i in range(past_days.shape[0]):
                if past_days[i] == 0:
                    continue
                # adding INTERVALS_MN//2 has we have a mean on INTERVALS_MN
                forecast_time = time_now_from_idx+timedelta(minutes=i*INTERVALS_MN + INTERVALS_MN//2)
                forecast.append((forecast_time, forecast_values[i]))

            # complement with the future if there is not enough data at the end
            if forecast[-1][0] < time_now + timedelta(hours=future_needed_in_hours):
                forecast.append((time_now + timedelta(hours=future_needed_in_hours), forecast[-1][1]))

            _LOGGER.debug(f"compute_now_forecast A GOOD ONE  {past_days.shape[0]}")

            if set_as_current:
                self._current_forecast = forecast

            return forecast

        _LOGGER.debug("compute_now_forecast nothing works!")
        return []



    def _get_values(self, start_idx, end_idx):
        if self.values is None:
            _LOGGER.info(f"NO VALUES IN _get_values")
            return None, None

        start_idx = _sanitize_idx(start_idx)
        end_idx = _sanitize_idx(end_idx)

        if end_idx > start_idx:
            return self.values[0][start_idx:end_idx+1], self.values[1][start_idx:end_idx+1]
        else:
            return np.concatenate((self.values[0][start_idx:], self.values[0][:end_idx+1])), np.concatenate((self.values[1][start_idx:], self.values[1][:end_idx+1]))


    async def save_values(self, file_path: str = None, for_reset: bool=False) -> None:
            def _save_values_to_file(path: str, values) -> None:
                """Write numpy."""
                try:
                    np.save(path, values)
                    _LOGGER.info(f"Write numpy SUCCESS for {path} for reset {for_reset}")
                except:
                    _LOGGER.info(f"Write numpy FAILED for {path} for reset {for_reset}")
                    pass

            if file_path is None:
                file_path = self.file_path

            if self.hass is None:
                _save_values_to_file(file_path, self.values)
            else:
                await self.hass.async_add_executor_job(
                    _save_values_to_file, file_path, self.values
                )

    def read_value(self):

        try:
            ret = np.load(self.file_path)
        except:
            ret = None

        return ret

    async def read_values_async(self):

            def _load_values_from_file(path: str) -> Any | None:
                """Read numpy."""
                ret = None
                try:
                    ret =  np.load(path)
                except:
                    ret = None

                return ret

            if self.hass is None:
                return _load_values_from_file(self.file_path)
            else:
                return await self.hass.async_add_executor_job(
                    _load_values_from_file, self.file_path)

    def is_time_in_current_interval(self, time: datetime) -> bool:

        if self._current_idx is None or self._current_days is None:
            return False

        idx, days = self.get_index_from_time(time)

        if idx == self._current_idx and days == self._current_days:
            return True

        return False

    def get_current_interval_value(self) -> tuple[datetime | None, float | None]:

        if self._current_values and self._current_idx is not None and self._current_days is not None:
            from_time = self.get_utc_time_from_index(self._current_idx, self._current_days)
            up_to_time = from_time + timedelta(minutes=INTERVALS_MN)

            nv = get_average_time_series(self._current_values, first_timing=from_time, last_timing=up_to_time)

            return from_time, nv

        return None, None


    def _cache_current_vals(self, extend_but_not_cover_idx=None) -> bool:

        done_something = False
        from_time, nv = self.get_current_interval_value()

        if from_time is not None:
            done_something = True
            if self.values is None:
                self.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)


            self.values[0][self._current_idx] = nv
            self.values[1][self._current_idx] = self._current_days

            if extend_but_not_cover_idx is not None and extend_but_not_cover_idx != _sanitize_idx(self._current_idx + 1):
                # used to make the ring buffer contiguous, in case there was a hole
                if extend_but_not_cover_idx <= self._current_idx:
                    # we have circled around the ring buffer
                    extend_but_not_cover_idx = extend_but_not_cover_idx + BUFFER_SIZE_IN_INTERVALS

                last_value = self._current_values[-1][1]

                for delta in range(1, extend_but_not_cover_idx - self._current_idx):
                    new_idx, new_days = self.get_index_with_delta(self._current_idx, self._current_days, delta)
                    self.values[0][new_idx] = last_value
                    self.values[1][new_idx] = new_days

        return done_something

    def store_and_flush_current_vals(self, extend_but_not_cover_idx=None) -> bool:

        done_something = self._cache_current_vals(extend_but_not_cover_idx=extend_but_not_cover_idx)

        self._current_values = []
        self._current_idx = None
        self._current_days = None

        return done_something

    def get_current_non_stored_val_at_time(self, time: datetime) -> tuple[float | None, float | None]:

        if self._current_idx is None or self._current_values is None or len(self._current_values) == 0:
            return None, None

        idx, days = self.get_index_from_time(time)

        if idx == self._current_idx and days == self._current_days:
            mean_v = get_average_time_series(self._current_values, last_timing=time)
            duration_s = (time - self._current_values[0][0]).total_seconds()
            return mean_v, duration_s

        return None, None

    def get_closest_stored_value(self, time: datetime) -> tuple[datetime | None, str | float | None]:

        if self.values is None:
            return None, None

        val = None
        time_out = None

        idx, days = self.get_index_from_time(time)

        if self._current_idx is not None and _sanitize_idx(self._current_idx + 1) == idx:
            time_out, val = self.get_current_interval_value()
        elif idx == self._current_idx and days == self._current_days:
            # get the previous one
            idx = _sanitize_idx(idx - 1)

        if val is None:
            v = self.values[0][idx]
            d = self.values[1][idx]
            if d > 0 and (d == days or d == days - 1) and v is not None:
                val = v
                time_out = self.get_utc_time_from_index(idx, d)

        if val is None:
            time_out = None

        return time_out, val


    def add_value(self, time: datetime, value: float) -> bool:

        something_done  = False

        idx, days = self.get_index_from_time(time) # idx is % BUFFER_SIZE_IN_INTERVALS, begining of the interval minutes

        if self._current_idx is None:
            self._current_idx = idx
            self._current_days = days
            self._current_values = []

        if self._current_idx != idx or self._current_days != days:
            something_done = self._cache_current_vals(extend_but_not_cover_idx=idx)
            self._current_idx = idx
            self._current_days = days
            self._current_values = [(time, value)]
        else:
            self._current_values.append((time, value))

        return something_done


    def get_index_from_time(self, time: datetime) -> tuple[int, int]:
        days = (time - BEGINING_OF_TIME).days
        idx = int((time-BEGINING_OF_TIME).total_seconds()//(60*INTERVALS_MN))%(BUFFER_SIZE_IN_INTERVALS)
        return idx, days

    def get_index_with_delta(self, idx:int, days:int, delta_idx:int) -> tuple[int, int]:
        init_time = self.get_utc_time_from_index(idx, days)
        next_time = init_time + timedelta(minutes=delta_idx*INTERVALS_MN)
        return  self.get_index_from_time(next_time)


    def get_utc_time_from_index(self, idx:int, days:int) -> datetime:
        days_index = int((days*24*60)//INTERVALS_MN)%(BUFFER_SIZE_IN_INTERVALS)
        if idx >= days_index:
            num_intervals = int(idx) - days_index
            return BEGINING_OF_TIME + timedelta(days=int(days)) + timedelta(minutes=int(num_intervals * INTERVALS_MN))
        else:
            num_intervals = (BUFFER_SIZE_IN_INTERVALS - days_index + idx)
            return BEGINING_OF_TIME + timedelta(days=int(days-1)) + timedelta(minutes=int(num_intervals * INTERVALS_MN))


    async def init(self, time:datetime, for_reset:bool = False, reset_for_switch_device:AbstractLoad | None = None) -> tuple[datetime | None, datetime | None]:

        if self._init_done:
            return None, None

        self._init_done = True

        if for_reset is False and self.values is not None:
            return None, None

        now_idx, now_days = self.get_index_from_time(time)
        time_now_idx = self.get_utc_time_from_index(now_idx, now_days)

        if for_reset:
            self.values = None
        else:
            await aiofiles.os.makedirs(self.storage_path, exist_ok=True)
            self.values = await self.read_values_async()

        last_bad_idx = None
        num_slots_before_now_idx = 0
        do_save = False
        if self.values is not None:

            if self.values.shape[0] != 2 or self.values.shape[1] != BUFFER_SIZE_IN_INTERVALS:
                _LOGGER.warning("Error loading forecast values for %s shape %s, resetting", self.entity_id, self.values.shape)
                self.values = None

        if self.values is not None:

            last_bad_idx = now_idx

            while True:

                try_prev_idx = last_bad_idx - 1
                if try_prev_idx < 0:
                    try_prev_idx = self.values.shape[1] - 1

                if self.values[1][try_prev_idx] == 0 or now_days - self.values[1][try_prev_idx] >= BUFFER_SIZE_DAYS:
                    last_bad_idx = try_prev_idx
                    num_slots_before_now_idx += 1
                else:
                    break

            # not sure why we would need to reset all in case we have a 1 week gap
            # if num_slots_before_now_idx*60*INTERVALS_MN > MAX_SENSOR_HISTORY_S:
            #    last_bad_idx = None
        else:
            do_save = True
            self.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)


        end_time = time
        if last_bad_idx is not None:
            start_time = time_now_idx - timedelta(minutes=(num_slots_before_now_idx)*INTERVALS_MN)
        else:
            start_time = time - timedelta(days=BUFFER_SIZE_DAYS-1)

        states : list[LazyState] = await self.forecast.load_from_history(self.entity_id, start_time, end_time)

        # states : ordered LazySates
        # we will fill INTERVALS_IN_MN slots with the mean of the values in the state
        # the last remaining for the current one will be put as current value
        self._current_idx = None
        self._current_days = None
        history_start = None
        history_end = None

        if states:

            # in states we have "LazyState" objects ... with no attribute
            # bewarei n homeassitant the states will "only change" meaning if ther eis no change, no measure, it is VERY important
            # to factor that in teh add value below!
            real_state = self.hass.states.get(self.entity_id)
            if real_state is None:
                state_attr = {}
            else:
                state_attr = real_state.attributes

            history_start = states[0].last_changed
            history_end = states[-1].last_changed

            for s in states:
                if s is None:
                    continue
                if s.last_changed < start_time:
                    continue
                if s.last_changed > end_time:
                    continue

                value = None
                if s.state is  None or s.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                    # we may forget what is between this "wrong measure' and the time before .. ti be seen if we don't use the last good value
                    pass
                else:
                    try:
                        if reset_for_switch_device is None:
                            value = float(s.state)
                        else:
                            value = float(reset_for_switch_device.get_power_from_switch_state(s.state))
                    except:
                        value = None
                        _LOGGER.warning("Error loading lazy safe value for %s state %s", self.entity_id, s.state)
                        # is it the same as a bad state above?

                if value is not None:
                    value, _ = convert_power_to_w(value, state_attr)

                if value is not None:
                    if self.add_value(s.last_changed, value):
                        # need to save if one value added, only do  it once at the end
                        do_save = True
                else:

                    if self.is_time_in_current_interval(s.last_changed):
                        #ok the current interval covers this, forget this bad value:
                        pass
                    else:
                        #forget history before this bad value
                        if self.store_and_flush_current_vals():
                            do_save = True

                    # possibly a wrong state
                    _LOGGER.warning("Error loading lazy safe value for %s", self.entity_id)

            if self._current_idx is not None and self._current_idx != now_idx:
                if self.store_and_flush_current_vals():
                    do_save = True

            self._current_idx = None
            self._current_days = None

            if for_reset is False and do_save:
                await self.save_values()

        return history_start, history_end




































