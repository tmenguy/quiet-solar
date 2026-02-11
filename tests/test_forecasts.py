import asyncio
import os
import pickle
from datetime import datetime, timedelta
from unittest import TestCase

import logging, sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d - %(message)s"

root = logging.getLogger()
if not root.handlers:  # avoid duplicate handlers on re-run
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)

root.setLevel(logging.DEBUG)


import pytz

import numpy as np

from custom_components.quiet_solar.const import FLOATING_PERIOD_S, CONSTRAINT_TYPE_MANDATORY_END_TIME, \
    CONSTRAINT_TYPE_FILLER_AUTO, \
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE, CONF_HOME_START_OFF_PEAK_RANGE_1, CONF_HOME_END_OFF_PEAK_RANGE_1, \
    CONF_HOME_START_OFF_PEAK_RANGE_2, CONF_HOME_END_OFF_PEAK_RANGE_2, CONF_HOME_PEAK_PRICE, CONF_HOME_OFF_PEAK_PRICE, \
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN, DOMAIN, FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER
from custom_components.quiet_solar.ha_model.home import QSHomeConsumptionHistoryAndForecast, BUFFER_SIZE_IN_INTERVALS, \
    INTERVALS_MN, \
    BUFFER_SIZE_DAYS, QSHome, QSSolarHistoryVals
from custom_components.quiet_solar.ha_model.solar import QSSolarProvider, QSSolarProviderSolcastDebug
from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.commands import copy_command, CMD_AUTO_GREEN_ONLY, CMD_ON
from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint, \
    MultiStepsPowerLoadConstraintChargePercent, LoadConstraint, TimeBasedSimplePowerLoadConstraint
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import PeriodSolver


root = logging.getLogger()
if not root.handlers:  # avoid duplicate handlers on re-run
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)

root.setLevel(logging.DEBUG)

def _util_constraint_save_dump(time, cs):
    dc_dump = cs.to_dict()
    load = cs.load
    cs_load = LoadConstraint.new_from_saved_dict(time, load, dc_dump)
    assert cs == cs_load

class TestForecast(TestCase):

    def test_read_solar(self):

        # This is a simple test to check if the forecast is working correctly
        # The forecast is a simple function that returns the sum of two numbers
        # The test checks if the sum of 1 and 2 is 3

        test_folder = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(test_folder, "data", "2024_20_8_before_reset", "solar_forecast.pickle")

        solar_provider = QSSolarProviderSolcastDebug(file_path)


        assert solar_provider.solar_forecast

        assert len(solar_provider.solar_forecast) >= 2

        assert   (solar_provider.solar_forecast[-1][0] - solar_provider.solar_forecast[0][0]).total_seconds() >= FLOATING_PERIOD_S


    def test_consumption_data_storage(self):

        async def _async_test():
            test_folder = os.path.dirname(os.path.realpath(__file__))
            conso_path = os.path.join(test_folder, "scratchpad", "test_storage")


            time = datetime(year=2024, month=8, day=20, hour=10, minute=5, second=30, microsecond=0, tzinfo=pytz.UTC)

            # nothing is here
            conso = QSHomeConsumptionHistoryAndForecast(home=None, storage_path=conso_path)
            conso.home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=conso)
            await conso.home_non_controlled_consumption.init(time)

            now_idx, day_idx = conso.home_non_controlled_consumption.get_index_from_time(time)
            t = conso.home_non_controlled_consumption.get_utc_time_from_index(now_idx, day_idx)
            assert t == datetime(year=2024, month=8, day=20, hour=10, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)

            # now we will add data
            conso.home_non_controlled_consumption._current_idx = None
            for num_minutes in range((BUFFER_SIZE_IN_INTERVALS + 1000)*INTERVALS_MN):
                t = time + timedelta(minutes=num_minutes)
                conso.home_non_controlled_consumption.add_value(t, 1000)

            mi = np.min(conso.home_non_controlled_consumption.values[0])
            assert mi == 1000.0

            mx = np.max(conso.home_non_controlled_consumption.values[0])
            assert mx == 1000.0

            mi = np.min(conso.home_non_controlled_consumption.values[1])
            assert mi >= day_idx

            mx = np.max(conso.home_non_controlled_consumption.values[1])
            assert mx >= day_idx + BUFFER_SIZE_DAYS

            assert mx - mi == BUFFER_SIZE_DAYS


            pass

        asyncio.run(_async_test())


    def test_read_passed_consumption_data(self):

        async def _async_test():
            test_folder = os.path.dirname(os.path.realpath(__file__))
            conso_path = os.path.join(test_folder, "data", "2024_20_8_before_reset")
            debug_conf_file = os.path.join(conso_path, "debug_conf.pickle")

            debug_conf = None
            if os.path.isfile(debug_conf_file):
                with open(debug_conf_file, 'rb') as file:
                    debug_conf = pickle.load(file)

            time = debug_conf["now"]


            kwargs = {
                CONF_HOME_START_OFF_PEAK_RANGE_1: "14:10:00",
                CONF_HOME_END_OFF_PEAK_RANGE_1: "17:10:00",
                CONF_HOME_START_OFF_PEAK_RANGE_2:"02:10:00",
                CONF_HOME_END_OFF_PEAK_RANGE_2 : "07:10:00",
                CONF_HOME_PEAK_PRICE : 0.27,
                CONF_HOME_OFF_PEAK_PRICE : 0.2068,
            }

            home = QSHome(hass=None, config_entry=None, name="test_home", **kwargs)

            conso = QSHomeConsumptionHistoryAndForecast(home=None, storage_path=conso_path)
            conso.home_non_controlled_consumption = QSSolarHistoryVals(entity_id=FULL_HA_SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, forecast=conso)
            await conso.home_non_controlled_consumption.init(time)

            conso_forecast = conso.home_non_controlled_consumption.compute_now_forecast(time_now=time, history_in_hours=24, future_needed_in_hours=FLOATING_PERIOD_S // 3600)

            assert conso_forecast[-1][0] >= time + timedelta(seconds=FLOATING_PERIOD_S)

            assert conso_forecast[0][0] <= time

            solar_file_path = os.path.join(test_folder, "data", "2024_20_8_before_reset", "solar_forecast.pickle")

            solar_provider = QSSolarProviderSolcastDebug(solar_file_path)

            solar_forecast = solar_provider.get_forecast(start_time=time, end_time=time + timedelta(seconds=FLOATING_PERIOD_S))

            assert solar_forecast[0][0] <= time

            assert solar_forecast[-1][0] >= time + timedelta(seconds=FLOATING_PERIOD_S - 10*60)




            for j in range(3):


                steps = []
                for a in range(7, 32 + 1):
                    steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=a*230*3))

                charger = TestLoad(name="charger",
                                   min_a=7,
                                   max_a=32,
                                   min_p=steps[0].power_consign,
                                   max_p=steps[-1].power_consign)

                cumulus = TestLoad(name="cumulus",
                                   min_a=6.5,
                                   max_a=6.5,
                                   min_p=1500,
                                   max_p=1500)


                charge_mandatory_end = time + timedelta(hours=11)
                cumulus_end = time + timedelta(hours=8)
                charge_manual_end = time + timedelta(minutes=5)
                car_capacity = 22000
                target_mandatory = 80
                target_best_effort = 100
                target_manual = 100
                cumulus_target_s = 3600*3

                car_charge_mandatory = MultiStepsPowerLoadConstraintChargePercent(
                    time=time,
                    type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                    total_capacity_wh=car_capacity,
                    load=charger,
                    end_of_constraint=charge_mandatory_end,
                    initial_value=0,
                    target_value=target_mandatory,
                    power_steps=steps,
                    support_auto=True,
                )
                charger.push_live_constraint(time, car_charge_mandatory)
                _util_constraint_save_dump(time, car_charge_mandatory)
                car_charge_as_best = MultiStepsPowerLoadConstraintChargePercent(
                    time=time,
                    type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
                    total_capacity_wh=car_capacity,
                    load=charger,
                    initial_value=None,
                    target_value=target_best_effort,
                    power_steps=steps,
                    support_auto=True,
                )
                charger.push_live_constraint(time, car_charge_as_best)
                _util_constraint_save_dump(time, car_charge_as_best)



                if j == 1:
                    car_charge_manual = MultiStepsPowerLoadConstraintChargePercent(
                        time=time,
                        type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
                        total_capacity_wh=car_capacity,
                        load=charger,
                        from_user=True,
                        end_of_constraint=time,
                        initial_value=0,
                        target_value=target_manual,
                        power_steps=steps,
                        support_auto=True,
                    )
                    charger.push_live_constraint(time, car_charge_manual)
                    _util_constraint_save_dump(time, car_charge_manual)
                    charge_manual_end = car_charge_manual.end_of_constraint
                elif j == 2:
                    load_mandatory = TimeBasedSimplePowerLoadConstraint(
                        type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
                        time=time,
                        load=cumulus,
                        from_user=False,
                        end_of_constraint=cumulus_end,
                        power=1500,
                        initial_value=0,
                        target_value=cumulus_target_s,
                    )
                    cumulus.push_live_constraint(time, load_mandatory)
                    _util_constraint_save_dump(time, load_mandatory)



                battery = Battery(name="battery")
                battery.max_charging_power = 10500
                battery.max_discharging_power = 10500
                battery.capacity = 21000
                battery._current_charge_value = 10000

                end_time = time + timedelta(seconds=FLOATING_PERIOD_S)
                tarrifs = home.get_tariffs(start_time=time, end_time=end_time)

                if j in [0,1]:
                    actionnable_loads = [charger]
                else:
                    actionnable_loads = [charger, cumulus]

                s = PeriodSolver(
                    start_time=time,
                    end_time=end_time,
                    tariffs=tarrifs,
                    actionable_loads=actionnable_loads,
                    battery=battery,
                    pv_forecast=solar_forecast,
                    unavoidable_consumption_forecast=conso_forecast
                )
                cmds, battery_comands = s.solve(with_self_test=True)

                assert cmds is not None

                if j in [0,1]:
                    assert len(cmds) == 1
                    cmds_charger = cmds[0][1]
                    cmds_cumulus = []
                else:
                    assert len(cmds) == 2
                    if  cmds[0][0].name == charger.name:
                        cmds_charger = cmds[0][1]
                        cmds_cumulus = cmds[1][1]
                    else:
                        cmds_charger = cmds[1][1]
                        cmds_cumulus = cmds[0][1]
                        assert len(cmds_cumulus) > 1

                assert len(cmds_charger) > 1


                target_energy_mandatory = car_capacity * target_mandatory / 100.0
                target_energy_best_effort = car_capacity * target_best_effort / 100.0
                target_energy_manual = car_capacity * target_manual / 100.0

                target_cumulus = cumulus_target_s

                if j == 2:
                    for i in range(1, len(cmds_cumulus)):
                        cmd_duration = (cmds_cumulus[i][0] - cmds_cumulus[i - 1][0]).total_seconds()
                        cmd_added_energy = (cmds_cumulus[i - 1][1].power_consign * cmd_duration) / 3600.0

                        if cmds_cumulus[i][0] <= cumulus_end:
                            if cmds_cumulus[i - 1][1].is_like(CMD_ON):
                                target_cumulus -= cmd_duration

                        if target_cumulus > 0 and cmds_cumulus[i][0] > cumulus_end:
                            assert False

                    assert target_cumulus <= 0


                for i in range(1, len(cmds_charger)):
                    cmd_duration = (cmds_charger[i][0] - cmds_charger[i-1][0]).total_seconds()
                    cmd_added_energy = (cmds_charger[i - 1][1].power_consign * cmd_duration) / 3600.0

                    if cmds_charger[i][0] <= charge_manual_end:
                        target_energy_manual -= cmd_added_energy

                    if cmds_charger[i][0] <= charge_mandatory_end:
                        target_energy_mandatory -= cmd_added_energy

                    if j == 2:
                        if cmds_charger[i][0] <= cumulus_end:
                            if cmds_charger[i-1][1].is_like(CMD_ON):
                                target_cumulus -= cmd_duration

                    target_energy_best_effort -= cmd_added_energy

                    if j == 1:
                        if target_energy_manual > 0 and cmds_charger[i][0] > charge_manual_end:
                            assert False

                    if target_energy_mandatory > 0 and cmds_charger[i][0] > charge_mandatory_end:
                        assert False

                if j == 1:
                    assert target_energy_manual <= 0
                assert target_energy_mandatory <= 0
                assert target_energy_best_effort <= 0


        asyncio.run(_async_test())





