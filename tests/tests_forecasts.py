import asyncio
import os
import pickle
from datetime import datetime, timedelta
from unittest import TestCase

import pytz

import numpy as np

from quiet_solar.const import FLOATING_PERIOD_S
from quiet_solar.ha_model.home import QSHomeConsumptionHistoryAndForecast, BUFFER_SIZE_IN_INTERVALS, INTERVALS_MN, \
    BUFFER_SIZE_DAYS
from quiet_solar.ha_model.solar import QSSolarProvider, QSSolarProviderSolcastDebug
from quiet_solar.home_model.battery import Battery
from quiet_solar.home_model.commands import copy_command, CMD_AUTO_GREEN_ONLY
from quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint, \
    MultiStepsPowerLoadConstraintChargePercent, LoadConstraint
from quiet_solar.home_model.load import TestLoad
from quiet_solar.home_model.solver import PeriodSolver

def test_constraint_save_dump(cs):
    dc_dump = cs.to_dict()
    load = cs.load
    cs_load = LoadConstraint.new_from_saved_dict(load, dc_dump)
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
            await conso.init_forecasts(time)

            now_idx, day_idx = conso.home_non_controlled_consumption.get_index_from_time(time)
            t = conso.home_non_controlled_consumption.get_utc_time_from_index(now_idx, day_idx)
            assert t == datetime(year=2024, month=8, day=20, hour=10, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)

            # now we will add data
            conso.home_non_controlled_consumption._current_idx = None
            for num_minutes in range((BUFFER_SIZE_IN_INTERVALS + 1000)*INTERVALS_MN):
                t = time + timedelta(minutes=num_minutes)
                await conso.home_non_controlled_consumption.add_value(t, 1000, False)

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
            conso_path_file = os.path.join(conso_path, "qs_home_non_controlled_consumption_power.npy")
            debug_conf_file = os.path.join(conso_path, "debug_conf.pickle")

            debug_conf = None
            if os.path.isfile(debug_conf_file):
                with open(debug_conf_file, 'rb') as file:
                    debug_conf = pickle.load(file)

            time = debug_conf["now"]

            conso = QSHomeConsumptionHistoryAndForecast(home=None, storage_path=conso_path)
            await conso.init_forecasts(time)

            conso_forecast = await conso.home_non_controlled_consumption.get_forecast(time_now=time, history_in_hours=24, futur_needed_in_hours=FLOATING_PERIOD_S//3600)

            assert conso_forecast[-1][0] >= time + timedelta(seconds=FLOATING_PERIOD_S)

            assert conso_forecast[0][0] <= time

            solar_file_path = os.path.join(test_folder, "data", "2024_20_8_before_reset", "solar_forecast.pickle")

            solar_provider = QSSolarProviderSolcastDebug(solar_file_path)

            solar_forecast = solar_provider.get_forecast(start_time=time, end_time=time + timedelta(seconds=FLOATING_PERIOD_S))

            assert solar_forecast[0][0] <= time

            assert solar_forecast[-1][0] >= time + timedelta(seconds=FLOATING_PERIOD_S - 10*60)

            tarrifs = [(timedelta(hours=0), 0.27/1000.0),
                       (timedelta(hours=2, minutes=10), 0.2068/1000.0),
                       (timedelta(hours=7, minutes=10), 0.27/1000.0),
                       (timedelta(hours=14, minutes=10), 0.2068/1000.0),
                       (timedelta(hours=17, minutes=10), 0.27/1000.0)
                       ]

            charger = TestLoad(name="charger")

            steps = []
            for a in range(7, 32 + 1):
                steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=a*230*3))

            charge_mandatory_end = time + timedelta(hours=11)
            car_capacity = 22000
            target_mandatory = 80
            target_best_effort = 100

            car_charge_mandatory = MultiStepsPowerLoadConstraintChargePercent(
                    total_capacity_wh=car_capacity,
                    load=charger,
                    mandatory=True,
                    end_of_constraint=charge_mandatory_end,
                    initial_value=0,
                    target_value=target_mandatory,
                    power_steps=steps,
                    support_auto=True,
            )
            charger.push_live_constraint(car_charge_mandatory)
            test_constraint_save_dump(car_charge_mandatory)
            car_charge_as_best = MultiStepsPowerLoadConstraintChargePercent(
                    total_capacity_wh=car_capacity,
                    load=charger,
                    mandatory=False,
                    end_of_constraint=None,
                    initial_value=0,
                    target_value=target_best_effort,
                    power_steps=steps,
                    support_auto=True,
            )
            charger.push_live_constraint(car_charge_as_best)
            test_constraint_save_dump(car_charge_as_best)

            battery = Battery(name="battery")
            battery.max_charging_power = 10500
            battery.max_discharging_power = 10500
            battery.capacity = 21000
            battery._current_charge_value = 10000

            s = PeriodSolver(
                start_time=time,
                end_time=time + timedelta(seconds=FLOATING_PERIOD_S),
                tariffs=tarrifs,
                actionable_loads=[charger],
                battery=battery,
                pv_forecast=solar_forecast,
                unavoidable_consumption_forecast=conso_forecast
            )
            cmds, battery_comands = s.solve()

            assert cmds is not None

            cmds_charger = cmds[0][1]

            assert len(cmds_charger) > 0


            target_energy_mandatory = car_capacity * target_mandatory / 100.0
            target_energy_best_effort = car_capacity * target_best_effort / 100.0

            for i in range(1, len(cmds_charger)):
                cmd_duration = (cmds_charger[i][0] - cmds_charger[i-1][0]).total_seconds()
                cmd_added_energy = (cmds_charger[i - 1][1].power_consign * cmd_duration) / 3600.0

                if cmds_charger[i][0] <= charge_mandatory_end:
                    target_energy_mandatory -= cmd_added_energy

                target_energy_best_effort -= cmd_added_energy

                if target_energy_mandatory > 0 and cmds_charger[i][0] > charge_mandatory_end:
                    assert False


            assert target_energy_mandatory <= 0
            assert target_energy_best_effort <= 0


        asyncio.run(_async_test())





