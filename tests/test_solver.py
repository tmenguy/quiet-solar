from unittest import TestCase

import pytz

from quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint, TimeBasedSimplePowerLoadConstraint
from quiet_solar.home_model.load import TestLoad
from quiet_solar.home_model.solver import PeriodSolver
from datetime import datetime
from datetime import timedelta

from quiet_solar.home_model.commands import LoadCommand


class TestSolver(TestCase):

    def test_solve(self):
        # This is a simple test to check if the solver is working correctly
        # The solver is a simple function that returns the sum of two numbers
        # The test checks if the sum of 1 and 2 is 3

        dt = datetime(year=2024, month=6, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)

        start_time = dt
        end_time = dt + timedelta(days=1)

        tarrifs = [( 0.27, timedelta(hours=0)),
                   (0.2068, timedelta(hours=2, minutes=10)),
                   (0.27, timedelta(hours=7, minutes=10)),
                   (0.2068, timedelta(hours=14, minutes=10)),
                   (0.27, timedelta(hours=17, minutes=10))
                   ]

        car = TestLoad(name="car")
        pool = TestLoad(name="pool")
        cumulus_parents = TestLoad(name="cumulus_parents")
        cumulus_children = TestLoad(name="cumulus_children")

        dt + timedelta(hours=1)

        car_steps = []
        for a in range(7, 33):
            car_steps.append(LoadCommand(command="ON_WITH_VAL", power_consign=a * 3 * 230, param=a))

        car_charge_mandatory = MultiStepsPowerLoadConstraint(
            load = car,
            mandatory = True,
            end_of_constraint = dt + timedelta(hours=11),
            initial_value = 10000,
            target_value = 16000,
            power_steps = car_steps,
            support_auto = True
        )
        car.push_live_constraint(car_charge_mandatory)

        car_charge_best_effort = MultiStepsPowerLoadConstraint(
            load = car,
            mandatory = False,
            end_of_constraint = None,
            initial_value = 10000,
            target_value = 22000,
            power_steps = car_steps,
            support_auto = True
        )
        car.push_live_constraint(car_charge_best_effort)


        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            load = pool,
            mandatory = True,
            end_of_constraint = dt + timedelta(hours=23),
            initial_value = 0,
            target_value = 10*3600,
            power = 1430,
        )
        pool.push_live_constraint(pool_constraint)

        cumulus_parents_constraint = TimeBasedSimplePowerLoadConstraint(
            load = cumulus_parents,
            mandatory = True,
            end_of_constraint = dt + timedelta(hours=20),
            initial_value = 0,
            target_value = 3*3600,
            power = 2000,
        )
        cumulus_parents.push_live_constraint(cumulus_parents_constraint)

        cumulus_children_constraint = TimeBasedSimplePowerLoadConstraint(
            load = cumulus_children,
            mandatory = False,
            end_of_constraint = dt + timedelta(hours=17),
            initial_value = 0,
            target_value = 1*3600,
            power = 2000,
        )
        cumulus_children.push_live_constraint(cumulus_children_constraint)

        unavoidable_consumption_forecast = [(300, dt),
                       (300, dt + timedelta(hours=1)),
                       (300, dt + timedelta(hours=2)),
                       (300, dt + timedelta(hours=3)),
                       (380, dt + timedelta(hours=4)),
                       (300, dt + timedelta(hours=5)),
                       (300, dt + timedelta(hours=6)),
                       (390, dt + timedelta(hours=7)),
                       (900, dt + timedelta(hours=8)),
                       (700, dt + timedelta(hours=9)),
                       (400, dt + timedelta(hours=10)),
                       (600, dt + timedelta(hours=11)),
                       (1300, dt + timedelta(hours=12)),
                       (500, dt + timedelta(hours=13)),
                       (800, dt + timedelta(hours=14)),
                       (400, dt + timedelta(hours=15)),
                       (1200, dt + timedelta(hours=16)),
                       (500, dt + timedelta(hours=17)),
                       (500, dt + timedelta(hours=18)),
                       (800, dt + timedelta(hours=19)),
                       (800, dt + timedelta(hours=20)),
                       (700, dt + timedelta(hours=21)),
                       (650, dt + timedelta(hours=22)),
                       (450, dt + timedelta(hours=23))
                       ]

        pv_forecast = [(0, dt),
                       (0, dt + timedelta(hours=1)),
                       (0, dt + timedelta(hours=2)),
                       (0, dt + timedelta(hours=3)),
                       (0, dt + timedelta(hours=4)),
                       (0, dt + timedelta(hours=5)),
                       (690, dt + timedelta(hours=6)),
                       (2250, dt + timedelta(hours=7)),
                       (2950, dt + timedelta(hours=8)),
                       (5760, dt + timedelta(hours=9)),
                       (8210, dt + timedelta(hours=10)),
                       (9970, dt + timedelta(hours=11)),
                       (9760, dt + timedelta(hours=12)),
                       (9840, dt + timedelta(hours=13)),
                       (7390, dt + timedelta(hours=14)),
                       (8420, dt + timedelta(hours=15)),
                       (9360, dt + timedelta(hours=16)),
                       (6160, dt + timedelta(hours=17)),
                       (3510, dt + timedelta(hours=18)),
                       (960, dt + timedelta(hours=19)),
                       (560, dt + timedelta(hours=20)),
                       (0, dt + timedelta(hours=21)),
                       (0, dt + timedelta(hours=22)),
                       (0, dt + timedelta(hours=23))
                       ]

        s = PeriodSolver(
            start_time = start_time,
            end_time = end_time,
            tariffs = tarrifs,
            actionable_loads = [car, pool, cumulus_parents, cumulus_children],
            battery = None,
            pv_forecast = pv_forecast,
            unavoidable_consumption_forecast = unavoidable_consumption_forecast
        )

        s.solve()

