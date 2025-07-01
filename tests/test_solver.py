import asyncio
from unittest import TestCase

import pytz

from custom_components.quiet_solar.const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER_AUTO, FLOATING_PERIOD_S
from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint, TimeBasedSimplePowerLoadConstraint, \
    LoadConstraint, MultiStepsPowerLoadConstraintChargePercent
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import PeriodSolver
from datetime import datetime
from datetime import timedelta

from custom_components.quiet_solar.home_model.commands import LoadCommand, copy_command, CMD_AUTO_GREEN_ONLY, CMD_IDLE


def _util_constraint_save_dump(time, cs):
    dc_dump = cs.to_dict()
    load = cs.load
    cs_load = LoadConstraint.new_from_saved_dict(time, load, dc_dump)
    assert cs == cs_load

class TestSolver(TestCase):


    def test_solve(self):
        # This is a simple test to check if the solver is working correctly
        # The solver is a simple function that returns the sum of two numbers
        # The test checks if the sum of 1 and 2 is 3

        dt = datetime(year=2024, month=6, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)

        start_time = dt
        end_time = dt + timedelta(days=1)

        tarrifs = 0.27/1000.0

        car = TestLoad(name="car")
        pool = TestLoad(name="pool")
        cumulus_parents = TestLoad(name="cumulus_parents")
        cumulus_children = TestLoad(name="cumulus_children")


        car_steps = []
        for a in range(7, 33):
            car_steps.append(LoadCommand(command="ON_WITH_VAL", power_consign=a * 3 * 230))

        car_charge_mandatory = MultiStepsPowerLoadConstraint(
            time=dt,
            load = car,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint = dt + timedelta(hours=11),
            initial_value = 10000,
            target_value = 16000,
            power_steps = car_steps,
            support_auto = True
        )
        car.push_live_constraint(dt, car_charge_mandatory)




        _util_constraint_save_dump(dt, car_charge_mandatory)


        car_charge_best_effort = MultiStepsPowerLoadConstraint(
            time=dt,
            load = car,
            type=CONSTRAINT_TYPE_FILLER_AUTO,
            end_of_constraint = None,
            initial_value = None,
            target_value = 22000,
            power_steps = car_steps,
            support_auto = True
        )
        car.push_live_constraint(dt, car_charge_best_effort)

        _util_constraint_save_dump(dt, car_charge_best_effort)


        pool_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load = pool,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint = dt + timedelta(hours=23),
            initial_value = 0,
            target_value = 10*3600,
            power = 1430,
        )
        pool.push_live_constraint(dt, pool_constraint)

        _util_constraint_save_dump(dt, pool_constraint)

        cumulus_parents_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load = cumulus_parents,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            end_of_constraint = dt + timedelta(hours=20),
            initial_value = 0,
            target_value = 3*3600,
            power = 2000,
        )
        cumulus_parents.push_live_constraint(dt, cumulus_parents_constraint)

        _util_constraint_save_dump(dt, cumulus_parents_constraint)

        cumulus_children_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load = cumulus_children,
            type=CONSTRAINT_TYPE_FILLER_AUTO,
            end_of_constraint = dt + timedelta(hours=17),
            initial_value = 0,
            target_value = 1*3600,
            power = 2000,
        )
        cumulus_children.push_live_constraint(dt, cumulus_children_constraint)

        _util_constraint_save_dump(dt, cumulus_children_constraint)

        unavoidable_consumption_forecast = [(dt, 300),
                       (dt + timedelta(hours=1) ,  300  ),
                       (dt + timedelta(hours=2) ,  300 ),
                       (dt + timedelta(hours=3) ,  300 ),
                       (dt + timedelta(hours=4) ,  380 ),
                       (dt + timedelta(hours=5) ,  300 ),
                       (dt + timedelta(hours=6) ,  300 ),
                       (dt + timedelta(hours=7) ,  390 ),
                       (dt + timedelta(hours=8) ,  900 ),
                       (dt + timedelta(hours=9) ,  700 ),
                       (dt + timedelta(hours=10),  400  ),
                       (dt + timedelta(hours=11),  600  ),
                       (dt + timedelta(hours=12) , 1300),
                       (dt + timedelta(hours=13),  500  ),
                       (dt + timedelta(hours=14),  800  ),
                       (dt + timedelta(hours=15),  400  ),
                       (dt + timedelta(hours=16) , 1200),
                       (dt + timedelta(hours=17),  500  ),
                       (dt + timedelta(hours=18),  500  ),
                       (dt + timedelta(hours=19),  800  ),
                       (dt + timedelta(hours=20),  800  ),
                       (dt + timedelta(hours=21),  700  ),
                       (dt + timedelta(hours=22),  650  ),
                       (dt + timedelta(hours=23),  450  )
                       ]

        pv_forecast = [(dt                      ,0          ),
                       (dt + timedelta(hours=1) ,0          ),
                       (dt + timedelta(hours=2) ,0          ),
                       (dt + timedelta(hours=3) ,0          ),
                       (dt + timedelta(hours=4) ,0          ),
                       (dt + timedelta(hours=5) ,0          ),
                       (dt + timedelta(hours=6) ,690        ),
                       (dt + timedelta(hours=7) ,2250       ),
                       (dt + timedelta(hours=8) ,2950       ),
                       (dt + timedelta(hours=9) ,5760       ),
                       (dt + timedelta(hours=10),8210       ),
                       (dt + timedelta(hours=11),9970       ),
                       (dt + timedelta(hours=12),9760       ),
                       (dt + timedelta(hours=13),9840       ),
                       (dt + timedelta(hours=14),7390       ),
                       (dt + timedelta(hours=15),8420       ),
                       (dt + timedelta(hours=16),9360       ),
                       (dt + timedelta(hours=17),6160       ),
                       (dt + timedelta(hours=18),3510       ),
                       (dt + timedelta(hours=19),960        ),
                       (dt + timedelta(hours=20),560        ),
                       (dt + timedelta(hours=21),0          ),
                       (dt + timedelta(hours=22),0          ),
                       (dt + timedelta(hours=23),0          )
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


    def test_auto_cmds(self):

        async def _async_test():

            time = datetime.now(pytz.UTC)
            car_capacity = 22000
            target_best_effort = 22000
            charger = TestLoad(name="charger")
            steps = []
            for a in range(7, 32 + 1):
                steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=a * 230 * 3))


            car_charge_as_best = MultiStepsPowerLoadConstraintChargePercent(
                time=time,
                type=CONSTRAINT_TYPE_FILLER_AUTO,
                total_capacity_wh=car_capacity,
                load=charger,
                initial_value=None,
                target_value=target_best_effort,
                power_steps=steps,
                support_auto=True,
            )
            charger.push_live_constraint(time, car_charge_as_best)

            s = PeriodSolver(
                start_time=time,
                end_time=time + timedelta(seconds=FLOATING_PERIOD_S),
                tariffs=None,
                actionable_loads=[charger],
                pv_forecast=None,
                unavoidable_consumption_forecast=None
            )
            cmds, battery_comands = s.solve()

            assert cmds is not None

            cmds_charger = cmds[0][1]

            assert len(cmds_charger)== 1

            assert cmds_charger[0][1] == CMD_AUTO_GREEN_ONLY

        asyncio.run(_async_test())
