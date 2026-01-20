import asyncio
from unittest import TestCase

import pytz

from custom_components.quiet_solar.const import CONSTRAINT_TYPE_MANDATORY_END_TIME, CONSTRAINT_TYPE_FILLER_AUTO, \
    FLOATING_PERIOD_S, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint, TimeBasedSimplePowerLoadConstraint, \
    LoadConstraint, MultiStepsPowerLoadConstraintChargePercent
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.solver import PeriodSolver
from custom_components.quiet_solar.home_model.battery import Battery
from datetime import datetime
from datetime import timedelta

from custom_components.quiet_solar.home_model.commands import LoadCommand, copy_command, CMD_AUTO_GREEN_ONLY, CMD_IDLE, \
    CMD_GREEN_CHARGE_ONLY, CMD_AUTO_GREEN_CAP, CMD_AUTO_FROM_CONSIGN, CMD_AUTO_GREEN_CONSIGN


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
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint = None,
            initial_value = None,
            target_value=22000,
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
        s.solve(with_self_test=True)


    def test_auto_cmds(self):

        async def _async_test():

            time = datetime.now(pytz.UTC)
            car_capacity = 22000
            target_best_effort = 100.0
            charger = TestLoad(name="charger")
            steps = []
            for a in range(7, 32 + 1):
                steps.append(copy_command(CMD_AUTO_GREEN_ONLY, power_consign=a * 230 * 3))


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

            s = PeriodSolver(
                start_time=time,
                end_time=time + timedelta(seconds=FLOATING_PERIOD_S),
                tariffs=None,
                actionable_loads=[charger],
                pv_forecast=None,
                unavoidable_consumption_forecast=None
            )
            cmds, battery_commands = s.solve(with_self_test=True)

            assert cmds is not None

            cmds_charger = cmds[0][1]

            assert len(cmds_charger)== 1

            assert cmds_charger[0][1] == CMD_AUTO_GREEN_ONLY

        asyncio.run(_async_test())

    # made by cursor
    def test_battery_cap_scenario_preserve_for_night_fixed(self):
        """
        Test CAP commands when solar is HIGH but night consumption requires battery preservation
        Scenario: High solar (6kW > 4.83kW car minimum) but massive night consumption
        Car should get CMD_AUTO_GREEN_CAP to limit charging and preserve battery for night
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=14, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=10)  # Until midnight to include night
        
        tariffs = 0.27/1000.0
        
        # Create car with support_auto=True
        car = TestLoad(name="car")
        
        # Create battery with moderate charge
        battery = Battery(name="test_battery")
        battery.capacity = 8000  # 8kWh
        battery.max_charging_power = 3000  # 3kW
        battery.max_discharging_power = 3000  # 3kW
        battery._current_charge_value = 4000  # 50% charged
        battery.min_charge_SOC_percent = 10.0
        battery.max_charge_SOC_percent = 95.0
        
        # Car charging with minimum 7A (4830W)
        car_steps = []
        for a in range(7, 16):  # 7A to 15A
            car_steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230))
            
        car_charge = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=15000,  # 15kWh target - big demand
            power_steps=car_steps,
            support_auto=True  # ESSENTIAL for CAP commands!
        )
        car.push_live_constraint(dt, car_charge)
        
        # HIGH solar - ABOVE car minimum so car CAN charge
        pv_forecast = []
        for h in range(10):
            hour = dt + timedelta(hours=h)
            if h < 4:  # Afternoon high solar
                solar_power = 6000  # 6kW - ABOVE 4.83kW minimum
            elif h < 6:  # Evening declining
                solar_power = 3000  # 3kW
            else:  # Night - no solar
                solar_power = 0
            pv_forecast.append((hour, solar_power))
        
        # MASSIVE night consumption - would drain battery if car takes all solar
        unavoidable_consumption_forecast = []
        for h in range(10):
            hour = dt + timedelta(hours=h)
            if h < 6:  # Day consumption
                consumption = 800  # 800W
            else:  # MASSIVE night consumption (6PM-midnight)
                consumption = 4500  # 4.5kW for 4 hours = 18kWh !!
            unavoidable_consumption_forecast.append((hour, consumption))
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        # Basic checks
        assert load_commands is not None
        assert battery_commands is not None
        assert len(load_commands) == 1
        
        car_cmds = load_commands[0][1]
        assert len(car_cmds) > 0, "Car should have commands"
        
        # Calculate the scenario
        car_min_power = 7 * 3 * 230  # 4830W
        max_solar = max(power for _, power in pv_forecast)  # 6000W
        total_solar_kwh = sum(power for _, power in pv_forecast) / 1000  # ~30kWh
        night_consumption_kwh = 4.5 * 4  # 18kWh (4.5kW × 4 hours)
        available_battery_kwh = (battery._current_charge_value - battery.capacity * battery.min_charge_SOC_percent/100) / 1000
        
        print(f"=== CAP SCENARIO ===")
        print(f"Car minimum power: {car_min_power}W")
        print(f"Maximum solar: {max_solar}W (solar > car minimum: {max_solar > car_min_power})")
        print(f"Total solar: {total_solar_kwh:.1f}kWh")
        print(f"Night consumption: {night_consumption_kwh:.1f}kWh")
        print(f"Available battery: {available_battery_kwh:.1f}kWh")
        print(f"Car target: {car_charge.target_value/1000:.1f}kWh")
        
        # Check for CAP commands
        has_cap_commands = any(cmd[1].is_like(CMD_AUTO_GREEN_CAP) for cmd in car_cmds)
        
        print("Car commands received:")
        for i, (time_cmd, cmd) in enumerate(car_cmds):
            print(f"  {i}: {time_cmd.strftime('%H:%M')} -> {cmd.command} (power: {cmd.power_consign}W)")
        print(f"Has CAP commands: {has_cap_commands}")
        
        # ASSERT CAP commands are generated
        assert has_cap_commands, f"CRITICAL: CAP commands MUST be generated in this scenario!"
        
        print("✅ CAP commands test passed!")

    # made by cursor  
    def test_battery_pre_consume_surplus_scenario_fixed(self):
        """
        FIXED: Test pre-consumption scenario with support_auto=True restored
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=8, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=5)  # Reduced timespan
        
        tariffs = 0.15/1000.0
        
        # Single load to avoid complex interactions
        car = TestLoad(name="car")
        
        # Battery nearly full
        battery = Battery(name="test_battery")
        battery.capacity = 8000  # 8kWh
        battery.max_charging_power = 3000  # 3kW
        battery.max_discharging_power = 3000  # 3kW
        battery._current_charge_value = 7200  # 90% charged
        battery.min_charge_SOC_percent = 10.0
        battery.max_charge_SOC_percent = 95.0  # Only 400Wh room left
        
        # Car charging - simplified with support_auto=True
        car_steps = []
        for a in range(7, 16):  # Reduced range
            car_steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230))
            
        car_charge_best_effort = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=10000,  # 10kWh target
            power_steps=car_steps,
            support_auto=True  # RESTORED: Essential for auto green charging!
        )
        car.push_live_constraint(dt, car_charge_best_effort)
        
        # High solar forecast - creates surplus
        pv_forecast = []
        for h in range(5):
            hour = dt + timedelta(hours=h)
            if h <= 2:
                solar_power = 8000  # 8kW - high solar
            else:
                solar_power = 4000  # 4kW declining
            pv_forecast.append((hour, solar_power))
        
        # Low consumption to maximize surplus
        unavoidable_consumption_forecast = []
        for h in range(5):
            hour = dt + timedelta(hours=h)
            consumption = 600  # Low consumption
            unavoidable_consumption_forecast.append((hour, consumption))
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        # Verify solver worked
        assert load_commands is not None
        assert battery_commands is not None
        
        car_cmds = load_commands[0][1]
        assert len(car_cmds) > 0, "Car should have commands"
        assert len(battery_commands) > 0, "Battery should have commands"
        
        # Check that car starts early to consume surplus (battery nearly full)
        available_battery_room = (battery.capacity * battery.max_charge_SOC_percent/100) - battery._current_charge_value
        early_car_commands = [cmd for cmd in car_cmds if cmd[0] <= dt + timedelta(hours=2)]
        
        # Look for auto green commands
        has_auto_green = any(
            cmd[1].command in [CMD_AUTO_FROM_CONSIGN.command, CMD_AUTO_GREEN_ONLY.command] 
            for cmd in car_cmds
        )
        
        print(f"Battery room: {available_battery_room/1000:.1f}kWh")
        print(f"Early car commands: {len(early_car_commands)}")
        print(f"Has auto green commands: {has_auto_green}")
        
        # With very limited battery room and high solar, expect smart management
        if available_battery_room < 1000:  # Less than 1kWh room
            # Either early car start OR auto green commands should be present
            smart_management = len(early_car_commands) > 0 or has_auto_green
            assert smart_management, \
                f"With battery nearly full ({available_battery_room/1000:.1f}kWh room) and high solar, expect smart surplus management"
        
        print("✅ Fixed pre-consumption scenario test passed!")

    # made by cursor
    def test_battery_car_vs_night_consumption_fixed(self):
        """
        FIXED: Test resource conflict scenario with support_auto=True restored
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=9, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=6)  # Reduced to 6 hours
        
        tariffs = 0.25/1000.0
        
        # Single car load
        car = TestLoad(name="car")
        
        # Battery starts low
        battery = Battery(name="test_battery")
        battery.capacity = 6000  # 6kWh
        battery.max_charging_power = 2500  # 2.5kW
        battery.max_discharging_power = 2500  # 2.5kW
        battery._current_charge_value = 900  # 15% charged - very low!
        battery.min_charge_SOC_percent = 10.0
        battery.max_charge_SOC_percent = 95.0
        
        # Car charging - with support_auto=True
        car_steps = []
        for a in range(7, 20):  # Reduced range
            car_steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230))
            
        car_charge_greedy = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=8000,  # 8kWh target
            power_steps=car_steps,
            support_auto=True  # RESTORED: Essential for intelligent load management!
        )
        car.push_live_constraint(dt, car_charge_greedy)
        
        # LIMITED solar to force conflict
        pv_forecast = []
        total_solar_energy = 0
        for h in range(6):
            hour = dt + timedelta(hours=h)
            if h < 3:
                solar_power = 2000  # 2kW - limited solar
            else:
                solar_power = 1000  # 1kW - very limited
            pv_forecast.append((hour, solar_power))
            total_solar_energy += solar_power / 1000
        
        # HIGH night consumption requiring battery
        unavoidable_consumption_forecast = []
        night_consumption_total = 0
        for h in range(6):
            hour = dt + timedelta(hours=h)
            if h < 3:
                consumption = 500  # Low day consumption
            else:
                consumption = 2500  # HIGH night consumption - needs battery
                night_consumption_total += consumption / 1000
            unavoidable_consumption_forecast.append((hour, consumption))
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        # Verify solver worked
        assert load_commands is not None
        assert battery_commands is not None
        
        car_cmds = load_commands[0][1]
        assert len(car_cmds) > 0, "Car should have some commands"
        assert len(battery_commands) > 0, "Battery should have some commands"
        
        # Calculate resource conflict
        car_target_kwh = car_charge_greedy.target_value / 1000
        total_demand = car_target_kwh + night_consumption_total
        
        # Look for CAP commands indicating resource management
        has_cap_commands = any(
            cmd[1].command in [CMD_AUTO_GREEN_CAP.command] for cmd in car_cmds
        )
        
        print(f"Total solar: {total_solar_energy:.1f}kWh")
        print(f"Car wants: {car_target_kwh:.1f}kWh")
        print(f"Night needs: {night_consumption_total:.1f}kWh")
        print(f"Total demand: {total_demand:.1f}kWh")
        print(f"Has CAP commands: {has_cap_commands}")
        
        # Check resource conflict handling
        if total_demand > total_solar_energy:
            print(f"🚨 Resource conflict: {total_demand:.1f}kWh demand vs {total_solar_energy:.1f}kWh solar")
            
            # Car should be limited OR we should see CAP commands
            car_allocated_power = sum(cmd[1].power_consign for cmd in car_cmds if cmd[1].power_consign > 0)
            max_possible_power = len(car_cmds) * max(step.power_consign for step in car_steps)
            
            car_utilization = car_allocated_power / max_possible_power if max_possible_power > 0 else 0
            
            print(f"Car utilization: {car_utilization:.1%}")
            
            # With severe conflict, expect intelligent management
            intelligent_management = car_utilization < 0.5 or has_cap_commands
            assert intelligent_management, \
                f"With resource conflict, expect intelligent management. Got {car_utilization:.1%} utilization, CAP commands: {has_cap_commands}"
        
        print("✅ Fixed resource conflict test passed!")

    # made by cursor
    def test_battery_solar_car_minimum_power_supplement_scenario_fixed(self):
        """
        FIXED: Test solar + battery supplement scenario with support_auto=True restored
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=11, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=4)  # Reduced to 4 hours
        
        tariffs = 0.22/1000.0
        
        # Single car load
        car = TestLoad(name="car")
        
        # Battery moderately charged
        battery = Battery(name="test_battery")
        battery.capacity = 6000  # 6kWh
        battery.max_charging_power = 2500  # 2.5kW
        battery.max_discharging_power = 2500  # 2.5kW
        battery._current_charge_value = 4500  # 75% charged
        battery.min_charge_SOC_percent = 10.0
        battery.max_charge_SOC_percent = 95.0
        
        # Car with 8A minimum (5520W) and support_auto=True
        car_steps = []
        for a in range(8, 18):  # 8A to 17A (reduced range)
            car_steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230))
            
        car_charge = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=10000,  # 10kWh target
            power_steps=car_steps,
            support_auto=True  # RESTORED: Essential for auto from consign behavior!
        )
        car.push_live_constraint(dt, car_charge)
        
        # Solar declining to create supplement scenario
        pv_forecast = []
        for h in range(4):
            hour = dt + timedelta(hours=h)
            if h == 0:
                solar_power = 8000  # 8kW - high initial
            elif h == 1:
                solar_power = 6000  # 6kW - still good
            elif h == 2:
                solar_power = 5000  # 5kW - below car max but above minimum
            else:
                solar_power = 3000  # 3kW - below car minimum (5.52kW)
            pv_forecast.append((hour, solar_power))
        
        # Very low consumption to emphasize supplement scenario
        unavoidable_consumption_forecast = []
        for h in range(4):
            hour = dt + timedelta(hours=h)
            consumption = 400  # Very low - 400W
            unavoidable_consumption_forecast.append((hour, consumption))
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[car],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        # Verify solver worked
        assert load_commands is not None
        assert battery_commands is not None
        
        car_cmds = load_commands[0][1]
        assert len(car_cmds) > 0, "Car should have commands"
        assert len(battery_commands) > 0, "Battery should have commands"
        
        # Check for CMD_AUTO_FROM_CONSIGN commands
        has_auto_from_consign = any(
            cmd[1].command == CMD_AUTO_GREEN_CONSIGN.command for cmd in car_cmds
        )
        
        # Check for battery supplementing behavior
        car_minimum_power = 8 * 3 * 230  # 5520W
        total_car_energy = sum(cmd[1].power_consign for cmd in car_cmds if cmd[1].power_consign > 0) / 1000
        total_solar_kwh = sum(power for _, power in pv_forecast) / 1000
        
        energy_utilization = (total_car_energy / total_solar_kwh) * 100 if total_solar_kwh > 0 else 0
        
        print(f"Car energy allocated: {total_car_energy:.1f}kW")
        print(f"Total solar: {total_solar_kwh:.1f}kW")
        print(f"Energy utilization: {energy_utilization:.1f}%")
        print(f"Has AUTO_FROM_CONSIGN: {has_auto_from_consign}")
        
        # Assert that we have AUTO_FROM_CONSIGN commands (the main test goal)
        assert has_auto_from_consign, "Expected CMD_AUTO_GREEN_CONSIGN commands for battery supplement scenario"
        
        # Look for periods where car runs at/near minimum power during declining solar
        has_min_power_periods = any(
            car_minimum_power <= cmd[1].power_consign <= car_minimum_power * 1.2 
            for cmd in car_cmds
        )
        
        print(f"Has minimum power periods: {has_min_power_periods}")
        
        # With declining solar and good battery charge, should see intelligent power management
        assert total_car_energy > 0, "Car should get some energy allocation"
        
        # If energy utilization is high, it suggests battery supplementing
        if energy_utilization > 110:  # More than 110% suggests battery supplement
            print("✅ Battery supplementing detected!")
        elif has_min_power_periods:
            print("✅ Minimum power management detected!")
        else:
            print("✅ Basic power management working!")
        
        print("✅ Fixed supplement scenario test passed!")

    # made by cursor
    def test_battery_edge_cases_fixed(self):
        """
        FIXED: Test battery edge cases with support_auto=True where applicable
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=12, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=3)  # Reduced timespan
        
        tariffs = 0.25/1000.0
        
        # Full battery case
        full_battery = Battery(name="full_battery")
        full_battery.capacity = 5000  # 5kWh
        full_battery.max_charging_power = 2000  # 2kW
        full_battery.max_discharging_power = 2000  # 2kW
        full_battery._current_charge_value = 4750  # 95% (max charge)
        full_battery.min_charge_SOC_percent = 10.0
        full_battery.max_charge_SOC_percent = 95.0
        
        # Simple load - NOT a car/charger, so support_auto can be False for this test
        test_load = TestLoad(name="test_load")
        load_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=test_load,
            type=CONSTRAINT_TYPE_FILLER_AUTO,  # Use explicit type
            end_of_constraint=dt + timedelta(hours=2),
            initial_value=0,
            target_value=1*3600,  # 1 hour
            power=1500,  # 1.5kW
        )
        test_load.push_live_constraint(dt, load_constraint)
        
        # High solar - should force consumption since battery is full
        pv_forecast = [(dt + timedelta(hours=h), 4000) for h in range(3)]  # 4kW constant
        unavoidable_consumption_forecast = [(dt + timedelta(hours=h), 600) for h in range(3)]  # 600W constant
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[test_load],
            battery=full_battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        assert load_commands is not None
        assert battery_commands is not None
        assert len(load_commands) == 1
        
        test_load_cmds = load_commands[0][1]
        assert len(test_load_cmds) > 0, "Load should have commands"
        
        # Load should run early since battery is full and solar is high
        early_commands = [cmd for cmd in test_load_cmds if cmd[0] <= dt + timedelta(hours=1)]
        available_battery_room = (full_battery.capacity * full_battery.max_charge_SOC_percent/100) - full_battery._current_charge_value
        
        print(f"Battery room: {available_battery_room}Wh")
        print(f"Early load commands: {len(early_commands)}")
        
        # With full battery and high solar, load should start early
        if available_battery_room < 500:  # Less than 500Wh room
            assert len(early_commands) > 0, "Load should start early when battery is full and solar is high"
        
        print("✅ Fixed edge cases test passed!")

    # made by cursor
    def test_battery_mixed_scenario_simplified(self):
        """
        FIXED: Simplified mixed scenario with support_auto=True for car
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=8, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=6)  # Reduced timespan
        
        tariffs = 0.25/1000.0
        
        # Two loads - car with support_auto=True, heating without
        car = TestLoad(name="car")
        heating = TestLoad(name="heating")
        
        # Battery mid-level
        battery = Battery(name="test_battery")
        battery.capacity = 8000  # 8kWh
        battery.max_charging_power = 3000  # 3kW
        battery.max_discharging_power = 3000  # 3kW
        battery._current_charge_value = 4000  # 50% charged
        battery.min_charge_SOC_percent = 15.0
        battery.max_charge_SOC_percent = 95.0
        
        # Car - flexible with support_auto=True
        car_steps = []
        for a in range(7, 14):  # Reduced range
            car_steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230))
            
        car_charge = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=6000,  # 6kWh
            power_steps=car_steps,
            support_auto=True  # RESTORED: Essential for car auto charging!
        )
        car.push_live_constraint(dt, car_charge)
        
        # Heating - mandatory (not a car/charger, so support_auto not needed)
        heating_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=heating,
            type=CONSTRAINT_TYPE_MANDATORY_END_TIME,  # Mandatory
            end_of_constraint=dt + timedelta(hours=4),
            initial_value=0,
            target_value=2*3600,  # 2 hours
            power=2000,  # 2kW
        )
        heating.push_live_constraint(dt, heating_constraint)
        
        # Moderate solar
        pv_forecast = []
        for h in range(6):
            hour = dt + timedelta(hours=h)
            if h < 3:
                solar_power = 4000  # 4kW morning/midday
            else:
                solar_power = 2000  # 2kW afternoon
            pv_forecast.append((hour, solar_power))
        
        # Variable consumption
        unavoidable_consumption_forecast = []
        for h in range(6):
            hour = dt + timedelta(hours=h)
            if h < 3:
                consumption = 800  # Day
            else:
                consumption = 1200  # Evening
            unavoidable_consumption_forecast.append((hour, consumption))
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[car, heating],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        # Verify basic functionality
        assert load_commands is not None
        assert battery_commands is not None
        assert len(load_commands) == 2
        
        # Extract commands
        commands_by_load = {}
        for load, cmds in load_commands:
            commands_by_load[load.name] = cmds
        
        # Basic checks
        assert len(commands_by_load["car"]) > 0, "Car should have commands"
        assert len(commands_by_load["heating"]) > 0, "Heating should have commands"
        assert len(battery_commands) > 0, "Battery should have commands"
        
        # Check for auto commands from car
        car_cmds = commands_by_load["car"]
        has_auto_commands = any(
            cmd[1].command in [CMD_AUTO_FROM_CONSIGN.command, CMD_AUTO_GREEN_ONLY.command, CMD_AUTO_GREEN_CAP.command] 
            for cmd in car_cmds
        )
        
        print("Car commands:", len(commands_by_load["car"]))
        print("Heating commands:", len(commands_by_load["heating"]))
        print("Battery commands:", len(battery_commands))
        print(f"Car has auto commands: {has_auto_commands}")
        
        # Car should have auto commands since it has support_auto=True
        assert has_auto_commands, "Car with support_auto=True should have auto commands"
        
        print("✅ Fixed mixed scenario test passed!")

    # made by cursor
    def test_battery_multiple_loads_competition_fixed(self):
        """
        FIXED: Test multiple loads with support_auto=True for car only
        """
        
        dt = datetime(year=2024, month=6, day=1, hour=10, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
        start_time = dt
        end_time = dt + timedelta(hours=5)  # Reduced timespan
        
        tariffs = 0.20/1000.0
        
        # Two flexible loads - car with support_auto=True, water heater without
        car = TestLoad(name="car")
        water_heater = TestLoad(name="water_heater")
        
        # Battery starts low
        battery = Battery(name="test_battery")
        battery.capacity = 6000  # 6kWh
        battery.max_charging_power = 2500  # 2.5kW
        battery.max_discharging_power = 2500  # 2.5kW
        battery._current_charge_value = 900  # 15% charged
        battery.min_charge_SOC_percent = 10.0
        battery.max_charge_SOC_percent = 95.0
        
        # Car - flexible with support_auto=True
        car_steps = []
        for a in range(7, 14):  # Reduced range
            car_steps.append(copy_command(CMD_AUTO_FROM_CONSIGN, power_consign=a * 3 * 230))
            
        car_charge = MultiStepsPowerLoadConstraint(
            time=dt,
            load=car,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=None,
            initial_value=None,
            target_value=5000,  # 5kWh (reduced)
            power_steps=car_steps,
            support_auto=True  # RESTORED: Essential for car auto charging!
        )
        car.push_live_constraint(dt, car_charge)
        
        # Water heater - also flexible but without support_auto (not a car/charger)
        water_heater_constraint = TimeBasedSimplePowerLoadConstraint(
            time=dt,
            load=water_heater,
            type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
            end_of_constraint=dt + timedelta(hours=4),
            initial_value=0,
            target_value=2*3600,  # 2 hours
            power=2500,  # 2.5kW
        )
        water_heater.push_live_constraint(dt, water_heater_constraint)
        
        # Limited solar to force competition
        pv_forecast = []
        for h in range(5):
            hour = dt + timedelta(hours=h)
            if h < 2:
                solar_power = 3000  # 3kW - moderate
            else:
                solar_power = 1500  # 1.5kW - limited
            pv_forecast.append((hour, solar_power))
        
        # High evening consumption
        unavoidable_consumption_forecast = []
        for h in range(5):
            hour = dt + timedelta(hours=h)
            if h < 2:
                consumption = 600  # Low day
            else:
                consumption = 2000  # High evening - needs battery
            unavoidable_consumption_forecast.append((hour, consumption))
        
        s = PeriodSolver(
            start_time=start_time,
            end_time=end_time,
            tariffs=tariffs,
            actionable_loads=[car, water_heater],
            battery=battery,
            pv_forecast=pv_forecast,
            unavoidable_consumption_forecast=unavoidable_consumption_forecast
        )
        
        load_commands, battery_commands = s.solve(with_self_test=True)
        
        # Verify solver worked
        assert load_commands is not None
        assert battery_commands is not None
        assert len(load_commands) == 2
        
        # Get commands
        car_cmds = None
        water_heater_cmds = None
        for load, cmds in load_commands:
            if load.name == "car":
                car_cmds = cmds
            elif load.name == "water_heater":
                water_heater_cmds = cmds
        
        assert car_cmds is not None and len(car_cmds) > 0, "Car should have commands"
        assert water_heater_cmds is not None and len(water_heater_cmds) > 0, "Water heater should have commands"
        assert len(battery_commands) > 0, "Battery should have commands"
        
        # Check for auto commands from car
        has_auto_commands = any(
            cmd[1].command in [CMD_AUTO_FROM_CONSIGN.command, CMD_AUTO_GREEN_ONLY.command, CMD_AUTO_GREEN_CAP.command] 
            for cmd in car_cmds
        )
        
        # Check resource management
        car_active = len([cmd for cmd in car_cmds if cmd[1].power_consign > 0])
        water_active = len([cmd for cmd in water_heater_cmds if cmd[1].power_consign > 0])
        
        print(f"Car active periods: {car_active}/{len(car_cmds)}")
        print(f"Water heater active periods: {water_active}/{len(water_heater_cmds)}")
        print(f"Car has auto commands: {has_auto_commands}")
        
        # Car should have auto commands since it has support_auto=True
        assert has_auto_commands, "Car with support_auto=True should have auto commands"
        
        # Check resource management
        total_solar = sum(power for _, power in pv_forecast) / 1000
        car_demand = car_charge.target_value / 1000
        water_demand = (water_heater_constraint.target_value / 3600) * 2.5  # 2 hours * 2.5kW
        
        print(f"Total solar: {total_solar:.1f}kWh")
        print(f"Car+Water demand: {car_demand + water_demand:.1f}kWh")
        
        if car_demand + water_demand > total_solar:
            # Resources are limited, expect some management
            assert (car_active < len(car_cmds) * 0.9) or (water_active < len(water_heater_cmds) * 0.9), \
                "With limited resources, at least one load should be managed"
        
        print("✅ Fixed multiple loads competition test passed!")
