from custom_components.quiet_solar.home_model.constraints import MultiStepsPowerLoadConstraint, TimeBasedSimplePowerLoadConstraint, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.commands import CMD_AUTO_GREEN_CAP, CMD_AUTO_FROM_CONSIGN, LoadCommand, copy_command
from custom_components.quiet_solar.home_model.battery import Battery
from datetime import datetime, timedelta
import pytz


def test_cap_command_preserve_battery():
    time = datetime.now(pytz.UTC)
    load = TestLoad(name="load")
    steps = [LoadCommand(command="ON_WITH_VAL", power_consign=1000)]
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
        end_of_constraint=time + timedelta(hours=1),
        initial_value=0,
        target_value=3600,
        power_steps=steps,
        support_auto=True,
    )
    out_c, solved, changed, remaining, commands, delta_power = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=-500,
        power_slots_duration_s=[900.0],
        existing_commands=[None],
        allow_change_state=True,
        time=time,
    )
    assert commands[0].command == CMD_AUTO_GREEN_CAP.command


def test_preconsume_surplus():
    time = datetime.now(pytz.UTC)
    load = TestLoad(name="load")
    steps = [LoadCommand(command="ON_WITH_VAL", power_consign=1000)]
    constraint = MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        type=CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
        end_of_constraint=time + timedelta(hours=1),
        initial_value=0,
        target_value=3600,
        power_steps=steps,
        support_auto=True,
    )
    out_c, solved, changed, remaining, commands, delta_power = constraint.adapt_repartition(
        first_slot=0,
        last_slot=0,
        energy_delta=500,
        power_slots_duration_s=[900.0],
        existing_commands=[None],
        allow_change_state=True,
        time=time,
    )
    assert commands[0].command == CMD_AUTO_FROM_CONSIGN.command
