from datetime import datetime
from datetime import timedelta
import numpy.typing as npt
import numpy as np
import pytz

from .battery import Battery, CMD_FORCE_CHARGE, CMD_FORCE_DISCHARGE
from .constraints import LoadConstraint, DATETIME_MAX_UTC
from .load import AbstractLoad
from .commands import LoadCommand, CMD_AUTO_FROM_CONSIGN, copy_command, CMD_IDLE
from ..const import CONSTRAINT_TYPE_BEFORE_BATTERY_AUTO_GREEN


class PeriodSolver(object):

    def __init__(self,
                 start_time: datetime | None = None,
                 end_time: datetime | None = None,
                 tariffs: list[tuple[timedelta, float]] | float | None = None,
                 actionable_loads: list[AbstractLoad] = None,
                 battery: Battery | None = None,
                 pv_forecast: list[tuple[datetime, float]] = None,
                 unavoidable_consumption_forecast: list[tuple[datetime, float]] = None,
                 step_s: timedelta = timedelta(seconds=900),
                 ) -> None:

        """
        :param start_time: datetime: The start time of the period to solve.
        :param end_time: datetime: The end time of the day.
        :param tariffs: list of tuples, each tuple contains the following:
            - float: the price of the period per kWh
            - timedelta: the offset of the start time in a current day
        :param loads: list of AbstractLoad that have some constraint in it and their current states as constraint
        :param pv_forecast: list of time sorted tuples, each tuple contains the following:
            - float: the forecasted power output of the PV, in W
            - datetime: the time of the forecast
        :param unavoidable_consumption_forecast: list time sorted of tuples, each tuple contains the following:
            - float: the forecasted unavoidable power consumption in W
            - datetime: the time of the forecast
        """

        if start_time is None:
            start_time = datetime.now(tz=pytz.UTC)

        if end_time is None:
            end_time = start_time + timedelta(days=1)

        self._start_time = start_time
        self._end_time = end_time
        self._step_s = step_s.total_seconds()
        self._tariffs: list[tuple[datetime, float]] | None = None
        self._loads: list[AbstractLoad] = actionable_loads
        self._pv_forecast: list[datetime, float] | None = pv_forecast
        self._ua_forecast: list[datetime, float] | None = unavoidable_consumption_forecast
        self._battery = battery

        if not tariffs:
            self._tariffs = [(start_time, 0.2)]
        elif isinstance(tariffs, float):
            self._tariffs = [(start_time, tariffs)]
        elif len(tariffs) == 1:
            self._tariffs = [(start_time, tariffs[0][1])]
        else:
            self._tariffs = tariffs

        self._prices_ordered_values : list[float] = sorted(list(set([ p[1] for p in self._tariffs])))

        # first lay off the time scales and slots, to match the constraints and tariffs timelines
        self._time_slots, self._active_constraints = self.create_time_slots(self._start_time, self._end_time)
        self._prices, self._durations_s, self._available_power = self.create_power_slots(self._time_slots)


    def create_time_slots(self, start_time: datetime, end_time: datetime) -> tuple[list[datetime], list[LoadConstraint]]:
        """
        Create the time slots for the forecast.

        Args:
            start_time: datetime: The start time of the solving period.
            end_time: datetime: The end time of the solving period.

        Returns:
            list: A list of time slots.
        """

        #gather all anchor point first:
        anchors = set()
        anchors.add(start_time)
        anchors.add(end_time)
        for tariff in self._tariffs:
            if tariff[0] >= start_time and tariff[0] <= end_time:
                anchors.add(tariff[0])

        active_constraints = []
        for load in self._loads:
            for constraint in load.get_active_constraint_generator(start_time, end_time):
                if constraint.end_of_constraint != DATETIME_MAX_UTC:
                    anchors.add(constraint.end_of_constraint)
                active_constraints.append(constraint)

        core_anchors = sorted(list(anchors))

        for i in range(len(core_anchors)-1):
            begin_slot = core_anchors[i]
            end_slot = core_anchors[i+1]
            slots_s = (end_slot - begin_slot).total_seconds()

            num_slots = int(slots_s // self._step_s)

            if num_slots > 0:
                left_over = slots_s % self._step_s
                if left_over < self._step_s//3:
                    #merge the last one with the previous one
                    num_slots -= 1

                for j in range(num_slots):
                    anchors.add(begin_slot + timedelta(seconds=(j+1)*self._step_s))

        anchors = sorted(list(anchors))

        return anchors, active_constraints

    def create_power_slots(self, anchors: list[datetime]) -> tuple[npt.NDArray[np.float64],
                                                             npt.NDArray[np.float64],
                                                             npt.NDArray[np.float64]]:
        """
        Create the power slots for the forecast.

        Args:
            anchors: list: The list of time slots.

        Returns:
            list: A list of power slots.
        """
        prices :npt.NDArray[np.float64]  = np.zeros(len(anchors)-1, dtype=np.float64)
        durations_s: npt.NDArray[np.float64] = np.zeros(len(anchors)-1, dtype=np.float64)
        available_power: npt.NDArray[np.float64] = np.zeros(len(anchors)-1, dtype=np.float64)

        #anchors contain start stop and everything needed to create the slots
        i_tariff = 0
        i_ua = -1 #the latest entry used in the previosu slot
        i_pv = -1


        for i in range(len(anchors)-1):
            begin_slot = anchors[i]
            end_slot = anchors[i+1]

            slots_s = (end_slot - begin_slot).total_seconds()
            durations_s[i] =  slots_s

            # by construction slots have been computed to be inside core anchors ie always inside the same tariff
            # < end_slot and not <= end_slot because we don't want to get the next tarif as the slot is inside
            while i_tariff < len(self._tariffs)-1 and self._tariffs[i_tariff+1][0] < end_slot:
                i_tariff += 1

            prices[i] = self._tariffs[i_tariff][1]

            #now compute the best power for the slots
            i_ua, ua_power = self._power_slot_from_forecast(self._ua_forecast, begin_slot, end_slot, i_ua)
            i_pv, pv_power = self._power_slot_from_forecast(self._pv_forecast, begin_slot, end_slot, i_pv)

            available_power[i] = ua_power - pv_power

        return prices, durations_s, available_power



    def _power_slot_from_forecast(self, forecast, begin_slot, end_slot, last_end):

        if not forecast:
            return last_end, 0.0

        prev_end = last_end
        # <= end_slot and not < end_slot because an exact value on the end of the slot has to be counted "in"
        while (last_end < len(forecast) - 1 and
               forecast[last_end + 1][0] <= end_slot):
            last_end += 1
        # get all the power data in the slot:
        power = []
        if prev_end >= 0:
            if forecast[prev_end][0] == begin_slot:
                power.append(forecast[prev_end][1])
            elif forecast[prev_end][0] < begin_slot and prev_end < len(forecast) - 1:
                adapted_power = forecast[prev_end][1]
                adapted_power += ((forecast[prev_end + 1][1] - forecast[prev_end][1]) *
                                  (begin_slot - forecast[prev_end][0]).total_seconds() /
                                  (forecast[prev_end + 1][0] - forecast[prev_end][0]).total_seconds())
                power.append(adapted_power)
        for j in range(prev_end + 1, last_end + 1):
            power.append(forecast[j][1])
        # if i_ua is not the exact end, we need to adapt the power by computing an additional value
        # (else by construction if self._ua_forecast[i_ua][1] == end_slot it is inside already and in the power values)
        if last_end < len(forecast) - 1 and forecast[last_end][0] < end_slot:
            adapted_power = forecast[last_end][1]
            adapted_power += ((forecast[last_end + 1][1] - forecast[last_end][1]) *
                              (end_slot - forecast[last_end][0]).total_seconds() /
                              (forecast[last_end + 1][0] - forecast[last_end][0]).total_seconds())
            power.append(adapted_power)

        if len(power) == 0 and prev_end == last_end and last_end == len(forecast) - 1:
            power.append(forecast[prev_end][1])

        return last_end, sum(power) / len(power)




    def solve(self) -> tuple[list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]], list[tuple[datetime, LoadCommand]]]:
        """
        Solve the day for the given loads and constraints.

        Args:
            start_time: datetime: The start time of the day.
            end_time: datetime: The end time of the day.

        Returns:
            dict: A dictionary with the scheduled loads.
        """

        # now we can optimize and laid out the constraints we have on the available slots
        # then optimize battery usage on top of that and iterate, may need linear programming to solve
        # but seeing all the forecast uncertainties, may be better to use a fast and simple approach here

        # first get ordered constraints, and they will fill one by one the slots with minimizing the cost for the constraint
        # special treatment for "dynamically adaptable load ex : a car with variable power charge, or the battery itself
        # the battery should have a "soft" constraint to cover the computed left unavidable consmuption (ie sum on the
        # positive numbers in available power)

        #ordering constraints: what are the mandatory constraints that can be filled "quickly" and easily compared to now and their expiration date
        constraints = []
        for c in self._active_constraints:
            if c.is_before_battery:
                constraints.append((c, c.score()))

        constraints= sorted(constraints, key=lambda x: x[1], reverse=True)

        actions = {}

        for c , _ in constraints:
            is_solved, out_commands, out_power = c.compute_best_period_repartition(
                do_use_available_power_only=c.type <= CONSTRAINT_TYPE_BEFORE_BATTERY_AUTO_GREEN,
                prices = self._prices,
                power_slots_duration_s = self._durations_s,
                power_available_power = self._available_power,
                prices_ordered_values = self._prices_ordered_values,
                time_slots = self._time_slots
            )
            self._available_power = self._available_power + out_power
            actions.setdefault(c.load, []).append(out_commands)


        # ok we have solved our mandatory constraints
        # now try to do our best with the non mandatory ones and the variables ones (car) .... and battery

        # try to see the best way of using the battery to cover the consumption at the best price, first by maximizing reuse
        battery_commands = np.zeros(len(self._available_power), dtype=np.float64)

        if self._battery is not None:
            battery_charge = np.zeros(len(self._available_power), dtype=np.float64)
            prev_battery_charge = self._battery.current_charge
            if prev_battery_charge is None:
                prev_battery_charge = 0.0


            # first try to fill the battery to the max of what is left
            total_chargeable_energy = 0
            for i in range(len(self._available_power)):

                charged_energy = 0.0
                charging_power = self._battery.get_best_charge_power(-self._available_power[i], float(self._durations_s[i]), current_charge=prev_battery_charge)
                if charging_power > 0:
                    battery_commands[i] = charging_power
                    charged_energy = (charging_power * float(self._durations_s[i])) / 3600.0
                    total_chargeable_energy += charged_energy
                    self._available_power[i] += charging_power

                battery_charge[i] = prev_battery_charge + charged_energy
                prev_battery_charge = battery_charge[i]


            # now try to discharge the battery to cover the consumption at the best prices
            total_discharged_energy = 0

            for i in range(len(self._available_power)):

                if battery_commands[i] == 0.0:

                    current_battery_charge = battery_charge[i] - total_discharged_energy
                    charging_power = self._battery.get_best_discharge_power(float(self._available_power[i]), float(self._durations_s[i]), current_charge=float(current_battery_charge))

                    if charging_power > 0:
                        total_discharged_energy += (charging_power*self._durations_s[i]/3600.0)
                        battery_commands[i] = -charging_power
                        self._available_power[i] -= charging_power

                battery_charge[i] = battery_charge[i] - total_discharged_energy



        # we may have charged "too much" and if it covers everything, we could remove some charging
        # ...or on the contrary : can we optimise some "expensive" consumption by charging at cheap times?

        #high_prices = np.where(self._prices == self._prices_ordered_values[-1])[0]
        #high_price_energy = np.sum((np.clip(self._available_power[high_prices], 0, None) * self._durations_s[high_prices]) / 3600.0)


        # for the slots where we still have some surplus production : it means the battery was at full capacity ... we can do now other non mandatory stuffs
        # like surplus to the car or other non mandatory constraints, non mandatory constraint are only consuming free electricity ... if anything is left :)

        constraints = []
        for c in self._active_constraints:
            if c.is_before_battery is False:
                constraints.append((c, c.score()))

        constraints = sorted(constraints, key=lambda x: x[1], reverse=True)

        for c , _ in constraints:
            is_solved, out_commands, out_power = c.compute_best_period_repartition(
                do_use_available_power_only=True,
                prices = self._prices,
                power_slots_duration_s = self._durations_s,
                power_available_power = self._available_power,
                prices_ordered_values = self._prices_ordered_values,
                time_slots = self._time_slots
            )
            self._available_power = self._available_power + out_power
            actions.setdefault(c.load, []).append(out_commands)


        # now will be time to layout the commands for the constraints and their respective loads
        # commands are to be sent as change of state for the load attached to the constraints
        output_cmds = []
        num_slots = len(self._durations_s)

        for load, commands_lists in actions.items():
            lcmd = []
            current_command = None
            for s in range(num_slots):
                s_cmd = CMD_IDLE
                for cmds in commands_lists:
                    cmd = cmds[s]
                    #only one should be not NULL...hum why?
                    if cmd is not None:
                        s_cmd = cmd
                        break

                if s_cmd != current_command or (s_cmd.power_consign != current_command.power_consign):
                    lcmd.append((self._time_slots[s], s_cmd))
                    current_command = s_cmd

            output_cmds.append((load, lcmd))

        bcmd = []
        if self._battery is not None:
            # this would be for battery grid charging commands ... not supported for now
            current_command = None
            for s in range(battery_commands.shape[0]):

                charge = float(battery_commands[s])
                if True:
                    param = float(charge)
                    base_cmd = CMD_FORCE_CHARGE #from grid for ex
                    if param < 0.0:
                        param = 0.0 - float(charge)
                        base_cmd = CMD_FORCE_DISCHARGE

                    cmd = copy_command(base_cmd, power_consign=param)

                else:
                    cmd = copy_command(CMD_AUTO_FROM_CONSIGN)

                if cmd != current_command:
                    bcmd.append((self._time_slots[s], cmd))
                    current_command = cmd

        # cmds are correctly ordered by time for each load by construction
        return output_cmds, bcmd





























