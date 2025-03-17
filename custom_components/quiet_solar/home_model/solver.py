import logging
from datetime import datetime
from datetime import timedelta
import numpy.typing as npt
import numpy as np
import pytz

from .battery import Battery, CMD_FORCE_CHARGE, CMD_FORCE_DISCHARGE
from .constraints import LoadConstraint, DATETIME_MAX_UTC
from .load import AbstractLoad
from .commands import LoadCommand, CMD_AUTO_FROM_CONSIGN, copy_command, CMD_IDLE, CMD_GREEN_CHARGE_AND_DISCHARGE, \
    CMD_GREEN_CHARGE_ONLY
from ..const import CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN

_LOGGER = logging.getLogger(__name__)

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
            - float: the price of the period per Wh
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
            self._tariffs = [(start_time, 0.2/1000.0)]
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
        i_ua = -1 #the latest entry used in the previous slot
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


        if self._loads and len(self._loads) > 0:
            home = self._loads[0].home
            # propagate amps limits to the topology
            if home:
                home.allocate_phase_amps_budget(self._start_time)
        else:
            _LOGGER.info(f"solve: NO LOADS!")

        #ordering constraints: what are the mandatory constraints that can be filled "quickly" and easily compared to now and their expiration date
        constraints = []
        for c in self._active_constraints:
            if c.is_before_battery:
                constraints.append((c, c.score()))

        constraints= sorted(constraints, key=lambda x: x[1], reverse=True)

        actions = {}

        for c , _ in constraints:
            is_solved, out_commands, out_power = c.compute_best_period_repartition(
                do_use_available_power_only= not c.is_mandatory,
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
        battery_commands = None

        if self._battery is not None:

            # the battery commands are:
            # - CMD_GREEN_CHARGE_AND_DISCHARGE: charge on solar only (not from grid) discharge when needed
            # - CMD_GREEN_CHARGE_ONLY: do not discharge the battery, charge at maximum from solar
            # - CMD_FORCE_CHARGE: charge the battery according to the power consign value



            init_battery_charge =  self._battery.current_charge
            if init_battery_charge is None:
                init_battery_charge = 0.0

            continue_optimizing = True
            limited_discharge_per_price = {}
            
            num_optim_tries = 0


            while continue_optimizing:
            
                num_optim_tries += 1
                # first try to fill the battery to the max of what is left and consume on all slots : if teh battery covers all (ie no grid need) fine, we need to let the battery do its job
                # in automatic mode, it will discharge when needed, and charge when it can
                # if it is not teh case : remove battery discharge from "lower prices", as much as we can until teh total price decrease ... do that little by little (well limit the number of steps for computation)
                # if the battery "not used energy" from this pass is
                remaining_grid_price  = 0
                remaining_grid_energy = 0
                prev_battery_charge = init_battery_charge
                battery_charge = np.zeros(len(self._available_power), dtype=np.float64)
                battery_commands = [CMD_GREEN_CHARGE_AND_DISCHARGE] * len(self._available_power)
                available_power = np.copy(self._available_power)
                prices_discharged_energy_buckets = {}
                prices_remaining_grid_energy_buckets = {}


                for i in range(len(available_power)):

                    charging_power = 0.0

                    cmd = None

                    if available_power[i] < 0.0:

                        charging_power = self._battery.get_best_charge_power(0.0 - available_power[i], float(self._durations_s[i]), current_charge=prev_battery_charge)
                        if charging_power < 0.0:
                            charging_power = 0.0

                    elif available_power[i] > 0.0:

                        # discharge....
                        charging_power = self._battery.get_best_discharge_power(float(available_power[i]),
                                                                                float(self._durations_s[i]),
                                                                                current_charge=float(prev_battery_charge))
                        if charging_power > 0:
                            charging_power = 0.0 - charging_power
                        else:
                            charging_power = 0.0

                    charged_energy = (charging_power * float(self._durations_s[i])) / 3600.0
                    
                    
                    limit_discharge = limited_discharge_per_price.get(self._prices[i], None)
                    if limit_discharge is not None:
                        if limit_discharge + min(0.0, charged_energy) <= 0.0:
                            # we need to .... forbid discharge
                            charged_energy = max(0.0, charged_energy)
                            charging_power = max(0.0, charging_power)
                            cmd = CMD_GREEN_CHARGE_ONLY
                                
                        limited_discharge_per_price[self._prices[i]] = max(0.0, limit_discharge + min(charged_energy,0.0))


                    if charged_energy < 0.0:
                        prices_discharged_energy_buckets[self._prices[i]] = prices_discharged_energy_buckets.get(self._prices[i], 0.0) - charged_energy 

                    available_power[i] += charging_power

                    if cmd is not None:
                        battery_commands[i] = cmd

                    battery_charge[i] = prev_battery_charge + charged_energy
                    prev_battery_charge = battery_charge[i]


                    if available_power[i] > 0:
                        # we will ask it from the grid:
                        grid_nrj = ((available_power[i]*float(self._durations_s[i])) / 3600.0)
                        remaining_grid_energy += grid_nrj
                        remaining_grid_price += self._prices[i]*grid_nrj
                        prices_remaining_grid_energy_buckets[self._prices[i]] = prices_remaining_grid_energy_buckets.get(self._prices[i], 0.0) + grid_nrj



                if num_optim_tries > 1:
                    # stop after a first pass of optimization
                    continue_optimizing = False

                #the goal is to lower as much as possible the remaining_grid_price ... by, if possible, discharging more in the "high" prices and less in the low prices
                if remaining_grid_energy <= 100 or len(self._prices_ordered_values) <= 1: # 100Wh hum or use a bit of a buffer here for uncertainty?
                    # fantastic we do nothing
                    continue_optimizing = False
                elif continue_optimizing:
                    # will be time now to see if one of the most expensive buckets should be covered my moving discharge for a least expensive one...we may have to do a few tries... well let's be aggressive here and do only one pass
                    # not optimal but will work pretty well in practice
                    limited_discharge_per_price = {}
                    have_an_optim = False
                    for price_idx in range(len(self._prices_ordered_values) - 1, 0, -1): # no need to go to the least expensive
                    
                        price = self._prices_ordered_values[price_idx]
                        energy_to_cover = prices_remaining_grid_energy_buckets.get(price, 0.0)
                        
                        if energy_to_cover <= 0.0:
                            continue
                        else:
                            # we have some energy to cover
                            # take the least expensive energy to cover first
                            for sub_price_idx in range(0, price_idx):

                                sub_price = self._prices_ordered_values[sub_price_idx]
                                energy_can_still_be_discharged = prices_discharged_energy_buckets.get(sub_price, 0.0)
                                if energy_can_still_be_discharged > 0.0:
                                    # we have some energy to cover
                                    if energy_can_still_be_discharged >= energy_to_cover:
                                        _LOGGER.info(f"==> Battery: partial price cover {energy_to_cover} < {energy_can_still_be_discharged}")
                                        # we can cover it
                                        energy_can_still_be_discharged = energy_can_still_be_discharged - energy_to_cover
                                        energy_to_cover = 0

                                        
                                    else:
                                        _LOGGER.info(f"==> Battery: complete price cover {energy_to_cover} > {energy_can_still_be_discharged}")
                                        energy_can_still_be_discharged = 0
                                        energy_to_cover -= energy_can_still_be_discharged

                                    limited_discharge_per_price[sub_price] = energy_can_still_be_discharged
                                    have_an_optim = True
                                    prices_remaining_grid_energy_buckets[sub_price] = prices_remaining_grid_energy_buckets.get(sub_price, 0.0) + prices_discharged_energy_buckets.get(sub_price, 0.0) - energy_can_still_be_discharged
                                    
                                prices_discharged_energy_buckets[sub_price] = energy_can_still_be_discharged

                                if energy_to_cover <= 0:
                                    break

                    if have_an_optim is False:
                        continue_optimizing = False

                if continue_optimizing is False:
                    # now we have the best battery commands for the period
                    self._available_power = available_power


        # we may have charged "too much" and if it covers everything, we could remove some charging
        # ...or on the contrary : can we optimise some "expensive" consumption by charging at cheap times?

        #high_prices = np.where(self._prices == self._prices_ordered_values[-1])[0]
        #high_price_energy = np.sum((np.clip(self._available_power[high_prices], 0, None) * self._durations_s[high_prices]) / 3600.0)


        # for the slots where we still have some surplus production : it means the battery was at full capacity ... we can do now other non mandatory stuffs
        # like surplus to the car or other non mandatory constraints, non mandatory constraint are only consuming free electricity ... if anything is left :)
        # should ONLY be done if the battery is full and we have free electricity by nature here the battery has been charged as much as it can with solar

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
        if self._battery is not None and battery_commands is not None:
            # setup battery comands
            current_command = None
            for s in range(num_slots):
                cmd = battery_commands[s]
                if cmd != current_command or (cmd.power_consign != current_command.power_consign):
                    bcmd.append((self._time_slots[s], cmd))
                    current_command = cmd

        if len(bcmd) == 0:
            bcmd = [(self._time_slots[0], CMD_GREEN_CHARGE_AND_DISCHARGE)]

        # cmds are correctly ordered by time for each load by construction
        return output_cmds, bcmd





























