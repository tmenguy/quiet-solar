import logging
from datetime import datetime
from datetime import timedelta
import numpy.typing as npt
import numpy as np
import pytz

from .battery import Battery
from .constraints import LoadConstraint, DATETIME_MAX_UTC
from .load import AbstractLoad
from .commands import LoadCommand, copy_command, CMD_IDLE, CMD_GREEN_CHARGE_AND_DISCHARGE, \
    CMD_GREEN_CHARGE_ONLY, merge_commands, CMD_AUTO_GREEN_CAP, CMD_AUTO_GREEN_ONLY, copy_command_and_change_type


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


        if pv_forecast is None:
            _LOGGER.warning("PeriodSolver: NO SOLAR FORECAST FROM INPUT")
        elif np.sum(self._available_power) == 0.0:
            _LOGGER.warning("PeriodSolver: NO SOLAR FORECAST 0 SUM")


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

            constraints = load.get_for_solver_constraints(start_time, end_time)
            for constraint in constraints:
                if constraint.end_of_constraint != DATETIME_MAX_UTC and constraint.end_of_constraint <= end_time:
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



    def _merge_commands_slots_for_load(self, loads, constraint, first_slot, last_slot, new_command_list, prio_on_new=False):

        if new_command_list is None:
            return

        load = constraint.load

        existing_cmds = loads.get(load, None)
        if existing_cmds is None:
            existing_cmds = new_command_list
            loads[load] = new_command_list

        default_cmd = CMD_IDLE
        if constraint.support_auto:
            # if the constraint supports auto, we should use the auto green command as default
            default_cmd = CMD_AUTO_GREEN_ONLY

        # for s in range(len(new_command_list)):
        for s in range(len(new_command_list)):

            new_cmd = new_command_list[s]

            prev_cmd = existing_cmds[s]
            if prev_cmd is None:
                prev_cmd = copy_command(default_cmd)

            cmd = None

            if new_cmd is None:
                cmd = prev_cmd
            elif prio_on_new:
                cmd = new_cmd

            if cmd is None:
                cmd = merge_commands(prev_cmd, new_cmd)

            existing_cmds[s] = cmd

    def _battery_get_charging_power(self, limited_discharge_per_price = None):

        available_power_list = self._available_power
        battery_charge_power = np.zeros(len(available_power_list), dtype=np.float64)
        battery_charge = np.zeros(len(available_power_list), dtype=np.float64)
        battery_commands = [CMD_GREEN_CHARGE_AND_DISCHARGE] * len(available_power_list)
        prices_discharged_energy_buckets = {}
        prices_remaining_grid_energy_buckets = {}
        remaining_grid_price = 0
        remaining_grid_energy = 0
        excess_solar_energy = 0

        if self._battery:

            init_battery_charge = self._battery.current_charge
            if init_battery_charge is None:
                init_battery_charge = self._battery.get_value_empty()

            prev_battery_charge = init_battery_charge

            for i in range(len(available_power_list)):

                charging_power = 0.0
                available_power = available_power_list[i]

                if available_power < 0.0:

                    charging_power = self._battery.get_best_charge_power(0.0 - available_power,
                                                                         float(self._durations_s[i]),
                                                                         current_charge=prev_battery_charge)
                    if charging_power < 0.0:
                        charging_power = 0.0

                elif available_power > 0.0:

                    # discharge....
                    charging_power = self._battery.get_best_discharge_power(float(available_power),
                                                                            float(self._durations_s[i]),
                                                                            current_charge=float(prev_battery_charge))
                    if charging_power > 0:
                        charging_power = 0.0 - charging_power
                    else:
                        charging_power = 0.0

                battery_charge_power[i] = charging_power

                charged_energy = (charging_power * float(self._durations_s[i])) / 3600.0

                if limited_discharge_per_price is not None:
                    limit_discharge = limited_discharge_per_price.get(self._prices[i], None)
                    if limit_discharge is not None:
                        if limit_discharge + min(0.0, charged_energy) <= 0.0:
                            # we need to ... forbid discharge to keep it when we will need it for bigger prices
                            charged_energy = max(0.0, charged_energy)
                            charging_power = max(0.0, charging_power)
                            battery_commands[i] = CMD_GREEN_CHARGE_ONLY

                        limited_discharge_per_price[self._prices[i]] = max(0.0, limit_discharge + min(charged_energy, 0.0))

                if charged_energy < 0.0:
                    prices_discharged_energy_buckets[self._prices[i]] = prices_discharged_energy_buckets.get(
                        self._prices[i], 0.0) - charged_energy

                grid_nrj = (((available_power + charging_power) * float(self._durations_s[i])) / 3600.0)
                if grid_nrj > 0:
                    # we will ask it from the grid:
                    remaining_grid_energy += grid_nrj
                    remaining_grid_price += self._prices[i] * grid_nrj
                    prices_remaining_grid_energy_buckets[self._prices[i]] = prices_remaining_grid_energy_buckets.get(
                        self._prices[i], 0.0) + grid_nrj
                elif grid_nrj < 0:
                    excess_solar_energy = excess_solar_energy - grid_nrj


                battery_charge[i] = prev_battery_charge + charged_energy
                prev_battery_charge = battery_charge[i]


        return battery_charge_power, battery_charge, battery_commands, prices_discharged_energy_buckets, prices_remaining_grid_energy_buckets, excess_solar_energy, remaining_grid_energy


    def _prepare_battery_segmentation(self):

        to_shave_segment = None
        energy_delta = None

        num_slots = len(self._available_power)
        # check battery: if we give back to grid and battery is full: we should consume more from the grid to avoid giving back to grid, so we should not discharge the battery
        battery_charge_power, battery_charge, battery_commands, prices_discharged_energy_buckets, prices_remaining_grid_energy_buckets, excess_solar_energy, remaining_grid_energy = self._battery_get_charging_power()

        empty_segments = [[None, num_slots - 1]]
        for i in range(len(self._available_power)):

            current_charge = float(battery_charge[i])
            if self._battery.is_value_empty(current_charge*0.9): #be a bit pessimistic here too ... 10%
                if empty_segments[-1][0] is None:
                    empty_segments[-1][0] = i
                else:
                    empty_segments[-1][1] = i
            else:
                if empty_segments[-1][0] is not None:
                    empty_segments.append([None, num_slots - 1])
        # if the last segment is empty, we remove it
        if empty_segments[-1][0] is None:
            empty_segments.pop()

        if len(empty_segments) == 0 or (
                len(empty_segments) == 1 and (empty_segments[0][0] == 0 and empty_segments[0][1] == num_slots - 1)):
            # not empty segments, so we can use the battery as we want ... or all empty we can do nothing
            pass
        else:
            energy_to_get_back = [0.0] * len(empty_segments)
            segments_to_shave = [None] * len(empty_segments)
            for s_idx, s in enumerate(empty_segments):

                for i in range(s[0], s[1] + 1):
                    energy_to_get_back[s_idx] += max(0.0, self._available_power[i]) * self._durations_s[i] / 3600.0

                if s_idx == 0:
                    if s[0] > 0:
                        segments_to_shave[s_idx] = [0, s[0] - 1]
                else:
                    segments_to_shave[s_idx] = [empty_segments[s_idx - 1][1] + 1, s[0] - 1]

                if segments_to_shave[s_idx] is not None:
                    for i in range(segments_to_shave[s_idx][0], segments_to_shave[s_idx][1] + 1):
                        energy_to_get_back[s_idx] += max(0.0, self._available_power[i]) * self._durations_s[i] / 3600.0

            for s_idx in range(len(segments_to_shave)):
                s = segments_to_shave[s_idx]
                if s is None:
                    continue

                to_shave_segment = [s[0], empty_segments[s_idx][1]]

                init_battery_charge = self._battery.current_charge
                if init_battery_charge is None:
                    init_battery_charge = self._battery.get_value_empty()

                energy_delta = -min(self._battery.get_value_full() - min(float(battery_charge[to_shave_segment[0]]), init_battery_charge), energy_to_get_back[s_idx]*1.1) # bump a bit what need to be reclaimed, 10% here ...
                break

        return to_shave_segment, energy_delta


    def _constraints_delta(self, energy_delta, constraints, constraints_evolution, constraints_bounds, actions, seg_start, seg_end, allow_change_state):

        solved = False
        has_changed = False

        if energy_delta != 0:

            if energy_delta > 0:
                # mor consumption, start with the more important ones
                constraints = sorted(constraints, key=lambda x: x[1], reverse=True)
            else:
                # energy to reclaim
                # start with the less important constraints first, to be reduced if needed
                constraints = sorted(constraints, key=lambda x: x[1], reverse=False)

            orig_energy_delta = energy_delta


            if energy_delta > 0:
                _LOGGER.info(
                    f"_constraints_delta: trying to consume more: {energy_delta}Wh from {self._time_slots[seg_start]} to {self._time_slots[seg_end]} for loads {[ f"{c.load.name} {score_c}" for c, score_c in constraints]}")
            else:
                _LOGGER.info(
                    f"_constraints_delta: trying to reclaim: {energy_delta}Wh from {self._time_slots[seg_start]} to {self._time_slots[seg_end]} for loads {[ f"{c.load.name} {score_c}" for c, score_c in constraints]}")
            load_to_re_adapt = set()

            for ci, _ in constraints:
                out_c = constraints_evolution[ci]
                first_slot, last_slot, first_change_slot, last_change_slot = constraints_bounds.get(ci, (None, None, None, None))

                if first_slot is None or last_slot is None:
                    _LOGGER.warning(f"_constraints_delta: constraint {ci} has no bounds, skipping")
                    continue

                out_commands = actions.get(ci.load, None)

                if seg_start > last_slot or first_slot > seg_end or out_commands is None:
                    # segment is not in the range of the constraint, skip it
                    continue
                st = max(seg_start, first_slot)
                nd = min(seg_end, last_slot)
                if st > nd:
                    continue


                # energy_delta can be negative or positive, negative means reduce the consumed energy bay the constraint
                init_energy_delta = energy_delta
                out_c_adapted, solved, has_changes, energy_delta, out_commands_adapted, out_delta_power = out_c.adapt_repartition(
                    first_slot=st,
                    last_slot=nd,
                    energy_delta=energy_delta,
                    power_slots_duration_s=self._durations_s,
                    existing_commands=out_commands,
                    allow_change_state=allow_change_state,
                    time=self._start_time)
                if has_changes:
                    _LOGGER.info(
                        f"_constraints_delta: {ci.load.name} remaining: {energy_delta} init: {init_energy_delta} Wh orig ask: {orig_energy_delta}Wh from {self._time_slots[st]} to {self._time_slots[nd]}")
                    has_changed = True
                    constraints_evolution[ci] = out_c_adapted
                    self._available_power = self._available_power + out_delta_power
                    self._merge_commands_slots_for_load(actions, ci, st, nd, out_commands_adapted, prio_on_new=True)
                else:
                    _LOGGER.info(
                        f"_constraints_delta: {ci.load.name} no change, energy delta: {energy_delta} Wh orig ask: {orig_energy_delta}Wh from {self._time_slots[st]} to {self._time_slots[nd]}")

                if ci.support_auto:
                    load_to_re_adapt.add(ci.load)


                if solved:
                    break

            # adapt all constraints that would need caping to respect less consumption
            if orig_energy_delta < 0 and len(load_to_re_adapt) > 1:
                # need to CAP all auto load if there is a CAP somewhere
                for i in range(seg_start, seg_end + 1):
                    has_cap = False

                    for load in load_to_re_adapt:
                        cmds = actions.get(load,None)
                        if cmds is not None:
                            cmd = cmds[i]
                            if cmd is not None and cmd.is_like(CMD_AUTO_GREEN_CAP):
                                # we need to cap it
                                has_cap = True
                                break

                    if has_cap:
                        for load in load_to_re_adapt:
                            cmds = actions.get(load, None)
                            if cmds is not None:
                                cmd = cmds[i]
                                if cmd is None:
                                    cmds[i] = copy_command(CMD_AUTO_GREEN_CAP)
                                elif cmd.is_like(CMD_AUTO_GREEN_ONLY):
                                    # we need to cap it
                                    cmds[i] = copy_command_and_change_type(cmd, CMD_AUTO_GREEN_CAP.command)

        return solved, has_changed, energy_delta



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

        constraints_evolution = {}
        constraints_bounds = {}


        if self._loads and len(self._loads) > 0:
            home = self._loads[0].home
            # propagate amps limits to the topology

            if home:
                load_set = set()
                for c in self._active_constraints:
                    load_set.add(c.load)

                home.budget_for_loads(self._start_time, list(load_set))
        else:
            _LOGGER.info(f"solve: NO LOADS!")

        #ordering constraints: what are the mandatory constraints that can be filled "quickly" and easily compared to now and their expiration date


        actions = {}
        num_slots = len(self._durations_s)

        constraints = []
        for c in self._active_constraints:
            if c.is_before_battery:
                constraints.append((c, c.score(self._start_time)))

        constraints= sorted(constraints, key=lambda x: x[1], reverse=True)

        for ci , _ in constraints:

            c = constraints_evolution.get(ci,ci)
            out_c, is_solved, out_commands, out_power, first_slot, last_slot, min_idx_with_energy_impact, max_idx_with_energy_impact = c.compute_best_period_repartition(
                do_use_available_power_only=not c.is_mandatory,
                prices = self._prices,
                power_slots_duration_s = self._durations_s,
                power_available_power = self._available_power,
                prices_ordered_values = self._prices_ordered_values,
                time_slots = self._time_slots
            )
            constraints_evolution[ci] = out_c
            constraints_bounds[ci] = (first_slot, last_slot, min_idx_with_energy_impact, max_idx_with_energy_impact)
            self._available_power = self._available_power + out_power
            self._merge_commands_slots_for_load(actions, ci, first_slot, last_slot, out_commands)

        constraints = []
        for c in self._active_constraints:
            if c.is_before_battery and c.is_mandatory is False:
                constraints.append((c, c.score(self._start_time)))

        if len(constraints) > 0 and self._battery is not None:

            # we have some spots where we need grid energy...so we consume perhaps too much already with the non mandatory constraints

            # segment battery usage by spots where the battery will be empty, and try to cap the commands sent
            # for non mandatory constraints, we will try to cap the commands sent to the battery so it can charge more

            while True:
                # check battery: if we give back to grid and battery is full: we should consume more from the grid to avoid giving back to grid, so we should not discharge the battery
                to_shave_segment, energy_delta = self._prepare_battery_segmentation()

                if to_shave_segment is None:
                    _LOGGER.info(
                        f"solve: No segment to shave for battery low points")
                    break
                else:

                    # now we need to get the commands that are coming from the non mandatory constraints
                    # and try to limit them so the battery can charge more and reclaim energy_to_get_back[s_idx] in the process
                    # to not ask anything from the grid
                    solved, has_changed, energy_delta = self._constraints_delta(energy_delta,
                                                                                constraints,
                                                                                constraints_evolution,
                                                                                constraints_bounds,
                                                                                actions,
                                                                                to_shave_segment[0],
                                                                                to_shave_segment[1],
                                                                                allow_change_state=True)

                    if has_changed:
                        _LOGGER.info(
                            f"solve: cap succeeded in saving some battery, left to save: {energy_delta} Wh {solved} / {has_changed}")

                    if solved or has_changed is False:
                        _LOGGER.info(
                            f"solve: cap no more constraints to adapt, energy left to save: {energy_delta} Wh {solved} / {has_changed}")
                        break


        # try to see the best way of using the battery to cover the consumption at the best price, first by maximizing reuse
        battery_commands = None
        battery_charge = None

        if self._battery is not None:

            # the battery commands are:
            # - CMD_GREEN_CHARGE_AND_DISCHARGE: charge on solar only (not from grid) discharge when needed
            # - CMD_GREEN_CHARGE_ONLY: do not discharge the battery, charge at maximum from solar
            # - CMD_FORCE_CHARGE: charge the battery according to the power consign value

            continue_optimizing = True
            limited_discharge_per_price = {}
            
            num_optim_tries = 0

            # first try to fill the battery to the max of what is left and consume on all slots : if the battery covers all (ie no grid need) fine, we need to let the battery do its job
            # in automatic mode, it will discharge when needed, and charge when it can
            # if it is not the case :
            # first, if usefull try to limit some non mandatory loads so the battery can charge more
            # if not enough remove battery discharge from "lower prices", as much as we can until the total price decrease ... do that little by little (well limit the number of steps for computation)
            # if the battery "not used energy" from this pass is

            while continue_optimizing:
            
                num_optim_tries += 1

                battery_charge_power, battery_charge, battery_commands, prices_discharged_energy_buckets, prices_remaining_grid_energy_buckets, excess_solar_energy, remaining_grid_energy = self._battery_get_charging_power(limited_discharge_per_price=limited_discharge_per_price)

                if num_optim_tries > 1:
                    # stop after a first pass of optimization
                    continue_optimizing = False

                #the goal is to lower as much as possible the remaining_grid_energy ... by, if possible, discharging more in the "high" prices and less in the low prices
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
                                        _LOGGER.info(f"solve:==> Battery: partial price cover {energy_to_cover} < {energy_can_still_be_discharged}")
                                        # we can cover it
                                        energy_can_still_be_discharged = energy_can_still_be_discharged - energy_to_cover
                                        energy_to_cover = 0

                                        
                                    else:
                                        _LOGGER.info(f"solve:==> Battery: complete price cover {energy_to_cover} > {energy_can_still_be_discharged}")
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
                    self._available_power = self._available_power + battery_charge_power


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
                constraints.append((c, c.score(self._start_time)))

        constraints = sorted(constraints, key=lambda x: x[1], reverse=True)

        for ci , _ in constraints:
            c = constraints_evolution.get(ci, ci)
            out_c, is_solved, out_commands, out_power, first_slot, last_slot, min_idx_with_energy_impact, max_idx_with_energy_impact = c.compute_best_period_repartition(
                do_use_available_power_only=True,
                prices = self._prices,
                power_slots_duration_s = self._durations_s,
                power_available_power = self._available_power,
                prices_ordered_values = self._prices_ordered_values,
                time_slots = self._time_slots
            )
            constraints_evolution[ci] = out_c
            constraints_bounds[ci] = (first_slot, last_slot, min_idx_with_energy_impact, max_idx_with_energy_impact)
            self._available_power = self._available_power + out_power
            self._merge_commands_slots_for_load(actions, ci, first_slot, last_slot, out_commands)


        if self._battery is not None and battery_charge is not None:
            # We may have a path here if we still do have some surplus and battery is full : we may be ok to force a bit some loads to consume more and use the battery for a time so the battery because the battery could fill itself back with solar
            # and we won't give back anything to the grid
            energy_given_back_to_grid = 0.0

            #limit this to the next 6 hours
            duration_s = 0.0
            first_surplus_index = None
            last_surplus_index = None
            for i in range(num_slots):
                if self._available_power[i] < 0.0 and self._battery.is_value_full(battery_charge[i]):
                    energy_given_back_to_grid += ((self._available_power[i] * float(self._durations_s[i])) / 3600.0)
                    if first_surplus_index is None:
                        first_surplus_index = i
                    last_surplus_index = i

                duration_s += self._durations_s[i]
                if duration_s > 6*3600:
                    break

            if energy_given_back_to_grid < 0.0 and first_surplus_index is not None and last_surplus_index is not None:

                energy_to_be_spent = (-energy_given_back_to_grid/2.0)*0.8 # try to reuse 80% of the estimated energy given back to the grid, so we can try to force some loads to consume more

                probe_window_start = first_surplus_index  # we may want to grab before the battery is full, take 1 or 2 hours

                if first_surplus_index > 0:
                    duration_s = 0.0
                    for i in range(first_surplus_index - 1, -1, -1):
                        duration_s += self._durations_s[i]
                        if duration_s > 2 * 3600:
                            break
                        probe_window_start = i

                probe_window_end = last_surplus_index

                # we have some surplus, limit the windw to reclaim energy so what is left can fill the battery
                nrj_to_recharge = (-energy_given_back_to_grid/2.0)
                for i in range(last_surplus_index, -1, -1):
                    if self._available_power[i] < 0.0:
                        nrj_to_recharge += ((self._available_power[i] * float(self._durations_s[i])) / 3600.0) # self._available_power[i] negative
                    probe_window_end = i
                    if nrj_to_recharge <= 0.0:
                        break

                if probe_window_end >= probe_window_start:
                    # all the mandatory are covered as they can be, now we can try to force some loads to consume more energy
                    # we have some energy given back to the grid, so we can try to force some loads to consume more
                    # this is only possible if the battery is full and we have some surplus

                    _LOGGER.info(
                        f"solve:Estimated Energy given back to the grid for the next 6 hours: {energy_given_back_to_grid} Wh get back {energy_to_be_spent} Wh")

                    # if possible we can bump any possible constraint has we know it is energy that we can use
                    # even if they are already met ? in fact the only issue would be to consume energy of a met constraint that has been met before the probe_window_start!

                    constraints = []
                    all_c = []
                    # if possible we can bump any possible contraint has we know it is energy that we can use
                    for c in self._active_constraints:
                        c_now = constraints_evolution.get(c, c)
                        first_slot, last_slot, min_idx_with_energy_impact, max_idx_with_energy_impact = constraints_bounds.get(c, (None, None, -1, -1))
                        add_to_probe = True
                        if c_now.is_constraint_met(self._start_time) and max_idx_with_energy_impact < probe_window_start:
                            # we don't wan't to give a constraint some more energy AFTER it has been completed
                            add_to_probe = False

                        if add_to_probe:
                            constraints.append((c, c.score(self._start_time)))

                        all_c.append((c, c.score(self._start_time), c_now.is_constraint_met(self._start_time), c.is_mandatory))

                    _LOGGER.info(
                        f"solve:Estimated Energy given back all cts: {[f"{c.load.name} met:{met} mandatory:{mand} score:{score}" for c, score, met, mand in all_c]}")

                    if len(constraints) > 0:

                        while True:

                            # instead of surplus index I could go to first_surplus_index = 0 ti start now to consume battery energy
                            # as we will get a lot more surplus ...
                            solved, has_changed, energy_to_be_spent = self._constraints_delta(energy_to_be_spent,
                                                                                              constraints,
                                                                                              constraints_evolution,
                                                                                              constraints_bounds,
                                                                                              actions,
                                                                                              probe_window_start,
                                                                                              probe_window_end,
                                                                                              True)
                            if has_changed:
                                _LOGGER.info(
                                    f"solve: Surplus succeeded in consuming more for surplus {energy_to_be_spent} Wh {solved} / {has_changed}")

                            if solved or has_changed is False:
                                _LOGGER.info(
                                    f"solve: Surplus No more constraints to adapt, energy to be spent: {energy_to_be_spent} Wh {solved} / {has_changed}")
                                break


        # we have now all the constraints solved, and the battery commands computed
        # now will be time to layout the commands for the constraints and their respective loads
        # commands are to be sent as change of state for the load attached to the constraints
        output_cmds = []

        for load, command_list in actions.items():
            lcmd = []
            current_command = None
            for s in range(num_slots):
                s_cmd = command_list[s]
                if s_cmd is None:
                    s_cmd = CMD_IDLE

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





























