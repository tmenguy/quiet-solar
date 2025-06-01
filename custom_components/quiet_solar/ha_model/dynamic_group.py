import copy
import logging

from datetime import datetime

import math

from ..const import CONF_DYN_GROUP_MAX_PHASE_AMPS
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice

_LOGGER = logging.getLogger(__name__)

class QSDynamicGroup(HADeviceMixin, AbstractDevice):

    # this class will allow to group Loads togther so the sum of their current can be limited below
    # a setting CONF_DYN_GROUP_MAX_CURRENT
    # It will also allow budgeting of Loads under this same constraint at planning/solving phase
    # but also at dynamic phase, for example allocating solar energy rightfully among connected cars
    # It is not a load but is a way to add a topology to the network of loads than are only leaves
    # of the network . Only Loads have constraints that need to be fullfilled.
    # The home is the root of this tree.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dyn_group_max_phase_current_conf = kwargs.pop(CONF_DYN_GROUP_MAX_PHASE_AMPS, 32)
        self._childrens : list[AbstractDevice] = []
        self.charger_group = None

        self._dyn_group_max_phase_current: list[float|int] | None = None

    @property
    def device_is_3p(self) -> bool:
        if super().device_is_3p:
            return True

        for device in self._childrens:
            if device.device_is_3p:
                return True
        return False

    @property
    def dyn_group_max_phase_current(self) -> list[float|int]:
        if self._dyn_group_max_phase_current is None:
            # we have not been set yet
            if self.device_is_3p:
                self._dyn_group_max_phase_current = (self.dyn_group_max_phase_current_conf, self.dyn_group_max_phase_current_conf, self.dyn_group_max_phase_current_conf)
            else:
                self._dyn_group_max_phase_current = (self.dyn_group_max_phase_current_conf,0, 0)

        return self._dyn_group_max_phase_current



    def is_delta_current_acceptable(self, delta_amps: list[float|int], new_amps_consumption: list[float|int] | None, time:datetime) -> (bool, list[float|int]):

        if new_amps_consumption is not None and self.is_amps_greater(new_amps_consumption, self.dyn_group_max_phase_current):
            return False, self.diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)

        phases_amps = self.get_device_worst_phase_amp_consumption(tolerance_seconds=None, time=time)

        new_amps = [0.0, 0.0, 0.0]
        for i in range(3):
            new_amps[i] = delta_amps[i] + phases_amps[i]


        if self.is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            return False, self.diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)

        if self.father_device is None or self == self.home:
            return True, self.diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)
        else:
            return self.father_device.is_delta_current_acceptable(delta_amps=delta_amps, time=time)

    def is_current_acceptable(self, new_amps: list[float|int], estimated_current_amps: list[float|int] | None, time:datetime) -> bool:
        return self.is_current_acceptable_and_diff(new_amps, estimated_current_amps, time)[0]

    def is_current_acceptable_and_diff(self, new_amps: list[float|int], estimated_current_amps: list[float|int] | None, time:datetime) -> (bool, list[float|int]):

        if self.is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            return False, self.diff_amps(new_amps, self.dyn_group_max_phase_current)

        if estimated_current_amps is None:
            estimated_current_amps = 0.0

        phases_amps = self.get_device_worst_phase_amp_consumption(tolerance_seconds=None, time=time)

        current_phases = [0.0, 0.0, 0.0]
        phases_for_delta = [0.0, 0.0, 0.0]
        if phases_amps is not None and estimated_current_amps is not None:
            phases_for_delta = estimated_current_amps
            for i in range(3):
                current_phases[i] = max(phases_amps[i], estimated_current_amps[i])
        elif estimated_current_amps is not None:
            phases_for_delta = estimated_current_amps
            current_phases = copy.copy(estimated_current_amps)
        elif phases_amps is not None:
            phases_for_delta = phases_amps
            current_phases = copy.copy(phases_amps)


        # get the best possible delta amps as it is a computation coming from outside that is done vs an estimated one
        delta_amps = [0.0, 0.0, 0.0]
        for i in range(3):
            if new_amps[i] is not None:
                delta_amps[i] = new_amps[i] - phases_for_delta[i]


        # recompute the new amps based on the delta and the "worst" current
        new_amps:list[float|int] = [0.0, 0.0, 0.0]
        for i in range(3):
            new_amps[i] = delta_amps[i] + current_phases[i]

        if self.is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            return False, self.diff_amps(new_amps, self.dyn_group_max_phase_current)

        if self.father_device is None or self == self.home:
            return True, self.diff_amps(new_amps, self.dyn_group_max_phase_current)
        else:
            return self.father_device.is_delta_current_acceptable(delta_amps=delta_amps,
                                                                  new_amps_consumption=new_amps,
                                                                  time=time)


    def get_device_power_latest_possible_valid_value(self,
                                                     tolerance_seconds: float | None,
                                                     time:datetime,
                                                     ignore_auto_load:bool= False) -> float:
        if self.accurate_power_sensor is not None:
            p =  self.get_sensor_latest_possible_valid_value(self.accurate_power_sensor, tolerance_seconds, time)
            if p is None:
                return 0.0
            return p
        else:
            power = 0.0
            for device in self._childrens:
                if isinstance(device, HADeviceMixin):
                    p = device.get_device_power_latest_possible_valid_value(tolerance_seconds, time, ignore_auto_load)
                    if p is not None:
                        power += p
            return power

    def get_min_max_power(self) -> (float, float):
        if len(self._childrens) == 0:
            return super().get_min_max_power()

        min_p = 1e12
        max_p = 0
        for device in self._childrens:
            min_p_d, max_p_d = device.get_min_max_power()
            min_p = min(min_p, min_p_d)
            max_p = max(max_p, max_p_d)

        return min_p, max_p

    def get_min_max_phase_amps_for_budgeting(self) -> ( list[float|int],  list[float|int]):
        if len(self._childrens) == 0:
            min_p, max_p =  super().get_min_max_phase_amps_for_budgeting()
        else:
            min_p = [1e12, 1e12, 1e12]
            max_p = [0, 0, 0]
            for device in self._childrens:
                min_p_d, max_p_d = device.get_min_max_phase_amps_for_budgeting()
                for i in range(3):
                    min_p[i] = min(min_p[i], min_p_d[i])
                    max_p[i] = max(max_p[i], max_p_d[i])

        return min_p, max_p

    def get_evaluated_needed_phase_amps_for_budgeting(self, time:datetime) -> list[float|int]:

        device_needed_amp = [0,0,0]
        for device in self._childrens:
            dn = device.get_evaluated_needed_phase_amps_for_budgeting(time)
            for i in range(3):
                device_needed_amp[i] += dn[i]

        return device_needed_amp


    def is_as_fast_as_possible_constraint_active(self, time:datetime) -> bool:
        for device in self._childrens:
            if device.is_as_fast_as_possible_constraint_active(time):
                return True
        return False

    def is_consumption_optional(self, time:datetime) -> bool:
        for device in self._childrens:
            if device.is_consumption_optional(time) is False:
                return False
        return True

    # the idea is that if we have a 3p home we allocate as if phases were balanced for loads that are not 3p
    def allocate_phase_amps_budget(self, time:datetime, from_father_budget:list[float|int] | None = None) -> list[float|int]:

        init_from_father_budget = from_father_budget
        if from_father_budget is None:
            from_father_budget = self.dyn_group_max_phase_current
        else:
            from_father_budget = [ min(from_father_budget[i], self.dyn_group_max_phase_current[i]) for i in range(3) ]

        if from_father_budget is None:
            from_father_budget = [1e8, 1e8, 1e8] # a lot of amps :)


        _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} father budget {init_from_father_budget} => {from_father_budget}")

        self.device_phase_amps_budget = from_father_budget

        # allocate this current budget to all children ... need to understand their needs a bit
        budgets = []
        current_budget_spend = [0.0, 0.0, 0.0]

        budget_as_fast_cluster = {
            "budgets" :[],
            "sum_needed" : [0.0, 0.0, 0.0],
            "name": "as_fast"
        }
        budget_to_be_done_cluster = {
            "budgets" :[],
            "sum_needed" : [0.0, 0.0, 0.0],
            "name": "tbd"
        }
        budget_optional_cluster = {
            "budgets" :[],
            "sum_needed" : [0.0, 0.0, 0.0],
            "name": "optional"
        }

        for device in self._childrens:
            device_needed_amps = device.get_evaluated_needed_phase_amps_for_budgeting(time)
            min_a, max_a = device.get_min_max_phase_amps_for_budgeting()


            if device_needed_amps is None or sum(device_needed_amps) == 0:
                cur_amps = (0.0, 0.0, 0.0)
            else:
                cur_amps = min_a

            # allocate first at minimum
            c_budget = {
                "device": device,
                "3p":device.device_is_3p,
                "needed_amp": device_needed_amps,
                "min_amp": min_a,
                "max_amp": max_a,
                "current_budget":cur_amps
            }

            for i in range(3):
                current_budget_spend[i] += cur_amps[i]

            budget_type = None
            if sum(cur_amps) > 0 and device.is_as_fast_as_possible_constraint_active(time):
                budget_type = budget_as_fast_cluster
            elif sum(cur_amps) > 0 and device.is_consumption_optional(time) is False:
                budget_type = budget_to_be_done_cluster
            elif sum(cur_amps) > 0:
                budget_type = budget_optional_cluster

            if budget_type is not None:
                for i in range(3):
                    budget_type["sum_needed"][i] += device_needed_amps[i]
                budget_type["budgets"].append(c_budget)

            budgets.append(c_budget)

        if self.is_amps_greater(current_budget_spend, from_father_budget):
            # ouch bad we are already over budget ....
            _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name}: initial over amp budget! {current_budget_spend} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]

            current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)

            if self.is_amps_greater(current_budget_spend,from_father_budget):
                _LOGGER.info(
                    f"allocate_phase_amps_budget for a group: {self.name} : initial over amp budget even after shaving {current_budget_spend} > {from_father_budget}")

        # everyone has its minimum already allocated
        if self.is_amps_greater(from_father_budget, current_budget_spend):
            # ok let's put the rest of the budget on the loads if they can get it
            budget_to_allocate = [from_father_budget[i] - current_budget_spend[i] for i in range(3)]

            _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} : budget_to_allocate {budget_to_allocate}")

            clusters = [budget_as_fast_cluster, None, budget_to_be_done_cluster, budget_optional_cluster]

            for i in range(len(clusters)):

                cluster_budget = clusters[i]

                if cluster_budget is None:
                    continue

                if sum(cluster_budget["sum_needed"]) == 0:
                    continue

                do_spread = True
                while do_spread:
                    one_modif = False
                    for c_budget in cluster_budget["budgets"]:
                        if sum(c_budget["current_budget"]) == 0:
                            continue

                        if self.is_amps_greater(c_budget["max_amp"], c_budget["current_budget"]):
                            # 1 amp per one amp budget adaptation ... but relative to the cluster needs
                            delta_budget = 1
                            c_cop = copy.deepcopy(c_budget["current_budget"])
                            if c_budget["3p"]:
                                phase_range = [0,1,2]
                            else:
                                good_phase = c_budget["current_budget"].index(max(c_budget["current_budget"]))
                                phase_range = [good_phase]

                            c_budget["current_budget"] = [
                                min(c_budget["max_amp"][i], c_budget["current_budget"][i] + delta_budget) for i in
                                phase_range]

                            budget_to_allocate = [budget_to_allocate[i] - (c_budget["current_budget"][i] - c_cop[i]) for i in phase_range]
                            current_budget_spend = [current_budget_spend[i] + (c_budget["current_budget"][i] - c_cop[i]) for i in phase_range]

                            one_modif = True

                        if max(budget_to_allocate) <= 0.0001:
                            do_spread = False
                            break

                    if one_modif is False or do_spread is False:
                        break

                if max(budget_to_allocate) <= 0.0001:
                    break


        # now clean a bit the budgets to get to integer for amps

        # ok we do have now good allocation for all budgets
        allocated_final_budget = [0.0,0.0,0.0]
        for c_budget in budgets:
            added_b = c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])
            allocated_final_budget = [allocated_final_budget[i] + added_b[i] for i in range(3)]



        if self.is_amps_greater(allocated_final_budget, from_father_budget):
            _LOGGER.warning(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed last resort shaving {allocated_final_budget} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            new_current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)

            if self.is_amps_greater(new_current_budget_spend, from_father_budget):
                _LOGGER.error(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
                raise ValueError(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
            else:
                allocated_final_budget = [0.0, 0.0, 0.0]
                for c_budget in budgets:
                    added_b = c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])
                    allocated_final_budget = [allocated_final_budget[i] + added_b[i] for i in range(3)]


        self.device_phase_amps_budget = allocated_final_budget

        _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} father budget {init_from_father_budget} => {from_father_budget} => {allocated_final_budget}")

        return allocated_final_budget

    def  _shave_phase_amps_clusters(self, cluster_list_to_shave, current_budget_spend:list[float|int], from_father_budget:list[float|int]) -> list[float|int]:
        for shaved_cluster in cluster_list_to_shave:
            if sum(shaved_cluster["sum_needed"]) == 0:
                continue

            for c_budget in shaved_cluster["budgets"]:
                if sum(c_budget["current_budget"]) == 0:
                    continue

                # by construction all are set at their minimum
                shaved_cluster["sum_needed"] = [shaved_cluster["sum_needed"][i] - c_budget["needed_amp"][i] for i in range(3)]
                c_budget["needed_amp"] = [0,0,0]
                current_budget_spend = [current_budget_spend[i] - c_budget["current_budget"][i] for i in range(3)]
                c_budget["current_budget"] = [0,0,0]


                if not self.is_amps_greater(current_budget_spend, from_father_budget):
                    break

            if not self.is_amps_greater(current_budget_spend, from_father_budget):
                break

        return current_budget_spend





