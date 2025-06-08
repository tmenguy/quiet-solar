import copy
import logging

from datetime import datetime


from typing import Any

from ..const import CONF_DYN_GROUP_MAX_PHASE_AMPS
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice, is_amps_greater, diff_amps, add_amps, max_amps, min_amps, is_amps_zero

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
    def physical_num_phases(self) -> int:
        if super().physical_num_phases == 3:
            return 3

        for device in self._childrens:
            if device.physical_num_phases == 3:
                return 3
        return 1

    @property
    def dyn_group_max_phase_current(self) -> list[float|int]:
        if self._dyn_group_max_phase_current is None:
            # we have not been set yet
            if self.physical_3p:
                self._dyn_group_max_phase_current = [self.dyn_group_max_phase_current_conf, self.dyn_group_max_phase_current_conf, self.dyn_group_max_phase_current_conf]
            else:
                self._dyn_group_max_phase_current = [0, 0, 0]
                self._dyn_group_max_phase_current[self.mono_phase_index] = self.dyn_group_max_phase_current_conf

        return self._dyn_group_max_phase_current



    def is_delta_current_acceptable(self, delta_amps: list[float|int], new_amps_consumption: list[float|int] | None, time:datetime) -> (bool, list[float|int]):

        if new_amps_consumption is not None and is_amps_greater(new_amps_consumption, self.dyn_group_max_phase_current):
            return False, diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)

        phases_amps = self.get_device_amps_consumption(tolerance_seconds=None, time=time)

        new_amps = add_amps(delta_amps, phases_amps)

        if is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            return False, diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)

        if self.father_device is None or self == self.home:
            return True, diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)
        else:
            return self.father_device.is_delta_current_acceptable(delta_amps=delta_amps, time=time)

    def is_current_acceptable(self, new_amps: list[float|int], estimated_current_amps: list[float|int] | None, time:datetime) -> bool:
        return self.is_current_acceptable_and_diff(new_amps, estimated_current_amps, time)[0]

    def is_current_acceptable_and_diff(self, new_amps: list[float|int], estimated_current_amps: list[float|int] | None, time:datetime) -> (bool, list[float|int]):

        if is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            return False, diff_amps(new_amps, self.dyn_group_max_phase_current)

        if estimated_current_amps is None:
            estimated_current_amps = 0.0

        phases_amps = self.get_device_amps_consumption(tolerance_seconds=None, time=time)

        current_phases = [0.0, 0.0, 0.0]
        phases_for_delta = [0.0, 0.0, 0.0]
        if phases_amps is not None and estimated_current_amps is not None:
            phases_for_delta = estimated_current_amps
            current_phases = max_amps(phases_amps, estimated_current_amps)
        elif estimated_current_amps is not None:
            phases_for_delta = estimated_current_amps
            current_phases = copy.copy(estimated_current_amps)
        elif phases_amps is not None:
            phases_for_delta = phases_amps
            current_phases = copy.copy(phases_amps)


        # get the best possible delta amps as it is a computation coming from outside that is done vs an estimated one
        delta_amps = diff_amps(new_amps, phases_for_delta)


        # recompute the new amps based on the delta and the "worst" current
        new_amps = add_amps(delta_amps, current_phases)

        if is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            return False, diff_amps(new_amps, self.dyn_group_max_phase_current)

        if self.father_device is None or self == self.home:
            return True, diff_amps(new_amps, self.dyn_group_max_phase_current)
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
                min_p = min_amps(min_p, min_p_d)
                max_p = max_amps(max_p, max_p_d)

        return min_p, max_p

    def get_evaluated_needed_phase_amps_for_budgeting(self, time:datetime) -> list[float|int]:

        device_needed_amp = [0,0,0]
        for device in self._childrens:
            dn = device.get_evaluated_needed_phase_amps_for_budgeting(time)
            device_needed_amp = add_amps(device_needed_amp, dn)


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
            from_father_budget = min_amps(from_father_budget, self.dyn_group_max_phase_current)

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

            if device_needed_amps is None or is_amps_zero(device_needed_amps):
                cur_amps = [0.0, 0.0, 0.0]
            else:
                cur_amps = min_a

            # allocate first at minimum
            c_budget: dict[str, Any] = {
                "device": device,
                "needed_amp": device_needed_amps,
                "min_amp": min_a,
                "max_amp": max_a,
                "current_budget":cur_amps
            }

            current_budget_spend = add_amps(current_budget_spend, cur_amps)

            budget_type = None
            if not is_amps_zero(cur_amps):
                if device.is_as_fast_as_possible_constraint_active(time):
                    budget_type = budget_as_fast_cluster
                elif device.is_consumption_optional(time) is False:
                    budget_type = budget_to_be_done_cluster
                else:
                    budget_type = budget_optional_cluster


            if budget_type is not None:
                budget_type["sum_needed"] = add_amps(budget_type["sum_needed"], device_needed_amps)
                budget_type["budgets"].append(c_budget)

            budgets.append(c_budget)

        if is_amps_greater(current_budget_spend, from_father_budget):
            # ouch bad we are already over budget ....
            _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name}: initial over amp budget! {current_budget_spend} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)
            if is_amps_greater(current_budget_spend,from_father_budget):
                _LOGGER.info(
                    f"allocate_phase_amps_budget for a group: {self.name} : initial over amp budget even after shaving {current_budget_spend} > {from_father_budget}")

        # everyone has its minimum already allocated
        if is_amps_greater(from_father_budget, current_budget_spend):
            # ok let's put the rest of the budget on the loads if they can get it
            budget_to_allocate = diff_amps(from_father_budget, current_budget_spend)

            _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} : budget_to_allocate {budget_to_allocate}")

            clusters = [budget_as_fast_cluster, budget_to_be_done_cluster, budget_optional_cluster]

            for cluster_budget in clusters:

                if cluster_budget is None:
                    continue

                if is_amps_zero(cluster_budget["sum_needed"]):
                    continue

                do_spread = True
                while do_spread:
                    one_modif = False
                    for c_budget in cluster_budget["budgets"]:
                        if is_amps_zero(c_budget["current_budget"]):
                            continue

                        if is_amps_greater(c_budget["max_amp"], c_budget["current_budget"]):

                            budget_to_allocate = add_amps(budget_to_allocate, c_budget["current_budget"])
                            current_budget_spend = diff_amps(current_budget_spend, c_budget["current_budget"])

                            device = c_budget["device"]
                            c_budget["current_budget"] = device.update_amps_with_delta(from_amps=c_budget["current_budget"], delta=1, is_3p=device.physical_3p)
                            c_budget["current_budget"] = min_amps(c_budget["current_budget"], c_budget["max_amp"])

                            budget_to_allocate = diff_amps(budget_to_allocate, c_budget["current_budget"])
                            current_budget_spend = add_amps(current_budget_spend, c_budget["current_budget"])

                            one_modif = True

                        if max(budget_to_allocate) <= 0.01:
                            do_spread = False
                            break

                    if one_modif is False or do_spread is False:
                        break

                if max(budget_to_allocate) <= 0.01:
                    break


        # now clean a bit the budgets to get to integer for amps

        # ok we do have now good allocation for all budgets
        allocated_final_budget = [0.0,0.0,0.0]
        for c_budget in budgets:
            added_b = c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])
            allocated_final_budget = add_amps(allocated_final_budget, added_b)


        if is_amps_greater(allocated_final_budget, from_father_budget):
            _LOGGER.warning(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed last resort shaving {allocated_final_budget} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            new_current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)

            if is_amps_greater(new_current_budget_spend, from_father_budget):
                _LOGGER.error(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
                raise ValueError(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
            else:
                allocated_final_budget = [0.0, 0.0, 0.0]
                for c_budget in budgets:
                    added_b = c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])
                    allocated_final_budget = add_amps(allocated_final_budget, added_b)


        self.device_phase_amps_budget = allocated_final_budget

        _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} father budget {init_from_father_budget} => {from_father_budget} => {allocated_final_budget}")

        return allocated_final_budget

    def  _shave_phase_amps_clusters(self, cluster_list_to_shave, current_budget_spend:list[float|int], from_father_budget:list[float|int]) -> list[float|int]:
        for shaved_cluster in cluster_list_to_shave:
            if is_amps_zero(shaved_cluster["sum_needed"]):
                continue

            for c_budget in shaved_cluster["budgets"]:
                if is_amps_zero(c_budget["current_budget"]):
                    continue

                # by construction all are set at their minimum
                shaved_cluster["sum_needed"] = diff_amps(shaved_cluster["sum_needed"], c_budget["needed_amp"])
                c_budget["needed_amp"] = [0,0,0]
                current_budget_spend = diff_amps(current_budget_spend, c_budget["current_budget"])
                c_budget["current_budget"] = [0,0,0]


                if not is_amps_greater(current_budget_spend, from_father_budget):
                    break

            if not is_amps_greater(current_budget_spend, from_father_budget):
                break

        return current_budget_spend





