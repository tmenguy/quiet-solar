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
        self.dyn_group_max_phase_current = kwargs.pop(CONF_DYN_GROUP_MAX_PHASE_AMPS, 32)
        self._childrens : list[AbstractDevice] = []
        self.charger_group = None

    @property
    def device_is_3p(self) -> bool:
        if super().device_is_3p:
            return True

        for device in self._childrens:
            if device.device_is_3p:
                return True
        return False

    def is_delta_current_acceptable(self, delta_amps: float, new_amps_consumption: float | None, time:datetime) -> bool:

        if new_amps_consumption is not None and new_amps_consumption > self.dyn_group_max_phase_current:
            return False

        worst_amps = self.get_device_worst_phase_amp_consumption(tolerance_seconds=None, time=time)

        if worst_amps + delta_amps > self.dyn_group_max_phase_current:
            return False

        if self.father_device is None or self == self.home:
            return True
        else:
            return self.father_device.is_delta_current_acceptable(delta_amps=delta_amps, time=time)

    def is_current_acceptable(self, new_amps: float, estimated_current_amps: float | None, time:datetime) -> bool:

        if new_amps > self.dyn_group_max_phase_current:
            return False

        if estimated_current_amps is None:
            estimated_current_amps = 0.0

        measured_amps = self.get_device_worst_phase_amp_consumption(tolerance_seconds=None, time=time)

        if measured_amps == 0.0:
            current_amps = estimated_current_amps
        elif estimated_current_amps == 0.0:
            current_amps = measured_amps
        else:
            current_amps = min(measured_amps, estimated_current_amps)

        # maximize the delta
        delta_amps = new_amps - current_amps

        new_amps = delta_amps + max(estimated_current_amps, measured_amps)

        if new_amps > self.dyn_group_max_phase_current:
            return False

        if self.father is None or self == self.home:
            return True
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

    def get_min_max_phase_amps_for_budgeting(self) -> (float, float):
        if len(self._childrens) == 0:
            min_p, max_p =  super().get_min_max_phase_amps_for_budgeting()
        else:
            min_p = 1e12
            max_p = 0
            for device in self._childrens:
                min_p_d, max_p_d = device.get_min_max_phase_amps_for_budgeting()
                min_p = min(min_p, min_p_d)
                max_p = max(max_p, max_p_d)

        if self.father_device.device_is_3p and not self.device_is_3p:
            min_p = min_p / 3.0
            max_p = max_p / 3.0

        return min_p, max_p

    def get_evaluated_needed_phase_amps_for_budgeting(self, time:datetime) -> float:

        device_needed_amp = 0.0
        for device in self._childrens:
            device_needed_amp += device.get_evaluated_needed_phase_amps_for_budgeting(time)

        if self.father_device.device_is_3p and not self.device_is_3p:
            device_needed_amp = device_needed_amp / 3.0

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
    def allocate_phase_amps_budget(self, time:datetime, from_father_budget:float|None = None) -> float:


        init_from_father_budget = from_father_budget
        if from_father_budget is None:
            from_father_budget = self.dyn_group_max_phase_current
        else:
            from_father_budget = min(from_father_budget, self.dyn_group_max_phase_current)

        if from_father_budget is None:
            from_father_budget = 1e8 # a lot of amps :)

        if self.father_device.device_is_3p and not self.device_is_3p:
            # we have been counting in budget only a third of the need for single phase load in a 3 phase father group (or home)
            from_father_budget = 3*from_father_budget

        _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} father budget {init_from_father_budget} => {from_father_budget}")

        self.device_phase_amps_budget = from_father_budget

        # allocate this current budget to all children ... need to understand their needs a bit
        budgets = []
        current_budget_spend = 0

        budget_as_fast_cluster = {
            "budgets" :[],
            "sum_needed" : 0,
            "name": "as_fast"
        }
        budget_to_be_done_cluster = {
            "budgets" :[],
            "sum_needed" : 0,
            "name": "tbd"
        }
        budget_optional_cluster = {
            "budgets" :[],
            "sum_needed" : 0,
            "name": "optional"
        }

        for device in self._childrens:
            device_needed_amp = device.get_evaluated_needed_phase_amps_for_budgeting(time)
            min_a, max_a = device.get_min_max_phase_amps_for_budgeting()

            min_a = math.ceil(min_a)
            max_a = math.ceil(max_a)

            if device_needed_amp == 0:
                cur_amps = 0.0
            else:
                cur_amps = min_a

            # allocate first at minimum
            c_budget = {
                "device": device,
                "needed_amp": device_needed_amp,
                "min_amp": min_a,
                "max_amp": max_a,
                "current_budget":cur_amps
            }
            current_budget_spend += cur_amps

            if cur_amps > 0 and device.is_as_fast_as_possible_constraint_active(time):
                budget_as_fast_cluster["sum_needed"] += device_needed_amp
                budget_as_fast_cluster["budgets"].append(c_budget)
            elif cur_amps > 0 and device.is_consumption_optional(time) is False:
                budget_to_be_done_cluster["sum_needed"] += device_needed_amp
                budget_to_be_done_cluster["budgets"].append(c_budget)
            elif cur_amps > 0:
                budget_optional_cluster["sum_needed"] += device_needed_amp
                budget_optional_cluster["budgets"].append(c_budget)

            budgets.append(c_budget)

        if current_budget_spend > from_father_budget:
            # ouch bad we are already over budget ....
            _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name}: initial over amp budget! {current_budget_spend} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)
            if current_budget_spend > from_father_budget:
                _LOGGER.info(
                    f"allocate_phase_amps_budget for a group: {self.name} : initial over amp budget even after shaving {current_budget_spend} > {from_father_budget}")

        # everyone has its minimum already allocated
        if current_budget_spend < from_father_budget:
            # ok let's put the rest of the budget on the loads if they can get it
            budget_to_allocate = from_father_budget - current_budget_spend

            _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} : budget_to_allocate {budget_to_allocate}")

            clusters = [budget_as_fast_cluster, None, budget_to_be_done_cluster, budget_optional_cluster]

            for i in range(len(clusters)):

                cluster_budget = clusters[i]

                if cluster_budget is None:
                    continue

                if cluster_budget["sum_needed"] == 0:
                    continue

                do_spread = True
                while do_spread:
                    one_modif = False
                    for c_budget in cluster_budget["budgets"]:
                        if c_budget["current_budget"] == 0:
                            continue

                        if c_budget["current_budget"] < c_budget["max_amp"]:
                            # 1 amp per one amp budget adaptation ... but relative to the cluster needs
                            delta_budget = math.ceil(min(min(c_budget["max_amp"] - c_budget["current_budget"], 1), budget_to_allocate))
                            c_budget["current_budget"] += delta_budget
                            budget_to_allocate -= delta_budget
                            current_budget_spend += delta_budget
                            one_modif = True

                        if budget_to_allocate <= 0.0001:
                            do_spread = False
                            break

                    if one_modif is False or do_spread is False:
                        do_spread = False
                        break

                # specific case for the as fast: we may want to remove any optional budget to better fill the as fast
                if cluster_budget["name"] == "as_fast" and budget_to_allocate <= 0.0001:
                    missing_budget = 0
                    for c_budget in cluster_budget["budgets"]:
                        if c_budget["current_budget"] == 0:
                            continue

                        if c_budget["current_budget"] < c_budget["max_amp"]:
                            missing_budget += c_budget["max_amp"] - c_budget["current_budget"]

                    if missing_budget > 0.9:
                        #missing one amp, we will have to "shave" the optional cluster and then the to be done cluster
                        for shaved_cluster in [budget_optional_cluster, budget_to_be_done_cluster]:
                            if shaved_cluster["sum_needed"] == 0:
                                continue

                            for c_budget in shaved_cluster["budgets"]:
                                if c_budget["current_budget"] == 0:
                                    continue

                                # by construction optional and to be done are set at their minimum
                                shaved_cluster["sum_needed"] -= c_budget["needed_amp"]
                                c_budget["needed_amp"] = 0

                                missing_budget -= c_budget["current_budget"]
                                budget_to_allocate += c_budget["current_budget"]
                                current_budget_spend -= c_budget["current_budget"]
                                c_budget["current_budget"] = 0

                                if missing_budget <= 0.0001:
                                    break

                            if missing_budget <= 0.0001:
                                break
                        if budget_to_allocate >= 0.9:
                            # ok we will need to redo the fast cluster
                            budget_as_fast_cluster["name"] = "as_fast_redone"
                            clusters[1] = budget_as_fast_cluster
                            _LOGGER.info(f"allocate_phase_amps_budget REDO for fast")

                if budget_to_allocate <= 0.0001:
                    break


        # now clean a bit the budgets to get to integer for amps

        # ok we do have now good allocation for all budgets
        allocated_final_budget = 0
        for c_budget in budgets:
            allocated_final_budget += c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])



        if allocated_final_budget > from_father_budget:
            _LOGGER.warning(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed last resort shaving {allocated_final_budget} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            new_current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)

            if new_current_budget_spend > from_father_budget:
                _LOGGER.error(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
                raise ValueError(f"allocate_phase_amps_budget for a group: {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
            else:
                allocated_final_budget = 0
                for c_budget in budgets:
                    allocated_final_budget += c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])


        self.device_phase_amps_budget = allocated_final_budget

        _LOGGER.info(f"allocate_phase_amps_budget for a group: {self.name} father budget {init_from_father_budget} => {from_father_budget} => {allocated_final_budget}")

        if self.father_device.device_is_3p and not self.device_is_3p:
            allocated_final_budget = allocated_final_budget / 3.0

        return allocated_final_budget

    def _shave_phase_amps_clusters(self, cluster_list_to_shave, current_budget_spend, from_father_budget):
        for shaved_cluster in cluster_list_to_shave:
            if shaved_cluster["sum_needed"] == 0:
                continue

            for c_budget in shaved_cluster["budgets"]:
                if c_budget["current_budget"] == 0:
                    continue

                # by construction all are set at their minimum
                shaved_cluster["sum_needed"] -= c_budget["needed_amp"]
                c_budget["needed_amp"] = 0
                current_budget_spend -= c_budget["current_budget"]
                c_budget["current_budget"] = 0

                if current_budget_spend <= from_father_budget:
                    break

            if current_budget_spend <= from_father_budget:
                break
        return current_budget_spend





