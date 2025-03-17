import logging

from datetime import datetime


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
        for device in self._childrens:
            if device.device_is_3p:
                return True
        return False

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

    def get_min_max_phase_amps(self) -> (float, float):
        if len(self._childrens) == 0:
            return super().get_min_max_phase_amps()

        min_p = 1e12
        max_p = 0
        for device in self._childrens:
            min_p_d, max_p_d = device.get_min_max_phase_amps()
            min_p = min(min_p, min_p_d)
            max_p = max(max_p, max_p_d)

        return min_p, max_p

    def get_evaluated_needed_phase_amps(self, time:datetime) -> float:

        device_needed_amp = 0.0
        for device in self._childrens:
            device_needed_amp += device.get_evaluated_needed_phase_amps(time)

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


    def allocate_phase_amps_budget(self, time:datetime, from_father_budget:float|None = None) -> float:

        if from_father_budget is None:
            from_father_budget = self.dyn_group_max_phase_current
        else:
            from_father_budget = min(from_father_budget, self.dyn_group_max_phase_current)

        if from_father_budget is None:
            from_father_budget = 1e9 # a lot of amps :)

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
            device_needed_amp = device.get_evaluated_needed_phase_amps(time)
            min_a, max_a = device.get_min_max_phase_amps()

            if device_needed_amp == 0:
                cur_amps = 0.0
            else:
                cur_amps = min_a

            # allocate first at minumum
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
            _LOGGER.info(f"QSDynamicGroup {self.name}: initial over amp budget! {current_budget_spend} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)
            if current_budget_spend > from_father_budget:
                _LOGGER.info(
                    f"QSDynamicGroup {self.name} : initial over amp budget even after shaving {current_budget_spend} > {from_father_budget}")

        # everyone has its minimum already allocated
        if current_budget_spend < from_father_budget:
            # ok let's put the rest of the budget on the loads if they can get it
            budget_to_allocate = from_father_budget - current_budget_spend

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
                            delta_budget = min(min(c_budget["max_amp"] - c_budget["current_budget"], (1.0*c_budget["needed_amp"])/(cluster_budget["sum_needed"])), budget_to_allocate)
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

                if budget_to_allocate <= 0.0001:
                    break


        # now clean a bit the budgets to get to integer for amps

        # ok we do have now good allocation for all budget
        allocated_final_budget = 0
        for c_budget in budgets:
            allocated_final_budget += c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])



        if allocated_final_budget > from_father_budget:
            _LOGGER.warning(f"QSDynamicGroup {self.name} allocated more than allowed last resort shaving {allocated_final_budget} > {from_father_budget}")
            cluster_list_to_shave = [budget_optional_cluster, budget_to_be_done_cluster, budget_as_fast_cluster]
            new_current_budget_spend = self._shave_phase_amps_clusters(cluster_list_to_shave, current_budget_spend, from_father_budget)

            if new_current_budget_spend > from_father_budget:
                _LOGGER.error(f"QSDynamicGroup {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
                raise ValueError(f"QSDynamicGroup {self.name} allocated more than allowed!! {new_current_budget_spend} > {from_father_budget}")
            else:
                allocated_final_budget = 0
                for c_budget in budgets:
                    allocated_final_budget += c_budget["device"].allocate_phase_amps_budget(time, c_budget["current_budget"])


        self.device_phase_amps_budget = allocated_final_budget
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





