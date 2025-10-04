import copy
import logging

from datetime import datetime


from typing import Any

from ..const import CONF_DYN_GROUP_MAX_PHASE_AMPS, CONF_TYPE_NAME_QSDynamicGroup
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice
from ..home_model.home_utils import is_amps_zero, are_amps_equal, is_amps_greater, add_amps, diff_amps, min_amps, \
    max_amps

_LOGGER = logging.getLogger(__name__)

class QSDynamicGroup(HADeviceMixin, AbstractDevice):

    conf_type_name = CONF_TYPE_NAME_QSDynamicGroup

    # this class will allow to group Loads togther so the sum of their current can be limited below
    # a setting CONF_DYN_GROUP_MAX_CURRENT
    # It will also allow budgeting of Loads under this same constraint at planning/solving phase
    # but also at dynamic phase, for example allocating solar energy rightfully among connected cars
    # It is not a load but is a way to add a topology to the network of loads than are only leaves
    # of the network . Only Loads have constraints that need to be fullfilled.
    # The home is the root of this tree.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dyn_group_max_phase_current_conf = kwargs.pop(CONF_DYN_GROUP_MAX_PHASE_AMPS, 54)
        self._childrens : list[AbstractDevice] = []
        self.charger_group = None
        self._dyn_group_max_phase_current: list[float|int] | None = None
        self.available_amps_for_group: list[list[float|int]| None] | None = None

    def update_available_amps_for_group(self, idx:int, amps:list[float | int], add:bool):
        """Update the available amps for the group based on the device's configuration."""
        if self.available_amps_for_group is not None and idx < len(self.available_amps_for_group):
            if add:
                self.available_amps_for_group[idx] = add_amps(self.available_amps_for_group[idx], amps)
            else:
                self.available_amps_for_group[idx] = diff_amps(self.available_amps_for_group[idx], amps)

        if self.father_device is not None and self.father_device != self:
            return self.father_device.update_available_amps_for_group(idx, amps, add)

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

    def is_delta_current_acceptable(self, delta_amps: list[float|int], time:datetime, new_amps_consumption: list[float|int] | None = None) -> tuple[bool, list[float|int]]:

        if new_amps_consumption is not None and is_amps_greater(new_amps_consumption, self.dyn_group_max_phase_current):
            _LOGGER.info(
                f"is_delta_current_acceptable: group {self.name} not acceptable for new amps {new_amps_consumption} at start with dyn_max_phase_current {self.dyn_group_max_phase_current} at time {time}")

            return False, diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)

        phases_amps = self.get_device_amps_consumption(tolerance_seconds=None, time=time)

        new_amps = add_amps(delta_amps, phases_amps)

        if is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            _LOGGER.info(
                f"is_delta_current_acceptable: group {self.name} not acceptable for new amps {new_amps_consumption} at recompute with dyn_max_phase_current {self.dyn_group_max_phase_current} phases_amps {phases_amps} delta_amps {delta_amps} at time {time}")
            return False, diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)

        if self.father_device is None or self == self.home:
            return True, diff_amps(new_amps_consumption, self.dyn_group_max_phase_current)
        else:
            return self.father_device.is_delta_current_acceptable(delta_amps=delta_amps, time=time)

    def is_current_acceptable(self, new_amps: list[float|int], estimated_current_amps: list[float|int] | None, time:datetime) -> bool:
        res = self.is_current_acceptable_and_diff(new_amps, estimated_current_amps, time)[0]
        if res is False:
            _LOGGER.info(f"is_current_acceptable: group {self.name} not acceptable for new amps {new_amps} with estimated current amps {estimated_current_amps} at time {time}")
        return res

    def is_current_acceptable_and_diff(self, new_amps: list[float|int], estimated_current_amps: list[float|int] | None, time:datetime) -> tuple[bool, list[float|int]]:

        if is_amps_greater(new_amps, self.dyn_group_max_phase_current):
            _LOGGER.info(
                f"is_current_acceptable_and_diff: group {self.name} not acceptable for new amps {new_amps} at start with dyn_max_phase_current {self.dyn_group_max_phase_current} with estimated current amps {estimated_current_amps} at time {time}")
            return False, diff_amps(new_amps, self.dyn_group_max_phase_current)

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
            _LOGGER.info(
                f"is_current_acceptable_and_diff: group {self.name} not acceptable for new amps {new_amps} at recompute with dyn_max_phase_current {self.dyn_group_max_phase_current} with estimated current amps {estimated_current_amps} and computed_phases {current_phases} and delta_amps {delta_amps} at time {time}")
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


    # the idea is that if we have a 3p home we allocate as if phases were balanced for loads that are not 3p
    def prepare_slots_for_amps_budget(self, time: datetime, num_slots: int,
                                      from_father_budget: list[float | int] | None = None):

        init_from_father_budget = from_father_budget
        if from_father_budget is None:
            from_father_budget = self.dyn_group_max_phase_current
        else:
            from_father_budget = min_amps(from_father_budget, self.dyn_group_max_phase_current)

        if from_father_budget is None:
            from_father_budget = [1e8, 1e8, 1e8] # a lot of amps :)

        self.available_amps_for_group = [copy.copy(from_father_budget) for _ in range(num_slots)]

        _LOGGER.debug(
            f"prepare_slots_for_amps_budget for a group: {self.name} father budget {init_from_father_budget} => {from_father_budget}")

        for device in self._childrens:
            # we need to prepare the slots for the device
            device.prepare_slots_for_amps_budget(time, num_slots, from_father_budget)

    def  _shave_phase_amps_clusters(self, cluster_list_to_shave, current_budget_spend:list[float|int], from_father_budget:list[float|int]) -> list[float|int]:

        _LOGGER.info(f"_shave_phase_amps_clusters begin for a group: {self.name} current budget spend {current_budget_spend} from father budget {from_father_budget}")

        for can_clear_budget in [False, True]:
            for shaved_cluster in cluster_list_to_shave:
                if is_amps_zero(shaved_cluster["sum_needed"]):
                    continue

                while True:
                    one_modif = False
                    for c_budget in shaved_cluster["budgets"]:
                        if is_amps_zero(c_budget["current_budget"]):
                            continue

                        if can_clear_budget:
                            shaved_cluster["sum_needed"] = diff_amps(shaved_cluster["sum_needed"], c_budget["needed_amp"])
                            c_budget["needed_amp"] = [0,0,0]
                            current_budget_spend = diff_amps(current_budget_spend, c_budget["current_budget"])
                            c_budget["current_budget"] = [0,0,0]
                        else:

                            if is_amps_greater(c_budget["current_budget"], c_budget["min_amp"]):
                                init_budget = copy.copy(c_budget["current_budget"])

                                current_budget_spend = diff_amps(current_budget_spend, c_budget["current_budget"])
                                device = c_budget["device"]
                                c_budget["current_budget"] = device.update_amps_with_delta(
                                    from_amps=c_budget["current_budget"], delta=-1, is_3p=device.physical_3p)
                                c_budget["current_budget"] = max_amps(c_budget["current_budget"], c_budget["min_amp"])
                                current_budget_spend = add_amps(current_budget_spend, c_budget["current_budget"])

                                if not are_amps_equal(init_budget, c_budget["current_budget"]):
                                    one_modif = True

                        if not is_amps_greater(current_budget_spend, from_father_budget):
                            break

                    # out of the while loop if shaved enough or no more modification possible
                    if not is_amps_greater(current_budget_spend, from_father_budget) or one_modif is False:
                        break

                if not is_amps_greater(current_budget_spend, from_father_budget):
                    break

            if not is_amps_greater(current_budget_spend, from_father_budget):
                break

        _LOGGER.info(f"_shave_phase_amps_clusters results for a group: {self.name} current budget spend {current_budget_spend} from father budget {from_father_budget}")


        return current_budget_spend





