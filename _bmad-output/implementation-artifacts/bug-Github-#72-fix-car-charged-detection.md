# Bug Fix: Car charged detection at 100% target

issue: 72
branch: "QS_72"
Status: in-progress

## Story

As a Quiet Solar user with an EV charger that lacks an accurate power sensor,
I want the system to detect when my car reaches 100% SOC using group sensor fallback and energy integration,
so that charging stops promptly instead of waiting 20+ minutes for an external API update.

## Bug Description

When a car has a 100% SOC target (the default filler constraint), `is_car_charged` caps the result at `min(current_charge, 99)`, making it impossible for the constraint to be met through SOC alone. The only escape is `is_car_stopped_asking_current` returning True, which requires 20 minutes of "SuspendedEV" OCPP status.

In the Twingo case, the car charged from 79% to 99% (sensor) / 102% (computed) over ~2 hours of solar charging, yet the constraint was never met until the Renault API updated to 100% at 14:26 -- about 22 minutes of unnecessary charging past the real 100% point.

Critically, `result_calculus` (power integration) could have detected completion earlier, but it relies on `get_device_real_energy` which returns None when the charger has no accurate power sensor. The same blind spot affects `is_charging_power_zero`. Both need group sensor fallback before Fix C can work.

## Root Cause

Three compounding issues:

1. **No group sensor fallback for energy computation** -- `_compute_added_charge_update` returns None for `result_calculus` when the charger has no power sensor, even though the parent group sensor could provide the data.
2. **No group sensor fallback for power-zero detection** -- `is_charging_power_zero` only checks the charger's own sensor, returns None when unavailable. The group fallback logic at lines 4598-4625 in `update_value_callback` is duplicated inline and missing safety guards.
3. **Unconditional cap at 99** -- `is_car_charged` always returns `min(current_charge, 99)` for 100% targets, with no escape path using computed energy or power-zero signals.

## Fix Plan

### Task 0: Add `_can_use_group_power_sensor()` helper

**File:** `custom_components/quiet_solar/ha_model/charger.py`

Add a helper method on `QSChargerGeneric` to centralize the group sensor eligibility guard:

```python
def _can_use_group_power_sensor(self) -> bool:
    return (
        self.accurate_power_sensor is None
        and self.father_device is not None
        and self.father_device is not self.home
        and self.father_device.accurate_power_sensor is not None
        and self.charger_group.dync_group_chargers_only
    )
```

Guards: `father_device` is not `home` (otherwise measures entire house), and `dync_group_chargers_only` is True (otherwise non-charger children pollute the reading).

### Task 1: Hardening A -- Group sensor fallback for `_compute_added_charge_update`

**File:** `custom_components/quiet_solar/ha_model/charger.py`

In `_compute_added_charge_update`, decide upfront which device to query for energy:

- If `_can_use_group_power_sensor()` AND no other charger in the group is charge-enabled during the period: get energy from the group instead of the charger.
- Otherwise: use the charger itself (existing behavior).

```python
energy_source = self
if self._can_use_group_power_sensor():
    other_charger_charging = any(
        c.is_charge_enabled(time=end_time, for_duration=(end_time - start_time).total_seconds()) is True
        for c in self.charger_group._chargers
        if c is not self
    )
    if not other_charger_charging:
        energy_source = self.father_device

added_nrj = energy_source.get_device_real_energy(
    start_time=start_time, end_time=end_time, clip_to_zero_under_power=self.charger_consumption_W
)
```

### Task 2: Hardening B -- `is_charging_power_zero` absorbs group fallback

**File:** `custom_components/quiet_solar/ha_model/charger.py`

Harden `is_charging_power_zero` to use group sensor when charger has no own power data:

1. If charger has its own power data: use it directly (existing logic).
2. If `_can_use_group_power_sensor()`:
   - Group zero -> True (no power to any charger in group)
   - Group non-zero AND no other charger charging -> False (all group power is this charger)
   - Group non-zero AND other chargers charging -> None (can't tell)
3. Otherwise -> None.

```python
def is_charging_power_zero(self, time: datetime, for_duration: float) -> bool | None:
    val = self.get_average_power(for_duration, time, use_fallback_command=False)
    if val is not None:
        return self.dampening_power_value_for_car_consumption(val) == 0.0

    if self._can_use_group_power_sensor():
        father_is_zero = self.is_charger_group_power_zero(time=time, for_duration=for_duration)
        if father_is_zero is True:
            return True
        if father_is_zero is False:
            other_charger_charging = any(
                c.is_charge_enabled(time=time, for_duration=for_duration) is True
                for c in self.charger_group._chargers
                if c is not self
            )
            if not other_charger_charging:
                return False

    return None
```

Then simplify `update_value_callback` lines 4598-4625 to just call `self.is_charging_power_zero(...)`, removing the duplicated inline logic and fixing the latent missing guards.

### Task 3: Fix C -- `is_car_charged` combined escape hatch

**File:** `custom_components/quiet_solar/ha_model/charger.py`

1. Pass `result_calculus` as optional parameter to `is_car_charged` from `constraint_update_value_callback_soc` (the other two call sites keep default None).

2. In the 100% target branch, add a combined escape where all three signals must agree:

```python
else:
    if is_target_percent and target_charge >= 100:
        if (
            result_calculus is not None
            and result_calculus >= target_charge
            and current_charge >= 99
            and self.is_charging_power_zero(
                time=time,
                for_duration=CHARGER_STOP_CAR_ASKING_FOR_CURRENT_TO_STOP_S,
            ) is True
        ):
            result = target_charge
        else:
            result = min(current_charge, 99)
```

Conservative: `result_calculus >= 100` (enough energy delivered), `current_charge >= 99` (sensor near target), AND `is_charging_power_zero` for full 20-minute window (car stopped accepting power).

## Dependency Order

Tasks must be implemented in order:

1. **Task 0** (guard helper) -- prerequisite for everything
2. **Task 1** (Hardening A) -- makes `result_calculus` available
3. **Task 2** (Hardening B) -- makes `is_charging_power_zero` reliable
4. **Task 3** (Fix C) -- uses both hardened signals

## Acceptance Criteria

- [ ] `_can_use_group_power_sensor()` returns True only when charger has no own sensor, father is not home, father has a sensor, and group is chargers-only
- [ ] `_compute_added_charge_update` falls back to group sensor energy when eligible and no other charger is charging
- [ ] `is_charging_power_zero` returns conclusive True/False via group sensor when eligible
- [ ] `update_value_callback` inline group logic replaced by single `is_charging_power_zero` call
- [ ] `is_car_charged` detects 100% completion when all three signals (calculus, sensor, power-zero) agree
- [ ] Other `is_car_charged` call sites unaffected (no `result_calculus` available, default None)
- [ ] All existing tests pass
- [ ] New tests cover group sensor fallback paths and combined escape hatch

## References

- External plan: `.cursor/plans/fix_car_charged_detection_82373539.plan.md`
- Main file: `custom_components/quiet_solar/ha_model/charger.py`
