# Charger control logic fixes for out-of-charge current change

## Problem statement

Some chargers (notably OCPP-based) exhibit this behavior: when they receive an amps/current command while NOT actively charging, they either ignore it or reset their internal value to max (32A). This causes a current spike on the next charge start -- whether that start is triggered by quiet-solar or by a fresh car plug-in.

The issue manifested in two ways:

1. **Stopping charge without pre-setting amps to min**: The charger "remembers" whatever amps it had at the time of the stop, and starts at that level next time.
2. **Sending amps commands to an idle charger**: Multiple code paths could send amps to a charger that wasn't actively charging, triggering the 32A reset on misbehaving hardware.

## Changes overview

All changes are in `custom_components/quiet_solar/ha_model/charger.py` and `tests/test_charger_coverage_deep.py`.

---

## 1. Restructured `_ensure_correct_state` state machine

The charge state + amps section of `_ensure_correct_state` was restructured around four clear cases based on the charge state transition. Previously, the flow was "check amps, then check charge state" regardless of direction. Now the transition type drives the logic:

### Case 1: Charging and want to stop (`currently_charging and not want_charge`)

Amps are reduced to `charger_default_idle_charge` (min) FIRST. The stop command is only sent once amps are confirmed by the sensor. This is a multi-cycle state machine:

- **Cycle N**: Amps section sends min_charge command (charger is still enabled, guard passes). Stop is NOT sent because amps not yet confirmed by sensor.
- **Cycle N+k**: Sensor reads amps = min_charge. Stop is now allowed and sent.
- **Fallback**: If `can_launch()` returns False (retries exhausted), stop is sent anyway with a warning log.

```python
if currently_charging and not want_charge:
    amps_confirmed = (charging_current_amp == self._expected_amperage.value)
    amps_retries_exhausted = not self._expected_amperage.can_launch()

    if amps_confirmed or amps_retries_exhausted:
        # Amps are at target (or retries exhausted) -> proceed with stop
        if self._expected_charge_state.is_ok_to_launch(value=False, time=time):
            await self.stop_charge(time=time)
    else:
        # Amps not yet at target -> send amps command, delay stop
        if self._expected_amperage.is_ok_to_launch(...):
            await self.set_charging_current(current=self._expected_amperage.value, ...)
```

### Case 2: Not charging and want to start (`not currently_charging and want_charge`)

Just send `start_charge`. Amps are NOT sent here -- they will be handled in Case 3 once the charger confirms it is enabled. This prevents sending amps to a stopped charger.

```python
elif not currently_charging and want_charge:
    if self._expected_charge_state.is_ok_to_launch(value=True, time=time):
        await self.start_charge(time=time)
```

### Case 3: Charging and want to charge -- steady state (`currently_charging and want_charge`)

Adapt amps if needed. This is the only place where `set_charging_current` is called during normal charging operation. Includes periodic refresh to re-send amps to the charger for robustness.

```python
elif currently_charging and want_charge:
    await self._expected_charge_state.success(time=time)
    if charging_current_amp == self._expected_amperage.value:
        # Amps match: periodic refresh
    else:
        # Amps mismatch: send correction
        await self.set_charging_current(current=self._expected_amperage.value, ...)
```

### Case 4: Not charging and don't want to charge -- steady state (`else`)

Do nothing. No amps commands, no start/stop. Mark charge state as success.

```python
else:
    await self._expected_charge_state.success(time=time)
```

### State machine flow

**Stop transition (multi-cycle):**

```
Charging at XA
  -> Budget says stop, _expected_amperage = min
  -> Amps command sent (min), stop delayed
  -> [sensor confirms min] or [retries exhausted]
  -> stop_charge sent
  -> Stopped (safe, charger remembers min amps)
```

**Start transition (multi-cycle):**

```
Stopped (amps = min from previous stop)
  -> Budget says start
  -> start_charge sent (no amps sent)
  -> [sensor confirms charge_enabled]
  -> Case 3: amps command sent (desired value)
  -> Charging at desired amps
```

**Fresh plug-in (hardware-initiated):**

```
Idle (amps = min, protected by low-level guard)
  -> User plugs in car
  -> Charger auto-starts at remembered amps (min) -- safe
  -> _ensure_correct_state detects enabled, sends desired amps (Case 3)
  -> Charging at desired amps
```

---

## 2. `can_set_amps_when_not_charging()` guard

### New overridable method on `QSChargerGeneric`

```python
def can_set_amps_when_not_charging(self) -> bool:
    """Whether this charger accepts amp commands while not actively charging."""
    return True
```

Overridden in `QSChargerOCPP` to return `False`. Wallbox chargers support setting amps while idle, so they inherit the default `True`.

### Guard in `set_max_charging_current` and `set_charging_current`

Both high-level methods check `can_set_amps_when_not_charging()` + `is_charge_enabled()` at the top. If the charger doesn't support amps while idle and is not actively charging, the command is blocked and returns `False`.

```python
async def set_max_charging_current(self, current, time, ...):
    if not self.can_set_amps_when_not_charging():
        is_charging = self.is_charge_enabled(time)
        if not is_charging:
            return False
    # ... proceed with low-level call
```

This catches all code paths that could send amps to an idle charger:
- The `for_default_when_unplugged` path (when car is unplugged)
- Any future code calling these methods
- The `_ensure_correct_state` restructure already prevents it at the state machine level, so this is a safety net

The low-level methods (`low_level_set_max_charging_current`, `low_level_set_charging_current`) remain pure "send command" ports with no business logic.

---

## 3. `can_launch()` method on `QSStateCmd`

Added `can_launch()` to `QSStateCmd` as a public method to check whether retries are exhausted, avoiding direct access to the private `_num_launched` field:

```python
def can_launch(self) -> bool:
    """Whether retries are not yet exhausted."""
    return self._num_launched <= STATE_CMD_RETRY_NUMBER
```

Used inside `is_ok_to_launch()` (replacing the inline check) and in `_ensure_correct_state` for the stop-gate fallback logic.

---

## Files changed

| File | Change |
|------|--------|
| `charger.py` `QSStateCmd` | Added `can_launch()` method, used in `is_ok_to_launch()` |
| `charger.py` `QSChargerGeneric._ensure_correct_state` | Restructured into 4 cases: stop transition, start transition, charging steady state, idle steady state |
| `charger.py` `QSChargerGeneric.set_max_charging_current` | Added `can_set_amps_when_not_charging` guard at top |
| `charger.py` `QSChargerGeneric.set_charging_current` | Added `can_set_amps_when_not_charging` guard at top |
| `charger.py` `QSChargerGeneric` | Added `can_set_amps_when_not_charging()` method (default `True`) |
| `charger.py` `QSChargerOCPP` | Override `can_set_amps_when_not_charging()` -> `False` |
| `test_charger_coverage_deep.py` | Added `TestCanSetAmpsWhenNotCharging` (8 tests) and `TestStopChargeGatedBehindAmps` (5 tests) |

---

## Test coverage

### `TestCanSetAmpsWhenNotCharging` (8 tests)

- Generic charger returns `True` for `can_set_amps_when_not_charging`
- OCPP charger returns `False`
- Wallbox charger returns `True` (inherits default)
- `set_max_charging_current` blocked for OCPP when not charging, allowed when charging
- `set_charging_current` allowed for Wallbox even when not charging, allowed when charging
- Generic charger allows amps when not charging

### `TestStopChargeGatedBehindAmps` (5 tests)

- Stop is delayed when amps not yet confirmed at min
- Stop proceeds once amps are confirmed by sensor
- Stop proceeds as fallback when retries are exhausted
- Generic chargers also gate stop behind amps confirmation
- Full stop-start cycle: low_level amps never called while idle for OCPP
