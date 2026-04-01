# Bug Fix: Car does not stop charging at target SOC -- idle command not physically enforced on OCPP charger

Status: implemented
issue: 99
branch: "QS_99"

## Story

As a Quiet Solar user with an OCPP-connected EV charger,
I want the system to physically stop charging when my car reaches the target SOC,
so that the car does not charge past 95% all the way to 100%, defeating the constraint system.

## Bug Description

When a car (Renault Zoe) reaches its target SOC (95%), the system correctly detects the constraint is met and sends an internal `idle` command, but the OCPP wallbox continues to charge at the previously set amperage (28A). The car charged from 95% all the way to 100%.

**Timeline (2026-04-01):**
1. **09:53:34** -- Zoe SOC reaches 95%: `is_car_charged True`, constraint marked as met
2. **09:53:34** -- `Constraint for wallbox 3 portail update callback asked for stop`
3. **09:53:34** -- COMPLETED notification sent, DO SOLVE triggered
4. **09:53:34** -- `launch_command: idle for wallbox 3 portail`
5. **09:53:34** -- `probe_and_enforce_stopped_charge_command_state: not a running command idle`
6. **09:53:34** -- **`Command already set idle for wallbox 3 portail`** <-- BUG: skips execute_command
7. **09:54+ onwards** -- OCPP MeterValues show charger STILL drawing ~20A per phase at 28A offered

No OCPP `SetChargingProfile` or `RemoteStopTransaction` is ever sent after the idle decision.

## Root Cause Analysis

**File:** `ha_model/charger.py`

The bug is a state-comparison error in the probe path that causes `launch_command` to incorrectly believe the idle command is already active.

### Code flow

1. `launch_command` (load.py:581) calls `probe_if_command_set(time, idle_command)` before `execute_command`
2. `probe_if_command_set` (charger.py:4893) calls:
   - `_probe_and_enforce_stopped_charge_command_state(time, command=idle, probe_only=True)` (line 4900)
   - `_ensure_correct_state(time, probe_only=True)` (line 4901)
3. In `_probe_and_enforce_stopped_charge_command_state` (line 4843):
   ```python
   if probe_only is False and handled is True:  # <-- BUG: skipped when probe_only=True
       self._expected_charge_state.set(False, time)
       self._expected_amperage.set(self.charger_default_idle_charge, time)
       self._expected_num_active_phases.set(self.current_num_phases, time)
   ```
   With `probe_only=True`, expected state is NOT updated:
   - `_expected_charge_state.value` remains `True` (from active charging)
   - `_expected_amperage.value` remains `28` (from active charging)
4. In `_ensure_correct_state(probe_only=True)`:
   - `want_charge = True` (stale), `currently_charging = True` (actual) -> STEADY STATE branch
   - `charging_current_amp (28A) == _expected_amperage.value (28A)` -> match
   - Returns `True` ("state is correct")
5. Back in `launch_command`: `is_command_set = True` -> logs "Command already set" -> skips `execute_command`
6. **Result**: no OCPP command is ever sent, charger keeps charging at 28A

### Why it only affects idle/off transitions

When `execute_command` runs (the path that works), it calls `_reset_state_machine()` first (line 4874), which clears all expected state, then calls `_probe_and_enforce_stopped_charge_command_state(probe_only=False)` which correctly sets expected state to idle values. The subsequent `_ensure_correct_state(probe_only=False)` sees the mismatch and sends the stop command.

The probe path skips all of this because expected state is never updated, so the stale "charging at 28A" expected state matches the actual "charging at 28A" state.

### Secondary effect (filler constraint)

After the bad ack, `running_command` is cleared and `current_command` is idle. A filler constraint at 95/95 is immediately met, so the SOC callback (and `dyn_handle`) may never run again for that load -- no second chance to call `execute_command`.

## Fix Plan (from Cursor plan -- primary authority)

Two changes in `ha_model/charger.py`:

### Change 1: `probe_if_command_set` -- declare desired state before probing (line 4900)

The fix is in the **caller** (`probe_if_command_set`), not in `_probe_and_enforce_stopped_charge_command_state` itself. The `probe_only` contract on `_probe_and_enforce_stopped_charge_command_state` is correct -- it controls whether expected state is updated. The bug is that `probe_if_command_set` asks to NOT update state (`probe_only=True`) and then expects `_ensure_correct_state` to give a meaningful answer based on that stale state.

The semantic split should be:
- `_probe_and_enforce_stopped_charge_command_state(probe_only=False)` -- declare desired state for idle/off commands
- `_probe_and_enforce_stopped_charge_command_state(probe_only=True)` -- keep for ON/AUTO commands (no state mutation needed)
- `_ensure_correct_state(probe_only=True)` -- compare desired vs. reality, no hardware commands

**Important**: the `probe_only` flag must be command-gated, not unconditionally False. `is_in_state_reset()` (line 4837) forces `handled=True` for ALL commands including ON/AUTO. Without the guard, ON/AUTO would falsely declare idle target state and be acked without executing.

```python
# Before (line 4900):
self._probe_and_enforce_stopped_charge_command_state(time, command=command, probe_only=True)

# After:
declare_idle_target = command is None or command.is_off_or_idle()
self._probe_and_enforce_stopped_charge_command_state(
    time,
    command=command,
    probe_only=not declare_idle_target,
)
```

### Change 2: Gate amps/phases update on actual charge-state transition (line 4843-4846)

`QSStateCmd.set()` returns `True` only when the value **changes**; if `_expected_charge_state` is already `False`, calling `set(False, ...)` is a no-op and returns `False`. Use that return value to avoid resetting `_expected_amperage` and `_expected_num_active_phases` when we are not actually entering a new "want stop" expectation (repeated probes, double calls, etc.). This prevents `set()` from calling `reset()` on the amperage/phases `QSStateCmd` objects, which would wipe their retry counters (`_num_launched`) and success tracking.

```python
# Before (lines 4843-4846):
if probe_only is False and handled is True:
    self._expected_charge_state.set(False, time)
    self._expected_amperage.set(self.charger_default_idle_charge, time)
    self._expected_num_active_phases.set(self.current_num_phases, time)

# After:
if probe_only is False and handled is True:
    if self._expected_charge_state.set(False, time):
        self._expected_amperage.set(self.charger_default_idle_charge, time)
        self._expected_num_active_phases.set(self.current_num_phases, time)
```

**Why this is safe for `execute_command` after `_reset_state_machine()`:** reset clears inner `QSStateCmd` objects; the next access recreates them with `value = None`. Then `set(False, time)` is a real change (`None` != `False`), returns `True`, so amps and phases are still set on the first stop after reset -- same as today.

**Trade-off:** if `_expected_charge_state` were already `False` but `_expected_amperage` were inconsistent (bug or partial state), this would not "repair" amps/phases on a later idle probe. That situation should be rare; the preference is to avoid unnecessary churn on the other two registers.

### Safety analysis (all 4 call sites of `_probe_and_enforce_stopped_charge_command_state`)

1. **`execute_command`** (line 4878): already `probe_only=False`. After `_reset_state_machine()`, `_expected_charge_state.value` is `None`, so `set(False)` returns `True` -- amps/phases still set. No behavior change.
2. **`ensure_correct_state`** (line 4335): always called with `probe_only=False` from `dyn_handle` -- no change
3. **`probe_if_command_set`** (line 4900): **change 1** -- now command-gated `probe_only`
   - For `idle`/`off` commands: `probe_only=False`, `handled=True`, charge state declared. Amps/phases only set on actual transition (**change 2**).
   - For `auto`/`on` commands: `probe_only=True` (preserved), guarded block never executes -- no impact
   - **Edge case**: `is_in_state_reset()` forces `handled=True` for ON/AUTO too, but the command-based guard keeps `probe_only=True` for those, preventing false idle-state declaration
   - `QSStateCmd.set()` is a no-op when value unchanged (no side effects on retry counters or `is_ok_to_set` rate limiting)
4. **`get_stable_dynamic_charge_status`** (line 2310): keeps `probe_only=True` -- always preceded by `ensure_correct_state(probe_only=False)` in the `dyn_handle` loop, so expected state is already set; the `.set()` call would be a no-op anyway

### Verification of fix with bug scenario

After fix, when idle command arrives while charging at 28A:
1. `_probe_and_enforce_stopped_charge_command_state(probe_only=False)`:
   - `handled = True` (idle command)
   - `_expected_charge_state.set(False, time)` returns `True` (was `True` -> `False`)
   - Sets `_expected_amperage = idle_charge` (e.g., 6A), `_expected_num_active_phases = current`
2. `_ensure_correct_state(probe_only=True)`:
   - `want_charge = False`, `currently_charging = True` -> TRANSITION branch
   - `amps_confirmed = (28A == 6A)` -> False
   - `one_bad = True`
   - Returns `False` (no hardware commands sent because `probe_only=True`)
3. `launch_command`: `is_command_set = False` -> calls `execute_command`
4. `execute_command`: `_reset_state_machine()` -> `_probe_and_enforce_stopped_charge_command_state(probe_only=False)` -> `_ensure_correct_state(probe_only=False)` -> sends OCPP stop/reduce command

## Acceptance Criteria

1. **AC1**: When car reaches target SOC and solver sends idle command, the charger physically stops charging (OCPP SetChargingProfile or RemoteStopTransaction is sent)
2. **AC2**: `probe_if_command_set` returns False when transitioning from active charging to idle (so `execute_command` runs)
3. **AC3**: `probe_if_command_set` returns True when charger is already idle and idle command is re-sent (optimization preserved)
4. **AC4**: ON/AUTO command probe behavior is unchanged (no regression)
5. **AC5**: The `_expected_amperage`, `_expected_charge_state`, and `_expected_num_active_phases` are correctly set to idle values when probing an idle command on an active charger
6. **AC6**: Repeated idle probes when already idle do NOT reset `_expected_amperage` / `_expected_num_active_phases` retry counters

## Tasks / Subtasks

- [x] Task 1: Fix `probe_if_command_set` -- declare desired state before probing (AC: 1, 2, 5)
  - [x] Change line 4900: remove `probe_only=True` (defaults to False)
  - [x] Keep `_ensure_correct_state(probe_only=True)` on line 4901 unchanged (compare only, no hardware commands)
- [x] Task 2: Gate amps/phases update on actual charge-state transition (AC: 6)
  - [x] Change lines 4843-4846: use `_expected_charge_state.set()` return value to gate `_expected_amperage` and `_expected_num_active_phases` updates
- [x] Task 3: Add tests for idle command probe path (AC: 2, 3, 4, 5, 6)
  - [x] Test: probe idle command while charger is actively charging at 28A -> returns False (command NOT already set)
  - [x] Test: probe idle command while charger is already idle -> returns True (command already set, optimization works)
  - [x] Test: probe ON/AUTO command -> behavior unchanged (handled=False, no state update)
  - [x] Test: repeated idle probe does not reset amperage/phases retry counters (AC6)
  - [x] Test: after probe returns False, execute_command correctly sends stop command
- [x] Task 4: Integration test for full SOC-reached flow (AC: 1)
  - [x] Test: car reaches target SOC -> constraint met -> idle command -> charger stops (end-to-end)

## Dev Notes

### Key file locations

- **Bug location 1**: `ha_model/charger.py:4900` (`probe_if_command_set`, the `probe_only=True` call)
- **Bug location 2**: `ha_model/charger.py:4843-4846` (`_probe_and_enforce_stopped_charge_command_state`, unconditional amps/phases overwrite)
- **Probe method**: `ha_model/charger.py:4893-4911` (`probe_if_command_set`)
- **QSStateCmd.set()**: `ha_model/charger.py:246-259` (returns True only on value change; calls reset() which wipes retry counters)
- **Execute path (working)**: `ha_model/charger.py:4850-4891` (`execute_command`)
- **State machine reset**: `ha_model/charger.py:2631-2636` (`_reset_state_machine`)
- **Launch command (trigger)**: `home_model/load.py:548-604` (`launch_command`)
- **Ensure correct state**: `ha_model/charger.py:4340-4505` (`_ensure_correct_state`)

### Architecture constraints

- `launch_command` is in `home_model/load.py` (domain layer) -- it calls `probe_if_command_set` and `execute_command` which are overridden in `ha_model/charger.py` (HA layer). The fix is entirely in the HA layer.
- The fix does not change the domain/HA boundary.

### Testing patterns

- Use `create_test_charger_double()` from `tests/factories.py` for charger test doubles
- Use `create_load_command()` for creating idle/on commands
- Use `create_state_cmd()` for setting up `_expected_amperage` etc.
- Mock `_do_update_charger_state`, `is_optimistic_plugged`, `is_charge_enabled`, `get_charging_current` for state setup
- See existing charger tests for patterns

### Related work

- This is a correctness bug in the charger state machine, not a regression from recent changes
- The `probe_only` parameter contract is correct -- it controls state updates in `_probe_and_enforce_stopped_charge_command_state`. The bug was the caller (`probe_if_command_set`) passing `probe_only=True` when it should declare desired state before probing

### References

- [Source: ha_model/charger.py#_probe_and_enforce_stopped_charge_command_state] -- lines 4814-4848
- [Source: ha_model/charger.py#probe_if_command_set] -- lines 4893-4911
- [Source: ha_model/charger.py#execute_command] -- lines 4850-4891
- [Source: ha_model/charger.py#_reset_state_machine] -- lines 2631-2636
- [Source: home_model/load.py#launch_command] -- lines 548-604
- [Source: ha_model/charger.py#_ensure_correct_state] -- lines 4340-4505
- [Source: ha_model/charger.py#get_stable_dynamic_charge_status] -- line 2310 (4th call site, keeps probe_only=True)
- [External plan: fix_charger_not_stopping_350bd288.plan.md] -- Cursor plan used as primary authority

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Root cause confirmed by tracing the exact code path: `probe_only=True` in `probe_if_command_set` prevents expected-state update, causing stale state comparison to return True
- Two-part fix: (1) remove `probe_only=True` from line 4900, (2) gate amps/phases on `set()` return value at line 4843
- Preserves the contract of `_probe_and_enforce_stopped_charge_command_state` -- `probe_only` still controls state updates for other callers (e.g., `get_stable_dynamic_charge_status`)
- All 4 call sites analyzed for safety (see fix plan); `QSStateCmd.set()` is a no-op when value unchanged but calls `reset()` (wiping retry counters) when value changes -- hence the gate
- ON/AUTO commands unaffected because `handled=False` for those, so the guarded block never executes regardless
- Cursor plan used as primary authority for fix approach

### File List

- `custom_components/quiet_solar/ha_model/charger.py` (modify -- lines 4843-4846 and 4900)
- `tests/` (add tests for probe path with idle command)
