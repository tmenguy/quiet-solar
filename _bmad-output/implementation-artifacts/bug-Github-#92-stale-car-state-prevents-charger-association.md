# Bug #92: Stale car state prevents automatic charger association after API reconnect

Status: ready-for-dev
issue: 92
branch: "QS_92"

## Story

As a Quiet Solar user,
I want the system to automatically associate my known car with its charger once the car API confirms the car is plugged in,
so that I don't need to manually reset the car card every time the API reconnects after a delay.

## Bug Summary

When a known car arrives home and is plugged into a charger, a guest car is initially shown as connected (expected). However, once the car API correctly reports the car as plugged in, the system does **not** automatically replace the guest car with the real car. Manually resetting the car card (red button) fixes the association immediately.

**Reported scenario (Twingo + charger 3):**
1. Twingo arrives home, Magali plugs it into charger 3
2. Guest car appears as connected (expected, API not yet updated)
3. Renault API eventually reports Twingo as plugged in
4. System does NOT automatically associate Twingo with charger 3
5. Manual reset via car card red button fixes it immediately

## Root Cause Analysis

Three issues combine to prevent the guest-to-known-car transition.

### Root Cause 1: `can_exit_stale_percent_mode` deadlock for plugged-but-unattached cars

**Location:** `ha_model/car.py`, `can_exit_stale_percent_mode()` (lines 825-838)

```python
if self.charger is not None:
    # Connected: plug=plugged AND home=home
    ...
else:
    # Not connected: plug=unplugged
    raw_plugged = self._get_raw_is_car_plugged(time)
    if self.car_plugged is None:
        return True
    return raw_plugged is False  # <-- BUG: car IS plugged!
```

When a car enters stale-percent mode (SOC sensor stale > 1h) and then the API recovers reporting `plugged=True`, the car **cannot exit stale mode** because:
- `self.charger is None` (the guest car is on the charger, not this car)
- The "not connected" exit path requires `raw_plugged is False`
- But the car IS plugged, so `raw_plugged is True` → returns `False` → stays stale

This keeps `car_api_stale_percent_mode = True` indefinitely. While it doesn't directly block scoring, it creates downstream charge-planning issues (poisoned SOC) and a permanently stale UI state. The existing test `test_exit_not_connected_plug_true_blocks` codifies this as intentional, but the logic is flawed for this scenario.

### Root Cause 2: `for_duration` scoring gap — `False` vs `None` fallback

**Location:** `ha_model/charger.py`, `get_car_score()` (lines 2722-2727)

```python
car_plug_res = car.is_car_plugged(time=time, for_duration=CHARGER_CHECK_STATE_WINDOW_S)
if car_plug_res is None:       # <-- Only falls back on None, NOT on False
    car_plug_res = car.is_car_plugged(time=time)
```

The fallback to an instant check only triggers on `None` (no sensor data), NOT on `False` (sensor exists but duration < 15s). When the car API first reports `plugged=True`, the contiguous "on" duration starts near 0 — so for the first ~15 seconds, `is_car_plugged(for_duration=15)` returns `False`, the fallback doesn't trigger, and `score_plug_bump = 0`.

The same pattern exists for `car_home_res` at lines 2783-2788.

Combined with the **charger plug duration mismatch** (lines 2748-2757): the charger has been plugged for hours (with the guest car) while the car API just started reporting plugged. The time difference far exceeds `CHARGER_LONG_CONNECTION_S` (20 min), so `score_plug_time_bump = 0`.

Result: the known car scores 0 on every cycle until both plug and home durations exceed 15 seconds. If both stay at 0, the gate at line 2794 (`score_plug_bump > 0 AND (score_dist_bump > 0 OR score_plug_time_bump > 0)`) blocks the car permanently. The scoring entirely depends on `score_dist_bump` (GPS distance), which requires the `for_duration` home check to pass — same issue.

### Root Cause 3: Stale user-originated state persisted across reboots — no lifecycle reset

**Location:** `ha_model/charger.py`, `get_best_car()` (lines 2874-2877), `home_model/load.py` (lines ~420-423)

```python
if car.get_user_originated("charger_name") == FORCE_CAR_NO_CHARGER_CONNECTED:
    _LOGGER.info("get_best_car: FORCE_CAR_NO_CHARGER_CONNECTED car: %s", car.name)
    continue  # Car is permanently skipped!
```

The `_user_originated` dict is persisted across reboots (via `load.py` lines ~420-423). If a previous session set `FORCE_CAR_NO_CHARGER_CONNECTED` on a car (or any other manual override), it persists indefinitely — even after the car has left home and come back. There is no lifecycle mechanism to auto-reset stale user state when the car starts a new "at home" session.

More broadly, ALL user-originated state (`FORCE_CAR_NO_CHARGER_CONNECTED`, manual charger assignments, inferred flags) should be treated as session-scoped to one "at home" period. When the car leaves home for real, those overrides are stale and should be cleared before the car's next arrival.

### Why manual reset fixes it

`user_clean_and_reset()` calls `clear_all_user_originated()`, which removes persisted flags (including `FORCE_CAR_NO_CHARGER_CONNECTED`) and inferred flags. The next 7-second cycle re-evaluates from scratch with fresh API data, and the known car can finally score > 0. The fix is to automate this same cleanup on departure.

### Scoring Context (not a bug, but relevant)

The long-relationship bonus (line 2711-2716) already correctly excludes invited/guest cars (`if car.car_is_invited is False`). Guest cars score -1.0 and are only used as a last-resort fallback in `get_best_car()`. The issue is not that the guest car scores too high, but that the known car scores 0 due to Root Causes 1-3.

## Acceptance Criteria

1. **AC1**: When a charger has a guest car attached and the car API reports a known car as plugged+home, the system automatically replaces the guest car with the known car within 2 solver cycles
2. **AC2**: A car in stale-percent mode that is plugged+home but not yet attached to a charger can exit stale mode
3. **AC3**: A car whose API just started reporting `plugged=True` (duration < 15s) gets a non-zero score via instant-check fallback
4. **AC4**: When a car has been confirmed not-home for 15 minutes (home→not-home transition), all user-originated state and inferred flags are automatically cleared, so the car starts fresh on its next arrival
5. **AC5**: User-originated state is NOT cleared while the car is still home (even if unplugged) — the user's manual overrides remain valid for the current "at home" session
6. **AC6**: Manual car selection (user-originated) still takes highest priority while the car is home and is NOT affected by these changes
7. **AC7**: Existing stale mode behavior (inferred flags during genuine API outage) continues to work correctly
8. **AC8**: All existing tests pass; new tests cover the guest-to-known-car transition scenarios

## Tasks / Subtasks

- [ ] Task 1: Fix `can_exit_stale_percent_mode` deadlock (AC: 2)
  - [ ] 1.1: In `can_exit_stale_percent_mode()` (car.py lines 825-838), change the `else` (not connected) branch to also allow exit when the car reports both `plugged=True` AND `home=True` — this means the car is at a charger but not yet attached in QS
  - [ ] 1.2: Update existing test `test_exit_not_connected_plug_true_blocks` to verify the new "plugged+home allows exit" behavior
  - [ ] 1.3: Add test: car in stale-percent mode, not connected, API reports plugged+home → can exit stale
  - [ ] 1.4: Add test: car in stale-percent mode, not connected, API reports plugged but NOT home → cannot exit stale (safety: car might be plugged elsewhere)

- [ ] Task 2: Fix `for_duration` scoring gap with instant-check fallback (AC: 1, 3)
  - [ ] 2.1: In `get_car_score()` (charger.py lines 2722-2727), change the `car_plug_res` fallback to trigger on `False` as well (not just `None`). When `for_duration` returns `False` but instant check returns `True`, set `score_plug_bump = 2` (reduced weight, lower than the duration-confirmed `5`)
  - [ ] 2.2: Apply the same pattern for `car_home_res` (lines 2783-2788): fall back to instant check on `False`, use reduced `score_dist_bump` weight
  - [ ] 2.3: Write tests: car API just started reporting plugged (duration < 15s) → instant fallback gives non-zero score
  - [ ] 2.4: Write tests: car API reports plugged for > 15s → full score (no change to existing behavior)

- [ ] Task 3: Auto-reset car on departure — clear stale user-originated state (AC: 4, 5)
  - [ ] 3.1: Add a new constant `CAR_NOT_HOME_AUTO_RESET_S = 15 * 60` (15 minutes) in `const.py`
  - [ ] 3.2: Add tracking state in car.py: `_car_not_home_since: datetime | None = None` to record the home→not-home transition timestamp
  - [ ] 3.3: In `_update_car_api_staleness()` (or `update_states()`), add departure detection logic:
    - Use `_get_raw_is_car_home(time)` (NOT the inferred version) to check actual API state
    - When raw home transitions from `True` to `False`: record `_car_not_home_since = time`
    - When raw home is `True`: reset `_car_not_home_since = None`
    - When `_car_not_home_since` is set and `time - _car_not_home_since > CAR_NOT_HOME_AUTO_RESET_S`: perform auto-reset
  - [ ] 3.4: The auto-reset should perform a full car reset — same as the red button (`user_clean_and_reset()`). This clears user-originated state, inferred flags, constraints, charge targets, detaches charger (no-op if car is away), and recomputes person allocation. One mechanism, clean slate for the next arrival. Log at info level when this happens.
  - [ ] 3.5: Handle edge case: if `car_tracker` is None (no home sensor), skip this mechanism entirely — cannot detect departure
  - [ ] 3.6: Write test: car home → car leaves → 15 min passes → user-originated state cleared (including `FORCE_CAR_NO_CHARGER_CONNECTED`)
  - [ ] 3.7: Write test: car home → car leaves → only 10 min → user-originated state preserved
  - [ ] 3.8: Write test: car home → brief GPS glitch (not-home for 5 min then back) → user-originated state preserved
  - [ ] 3.9: Write test: car with no tracker → no auto-reset attempted

- [ ] Task 4: Add diagnostic logging (AC: all)
  - [ ] 4.1: Add debug-level logging in `get_best_car()` when the generic/guest car is returned as fallback despite real cars existing
  - [ ] 4.2: Add debug-level logging in `get_car_score()` when a car's score is 0 despite having some positive sub-scores, and when instant-check fallback is used
  - [ ] 4.3: Add info-level logging when departure auto-reset triggers

- [ ] Task 5: Integration test — full guest-to-known-car transition (AC: 1, 8)
  - [ ] 5.1: Write an end-to-end scenario test: charger plugged → guest car attached → car API reports plugged+home → known car replaces guest car automatically
  - [ ] 5.2: Write scenario: car had FORCE_NO_CHARGER from previous session → car left home for 15 min → came back → flag cleared → car scores normally
  - [ ] 5.3: Verify manual selection still overrides everything while car is home
  - [ ] 5.4: Run full quality gate (`python scripts/qs/quality_gate.py`)

## Dev Notes

### Architecture Constraints
- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. All fixes are in `ha_model/` which is fine since the scoring and stale logic live there.
- **Solver step**: `SOLVER_STEP_S = 900` — do not change.
- **Logging**: Use lazy `%s` formatting, no f-strings in log calls, no periods at end.
- **Async**: No blocking calls in async code.
- **Constants**: All config keys in `const.py`. One new constant needed: `CAR_NOT_HOME_AUTO_RESET_S`.

### Key Code Locations

| Area | File | Lines | Method |
|------|------|-------|--------|
| Stale percent exit | `ha_model/car.py` | 825-838 | `can_exit_stale_percent_mode()` |
| Stale detection | `ha_model/car.py` | 621-703 | `_update_car_api_staleness()` |
| Exit stale mode | `ha_model/car.py` | 731-737 | `_exit_stale_mode()` |
| Inferred plugged | `ha_model/car.py` | 1148-1154 | `is_car_plugged()` |
| Inferred home | `ha_model/car.py` | 1207-1213 | `is_car_home()` |
| Clear inferred flags | `ha_model/car.py` | 784-787 | `clear_inferred_flags()` |
| Scoring algorithm | `ha_model/charger.py` | 2677-2810 | `get_car_score()` |
| Plug duration fallback | `ha_model/charger.py` | 2722-2727 | `get_car_score()` |
| Home duration fallback | `ha_model/charger.py` | 2783-2788 | `get_car_score()` |
| Score gate | `ha_model/charger.py` | 2794-2798 | `get_car_score()` |
| Score > 0 filter | `ha_model/charger.py` | 2880-2882 | `get_best_car()` |
| FORCE_NO_CHARGER skip | `ha_model/charger.py` | 2874-2877 | `get_best_car()` |
| Best car selection | `ha_model/charger.py` | 2812-2968 | `get_best_car()` |
| Car swap decision | `ha_model/charger.py` | 3153-3218 | `check_load_activity_and_constraints()` |
| Attach/detach car | `ha_model/charger.py` | 3755-3785 | `attach_car()`, `detach_car()` |
| Guest car creation | `ha_model/charger.py` | 2055-2057 | `__init__()` |
| User originated persist | `home_model/load.py` | ~420-423 | persistence logic |
| Clear all user originated | `home_model/load.py` | 169-170 | `clear_all_user_originated()` |
| Car user_clean_and_reset | `ha_model/car.py` | 2209-2224 | `user_clean_and_reset()` |
| Base user_clean_and_reset | `home_model/load.py` | 304-307 | `user_clean_and_reset()` |
| Reset button | `button.py` | 162-174, 263-277 | `async_press()` |

### Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `CAR_API_STALE_THRESHOLD_S` | 21600 (6h) | `const.py` ~104 |
| `CAR_SOC_STALE_THRESHOLD_S` | 3600 (1h) | `const.py` ~105 |
| `CAR_CHARGER_LONG_RELATIONSHIP_S` | 3600 (1h) | `charger.py` ~190 |
| `CHARGER_LONG_CONNECTION_S` | 1200 (20min) | `charger.py` ~191 |
| `CHARGER_CHECK_STATE_WINDOW_S` | 15 (s) | `charger.py` ~178 |
| `CAR_NOT_HOME_AUTO_RESET_S` | 900 (15min) | `const.py` (NEW) |

### Testing Approach
- Use `create_test_car_double()` and `create_test_charger_double()` from `tests/factories.py` for lightweight test doubles
- Use `freezegun` for time-dependent scenarios (stale transitions, duration thresholds)
- **Existing test to update**: `test_exit_not_connected_plug_true_blocks` — currently codifies the deadlock as intentional; must be updated to allow exit when plugged+home
- Test scenarios must cover:
  - Stale-percent car, not connected, API reports plugged+home → can exit stale
  - Stale-percent car, not connected, API reports plugged but NOT home → stays stale
  - Car API just started reporting plugged (duration < 15s) → instant fallback gives reduced score > 0
  - Car API reports plugged for > 15s → full score (unchanged behavior)
  - Car leaves home for 15 min → all user-originated state auto-cleared
  - Car leaves home for < 15 min (GPS glitch) → state preserved
  - Car still home but unplugged → state preserved (not-home is the trigger, not unplugged)
  - Car with no tracker → auto-reset skipped
  - Full guest-to-real-car swap scenario end-to-end
  - Manual selection still overrides everything while car is home
- 100% coverage required

### Risk Assessment
- **Low risk**: Task 3 (departure auto-reset) — only triggers after 15 min confirmed not-home; uses raw API (no inferred override); clears state that would be cleared by manual reset anyway
- **Medium risk**: Task 1 (stale percent exit) — changes a state machine exit condition; must not allow exit when car is plugged at a different location (hence the plugged+home double-check)
- **Medium risk**: Task 2 (scoring fallback) — changes score weights; the reduced weight (2 vs 5) ensures instant-only confirmation doesn't dominate duration-confirmed results
- **Regression risk**: Ensure cars with no plug sensor (`car_plugged is None`) still use the existing `None` fallback path unchanged
- **Regression risk**: Ensure cars with no tracker (`car_tracker is None`) skip the departure auto-reset entirely

### Project Structure Notes
- All changes confined to `ha_model/` layer (car.py, charger.py) — no architecture boundary violations
- One new constant in `const.py`: `CAR_NOT_HOME_AUTO_RESET_S`
- No translation changes needed

### References
- [Source: ha_model/car.py:825-838 - `can_exit_stale_percent_mode()` deadlock]
- [Source: ha_model/charger.py:2722-2727 - `get_car_score()` plug duration fallback]
- [Source: ha_model/charger.py:2783-2788 - `get_car_score()` home duration fallback]
- [Source: ha_model/charger.py:2874-2877 - `get_best_car()` FORCE_CAR_NO_CHARGER_CONNECTED skip]
- [Source: ha_model/car.py:731-737 - `_exit_stale_mode()` flag clearing]
- [Source: home_model/load.py:169-170 - `clear_all_user_originated()`]
- [Source: ha_model/car.py:2209-2224 - `user_clean_and_reset()` — reference for what a full reset does]
- [Source: Cursor plan analysis - cross-validated root causes]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
