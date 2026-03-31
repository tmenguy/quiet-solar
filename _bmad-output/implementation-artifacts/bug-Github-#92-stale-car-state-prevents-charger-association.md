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

The bug is in the car-charger scoring and association flow. There are **two interacting failure modes**, both in the transition from guest car to known car.

### Failure Mode 1: Guest car "long relationship" score blocks replacement

**Location:** `ha_model/charger.py`, `get_car_score()` method (~line 2693, 2711-2716)

When the guest car has been attached for over 1 hour (`CAR_CHARGER_LONG_RELATIONSHIP_S = 3600`), it receives a score of `max_score - 1.0` (nearly the highest possible score). The known car's dynamic score (based on plug state + distance/timing) can never exceed this because the dynamic formula caps well below `max_score - 1`. Result: guest car wins scoring indefinitely once the 1-hour threshold is crossed.

**Why manual reset fixes it:** Reset detaches the guest car, clearing `car_attach_time`. On the next cycle, the guest car no longer has a long relationship bonus, so the known car's score wins.

### Failure Mode 2: `check_manual_assignment_contradiction()` only runs for attached cars

**Location:** `ha_model/car.py`, `_update_car_api_staleness()` (~line 653-660)

The contradiction check that sets `_car_api_inferred_plugged = True` only runs when the car is already attached to a charger (`if self.charger is not None`). Since the guest car is attached (not the known car), the known car never gets its inferred flags set. If the API is slow to report `plugged=True` or reports `plugged=False` briefly, the known car gets `score_plug_bump = 0` and cannot score high enough to replace the guest.

### Failure Mode 3: Stale exit clears inferred flags before API catches up

**Location:** `ha_model/car.py`, `_exit_stale_mode()` (~line 731-737)

If the car was previously stale with inferred flags set, and the API reconnects (sensors update), `_exit_stale_mode()` clears `_car_api_inferred_home` and `_car_api_inferred_plugged`. If the API's first update after reconnect still has stale/wrong data, the car loses its inferred override and gets score 0.

### Scoring Gate Condition

**Location:** `ha_model/charger.py` (~line 2794-2798, ~line 2880-2882)

The critical gate: a car MUST have `score_plug_bump > 0 AND (score_dist_bump > 0 OR score_plug_time_bump > 0)` to get any non-zero score. If the car API says `plugged=False`, `score_plug_bump = 0` and the car is excluded entirely from consideration.

## Acceptance Criteria

1. **AC1**: When a charger has a guest car attached and the car API reports a known car as plugged, the system automatically replaces the guest car with the known car within 2 solver cycles
2. **AC2**: A guest car that has been attached for more than 1 hour does NOT block replacement by a known car whose API confirms it is plugged
3. **AC3**: If the car API was stale and reconnects reporting `plugged=True`, the known car is associated within 2 solver cycles without manual intervention
4. **AC4**: Manual car selection (user-originated) still takes highest priority and is NOT affected by these changes
5. **AC5**: Existing stale mode behavior (inferred flags during genuine API outage) continues to work correctly
6. **AC6**: All existing tests pass; new tests cover the guest-to-known-car transition scenarios

## Tasks / Subtasks

- [ ] Task 1: Fix guest car long-relationship score blocking known car replacement (AC: 1, 2)
  - [ ] 1.1: In `get_car_score()` (~line 2711-2716 in charger.py), exclude guest/invited cars from the long-relationship score bonus. An invited car should never get `max_score - 1` just for being attached a long time — it should always yield to a real car with a positive dynamic score.
  - [ ] 1.2: Add a check: `if is_long_time_attached and not car.car_is_invited:` before granting the long-relationship bonus
  - [ ] 1.3: Write tests for guest car scoring when attached > 1 hour vs known car with positive plug score

- [ ] Task 2: Improve scoring to prefer known cars over guest cars when plug evidence exists (AC: 1, 3)
  - [ ] 2.1: In the scoring algorithm, ensure that when a known car has `score_plug_bump > 0` (API confirms plugged), it always scores higher than the guest car's fallback score
  - [ ] 2.2: Consider adding a "known car plugged bonus" that elevates known-car scores above guest-car scores when plug evidence is strong
  - [ ] 2.3: Write tests for known-car-vs-guest-car scoring comparison

- [ ] Task 3: Handle API reconnect transition gracefully (AC: 3, 5)
  - [ ] 3.1: Review `_exit_stale_mode()` (car.py ~line 731-737) to ensure inferred flags are not cleared prematurely when the API's first post-reconnect value might still be stale
  - [ ] 3.2: Consider a grace period after stale exit where the car retains inferred flags until at least one fresh API reading confirms the actual state
  - [ ] 3.3: Write tests for the stale→non-stale transition with delayed API truth

- [ ] Task 4: Add diagnostic logging for car-charger association decisions (AC: all)
  - [ ] 4.1: Add debug-level logging in `get_best_car()` showing the winning car, its score, and the runner-up car with score, especially when a guest car wins over a known car
  - [ ] 4.2: Log when a guest car's long-relationship bonus causes it to win over a known car with positive plug evidence

- [ ] Task 5: Verify manual selection and existing stale behavior are preserved (AC: 4, 5, 6)
  - [ ] 5.1: Write/verify tests that user-originated car selection still wins over everything
  - [ ] 5.2: Write/verify tests that stale mode inferred flags still work during genuine API outages
  - [ ] 5.3: Run full quality gate

## Dev Notes

### Architecture Constraints
- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. All fixes are in `ha_model/` which is fine since the scoring and stale logic live there.
- **Solver step**: `SOLVER_STEP_S = 900` — do not change.
- **Logging**: Use lazy `%s` formatting, no f-strings in log calls, no periods at end.
- **Async**: No blocking calls in async code.

### Key Code Locations

| Area | File | Lines | Method |
|------|------|-------|--------|
| Scoring algorithm | `ha_model/charger.py` | ~2677-2810 | `get_car_score()` |
| Long-relationship gate | `ha_model/charger.py` | ~2693, 2711-2716 | `get_car_score()` |
| Score > 0 gate | `ha_model/charger.py` | ~2794-2798, 2880-2882 | `get_car_score()`, score collection |
| Best car selection | `ha_model/charger.py` | ~2812-2968 | `get_best_car()` |
| Car swap decision | `ha_model/charger.py` | ~3153-3218 | `check_load_activity_and_constraints()` |
| Attach car | `ha_model/charger.py` | ~3755-3776 | `attach_car()` |
| Detach car | `ha_model/charger.py` | ~3778-3785 | `detach_car()` |
| Guest car creation | `ha_model/charger.py` | ~2055-2057 | `__init__()` |
| Stale detection | `ha_model/car.py` | ~621-703 | `_update_car_api_staleness()` |
| Exit stale mode | `ha_model/car.py` | ~731-737 | `_exit_stale_mode()` |
| Contradiction check | `ha_model/car.py` | ~739-782 | `check_manual_assignment_contradiction()` |
| Inferred plugged | `ha_model/car.py` | ~1148-1154 | `is_car_plugged()` |
| Inferred home | `ha_model/car.py` | ~1207-1213 | `is_car_home()` |
| Clear inferred flags | `ha_model/car.py` | ~784-787 | `clear_inferred_flags()` |
| Reset button | `button.py` | ~162-174, 263-277 | `async_press()` |
| Charger reset | `ha_model/charger.py` | ~2103-2106 | `user_clean_and_reset()` |

### Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `CAR_API_STALE_THRESHOLD_S` | 21600 (6h) | `const.py` ~104 |
| `CAR_SOC_STALE_THRESHOLD_S` | 3600 (1h) | `const.py` ~105 |
| `CAR_CHARGER_LONG_RELATIONSHIP_S` | 3600 (1h) | `charger.py` ~190 |
| `CHARGER_LONG_CONNECTION_S` | 1200 (20min) | `charger.py` ~191 |
| `CHARGER_CHECK_STATE_WINDOW_S` | 15 (s) | `charger.py` ~178 |

### Testing Approach
- Use `create_test_car_double()` and `create_test_charger_double()` from `tests/factories.py` for lightweight test doubles
- Use `freezegun` for time-dependent scenarios (stale transitions, long-relationship thresholds)
- Test scenarios must cover:
  - Guest car attached < 1h, known car API reports plugged → known car wins
  - Guest car attached > 1h, known car API reports plugged → known car still wins (the fix)
  - Stale car exits stale, API reports plugged → association happens
  - Stale car exits stale, API reports NOT plugged → guest stays (correct behavior)
  - User manual selection still overrides everything
- 100% coverage required

### Risk Assessment
- **Low risk**: Task 1 (guest car scoring fix) is a targeted change to an `if` condition
- **Medium risk**: Task 2 (scoring preference) and Task 3 (stale exit grace period) touch more complex state machine logic — test thoroughly
- **Regression risk**: Ensure that non-invited cars still get the long-relationship bonus (only invited/guest cars should be excluded)

### Project Structure Notes
- All changes confined to `ha_model/` layer (car.py, charger.py) — no architecture boundary violations
- No new config keys needed — this is a logic fix, not a feature
- No translation changes needed

### References
- [Source: ha_model/charger.py - `get_car_score()` scoring algorithm]
- [Source: ha_model/charger.py - `get_best_car()` bipartite matching]
- [Source: ha_model/car.py - `_update_car_api_staleness()` stale state machine]
- [Source: ha_model/car.py - `_exit_stale_mode()` flag clearing]
- [Source: ha_model/car.py - `check_manual_assignment_contradiction()` Feature B]
- [Source: button.py - reset flow]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
