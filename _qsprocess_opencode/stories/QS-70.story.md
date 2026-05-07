# Story: Fix cumulus rapid cycling caused by filler-constraint infinite loop and bistate metrics reset

Status: done

issue: 70
branch: "QS_70"

## Story

As a Quiet Solar user with a Cumulus (hot water tank) and EV chargers,
I want the solver to not rapid-cycle my cumulus on/off hundreds of times when fully-charged cars stay plugged in,
So that my equipment is protected from excessive switching and the UI correctly shows completed runtime hours.

## Root Cause Analysis

Two bugs combine to cause the cumulus to cycle 100+ times in ~4 hours:

### Bug 1: Charger filler constraint infinite loop (primary)

When a car reaches 100% charge but stays plugged in, `check_load_activity_and_constraints` in `charger.py` (line ~3666) keeps pushing an already-met filler constraint every cycle:

1. Condition `realized_charge_target + 1.5 >= target_charge and target_charge < max_target_charge` is True when car is at 100% and `target_charge` (95%) < `max_target_charge` (100%)
2. Creates constraint with `initial_value=100.0, target_value=100.0` -- immediately met
3. `push_live_constraint` accepts it because previous filler was already removed
4. `update_live_constraints` detects it as met, removes it, sets `force_solving = True`
5. Next 7-second cycle: same filler pushed again -- infinite loop causing solver re-runs every 7 seconds

### Bug 2: Bistate metrics reset on constraint removal

`update_current_metrics` in `bistate_duration.py` (line ~58) resets `qs_bistate_current_on_h` to 0 when the active constraint is removed, with no fallback to `_last_completed_constraint`. The pool.py version already handles this correctly.

### Combined impact

The solver running every 7 seconds generates new plans each time. Under marginal surplus conditions, the cumulus alternates ON/OFF (classic control oscillation). The UI shows 0h instead of the completed 3h.

## Acceptance Criteria

1. **Given** a fully-charged car (realized >= target) stays plugged in
   **When** `check_load_activity_and_constraints` runs
   **Then** no filler constraint is created (type is set to None), preventing the push-remove-push loop

2. **Given** a bistate load (cumulus) completes its constraint and it is removed
   **When** `update_current_metrics` is called
   **Then** it falls back to `_last_completed_constraint` (like pool.py does) and shows the completed hours until the day cycle ends

3. **Given** the fixes are applied
   **When** quality gates run
   **Then** all tests pass with 100% coverage, ruff and mypy are clean

## Tasks / Subtasks

- [x] Task 1: Skip filler constraint when already met (AC: #1)
  - [x] 1.1 In `charger.py` at line ~3670, after `target_charge = max_target_charge`, add guard: `if realized_charge_target >= target_charge: type = None`
  - [x] 1.2 Add unit test: fully-charged car plugged in does not push filler constraint
  - [x] 1.3 Add unit test: partially-charged car still pushes filler constraint as before

- [x] Task 2: Bistate metrics fallback to completed constraint (AC: #2)
  - [x] 2.1 In `bistate_duration.py` `update_current_metrics`, add `end_range` parameter handling matching pool.py pattern
  - [x] 2.2 Build `ct_to_probe` list: extend with `_constraints`, elif fallback to `_last_completed_constraint`
  - [x] 2.3 Filter by current day window (start_day to end_day) before accumulating metrics
  - [x] 2.4 Add unit test: metrics show completed hours after constraint removal (via _last_completed_constraint)
  - [x] 2.5 Add unit test: metrics show active constraint hours when constraint is still live

- [x] Task 3: Quality gates (AC: #3)
  - [x] 3.1 Run `python scripts/qs/quality_gate.py` -- all checks pass

## Key Files

- `custom_components/quiet_solar/ha_model/charger.py` lines 3640-3692 -- filler constraint creation
- `custom_components/quiet_solar/ha_model/bistate_duration.py` lines 56-64 -- UI metrics reset bug
- `custom_components/quiet_solar/ha_model/pool.py` lines 55-90 -- reference correct implementation
- `custom_components/quiet_solar/home_model/load.py` lines 1305-1373 -- `push_live_constraint`

## Dev Notes

- The external plan file is the primary authority for root cause analysis and fix ordering
- Fix 1 (charger guard) is the priority -- it eliminates the infinite loop entirely
- Fix 2 (bistate metrics) is a UI correctness fix that should follow the pool.py pattern closely
- The `end_range` / day-window logic in pool.py `update_current_metrics` is the reference implementation
