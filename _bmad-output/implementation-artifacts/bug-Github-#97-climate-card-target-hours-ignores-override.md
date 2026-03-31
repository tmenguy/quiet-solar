# Bug Fix: Climate card target hours ignores user override duration

Status: dev-complete
issue: 97
branch: "QS_97"

## Story

As a Quiet Solar user with a climate entity,
I want the target hours display to reflect my override duration,
so that when I override from 1h to 2h, the card shows 2h as the target (matching the From/To window).

## Bug Description

When a user overrides the scheduled duration for a climate entity (e.g., from 1h to 2h), the From/To times correctly reflect the 2h window, but the "Target Hours" display still shows the original 1h value.

**Example from "Clim Arthur":**
- From: 21:06, To: 23:06 (2h window -- correct, matches override)
- Target Hours: 1h (wrong -- should show 2h to match the override)

The actual/target hours attribute should reflect the user-overridden duration, not the original scheduled value.

## Root Cause Analysis

In `ha_model/bistate_duration.py`, `update_current_metrics()` computes target (`duration_s`) and actual (`run_s`) using mode-specific logic:

- **Calendar path** (lines 76-115): fetches calendar events and sums their durations to compute `duration_s`. The override constraint's `target_value` is never consulted -- the target comes purely from calendar event durations.
- **Default path** (lines 116-144): day-filters `_constraints` by `end_of_constraint > today_utc and end_of_constraint <= tomorrow_utc`. Cross-midnight overrides can fall outside this window.

In both paths, when a user override is active, the override constraint has its own `target_value` (the override duration budget) and `current_value` (actual runtime so far), but these are not used directly. The From/To sensors work correctly because they read from `constraint.end_of_constraint` directly without day filtering (`load.py:1528-1535`).

**Key insight**: The override constraint is created with correct values at `bistate_duration.py:472-486`:
```python
override_constraint = TimeBasedSimplePowerLoadConstraint(
    ...
    load_info={"originator": "user_override"},
    target_value=3600.0 * self.override_duration,
)
```

The fix should read these values directly when an override is active.

## Fix Plan (from external plan -- primary authority)

Add an **early-return short-circuit** at the top of `update_current_metrics()`: scan `self._constraints` for an active override constraint with `load_info.get("originator") == "user_override"`. If found, set metrics directly from the constraint's `target_value` and `current_value`, bypassing all calendar/default logic.

### Change location

`ha_model/bistate_duration.py` -- `update_current_metrics()`, insert after line 74 (after `run_s = 0.0`, before the `if self._is_calendar_based_mode(...)` block):

```python
        # Short-circuit: during a running user override, show override progress only
        for ct in self._constraints:
            if (
                ct.is_constraint_active_for_time_period(time)
                and ct.load_info is not None
                and ct.load_info.get("originator", "") == "user_override"
            ):
                self.qs_bistate_current_on_h = ct.current_value / 3600.0
                self.qs_bistate_current_duration_h = ct.target_value / 3600.0
                return
```

No card-side changes needed -- the card already reads `qs_bistate_current_duration_h` as the target in non-default modes.

## Acceptance Criteria

1. **AC1**: When user overrides climate duration to Xh, `qs_bistate_current_duration_h` sensor shows X (not the original scheduled value)
2. **AC2**: `qs_bistate_current_on_h` reflects actual accumulated runtime from the override constraint's `current_value`
3. **AC3**: From/To times continue to match the override window (no regression)
4. **AC4**: After the override constraint is met (completed), normal calendar/default metrics resume
5. **AC5**: Works for both calendar-based modes (`bistate_mode_auto`, `bistate_mode_exact_calendar`) and default mode
6. **AC6**: Existing non-override constraint metrics are unaffected

## Tasks / Subtasks

- [x] Task 1: Add override short-circuit in `update_current_metrics()` (AC: 1, 2, 4, 5)
  - [x] Insert override detection loop before calendar/default branching
  - [x] Use `is_constraint_active_for_time_period(time)` + `load_info["originator"] == "user_override"` to identify active override
  - [x] Set `qs_bistate_current_on_h` and `qs_bistate_current_duration_h` from constraint values and return early
- [x] Task 2: Add tests for override metrics (AC: 1, 2, 4, 5, 6)
  - [x] Test: active user override in default mode -> metrics reflect override values
  - [x] Test: active user override in calendar mode -> metrics reflect override values (not calendar events)
  - [x] Test: completed (met) override -> falls through to normal metrics
  - [x] Test: no override present -> normal metrics unchanged

## Dev Notes

### Architecture constraints

- `update_current_metrics()` is in `ha_model/bistate_duration.py` (HA layer) -- OK to reference `_constraints` which come from `home_model/load.py`
- `is_constraint_active_for_time_period(start_time, end_time=None)` is defined in `home_model/constraints.py:485` -- returns False if constraint is met (completed) or hasn't started
- The `load_info` dict is set during constraint creation in `check_load_activity_and_constraints()` at line 479: `load_info={"originator": "user_override"}`

### Key file locations

- **Bug location**: `ha_model/bistate_duration.py:66-147` (`update_current_metrics`)
- **Override constraint creation**: `ha_model/bistate_duration.py:472-486`
- **Constraint active check**: `home_model/constraints.py:485-495`
- **From/To sensor source**: `home_model/load.py:1528-1535`
- **Card display**: `ui/resources/qs-climate-card.js:100-125`
- **Sensor entity**: `sensor.py:76-84` (creates `qs_bistate_current_duration_h`)

### Testing patterns

- Use `TimeBasedSimplePowerLoadConstraint` with `load_info={"originator": "user_override"}` to simulate override
- Mock `is_constraint_active_for_time_period` or set constraint times appropriately
- See `tests/ha_tests/test_bistate_duration.py` for existing `update_current_metrics` tests (e.g., #95 pool target offset tests)
- Calendar-mode tests need `_is_calendar_based_mode` returning True + mocked `get_next_scheduled_events`

### Previous related work

- **#95** (`bug-Github-#95-pool-target-offset-slider-mid-run.md`): Fixed similar `update_current_metrics` bugs in default/pool path -- added day lower bound and same-end-date guard. This fix is additive and does not conflict.
- **#78/#80**: Refactored metrics tracking into `bistate_duration.py`

### References

- [Source: ha_model/bistate_duration.py#update_current_metrics] -- lines 66-147
- [Source: ha_model/bistate_duration.py#check_load_activity_and_constraints] -- lines 338-493
- [Source: home_model/constraints.py#is_constraint_active_for_time_period] -- lines 485-495
- [Source: home_model/load.py#constraint_start_end_times] -- lines 1528-1535
- [External plan: fix_override_target_hours_28559f4b.plan.md] -- Cursor plan used as primary authority

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- External plan validated against codebase: method signatures, line numbers, and `load_info` field confirmed
- `is_constraint_active_for_time_period` correctly returns False for met constraints, ensuring fallback to normal metrics after override completion

### File List

- `custom_components/quiet_solar/ha_model/bistate_duration.py` (modify)
- `tests/ha_tests/test_bistate_duration.py` (modify)
