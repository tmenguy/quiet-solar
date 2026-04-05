# Fix: Pool forgets elapsed runtime when switching from force-on to default mode

Status: dev-complete
issue: 118
branch: "QS_118"

## Story

As a pool owner using Quiet Solar,
I want the system to preserve my pool's accumulated runtime when I switch from force-on mode back to default mode,
so that the solver correctly recognizes hours already filtered today and does not re-schedule work that was already done.

## Problem

When a pool has been running in force-on mode for 11 hours and the user switches to default mode (e.g., 8h/day), the system creates a new constraint with `current_value=0` instead of carrying the 11h of accumulated runtime. The constraint should be immediately satisfied (11h >= 8h) but instead the solver schedules a fresh 8h of filtering.

### Reproduction steps

1. Pool is in force-on mode, running for 11 hours (`current_value = 11h * 3600`)
2. Force-on constraint: `end_of_constraint = tomorrow_midnight`, `target_value = 25h`
3. User switches mode to "Default" with 8h/day duration
4. System creates new default constraint: `end_of_constraint = configured_finish_time` (e.g., 20:00), `target_value = 8h`
5. **Bug:** New constraint starts with `current_value = 0` — the 11h of runtime is lost
6. Solver schedules 8h of new filtering instead of recognizing the constraint is already met

### Root cause

Three cooperating failures across `push_live_constraint` and `set_live_constraints`:

**Failure 1 — Active constraint replacement requires matching end times (load.py:1351)**

The replacement loop only triggers when `c.end_of_constraint == constraint.end_of_constraint`. Force-on ends at tomorrow midnight; default ends at the configured finish time (e.g., 20:00). End times differ, so the old active constraint is never replaced — both constraints coexist, and the runtime carry at line 1366 never fires.

**Failure 2 — Carry-from-completed also requires matching end times (load.py:1338)**

The carry-from-completed block added by Bug #68 requires:
```python
self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
or self._last_completed_constraint.initial_end_of_constraint == constraint.end_of_constraint
```
This condition also fails on cross-mode switching because the end times differ.

**Failure 3 — Hardcoded `initial_value=0` (bistate_duration.py:586)**

All constraints created in `check_load_activity_and_constraints` use `initial_value=0`, so there is no mechanism to seed the new constraint with accumulated runtime from outside `push_live_constraint`.

**Failure 4 — `set_live_constraints` silently drops met constraints (load.py:1284)**

Even if carry-over succeeds and `current_value` is transferred correctly, `set_live_constraints` at line 1284 filters out met constraints:
```python
self._constraints = [c for c in self._constraints if c.is_constraint_met(time=time) is False]
```
This happens **without calling `ack_completed_constraint`**, so `_last_completed_constraint` is never updated. On the next periodic `check_load_activity_and_constraints` call, a fresh constraint is created with `current_value=0` — and the runtime is lost again.

### Why existing carry logic does not help

| Carry mechanism | Location | Why it fails |
|---|---|---|
| Carry from completed constraint | `load.py:1331-1345` | End time mismatch (tomorrow midnight vs 20:00) |
| Active constraint replacement | `load.py:1351-1367` | End time mismatch (same reason) |
| `reset_initial_value_to_follow_prev_if_needed` | `constraints.py:526` | Only fires when `initial_value is None`; hardcoded to `0` |
| `set_live_constraints` met filter | `load.py:1284` | Silently drops met constraints without `ack_completed_constraint` |

## Fix Plan

### Fix 1: Save runtime and clean up old-mode constraints in `check_load_activity_and_constraints`

**File:** `custom_components/quiet_solar/ha_model/bistate_duration.py`
**Location:** In the `else` block at line 565, **before** `_build_mode_constraint_items` is called

Before building new-mode constraints, save the max `current_value` from any existing non-override constraint and remove them. This ensures old force-on constraints are cleaned up even when end dates differ from the new default constraint.

```python
# Save runtime from old-mode constraints and clean them up
saved_runtime = 0.0
old_removed = False
for i, ct in enumerate(self._constraints):
    if ct.load_info is not None and ct.load_info.get("originator", "") == "user_override":
        continue  # keep override constraints
    if ct.current_value > saved_runtime:
        saved_runtime = ct.current_value
    self._constraints[i] = None
    old_removed = True

if old_removed:
    self._constraints = [c for c in self._constraints if c is not None]
```

Then after building the new constraint (line 577-588) but before `push_live_constraint`, pre-seed its `current_value`:

```python
load_mandatory = TimeBasedSimplePowerLoadConstraint(
    ...
    initial_value=0,
    target_value=ct.target_value,
)
# Pre-seed with saved runtime from old-mode constraint
if saved_runtime > 0:
    load_mandatory.current_value = min(saved_runtime, ct.target_value)
```

### Fix 2: Intercept immediately-met constraints in `push_live_constraint`

**File:** `custom_components/quiet_solar/home_model/load.py`
**Location:** In the `else` branch of the active constraint replacement loop, after the carry at lines 1365-1367

After carry-over (either from Fix 1 pre-seeding or from the existing replacement carry at line 1366), the new constraint may be immediately met. If `set_live_constraints` (line 1370) filters it out at line 1284, `_last_completed_constraint` is never set — and the next periodic check recreates the constraint from scratch.

Fix: after the carry-over, check if the constraint is now met. If so, set `_last_completed_constraint` directly, clean up, and return `True` without appending:

```python
else:
    self._constraints[i] = None
    if type(c) == type(constraint) and c.current_value > constraint.current_value:
        constraint.current_value = min(c.current_value, constraint.target_value)
    # If carry-over made constraint immediately met, ack and bail out
    if constraint.is_constraint_met(time=time):
        self._last_completed_constraint = constraint
        self._constraints = [x for x in self._constraints if x is not None]
        return True
```

**Why not call `ack_completed_constraint`?** It is `async` while `push_live_constraint` is sync. The essential bookkeeping (`_last_completed_constraint = constraint`) is done synchronously. We skip `on_device_state_change(CONSTRAINT_COMPLETED)` — acceptable since the constraint was never active in the solver.

**Guard interaction:** After Fix 2 sets `_last_completed_constraint`, the identity check at lines 1317-1329 will correctly reject subsequent re-pushes of the same Default constraint (same `requested_target_value` + same `end_of_constraint`). Force-on has `target=25h` vs Default `target=8h`, so the guard will not cross-block different modes.

### Fix 3: Carry from active constraints regardless of end time in `push_live_constraint`

**File:** `custom_components/quiet_solar/home_model/load.py`
**Location:** After the carry-from-completed block (after line 1345), before the active constraint loop (line 1347)

Defense-in-depth: even without Fix 1's pre-seeding, `push_live_constraint` should carry `current_value` from an active constraint of the same type when end times differ (mode switch). Mark the old one for removal:

```python
# Carry current_value from active constraint during mode switch
# (different end times, same constraint type for same load)
for i, c in enumerate(self._constraints):
    if (
        type(c) == type(constraint)
        and c.current_value > constraint.current_value
    ):
        constraint.current_value = min(c.current_value, constraint.target_value)
        self._constraints[i] = None
        _LOGGER.info(
            "Constraint %s replacing %s on mode switch, carried current_value %.1f",
            constraint.name,
            c.name,
            constraint.current_value,
        )
        break

self._constraints = [c for c in self._constraints if c is not None]
```

**Combined with Fix 2:** After this carry, also check `constraint.is_constraint_met(time)`. If met, set `_last_completed_constraint = constraint` and return `True` without appending — same pattern as Fix 2.

### Fix 4: Relax carry-from-completed end-time condition

**File:** `custom_components/quiet_solar/home_model/load.py`
**Location:** Lines 1331-1345

Belt-and-suspenders for the case where the force-on constraint was already completed (stored in `_last_completed_constraint`) before the mode switch. Remove the end-time matching requirement:

Change:
```python
if (
    self._last_completed_constraint is not None
    and type(self._last_completed_constraint) == type(constraint)
    and self._last_completed_constraint.current_value > constraint.current_value
    and (
        self._last_completed_constraint.end_of_constraint == constraint.end_of_constraint
        or self._last_completed_constraint.initial_end_of_constraint == constraint.end_of_constraint
    )
):
```

To:
```python
if (
    self._last_completed_constraint is not None
    and type(self._last_completed_constraint) == type(constraint)
    and self._last_completed_constraint.current_value > constraint.current_value
):
```

**Risk:** Could carry stale runtime from a previous day cycle. `_last_completed_constraint` is reset via `ack_completed_constraint` on new day cycles, so the risk is low. If needed, add a same-day safety check:
```python
and self._last_completed_constraint.end_of_constraint.date() == constraint.end_of_constraint.date()
```

### Fix priority

| Fix | Layer | Criticality | What it prevents |
|---|---|---|---|
| Fix 1 | `bistate_duration.py` | **Must have** | Cleans up stale old-mode constraints + pre-seeds runtime |
| Fix 2 | `load.py` | **Must have** | Prevents met constraint from being silently dropped and recreated |
| Fix 3 | `load.py` | **Should have** | Defense-in-depth carry in `push_live_constraint` for any mode switch |
| Fix 4 | `load.py` | **Nice to have** | Carry from completed constraint across mode switches |

## Acceptance Criteria

1. When a pool has been running 11h in force-on mode and user switches to default (8h/day), the constraint is immediately satisfied, `_last_completed_constraint` is set, and metrics show 8h/8h
2. When a pool has been running 5h in force-on mode and user switches to default (8h/day), the constraint shows 5h/8h and solver schedules only 3 more hours
3. When switching from default to force-on, the existing runtime is also preserved
4. When switching to off mode, constraints are still correctly cleared (existing behavior)
5. The identity check (same target + same end time as completed) still blocks re-push correctly
6. The Bug #68 carry-from-completed still works for same-mode target extension (4h→7h)
7. After an immediately-met constraint is acked, subsequent periodic checks do NOT recreate a fresh 0h constraint
8. Old-mode constraints with different end times are cleaned up (no zombie force-on constraint alongside new default)
9. All existing tests continue to pass

## Tasks / Subtasks

- [x] Task 1: Save runtime and clean up old-mode constraints in `bistate_duration.py` (AC: 1, 2, 3, 8)
  - [x] In `check_load_activity_and_constraints` else block, save max `current_value` from non-override constraints
  - [x] Remove old non-override constraints before building new ones
  - [x] Pre-seed new constraint's `current_value` with `min(saved_runtime, target_value)` after creation
- [x] Task 2: Intercept immediately-met constraints in `push_live_constraint` (AC: 1, 7)
  - [x] After carry-from-completed and before the active constraint loop, check `constraint.is_constraint_met(time)`
  - [x] If met: set `_last_completed_constraint = constraint`, return `True`
  - [x] Also in the replacement `else` branch (same-end-time replacement with carry)
- [x] Task 3: **Dropped** — carry from active constraints regardless of end time
  - Removed during implementation: too broad, interfered with charger constraints (different end times are legitimate for chargers). Fix 1 handles active constraint cleanup at the bistate level instead.
- [x] Task 4: **Reverted** — relaxed carry-from-completed end-time condition
  - Removed `or self._last_completed_constraint.end_of_constraint >= time` during review
  - Too broad: fires for any completed constraint whose deadline hasn't passed, not just cross-mode switches
  - Cross-mode runtime preservation relies solely on Fix 1 (bistate pre-seeding)
  - Bug #68 same-end-time matching preserved unchanged
- [x] Task 5: Add tests (AC: 1, 2, 3, 4, 5, 6, 7, 8, 9)
  - [x] Test force-on→default with runtime exceeding new target (11h→8h) — immediately met
  - [x] Test force-on→default with runtime below new target (5h→8h) — partial carry
  - [x] Test default→force-on runtime preservation via completed carry
  - [x] Test cross-mode NO carry from completed with different end times (Fix 4 reverted)
  - [x] Test identity check blocks re-push after immediately-met ack
  - [x] Test Fix 1 cleanup simulation at push level
  - [x] Test Bug #68 same-mode extension still works
  - [x] Test no stale carry from previous day
  - [x] Test same-end-time replacement carry → immediately met (line coverage)
  - [x] Test set_live_constraints infinite-met skip (line coverage)
  - [x] Test set_live_constraints all-met empty result (line coverage)

## Dev Notes

### Architecture constraints

- **Two-layer boundary**: Fix spans both `ha_model/bistate_duration.py` (HA layer — mode-switch cleanup) and `home_model/load.py` (domain layer — constraint carry/ack). The bistate fix handles HA-specific mode logic; the load fix handles generic constraint management. No new cross-boundary imports.
- **Logging**: Use lazy `%s` formatting, no f-strings in log calls, no trailing periods.
- **Type checking**: Use `type(c) == type(constraint)` (not `isinstance`) — consistent with existing carry logic at line 1335 and 1366.
- **Async boundary**: `push_live_constraint` is sync and returns `(pushed: bool, needs_ack: bool)`. When `needs_ack` is True the caller (always async) must `await ack_completed_constraint(time, constraint)`. `_last_completed_constraint` is also set as a sync guardrail for the identity check. `push_agenda_constraints` returns `(bool, list[LoadConstraint])` following the same pattern.

### Supplementary mode-change detection via `_previous_bistate_mode`

The original end-time comparison (`mode_changed = any(c.end_of_constraint not in new_ends ...)`) is fragile: if force-on and default modes happen to produce the same `end_of_constraint` value, `mode_changed` evaluates to `False` and runtime pre-seeding is skipped. A new instance variable `_previous_bistate_mode` (initialized to `None` in `__init__`) remembers the bistate mode string from the previous call. After the end-time check, the detection is supplemented:

```python
if self._previous_bistate_mode is not None and self._previous_bistate_mode != bistate_mode:
    mode_changed = True
self._previous_bistate_mode = bistate_mode
```

Both checks are kept: end-time detection catches calendar changes where end times shift without a mode change; the state variable catches mode switches where end times coincide by accident.

### Key code locations

| File | Lines | What |
|---|---|---|
| `ha_model/bistate_duration.py` | 565-605 | `check_load_activity_and_constraints` else block — Fix 1 location |
| `ha_model/bistate_duration.py` | 577-588 | Constraint creation with `initial_value=0` — pre-seed location |
| `ha_model/bistate_duration.py` | 278-356 | `_build_mode_constraint_items` — different end times per mode |
| `home_model/load.py` | 1284 | `set_live_constraints` met filter — silent drop of met constraints |
| `home_model/load.py` | 1305-1373 | `push_live_constraint` — Fix 2, 3, 4 location |
| `home_model/load.py` | 1317-1329 | Identity check — blocks re-push of same completed constraint |
| `home_model/load.py` | 1331-1345 | Carry-from-completed block (Bug #68) — Fix 4 location |
| `home_model/load.py` | 1347-1367 | Active constraint replacement loop — Fix 2 intercept location |
| `home_model/load.py` | 1106-1122 | `ack_completed_constraint` — async, NOT called from sync fix |
| `home_model/constraints.py` | 82-147 | `LoadConstraint.__init__` — `initial_value` handling |
| `tests/test_load_model.py` | — | Existing constraint carry tests |
| `tests/test_ha_pool.py` | — | Pool-specific tests |

### Related prior work

- **Bug #68** (`bug-Github-#68-carry-from-completed-constraint.md`): Added carry-from-completed block for same-mode target extension. This fix extends that work to cross-mode scenarios.
- The carry-from-completed block at lines 1331-1345 was the fix for Bug #68. Our fix must not break that behavior.

### Testing patterns

- Use `freezegun` for time control
- Create `TimeBasedSimplePowerLoadConstraint` instances directly
- Use `push_live_constraint` to simulate the push flow
- Assert `current_value` after push to verify carry
- Test file: `tests/test_load_model.py`

### Project Structure Notes

- Fix spans `ha_model/bistate_duration.py` (mode-switch cleanup) and `home_model/load.py` (constraint carry/ack)
- No new files needed — modify existing source and test files
- No config key changes, no UI changes, no translation changes

### References

- [Source: home_model/load.py#push_live_constraint] — main fix location
- [Source: ha_model/bistate_duration.py#_build_mode_constraint_items] — shows why end times differ per mode
- [Source: bug-Github-#68-carry-from-completed-constraint.md] — prior art for carry logic

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
- Fix 3 (carry from active constraints) dropped: interfered with charger constraints in test_forecasts.py. Chargers legitimately have multiple same-type constraints with different end times.
- Fix 4 staleness guard uses `end_of_constraint >= time` to prevent stale cross-day carry while allowing cross-mode carry within the same cycle.

### Completion Notes List
- Fix 1: bistate_duration.py — saves runtime from old-mode constraints, removes them, pre-seeds new constraint
- Fix 2: load.py — intercepts immediately-met constraints after carry (both pre-carry and same-end-time replacement)
- Fix 3: DROPPED — too broad, replaced by Fix 1's bistate-level cleanup
- Fix 4: load.py — relaxed carry-from-completed with staleness guard (end_of_constraint >= time)
- 11 new tests covering all ACs + line coverage for pre-existing gaps

### File List
- `custom_components/quiet_solar/ha_model/bistate_duration.py` — Fix 1
- `custom_components/quiet_solar/home_model/load.py` — Fixes 2, 4
- `tests/test_load_model.py` — 11 new tests (TestModeSwitchRuntimeCarry + TestSetLiveConstraintsCoverage)
