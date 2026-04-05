# Bug Fix: Fix solver force-solving loop

Status: done
issue: 120
branch: "QS_120"

## Story

As a Quiet Solar user with agenda (calendar) constraints on my charger or pool,
I want the solver to stop re-running every 7-second tick after a constraint is met,
so that the system operates efficiently without a CPU-burning feedback loop until the calendar event ends.

## Root Cause Analysis

The solver runs at every 7-second load-management tick because of a feedback loop between two independent triggers in `update_loads()` (`ha_model/home.py:2791`):

**Trigger 1 -- `update_live_constraints`** (`home_model/load.py:1398`):
The `len(constraints) != len(self._constraints)` check at line 1525 sets `force_solving = True` whenever ANY constraint is skipped/removed -- including repeat completions of re-pushed agenda constraints that were already met.

**Trigger 2 -- `push_agenda_constraints`** (`home_model/load.py:918`):
When a constraint was removed from `_constraints` (by the previous `update_live_constraints`), `push_agenda_constraints` can't find it via `eq_no_current`, so it calls `push_live_constraint`. If `push_live_constraint` returns `(True, True)` (accepted but immediately met), `push_agenda_constraints` counts it as `res = True` at line 966. The caller `check_load_activity_and_constraints` then calls `force_next_solve()`, which sets `_last_solve_done = None`, triggering the solve at line 2840.

**The loop:**
1. Constraint B is met, removed by `update_live_constraints` (trigger 1 fires)
2. Next cycle: calendar re-pushes B. `push_agenda_constraints` doesn't find B in `_constraints`, calls `push_live_constraint`. Either Guard #1 catches it `(False, False)` or it gets through to Guard #2 `(True, True)`. If `(True, True)`: trigger 2 fires.
3. B may or may not re-enter `_constraints`; either way at least one trigger fires. Loop continues until calendar event end time passes.

## Acceptance Criteria

1. After a constraint is met and acknowledged, re-pushing the same constraint (same `requested_target_value` and same `end_of_constraint` or `initial_end_of_constraint` as `_last_completed_constraint`) does NOT cause `force_solving` to become True.
2. `push_live_constraint` returns `(False, True)` (not `(True, True)`) when a constraint is immediately met via carry-over at line 1361-1363 -- no change was made to `_constraints`.
3. The other `return True, True` at line 1390 remains unchanged -- there `_constraints` WAS modified.
4. All existing tests for `set_live_constraints`, `push_live_constraint`, and `push_agenda_constraints` continue to pass.
5. New tests cover both fix paths: (a) `set_live_constraints` filtering out already-completed constraints, (b) `push_live_constraint` returning `(False, True)` for dead-on-arrival constraints.
6. 100% test coverage maintained.

## Tasks / Subtasks

- [x] Task 1: Fix `set_live_constraints` -- filter out already-completed constraints (AC: #1)
  - [x] 1.1: In `home_model/load.py`, method `set_live_constraints`, after the existing met-constraint filter at line 1290, add filtering for constraints matching `_last_completed_constraint`
  - [x] 1.2: Match on `requested_target_value` AND (`end_of_constraint` == lc.`end_of_constraint` OR `end_of_constraint` == lc.`initial_end_of_constraint`)
  - [x] 1.3: Add warning log when constraints are removed by this filter
- [x] Task 2: Fix `push_live_constraint` -- correct return for dead-on-arrival (AC: #2, #3)
  - [x] 2.1: At line 1361-1363, change `return True, True` to `return False, True`
  - [x] 2.2: Verify the other `return True, True` at line 1390 stays unchanged
- [x] Task 3: Update tests (AC: #4, #5, #6)
  - [x] 3.1: Add test for `set_live_constraints` filtering already-completed constraints
  - [x] 3.2: Add test for `push_live_constraint` returning `(False, True)` for dead-on-arrival
  - [x] 3.3: Add integration-level test showing the full loop is broken (agenda re-push after constraint met does not trigger force solve)
  - [x] 3.4: Run full quality gate to verify 100% coverage and all existing tests pass

## Dev Notes

### Fix 1: `set_live_constraints` -- filter out already-completed constraints

**File**: `home_model/load.py`, method `set_live_constraints`

`set_live_constraints` is the canonical normalization function -- every path that modifies `_constraints` goes through it (`push_live_constraint`, `push_agenda_constraints`, `update_live_constraints`, bistate/charger code). It already filters met constraints at line 1290:

```python
self._constraints = [c for c in self._constraints if c.is_constraint_met(time=time) is False]
```

Add filtering for constraints that match `_last_completed_constraint` immediately after:

```python
self._constraints = [c for c in self._constraints if c.is_constraint_met(time=time) is False]

# Filter out constraints matching the last completed one
if self._last_completed_constraint is not None:
    lc = self._last_completed_constraint
    before = len(self._constraints)
    self._constraints = [
        c for c in self._constraints
        if not (
            c.requested_target_value == lc.requested_target_value
            and (
                c.end_of_constraint == lc.end_of_constraint
                or c.end_of_constraint == lc.initial_end_of_constraint
            )
        )
    ]
    if len(self._constraints) != before:
        _LOGGER.warning(
            "set_live_constraints: removed %d already-completed "
            "constraint(s) matching last completed %s for %s",
            before - len(self._constraints),
            lc.name,
            self.name,
        )
```

This is the right place because:
- `set_live_constraints` is the normalization layer, called by all constraint-modifying paths
- The constraint never survives into `_constraints` regardless of which upstream path let it through
- The warning log reveals upstream gaps (Guard #1 failures, `push_agenda_constraints` mismatches, etc.)
- `update_live_constraints` needs no changes -- the `len` check stays correct since it only sees genuine constraints

### Fix 2: `push_live_constraint` -- correct return semantics for dead-on-arrival

**File**: `home_model/load.py`, method `push_live_constraint`, line 1361-1363

The docstring says `pushed is True if a change was made`. But at line 1361-1363, the constraint was NOT added to `_constraints` -- it was tested against carry-over, found immediately met, and only `_last_completed_constraint` was updated. No solver input changed.

```python
# Current (line 1361-1363):
if constraint.is_constraint_met(time=time):
    self._last_completed_constraint = constraint
    return True, True   # WRONG: no change was made to _constraints

# Fix:
if constraint.is_constraint_met(time=time):
    self._last_completed_constraint = constraint
    return False, True  # CORRECT: needs_ack but no solver-relevant change
```

The other `return True, True` at line 1390 is correct -- there `_constraints` WAS modified (existing constraint set to None and compacted).

All callers handle `(False, True)` correctly since they check `pushed` and `needs_ack` independently:
```python
pushed, needs_ack = self.push_live_constraint(time, ct)
if needs_ack:
    await self.ack_completed_constraint(time, ct)
if pushed:
    do_force_solve = True  # <-- skipped when pushed=False
```

### Why both fixes together solve the loop

- **Fix 1** filters out already-completed constraints in `set_live_constraints`. The constraint never survives into `_constraints` regardless of which upstream path pushed it. `update_live_constraints` is untouched -- its `len` check stays intact and only fires for genuine changes.
- **Fix 2** corrects `push_live_constraint` to return `(False, True)` -- no solver-input change was made, so callers don't count it as a change, preventing spurious `force_next_solve()` calls.
- Together they close both trigger paths. The solver only runs when there is a genuine change.

### Architecture compliance

- Both changes are in `home_model/load.py` -- pure domain logic, no HA imports.
- Logging uses lazy `%s` format, no f-strings, no periods at end.
- No constants changed; `SOLVER_STEP_S` untouched.

### Testing approach

- Use `MinimalTestLoad` and `create_constraint()` from `tests/factories.py`.
- Use `freezegun` for time-dependent assertions.
- Existing tests in `tests/test_load_model.py` (classes around line 2234+ for `set_live_constraints`, line 2448+ for `push_live_constraint`) and `tests/test_coverage_load.py` (line 522 for `push_agenda_constraints`).
- New tests should follow the same class/fixture patterns.

### Key files

| File | Role |
|------|------|
| `custom_components/quiet_solar/home_model/load.py` | Both fixes (set_live_constraints, push_live_constraint) |
| `tests/test_load_model.py` | Primary test file for set_live_constraints and push_live_constraint |
| `tests/test_coverage_load.py` | Additional coverage tests for push_agenda_constraints |
| `tests/factories.py` | Test factories -- use `MinimalTestLoad`, `create_constraint()` |

### References

- `home_model/load.py#set_live_constraints` — met-constraint filter (`is_constraint_met` list comprehension)
- `home_model/load.py#push_live_constraint` — dead-on-arrival return (`return False, True` after `is_constraint_met` check)
- `home_model/load.py#push_live_constraint` — correct `return True, True` (constraint replaced via `self._constraints[i] = None`)
- `home_model/load.py#update_live_constraints` — `len(constraints) != len(self._constraints)` check that triggers `force_solving`
- `home_model/load.py#push_agenda_constraints` — `res = pushed or res` aggregation loop

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References

### Completion Notes List
- Fix 1: Added already-completed constraint filter in `set_live_constraints` (load.py:1293-1311)
- Fix 2: Changed dead-on-arrival return from `(True, True)` to `(False, True)` in `push_live_constraint` (load.py:1385)
- Updated 3 existing tests that depended on old `(True, True)` return semantics
- Added 7 new tests covering both fix paths and the integration loop break

### File List
- `custom_components/quiet_solar/home_model/load.py` — both fixes
- `tests/test_load_model.py` — new tests + updated existing tests
- `tests/test_charger_coverage_deep.py` — updated assertion for dead-on-arrival behavior
