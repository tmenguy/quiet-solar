# Bug Fix: Pool target offset by already-run hours when slider is adjusted mid-run (regression)

Status: in-progress
issue: 101
branch: "QS_101"

## Story

As a pool owner using Quiet Solar,
I want the target hours slider to set an absolute daily target,
so that adjusting it while the pool is running shows the value I set, not an inflated number.

## Bug Description

Regression of #95. When the pool is running and the user adjusts the target hours slider (ring handle), the new target is offset by the hours already run instead of being set to the slider value directly.

**Steps to reproduce:**
1. Pool is running (ON), has run for 8h, target is 12h
2. Move the target ring handle to 10h → target sensor shows 18h (10 + 8)
3. Move handle to 4h → target shows 12h (4 + 8)
4. Move handle to 0h → target shows 8h (0 + 8)

**Key user detail:** Bug occurs specifically while the pool is ON (running).

**Expected:** `qs_bistate_current_duration_h` sensor = handle position value (absolute target).
**Actual:** `qs_bistate_current_duration_h` = handle_value + already_run_hours.

**Related issues:** #95 (original fix), #78 (daily target/actual metrics), #80 (calendar precompute refactor), #68 (carry-from-completed constraint).

## Root Cause Analysis

### Confirmed root cause: day-rollover lcc double-count

GitHub #95 fixed same-end-date and yesterday-active leaks in `update_current_metrics`, but a **day-rollover case** remains.

`update_current_metrics()` default path sums:
- All **active** constraints with `today_utc < end <= tomorrow_utc`
- `_last_completed_constraint` (lcc) in the same window when not `already_absorbed`

`already_absorbed` (lines 142-147) is true only if some active constraint shares `end_of_constraint` (or `initial_end_of_constraint`) with lcc.

For pools with `default_on_finish_time = 00:00` (common setup):
- **Yesterday's** completed constraint ends at **local midnight** → `lcc.end_of_constraint == today_utc`
- **Today's** active constraint ends at **next local midnight** → `active.end_of_constraint == tomorrow_utc`

Those two instants are **never equal**, so `already_absorbed` is false while **both** pass the day-window checks:
- lcc: `end >= today_utc` (midnight >= midnight) ✓ AND `end <= tomorrow_utc` ✓
- active: `end > today_utc` ✓ AND `end <= tomorrow_utc` ✓

Result: `qs_bistate_current_duration_h = lcc.target + active.target` and `qs_bistate_current_on_h = lcc.current + active.current`.

This matches the user description of `target = new_target + current` when the lcc's values line up numerically with the already-run hours.

### Why #95 didn't catch this

The #95 fix assumed "same cycle" implies **identical end timestamps**. The day-rollover case has lcc ending at `today_utc` and the active constraint ending at `tomorrow_utc` — different timestamps but representing consecutive daily cycles of the same pool.

### Status of #95 fixes

All three fixes are still in place and correct for their intended scenarios:
1. Day lower bound on active constraints loop (line 132): `ct.end_of_constraint > today_utc` ✓
2. `already_absorbed` guard for lcc (lines 142-147) ✓
3. `>=` boundary for lcc (line 151) ✓

## Acceptance Criteria

1. **AC1**: When user adjusts ring handle to X hours while pool is running, `qs_bistate_current_duration_h` = X (not X + lcc.target)
2. **AC2**: `qs_bistate_current_on_h` (actual run hours) = only today's active constraint runtime (not + lcc.current)
3. **AC3**: Intra-day multi-segment metrics still work: lcc completed at 08:00 + active until 17:00 → both counted (lcc.end ≠ today_utc rollover boundary)
4. **AC4**: Existing tests pass: `test_pool_update_current_metrics_completed_and_active_sums_both` (different end dates, both counted)
5. **AC5**: Existing midnight-boundary test passes: `test_default_mode_exact_midnight_lcc_included`
6. **AC6**: 100% test coverage maintained

## Fix Plan

### Proposed fix (metrics only, localized to `update_current_metrics`)

**File:** `custom_components/quiet_solar/ha_model/bistate_duration.py`

Extend the "skip lcc" logic alongside existing `already_absorbed` so that lcc is **not counted** when it represents the **previous local day's rollover** and a same-type active constraint represents today's cycle:

**Condition:** `lcc.end_of_constraint == today_utc` (exact rollover boundary) AND there exists an active constraint of `type(ct) == type(lcc)` with `end_of_constraint > today_utc` in the today/tomorrow window.

**Effect:** Metrics show only today's target/actual for the running constraint, while intra-day "completed at 08:00 + active until 17:00" cases stay valid because lcc.end is NOT the `today_utc` rollover boundary.

Keep the change localized to `update_current_metrics` (per project boundary: no HA imports in `home_model/load.py`).

## Tasks / Subtasks

- [ ] Task 1: Implement rollover skip in `update_current_metrics` default path (AC: 1, 2, 3)
  - [ ] 1.1: Next to the existing `already_absorbed` check (~line 142), add a `rollover_from_previous_day` guard: skip lcc when `lcc.end_of_constraint == today_utc` AND an active same-type constraint exists in the day window
  - [ ] 1.2: Combine with existing `already_absorbed` as: `if not already_absorbed and not rollover_from_previous_day:`

- [ ] Task 2: Add regression test — lcc.end==today_utc + active.end==tomorrow_utc (AC: 1, 2, 3)
  - [ ] 2.1: In `tests/test_ha_pool.py` (or `test_bug_78_daily_metrics.py`): lcc with `end_of_constraint = today_utc`, non-zero target/current (yesterday's completed cycle). Active constraint with `end_of_constraint = tomorrow_utc`, target = slider value, current = some runtime.
  - [ ] 2.2: Assert `qs_bistate_current_duration_h == active.target / 3600` and `qs_bistate_current_on_h == active.current / 3600` (lcc NOT included)

- [ ] Task 3: Verify existing tests pass — no regression on intra-day cases (AC: 3, 4, 5)
  - [ ] 3.1: `test_pool_update_current_metrics_completed_and_active_sums_both` — lcc at 08:00, active at 17:00, both counted (lcc.end ≠ today_utc)
  - [ ] 3.2: `test_default_mode_exact_midnight_lcc_included` — lcc at exact midnight, no active → lcc counted
  - [ ] 3.3: All 48 existing regression tests pass

- [ ] Task 4: Run full quality gate (AC: 6)
  - [ ] 4.1: `python scripts/qs/quality_gate.py`

## Dev Notes

### Key Files

| File | Lines | Role |
|------|-------|------|
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 127-158 | `update_current_metrics()` default path — **fix goes here** |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 142-147 | `already_absorbed` guard — rollover guard added alongside |
| `tests/test_ha_pool.py` | all | Pool regression tests — new test added here |
| `tests/test_bug_78_daily_metrics.py` | all | Bistate daily metrics tests — verify no regression |

### Architecture Constraints

- **Two-layer boundary**: `home_model/load.py` NEVER imports `homeassistant.*`. Fix stays in `ha_model/bistate_duration.py`.
- Do NOT modify `push_live_constraint()`, constraint classes, or solver.
- Lazy logging: `_LOGGER.debug("msg %s", var)` — no f-strings in log calls.

### Risk Note

Tie the skip to `lcc.end == today_utc` (and same constraint type) so multi-segment same-day metrics from different wall-clock ends remain unchanged. The exact `today_utc` comparison avoids catching intra-day completions (e.g., lcc.end = 08:00 UTC).

### Constraint Lifecycle Reference

1. **Created**: `TimeBasedSimplePowerLoadConstraint(target_value=N, initial_value=0)`
2. **Pushed**: `push_live_constraint()` — carry-over and replacement logic
3. **Updated**: `update_live_constraints()` — increments `current_value`
4. **Completed**: `ack_completed_constraint()` sets `_last_completed_constraint`
5. **Metrics**: `update_current_metrics()` sums active + lcc for today — **double-count bug is here**

### Test Infrastructure

- Use `MinimalTestHome` / `MinimalTestLoad` from `tests/factories.py`
- Use `create_constraint()` factory with `TimeBasedSimplePowerLoadConstraint`
- Use `freezegun` for time-dependent scenarios
- asyncio_mode=auto — no `@pytest.mark.asyncio` decorator needed

### Related Bug Story References

- `bug-Github-#95-pool-target-offset-slider-mid-run.md` — original fix (three bugs in `update_current_metrics`)
- `bug-Github-#78-fix-bistate-daily-target-actual-hours.md` — daily target/actual metrics
- `bug-Github-#68-carry-from-completed-constraint.md` — carry-over logic for completed constraints

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Story created from GitHub issue #101 (regression of #95)
- Root cause identified via Cursor plan: day-rollover lcc double-count — lcc.end == today_utc while active.end == tomorrow_utc, so `already_absorbed` never fires
- Confirmed #95 fix is still in place but doesn't cover this cross-day boundary case
- Fix is localized to `update_current_metrics()` default path — add rollover guard alongside `already_absorbed`
- All 48 existing regression tests pass (22 pool, 26 bistate daily metrics)

### File List

- `custom_components/quiet_solar/ha_model/bistate_duration.py` — fix in `update_current_metrics()` default path
- `tests/test_ha_pool.py` — new regression test for day-rollover lcc
