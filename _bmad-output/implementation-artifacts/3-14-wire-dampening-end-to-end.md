# Story 3.14: Solar Forecast Scoring and Provider Auto-Selection

Status: in-progress

## Story

As TheAdmin,
I want the solar forecast scoring system to record real production data, compute meaningful accuracy scores per provider, persist them across restarts, and show me which provider is currently active,
so that auto-provider selection uses real accuracy data and I can monitor forecast quality.

## Context

Story 3.7 built the full solar forecast infrastructure including multi-provider support, staleness detection, health monitoring, scoring, and dampening (MOS linear regression). After real-world testing, **solar forecast dampening was removed entirely** — the MOS correction added complexity without sufficient benefit for the use case. This story:

1. Removes all solar forecast dampening code, sensors, switches, and tests
2. Fixes the scoring infrastructure to work correctly end-to-end (actuals recording, forecast snapshots, score computation)
3. Refactors time series utilities into `home_utils.py` for reuse

GitHub Issue: #43

## Scope Change from Original

The original story title was "Wire Dampening End-to-End with Resolution-Independent Storage". During implementation, the decision was made to **remove solar forecast dampening entirely** rather than wire it up. The scoring infrastructure (MAE computation, provider ranking, auto-selection) is kept and improved. Load dampening (unrelated) is untouched.

## Acceptance Criteria

### Time Series Utils Refactoring

1. **Given** the functions `align_time_series_and_values`, `get_slots_from_time_series`, `get_value_from_time_series` currently live in `home_model/load.py`
   **When** the refactoring is complete
   **Then** they are in `home_model/home_utils.py` alongside `get_average_time_series`
   **And** `load.py` re-exports them for backward compatibility
   **And** all existing imports from `home.py`, `solar.py`, and `load.py` still work

2. **Given** `PeriodSolver._power_slot_from_forecast` is a private method on the solver
   **When** the refactoring is complete
   **Then** it exists as `slot_value_from_time_series` in `home_model/home_utils.py` as a standalone function
   **And** `PeriodSolver` calls the new function instead of its private method
   **And** the function signature is self-contained (no `self` dependency)

3. **Given** the new `align_time_series_on_time_slots` function in `home_utils.py`
   **When** called with a time series and a list of time slot boundaries (like solver anchors)
   **Then** it loops over consecutive slot pairs, calls `slot_value_from_time_series` for each, and returns an aligned time series `list[tuple[datetime, float]]` with one value per slot
   **And** `PeriodSolver.create_power_slots` keeps its own loop (it does tariff lookup and other work around the forecast calls) but calls `slot_value_from_time_series` instead of `_power_slot_from_forecast`

4. **Given** all refactored time series utils
   **When** the test suite runs
   **Then** there is a thorough, dedicated test suite covering `slot_value_from_time_series`, `align_time_series_on_time_slots`, `get_average_time_series`, `align_time_series_and_values`, `get_slots_from_time_series`, `get_value_from_time_series` — including edge cases (empty series, single point, gaps, boundary alignment, resolution mismatch)

### Scoring Infrastructure (Dampening Removed)

5. **Given** the system records actual solar production and forecast history
   **When** `QSHomeSolarAndConsumptionHistoryAndForecast` updates each cycle
   **Then** actual solar production is stored in `solar_production_history` (a `QSSolarHistoryVals` ring buffer at 15-min intervals, 560 days)
   **And** forecast values at multiple lookahead intervals (15mn, 1h, 4h, 8h, 12h, 18h, 24h) are recorded via `QSforecastValueSensor` probers into per-sensor and per-provider `QSSolarHistoryVals` ring buffers
   **And** this data is shared across all providers via the home's forecast handler

6. **Given** the scoring cycle runs (at 00:00 and 12:00 local time)
   **When** `compute_score(time)` is called on each provider
   **Then** it retrieves the last 24h of actual production from `solar_production_history.get_historical_data(time, past_hours=24)`
   **And** it retrieves the last 24h of stored forecast history from the provider's forecast history ring buffers (preferring >= 8h lookahead, falling back to the largest available)
   **And** it computes MAE (Mean Absolute Error) over daytime slots (where forecast or actual > 0)
   **And** the score is stored as `provider.score`

7. **Given** the solar plant uses provider auto-selection
   **When** the scoring cycle completes
   **Then** `auto_select_best_provider` picks the provider with the lowest MAE score
   **And** this works regardless of whether the user has selected "auto" mode or a specific provider

### Dampening Removal

9. **Given** the solar forecast system
   **When** the codebase is inspected
   **Then** there are NO dampening coefficients, NO MOS regression computation, NO dampening enable/disable switches
   **And** `get_forecast` and `get_value_from_current_forecast` use `self.solar_forecast` directly (no `_get_effective_forecast`)
   **And** `QSSolarProvider` has no `dampening_enabled`, `_dampening_coefficients`, `_dampening_coefficients_per_tod`, `score_dampened` fields
   **And** no `Platform.SWITCH` is registered for solar providers

### Sensor Changes

10. **Given** a solar forecast provider
    **When** sensors are registered
    **Then** there is a single "Forecast Score (ProviderName)" sensor per provider using `SENSOR_SOLAR_FORECAST_SCORE_PREFIX`
    **And** its value is the MAE score (`provider.score`)
    **And** the old "Dampened Score" and "No Dampening Score" sensors are removed

11. **Given** the solar plant uses provider auto-selection
    **When** sensors are registered
    **Then** a sensor shows the name of the currently active solar provider

### Dashboard

12. **Given** the solar section of the dashboard
    **When** the dashboard is rendered
    **Then** the score sensor and active provider sensor are displayed
    **And** there are no dampening switch entries
    **And** the old dampened/no-dampening score sensor entries are removed

### Persistence

13. **Given** HA restarts
    **When** the solar provider initializes
    **Then** the `QSSolarHistoryVals` ring buffers (production history + forecast histories) are loaded from their numpy files on disk
    **And** scores start as `None` and are recomputed on the next scoring cycle (00:00 or 12:00) using the persisted ring buffer data
    **And** no separate scoring persistence is needed — the ring buffers ARE the persistence

14. **Given** the user presses "Full Reset Quiet Solar History from DB"
    **When** the reset completes
    **Then** the scoring state is cleared via `reset_scoring()` and ring buffers are cleared and re-bootstrapped via `solar_forecast_set_and_reset(for_reset=True)`

## Tasks / Subtasks

- [x] **Task 1: Move time series utils to home_utils.py** (AC: 1)
  - [x] 1.1 Move `align_time_series_and_values`, `get_slots_from_time_series`, `get_value_from_time_series` from `load.py` to `home_utils.py`
  - [x] 1.2 Add re-exports in `load.py` for backward compatibility
  - [x] 1.3 Update imports in `ha_model/solar.py` and `ha_model/home.py` to import from `home_model.home_utils`

- [x] **Task 2: Extract `_power_slot_from_forecast` to `home_utils.py`** (AC: 2)
  - [x] 2.1 Create `slot_value_from_time_series` in `home_utils.py` — same logic but standalone (no `self`)
  - [x] 2.2 Replace `PeriodSolver._power_slot_from_forecast` with a call to the new function

- [x] **Task 3: Create `align_time_series_on_time_slots`** (AC: 3)
  - [x] 3.1 Implement in `home_utils.py`
  - [x] 3.2 `PeriodSolver.create_power_slots` calls `slot_value_from_time_series` instead of `_power_slot_from_forecast`

- [x] **Task 4: Thorough test suite for all time series utils** (AC: 4)
  - [x] 4.1 Tests for `slot_value_from_time_series`
  - [x] 4.2 Tests for `align_time_series_on_time_slots`
  - [x] 4.3 Tests for moved functions — verify they still work from both import paths
  - [x] 4.4 Verify `PeriodSolver.create_power_slots` still produces identical results after refactor

- [x] **Task 5: Wire actuals + forecast history recording** (AC: 5)
  - [x] 5.1 `QSHomeSolarAndConsumptionHistoryAndForecast` records solar production into `solar_production_history` ring buffer
  - [x] 5.2 `QSforecastValueSensor` probers sample forecast at multiple lookahead intervals (15mn–24h)
  - [x] 5.3 Per-sensor and per-provider forecast history stored in `QSSolarHistoryVals` ring buffers
  - [x] 5.4 `solar_forecast_set_and_reset` creates/resets all ring buffers

- [x] **Task 6: Wire scoring cycle** (AC: 6)
  - [x] 6.1 `_run_scoring_cycle` runs at 00:00 and 12:00 local time
  - [x] 6.2 `compute_score(time)` fetches 24h actuals and 24h forecast history from ring buffers
  - [x] 6.3 MAE computed over daytime slots, preferring >= 8h lookahead forecast sensor

- [x] **Task 7: Remove all solar forecast dampening** (AC: 7, 9)
  - [x] 7.1 Remove from `QSSolarProvider.__init__`: `dampening_enabled`, `_dampening_coefficients`, `_dampening_coefficients_per_tod`, `_last_dampening_date`, `score_dampened`
  - [x] 7.2 Remove methods: `compute_dampening`, `_apply_dampening`, `_get_effective_forecast`, `get_dampening_attributes`, `_group_by_day`
  - [x] 7.3 Rename `score_raw` → `score`, `reset_dampening_buffers` → `reset_scoring_buffers`, `reset_dampening` → `reset_scoring`
  - [x] 7.4 Simplify `get_active_score` to return `self.score` directly
  - [x] 7.5 Simplify `get_forecast` and `get_value_from_current_forecast` to use `self.solar_forecast` directly
  - [x] 7.6 Remove `Platform.SWITCH` from `get_platforms()`
  - [x] 7.7 Rename constants: `DAMPENING_*` → `SCORING_*`

- [x] **Task 8: Update sensors** (AC: 10, 11)
  - [x] 8.1 Replace two per-provider sensors (dampened + no-dampening) with single "Forecast Score" sensor using `SENSOR_SOLAR_FORECAST_SCORE_PREFIX`
  - [x] 8.2 Keep active provider sensor
  - [x] 8.3 Remove `SENSOR_SOLAR_FORECAST_SCORE_DAMPENED_PREFIX` and `SENSOR_SOLAR_FORECAST_SCORE_NO_DAMPENING_PREFIX` from const.py

- [x] **Task 9: Update switches** (AC: 9)
  - [x] 9.1 Remove `create_ha_switch_for_QSSolar` entirely
  - [x] 9.2 Remove `SWITCH_SOLAR_DAMPENING_PREFIX` from const.py and switch.py

- [x] **Task 10: Dashboard updates** (AC: 12)
  - [x] 10.1 Remove dampening switch block from both dashboard templates
  - [x] 10.2 Replace dampened/no-dampening score with single score + active_provider
  - [x] 10.3 Update both `quiet_solar_dashboard_template.yaml.j2` and `quiet_solar_dashboard_template_standard_ha.yaml.j2`

- [x] **Task 11: Update tests** (AC: all)
  - [x] 11.1 Remove all dampening test classes: TestDampening, TestGetDampeningAttributes, TestRealisticDampeningScenarios (~31 tests)
  - [x] 11.2 Rename: `score_raw` → `score`, `DAMPENING_*` → `SCORING_*`, `reset_dampening` → `reset_scoring`
  - [x] 11.3 Update sensor tests: single score sensor per provider
  - [x] 11.4 Update dashboard tests: no dampening switch/score assertions
  - [x] 11.5 Update home.py test: `reset_dampening` → `reset_scoring`
  - [x] 11.6 Maintain 100% coverage

- [x] **Task 12: Update story artifacts** (AC: N/A)
  - [x] 12.1 Rewrite this story (3.14) to reflect scope change
  - [x] 12.2 Add note to story 3.7 that AC4 (dampening) was later removed

## Dev Notes

### Dampening Removal Rationale

After real-world testing, MOS linear regression dampening for solar forecasts added complexity without sufficient benefit:
- Solar forecast providers (Solcast, Open-Meteo) already apply their own corrections
- The MOS approach needed weeks of data accumulation before being useful
- Simple MAE scoring + auto-provider selection achieves the goal of using the best forecast
- Load dampening (separate system) remains untouched — it serves a different purpose

### Architecture

- **Two-layer storage**: `QSSolarHistoryVals` ring buffers (numpy, 15-min intervals, 560 days) persist production actuals and forecast history per-sensor and per-provider
- **Probers**: `QSforecastValueSensor` samples forecast values at 7 lookahead intervals (15mn, 1h, 4h, 8h, 12h, 18h, 24h) into ring buffers
- **Scoring**: `compute_score(time)` on `QSSolarProvider` fetches 24h actuals + 24h forecast history from ring buffers, computes MAE over daytime slots
- **Persistence**: Ring buffers saved as numpy files — no separate scoring persistence needed
- Constants renamed from `DAMPENING_*` to `SCORING_*` for remaining infrastructure
- `home.py` calls `reset_scoring()` instead of `reset_dampening()`
- Circular import between `solar.py` and `home.py` resolved with deferred import in `QSSolar.__init__`

### Quality Gates

- 4202 tests pass
- 100% coverage
- ruff check clean
- ruff format clean
- mypy clean
- translations valid

### References

- [Source: ha_model/solar.py] — scoring infrastructure, actuals recording, forecast snapshots
- [Source: sensor.py] — single score sensor per provider + active provider sensor
- [Source: switch.py] — dampening switches removed
- [Source: const.py] — SCORING_* constants, removed DAMPENING_* constants
- [Source: ui/quiet_solar_dashboard_template*.yaml.j2] — updated dashboard templates
- [Source: tests/test_solar_forecast_resilience.py] — updated tests
- [Source: _bmad-output/implementation-artifacts/3-7-fm-001-solar-forecast-api-resilience.md] — original dampening story

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None — quality gates passed after implementation.

### Completion Notes List
- All dampening code, sensors, switches, and tests removed
- Scoring infrastructure wired end-to-end: probers-based forecast history → ring buffer storage → MAE scoring at 00:00/12:00
- Production actuals + forecast history stored in `QSSolarHistoryVals` ring buffers (numpy, 15-min, 560 days)
- `QSforecastValueSensor` probers sample forecast at 7 lookahead intervals per sensor per provider
- `compute_score(time)` fetches 24h actuals + 24h forecast from ring buffers, computes MAE over daytime slots
- Code review found and fixed: circular import, 10+ bugs (None guards, wrong attribute chains, dict iteration, missing args), behavioral regression in `get_slots_from_time_series`
- 4202 tests, 100% coverage, all quality gates pass

### File List
- `custom_components/quiet_solar/ha_model/solar.py` — Removed dampening, wired scoring with ring buffer history, compute_score(time)
- `custom_components/quiet_solar/ha_model/home.py` — Forecast history ring buffers, probers, solar_forecast_set_and_reset, reset_scoring
- `custom_components/quiet_solar/home_model/home_utils.py` — Moved time series utils, new slot_value_from_time_series, align_time_series_on_time_slots
- `custom_components/quiet_solar/home_model/load.py` — Re-exports for backward compatibility
- `custom_components/quiet_solar/home_model/solver.py` — Uses slot_value_from_time_series, None guards
- `custom_components/quiet_solar/const.py` — Removed dampening constants, renamed to SCORING_*, single score prefix
- `custom_components/quiet_solar/sensor.py` — Single score sensor per provider, active provider sensor
- `custom_components/quiet_solar/switch.py` — Removed solar dampening switches
- `tests/test_solar_forecast_scoring.py` — 34 new tests for scoring, forecast history, probers
- `tests/test_solar_forecast_resilience.py` — Rewritten for new compute_score(time) API
- `tests/test_time_series_utils.py` — 47+ tests for all time series utils
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2` — Removed dampening UI
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` — Removed dampening UI
- `tests/test_solar_forecast_resilience.py` — Removed 31 dampening tests, updated scoring references
- `tests/test_dashboard_rendering.py` — Updated sensor assertions
- `tests/ha_tests/test_home_misc.py` — reset_dampening → reset_scoring
- `tests/test_solar_forecast_solar_provider.py` — Updated attribute checks
