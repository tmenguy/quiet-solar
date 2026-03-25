# Story 3.14: Wire Dampening End-to-End with Resolution-Independent Storage

Status: ready-for-dev

## Story

As TheAdmin,
I want the solar forecast dampening system to actually work — recording real production data, computing meaningful scores and dampening coefficients per time-of-day, persisting them across restarts, and showing me detailed accuracy metrics,
so that forecast accuracy improves over time, I can monitor dampening behavior, and auto-provider selection uses real accuracy data.

## Context

Story 3.7 built the dampening infrastructure (scoring, MOS regression, persistence methods, dampening application). All the methods exist and are tested in isolation. But **none of the data-feeding or persistence wiring was ever connected in production code**, and the storage design is coupled to provider step size. This story completes what 3.7 started with a better architecture.

GitHub Issue: #43

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

### Dampening Storage

5. **Given** the system records actual solar production
   **When** each forecast update cycle runs
   **Then** actual production is stored as a continuous rolling time series in 5-minute slots (using the new alignment utils to downsample from raw sensor updates)
   **And** the rolling buffer covers 8 days
   **And** on first startup, `get_historical_solar_fallback` bootstraps the buffer from HA sensor history (15-min granularity is acceptable)

6. **Given** the system records forecast snapshots
   **When** a new forecast is received from a provider
   **Then** the forecast is stored in one of 12 rolling buffers indexed by 2-hour time-of-day slots (00:00, 02:00, 04:00, ..., 22:00)
   **And** each buffer holds 8 days of forecast time series
   **And** the buffer chosen is the one matching the current 2-hour window (e.g., forecast at 08:45 goes into the 08:00 buffer)

7. **Given** dampening is computed for a forecast at time T
   **When** the system looks up the appropriate history
   **Then** it uses the forecast rolling buffer closest to T's 2-hour slot (e.g., 08:00 buffer for T=08:45)
   **And** it shifts past days' timestamps forward to align with today's forecast timing
   **And** it uses `align_time_series_on_time_slots` to align both stored forecasts and stored actuals onto the current forecast's time slots
   **And** it computes per-slot dampening coefficients via MOS linear regression on the aligned data

8. **Given** a new forecast arrives for a provider
   **When** dampening is applied
   **Then** dampening coefficients are recomputed from scratch using the current forecast's timing and the matching 2-hour history buffer — there is no coefficient caching between forecasts
   **And** the computed coefficients are stored only for the "Dampened Score" sensor attributes (monitoring)

9. **Given** a provider's forecast resolution changes (e.g., 30min to 1h)
   **When** dampening is computed
   **Then** previously stored raw history is re-aligned to the new resolution on-the-fly (not discarded)
   **And** this works naturally because coefficients are always recomputed from raw history, never cached

### Persistence

10. **Given** `compute_dampening()` produces new coefficients
    **When** the computation completes
    **Then** coefficients and the rolling buffers (actuals + forecast history) are persisted to disk

11. **Given** HA restarts
    **When** the solar provider initializes
    **Then** dampening coefficients and rolling buffers are loaded from disk
    **And** dampening is applied immediately without needing fresh data
    **And** if the persisted format is incompatible (old format), it is discarded gracefully

12. **Given** the user presses "Full Reset Quiet Solar History from DB"
    **When** the reset completes
    **Then** the dampening rolling buffers (actuals + forecasts) are also cleared and re-bootstrapped from HA sensor history where possible

### Scoring and Dampening Always Computed

13. **Given** any provider, regardless of the dampening switch state
    **When** the daily scoring cycle runs
    **Then** both raw score (forecast vs actual) and dampened score (dampened forecast vs actual) are always computed
    **And** dampening coefficients are always computed
    **And** the dampening switch (`switch.qs_solar_dampening_<provider>`) only controls whether the dampened forecast is used by the rest of Quiet Solar — it does NOT gate computation

### Sensor Changes

14. **Given** a solar forecast provider
    **When** sensors are registered
    **Then** the sensor currently named "Score (ProviderName)" is renamed to "Dampened Score (ProviderName)"
    **And** its value is the MAE score of the dampened forecast vs actuals
    **And** its attributes contain the dampening coefficients for the current day and for each of the 12 two-hour slots (for monitoring)
    **And** a new sensor "No Dampening Score (ProviderName)" is added with value = MAE score of raw forecast vs actuals
    **And** the existing "Raw Score" and "Dampened Score" sensors from Story 3.7 are replaced by the above two

15. **Given** the solar plant uses provider auto-selection
    **When** sensors are registered
    **Then** a new sensor shows the name of the currently active solar provider (useful in auto mode)

### Dashboard

16. **Given** the solar section of the dashboard
    **When** the dashboard is rendered
    **Then** the new "Dampened Score", "No Dampening Score", and active provider sensors are displayed
    **And** the old score sensor entries are removed

## Tasks / Subtasks

- [ ] **Task 1: Move time series utils to home_utils.py** (AC: 1)
  - [ ] 1.1 Move `align_time_series_and_values`, `get_slots_from_time_series`, `get_value_from_time_series` from `load.py` to `home_utils.py`
  - [ ] 1.2 Add re-exports in `load.py` for backward compatibility
  - [ ] 1.3 Update imports in `ha_model/solar.py` (lines 41-43) and `ha_model/home.py` (line 81) to import from `home_model.home_utils`

- [ ] **Task 2: Extract `_power_slot_from_forecast` to `home_utils.py`** (AC: 2)
  - [ ] 2.1 Create `slot_value_from_time_series(forecast, begin_slot, end_slot, last_end, geometric_smoothing=False)` in `home_utils.py` — same logic as `_power_slot_from_forecast` (solver.py:224-264) but standalone (no `self`)
  - [ ] 2.2 Replace `PeriodSolver._power_slot_from_forecast` with a call to the new function

- [ ] **Task 3: Create `align_time_series_on_time_slots`** (AC: 3)
  - [ ] 3.1 Implement in `home_utils.py`: given a time series and a list of time slot boundaries (anchors), loop over consecutive pairs calling `slot_value_from_time_series`, return `list[tuple[datetime, float]]`
  - [ ] 3.2 `PeriodSolver.create_power_slots` keeps its loop but calls `slot_value_from_time_series` instead of `_power_slot_from_forecast`

- [ ] **Task 4: Thorough test suite for all time series utils** (AC: 4)
  - [ ] 4.1 Tests for `slot_value_from_time_series`: empty forecast, single point, exact boundaries, interpolation, geometric smoothing, gaps
  - [ ] 4.2 Tests for `align_time_series_on_time_slots`: multiple slots, single slot, misaligned boundaries, different resolutions
  - [ ] 4.3 Tests for moved functions: verify they still work identically from both import paths
  - [ ] 4.4 Verify `PeriodSolver.create_power_slots` still produces identical results after refactor

- [ ] **Task 5: Implement rolling buffer storage for actuals** (AC: 5, 12)
  - [ ] 5.1 Replace `_actual_history: np.ndarray` with a continuous rolling time series in 5-min slots, covering 8 days
  - [ ] 5.2 On each forecast update cycle, record actual production from the inverter sensor using `align_time_series_on_time_slots` to downsample to 5-min
  - [ ] 5.3 On first startup (empty buffer), bootstrap from `get_historical_solar_fallback` (15-min granularity OK)
  - [ ] 5.4 Wire into "Full Reset" (`reset_forecasts`) to clear and re-bootstrap

- [ ] **Task 6: Implement 12x rolling buffer storage for forecast snapshots** (AC: 6, 8)
  - [ ] 6.1 Replace `_forecast_history: np.ndarray` with 12 rolling buffers indexed by 2-hour slots (00:00, 02:00, ..., 22:00), each covering 8 days of forecast time series
  - [ ] 6.2 On each forecast update, store the full forecast in the buffer matching the current 2-hour window
  - [ ] 6.3 No coefficient caching — recompute dampening from raw history each time a forecast is applied; store computed coefficients only as sensor attributes for monitoring
  - [ ] 6.4 Wire into "Full Reset" to clear all buffers

- [ ] **Task 7: Rewrite dampening computation with on-the-fly alignment** (AC: 7, 9, 13)
  - [ ] 7.1 When computing dampening for a given forecast: select the matching 2-hour forecast buffer, shift past timestamps to align with today, use `align_time_series_on_time_slots` to project both forecast history and actuals onto the current forecast's time slots
  - [ ] 7.2 Compute per-slot MOS linear regression on the aligned data (same algorithm as current `compute_dampening` but on aligned data)
  - [ ] 7.3 Compute both raw and dampened scores always, regardless of dampening switch state
  - [ ] 7.4 Remove the `if provider.dampening_enabled` gate from `_run_daily_scoring_cycle` around `compute_dampening` — always compute, only gate application

- [ ] **Task 8: Update persistence** (AC: 10, 11)
  - [ ] 8.1 Update `save_dampening()` to persist: dampening coefficients per 2-hour slot, actuals rolling buffer, forecast rolling buffers
  - [ ] 8.2 Update `load_dampening()` to restore all of the above
  - [ ] 8.3 Handle old format gracefully (discard and start fresh)

- [ ] **Task 9: Sensor changes** (AC: 14, 15)
  - [ ] 9.1 Rename "Score (ProviderName)" sensor to "Dampened Score (ProviderName)" — value = dampened MAE score; attributes = dampening coefficients (daily summary + per 2-hour slot values)
  - [ ] 9.2 Add "No Dampening Score (ProviderName)" sensor — value = raw MAE score
  - [ ] 9.3 Remove the old "Raw Score" and "Dampened Score" sensors (replaced by the above)
  - [ ] 9.4 Add "Active Solar Provider" sensor on the solar plant — value = name of currently active provider
  - [ ] 9.5 Update `strings.json` and run `bash scripts/generate-translations.sh`

- [ ] **Task 10: Dashboard updates** (AC: 16)
  - [ ] 10.1 Update solar section in `ui/quiet_solar_dashboard_template.yaml.j2` to display new sensors
  - [ ] 10.2 Update `ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` similarly
  - [ ] 10.3 Remove references to old score sensor prefixes

- [ ] **Task 11: Tests** (AC: all)
  - [ ] 11.1 Update all existing dampening/scoring tests for new storage format
  - [ ] 11.2 Test: actuals recording on forecast update, including 5-min downsampling
  - [ ] 11.3 Test: forecast snapshot stored in correct 2-hour buffer
  - [ ] 11.4 Test: dampening computation uses correct buffer and alignment
  - [ ] 11.5 Test: coefficients recomputed on each new forecast (no stale coefficients)
  - [ ] 11.6 Test: resolution change preserves history
  - [ ] 11.7 Test: persistence round-trip (save, restart, load, apply)
  - [ ] 11.8 Test: bootstrap from `get_historical_solar_fallback`
  - [ ] 11.9 Test: full reset clears dampening buffers
  - [ ] 11.10 Test: scores always computed regardless of switch state
  - [ ] 11.11 Test: new sensors return correct values and attributes
  - [ ] 11.12 Maintain 100% coverage

## Dev Notes

### Time Series Utils Consolidation

Currently `home_utils.py` has `get_average_time_series` while `load.py` has 3 other time series functions. After this refactor, all time series utilities live in `home_utils.py`:

| Function | Current Location | New Location |
|----------|-----------------|-------------|
| `get_average_time_series` | `home_model/home_utils.py:78` | stays |
| `align_time_series_and_values` | `home_model/load.py:1578` | `home_model/home_utils.py` |
| `get_slots_from_time_series` | `home_model/load.py:1722` | `home_model/home_utils.py` |
| `get_value_from_time_series` | `home_model/load.py:1748` | `home_model/home_utils.py` |
| `slot_value_from_time_series` | `home_model/solver.py:224` (as `_power_slot_from_forecast`) | `home_model/home_utils.py` |
| `align_time_series_on_time_slots` | NEW | `home_model/home_utils.py` |

**Imports to update:**
- `ha_model/solar.py` lines 41-43: imports all 3 from `load.py`
- `ha_model/home.py` line 81: imports `get_value_from_time_series` from `load.py`
- `load.py` itself: uses all 3 internally — add re-exports
- `home_model/solver.py`: already imports `get_average_time_series` from `home_utils`; add `slot_value_from_time_series`, `align_time_series_on_time_slots`

### `align_time_series_on_time_slots` Design

Pattern from `create_power_slots` (solver.py:179-222):
```python
# Current pattern in solver:
for i in range(len(anchors) - 1):
    begin_slot = anchors[i]
    end_slot = anchors[i + 1]
    i_ua, ua_power = self._power_slot_from_forecast(forecast, begin_slot, end_slot, i_ua)
```

New generic function:
```python
def align_time_series_on_time_slots(
    time_series: list[tuple[datetime, float]],
    slot_boundaries: list[datetime],
    geometric_smoothing: bool = False,
) -> list[tuple[datetime, float]]:
    """Align a time series onto time slots, returning one averaged value per slot."""
```

### Forecast Rolling Buffers: 12 x 2-Hour Slots

Some providers update past forecast values during the day to be closer to actuals. To capture the forecast *as it was* at different times of day, we store 12 independent rolling buffers:

| Buffer Index | Time-of-Day | Stores forecasts received between |
|-------------|-------------|----------------------------------|
| 0 | 00:00 | 23:00 - 01:00 |
| 1 | 02:00 | 01:00 - 03:00 |
| ... | ... | ... |
| 11 | 22:00 | 21:00 - 23:00 |

Each buffer is a rolling 8-day list of `list[tuple[datetime, float]]`. When computing dampening at time T, we pick buffer `floor(T.hour / 2)`, shift historical timestamps to today, and align. Coefficients are recomputed from scratch each time — no caching between forecasts. The computed coefficients are stored only as attributes on the "Dampened Score" sensor for monitoring purposes.

### HA Past Forecast Data

**Finding: external HA integrations do NOT expose past forecast data.** Solcast, OpenWeather, and Forecast.Solar coordinators only provide the current forecast. There is no way to bootstrap forecast history from HA. The 12-buffer approach starts recording from when the story is deployed.

### Actuals Data Source

Actual solar production from `QSSolarHistoryVals` (home.py:3297):
- Entity: `solar_plant.solar_inverter_input_active_power`
- Resolution: 5-minute intervals (`INTERVALS_MN`)
- Access: `self.home._consumption_forecast.solar_production_history`
- Bootstrap: `get_historical_solar_fallback()` (solar.py:232) returns past day patterns from this history at 15-min granularity

### Current Sensor Layout (to be changed)

Currently in `sensor.py:373-430`, three sensors per provider:
- `qs_solar_forecast_score_{name}` — "Score (Name)" — `provider.get_active_score()`
- `qs_solar_forecast_score_raw_{name}` — "Raw Score (Name)" — `provider.score_raw`
- `qs_solar_forecast_score_dampened_{name}` — "Dampened Score (Name)" — `provider.score_dampened`

Constants in `const.py`:
- `SENSOR_SOLAR_FORECAST_SCORE_PREFIX`
- `SENSOR_SOLAR_FORECAST_SCORE_RAW_PREFIX`
- `SENSOR_SOLAR_FORECAST_SCORE_DAMPENED_PREFIX`

**New layout:** Two sensors per provider + one on the solar plant:
- "Dampened Score (Name)" — dampened MAE + dampening params in attributes
- "No Dampening Score (Name)" — raw MAE
- "Active Solar Provider" — on solar plant, shows active provider name

### Dashboard Templates

Solar section in both jinja templates (`ui/quiet_solar_dashboard_template*.yaml.j2`) currently iterates entities by prefix:
- `qs_solar_dampening_*` — dampening switches
- `qs_solar_forecast_score_*` (excluding raw/dampened) — score sensors

Update to show the new sensor names and the active provider sensor.

### Architecture Constraints

- All new utils in `home_model/home_utils.py` — domain layer, pure Python, NO `homeassistant.*` imports
- `ha_model/solar.py` — HA bridge layer, may import from both
- All external I/O async (`save_dampening`/`load_dampening`)
- Lazy logging with `%s`, no f-strings in log calls
- 100% test coverage mandatory
- `strings.json` is the source — run `bash scripts/generate-translations.sh` after edits

### References

- [Source: home_model/load.py:1578-1801] — functions to move
- [Source: home_model/solver.py:179-264] — `create_power_slots` and `_power_slot_from_forecast` patterns
- [Source: home_model/home_utils.py:78-143] — `get_average_time_series`
- [Source: ha_model/solar.py:340-777] — all dampening infrastructure in QSSolarProvider
- [Source: ha_model/solar.py:266-291] — `_run_daily_scoring_cycle`
- [Source: ha_model/home.py:3297-3358] — `QSSolarHistoryVals` and `get_historical_solar_pattern`
- [Source: ha_model/solar.py:232-242] — `get_historical_solar_fallback`
- [Source: sensor.py:373-430] — current score sensor creation
- [Source: switch.py:95-114] — dampening switch
- [Source: button.py:38-44] — reset history button
- [Source: ui/quiet_solar_dashboard_template.yaml.j2:247-275] — solar dashboard section
- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.7:703-710] — original dampening acceptance criteria
- [Source: _bmad-output/project-context.md] — architecture and testing rules

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
