# Feature: Add Solar Forecast Dampening Buttons and Split Score Sensors

Status: done
issue: 124
branch: "QS_124"

## Story

As TheAdmin,
I want "Compute Dampening 1-Day", "Compute Dampening 7-Day", and "Reset Dampening" buttons on the solar plant dashboard, with split raw/dampened score sensors per provider that persist dampening coefficients in their attributes,
so that I can manually apply MOS linear regression correction to improve forecast accuracy, see the impact on scores, and have the solver use dampened forecasts automatically.

## Context

Story 3.7 originally included automatic solar forecast dampening (MOS linear regression at midnight). Story 3.14 removed it — too complex when automatic. This feature brings dampening back as **manual buttons** with two window sizes (1-day ratio correction, 7-day linear regression), split score sensors (raw + dampened), and coefficients persisted via sensor attributes using the existing RestoreEntity infrastructure.

The scoring infrastructure (MAE computation, provider ranking, auto-selection) remains fully intact from Story 3.14. This feature builds on top of it.

## Acceptance Criteria

### AC1 — Compute Dampening 1-Day Button

**Given** the solar plant dashboard is displayed
**When** TheAdmin presses the "Compute Dampening (1 Day)" button
**Then** the system determines the reference day: if local time >= 22:00, use today; otherwise use yesterday
**And** for each provider, it retrieves 1 day of actual production and stored forecast aligned to the ring buffer's 15-minute grid (`NUM_INTERVAL_PER_HOUR = 4`, 96 slots/day)
**And** for each 15-min slot `k` (0-95): computes ratio correction `a_k = actual_k / forecast_k`, `b_k = 0` (skip slot if `forecast_k < 10W`, use identity)
**And** physical guards are enforced: `a_k` clamped to `[0.1, 3.0]`, nighttime slots (both values zero) → identity `(1.0, 0.0)`
**And** coefficients are stored on the provider and immediately applied to `get_forecast()` and `get_value_from_current_forecast()`
**And** a dampened MAE score is computed and stored as `provider.score_dampened`

### AC2 — Compute Dampening 7-Day Button

**Given** the solar plant dashboard is displayed
**When** TheAdmin presses the "Compute Dampening (7 Day)" button
**Then** the system determines the reference day (same logic as AC1) and uses a 7-day window ending at that day
**And** for each provider, it retrieves 7 days of actual production and stored forecast aligned to the 15-minute grid
**And** for each 15-min slot `k` (0-95) with >= 3 data points: computes `numpy.polyfit(forecast_vals, actual_vals, 1)` → `[a_k, b_k]`
**And** slots with < 3 data points get identity `(1.0, 0.0)`
**And** same physical guards as AC1
**And** coefficients stored and dampened MAE score computed

### AC3 — Reset Dampening Button

**Given** any dampening state on any provider
**When** TheAdmin presses the "Reset Dampening" button
**Then** all providers get identity coefficients: `{k: (1.0, 0.0) for k in range(96)}`
**And** `score_dampened` is set to `None` for all providers
**And** `get_forecast()` returns the raw forecast (identity transform applied = no change)

### AC4 — Score Sensor Split

**Given** solar forecast providers are configured
**When** sensors are registered
**Then** the existing per-provider score sensor is renamed to "Forecast Raw Score (Provider)" (entity key unchanged to avoid ID breakage)
**And** a NEW per-provider "Forecast Dampened Score (Provider)" sensor is created with:
  - `native_value` = `provider.score_dampened` (MAE with dampening applied)
  - `extra_state_attributes` = `{"dampening_coefficients": {slot_str: [a_k, b_k], ...}}` containing the full coefficient dict

### AC5 — Dampening Persistence via Sensor Attributes

**Given** dampening coefficients are stored in the dampened score sensor attributes
**When** Home Assistant restarts
**Then** `QSBaseSensorSolarDampenedScoreRestore.async_added_to_hass()` reads coefficients from restored `extra_state_attributes`
**And** calls `provider.set_dampening_coefficients()` to restore the provider's dampening state
**And** `provider.score_dampened` is restored from the sensor's `native_value`
**And** subsequent calls to `get_forecast()` apply the restored dampening

### AC6 — Auto-Mode Uses Dampened Scores

**Given** the provider selection mode is "auto"
**When** `auto_select_best_provider()` runs
**Then** it uses `provider.get_active_score()` which returns `score_dampened` when available, falling back to `score` (raw)
**And** the provider with the lowest dampened MAE is selected

### AC7 — Dashboard UI Integration

**Given** the solar plant dashboard is generated
**When** the template renders the solar device section
**Then** dampened score sensors appear alongside raw score sensors
**And** "Compute Dampening (1 Day)", "Compute Dampening (7 Day)", and "Reset Dampening" buttons appear below the existing "Recompute Forecast Scores" button

## Tasks / Subtasks

- [x] Task 1: Add constants (AC: 1, 2, 3, 4)
  - [x] 1.1 In `const.py` (~line 279), add:
    - `BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY = "qs_solar_compute_dampening_1day"`
    - `BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY = "qs_solar_compute_dampening_7day"`
    - `BUTTON_SOLAR_RESET_DAMPENING = "qs_solar_reset_dampening"`
    - `SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX = "qs_solar_forecast_dampened_score_"`
    - Also added `NUM_INTERVAL_PER_HOUR`, `INTERVALS_MN`, `NUM_INTERVALS_PER_DAY` (moved from home.py)
  - [x] 1.2 Keep `SENSOR_SOLAR_FORECAST_SCORE_PREFIX` unchanged (preserves raw score entity IDs)

- [x] Task 2: Add dampening fields and methods to QSSolarProvider (AC: 1, 2, 3, 5, 6)
  - [x] 2.1 Add fields in `QSSolarProvider.__init__()` (`ha_model/solar.py`, after line 399):
    - `_dampening_coefficients: dict[int, tuple[float, float]] = {}` — key = 15-min slot index 0-95
    - `score_dampened: float | None = None`
  - [x] 2.2 Add property `has_dampening` → `bool(self._dampening_coefficients)`
  - [x] 2.3 Add property `dampening_coefficients` → `dict(self._dampening_coefficients)` (copy)
  - [x] 2.4 Add `set_dampening_coefficients(coefficients: dict[int, tuple[float, float]])` — used by sensor restore
  - [x] 2.5 Add `_time_to_slot_index(time: datetime) -> int`:
    - Convert UTC time to local time: `local = time.astimezone(dt_util.get_default_time_zone())`
    - Return `local.hour * NUM_INTERVAL_PER_HOUR + local.minute // INTERVALS_MN` (0-95)
    - Import `NUM_INTERVAL_PER_HOUR` and `INTERVALS_MN` from `home.py` (or use inline `4` and `15`)
  - [x] 2.6 Add `_get_dampened_value(time: datetime, raw_value: float) -> float`:
    - Look up `(a_k, b_k) = _dampening_coefficients.get(slot, (1.0, 0.0))`
    - Return `max(0.0, a_k * raw_value + b_k)`

- [x] Task 3: Implement compute_dampening (AC: 1, 2)
  - [x] 3.1 Add `compute_dampening(time: datetime, num_days: int) -> bool` on `QSSolarProvider`:
    - Determine reference day: local_time = time in local tz. If `local_time.hour >= 22`, ref_day = today; else ref_day = yesterday
    - Compute ref_end = ref_day at 23:59 local (end of reference day), ref_start = ref_end - `num_days` days
    - Get actuals: `forecast_handler.solar_production_history.get_historical_data(ref_end_utc, past_hours=num_days * 24)`
    - Get stored forecast: same approach as `compute_score` (lines 549-573) — pick 8h+ forecast type from `forecast_handler.get_forecast_histories_for_provider(self.provider_name)`, call `.get_historical_data(ref_end_utc, past_hours=num_days * 24)`
    - Build per-slot data: for each aligned (timestamp, forecast_val, actual_val) triple, compute slot index via `_time_to_slot_index(timestamp)`, group into `slots: dict[int, list[tuple[float, float]]]`
    - For each slot 0-95:
      - If `num_days == 1`: ratio correction `a_k = actual / forecast` if `forecast > 10`, else identity. `b_k = 0`
      - If `num_days >= 7` and `len(points) >= 3`: `a_k, b_k = numpy.polyfit(forecasts, actuals, 1)`. Apply guards: `a_k = max(0.1, min(3.0, a_k))`. Nighttime (all zero) → identity
      - If < 3 points: identity `(1.0, 0.0)`
    - Store in `self._dampening_coefficients`
    - Call `self.compute_dampened_score(time)` to compute the new dampened MAE
    - Return True on success

- [x] Task 4: Implement compute_dampened_score (AC: 1, 2, 4)
  - [x] 4.1 Add `compute_dampened_score(time: datetime) -> bool` on `QSSolarProvider`:
    - Same data retrieval as `compute_score` (24h window of actuals + forecast)
    - Apply `_get_dampened_value` to each forecast point before comparing
    - Store result in `self.score_dampened`
    - Return True on success

- [x] Task 5: Implement reset_dampening (AC: 3)
  - [x] 5.1 Add `reset_dampening()` on `QSSolarProvider`:
    - Set `_dampening_coefficients = {k: (1.0, 0.0) for k in range(24 * NUM_INTERVAL_PER_HOUR)}`
    - Set `score_dampened = None`

- [x] Task 6: Modify forecast access to apply dampening (AC: 1, 2, 3)
  - [x] 6.1 Modify `QSSolarProvider.get_forecast()` (line 422): when `_dampening_coefficients` is non-empty, apply `_get_dampened_value(t, v)` to each `(t, v)` point before returning
  - [x] 6.2 Modify `QSSolarProvider.get_value_from_current_forecast()` (line 427): same dampening application
  - [x] 6.3 Modify `QSSolarProvider.get_active_score()` (line 418): return `self.score_dampened` if not None, else `self.score`

- [x] Task 7: Add orchestration methods on QSSolar (AC: 1, 2, 3)
  - [x] 7.1 Add `async compute_dampening_all_providers(num_days: int, time: datetime | None = None)` on `QSSolar` (near `force_scoring_cycle`, line 294): iterate providers, call `compute_dampening(time, num_days)`, log results
  - [x] 7.2 Add `async reset_dampening_all_providers(time: datetime | None = None)` on `QSSolar`: iterate providers, call `reset_dampening()`, log confirmation

- [x] Task 8: Rename raw score sensor and add dampened score sensor (AC: 4, 5)
  - [x] 8.1 In `sensor.py` `create_ha_sensor_for_QSSolar()` (line 388): change raw score sensor `name` to `f"Forecast Raw Score ({provider_name})"` (keep `key` = `SENSOR_SOLAR_FORECAST_SCORE_PREFIX + safe_name` unchanged)
  - [x] 8.2 Add dampened score sensor per provider using `value_fn_and_attr`:
    ```python
    def _dampened_score_and_attr(device, key, prov=provider):
        value = prov.score_dampened
        attrs = {}
        if prov.has_dampening:
            coeffs = {str(slot): [float(a), float(b)] for slot, (a, b) in prov.dampening_coefficients.items()}
            attrs["dampening_coefficients"] = coeffs
        return value, attrs
    ```
    Key: `SENSOR_SOLAR_FORECAST_DAMPENED_SCORE_PREFIX + safe_name`, name: `f"Forecast Dampened Score ({provider_name})"`
  - [x] 8.3 Create `QSBaseSensorSolarDampenedScoreRestore(QSBaseSensorRestore)`:
    - `__init__`: accept `provider: QSSolarProvider` parameter (same pattern as `QSBaseSensorSolarScoreRestore`, line 658)
    - `async_added_to_hass`: call `super()`, restore `provider.score_dampened` from `_attr_native_value`, parse `dampening_coefficients` from `_attr_extra_state_attributes` and call `provider.set_dampening_coefficients()`
  - [x] 8.4 Use `QSBaseSensorSolarDampenedScoreRestore` as the entity class for dampened score sensors

- [x] Task 9: Create button entities (AC: 1, 2, 3, 7)
  - [x] 9.1 In `button.py` `create_ha_button_for_QSSolar()` (line 83), add 3 new `QSButtonEntityDescription`:
    - `BUTTON_SOLAR_COMPUTE_DAMPENING_1DAY` → `async_press=lambda x: x.device.compute_dampening_all_providers(num_days=1)`
    - `BUTTON_SOLAR_COMPUTE_DAMPENING_7DAY` → `async_press=lambda x: x.device.compute_dampening_all_providers(num_days=7)`
    - `BUTTON_SOLAR_RESET_DAMPENING` → `async_press=lambda x: x.device.reset_dampening_all_providers()`
  - [x] 9.2 Add translations in `strings.json` under `entity.button`:
    - `"qs_solar_compute_dampening_1day": {"name": "Compute dampening (1 day)"}`
    - `"qs_solar_compute_dampening_7day": {"name": "Compute dampening (7 day)"}`
    - `"qs_solar_reset_dampening": {"name": "Reset dampening"}`
  - [x] 9.3 Run `bash scripts/generate-translations.sh`

- [x] Task 10: Update dashboard template (AC: 7)
  - [x] 10.1 In `ui/quiet_solar_dashboard_template.yaml.j2` (after score loop at line 263-268): add loop for `qs_solar_forecast_dampened_score_*` entities
  - [x] 10.2 After the recompute scores button (line 269-272): add entries for `qs_solar_compute_dampening_1day`, `qs_solar_compute_dampening_7day`, `qs_solar_reset_dampening`
  - [x] 10.3 Same changes in `ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` if it has a matching solar section

- [x] Task 11: Write tests (AC: all)
  - [x] 11.1 Test `compute_dampening` 1-day mode: mock 1 day of forecast+actual data at 15-min resolution, verify ratio correction per slot, verify physical guards (nighttime identity, a_k bounds, skip low forecast)
  - [x] 11.2 Test `compute_dampening` 7-day mode: mock 7 days, verify `numpy.polyfit` regression per slot, verify min 3 data points guard, verify nighttime identity
  - [x] 11.3 Test day selection logic: verify pressing at 23:00 uses today, pressing at 16:00 uses yesterday
  - [x] 11.4 Test `reset_dampening`: verify identity coefficients stored for all 96 slots, `score_dampened` cleared
  - [x] 11.5 Test `_get_dampened_value`: verify correct slot lookup, value transformation, non-negative clamp
  - [x] 11.6 Test `get_forecast()` and `get_value_from_current_forecast()`: verify dampened output when coefficients active, raw output when empty
  - [x] 11.7 Test `get_active_score()`: returns `score_dampened` when available, falls back to `score`
  - [x] 11.8 Test sensor restore: create `QSBaseSensorSolarDampenedScoreRestore`, simulate persisted attrs with coefficients, verify `provider.set_dampening_coefficients()` called on restore
  - [x] 11.9 Test button integration: verify button press triggers correct `compute_dampening_all_providers(num_days=N)` or `reset_dampening_all_providers()`

## Dev Notes

### Architecture Compliance

- **Two-layer boundary**: Dampening computation lives in `ha_model/solar.py` (needs ring buffer access + HA sensor persistence). Uses `numpy.polyfit` for regression. No scipy (not a HA dependency).
- **15-minute grid alignment**: Coefficients use the same slot indexing as `QSSolarHistoryVals` ring buffers: `slot = hour * NUM_INTERVAL_PER_HOUR + minute // INTERVALS_MN` (0-95). Import `NUM_INTERVAL_PER_HOUR` from `ha_model/home.py` or use the value `4` inline with a comment.
- **Async rules**: `compute_dampening_all_providers()` and `reset_dampening_all_providers()` must be async. The numpy regression itself is fast for 96 slots × 7 points — no need for `async_add_executor_job`.
- **Logging**: Use lazy `%s` formatting, no f-strings in log calls, no trailing periods.
- **Constants**: All new keys in `const.py`. Never hardcode strings.
- **Translations**: Edit `strings.json` only, then run `bash scripts/generate-translations.sh`.

### Existing Patterns to Follow

| Pattern | Source | Use For |
|---------|--------|---------|
| Button creation | `button.py:83-94` (recompute scores) | 3 new buttons |
| Score sensor with restore | `sensor.py:388-404` + `QSBaseSensorSolarScoreRestore:658-680` | Dampened score sensor |
| `value_fn_and_attr` | `QSSensorEntityDescription.value_fn_and_attr` (sensor.py:527) | Return score + coefficients |
| Restore from attributes | `QSBaseSensorRestore.async_added_to_hass` (sensor.py:601-631) | Restore coefficients at boot |
| Scoring data access | `compute_score` (solar.py:538-588) | Same ring buffer access for dampening |
| Dashboard entity loop | `quiet_solar_dashboard_template.yaml.j2:263-268` | Dampened score loop |

### Key File Locations

| File | Purpose | Key Lines |
|------|---------|-----------|
| `const.py` | Constants | 233 (score prefix), 279 (solar button) |
| `ha_model/solar.py` | QSSolarProvider + QSSolar | 378-617 (provider), 83-376 (solar), 294-304 (force_scoring), 418-420 (get_active_score), 422-429 (get_forecast), 538-588 (compute_score) |
| `ha_model/home.py` | Ring buffer + history | NUM_INTERVAL_PER_HOUR (line 3023), get_historical_data (line 3580), get_forecast_histories_for_provider (line 3063) |
| `sensor.py` | Sensor creation + restore | 388-404 (score sensors), 520-528 (description), 543-572 (update_callback), 575-631 (restore), 658-680 (score restore) |
| `button.py` | Button entities | 83-94 (create_ha_button_for_QSSolar) |
| `strings.json` | Translations | entity.button section (~line 865) |
| `ui/quiet_solar_dashboard_template.yaml.j2` | Dashboard | 247-273 (solar section) |

### What NOT to Do

- Do NOT add automatic/scheduled dampening — manual button press only
- Do NOT add per-provider dampening switches (removed in Story 3.14)
- Do NOT use `scipy` — use `numpy.polyfit` only
- Do NOT change existing score sensor entity key — only rename the display `name`
- Do NOT use config entry options for persistence — use sensor attribute RestoreEntity pattern
- Do NOT add a binary_sensor for dampening status — coefficients visible in sensor attrs is sufficient

### References

- [Source: epics.md#Story-3.7, lines 709-751] — Original dampening spec (MOS regression)
- [Source: epics.md#Story-3.14, lines 776-786] — Dampening removal scope change
- [Source: ha_model/solar.py#QSSolarProvider, lines 378-617] — Provider class
- [Source: ha_model/solar.py#QSSolar, lines 83-376] — Solar device class
- [Source: ha_model/solar.py#compute_score, lines 538-588] — Scoring data access pattern
- [Source: sensor.py#QSBaseSensorSolarScoreRestore, lines 658-680] — Score restore pattern
- [Source: sensor.py#QSExtraStoredData, lines 575-599] — Extra stored data for attrs
- [Source: ha_model/home.py#NUM_INTERVAL_PER_HOUR, line 3023] — Ring buffer resolution
- [Source: ha_model/home.py#get_historical_data, line 3580] — Historical data retrieval
- [Source: button.py#create_ha_button_for_QSSolar, lines 83-94] — Button pattern
- [Source: _bmad-output/project-context.md] — Full 42-rule set

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References

### Completion Notes List
- All 11 tasks completed with 61 tests and 100% coverage
- Moved `NUM_INTERVAL_PER_HOUR`, `INTERVALS_MN`, `NUM_INTERVALS_PER_DAY` from `home.py` to `const.py` per user direction
- Updated existing button test to account for 3 new buttons

### Review Fixes (PR #125 review)
- Re-select active provider after `compute_dampening_all_providers` in auto mode
- Refresh `score_dampened` during `_run_scoring_cycle` when dampening active (prevents stale dampened MAE)
- Skip dampening on historical fallback forecast data (`_using_historical_fallback` flag)
- Remove unused `time` parameter from `reset_dampening_all_providers`
- Fix "7 day" → "7 days" pluralization in button label
- Extract `_get_scoring_data` shared helper (dedup forecast lookup across `compute_score`, `compute_dampened_score`, `_get_historical_data_for_dampening`)
- Remove 6 dev-only symlinks from commit
- Fix button integration tests to verify button→method wiring (not just mock)

### File List
- `custom_components/quiet_solar/const.py` — new constants (buttons, sensor prefix, ring buffer resolution)
- `custom_components/quiet_solar/ha_model/solar.py` — dampening fields, compute/reset/score methods, forecast dampening
- `custom_components/quiet_solar/ha_model/home.py` — import ring buffer constants from const.py
- `custom_components/quiet_solar/sensor.py` — dampened score sensor + restore class, raw score rename
- `custom_components/quiet_solar/button.py` — 3 new dampening buttons
- `custom_components/quiet_solar/strings.json` — button translations
- `custom_components/quiet_solar/translations/en.json` — generated
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2` — dampened scores + buttons
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` — same
- `tests/test_solar_dampening.py` — 61 tests covering all ACs
- `tests/test_platform_button.py` — updated solar button count assertion
