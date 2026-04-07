# Feature: Add Auto-Dampening and Reset Dampening Buttons to Solar Plant Dashboard UI

Status: ready-for-dev
issue: 124
branch: "QS_124"

## Story

As TheAdmin,
I want "Compute Dampening" and "Reset Dampening" buttons on the solar plant dashboard for each provider,
so that I can manually apply MOS linear regression correction to improve forecast accuracy when I choose, without the complexity of automatic dampening.

## Context

Story 3.7 originally included automatic solar forecast dampening (MOS linear regression running at midnight). Story 3.14 removed it entirely — it added complexity without sufficient benefit when automatic. The user now wants dampening back as a **manual, button-triggered** operation: press a button to compute coefficients, press another to clear them.

The scoring infrastructure (MAE computation, provider ranking, auto-selection) remains fully intact from Story 3.14. This feature builds on top of it.

## Acceptance Criteria

### AC1 — Compute Dampening Button

**Given** the solar plant dashboard is displayed
**When** TheAdmin presses the "Compute Dampening" button
**Then** for each configured provider, the system computes MOS linear regression coefficients `(a_k, b_k)` per time-of-day slot `k` using 7 days of (forecast_k, actual_k) data from the existing ring buffers
**And** dampened forecast = `max(0, a_k * raw_forecast_k + b_k)` (clamped to non-negative)
**And** physical guards are enforced: nighttime steps use identity `(1.0, 0.0)`, `a_k` bounded to `[0.1, 3.0]`, minimum 3 data points required per slot (otherwise identity)
**And** coefficients are stored on the provider and immediately applied to `get_forecast()` and `get_value_from_current_forecast()`
**And** a log message confirms dampening was computed with the number of slots and average correction factor

### AC2 — Reset Dampening Button

**Given** dampening coefficients are currently active for one or more providers
**When** TheAdmin presses the "Reset Dampening" button
**Then** all dampening coefficients are cleared for all providers
**And** `get_forecast()` and `get_value_from_current_forecast()` revert to returning the raw forecast directly
**And** a log message confirms dampening was reset

### AC3 — Dampening Persistence

**Given** dampening coefficients have been computed
**When** Home Assistant restarts
**Then** the dampening coefficients are restored from persistent storage (config entry `options` or similar mechanism)
**And** the forecast continues to use the dampened values without manual re-trigger

**Given** dampening has been reset
**When** Home Assistant restarts
**Then** no dampening coefficients are loaded (raw forecast is used)

### AC4 — Dashboard UI Integration

**Given** the solar plant dashboard is generated
**When** the template renders the solar device section
**Then** "Compute Dampening" and "Reset Dampening" buttons appear below the existing "Recompute Forecast Scores" button
**And** both buttons are always visible (no conditional display)

### AC5 — Dampening Status Sensor

**Given** dampening is active for a provider
**When** TheAdmin checks the dashboard
**Then** a `binary_sensor.qs_solar_dampening_active` sensor shows `on` if any provider has active dampening coefficients, `off` otherwise
**And** the sensor name is "Solar Dampening Active"

## Tasks / Subtasks

- [ ] Task 1: Add dampening constants and storage fields (AC: 1, 2, 3)
  - [ ] 1.1 Add constants in `const.py`: `BUTTON_SOLAR_COMPUTE_DAMPENING = "qs_solar_compute_dampening"`, `BUTTON_SOLAR_RESET_DAMPENING = "qs_solar_reset_dampening"`, `BINARY_SENSOR_SOLAR_DAMPENING_ACTIVE = "qs_solar_dampening_active"`
  - [ ] 1.2 Add dampening fields to `QSSolarProvider.__init__()` in `ha_model/solar.py` (around line 399): `_dampening_coefficients: dict[int, tuple[float, float]] = {}` (key = time-of-day slot index, value = (a_k, b_k))
  - [ ] 1.3 Add property `has_dampening` on `QSSolarProvider` returning `bool(self._dampening_coefficients)`

- [ ] Task 2: Implement MOS linear regression computation (AC: 1)
  - [ ] 2.1 Add method `compute_dampening(time: datetime) -> bool` on `QSSolarProvider`:
    - Retrieve 7 days of actual production from `solar_production_history.get_historical_data(time, past_hours=168)`
    - Retrieve 7 days of stored forecast from provider's forecast history ring buffers (same pattern as `compute_score`, lines 538-588)
    - Group aligned (forecast, actual) pairs by time-of-day slot (use the provider's temporal resolution, typically 15-min or 1-hour)
    - For each slot with >= 3 data points: compute linear regression `actual = a_k * forecast + b_k` using `numpy.polyfit(forecast, actual, 1)` or `scipy.stats.linregress`
    - Apply physical guards: `a_k = max(0.1, min(3.0, a_k))`, nighttime slots (both forecast and actual == 0 for all points) get identity `(1.0, 0.0)`
    - Store in `self._dampening_coefficients`
  - [ ] 2.2 Add method `reset_dampening()` on `QSSolarProvider`: clear `_dampening_coefficients`

- [ ] Task 3: Modify forecast access to apply dampening (AC: 1, 2)
  - [ ] 3.1 Modify `QSSolarProvider.get_forecast()` (line 422): if `_dampening_coefficients` is non-empty, apply `(a_k, b_k)` to each forecast value based on its time-of-day slot before returning. Use `_get_dampened_value(time, raw_value)` helper
  - [ ] 3.2 Modify `QSSolarProvider.get_value_from_current_forecast()` (line 427): same dampening application
  - [ ] 3.3 Add private helper `_get_dampened_value(time: datetime, raw_value: float) -> float`: compute slot index from time, look up `(a_k, b_k)`, return `max(0, a_k * raw_value + b_k)`
  - [ ] 3.4 Add private helper `_time_to_slot_index(time: datetime, resolution_minutes: int = 60) -> int`: convert time-of-day to slot index (e.g., for 1h resolution: 0-23; for 15-min resolution: 0-95)
  - [ ] 3.5 Store `_dampening_resolution_minutes: int` alongside coefficients so the slot index mapping is consistent

- [ ] Task 4: Add orchestration methods on QSSolar (AC: 1, 2)
  - [ ] 4.1 Add `async compute_dampening_all_providers()` on `QSSolar` (near `force_scoring_cycle`, line 294): iterate all providers, call `compute_dampening(now)`, log results
  - [ ] 4.2 Add `async reset_dampening_all_providers()` on `QSSolar`: iterate all providers, call `reset_dampening()`, log confirmation

- [ ] Task 5: Add persistence for dampening coefficients (AC: 3)
  - [ ] 5.1 Add `_salvable_dampening` property on `QSSolarProvider`: serialize `_dampening_coefficients` and `_dampening_resolution_minutes` to a JSON-compatible dict (follow the car dampening persistence pattern in `car.py` lines 180-183)
  - [ ] 5.2 In `QSSolar.__init__` (line 130): load dampening from config entry options if present, restore to each provider
  - [ ] 5.3 After `compute_dampening` and `reset_dampening`: save to config entry options via `hass.config_entries.async_update_entry(entry, options={...})` (follow existing persistence patterns)

- [ ] Task 6: Create button entities (AC: 1, 2, 4)
  - [ ] 6.1 In `button.py` `create_ha_button_for_QSSolar()` (line 83): add two new `QSButtonEntityDescription` entries:
    - `BUTTON_SOLAR_COMPUTE_DAMPENING` with `async_press=lambda x: x.device.compute_dampening_all_providers()`
    - `BUTTON_SOLAR_RESET_DAMPENING` with `async_press=lambda x: x.device.reset_dampening_all_providers()`
  - [ ] 6.2 Add translations in `strings.json` under `entity.button`:
    - `"qs_solar_compute_dampening": {"name": "Compute dampening"}`
    - `"qs_solar_reset_dampening": {"name": "Reset dampening"}`
  - [ ] 6.3 Run `bash scripts/generate-translations.sh` to sync translations

- [ ] Task 7: Create dampening active binary sensor (AC: 5)
  - [ ] 7.1 Add `BINARY_SENSOR_SOLAR_DAMPENING_ACTIVE` in `const.py`
  - [ ] 7.2 Create binary sensor entity in `binary_sensor.py` (follow `qs_solar_forecast_ok` pattern): state = `any(p.has_dampening for p in device.solar_forecast_providers.values())`
  - [ ] 7.3 Add `Platform.BINARY_SENSOR` to `QSSolar.get_platforms()` (line 371)
  - [ ] 7.4 Add translation in `strings.json`: `"qs_solar_dampening_active": {"name": "Solar dampening active"}`

- [ ] Task 8: Update dashboard template (AC: 4)
  - [ ] 8.1 In `ui/quiet_solar_dashboard_template.yaml.j2` (after the recompute scores button at line 269-272): add entries for `qs_solar_compute_dampening`, `qs_solar_reset_dampening`, and `qs_solar_dampening_active`
  - [ ] 8.2 Do the same in `ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` if it has a matching solar section

- [ ] Task 9: Write tests (AC: 1, 2, 3, 5)
  - [ ] 9.1 Test `compute_dampening`: mock 7 days of forecast and actual data, verify coefficients are computed correctly, verify physical guards (nighttime identity, a_k bounds, min 3 data points)
  - [ ] 9.2 Test `reset_dampening`: verify coefficients are cleared, forecast returns raw values
  - [ ] 9.3 Test `_get_dampened_value`: verify correct slot lookup and value transformation, verify non-negative clamp
  - [ ] 9.4 Test forecast modification: verify `get_forecast()` and `get_value_from_current_forecast()` apply dampening when active, return raw when inactive
  - [ ] 9.5 Test persistence: verify dampening coefficients survive simulated restart (save to config entry, reload)
  - [ ] 9.6 Test binary sensor: verify `qs_solar_dampening_active` reflects actual dampening state
  - [ ] 9.7 Test button press integration: verify button press triggers compute/reset flow

## Dev Notes

### Architecture Compliance

- **Two-layer boundary**: Dampening computation uses `numpy` (allowed in domain logic) but accesses ring buffers via `QSSolarProvider` which lives in `ha_model/`. The computation can stay in `ha_model/solar.py` since it needs HA config entry access for persistence. If a pure-Python helper is extracted later, it should go to `home_model/home_utils.py`.
- **Async rules**: `compute_dampening_all_providers()` and `reset_dampening_all_providers()` must be async since they may write to config entries. The regression computation itself (numpy) is CPU-bound — wrap in `hass.async_add_executor_job()` if it takes >100ms.
- **Logging**: Use lazy `%s` formatting, no f-strings in log calls, no trailing periods.
- **Constants**: All new keys go in `const.py`. Never hardcode strings.
- **Translations**: Edit `strings.json` only, then run `bash scripts/generate-translations.sh`. Never edit `translations/en.json` directly.

### Existing Patterns to Follow

- **Button creation**: Follow `create_ha_button_for_QSSolar()` in `button.py:83-94` — the "Recompute Forecast Scores" button is the exact pattern.
- **Binary sensor**: Follow `qs_solar_forecast_ok` binary sensor pattern in `binary_sensor.py` and `sensor.py`.
- **Config persistence**: Follow `car.py:180-183` for `_salvable_dampening` dict pattern. Use `hass.config_entries.async_update_entry()`.
- **Dashboard template**: Follow the entity rendering pattern at `quiet_solar_dashboard_template.yaml.j2:269-272`.
- **Scoring data access**: `compute_dampening` should access the same ring buffers as `compute_score` (lines 538-588 in `solar.py`) — use `forecast_handler.solar_production_history` for actuals and `forecast_handler.get_forecast_histories_for_provider()` for stored forecasts.

### Key File Locations

| File | Purpose | Key Lines |
|------|---------|-----------|
| `const.py` | Constants | 279 (existing solar button), 94-107 (solar config) |
| `ha_model/solar.py` | Provider + Solar classes | 378-617 (QSSolarProvider), 83-376 (QSSolar), 294-304 (force_scoring_cycle), 538-588 (compute_score) |
| `button.py` | Button entity creation | 83-94 (create_ha_button_for_QSSolar) |
| `strings.json` | Translation strings | entity.button section (line ~865) |
| `sensor.py` | Sensor creation | 368-459 (solar sensors) |
| `binary_sensor.py` | Binary sensor creation | (follow forecast_ok pattern) |
| `ui/quiet_solar_dashboard_template.yaml.j2` | Dashboard | 247-273 (solar section) |
| `home_model/home_utils.py` | Time series utilities | align/slot functions |

### MOS Linear Regression Reference

From the original Story 3.7 acceptance criteria (epics.md lines 744-751):
- For each time step k: compute `(a_k, b_k)` via linear regression on 7 days of `(forecast_k, actual_k)` data
- Dampened forecast = `max(0, a_k * raw_forecast_k + b_k)`
- Physical guards: nighttime steps → identity `(1.0, 0.0)`, `a_k` bounded to `[0.1, 3.0]`, minimum 3 data points per step
- Use `numpy.polyfit(forecast_values, actual_values, 1)` which returns `[a_k, b_k]`

### What NOT to Do

- Do NOT add automatic/scheduled dampening computation — this is manual only (button press)
- Do NOT add per-provider dampening switches (the old `switch.qs_solar_dampening_<provider>` approach was removed)
- Do NOT add `score_raw` / `score_dampened` split sensors — keep the single score sensor
- Do NOT modify the scoring cycle — dampening and scoring are independent operations

### References

- [Source: epics.md#Story-3.7, lines 709-751] — Original dampening spec
- [Source: epics.md#Story-3.14, lines 776-786] — Dampening removal scope change
- [Source: ha_model/solar.py#QSSolarProvider, lines 378-617] — Provider class
- [Source: ha_model/solar.py#QSSolar, lines 83-376] — Solar device class
- [Source: button.py#create_ha_button_for_QSSolar, lines 83-94] — Button pattern
- [Source: _bmad-output/project-context.md] — Full 42-rule set

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
