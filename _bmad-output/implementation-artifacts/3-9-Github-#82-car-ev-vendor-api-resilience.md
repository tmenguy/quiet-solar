# Story 3.9: FM-008 — Car/EV Vendor API Resilience

Status: ready-for-dev
issue: 82
branch: "QS_82"

## Story

As TheAdmin,
I want the system to detect stale car API data, infer correct state from manual user actions, offer a manual car mode when the API is unreliable for days, and handle the cascading impact across charging, prediction, and person tracking,
So that an unreliable car vendor API doesn't silently degrade the entire household optimization.

## Real-World Problem (Magali's Scenario)

The Renault Twingo APIs sometimes "stall" for hours or days — the car's HA sensors (home, plug, odometer, position, SOC) stop updating. When stalled:

1. Car shows `not_home` and `unplugged` even though Magali drove home and plugged it in
2. Charger detects power draw but can't match it to the Twingo (stale GPS/plug data) -> treated as guest car
3. Magali manually assigns the charger to the Twingo via the car card
4. **Problem**: SOC is stale/wrong, but the system doesn't know it — charges to wrong target
5. **Need**: When Magali manually assigns charger to a stale car, the system should (a) infer the car IS home and plugged, (b) use "stale percent" constraint mode (percent constraints starting at 0%, bypassing stale SOC), (c) show a red border around the entire car card

## Acceptance Criteria

### AC1: Stale Car API Detection

**Given** a car entity's API data stops updating (all tracked sensors — home, plug, odometer, position, SOC — haven't changed for longer than a configurable threshold, default ~4 hours)
**When** the system detects staleness
**Then** the car is flagged as `_car_api_stale = True`
**And** `binary_sensor.qs_<car>_api_ok` turns off
**And** the car card gets a red border around the entire card
**And** the SOC widget in the dashboard header shows an error/warning state
**And** the car auto-switches to "stale percent" constraint mode: percent constraints with initial SOC forced to 0% (bypassing stale SOC sensor), display shows `+XX%` (delta added by current constraint)

### AC2: Manual Assignment Infers State for Stale Cars

**Given** Magali manually assigns a charger to a car whose API data is stale
**When** the manual assignment is made (via `set_user_originated("car_name", ...)` on charger or `set_user_originated("charger_name", ...)` on car)
**Then** the system infers: car IS home, car IS plugged (overriding stale sensor data)
**And** the car uses "stale percent" constraint mode: `MultiStepsPowerLoadConstraintChargePercent` with `car_initial_value = 0` (bypass stale SOC)
**And** the car card center display shows `+XX%` (percent added by current constraint) instead of absolute `XX%`
**And** the car card has a red border indicating unreliable data

### AC3: Manual Mode for Extended Staleness

**Given** the car API has been stale for an extended period
**When** TheAdmin toggles `switch.qs_<car>_manual_mode`
**Then** the system bypasses the vendor API entirely
**And** TheAdmin can enter current SOC via `number.qs_<car>_manual_soc`
**And** the system tracks SOC by accumulating charger energy delivery (kWh to % using battery capacity)
**And** car presence is inferred from charger state (plugged = home, unplugged = away)
**And** person forecast for this car uses conservative defaults (always assume full charge needed)

### AC4: Auto-Recovery When API Resumes

**Given** the car API was stale and recovers
**When** the system detects recovery — defined as: at least one of the "critical" sensors (home tracker OR plug state) has a fresh timestamp, AND if the car was manually assigned to a charger during staleness, the plug sensor must be among the fresh ones
**Then** `binary_sensor.qs_<car>_api_ok` turns on
**And** the stale flag auto-clears, the car card removes the red border
**And** if the car was in "stale percent" mode (manual override / inferred state), the assigned person is notified (CC-001): "Your {car_name}'s data has recovered — charging will resume using live data"
**And** constraints are reassessed: real SOC is re-read, stale percent constraints are replaced with normal percent constraints using actual SOC as initial value
**And** if manual mode (switch) is on, TheAdmin is notified suggesting to exit manual mode

**Note on recovery sensors**: Not all sensors carry the same weight. An odometer or range update alone is not sufficient to exit stale mode — the home tracker and plug state are the "critical" sensors because they directly affect charger assignment and constraint behavior. Recovery requires at least one critical sensor to be fresh. If the car was manually forced onto a charger (inferred plugged), the plug sensor specifically must recover before we trust it again.

### AC5: Notifications

**Given** a car becomes stale or recovers
**When** the state transition happens
**Then** TheAdmin is notified with specific stale data points (which sensors, how long stale)
**And** Magali is notified if her car is affected, with plain-language explanation: "Your Twingo's data is stale — charging will use conservative estimates. You can enter the current battery level manually."

## Tasks / Subtasks

### Task 1: Add Stale Detection Infrastructure to QSCar (AC: #1)

- [ ] 1.1 Add constant `CAR_API_STALE_THRESHOLD_S = 14400` (4 hours) to `car.py` — code-level tunable, not user-configurable via config flow
- [ ] 1.2 Classify sensors into "critical" (home tracker, plug state) and "supplementary" (odometer, range, SOC) for recovery logic
- [ ] 1.3 Track list of "API sensors" on QSCar: `car_tracker`, `car_plugged`, `car_charge_percent_sensor`, `car_odometer_sensor`, `car_estimated_range_sensor`
- [ ] 1.4 Implement `_check_car_api_staleness(self, time)` method:
  - For each API sensor, get its `last_updated` timestamp via `get_sensor_latest_possible_valid_time_value_attr()`
  - If ALL sensors have `last_updated` older than threshold -> `_car_api_stale = True`
  - Track `_was_car_api_stale` for transition detection (same pattern as solar `_was_stale`)
  - Track `_car_api_stale_since` timestamp for notification detail
- [ ] 1.5 Call `_check_car_api_staleness()` from `update_current_metrics()` (called every state cycle)
- [ ] 1.6 Add tests for stale detection: fresh data, all stale, partial stale (should NOT trigger), threshold boundary

### Task 2: Implement "Stale Percent" Constraint Mode (AC: #1, #2)

**Key concept**: When the car API is stale, we do NOT switch to energy mode. Instead we use a "stale percent" mode — still uses `MultiStepsPowerLoadConstraintChargePercent` but forces `car_initial_value = 0` (bypass stale SOC sensor). The display shows `+XX%` (delta added) instead of absolute `XX%`. This is better than energy mode because percent is more meaningful to users. The target percent handler continues to work as before — user can still set a target SOC%.

**CRITICAL**: In stale mode the SOC sensor is poisoned. The ONLY source of charge progress is energy delivered by the charger (converted to % via battery capacity). This affects multiple call sites — see the full SOC bypass audit below.

#### 2a: Constraint initialization (charger.py:3220-3239)

- [ ] 2.1 Add `_car_api_stale_percent_mode` flag on QSCar (True when stale AND can_use_charge_percent_constraints_static)
- [ ] 2.2 Keep `_use_percent_mode = True` when API is stale (do NOT fall back to energy mode for staleness). The existing `CAR_INVALID_DURATION_PERCENT_SENSOR_FOR_ENERGY_MODE_S` continues to handle SOC-sensor-only staleness as before — this story's stale detection is a separate, broader mechanism.
- [ ] 2.3 Modify the constraint building in `charger.py:3220-3239`: when `car._car_api_stale_percent_mode` is True:
  - Still use `MultiStepsPowerLoadConstraintChargePercent` (percent class)
  - But set `car_initial_value = 0` instead of `self.car.get_car_charge_percent(time)`
  - `car_current_charge_value = 0` as well
  - Target charge: keep the user's target % as-is (target handler works normally)

#### 2b: Constraint update callback — NEVER read SOC sensor in stale mode (charger.py:4521-4680)

- [ ] 2.4 In `constraint_update_value_callback_soc()` (charger.py:4550): when `car._car_api_stale_percent_mode` is True:
  - **Skip** `sensor_result = self.car.get_car_charge_percent(time, ...)` entirely — set `sensor_result = None`
  - This forces the existing fallback to `result = result_calculus` (line 4575) which uses `_compute_added_charge_update()` — energy-based delta converted to % via battery capacity
  - The `is_car_charge_growing()` check (line 4604) also uses SOC sensor — skip it in stale mode (the calculus path doesn't need it)
  - The "expected charge state" power check (lines 4622-4656) remains valid — it checks charger power, not SOC

#### 2c: SOC accessors that must be bypassed in stale mode

Full audit of `get_car_charge_percent()` callers that need stale-aware behavior:

| File:Line | Usage | Stale mode behavior |
|-----------|-------|-------------------|
| `car.py:576` | Odometer/range estimation | Return None or skip — range is unreliable when SOC is stale |
| `car.py:595` | Person mileage/range forecast | Use conservative defaults (full charge needed) |
| `car.py:1034` | Car state checks | Skip SOC-dependent checks |
| `car.py:1073` | Charge type determination | Use stale-aware path |
| `car.py:1129` | Autonomy to target SOC calc | Return None — autonomy is unknown when stale |
| `car.py:1431` | Dynamic charging priority scoring | Use conservative score (treat as fully discharged) |
| `charger.py:2573` | `get_stable_dynamic_charge_status()` priority | Use conservative score |
| `charger.py:3226` | **Constraint init** | Force `car_initial_value = 0` (Task 2a) |
| `charger.py:4550` | **Constraint update loop** | Skip sensor, use calculus only (Task 2b) |
| `sensor.py:144-153` | HA `qs_car_soc_percent` sensor | Keep exposing last known value but add stale attribute |

- [ ] 2.5 Implement stale-aware behavior for each caller above. The simplest approach: make `get_car_charge_percent()` return None when `_car_api_stale_percent_mode` is True, and ensure every caller already handles None gracefully (most already do — verify each one).
- [ ] 2.6 Add a method or sensor attribute exposing the `+XX%` delta value (= `ct.current_value` since init was 0) for the UI
- [ ] 2.7 Add tests: stale API uses percent constraints with initial=0 and calculus-only updates; SOC sensor is never read during stale; non-stale uses actual SOC; stale recovery re-reads real SOC and rebuilds constraint

### Task 3: Infer State from Manual Charger Assignment (AC: #2)

- [ ] 3.1 Add `_car_api_inferred_home` and `_car_api_inferred_plugged` flags on QSCar
- [ ] 3.2 In charger's manual assignment path (when `set_user_originated("car_name", ...)` is called and the car has `_car_api_stale`):
  - Set `car._car_api_inferred_home = True` and `car._car_api_inferred_plugged = True`
  - Log info: "Car %s manually assigned to charger %s while API stale — inferring home and plugged"
- [ ] 3.3 Modify `is_car_home()` to return True if `_car_api_inferred_home` is True (when stale)
- [ ] 3.4 Modify `is_car_plugged()` to return True if `_car_api_inferred_plugged` is True (when stale)
- [ ] 3.5 Clear inferred flags when car is detached from charger or API recovers
- [ ] 3.6 Add tests: manual assign while stale -> inferred home/plugged; detach -> flags cleared; recovery -> flags cleared

### Task 4: Manual Mode Entities (AC: #3)

- [ ] 4.1 Add `CONF_CAR_MANUAL_MODE` and `CONF_CAR_MANUAL_SOC` constants to `const.py`
- [ ] 4.2 Create `switch.qs_<car>_manual_mode` entity in `switch.py`
  - When toggled on: bypass all API sensor readings, use manual SOC + charger inference
  - When toggled off: resume normal API-based operation
- [ ] 4.3 Create `number.qs_<car>_manual_soc` entity in `number.py`
  - Range: 0-100, step: 1
  - Only effective when manual mode is on
- [ ] 4.4 Implement SOC tracking by energy accumulation in manual mode:
  - When charger delivers energy, convert kWh to % using `car_battery_capacity`
  - Formula: `delta_soc = (energy_kwh / battery_capacity_kwh) * 100`
  - Update `manual_soc` accordingly
- [ ] 4.5 In manual mode, `is_car_home()` and `is_car_plugged()` infer from charger state
- [ ] 4.6 In manual mode, person forecast uses conservative defaults (full charge needed)
- [ ] 4.7 Add tests for manual mode: enable/disable, SOC accumulation, charger-based inference

### Task 5: Binary Sensor for API Health (AC: #1, #4)

- [ ] 5.1 Add `BINARY_SENSOR_CAR_API_OK = "qs_car_api_ok"` constant to `const.py`
- [ ] 5.2 Register `binary_sensor.qs_<car>_api_ok` in `binary_sensor.py`
  - State: `not self._car_api_stale`
  - Device class: `connectivity`
  - Attributes: `stale_since`, `stale_sensors` (list of which sensors are stale)
- [ ] 5.3 Add tests for sensor state transitions

### Task 6: Car Card UI Indicators (AC: #1, #2)

- [ ] 6.1 Add `api_ok` entity reference to car card JS entity discovery
- [ ] 6.2 When `api_ok` is `off`:
  - Add a red border (`border: 2px solid red` or similar) around the entire card container
  - Center SOC display switches to `+XX%` format showing delta percent added (read from the stale percent delta sensor/attribute)
  - Example: if constraint started at 0% and charger has delivered enough for 15%, display shows `+15%`
- [ ] 6.3 When `api_ok` returns to `on`: remove red border, revert to normal absolute `XX%` display
- [ ] 6.4 Dashboard header SOC widget: when `api_ok` is `off`, show the value with error styling (red text or warning icon)

### Task 7: Notifications (AC: #5)

- [ ] 7.1 On stale transition (`_was_car_api_stale` changed to True):
  - Notify TheAdmin: "Car {name} API data is stale since {time}. Affected sensors: {list}. Charging will use conservative energy-based estimates."
  - Notify Magali (if her car): "Your {car_name}'s data is stale — charging will use conservative estimates. You can enter the current battery level manually."
- [ ] 7.2 On recovery transition:
  - Notify TheAdmin: "Car {name} API data has recovered."
  - If manual mode is on: "Car {name} API has recovered. Consider disabling manual mode."
- [ ] 7.3 Use existing notification pattern from `async_notify_all_mobile_apps()` (see Story 3.3 patterns)
- [ ] 7.4 Add tests for notification triggers

### Task 8: Auto-Recovery (AC: #4)

- [ ] 8.1 In `_check_car_api_staleness()`, detect recovery using tiered sensor logic:
  - **Entry to stale**: ALL tracked sensors stale > threshold (unchanged)
  - **Exit from stale**: at least one "critical" sensor (home tracker OR plug state) has a fresh timestamp
  - **Extra rule**: if `_car_api_inferred_plugged` is True (car was manually forced onto charger), the plug sensor specifically must be fresh before exiting stale
  - This prevents: odometer updates alone clearing stale while home/plug are still wrong
- [ ] 8.2 On recovery:
  - Auto-clear `_car_api_stale`, `_car_api_inferred_home`, `_car_api_inferred_plugged`, `_car_api_stale_percent_mode`
  - Re-read real SOC from API sensor, replace stale percent constraints (init=0) with normal percent constraints (init=real SOC)
  - Notify assigned person (CC-001): "Your {car_name}'s data has recovered — charging will resume using live data"
  - If manual mode (switch) is on: notify TheAdmin suggesting to exit manual mode (don't auto-disable)
- [ ] 8.3 Add tests for recovery scenarios:
  - Odometer-only update does NOT clear stale
  - Home tracker update clears stale (when no inferred plug)
  - Plug update clears stale when car was manually assigned (inferred plug)
  - Constraint reassessment on recovery, person notification, manual mode persistence

## Dev Notes

### Architecture Constraints

- **Two-layer boundary**: All staleness logic lives in `ha_model/car.py` (HA layer). If any pure-logic helper is needed, place it in `home_model/` with zero HA imports.
- **Async rules**: `_check_car_api_staleness()` is called from `update_current_metrics()` which runs in the state polling cycle (~4s). Keep it lightweight — no blocking calls.
- **Logging**: Lazy `%s` format, no f-strings, no trailing periods in log messages.
- **Constants**: ALL new config keys go in `const.py`. Never hardcode strings.

### Existing Patterns to Reuse

| Pattern | Source | Reuse for |
|---------|--------|-----------|
| Solar stale detection | `solar.py:397-403` (`is_stale`, `_was_stale`) | Car API stale detection with transition tracking |
| Percent mode sensor getter | `car.py:536-566` (`car_use_percent_mode_sensor_state_getter`) | Stale percent mode keeps percent ON but forces initial SOC to 0 |
| Sensor timestamp query | `device.py` (`get_sensor_latest_possible_valid_time_value_attr`) | Get last update time for each API sensor |
| Manual charger assignment | `charger.py:2695-2709` (`get_user_originated("car_name")`) | Detection point for "manual assign while stale" |
| Notification pattern | `home.py` (`async_notify_all_mobile_apps`) | Stale/recovery notifications |
| Binary sensor registration | `binary_sensor.py:76-81` (existing car binary sensors) | New `api_ok` binary sensor |
| Constraint initial value | `charger.py:3220-3239` (constraint building) | Override `car_initial_value = 0` for stale percent mode |

### Key Code Locations

| File | Lines | What |
|------|-------|------|
| `ha_model/car.py` | 62 | `CAR_INVALID_DURATION_PERCENT_SENSOR_FOR_ENERGY_MODE_S = 3600` |
| `ha_model/car.py` | 536-566 | `car_use_percent_mode_sensor_state_getter()` — extend this |
| `ha_model/car.py` | 876 | `is_car_plugged()` — add inferred flag check |
| `ha_model/car.py` | 927 | `is_car_home()` — add inferred flag check |
| `ha_model/car.py` | 951 | `get_car_charge_percent()` — manual SOC override |
| `ha_model/car.py` | 1646-1664 | `can_use_charge_percent_constraints()` / `_static()` |
| `ha_model/charger.py` | 2695-2709 | Manual car assignment scoring |
| `ha_model/charger.py` | 3220-3239 | Constraint class selection (percent vs energy) |
| `ha_model/solar.py` | 397-403 | `is_stale` pattern to follow |
| `ha_model/solar.py` | 469-491 | Stale transition detection pattern |
| `ui/resources/qs-car-card.js` | 143-188 | Percent/energy mode display logic |
| `ui/resources/qs-car-card.js` | 232-275 | Circular SVG gauge |
| `const.py` | 220 | `BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS` |
| `const.py` | 321 | `CAR_USE_PERCENT_MODE_SENSOR` |
| `binary_sensor.py` | 76-81 | Existing car binary sensor registration |

### Anti-Patterns to Avoid

1. **Do NOT add staleness checks inside the solver** — the solver operates on constraints. Staleness changes the constraint initial value (SOC→0%) BEFORE the solver runs. Do NOT switch to energy mode for staleness — use "stale percent" (same percent constraint class, just init=0%).
2. **Do NOT modify `home_model/` imports** — staleness detection uses HA entity timestamps, which is HA-layer concern.
3. **Do NOT auto-disable manual mode on recovery** — the user explicitly enabled it, they should explicitly disable it. Only notify.
4. **Do NOT treat "some sensors stale" as "API stale"** — ALL tracked API sensors must be stale to flag the API as stale. A single fresh sensor means the API is still working (partial data is better than no data).
5. **Do NOT modify the charger scoring algorithm** — the manual assignment override already exists and works. This story adds inference ON TOP of existing assignment.
6. **Do NOT create new JS card files** — modify the existing `qs-car-card.js`.
7. **Do NOT read the car SOC sensor anywhere in stale mode** — the SOC sensor is poisoned when the API is stale. The ONLY source of charge progress is energy delivered by the charger (via `_compute_added_charge_update()`). Every call to `get_car_charge_percent()` must be stale-aware.
8. **Do NOT change the target percent handler** — user can still set target SOC% normally even in stale mode. Only the initial value and progress tracking change.

### Previous Story Intelligence

From **Story 3.7** (Solar Forecast API Resilience):
- Stale detection pattern: track `_latest_successful_time`, `_was_stale` flag, transition detection
- Binary sensor for health: `binary_sensor.qs_solar_forecast_ok` pattern
- Notification on state transition (stale/recovery)
- All followed lazy logging, 100% coverage

From **Story 3.14** (Solar Scoring):
- Time series utilities moved to `home_utils.py` — reuse if needed
- Ring buffer pattern for historical data

From **Story 3.3** (Grid Outage):
- `async_notify_all_mobile_apps()` notification pattern with per-app failure isolation
- Plain-language messages for Magali
- Hardened exception handling: catch specific exceptions, not bare `except`

### Testing Strategy

- **Unit tests** (`@pytest.mark.unit`): stale detection logic, flag transitions, inferred state, manual mode SOC accumulation
- **Integration tests** (`@pytest.mark.integration`): binary sensor state, switch/number entity behavior, notification triggers
- **Scenario tests**: full flow — car goes stale → manual assign → stale percent mode (init=0%, +XX% display) → charger delivers energy → API recovers → normal percent mode resumes with real SOC
- Use `freezegun` for time-dependent staleness threshold tests
- Use FakeHass for domain logic tests, real HA fixtures for entity integration tests
- 100% coverage required — no exceptions

### Project Structure Notes

- New constants in `const.py` — follow `CONF_CAR_*` naming pattern
- New entities in `switch.py`, `number.py`, `binary_sensor.py` — follow existing car entity registration patterns
- Car card changes in `ui/resources/qs-car-card.js` — no new JS files
- Dashboard changes in `ui/dashboard.py` if the SOC widget error styling requires template changes
- No config flow changes needed — stale threshold is a code-level constant (`CAR_API_STALE_THRESHOLD_S`)

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.9] — Original story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#Decision 1] — Resilience fallback table
- [Source: _bmad-output/planning-artifacts/architecture.md#Architectural Gaps #3] — Failure mode gap
- [Source: _bmad-output/implementation-artifacts/3-7-fm-001-solar-forecast-api-resilience.md] — Solar stale detection pattern
- [Source: _bmad-output/implementation-artifacts/3-3-fm-005-grid-outage-verification.md] — Notification pattern
- [Source: _bmad-output/project-context.md] — 42-rule code quality set

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
