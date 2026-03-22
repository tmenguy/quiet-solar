in# Failure Mode Catalog

Living document per NFR20. Updated whenever a new failure mode is discovered or an existing response changes.

Last updated: 2026-03-20

## Priority Levels

| Priority | User Impact | Examples |
|----------|-------------|----------|
| **P1** | Missed commitment — car not charged, comfort violated | Charger failure, car API stale, solver infeasibility, grid outage |
| **P2** | Degraded optimization — higher cost, less solar use | Solar forecast stale, prediction confidence low |
| **P3** | Reduced visibility — missing data, incomplete history | Numpy corruption, HA state temporarily unavailable |

---

## Failure Mode Entries

### FM-001: Solar Forecast API Failure

| Field | Value |
|-------|-------|
| **Dependency** | Solcast / OpenMeteo via HA integration |
| **Priority** | P2 — degraded optimization (uses stale or no forecast) |
| **Failure Signature** | API timeout, HTTP error, or orchestrator returns invalid data |
| **Fallback Behavior** | Use last successful forecast. If stale >6h, fall back to historical consumption patterns (numpy ring buffer, 560 days). Multi-provider: if one provider fails, continue with remaining providers. |
| **Recovery Path** | Auto-retry on next forecast polling cycle (~30s). Valid orchestrators re-included. Failed provider re-probed periodically. |
| **Implementation Status** | Partial |
| **Test Coverage** | None |

**Current code:** `ha_model/solar.py` — validation probe at init filters invalid orchestrators (`get_power_series_from_orchestrator` wrapped in try/except). Invalid providers removed from `self.orchestrators` list. Supports multiple orchestrators already.

**Multi-provider resilience strategy (future):**
- Support multiple forecast providers simultaneously (e.g., Solcast + OpenMeteo)
- Best-provider selection: compare forecast accuracy against actual production over rolling window, weight providers by accuracy
- Auto-dampening: if one provider consistently over/under-predicts, reduce its weight automatically
- Provider failover: if primary provider fails, seamlessly continue with secondary without interruption
- This turns FM-001 from a single point of failure into a redundant system

**Gaps for Story 3.2:**
- No continuous staleness detection (>6h threshold from AR5 not implemented)
- No fallback to historical patterns when forecast is stale
- One-time validation only, no runtime monitoring
- No provider accuracy comparison or weighting
- No auto-dampening based on forecast vs. actual production
- No automatic failover between providers

---

### FM-002: Charger Communication Failure

| Field | Value |
|-------|-------|
| **Dependency** | OCPP / Wallbox / Generic charger protocols via HA |
| **Priority** | P1 — missed commitment (car may not charge) |
| **Failure Signature** | Command not ACKed within `CHARGER_STATE_REFRESH_INTERVAL_S` (14s) |
| **Fallback Behavior** | Retry up to 3 times, then mark charger unavailable. Solver excludes unavailable chargers. Notify TheAdmin. |
| **Recovery Path** | Auto-recover when charger responds. Re-include in next solver cycle. |
| **Implementation Status** | Partial |
| **Test Coverage** | Partial — wallbox fallback enum tested, basic charger scenarios covered |

**Current code:** `ha_model/charger.py` — OCPP notification handling with exception logging. Wallbox conditional import with fallback StrEnum. Service call wrappers catch exceptions.

**Gaps for Story 3.2:**
- No retry loop with exponential backoff (NFR12 requires configurable count + backoff)
- No explicit "mark unavailable" state transition
- No admin notification on persistent failure (FR29)

---

### FM-003: HA Device Communication Failure (State + Commands)

| Field | Value |
|-------|-------|
| **Dependency** | Home Assistant entity state system + service call infrastructure |
| **Priority** | P1 when command fails on a load (device doesn't do what solver asked) / P3 when state data is stale or unavailable |
| **Failure Signature** | Entity state = `STATE_UNKNOWN` or `STATE_UNAVAILABLE`. Service call raises exception or silently fails. Device state doesn't change after command. State timestamps stop updating (stale). |
| **Fallback Behavior** | **State side:** filter invalid values from history, continue with last-known-good. **Command side:** detect command not applied (state didn't change within expected window), alert human to intervene physically. |
| **Recovery Path** | Auto-recover when entity returns valid state or device acknowledges command. Admin/Magali can force the physical device manually if automated commands fail. |
| **Implementation Status** | Partial (state filtering implemented, command failure detection not implemented) |
| **Test Coverage** | Partial (state filtering covered, command verification not covered) |

**This applies to ANY HA-connected device** — chargers, switches, climate controllers, pool pumps, etc. The charger-specific case (FM-002) is a specialization with protocol-level detail, but the generic problem is: the solver sends a command, and the physical device doesn't do it.

**Failure scenarios:**
| Scenario | Impact | Current Handling |
|----------|--------|-----------------|
| State = unavailable/unknown | Device uses last-known-good values | Implemented — state filtering in bistate_duration.py, person.py, home.py |
| State timestamps stop updating (stale) | System operates on outdated data, makes wrong decisions | Not Implemented — no staleness detection |
| Service call raises exception | Command lost, device doesn't change state | Partial — some try/except with logging, but no retry or human alert |
| Service call succeeds but device doesn't react | Silent failure — solver thinks command was applied | Not Implemented — no command verification |
| Device responds but with wrong/garbage values | Corrupted data enters solver decisions | Not Implemented — no value sanity checking |

**Current code:** `ha_model/bistate_duration.py`, `ha_model/person.py`, `ha_model/home.py` — state filtering for `STATE_UNKNOWN`/`STATE_UNAVAILABLE`. Service calls in `ha_model/charger.py` and `ha_model/home.py` wrapped in try/except with logging.

**Gaps for Story 3.2:**
- Command verification: after sending a command, check that device state changed within expected window
- If command not applied: retry once, then alert TheAdmin/Magali to physically intervene ("The pool pump didn't turn on — please check the breaker or switch it manually")
- Stale state detection: compare `last_updated` timestamp against configurable threshold per device type
- Value sanity checking: reject obviously wrong values (negative power, SOC > 100%, temperature outside physical range)
- Per-device health binary sensor: `binary_sensor.qs_<device>_communication_ok` for dashboard visibility

---

### FM-004: Solver Infeasibility

| Field | Value |
|-------|-------|
| **Dependency** | Internal solver (`home_model/solver.py`) |
| **Priority** | P1 — missed commitment (no valid plan means no device commands) |
| **Failure Signature** | No valid plan satisfies all active constraints |
| **Fallback Behavior** | Mandatory constraints get priority, filler constraints dropped. Battery held at current SOC. |
| **Recovery Path** | Re-solve on next triggered evaluation or 5-minute periodic fallback. |
| **Implementation Status** | Partial |
| **Test Coverage** | Partial — constraint priority logic tested, no explicit infeasibility scenario |

**Current code:** `home_model/solver.py` — constraint priority tiers exist (mandatory urgent > mandatory deadline > green > filler). Solver processes constraints in priority order.

**Gaps for Story 3.2:**
- No explicit infeasibility detection and handler
- No safe-default fallback when solver cannot produce any valid plan
- No logging of infeasibility causes for troubleshooting (feeds Story 3.4)

---

### FM-005: Grid Outage

| Field | Value |
|-------|-------|
| **Dependency** | Grid power sensor entity |
| **Priority** | P1 — missed commitment (household on battery + solar only) |
| **Failure Signature** | Grid power sensor drops to zero or off-grid entity triggers |
| **Fallback Behavior** | Off-grid mode: reduce consumption to solar + battery capacity. Emergency broadcast to mobile apps. |
| **Recovery Path** | Auto-detect grid restoration. Broadcast recovery notification. Resume normal operation. |
| **Implementation Status** | Implemented |
| **Test Coverage** | Fully Verified (Story 3.3, 2026-03-22) |

**Current code:** Extensive off-grid support:
- `const.py` — `CONF_OFF_GRID_ENTITY`, `OFF_GRID_MODE_AUTO`/`FORCE_OFF_GRID`/`FORCE_ON_GRID`
- `config_flow.py` — off-grid entity configuration with state value and inversion
- `ha_model/home.py` — off-grid detection, mode switching, emergency broadcast (`async_notify_all_mobile_apps`)
- `binary_sensor.py` — `BINARY_SENSOR_HOME_IS_OFF_GRID`, `BINARY_SENSOR_HOME_REAL_OFF_GRID`
- `home_model/solver.py` — off-grid constraint filtering, best-effort load exclusion, battery depletion with min SOC

**Verified in Story 3.3:**
- Emergency broadcast sends critical push to all mobile apps with per-app failure isolation
- Notification messages are plain-language (Magali-friendly)
- Load shedding prioritizes mandatory over filler/best-effort loads
- Solver uses only solar + battery capacity in off-grid (no grid import)
- Battery min SOC respected during off-grid depletion
- 3-minute transition gate blocks solver until loads acknowledge
- On-grid restoration clears gate and resumes normal operation
- Force-on-grid overrides real off-grid state (safety override)
- Unavailable/unknown entity state defaults to on-grid (safe default)

---

### FM-006: Numpy Persistence Corruption

| Field | Value |
|-------|-------|
| **Dependency** | numpy `.npy` file storage (ring buffers) |
| **Priority** | P3 — reduced visibility (loses historical data, cold start) |
| **Failure Signature** | Corrupted `.npy` file from partial write, disk full, or permission error |
| **Fallback Behavior** | `np.load()` wrapped in try/except, returns None on failure. System operates without historical data (cold start). Falls back to conservative estimates. |
| **Recovery Path** | Ring buffer rebuilds over time as new data accumulates. Full 560-day window takes ~1.5 years to restore. |
| **Implementation Status** | Implemented |
| **Test Coverage** | Partial |

**Current code:** `ha_model/home.py:3764-3769` and `3776-3779` — `np.load()` wrapped in bare `except:` returning None. System continues without historical data.

**Gaps for Story 3.2:**
- Replace bare `except:` with specific exception types (`OSError`, `ValueError`, `pickle.UnpicklingError`)
- Add warning log when persistence fails (currently silent)

---

### FM-007: Prediction Confidence Degradation

| Field | Value |
|-------|-------|
| **Dependency** | Internal prediction engine (`ha_model/person.py`) |
| **Priority** | P2 — degraded optimization (over-charges to compensate for uncertainty) |
| **Failure Signature** | Insufficient mileage history, holiday period, major life change (new job/commute) |
| **Fallback Behavior** | When history is sparse or pattern match score is low, fall back to conservative defaults: charge to higher SOC than predicted necessary. |
| **Recovery Path** | Predictions self-correct as new patterns accumulate in the 30-day mileage window. |
| **Implementation Status** | Partial |
| **Test Coverage** | Partial — person prediction tests exist, no explicit low-confidence scenario |

**Current code:** `ha_model/person.py` — `_get_best_week_day_guess()` returns None when no matching weekday data exists. `_compute_person_next_need()` handles None by using defaults.

**Gaps for Story 3.2:**
- No explicit confidence score calculation
- No threshold-based behavior change (e.g., confidence < 50% → use max observed mileage)
- No logging of prediction confidence for observability

---

### FM-008: Car / EV Vendor API Failure

| Field | Value |
|-------|-------|
| **Dependency** | EV vendor cloud APIs (Renault/Twingo, Tesla, etc.) via HA integrations |
| **Priority** | P1 — missed commitment (cascading failures across charging, prediction, and person tracking) |
| **Failure Signature** | API returns stale data (timestamps stop updating), HTTP errors, or entity becomes unavailable. Can persist for days without detection. |
| **Fallback Behavior** | See cascading impact table below. |
| **Recovery Path** | Auto-recover when API returns fresh data. Stale detection should trigger admin notification and allow manual override (force-remove car from charger assignment, manual SOC entry). |
| **Implementation Status** | Partial |
| **Test Coverage** | Partial — charging percent-to-energy fallback exists, no stale detection |

**Cascading impact — this failure mode affects multiple subsystems:**

| Subsystem | Impact | Current Handling |
|-----------|--------|-----------------|
| **Car location (GPS)** | Unknown which charger the car is connected to. Cannot auto-assign charger. | None — system assumes last known location |
| **Car SOC** | Stale or unknown SOC. Cannot determine if charging is progressing. Cannot calculate remaining charge time. | Partial — charger falls back from percent-based to energy-based charging |
| **Car presence (home/away)** | Cannot determine if car is home. Affects charger assignment and constraint creation. | None — uses last known state |
| **Odometer / mileage** | No mileage data. Person forecast (`_get_best_week_day_guess`) cannot compute next trip prediction. | None — returns None, uses conservative defaults |
| **Person forecast** | No home/away transitions. `_compute_person_next_need` cannot build daily patterns. Mileage history stops accumulating. | Partial — handles None from prediction, but doesn't detect WHY data is missing |
| **Solver constraints** | Stale constraints based on old SOC/location. May charge an absent car or skip a present one. | None |

**Current code:** `ha_model/car.py` — reads SOC, location, odometer from HA entities. Charger has percent-to-energy fallback when SOC entity is unavailable. No staleness detection on car API data.

**Real-world incident:** Twingo (Renault) API stopped responding for several days. System did not detect the stale state. No way to reset, manually override, or force-remove the stale car from the planning cycle.

**Manual / degraded mode (recovery when API is unreliable for days):**

When the car API is confirmed stale (by detection or by TheAdmin), the system should offer a **"manual car mode"** that bypasses the vendor API entirely:

| Data point | Normal mode (API) | Manual / degraded mode |
|-----------|-------------------|----------------------|
| **SOC** | Read from car entity | TheAdmin enters current SOC manually (number entity). System then tracks SOC by accumulating charger energy delivery (kWh → % using battery capacity). |
| **Location / presence** | GPS from car entity | Infer from charger: if car is plugged into a known charger → car is home. If unplugged → car is away. |
| **Odometer / mileage** | Read from car entity | Suspended — no mileage data available. |
| **Person forecast** | Based on mileage + presence history | Suspended for this car — use conservative defaults (always assume next trip needs full charge). Can still use person GPS/phone presence independently. |
| **Charging control** | SOC-based (charge to target %) | Energy-based (charge X kWh from manual SOC). Already partially implemented as percent-to-energy fallback. |

**Entering manual mode:** TheAdmin toggles a switch entity (`switch.qs_<car>_manual_mode`), enters current SOC via number entity, and the system takes over from there. When the API recovers (fresh timestamps detected), the system can suggest exiting manual mode.

**This pattern could generalize** to other devices with unreliable cloud APIs — any device could have a "manual mode" where the human provides the data the API should have provided.

**Gaps for Story 3.2:**
- No staleness detection on car API data (compare last_updated timestamp against threshold)
- No admin notification when car data goes stale
- No manual car mode (switch entity + manual SOC entry + energy-based SOC tracking)
- No charger-based presence inference (plugged = home)
- No suspension of person forecast when car data is unreliable
- No auto-detection of API recovery to suggest exiting manual mode
- No mechanism to reset stale state when API recovers
- Person forecast should detect "no fresh car data" and fall back explicitly rather than silently degrading

---

## Cross-Cutting Resilience Requirements

### CC-001: Magali Communication on P1 Failures

When a P1 failure affects a household member's commitment (car not charging, load not running), **Magali must be notified** in plain language — not just TheAdmin. She needs to know:
- What happened ("Your car isn't charging because the charger lost connection")
- What she can do ("You can force-charge from the app, or plug into the other charger")
- Whether the system is handling it ("The system will retry automatically, but if it's not resolved by 7 AM, you'll need to take action")

**Applies to:** FM-002 (charger), FM-004 (solver), FM-005 (grid), FM-008 (car API)

### CC-002: Manual Override / Recovery Actions

Every P1 failure mode must have a **simple, immediate manual override** accessible from the HA UI — not just for TheAdmin but designed so Magali can use it under stress:

| Override Action | Failure Mode | Implementation |
|----------------|-------------|----------------|
| Force-start charge (ignore solver) | FM-002, FM-004, FM-008 | Button entity per charger |
| Force-stop charge | FM-002, FM-008 | Button entity per charger |
| Force load ON/OFF | FM-004, FM-005 | Switch entity per controllable load |
| Manual SOC entry | FM-008 | Number entity per car (when API SOC is stale) |
| Remove car from planning | FM-008 | Button entity per car |
| Force on-grid / off-grid | FM-005 | Select entity (already exists: `SELECT_OFF_GRID_MODE`) |

**Implementation Status:** Partial — off-grid mode select exists, some button entities exist, but no unified "override panel" and no stale-car manual controls.

### CC-003: Admin Resilience Dashboard

TheAdmin needs a **single dashboard view** showing all active issues, their severity, and available remediation actions:

| Section | Content | Priority |
|---------|---------|----------|
| **Active Alerts** | Current P1/P2/P3 failures with timestamp, affected devices, plain-language description | All |
| **P1 Actions** | One-click manual overrides (force charge, force load, manual SOC) | P1 |
| **P2 Status** | Solar forecast freshness, prediction confidence per person, provider health | P2 |
| **P3 Status** | Data persistence health, HA entity availability, historical data completeness | P3 |
| **Recovery Log** | Recent failures that auto-recovered, with duration and impact | All |

**Implementation Status:** Not Implemented

**Gaps for Story 3.2 / 3.4:**
- Sensor entities exposing failure state per dependency (binary_sensor per FM-XXX)
- Centralized failure registry that tracks active failures with timestamps
- Dashboard card (Lovelace) showing active issues and actions
- Notification routing: P1 → Magali + TheAdmin, P2/P3 → TheAdmin only

---

## Gap Analysis Summary (Story 3.2 Scope)

| ID | Gap | Failure Mode | NFR/FR |
|----|-----|-------------|--------|
| G1 | Solar forecast staleness detection (>6h threshold) | FM-001 | AR5, NFR13 |
| G2 | Solar forecast fallback to historical patterns | FM-001 | AR5, FR21, FR22 |
| G3 | Charger retry with exponential backoff | FM-002 | NFR12 |
| G4 | Charger unavailable state + admin notification | FM-002 | FR29 |
| G5 | Solver infeasibility detection + safe defaults | FM-004 | AR5 |
| ~~G6~~ | ~~Emergency broadcast verification in off-grid~~ | ~~FM-005~~ | ~~AR5, FR19~~ | **CLOSED** (Story 3.3) |
| G7 | Numpy bare except → specific exceptions + logging | FM-006 | Code quality |
| G8 | Prediction confidence scoring | FM-007 | AR5 |
| G9 | Car API staleness detection (timestamp threshold) | FM-008 | NFR13, NFR24 |
| G10 | Admin notification on stale car data | FM-008 | FR29 |
| G11 | Manual override to force-remove stale car from planning | FM-008 | FR24 |
| G12 | Person forecast detection of missing car data | FM-008 | FR13, FR24 |
| G13 | Solar multi-provider accuracy weighting | FM-001 | FR21 |
| G14 | Solar auto-dampening and failover | FM-001 | FR21, FR22 |
| G15 | Command verification (check device reacted after service call) | FM-003 | NFR12, FR24 |
| G16 | Human alert when command not applied ("check the device manually") | FM-003 | FR29, CC-001 |
| G17 | Per-device stale state detection (last_updated threshold) | FM-003 | NFR13, FR24 |
| G18 | Value sanity checking (reject obviously wrong readings) | FM-003 | FR24 |
| G19 | Per-device health binary sensor | FM-003 | FR39 |
| G20 | Magali notification on P1 failures (plain language) | CC-001 | FR26, FR28 |
| G21 | Manual override buttons/entities for P1 recovery | CC-002 | FR27 |
| G22 | Admin resilience dashboard with active alerts | CC-003 | FR39, FR41 |
| G23 | Failure registry (binary sensors per FM, centralized tracking) | CC-003 | FR25a |

---

## How to Add a New Failure Mode

When a new failure mode is discovered (NFR20):

1. Assign the next ID: `FM-XXX`
2. Copy the entry template below
3. Fill in all fields — leave none blank
4. Set Implementation Status to "Not Implemented"
5. Add to the Gap Analysis Summary table if implementation work is needed
6. Commit with message: `docs: add FM-XXX to failure mode catalog`

### Entry Template

```markdown
### FM-XXX: [Title]

| Field | Value |
|-------|-------|
| **Dependency** | |
| **Priority** | P1/P2/P3 |
| **Failure Signature** | |
| **Fallback Behavior** | |
| **Recovery Path** | |
| **Implementation Status** | Not Implemented / Partial / Implemented |
| **Test Coverage** | None / Partial / Covered |

**Current code:** [description of existing handling, if any]

**Gaps for Story 3.2:** [what needs to be implemented]
```
