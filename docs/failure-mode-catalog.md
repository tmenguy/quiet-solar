# Failure Mode Catalog

Living document per NFR20. Updated whenever a new failure mode is discovered or an existing response changes.

Last updated: 2026-03-20

## Priority Levels

| Priority | User Impact | Examples |
|----------|-------------|----------|
| **P1** | Missed commitment — car not charged, comfort violated | Charger failure, solver infeasibility, grid outage |
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
| **Fallback Behavior** | Use last successful forecast. If stale >6h, fall back to historical consumption patterns (numpy ring buffer, 560 days). |
| **Recovery Path** | Auto-retry on next forecast polling cycle (~30s). Valid orchestrators re-included. |
| **Implementation Status** | Partial |
| **Test Coverage** | None |

**Current code:** `ha_model/solar.py` — validation probe at init filters invalid orchestrators (`get_power_series_from_orchestrator` wrapped in try/except). Invalid providers removed from `self.orchestrators` list.

**Gaps for Story 3.2:**
- No continuous staleness detection (>6h threshold from AR5 not implemented)
- No fallback to historical patterns when forecast is stale
- One-time validation only, no runtime monitoring

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

### FM-003: HA State Unavailability

| Field | Value |
|-------|-------|
| **Dependency** | Home Assistant entity state system |
| **Priority** | P3 — reduced visibility (device continues with last-known-good) |
| **Failure Signature** | Entity state = `STATE_UNKNOWN` or `STATE_UNAVAILABLE` |
| **Fallback Behavior** | Filter invalid values from history. Device continues with last-known-good state. |
| **Recovery Path** | Auto-recover when entity returns valid state. |
| **Implementation Status** | Implemented |
| **Test Coverage** | Covered |

**Current code:** `ha_model/bistate_duration.py`, `ha_model/person.py`, `ha_model/home.py` — all check for `STATE_UNKNOWN`/`STATE_UNAVAILABLE` before processing. Invalid states skipped or mapped to defaults.

**Gaps for Story 3.2:** None — this failure mode is adequately handled.

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
| **Test Coverage** | Covered |

**Current code:** Extensive off-grid support:
- `const.py` — `CONF_OFF_GRID_ENTITY`, `OFF_GRID_MODE_AUTO`/`FORCE_OFF_GRID`/`FORCE_ON_GRID`
- `config_flow.py` — off-grid entity configuration with state value and inversion
- `ha_model/home.py` — off-grid detection and mode switching
- `binary_sensor.py` — `BINARY_SENSOR_HOME_IS_OFF_GRID`, `BINARY_SENSOR_HOME_REAL_OFF_GRID`

**Gaps for Story 3.2:**
- Verify emergency broadcast implementation
- Verify load shedding prioritization in off-grid mode

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

## Gap Analysis Summary (Story 3.2 Scope)

| ID | Gap | Failure Mode | NFR/FR |
|----|-----|-------------|--------|
| G1 | Solar forecast staleness detection (>6h threshold) | FM-001 | AR5, NFR13 |
| G2 | Solar forecast fallback to historical patterns | FM-001 | AR5, FR21, FR22 |
| G3 | Charger retry with exponential backoff | FM-002 | NFR12 |
| G4 | Charger unavailable state + admin notification | FM-002 | FR29 |
| G5 | Solver infeasibility detection + safe defaults | FM-004 | AR5 |
| G6 | Emergency broadcast verification in off-grid | FM-005 | AR5, FR19 |
| G7 | Numpy bare except → specific exceptions + logging | FM-006 | Code quality |
| G8 | Prediction confidence scoring | FM-007 | AR5 |

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
