# Bug Fix: Fix solar forecast scores, prober persistence, and entity lookup

Status: done
issue: 84
branch: "QS_84"

## Story

As a Quiet Solar user,
I want forecast scores to compute correctly and forecast probers to survive reboots,
so that forecast accuracy is tracked, scoring sensors are available, and forecast graphs have no post-restart gaps.

## Bug Description

### Bug 1 (Critical): QSforecastValueSensor state lost on reboot

**File:** `ha_model/home.py:156-199`

`QSforecastValueSensor._stored_values` is an in-memory `list[tuple[datetime, float]]` initialized to `[]` in `__init__` (line 170). On every HA restart, all probers start empty.

**Impact:**
- 8h prober needs 8 hours of continuous running before it covers the current time slot
- During recovery, `get_value_from_time_series` (`home_model/home_utils.py:538+`) with `idx <= 0` returns the first stored value (a future nighttime 0)
- Causes partial forecast graphs — every restart creates a gap of `delta` hours
- Example: restart at 15:56 UTC → 8h (23:56 = night) and 12h (03:56 = night) show 0, while 18h and 24h land on next-day daytime

### Bug 2 (Critical): Forecast history entity lookup uses wrong device — scores permanently broken

**File:** `ha_model/home.py:3105,3126`

```python
ha_entity = self.home.ha_entities.get(forecast_entity_name, None)
```

Forecast sensor entities are registered on the **Solar** device (via `entity.py:128-133` → `device.attach_exposed_has_entity(self)`), but the lookup searches `self.home.ha_entities` (the **Home** device). Result is always `None`.

**Additionally:** Even if lookup succeeded, the code passes the HA entity **object** to `QSSolarHistoryVals(entity_id=ha_entity)` which expects a **string** entity_id (used at lines 4298 and 4312 for `load_from_history()` and `hass.states.get()`).

**Impact:**
- `solar_forecast_history` and `solar_forecast_history_per_provider` stay `None`
- `get_forecast_histories_for_provider()` returns `{}`
- `compute_score()` never finds past data → always returns `False`
- Score sensors permanently **unavailable** (`score=nan is_score_computed=False`)

### Bug 3 (Cosmetic): Entity naming redundancy — DEFERRED

`device_id = f"qs_{slugify(name)}_{device_type}"` produces `"qs_solar_solar"`. Combined with forecast keys like `"qs_solar_forecast_15mn"`, entity IDs become `quiet_solar.qs_solar_solar_qs_solar_forecast_15mn`. Functional but confusing. **Deferred** to separate story due to entity ID migration complexity.

## Acceptance Criteria

1. After HA restart, forecast sensors for all horizons immediately show restored values (no gap)
2. Score sensors become available after the first scoring cycle (00:00 or 12:00 local)
3. Log shows `is_score_computed=True` at scoring boundaries
4. Forecast graphs have no post-restart gaps
5. Prober serialization round-trips correctly (serialize → restore produces identical state)
6. Entity lookup correctly finds forecast sensors on the Solar device
7. `QSSolarHistoryVals` receives a string entity_id, not an object

## Tasks / Subtasks

- [x] Task 1: Add serialize/restore to QSforecastValueSensor (AC: 1, 4, 5)
  - [x] 1.1 Add `serialize_stored_values() -> list[list]` method returning `[[iso_str, float], ...]`
  - [x] 1.2 Add `restore_stored_values(data: list[list])` class method rebuilding `_stored_values`
  - [x] 1.3 Add `qs_prober` optional field to `QSSensorEntityDescription` dataclass
  - [x] 1.4 Wire prober references into forecast sensor entity descriptions (solar + home)
  - [x] 1.5 Create `QSBaseSensorForecastRestore` extending `QSBaseSensorRestore` for forecast sensors
  - [x] 1.6 Use `QSBaseSensorForecastRestore` for forecast sensor entities in `_get_forecast_sensors_*` functions
- [x] Task 2: Fix forecast entity lookup for scoring (AC: 2, 3, 6, 7)
  - [x] 2.1 At line 3105: change `self.home.ha_entities` → `self.home.solar_plant.ha_entities`
  - [x] 2.2 At line 3105: change `entity_id=ha_entity` → `entity_id=ha_entity.entity_id`
  - [x] 2.3 At line 3126: same two changes for per-provider lookup
  - [x] 2.4 Add null guard for `self.home.solar_plant` (property can return None based on home_mode)
- [x] Task 3: Add tests (AC: 5, 6, 7)
  - [x] 3.1 Test prober serialization round-trip
  - [x] 3.2 Test prober restore from serialized data
  - [x] 3.3 Test entity lookup uses solar_plant.ha_entities
  - [x] 3.4 Test entity_id passed as string to QSSolarHistoryVals
- [x] Task 4: Document deferred entity naming issue
  - [x] 4.1 Add entry to `_bmad-output/implementation-artifacts/deferred-work.md`

## Dev Notes

### Existing RestoreEntity Pattern (REUSE THIS)

The codebase already has a working restore pattern in `sensor.py`:

- **`QSExtraStoredData`** (lines 546-569): Custom `ExtraStoredData` with `native_value` + `native_attr`. Has `as_dict()` / `from_dict()`.
- **`QSBaseSensorRestore`** (lines 572-601): Inherits `QSBaseSensor, RestoreEntity`. Restores state in `async_added_to_hass()` via `self.async_get_last_extra_data()`.
- Import already exists: `from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity` (line 26)

**Strategy:** Extend `QSBaseSensorRestore` with a `QSBaseSensorForecastRestore` subclass that also persists prober `_stored_values` via extra state attributes. The prober data fits naturally into the `native_attr` dict of `QSExtraStoredData`.

### QSSensorEntityDescription Extension

Current dataclass (`sensor.py:492-498`):
```python
@dataclass(frozen=True, kw_only=True)
class QSSensorEntityDescription(SensorEntityDescription):
    qs_is_none_unavailable: bool = False
    value_fn: Callable[[AbstractDevice, str], Any] | None = None
    value_fn_and_attr: Callable[[AbstractDevice, str], tuple[Any, Any]] | None = None
```

Add: `qs_prober: QSforecastValueSensor | None = None` — this links the entity description to its prober instance so the restore sensor can access it.

**NOTE:** The dataclass is `frozen=True` so the prober must be set at creation time, not mutated later. The prober instances are created in `QSHome.__init__` / `QSSolar.__init__` (before sensor entities), so they're available when entity descriptions are built.

### Forecast Sensor Entity Creation Points

Forecast sensors are created in these `sensor.py` functions — all currently use `QSBaseSensor` and must switch to `QSBaseSensorForecastRestore`:

| Function | Lines | Device | Sensors |
|----------|-------|--------|---------|
| `_get_forecast_sensors_home` | 343-354 | QSHome | Home non-controlled consumption forecasts |
| `_get_forecast_sensors_solar` | 401-412 | QSSolar | Aggregate solar forecasts |
| `_get_forecast_sensors_solar` | 414-429 | QSSolar | Per-provider solar forecasts |

### solar_plant Property Warning

`QSHome.solar_plant` (line 441-449) returns `None` for home modes without solar. The entity lookup fix **must** guard against `solar_plant is None`:
```python
solar = self.home.solar_plant
if solar is not None:
    ha_entity = solar.ha_entities.get(forecast_entity_name, None)
```

### QSSolarHistoryVals entity_id Usage

`QSSolarHistoryVals.__init__` (line 3439-3452) stores `entity_id: str`. Used at:
- Line 4298: `await load_from_history(self.hass, self.entity_id, start_time, end_time)`
- Line 4312: `real_state = self.hass.states.get(self.entity_id)`

Both require a string like `"sensor.quiet_solar_qs_solar_solar_qs_solar_forecast_15mn"`. The HA entity object's `.entity_id` property (entity.py:122) provides this string.

### Test Patterns

- Use `pytest` functions (not `unittest.TestCase`) for new tests
- `conftest.py` has `FakeHass`, factory fixtures, mock patterns
- `factories.py` has `create_minimal_home_model()` for lightweight home instances
- Existing forecast tests in `test_forecasts.py`, `test_solar_forecast_scoring.py`
- Mock prober with known `_stored_values` for serialization tests
- For entity lookup tests: create a `FakeQSSolar` with populated `ha_entities` dict

### Architecture Constraints

- **Two-layer boundary**: `home_model/` NEVER imports `homeassistant.*`. The `QSforecastValueSensor` class lives in `ha_model/home.py` (HA layer), so using HA types is allowed.
- **Logging**: lazy `%s` formatting, no f-strings in log calls
- **Async**: no blocking calls in async code

### Project Structure Notes

- All changes stay within `ha_model/` and `sensor.py` (HA integration layer) — no domain layer boundary violations
- Tests go in `tests/` root directory following `test_bug_84_*.py` naming pattern

### References

- [Source: ha_model/home.py:156-199] QSforecastValueSensor class
- [Source: ha_model/home.py:3065-3130] solar_forecast_set_and_reset method
- [Source: ha_model/home.py:3438-3452] QSSolarHistoryVals class
- [Source: ha_model/home.py:441-449] solar_plant property
- [Source: sensor.py:492-498] QSSensorEntityDescription dataclass
- [Source: sensor.py:501-543] QSBaseSensor class
- [Source: sensor.py:546-601] QSExtraStoredData + QSBaseSensorRestore classes
- [Source: sensor.py:343-429] Forecast sensor entity creation functions
- [Source: entity.py:128-133] Entity registration via attach_exposed_has_entity
- [Source: home_model/home_utils.py:538+] get_value_from_time_series function

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
