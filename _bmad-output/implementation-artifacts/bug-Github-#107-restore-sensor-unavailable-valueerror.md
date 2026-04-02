# Bug Fix: QSRestoreBaseSensor returns 'unavailable' string causing ValueError at boot

issue: 107
branch: "QS_107"

Status: ready-for-dev

## Story

As a Quiet Solar user,
I want sensor state restoration at boot to handle 'unavailable'/'unknown' string states gracefully,
so that numeric sensors don't crash with ValueError when HA restores a non-numeric state string.

## Bug Description

At boot time, solar forecast sensors using `QSBaseSensorRestore` fail with `ValueError` because the restored state is the string `"unavailable"` instead of `None`/numeric.

**Traceback:**
```
ValueError: could not convert string to float: 'unavailable'
ValueError: Sensor sensor.qs_peyrebelle_home_qs_no_control_forecast_now has device class 'power',
state class 'measurement' unit 'W' and suggested precision 'None' thus indicating it has a numeric
value; however, it has the non-numeric value: 'unavailable' (<class 'str'>)
```

**Affected entities:** Solar forecast sensors (4h, 8h, 12h, 18h, 24h), `no_control_forecast_now`, forecast age, forecast score, active provider.

## Root Cause Analysis

### The write path (line 554)

When `qs_is_none_unavailable=True` and the sensor value is `None`, `QSBaseSensor.async_update_callback()` at `sensor.py:554` sets:
```python
self._attr_native_value = STATE_UNAVAILABLE  # the string "unavailable"
```

This string gets persisted via `QSExtraStoredData.extra_restore_state_data` → HA's `RestoreEntity` storage.

### The read path (lines 617-618) - THE BUG

`QSBaseSensorRestore.async_added_to_hass()` at `sensor.py:617-618` restores the value without filtering:
```python
self._attr_native_value = last_sensor_state.native_value  # "unavailable" string passed through!
```

No check for `STATE_UNAVAILABLE` or `STATE_UNKNOWN` — the raw string is assigned as the sensor's numeric value, causing HA to raise `ValueError` when it tries to format a numeric sensor with a non-numeric value.

## Acceptance Criteria

1. **AC1**: When `QSBaseSensorRestore.async_added_to_hass()` restores a `native_value` that is `"unavailable"` or `"unknown"`, it MUST convert it to `None` (not assign the string).
2. **AC2**: When restored `native_value` is a normal value (numeric string, `None`, etc.), behavior is unchanged.
3. **AC3**: `QSBaseSensorForecastRestore` (which calls `super().async_added_to_hass()`) inherits the fix automatically.
4. **AC4**: All existing tests pass. New tests cover the unavailable/unknown restoration paths.
5. **AC5**: 100% test coverage maintained.

## Tasks / Subtasks

- [ ] Task 1: Filter unavailable/unknown in `async_added_to_hass` (AC: #1, #2, #3)
  - [ ] 1.1 Add `STATE_UNKNOWN` to the imports from `homeassistant.const` in `sensor.py` (line 18-26). `STATE_UNAVAILABLE` is already imported.
  - [ ] 1.2 In `QSBaseSensorRestore.async_added_to_hass()` (line 617), add a guard: if `last_sensor_state.native_value` is in `(STATE_UNAVAILABLE, STATE_UNKNOWN)`, set `self._attr_native_value = None` instead.
- [ ] Task 2: Add tests (AC: #4, #5)
  - [ ] 2.1 Test: restoring `"unavailable"` native_value results in `None`
  - [ ] 2.2 Test: restoring `"unknown"` native_value results in `None`
  - [ ] 2.3 Test: restoring a normal value (e.g., `"42.5"`) works unchanged
  - [ ] 2.4 Test: restoring `None` works unchanged
  - [ ] 2.5 Test: `QSBaseSensorForecastRestore` also filters (inherits via super)
- [ ] Task 3: Run quality gates and verify

## Dev Notes

### Exact fix location

**File:** `custom_components/quiet_solar/sensor.py`

**Import fix (line ~18-26):** Add `STATE_UNKNOWN` to the existing import block:
```python
from homeassistant.const import (
    PERCENTAGE,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,        # ADD THIS
    EntityCategory,
    ...
)
```

**Logic fix (lines 606-618):** In `QSBaseSensorRestore.async_added_to_hass()`:
```python
async def async_added_to_hass(self) -> None:
    """Restore sensor state on startup, filtering invalid state strings."""
    await super().async_added_to_hass()

    self._attr_native_value = None
    self._attr_extra_state_attributes = {}

    last_sensor_state = await self.async_get_last_sensor_data()
    if not last_sensor_state:
        return

    # Filter out HA state strings that aren't valid sensor values
    if last_sensor_state.native_value not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        self._attr_native_value = last_sensor_state.native_value
    self._attr_extra_state_attributes = last_sensor_state.native_attr
```

### Architecture compliance

- Fix is in `ha_model` layer (sensor.py) — correct, no domain boundary violation
- Uses `STATE_UNAVAILABLE` / `STATE_UNKNOWN` constants from `homeassistant.const` — follows const rule
- No lazy-logging violations (no new log calls needed)

### Test patterns

- Existing test pattern for `async_added_to_hass`: see `test_platform_sensor.py:322` (`test_qs_load_sensor_current_constraints_async_added_to_hass`) — uses `patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock)` to mock restore data
- `QSExtraStoredData` valid/invalid dict tests exist at lines 296-318
- Follow the same mock pattern: create a `QSExtraStoredData` with the target value, patch `async_get_last_extra_data` to return it, call `await sensor.async_added_to_hass()`, assert `_attr_native_value`

### Sensors affected (all use QSBaseSensorRestore or QSBaseSensorForecastRestore)

| Sensor | Class | Lines |
|--------|-------|-------|
| `qs_no_control_forecast_now/15mn/30mn/1h/3h/6h` | `QSBaseSensorForecastRestore` | 346-362 |
| `qs_solar_forecast_15mn/1h/4h/8h/12h/18h/24h` | `QSBaseSensorForecastRestore` | 413-429 |
| Per-provider forecast sensors | `QSBaseSensorForecastRestore` | 431-450 |
| Forecast age | `QSBaseSensorRestore` | 382-384 |
| Forecast score (per-provider) | `QSBaseSensorRestore` | 391-399 |
| Active provider | `QSBaseSensorRestore` | 402-410 |

### References

- [Source: custom_components/quiet_solar/sensor.py#QSBaseSensorRestore] — lines 594-618
- [Source: custom_components/quiet_solar/sensor.py#QSBaseSensorForecastRestore] — lines 625-647
- [Source: custom_components/quiet_solar/sensor.py#QSExtraStoredData] — lines 568-591
- [Source: custom_components/quiet_solar/sensor.py#async_update_callback-unavailable] — line 554
- [Source: tests/test_platform_sensor.py#restore-tests] — lines 296-369

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Story created from GitHub issue #107 analysis + codebase exploration
- Root cause identified: no filtering in async_added_to_hass restore path
- Fix is minimal (2-line change + import), low regression risk

### File List

- `custom_components/quiet_solar/sensor.py` (modify)
- `tests/test_platform_sensor.py` (modify)
