# Bug Fix: Solar forecast score sensors unknown after reboot

Status: ready
issue: 104
branch: "QS_104"

## Story

As a Quiet Solar user with multiple solar forecast providers,
I want forecast score, active provider, and forecast age sensors to retain their values after a reboot,
so that the dashboard shows meaningful data immediately instead of "unknown" until 24h of history accumulates.

## Bug Description

After reboot the solar provider score sensors (`SENSOR_SOLAR_FORECAST_SCORE_PREFIX` per provider), `SENSOR_SOLAR_ACTIVE_PROVIDER`, and `SENSOR_SOLAR_FORECAST_AGE` are unknown because they use `QSBaseSensor` (no state restoration) instead of `QSBaseSensorRestore`.

**Steps to reproduce:**
1. System is running with multiple solar providers, scores are computed and visible
2. Reboot Home Assistant
3. Score sensors show "unknown", active provider sensor shows "unknown", forecast age shows "unknown"
4. Sensors remain unknown until enough historical data accumulates (~24h for scores)

**Expected:** Sensors restore their last-known values on reboot and update naturally once fresh data arrives.
**Actual:** Sensors are "unknown" until the system recomputes them from scratch.

## Root Cause Analysis

In `custom_components/quiet_solar/sensor.py`, function `create_ha_sensor_for_QSSolar()` (lines 367-407):

1. **Forecast age sensor** (line 382): `QSBaseSensor(...)` — no restore
2. **Per-provider score sensors** (line 397): `QSBaseSensor(...)` — no restore
3. **Active provider sensor** (line 407): `QSBaseSensor(...)` — no restore

`QSBaseSensor` does not extend `RestoreEntity`, so Home Assistant does not persist/restore these sensor values across reboots.

Meanwhile, the forecast power sensors (lines 422-425) correctly use `QSBaseSensorForecastRestore`, which extends `QSBaseSensorRestore → RestoreEntity` and survives reboot.

The score sensors are particularly affected because `QSSolarProvider.compute_score()` requires 24h of actuals from `solar_production_history` — data that isn't available after a fresh reboot until enough time passes.

## Fix

Switch all three sensor types from `QSBaseSensor` to `QSBaseSensorRestore` so their values are persisted and restored across reboots.

### Affected lines in `sensor.py`

| Sensor | Current (line) | Change to |
|--------|---------------|-----------|
| Forecast age | `QSBaseSensor(...)` (L382) | `QSBaseSensorRestore(...)` |
| Per-provider score | `QSBaseSensor(...)` (L397) | `QSBaseSensorRestore(...)` |
| Active provider | `QSBaseSensor(...)` (L407) | `QSBaseSensorRestore(...)` |

### Why `QSBaseSensorRestore` is the right choice

- Already proven: forecast power sensors use the derived `QSBaseSensorForecastRestore` class
- `QSBaseSensorRestore` handles `RestoreEntity` integration, `QSExtraStoredData` serialization, and `async_added_to_hass` restoration — all generic and applicable
- Restored values are a good approximation until fresh computation replaces them
- No need for `QSBaseSensorForecastRestore` — these sensors don't have `qs_prober` data to restore

### Implementation

The change is a 3-line edit in `sensor.py`:

```python
# Line 382: forecast age
entities.append(QSBaseSensorRestore(data_handler=device.data_handler, device=device, description=forecast_age_sensor))

# Line 397: per-provider score
entities.append(QSBaseSensorRestore(data_handler=device.data_handler, device=device, description=score_sensor))

# Line 407: active provider
entities.append(QSBaseSensorRestore(data_handler=device.data_handler, device=device, description=active_provider_sensor))
```

## Acceptance Criteria

- [ ] After reboot, `SENSOR_SOLAR_FORECAST_SCORE_PREFIX` sensors restore their last-known values
- [ ] After reboot, `SENSOR_SOLAR_ACTIVE_PROVIDER` sensor restores its last-known value
- [ ] After reboot, `SENSOR_SOLAR_FORECAST_AGE` sensor restores its last-known value
- [ ] Once fresh data arrives, sensors update normally (restored values are overwritten)
- [ ] Existing tests pass
- [ ] New tests verify restore behavior for these three sensor types

## Tasks

- [x] 1. Analyze root cause and confirm affected sensors
- [ ] 2. Change `QSBaseSensor` → `QSBaseSensorRestore` for the three sensor instantiations in `create_ha_sensor_for_QSSolar()`
- [ ] 3. Add tests verifying that the three sensors are instances of `QSBaseSensorRestore`
- [ ] 4. Run quality gates (`python scripts/qs/quality_gate.py`)

## Dev Notes

- The `QSBaseSensorRestore` class (sensor.py:590) already handles everything needed: `extra_restore_state_data`, `async_get_last_sensor_data`, and `async_added_to_hass` restoration
- No changes needed to `QSBaseSensorRestore` itself — it's generic enough
- The active provider string value and forecast age float value both serialize fine via `QSExtraStoredData`
