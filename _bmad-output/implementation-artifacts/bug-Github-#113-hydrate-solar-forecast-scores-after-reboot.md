# Bug Fix: Hydrate solar forecast scores and active provider from restored state after reboot

Status: ready-for-dev
issue: 113
branch: "QS_113"

## Story

As a Quiet Solar user with multiple solar forecast providers,
I want forecast scores and the active provider to survive HA reboots,
so that I don't see "Unavailable" scores or the wrong active provider until the next scoring cycle runs.

## Problem

After HA reboot, two things go wrong:

1. **Score sensors show Unavailable** despite having valid numeric scores before restart. `QSSolarProvider.score` (in-memory `float | None`) resets to `None` on boot. `QSBaseSensorRestore.async_added_to_hass` restores `_attr_native_value` from `QSExtraStoredData`, but nothing writes that value back into `provider.score`. The next `async_update_callback` reads `prov.score` via `value_fn`, gets `None`, and marks the entity Unavailable (`qs_is_none_unavailable=True`).

2. **Active provider defaults to first dict key** instead of the last active provider before reboot. `QSSolar.__init__` sets `_active_provider_name = next(iter(self.solar_forecast_providers))` (line 152 of `solar.py`). The active provider diagnostic sensor restores the correct string from HA state, but never writes it back into the model. The first `async_update_callback` publishes the wrong active name until `auto_select_best_provider` runs.

## Acceptance Criteria

1. After HA restart, per-provider forecast score sensors display the last persisted numeric value (not Unavailable) until `compute_score()` runs successfully and overwrites it.
2. After HA restart, the active provider diagnostic sensor and `device.active_provider_name` reflect the last active provider from before reboot (not the arbitrary first dict key).
3. Hydration is skip-safe: if restored value is `unavailable`, `unknown`, non-numeric (for scores), or not a valid provider name (for active provider), the default behavior is preserved.
4. Hydration only runs when the model value is still `None`/default — never clobbers a value already computed in the same boot.
5. The select entity remains the source of truth for auto vs pinned mode — hydration does NOT set `_provider_mode`.
6. 100% test coverage maintained.

## Tasks / Subtasks

- [ ] Task 1: Add `QSBaseSensorSolarScoreRestore` subclass (AC: #1, #3, #4)
  - [ ] 1.1 Create class in `sensor.py` extending `QSBaseSensorRestore`
  - [ ] 1.2 Accept `provider: QSSolarProvider` in constructor, store as `self._provider`
  - [ ] 1.3 In `async_added_to_hass`: call `await super().async_added_to_hass()` first, then if `self._provider.score is None` and `self._attr_native_value` is set, parse with `float()` in try/except (`TypeError`, `ValueError`), assign `self._provider.score` on success
- [ ] Task 2: Wire score sensor to new subclass (AC: #1)
  - [ ] 2.1 In `create_ha_sensor_for_QSSolar` (line ~400), replace `QSBaseSensorRestore` with `QSBaseSensorSolarScoreRestore` for the per-provider forecast score loop, passing the `provider` reference
- [ ] Task 3: Add `QSBaseSensorSolarActiveProviderRestore` subclass (AC: #2, #3, #4, #5)
  - [ ] 3.1 Create class in `sensor.py` extending `QSBaseSensorRestore`
  - [ ] 3.2 Accept `device: QSSolar` reference (already available via `self.device` from parent)
  - [ ] 3.3 In `async_added_to_hass`: call `await super().async_added_to_hass()` first, then if `self._attr_native_value` is a non-empty string that exists as a key in `device.solar_forecast_providers`, call `device._set_active_provider(name)` (or add a thin public wrapper — see dev notes)
  - [ ] 3.4 Skip if restored value is `unavailable`, `unknown`, empty, or not a configured provider name
  - [ ] 3.5 Do NOT set `_provider_mode` — only update `_active_provider_name`
- [ ] Task 4: Wire active provider sensor to new subclass (AC: #2)
  - [ ] 4.1 In `create_ha_sensor_for_QSSolar` (line ~412), replace `QSBaseSensorRestore` with `QSBaseSensorSolarActiveProviderRestore` for the active provider sensor
- [ ] Task 5: Tests for score hydration (AC: #1, #3, #4, #6)
  - [ ] 5.1 Test: numeric restore value hydrates `provider.score` when `score is None`
  - [ ] 5.2 Test: skip when `provider.score` is already set (don't clobber)
  - [ ] 5.3 Test: skip when restored value is `unavailable` or `unknown`
  - [ ] 5.4 Test: skip when restored value is non-numeric string
  - [ ] 5.5 Test: skip when no last extra data available
- [ ] Task 6: Tests for active provider hydration (AC: #2, #3, #4, #5, #6)
  - [ ] 6.1 Test: valid provider name from restore updates `device.active_provider_name`
  - [ ] 6.2 Test: unknown provider name (not in dict) is skipped — keeps default
  - [ ] 6.3 Test: `unavailable` / `unknown` / empty string skipped
  - [ ] 6.4 Test: does NOT change `_provider_mode`
  - [ ] 6.5 Test: skip when no last extra data available
- [ ] Task 7: Quality gate (AC: #6)
  - [ ] 7.1 Run `python scripts/qs/quality_gate.py` — pytest 100% coverage + ruff + mypy + translations

## Dev Notes

### Architecture and source tree

All changes are in the **HA integration layer** (`ha_model/` and platform files) — no domain layer (`home_model/`) changes needed. This is correct because `QSSolarProvider.score` lives on the HA-bridge object, and `_set_active_provider` is on `QSSolar` (also HA layer).

**Files to modify:**

| File | What changes |
|------|-------------|
| `custom_components/quiet_solar/sensor.py` | Add two new `QSBaseSensorRestore` subclasses; wire them in `create_ha_sensor_for_QSSolar` |
| `tests/test_platform_sensor.py` | Add restore tests for both new subclasses |

**Files to read (not modify):**

| File | Why |
|------|-----|
| `custom_components/quiet_solar/ha_model/solar.py` | Understand `QSSolarProvider.score`, `_set_active_provider`, `set_provider_mode` |
| `custom_components/quiet_solar/const.py` | `SENSOR_SOLAR_FORECAST_SCORE_PREFIX`, `SENSOR_SOLAR_ACTIVE_PROVIDER` |

### Key implementation details

**Existing subclass pattern to follow** (see `QSBaseSensorForecastRestore` at `sensor.py:627-649`, `QSDeviceSensorData` at `sensor.py:652-676`):
- Override `async_added_to_hass`
- Call `await super().async_added_to_hass()` first (fills `_attr_native_value`)
- Then hydrate model from the restored value

**Score sensor constructor** — the existing score sensor entity is created at `sensor.py:400`:
```python
entities.append(QSBaseSensorRestore(data_handler=device.data_handler, device=device, description=score_sensor))
```
Replace `QSBaseSensorRestore` with `QSBaseSensorSolarScoreRestore`. The new class needs the `provider` reference. Pass it as an extra constructor arg (like `QSBaseSensorForecastRestore` stores its prober reference).

**Active provider — calling `_set_active_provider`**: This is a private method on `QSSolar` (lines 213-217). Two options:
- Option A: Call `device._set_active_provider(name)` directly from the sensor subclass. The sensor already holds `self.device` which is the `QSSolar` instance. This is within the same integration, so accessing a private method is acceptable.
- Option B: Add a public `restore_active_provider(name: str)` wrapper on `QSSolar` that validates the name and delegates to `_set_active_provider`. Slightly cleaner but more code.
- **Recommended: Option A** — keep it simple. The sensor is tightly coupled to the device already.

**Why NOT use `set_provider_mode(name)` for hydration**: `set_provider_mode` also changes `_provider_mode` to the provider name (manual pin). We only want to restore the last *resolved* active provider, not change the mode. The select entity will restore the mode separately.

**`QSExtraStoredData.native_value` is `str | None`** — the score is stored as a string representation of the float. Must parse with `float()`.

### Testing patterns

Follow the established restore test pattern in `test_platform_sensor.py` (lines 479-561):

```python
def _make_restore_sensor(cls=QSBaseSensorRestore):
    """Create a minimal sensor instance for restore testing."""
    ...

async def _restore_with_value(native_value, cls=QSBaseSensorRestore):
    """Helper: restore a sensor with given native_value."""
    sensor = _make_restore_sensor(cls)
    stored_data = QSExtraStoredData(native_value=native_value, native_attr={"some_attr": "val"})
    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = MagicMock()
        mock_extra.return_value.as_dict.return_value = stored_data.as_dict()
        await sensor.async_added_to_hass()
    return sensor
```

For the score tests, the `_make_restore_sensor` helper will need to accept the new class and provide a mock `QSSolarProvider` with a `score` attribute. For active provider tests, the mock `device` needs `solar_forecast_providers` dict and `_set_active_provider` method (or use a real `QSSolar` if feasible).

### What NOT to change

- Do NOT modify `home_model/` — domain boundary stays clean
- Do NOT persist `_last_scoring_half_day` or scores to disk (deferred/cancelled per plan)
- Do NOT change the select entity or `set_provider_mode` behavior
- Do NOT change `compute_score` or `_run_scoring_cycle`
- Do NOT add `qs_is_none_unavailable=False` to existing sensors
- Do NOT modify `async_update_callback` — the existing None-to-Unavailable logic is correct once scores are hydrated

### Project Structure Notes

- Two new classes added to `sensor.py` — follows existing subclass pattern (3 existing subclasses already)
- Tests added to existing `test_platform_sensor.py` — follows established restore test section
- No new files, no new constants, no config changes

### References

- [Source: custom_components/quiet_solar/sensor.py#QSBaseSensorRestore] Lines 595-620 — base restore class
- [Source: custom_components/quiet_solar/sensor.py#QSBaseSensorForecastRestore] Lines 627-649 — subclass pattern to follow
- [Source: custom_components/quiet_solar/sensor.py#create_ha_sensor_for_QSSolar] Lines 387-412 — sensor creation to modify
- [Source: custom_components/quiet_solar/sensor.py#async_update_callback] Lines 537-566 — None/Unavailable handling
- [Source: custom_components/quiet_solar/ha_model/solar.py#__init__] Line 152 — active provider default
- [Source: custom_components/quiet_solar/ha_model/solar.py#_set_active_provider] Lines 213-217 — private switch method
- [Source: custom_components/quiet_solar/ha_model/solar.py#QSSolarProvider.score] Line 386 — score attribute
- [Source: tests/test_platform_sensor.py#restore_tests] Lines 479-561 — test patterns

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
