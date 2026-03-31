# Bug Fix: Solar provider select not persisted after reboot

Status: ready-for-dev
issue: 90
branch: "QS_90"

## Story

As a Quiet Solar user,
I want the solar forecast provider select entity to retain my choice after a Home Assistant reboot,
so that I do not have to re-select my preferred provider every time the system restarts.

## Bug Description

The select entity for choosing the solar forecast provider (or "auto") reverts to the default value after a Home Assistant reboot. Users who explicitly select a specific provider find the entity has switched back to "auto" on restart.

### Entity Involved

- **Entity key:** `qs_solar_provider_mode` (`SELECT_SOLAR_PROVIDER_MODE` in `const.py:235`)
- **Entity class:** `QSUserOverrideSelectRestore` (in `select.py:426-459`)
- **Device callback:** `QSSolar.set_provider_mode()` (in `ha_model/solar.py:204-211`)

### Persistence Mechanism

The entity uses `QSUserOverrideSelectRestore` which extends `QSBaseSelectRestore(QSBaseSelect, RestoreEntity)`. It stores the user's selection via HA's `ExtraStoredData` mechanism:

- `extra_restore_state_data` property (line 430) returns `QSExtraStoredDataSelect(self.user_selected_option)`
- `async_get_last_select_data()` (line 434) retrieves saved data on restore
- `async_added_to_hass()` (line 440) calls `super()` first, then restores the extra data and calls `device.set_provider_mode()`

### Root Cause Analysis

**Primary suspect — `async_update_callback` overrides restored state (select.py:344-354):**

```python
@callback
def async_update_callback(self, time: datetime) -> None:
    self._set_availabiltiy()
    self._attr_options = self.get_available_options()
    self._attr_current_option = self.get_current_option()  # reads device.provider_mode
    self.async_write_ha_state()
```

This callback is inherited from `QSBaseSelect` and `QSUserOverrideSelectRestore` does **not** override it. On every update tick (every ~4-7s), it:
1. Reads `device.provider_mode` directly
2. Overwrites `_attr_current_option`
3. Writes HA state (which includes `extra_restore_state_data`)

The `user_selected_option` attribute is only set by `async_select_option()` (user UI action or restore) but NOT by the periodic callback. If the callback fires after the device has been re-initialized (with default "auto") but **before** `async_added_to_hass()` restores the saved value, the extra data could be flushed with `user_selected_option = None`.

**Secondary suspects:**

1. **Initialization ordering (select.py:356-361):** `QSBaseSelect.async_added_to_hass()` sets `_attr_current_option = self.get_current_option()` (device default "auto") and calls `super().async_added_to_hass()` which registers the entity in HA's state machine. Between this registration and the extra data restore in `QSUserOverrideSelectRestore.async_added_to_hass()` (line 444), there is a window where state could be persisted with stale values.

2. **Provider name validation (solar.py:207):** `set_provider_mode()` silently falls back to "auto" if the saved provider name doesn't match any entry in `self.solar_forecast_providers`. If provider names change between reboots (unlikely but possible), the restored selection would be discarded.

3. **Missing `None` → default coercion (select.py:451-454):** When the restored `user_selected_option` is `None`, the code passes `None` to `async_select_option()` which then calls `device.set_provider_mode(None)` → falls back to "auto". But `_attr_current_option` remains `None` until the next `async_update_callback`, causing the entity state to briefly be `None`/unknown.

### Comparison with Working Selects

Other select entities (car stale mode, off-grid mode, bistate mode) use `QSSimpleSelectRestore` which restores via `async_get_last_state()` (basic HA state). This simpler mechanism avoids the `ExtraStoredData` complexity and may explain why those selects persist correctly while the solar provider does not.

## Acceptance Criteria

1. After HA reboot, the solar provider select entity retains the user's last explicit selection
2. Selecting a specific provider, rebooting, and checking the entity shows the same provider
3. Selecting "auto", rebooting, and checking the entity shows "auto"
4. The `user_selected_option` in `extra_restore_state_data` matches the displayed entity state at all times
5. The `async_update_callback` does not overwrite restored or user-selected state with stale device defaults
6. If a previously selected provider no longer exists after reboot, the entity falls back to "auto" gracefully and logs a warning

## Tasks / Subtasks

- [ ] Task 1: Reproduce and confirm the bug (AC: 1, 2)
  - [ ] 1.1 Add a test that creates a `QSUserOverrideSelectRestore` solar provider entity, sets a provider, simulates a reboot (save extra data, reinitialize, call `async_added_to_hass`), and checks the restored state
  - [ ] 1.2 Verify the test fails with current code (confirms the bug)

- [ ] Task 2: Fix persistence — ensure `async_update_callback` respects user override (AC: 4, 5)
  - [ ] 2.1 Override `async_update_callback` in `QSUserOverrideSelectRestore` to read from `user_selected_option` instead of blindly from `device.provider_mode` — when `user_selected_option` is set, it takes precedence
  - [ ] 2.2 Alternatively, ensure `device.provider_mode` is always in sync with `user_selected_option` so the callback reads the correct value

- [ ] Task 3: Fix restore initialization (AC: 1, 2, 3)
  - [ ] 3.1 In `async_added_to_hass`, handle `None` restored value by applying `qs_default_option` (same pattern as `QSSimpleSelectRestore` lines 393-397)
  - [ ] 3.2 After restoring `user_selected_option`, ensure `device.set_provider_mode()` receives the correct value and the device state is in sync before `async_write_ha_state()` is called

- [ ] Task 4: Add graceful fallback for missing providers (AC: 6)
  - [ ] 4.1 In `async_added_to_hass`, after restoring `user_selected_option`, check if the value is still in the current `options` list. If not, log a warning and fall back to `qs_default_option` ("auto")

- [ ] Task 5: Test coverage (AC: 1-6)
  - [ ] 5.1 Test: select provider, save state, restore → restored value matches
  - [ ] 5.2 Test: select "auto", save state, restore → "auto" is preserved
  - [ ] 5.3 Test: `async_update_callback` does not overwrite user selection
  - [ ] 5.4 Test: provider name no longer valid after restore → graceful fallback to "auto" with log warning
  - [ ] 5.5 Test: first boot (no saved state) → defaults to "auto"

## Dev Notes

### Architecture Constraints

- **Two-layer boundary:** `ha_model/solar.py` (HA bridge) and `select.py` (HA entity) are both in the HA layer. No domain boundary concerns for this fix.
- **Logging:** Use lazy `%s` formatting, no f-strings in log calls, no trailing periods.
- **Constants:** `SELECT_SOLAR_PROVIDER_MODE` and `SOLAR_PROVIDER_MODE_AUTO` are in `const.py` — use them, don't hardcode.
- **Translations:** If adding new log messages visible to users, edit `strings.json` not `translations/en.json`.

### Key Files

| File | Lines | Role |
|------|-------|------|
| `select.py` | 191-212 | `create_ha_select_for_QSSolar()` — entity creation |
| `select.py` | 271-364 | `QSBaseSelect` — base class with `async_update_callback` |
| `select.py` | 366-399 | `QSSimpleSelectRestore` — working persistence pattern to reference |
| `select.py` | 402-459 | `QSUserOverrideSelectRestore` + `QSExtraStoredDataSelect` — the bug area |
| `ha_model/solar.py` | 82-211 | `QSSolar` — device-side `set_provider_mode()` and init |
| `const.py` | 101, 235 | Constants for provider mode and select key |
| `entity.py` | 128-134 | `QSDeviceEntity.async_added_to_hass()` — base entity setup |

### Testing Patterns

- Use test factories from `tests/factories.py` for creating domain objects
- Use mock configs from `tests/ha_tests/const.py` for HA integration tests
- For restore entity tests, mock `async_get_last_extra_data()` to return saved state
- Reference `bug-Github-#84` (solar forecast scores/prober persistence) for a prior persistence fix pattern in the same codebase

### Previous Story Intelligence

`bug-Github-#84` fixed persistence for `QSforecastValueSensor` using `ExtraStoredData` — a very similar pattern. That fix successfully restored forecast prober data across reboots. The key difference: that fix created a new `QSBaseSensorForecastRestore` class. The solar provider select already has `QSUserOverrideSelectRestore` but the restore path may not correctly handle the interaction between `async_update_callback` and `extra_restore_state_data`.

### Project Structure Notes

- All code is within `custom_components/quiet_solar/` — HA integration layer
- Select entity persistence uses HA's `RestoreEntity` + `ExtraStoredData` mechanism from `homeassistant.helpers.restore_state`
- No domain layer changes needed — this is entirely an HA entity persistence bug

### References

- [Source: select.py:426-459] — `QSUserOverrideSelectRestore` persistence class
- [Source: select.py:344-354] — `async_update_callback` that may overwrite state
- [Source: select.py:379-399] — `QSSimpleSelectRestore` working pattern
- [Source: ha_model/solar.py:204-211] — `set_provider_mode()` with fallback logic
- [Source: _bmad-output/project-context.md] — project rules and testing standards

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Story created from user bug report: solar provider select not persisted after reboot
- Thorough analysis of persistence chain: entity → ExtraStoredData → device → callback loop
- Identified primary suspect: `async_update_callback` not respecting user override
- Cross-referenced with working `QSSimpleSelectRestore` pattern

### File List

- `custom_components/quiet_solar/select.py`
- `custom_components/quiet_solar/ha_model/solar.py`
- `custom_components/quiet_solar/const.py`
- `custom_components/quiet_solar/entity.py`
