# Bug Fix: Solar provider select not persisted after reboot

Status: done
issue: 90
branch: "QS_90"

## Story

As a Quiet Solar user,
I want the solar forecast provider select entity to retain my choice after a Home Assistant reboot,
so that I do not have to re-select my preferred provider every time the system restarts.

## Bug Description

The select entity for choosing the solar forecast provider (or "auto") reverted to the default after a Home Assistant reboot.

### Root Cause

The entity used `QSUserOverrideSelectRestore` with HA's `ExtraStoredData` mechanism, which was unnecessarily complex for this use case. The `device.provider_mode` is a stable user-set value that doesn't change dynamically — the simpler `QSSimpleSelectRestore` (basic HA state restoration via `async_get_last_state()`) is sufficient, and is what every other select entity in the codebase uses successfully.

### Fix

Switched from `QSUserOverrideSelectRestore` to `QSSimpleSelectRestore` in `create_ha_select_for_QSSolar()`. Removed the now-unused `QSUserOverrideSelectRestore` and `QSExtraStoredDataSelect` classes and their tests.

## Acceptance Criteria

1. After HA reboot, the solar provider select entity retains the user's last selection
2. Selecting a specific provider, rebooting, and checking the entity shows the same provider
3. Selecting "auto", rebooting, and checking the entity shows "auto"
4. All quality gates pass (100% coverage, ruff, mypy, translations)

## Changes

- `select.py:208` — switched `QSUserOverrideSelectRestore` to `QSSimpleSelectRestore`
- `select.py` — removed dead `QSUserOverrideSelectRestore` and `QSExtraStoredDataSelect` classes, cleaned up unused imports (`asdict`, `ExtraStoredData`)
- `tests/test_platform_select.py` — removed tests for dead classes, cleaned up unused imports
- `tests/test_solar_forecast_resilience.py` — added `test_set_provider_mode_none_falls_back_to_auto` to cover `set_provider_mode(None)` fallback path (previously covered by removed tests)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- Root cause identified by user: unnecessary complexity using `QSUserOverrideSelectRestore` when `QSSimpleSelectRestore` suffices
- One-line fix plus dead code cleanup
- All quality gates pass
