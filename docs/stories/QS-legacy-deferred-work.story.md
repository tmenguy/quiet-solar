# Deferred Work

## Deferred from: bug-Github-#84-fix-solar-forecast-scores-prober-persistence (2026-03-30)

- **Entity naming redundancy (Bug 3)**: `device_id = f"qs_{slugify(name)}_{device_type}"` produces `"qs_solar_solar"` for the solar device. Combined with forecast keys like `"qs_solar_forecast_15mn"`, entity IDs become `quiet_solar.qs_solar_solar_qs_solar_forecast_15mn`. Functional but confusing. Requires entity ID migration (HA `entity_registry.async_update_entity`) to avoid breaking existing users' automations and dashboards. Separate story recommended.

## Deferred from: code review of bug-Github-#64-pool-target-double-count-and-missing-handle (2026-03-29)

- No test for partial completion where `target_value != current_value` in pool metrics tests
- Day-boundary scenario untested with distinct `end_of_constraint` values for active vs completed constraints
- Multiple active constraints could double-count metrics in `QSPool.update_current_metrics` (by-design pool override sums all)
- Stale `_localTargetPct` not cleared on Reset button press in `qs-pool-card.js` (5-second drag state window)
- `handle.style.cursor` written to detached DOM node after async re-render in `qs-pool-card.js`
- `_isInteractingTarget` race with `set hass` re-render during drag in `qs-pool-card.js`
- `end_range` parameter of `update_current_metrics` has no test coverage
- `_last_completed_constraint` can be wiped by `constraint_reset_and_reset_commands_if_needed` before `update_current_metrics` reads it in the mode-off path
- No JS/frontend test coverage for pool card reset behavior (no JS test framework in project)
