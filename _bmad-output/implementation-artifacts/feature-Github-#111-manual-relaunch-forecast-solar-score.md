# Manual Relaunch Forecast Solar Provider Score Computation

Status: done
issue: 111
branch: "QS_111"

## Story

As a Quiet Solar user with multiple solar forecast providers,
I want a button on the solar plant device to manually trigger forecast score recomputation,
so that I can refresh the provider accuracy scores on demand without waiting for the next automatic half-day boundary (00:00 / 12:00).

## Context

The forecast scoring system (`QSSolar._run_scoring_cycle`) computes MAE scores for each solar forecast provider at half-day boundaries (00:00 and 12:00 local time). The `_last_scoring_half_day` guard prevents re-runs within the same half-day window. Users currently have no way to force a recomputation — for example after a provider API outage resolves, or after manually resetting history.

The project already has button entities on QSHome (reset history, debug dump, dashboard regeneration) and on chargers/cars (force charge, add default charge). There is no button factory for QSSolar yet — one must be created.

## Acceptance Criteria

1. **New button entity**: A button entity named "Recompute forecast scores" is created on the solar plant device (`QSSolar`).
2. **Triggers scoring**: Pressing the button calls a method on `QSSolar` that resets the `_last_scoring_half_day` guard and runs `_run_scoring_cycle(now)` for all providers, bypassing the half-day throttle.
3. **Score sensors update**: After pressing the button, the per-provider `qs_solar_forecast_score_*` sensors reflect the freshly computed scores.
4. **Force update**: After the scoring cycle completes, `force_update_all()` is called (already handled by `QSButtonEntity.async_press`).
5. **Dashboard integration**: The button appears in the solar device section of both dashboard templates (standard HA and custom).
6. **Translations**: The button has a proper translation entry in `strings.json`.
7. **100% test coverage**: All new code paths are covered by tests.

## Tasks / Subtasks

- [x] Task 1: Add constant and domain method (AC: 1, 2)
  - [x] 1.1: Add `BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES = "qs_solar_recompute_forecast_scores"` to `const.py` (after the existing `BUTTON_HOME_*` block, near line 277)
  - [x] 1.2: Add `force_scoring_cycle(self, time: datetime | None = None)` method to `QSSolar` in `ha_model/solar.py`. This method resets `self._last_scoring_half_day = None` and then calls `self._run_scoring_cycle(time)`. If `time` is None, use `dt_util.utcnow()`.

- [x] Task 2: Create button entity for QSSolar (AC: 1, 3)
  - [x] 2.1: Import `QSSolar` in `button.py`
  - [x] 2.2: Import `BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES` in `button.py`
  - [x] 2.3: Create `create_ha_button_for_QSSolar(device: QSSolar)` function that creates a single button with:
    - `key=BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES`
    - `translation_key=BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES`
    - `async_press=lambda x: x.device.force_scoring_cycle()`
  - [x] 2.4: Add `isinstance(device, QSSolar)` check in `create_ha_button()` dispatcher (after the `QSHome` check)

- [x] Task 3: Add translation (AC: 6)
  - [x] 3.1: Add entry in `strings.json` under `entity.button`:
    ```json
    "qs_solar_recompute_forecast_scores": {
      "name": "Recompute forecast scores"
    }
    ```
  - [x] 3.2: Run `bash scripts/generate-translations.sh` to sync `translations/en.json`

- [x] Task 4: Dashboard integration (AC: 5)
  - [x] 4.1: In `quiet_solar_dashboard_template_standard_ha.yaml.j2`, add the button entity in the solar device section (after the score sensors loop around line 464):
    ```jinja2
    {%- set ha_entity = device.ha_entities.get("qs_solar_recompute_forecast_scores") %}
    {%- if ha_entity != None %}
    {{ "- entity: " + ha_entity.entity_id }}
    {{ "  name: " + "'" + ha_entity.name + "'" }}
    {%- endif %}
    ```
  - [x] 4.2: Apply the same pattern in `quiet_solar_dashboard_template.yaml.j2` if it has a solar section

- [x] Task 5: Tests (AC: 7)
  - [x] 5.1: Test `force_scoring_cycle()` resets `_last_scoring_half_day` and calls `_run_scoring_cycle` — add to `tests/test_solar_forecast_scoring.py`
  - [x] 5.2: Test `force_scoring_cycle()` works even when scoring already ran this half-day (the guard was set)
  - [x] 5.3: Test `create_ha_button_for_QSSolar()` returns exactly 1 button entity with correct key — add to `tests/test_platform_button.py`
  - [x] 5.4: Test `create_ha_button()` dispatcher includes QSSolar buttons when given a QSSolar device
  - [x] 5.5: Test the button's `async_press` lambda calls `force_scoring_cycle()` on the device

## Dev Notes

### Architecture

This feature lives entirely in the **HA integration layer** (`ha_model/solar.py`, `button.py`, `sensor.py`, dashboard templates). The `force_scoring_cycle` method is on `QSSolar` (HA layer), not in `home_model/` — it orchestrates existing scoring logic that's already on `QSSolar`.

No domain-layer changes are needed. The scoring logic (`_run_scoring_cycle`, `compute_score`) already exists.

### Key design decisions

- **Method on QSSolar, not QSHome**: The scoring cycle is owned by `QSSolar`. While we could add a button on QSHome that delegates to `solar_plant.force_scoring_cycle()`, it's cleaner to put the button directly on the solar device — that's where the score sensors live.
- **Reset guard, then run**: `force_scoring_cycle` sets `_last_scoring_half_day = None` before calling `_run_scoring_cycle`, ensuring the throttle check passes regardless of when the button is pressed.
- **No availability gate**: The button is always available when the solar device exists and has providers. Unlike car charge buttons, there's no precondition that could make scoring impossible (it gracefully handles missing data by returning None scores).

### Existing patterns to follow

- **Button factory pattern**: See `create_ha_button_for_QSHome()` in `button.py:35-78` — simple list of `QSButtonEntityDescription` instances wrapped in `QSButtonEntity`.
- **Dispatcher pattern**: `create_ha_button()` at `button.py:178-196` uses `isinstance` checks to call device-specific factories.
- **Constants**: All button keys in `const.py:273-289` follow `BUTTON_{DEVICE}_{ACTION}` naming.
- **Translations**: `strings.json` button section at line 831-864 — each key maps to `{"name": "..."}`.
- **Dashboard template**: Solar section in standard template at lines 443-464 — entities listed by key lookup.

### Files to modify

| File | Change |
|------|--------|
| `custom_components/quiet_solar/const.py` | Add `BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES` constant |
| `custom_components/quiet_solar/ha_model/solar.py` | Add `force_scoring_cycle()` method to `QSSolar` |
| `custom_components/quiet_solar/button.py` | Add `create_ha_button_for_QSSolar()`, register in dispatcher |
| `custom_components/quiet_solar/strings.json` | Add button translation entry |
| `custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` | Add button to solar card |
| `custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2` | Add button to solar card (if applicable) |
| `tests/test_solar_forecast_scoring.py` | Test `force_scoring_cycle()` |
| `tests/test_platform_button.py` | Test QSSolar button creation and dispatch |

### References

- [Source: ha_model/solar.py:82-170] — `QSSolar.__init__()`, provider setup, `_last_scoring_half_day`
- [Source: ha_model/solar.py:270-317] — `update_forecast()`, `_run_scoring_cycle()`, `_scoring_half_day()`
- [Source: button.py:35-78] — `create_ha_button_for_QSHome()` pattern
- [Source: button.py:178-196] — `create_ha_button()` dispatcher
- [Source: button.py:232-284] — `QSButtonEntityDescription` and `QSButtonEntity` classes
- [Source: sensor.py:387-400] — per-provider score sensor creation
- [Source: const.py:273-289] — existing button constants
- [Source: strings.json:831-864] — existing button translations
- [Source: ui/quiet_solar_dashboard_template_standard_ha.yaml.j2:443-464] — solar dashboard section

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None

### Completion Notes List
- All 5 tasks completed with TDD approach (red-green-refactor)
- All quality gates pass: ruff format, ruff lint, mypy, translations, pytest 100% coverage
- No deviations from the original spec

### File List
- `custom_components/quiet_solar/const.py` — added `BUTTON_SOLAR_RECOMPUTE_FORECAST_SCORES`
- `custom_components/quiet_solar/ha_model/solar.py` — added `force_scoring_cycle()` method, `dt_util` import
- `custom_components/quiet_solar/button.py` — added `create_ha_button_for_QSSolar()`, `QSSolar` dispatch in `create_ha_button()`
- `custom_components/quiet_solar/strings.json` — added button translation
- `custom_components/quiet_solar/translations/en.json` — auto-generated
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2` — added button to solar section
- `custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2` — added button to solar section
- `tests/test_solar_forecast_scoring.py` — 3 tests for `force_scoring_cycle()`
- `tests/test_platform_button.py` — 3 tests for QSSolar button creation, dispatch, and press
