# Story 1.10: Lazy Logging Sweep (f-string cleanup)

Status: ready-for-dev

## Story

As TheAdmin,
I want all logging calls to use lazy formatting (`%s` style) instead of f-strings,
So that the codebase follows HA logging guidelines and avoids unnecessary string interpolation when log levels are disabled.

## Acceptance Criteria

1. **Given** the codebase contains `_LOGGER.<level>(f"...")` calls
   **When** the sweep is complete
   **Then** all logging calls use `_LOGGER.<level>("... %s", var)` lazy formatting
   **And** no f-string log calls remain in `custom_components/quiet_solar/`
   **And** all existing tests still pass
   **And** `ruff check`, `ruff format --check`, and `mypy` all pass

## Tasks / Subtasks

- [ ] Task 1: Convert f-string log calls in `ha_model/charger.py` (69 occurrences) (AC: #1)
  - [ ] 1.1 Convert all `_LOGGER.<level>(f"...")` to `_LOGGER.<level>("... %s", var)` in charger.py
  - [ ] 1.2 Run tests for charger: `pytest tests/ -k charger -q`
  - [ ] 1.3 Run quality gates

- [ ] Task 2: Convert f-string log calls in `home_model/load.py` (24 occurrences) (AC: #1)
  - [ ] 2.1 Convert all f-string log calls in load.py
  - [ ] 2.2 Run tests: `pytest tests/ -k load -q`

- [ ] Task 3: Convert f-string log calls in `ha_model/home.py` (23 occurrences) (AC: #1)
  - [ ] 3.1 Convert all f-string log calls in home.py
  - [ ] 3.2 Run tests: `pytest tests/ -k home -q`

- [ ] Task 4: Convert f-string log calls in remaining ha_model files (19 occurrences) (AC: #1)
  - [ ] 4.1 `ha_model/car.py` (8)
  - [ ] 4.2 `ha_model/battery.py` (8)
  - [ ] 4.3 `ha_model/device.py` (4)
  - [ ] 4.4 `ha_model/person.py` (3)
  - [ ] 4.5 `ha_model/solar.py` (2)
  - [ ] 4.6 `ha_model/dynamic_group.py` (2)
  - [ ] 4.7 `ha_model/bistate_duration.py` (2)
  - [ ] 4.8 `ha_model/on_off_duration.py` (1)

- [ ] Task 5: Convert f-string log calls in remaining files (20 occurrences) (AC: #1)
  - [ ] 5.1 `home_model/constraints.py` (8)
  - [ ] 5.2 `__init__.py` (6)
  - [ ] 5.3 `switch.py` (2)
  - [ ] 5.4 `home_model/solver.py` (1)
  - [ ] 5.5 `number.py` (1)
  - [ ] 5.6 `time.py` (1)
  - [ ] 5.7 `select.py` (1)
  - [ ] 5.8 `button.py` (1)

- [ ] Task 6: Final verification (AC: #1)
  - [ ] 6.1 Run `grep -r '_LOGGER\.\w\+(f"' custom_components/quiet_solar/` and confirm zero matches
  - [ ] 6.2 Run full quality gates: pytest (100% coverage), ruff check, ruff format, mypy

## Dev Notes

### This is a mechanical refactoring — no logic changes

Every change follows the exact same pattern:
```python
# BEFORE
_LOGGER.debug(f"Some message {variable} and {other}")

# AFTER
_LOGGER.debug("Some message %s and %s", variable, other)
```

Do NOT change log levels, message text, or log call locations. Only change the string formatting mechanism.

### Conversion Rules

1. **Simple variables**: `f"message {var}"` -> `"message %s", var`
2. **Expressions**: `f"message {obj.attr}"` -> `"message %s", obj.attr`
3. **Multiple variables**: `f"message {a} and {b}"` -> `"message %s and %s", a, b`
4. **Format specs**: `f"value {x:.2f}"` -> `"value %.2f", x` (preserve format specs in %s syntax)
5. **Ternary/complex expressions**: extract to a variable if needed, or use `%s` and let Python's str() handle it
6. **exc_info parameter**: preserve `exc_info=True` as-is — it's a kwarg, not part of the format string

### Edge Cases to Watch

- `_LOGGER.error("...", exc_info=True)` — `exc_info` is a keyword argument, keep it after the format args
- Some log calls may use `%s` mixed with f-strings — convert the whole call to `%s`
- Watch for log calls that span multiple lines — convert the entire call
- `repr()` calls inside f-strings: `f"state={state!r}"` -> `"state=%r", state`

### What NOT to Do

- Do NOT change any log message text (wording, punctuation, etc.)
- Do NOT change log levels (debug, info, warning, error)
- Do NOT add or remove log calls
- Do NOT refactor surrounding code
- Do NOT change any non-logging code
- Do NOT add comments or type annotations to code you didn't change

### File Inventory (167 total occurrences across 19 files)

| File | Count | Layer |
|------|-------|-------|
| `ha_model/charger.py` | 69 | HA bridge |
| `home_model/load.py` | 24 | Domain |
| `ha_model/home.py` | 23 | HA bridge |
| `ha_model/car.py` | 8 | HA bridge |
| `ha_model/battery.py` | 8 | HA bridge |
| `home_model/constraints.py` | 8 | Domain |
| `__init__.py` | 6 | Integration |
| `ha_model/device.py` | 4 | HA bridge |
| `ha_model/person.py` | 3 | HA bridge |
| `switch.py` | 2 | Platform |
| `ha_model/solar.py` | 2 | HA bridge |
| `ha_model/dynamic_group.py` | 2 | HA bridge |
| `ha_model/bistate_duration.py` | 2 | HA bridge |
| `number.py` | 1 | Platform |
| `time.py` | 1 | Platform |
| `select.py` | 1 | Platform |
| `button.py` | 1 | Platform |
| `ha_model/on_off_duration.py` | 1 | HA bridge |
| `home_model/solver.py` | 1 | Domain |

### Logging Rules (from project-context.md)

- Lazy logging with `%s`: `_LOGGER.warning("Message %s", variable)` — NOT f-strings
- No periods at end of log messages
- No integration names/domains in messages
- Use `debug` level for non-user-facing messages

### Architecture Constraints

- **Two-layer boundary**: domain logic (`home_model/`) NEVER imports `homeassistant.*`. This sweep touches both layers but only modifies logging format, not imports.
- Files in `home_model/` use plain Python `logging` module
- Files in `ha_model/` use the same `logging` module but may log HA-specific objects

### Previous Story Intelligence (Story 3.3)

Story 3.3 already fixed 2 f-string log calls in `ha_model/home.py` as part of the off-grid verification work:
- Line 959: `_LOGGER.warning(f"async_set_off_grid_mode: {off_grid}")` -> `_LOGGER.warning("async_set_off_grid_mode: %s", off_grid)`

Follow the exact same pattern for all remaining 167 calls.

### Verification Command

After all conversions, this grep must return zero matches:
```bash
grep -rn '_LOGGER\.\w\+(f"' custom_components/quiet_solar/
```

### References

- [Source: _bmad-output/project-context.md#Logging] — lazy logging rules
- [Source: _bmad-output/project-context.md#HA Anti-Patterns] — "NEVER use f-strings in logging calls"
- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.10] — story definition
- [Source: custom_components/quiet_solar/ha_model/home.py:959] — example fix from Story 3.3

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### Change Log

### File List
