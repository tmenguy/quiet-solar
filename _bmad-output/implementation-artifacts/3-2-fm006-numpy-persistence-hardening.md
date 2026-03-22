# Story 3.2: FM-006 — Numpy Persistence Hardening

Status: done

## Story

As TheDev,
I want numpy persistence failures to be caught with specific exception types and logged as warnings, with a health binary sensor exposing persistence state,
So that silent data loss is eliminated and persistence health is observable on the admin dashboard.

## Acceptance Criteria

1. **Given** a corrupted `.npy` file (partial write, disk full, permission error)
   **When** the system attempts to load it
   **Then** specific exceptions are caught (`OSError`, `ValueError`, `pickle.UnpicklingError`) instead of bare `except:`
   **And** a warning is logged identifying the corrupted file and exception type
   **And** the system continues without historical data using conservative estimates

2. **Given** a numpy save operation fails
   **When** the exception is caught
   **Then** specific exceptions are caught (`OSError`, `PermissionError`)
   **And** the failure is logged at warning level (not info)
   **And** the system continues operating

3. **Given** numpy persistence health is queried
   **When** TheAdmin checks the dashboard
   **Then** `binary_sensor.qs_persistence_health` reflects current persistence state
   **And** the sensor turns off when a load/save failure is detected

4. **Given** all logging in persistence code
   **When** the code is reviewed
   **Then** no f-strings are used in log calls (lazy logging with `%s` only)

## Tasks / Subtasks

- [x] Task 1: Replace bare `except:` with specific exception types in `QSSolarHistoryVals` (AC: #1, #2)
  - [x] 1.1 Fix `_save_values_to_file`: catch `OSError`, warning level, lazy logging
  - [x] 1.2 Fix `read_value`: catch `(OSError, ValueError, pickle.UnpicklingError)`, add warning log
  - [x] 1.3 Fix `_load_values_from_file` in `read_values_async`: same types, add warning log
  - [x] 1.4 Fix bare `except:` at float conversion in `init`: catch `(ValueError, TypeError)`
- [x] Task 2: Track persistence health on QSHome (AC: #3)
  - [x] 2.1 Add `qs_home_persistence_health` attribute to `QSHome` (default True)
  - [x] 2.2 Set False on load/save failure, True on successful save
  - [x] 2.3 Add `BINARY_SENSOR_HOME_PERSISTENCE_HEALTH` constant to `const.py`
  - [x] 2.4 Add binary sensor in `binary_sensor.py` using existing pattern
- [x] Task 3: Tests (AC: #1, #2, #3, #4)
  - [x] 3.1 Test: corrupted `.npy` file triggers specific exception catch + warning log
  - [x] 3.2 Test: save failure to bad path triggers specific exception catch + warning log
  - [x] 3.3 Test: `binary_sensor.qs_persistence_health` created and reflects persistence state
  - [x] 3.4 Updated existing tests to assert warning logs on failure
  - [x] 3.5 100% coverage maintained (3962 tests pass)

## Dev Notes

### Exact Code Locations (4 bare `except:` clauses to fix)

All in `custom_components/quiet_solar/ha_model/home.py` within `QSSolarHistoryVals` class:

**1. `_save_values_to_file` — lines 3747-3752:**
```python
# CURRENT (broken):
try:
    np.save(path, values)
    _LOGGER.info(f"Write numpy SUCCESS for {path} for reset {for_reset}")
except:
    _LOGGER.info(f"Write numpy FAILED for {path} for reset {for_reset}")
    pass
```
Fix: catch `(OSError, PermissionError)`, change to `_LOGGER.warning("Write numpy failed for %s (reset=%s): %s", path, for_reset, e)`, fix success log too.

**2. `read_value` — lines 3764-3767:**
```python
# CURRENT (broken):
try:
    ret = np.load(self.file_path)
except:
    ret = None
```
Fix: catch `(OSError, ValueError, pickle.UnpicklingError)`, add `_LOGGER.warning("Read numpy failed for %s: %s", self.file_path, e)`.

**3. `_load_values_from_file` in `read_values_async` — lines 3776-3779:**
```python
# CURRENT (broken):
try:
    ret = np.load(path)
except:
    ret = None
```
Fix: same as #2 but uses `path` parameter.

**4. Float conversion in `init` — lines 4037-4044:**
```python
# CURRENT (broken):
try:
    ...
    value = float(s.state)
    ...
except:
    value = None
    _LOGGER.warning(...)
```
Fix: catch `(ValueError, TypeError)` — these are the only exceptions `float()` raises on bad input.

### Exception Types Rationale

For `np.load()`:
- `OSError` — file not found, permission denied, disk I/O error (covers `FileNotFoundError`, `PermissionError`)
- `ValueError` — corrupted `.npy` header, wrong format, shape mismatch
- `pickle.UnpicklingError` — corrupted pickle data within numpy format

For `np.save()`:
- `OSError` — disk full, permission denied, path not found (covers `FileNotFoundError`, `PermissionError`)
- `PermissionError` — listed explicitly for clarity (subclass of `OSError`, but makes intent clear)

Note: `PermissionError` IS a subclass of `OSError`, so catching `(OSError, PermissionError)` is technically redundant. You can catch just `OSError` for save. Catching both is harmless and documents intent.

### Binary Sensor Pattern (copy existing)

Follow the exact pattern from `BINARY_SENSOR_HOME_IS_OFF_GRID`:

1. **`const.py`** — add constant:
   ```python
   BINARY_SENSOR_HOME_PERSISTENCE_HEALTH = "qs_home_persistence_health"
   ```

2. **`binary_sensor.py`** — add to `create_ha_binary_sensor_for_QSHome()`:
   ```python
   persistence_health = QSBinarySensorEntityDescription(
       key=BINARY_SENSOR_HOME_PERSISTENCE_HEALTH,
       translation_key=BINARY_SENSOR_HOME_PERSISTENCE_HEALTH,
   )
   entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=persistence_health))
   ```

3. **`ha_model/home.py`** — add attribute to `QSHome`:
   The binary sensor reads `getattr(self.device, self.entity_description.key, False)`, so the attribute name must match the constant key: `qs_home_persistence_health`. Set it to `True` (healthy) by default, flip to `False` on any load/save failure, reset to `True` on successful save.

### Logging Rules (MUST follow)

- Lazy logging with `%s`: `_LOGGER.warning("Message %s", variable)` — NOT f-strings
- No periods at end of log messages
- No integration names/domains in messages
- Log failure once, recovery once (don't spam)
- Use `warning` level for persistence failures (not `info` — this is actionable)
- Include the exception in the log: `_LOGGER.warning("...: %s", e)`

### What NOT to Do

- Do NOT add retry logic — that's out of scope (FM-006 is P3, retry is for P1 stories)
- Do NOT add atomic write patterns (write-to-temp + rename) — that's a future enhancement
- Do NOT change the ring buffer data structure or persistence format
- Do NOT modify the `read_value()` sync method signature or return type
- Do NOT add `import pickle` at module level if it's not already imported — use `pickle.UnpicklingError` with a local import or catch it via the numpy exception hierarchy
- Do NOT change the init flow that handles shape validation (lines 3967-3972) — that's already correct

### Import Note for `pickle.UnpicklingError`

`pickle` is a stdlib module. To catch `pickle.UnpicklingError`, add `import pickle` at the top of `home.py` (it's not currently imported). Alternatively, numpy wraps pickle errors, so you could test whether `ValueError` already covers all numpy corruption cases. If so, `(OSError, ValueError)` may be sufficient without importing pickle. Verify by checking numpy source or testing with a genuinely corrupted file.

### Existing Test Coverage (build on these)

- `tests/ha_tests/test_home_history_vals.py` — `test_history_vals_save_and_read()`, `test_history_vals_init_loads_existing_values()`, `test_history_vals_init_bad_shape_resets()`
- `tests/ha_tests/test_home_extended_coverage.py` — `test_save_values_no_hass()`, `test_read_value_missing_file()`, `test_read_value_existing_file()`, `test_save_values_custom_path()`, `test_read_values_async_no_hass()`
- `tests/ha_tests/test_home_coverage.py` — `test_save_values_exception_handling()` (saves to `/nonexistent_dir/test.npy`)

The existing `test_save_values_exception_handling` and `test_read_value_missing_file` tests verify that exceptions are caught — update them to assert the specific exception types and that warning logs are emitted (use `caplog` fixture).

### Project Structure Notes

Files to modify:
- `custom_components/quiet_solar/ha_model/home.py` — fix 4 bare `except:` clauses, add `persistence_healthy` attribute
- `custom_components/quiet_solar/const.py` — add `BINARY_SENSOR_HOME_PERSISTENCE_HEALTH`
- `custom_components/quiet_solar/binary_sensor.py` — add persistence health binary sensor to QSHome
- `tests/ha_tests/test_home_coverage.py` — update/add persistence failure tests
- `tests/ha_tests/test_home_extended_coverage.py` — update/add persistence failure tests

No new files needed.

### References

- [Source: docs/failure-mode-catalog.md#FM-006] — failure mode definition, gaps G7
- [Source: _bmad-output/planning-artifacts/architecture.md#Decision 1] — AR5 resilience fallback table, numpy persistence row (line 574)
- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.2] — story definition and CC impact
- [Source: _bmad-output/project-context.md#Error Handling] — no bare `except Exception` rule (line 106)
- [Source: _bmad-output/project-context.md#Logging] — lazy logging rules (lines 110-116)
- [Source: custom_components/quiet_solar/ha_model/home.py:3744-3786] — numpy persistence code
- [Source: custom_components/quiet_solar/binary_sensor.py:40-55] — binary sensor pattern for QSHome

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Replaced 4 bare `except:` clauses with specific exception types in `QSSolarHistoryVals`
- Added warning-level logging for all numpy load/save failures (load was previously silent)
- Fixed f-string logging to lazy `%s` format in save method
- Added `qs_home_persistence_health` attribute to `QSHome`, tracked on save/load success/failure
- Added `binary_sensor.qs_home_persistence_health` binary sensor via existing QSHome pattern
- Added 10 new tests: corrupted file, truncated file (EOFError), missing file with log assertion, persistence health flag on read/write success/failure, read success restores health, async read success restores health, async read corruption, binary sensor creation
- Updated 3 existing tests to assert warning log output using `caplog`
- Code review fixes: parenthesized except tuple, EOFError in read paths, read success restores health, BinarySensorDeviceClass.PROBLEM
- 3965 tests pass at 100% coverage, all quality gates green

### Change Log

- 2026-03-22: Story 3.2 implemented — numpy persistence hardening
- 2026-03-22: Code review fixes — EOFError handling, read health recovery, device_class, except tuple syntax

### File List

Modified files:
- `custom_components/quiet_solar/ha_model/home.py` — 4 bare `except:` fixed, `qs_home_persistence_health` attribute added
- `custom_components/quiet_solar/const.py` — added `BINARY_SENSOR_HOME_PERSISTENCE_HEALTH`
- `custom_components/quiet_solar/binary_sensor.py` — added persistence health binary sensor to QSHome
- `tests/ha_tests/test_home_extended_coverage.py` — 7 new/updated persistence tests
- `tests/ha_tests/test_home_coverage.py` — updated save exception test with log assertion
- `tests/ha_tests/test_binary_sensor.py` — updated home binary sensor test to verify persistence health
