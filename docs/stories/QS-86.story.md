# Bug #86: Saved Forecast Data Can't Be Restored (Numpy Types)

Status: done
issue: 86
branch: "QS_86"

## Story

As a Quiet Solar user,
I want forecast sensor data to be correctly serialized and restored across HA restarts,
so that I don't lose forecast history and don't see repeated "Bad data" errors in my logs.

## Bug Description

Home Assistant's `core.restore_state` storage fails to serialize sensor state because numpy scalar types (`numpy.int32`, `numpy.float64`) leak into `native_value` and `extra_restored_data` attributes. HA's JSON serializer only accepts native Python types (`int`, `float`, `str`, etc.) and rejects numpy scalars, causing repeated "Bad data" errors (33+ occurrences per restart cycle).

### Observed Behavior

```
Error writing config for core.restore_state: Bad data at
  $.data[201].extra_data.native_value=466(<class 'numpy.int32'>),
  $.data[202].extra_data.native_value=347(<class 'numpy.int32'>),
  $.data[202].extra_data.native_attr._qs_prober_data[0][1]=265(<class 'numpy.int32'>),
  $.data[202].extra_data.native_attr._qs_prober_data[1][1]=347(<class 'numpy.int32'>),
  ...
```

Affected paths:
- `extra_data.native_value` — forecast sensor values are `numpy.int32`
- `extra_data.native_attr._qs_prober_data[N][1]` — prober data tuples contain `numpy.int32`

### Expected Behavior

All values exposed to HA state/attributes must be native Python types. Forecast values should be `int` or `float`, not `numpy.int32` or `numpy.float64`.

### Labels

bug, area:car, area:charger, area:ui, area:battery, area:config

## Root Cause Analysis

### Root Cause 1: Forecast buffer uses `dtype=np.int32`

**File:** `ha_model/home.py`, lines ~4131, ~4312

The forecast value buffers are initialized as:
```python
self.values = np.zeros((2, BUFFER_SIZE_IN_INTERVALS), dtype=np.int32)
```

Any indexing of these arrays returns `numpy.int32` scalars, not Python `int`.

### Root Cause 2: Forecast list built from raw numpy indexing

**File:** `ha_model/home.py`, `compute_now_forecast()` method (~line 3975-4018)

```python
prev_val = self.values[0][prev_val_idx]       # numpy.int32
forecast.append((forecast_time, forecast_values[i]))  # numpy scalar
```

Values are appended to the forecast list directly from numpy array indexing without conversion. These tuples end up in `self._current_forecast` and are serialized via `serialize_stored_values()`.

### Root Cause 3: Serialization doesn't convert types

**File:** `ha_model/home.py`, `serialize_stored_values()` (~line 176-178)

```python
def serialize_stored_values(self) -> list[list]:
    return [[t.isoformat(), v] for t, v in self._stored_values]
```

The value `v` (which may be a numpy scalar) is passed through without `int()` or `float()` conversion.

### Root Cause 4: `_combine_stored_forecast_values()` returns numpy arrays

**File:** `ha_model/home.py`, `_combine_stored_forecast_values()` (~line 3164-3182)

This method performs numpy operations and returns numpy arrays that propagate `np.int32` scalars into downstream forecast computations.

### Root Cause 5: Prober data (`push_and_get`) stores numpy scalars

**File:** `ha_model/home.py`, `push_and_get()` (~line 196-219)

Values extracted from forecast tuples are stored in `_stored_values` and later serialized as `_qs_prober_data` in sensor attributes, where HA encounters the numpy types.

## Acceptance Criteria

- [x] AC1: All `native_value` values returned by forecast sensors are native Python `int` or `float`, never numpy scalars
- [x] AC2: All `_qs_prober_data` attribute values are native Python types
- [x] AC3: `serialize_stored_values()` produces JSON-serializable output with no numpy types
- [x] AC4: No "Bad data" errors in HA logs related to numpy types after restart
- [x] AC5: Existing forecast accuracy and behavior is unchanged (values are numerically identical)
- [x] AC6: All quality gates pass (pytest 100% coverage, ruff, mypy, translations)

## Technical Design

### Approach: Convert at serialization boundaries

Rather than converting numpy types throughout the entire computation pipeline (which would hurt performance and risk subtle bugs), convert at the **output boundaries** where values flow from the numpy domain to HA's serialization domain.

### Fix Points

**F1: `serialize_stored_values()` — convert values on serialization**

In `QSforecastValueSensor.serialize_stored_values()`, convert each value to native Python type:
```python
def serialize_stored_values(self) -> list[list]:
    return [[t.isoformat(), int(v) if isinstance(v, (int, np.integer)) else float(v)]
            for t, v in self._stored_values]
```

**F2: `compute_now_forecast()` — convert on forecast list construction**

When building forecast tuples, convert numpy scalars to Python `int`:
```python
forecast.append((forecast_time, int(forecast_values[i])))
```

Apply at all append points in the method (~lines 4006, 4014, 4018).

**F3: `push_and_get()` — convert values entering `_stored_values`**

When storing values in `_stored_values`, ensure they are native Python types:
```python
self._stored_values.append((future_time, int(future_val)))
```

**F4: Sensor `native_value` property — ensure native types**

In the sensor property that returns `native_value`, add a conversion guard for the forecast value before returning it to HA.

### Fix Order

1. F2 (forecast construction) — primary source of numpy scalars
2. F3 (prober data storage) — secondary source
3. F1 (serialization) — safety net at the serialization boundary
4. F4 (native_value) — final guard at the HA interface

## Tasks

- [x] T1: Identify all exact locations in `ha_model/home.py` where numpy scalars enter forecast tuples and `_stored_values`
- [x] T2: Add `int()` conversion at forecast tuple construction points in `compute_now_forecast()`
- [x] T3: Add `int()` conversion in `push_and_get()` when storing values
- [x] T4: Update `serialize_stored_values()` to convert numpy types as safety net
- [x] T5: Verify `native_value` property returns native Python types
- [x] T6: Write/update tests covering numpy type conversion at each fix point
- [x] T7: Run quality gates (`python scripts/qs/quality_gate.py`)

## Files to Modify

- `custom_components/quiet_solar/ha_model/home.py` — primary fix target (forecast construction, serialization, prober data)
- `tests/` — test coverage for numpy type conversion

## Out of Scope

- Changing the internal numpy `dtype` of forecast buffers (performance-sensitive)
- Converting numpy types deep within the solver pipeline (not needed if boundaries are handled)
- Constraint serialization (`to_dict()`) — separate concern, no reports of failures there yet
