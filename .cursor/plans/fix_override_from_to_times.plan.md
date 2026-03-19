---
name: Fix override From/To times
overview: "Fix two bugs causing wrong From/To times during override in Default: End+Duration mode. Bug 1 (From: --:--) is a parameter name typo in the backend. Bug 2 (To: 00:00) is the To value showing the override expiration time rather than the original constraint end."
todos:
  - id: fix-start-param
    content: "Fix parameter name bug: rename start_time=time to start_of_constraint=time in bistate_duration.py override constraint creation"
    status: pending
isProject: false
---

# Fix override From/To times

## Bug 1: From shows "--:--" during override

### Root cause

In [bistate_duration.py](custom_components/quiet_solar/ha_model/bistate_duration.py) line 320-333, the override constraint is created with a **wrong parameter name**:

```python
override_constraint = TimeBasedSimplePowerLoadConstraint(
    ...
    start_time=time,              # <--- BUG: wrong param name
    end_of_constraint=end_schedule,
    ...
)
```

The constructor chain is:

- `TimeBasedSimplePowerLoadConstraint.__init__(**kwargs)` -> `MultiStepsPowerLoadConstraint.__init__(power_steps, power, **kwargs)` -> `LoadConstraint.__init__(..., start_of_constraint=None, ..., **kwargs)`

The constructor expects `start_of_constraint`, not `start_time`. Because `LoadConstraint.__init__` has `**kwargs` at the end ([constraints.py](custom_components/quiet_solar/home_model/constraints.py) line 73), `start_time=time` is **silently absorbed** into `**kwargs` and discarded. The `start_of_constraint` parameter then defaults to `None`, which becomes `DATETIME_MIN_UTC` ([constraints.py](custom_components/quiet_solar/home_model/constraints.py) lines 109-110).

Later, in `update_live_constraints` ([load.py](custom_components/quiet_solar/home_model/load.py) lines 1382-1387):

```python
if c.current_start_of_constraint > DATETIME_MIN_UTC:
    self.next_or_current_constraint_start_time = c.current_start_of_constraint
```

Since `current_start_of_constraint == DATETIME_MIN_UTC`, the `>` check is `False`, so `next_or_current_constraint_start_time` stays `None`. The sensor `qs_next_or_current_constraint_start_time` then returns `"--:--"` via `get_readable_date_string(None)`.

Note: the non-override constraint creation at line 456-467 of the SAME file correctly uses `start_of_constraint=ct.start_schedule`. So this bug is specific to the override path.

### Fix

In [bistate_duration.py](custom_components/quiet_solar/ha_model/bistate_duration.py) line 328, rename `start_time=time` to `start_of_constraint=time`.

## Bug 2: To shows "00:00" (or near-midnight) during override

### Analysis

This is **not a code bug** -- it is working as designed, but the displayed value is confusing.

The override constraint is created with `end_of_constraint = time + timedelta(seconds=(3600.0 * self.override_duration))`. The default `override_duration` is 8 hours (from `MAX_USER_OVERRIDE_DURATION_S = 8*3600` divided by 3600).

So if you trigger an override at 16:00, the end is 00:00 next day. At 16:15, the end is 00:15. The "To:" field shows this override expiration time, which happens to be around midnight.

The constraints list is sorted by `end_of_constraint` ([load.py](custom_components/quiet_solar/home_model/load.py) line 1070). The override constraint (ending around midnight) typically has an earlier end than the next-day default constraint, so it ends up as `_constraints[0]` -- the one whose times are displayed.

**This is expected behavior**: during an override, the From/To shows the override's time window (start -> override expiration), not the original default schedule. The user sees the override's 8-hour window.

No code change needed for this -- it becomes clear once Bug 1 is fixed and the From time shows correctly (e.g. "From: 16:15 / To: 00:15" is self-explanatory).

## Summary


| Issue       | Cause                                                  | Fix                                    |
| ----------- | ------------------------------------------------------ | -------------------------------------- |
| From: --:-- | `start_time=time` should be `start_of_constraint=time` | One-line rename in bistate_duration.py |
| To: 00:00   | Override expires 8h after creation (~midnight)         | Expected behavior, no fix needed       |


## File to change

- [bistate_duration.py](custom_components/quiet_solar/ha_model/bistate_duration.py) line 328: rename `start_time=time` to `start_of_constraint=time`

