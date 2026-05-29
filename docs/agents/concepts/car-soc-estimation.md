---
title: Car SOC estimation
slug: car-soc-estimation
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/car.py
last_verified: 2026-05-30
---

# Car SOC estimation — the effective-SOC model

## TL;DR

A car's *effective* SOC is unified behind one accessor,
`QSCar.get_car_charge_percent`. When the SOC API is **failed**,
**inaccurate**, or **absent**, the car estimates SOC as `base + charged
energy since`, where the base is either a **manual entry** or the **last
valid sensor reading** captured at the fresh→stale edge. A genuine new
sensor reading always wins and clears the estimate.

## When you need this concept

- Touching `get_car_charge_percent` or any SOC read in the charge loop.
- Changing stale-percent-mode entry/exit or the manual-SOC entities.
- Reasoning about why the constraint seed (`car_initial_value`) is no
  longer forced to `0.0`.

## State model (persisted on `QSCar`)

- `_user_base_soc_value` — manually-entered baseline.
- `_last_valid_base_soc_value` — last valid real sensor value, captured at
  the fresh→stale transition (only when no user base exists).
- `_computed_added_delta_soc_percent` — accumulated, efficiency-aware
  percent added during the current plugged session (a **float**).
- `_user_base_soc_entry_sensor_value` — raw sensor value at the instant the
  user entered the manual value (recovery reference; may be `None`).
- `_user_base_soc_entry_api_stale` — the stale-percent state at manual entry;
  drives the 4-case recovery branch (`None` for pre-QS-243 blobs).
- `_delta_soc_last_integration_time` — **not persisted**; the dedicated
  integration cursor, re-anchored on reboot so downtime energy is not
  counted.

`user_set_manual_soc_percent` guards non-finite input (`math.isfinite`) and
`round()`s before clamping — a raw `number.set_value` can bypass the card's
finite/round guard.

## Accessors

- `get_car_charge_percent_raw_sensor` — the trusted-sensor read; efficiency
  learning and the charge callback use this so they never see the estimate.
- `_estimated_soc_percent` — `clamp(base + delta, 0, 100)`, or `None` with no
  base (pure-delta `+XX%`).
- `is_in_soc_estimation_mode` — True for a no-sensor car, in stale-percent
  mode, or with a manual base on a healthy API.
- `is_soc_sensor_distrusted` — True only in stale-percent mode or with no SOC
  sensor. A manual override on a healthy sensor is **not** distrust — the
  charger's zero-power hardware-fault check still runs (it is gated on distrust,
  not on `is_in_soc_estimation_mode`).
- `has_soc_estimate` — an absolute estimate exists (drives the `*` /
  `is_soc_estimated` sensor). Distinct from `is_in_soc_estimation_mode`:
  they diverge for the pure-delta case.
- `soc_integration_cursor` (property) + `accumulate_soc_delta(inc, time)` — the
  car's public accumulator API; the charger drives it through these instead of
  reaching into the underscore-private fields. `accumulate_soc_delta` clamps the
  pure-delta return to `[0, 100]`.

## Accumulator (charger callback is the sole writer)

`constraint_update_value_callback_soc` advances the car-level float
accumulator with its own cursor **every cycle** so sub-1%/cycle slices are
never lost and never double-counted across solver / constraint churn. The
first cycle (or post-reboot / post-base-set) only **anchors** the cursor. The
charger computes `inc` from `soc_integration_cursor` then calls
`accumulate_soc_delta(inc, time)`.

## Recovery and orthogonality

- **User-base recovery** (`_update_soc_estimation`, per cycle) is a 4-case
  state machine keyed on `_user_base_soc_entry_api_stale`:
  - *Case 1* — entered while stale: clear once the car has exited stale mode
    and a fresh valid value is available (live sensor wins).
  - *Case 2* — entered not-stale with a valid value: clear only when a fresh
    value *differs* from the entry reference (tolerant `round` compare); equal
    → keep base **and** delta.
  - *Case 3* — entered not-stale without a valid value: any valid value clears.
- **System-base recovery**: `_exit_stale_mode` clears the system base. It also
  clears the accumulator + cursor **only when no user override is active** — an
  override owns its own accumulator lifecycle, so a transient stale blip must
  not wipe its accumulated delta. The user base is never cleared by stale-mode
  exit.
- Estimation is **orthogonal** to `is_car_effectively_stale`: a manual
  override on a healthy API marks the car *estimated* (asterisk) but **not**
  API-stale.

## Capture at the fresh→stale edge

`_enter_stale_percent_mode(time)` wraps every stale-percent write site and
captures the last valid raw SOC into `_last_valid_base_soc_value` (zeroing
the accumulator) only on a genuine edge and only when no base exists. The
`for_init` restore path never captures — the persisted value is the truth.

## Gotchas

- `can_use_charge_percent_constraints` no longer requires a SOC sensor — a
  non-invited car with a true battery capacity estimates from energy.
- Clearing the estimate happens on `user_clean_and_reset` / the reset button,
  on the unplug edge, and on a **genuine plug-in** (charger.py
  `do_full_reset=True` branch). It is **never** in the low-level `car.reset()`:
  `attach_car` runs `reset()` on both a genuine plug-in and the boot re-attach,
  so only the charger can tell them apart. A plug-in resets the estimate (fresh
  charge session); a boot re-attach (`do_full_reset=False`) preserves the
  persisted estimate, so it survives an HA reboot.
