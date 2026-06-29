---
title: Car SOC estimation
slug: car-soc-estimation
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/car.py
last_verified: 2026-06-29
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
half-up-rounds before clamping — a raw `number.set_value` can bypass the card's
finite/round guard. The manual-SOC **number entity** tracks
`qs_car_manual_soc_percent` every update cycle (`qs_track_device_value`), so a
runtime reset (plug-in / reset button / recovery) is reflected in the card.

## Accessors

- `get_car_charge_percent_raw_sensor` — the trusted-sensor read; efficiency
  learning and the charge callback use this so they never see the estimate.
- `_estimated_soc_percent` — `clamp(base + delta, 0, 100)`, or `None` with no
  base (pure-delta `+XX%`).
- `is_in_soc_estimation_mode` — True for a no-sensor car, in stale-percent
  mode, or with a manual base on a healthy API. **Drives the `*` /
  `is_soc_estimated` binary sensor**: the asterisk means "the SOC number is
  being extrapolated/overridden", so a fresh SOC shows no asterisk and the
  pure-delta stale case (no absolute estimate) still shows it.
- `is_soc_sensor_distrusted` — True only in stale-percent mode or with no SOC
  sensor. A manual override on a healthy sensor is **not** distrust — the
  charger's zero-power hardware-fault check still runs (it is gated on distrust,
  not on `is_in_soc_estimation_mode`).
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
  - Force-Not-Stale is treated as "sensor trusted": recovery may proceed even
    when the SOC sensor is time-stale (the user has asserted it is reliable).
  The tolerant compare uses half-away-from-zero rounding (`_round_half_up`), not
  Python's banker's rounding, so an exact `.5` reading is not mis-binned.
- **System-base recovery**: `_exit_stale_mode` clears the system base. It also
  clears the accumulator + cursor **only when no user override is active** — an
  override owns its own accumulator lifecycle, so a transient stale blip must
  not wipe its accumulated delta. The user base is never cleared by stale-mode
  exit.
- **Manual stale-detection reset** (`reset_car_api_stale_detection`): three
  manual user actions give the live API a fresh chance by calling
  `_exit_stale_mode` and clearing `_was_car_api_stale` (the *detected* state
  only — `car_stale_mode_override` is preserved, so a Force-stale override
  still wins): the car's clean-and-reset button (`user_clean_and_reset`, which
  also covers the departure auto-reset path), a manual
  `FORCE_CAR_NO_CHARGER_CONNECTED` select on the car, and a manual
  charger-side car allocation that displaces the currently attached car
  (`user_set_selected_car_by_name`, including `CHARGER_NO_CAR_CONNECTED`).
  Automatic detaches (`detach_car` on its own) deliberately do **not** reset —
  they keep the stale episode.
- Estimation is **orthogonal** to `is_car_effectively_stale`: a manual
  override on a healthy API marks the car *estimated* (asterisk) but **not**
  API-stale.

## Manual charger assignment vs a wrong location tracker (QS-265)

A *manual charger assignment* is the user explicitly attaching a car to a
charger; it is distinct from a *manual SOC value* (the override above). When a
manually-assigned car's location tracker wrongly reports "away" (or the plug
sensor reports unplugged) while the SOC sensor is live:

- `check_charger_assignment_contradiction(..., manual=True)` **trusts the
  user**: on a contradiction it sets
  `_car_api_inferred_home`/`_car_api_inferred_plugged` (so the car keeps being
  managed and charged) and logs **one WARNING per contradiction episode**
  (deduped on the dedicated `_manual_contradiction_logged` key). It does **not** mark the car stale and
  does **not** notify. A manually-assigned car's staleness therefore depends
  only on its SOC sensor (the SOC-only stale entry) plus the all-sensors-dead
  and force paths. With a fresh SOC the car is not in estimation mode, so
  `get_car_charge_percent` returns the live sensor and the constraint seed is
  the real SOC (never force-init at 0), with no asterisk.
- **Flag lifecycle** (single owner, tri-state). The same call reconciles the
  override against the raw reads each cycle: an explicit `False` sets it (+ the
  per-episode WARNING); affirmative `True` reads on **every available sensor**
  clear it (the override never outlives the contradiction — an away→home blip on
  an attached, never-stale car drops it and a later genuine unplug is honored,
  and the next contradiction re-arms the WARNING); a `None` (unavailable) read
  is "no new info" and **holds** the current override, so a single-cycle sensor
  flicker does not drop a still-valid override (R4-SF1). To make this robust it
  runs every cycle in `_update_car_api_staleness` **before** the SOC-only stale
  entry and independent of stale-percent mode, so a manual car still gets the
  override when a SOC-stale entry and a tracker contradiction coincide on one
  cycle. It is skipped only while fully API-stale (all sensors dead), where
  there is no reliable raw signal to reconcile against.
- `manual=False` (auto-attached by plug-time correlation): a contradiction
  takes **no action** — identity is only a heuristic, so it neither marks the
  car stale nor sets the inferred flags.
- **Recovery** (`can_exit_stale_percent_mode`): a user-originated assignment
  recovers on SOC freshness alone (`return not self._is_soc_sensor_stale(time)`),
  ignoring the possibly-wrong raw home/plug readings that gate the non-manual
  connected/not-connected branches. The branch sits **after** the genuine
  all-data-dead guard and is itself guarded on a SOC sensor existing, so a
  no-SOC-sensor car (whose `_is_soc_sensor_stale` is vacuously False) never
  short-circuits to permanent recovery.
- **15-minute ceiling on the override.** `_check_departure_auto_reset` reads the
  **raw** tracker, so a manually-assigned car whose tracker is *persistently*
  (wrongly) "away" is auto-reset after `CAR_NOT_HOME_AUTO_RESET_S` (15 min):
  `user_clean_and_reset` wipes the user-originated marker and the inferred
  flags, ending the manual trust and reverting to pre-fix behavior. This is the
  deliberate upper bound from the story's adversarial-review notes ("trusted
  until the existing unplug/displacement resets fire") — a genuinely-away car
  must not be charged forever on a stale manual pick. The ceiling is
  **tracker-only**: a plug-only contradiction (tracker genuinely home, plug
  unplugged) is not time-bounded here — it relies on the charger-side detach
  (`charger._check_plugged_val`, which consults `is_car_plugged()` only when its
  own plug sensor is inconclusive).
- **Force-Not-Stale drops the override.** Selecting Force-Not-Stale means "trust
  live data", so `_update_car_api_staleness` clears the inferred flags whenever
  that override is active — the manual home/plug override never outlives an
  explicit Force-Not-Stale, even for a never-stale car (R3-SF1).

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
