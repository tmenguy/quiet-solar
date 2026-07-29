---
title: AbstractDevice / AbstractLoad
slug: load-base
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
last_verified: 2026-07-29
---

# AbstractDevice & AbstractLoad

## TL;DR

`AbstractDevice` is the base for all controllable devices. It owns
configuration, the command lifecycle (pending → launched → acked,
with stacking for busy devices), and switching-cost protection
(`num_max_on_off` daily budget plus a 10-minute hysteresis). It is
3-phase aware. `AbstractLoad` extends `AbstractDevice` and adds the
constraint-management surface (`get_for_solver_constraints()` is the
solver's entry point), plus green-only mode, user override state, and
external-control detection. **Both live in `home_model/load.py` —
strict zero-HA-import boundary.**

## When you need this concept

- Adding a new device type — you'll extend `AbstractLoad` (or, rarely,
  `AbstractDevice` if it doesn't participate in solving).
- Changing command lifecycle semantics.
- Touching switching-cost protection (the daily on/off budget +
  hysteresis pattern).
- Working on external-control detection or user-override handling.

## Core idea

**Behavioral contract**: `AbstractLoad` defines guarantees every
device type must honour. Device-specific tests validate *deviations*
from that contract — don't re-test the contract itself. This is a
contract-testing pattern.

Command lifecycle (`AbstractDevice`):

- `pending` — solver decided to run; queued for launch.
- `launched` / `running_command` — command dispatched via HA service
  call, ACK not yet observed.
- `acked` / `current_command` — `probe_if_command_set()` confirmed.
- Stacking: when a device is busy with a `running_command`, additional
  pending commands stack until ACK clears the slot — *unless* QS has
  lost control, in which case the newer command supersedes the stale
  one (see below).

Relaunch, escalation and supersession (`AbstractDevice`, QS-304):

- `check_and_relaunch_command(time)` is the whole retry lifecycle, and
  it is the *only* thing `QSHome.check_loads_commands` calls. Probe →
  ack-or-relaunch → escalate-or-re-arm. It returns
  `command_acked_or_good` for the caller's `all_ok` aggregation.
- **Saturating backoff, not a give-up.** `command_relaunch_delay_s()`
  returns `COMMAND_RELAUNCH_BASE_DELAY_S * min(n + 1,
  NUM_MAX_COMMAND_RELAUNCH)` — 50, 100, 150, 200, 250, 300 s and then
  300 s **forever**. Two distinct indices, deliberately named apart:
  the *delay* stops growing at `n == 5`; the *uncontrollable threshold*
  is `n >= 6` (1050 s cumulative). Before QS-304 the second index
  meant "give up", which bricked the load's command slot permanently.
- **One durable clock.** `unresponsive_since` is set once when the
  threshold is crossed (one ERROR, one push) and cleared by
  `_clear_unresponsive` — the single writer, so "one line in, one line
  out" cannot drift. It is deliberately NOT reset by
  `constraint_reset_and_reset_commands_if_needed`.
- **`is_uncontrollable` needs `running_command is not None`.** That
  conjunct is load-bearing, not defensive: several paths empty the slot
  with no successor (the `current_command == command` early-return, the
  override-suppression drop), and an `unresponsive_since`-only property
  would stay `True` forever with no retry left to clear it. If we are
  not waiting on anything, we are not uncontrollable.
- **Supersession + throttle.** A *differing* command against an
  uncontrollable load calls `abandon_running_command` and executes, at
  most once per `SUPERSEDE_MIN_INTERVAL_S` (300 s); inside the window it
  becomes `_stacked_command` (last-wins). An *equal* command is absorbed
  with no service call at any rung. A differing command against a
  *healthy* load is stacked, exactly as before.
- **`abandon_running_command` preserves the rung.**
  `running_command_num_relaunch` survives, along with
  `current_command` (= last *confirmed* command), `prev_command`,
  `num_on_off`, `unresponsive_since` and `_last_supersede_time`.
  Zeroing the rung would restart the ladder at 50 s after every
  supersede — ~1150 service calls/day instead of ~288.
- **Invariant:** anything meaning "we have lost control" reads
  `is_uncontrollable`, never `running_command_num_relaunch`. The
  counter is resettable; the clock is not.

Switching-cost protection (`AbstractDevice`):

- `num_max_on_off` — daily on/off budget.
- `CHANGE_ON_OFF_STATE_HYSTERESIS_S = 600` — minimum delay (10 min)
  between state changes for the same device.
- **Multi-pass adaptation**: try free transitions first; only spend
  the daily budget on the second pass.

3-phase awareness:

- Phase configuration tracked per device.
- Power → per-phase amperage conversion for budgeting checks.

## Key types / structures

- `AbstractDevice` — base. Config, lifecycle, switching cost,
  3-phase awareness.
- `AbstractLoad(AbstractDevice)` — adds constraint surface.
- `PilotedDevice(AbstractDevice)` — devices that pilot other devices
  (e.g., heat pump with aux heater). Tracks client list and per-slot
  demand counts.
- `get_for_solver_constraints()` — the solver's entry point for
  reading active constraints.
- `push_live_constraint(...)` — runtime constraint push.
- `push_agenda_constraints(...)` — calendar / schedule push.
- `external_user_initiated_state` — set when the device state
  changes without a command quiet-solar sent.
- `is_command_suppressed_by_override(time, command)` — hook checked at
  the `launch_command` drop point (after the stacked-command clear,
  before the same-command early-return): a suppressed command is
  DROPPED before `running_command` is set — no ack, no counter
  mutation, nothing for `check_commands` / `force_relaunch_command`
  to resurrect (QS-256). `force_relaunch_command` applies the same
  hook to a stale `running_command` and drops it (clears the running
  slot and relaunch counters — `current_command` is intentionally
  preserved as the last acked command of record, so a later
  non-suppressed launch can still compare against it) instead of
  retrying it against the override (review fix QS-256#02). Default
  False; bistate loads override it.
- `_restored_utc_datetime(value)` — restore-boundary parser for the
  stored override timestamps: tz-naive isoformat strings (legacy /
  hand-edited storage) are coerced to UTC so downstream datetime
  arithmetic never raises (review fix QS-256#02). The persisted payload
  keys themselves are the `STORAGE_KEY_*` constants in `const.py`
  (review fix QS-256#05).
- `unresponsive_since` / `is_uncontrollable` /
  `command_relaunch_delay_s()` / `abandon_running_command(time, reason)`
  / `check_and_relaunch_command(time)` / `_notify_unresponsive(time,
  command)` — the QS-304 lost-control surface. `_notify_unresponsive` is
  a documented no-op on `AbstractDevice` (the battery reaches the same
  driver and has no notification channel) and is overridden on
  `AbstractLoad` to push one `DEVICE_STATUS_CHANGE_ERROR`.
- `last_command_execution_time` — in-memory causality anchor, set
  only on real `execute_command` successes (via the shared
  `_anchor_causality_guard_if_executed` helper called from
  `launch_command` and `force_relaunch_command`, never on the
  probe-already-set branch) and
  initialized to "now" at storage restore when a `current_command` is
  restored. Never serialized. Cleared by `user_clean_and_reset`,
  which also clears ALL user-override fields (QS-256).

## Common mistakes

- Adding a device only in `ha_model/` without a `home_model/`
  counterpart. The solver can't see it — it's a ghost device.
- Calling `execute_command()` from a test and awaiting the ACK in
  the same call. ACK arrives asynchronously via
  `probe_if_command_set()`; tests need to advance time.
- Bypassing `num_max_on_off` for "important" devices. The whole
  hysteresis pattern depends on the budget being respected
  uniformly.
- Importing `homeassistant.*` into `home_model/load.py`. The two-
  layer boundary is non-negotiable.
- Keying "we have lost control" on `running_command_num_relaunch`.
  `abandon_running_command` no longer zeroes it, but `_ack_command`
  still does — read `is_uncontrollable` instead (QS-304).
- Adding a give-up that empties `running_command` without either
  clearing `unresponsive_since` (sensor pinned on, recovery edge
  unreachable) or deliberately keeping it (a probe that went
  *unavailable* is the device failing harder, not recovering — that is
  why the empty-slot clear is guarded on `current_command is not None`).
- Using a `MagicMock` as a load in a `check_loads_commands` /
  `update_loads` test. The resulting `TypeError` is swallowed by the
  per-load `except`, `all_ok` stays `True`, and your assertions never
  fire. Use `tests.factories.attach_minimal_load_to_home`.

## See also

- [ha-device-mixin.md](ha-device-mixin.md) — the HA-side counterpart.
- [constraints.md](constraints.md) — the constraint API.
- [piloted-device-and-heat-pump.md](piloted-device-and-heat-pump.md)
  — the `PilotedDevice` subclass.
- [external-control-detection.md](external-control-detection.md) — the
  external-state detection flow.
- [user-override.md](user-override.md) — user-originated state.
- [../principles/two-layer-boundary.md](../principles/two-layer-boundary.md)
  — why `home_model/` never imports HA.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the daily-budget + hysteresis pattern.
