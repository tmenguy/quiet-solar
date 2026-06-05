---
title: AbstractDevice / AbstractLoad
slug: load-base
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
last_verified: 2026-06-05
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
  pending commands stack until ACK clears the slot.

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
  slot and relaunch counters) instead of retrying it against the
  override (review fix QS-256#02). Default False; bistate loads
  override it.
- `_restored_utc_datetime(value)` — restore-boundary parser for the
  stored override timestamps: tz-naive isoformat strings (legacy /
  hand-edited storage) are coerced to UTC so downstream datetime
  arithmetic never raises (review fix QS-256#02).
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
