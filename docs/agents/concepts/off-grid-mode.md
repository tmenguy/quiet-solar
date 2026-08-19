---
title: Off-grid mode
slug: off-grid-mode
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
last_verified: 2026-08-19
---

# Off-grid mode

## TL;DR

When the grid power sensor drops to zero, `QSHome` detects an outage
and transitions into `QSHomeMode.OFF_GRID`: it sheds non-mandatory
loads, restricts consumption to (solar + battery) capacity, and
broadcasts a high-priority alarm to every household member's mobile
app. `FORCED_OFF_GRID` is the admin-pinned variant (used during
inverter maintenance). The mode reverses automatically when grid
power returns; recovery is also broadcast.

## When you need this concept

- Touching grid-outage detection, the consumption-shedding policy,
  or the broadcast notification flow.
- Adding a new device whose behaviour must differ when off-grid.
- Working on `QSHomeMode` state transitions.
- Debugging "the system didn't shed load when the power went out"
  issues.

## Core idea

Off-grid mode lives in `ha_model/home.py` on `QSHome`. The detection
trigger is "grid power sensor drops to zero" (with a small debounce
to avoid false positives from sensor noise). Once tripped:

1. **Load shedding**: `FILLER` and `FILLER_AUTO` constraints are
   suspended. `MANDATORY` constraints remain — they're still
   honoured but capped at (solar + battery) available power.
2. **Battery policy switch**: battery's normal "save for evening
   arbitrage" goal is overridden by "keep the house running".
3. **Broadcast**: every household member's `mobile_app` notification
   service receives a high-priority alarm-channel message.

Recovery: when the grid sensor returns to a non-zero stable reading,
mode reverts to `NORMAL`, suspended `FILLER` constraints resume, and
a "grid restored" broadcast goes out.

`FORCED_OFF_GRID` is a manual variant: an admin toggle that pins the
mode for maintenance windows. It does not auto-recover.

## Key types / structures

- `QSHomeMode` enum: `NORMAL`, `OFF_GRID`, `FORCED_OFF_GRID`.
- Off-grid detection logic on `QSHome` (in the state polling cycle).
- Broadcast helper: routes through every configured `QSPerson`'s
  mobile-app entity.
- `finish_off_grid_switch(time)` gates the transition on
  `check_loads_commands(time)`: the switch only completes once **every**
  load reports its command confirmed-or-good. QS-304 tightened that
  contract in two ways:
  - a load whose command never converges keeps reporting not-good
    forever, so it holds the transition back indefinitely rather than
    being silently given up on;
  - a load whose command cycle **raises** now also falsifies `all_ok`.
    It is equally unconfirmed, and treating it as OK let the transition
    complete with an unlanded command.

  So an unresponsive or exploding load blocks the off-grid switch. That
  is deliberate — switching to island power while a load is still in an
  unknown state is the riskier failure.

  Not to be confused with `HOME_MODE_OFF`: that is the *managing* mode
  switch, and QS-304 releases every device's lost-control state on its
  `update_loads` early return. `OFF_GRID` is an island-power state in
  which QS keeps managing loads, so nothing is released there.

## Lifecycle

```text
Grid sensor steady at 0W (debounced) → mode = OFF_GRID
  ↓
suspend FILLER constraints
  ↓
broadcast alarm to all household members
  ↓
... grid restored ...
  ↓
mode = NORMAL → resume FILLER → broadcast recovery
```

## Common mistakes

- Forgetting to test recovery. The "outage detected" path is easy;
  the "grid restored" path is where bugs hide.
- Treating `FORCED_OFF_GRID` like `OFF_GRID`. The admin pinned it
  for a reason — auto-recovery would defeat the purpose.
- Skipping the debounce. A flickering grid sensor causes alarm
  storms.
- Trying to keep `FILLER_AUTO` loads running on battery. The whole
  point of shedding is to extend battery life during the outage.

## Battery outage reserve is a native inverter concern (not QS-managed)

Holding a slice of battery charge so the house survives the *instant*
the grid drops is **not** something quiet-solar implements — it is a
native inverter / backup-gateway setting, and that is a deliberate
decision (QS-264).

- **Software enforcement blacks out the transition.** Any QS-side way
  of holding a reserve — forcing the battery's max-discharge power to
  0, or raising a minimum-SOC floor — is still *in place* at the
  instant the grid fails. QS only learns of the outage after the fact
  (state change → service call → inverter reacts), so the reserve it
  "saved" is locked behind the very block it must then race to lift.
  The release at t=0 cannot be QS's job.
- **The backup gateway already does it, instantly.** Inverters with a
  backup gateway (e.g. Huawei SmartGuard / Backup Box) keep two floors:
  a higher **on-grid reserve** the battery will not discharge below
  while the grid is up, and a lower **deep-discharge floor** it runs
  down to in island mode. On grid loss the gateway islands and releases
  the reserve in hardware — sub-second, no controller action.
- **On gateway-managed setups the setpoints are read-only anyway.**
  With a Huawei EMMA in front, the SOC setpoints are exposed read-only
  over local Modbus (the `wlcrs/huawei_solar` integration never creates
  a writable SOC entity for an EMMA device), so QS could not write them
  even if it wanted to.

User configuration (Huawei EMMA + SmartGuard, for support questions):
the reserve is **`Backup power SOC`**, set on the **inverter** (not the
battery, not the SmartGuard) and only visible once **`Off-grid mode`**
is enabled — FusionSolar app → Device Commissioning → connect to the
EMMA → `Monitor > SUN2000 > Set > Feature parameters`. `Backup power
SOC` takes precedence over the battery's `End-of-discharge SOC`:
on-grid the battery stops discharging at the reserve, and on grid loss
it runs the reserve down toward the deep floor. Guaranteed outage
runway ≈ Backup power SOC − End-of-discharge SOC (Huawei recommends a
20–30% reserve).

What QS must still avoid: do not pin the battery at max-discharge = 0
longer than necessary while on-grid — on inverters that honour that
limit in island mode it would defeat the native reserve. QS already
restores discharge on the off-grid transition
(`async_set_off_grid_mode` → `CMD_GREEN_CHARGE_AND_DISCHARGE`).

### Opt-in discharge-power floor (QS-349)

QS-264's stance is unchanged — the SOC *reserve* stays a native inverter
concern — but QS now offers an opt-in **discharge-power floor** as the
software safety net for inverters that honour max-discharge in island
mode. `CONF_BATTERY_MIN_DISCHARGE_POWER_VALUE` (default 0) sets
`Battery.min_discharging_power`; when > 0, the commands that used to
write `max_discharging_power = 0` (`CMD_GREEN_CHARGE_ONLY`,
`CMD_FORCE_CHARGE`) write the floor instead, so a grid outage during a
grid-charge phase cannot lock the battery at zero discharge. The floor
covers the pre-detection ride-through window; it does **not** run an
arbitrary house — above the floor the inverter's own overload protection
decides the outcome. The "do not pin at 0 longer than necessary"
obligation above is then structurally guaranteed whenever the floor is
set.

### Reaction ordering (QS-349, fix A + fix B)

Two reaction-latency fixes on the off-grid transition:

- **Battery first** (`async_set_off_grid_mode`): the
  `CMD_GREEN_CHARGE_AND_DISCHARGE` restore is launched **before** the
  load-shedding loop — the one command that keeps the house alive no
  longer sits behind every load's awaited probe + service call. The
  battery is not in `_all_loads`, so this is a pure reorder; shedding,
  the `_switch_to_off_grid_launched` gate, and `force_next_solve()` are
  unchanged. The restore call is wrapped in a log-only `try/except` so a
  raising battery command cannot cancel load shedding or the gate
  (do_reset is already committed — a swallowed error would otherwise
  leave loads running off-grid with no retry).
- **Notify off the critical path** (`_off_grid_entity_state_changed`):
  the mobile broadcast is scheduled as a background task **before** the
  apply is awaited, so notification delivery and the battery restore are
  concurrent and neither delays the other (see
  [notification-routing.md](notification-routing.md)).

**Manual island-mode validation** (reporter hardware: Huawei
SUN2000-12K-MAP0 + LUNA2000 + SmartGuard-63A + EMMA-A02) is a post-merge
follow-up — its outcome (whether max-discharge = 0 is honoured in island
mode, and floor = 300 W ride-through) will be recorded here once known.

## See also

- [qs-home-orchestrator.md](qs-home-orchestrator.md) — `QSHome` and
  its cycles.
- [notification-routing.md](notification-routing.md) — the
  per-person broadcast path.
- [../use-cases/off-grid-kicks-in.md](../use-cases/off-grid-kicks-in.md)
  — end-to-end outage scenario.
- [qs-home-orchestrator.md](qs-home-orchestrator.md) "Dashboard
  sections auto-migration" — QS-194 added migration logic to
  `QSHome.add_device` but no off-grid-mode-related behaviour was
  touched (re-verified under review-fix #03).
