---
title: QSHome (orchestrator)
slug: qs-home-orchestrator
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
last_verified: 2026-05-21
---

# QSHome — the orchestrator

## TL;DR

`QSHome` extends `QSDynamicGroup` (which extends `HADeviceMixin` +
`AbstractDevice`) and sits at the root of the dynamic-group tree.
`QSDataHandler` drives three async cycles against it: state polling
(~4s), load management (~7s), forecast refresh (~30s). The load
management cycle is where the solver runs, ACKs are checked, and
commands are launched. `QSHome` also owns the `QSHomeMode` state
(NORMAL / OFF_GRID / FORCED_OFF_GRID) and routes household-member
notifications.

## When you need this concept

- Touching the orchestration cycles (state polling, load management,
  forecast refresh).
- Working on `QSHomeMode` transitions (e.g., grid outage detection).
- Modifying notification routing.
- Anything that affects multiple device types at once.

## Core idea

Three cycles, three responsibilities:

- **State polling (~4s)** — `update_all_states()` reads HA entities
  into history, refreshes forecasts.
- **Load management (~7s)** — `update_loads()` updates constraints,
  checks command ACKs, triggers solver if needed, launches commands
  (max 1 per load per cycle, amp budget checked).
- **Forecast refresh (~30s)** — solar forecast provider polling
  (Solcast / OpenMeteo).

Each cycle is guarded by an `asyncio.Lock` to prevent re-entrance.
The locks are independent (state polling and load management can
overlap their *non-locked* sections), but a single cycle can't run
twice concurrently — important when HA's scheduler fires faster
than the cycle completes.

`QSHomeMode` is a state machine:

- `NORMAL` — default; full optimisation.
- `OFF_GRID` — grid outage detected (sensor dropped to 0W);
  emergency consumption reduction and broadcast to all mobile apps.
- `FORCED_OFF_GRID` — admin-pinned off-grid (used for maintenance).

## Key types / structures

- `QSHome(QSDynamicGroup)` — orchestrator class.
- `QSHomeMode` — `NORMAL` / `OFF_GRID` / `FORCED_OFF_GRID`.
- `update_all_states()` — the 4s cycle.
- `update_loads()` — the 7s cycle.
- `_state_lock`, `_loads_lock` — `asyncio.Lock` guards.

## Lifecycle

```text
QSDataHandler.async_setup_entry()
  → creates QSHome
  → registers async_track_time_interval for the 3 cycles
  → QSHome.async_init()
  → cycles start firing

per cycle:
  acquire lock → do work → release lock
```

## Common mistakes

- Adding a fourth cycle. Three is the design contract — anything
  that needs more frequent updates should hook into state polling.
- Sharing state between cycles without a lock. The locks are per-
  cycle; cross-cycle shared mutable state is a footgun.
- Blocking inside a cycle (sync I/O, long compute). All work must
  be async or routed through `hass.async_add_executor_job()`.
- Hard-coding the cycle intervals — they live in `const.py`.

## Dashboard sections auto-migration

When a new device type adds a new bundled `DASHBOARD_DEFAULT_SECTIONS`
entry (e.g. `water_boilers` added by QS-194), users who **previously
customised** their dashboard sections will not have the new section
in their stored list. `QSHome.add_device` runs
`_maybe_migrate_missing_default_section(device)` for every device it
accepts: if the device's requested section is one of
`DASHBOARD_DEFAULT_SECTIONS` and is missing from `self.dashboard_sections`,
it is appended **at runtime only** (the config entry is never modified —
the user's customisation stays user-owned). The complementary tier-1
diagnostic lives in `home_model/load.py:dashboard_section`: if section
resolution still fails (i.e. it's not a default-section name), a
single `_LOGGER.warning` surfaces the device and the unresolved
section so the issue can be diagnosed from HA logs.

## See also

- [dynamic-group-tree.md](dynamic-group-tree.md) — the topology
  `QSHome` roots.
- [off-grid-mode.md](off-grid-mode.md) — the `OFF_GRID` path.
- [notification-routing.md](notification-routing.md) — who gets
  notified about what.
- [solver.md](solver.md) — runs inside the load management cycle.
- [config-and-setup-flow.md](config-and-setup-flow.md) — how
  `QSDataHandler` brings `QSHome` to life.
- [../principles/event-driven-with-fallback.md](../principles/event-driven-with-fallback.md)
  — the 5-minute fallback timer logic.
