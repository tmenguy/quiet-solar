---
title: QSHome (orchestrator)
slug: qs-home-orchestrator
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
last_verified: 2026-07-04
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
- Public registry accessors — `get_car_by_name(name)`,
  `get_person_by_name(name)`, `get_heat_pumps()`. Callers outside
  `QSHome` should prefer these over reaching into the private
  `_cars` / `_persons` / `_heat_pumps` lists. Accessors return
  snapshot copies so external code cannot mutate the home's
  registry; the canonical mutation surface stays `add_device` /
  `remove_device`.

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

## Dashboard sections — init-time auto-include

`QSHome.__init__` deterministically builds `self.dashboard_sections`
on every load:

1. Read every `CONF_DASHBOARD_SECTION_NAME_<i>` /
   `CONF_DASHBOARD_SECTION_ICON_<i>` slot from `config_entry.data`
   into an in-memory list. Slot names go through
   `extract_name_and_index_from_dashboard_section_option` so the
   `"#N - <name>"` prefix is stripped consistently (no substring
   heuristic, no mis-strip on section names containing `" - "`).
2. **Auto-include**: walk `DASHBOARD_DEFAULT_SECTIONS` (the bundled
   defaults — `cars`, `climates`, `pools`, `water_boilers`,
   `radiators`, `others`, `settings`); for each entry missing from
   the list AND not listed in `CONF_DASHBOARD_SECTIONS_USER_REMOVED`,
   append it. Runtime-only — `config_entry.data` is NOT modified, so
   the user's persisted slot layout stays user-owned.
3. Run `_normalize_dashboard_sections_order` so bundled defaults
   appear in canonical const order regardless of the persisted slot
   ordering. Custom user sections (names not in
   `DASHBOARD_DEFAULT_SECTIONS_DICT`) are preserved at the tail in
   their original relative order.

This replaces the pre-QS-195 `_maybe_migrate_missing_default_section`
per-device mechanism. Why the swap:

- The previous approach mutated `dashboard_sections` lazily as each
  device was added, which created in-memory-vs-persisted divergence
  AND a per-device timing dependency on the home being constructed
  first. Silent early-returns (5+ different guards) could prevent
  the migration from firing under conditions that were hard to
  diagnose.
- The init-time approach runs once, deterministically, before any
  device is added. The dashboard YAML, the home options form, and
  every device's section dropdown read the same `home.dashboard_sections`
  list, so all three surfaces are always consistent.

Users still have an opt-out: list a section name in
`CONF_DASHBOARD_SECTIONS_USER_REMOVED` on the home config entry and
the init-time auto-include skips it (no UI for this — it's a
power-user knob for the rare case of deliberately removing a bundled
default).

The complementary tier-1 diagnostic lives in
`home_model/load.py:dashboard_section`: if a device's requested
section can't be resolved (e.g. a typo, a now-removed custom
section), a single `_LOGGER.warning` surfaces the device and the
unresolved section so the issue can be diagnosed from HA logs.

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
