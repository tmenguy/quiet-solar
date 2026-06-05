---
title: Bistate-duration devices
slug: bistate-duration-devices
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/bistate_duration.py
  - custom_components/quiet_solar/ha_model/on_off_duration.py
  - custom_components/quiet_solar/ha_model/pool.py
  - custom_components/quiet_solar/ha_model/climate_controller.py
  - custom_components/quiet_solar/ha_model/radiator.py
  - custom_components/quiet_solar/ha_model/bistate_transport.py
  - custom_components/quiet_solar/ha_model/water_boiler.py
last_verified: 2026-06-05
---

# Bistate-duration devices (pool, on/off duration, water boiler, climate, radiator)

## TL;DR

Bistate-duration devices are loads with two states (on / off) that
must run for a **specified duration** rather than be modulated. Pool
pumps, fixed-power boilers, water-boilers (cumulus / thermodynamic), climate splits, heating-only radiators, 
and miscellaneous on/off duration loads
all use this pattern.
`ha_model/bistate_duration.py` provides the shared base;
`ha_model/on_off_duration.py` and `ha_model/climate_controller.py` are
the original subclasses; `ha_model/pool.py` extends `on_off_duration`
with temperature-dependent filter-duration logic;
`ha_model/water_boiler.py` is a thin subclass that adds an
**optional** water-tank temperature sensor (plumbing only — no
temperature-aware control logic yet) plus its own config step,
dashboard section, and select-mode translation key.
`ha_model/radiator.py` adds a heating-only variant that can sit on
**either** a switch OR a climate entity. 
All inherit the
switching-cost protection pattern.

The per-backing difference (which HA entity is observed, which HA
service is called) is extracted into `ha_model/bistate_transport.py`
as a `BistateTransport` strategy — `SwitchTransport` flips a switch
via `switch.turn_on`/`turn_off`; `ClimateTransport` toggles a climate
entity via `climate.set_hvac_mode`. Each subclass picks a transport
in `__init__` and delegates `execute_command_system` to it. The host
keeps owning `_state_on` / `_state_off` / `_bistate_mode_*` and the
override state machine; the transport receives primitives.

The `_state_on` / `_state_off` shadow lives on `QSBiStateDuration`
as a property pair that delegates to `self._transport.state_on /
state_off` when the transport is set (and falls back to a per-
instance host field while `_transport is None`, during the base
ctor's seed assignments). A public `hvac_state_on` accessor exposes
the on-state string to the dashboard template — HA's Jinja sandbox
restricts leading-underscore attribute access.

The `_bistate_mode_on` / `_bistate_mode_off` strings drive the
**bistate-mode select** UI (Force ON / Force OFF entries) and follow
ONE OF TWO conventions across subclasses:

1. **Namespaced-literal convention** — `QSOnOffDuration`, `QSPool`,
   and `QSRadiator` use a per-class namespaced key (e.g.
   `"on_off_mode_on"`, `"radiator_mode_on"`). The select translates
   those keys to "Force ON" / "Force OFF" labels independently of
   the underlying HA state. The `_state_on/off` HA-state value
   remains the raw service-call value (e.g. `"on"`/`"off"` for
   switches, `"heat"`/`"off"` for climate-backed radiators). QS-204
   moved `QSRadiator` from the bare `"on"`/`"off"` literals to the
   namespaced form so the radiator-mode keys no longer collide with
   the underlying HA switch states.
2. **Raw HVAC convention** — `QSClimateDuration` mirrors the raw
   HVAC mode (`"heat"`, `"cool"`, `"fan_only"`, …) so the
   `climate_mode` translation can label each force-mode entry with
   the HVAC mode name.

Cross-subclass logic that compares `_state_on` ↔ `_bistate_mode_on`
must treat the two as decoupled. The divergence is documented in
`QSBiStateDuration`'s class docstring.

## When you need this concept

- Adding a new simple on/off device (irrigation, ventilation, fixed-
  power resistive heater).
- Touching pool-pump duration logic.
- Working on the bistate constraint type
  (`TimeBasedSimplePowerLoadConstraint`).
- Tweaking switching-cost / hysteresis behaviour for on/off
  devices.

## Core idea

A bistate-duration device has:

- A fixed power draw when on.
- A **target duration** (e.g., "run for 4 hours").
- The same `num_max_on_off` + 10-minute hysteresis as every other
  switching device.

The solver consumes `TimeBasedSimplePowerLoadConstraint` for these
— the constraint specifies "X hours at Y watts before deadline";
the solver picks the cheapest contiguous (or split) windows.

`QSPool` adds **temperature-dependent duration** on top of
`QSOnOffDuration`: warmer water → longer filter duration. The
extension overrides the duration calculation but inherits the rest
of the on/off behaviour unchanged.

**Override semantics (QS-256):**

- `probe_if_command_set` is truthful — it compares the entity state
  against `expected_state_from_command(command)` ONLY (the command's
  expected state, never the override state). Comparing against the
  override used to phantom-ack solver commands during an override.
- `is_command_suppressed_by_override` (the `launch_command` drop-point
  hook) returns True iff an override is active AND the command's
  expected state differs from it — with the degraded-override nuance:
  a non-mandatory override constraint lets off/idle commands through,
  mirroring `execute_command`'s interception block (which stays as the
  defensive guard for the `force_relaunch_command` path).
- An override to the idle state pushes a `TimeBasedHoldOffConstraint`
  (zero power, CMD_IDLE window) so the override ends through the
  constraint-ack path; if the computed end is already past, the
  override is reset directly instead.
- `use_saved_extra_device_info` drops a stored override that is
  already older than `override_duration` hours at restore time.

## Key types / structures

- `bistate_duration.py` — shared base (the lifecycle, the constraint
  interface, the bistate-mode signals, the override state machine).
- `bistate_transport.py` — strategy module. `BistateTransport`
  (abstract), `SwitchTransport`, `ClimateTransport`. Each transport
  owns the underlying HA `entity_id` and translates a `LoadCommand`
  (plus optional override-state) into the right service call.
- `QSOnOffDuration(QSBiStateDuration)` — simple switch loads;
  delegates to `SwitchTransport`.
- `QSClimateDuration(QSBiStateDuration)` — climate-entity-backed
  loads; delegates to `ClimateTransport`; keeps the runtime
  `climate_state_on/off` selects so users can flip seasonal HVAC
  modes (heat/cool) without restarting.
- `QSRadiator(QSBiStateDuration)` — heating-only variant; picks a
  switch- OR climate-transport at `__init__`; defaults the climate
  backing to `heat`/`off`; does NOT expose runtime HVAC selects
  (config-time only — heating-only).
- `QSPool(QSOnOffDuration)` — temperature-aware filter duration.
- `QSWaterBoiler(QSOnOffDuration)` — water boiler (cumulus or
  thermodynamic). Optional `water_boiler_temperature_sensor` field
  is plumbing-only in QS-194; a future story will introduce
  temperature-aware constraint logic, off-peak preference, and
  anti-legionella cycles. The constructor normalises an empty-string
  config value to `None` (the options-flow form can store `""` when
  the EntitySelector is cleared) so downstream consumers only ever
  see a real entity id or `None`.
- `TimeBasedSimplePowerLoadConstraint` (in
  `home_model/constraints.py`) — the constraint subclass these
  devices use.

## Adding a new bistate device

If your new device shares the bistate-duration lifecycle and only
differs in (a) which HA entity it observes and (b) which HA service
it calls, you can add it by:

1. If the backing isn't already covered by `SwitchTransport` or
   `ClimateTransport`, add a new `BistateTransport` subclass in
   `bistate_transport.py` exposing `default_state_on/off`,
   `mode_options`, and `execute(hass, command, override_state,
   state_on, state_off)`.
2. Create `ha_model/<device>.py` extending `QSBiStateDuration` and
   instantiating the right transport in `__init__`. Set
   `self._state_on/off`, `self._bistate_mode_on/off`,
   `self.bistate_entity = self._transport.entity`.
3. Register the new type in `entity.py` (`LOAD_TYPE_LIST`,
   `LOAD_NAMES`) and `config_flow.py` (`LOAD_TYPES_MENU` plus the
   `async_step_<type>` step).
4. Add the dashboard section in `const.py`
   (`DASHBOARD_DEFAULT_SECTIONS`, `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`)
   and the per-type Jinja branch in
   `ui/quiet_solar_dashboard_template.yaml.j2`.

`QSRadiator` is the worked example for a device that supports two
different backings (switch OR climate) — the constructor picks the
transport based on which `CONF_*` the user filled.

## Common mistakes

- Modulating power on a bistate device. The whole point is that
  it's on or off; if you can modulate, use
  [piloted-device-and-heat-pump.md](piloted-device-and-heat-pump.md)
  or a charger pattern instead.
- Forgetting the switching budget. On/off devices are the heaviest
  consumers of `num_max_on_off` — debugging "the pool never ran"
  often comes back to a tight daily limit.
- Hard-coding pool filter duration. The temperature-dependent
  helper on `QSPool` is the single source of truth.
- Putting backing-specific service-call logic on the host. The host
  (`QSBiStateDuration`) owns the bistate-mode signals; the transport
  owns the service-call. Crossing that boundary breaks
  `QSRadiator`'s ability to pick a transport at construction time.
- Dropping the currently-active constraint from the daily metrics
  when its finish time crosses local midnight. `update_current_metrics`
  filters constraints to a today-only window
  (`end_of_constraint <= tomorrow_utc`) in **both** the calendar and
  default/pool paths. An active constraint that started today but ends
  overnight (e.g. 06:30 tomorrow) falls outside that window, so the card
  ring (`qs_bistate_current_on_h`) and target
  (`qs_bistate_current_duration_h`) would show 0 while the
  `constraint_completion` sensor rises. `_overnight_active_constraint`
  returns the single qualifying constraint (or `None`). A constraint qualifies
  when `end_of_constraint > tomorrow_utc` (overnight finish),
  `current_start_of_constraint <= time` (already started — excludes a
  not-yet-started tomorrow-only constraint), `target_value is not None`, and
  `is_constraint_active_for_time_period(time)` (unmet/active). Both metric
  paths call it, skip the returned constraint in their in-window loop, and add
  its target/runtime back (QS-245).
  **Invariant — at most one active constraint at a time.** A load's constraints
  follow each other in time *by construction*: their active windows are sequential
  and non-overlapping (`push_live_constraint` chains them, each new one starting
  at/after the previous one's end). So a single constraint is "current" at any
  `time` — the helper reuses `get_current_active_constraint` (already filtered for
  unmet/active) and just asks whether that one finishes overnight. It **must not**
  be generalised to scan for several overnight constraints (that would only matter
  for overlapping windows, which cannot occur here).
- Mishandling `_last_completed_constraint` (lcc) at the local-midnight
  boundary. The default/pool path adds the lcc to the daily metrics following a
  single rule keyed on "is there already today content?" — where *today content*
  = any in-window constraint (`today_utc < end <= tomorrow_utc`) **or** the
  overnight active constraint:
  - `lcc.end == today_utc` (a cycle that ended exactly at local midnight, bug
    #101 / #95): surface it **only when there is no today content** — otherwise
    it is stale and would double-count yesterday on top of today's cycle
    (whether that cycle is in-window or finishes overnight, QS-245 fix #01).
  - `today_utc < lcc.end <= tomorrow_utc`: show it unless an active same-type
    constraint sharing its (initial) end date already absorbed its runtime
    (same-day cycle carry-over). Also drop it when an **overnight active
    constraint is running** (`overnight_ct is not None`, QS-247): for a
    non-midnight finish time the just-completed cycle is then the previous day's
    cycle of this single-daily-cycle load and would otherwise be counted on top
    of today's running cycle (the growing / overfull ring) — symmetric with the
    `lcc.end == today_utc` rule above.
  - anything ending before `today_utc`, the `DATETIME_MAX_UTC` sentinel, or
    after `tomorrow_utc`: excluded. Use `== today_utc` (not `<=`) for the
    boundary case so genuinely-old cycles are never resurrected.

## See also

- [load-base.md](load-base.md) — `AbstractLoad` base contract.
- [constraints.md](constraints.md) — the
  `TimeBasedSimplePowerLoadConstraint` subclass.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the daily-budget pattern these devices inherit.
