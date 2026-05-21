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
last_verified: 2026-05-21
---

# Bistate-duration devices (pool, on/off duration, climate, radiator)

## TL;DR

Bistate-duration devices are loads with two states (on / off) that
must run for a **specified duration** rather than be modulated. Pool
pumps, fixed-power boilers, climate splits, heating-only radiators,
and miscellaneous on/off duration loads all use this pattern.
`ha_model/bistate_duration.py` provides the shared base;
`ha_model/on_off_duration.py` and `ha_model/climate_controller.py` are
the original subclasses; `ha_model/pool.py` extends `on_off_duration`
with temperature-dependent filter-duration logic;
`ha_model/radiator.py` adds a heating-only variant that can sit on
**either** a switch OR a climate entity. All inherit the
switching-cost protection pattern.

The per-backing difference (which HA entity is observed, which HA
service is called) is extracted into `ha_model/bistate_transport.py`
as a `BistateTransport` strategy — `SwitchTransport` flips a switch
via `switch.turn_on`/`turn_off`; `ClimateTransport` toggles a climate
entity via `climate.set_hvac_mode`. Each subclass picks a transport
in `__init__` and delegates `execute_command_system` to it. The host
keeps owning `_state_on` / `_state_off` / `_bistate_mode_*` and the
override state machine; the transport receives primitives.

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

## See also

- [load-base.md](load-base.md) — `AbstractLoad` base contract.
- [constraints.md](constraints.md) — the
  `TimeBasedSimplePowerLoadConstraint` subclass.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the daily-budget pattern these devices inherit.
