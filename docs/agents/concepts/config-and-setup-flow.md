---
title: Config flow and setup
slug: config-and-setup-flow
kind: concept
covers:
  - custom_components/quiet_solar/config_flow.py
  - custom_components/quiet_solar/__init__.py
  - custom_components/quiet_solar/data_handler.py
  - custom_components/quiet_solar/const.py
last_verified: 2026-05-21
---

# Config flow and setup

## TL;DR

The config flow is hierarchical: a top-level menu routes to device-
type-specific steps, each of which composes a schema via the
shared `get_common_schema()` builder. `__init__.py:async_setup_entry`
hands control to `QSDataHandler`, which creates `QSHome` on the
first config entry and queues subsequent device entries.
`const.py` is the **single source of truth** for config keys —
strings are never hardcoded anywhere else.

## When you need this concept

- Adding a new device type (you'll add a `CONF_TYPE_NAME_QS<Name>`
  constant + a config-flow step + schema entries).
- Modifying the common schema (3-phase, calendar, groups,
  max on/off, etc.).
- Touching device construction or config-entry lifecycle.
- Working on UI translations (`strings.json`).

## Core idea

**Hierarchical menu pattern**:

```text
async_step_user (top-level menu)
  ├── async_step_<device_type> (per-type steps)
  │   └── get_common_schema() → vol.Schema(...)
  └── ConfigEntry created
```

**Common schema builder** (`get_common_schema()`):

- Composable field set: name, power, 3-phase config, calendar
  entity, group membership, max on/off, etc.
- Reused across every device-type step so the user experience is
  uniform.
- Entity selectors filter by unit of measurement and **exclude
  quiet-solar's own entities** (no self-referential setups).

**Data handler lifecycle** (`data_handler.py`):

1. First config entry → create `QSHome`.
2. Subsequent entries → queue device construction via
   `async_add_entry()` → `create_device_from_type()`.
3. Device added to home → platforms (sensor/switch/number/select/
   button) get factory calls → entities registered.

## Key types / structures

- `config_flow.py` — the flow steps.
- `get_common_schema()` — the field composer.
- `QSDataHandler` (in `data_handler.py`) — entry lifecycle manager.
- `create_device_from_type()` — the per-type construction switch.
- `CONF_*` constants in `const.py` — the only place config keys
  exist.

## Lifecycle

```text
HA UI → config_flow.py step → ConfigEntry created
  ↓
__init__.py:async_setup_entry → QSDataHandler.async_add_entry
  ↓
create_device_from_type(config_entry.data) → device class instance
  ↓
home.add_device(device) → device.get_platforms()
  ↓
HA platform setup → create_ha_<platform>(device) → entities registered
  ↓
device.attach_ha_state_to_probe(...) for each tracked entity
```

## Common mistakes

- Hardcoding a config key string instead of using a `CONF_*`
  constant. Breaks every test that depends on the constant.
- Adding a config-flow step that bypasses `get_common_schema()`. The
  user experience diverges across device types.
- Adding `CONF_*` keys directly in the flow file. Every key lives
  in `const.py` — the flow imports them.
- Editing `translations/en.json` directly. Always edit
  `strings.json` and run `bash scripts/generate-translations.sh`.
- Creating a device that doesn't register itself with `QSHome` via
  `add_device()`. The device is invisible to the solver.

## See also

- [load-base.md](load-base.md) — what each device extends.
- [ha-device-mixin.md](ha-device-mixin.md) — the bridge layer
  every HA device class uses.
- [qs-home-orchestrator.md](qs-home-orchestrator.md) — what
  `QSDataHandler` instantiates.
- [dashboard-and-cards.md](dashboard-and-cards.md) — step 0 of
  the device-type checklist (the `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`
  mapping) decides whether the new device appears on the dashboard.
- [../../workflow/project-rules.md](../../workflow/project-rules.md)
  — the `const.py` and translations rules.
- `async_step_water_boiler` (in `config_flow.py`) — the latest
  per-type step, added in QS-194. Mirrors `async_step_on_off_duration`
  with an extra optional `water_boiler_temperature_sensor` field.
  Uses the canonical `return await self.async_entry_next(...)`
  direct-return pattern (no intermediate variable). The optional
  temperature-sensor selector is surfaced **whenever there are live
  temperature entities OR a previously-configured id is stored** — the
  latter rule prevents a stranded entity id from becoming invisible
  after HA loses the sensor (the form's selector then includes the
  stored id so the user can replace or keep it). When the field was
  rendered, the per-step also `setdefault`s the key in `user_input`
  to an empty string before submission so the OptionsFlow merge can
  actually overwrite a stranded id with a cleared value
  (`QSWaterBoiler.__init__` normalises the empty string back to
  `None`).
