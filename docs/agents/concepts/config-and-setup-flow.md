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

**Cross-field XOR validation pattern** (`async_step_radiator`):

The radiator step shows BOTH a switch selector AND a climate
selector and validates exactly-one-set via an explicit if/else
inside the step. On misconfig it re-renders the form with
`errors={"base": "exactly_one_backing_required"}`. The same
constraint is enforced in `QSRadiator.__init__` via
`ServiceValidationError` for any non-UI code path (e.g. tests,
direct construction). This is the canonical pattern for "pick
exactly one of these N options" — voluptuous's `vol.Exclusive`
group only enforces "at most one", not "exactly one".

**Two-pass form pattern** (`async_step_climate`,
`async_step_radiator`):

When a step needs to render dropdowns derived from another field
(e.g. HVAC modes that depend on which climate entity the user
picked), the step issues a self-call with a sentinel token in
`user_input` (`"force_climate"`, `"force_radiator_climate"`). Pass 1
captures the climate entity into `self.config_entry.data`
(creation flow — `FakeConfigEntry`) or via `async_update_entry`
(options flow — real `ConfigEntry`). Pass 2 reads the available
HVAC modes from the registry and shows the mode selectors with a
suggested default.

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
