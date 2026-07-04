---
title: Config flow and setup
slug: config-and-setup-flow
kind: concept
covers:
  - custom_components/quiet_solar/config_flow.py
  - custom_components/quiet_solar/__init__.py
  - custom_components/quiet_solar/data_handler.py
  - custom_components/quiet_solar/const.py
last_verified: 2026-07-05
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
`async_step_radiator`, `async_step_car`):

When a step needs to render dropdowns derived from another field
(e.g. HVAC modes that depend on which climate entity the user
picked), Pass 1 writes the user's input into `config_entry.data`
via `async_update_entry` (for real options-flow entries) or by
direct assignment (for the `FakeConfigEntry` used during creation),
then renders the form directly. Pass 2 reads the available HVAC
modes from the registry and shows the mode selectors with a
suggested default. Because the Pass 1 values are now in
`config_entry.data`, every Pass 2 field (`CONF_NAME`,
`CONF_DEVICE_DASHBOARD_SECTION`, `CONF_POWER`, …) picks up the
correct default through the standard `get_common_schema`
machinery — no per-field plumbing. On any validation failure
(XOR misconfig, empty HVAC modes, identical ON/OFF modes) the
just-submitted payload is passed back into the form via a
`pending` kwarg so the user sees their own selections on the
re-render.

Single-mode and identical-mode HVAC validations live alongside the
XOR check: when fewer than two HVAC modes are advertised the step
surfaces `climate_modes_insufficient`; when both selectors carry
the same mode the step surfaces `hvac_modes_must_differ` — both
errors keep the user's selections through the re-render. The
heat-pump dropdown is also re-validated at submit time against
the LIVE `home.get_heat_pumps()` so a parallel admin action that
removed a heat-pump between Pass 1 and the final submit can't
silently persist a stale name.

User-explicit pilot clear is handled by the **centralized
clear-on-absence mechanism** (below), not a per-field hack: the
heat-pump dropdown is a plain `vol.Optional(CONF_DEVICE_TO_PILOT_NAME)`
selector, so when it was rendered and the submit omits it, the
generic path drops the persisted name. `_async_save_radiator_entry`
folds `_cleared_optional_keys(cleaned)` into its stale-key set so the
single `async_update_entry` write excludes the cleared key. The
genuinely-different `_radiator_orphan_pilot_keys` cleanup still runs
for the case the generic mechanism cannot see — all heat pumps
deleted, so the dropdown isn't even rendered — alongside the
render-side stale-name surfacing and the submit-time revalidation
against `home.get_heat_pumps()`.

**Centralized clear-on-absence for optional fields**
(`QSFlowHandlerMixin`, QS-251): clearing the ✕ on an optional
selector makes Home Assistant omit the key from `user_input`, but the
options-flow save is an additive merge
(`merged = dict(config_entry.data); merged.update(user_input)`), so a
missing key let the stale value survive — clearing nearly any optional
field silently reverted on save. The mixin overrides `async_show_form`
to record the optional marker keys of the rendered schema in
`_last_optional_keys` (bare `CONF_*` strings extracted from each
`vol.Optional` marker). The two terminal options-flow save sites then
drop any optional key that was rendered but omitted:
`QSOptionsFlowHandler._async_entry_next` calls `_merge_with_cleared`,
and `_async_save_radiator_entry` adds `_cleared_optional_keys` to its
stale set. Runtime-only keys (`measured_power`, `measured_charge_*`)
never appear in a form schema, so they are never in the cleared set
and survive the merge. Required fields and boolean `default=` fields
are always submitted by HA, so they are never falsely cleared. The
creation flow uses `async_create_entry(data=...)` with no merge, so the
captured set is simply unused there. This replaced two per-field
band-aids: the water-boiler `setdefault(CONF_WATER_BOILER_TEMPERATURE_SENSOR, "")`
and the radiator `explicit_pilot_clear` sentinel block.

**Multi-pass coverage:** the mechanism is applied at the two *terminal*
save sites **and** at the three *intermediate* multi-pass persists (car
dampening, climate/radiator HVAC reprompt, `_persist_radiator_pass1`).
At an intermediate write, `_last_optional_keys` still reflects the
just-submitted Pass-1 form (Pass 2 has not rendered yet), so routing the
write through `_merge_with_cleared` drops a Pass-1-cleared field before
Pass 2 can re-suggest it — no need to accumulate keys across passes. The
helper takes the merge `base` explicitly (it never reads `config_entry`),
so the same call serves both the terminal saves and the intermediate
persists in both the creation and options flows.

**Radiator validation-error re-render:** when a radiator submit trips a
validation error (XOR backing, HVAC modes), `_async_show_radiator_form`
re-renders with the rejected `pending` submission so the user keeps
their selections. `_suggest` normally falls back to `config_entry.data`,
but for an optional field the user *cleared* (present in
`_last_optional_keys` yet absent/empty in `pending`) that fallback is
suppressed so the selector renders empty — otherwise the error round-trip
would revive the stale value and re-persist it once the user fixes the
unrelated error. Other rejected fields are still re-prefilled.

**Home options-flow section editor** (`async_step_home`): the 8
dashboard-section slot suggestions (`CONF_DASHBOARD_SECTION_NAME_<i>`
+ `CONF_DASHBOARD_SECTION_ICON_<i>`) come from the **live**
`home.dashboard_sections` list — not from the stored
`config_entry.data` slot-by-slot and not from
`DASHBOARD_DEFAULT_SECTIONS[i]` indexed against `i`.
`QSHome.__init__` builds that live list with init-time auto-include
of every bundled default (minus
`CONF_DASHBOARD_SECTIONS_USER_REMOVED`) plus
`_normalize_dashboard_sections_order`, so the form reflects exactly
what's rendered on the main dashboard. Reading by index against the
persisted slots was the source of the QS-195 "multiple times others"
bug: a pre-QS-194 user whose stored slot 3 was `"others"` and slot
5's index default also resolved to `"others"` (current const) saw
`"others"` twice and no `"radiators"`. Slots beyond
`len(home.dashboard_sections)` default to `None`, so the user has at
least one empty slot for adding a custom section.

**Device-side `CONF_DEVICE_DASHBOARD_SECTION` dropdown**: the
`get_common_schema` builder reads its option list from
`home.dashboard_sections` directly with no per-step augmentation.
The pre-QS-195 augmentation that appended a missing bundled default
to the dropdown options (so the user could still pick e.g.
`water_boilers` after a QS-194 upgrade) is no longer needed —
`QSHome.__init__`'s auto-include guarantees the live home already
contains every bundled section. Every config-flow step that
includes this field (`home`, `battery`, `solar`, `person`, `car`,
`pool`, `water_boiler`, `on_off_duration`, `climate`, `heat_pump`,
`radiator`) must have a `device_dashboard_section` translation in
`strings.json` (literal in `person.data`, `[%key:...%]` references
in the others); without it the form renders the raw `device_dashboard_section`
key instead of the localised "Dashboard section" label.

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
- `STORAGE_KEY_*` constants in `const.py` — the persisted per-device
  storage payload keys (read/written by `home_model/load.py`'s
  save/restore methods). Part of the on-disk format: never change the
  values without a storage migration (QS-256).

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
  stored id so the user can replace or keep it). Clearing the field is
  now handled by the centralized clear-on-absence mechanism (see
  "Centralized clear-on-absence" above) — the former per-step
  `setdefault(..., "")` band-aid was removed in QS-251.
  `QSWaterBoiler.__init__` still normalises a pre-existing stored `""`
  back to `None` at construction time.
