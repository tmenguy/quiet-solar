---
title: Dashboard generation and JS Lovelace cards
slug: dashboard-and-cards
kind: concept
covers:
  - custom_components/quiet_solar/ui/dashboard.py
  - custom_components/quiet_solar/ui/quiet_solar_dashboard_template.yaml.j2
  - custom_components/quiet_solar/ui/quiet_solar_dashboard_template_standard_ha.yaml.j2
  - custom_components/quiet_solar/ui/resources/qs-car-card.js
  - custom_components/quiet_solar/ui/resources/qs-pool-card.js
  - custom_components/quiet_solar/ui/resources/qs-climate-card.js
  - custom_components/quiet_solar/ui/resources/qs-on-off-duration-card.js
last_verified: 2026-05-19
---

# Dashboard generation and JS Lovelace cards

## TL;DR

On first install, quiet-solar **auto-generates two storage-mode HA
Lovelace dashboards** by rendering two Jinja2 templates against the
live `QSHome`:

- **"Quiet Solar"** (`quiet-solar` URL, custom cards) — renders the
  four bundled JS Lovelace cards (`qs-car-card`, `qs-pool-card`,
  `qs-climate-card`, `qs-on-off-duration-card`), one per device type
  that has a dedicated card.
- **"Quiet Std"** (`quiet-solar-standard` URL, standard cards) —
  renders the same data using only built-in HA cards (no JS cards
  required). This is the fallback for households who want a
  zero-custom-resource setup.

Both dashboards iterate `home.dashboard_sections` (a list of
`(section_name, icon)` tuples derived from
`LOAD_TYPE_DASHBOARD_DEFAULT_SECTION` in `const.py`) and pull
devices via `home.get_devices_for_dashboard_section(name)`. Cards
within each section read the device's `ha_entities` dict to wire
specific entity IDs — the dashboard never hard-codes entity ids; it
always asks the device. JS resources live under
`ui/resources/`, are copied to `<config>/www/quiet_solar/` on every
HA start, and are registered as Lovelace resources with a
`?qs_tag=<epoch>` cache-buster so a component update reloads them in
the browser.

## When you need this concept

- Adding a new device type — decide whether it gets a dedicated JS
  card (rare) or just appears as a standard `entities` card; map it
  to a dashboard section via `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`
  in `const.py`.
- Modifying either Jinja2 template (e.g. surfacing a new entity in a
  card, changing section ordering, conditionally hiding a card).
- Touching the JS cards' input contract — i.e., the per-type
  `entities:` block in the template that the JS card consumes.
- Working on dashboard registration, restoration after restart, or
  the JS resource cache-busting.
- Debugging "my device appears under the wrong section" or "the
  dashboard didn't refresh after I added a charger" issues.

## Core idea

### Two dashboards, one source of truth

`ui/dashboard.py` defines two `dashboard_def` dicts
(`DASHBOARD_CUSTOM` and `DASHBOARD_STANDARD`) and a list
`ALL_DASHBOARDS = [DASHBOARD_CUSTOM, DASHBOARD_STANDARD]`.
`generate_dashboard_yaml(home)` walks that list, reads each
`template_filename`, renders it with HA's `Template` engine (so the
templates have full access to HA jinja helpers), parses the
resulting YAML with `yaml.safe_load`, and pushes the dict into a
**storage-mode** `LovelaceStorage` instance. Storage mode means no
YAML on disk and no `configuration.yaml` entries — the dashboard
lives entirely in HA's `.storage/` directory and survives restarts.

The "Quiet Std" template uses only built-in HA cards, so a household
that doesn't trust the bundled JS can switch to that dashboard
without losing any functionality (just visual polish).

### Section mapping in `const.py`

The dashboard is organized into **sections** (`cars`, `climates`,
`pools`, `others`, `settings`). Each device type has a default
section in `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`; TheAdmin can
override per-device via the `CONF_DASHBOARD_SECTION_NAME` config
field. `QSHome.dashboard_sections` is the in-memory list of active
sections (deduplicated, ordered as configured); the templates
iterate this list and call
`home.get_devices_for_dashboard_section(section_name)` to find
which devices to render in that section.

### Per-device-type card dispatch (custom template)

`quiet_solar_dashboard_template.yaml.j2` contains the per-type
switch:

```jinja
{%- if  device.device_type == "car" %}
- type: custom:qs-car-card
{%- elif  device.device_type == "pool" %}
- type: custom:qs-pool-card
{%- elif  device.device_type == "climate" %}
- type: custom:qs-climate-card
{%- elif  device.device_type == "on_off_duration" %}
- type: custom:qs-on-off-duration-card
{%- else %}
- type: entities
{%- endif %}
```

The four JS cards have card-specific YAML input shapes (e.g.,
`qs-car-card` expects `soc:`, `range_now:`, `charge_type:`, etc.).
Every key resolves to an entity ID via `device.ha_entities.get(...)`
— so the template translates "the device knows about a certain
state" into "the JS card reads this entity". If a key is missing on
the device, the template skips that line and the JS card simply
doesn't render that row.

### JS resources lifecycle

`async_update_resources(hass)` runs on **every** HA start. It walks
the bundled `ui/resources/` directory, copies each JS file to
`<config>/www/quiet_solar/<filename>`, and registers it as a
Lovelace resource at URL
`/local/quiet_solar/<filename>?qs_tag=<epoch>`. The `qs_tag` query
parameter changes on each start (epoch-derived), so browsers reload
the updated JS without a hard refresh. Dashboard **content** is NOT
touched on startup — only the JS resources — so any manual edits
TheAdmin made to the dashboards survive an upgrade.

### Tracking storage (survives restart)

`QS_DASHBOARDS_STORAGE_KEY` (`quiet_solar_dashboards`) records which
dashboards have been generated. On first install (no tracking data),
`async_auto_generate_if_first_install(home)` triggers the full
render. On subsequent starts, `async_restore_dashboards_and_update_resources(hass)`
just re-registers the panels (so they reappear in the sidebar) and
refreshes the JS resources — dashboard content is left alone.

## Key types / structures

- `generate_dashboard_yaml(home)` — main entry. Renders both Jinja
  templates and pushes the result into Lovelace storage.
- `async_auto_generate_if_first_install(home)` — gated by tracking
  data; runs the full generate once.
- `async_restore_dashboards_and_update_resources(hass)` — per-start
  re-registration + JS refresh.
- `async_update_resources(hass)` — copy + cache-bust JS card files.
- `async_unregister_dashboards(hass)` — full teardown (called from
  the data handler on integration removal).
- `DASHBOARD_CUSTOM`, `DASHBOARD_STANDARD`, `ALL_DASHBOARDS` —
  dashboard definitions in `ui/dashboard.py`.
- `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION` / `DASHBOARD_DEFAULT_SECTIONS`
  / `DASHBOARD_DEFAULT_SECTIONS_DICT` / `DASHBOARD_NO_SECTION`
  — section taxonomy in `const.py`.
- `QSHome.dashboard_sections` — in-memory ordered list of active
  sections.
- `QSHome.get_devices_for_dashboard_section(name)` — per-section
  device lookup the templates rely on.
- `HADeviceMixin.ha_entities` — the dict the templates query
  (`device.ha_entities.get("car_soc_percentage")` etc.).

## Lifecycle

```text
First install:
  QSDataHandler.async_setup_entry()
    → QSHome created → devices created → ha_entities populated
    → async_auto_generate_if_first_install(home)
      → no tracking data → generate_dashboard_yaml(home)
        → for each dashboard_def in ALL_DASHBOARDS:
            read template → Template.async_render(home=home)
            → yaml.safe_load → push into LovelaceStorage
        → _async_save_dashboard_tracking(hass) [records "done"]
        → async_update_resources(hass) [copy + cache-bust JS]

Subsequent HA starts:
  async_setup → async_restore_dashboards_and_update_resources(hass)
    → tracking data present → re-register panels (sidebar entries)
    → async_update_resources(hass) [refresh JS only]
    → dashboard CONTENT is never touched — TheAdmin's edits survive
```

## The four JS Lovelace cards

| Card | Card type | Device types | Source |
|---|---|---|---|
| Car | `custom:qs-car-card` | `QSCar` | `ui/resources/qs-car-card.js` |
| Pool | `custom:qs-pool-card` | `QSPool` | `ui/resources/qs-pool-card.js` |
| Climate | `custom:qs-climate-card` | `QSClimateDuration`, `QSHeatPump` | `ui/resources/qs-climate-card.js` |
| On-off duration | `custom:qs-on-off-duration-card` | `QSOnOffDuration` | `ui/resources/qs-on-off-duration-card.js` |

The cards are **outside the quality pipeline**: no JS tests, no JS
linter, no build step. The product brief explicitly says "don't
modify JS without explicit instruction" — this is a known gap
(architecture.md "UI Stack KNOWN GAP"). Any change to a card's
input contract (new YAML key, renamed key, type change) must be
mirrored in **both** templates AND the JS file, or the dashboard
silently degrades for one of the two configurations.

## Common mistakes

- **Forgetting the section mapping in `const.py`** when adding a
  new device type. The device exists and works, but is invisible
  on both dashboards — `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`
  defaulting to `None` puts it in `DASHBOARD_NO_SECTION`. The 7-
  file device-type checklist mentions this; the dashboard is
  step 0.
- **Hard-coding an entity ID in a template**. Always go through
  `device.ha_entities.get("key")` so the template stays portable
  across config changes. Adding `entity_id` strings directly
  breaks the moment HA re-allocates them.
- **Modifying one template and forgetting the other**. The custom
  and standard templates render the same data through different
  card families. A new entity in one must land in both, or the
  "Quiet Std" dashboard becomes incomplete.
- **Modifying a JS card without testing**. The cards are
  untested artifacts (KNOWN GAP). Changes must be visually
  verified against a real HA install — they cannot be caught by
  the quality gate.
- **Skipping the cache-buster**. If JS is updated without a new
  `qs_tag`, browsers serve a stale cached version. The
  `_generate_qs_tag()` helper uses `time.time()` so every restart
  produces a fresh tag — don't bypass it.
- **Calling `generate_dashboard_yaml` on every startup**. It
  overwrites manual edits TheAdmin made. Only the first-install
  helper triggers the full render; subsequent starts use the
  restore path.

## See also

- [config-and-setup-flow.md](config-and-setup-flow.md) — the 7-file
  device-type checklist, including the `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`
  step 0.
- [ha-device-mixin.md](ha-device-mixin.md) — the `ha_entities` dict
  the templates query.
- [qs-home-orchestrator.md](qs-home-orchestrator.md) — `QSHome`
  owns `dashboard_sections` and the per-section device lookup.
- [../personas/the-admin.md](../personas/the-admin.md) — the
  persona who lives on the dashboard.
- [../personas/magali.md](../personas/magali.md) — Magali doesn't
  open the dashboard, she taps the mobile app — but her override
  flow surfaces in the qs-car-card.
