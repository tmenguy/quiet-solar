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
  - custom_components/quiet_solar/ui/resources/qs-radiator-card.js
  - custom_components/quiet_solar/ui/resources/qs-water-boiler-card.js
  - custom_components/quiet_solar/ui/resources/shared/qs-card-styles.js
  - custom_components/quiet_solar/ui/resources/shared/qs-card-base.js
  - custom_components/quiet_solar/ui/resources/shared/qs-ring-duration-base.js
  - custom_components/quiet_solar/ui/resources/shared/qs-anim-flame.js
  - custom_components/quiet_solar/ui/resources/shared/qs-anim-wave.js
last_verified: 2026-05-29
---

# Dashboard generation and JS Lovelace cards

## TL;DR

On first install, quiet-solar **auto-generates two storage-mode HA
Lovelace dashboards** by rendering two Jinja2 templates against the
live `QSHome`:

- **"Quiet Solar"** (`quiet-solar` URL, custom cards) — renders the
  five bundled JS Lovelace cards (`qs-car-card`, `qs-pool-card`,
  `qs-climate-card`, `qs-on-off-duration-card`, `qs-radiator-card`),
  one per device type that has a dedicated card.
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
`radiators`, `pools`, `others`, `settings`). Each device type has a
default section in `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`; TheAdmin can
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
{%- elif  device.device_type == "radiator" %}
- type: custom:qs-radiator-card
{%- elif  device.device_type == "on_off_duration" %}
- type: custom:qs-on-off-duration-card
{%- else %}
- type: entities
{%- endif %}
```

The five JS cards have card-specific YAML input shapes (e.g.,
`qs-car-card` expects `soc:`, `range_now:`, `charge_type:`, etc.).
Every key resolves to an entity ID via `device.ha_entities.get(...)`
— so the template translates "the device knows about a certain
state" into "the JS card reads this entity". If a key is missing on
the device, the template skips that line and the JS card simply
doesn't render that row.

### Shared base modules (QS-199)

The six per-card JS files (~9,400 LOC pre-extraction) share a common
foundation that lives under `ui/resources/shared/`. Each card
imports from these modules; the Python copy step rewrites their
`from './shared/*.js'` import URLs with the same `?qs_tag=<epoch>`
cache-buster used for the top-level cards, so a component upgrade
invalidates both the cards AND their dependencies atomically.

Five shared modules:

- **`shared/qs-card-styles.js`** — exports `baseCardCSS(palette,
  options)`, the CSS string consumed by every card. Branded palette
  flows through the `palette` argument
  (`{primary, gradStart, gradEnd, animStart, animEnd}`); the shared
  CSS contains no card-branded literals (only neutral
  `rgba`/`var(--*)` and the four semantic-anchor colours —
  power-blue `#2196F3`, solar-green `#4CAF50`, override-orange
  `#FF9800`, and pill-on green `#2ecc71`/`#2ecc71aa` — pinned by
  `test_shared_css_has_no_branded_colour_literals`).
- **`shared/qs-card-base.js`** — exports `class QsCardBase extends
  HTMLElement` plus the geometry pure functions (`deg2rad`,
  `rad2deg`, `polar`, `arcPath`, `pctToDeg`). The class owns
  lifecycle, service callers (`_entity`, `_call`, `_press`,
  `_turnOn`, `_turnOff`, `_select`, `_setNumber`, `_setTime`),
  defensive utilities (`_escapeHtml`, `_safeNumber` —
  climate-hardened with trim + `Number.isFinite`, `_fmt`,
  `_isValidState`, `_formatTime`, `_parseTimeToMinutes`,
  `_formatHm`), the modal dialog (`_showDialog`), keyboard
  activation (`_registerKeyActivation`), a per-instance ID counter
  (`_instanceId`), the dashed-arc RAF helpers
  (`_startAnimation`/`_stopAnimation`), and **5 wire-helpers**
  (`_wireTargetHandle`, `_wireTimePicker`, `_wireResetButton`,
  `_wirePowerButton`, `_wireGreenButton`). Wire-helpers live on the
  base (not the sub-base) because the car card uses several of them
  too — it inherits `QsCardBase` directly. **QS-235** moved
  `_buildRingHTML` (the ring SVG builder) UP here from the duration
  sub-base — it is the single source of truth, with backward-compatible
  optional params (`handleLabel` / `bgStroke` / `handleFontSize` /
  `handleStroke` / `handleFill` / `animPathId` / `extraDefs`) that
  default to the duration-card output so the car can override them.
  QS-235 also added `_ringCarveCover({cx, cy, r, id, show})` + the
  shared `RING_BOTTOM_CARVE_CX/CY/R` constants (the unified
  bottom-center carve cover — see the QS-217 section), and generalized
  `_wireTargetHandle` (added `onCommit` / `pctToValue` / `valueToPct` /
  `onDragMove` / `fmtHandleText`) and `_wireTimePicker` (added
  `onAfterCommit` / `resetButton` / `title` / `bodyText`) in place so
  the car adopts them with no duration-card source change.
- **`shared/qs-ring-duration-base.js`** — exports `class
  QsRingDurationCardBase extends QsCardBase`. Adds `_clampMaxHours` /
  `_allowedHalfHours` (the hours-specific drag-range helpers),
  `_wireOverrideButton`, and `_wireBistateMode` (the latter wraps
  `_select` in try/catch/finally + 300ms cleanup setTimeout — M2
  hardening). QS-235 moved `_buildRingHTML` UP to `QsCardBase` (the
  five duration cards inherit it unchanged); the hours-/override-only
  helpers stay here because the car needs none of them.
- **`shared/qs-anim-flame.js`** — exports `FLAME_CONSTANTS` plus
  `class QsFlameEngine`. The engine is a pure state machine + path
  generator (`step(dt, fireOn, baseY, ts)` →
  `{shouldRegen}`; `generatePaths(baseY, isIdle)` →
  `string[layerCount]`). Per-card palette and per-layer constants
  (`LAYER_TEETH_COUNTS`, `LAYER_BASE_HEIGHTS`, `FLAME_FILLS`,
  `FLAME_GREY_FILLS`) flow through the constructor — both the
  radiator and climate cards keep their own copies as the source of
  truth.
- **`shared/qs-anim-wave.js`** — exports `WAVE_CONSTANTS` plus
  `generateWavePath(width, amplitude, frequency, phase, yOffset)`
  (a pure ~15-LOC helper). Deliberately NOT a class: each consuming
  card (pool, water-boiler, climate-cool, car) owns its RAF step
  body because the per-card variations (palette, bubble/steam/glow
  hooks, snow-pile particles, single-layer vs 3-layer) would turn
  any "engine" class into mostly feature-flag scaffolding.

**Inheritance hierarchy:**

- `QsCardBase extends HTMLElement` — used directly by: **car**.
- `QsRingDurationCardBase extends QsCardBase` — used by: **pool,
  on-off-duration, radiator, water-boiler, climate**.

**Decision rule going forward:** lifecycle + service callers +
helpers that mutate `this.*` flags → inheritance; pure pipeline
transforms (engines, geometry, CSS templates) → composition.

**Why particle systems stay per-card:** each particle system
(boiler bubbles/steam/surface glow, climate snow-pile + falling
flakes, car sparkles, car lightning, pool temp-tint) is used by
exactly one card and carries 5+ rounds of review-fix tuning baked
into its constants and spawn/advance/retire algorithm. Extracting a
generic `QsParticleEngine` would gain less than ~120 LOC of
deduplication at very high regression risk — every tuning hint
would ride into the engine as a constructor arg. The decision is
explicit and documented; future readers should not redo the
analysis without new evidence.

### JS resources lifecycle

`async_update_resources(hass)` runs on **every** HA start. It walks
the bundled `ui/resources/` directory recursively. For each `.js`
file:

1. It reads the file, rewrites any `from './shared/*.js'` import URL
   with a fresh `?qs_tag=<epoch>` cache-buster (regex always
   overwrites — never skips for an existing query string), and
   writes the rewritten content to `<config>/www/quiet_solar/<...>`.
2. **Top-level cards** (no parent directory under `ui/resources/`)
   are registered as Lovelace resources at URL
   `/local/quiet_solar/<filename>?qs_tag=<epoch>`.
3. **Files under `shared/`** are copied to
   `<config>/www/quiet_solar/shared/<filename>` but **not** registered
   with Lovelace — they are imported by the top-level cards as ES
   modules. The same `tag` value flows into both the registration
   URL and the rewritten import URLs so a browser reload invalidates
   everything atomically.

The import-URL rewrite (review-fix #01 hardening) matches **both**
`from './shared/x.js'` in top-level cards **and** sibling
`from './x.js'` imports *between* files inside `shared/` (so the
inheritance chain `card → qs-ring-duration-base.js → qs-card-base.js`
is fully cache-busted), tolerates the no-whitespace `from'./x.js'`
and dynamic `import('./x.js')` forms, preserves any pre-existing
non-`qs_tag` query params, and uses a literal (callable) replacement
so a tag value is never mis-parsed as a regex backreference. The tag
itself is `time.time_ns()` (nanosecond resolution) so two restarts in
the same wall-clock second still differ. Subdirectories are copied
*before* the top-level cards that import them (dependency order), each
file is written via a temp-file + `os.replace()` atomic swap, and a
non-UTF-8 `.js` file is byte-copied (no rewrite) rather than aborting
the recursion.

Dashboard **content** is NOT touched on startup — only the JS
resources — so any manual edits TheAdmin made to the dashboards
survive an upgrade.

All six cards route through the shared base: lifecycle, service
callers, defensive utilities, the modal dialog (`_showDialog`,
N12/N13/S16-hardened), keyboard activation, the five wire-helpers,
the ring HTML builder, the geometry helpers, and `baseCardCSS`. The
flame engine (`QsFlameEngine`) is consumed only by the radiator card;
the climate card keeps its own inline flame/snow/wind engines (the
four-backdrop dispatch + snow-pile particle system don't fit the
generic engine), and pool / water-boiler keep their own
`_generateWavePath` (a 2×-width GPU-scroll variant). Those three cards
therefore do not import the shared animation modules.

Cross-card hardening invariants worth knowing (review-fix #02): every
entity-derived string interpolated into `innerHTML` is escaped via
`_escapeHtml`; every custom `div` control that carries handlers is
focusable (`role="button"` + `tabindex` + `_registerKeyActivation`),
and a disconnected card drops its custom controls out of the tab order
(`tabindex="-1"` + `aria-disabled`); time parsing goes through the
hardened `_parseTimeToMinutes` (07:00 fallback for `unavailable`/
`unknown`); and the radiator keeps its RAF loop alive until
`QsFlameEngine.isIdle()` so an on→off transition settles to a clean
still silhouette instead of freezing mid-flicker.

Two subsystems were root-caused in review-fix #04 to stop a recurring
edge-case cycle:

- **Resource-update concurrency** — `async_update_resources` runs its
  whole sweep + copy + register under a module-level
  `_RESOURCE_UPDATE_LOCK`, so the two unlocked entry points
  (startup-restore and the Generate-Dashboard button) can't interleave
  their temp-file writes. The unique per-write temp name
  (`<pid>.<counter>.qstmp`) + the orphan sweep stay as defense-in-depth.
- **Drag-range vs gauge** — each card derives `maxHours` through the
  single `_clampMaxHours` helper on EVERY branch (default
  `max_default_hours` AND non-default runtime `targetHours`), and BOTH
  the gauge math and `_allowedHalfHours(maxHours)` consume that one
  value. `_clampMaxHours` rejects non-finite/non-positive input
  (`MAX_HOURS_DEFAULT` = 12), grid-aligns to the 0.5 snap step
  (`SNAP_STEP_HOURS`), floors at one step, and ceilings at
  `MAX_HOURS_CEILING` = 168. Because the gauge's 100% is itself a
  0.5-multiple, `gauge max == max(snap_list)` holds BY CONSTRUCTION —
  no top-of-ring dead zone for fractional configs, no huge-array tail,
  no Infinity hang. (Closed across rounds S8 → M1 → ES1 → #05 S1.)

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

## The five JS Lovelace cards

| Card | Card type | Device types | Source |
|---|---|---|---|
| Car | `custom:qs-car-card` | `QSCar` | `ui/resources/qs-car-card.js` |
| Pool | `custom:qs-pool-card` | `QSPool` | `ui/resources/qs-pool-card.js` |
| Climate | `custom:qs-climate-card` | `QSClimateDuration`, `QSHeatPump` | `ui/resources/qs-climate-card.js` |
| On-off duration | `custom:qs-on-off-duration-card` | `QSOnOffDuration` | `ui/resources/qs-on-off-duration-card.js` |
| Radiator | `custom:qs-radiator-card` | `QSRadiator` | `ui/resources/qs-radiator-card.js` (cloned from `qs-on-off-duration-card.js`; has S14-S17/N7 safety hardening the on/off card has not yet adopted; QS-201 added the heat-mode warm palette and a 3-layer flame backdrop that grows with `hoursRun/maxHours` and goes grey when off; QS-204 redesigned the backdrop from a sine-wave-tongue shape (visually identical to the pool card's water waves) to a peaked-teeth path that reads as flames in both running (orange, per-tooth flicker) and idle (grey, motionless) states. The redesign drops the global `translateX` scroll in favour of per-tooth independent tip flicker. Animation constants are duplicated in-file rather than shared, so a future pool-card refactor cannot silently break this card.) |
| Water boiler | `custom:qs-water-boiler-card` | `QSWaterBoiler` | `ui/resources/qs-water-boiler-card.js` |

The radiator template wires `backing_entity` (the underlying
switch/climate entity id) and `climate_hvac_mode_on` (the configured
HVAC ON mode — e.g. `"heat"`, `"auto"`) through the entities block.
The card uses those values to derive `running` during the cold-start
grace window when the QS command sensor hasn't published yet.

**`qs-water-boiler-card` initial release (QS-194).** The water-boiler
card was forked from `qs-on-off-duration-card` as its starting point
(water boilers are on/off-duration loads at heart) with one
boiler-specific extension: an optional `temperature_sensor` entity
that renders a water-tank temperature row at the top of the card.
Future iterations will add boiler-specific UI (anti-legionella
indicators, off-peak preference, water-usage tracking) without
churning every on/off-duration user. The custom template emits the
JS card's input contract via `key: value` pairs (e.g.
`temperature_sensor: {{ device.water_boiler_temperature_sensor }}`)
just like the other dedicated cards. The standard template still
falls back to plain `- entity:` rows — that's the no-JS variant.

New Jinja branches added by QS-194 use the idiomatic `is not none`
test (rather than the pre-existing `!= None`) and `{# NOTE: ... #}`
documentation comments (rather than `{# TODO: ... #}`); follow these
conventions for any future template additions.

**QS-200 — boiler card visual upgrade.** The water-boiler card
adopted three layered visual changes on top of the QS-194 baseline,
overriding the project-context "don't modify JS cards without explicit
instruction" rule via direct issue-#200 authorisation:

- **Heat palette.** The `const colors = { ... }` block was swapped
  from the cool-blue scheme (`#2196F3` / `#00bcd4` / `#8bc34a` /
  `#00e1ff` / `#0066ff`) to the climate card's `heat` scheme
  (`#FF5722` / `#D32F2F` / `#FF6E40` / `#E64A19`). The cool-blue
  hexes may still appear elsewhere in the file — e.g. `.power-btn.on`
  uses `#2196F3` as a semantic "power" anchor — and that's preserved.
- **Continuous RAF, mirroring the pool card.** `connectedCallback()`
  calls `_startAnimation()` directly with no `showAnimation` gate;
  `disconnectedCallback()` stops it. The wave is intrinsically
  visible at all times (cool blue when not running, translucent
  pool-blue when boiling) so the RAF loop itself is no longer
  conditional. `showAnimation` survives only as a render-time switch
  for the existing dashed-arc `<path id="running_anim">`. The
  boiler is therefore removed from `test_card_raf_idle_gated`'s
  parametrize list — that test pins the idle-gated model that the
  other duration cards still follow.
- **Dual-layer water cross-fade + bubbles + surface glow.** Inside
  a circular `clipPath` (radius `CLIP_R = 120`), the card renders
  six wave paths in pairs: `wave{0,1,2}_cool` (cool-blue HSLA) and
  `wave{0,1,2}_boil` (translucent pool-blue HSLA). Each pair shares geometry;
  only `fill` and `opacity` differ. The RAF loop lerps
  `_currentColorMix` toward `running ? 1 : 0` with the standard
  `LERP_RATE` envelope and updates per-frame opacity: cool layer
  gets `1 - colorMix`, boil layer gets `colorMix`. This avoids the
  yellow-green midpoint that an HSL hue lerp from cyan to orange
  would pass through, and eliminates the staircase a
  `COLOR_MIX_REGEN_THRESHOLD` would have introduced. A dynamic
  bubble system spawns `<circle>` nodes inside a bubble `<g>` layer
  (capped at `MAX_CONCURRENT_BUBBLES = 12`, spawn rate
  `BUBBLE_SPAWN_RATE_HZ = 6` Hz, paused when not running) — bubbles
  rise from the bottom, expand slightly, fade with life, and retire
  on surface contact or life expiry. A red surface glow
  (`SURFACE_GLOW_COLOR = '#FF3D00'`) traces wave 0 via a Gaussian
  blur + `mix-blend-mode: screen` filter; its `d` resyncs on
  amplitude/level regen, `transform` resyncs every frame to follow
  wave 0's `translateX`, and `opacity` is bound to `_currentColorMix`
  so it cross-fades in/out with the boiling state.

Per-instance SVG ids (`waterClipId`, `surfaceGlowFilterId`,
`bubbleLayerId`) are derived from `QsWaterBoilerCard._nextClipId` so
two boiler cards on the same dashboard never collide. The water
layer renders BEFORE the dashed ring / progress arc / handle in DOM
order (= lowest z), so the controls sit on top. The QS-194 optional
`temperature_sensor` row is untouched: water level, wave amplitude,
bubble rate, and surface-glow opacity are all independent of the
temperature sensor (boiling is binary, driven by `running === true`).

**QS-211 — boiling steam puffs.** A 4th boiling-state visual on top
of QS-200's waves + bubbles + surface-glow. When `running === true`:
soft white-translucent `<circle>` puffs spawn from the water surface
(`_waterBaseY`) at `STEAM_SPAWN_RATE_HZ = 1.5` (capped at
`MAX_CONCURRENT_STEAM = 8`), rise toward the top of the water clip
circle with a small sin-wobble, grow with `STEAM_RADIUS_GROW_PX_PER_S
= 4`, and fade in/hold/fade-out via a piecewise-linear life curve. A
single `<filter>` (`feGaussianBlur stdDeviation = 3.5`) is applied to
the `<g id="${steamLayerId}">` group (NOT per circle) for the soft
wispy look at constant per-frame cost. Per-instance unique IDs
(`_steamLayerId`, `_steamFilterId`) derived from
`QsWaterBoilerCard._nextClipId` so two boiler cards on the same
dashboard never collide. The `<circle>` fill is
`STEAM_FILL_COLOR = 'rgba(195,215,235,0.45)'` (the 0.45 alpha is
baked into the SVG fill literal — there is no separate
`STEAM_FILL_ALPHA` JS constant); the runtime per-frame `opacity`
attribute is `lifeOpacity * _currentColorMix` — assignment, not
compound — where `lifeOpacity` is the piecewise-linear lifeCurve(t)
with breakpoints at 0.15 (fade-in end) and 0.85 (fade-out start). The
effective rendered alpha multiplies through the SVG: `0.45 ×
_currentColorMix × lifeOpacity × rimOpacity`, peaking at 0.45 during
the hold band with a fully-ramped colorMix. Steam therefore
cross-fades with `_currentColorMix` and gracefully exits on
running→false: existing puffs continue rising while their opacity
ramps to 0 over the same ~1.5s envelope as bubbles. `_resetDomRefs()`
extends to null `_steamLayerEl` and clear `_steamPuffs`;
`disconnectedCallback` mirrors the bubble teardown to eagerly remove
puff DOM nodes. **QS-214** tinted the puff color to a cool blue-gray for visibility
against dark HA themes; unified the rise rate
(`STEAM_RISE_PX_PER_S_MIN = STEAM_RISE_PX_PER_S_MAX = 24`, with
`STEAM_MAX_LIFE_S = 8.0`); introduced the per-puff circle-aware
geometry (`localTopY = CENTER_CY - sqrt(CLIP_R² - dx²) +
STEAM_TOP_MARGIN_PX`) so puffs follow the arc shape rather than the
global apex; narrowed the spawn cx to a central band
(`±STEAM_SPAWN_CX_HALF_PX = ±35`) so every puff has a substantial
vertical rise regardless of water level (avoiding the "stops just
above the surface" edge-spawn artefact); and added a geometry-aware
per-puff fade — each puff stores `fadeBand = STEAM_RIM_FADE_FRACTION ×
riseBudget` at spawn time (where `riseBudget = cySpawn -
spawnLocalTopY`), then the per-frame factor `rimOpacity = clamp((p.cy
- localTopY) / p.fadeBand, 0, 1)` is multiplied into the opacity
alongside `lifeOpacity` and `_currentColorMix`. Center-spawned puffs
(long rise) get a long fade band; side puffs (short local rise) get
a proportionally shorter one, so the fade always fits the available
space. The visible result reads as "puffs rise visibly through ~40 %
of their rise budget at full opacity, then fade smoothly over the
upper 60 % as they approach the local rim". The disjunctive retire
(`p.cy < localTopY || p.life >= p.maxLife`) is preserved: by the time
geometric retire fires, opacity is already 0, so the remove is
invisible. **QS-214 also slows the wave speed.** The pool-card-inherited
`CALM_SPEED = 0.2` and `BOIL_SPEED = 1.6` read as pumped flow on the
boiler card, which the user flagged as "too fast — it's the speed
from the pool pump running". Slowed to `CALM_SPEED = 0.1` /
`BOIL_SPEED = 0.4` (~4× slower while boiling) so the surface reads
as a gentle simmer rather than pumped circulation. Bubble and steam
constants are unaffected.

**Steam puffs AND bubbles survive `_render()` innerHTML rewrites** —
`_render()` snapshots `_steamPuffs` / `_nextSteamAt` and `_bubbles` /
`_nextBubbleAt` before the rewrite, then re-attaches each preserved
particle's detached DOM node to the freshly-rendered steam / bubble
layer after `_resetDomRefs()`. Without this preservation, every HA
state push wiped all in-flight particles simultaneously: the "3-4
puffs all disappear at the exact same time" symptom for steam
(visible because puffs now live ~8 s with a long fade) and the
previously-accepted "barely-perceptible blip" for bubbles (which
respawned in ~167 ms because their life is only ~1.5 s). The
symmetric snapshot/restore for both subsystems also removes a
per-push DOM-thrash spike.

Review-fix #01 also caps the RAF step `dt` at `LERP_DT_CEIL` (`0.1s`)
in BOTH `qs-water-boiler-card.js` AND `qs-pool-card.js`. Without the
cap, the first frame after a multi-second hidden-tab window
produced a huge `dt` that advanced wave phase by hundreds of pixels
in one frame (visible "snap") and aged every bubble past
`BUBBLE_MAX_LIFE_S` simultaneously. After the cap, all step-loop
subsystems — phase advance, lerp envelope, bubble life — share the
same upper-bound envelope and the visual smoothly resumes from where
it left off. Pinned via the parametrized
`test_card_caps_raf_dt_against_hidden_tab`.

Review-fix #02 added one behavioral change to the boiler card and
several internal refactors:

- **N12 — reconnect re-prime.** `_stopAnimation()` now stashes
  `_runningAtStop`. On the first `_render()` after reconnect, if
  `_runningAtStop !== running`, `_needsAnimationPrime` is forced
  true so the wave snaps to the current target instead of lerping
  from the pre-disconnect state (otherwise a card detached while
  boiling, with the boiler turning off mid-detach, would visibly
  "calm down" on reattach despite the boiler having been off the
  whole detach window). The consume rule has gone through three
  revisions (see code comments in `_render()` for the full
  history):
  - Plan #02 N12: stash cleared unconditionally after the inner
    guard. Hole: mid-detach hass-pushes (which fire `set hass →
    _render` even while disconnected, since `set hass` doesn't gate
    on `this.isConnected`) consume the stash on a stable-running
    push, defeating the prime on the eventual reattach.
  - Plan #03 S1: clear moved INSIDE the inner guard's if-body.
    Closes the mid-detach-pushes hole, but opens a new one — on a
    reattach where `running` is unchanged at reattach, the stash
    leaks across renders and the next normal in-place state flip
    (hours later, no detach involved) falsely fires the prime.
  - Plan #04 M1: the entire consume is now gated by
    `_pendingReattachCheck`, a one-shot flag set in
    `connectedCallback` after `_startAnimation()`, cleared after
    the one-shot consume in `_render`. The stash is consumed
    EXACTLY ONCE on the first post-reattach render, regardless of
    inner-guard outcome. Mid-detach pushes see the flag false and
    skip the entire block, preserving the stash for the eventual
    reattach. Subsequent renders after the consume see the flag
    false and don't re-fire.
- **S3 — `_resetDomRefs()` helper.** Both `_invalidateWaveCache()`
  and the post-`innerHTML` cleanup block now share a single helper
  that nulls DOM-ref memo keys and resets `_bubbles = []`. A future
  memo-key addition lands at both call sites by construction.
- **S1 — `CENTER_CX`/`CENTER_CY` for the ring center.** The arc /
  handle / progress-ring center literal now uses the named
  constants, completing the migration started by review-fix #01 N4
  (water-clip cx/cy). The N4 finding had only migrated bubble
  spawn and clipPath markup; the ring geometry was left inlined.
- **N3 — pool-card dt-cap rationale note.** A code comment on
  `qs-pool-card.js`'s `dt` cap documents the trade-off: cross-card
  consistency over the prior "catch up after hidden tab" behavior.

### QS-210 — Climate card backdrops

The climate card (`qs-climate-card.js`) renders one of four backdrop
visuals inside the central ring's clip circle, selected from the
configured HVAC ON mode plus, for AUTO / HEAT_COOL, the backing
climate entity's current vs. resolved-target temperature:

- **HEAT** → flame (3-layer peaked-teeth engine, copied verbatim from
  `qs-radiator-card.js`'s QS-204 engine — same `LAYER_TEETH_COUNTS`,
  `LAYER_TIP_FLICKER_HZ`, `FLAME_FILLS`/`FLAME_GREY_FILLS`, base-Y
  envelope `FLAME_BASE_MIN_PCT..FLAME_BASE_MAX_PCT` tracked off
  `progressRatio`).
- **COOL** → snow pile + falling snowflakes (waves adapted from
  `qs-pool-card.js`, particle system three-way-inverted from
  `qs-water-boiler-card.js` bubbles — top-of-clip spawn at
  `CENTER_CY - CLIP_R + 8`, positive `vy`, retire on
  `cy >= surfaceY` or `life >= SNOW_MAX_LIFE_S`). Spawn `cx` is
  chord-bounded via `halfChord = Math.sqrt(CLIP_R² - dy²)` so flakes
  don't waste their lifetime spawning outside the visible clip.
  Off-state pile keeps scrolling at `CALM_SNOW_AMP` /
  `CALM_SNOW_SPEED` (matches the pool card's calm-water idiom).
- **AUTO / HEAT_COOL** → flame or snow based on
  `target > currentTemp`, with a `BACKDROP_DEADBAND_C = 0.2°C`
  hysteresis band. Inside the deadband the algorithm holds the
  previous resolved flame/snow; if there's no previous (cold start
  or transitioning from wind/none) the fallback uses the sign-based
  ternary `target > current ? 'flame' : 'snow'` so the very first
  paint still respects the user's intent. When no setpoint resolves
  at all, the algorithm falls through to `running ? 'wind' :
  'none'`. Dual setpoints (`target_temp_low` + `target_temp_high`)
  resolve via midpoint.
- **WIND** → 3 stroked sinusoidal wisps with a linear `translateX`
  scroll (no lerp envelope, no path regen — just modulo wrap to
  prevent float drift). Open SVG path (no closing `L … Z`) so it
  renders as stroked lines, not filled polygons.
- **fan_only / dry / off / unrecognised / null** → none (no
  backdrop). The four climate-entity attribute reads are gated on
  `needsTemps = climateStateOn === 'auto' || climateStateOn === 'heat_cool'` so
  the off-path skips them entirely.

The dashboard template emits the backing climate entity id via
`climate_entity: <entity_id>` in the `entities:` block, mirroring
the radiator card's `backing_entity` plumbing. `device.climate_entity`
is a plain string attribute on `QSClimateDuration` (set at
`ha_model/climate_controller.py`), so Jinja accesses it directly
without a helper. When the attribute is falsy the `{% if %}` guard
omits the line, and the JS card's defensive
`e.climate_entity ? this._hass?.states?.[…] : null` falls through
to `climateEntity = null` → all 4 attribute reads return `null` →
the AUTO branch lands on the `running ? 'wind' : 'none'` fallback.

Per-instance SVG ids (`climateClipId`, `snowLayerId`, `windLayerId`)
derive from `QsClimateCard._nextClipId` so multiple climate cards
on one dashboard never collide.

**Cache invalidation contract.** The three backdrop DOM-ref caches
(`_flameEls`, `_snowWaveEls`/`_snowLayerEl`, `_windWispEls`) are
invalidated UNCONDITIONALLY after every `_render()` innerHTML
rewrite (mirror of `qs-radiator-card.js:967-969`). Same-backdrop
re-renders (every hass push, ~once per second) detach all cached
elements; without the post-rewrite invalidation, RAF would tick on
orphan nodes and the visual would freeze after ~1 s. Animation
accumulators (`_currentFlameAmp`, `_currentSnowAmp`,
`_currentSnowSpeed`, `_snowWavePhase`, `_windPhase`, `_tipPhases`)
deliberately survive both `_stopAnimation` and the post-rewrite
invalidation so re-attach picks up where it left off without a
visible snap. QS-216 — snowflake array now survives innerHTML via a
snapshot/restore block bracketing the `_invalidate*Cache()` triplet.
`_invalidateSnowCache` itself still calls `.remove()` on each flake
and clears `_snowflakes = []` (it governs the
`disconnectedCallback` + real-backdrop-transition paths) but the
body MUST NOT null `b.el` — the truthy-branch restore filter
`(b => b?.el)` silently drops anything nulled there. Pinned by
`test_invalidate_snow_cache_does_not_null_el` (review-fix #01 S2).

**Snow-mode cold-start defence.** A cold start straight into
`'snow'` mode (e.g. `climate_state_on === 'cool'` on first paint)
hits the N5 backdrop-transition reset, which mutates
`this._currentSnowAmp` BEFORE `_startAnimation`'s lazy-init guard
runs. The lazy-init uses PER-FIELD `if (this.<X> == null)` guards
on `_currentSnowAmp` / `_currentSnowSpeed` / `_snowWavePhase` /
`_snowflakes` / `_nextSnowflakeAt` (not a single combined guard) so
the four sibling fields still initialise correctly. The N5 block
also defensively initialises `_snowflakes`, `_snowWavePhase`, and
`_nextSnowflakeAt` itself — belt + braces, robust to either
init-order.

**QS-216 — snowflake preservation across `innerHTML`.** Mirror of
the QS-214 boiler steam-puff and bubble preservation. `_render()`
snapshots `_snowflakes` and `_nextSnowflakeAt` BEFORE the
innerHTML rewrite, then re-attaches each preserved `<circle>` to
the freshly-rendered snow layer AFTER the three
`_invalidate*Cache()` calls. Without this, every HA state push
wiped every in-flight snowflake simultaneously (`_invalidateSnowCache`
calls `b.el.remove()` on each then sets `_snowflakes = []`) — the
same systemic wipe pathology QS-214 fixed for boiler steam. Unlike
the boiler, which always renders the steam layer, the climate
card emits the snow `<g>` only when `_backdrop === 'snow'`; the
restore's null-layer branch is therefore a real backdrop-transition
path (not just defensive) and explicitly removes the preserved
flakes' detached DOM so they don't leak.

### QS-217 — Override-button cover overlay

The semi-transparent bottom-center override "hand" button
(`<div id="override_btn">`) was hard to read against the coloured
animations (orange flames, pale-blue snow, blue water) painted by the
clipPath group below it. QS-217 fixes the legibility by drawing a
small **`<circle>` cover** with `fill="var(--card-background-color)"`
on top of the clipped animation group, visually erasing the
animation in a clean circular patch behind the button. The three
affected cards — `qs-radiator-card.js`, `qs-water-boiler-card.js`,
and `qs-climate-card.js` — each render the cover element
immediately after their `<g clip-path="url(#…ClipId)">` group and
before the outer-ring stroke, gated on `e.override_reset` (the same
truthy check that already controls the button DOM render).

**QS-235 unification.** The cover markup + geometry are now shared.
The geometry lives in the `RING_BOTTOM_CARVE_CX / RING_BOTTOM_CARVE_CY
/ RING_BOTTOM_CARVE_R` constants (`shared/qs-card-base.js`), and the
markup is emitted by the shared `_ringCarveCover({cx, cy, r, id,
show})` helper on `QsCardBase`. The car card reuses the SAME helper +
constants for its bottom-center `sun_btn_cover` (and the helper again,
with car-local geometry, for its `rabbit_btn_cover` / `time_btn_cover`).
There is no per-card `OVERRIDE_BTN_CARVE_*` / `SUN_BTN_CARVE_*`
duplication left:

```html
<g clip-path="url(#…ClipId)"> … animation paths … </g>
${this._ringCarveCover({ cx: RING_BOTTOM_CARVE_CX,
  cy: RING_BOTTOM_CARVE_CY, r: RING_BOTTOM_CARVE_R,
  id: 'override_btn_cover', show: e.override_reset })}
<path d="${bgPath}" … />  <!-- outer ring stroke -->
```

`RING_BOTTOM_CARVE_CY = 277` is the button centre y in SVG units
(derived from the CSS `.override-btn` position: button centre
`(150, 260)` CSS px, scaled by the SVG viewBox factor `320/300` →
`(160, 277.33)`, rounded to integer). `RING_BOTTOM_CARVE_R` is
the cover radius in SVG units; the integer is intentionally user-
tunable for visual iteration (tests pin the constant NAME only,
not the value). The cover applies uniformly across all backdrops
(including the climate card's flame / snow / wind / none).

**Why a cover overlay instead of a clipPath carve?** The first two
implementation rounds used an `<clipPath>` with three subpaths
(outer disc + carve disc + cancel subpath, evenodd fill rule) to
create a hole in the animation. Across two visual-iteration rounds
(R = 35 → 45 → 60) the user consistently reported a "lens" inside
the button: the intersection of the carve disc with the outer
clip disc is geometrically a lens (vesica piscis) shape, so the
visible HOLE always took on that shape regardless of R. The cover
overlay sidesteps the issue entirely — the patch is a `<circle>`
by construction, so the user sees a circle on screen. No lens
shape possible.

### QS-232 — Car-card electron-soup animation

QS-232 transplants the QS-200 / QS-211 / QS-214 boiler architecture
onto `qs-car-card.js`. The car's SOC reads as a translucent
"electron soup" inside the ring's clip circle: one slowly-moving
sine layer, lightning-blue sparkle particles popping inside the
soup ("electrons popping in the water"), and lightning bolts
flashing from the top of the clip circle to the soup surface when
the car is charging. The animation tracks the existing already-
mode-normalised `soc` variable (0–100), and the wave-level
mapping `0.2 + ratio × 0.6` matches the boiler envelope so the
visual character is consistent across the two water-style cards.

**Single-layer wave with dual-opacity cross-fade.** The issue
explicitly asks "do not superpose 3 layer of sin". The clipped
`<g clip-path="url(#${electronClipId})">` group contains ONE
`_generateWavePath(...)`-derived `d` string applied to TWO stacked
`<path>` siblings (`electron_wave_idle` and `electron_wave_charge`)
that share geometry but differ in `fill` (`IDLE_SOUP_COLOR`
`hsla(140, 80%, 50%, 0.12)` vs `CHARGE_SOUP_COLOR`
`hsla(180, 90%, 55%, 0.25)`) and `opacity`. The cross-fade is
binary: `_currentColorMix` lerps toward `_charging ? 1 : 0` with
`LERP_RATE = 2` (≈ 0.5 s time constant). Idle opacity =
`1 - colorMix`; charge opacity = `colorMix`. This is the same
dual-opacity idiom as the boiler's cool/boil cross-fade, adapted
to a single layer (boiler renders 6 paths; car renders 2).

**Sparkle palette switch + power-scaling.** Sparkles are always
visible. Idle: green (`SPARKLE_IDLE_COLOR` `hsla(140, 90%, 70%, 0.9)`),
very few (max 4), very small (radius 0.8–1.6 px), fixed spawn
rate `SPARKLE_IDLE_RATE_HZ = 0.6`. Charging: lightning blue
(`SPARKLE_CHARGE_COLOR` `#00E5FF`), with density, spawn-rate and
radius all scaling LINEARLY with charge power on the range
`[SPARKLE_POWER_MIN_W, SPARKLE_POWER_MAX_W] = [1500, 22000]` W —
12–28 max concurrent, 3–10 Hz spawn rate, 1.2–3.5 px radius. Below
1500 W (but still `_charging`) the values clamp to the
1500 W MIN endpoint per user intent (1500 W = MIN charging
density, NOT a ramp from 0 W). The idle→charging boundary at 50 W
is therefore a deliberate visible step in density and colour. Each
sparkle's `fill` attribute is decided ONCE at spawn from the
then-current `_charging` value and never re-assigned in the
advance loop — when `_charging` flips, new sparkles spawn with
the new colour while in-flight sparkles complete their natural
~0.45 s fade (a brief mixed-palette tail).

**Lightning bolts.** Only spawn when `charging && !degraded`. Each
bolt is a 3-segment jagged `<path d="M start L mid L end">`
(NOT polyline — `<path>` is consistent with the boiler precedent
and lets tests pin the `d=` attribute), starting near the top of
the clip circle, kinking laterally at the midpoint, and ending on
the water surface within the chord. Life is
`LIGHTNING_LIFE_S = 0.25` s with a three-phase opacity curve
(fade-in `[0, 0.10]`, hold `[0.10, 0.70]`, fade-out `[0.70, 1.0]`).
Spawn interval is randomised on
`[LIGHTNING_SPAWN_MIN_S, LIGHTNING_SPAWN_MAX_S] = [1.5, 3.0]` s, capped at
`MAX_CONCURRENT_LIGHTNING = 3` simultaneous bolts. The advance /
retire loop sits OUTSIDE the spawn gate so in-flight bolts
complete their fade naturally when charging stops. Glow is provided
by a single `<filter id="${lightningFilterId}">` with
`<feGaussianBlur stdDeviation="${LIGHTNING_GLOW_STDDEV}">` plus
`mix-blend-mode: screen` on the layer `<g>`.

**Wave fill — grain removed (QS-235 AC7).** The two wave `<path>`
elements now carry only `fill` (`IDLE_SOUP_COLOR` /
`CHARGE_SOUP_COLOR`) and `opacity`. The earlier `feTurbulence`
grain filter (a `<filter>` declaring `fractalNoise` composited
against `SourceGraphic`) was removed in QS-235 — a deliberate,
owner-authorized visual simplification isolated in its own commit
for a clean Phase-F smoke read. There is no per-instance grain id
or wave-fill `filter="url(#…)"` reference anymore.

**Energy-mode SOC on sensor dropout (QS-235 AC6 / review-fix #01
NTH1).** The `current_inputed_energy` read now goes through the
shared `_safeNumber(sensor, 0)`, which returns `0` (not `NaN`) for an
`unavailable` / `unknown` / missing sensor. In energy mode with no
valid target limit, the SOC readout and ring handle therefore show
`0` / `0 kWh` rather than the pre-QS-235 `--` / `-- kWh`. This is
intentional: besides being arguably more correct, it **hardens the
ring geometry** — the old `NaN` propagated through
`socPct → handlePct → pctToDeg → polar`, emitting a `cx="NaN"` handle
position; the `0` keeps the handle pinned at the bottom of the gauge.
Confirm the `0`-on-dropout readout in the Phase-F smoke.

**Estimated-SOC display + manual popup (QS-243).** The percent display is
keyed on the `is_soc_estimated` binary sensor, not raw staleness: an
absolute estimate renders as `NN%*` (the `*` flags an estimated value),
the pure-delta case (estimating with no base → SOC sensor unknown) renders
`+XX%`, and a live fresh sensor renders plain `NN%`. In percent mode the big
SOC number is clickable → a popup with an integer 0–100 input and **Save**
(`number.set_value` on `manual_soc`) / **Cancel** / **Reset**
(`button.press` on `reset_soc`). The template wires `is_soc_estimated`,
`manual_soc`, and `reset_soc` into the card. See
[car-soc-estimation.md](car-soc-estimation.md) for the backing model.

**Degraded state CSS filter.** When `degraded === true` (computed
as `isDisconnected || isFaulted || isStale` — `isOffGrid` is
explicitly excluded; the card-level pinkish `.off-grid` CSS class
already covers it), the clipped `<g>` carries the inline style
`filter: saturate(0.3) brightness(0.7);`. This is a runtime CSS
filter swap, NOT a third entry in the `_currentColorMix` lerp —
the lerp stays binary idle↔charge. Sparkles still spawn at idle
rate with idle-green colour (the CSS filter desaturates them);
lightning is suppressed entirely (`if (charging && !degraded)`
spawn gate), and in-flight bolts complete their natural fade.

**Continuous-RAF model.** The car card was migrated out of the
`_charging`-gated `test_card_raf_idle_gated` parametrize list
(previously gated on `charging`) and into the
`test_card_caps_raf_dt_against_hidden_tab` list (joining the
pool and water-boiler cards). `connectedCallback()` now calls
`this._startAnimation()` directly. The dt cap
`LERP_DT_CEIL = 0.1` clamps the first frame after a hidden-tab
return so phase advance and sparkle / lightning life increments
don't burst. The existing dashed-arc animation
(`<path id="charge_anim">`) is preserved verbatim, and
`showAnimation` remains its render-time switch.

**D17 — three carve-cover overlays.** Unlike the
boiler / radiator / climate cards which carry only one
`override_reset` button cover (QS-217), the car card has THREE
buttons sitting inside the clip disc: the `sun-btn`
(`<div id="sun_btn">`, Solar priority, bottom-center stack), the
`rabbit-btn` (`<div id="rabbit_btn">`, Force now, left column of
the mini-grid), and the `time-btn` (`<div id="time_btn">`,
Finish time, right column of the mini-grid). Each gets a
QS-217-style `<circle id="…_cover" fill="var(--card-background-color)"
pointer-events="none" />` drawn AFTER the closing `</g>` of the
clipped animation group and BEFORE the `<path d="${bgPath}"`
outer-ring stroke, gated on its corresponding render predicate —
`swPriority` for `sun_btn_cover`, `e.force_now` for
`rabbit_btn_cover`, `(tNext && e.schedule)` for `time_btn_cover`.
The CY/CX integer values are user-tunable; tests pin the constant
NAMES only (mirrors the QS-217 R-tuning iteration). The
`target_handle` circle (already on the ring stroke at radius
`ringCirc = 130`) does NOT need a carve — it sits outside the
clip disc by construction.

## Hardened JS-card patterns (QS-194 review-fix #03)

Every JS card in `ui/resources/` follows the same defensive
patterns after the cross-card audit:

- **RAF idle-gating (M4).** Each card defines `_startAnimation()` and
  `_stopAnimation()` helpers. The render path starts/stops the
  animation conditionally — on `showAnimation` for the
  duration-based cards (`qs-on-off-duration-card`, `qs-radiator-card`,
  `qs-climate-card`) — and continuously while connected for the
  three water-style cards (`qs-pool-card`, `qs-water-boiler-card`,
  and `qs-car-card` per QS-232) whose wave is intrinsically visible.
  `disconnectedCallback` always stops the loop. The
  `_charging`-gated form previously used by `qs-car-card.js` was
  retired by QS-232 — see "QS-232 — Car-card electron-soup
  animation" above.
- **try/finally around service calls (M2).** Every
  `_isProcessing*` flag setter is wrapped in `try { await
  this._select(...) } finally { setTimeout(() => _isProcessing... =
  false, ...) }` so a rejected HA service call can't wedge the
  card.
- **Interaction-flag reset on disconnect (S7).** Every card resets
  every `_isInteracting*` / `_isProcessing*` / `_modalOpen` flag in
  `disconnectedCallback` so a re-attach mid-interaction doesn't
  silently short-circuit `set hass` on stale state.
- **HTML escaping (S6).** Each card carries an `_escapeHtml(s)`
  helper applied to user-/third-party-controlled strings
  interpolated into `innerHTML` (card title, dialog title, dialog
  message, sensor unit).
- **Safe numeric coercion (S8, water-boiler).** The water-boiler
  card uses `_safeNumber(sensor, default)` instead of
  `Number(s?.state || N)` so degenerate states
  (`""` / `unknown` / `unavailable`) can't propagate `NaN` into SVG
  path attributes.
- **Duration-card `displayTargetHours` pattern (QS-237).** All five
  duration cards (pool, on_off_duration, radiator, water-boiler,
  climate) read `displayTargetHours = isDefaultMode ? defaultDuration
  : targetHours` and use that single value for both the handle and
  the big target-value span — so a drag-commit propagates instantly
  through HA's `default_on_duration` number-entity push, not via the
  lagging constraint `duration_limit`. Drag is gated on
  `isDefaultMode`; QS-237 brought pool into this pattern. Pool's
  drag gate is `isEnabled && isDefaultMode && !!e.default_on_duration`
  (review-fix #01 S1 + #02 N3): the `displayTargetHours > 0` term
  was deliberately dropped to keep drag-recovery reachable after a
  drag-to-zero commit on the half-hour snap grid, and the third
  `e.default_on_duration` term protects against a dashboard
  template that omits the number-entity key (which would silently
  no-op `_setNumber`). Pool's big-text renders
  `_fmt(displayTargetHours, false)` (un-rounded) so the committed
  display matches the `dragMove` live preview on the half-hour
  grid (review-fix #01 N6) — a card-local divergence from the
  family's default `round=true`. The water-fill `progressRatio` in
  default mode is additionally gated on a `defaultDurationKnown`
  flag (review-fix #02 N1) so the boot-window fallback
  `_safeNumber(sDefaultOnDuration, 1)` doesn't render a full water
  bowl when `hoursRun > 1`.
- **Zero-`maxHours` clamp (post-QS-195 user bug).** Both the
  radiator and water-boiler cards clamp `maxHours = targetHours > 0 ?
  targetHours : <fallback>` in the non-default-mode branch. A
  brand-new device whose constraint sensor reports `0` would
  otherwise divide by zero in `hoursToPct`, propagate `NaN` into the
  arc-path math, and emit `M ... A 130 130 0 0 1 NaN NaN` in the
  rendered SVG `d` attribute — the browser shows a "Configuration
  error" and the card refuses to render.
- **Arc-path NaN guard (defense-in-depth).** Both cards' `arcPath`
  helper short-circuits with `if (!Number.isFinite(a0) ||
  !Number.isFinite(a1)) return '';` so even if upstream math escapes
  the clamp, the SVG `d` attribute stays well-formed.
- **Local-state cleanup symmetry (S9).** Every Apply handler that
  sets a `_localFinishTimeMins` / `_localNextTimeMins` schedules a
  matching 5-second clear timer so out-of-band backend updates
  aren't masked indefinitely (mirrors the existing
  `_localTargetPct` pattern).
- **Modal dismiss path (N12) + activate try/finally (N13).** The
  shared `showDialog` helper falls back to a "Close" button when
  `buttons` is empty, and the per-button `activate` closure wraps
  `b.onClick?.()` in `try/finally` so a synchronous throw can't
  leave the modal locked open.

### Ring text readability — uniform shadow (QS-228)

Every QS Lovelace card declares
`--ring-text-shadow: 0 0 12px rgba(0,0,0,0.8), 0 2px 4px rgba(0,0,0,0.5)`
on `:host` and applies `text-shadow: var(--ring-text-shadow)` to every
inside-the-ring text class across all six cards (`qs-car-card`,
`qs-climate-card`, `qs-on-off-duration-card`, `qs-pool-card`,
`qs-radiator-card`, `qs-water-boiler-card`). The shadow originated in
QS-201 for the radiator's warm flame and the pool's animated water;
QS-228 extends it to the remaining cards for visual consistency. The
contract is pinned by
`tests/test_dashboard_rendering.py::TestDashboardTemplateRendering::test_all_qs_cards_apply_ring_text_shadow_for_readability`
(the per-card `CARDS_TO_RING_TEXT_CLASSES` mapping enumerates which
classes each card must shadow).

Follow these patterns when adding new JS cards. 

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
