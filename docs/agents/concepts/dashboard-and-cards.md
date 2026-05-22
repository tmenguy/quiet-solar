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
last_verified: 2026-05-22
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
  visible at all times (cool blue when not running, near-white
  translucent when boiling) so the RAF loop itself is no longer
  conditional. `showAnimation` survives only as a render-time switch
  for the existing dashed-arc `<path id="running_anim">`. The
  boiler is therefore removed from `test_card_raf_idle_gated`'s
  parametrize list — that test pins the idle-gated model that the
  other duration cards still follow.
- **Dual-layer water cross-fade + bubbles + surface glow.** Inside
  a circular `clipPath` (radius `CLIP_R = 120`), the card renders
  six wave paths in pairs: `wave{0,1,2}_cool` (cool-blue HSLA) and
  `wave{0,1,2}_boil` (near-white HSLA). Each pair shares geometry;
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
visible snap.

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

## Hardened JS-card patterns (QS-194 review-fix #03)

Every JS card in `ui/resources/` follows the same defensive
patterns after the cross-card audit:

- **RAF idle-gating (M4).** Each card defines `_startAnimation()` and
  `_stopAnimation()` helpers. The render path starts/stops the
  animation conditionally — on `showAnimation` for the
  duration-based cards, on `_charging` for the car card, and
  continuously while connected for the pool wave (intrinsically
  visible). `connectedCallback` no longer kicks off RAF
  unconditionally; `disconnectedCallback` always stops it.
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
