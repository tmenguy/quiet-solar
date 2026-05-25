/*
  QS Car Card - custom:qs-car-card
  Zero-build single-file Lit-style web component compatible with Home
  Assistant.

  QS-232 layers an "electron soup" SOC animation inside the ring,
  mirroring the QS-200 / QS-211 / QS-214 boiler architecture. The
  high-level design — single-layer sine wave inside a clipPath,
  idle↔charge cross-fade, lightning-blue sparkle particles, lightning
  bolts during charging, feTurbulence grain on the wave fill,
  degraded-state CSS desaturate, continuous RAF, and QS-217 carve
  covers extended to three inside-disc buttons — is documented in
  `docs/agents/concepts/dashboard-and-cards.md` (search for
  "QS-232 — Car-card electron-soup animation"). Tuning constants
  (sparkle power-scaling range, lightning spawn interval, life
  curves, palette colours) live in the constants block immediately
  below this header — those values are the source of truth; the
  docstring intentionally avoids duplicating them.

  Existing dashed-arc animation (`<path id="charge_anim">`) is
  preserved verbatim; `showAnimation` remains its render-time switch.
*/

const INVALID_STATES = ['unavailable', 'unknown', 'none'];

const ANIM_MIN_SPEED = 20;   // dash-units per second at minimum power
const ANIM_MAX_SPEED = 200;  // dash-units per second at maximum power
const ANIM_MIN_POWER_W = 500;
const ANIM_MAX_POWER_W = 22000;
const ANIM_SPEED_RANGE = ANIM_MAX_SPEED - ANIM_MIN_SPEED;
const ANIM_POWER_RANGE = ANIM_MAX_POWER_W - ANIM_MIN_POWER_W;

// === QS-232 electron-soup constants ===

// --- Geometry (must match the SVG <clipPath> circle attributes below) ---
const CENTER_CX = 160;              // SVG x-center of the ring / clip circle
const CENTER_CY = 160;              // SVG y-center of the ring / clip circle
const CLIP_R = 120;                 // soup clip circle radius (10 px inside ringCirc=130, same as boiler)
const WAVE_WIDTH = 480;             // single wave period (2× clip diameter)
const WAVE_BOTTOM_Y = 400;          // closing rectangle y; clipped by circle

// --- Wave palette (dual-opacity cross-fade idle ↔ charge) ---
// Review-fix #02: user reported the idle green was too dark / muddy
// to read at a glance. Bumped from `hsla(140, 80%, 50%, 0.12)` →
// `hsla(140, 85%, 60%, 0.30)` — slightly brighter, slightly more
// saturated, 2.5× more opaque. Charging blue bumped in parallel
// so the cross-fade contrast remains balanced.
const IDLE_SOUP_COLOR   = 'hsla(140, 85%, 60%, 0.30)';   // electric green, clearly visible
const CHARGE_SOUP_COLOR = 'hsla(180, 90%, 60%, 0.38)';   // bright cyan-blue, equally visible

// --- Animation tuning (lerp envelope; mirror boiler) ---
const LERP_RATE = 2;
const LERP_DT_CEIL = 0.1;
const AMP_REGEN_THRESHOLD = 0.25;
const LEVEL_REGEN_THRESHOLD = 0.01;
const PHASE_WRAP = 1e6;
const PHASE_TO_PX = 60;

// --- Calm vs charge targets (single layer, gentler than boiler) ---
const CALM_AMP = 1.2;
const CALM_SPEED = 0.03;            // very slow per issue brief
const CHARGE_AMP = 3.5;
const CHARGE_SPEED = 0.15;          // still gentle

// --- Sparkles ("electrons popping in the water") ---
// Idle: always visible, low rate, vivid yellow-green burst-shaped
// "electron pops" rather than smooth discs. Each sparkle is an
// 8-ray asterisk path (4 long rays in a cross + 4 short rays in
// an X) drawn with stroke (no fill) — see the spawn code in
// `_startAnimation` for the exact `d` builder. Mimics the
// little-explosion mental model of an electron flashing in the
// soup, vs. the smooth-disc look the previous `<circle>` form
// gave.
//
// Review-fix history:
// - #01 #2: radii roughly doubled (0.8/1.6 → 1.6/3.0).
// - #03:    bumped further — user reported sparkles became invisible
//           against the brighter idle soup palette.
// - #04:    shape changed `<circle>` → 8-ray `<path>` burst; colour
//           changed near-white mint → vivid yellow-green (hue 80)
//           per user feedback "they have to be green/yellow ... not
//           a disc as today but should be more like a little
//           explosion".
const SPARKLE_IDLE_MAX = 8;
const SPARKLE_IDLE_RATE_HZ = 1.5;
const SPARKLE_IDLE_RADIUS_MIN = 2.5;
const SPARKLE_IDLE_RADIUS_MAX = 5.0;
const SPARKLE_IDLE_COLOR = 'hsla(80, 100%, 65%, 0.95)';    // vivid yellow-green / lime, high contrast vs soup green
// Stroke-width for the asterisk-style sparkle path. Tuned so a 5 px
// sparkle still reads as a "spark" rather than a clump.
const SPARKLE_STROKE_WIDTH = 1.4;
// Charging endpoints — scale linearly on [SPARKLE_POWER_MIN_W, SPARKLE_POWER_MAX_W].
// Below MIN (but still `_charging`) the values clamp to the MIN-power
// endpoint per user intent (1500W = MIN charging density, NOT a ramp
// from 0W). The idle→charging boundary at 50W is a deliberate
// visible step in density and colour.
const SPARKLE_POWER_MIN_W = 1500;
const SPARKLE_POWER_MAX_W = 22000;
const SPARKLE_CHARGE_MAX_AT_MIN_POWER = 12;
const SPARKLE_CHARGE_MAX_AT_MAX_POWER = 28;
const SPARKLE_CHARGE_RATE_HZ_AT_MIN_POWER = 3;
const SPARKLE_CHARGE_RATE_HZ_AT_MAX_POWER = 10;
const SPARKLE_CHARGE_RADIUS_MIN_AT_MIN_POWER = 2.4;
const SPARKLE_CHARGE_RADIUS_MAX_AT_MIN_POWER = 4.5;
const SPARKLE_CHARGE_RADIUS_MIN_AT_MAX_POWER = 3.4;
const SPARKLE_CHARGE_RADIUS_MAX_AT_MAX_POWER = 6.5;
const SPARKLE_CHARGE_COLOR = '#00E5FF';   // electric "lightning blue"
const SPARKLE_LIFE_S = 0.45;

// --- Lightning bolts ("thunder from the top to the soup surface") ---
// Only spawn when `_charging === true && !degraded`. In-flight bolts
// complete their fade-out naturally on charging → false.
//
// Each bolt is a FILLED tapered polygon (Zeus-thunderbolt shape):
// wide at the top, zig-zags via two lateral kinks, narrows to a
// point at the soup surface. The previous form used a uniform-width
// stroked polyline (~1.8 px) and a near-white colour that bloomed
// to grey through the glow filter — user feedback: "they should be
// light and bright blue, wider at top and reducing to the bottom
// … like the ones from Zeus".
const MAX_CONCURRENT_LIGHTNING = 3;
const LIGHTNING_SPAWN_MIN_S = 1.5;
const LIGHTNING_SPAWN_MAX_S = 3.0;
const LIGHTNING_LIFE_S = 0.25;
// Fill colour for the bolt polygon. The name `LIGHTNING_STROKE_COLOR`
// is retained for backward-compat with the AC-6 test (which pins the
// constant NAME); semantically it's now a fill colour. Value is the
// "Light Blue A400" Material Design tone — saturated enough to read
// as electric blue even after the Gaussian-blur glow softens edges.
const LIGHTNING_STROKE_COLOR = '#33B5FF';
const LIGHTNING_GLOW_STDDEV = 2.5;
const LIGHTNING_TOP_MARGIN_PX = 4;
const LIGHTNING_LATERAL_JITTER_PX = 18;
// Tapering widths for the thunderbolt outline (in SVG units).
// `LIGHTNING_TOP_WIDTH` is the width at the top of the bolt; the
// path then narrows to a point at the soup surface via two
// intermediate widths (~55% and ~25% of the top, computed inline).
const LIGHTNING_TOP_WIDTH = 9;
// Review-fix #01 #16: removed unused `LIGHTNING_SEGMENTS = 3` —
// the lightning path hardcodes its skeleton inline rather than
// driving an emission loop, so the constant served no purpose.
// Review-fix #02 user follow-up: removed `LIGHTNING_STROKE_WIDTH`
// — the new tapered-polygon shape has no uniform stroke; widths
// vary along the skeleton via `LIGHTNING_TOP_WIDTH` and its
// inline percentages.

// --- Inside-disc button carve-out covers (QS-232 D17) ---
// Mirror of QS-217 for THREE buttons that sit inside the clip disc.
// The sun-btn (Solar priority, bottom-center) now follows the EXACT
// same pattern as `override-btn` on the boiler/radiator/climate
// cards — absolute-positioned at `bottom: 15px; left: 50%;` in the
// .ring container, carved with CY=277 / R=35 (matching
// OVERRIDE_BTN_CARVE_* on those cards). Rabbit-btn (Force now, left
// column of mini-grid) and time-btn (Finish time, right column) sit
// inside the .stack flow and are carved with CY=215 / R=32.
//
// Review-fix iteration history:
// - QS-232 initial:    SUN=267, RABBIT/TIME=180,180  R=32
// - Review-fix #01:    SUN=285, RABBIT/TIME=195,195  R=32 (still too high)
// - Review-fix #02:    SUN=288, RABBIT/TIME=192,192  R=38 (R bump made things worse)
// - Review-fix #03:    SUN=277 R=35, RABBIT/TIME=215 R=32 — sun matches
//                      OVERRIDE_BTN_CARVE_CY/R from the other cards.
// - Review-fix #05:    RABBIT/TIME=200 R=32 — user "move the holes
//                      UP by 15 pixels" from the 215 deployment.
// - Review-fix #06:    RABBIT/TIME=206 R=32 — user measured "hole
//                      apex 22 image px above button apex, lower
//                      by 11 image px to align". With a ~1.9× retina
//                      screenshot scale (the user measured the
//                      circle diameter at ~490 image px ÷ ~258 CSS
//                      px ring-stroke diameter), 11 image px ≈
//                      5.5 CSS px ≈ 6 SVG units → 200 + 6 = 206.
//                      Sun stays at 277 (matches override-btn).
// - The sun-btn DOM was restructured to a direct child of the
//   .ring container (out of `.center > .stack > .center-controls`)
//   so its CSS layout matches override-btn exactly.
const SUN_BTN_CARVE_CX = 160;       // bottom-center of ring
const SUN_BTN_CARVE_CY = 277;       // matches OVERRIDE_BTN_CARVE_CY in other cards
const SUN_BTN_CARVE_R  = 35;        // matches OVERRIDE_BTN_CARVE_R in other cards
const RABBIT_BTN_CARVE_CX = 96;     // left column of mini-grid
const RABBIT_BTN_CARVE_CY = 206;
const RABBIT_BTN_CARVE_R  = 32;
const TIME_BTN_CARVE_CX = 224;      // right column of mini-grid
const TIME_BTN_CARVE_CY = 206;
const TIME_BTN_CARVE_R  = 32;

class QsCarCard extends HTMLElement {
  constructor() {
    super();
    this._chargePower = 0;
    this._charging = false;
  }

  // QS-232: generate an SVG path for a sine-wave-based closed shape
  // (ported verbatim from `qs-water-boiler-card.js` / `qs-pool-card.js`).
  // Emits TWO repetitions of the wave (path extent = [0, 2*width]) so
  // the translated path always covers the clip region regardless of
  // scroll offset. The car uses ONE call site with `frequency = 2`
  // (single layer per D1).
  _generateWavePath(width, amplitude, frequency, phase, yOffset) {
    const points = [];
    const stepsPerPeriod = 60;
    const totalSteps = stepsPerPeriod * 2;
    const totalWidth = 2 * width;
    for (let i = 0; i <= totalSteps; i++) {
      const x = (i / stepsPerPeriod) * width;
      const y = yOffset + amplitude * Math.sin((x / width) * frequency * 2 * Math.PI + phase);
      points.push(`${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    return `M ${points[0]} ` +
           points.slice(1).map(p => `L ${p}`).join(' ') +
           ` L ${totalWidth.toFixed(2)} ${WAVE_BOTTOM_Y} L 0 ${WAVE_BOTTOM_Y} Z`;
  }

  // QS-232: reset cached DOM refs and the logical sparkle / lightning
  // arrays. Shared between `_invalidateAnimCache()` (full memo reset
  // on (re-)connect) and the post-`innerHTML` cleanup block in
  // `_render()`. Factored out so a future memo-key addition lands at
  // BOTH call sites — without the helper, the two paths would drift
  // silently and the post-render block would carry stale refs into
  // the next RAF frame (boiler precedent —
  // `test_water_boiler_card_factors_reset_dom_refs_helper`).
  _resetDomRefs() {
    this._waveEls = null;             // [idleWave, chargeWave]
    // Review-fix #01 #14: removed `_grainFilterEl` — no code path
    // ever read or lazily-resolved it. The grain filter is declared
    // once in `<defs>` and referenced by the wave paths via
    // `filter="url(#${grainFilterId})"`; no JS-side ref needed.
    this._sparkleLayerEl = null;
    this._lightningLayerEl = null;
    this._sparkles = [];
    this._lightningBolts = [];
  }

  // QS-232: clear wave-path memoization keys and cached DOM refs.
  // Called on every (re-)connect and after each _render() innerHTML
  // rewrite. Mirrors boiler `_invalidateWaveCache` (it nulls
  // `_lastWaterBaseY` / `_lastAmplitude` for full invalidation; the
  // post-innerHTML path syncs them to the current render state, so
  // they stay outside `_resetDomRefs()`).
  _invalidateAnimCache() {
    this._lastWaterBaseY = null;
    this._lastAmplitude = null;
    this._resetDomRefs();
  }

  // QS-232: continuous RAF while connected, mirroring
  // `qs-water-boiler-card.js`. The electron-soup wave is intrinsically
  // visible at all times (idle green when not charging, charging blue
  // when `_charging`, desaturated grey when `degraded`) so RAF is no
  // longer gated on `_charging`. Calm vs. charging is amplitude /
  // speed / color-mix lerp, not RAF on/off. The existing dashed-arc
  // animation (`<path id="charge_anim">`) is preserved verbatim and
  // continues to be driven by `_animOffset` inside the step closure
  // below — `showAnimation` remains its render-time switch.
  _startAnimation() {
    if (this._animRaf != null) return;
    // Review-fix #01 #22: bail if `_root` isn't yet attached. A
    // `connectedCallback` that fires before `setConfig` would
    // otherwise spin RAF with every DOM lookup returning null. The
    // re-entry happens via `setConfig` → `_render` (which doesn't
    // call `_startAnimation` directly today) — for safety, we also
    // re-attempt from `connectedCallback` if it runs first.
    if (!this._root) return;
    // Lazy-init: each RAF-state field guards itself individually.
    // Review-fix #02 #1: the previous form bundled ALL five
    // RAF-state fields (`_currentAmplitude`, `_wavePhase`,
    // `_nextSparkleAt`, `_nextLightningAt`, `_sparkles`,
    // `_lightningBolts`) inside `if (this._currentAmplitude == null)`,
    // and review-fix #01 #4 added a `_needsAnimationPrime` consumer
    // in `_render()` that pre-seeds `_currentAmplitude` / `_currentSpeed`
    // / `_currentColorMix`. Combined effect: the lazy-init guard
    // evaluated FALSE on first `_startAnimation()`, leaving
    // `_wavePhase` / `_nextSparkleAt` / `_nextLightningAt` /
    // `_sparkles` / `_lightningBolts` undefined. First RAF tick:
    // `_wavePhase += dt` = NaN, sparkle/lightning spawn loops never
    // enter (since `NaN <= 0 === false`). User-visible symptom:
    // "no sparkles at all". Per-field guards eliminate the coupling.
    if (this._currentAmplitude == null) this._currentAmplitude = CALM_AMP;
    if (this._currentSpeed     == null) this._currentSpeed     = CALM_SPEED;
    if (this._currentColorMix  == null) this._currentColorMix  = 0;
    if (this._wavePhase        == null) this._wavePhase        = 0;
    if (this._sparkles         == null) this._sparkles         = [];
    if (this._nextSparkleAt    == null) this._nextSparkleAt    = 0;
    if (this._lightningBolts   == null) this._lightningBolts   = [];
    if (this._nextLightningAt  == null) this._nextLightningAt  = LIGHTNING_SPAWN_MIN_S +
      Math.random() * (LIGHTNING_SPAWN_MAX_S - LIGHTNING_SPAWN_MIN_S);
    // `_needsAnimationPrime` is armed by `setConfig` (review-fix #01 #4)
    // and consumed by `_render`'s prime block; only set here on the
    // very first lazy-init if neither `setConfig` nor `_render` has
    // touched it yet.
    if (this._needsAnimationPrime == null) this._needsAnimationPrime = true;
    this._lastAnimTs = null;
    this._invalidateAnimCache();

    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      // QS-232 (mirror QS-200 fix #02 S6): cap `dt` against hidden-tab
      // return. Without this cap, the first frame after a multi-second
      // tab-hidden window produces a huge dt, snapping wave phase by
      // hundreds of pixels in one frame and aging every sparkle past
      // SPARKLE_LIFE_S simultaneously. The cap matches LERP_DT_CEIL
      // so every step-loop subsystem shares the same envelope.
      let dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      dt = Math.min(dt, LERP_DT_CEIL);
      this._lastAnimTs = ts;

      // --- Existing dashed-arc animation (preserved from the prior
      //     car-card revision). The dashed `<path id="charge_anim">`
      //     still drives off `showAnimation`, which is applied as a
      //     render-time switch on the element. The RAF loop itself
      //     is no longer gated on `_charging` — QS-232 D9 migrated
      //     the card to the continuous-RAF model.
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const cp = this._chargePower || 0;
      const dashSpeed = Math.min(ANIM_MAX_SPEED, Math.max(ANIM_MIN_SPEED, ANIM_MIN_SPEED + (cp - ANIM_MIN_POWER_W) * ANIM_SPEED_RANGE / ANIM_POWER_RANGE));
      this._animOffset = ((this._animOffset || 0) + dashSpeed * dt) % patternLen;
      const dashEl = this._root?.getElementById('charge_anim');
      if (dashEl) {
        dashEl.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }

      // --- Lerp amplitude / speed / colorMix toward charge targets.
      // The lerp is binary (idle ↔ charge); the degraded palette is
      // a separate runtime path via CSS filter on the clip group (D8).
      const charging = this._charging === true;
      const degraded = this._degraded === true;
      const targetAmplitude = charging ? CHARGE_AMP : CALM_AMP;
      const targetSpeed     = charging ? CHARGE_SPEED : CALM_SPEED;
      const targetColorMix  = charging ? 1 : 0;
      const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
      this._currentAmplitude += (targetAmplitude - this._currentAmplitude) * lerpFactor;
      this._currentSpeed     += (targetSpeed - this._currentSpeed) * lerpFactor;
      this._currentColorMix  += (targetColorMix - this._currentColorMix) * lerpFactor;
      this._wavePhase += this._currentSpeed * dt;
      this._wavePhase = ((this._wavePhase % PHASE_WRAP) + PHASE_WRAP) % PHASE_WRAP;

      // --- Lazy-resolve wave / sparkle / lightning DOM refs once per
      //     innerHTML rewrite (two wave nodes — idle + charge).
      //     The inner `?? null` is intentional: `_root?.getElementById`
      //     can return `undefined` (when `_root` itself is null), and
      //     the retry guard below distinguishes a successful cache
      //     (truthy elements) from a stale `[null, null]` cache hit.
      //     Review-fix #02 #10: retry the lookup if EITHER slot is
      //     null (the previous form cached `[null, null]` — truthy —
      //     permanently if both lookups missed during a transient
      //     detach).
      // Review-fix #02 #12: align all three DOM-ref lookups on the
      // block-form (was: inline `?? (... = ...)` for sparkle/lightning
      // and block-form for `_waveEls`). One shape across all layers
      // makes the retry-on-null behaviour easier to reason about.
      if (!this._waveEls || !this._waveEls[0] || !this._waveEls[1]) {
        const idleEl   = this._root?.getElementById('electron_wave_idle')   ?? null;
        const chargeEl = this._root?.getElementById('electron_wave_charge') ?? null;
        this._waveEls = [idleEl, chargeEl];
      }
      if (!this._sparkleLayerEl) {
        this._sparkleLayerEl = this._sparkleLayerId
          ? (this._root?.getElementById(this._sparkleLayerId) ?? null)
          : null;
      }
      if (!this._lightningLayerEl) {
        this._lightningLayerEl = this._lightningLayerId
          ? (this._root?.getElementById(this._lightningLayerId) ?? null)
          : null;
      }
      const sparkleLayer = this._sparkleLayerEl;
      const lightningLayer = this._lightningLayerEl;

      // --- Wave transform (single layer translateX).
      // Path extent is [0, 2*WAVE_WIDTH] so the translated path always
      // covers the clip region.
      const raw = this._wavePhase * PHASE_TO_PX;
      const scrollOffset = ((raw % WAVE_WIDTH) + WAVE_WIDTH) % WAVE_WIDTH;
      const tx = -CLIP_R - scrollOffset;
      const txStr = `translateX(${tx.toFixed(1)}px)`;
      const idleWave   = this._waveEls[0];
      const chargeWave = this._waveEls[1];
      if (idleWave)   idleWave.style.transform   = txStr;
      if (chargeWave) chargeWave.style.transform = txStr;

      // --- Wave path regen (throttled by amplitude / soup-level delta).
      const waterBaseY = this._waterBaseY;
      // Review-fix #02 #11: use `Number.isFinite(...)` instead of
      // `!Number.isNaN(...)` so a spurious `Infinity` doesn't slip
      // through the guard. Asymmetric otherwise with the
      // `Number.isFinite(soc)` guard in `_render` (fix #01 #3) and
      // the call-site guard on `initialWavePath` (fix #01 #5).
      const hasValidBase = waterBaseY != null && Number.isFinite(waterBaseY);
      // Review-fix #02 #5: the in-RAF read AND write must guard
      // against non-finite `_lastAmplitude`. Fix-#01-#8 only patched
      // the post-innerHTML write at the end of `_render`; if NaN
      // ever lands in `_lastAmplitude` here mid-frame, all
      // subsequent regen checks evaluate `Math.abs(_ - NaN) = NaN`
      // (always `< threshold`), freezing the wave-path regen for
      // the lifetime of the card.
      const ampDelta = (this._lastAmplitude == null || !Number.isFinite(this._lastAmplitude))
          ? Infinity
          : Math.abs(this._currentAmplitude - this._lastAmplitude);
      const levelChanged = hasValidBase &&
          Math.abs(waterBaseY - (this._lastWaterBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
      if (hasValidBase && (levelChanged || ampDelta > AMP_REGEN_THRESHOLD)) {
        this._lastWaterBaseY = waterBaseY;
        this._lastAmplitude = Number.isFinite(this._currentAmplitude) ? this._currentAmplitude : CALM_AMP;
        const d = this._generateWavePath(WAVE_WIDTH, this._currentAmplitude, 2, 0, waterBaseY);
        if (idleWave)   idleWave.setAttribute('d', d);
        if (chargeWave) chargeWave.setAttribute('d', d);
      }

      // --- Per-frame wave opacity (idle ↔ charge cross-fade).
      // Dual-layer opacity rather than HSL lerp — passing through
      // yellow at the midpoint is artifact-prone. Review-fix #01 #18:
      // clamp `_currentColorMix` to `[0, 1]` for opacity emission so
      // floating-point lerp drift (e.g., `1.0000000001`) doesn't
      // surface as ugly `"-0.000"` opacity tokens.
      // Review-fix #02 #9: `Math.max(0, Math.min(1, NaN)) === NaN`,
      // so the clamp alone doesn't trap NaN. Wrap with a
      // `Number.isFinite` ternary so a poisoned lerp falls back to
      // the idle palette (`mix = 0`).
      const mix = Number.isFinite(this._currentColorMix)
        ? Math.max(0, Math.min(1, this._currentColorMix)) : 0;
      const idleOpacity   = (1 - mix).toFixed(3);
      const chargeOpacity = mix.toFixed(3);
      if (idleWave)   idleWave.setAttribute('opacity', idleOpacity);
      if (chargeWave) chargeWave.setAttribute('opacity', chargeOpacity);

      // === QS-232 sparkle subsystem ===========================
      // Always visible. Colour / density / spawn-rate / radius
      // depend on `_charging` state. Spawns advance even when idle
      // (low rate). The advance + retire loop runs every frame for
      // graceful exit on `_charging` flips (in-flight sparkles
      // complete their natural fade).
      if (sparkleLayer && hasValidBase) {
        // Power-scaling for the charging endpoints (clamped to
        // [SPARKLE_POWER_MIN_W, SPARKLE_POWER_MAX_W]).
        // Review-fix #01 #20: defensively floor negative chargePower
        // (V2L discharge / sensor glitch) to 0 at the source so
        // downstream scaling never receives a negative `power`.
        let sparkleMax, sparkleRateHz, rMin, rMax;
        if (charging) {
          const power = Math.max(0, this._chargePower || 0);
          const clamped = Math.max(SPARKLE_POWER_MIN_W, Math.min(SPARKLE_POWER_MAX_W, power));
          const t = (clamped - SPARKLE_POWER_MIN_W) / (SPARKLE_POWER_MAX_W - SPARKLE_POWER_MIN_W);
          sparkleMax = SPARKLE_CHARGE_MAX_AT_MIN_POWER + t * (SPARKLE_CHARGE_MAX_AT_MAX_POWER - SPARKLE_CHARGE_MAX_AT_MIN_POWER);
          sparkleRateHz = SPARKLE_CHARGE_RATE_HZ_AT_MIN_POWER + t * (SPARKLE_CHARGE_RATE_HZ_AT_MAX_POWER - SPARKLE_CHARGE_RATE_HZ_AT_MIN_POWER);
          rMin = SPARKLE_CHARGE_RADIUS_MIN_AT_MIN_POWER + t * (SPARKLE_CHARGE_RADIUS_MIN_AT_MAX_POWER - SPARKLE_CHARGE_RADIUS_MIN_AT_MIN_POWER);
          rMax = SPARKLE_CHARGE_RADIUS_MAX_AT_MIN_POWER + t * (SPARKLE_CHARGE_RADIUS_MAX_AT_MAX_POWER - SPARKLE_CHARGE_RADIUS_MAX_AT_MIN_POWER);
        } else {
          sparkleMax = SPARKLE_IDLE_MAX;
          sparkleRateHz = SPARKLE_IDLE_RATE_HZ;
          rMin = SPARKLE_IDLE_RADIUS_MIN;
          rMax = SPARKLE_IDLE_RADIUS_MAX;
        }
        // Review-fix #01 #19: `Math.floor(sparkleMax)` for the
        // concurrent-cap comparison so a fractional max (e.g. 18.5
        // from linear interpolation between 12 and 28) doesn't read
        // confusingly when compared against `length`.
        const sparkleCap = Math.floor(sparkleMax);

        this._nextSparkleAt -= dt;
        while (this._nextSparkleAt <= 0 && this._sparkles.length < sparkleCap) {
          // Spawn position: random inside the water portion of the
          // clip circle. `continue`-with-cadence-advance pattern
          // (no `break`) avoids a per-frame spin-loop on degenerate
          // empty soup, mirroring QS-214 review-fix #01 #5.
          const cyMin = this._waterBaseY + 2;
          const cyMax = CENTER_CY + CLIP_R - 4;
          if (cyMax <= cyMin) {
            this._nextSparkleAt += 1 / sparkleRateHz;
            continue;
          }
          const cy = cyMin + Math.random() * (cyMax - cyMin);
          const dy = cy - CENTER_CY;
          const chordHalf = Math.sqrt(Math.max(0, CLIP_R * CLIP_R - dy * dy));
          if (chordHalf < 2) {
            this._nextSparkleAt += 1 / sparkleRateHz;
            continue;
          }
          const cx = CENTER_CX + (Math.random() * 2 - 1) * chordHalf * 0.9;
          const r = rMin + Math.random() * (rMax - rMin);
          // Fill decided ONCE at spawn — never re-assigned in advance.
          const fill = charging ? SPARKLE_CHARGE_COLOR : SPARKLE_IDLE_COLOR;
          // Review-fix #02 user follow-up: sparkles are now 8-ray
          // asterisk bursts (4 long axial rays + 4 short diagonal
          // rays) drawn with stroke, evoking a small electron-soup
          // flash. The path origin is the spawn (cx, cy); ray length
          // scales with `r` from the power-scaling formula.
          const longR  = r;
          const shortR = r * 0.6;
          const cxs = cx.toFixed(2);
          const cys = cy.toFixed(2);
          const d = `M ${cxs},${(cy - longR).toFixed(2)} L ${cxs},${(cy + longR).toFixed(2)}`
                  + ` M ${(cx - longR).toFixed(2)},${cys} L ${(cx + longR).toFixed(2)},${cys}`
                  + ` M ${(cx - shortR).toFixed(2)},${(cy - shortR).toFixed(2)} L ${(cx + shortR).toFixed(2)},${(cy + shortR).toFixed(2)}`
                  + ` M ${(cx - shortR).toFixed(2)},${(cy + shortR).toFixed(2)} L ${(cx + shortR).toFixed(2)},${(cy - shortR).toFixed(2)}`;
          const el = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          el.setAttribute('d', d);
          el.setAttribute('stroke', fill);
          el.setAttribute('stroke-width', String(SPARKLE_STROKE_WIDTH));
          el.setAttribute('stroke-linecap', 'round');
          el.setAttribute('fill', 'none');
          el.setAttribute('pointer-events', 'none');
          el.setAttribute('opacity', '0');
          sparkleLayer.appendChild(el);
          this._sparkles.push({el, cx, cy, r, life: 0, maxLife: SPARKLE_LIFE_S});
          this._nextSparkleAt += 1 / sparkleRateHz;
        }
        if (this._nextSparkleAt < 0) this._nextSparkleAt = 0;

        // Advance + retire active sparkles. Three-phase opacity
        // curve: fade-in [0, 0.20], hold [0.20, 0.50], fade-out
        // [0.50, 1.0]. No rising motion (unlike boiler bubbles).
        // Review-fix #01 #21: retire slightly before maxLife so a
        // sparkle whose `lifeT` lands exactly on `1.0` doesn't
        // linger one frame at opacity 0 before the next-tick retire.
        const aliveSparkles = [];
        for (const s of this._sparkles) {
          s.life += dt;
          if (s.life >= s.maxLife * 0.999) {
            s.el.remove();
            continue;
          }
          const lifeT = s.life / s.maxLife;
          let opacity;
          if (lifeT < 0.20) opacity = lifeT / 0.20;
          else if (lifeT < 0.50) opacity = 1;
          else opacity = Math.max(0, 1 - (lifeT - 0.50) / 0.50);
          s.el.setAttribute('opacity', opacity.toFixed(3));
          aliveSparkles.push(s);
        }
        this._sparkles = aliveSparkles;
      }

      // === QS-232 lightning subsystem =========================
      // Only spawn when `charging && !degraded`. The advance/retire
      // loop runs unconditionally for graceful exit (in-flight bolts
      // complete their fade-out ~0.25s).
      if (lightningLayer && hasValidBase) {
        if (charging && !degraded) {
          this._nextLightningAt -= dt;
          while (this._nextLightningAt <= 0 && this._lightningBolts.length < MAX_CONCURRENT_LIGHTNING) {
            // Tapered Zeus-thunderbolt path. 4-node skeleton from
            // top (wide) to soup tip (point), with lateral jitter at
            // the two midpoints for the zig-zag. The outline is a
            // closed polygon: left side top→bottom, tip, right side
            // bottom→top.
            const topY = CENTER_CY - CLIP_R + LIGHTNING_TOP_MARGIN_PX;
            const startCx = CENTER_CX + (Math.random() * 2 - 1) * (CLIP_R * 0.4);
            const dy = this._waterBaseY - CENTER_CY;
            const chordHalf = Math.sqrt(Math.max(0, CLIP_R * CLIP_R - dy * dy));
            if (chordHalf < 2) {
              this._nextLightningAt += LIGHTNING_SPAWN_MIN_S +
                Math.random() * (LIGHTNING_SPAWN_MAX_S - LIGHTNING_SPAWN_MIN_S);
              continue;
            }
            const endCx = CENTER_CX + (Math.random() * 2 - 1) * chordHalf * 0.7;
            const totalLen = this._waterBaseY - topY;
            // Two lateral jitters at 33% and 67% down the bolt —
            // smaller than the original mid-kink so the overall
            // shape still reads as "going down" rather than
            // "sideways squiggle".
            const j1 = (Math.random() * 2 - 1) * LIGHTNING_LATERAL_JITTER_PX * 0.5;
            const j2 = (Math.random() * 2 - 1) * LIGHTNING_LATERAL_JITTER_PX * 0.5;
            const x1 = startCx + (endCx - startCx) * 0.33 + j1;
            const x2 = startCx + (endCx - startCx) * 0.67 + j2;
            const y1 = topY + totalLen * 0.33;
            const y2 = topY + totalLen * 0.67;
            // Half-widths along the skeleton. Top is the widest; the
            // path narrows to a point at the tip (water surface).
            const h0 = LIGHTNING_TOP_WIDTH * 0.50;
            const h1 = LIGHTNING_TOP_WIDTH * 0.30;
            const h2 = LIGHTNING_TOP_WIDTH * 0.15;
            const d = `M ${(startCx - h0).toFixed(2)},${topY.toFixed(2)}`
                    + ` L ${(x1 - h1).toFixed(2)},${y1.toFixed(2)}`
                    + ` L ${(x2 - h2).toFixed(2)},${y2.toFixed(2)}`
                    + ` L ${endCx.toFixed(2)},${this._waterBaseY.toFixed(2)}`
                    + ` L ${(x2 + h2).toFixed(2)},${y2.toFixed(2)}`
                    + ` L ${(x1 + h1).toFixed(2)},${y1.toFixed(2)}`
                    + ` L ${(startCx + h0).toFixed(2)},${topY.toFixed(2)}`
                    + ` Z`;
            const el = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            el.setAttribute('d', d);
            el.setAttribute('fill', LIGHTNING_STROKE_COLOR);
            el.setAttribute('stroke', 'none');
            el.setAttribute('pointer-events', 'none');
            el.setAttribute('opacity', '0');
            lightningLayer.appendChild(el);
            this._lightningBolts.push({el, life: 0, maxLife: LIGHTNING_LIFE_S});
            // Review-fix #01 #12: use `+=` (cadence accumulation) for
            // symmetry with the degenerate-chord `continue` branch
            // above AND with the sparkle subsystem — both spawn
            // paths advance the cadence counter identically, and the
            // post-loop clamp `if (… < 0) … = 0` below absorbs any
            // backlog drift.
            this._nextLightningAt += LIGHTNING_SPAWN_MIN_S +
              Math.random() * (LIGHTNING_SPAWN_MAX_S - LIGHTNING_SPAWN_MIN_S);
          }
        }
        // Review-fix #02 #8: hoist the cap-recovery clamp out of the
        // `if (charging && !degraded)` spawn gate so the structure
        // mirrors the sparkle subsystem above. Currently safe because
        // the `-= dt` decrement is also gated, but a future refactor
        // moving the decrement out would create a permanent
        // negative-drift bug.
        if (this._nextLightningAt < 0) this._nextLightningAt = 0;

        // Advance + retire (runs every frame, graceful exit).
        // Three-phase opacity curve: fade-in [0, 0.10], hold
        // [0.10, 0.70], fade-out [0.70, 1.0].
        const aliveBolts = [];
        for (const b of this._lightningBolts) {
          b.life += dt;
          if (b.life >= b.maxLife) {
            b.el.remove();
            continue;
          }
          const lifeT = b.life / b.maxLife;
          let opacity;
          if (lifeT < 0.10) opacity = lifeT / 0.10;
          else if (lifeT < 0.70) opacity = 1;
          else opacity = Math.max(0, 1 - (lifeT - 0.70) / 0.30);
          b.el.setAttribute('opacity', opacity.toFixed(3));
          aliveBolts.push(b);
        }
        this._lightningBolts = aliveBolts;
      }

      this._animRaf = requestAnimationFrame(step);
    };
    this._animRaf = requestAnimationFrame(step);
  }

  _stopAnimation() {
    if (this._animRaf != null) cancelAnimationFrame(this._animRaf);
    this._animRaf = null;
    this._lastAnimTs = null;
  }

  connectedCallback() {
    // QS-232: continuous RAF while connected, mirroring
    // qs-water-boiler-card.js / qs-pool-card.js. The electron-soup
    // wave is intrinsically visible at all times (idle green / charge
    // blue / degraded grey) so RAF is no longer gated on `_charging`.
    this._startAnimation();
    // Review-fix #01 #7: re-prime on reconnect ONLY when the implied
    // colorMix state diverged from the actual `_charging` during the
    // detached interval. Without this, the lazy-init guard in
    // `_startAnimation()` preserves the pre-detach `_currentColorMix`
    // across reconnect, so a state-flip mid-detach causes the soup to
    // visibly lerp on reattach rather than snap. The `?? 0` fallback
    // covers the very first connect (before lazy-init has run), where
    // both _charging and _currentColorMix are still at their initial
    // values — the equation `(false ? 1 : 0) !== Math.round(0)`
    // evaluates `0 !== 0`, so no spurious prime fires.
    //
    // Review-fix #02 #4: arming the flag isn't enough — `_render`
    // consumes it. Without an explicit `_render` call, the flag
    // sits until the next hass push (which can be seconds away)
    // and the RAF lerps visibly during the gap. Call `_render`
    // immediately so the prime takes effect on the next paint.
    const impliedMix = Math.round(this._currentColorMix ?? 0);
    if ((this._charging ? 1 : 0) !== impliedMix) {
      this._needsAnimationPrime = true;
      this._render();
    }
  }

  disconnectedCallback() {
    this._stopAnimation();
    // S7: reset interaction flags so a re-attach after mid-interaction
    // doesn't silently short-circuit `set hass` on stale flags.
    this._isInteracting = false;
    this._isInteractingCharger = false;
    this._isInteractingPerson = false;
    this._isInteractingTarget = false;
    this._modalOpen = false;
    // QS-232: eagerly tear down sparkle and lightning DOM nodes.
    // Without this they would be GC'd along with the shadow root, but
    // explicit cleanup is cheap and avoids dangling SVG nodes during
    // a rapid detach/attach. Optional-chaining shape matches the
    // boiler bubble / steam teardown pattern.
    this._sparkles?.forEach(s => s.el?.remove?.());
    this._sparkles = [];
    this._lightningBolts?.forEach(b => b.el?.remove?.());
    this._lightningBolts = [];
  }
  static getStubConfig() {
    return { name: "QS Car", entities: {} };
  }

  setConfig(config) {
    if (!config || !config.entities) throw new Error("entities is required");
    this._config = config;
    this._root = this.attachShadow({ mode: "open" });
    // Review-fix #01 #4: arm the priming flag so the FIRST `_render()`
    // below — which runs from `setConfig`, BEFORE `connectedCallback`
    // fires — primes `_currentAmplitude` / `_currentSpeed` /
    // `_currentColorMix` against the about-to-be-computed `_charging`
    // state. Without this, the first paint always uses
    // `initialAmp = CALM_AMP` and `initialColorMix = 0` (idle palette)
    // regardless of `_charging`, and either the RAF lerps visibly
    // over ~1.5 s or a quick hass push snaps the values
    // (green-to-blue flash).
    this._needsAnimationPrime = true;
    this._render();
    // Review-fix #02 #3: complete the fix-#01-#22 fix. `_startAnimation`
    // bails early on `!this._root`; if `connectedCallback` fires
    // BEFORE `setConfig` (uncommon but valid lifecycle), the bail
    // would never be retried. Re-invoke here once `_root` is set.
    if (this.isConnected && this._animRaf == null) {
      this._startAnimation();
    }
  }

  // S6: defence-in-depth HTML escaping for user-/3rd-party-controlled
  // strings interpolated into innerHTML.
  _escapeHtml(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._root) return;
    // Avoid re-rendering while user is interacting with selects or a modal is open
    if (this._isInteracting || this._isInteractingCharger || this._isInteractingPerson || this._modalOpen || this._isInteractingTarget) return;
    this._render();
  }

  getCardSize() { return 6; }

  _entity(id) { return id ? this._hass?.states?.[id] : undefined; }

  _call(domain, service, data) {
    return this._hass.callService(domain, service, data);
  }

  _press(entity_id) { return this._call('button', 'press', { entity_id }); }
  _turnOn(entity_id) { return this._call('switch', 'turn_on', { entity_id }); }
  _turnOff(entity_id) { return this._call('switch', 'turn_off', { entity_id }); }
  _select(entity_id, option) { return this._call('select', 'select_option', { entity_id, option }); }
  _setTime(entity_id, value) { return this._call('time', 'set_value', { entity_id, time: value }); }

  _percent(num) {
    const n = Number(num);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(100, n));
  }

  // Format a number for display, replacing NaN/null/undefined with "--"
  _fmt(num, round = true) {
    const n = Number(num);
    if (num == null || Number.isNaN(n)) return '--';
    return round ? Math.round(n) : n;
  }

  _render() {
      const cfg = this._config || {};
      const e = cfg.entities || {};

      const sSoc = this._entity(e.soc);
      const sCurrentInputedEnergy = this._entity(e.current_inputed_energy);
      const sPower = this._entity(e.power);
      const selCharger = this._entity(e.charger_select);
      const selPerson = this._entity(e.attached_person);

      const selLimit = this._entity(e.next_limit);
      const swPriority = this._entity(e.bump_priority);
      const tNext = this._entity(e.next_time);
      const sChargeType = this._entity(e.charge_type);
      const sChargeTime = this._entity(e.charge_time);
      const sRangeNow = this._entity(e.range_now);
      const sRangeTarget = this._entity(e.range_target);
      const sPersonForecast = this._entity(e.person_forecast);
      const sUsePercentMode = this._entity(e.use_percent_mode);
      const sIsOffGrid = this._entity(e.is_off_grid);
      const sCarIsStale = this._entity(e.car_is_stale);

      const title = (cfg.title || cfg.name) || (sSoc ? (sSoc.attributes.friendly_name || sSoc.entity_id) : "Car");

      // Check if system is off-grid
      const isOffGrid = sIsOffGrid?.state === 'on';
      // Check if car API data is stale
      const isStale = sCarIsStale?.state === 'on';
      let soc = this._percent(sSoc?.state);
      const power = sPower?.state || "0";
      this._chargePower = Number(power) || 0;
      const target = selLimit?.state || "";
      const charging = (Number(power) > 50);
      this._charging = charging;
      // QS-232: per-instance unique SVG IDs so two car cards on the
      // same dashboard don't collide (cheap insurance — households
      // can plausibly have ≥ 2 cars on one dashboard). Mirrors the
      // boiler / pool _nextClipId pattern.
      if (!this._electronClipId) {
        QsCarCard._nextClipId = (QsCarCard._nextClipId || 0) + 1;
        const uid = QsCarCard._nextClipId;
        this._electronClipId    = `car_eClip_${uid}`;
        this._sparkleLayerId    = `car_sparkLayer_${uid}`;
        this._lightningLayerId  = `car_lightningLayer_${uid}`;
        this._grainFilterId     = `car_grainFilter_${uid}`;
        this._lightningFilterId = `car_lightningFilter_${uid}`;
      }
      const electronClipId    = this._electronClipId;
      const sparkleLayerId    = this._sparkleLayerId;
      const lightningLayerId  = this._lightningLayerId;
      const grainFilterId     = this._grainFilterId;
      const lightningFilterId = this._lightningFilterId;
      const carChargeTypeIcons = {
          "Unknown": "mdi:help-circle-outline",
          "Not Plugged": "mdi:power-plug-off",
          "Faulted": "mdi:emoticon-dead",
          "No Power To Car": "mdi:flash-off",
          "Not Charging": "mdi:battery-off",
          "Target Met": "mdi:battery-high",
          "As Fast As Possible": "mdi:rabbit",
          "Scheduled": "mdi:clock-outline",
          "Solar Priority": "mdi:solar-power",
          "Solar": "mdi:white-balance-sunny",
          "Person Automated": "mdi:auto-fix",
      };
      const iconForChargeType = (str) => carChargeTypeIcons[str];
      const chargeIcon = iconForChargeType(sChargeType?.state);
      const chargeTime = sChargeTime?.state || '';
      const chargeIconLabel = 'Mode';
      const chargeTimeLabel = 'Finish';

      const isNumberLike = (v) => v != null && v !== '' && !Number.isNaN(Number(v));
      const normState = (s) => String(s || '').toLowerCase();
      const validState = (s) => s != null && !INVALID_STATES.includes(normState(s));
      const rangeNowStr = (sRangeNow && isNumberLike(sRangeNow.state)) ? `${this._fmt(sRangeNow.state)} km` : '';
      const rangeTargetStr = (sRangeTarget && isNumberLike(sRangeTarget.state)) ? `${this._fmt(sRangeTarget.state)} km` : '';

      const parseTargetPercent = (txt) => {
          if (!txt) return undefined;
          const m = String(txt).match(/(\d+)\s*%?/);
          return m ? Number(m[1]) : undefined;
      };

      const parseTargetEnergy = (txt) => {
          if (!txt) return undefined;
          const m = String(txt).match(/([\d.]+)\s*kWh/i);
          return m ? Number(m[1]) : undefined;
      };

      // Check if we're in energy mode (energy mode = NOT use_percent_mode)
      // use_percent_mode entity returns 'on' when percent constraints can be used
      const usePercentModeState = sUsePercentMode?.state;
      const useEnergyMode = usePercentModeState !== 'on';

      // Font size for energy unit (kWh) relative to the number
      const energyUnitFontSize = 0.4; // 40% of the number size

      // Get target value based on mode
      const isStalePercentMode = isStale && !useEnergyMode;
      let targetPct, displayTargetValue, maxCircleValue, displaySocValue;
      if (isStalePercentMode) {
          // Stale-percent mode: show +XX% based on energy delivered
          const energyRaw = sCurrentInputedEnergy?.state;
          const energyAvailable = isNumberLike(energyRaw);
          const energyWh = energyAvailable ? Number(energyRaw) : 0;
          const rawCap = Number(e.car_battery_capacity_kwh);
          const batteryWh = (rawCap > 0 ? rawCap : 100) * 1000;
          const pctAdded = Math.max(0, (energyWh / batteryWh) * 100);
          // When charging with unavailable energy, ensure a minimum arc so animation is visible
          const minStaleChargingSoc = (charging && isStalePercentMode) ? 3 : 0;
          soc = Math.max(minStaleChargingSoc, Math.min(100, pctAdded));
          targetPct = parseTargetPercent(target);
          maxCircleValue = 100;
          // If energy data is unavailable, show contextual indicator instead of +0%
          if (!energyAvailable && charging) {
              // ⚡ (U+26A1) is widely supported; replace with 'charging' text if emoji issues arise
              displaySocValue = '\u26A1';
          } else if (!energyAvailable && !charging) {
              displaySocValue = '--';
          } else {
              displaySocValue = `+${this._fmt(pctAdded)}%`;
          }
          displayTargetValue = `${this._fmt(targetPct ?? 0)}%`;
      } else if (useEnergyMode) {
          const targetEnergy = parseTargetEnergy(target);
          // Use car battery capacity from config as max circle value (in kWh)
          // Fall back to parsing from last option if not provided
          const configBatteryCapacity = e.car_battery_capacity_kwh;
          if (configBatteryCapacity != null && configBatteryCapacity > 0) {
              maxCircleValue = Number(configBatteryCapacity);
          } else {
              // Fallback: get max energy from last option
              const limitOptions = selLimit?.attributes?.options || [];
              const lastOption = limitOptions.length > 0 ? limitOptions[limitOptions.length - 1] : "100kWh";
              maxCircleValue = parseTargetEnergy(lastOption) || 100;
          }

          // Use current_inputed_energy for energy mode (value is in Wh, convert to kWh)
          const energyValue = Number(sCurrentInputedEnergy?.state || 0);
          const socKwhNum = Math.round(energyValue / 1000);
          const socKwh = this._fmt(socKwhNum);

          // Calculate percentage for circle (0 to maxCircleValue)
          const socPct = maxCircleValue > 0 ? (socKwhNum / maxCircleValue) * 100 : 0;
          targetPct = targetEnergy != null && maxCircleValue > 0 ? (targetEnergy / maxCircleValue) * 100 : socPct;

          displaySocValue = `${socKwh}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>`;

          let toBeDisplayedvalue = this._fmt(targetEnergy ?? socKwhNum);
          displayTargetValue = `${toBeDisplayedvalue}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>`;

          // Override soc for display
          soc = Math.max(0, Math.min(100, socPct));
      } else {
          targetPct = parseTargetPercent(target);
          maxCircleValue = 100;
          displayTargetValue = `${this._fmt(targetPct ?? soc)}%`;
          displaySocValue = `${this._fmt(soc)}%`;
      }

      const formatHm = (mins) => {
          if (mins == null) return '';
          const h = String(Math.floor(mins / 60)).padStart(2, '0');
          const m = String(mins % 60).padStart(2, '0');
          return `${h}:${m}`;
      };

      const parseTimeToMinutes = (txt) => {
          if (!txt) return 420; // 07:00
          const parts = String(txt).split(':').map(Number);
          const h = parts[0] || 0, m = parts[1] || 0;
          return h * 60 + m;
      };
      const nextTimeStr = tNext?.state || '07:00:00';
      const nextTimeMins = this._localNextTimeMins != null ? this._localNextTimeMins : parseTimeToMinutes(nextTimeStr);

      const css = `
      :host { --pad: 18px; --ring-text-shadow: 0 0 12px rgba(0,0,0,0.8), 0 2px 4px rgba(0,0,0,0.5); display:block; }
      .card { padding: var(--pad); }
      .card.stale { border: 3px solid var(--error-color, #db4437); }
      .card.off-grid { background: rgba(244, 67, 54, 0.08); }
      .card-title { text-align:center; font-weight:800; font-size: 1.6rem; margin: 0px 0 0px; }
      .top { display:flex; gap:12px; flex-wrap:wrap; }
      .below { display:flex; align-items:center; justify-content:center; margin-top: 0px; width:260px; margin-left:auto; margin-right:auto; }
      .below .pill { width:100%; }
      .forecast-row { text-align:center; width:260px; margin: 4px auto 0; color: var(--secondary-text-color); font-weight:600; font-size: .85rem; }
      .below-line { width:260px; margin: 8px auto 0; display:grid; grid-template-columns: 1fr auto; align-items:center; column-gap:12px; }
      .below-line.full { display:block; }
      .below-line.full > button { width: 100%; justify-content: center; position: relative; }
      .below-line.full > button.align-left { justify-content: flex-start; }
      .below-line.full > button .btn-center { position: absolute; left: 50%; transform: translateX(-50%); }
      .below-line .time-row { justify-self: end; margin-top: 0; }
      .btn-clock { display:flex; align-items:center; gap:8px; }
      .pill { display:flex; align-items:center; gap:8px; border-radius: 28px; height:40px; min-height:40px; padding:0 12px; border:1px solid var(--divider-color);
              background: var(--ha-card-background, var(--card-background-color)); box-sizing: border-box; cursor: pointer; touch-action: manipulation; }
      .pill .dot { width:12px; height:12px; border-radius:50%; background: var(--divider-color); box-shadow: 0 0 8px rgba(0,0,0,.25) inset; }
      .pill.on { background: rgba(56,142,60,0.15); border-color: rgba(56,142,60,.35); }
      .pill.on .dot { background: #2ecc71; box-shadow: 0 0 12px #2ecc71aa; }
      .pill { position: relative; }
      .pill select { appearance:none; background: transparent; color: var(--primary-text-color); border: none; font-weight:700; position: absolute; left:0; top:0; width:100%; height:100%; text-align:center; text-align-last:center; padding: 0 12px 0 40px; border-radius: 28px; cursor: pointer; z-index:1; box-sizing: border-box; }

      .hero { margin-top: 0px; display:flex; align-items:center; justify-content:center; }
      .hero .side { text-align:center; color: var(--secondary-text-color); font-weight:600; }
      .hero .side .value { display:block; font-size:1.2rem; color: var(--primary-text-color); }
      .ring { position: relative; width:300px; height:300px; margin: 0 auto; }
      .ring .center { position:absolute; inset:0; display:grid; place-items:center; text-align:center; pointer-events: none; transform: translateY(16px); }
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; text-shadow: var(--ring-text-shadow); }
      .ring ha-icon { --mdc-icon-size: 32px; color: var(--secondary-text-color); margin-bottom: 6px; }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; text-shadow: var(--ring-text-shadow); }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; text-shadow: var(--ring-text-shadow); }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; text-align:center; width: 180px; margin: 0 auto; }
      .ring .soc-block { display:flex; flex-direction:column; align-items:center; gap:2px; margin-top:4px; margin-bottom:8px; }
      .ring .soc-block .charge-type-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); margin-bottom: 2px; }
      .ring .stack > * { text-align:center; }
      .ring .mini-grid { display:grid; grid-template-columns: repeat(3, 60px); grid-auto-rows: auto; width:180px; margin: 0 auto; justify-items:center; align-items:center; row-gap:4px; column-gap:0; }
      .ring .mini-grid.extra { row-gap:0; margin-top:2px; margin-bottom:6px; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:0; }
      .ring .target-cell { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .mini-title { color: var(--secondary-text-color); font-weight:700; font-size: .7rem; letter-spacing:.2px; white-space: nowrap; text-shadow: var(--ring-text-shadow); }
      .ring .mini-value { color: var(--primary-text-color); font-weight:800; font-size: .95rem; line-height: 1.1; white-space: pre-line; text-shadow: var(--ring-text-shadow); }
      .ring .mini-icon { --mdc-icon-size: 18px; color: var(--primary-text-color); }
      .ring .mini-range { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; text-shadow: var(--ring-text-shadow); }
      .ring .mini-range-now { color: var(--primary-text-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; transform: translateY(-8px); text-shadow: var(--ring-text-shadow); }
      .ring .mini-range-target { color: var(--primary-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; margin-bottom:0; text-shadow: var(--ring-text-shadow); }
      .disabled .ring .mini-range-target { color: var(--secondary-text-color); }
      .ring .mini-range:empty, .ring .mini-range-now:empty, .ring .mini-range-target:empty { display:none; }
      .ring .center-controls { display:flex; align-items:center; justify-content:center; margin-top: 6px; }
      /* Mobile touch fix: touch-action:none on the SVG (not the inner <circle>) prevents the
         browser from initiating scroll/pan gestures when dragging the ring handle. SVG child
         elements like <circle> don't reliably honor touch-action on iOS Safari / HA Companion. */
      .ring svg { touch-action: none; }
      /* Mobile touch fix: touch-action:manipulation removes the 300ms tap delay that mobile
         browsers impose for double-tap detection, making button taps register immediately.
         Without this, a hass re-render can destroy the DOM node before the synthetic click fires. */
      /* QS-232 review-fix #03: sun-btn now uses the same absolute
         positioning as override-btn on boiler/radiator/climate cards.
         Bottom: 15px from .ring bottom, centered horizontally. */
      .ring .sun-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:flex; align-items:center; justify-content:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      /* QS-232 review-fix #04: icon bumped 26 → 30 px so the sun
         renders at the same visual size as the hand on the
         override-btn in climate/radiator/boiler.
         Review-fix #07: kept 'transform: translateY(3px)' on the icon
         (same as the rabbit-btn icon) — the mdi:weather-sunny glyph
         has rays extending upward more than downward, so its bounding-
         box center sits above its visual center. The 3 px nudge
         pushes the icon down so it reads as centered in the 50×50
         button shell. */
      .ring .sun-btn ha-icon { --mdc-icon-size: 30px; color: var(--secondary-text-color); display:flex; align-items:center; justify-content:center; line-height:1; transform: translateY(3px); }
      .ring .sun-btn.on { border-color: rgba(255,202,40,.45); background: rgba(255,202,40,.14); box-shadow: 0 0 0 3px rgba(255,202,40,.20), 0 0 16px #FFCA28; }
      .ring .sun-btn.on ha-icon { color: #FFCA28; }
      /* QS-232 review-fix #04: invisible spacer that preserves the
         .stack's vertical balance after the sun-btn was moved out of
         .center-controls to absolute positioning. Without this, the
         shorter stack gets grid-centered higher in .center, pushing
         the soc-block (89%, range) and the mini-grid buttons (rabbit,
         time) visibly downward.
         Review-fix #02 #17 — components of the 74 px height:
           - 6 px margin-top on the original .center-controls block
           - 14 px label ("Solar priority", font-size 0.7rem,
             font-weight 700)
           - 4 px gap between label and button (column-gap: 4)
           - 50 px sun-btn button height
         If any of those four numbers changes (icon-size, label
         typography, button dimensions), this height MUST be updated
         in lockstep, otherwise the layout drifts. */
      .ring .sun-btn-spacer { width: 50px; height: 74px; pointer-events: none; }
      .ring .rabbit-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .rabbit-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; transform: translateY(3px); }
      .ring .rabbit-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .rabbit-btn.on ha-icon { color: #2196F3; }
      .ring .time-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; font-size: 0.99rem; font-weight: 800; color: var(--primary-color); line-height: 1; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .time-btn:hover { border-color: var(--primary-color); background: rgba(255,255,255,.08); }
      .ring .time-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .time-btn.on { color: #2196F3; }

      .grid { display:grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap:16px; margin-top: 16px; }
      @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
      .section { border-radius: 20px; background: var(--card-background-color); padding:16px; border: 1px solid var(--divider-color); }
      .row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:8px 0; }
      select, input[type="time"] {
        background: var(--ha-card-background, var(--card-background-color));
        color: var(--primary-text-color);
        border:1px solid var(--divider-color);
        border-radius:14px;
        padding:10px 12px;
        font-weight:700;
        font-size: .95rem;
        font-family: inherit;
        -webkit-appearance: none;
        appearance: none;
        outline: none;
        line-height: 1.2;
      }
      /* Charger select full width */
      .below select { width: 100%; height: 40px; min-height: 40px; }
      /* Hour/minute compact width */
      .below-line .time-row select { width: auto; min-width: 64px; height: 40px; min-height: 40px; text-align: center; text-align-last: center; }
      /* Focus state for consistency */
      select:focus { border-color: var(--primary-color); box-shadow: 0 0 0 2px rgba(0,0,0,0), 0 0 0 3px color-mix(in srgb, var(--primary-color) 30%, transparent); }
      .actions { display:grid; grid-template-columns: repeat(2,minmax(180px,1fr)); gap:12px; margin-top:8px; }
      button { border:none; border-radius:18px; padding:14px 16px; font-weight:700; cursor:pointer; font-size: .95rem; }
      button.pill { height: 40px; min-height: 40px; display:flex; align-items:center; }
      .primary { background: var(--primary-color); color: var(--text-primary-color, #fff); }
      .secondary { background: rgba(255,255,255,.06); color: var(--primary-text-color); border:1px solid var(--divider-color); }
      .danger { background: var(--error-color); color: #fff; }
      /* Outline style variant */
      button.outline { background: transparent !important; border-width: 2px; }
      .danger.outline { color: var(--error-color) !important; border-color: var(--error-color) !important; }
      /* Disconnected visual mode (greyscale) */
      .disabled .ring .target-value { color: var(--secondary-text-color); }
      .disabled .primary { background: var(--divider-color); color: var(--primary-text-color); }
      .disabled .secondary { background: rgba(255,255,255,.04); border-color: var(--divider-color); color: var(--secondary-text-color); }
      /* Keep reset/danger buttons red even in disabled mode */
      .disabled .danger { background: var(--error-color); color: #fff; }
      .disabled .danger.outline { background: transparent !important; color: var(--error-color) !important; border-color: var(--error-color) !important; }
      .disabled .chip, .disabled .pill { border-color: var(--divider-color); }
      .disabled .sun-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .sun-btn ha-icon { color: var(--secondary-text-color); }
      .disabled .sun-btn.on { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .sun-btn.on ha-icon { color: var(--secondary-text-color); }
      .disabled .rabbit-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .rabbit-btn ha-icon { color: var(--secondary-text-color); }
      .disabled .rabbit-btn.on { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .rabbit-btn.on ha-icon { color: var(--secondary-text-color); }
      .disabled .time-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; cursor: not-allowed; color: var(--secondary-text-color); }
      .disabled .progress > div { background: var(--divider-color); }
      .disabled #force, .disabled #schedule_inline { pointer-events: none; cursor: not-allowed; }
      .live { display:grid; gap:10px; }
      .fault .card-title { color: var(--error-color); }
      .fault .time-btn { color: var(--secondary-text-color); }
      .progress { height:10px; border-radius:999px; background: var(--divider-color); overflow:hidden; }
      .progress > div { height:100%; background: var(--accent-color); width:${soc}% }
      .menu { text-align:right; }
      .quick { display:flex; gap:10px; flex-wrap:wrap; margin-top: 10px; }
      .chip { padding:10px 14px; border-radius: 999px; background: rgba(255,255,255,.06); border:1px solid var(--divider-color); font-weight:700; }
      .time-row { display:grid; grid-template-columns: auto auto auto; align-items:center; gap: 12px; margin-top: 6px; justify-content:center; }
      input[type="range"]{ width:100%; height:6px; border-radius:999px; background: var(--divider-color); outline:none; -webkit-appearance:none; appearance:none; }
      input[type="range"]::-webkit-slider-thumb{ -webkit-appearance:none; appearance:none; width:20px; height:20px; border-radius:50%; background: var(--accent-color); box-shadow: 0 0 0 4px rgba(0,0,0,.15); cursor:pointer; }

      /* Themed confirm dialog */
      #target_handle { touch-action: none; }
      .modal { position:absolute; inset:0; background: rgba(0,0,0,.45); display:flex; align-items:center; justify-content:center; z-index: 50; touch-action: manipulation; }
      .dialog { background: var(--card-background-color); color: var(--primary-text-color); border:1px solid var(--divider-color); border-radius: 16px; padding: 16px; width: min(92%, 360px); box-shadow: 0 10px 30px rgba(0,0,0,.3); }
      .dialog h3 { margin: 0 0 8px; font-size: 1.1rem; font-weight:800; text-align:left; }
      .dialog p { margin: 0 0 14px; line-height:1.4; color: var(--secondary-text-color); white-space: pre-line; }
      .dialog .time-picker { display:flex; align-items:center; justify-content:center; gap:12px; margin: 16px 0; }
      .dialog .time-picker select { width: auto; min-width: 64px; height: 40px; text-align: center; text-align-last: center; }
      .dialog .time-picker span { font-weight:700; font-size: 1.2rem; }
      .dialog .actions { display:flex; gap:10px; justify-content:flex-end; margin-top: 6px; }
      .btn { border:none; border-radius:12px; padding:10px 14px; font-weight:700; cursor:pointer; touch-action: manipulation; min-height: 44px; -webkit-tap-highlight-color: transparent; }
      .btn.secondary { background: rgba(255,255,255,.06); color: var(--primary-text-color); border:1px solid var(--divider-color); }
      .btn.primary { background: var(--primary-color); color:#fff; }
      .btn.danger { background: var(--error-color); color:#fff; }
    `;

      const ringCirc = 130; // bigger radius
      const gapDeg = 60;
      const rangeDeg = 360 - gapDeg;
      // Place the gap centered at bottom. Start at bottom-left, end at bottom-right.
      // use degree as 0 bottom, positive clockwise , contrary to radian beware in svg : y are downward
      // degree part of [0,360[
      const startDeg = gapDeg / 2;   // bottom-left
      const endDeg = startDeg + rangeDeg; // bottom-right (wraps past 360)

      const deg2rad = (d) => ((270 - d) * Math.PI) / 180;
      const rad2deg = (r) => {
          if (r < 0) r += 2 * Math.PI;
          return (((270 - ((r * 180) / Math.PI)) + 360) % 360);
      };
      const polar = (cx, cy, r, deg) => ({x: cx + r * Math.cos(deg2rad(deg)), y: cy - r * Math.sin(deg2rad(deg))});
      const arcPath = (cx, cy, r, a0, a1) => {
          // BH defense-in-depth: a non-finite angle (NaN / Infinity)
          // would render as `A 130 130 0 0 1 NaN NaN` in the SVG `d`
          // attribute and trigger a browser "Configuration error".
          // Return empty string so the consumer omits the <path> tag.
          if (!Number.isFinite(a0) || !Number.isFinite(a1)) return '';
          const p0 = polar(cx, cy, r, a0);
          const p1 = polar(cx, cy, r, a1);
          let delta = a1 - a0;
          if (delta < 0) delta += 360;
          const laf = delta > 180 ? 1 : 0;
          return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${laf} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
      };

      const pctToDeg = (p) => startDeg + (Math.max(0, Math.min(100, p)) / 100) * rangeDeg;

      const socEndDeg = pctToDeg(soc);
      const handlePct =
          this._targetDragPct != null ? this._targetDragPct :
              (this._localTargetPct != null ? this._localTargetPct : (targetPct ?? soc));
      const handleDeg = pctToDeg(handlePct);
      // Enlarge gauge
      const center = {cx: 160, cy: 160};
      const arcLen = 2 * Math.PI * ringCirc * (rangeDeg / 360);
      const segLen = arcLen * (Math.max(0, Math.min(100, soc)) / 100);
      let dashLen = Math.round(segLen * 0.22);
      let gapLen = Math.round(segLen * 0.28);
      dashLen = Math.max(6, dashLen);
      gapLen = Math.max(6, gapLen);
      const patternLen = dashLen + gapLen;
      // Ensure pattern repeats within the segment for visible motion
      if (patternLen >= segLen - 4) {
          const scale = (segLen - 4) / patternLen;
          dashLen = Math.max(4, Math.round(dashLen * scale));
          gapLen = Math.max(4, Math.round(gapLen * scale));
      }
      // Persist animation pattern for smoothness across re-renders
      this._animPatternLen = Math.max(8, dashLen + gapLen);
      // showAnimation will be defined after connection state is resolved
      const handlePos = polar(center.cx, center.cy, ringCirc, handleDeg);
      const bgPath = arcPath(center.cx, center.cy, ringCirc, startDeg, endDeg);
      const socPath = arcPath(center.cx, center.cy, ringCirc, startDeg, socEndDeg);
      const gradGreenId = `gradG-${Math.floor(Math.random() * 1e6)}`;
      const gradChargeId = `gradC-${Math.floor(Math.random() * 1e6)}`;
      const gradDisabledId = `gradD-${Math.floor(Math.random() * 1e6)}`;
      const gradFaultId = `gradF-${Math.floor(Math.random() * 1e6)}`;
      const gradStaleId = `gradS-${Math.floor(Math.random() * 1e6)}`;
      const isDisconnected = (sChargeType?.state === 'Not Plugged' || sChargeType?.state === 'Unknown');
      const chargeTypeState = (sChargeType?.state || '').toLowerCase();
      const isFaulted = chargeTypeState === 'faulted' || chargeTypeState === 'unknown' || chargeTypeState === 'no power to car';

      const chargerOptions = selCharger?.attributes?.options || [];
      const chargerState = (selCharger?.state || '').trim();
      const stateLc = chargerState.toLowerCase();
      const invalidStates = ['unavailable', 'unknown', 'none', 'not plugged', 'not_plugged', 'not connected', 'not_connected'];
      const shouldShowPlaceholder = isDisconnected || !chargerState || invalidStates.includes(stateLc) || !chargerOptions.includes(chargerState);
      const chargerOptionsHtml = shouldShowPlaceholder
          ? [`<option value="" selected>No connected Charger</option>`, ...chargerOptions.map(o => `<option>${o}</option>`)].join('')
          : chargerOptions.map(o => `<option ${o === chargerState ? 'selected' : ''}>${o}</option>`).join('');

      // Person selector options
      const personOptions = selPerson?.attributes?.options || [];
      const personState = (selPerson?.state || '').trim();
      const personStateLc = personState.toLowerCase();
      const personInvalidStates = INVALID_STATES;
      const shouldShowPersonPlaceholder = !personState || personInvalidStates.includes(personStateLc) || !personOptions.includes(personState);
      const personOptionsHtml = shouldShowPersonPlaceholder
          ? [`<option value="" selected>No person attached</option>`, ...personOptions.map(o => `<option>${o}</option>`)].join('')
          : personOptions.map(o => `<option ${o === personState ? 'selected' : ''}>${o}</option>`).join('');

      // Person forecast string
      const personForecastStr = sPersonForecast?.state || '';
      const validPersonForecast = personForecastStr && personForecastStr.toLowerCase() !== 'none' && personForecastStr.toLowerCase() !== 'unknown' && personForecastStr.toLowerCase() !== 'unavailable' && personForecastStr.trim() !== '';
      const forecastDisplay = validPersonForecast ? personForecastStr : 'None';

      const activeGradId = isFaulted ? gradFaultId : (isStale ? gradStaleId : (isDisconnected ? gradDisabledId : (charging ? gradChargeId : gradGreenId)));
      // In stale-percent mode, show animation whenever charging regardless of arc size
      const showAnimation = (charging && !shouldShowPlaceholder && (segLen > 6 || isStalePercentMode));

      //const forecastedPersonStr = sForecastedPerson?.state;
      //const showForecastedPerson = forecastedPersonStr && forecastedPersonStr.toLowerCase() !== 'none' && forecastedPersonStr.toLowerCase() !== 'unknown' && forecastedPersonStr.toLowerCase() !== 'unavailable' && forecastedPersonStr.trim() !== '';
      //const displayTitle = showForecastedPerson ? `${title} (${forecastedPersonStr})` : title;
      const displayTitle = title;

      // QS-232 D8: degraded state combines disconnected / faulted /
      // stale. `isOffGrid` is excluded — off-grid keeps the
      // card-level pinkish tint already applied via the `.off-grid`
      // CSS class; the soup is unaffected.
      const degraded = isDisconnected || isFaulted || isStale;
      // Stash on the instance so the RAF step closure can read it
      // without re-deriving (and so degraded-state lightning
      // suppression has a consistent value across the frame).
      this._degraded = degraded;

      // QS-232 D15: water surface y from SOC fill ratio. Matches the
      // boiler envelope (`0.2 + ratio × 0.6`) so the visual character
      // is consistent across the two water-style cards.
      // Review-fix #01 #3: guard `soc` against NaN (a non-numeric
      // `current_inputed_energy` state in `useEnergyMode`, or a
      // corrupted percent reading) so the resulting `_waterBaseY`
      // never propagates `NaN` into the SVG `d` attribute. Mirrors
      // the boiler's `Number.isFinite(displayTargetHours)` guard.
      const safeSoc = Number.isFinite(soc) ? soc : 0;
      const progressRatio = Math.max(0, Math.min(1, safeSoc / 100));
      this._waterBaseY = CENTER_CY + CLIP_R - (0.2 + progressRatio * 0.6) * 2 * CLIP_R;

      // QS-232: prime the wave animation state to the actual charging
      // targets on the first render after connect, avoiding the
      // ~1.5s boot transient. Without this, a card mounted while the
      // car is already charging would visibly lerp up from CALM.
      if (this._needsAnimationPrime) {
        this._currentAmplitude = charging ? CHARGE_AMP : CALM_AMP;
        this._currentSpeed     = charging ? CHARGE_SPEED : CALM_SPEED;
        this._currentColorMix  = charging ? 1 : 0;
        this._needsAnimationPrime = false;
      }

      // QS-232: pre-generate the initial wave path so the SVG renders
      // with soup immediately, avoiding the empty-`d=""` flash between
      // the innerHTML rewrite and the first RAF tick. Both idle/charge
      // siblings share the same `d`; only fill + opacity differ.
      // Review-fix #01 #5: belt-and-braces guard at the call site —
      // even with the upstream `safeSoc` guard, a regression that
      // bypasses it would still poison the emitted `d` attribute.
      // Wrapping the `_generateWavePath(...)` call with a
      // `Number.isFinite(this._waterBaseY)` ternary makes the empty-
      // string fallback explicit at the call site.
      // Review-fix #02 #2: `??` only traps null/undefined; a runtime
      // NaN in either `_currentAmplitude` or `_currentColorMix` would
      // propagate into the initial wave path's `d` attribute and the
      // opacity attribute strings. Mirror the fix-#01-#8 finite-guard
      // shape so NaN falls back to the calm/idle defaults.
      const initialAmp = Number.isFinite(this._currentAmplitude)
        ? this._currentAmplitude : CALM_AMP;
      const initialColorMix = Number.isFinite(this._currentColorMix)
        ? this._currentColorMix : 0;
      const initialWavePath = Number.isFinite(this._waterBaseY)
        ? this._generateWavePath(WAVE_WIDTH, initialAmp, 2, 0, this._waterBaseY)
        : '';
      // Review-fix #02 #6: clamp `initialColorMix` to [0, 1] before
      // `.toFixed(3)` so floating-point lerp drift outside the
      // envelope (e.g., `1.0000000001`) doesn't surface as an ugly
      // `"-0.000"` opacity token at the initial-paint sites
      // (review-fix #01 #18 clamped only the RAF-step emission).
      const initialMix = Math.max(0, Math.min(1, initialColorMix));
      const initialIdleOpacity   = (1 - initialMix).toFixed(3);
      const initialChargeOpacity = initialMix.toFixed(3);

      // QS-232 review-fix #03: snapshot the in-flight sparkles AND
      // lightning bolts BEFORE the innerHTML rewrite below. The
      // rewrite detaches every DOM node under `this._root`, including
      // the sparkle and lightning layers' children. Without this
      // snapshot, every HA state push (which fires `set hass →
      // _render`) wipes the entire particle arrays — the user-
      // visible "no sparkles at all" symptom on a fast-updating car
      // (the original story dropped snapshot/restore on the theory
      // that 0.45 s lifetime made the wipe imperceptible, but in
      // practice with HA pushing every 100-500 ms the sparkles never
      // accumulate). Mirrors the QS-214 boiler snapshot/restore.
      // Review-fix #02 #18: defensive `?? []` so a future caller
      // that dereferences `.length` on the snapshot result can't
      // crash when `_render` runs from `setConfig` BEFORE
      // `_startAnimation` has lazy-init'd the particle arrays.
      const preservedSparkles      = this._sparkles ?? [];
      const preservedNextSparkleAt = this._nextSparkleAt;
      const preservedLightningBolts  = this._lightningBolts ?? [];
      const preservedNextLightningAt = this._nextLightningAt;

      this._root.innerHTML = `
      <ha-card class="card ${isDisconnected ? 'disabled' : ''} ${isFaulted ? 'fault' : ''} ${isOffGrid ? 'off-grid' : ''} ${isStale ? 'stale' : ''}">
        <style>${css}</style>
        <div class="card-title">${this._escapeHtml(displayTitle)}</div>
        <div class="top"></div>

        <div class="hero">
          <div class="ring" title="${soc}%">
            <svg viewBox="0 0 320 320" width="300" height="300" style="touch-action: none;" aria-hidden="true">
              <defs>
                <linearGradient id="${gradGreenId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#00bcd4"/>
                  <stop offset="100%" stop-color="#8bc34a"/>
                </linearGradient>
                <linearGradient id="${gradChargeId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#00e1ff"/>
                  <stop offset="100%" stop-color="#0066ff"/>
                </linearGradient>
                <linearGradient id="${gradDisabledId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#6b6b6b" stop-opacity="0.65"/>
                  <stop offset="100%" stop-color="#a0a0a0" stop-opacity="0.75"/>
                </linearGradient>
                <linearGradient id="${gradFaultId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#ff8a80"/>
                  <stop offset="100%" stop-color="#ff1744"/>
                </linearGradient>
                <linearGradient id="${gradStaleId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#ffa726"/>
                  <stop offset="100%" stop-color="#ff8f00"/>
                </linearGradient>
                <filter id="chargeGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="${grainFilterId}" x="-20%" y="-20%" width="140%" height="140%">
                  <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="3" result="noise" />
                  <feComposite in="noise" in2="SourceGraphic" operator="in" result="grain" />
                  <feMerge>
                    <feMergeNode in="SourceGraphic" />
                    <feMergeNode in="grain" />
                  </feMerge>
                </filter>
                <filter id="${lightningFilterId}" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="${LIGHTNING_GLOW_STDDEV}" />
                </filter>
                <clipPath id="${electronClipId}">
                  <circle cx="${CENTER_CX}" cy="${CENTER_CY}" r="${CLIP_R}" />
                </clipPath>
              </defs>
              <g clip-path="url(#${electronClipId})" style="${degraded ? 'filter: saturate(0.3) brightness(0.7);' : ''}">
                <path id="electron_wave_idle" d="${initialWavePath}" fill="${IDLE_SOUP_COLOR}" opacity="${initialIdleOpacity}" filter="url(#${grainFilterId})" pointer-events="none" style="will-change: transform;" />
                <path id="electron_wave_charge" d="${initialWavePath}" fill="${CHARGE_SOUP_COLOR}" opacity="${initialChargeOpacity}" filter="url(#${grainFilterId})" pointer-events="none" style="will-change: transform;" />
                <g id="${sparkleLayerId}" pointer-events="none"></g>
                <g id="${lightningLayerId}" filter="url(#${lightningFilterId})" pointer-events="none" style="mix-blend-mode: screen; will-change: opacity;"></g>
              </g>
              ${swPriority ? `<circle id="sun_btn_cover" cx="${SUN_BTN_CARVE_CX}" cy="${SUN_BTN_CARVE_CY}" r="${SUN_BTN_CARVE_R}" fill="var(--card-background-color)" pointer-events="none" />` : ''}
              ${e.force_now ? `<circle id="rabbit_btn_cover" cx="${RABBIT_BTN_CARVE_CX}" cy="${RABBIT_BTN_CARVE_CY}" r="${RABBIT_BTN_CARVE_R}" fill="var(--card-background-color)" pointer-events="none" />` : ''}
              ${(tNext && e.schedule) ? `<circle id="time_btn_cover" cx="${TIME_BTN_CARVE_CX}" cy="${TIME_BTN_CARVE_CY}" r="${TIME_BTN_CARVE_R}" fill="var(--card-background-color)" pointer-events="none" />` : ''}
              <path d="${bgPath}" stroke="${isFaulted ? 'rgba(244,67,54,0.35)' : 'var(--divider-color)'}" stroke-width="14" fill="none" stroke-linecap="round" />
              <path d="${socPath}" stroke="url(#${activeGradId})" stroke-width="14" fill="none" stroke-linecap="round" ${showAnimation ? 'stroke-opacity="0.35"' : ''} />
              ${showAnimation ? `
              <path id="charge_anim"
                    d="${socPath}"
                    stroke="url(#${isFaulted ? gradFaultId : gradChargeId})"
                    stroke-width="16"
                    fill="none"
                    stroke-linecap="round"
                    stroke-dasharray="${dashLen} ${gapLen}"
                    stroke-opacity="1"
                    filter="url(#chargeGlow)"
                    style="mix-blend-mode:screen; will-change: stroke-dashoffset"
              />
              ` : ''}
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="${isDisconnected ? 'var(--divider-color)' : 'var(--primary-color)'}" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="${isDisconnected ? 'var(--secondary-text-color)' : 'var(--primary-color)'}" font-size="${useEnergyMode ? '11' : '13'}" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${useEnergyMode ? this._fmt(parseTargetEnergy(target) ?? (Number(sCurrentInputedEnergy?.state || 0) / 1000)) : this._fmt(targetPct ?? soc)}</text>
            </svg>
            <div class="center">
              <div class="stack">
                <div class="soc-block">
                  ${chargeIcon ? `<ha-icon class="charge-type-icon" icon="${chargeIcon}"></ha-icon>` : ''}
                  <div class="pct" style="margin-bottom:0;">${displaySocValue}</div>
                  ${useEnergyMode ? '' : `<div class="mini-range-now" aria-label="current range">${rangeNowStr}</div>`}
                </div>
                <div class="target-block">
                <div class="mini-grid">
                  <div class="mini-title">Force Now</div>
                  <div class="mini-title">${useEnergyMode ? 'Target Energy' : 'Target SOC'}</div>
                  <div class="mini-title">${chargeTimeLabel}</div>

                  ${e.force_now ? `<div id="rabbit_btn" class="rabbit-btn ${sChargeType?.state === 'As Fast As Possible' ? 'on' : ''}"><ha-icon icon="mdi:rabbit"></ha-icon></div>` : ''}
                  <div class="target-cell">
                    <div id="target_value" class="target-value">${displayTargetValue}</div>
                    ${useEnergyMode ? '' : `<div class="mini-range-target" aria-label="range at target">${rangeTargetStr}</div>`}
                  </div>
                  ${(tNext && e.schedule) ? `<div id="time_btn" class="time-btn ${chargeTime && chargeTime !== '--:--' ? 'on' : ''}">${chargeTime}</div>` : ''}
                </div>
                </div>
                <!-- QS-232 review-fix #04: invisible spacer preserves .stack's vertical balance
                     after sun-btn was moved to absolute positioning. Without this, the soc-block
                     and mini-grid visibly shift downward. -->
                <div class="sun-btn-spacer" aria-hidden="true"></div>
              </div>
            </div>
            <!-- QS-232 review-fix #03: sun-btn moved out of .center > .stack > .center-controls
                 to a direct child of .ring, matching the override-btn placement pattern in
                 boiler/radiator/climate cards (position: absolute; bottom: 15px;). -->
            ${swPriority ? `<div id="sun_btn" class="sun-btn ${swPriority?.state === 'on' ? 'on' : ''}"><ha-icon icon="mdi:weather-sunny"></ha-icon></div>` : ''}
          </div>
        </div>

        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:account"></ha-icon>
            <select id="person">
              ${personOptionsHtml}
            </select>
          </div>
        </div>
        <div class="forecast-row">Forecast: ${forecastDisplay}</div>
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:ev-station"></ha-icon>
            <select id="charger">
              ${chargerOptionsHtml}
            </select>
          </div>
        </div>
        <div class="below-line full">
           <button id="reset" class="danger pill outline">Reset</button>
        </div>
      </ha-card>
    `;

      // QS-232: innerHTML rewrite just replaced the wave <path> nodes
      // and the sparkle/lightning layer <g>s. Sync the memo keys to
      // the freshly-rendered state (prevents a redundant regen on the
      // next frame) and null cached DOM refs + clear the logical
      // particle arrays via _resetDomRefs() so subsequent spawns
      // don't try to advance dead nodes.
      this._lastWaterBaseY = this._waterBaseY;
      // Review-fix #01 #8: `??` does NOT trap NaN — if
      // `_currentAmplitude` ever became NaN, the prior form would
      // assign NaN to `_lastAmplitude`, making `ampDelta` always NaN
      // (i.e. always `< AMP_REGEN_THRESHOLD`), and the wave-path
      // regen would freeze for the lifetime of the card. Use an
      // explicit `Number.isFinite` check instead.
      this._lastAmplitude  = Number.isFinite(this._currentAmplitude) ? this._currentAmplitude : CALM_AMP;
      this._resetDomRefs();

      // QS-232 review-fix #03: restore the preserved sparkles AND
      // lightning bolts into the FRESH layers (their DOM identity
      // changed via innerHTML — same `id`, new element). For each
      // preserved particle, re-attach its detached `el` to the new
      // layer; the per-frame state (cy, life, …) survived in the
      // JS array so the RAF advance loop picks up exactly where it
      // left off, with no visual blink. Cadence counters
      // (`_nextSparkleAt`, `_nextLightningAt`) are restored
      // INSIDE the `length > 0` branches below; when the snapshot
      // arrays are empty (common — sparkles live ~0.45 s and
      // hass-pushes fire every 100–500 ms, so most renders snapshot
      // 0–3 sparkles), the cadence counters survive the rebuild
      // because `_resetDomRefs` doesn't touch them. Net effect: a
      // spawn cadence is continuous regardless of whether the
      // particle arrays were populated at snapshot time. If a new
      // layer isn't found (defensive — e.g. shadow root was torn
      // down mid-flight), preserved particles are dropped to avoid
      // leaking detached DOM. Mirrors the QS-214 boiler
      // steam/bubble snapshot/restore pattern.
      if (preservedSparkles?.length) {
        const newSparkleLayer = this._sparkleLayerId
          ? this._root.getElementById(this._sparkleLayerId)
          : null;
        if (newSparkleLayer) {
          for (const s of preservedSparkles) {
            if (s?.el) newSparkleLayer.appendChild(s.el);
          }
          // Review-fix #02 #16: drop preserved sparkles whose `cy`
          // is now ABOVE the new `_waterBaseY` (i.e., the SOC dropped
          // enough that the soup surface rose past the sparkle's
          // frozen y-position). Otherwise the sparkle would visually
          // float in the air above the soup for the remainder of its
          // ~0.45s life. The DOM `<circle>` is removed too so it
          // doesn't paint during the brief gap.
          this._sparkles = preservedSparkles.filter(s => {
            if (!s?.el) return false;
            if (s.cy < this._waterBaseY) {
              s.el.remove();
              return false;
            }
            return true;
          });
          this._sparkleLayerEl = newSparkleLayer;
          this._nextSparkleAt = preservedNextSparkleAt;
        } else {
          for (const s of preservedSparkles) { s?.el?.remove(); }
          this._sparkles = [];
        }
      }
      if (preservedLightningBolts?.length) {
        const newLightningLayer = this._lightningLayerId
          ? this._root.getElementById(this._lightningLayerId)
          : null;
        if (newLightningLayer) {
          for (const b of preservedLightningBolts) {
            if (b?.el) newLightningLayer.appendChild(b.el);
          }
          this._lightningBolts = preservedLightningBolts.filter(b => b?.el);
          this._lightningLayerEl = newLightningLayer;
          this._nextLightningAt = preservedNextLightningAt;
        } else {
          for (const b of preservedLightningBolts) { b?.el?.remove(); }
          this._lightningBolts = [];
        }
      }

      // old buttons:
/*      <div className="below-line">
          <button id="schedule_inline" className="secondary btn-clock pill">
              <ha-icon icon="mdi:clock-outline"></ha-icon>
              <span>Charge at:</span></button>
          <div className="time-row" style="margin-top:0;">
              <select id="hour_select">
                  ${Array.from({length: 24}, (_, h) => `<option value="${h}" ${Math.floor(nextTimeMins / 60) === h ? 'selected' : ''}>${String(h).padStart(2, '0')}</option>`).join('')}
              </select>
              <span style="font-weight:700;">:</span>
              <select id="minute_select">
                  ${[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map(m => `<option value="${m}" ${(nextTimeMins % 60) === m ? 'selected' : ''}>${String(m).padStart(2, '0')}</option>`).join('')}
              </select>
          </div>
      </div>
      <div className="below-line full">
          <button id="force" className="primary pill align-left">
              <ha-icon icon="mdi:rabbit"></ha-icon>
              <span className="btn-center">Force charge now</span></button>
      </div>*/

      // Wire events
      const root = this._root;
      const ids = (k) => root.getElementById(k);
      const withEntityId = (id) => ({entity_id: id});

      if (selPerson) {
          const personSel = ids('person');
          const startP = () => {
              this._isInteractingPerson = true;
          };
          const endP = () => {
              this._isInteractingPerson = false;
              this._render();
          };
          personSel?.addEventListener('focus', startP);
          personSel?.addEventListener('blur', endP);
          personSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return; // ignore placeholder
              await this._select(e.attached_person, option);
              // Clear the interaction flag and force re-render to show the updated state
              this._isInteractingPerson = false;
              this._render();
          });
          const personPill = personSel?.closest('.pill');
          if (personPill && personSel) {
              personPill.addEventListener('click', (ev) => {
                  if (ev.target.tagName === 'SELECT') return;
                  try { personSel.showPicker(); } catch (_) { personSel.focus(); }
              });
          }
      }

      if (selCharger) {
          const chargerSel = ids('charger');
          const startC = () => {
              this._isInteractingCharger = true;
          };
          const endC = () => {
              this._isInteractingCharger = false;
              this._render();
          };
          chargerSel?.addEventListener('focus', startC);
          chargerSel?.addEventListener('blur', endC);
          chargerSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return; // ignore placeholder
              await this._select(e.charger_select, option);
              // Clear the interaction flag and force re-render to show the updated state
              this._isInteractingCharger = false;
              this._render();
          });
          const chargerPill = chargerSel?.closest('.pill');
          if (chargerPill && chargerSel) {
              chargerPill.addEventListener('click', (ev) => {
                  if (ev.target.tagName === 'SELECT') return;
                  try { chargerSel.showPicker(); } catch (_) { chargerSel.focus(); }
              });
          }
      }

      if (selLimit) {
          ids('limit')?.addEventListener('change', (ev) => {
              const option = ev.target.value;
              this._select(e.next_limit, option);
          });
      }

      if (tNext) {
          const hourSel = ids('hour_select');
          const minSel = ids('minute_select');
          const update = () => {
              const h = Number(hourSel?.value ?? 0);
              const m = Number(minSel?.value ?? 0);
              const mins = h * 60 + m;
              const hm = formatHm(mins);
              const val = hm + ':00';
              this._localNextTimeMins = mins; // keep local until HA push comes back to avoid select jumping
              // S9: clear the local override after a grace period.
              if (this._localNextTimeClearTimer) {
                  clearTimeout(this._localNextTimeClearTimer);
              }
              this._localNextTimeClearTimer = setTimeout(() => {
                  this._localNextTimeMins = null;
                  this._render();
              }, 5000);
              this._setTime(e.next_time, val);
          };
          const startInteract = () => {
              this._isInteracting = true;
          };
          const endInteract = () => {
              this._isInteracting = false;
              this._render();
          };
          hourSel?.addEventListener('focus', startInteract);
          minSel?.addEventListener('focus', startInteract);
          hourSel?.addEventListener('blur', endInteract);
          minSel?.addEventListener('blur', endInteract);
          hourSel?.addEventListener('change', update);
          minSel?.addEventListener('change', update);
      }

      // Mobile touch fix: every button below uses a dual click + touchend pattern.
      // On mobile, the browser synthesizes "click" from touchstart/touchend with up to a
      // 300ms delay. If a hass re-render (innerHTML replacement) occurs in that window, the
      // DOM node is destroyed before the synthetic click fires, so the tap is lost. The
      // touchend handler fires immediately, calls preventDefault() to suppress the delayed
      // synthetic click (avoiding double-fire on desktop), and invokes the action directly.
      if (swPriority) {
          const togglePriority = async () => {
              const btn = ids('sun_btn');
              try {
                  if (swPriority.state === 'on') {
                      await this._turnOff(e.bump_priority);
                      btn?.classList.remove('on');
                  } else {
                      await this._turnOn(e.bump_priority);
                      btn?.classList.add('on');
                  }
              } catch (_) {
                  // ignore errors; HA state will resync UI on next render
              }
          };
          ids('priority')?.addEventListener('click', togglePriority);
          const sbtn = ids('sun_btn');
          if (sbtn) {
              sbtn.style.pointerEvents = 'auto';
              sbtn.addEventListener('click', togglePriority);
              sbtn.addEventListener('touchend', (ev) => { ev.preventDefault(); togglePriority(); });
          }
      }

      // Rabbit button for force now
      if (e.force_now) {
          const rbtn = ids('rabbit_btn');
          if (rbtn) {
              rbtn.style.pointerEvents = 'auto';
              const rbtnAction = async () => {
                  if (this._root?.querySelector('.disabled')) return;

                  // Check if already in "As Fast As Possible" mode
                  const isAlreadyForcing = sChargeType?.state === 'As Fast As Possible';

                  if (isAlreadyForcing && e.clean_constraints) {
                      showDialog({
                          title: 'Stop Force Charging',
                          message: 'This will stop the current charge.\nOk to proceed?',
                          buttons: [
                              {text: 'Cancel', variant: 'secondary'},
                              {
                                  text: 'Reset', variant: 'danger', onClick: async () => {
                                      await this._press(e.clean_constraints);
                                  }
                              },
                          ]
                      });
                  } else {
                      showDialog({
                          title: 'Force charge now',
                          message: 'Start full-speed charge immediately?\nThis will use maximum available power.',
                          buttons: [
                              {text: 'Cancel', variant: 'secondary'},
                              {
                                  text: 'Start', variant: 'primary', onClick: async () => {
                                      await this._press(e.force_now);
                                  }
                              },
                          ]
                      });
                  }
              };
              rbtn.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); rbtnAction(); });
              rbtn.addEventListener('touchend', (ev) => { ev.preventDefault(); rbtnAction(); });
          }
      }

      // Time button for finish time
      if (tNext && e.schedule) {
          const tbtn = ids('time_btn');
          if (tbtn) {
              tbtn.style.pointerEvents = 'auto';
              const tbtnAction = async () => {
                  if (this._root?.querySelector('.disabled')) return;

                  let defaultHour, defaultMin;
                  if (chargeTime && chargeTime !== '--:--' && chargeTime.includes(':')) {
                      const chargeMins = parseTimeToMinutes(chargeTime);
                      defaultHour = Math.floor(chargeMins / 60);
                      defaultMin = chargeMins % 60;
                      defaultMin = Math.ceil(defaultMin / 5) * 5;
                      if (defaultMin === 60) {
                          defaultMin = 0;
                          defaultHour = (defaultHour + 1) % 24;
                      }
                  } else {
                      defaultHour = Math.floor(nextTimeMins / 60);
                      defaultMin = nextTimeMins % 60;
                      defaultMin = Math.ceil(defaultMin / 5) * 5;
                      if (defaultMin === 60) {
                          defaultMin = 0;
                          defaultHour = (defaultHour + 1) % 24;
                      }
                  }

                  const customContent = `
            <p>Select the next time the charge of the car should end:</p>
            <div class="time-picker">
              <select id="dialog_hour_select">
                ${Array.from({length: 24}, (_, h) => `<option value="${h}" ${defaultHour === h ? 'selected' : ''}>${String(h).padStart(2, '0')}</option>`).join('')}
              </select>
              <span>:</span>
              <select id="dialog_minute_select">
                ${[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map(m => `<option value="${m}" ${defaultMin === m ? 'selected' : ''}>${String(m).padStart(2, '0')}</option>`).join('')}
              </select>
            </div>
          `;

                  const dialog = showDialog({
                      title: 'Charge Finish Time',
                      customContent: customContent,
                      buttons: [
                          {text: 'Cancel', variant: 'secondary'},
                          {
                              text: 'Reset',
                              variant: 'danger',
                              onClick: async () => {
                                  if (e.clean_constraints) await this._press(e.clean_constraints);
                              }
                          },
                          {
                              text: 'Apply',
                              variant: 'primary',
                              onClick: async () => {
                                  const dialogRoot = dialog.querySelector('.dialog');
                                  const hourSel = dialogRoot?.querySelector('#dialog_hour_select');
                                  const minSel = dialogRoot?.querySelector('#dialog_minute_select');
                                  const h = Number(hourSel?.value ?? 0);
                                  const m = Number(minSel?.value ?? 0);
                                  const mins = h * 60 + m;
                                  const hm = formatHm(mins);
                                  const val = hm + ':00';
                                  this._localNextTimeMins = mins;
                                  // S9: clear the local override after a
                                  // grace period so out-of-band backend
                                  // updates aren't masked indefinitely.
                                  if (this._localNextTimeClearTimer) {
                                      clearTimeout(this._localNextTimeClearTimer);
                                  }
                                  this._localNextTimeClearTimer = setTimeout(() => {
                                      this._localNextTimeMins = null;
                                      this._render();
                                  }, 5000);
                                  await this._setTime(e.next_time, val);
                                  await this._press(e.schedule);
                              }
                          },
                      ]
                  });
              };
              tbtn.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); tbtnAction(); });
              tbtn.addEventListener('touchend', (ev) => { ev.preventDefault(); tbtnAction(); });
          }
      }
      const showDialog = (opts) => {
          const {title, message, buttons, customContent} = opts;
          const wrap = document.createElement('div');
          wrap.className = 'modal';
          // S6: escape user-controlled `title` and `message`.
          const contentHtml = customContent || `<p>${this._escapeHtml(message)}</p>`;
          wrap.innerHTML = `<div class="dialog"><h3>${this._escapeHtml(title)}</h3>${contentHtml}<div class="actions"></div></div>`;
          const actions = wrap.querySelector('.actions');
          this._modalOpen = true;
          buttons.forEach(b => {
              const el = document.createElement('button');
              el.className = `btn ${b.variant || 'secondary'}`;
              el.textContent = b.text;
              let activated = false;
              const activate = () => {
                  if (activated) return;
                  activated = true;
                  if (b.onClick) b.onClick();
                  wrap.remove();
                  this._modalOpen = false;
                  this._render();
              };
              el.addEventListener('click', activate);
              el.addEventListener('touchend', (ev) => {
                  ev.preventDefault();
                  activate();
              });
              actions.appendChild(el);
          });
          this._root.appendChild(wrap);
          return wrap;
      };

      if (e.force_now) {
          const forceBtn = ids('force');
          const forceAction = async () => {
              if (this._root?.querySelector('.disabled')) return;
              showDialog({
                  title: 'Force charge now',
                  message: 'Start full-speed charge immediately?\nThis will use maximum available power.',
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
                      {
                          text: 'Start', variant: 'primary', onClick: async () => {
                              await this._press(e.force_now);
                          }
                      },
                  ]
              });
          };
          forceBtn?.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); forceAction(); });
          forceBtn?.addEventListener('touchend', (ev) => { ev.preventDefault(); forceAction(); });
      }

      if (e.schedule) {
          const schedBtn = ids('schedule_inline');
          const schedAction = async () => {
              if (this._root?.querySelector('.disabled')) return;
              const limit = selLimit?.state || '';
              const time = tNext?.state || '';
              showDialog({
                  title: 'Add scheduled charge',
                  message: `Schedule a charge for ${time} at target SOC: ${limit} .\nProceed?`,
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
                      {
                          text: 'Add', variant: 'primary', onClick: async () => {
                              await this._press(e.schedule);
                          }
                      },
                  ]
              });
          };
          schedBtn?.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); schedAction(); });
          schedBtn?.addEventListener('touchend', (ev) => { ev.preventDefault(); schedAction(); });
      }

      if (e.reset) {
          const resetBtn = ids('reset');
          const resetAction = async () => {
              showDialog({
                  title: 'Reset car state',
                  message: 'This will reset internal state for this car and cannot be undone.\nProceed?',
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
                      {
                          text: 'Reset', variant: 'danger', onClick: async () => {
                              await this._press(e.reset);
                          }
                      },
                  ]
              });
          };
          resetBtn?.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); resetAction(); });
          resetBtn?.addEventListener('touchend', (ev) => { ev.preventDefault(); resetAction(); });
      }

      // Quick percent chips
      const findOptionByPercent = (percent) => {
          const options = selLimit?.attributes?.options || [];
          const p = String(percent);
          return options.find(o => o.startsWith(p) || o.startsWith(p + "%") || o.includes(`${p}%`));
      };

      const findOptionByEnergy = (energy) => {
          const options = selLimit?.attributes?.options || [];
          const e = String(Math.round(energy));
          return options.find(o => o.startsWith(e) || o.includes(`${e}kWh`) || o.includes(`${e} kWh`));
      };

      const parseEnergyFromOption = (opt) => {
          const m = String(opt).match(/([\d.]+)\s*kWh/i);
          return m ? Number(m[1]) : undefined;
      };

      const parsePctFromOption = (opt) => {
          const m = String(opt).match(/(\d+)\s*%/);
          return m ? Number(m[1]) : undefined;
      };

      const allowedPercents = (selLimit?.attributes?.options || [])
          .map(parsePctFromOption)
          .filter(v => v != null && !Number.isNaN(v))
          .sort((a, b) => a - b);

      const allowedEnergies = (selLimit?.attributes?.options || [])
          .map(parseEnergyFromOption)
          .filter(v => v != null && !Number.isNaN(v))
          .sort((a, b) => a - b);

      // Generate fallback energy values with 2 kWh increments (matching get_car_next_charge_values_options_energy)
      const generateEnergyFallback = (maxKwh) => {
          const values = [];
          for (let v = 0; v <= maxKwh; v += 2) {
              values.push(v);
          }
          if (!values.includes(maxKwh)) {
              values.push(maxKwh);
          }
          return values;
      };

      const allowedOrDefault = useEnergyMode
          ? (allowedEnergies.length ? allowedEnergies : generateEnergyFallback(maxCircleValue))
          : (allowedPercents.length ? allowedPercents : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]);

      root.querySelectorAll('.chip[data-pct]')?.forEach(el => {
          el.addEventListener('click', () => {
              const pct = Number(el.getAttribute('data-pct'));
              const opt = findOptionByPercent(pct);
              if (opt) this._select(e.next_limit, opt);
          });
      });

      // Drag target handle on ring
      const svg = this._root.querySelector('.ring svg');
      const handle = this._root.getElementById('target_handle');
      if (svg && handle) {
          const pt = svg.createSVGPoint();
          const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
          const onMove = (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              const e2 = ev.touches ? ev.touches[0] : ev;
              pt.x = e2.clientX;
              pt.y = e2.clientY;
              const cursor = pt.matrixTransform(svg.getScreenCTM().inverse());
              const dx = cursor.x - center.cx;
              const dy = cursor.y - center.cy;
              // Convert to [0,360]
              // -dy because svg y are downward
              let ang = rad2deg(Math.atan2(-dy, dx));
              // Map to arc domain [startDeg, endDeg] allowing values > 360
              let a = ang;
              if (a < startDeg) a = startDeg;
              if (a > endDeg) a = endDeg;
              // Snap to available select options
              const rawPct = ((a - startDeg) / rangeDeg) * 100;
              const list = allowedOrDefault;
              const snapValue = list.reduce((best, v) => Math.abs(v - (useEnergyMode ? (rawPct / 100 * maxCircleValue) : rawPct)) < Math.abs(best - (useEnergyMode ? (rawPct / 100 * maxCircleValue) : rawPct)) ? v : best, list[0]);

              // Store the percentage for the circle (0-100)
              const displayPct = useEnergyMode ? (snapValue / maxCircleValue * 100) : snapValue;
              this._targetDragPct = displayPct;
              this._targetDragValue = snapValue; // Store actual value (percent or kWh)
              this._isInteractingTarget = true;

              // Update handle and target label without full render to keep it smooth
              const angSnap = startDeg + (displayPct / 100) * rangeDeg;
              const pos = polar(center.cx, center.cy, ringCirc, angSnap);
              handle.setAttribute('cx', pos.x.toFixed(2));
              handle.setAttribute('cy', pos.y.toFixed(2));
              const handleText = this._root.getElementById('target_handle_text');
              if (handleText) {
                  handleText.setAttribute('x', pos.x.toFixed(2));
                  handleText.setAttribute('y', pos.y.toFixed(2));
                  handleText.textContent = this._fmt(snapValue);
              }
              const tv = this._root.getElementById('target_value');
              if (tv) tv.innerHTML = useEnergyMode ? `${this._fmt(snapValue)}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>` : `${this._fmt(snapValue)}%`;
          };
          const onUp = async (ev) => {
              if (this._upInProgress) return;
              this._upInProgress = true;

              if (ev) {
                  ev.stopPropagation();
                  ev.preventDefault();
              }

              const dragPct = this._targetDragPct;
              const dragValue = this._targetDragValue;

              if (dragValue != null) {
                  const opt = useEnergyMode ? findOptionByEnergy(dragValue) : findOptionByPercent(dragValue);
                  if (opt) await this._select(e.next_limit, opt);
                  this._localTargetPct = dragPct;
                  this._pendingClearLocalTarget && clearTimeout(this._pendingClearLocalTarget);
                  this._pendingClearLocalTarget = setTimeout(() => {
                      this._localTargetPct = null;
                      this._pendingClearLocalTarget = null;
                      this._render();
                  }, 5000);
              }
              this._targetDragPct = null;
              this._targetDragValue = null;
              this._isInteractingTarget = false;
              this._upInProgress = false;
              handle.style.cursor = 'grab';
          };

          if (window.PointerEvent) {
              const onPointerMove = (ev) => onMove(ev);
              const onPointerUp = async (ev) => {
                  try { handle.releasePointerCapture(ev.pointerId); } catch (_) {}
                  handle.removeEventListener('pointermove', onPointerMove);
                  handle.removeEventListener('pointerup', onPointerUp);
                  handle.removeEventListener('pointercancel', onPointerUp);
                  await onUp(ev);
              };
              const onPointerDown = (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  this._isInteractingTarget = true;
                  try { handle.setPointerCapture(ev.pointerId); } catch (_) {}
                  handle.addEventListener('pointermove', onPointerMove);
                  handle.addEventListener('pointerup', onPointerUp);
                  handle.addEventListener('pointercancel', onPointerUp);
                  handle.style.cursor = 'grabbing';
              };
              handle.addEventListener('pointerdown', onPointerDown);
          } else {
              const onDown = (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  this._isInteractingTarget = true;
                  document.addEventListener('mousemove', onMove);
                  document.addEventListener('touchmove', onMove, {passive: false});
                  document.addEventListener('mouseup', onUpLegacy);
                  document.addEventListener('touchend', onUpLegacy);
                  handle.style.cursor = 'grabbing';
              };
              const onUpLegacy = async (ev) => {
                  document.removeEventListener('mousemove', onMove);
                  document.removeEventListener('touchmove', onMove);
                  document.removeEventListener('mouseup', onUpLegacy);
                  document.removeEventListener('touchend', onUpLegacy);
                  await onUp(ev);
              };
              handle.addEventListener('mousedown', onDown);
              handle.addEventListener('touchstart', onDown, {passive: false});
          }
      }
  }
}

if (!customElements.get('qs-car-card')) {
    customElements.define('qs-car-card', QsCarCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-car-card')) {
    window.customCards.push({
        type: 'qs-car-card',
        name: 'QS Car Card',
        description: 'Quiet Solar car control card',
    });
}
