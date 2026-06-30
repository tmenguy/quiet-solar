/*
  QS Car Card - custom:qs-car-card
  Zero-build single-file Lit-style web component compatible with Home
  Assistant.

  QS-232 layers an "electron soup" SOC animation inside the ring,
  mirroring the QS-200 / QS-211 / QS-214 boiler architecture. The
  high-level design — single-layer sine wave inside a clipPath,
  idle↔charge cross-fade, lightning-blue sparkle particles, lightning
  bolts during charging, degraded-state CSS desaturate, continuous
  RAF, and QS-217 carve covers extended to three inside-disc buttons
  — is documented in
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
// constant NAME); semantically it's now a fill colour.
// Iteration history:
// - Initial:                 `#E0F7FF` near-white (read as grey through glow)
// - Review-fix #02 user #1:  `#33B5FF` electric blue ("too blue")
// - Review-fix #02 user #2:  `#E580FF` very bright purple ("purple is ugly")
// - Review-fix #02 user #3:  `#00E5FF` — same `SPARKLE_CHARGE_COLOR`
//                            electric-cyan as the charging sparkles,
//                            so bolts + sparkles share one bright-
//                            blue palette during charging discharge.
const LIGHTNING_STROKE_COLOR = '#00E5FF';
const LIGHTNING_TOP_MARGIN_PX = 4;
const LIGHTNING_LATERAL_JITTER_PX = 18;
// Tapering widths for the thunderbolt outline (in SVG units).
// `LIGHTNING_TOP_WIDTH` is the width at the top of the bolt; the
// path then narrows to a point at the soup surface via two
// intermediate widths (~55% and ~25% of the top, computed inline).
const LIGHTNING_TOP_WIDTH = 9;
// Branch (ramification) count per bolt. Real lightning has many
// smaller forks off the main channel — these short tapered branches
// give the bolt the characteristic "lightning tree" look.
const LIGHTNING_BRANCH_MIN_COUNT = 3;
const LIGHTNING_BRANCH_MAX_COUNT = 6;
const LIGHTNING_BRANCH_MIN_LEN = 8;
const LIGHTNING_BRANCH_MAX_LEN = 22;
// Review-fix #01 #16: removed unused `LIGHTNING_SEGMENTS = 3` —
// the lightning path hardcodes its skeleton inline rather than
// driving an emission loop, so the constant served no purpose.
// Review-fix #02 user follow-up: removed `LIGHTNING_STROKE_WIDTH`
// — the new tapered-polygon shape has no uniform stroke.
// Review-fix #02 user follow-up #2: removed `LIGHTNING_GLOW_STDDEV`
// and the associated `<filter>` in `<defs>` — user said "no need of
// the glow effect for the lightning bolts". The sharp filled
// polygon now reads as bright-edged plasma without the soft blur.

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
// QS-235 — the bottom-center sun-btn cover now uses the shared
// `RING_BOTTOM_CARVE_CX/CY/R` constants (`shared/qs-card-base.js`,
// imported above) — the same set the duration cards use for their
// `override_btn_cover`. The per-card `SUN_BTN_CARVE_*` duplicate is
// removed. The rabbit/time covers keep their own car-local geometry.
const RABBIT_BTN_CARVE_CX = 96;     // left column of mini-grid
const RABBIT_BTN_CARVE_CY = 206;
const RABBIT_BTN_CARVE_R  = 32;
const TIME_BTN_CARVE_CX = 224;      // right column of mini-grid
const TIME_BTN_CARVE_CY = 206;
const TIME_BTN_CARVE_R  = 32;

// QS-199 — shared module imports. The car card extends `QsCardBase`
// directly (NOT `QsRingDurationCardBase` — car uses a full-circle ring
// with 3 inside-disc buttons instead of one override-btn at the
// bottom). Inherits lifecycle, service callers, defensive utilities,
// modal dialog, keyboard activation, and the 5 wire-helpers. Car
// retains its own ring HTML (full-circle, sun/rabbit/time-btn carve
// covers), sparkle system, lightning system.
import { baseCardCSS } from './shared/qs-card-styles.js';
import { QsCardBase, polar, arcPath, pctToDeg, RING_BOTTOM_CARVE_CX, RING_BOTTOM_CARVE_CY, RING_BOTTOM_CARVE_R } from './shared/qs-card-base.js';

class QsCarCard extends QsCardBase {
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
            // Tapered Zeus-thunderbolt with branching ramifications.
            // 4-node main skeleton from top (wide) to soup tip
            // (point), with lateral jitter at the two midpoints for
            // the zig-zag. Plus N small tapered branches forking off
            // the main spine — gives the bolt the characteristic
            // "lightning tree" look of a real discharge.
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
            const j1 = (Math.random() * 2 - 1) * LIGHTNING_LATERAL_JITTER_PX * 0.5;
            const j2 = (Math.random() * 2 - 1) * LIGHTNING_LATERAL_JITTER_PX * 0.5;
            const x1 = startCx + (endCx - startCx) * 0.33 + j1;
            const x2 = startCx + (endCx - startCx) * 0.67 + j2;
            const y1 = topY + totalLen * 0.33;
            const y2 = topY + totalLen * 0.67;
            // Half-widths along the main skeleton (50/30/15/0 of top).
            const h0 = LIGHTNING_TOP_WIDTH * 0.50;
            const h1 = LIGHTNING_TOP_WIDTH * 0.30;
            const h2 = LIGHTNING_TOP_WIDTH * 0.15;
            // Main bolt outline — left edge top→bottom, tip, right
            // edge bottom→top, closed.
            let d = `M ${(startCx - h0).toFixed(2)},${topY.toFixed(2)}`
                  + ` L ${(x1 - h1).toFixed(2)},${y1.toFixed(2)}`
                  + ` L ${(x2 - h2).toFixed(2)},${y2.toFixed(2)}`
                  + ` L ${endCx.toFixed(2)},${this._waterBaseY.toFixed(2)}`
                  + ` L ${(x2 + h2).toFixed(2)},${y2.toFixed(2)}`
                  + ` L ${(x1 + h1).toFixed(2)},${y1.toFixed(2)}`
                  + ` L ${(startCx + h0).toFixed(2)},${topY.toFixed(2)}`
                  + ` Z`;
            // Branches: short tapered triangles forking from random
            // positions along the main spine. Each branch is a
            // separate sub-path appended to the compound `d`.
            const branchCount = LIGHTNING_BRANCH_MIN_COUNT +
              Math.floor(Math.random() * (LIGHTNING_BRANCH_MAX_COUNT - LIGHTNING_BRANCH_MIN_COUNT + 1));
            for (let bi = 0; bi < branchCount; bi++) {
              // Origin along the main spine — `t` in [0.15, 0.85].
              const t = 0.15 + Math.random() * 0.70;
              // Linear interp through the 4-node skeleton to find
              // origin point + local main-bolt half-width.
              let ox, oy, originHalfWidth;
              if (t < 0.33) {
                const u = t / 0.33;
                ox = startCx + (x1 - startCx) * u;
                oy = topY + (y1 - topY) * u;
                originHalfWidth = h0 + (h1 - h0) * u;
              } else if (t < 0.67) {
                const u = (t - 0.33) / 0.34;
                ox = x1 + (x2 - x1) * u;
                oy = y1 + (y2 - y1) * u;
                originHalfWidth = h1 + (h2 - h1) * u;
              } else {
                const u = (t - 0.67) / 0.33;
                ox = x2 + (endCx - x2) * u;
                oy = y2 + (this._waterBaseY - y2) * u;
                originHalfWidth = h2 * (1 - u);
              }
              // Branch direction — biased outward+downward.
              // dirAngle = 90° is straight down. Range covers down ±
              // ~70° to give some near-horizontal forks too.
              const angleDeg = -70 + Math.random() * 140;
              const dirAngle = (90 + angleDeg) * Math.PI / 180;
              const branchLen = LIGHTNING_BRANCH_MIN_LEN +
                Math.random() * (LIGHTNING_BRANCH_MAX_LEN - LIGHTNING_BRANCH_MIN_LEN);
              const tipX = ox + Math.cos(dirAngle) * branchLen;
              const tipY = oy + Math.sin(dirAngle) * branchLen;
              // Branch origin width — ~70% of the main bolt's local
              // half-width, perpendicular to the branch direction.
              const branchHalfW = Math.max(0.6, originHalfWidth * 0.7);
              const blen = Math.sqrt(
                (tipX - ox) * (tipX - ox) + (tipY - oy) * (tipY - oy)
              );
              if (blen < 0.5) continue;
              // Perpendicular unit vector to the branch direction.
              const px = -(tipY - oy) / blen;
              const py = (tipX - ox) / blen;
              const lx = ox + px * branchHalfW;
              const ly = oy + py * branchHalfW;
              const rx = ox - px * branchHalfW;
              const ry = oy - py * branchHalfW;
              // Tapered triangular branch: origin-left, tip, origin-right.
              d += ` M ${lx.toFixed(2)},${ly.toFixed(2)}`
                +  ` L ${tipX.toFixed(2)},${tipY.toFixed(2)}`
                +  ` L ${rx.toFixed(2)},${ry.toFixed(2)}`
                +  ` Z`;
            }
            const el = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            el.setAttribute('d', d);
            el.setAttribute('fill', LIGHTNING_STROKE_COLOR);
            el.setAttribute('stroke', 'none');
            el.setAttribute('fill-rule', 'nonzero');
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
    // QS-199 review-fix S13: chain to the base so its _stopAnimation +
    // standard flag-reset (and any future base teardown) always runs.
    super.disconnectedCallback();
    // Car-specific interaction flags not reset by the base.
    this._isInteracting = false;
    this._isInteractingCharger = false;
    this._isInteractingPerson = false;
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

  // QS-199 — _escapeHtml, _entity, _call, _press, _turnOn, _turnOff,
  // _select, _setNumber, _setTime, _fmt all inherited from QsCardBase
  // (in shared/qs-card-base.js). Local definitions removed for AC1.

  // Override `set hass` to add the car-specific `_isInteracting`,
  // `_isInteractingCharger`, `_isInteractingPerson` guards.
  set hass(hass) {
    this._hass = hass;
    if (!this._root) return;
    // QS-271 — also defer the repaint while a tap is in flight (state is
    // stored above; only the _render() repaint is deferred).
    if (this._isInteracting || this._isInteractingCharger || this._isInteractingPerson || this._modalOpen || this._isInteractingTarget || this._isPressInFlight()) return;
    this._render();
  }

  getCardSize() { return 6; }

  _percent(num) {
    const n = Number(num);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(100, n));
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
      const sChargeOrigin = this._entity(e.charge_origin);
      const sUsePercentMode = this._entity(e.use_percent_mode);
      const sIsOffGrid = this._entity(e.is_off_grid);
      const sCarIsStale = this._entity(e.car_is_stale);
      // QS-243 — estimated-SOC wiring: the asterisk flag, the manual-SOC
      // number entity (popup prefill + write target), and the reset button.
      const sIsSocEstimated = this._entity(e.is_soc_estimated);
      const sManualSoc = this._entity(e.manual_soc);

      const title = (cfg.title || cfg.name) || (sSoc ? (sSoc.attributes.friendly_name || sSoc.entity_id) : "Car");

      // Check if system is off-grid
      const isOffGrid = sIsOffGrid?.state === 'on';
      // Check if car API data is stale
      const isStale = sCarIsStale?.state === 'on';
      let soc = this._percent(sSoc?.state);
      // QS-235 AC6 — guard-shaped sensor read via the shared `_safeNumber`
      // (trims, rejects unknown/unavailable/±Infinity) instead of the raw
      // `Number(sPower?.state || "0")` pattern.
      const powerW = this._safeNumber(sPower, 0);
      this._chargePower = powerW;
      const target = selLimit?.state || "";
      const charging = (powerW > 50);
      this._charging = charging;
      // QS-232: per-instance unique SVG IDs so two car cards on the
      // same dashboard don't collide (cheap insurance — households
      // can plausibly have ≥ 2 cars on one dashboard). Mirrors the
      // boiler / pool _nextClipId pattern.
      if (!this._electronClipId) {
        QsCarCard._nextClipId = (QsCarCard._nextClipId || 0) + 1;
        const uid = QsCarCard._nextClipId;
        this._electronClipId   = `car_eClip_${uid}`;
        this._sparkleLayerId   = `car_sparkLayer_${uid}`;
        this._lightningLayerId = `car_lightningLayer_${uid}`;
        // Review-fix #02 user follow-up #2: `_lightningFilterId`
        // removed alongside the lightning glow filter — the new
        // sharp purple bolt doesn't use a `<filter>`.
      }
      const electronClipId   = this._electronClipId;
      const sparkleLayerId   = this._sparkleLayerId;
      const lightningLayerId = this._lightningLayerId;
      const carChargeTypeIcons = {
          "Unknown": "mdi:help-circle-outline",
          "Not Plugged": "mdi:power-plug-off",
          "Faulted": "mdi:emoticon-dead",
          "No Power To Car": "mdi:flash-off",
          "Not Charging": "mdi:battery-off",
          "Target Met": "mdi:battery-high",
          "Manual": "mdi:hand-back-right",
          "Calendar": "mdi:calendar",
          "As Fast As Possible": "mdi:rabbit",
          "Manual As Fast As Possible": "mdi:rabbit",
          "Solar Priority": "mdi:solar-power",
          "Solar": "mdi:white-balance-sunny",
          "Person Automated": "mdi:account-clock",
      };
      const iconForChargeType = (str) => carChargeTypeIcons[str];
      // QS-280: the rabbit (charge-as-fast-as-possible) button covers both
      // as-fast charge-type strings — solver-driven "As Fast As Possible" and
      // user-force "Manual As Fast As Possible" (what get_charge_type() returns
      // after a user press). The set is matched directly here, decoupled from
      // the icon map, so the icon for either state can change independently
      // without silently breaking the button's lit state or its Stop dialog.
      // These are the rendered values of the CAR_CHARGE_TYPE_*AS_FAST_AS_POSSIBLE
      // constants in const.py and must stay in sync with them.
      const AS_FAST_STATES = ['As Fast As Possible', 'Manual As Fast As Possible'];
      const isAsFastState = (s) => AS_FAST_STATES.includes(s);
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

      // Get target value based on mode.
      // QS-243 — display is keyed on the estimate state, not raw staleness:
      //   - `hasSocEstimate` (absolute estimate) → `NN%*` via the percent
      //     branch (the SOC sensor already returns the estimate);
      //   - pure-delta (estimating, no base → SOC sensor unknown) → `+XX%`;
      //   - a real fresh sensor → plain `NN%`.
      const hasSocEstimate = sIsSocEstimated?.state === 'on';
      const socNumeric = isNumberLike(sSoc?.state);
      const isStalePercentMode = !useEnergyMode && !hasSocEstimate && !socNumeric;
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
          // QS-235 AC6 — guard-shaped sensor read via `_safeNumber`.
          const energyValue = this._safeNumber(sCurrentInputedEnergy, 0);
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
          // QS-243 — append an asterisk when the SOC is estimated rather
          // than read live from the car's sensor.
          displaySocValue = `${this._fmt(soc)}%${hasSocEstimate ? '*' : ''}`;
      }

      // QS-199 review-fix #02 S2 — use the inherited hardened
      // `_parseTimeToMinutes` / `_formatHm` (QsCardBase). The car's old
      // local `parseTimeToMinutes` only guarded `if (!txt)`, so an
      // `unavailable`/`unknown` next-charge time (truthy) coerced via
      // `Number('unavailable')` → NaN → 0 → midnight. The base version
      // falls back to the documented 07:00 for invalid states (S16).
      const nextTimeStr = tNext?.state || '07:00:00';
      const nextTimeMins = this._localNextTimeMins != null ? this._localNextTimeMins : this._parseTimeToMinutes(nextTimeStr);

      // QS-199 review-fix M3 — common rules come from the shared
      // baseCardCSS(palette); the car appends its bespoke layout
      // (sun/rabbit/time inside-disc buttons, mini-grid, soc-block,
      // forecast-row, progress, sections, range sliders) plus a few
      // cascade overrides of the common rules (`.below` margin-top,
      // `.ring .center` translate, `.ring .target-value` colour) AFTER
      // the base block so they win. Car uses the same blue/cyan palette
      // family as on-off-duration.
      const colors = {
        primary: '#2196F3',
        gradStart: '#00bcd4',
        gradEnd: '#8bc34a',
        animStart: '#00e1ff',
        animEnd: '#0066ff',
      };
      const css = baseCardCSS(colors) + `
      /* ----- car-specific cascade overrides of common rules ----- */
      .card.stale { border: 3px solid var(--error-color, #db4437); }
      .below { margin-top: 0px; }
      .ring .center { transform: translateY(16px); }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; }

      /* ----- car-specific extras ----- */
      .forecast-row { text-align:center; width:260px; margin: 4px auto 0; color: var(--secondary-text-color); font-weight:600; font-size: .85rem; }
      .below-line.full > button.align-left { justify-content: flex-start; }
      .below-line.full > button .btn-center { position: absolute; left: 50%; transform: translateX(-50%); }
      .below-line .time-row { justify-self: end; margin-top: 0; }
      .btn-clock { display:flex; align-items:center; gap:8px; }

      .hero .side { text-align:center; color: var(--secondary-text-color); font-weight:600; }
      .hero .side .value { display:block; font-size:1.2rem; color: var(--primary-text-color); }
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; text-shadow: var(--ring-text-shadow); }
      /* QS-243 M1 -- .ring .center is pointer-events:none (so the SVG gauge
         stays interactive); the editable SOC variant must re-enable pointer
         events or its click/tap handlers never fire. The non-editable SOC text
         stays click-through. The .disabled guard still no-ops it when the car
         is not plugged into a charger. */
      .ring .pct.soc-editable { pointer-events: auto; cursor: pointer; }
      .ring ha-icon { --mdc-icon-size: 32px; color: var(--secondary-text-color); margin-bottom: 6px; }
      .ring .soc-block { display:flex; flex-direction:column; align-items:center; gap:2px; margin-top:4px; margin-bottom:8px; }
      .ring .soc-block .charge-type-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); margin-bottom: 2px; }
      .ring .stack { width: 180px; gap:4px; }
      .ring .stack > * { text-align:center; }
      .ring .mini-grid { display:grid; grid-template-columns: repeat(3, 60px); grid-auto-rows: auto; width:180px; margin: 0 auto; justify-items:center; align-items:center; row-gap:4px; column-gap:0; }
      .ring .mini-grid.extra { row-gap:0; margin-top:2px; margin-bottom:6px; }
      .ring .target-block { gap:0; }
      .ring .target-cell { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .mini-title { color: var(--secondary-text-color); font-weight:700; font-size: .7rem; letter-spacing:.2px; white-space: nowrap; text-shadow: var(--ring-text-shadow); }
      .ring .mini-value { color: var(--primary-text-color); font-weight:800; font-size: .95rem; line-height: 1.1; white-space: pre-line; text-shadow: var(--ring-text-shadow); }
      .ring .mini-icon { --mdc-icon-size: 18px; color: var(--primary-text-color); }
      .ring .mini-range { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; text-shadow: var(--ring-text-shadow); }
      .ring .mini-range-now { color: var(--primary-text-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; transform: translateY(-8px); text-shadow: var(--ring-text-shadow); }
      .ring .mini-range-target { color: var(--primary-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; margin-bottom:0; text-shadow: var(--ring-text-shadow); }
      .disabled .ring .mini-range-target { color: var(--secondary-text-color); }
      .ring .mini-range:empty, .ring .mini-range-now:empty, .ring .mini-range-target:empty { display:none; }
      /* QS-232 review-fix #03: sun-btn uses the same absolute positioning
         as override-btn on the duration cards (bottom: 15px, centered). */
      .ring .sun-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:flex; align-items:center; justify-content:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      /* QS-232 review-fix #04/#07: icon 30px + 3px down-nudge so the
         mdi:weather-sunny glyph reads as centered in the 50×50 shell. */
      .ring .sun-btn ha-icon { --mdc-icon-size: 30px; color: var(--secondary-text-color); display:flex; align-items:center; justify-content:center; line-height:1; transform: translateY(3px); }
      .ring .sun-btn.on { border-color: rgba(255,202,40,.45); background: rgba(255,202,40,.14); box-shadow: 0 0 0 3px rgba(255,202,40,.20), 0 0 16px #FFCA28; }
      .ring .sun-btn.on ha-icon { color: #FFCA28; }
      /* QS-232 review-fix #04: invisible spacer preserving the .stack's
         vertical balance after the sun-btn moved to absolute positioning.
         74px = 6 margin + 14 label + 4 gap + 50 button. Update in lockstep
         if any of those four change. */
      .ring .sun-btn-spacer { width: 50px; height: 74px; pointer-events: none; }
      .ring .rabbit-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .rabbit-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; transform: translateY(3px); }
      .ring .rabbit-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .rabbit-btn.on ha-icon { color: #2196F3; }
      .ring .time-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; font-size: 0.99rem; font-weight: 800; color: var(--primary-color); line-height: 1; margin-top: 0; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .time-btn:hover { border-color: var(--primary-color); background: rgba(255,255,255,.08); }
      .ring .time-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; color: #2196F3; }

      .grid { display:grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap:16px; margin-top: 16px; }
      @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
      .section { border-radius: 20px; background: var(--card-background-color); padding:16px; border: 1px solid var(--divider-color); }
      .row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:8px 0; }
      input[type="time"] {
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
      .below-line .time-row select { width: auto; min-width: 64px; height: 40px; min-height: 40px; text-align: center; text-align-last: center; }
      .actions { display:grid; grid-template-columns: repeat(2,minmax(180px,1fr)); gap:12px; margin-top:8px; }
      .primary { background: var(--primary-color); color: var(--text-primary-color, #fff); }
      .secondary { background: rgba(255,255,255,.06); color: var(--primary-text-color); border:1px solid var(--divider-color); }
      /* Disconnected visual mode (greyscale) */
      .disabled .ring .target-value { color: var(--secondary-text-color); }
      /* QS-253: disabled greying is scoped to in-ring controls only. The
         below-ring controls (person / charger selects, Reset) render at full
         enabled fidelity even when the car is disconnected. */
      .disabled .ring .sun-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .ring .sun-btn ha-icon { color: var(--secondary-text-color); }
      .disabled .ring .sun-btn.on { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .ring .sun-btn.on ha-icon { color: var(--secondary-text-color); }
      .disabled .ring .rabbit-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .ring .rabbit-btn ha-icon { color: var(--secondary-text-color); }
      .disabled .ring .rabbit-btn.on { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .ring .rabbit-btn.on ha-icon { color: var(--secondary-text-color); }
      .disabled .ring .time-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; cursor: not-allowed; color: var(--secondary-text-color); }
      .disabled .progress > div { background: var(--divider-color); }
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
    `;

      const ringCirc = 130; // bigger radius
      const gapDeg = 60;
      const rangeDeg = 360 - gapDeg;
      // Place the gap centered at bottom. Start at bottom-left, end at bottom-right.
      // use degree as 0 bottom, positive clockwise , contrary to radian beware in svg : y are downward
      // degree part of [0,360[
      const startDeg = gapDeg / 2;   // bottom-left
      const endDeg = startDeg + rangeDeg; // bottom-right (wraps past 360)

      // QS-199 — geometry helpers (deg2rad / rad2deg / polar / arcPath /
      // pctToDeg) imported from shared/qs-card-base.js. The shared
      // `pctToDeg(p, startDeg, rangeDeg)` takes the range explicitly.
      const socEndDeg = pctToDeg(soc, startDeg, rangeDeg);
      const handlePct =
          this._targetDragPct != null ? this._targetDragPct :
              (this._localTargetPct != null ? this._localTargetPct : (targetPct ?? soc));
      const handleDeg = pctToDeg(handlePct, startDeg, rangeDeg);
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
      // S3: escape entity-derived option values before innerHTML interpolation.
      const chargerOptionsHtml = shouldShowPlaceholder
          ? [`<option value="" selected>No connected Charger</option>`, ...chargerOptions.map(o => `<option>${this._escapeHtml(o)}</option>`)].join('')
          : chargerOptions.map(o => `<option ${o === chargerState ? 'selected' : ''}>${this._escapeHtml(o)}</option>`).join('');

      // Person selector options
      const personOptions = selPerson?.attributes?.options || [];
      const personState = (selPerson?.state || '').trim();
      const personStateLc = personState.toLowerCase();
      const personInvalidStates = INVALID_STATES;
      const shouldShowPersonPlaceholder = !personState || personInvalidStates.includes(personStateLc) || !personOptions.includes(personState);
      // S3: escape entity-derived option values before innerHTML interpolation.
      const personOptionsHtml = shouldShowPersonPlaceholder
          ? [`<option value="" selected>No person attached</option>`, ...personOptions.map(o => `<option>${this._escapeHtml(o)}</option>`)].join('')
          : personOptions.map(o => `<option ${o === personState ? 'selected' : ''}>${this._escapeHtml(o)}</option>`).join('');

      // Charge-origin string (origin-responsive context line — replaces the
      // raw person-forecast string in the .forecast-row).
      const chargeOriginStr = sChargeOrigin?.state || '';
      const validChargeOrigin = chargeOriginStr && chargeOriginStr.toLowerCase() !== 'none' && chargeOriginStr.toLowerCase() !== 'unknown' && chargeOriginStr.toLowerCase() !== 'unavailable' && chargeOriginStr.trim() !== '';
      // S3: escape — `forecastDisplay` is interpolated into the .forecast-row innerHTML.
      const forecastDisplay = validChargeOrigin ? this._escapeHtml(chargeOriginStr) : 'None';

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
      // QS-199 review-fix #04 CR1 — when disconnected the card is
      // pointer-disabled (`.disabled` CSS sets `pointer-events: none`),
      // so the custom `div role="button"` controls (sun / rabbit / time)
      // must ALSO drop out of the keyboard tab order and report
      // `aria-disabled`, otherwise they're dead keyboard tab stops whose
      // action just bails. Native `<button>`s (reset) handle this
      // themselves; only the custom divs need the explicit attrs.
      const ctrlTabAttrs = isDisconnected ? 'tabindex="-1" aria-disabled="true"' : 'tabindex="0"';
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

      // QS-235 AC2 — the bg-arc / progress-arc / dash-anim / handle are
      // built by the shared `_buildRingHTML`. The builder emits its own
      // <defs> for the green + charge gradients (via the standard
      // gradGreenId / gradRunningId params) plus a per-instance glow; the
      // car's disabled / fault / stale gradients + the soup clipPath are
      // injected via `extraDefs` (forward-referenced by the clip <g>
      // rendered just above ${ringMarkup} — valid SVG). The car follows
      // the radiator composition pattern: clip backdrop <g> + carves
      // first, then ${ringMarkup}.
      const extraDefs = `
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
                <clipPath id="${electronClipId}">
                  <circle cx="${CENTER_CX}" cy="${CENTER_CY}" r="${CLIP_R}" />
                </clipPath>`;
      const ringMarkup = this._buildRingHTML({
          palette: colors, ringCirc, center,
          progressPath: socPath, bgPath,
          handlePos, handlePct,
          showAnimation, canDragHandle: true,
          gradGreenId, gradRunningId: gradChargeId, activeGradId,
          dashLen, gapLen,
          // QS-235 — car-specific overrides over the duration defaults.
          // SF1 — the moving dash matches the static arc when faulted:
          // faulted-while-charging draws the dash with `gradFaultId`
          // (red, defined in extraDefs), else `gradChargeId` (cyan/blue,
          // defined by the builder via the `gradRunningId` param). This
          // restores the pre-refactor `stroke="url(#${isFaulted ?
          // gradFaultId : gradChargeId})"` exactly.
          animGradId: isFaulted ? gradFaultId : gradChargeId,
          bgStroke: isFaulted ? 'rgba(244,67,54,0.35)' : 'var(--divider-color)',
          handleFontSize: useEnergyMode ? '11' : '13',
          handleStroke: isDisconnected ? 'var(--divider-color)' : 'var(--primary-color)',
          handleFill: isDisconnected ? 'var(--secondary-text-color)' : 'var(--primary-color)',
          // Handle TEXT shows the TRUE (unclamped) target — energy or
          // percent — mirroring the duration-card round-trip invariant
          // (position is clamped via pctToDeg, label is not).
          handleLabel: useEnergyMode
              ? this._fmt(parseTargetEnergy(target) ?? (this._safeNumber(sCurrentInputedEnergy, 0) / 1000))
              : this._fmt(targetPct ?? soc),
          animPathId: 'charge_anim',
          extraDefs,
      });

      this._root.innerHTML = `
      <ha-card class="card ${isDisconnected ? 'disabled' : ''} ${isFaulted ? 'fault' : ''} ${isOffGrid ? 'off-grid' : ''} ${isStale ? 'stale' : ''}">
        <style>${css}</style>
        <div class="card-title">${this._escapeHtml(displayTitle)}</div>
        <div class="top"></div>

        <div class="hero">
          <div class="ring" title="${soc}%">
            <svg viewBox="0 0 320 320" width="300" height="300" style="touch-action: none;" aria-hidden="true">
              <g clip-path="url(#${electronClipId})" style="${degraded ? 'filter: saturate(0.3) brightness(0.7);' : ''}">
                <path id="electron_wave_idle" d="${initialWavePath}" fill="${IDLE_SOUP_COLOR}" opacity="${initialIdleOpacity}" pointer-events="none" style="will-change: transform;" />
                <path id="electron_wave_charge" d="${initialWavePath}" fill="${CHARGE_SOUP_COLOR}" opacity="${initialChargeOpacity}" pointer-events="none" style="will-change: transform;" />
                <g id="${sparkleLayerId}" pointer-events="none"></g>
                <g id="${lightningLayerId}" pointer-events="none" style="mix-blend-mode: screen; will-change: opacity;"></g>
              </g>
              ${this._ringCarveCover({ cx: RING_BOTTOM_CARVE_CX, cy: RING_BOTTOM_CARVE_CY, r: RING_BOTTOM_CARVE_R, id: 'sun_btn_cover', show: swPriority })}
              ${this._ringCarveCover({ cx: RABBIT_BTN_CARVE_CX, cy: RABBIT_BTN_CARVE_CY, r: RABBIT_BTN_CARVE_R, id: 'rabbit_btn_cover', show: e.force_now })}
              ${this._ringCarveCover({ cx: TIME_BTN_CARVE_CX, cy: TIME_BTN_CARVE_CY, r: TIME_BTN_CARVE_R, id: 'time_btn_cover', show: (tNext && e.schedule) })}
              ${ringMarkup}
            </svg>
            <div class="center">
              <div class="stack">
                <div class="soc-block">
                  ${chargeIcon ? `<ha-icon class="charge-type-icon" icon="${chargeIcon}"></ha-icon>` : ''}
                  <div class="pct${(!useEnergyMode && e.manual_soc) ? ' soc-editable' : ''}" id="soc_pct"${(!useEnergyMode && e.manual_soc) ? ` role="button" ${ctrlTabAttrs} aria-label="Edit charge percent"` : ''} style="margin-bottom:0;${(!useEnergyMode && e.manual_soc) ? ' cursor: pointer;' : ''}">${displaySocValue}</div>
                  ${useEnergyMode ? '' : `<div class="mini-range-now" aria-label="current range">${rangeNowStr}</div>`}
                </div>
                <div class="target-block">
                <div class="mini-grid">
                  <div class="mini-title">Force Now</div>
                  <div class="mini-title">${useEnergyMode ? 'Target Energy' : 'Target SOC'}</div>
                  <div class="mini-title">${chargeTimeLabel}</div>

                  ${e.force_now ? `<div id="rabbit_btn" class="rabbit-btn ${isAsFastState(sChargeType?.state) ? 'on' : ''}" role="button" ${ctrlTabAttrs} aria-label="Charge as fast as possible"><ha-icon icon="mdi:rabbit"></ha-icon></div>` : ''}
                  <div class="target-cell">
                    <div id="target_value" class="target-value">${displayTargetValue}</div>
                    ${useEnergyMode ? '' : `<div class="mini-range-target" aria-label="range at target">${rangeTargetStr}</div>`}
                  </div>
                  ${(tNext && e.schedule) ? `<div id="time_btn" class="time-btn ${chargeTime && chargeTime !== '--:--' ? 'on' : ''}" role="button" ${ctrlTabAttrs} aria-label="Set next charge time">${this._escapeHtml(chargeTime)}</div>` : ''}
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
            ${swPriority ? `<div id="sun_btn" class="sun-btn ${swPriority?.state === 'on' ? 'on' : ''}" role="button" ${ctrlTabAttrs} aria-label="Toggle solar priority"><ha-icon icon="mdi:weather-sunny"></ha-icon></div>` : ''}
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
        <div class="forecast-row">${forecastDisplay}</div>
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
              // QS-271 — defer re-render across the click→showPicker tap.
              this._armPressGuard(personPill);
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
              // QS-271 — defer re-render across the click→showPicker tap.
              this._armPressGuard(chargerPill);
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
              const hm = this._formatHm(mins);
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
      // QS-235 AC4 — the solar-priority (sun) button adopts the shared
      // `_wireGreenButton` (toggle the `bump_priority` switch with the
      // click + touchend + keyboard-activation plumbing). The dead
      // `ids('priority')` listener (no matching DOM element) is dropped.
      if (swPriority) {
          this._wireGreenButton({ buttonEl: ids('sun_btn'), swEntity: swPriority, entityId: e.bump_priority });
      }

      // Rabbit button for force now
      if (e.force_now) {
          const rbtn = ids('rabbit_btn');
          if (rbtn) {
              const rbtnAction = async () => {
                  if (this._root?.querySelector('.disabled')) return;

                  // Check if already in "As Fast As Possible" mode
                  const isAlreadyForcing = isAsFastState(sChargeType?.state);

                  if (isAlreadyForcing && e.clean_constraints) {
                      this._showDialog({
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
                      this._showDialog({
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
              // QS-271 — chokepoint. stopPropagation:true so the tap
              // doesn't bubble into the ring/center handlers; keyboard:true
              // for the focusable div (S5).
              this._wireTap(rbtn, rbtnAction, { keyboard: true, stopPropagation: true });
          }
      }

      // QS-235 AC5 — the finish-time button adopts the generalized shared
      // `_wireTimePicker`. The car-specific bits map onto its new optional
      // params: `onAfterCommit` (await `_press(schedule)` after the
      // `_setTime` write), `resetButton` (a 3rd dialog button → reset via
      // `clean_constraints`), and the `title` / `bodyText` overrides. The
      // dialog pre-selects the next charge-end time (preferring the live
      // `chargeTime`, falling back to `nextTimeMins`), rounded UP to the
      // nearest 5-minute select option — so the car PRE-rounds `currentMins`
      // (the base picker computes `currentMins % 60` against the 5-minute
      // option grid). Wiring is gated on `!isDisconnected`, preserving the
      // old `.disabled` bail (an unplugged car can't schedule a charge).
      if (tNext && e.schedule && !isDisconnected) {
          const rawFinishMins = (chargeTime && chargeTime !== '--:--' && chargeTime.includes(':'))
              ? this._parseTimeToMinutes(chargeTime)
              : nextTimeMins;
          let finishHour = Math.floor(rawFinishMins / 60);
          let finishMin = Math.ceil((rawFinishMins % 60) / 5) * 5;
          if (finishMin === 60) {
              finishMin = 0;
              finishHour = (finishHour + 1) % 24;
          }
          this._wireTimePicker({
              buttonEl: ids('time_btn'),
              entityId: e.next_time,
              currentMins: finishHour * 60 + finishMin,
              localStateKey: '_localNextTimeMins',
              clearTimerKey: '_localNextTimeClearTimer',
              onAfterCommit: () => this._press(e.schedule),
              resetButton: {
                  text: 'Reset',
                  variant: 'danger',
                  onClick: async () => { if (e.clean_constraints) await this._press(e.clean_constraints); },
              },
              title: 'Charge Finish Time',
              bodyText: 'Select the next time the charge of the car should end:',
          });
      }
      // QS-199 review-fix M3/M5 — the inline `showDialog` closure (which
      // lacked the N12 close-fallback, N13 try/finally, and S16 keyboard
      // activation) was deleted. All dialogs now route through the
      // hardened `this._showDialog(...)` on QsCardBase.

      if (e.force_now) {
          const forceBtn = ids('force');
          const forceAction = async () => {
              if (this._root?.querySelector('.disabled')) return;
              this._showDialog({
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
          // QS-271 — chokepoint. keyboard:false (matches prior behaviour).
          this._wireTap(forceBtn, forceAction, { keyboard: false, stopPropagation: true });
      }

      if (e.schedule) {
          const schedBtn = ids('schedule_inline');
          const schedAction = async () => {
              if (this._root?.querySelector('.disabled')) return;
              const limit = selLimit?.state || '';
              const time = tNext?.state || '';
              this._showDialog({
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
          // QS-271 — chokepoint. keyboard:false (matches prior behaviour).
          this._wireTap(schedBtn, schedAction, { keyboard: false, stopPropagation: true });
      }

      // QS-243 — the big SOC number opens a manual-charge popup with an
      // integer 0–100 input and Save / Cancel / Reset actions. Save writes
      // the manual-SOC number entity; Reset presses the reset button.
      if (!useEnergyMode && e.manual_soc) {
          const socEl = ids('soc_pct');
          const openSocDialog = () => {
              if (this._root?.querySelector('.disabled')) return;
              const manualRaw = sManualSoc?.state;
              const cur = isNumberLike(manualRaw)
                  ? Math.round(Number(manualRaw))
                  : (isNumberLike(sSoc?.state) ? Math.round(Number(sSoc.state)) : 0);
              const content = `<div class="qs-soc-edit" style="text-align:center; padding: 8px 0;">`
                  + `<input id="qs_soc_input" type="number" min="0" max="100" step="1" inputmode="numeric" value="${cur}" `
                  + `style="font-size:1.4rem; width:96px; text-align:center; padding:6px;" /> %</div>`;
              const buttons = [
                  {
                      text: 'Save', variant: 'primary', onClick: async () => {
                          const inp = this._root?.querySelector('#qs_soc_input');
                          let v = inp ? Number(inp.value) : NaN;
                          // QS-243 N1 — never silently discard: invalid / non-finite
                          // input falls back to the prefilled current value and is
                          // still clamped + saved (the dialog always commits a sane
                          // value rather than closing with no save).
                          if (!Number.isFinite(v)) v = cur;
                          v = Math.max(0, Math.min(100, Math.round(v)));
                          await this._setNumber(e.manual_soc, v);
                      }
                  },
                  { text: 'Cancel', variant: 'secondary' },
              ];
              if (e.reset_soc) {
                  buttons.push({
                      text: 'Reset', variant: 'secondary', onClick: async () => {
                          await this._press(e.reset_soc);
                      }
                  });
              }
              // `_showDialog`'s per-button `activate()` removes the modal in a
              // finally block after `onClick` resolves, so Save / Cancel / Reset
              // all dismiss the popup — no explicit close needed here (N6).
              this._showDialog({ title: 'Set charge percent', customContent: content, buttons });
          };
          // QS-271 — chokepoint. keyboard:true, stopPropagation:true.
          this._wireTap(socEl, openSocDialog, { keyboard: true, stopPropagation: true });
      }

      // QS-235 AC4 — the reset button adopts the shared `_wireResetButton`
      // (confirmation dialog → `_press(reset)`). The native `<button
      // id="reset">` is keyboard-native, so the helper intentionally
      // does not register an extra Enter/Space hook.
      if (e.reset) {
          this._wireResetButton({ buttonEl: ids('reset'), entityId: e.reset });
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

      // QS-235 AC1 — the ~120-LOC inline pointer-drag block is replaced by
      // the shared `_wireTargetHandle` (byte-identical pointer/mouse/touch
      // plumbing + the S17 try/finally the car previously lacked). The car
      // layers four callbacks over that plumbing:
      //   - pctToValue / valueToPct — the %↔kWh mapping (branch on
      //     `useEnergyMode` against `maxCircleValue`),
      //   - onDragMove — the live `#target_value` label update,
      //   - onCommit — map the snapped value → a select option
      //     (`findOptionByEnergy` / `findOptionByPercent`) then
      //     `_select(next_limit, opt)`,
      //   - fmtHandleText — keep the handle text rounded (the car shows
      //     integers; the duration default shows one decimal).
      const svg = this._root.querySelector('.ring svg');
      const handle = this._root.getElementById('target_handle');
      this._wireTargetHandle({
          ringSvg: svg,
          handle,
          center,
          ringCirc,
          startDeg, endDeg, rangeDeg,
          allowedValues: allowedOrDefault,
          pctToValue: (rawPct) => (useEnergyMode ? (rawPct / 100 * maxCircleValue) : rawPct),
          valueToPct: (v) => (useEnergyMode ? (v / maxCircleValue * 100) : v),
          fmtHandleText: (v) => this._fmt(v),
          onDragMove: (v) => {
              const tv = this._root.getElementById('target_value');
              if (tv) tv.innerHTML = useEnergyMode ? `${this._fmt(v)}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>` : `${this._fmt(v)}%`;
          },
          onCommit: async (v) => {
              const opt = useEnergyMode ? findOptionByEnergy(v) : findOptionByPercent(v);
              if (opt) await this._select(e.next_limit, opt);
          },
      });
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
