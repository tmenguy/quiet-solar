/*
  QS Car Card - custom:qs-car-card
  Zero-build single-file Lit-style web component compatible with Home Assistant
*/

const INVALID_STATES = ['unavailable', 'unknown', 'none'];

const ANIM_MIN_SPEED = 20;   // dash-units per second at minimum power
const ANIM_MAX_SPEED = 200;  // dash-units per second at maximum power
const ANIM_MIN_POWER_W = 500;
const ANIM_MAX_POWER_W = 22000;
const ANIM_SPEED_RANGE = ANIM_MAX_SPEED - ANIM_MIN_SPEED;
const ANIM_POWER_RANGE = ANIM_MAX_POWER_W - ANIM_MIN_POWER_W;

// --- QS-224 battery animation constants -----------------------------------
// Battery shape derived from mdi:battery-outline (24×24 mdi viewBox).
// The icon is body 12×18 at (6,4)..(18,22) plus terminal nub 6×2 at
// (9,2)..(15,4), with corner radius 2 mdi-units (~2/24 ≈ 8.3% of side).
// The user pins vertical span at "+1/5 / −1/5 of diameter" = 60% of
// the ring's visible diameter (ringCirc * 2 = 260 SVG-px) — total
// height 156 SVG-px split 18:2 between body and terminal nub.

const CENTER_CX = 160;                                       // SVG x-center
const CENTER_CY = 160;                                       // SVG y-center
const BATTERY_TOTAL_HEIGHT = 156;                            // = 0.6 × 2 × ringCirc (130), per issue's "+1/5 / −1/5" rule
const MDI_UNIT_PX = BATTERY_TOTAL_HEIGHT / 20;               // = 7.8 (20 mdi-units total height: body 18 + terminal 2)
const BATTERY_BODY_HEIGHT = 18 * MDI_UNIT_PX;                // = 140.4
const BATTERY_BODY_WIDTH  = 12 * MDI_UNIT_PX;                // = 93.6   (body 12:18 ratio preserved)
const BATTERY_TERMINAL_WIDTH  = 6 * MDI_UNIT_PX;             // = 46.8
const BATTERY_TERMINAL_HEIGHT = 2 * MDI_UNIT_PX;             // = 15.6
const BATTERY_CORNER_R = 2 * MDI_UNIT_PX;                    // = 15.6
const BATTERY_TOP_Y       = CENTER_CY - BATTERY_TOTAL_HEIGHT / 2;       // = 82
const BATTERY_BOTTOM_Y    = CENTER_CY + BATTERY_TOTAL_HEIGHT / 2;       // = 238
const BATTERY_BODY_TOP_Y  = BATTERY_TOP_Y + BATTERY_TERMINAL_HEIGHT;    // = 97.6
const BATTERY_BODY_LEFT_X  = CENTER_CX - BATTERY_BODY_WIDTH / 2;        // = 113.2
const BATTERY_BODY_RIGHT_X = CENTER_CX + BATTERY_BODY_WIDTH / 2;        // = 206.8
const BATTERY_TERMINAL_LEFT_X  = CENTER_CX - BATTERY_TERMINAL_WIDTH / 2; // = 136.6
const BATTERY_TERMINAL_RIGHT_X = CENTER_CX + BATTERY_TERMINAL_WIDTH / 2; // = 183.4

// Wave geometry — reuse the pool/boiler defaults verbatim. The wave path
// is rendered with extent [0, 2*WAVE_WIDTH] and translated; absolute
// width is decoupled from battery body width.
const WAVE_WIDTH = 480;                                      // single wave period; same as pool/boiler
const WAVE_BOTTOM_Y = 400;                                   // closing rect y; below viewBox (320)
const LAYER_SCROLL_OFFSET = 1.2;                             // per-layer extra scroll phase
const LAYER_PHASE_OFFSET  = 2.1;                             // per-layer static sine phase

// Water palettes — opacity cross-fade between idle/green and charging/blue.
// Review-fix #01 FX-01 + FX-02: collapsed from 3 layers to 1 with a
// BRIGHT (lightness ≥ 85%) + VERY TRANSLUCENT (alpha ≤ 0.35) palette.
// The original mid-saturation mid-lightness 3-layer stack read as a
// dark muddy green and obscured the SOC arc; the user wants the
// water to read as "fresh / lively" and the existing ring + SOC
// arc to remain clearly visible THROUGH the water. The single-layer
// design avoids the multi-path overlay density that defeated the
// "very translucent" goal. Loop counts in `_render()` and
// `_startAnimation()` are driven by `IDLE_WATER_COLORS.length` so
// adding entries back later is a one-line change.
const IDLE_WATER_COLORS = [
  'hsla(135, 95%, 88%, 0.30)',   // bright vibrant green, very translucent
];
const CHARGING_WATER_COLORS = [
  'hsla(205, 95%, 88%, 0.32)',   // bright sky-blue, slightly more present while charging
];

// Animation tuning — mirror the boiler exactly (gentle calm, more
// pronounced when active). Documented at QS-200 in
// docs/agents/concepts/dashboard-and-cards.md.
const LERP_RATE = 2;                                         // exp lerp time-constant (~95% in ~1.5s)
const LERP_DT_CEIL = 0.1;                                    // s; clamp dt against hidden-tab return
const AMP_REGEN_THRESHOLD = 0.25;                            // amp delta for path regen
const LEVEL_REGEN_THRESHOLD = 0.01;                          // px; water level delta for path regen
const PHASE_WRAP = 1e6;                                      // wrap _wavePhase to preserve float precision
const PHASE_TO_PX = 60;                                      // scroll px per phase-unit
const CALM_AMP = 1.5;                                        // = boiler CALM_AMP
const CHARGING_AMP = 8;                                      // = boiler BOIL_AMP
const CALM_SPEED = 0.1;                                      // = boiler CALM_SPEED
const CHARGING_SPEED = 0.4;                                  // = boiler BOIL_SPEED

// Lightning subsystem — charging-only particle system, HARD CAP at
// MAX_CONCURRENT_LIGHTNING (spawn rejected when len >= cap, no overflow).
// Architecture mirrors the boiler bubble/steam subsystems: create/destroy
// (NOT pool/reuse) — matches project precedent at qs-water-boiler-card.js
// :413-461. Spawn rate × life ≈ 1.5 avg, occasional 3-cap.
const MAX_CONCURRENT_LIGHTNING = 3;
const LIGHTNING_SPAWN_RATE_HZ = 3;
const LIGHTNING_LIFE_S = 0.5;
const LIGHTNING_HEIGHT_MIN_PX = 18;
const LIGHTNING_HEIGHT_MAX_PX = 30;
const LIGHTNING_ZIGZAG_AMPLITUDE_PX = 6;                     // horizontal jitter per segment
const LIGHTNING_SEGMENTS = 4;                                // zigzag segments per bolt (5 vertices)
const LIGHTNING_STROKE_WIDTH = 1.5;
const LIGHTNING_STROKE_COLOR = '#FFFFFF';                    // white core
const LIGHTNING_GLOW_COLOR = '#80D4FF';                      // blue-white halo
const LIGHTNING_GLOW_STDDEV = 2.5;
// Life curve breakpoints (piecewise linear). Mirrors boiler steam at
// qs-water-boiler-card.js:573-577 (0.15/0.85 → here 0.2/0.7).
const LIGHTNING_FADE_IN_FRACTION = 0.2;                      // 0 → 0.2 fade-in
const LIGHTNING_FADE_OUT_FRACTION = 0.7;                     // 0.2 → 0.7 hold, 0.7 → 1.0 fade-out

// Battery outline appearance — alpha lerped between idle/charging via
// JS rgba interpolation (NOT CSS color-mix, for browser compat).
// Review-fix #01 FX-03: the user reported the outline was invisible
// against the ring and the (post-FX-01 brighter) water. Three knobs
// were tuned together:
//   1. STROKE color swapped from near-white `rgba(255,255,255,…)` to
//      a neutral grey `rgba(180,180,180,…)` so the silhouette reads
//      against both the ring and the bright translucent water.
//   2. STROKE_WIDTH widened from a literal `2` to `MDI_UNIT_PX`
//      (= 1 mdi-unit ≈ 7.8 px) — the canonical mdi:battery-outline
//      stroke ratio (1/12 of body width).
//   3. ALPHA range bumped from `0.30 / 0.55` to `0.55 / 0.80` so the
//      wider grey stroke reads at idle and pops while charging.
const BATTERY_OUTLINE_STROKE_IDLE_ALPHA = 0.55;
const BATTERY_OUTLINE_STROKE_CHARGING_ALPHA = 0.80;
const BATTERY_OUTLINE_STROKE_WIDTH = MDI_UNIT_PX;

class QsCarCard extends HTMLElement {
  constructor() {
    super();
    this._chargePower = 0;
    this._charging = false;
  }

  // QS-224: SVG path `d` for the body-only rounded-rectangle (terminal
  // NOT included). Used as the <clipPath> contents — water and
  // lightning are clipped to the body interior so the terminal nub
  // never shows water bleed or stray bolts. Trace the rounded-rect
  // clockwise from top-left-after-corner.
  _generateBatteryBodyClipPath() {
    const r = BATTERY_CORNER_R;
    const left = BATTERY_BODY_LEFT_X;
    const right = BATTERY_BODY_RIGHT_X;
    const top = BATTERY_BODY_TOP_Y;
    const bottom = BATTERY_BOTTOM_Y;
    return (
      `M ${(left + r).toFixed(2)} ${top.toFixed(2)} ` +
      `H ${(right - r).toFixed(2)} ` +
      `Q ${right.toFixed(2)} ${top.toFixed(2)} ${right.toFixed(2)} ${(top + r).toFixed(2)} ` +
      `V ${(bottom - r).toFixed(2)} ` +
      `Q ${right.toFixed(2)} ${bottom.toFixed(2)} ${(right - r).toFixed(2)} ${bottom.toFixed(2)} ` +
      `H ${(left + r).toFixed(2)} ` +
      `Q ${left.toFixed(2)} ${bottom.toFixed(2)} ${left.toFixed(2)} ${(bottom - r).toFixed(2)} ` +
      `V ${(top + r).toFixed(2)} ` +
      `Q ${left.toFixed(2)} ${top.toFixed(2)} ${(left + r).toFixed(2)} ${top.toFixed(2)} Z`
    );
  }

  // QS-224: SVG path `d` for the COMBINED body+terminal silhouette
  // (single closed path, no seam). Drawn as a stroked outline only
  // (fill="none"). The path traces the mdi:battery-outline shape:
  // terminal-nub on top (small rectangle) seamlessly joined to the
  // rounded-rect body below.
  _generateBatteryOutlinePath() {
    const r = BATTERY_CORNER_R;
    const left = BATTERY_BODY_LEFT_X;
    const right = BATTERY_BODY_RIGHT_X;
    const top = BATTERY_BODY_TOP_Y;
    const bottom = BATTERY_BOTTOM_Y;
    const termLeft = BATTERY_TERMINAL_LEFT_X;
    const termRight = BATTERY_TERMINAL_RIGHT_X;
    const termTop = BATTERY_TOP_Y;
    // Trace clockwise from body-top-left-after-corner:
    //   1. across body top to where the terminal joins (termLeft)
    //   2. up the terminal left edge
    //   3. across the terminal top
    //   4. down the terminal right edge
    //   5. across body top from termRight back to body-top-right-before-corner
    //   6. arc the top-right body corner
    //   7. down the right body edge
    //   8. arc the bottom-right corner
    //   9. across the body bottom
    //  10. arc the bottom-left corner
    //  11. up the left body edge
    //  12. arc the top-left body corner, close
    return (
      `M ${(left + r).toFixed(2)} ${top.toFixed(2)} ` +
      `H ${termLeft.toFixed(2)} ` +
      `V ${termTop.toFixed(2)} ` +
      `H ${termRight.toFixed(2)} ` +
      `V ${top.toFixed(2)} ` +
      `H ${(right - r).toFixed(2)} ` +
      `Q ${right.toFixed(2)} ${top.toFixed(2)} ${right.toFixed(2)} ${(top + r).toFixed(2)} ` +
      `V ${(bottom - r).toFixed(2)} ` +
      `Q ${right.toFixed(2)} ${bottom.toFixed(2)} ${(right - r).toFixed(2)} ${bottom.toFixed(2)} ` +
      `H ${(left + r).toFixed(2)} ` +
      `Q ${left.toFixed(2)} ${bottom.toFixed(2)} ${left.toFixed(2)} ${(bottom - r).toFixed(2)} ` +
      `V ${(top + r).toFixed(2)} ` +
      `Q ${left.toFixed(2)} ${top.toFixed(2)} ${(left + r).toFixed(2)} ${top.toFixed(2)} Z`
    );
  }

  // QS-224: ported verbatim from qs-pool-card.js:49-62. Emits TWO
  // repetitions of the wave (path extent = [0, 2*width]) so the
  // translated path always covers the clip region regardless of
  // scroll offset.
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

  // QS-224: reset cached DOM refs and the logical lightning array.
  // Shared between `_invalidateWaveCache()` (full memo reset on
  // (re-)connect) and the post-`innerHTML` cleanup block in
  // `_render()`. Mirrors qs-water-boiler-card.js:192-199.
  _resetDomRefs() {
    this._waveEls = null;
    this._lightningLayerEl = null;
    this._batteryOutlineEl = null;
    this._lightning = [];
  }

  // QS-224: clear the wave-path memoization keys and cached DOM refs.
  // Called on every (re-)connect and after each _render() innerHTML
  // rewrite. Mirrors qs-water-boiler-card.js:209-213.
  _invalidateWaveCache() {
    this._lastWaterBaseY = null;
    this._lastAmplitude = null;
    this._resetDomRefs();
  }

  // QS-224: continuous RAF while connected, mirroring qs-water-boiler-card.js.
  // The battery+water animation is intrinsically visible at all times
  // (calm green when not charging, electric blue with lightning when
  // charging) so RAF is no longer gated on `_charging`. Calm vs charging
  // is amplitude / speed / color-mix lerp, not RAF on/off.
  _startAnimation() {
    if (this._animRaf != null) return;
    // Initialize wave animation state ONLY on the first-ever connect
    // so that detach/re-attach preserves _currentAmplitude /
    // _currentSpeed / _wavePhase. Without this guard, a charging wave
    // would visibly snap back to CALM on each reconnect and re-lerp
    // up over ~1.5s. Mirrors qs-water-boiler-card.js:227-242.
    if (this._currentAmplitude == null) {
      this._currentAmplitude = CALM_AMP;
      this._currentSpeed = CALM_SPEED;
      this._wavePhase = 0;
      this._currentColorMix = 0;
      this._lightning = [];
      this._nextLightningAt = 0;
      this._needsAnimationPrime = true;
    }
    this._lastAnimTs = null;
    this._invalidateWaveCache();

    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      // Cap dt against hidden-tab return (S6 pattern from boiler /
      // pool — bounds phase advance, lerp envelope, and lightning life
      // by the same upper envelope).
      let dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      dt = Math.min(dt, LERP_DT_CEIL);
      this._lastAnimTs = ts;

      // --- Existing dashed-arc animation (preserved verbatim).
      // The charging-progress dash still drives off `showAnimation`
      // (applied as a render-time switch on `<path id="charge_anim">`).
      // The RAF loop itself no longer gates on `_charging`.
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const cp = this._chargePower || 0;
      const speed = Math.min(ANIM_MAX_SPEED, Math.max(ANIM_MIN_SPEED, ANIM_MIN_SPEED + (cp - ANIM_MIN_POWER_W) * ANIM_SPEED_RANGE / ANIM_POWER_RANGE));
      this._animOffset = ((this._animOffset || 0) + speed * dt) % patternLen;
      const dashEl = this._root?.getElementById('charge_anim');
      if (dashEl) {
        dashEl.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }

      // --- Lerp amplitude / speed / colorMix toward charging targets.
      const charging = this._charging === true;
      const targetAmplitude = charging ? CHARGING_AMP : CALM_AMP;
      const targetSpeed     = charging ? CHARGING_SPEED : CALM_SPEED;
      const targetColorMix  = charging ? 1 : 0;
      const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
      this._currentAmplitude += (targetAmplitude - this._currentAmplitude) * lerpFactor;
      this._currentSpeed     += (targetSpeed     - this._currentSpeed)     * lerpFactor;
      this._currentColorMix  += (targetColorMix  - this._currentColorMix)  * lerpFactor;
      this._wavePhase += this._currentSpeed * dt;
      this._wavePhase = ((this._wavePhase % PHASE_WRAP) + PHASE_WRAP) % PHASE_WRAP;

      // Lazy-resolve wave / lightning layer / outline DOM refs.
      // Review-fix #01 FX-07: the resolve guard checks `null` on
      // EACH cached entry (not just the array's existence) — if the
      // first RAF tick lands before `_render()` populates the
      // shadow root, every `getElementById` returns `null` and the
      // resulting all-nulls array is truthy, which would lock the
      // cache permanently. Re-querying when any entry is null
      // covers the eager-RAF case (and is idempotent once the refs
      // are populated). The layer-count loop is driven by
      // `IDLE_WATER_COLORS.length` per FX-02.
      const waveLayerCount = IDLE_WATER_COLORS.length;
      if (!this._waveEls || this._waveEls.some(el => el == null)) {
        const refs = [];
        for (let i = 0; i < waveLayerCount; i++) {
          refs.push(this._root?.getElementById(`wave${i}_idle`) ?? null);
          refs.push(this._root?.getElementById(`wave${i}_charge`) ?? null);
        }
        this._waveEls = refs;
      }
      // Same null-recheck idiom for `_lightningLayerEl` and
      // `_batteryOutlineEl` — the prior `el ?? (el = …)` pattern
      // re-ran `getElementById` every frame when the result was
      // `null` (assignment-as-cache short-circuits to the truthy
      // result only). The explicit null check is clearer.
      if (this._lightningLayerEl == null && this._lightningLayerId) {
        this._lightningLayerEl =
          this._root?.getElementById(this._lightningLayerId) ?? null;
      }
      const lightningLayer = this._lightningLayerEl;
      if (this._batteryOutlineEl == null) {
        this._batteryOutlineEl =
          this._root?.getElementById('battery_outline') ?? null;
      }
      const outlineEl = this._batteryOutlineEl;

      // --- Wave transforms (per-layer translateX).
      // Loop bound from `waveLayerCount = IDLE_WATER_COLORS.length`
      // (FX-02) — same idiom for the regen and opacity loops below.
      for (let i = 0; i < waveLayerCount; i++) {
        const phaseOffset = i * LAYER_SCROLL_OFFSET;
        const raw = (this._wavePhase + phaseOffset) * PHASE_TO_PX;
        const scrollOffset = ((raw % WAVE_WIDTH) + WAVE_WIDTH) % WAVE_WIDTH;
        // Align path start with the battery body left edge, then
        // scroll within one period. The path extent is [0, 2*WAVE_WIDTH]
        // so any scroll value still covers the visible body.
        const tx = BATTERY_BODY_LEFT_X - WAVE_WIDTH - scrollOffset;
        const txStr = `translateX(${tx.toFixed(1)}px)`;
        const idleEl   = this._waveEls[i * 2];
        const chargeEl = this._waveEls[i * 2 + 1];
        if (idleEl)   idleEl.style.transform = txStr;
        if (chargeEl) chargeEl.style.transform = txStr;
      }

      // --- Wave path regen (throttled by amplitude / water-level delta).
      const waterBaseY = this._waterBaseY;
      const hasValidBase = waterBaseY != null && !Number.isNaN(waterBaseY);
      const ampDelta = this._lastAmplitude == null
          ? Infinity
          : Math.abs(this._currentAmplitude - this._lastAmplitude);
      const levelChanged = hasValidBase &&
          Math.abs(waterBaseY - (this._lastWaterBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
      if (hasValidBase && (levelChanged || ampDelta > AMP_REGEN_THRESHOLD)) {
        this._lastWaterBaseY = waterBaseY;
        this._lastAmplitude = this._currentAmplitude;
        for (let i = 0; i < waveLayerCount; i++) {
          const phaseOffset = i * LAYER_PHASE_OFFSET;
          const freq = 2 + i;
          const d = this._generateWavePath(WAVE_WIDTH, this._currentAmplitude, freq, phaseOffset, waterBaseY);
          const idleEl   = this._waveEls[i * 2];
          const chargeEl = this._waveEls[i * 2 + 1];
          if (idleEl)   idleEl.setAttribute('d', d);
          if (chargeEl) chargeEl.setAttribute('d', d);
        }
      }

      // --- Per-frame wave opacity update (idle ↔ charge cross-fade).
      // Per-PATH opacity (NOT group opacity on parent <g>) per AC-3.
      const idleOpacity   = (1 - this._currentColorMix).toFixed(3);
      const chargeOpacity = this._currentColorMix.toFixed(3);
      for (let i = 0; i < waveLayerCount; i++) {
        const idleEl   = this._waveEls[i * 2];
        const chargeEl = this._waveEls[i * 2 + 1];
        if (idleEl)   idleEl.setAttribute('opacity', idleOpacity);
        if (chargeEl) chargeEl.setAttribute('opacity', chargeOpacity);
      }

      // --- Battery outline alpha lerp (idle 0.55 → charging 0.80).
      // Review-fix #01 FX-03: stroke RGB is `180,180,180` (neutral
      // grey) — was `255,255,255` (white) — and the alpha range is
      // wider so the (also-widened) stroke is always readable.
      // JS rgba interpolation (NOT CSS color-mix, for browser compat).
      if (outlineEl) {
        const outlineAlpha = BATTERY_OUTLINE_STROKE_IDLE_ALPHA
          + this._currentColorMix
            * (BATTERY_OUTLINE_STROKE_CHARGING_ALPHA - BATTERY_OUTLINE_STROKE_IDLE_ALPHA);
        outlineEl.setAttribute('stroke', `rgba(180,180,180,${outlineAlpha.toFixed(3)})`);
      }

      // === QS-224 lightning particle system: charging-only spawn,
      // HARD CAP at MAX_CONCURRENT_LIGHTNING. Advance + retire runs
      // unconditionally so bolts in flight when charging→false
      // continue their life curve and retire naturally.
      if (lightningLayer) {
        if (charging) {
          this._nextLightningAt -= dt;
          while (this._nextLightningAt <= 0
                 && this._lightning.length < MAX_CONCURRENT_LIGHTNING) {
            const h = LIGHTNING_HEIGHT_MIN_PX
                    + Math.random() * (LIGHTNING_HEIGHT_MAX_PX - LIGHTNING_HEIGHT_MIN_PX);
            const cxStart = BATTERY_BODY_LEFT_X + 10
                          + Math.random() * (BATTERY_BODY_WIDTH - 20);
            const cyStart = BATTERY_BODY_TOP_Y + 5
                          + Math.random() * (BATTERY_BODY_HEIGHT - h - 10);
            // Build zigzag points: 5 vertices, alternating
            // ±LIGHTNING_ZIGZAG_AMPLITUDE_PX.
            const points = [];
            for (let i = 0; i <= LIGHTNING_SEGMENTS; i++) {
              const t = i / LIGHTNING_SEGMENTS;
              const dx = (i % 2 === 0 ? 1 : -1) * LIGHTNING_ZIGZAG_AMPLITUDE_PX;
              points.push(`${(cxStart + dx).toFixed(2)},${(cyStart + t * h).toFixed(2)}`);
            }
            // createElementNS REQUIRED for SVG inside a Shadow DOM
            // (matches boiler bubble/steam precedent).
            const el = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            el.setAttribute('points', points.join(' '));
            el.setAttribute('fill', 'none');
            el.setAttribute('stroke', LIGHTNING_STROKE_COLOR);
            el.setAttribute('stroke-width', String(LIGHTNING_STROKE_WIDTH));
            el.setAttribute('stroke-linejoin', 'round');
            el.setAttribute('stroke-linecap', 'round');
            el.setAttribute('pointer-events', 'none');
            el.setAttribute('opacity', '0');
            lightningLayer.appendChild(el);
            this._lightning.push({ el, life: 0, maxLife: LIGHTNING_LIFE_S });
            this._nextLightningAt += 1 / LIGHTNING_SPAWN_RATE_HZ;
          }
          // Review-fix #01 FX-06: ONLY clamp when not capped, so the
          // accumulated spawn debt is preserved across cap-blocked
          // frames and a freed slot is filled immediately. Previously
          // the unconditional clamp discarded the debt when the cap
          // was hit, making the next post-retire spawn wait the full
          // `1/SPAWN_RATE_HZ` window.
          if (this._lightning.length < MAX_CONCURRENT_LIGHTNING
              && this._nextLightningAt < 0) {
            this._nextLightningAt = 0;
          }
        }

        // Advance + retire — graceful exit on charging→false.
        const aliveLightning = [];
        for (const b of this._lightning) {
          b.life += dt;
          if (b.life >= b.maxLife) { b.el.remove(); continue; }
          const lifeT = b.life / b.maxLife;
          let lifeOpacity;
          if (lifeT < LIGHTNING_FADE_IN_FRACTION) {
            lifeOpacity = lifeT / LIGHTNING_FADE_IN_FRACTION;
          } else if (lifeT < LIGHTNING_FADE_OUT_FRACTION) {
            lifeOpacity = 1;
          } else {
            lifeOpacity = Math.max(
              0,
              1 - (lifeT - LIGHTNING_FADE_OUT_FRACTION) / (1 - LIGHTNING_FADE_OUT_FRACTION),
            );
          }
          b.el.setAttribute('opacity', (lifeOpacity * this._currentColorMix).toFixed(3));
          aliveLightning.push(b);
        }
        this._lightning = aliveLightning;
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
    // QS-224: continuous RAF while connected, mirroring
    // qs-water-boiler-card.js / qs-pool-card.js. The battery+water
    // animation is intrinsically visible at all times (calm green when
    // not charging, electric blue with lightning when charging) so RAF
    // is no longer gated on `_charging`. See AC-7 in
    // docs/stories/QS-224.story.md and the QS-224 paragraph in
    // docs/agents/concepts/dashboard-and-cards.md.
    this._startAnimation();
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
    // QS-224: eagerly tear down lightning DOM nodes. Mirrors
    // qs-water-boiler-card.js:637-644.
    this._lightning?.forEach(b => b.el?.remove?.());
    this._lightning = [];
  }
  static getStubConfig() {
    return { name: "QS Car", entities: {} };
  }

  setConfig(config) {
    if (!config || !config.entities) throw new Error("entities is required");
    this._config = config;
    this._root = this.attachShadow({ mode: "open" });
    this._render();
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
      // QS-224: continuous RAF while connected — the battery+water
      // animation must keep ticking gently (calm green waves) even
      // when idle, and the charging state is reflected via
      // amplitude/speed/colorMix lerp inside the RAF step function.
      // The RAF loop is started in `connectedCallback()`.
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
      :host { --pad: 18px; display:block; }
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
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; }
      .ring ha-icon { --mdc-icon-size: 32px; color: var(--secondary-text-color); margin-bottom: 6px; }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; text-align:center; width: 180px; margin: 0 auto; }
      .ring .soc-block { display:flex; flex-direction:column; align-items:center; gap:2px; margin-top:4px; margin-bottom:8px; }
      .ring .soc-block .charge-type-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); margin-bottom: 2px; }
      .ring .stack > * { text-align:center; }
      .ring .mini-grid { display:grid; grid-template-columns: repeat(3, 60px); grid-auto-rows: auto; width:180px; margin: 0 auto; justify-items:center; align-items:center; row-gap:4px; column-gap:0; }
      .ring .mini-grid.extra { row-gap:0; margin-top:2px; margin-bottom:6px; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:0; }
      .ring .target-cell { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .mini-title { color: var(--secondary-text-color); font-weight:700; font-size: .7rem; letter-spacing:.2px; white-space: nowrap; }
      .ring .mini-value { color: var(--primary-text-color); font-weight:800; font-size: .95rem; line-height: 1.1; white-space: pre-line; }
      .ring .mini-icon { --mdc-icon-size: 18px; color: var(--primary-text-color); }
      .ring .mini-range { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .mini-range-now { color: var(--primary-text-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; transform: translateY(-8px); }
      .ring .mini-range-target { color: var(--primary-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; margin-bottom:0; }
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
      .ring .sun-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .sun-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; transform: translateY(3px); }
      .ring .sun-btn.on { border-color: rgba(255,202,40,.45); background: rgba(255,202,40,.14); box-shadow: 0 0 0 3px rgba(255,202,40,.20), 0 0 16px #FFCA28; }
      .ring .sun-btn.on ha-icon { color: #FFCA28; }
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

      // === QS-224 battery+water layer ============================
      // Water level from SOC (same value the SOC arc uses — no
      // raw-SOC bypass of the stale-percent override). The issue's
      // literal "+1/5 / −1/5 of diameter" rule maps to the pool's
      // `(0.2 + ratio × 0.6)` formula applied to the battery body.
      const progressRatio = Math.max(0, Math.min(1, soc / 100));
      const waterBaseY = BATTERY_BOTTOM_Y
                       - (0.2 + progressRatio * 0.6) * BATTERY_BODY_HEIGHT;
      this._waterBaseY = waterBaseY;

      // QS-224: prime the wave animation state to the actual charging
      // targets on the first render after connect, avoiding the
      // ~1.5s boot transient. Mirror qs-water-boiler-card.js:1072-1077.
      // Review-fix #01 FX-05: also prime when `_currentAmplitude` is
      // still uninitialised — the very first `_render` is called by
      // `setConfig` BEFORE `connectedCallback → _startAnimation`
      // initialises any animation state, so `_needsAnimationPrime`
      // is still `undefined` on that path. Without the
      // `|| _currentAmplitude == null` clause, an actively-charging
      // car shows calm green on its very first paint and only the
      // second `_render` snaps to charging.
      if (this._needsAnimationPrime || this._currentAmplitude == null) {
        this._currentAmplitude = charging ? CHARGING_AMP : CALM_AMP;
        this._currentSpeed     = charging ? CHARGING_SPEED : CALM_SPEED;
        this._currentColorMix  = charging ? 1 : 0;
        this._needsAnimationPrime = false;
      }

      // QS-224: pre-generate the initial wave path `d` strings so
      // the SVG renders with water immediately, avoiding an empty-`d`
      // flash between the innerHTML rewrite and the first RAF tick.
      // The idle/charge siblings of each layer share the same `d`.
      // Review-fix #01 FX-02: layer count driven by
      // `IDLE_WATER_COLORS.length` (= 1 today) so a future palette
      // tweak that re-adds layers doesn't require touching the loop.
      const initialAmp = this._currentAmplitude ?? CALM_AMP;
      const initialColorMix = this._currentColorMix ?? 0;
      const initialWavePaths = IDLE_WATER_COLORS.map((_color, i) => {
        const freq = 2 + i;
        const phaseOffset = i * LAYER_PHASE_OFFSET;
        return this._generateWavePath(WAVE_WIDTH, initialAmp, freq, phaseOffset, waterBaseY);
      });
      const initialIdleOpacity   = (1 - initialColorMix).toFixed(3);
      const initialChargeOpacity = initialColorMix.toFixed(3);
      const initialOutlineStrokeAlpha = BATTERY_OUTLINE_STROKE_IDLE_ALPHA
        + initialColorMix
          * (BATTERY_OUTLINE_STROKE_CHARGING_ALPHA - BATTERY_OUTLINE_STROKE_IDLE_ALPHA);

      // Per-instance unique SVG ids (mirror gradGreenId / gradChargeId
      // pattern at lines 884-888) so two car cards on the same
      // dashboard don't collide on clipPath / filter / layer ids.
      const batteryClipId    = `batClip-${Math.floor(Math.random() * 1e6)}`;
      const lightningGlowId  = `lightG-${Math.floor(Math.random() * 1e6)}`;
      const lightningLayerId = `lightL-${Math.floor(Math.random() * 1e6)}`;
      this._lightningLayerId = lightningLayerId;
      const batteryBodyClipD = this._generateBatteryBodyClipPath();
      const batteryOutlineD  = this._generateBatteryOutlinePath();

      // Review-fix #01 FX-04: snapshot in-flight lightning bolts AND
      // the spawn cadence counter BEFORE the innerHTML rewrite below.
      // The rewrite detaches every DOM node under `this._root`,
      // including the `<polyline>`s appended to the lightning layer.
      // Without this snapshot, every HA state push during charging
      // wiped all in-flight bolts simultaneously (the "particles all
      // disappear at the exact same time" symptom QS-214 fixed on
      // the boiler card). The `b.el` references survive detachment
      // (JS still holds them); they get re-attached to the new
      // layer below. Mirrors qs-water-boiler-card.js:1216-1219 for
      // the snapshot side and 1418-1437 for the restore side.
      const preservedLightning = this._lightning;
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
                <clipPath id="${batteryClipId}">
                  <path d="${batteryBodyClipD}" />
                </clipPath>
                <filter id="${lightningGlowId}" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="${LIGHTNING_GLOW_STDDEV}" result="blur" />
                  <feFlood flood-color="${LIGHTNING_GLOW_COLOR}" flood-opacity="1" />
                  <feComposite in2="blur" operator="in" result="glow" />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <g clip-path="url(#${batteryClipId})">
                <path id="wave0_idle"   d="${initialWavePaths[0]}" fill="${IDLE_WATER_COLORS[0]}"     opacity="${initialIdleOpacity}"   pointer-events="none" style="will-change: transform;" />
                <path id="wave0_charge" d="${initialWavePaths[0]}" fill="${CHARGING_WATER_COLORS[0]}" opacity="${initialChargeOpacity}" pointer-events="none" style="will-change: transform;" />
                <g id="${lightningLayerId}" filter="url(#${lightningGlowId})" pointer-events="none"></g>
              </g>
              <path id="battery_outline" d="${batteryOutlineD}" fill="none"
                    stroke="rgba(255,255,255,${initialOutlineStrokeAlpha.toFixed(3)})"
                    stroke-width="${BATTERY_OUTLINE_STROKE_WIDTH}"
                    stroke-linejoin="round" pointer-events="none" />
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

                  <div id="rabbit_btn" class="rabbit-btn ${sChargeType?.state === 'As Fast As Possible' ? 'on' : ''}"><ha-icon icon="mdi:rabbit"></ha-icon></div>
                  <div class="target-cell">
                    <div id="target_value" class="target-value">${displayTargetValue}</div>
                    ${useEnergyMode ? '' : `<div class="mini-range-target" aria-label="range at target">${rangeTargetStr}</div>`}
                  </div>
                  <div id="time_btn" class="time-btn ${chargeTime && chargeTime !== '--:--' ? 'on' : ''}">${chargeTime}</div>
                </div>
                </div>
                <div class="center-controls" style="flex-direction:column; gap:4px;">
                  <div class="mini-title">Solar priority</div>
                  <div id="sun_btn" class="sun-btn ${swPriority?.state === 'on' ? 'on' : ''}"><ha-icon icon="mdi:weather-sunny"></ha-icon></div>
                </div>
              </div>
            </div>
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

      // QS-224: innerHTML rewrite detached the previous wave / outline
      // / lightning DOM nodes. Sync RAF's memo keys to the rendered
      // state (prevents a redundant regen on the next frame) and null
      // the cached DOM refs so RAF re-resolves them lazily. The new
      // wave nodes already carry the freshly-generated `d`
      // (initialWavePaths) and the correct `fill`
      // (IDLE_/CHARGING_WATER_COLORS), so the next RAF tick can
      // proceed without an extra regen.
      this._lastWaterBaseY = this._waterBaseY;
      this._lastAmplitude  = initialAmp;
      this._waveEls = null;
      this._lightningLayerEl = null;
      this._batteryOutlineEl = null;

      // Review-fix #01 FX-04: restore the preserved lightning bolts
      // into the FRESH lightning layer. Their DOM identity changed
      // via innerHTML (same `id`, new element); for each preserved
      // bolt, re-attach its detached `el` to the new layer. The
      // per-frame state (life, maxLife) survived in the JS array, so
      // the RAF advance loop picks up exactly where it left off with
      // no visual blink. The spawn cadence counter is also restored
      // so the next spawn doesn't reset to 0 (which would burst-
      // spawn up to the cap on every HA state push). If no layer
      // exists in the fresh markup (defensive — e.g. a future
      // template change removed it), drop the preserved bolts via
      // explicit `el.remove()` so detached DOM nodes don't leak.
      if (preservedLightning?.length) {
        const newLightningLayer = this._lightningLayerId
          ? this._root.getElementById(this._lightningLayerId)
          : null;
        if (newLightningLayer) {
          for (const b of preservedLightning) {
            if (b?.el) newLightningLayer.appendChild(b.el);
          }
          this._lightning = preservedLightning.filter(b => b?.el);
          this._lightningLayerEl = newLightningLayer;
          this._nextLightningAt = preservedNextLightningAt;
        } else {
          for (const b of preservedLightning) { b?.el?.remove(); }
          this._lightning = [];
        }
      } else {
        this._lightning = [];
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
