/*
  QS Water Boiler Card - custom:qs-water-boiler-card
  Zero-build single-file Lit-style web component compatible with Home Assistant.

  Dedicated card for QSWaterBoiler (cumulus / thermodynamic boiler) loads.
  Originally forked from `qs-on-off-duration-card.js` (QS-194) with one
  boiler-specific addition: an optional water-tank temperature row.

  QS-200 layered three visual upgrades on top:
  - Heat palette (red/orange) replacing the cool-blue scheme.
  - Water animation inside the ring, mirroring `qs-pool-card.js` — three
    sine-wave layers in a circular clipPath, water level driven by
    progress.
  - "Boiling" state when `running === true`: water cross-fades from cool
    blue to near-white translucent (dual-layer opacity, not HSL lerp),
    bubbles rise bottom→top, and a red Gaussian-blurred glow with
    `mix-blend-mode: screen` traces the water surface.

  QS-211 added a 4th boiling-state visual on top of QS-200:
  - "Steam" layer above the water — soft white-translucent puffs
    spawn at the surface, rise to the top of the clip circle with a
    gentle sin-wobble, grow and fade with a piecewise-linear life
    curve. Single Gaussian-blur filter applied to the layer group
    (NOT per particle). Cross-fades with `_currentColorMix` like the
    other boiling visuals; graceful exit on running→false.

  Other lit features:
  - Optional `temperature_sensor` entity row, rendered at the top of
    the card when configured (the QS-194 customisation).
  All other entity keys match the on/off-duration card's input shape.
*/

// --- Geometry (must match the SVG <clipPath> circle attributes below) ---
const CENTER_CX = 160;              // SVG x-center of the ring / clip circle
const CENTER_CY = 160;              // SVG y-center of the ring / clip circle
const CLIP_R = 120;                 // water clip circle radius (10px inside ringCirc=130 — same as pool, intentional)
const WAVE_WIDTH = 480;             // single wave period (2× clip diameter)
const WAVE_BOTTOM_Y = 400;          // closing rectangle y; clipped by circle

// --- Per-layer wave offsets ---
const LAYER_SCROLL_OFFSET = 1.2;
const LAYER_PHASE_OFFSET = 2.1;

// --- Water palettes ---
const COOL_WATER_COLORS = [
  'hsla(185, 60%, 22%, 0.55)',
  'hsla(185, 60%, 20%, 0.45)',
  'hsla(185, 60%, 18%, 0.35)',
];
const BOIL_WATER_COLORS = [
  'hsla(0, 0%, 95%, 0.65)',
  'hsla(0, 0%, 90%, 0.55)',
  'hsla(0, 0%, 85%, 0.45)',
];

// --- Animation tuning (lerp envelope; mirror pool) ---
const LERP_RATE = 2;
const LERP_DT_CEIL = 0.1;
const AMP_REGEN_THRESHOLD = 0.25;
const LEVEL_REGEN_THRESHOLD = 0.01;
const PHASE_WRAP = 1e6;
const PHASE_TO_PX = 60;

// --- Calm vs boil targets ---
const CALM_AMP = 1.5;
const BOIL_AMP = 8;
const CALM_SPEED = 0.2;
const BOIL_SPEED = 1.6;

// --- Bubbles ---
const MAX_CONCURRENT_BUBBLES = 12;
const BUBBLE_SPAWN_RATE_HZ = 6;
const BUBBLE_RADIUS_MIN = 1.5;
const BUBBLE_RADIUS_MAX = 4;
const BUBBLE_SPEED_PX_PER_S_MIN = 40;
const BUBBLE_SPEED_PX_PER_S_MAX = 80;
const BUBBLE_MAX_LIFE_S = 2.5;
const BUBBLE_FILL_COLOR = 'rgba(255,255,255,0.85)';

// --- Steam (QS-211) ---
// Wispy puffs rising from the water surface (_waterBaseY) to the top
// of the clip circle (CENTER_CY - CLIP_R) while the boiler is running.
// Architecture mirrors the bubble subsystem above: spawn-gated on
// `boiling`, capped at MAX_CONCURRENT_STEAM, advance/retire outside
// the gate for graceful exit. A single feGaussianBlur is applied to
// the layer <g> (NOT per particle) for the soft wispy look without
// per-circle filter cost.
const MAX_CONCURRENT_STEAM = 8;
const STEAM_SPAWN_RATE_HZ = 1.5;
const STEAM_RADIUS_MIN = 4;
const STEAM_RADIUS_MAX = 10;
const STEAM_RISE_PX_PER_S_MIN = 10;
const STEAM_RISE_PX_PER_S_MAX = 22;
const STEAM_DRIFT_PX_PER_S = 6;
const STEAM_DRIFT_FREQ_HZ = 0.4;
const STEAM_RADIUS_GROW_PX_PER_S = 4;
const STEAM_MAX_LIFE_S = 4.5;
const STEAM_FILL_COLOR = 'rgba(255,255,255,0.45)';
const STEAM_BLUR_STDDEV = 3.5;
const STEAM_TOP_MARGIN_PX = 4;

// --- Surface glow ---
const SURFACE_GLOW_COLOR = '#FF3D00';
const SURFACE_GLOW_STROKE_WIDTH = 4;
const SURFACE_GLOW_BLUR_STDDEV = 6;

class QsWaterBoilerCard extends HTMLElement {

  // Generate an SVG path for a sine-wave-based closed shape (ported
  // verbatim from `qs-pool-card.js`). Emits TWO repetitions of the wave
  // (path extent = [0, 2*width]) so the translated path always covers
  // the clip region regardless of scroll offset.
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

  // Reset cached DOM refs and the logical bubble array. Shared between
  // `_invalidateWaveCache()` (full memo reset on (re-)connect) and the
  // post-`innerHTML` cleanup block in `_render()`. Factored out so a
  // future memo-key addition (a new `_fooEl`) lands at BOTH call
  // sites — without the helper, the two paths would silently drift
  // and the post-render block would carry stale refs into the next
  // RAF frame.
  // Review-fix #03 N1: this helper ALSO resets `this._bubbles = []`.
  // That's a no-op at today's two call sites (both happen after the
  // shadow root was rewritten / bubbles already drained on disconnect)
  // but the bubble wipe is intentional: any caller using this helper
  // to force a "fresh wave canvas" wants the bubble layer reset too.
  // If a future caller needs DOM-only reset WITHOUT touching bubbles,
  // split this into `_resetWaveDomRefs()` + explicit
  // `this._bubbles = []` at the call sites.
  // QS-211: the same pattern is extended for the steam subsystem —
  // `this._steamLayerEl` (cached DOM ref) and `this._steamPuffs`
  // (logical particle array) are both reset here so the next RAF
  // tick after an innerHTML rewrite or full memo invalidation starts
  // with a clean slate (mirrors the QS-200 N1 note for bubbles).
  _resetDomRefs() {
    this._waveEls = null;
    this._bubbleLayerEl = null;
    this._surfaceGlowEl = null;
    this._bubbles = [];
    this._steamLayerEl = null;
    this._steamPuffs = [];
  }

  // Clear the wave-path memoization keys and cached DOM refs. Called on
  // every (re-)connect and after each _render() innerHTML rewrite.
  // QS-200 additions vs. pool: `_bubbleLayerEl`, `_surfaceGlowEl` (new
  // DOM refs). Wave opacity is updated unconditionally per frame so it
  // doesn't need a memo key. `_lastWaterBaseY` / `_lastAmplitude` get
  // nulled here (full invalidation) but are SYNCED to the current
  // render state in the post-innerHTML block — that's why they stay
  // outside `_resetDomRefs()`.
  _invalidateWaveCache() {
    this._lastWaterBaseY = null;
    this._lastAmplitude = null;
    this._resetDomRefs();
  }

  // QS-200: continuous RAF while connected, mirroring `qs-pool-card.js`.
  // The water is intrinsically visible at all times (cool blue when not
  // running, near-white translucent boiling when running) so RAF is no
  // longer gated on `showAnimation`. Calm vs. boiling is amplitude /
  // speed / color-mix lerp, not RAF on/off.
  _startAnimation() {
    if (this._animRaf != null) return;
    // Initialize wave animation state ONLY on the first-ever connect, so
    // that detach/re-attach (HA dashboard rearrangement, tab navigation)
    // preserves _currentAmplitude/_currentSpeed/_wavePhase. Without this
    // guard, a boiling wave would visibly snap back to CALM on each
    // reconnect and re-lerp up over ~1.5s.
    if (this._currentAmplitude == null) {
      this._currentAmplitude = CALM_AMP;
      this._currentSpeed = CALM_SPEED;
      this._wavePhase = 0;
      this._currentColorMix = 0;
      this._bubbles = [];
      this._nextBubbleAt = 0;
      // Tell the next _render() to prime amp/speed/colorMix directly to
      // the actual running-state targets, skipping the 1.5s lerp envelope.
      this._needsAnimationPrime = true;
      // QS-211: steam particle array + spawn cadence counter. Mirrors
      // the bubble lazy-init above; runs once on first connect and is
      // preserved across detach/re-attach.
      this._steamPuffs = [];
      this._nextSteamAt = 0;
    }
    this._lastAnimTs = null;
    this._invalidateWaveCache();

    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      // S6: cap `dt` against hidden-tab return. Without this, the first
      // frame after a multi-second tab-hidden window produces a huge
      // `dt`, snapping wave phase by hundreds of pixels in one frame
      // and aging every bubble past `BUBBLE_MAX_LIFE_S` simultaneously.
      // The cap matches `LERP_DT_CEIL` so all step-loop subsystems are
      // bounded by the same envelope.
      let dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      dt = Math.min(dt, LERP_DT_CEIL);
      this._lastAnimTs = ts;

      // --- Existing dashed-arc animation (preserved verbatim).
      // The running-progress dash still drives off `showAnimation`,
      // which is now applied as a render-time switch on the
      // `<path id="running_anim">` element. The RAF loop itself no
      // longer gates on it.
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const dashSpeed = 80; // dash units per second
      this._animOffset = ((this._animOffset || 0) + dashSpeed * dt) % patternLen;
      const dashPath = this._root?.getElementById('running_anim');
      if (dashPath) {
        dashPath.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }

      // --- Lerp amplitude / speed / colorMix toward boiling targets.
      // `dt` is already clamped at LERP_DT_CEIL above (S6), so the
      // lerpFactor envelope, phase advance, and bubble life all share
      // the same upper bound — no per-system local clamp needed.
      // N7: `_currentColorMix` is initialised to 0 in the lazy-init
      // block at the top of _startAnimation, so the lerp can read it
      // directly without an `?? 0` fallback (matches amp/speed style).
      const boiling = this._running === true;
      const targetAmplitude = boiling ? BOIL_AMP : CALM_AMP;
      const targetSpeed = boiling ? BOIL_SPEED : CALM_SPEED;
      const targetColorMix = boiling ? 1 : 0;
      const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
      this._currentAmplitude += (targetAmplitude - this._currentAmplitude) * lerpFactor;
      this._currentSpeed += (targetSpeed - this._currentSpeed) * lerpFactor;
      this._currentColorMix += (targetColorMix - this._currentColorMix) * lerpFactor;
      this._wavePhase += this._currentSpeed * dt;
      // N6: sign-safe modulo for consistency with the `scrollOffset`
      // wrap below. Phase is monotonically non-decreasing today (speed
      // is always positive), so the prior `%` worked — the new form is
      // robust to any sign / magnitude of accumulated phase.
      this._wavePhase = ((this._wavePhase % PHASE_WRAP) + PHASE_WRAP) % PHASE_WRAP;

      // Lazy-resolve wave / bubble layer / surface glow DOM refs once per
      // innerHTML rewrite. Six wave nodes (3 cool + 3 boil pairs) — the
      // pair shares geometry; only `fill` differs.
      if (!this._waveEls) {
        const cool0 = this._root?.getElementById('wave0_cool') ?? null;
        const boil0 = this._root?.getElementById('wave0_boil') ?? null;
        const cool1 = this._root?.getElementById('wave1_cool') ?? null;
        const boil1 = this._root?.getElementById('wave1_boil') ?? null;
        const cool2 = this._root?.getElementById('wave2_cool') ?? null;
        const boil2 = this._root?.getElementById('wave2_boil') ?? null;
        this._waveEls = [cool0, boil0, cool1, boil1, cool2, boil2];
      }
      const surfaceGlow = this._surfaceGlowEl
        ?? (this._surfaceGlowEl = this._root?.getElementById('surface_glow') ?? null);
      const bubbleLayer = this._bubbleLayerEl
        ?? (this._bubbleLayerEl = this._bubbleLayerId
              ? (this._root?.getElementById(this._bubbleLayerId) ?? null)
              : null);

      // --- Wave transforms (per-layer translateX).
      // Path extent is [0, 2*WAVE_WIDTH] so the translated path always
      // covers the clip region. We offset by -CLIP_R to align the path
      // start with the clip's left edge, then scroll within one period.
      // Same `tx` applied to both cool+boil siblings of each layer, and
      // to surface_glow (locked to wave 0's tx so the glow trace stays
      // with the visible crest).
      // N13: initialised to a safe `translateX(0px)` so a future early
      // return / throw before the i===0 iteration can't leave
      // `surfaceGlow.style.transform = ''` (which CSS resets to `none`
      // — the glow would snap to SVG origin while waves stay translated).
      let wave0Tx = 'translateX(0px)';
      for (let i = 0; i < 3; i++) {
        const phaseOffset = i * LAYER_SCROLL_OFFSET;
        const raw = (this._wavePhase + phaseOffset) * PHASE_TO_PX;
        const scrollOffset = ((raw % WAVE_WIDTH) + WAVE_WIDTH) % WAVE_WIDTH;
        const tx = -CLIP_R - scrollOffset;
        const txStr = `translateX(${tx.toFixed(1)}px)`;
        if (i === 0) wave0Tx = txStr;
        const coolEl = this._waveEls[i * 2];
        const boilEl = this._waveEls[i * 2 + 1];
        if (coolEl) coolEl.style.transform = txStr;
        if (boilEl) boilEl.style.transform = txStr;
      }

      // --- Wave path regen (throttled by amplitude / water-level delta).
      // Both cool and boil siblings share the same `d` string; only fill
      // differs (already set in _render()).
      const waterBaseY = this._waterBaseY;
      const hasValidBase = waterBaseY != null && !Number.isNaN(waterBaseY);
      const ampDelta = this._lastAmplitude == null
          ? Infinity
          : Math.abs(this._currentAmplitude - this._lastAmplitude);
      const levelChanged = hasValidBase &&
          Math.abs(waterBaseY - (this._lastWaterBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
      let regenerated = false;
      if (hasValidBase && (levelChanged || ampDelta > AMP_REGEN_THRESHOLD)) {
        this._lastWaterBaseY = waterBaseY;
        this._lastAmplitude = this._currentAmplitude;
        for (let i = 0; i < 3; i++) {
          const phaseOffset = i * LAYER_PHASE_OFFSET;
          const freq = 2 + i; // integer freq → seamless wrap at WAVE_WIDTH
          const d = this._generateWavePath(WAVE_WIDTH, this._currentAmplitude, freq, phaseOffset, waterBaseY);
          const coolEl = this._waveEls[i * 2];
          const boilEl = this._waveEls[i * 2 + 1];
          if (coolEl) coolEl.setAttribute('d', d);
          if (boilEl) boilEl.setAttribute('d', d);
        }
        regenerated = true;
      }

      // --- Surface glow sync (per AC-8).
      // d: only resynced when wave 0 was regenerated (geometry hasn't
      //    changed otherwise — opacity / transform updates handle the
      //    visible motion). First-frame parity (review-fix #02 N4):
      //    on the first RAF tick after `_render()`, neither `levelChanged`
      //    nor `ampDelta > AMP_REGEN_THRESHOLD` is true (memo keys
      //    were just synced post-innerHTML), so we DON'T enter the
      //    regen branch — but the SVG markup anchors both wave 0 and
      //    surface_glow to `initialWavePaths[0]`, so they're already
      //    parity-aligned at paint time. Any future refactor that
      //    diverges those two initial `d` strings must also force a
      //    surface_glow.setAttribute('d', ...) on the first frame.
      // transform: synced EVERY frame so the glow trace stays locked to
      //            wave 0's translated crest.
      // opacity: bound to colorMix every frame (0 when calm, 1 when boiling).
      if (surfaceGlow) {
        if (regenerated && this._waveEls?.[0]) {
          const wave0d = this._waveEls[0].getAttribute('d');
          if (wave0d) surfaceGlow.setAttribute('d', wave0d);
        }
        surfaceGlow.style.transform = wave0Tx;
        surfaceGlow.setAttribute('opacity', this._currentColorMix.toFixed(3));
      }

      // --- Per-frame wave opacity update (cool ↔ boil cross-fade).
      // Dual-layer opacity rather than HSL lerp — see story D2 / D11
      // (HSL lerp from cyan to orange passes through yellow-green at
      // the midpoint; opacity cross-fade is artifact-free).
      const coolOpacity = (1 - this._currentColorMix).toFixed(3);
      const boilOpacity = this._currentColorMix.toFixed(3);
      for (let i = 0; i < 3; i++) {
        const coolEl = this._waveEls[i * 2];
        const boilEl = this._waveEls[i * 2 + 1];
        if (coolEl) coolEl.setAttribute('opacity', coolOpacity);
        if (boilEl) boilEl.setAttribute('opacity', boilOpacity);
      }

      // === AC-7 bubble system: dynamic spawn, soft-capped at MAX_CONCURRENT_BUBBLES ===
      if (bubbleLayer) {
        // Spawn cadence (only while boiling, capped at MAX_CONCURRENT_BUBBLES).
        // N7: reuse the `boiling` const computed above instead of
        //     re-evaluating `this._running === true`.
        // N6: drop the `?? 0` defence — the lazy-init in
        //     _startAnimation guarantees `_nextBubbleAt = 0` before
        //     the first step runs (matches the N7 simplification
        //     applied to `_currentColorMix` in fix #01).
        if (boiling) {
          this._nextBubbleAt -= dt;
          while (this._nextBubbleAt <= 0 && this._bubbles.length < MAX_CONCURRENT_BUBBLES) {
            const cx = CENTER_CX - CLIP_R + 8 + Math.random() * (2 * (CLIP_R - 8));
            const cy = CENTER_CY + CLIP_R - 8;
            const r = BUBBLE_RADIUS_MIN + Math.random() * (BUBBLE_RADIUS_MAX - BUBBLE_RADIUS_MIN);
            const vy = BUBBLE_SPEED_PX_PER_S_MIN + Math.random() * (BUBBLE_SPEED_PX_PER_S_MAX - BUBBLE_SPEED_PX_PER_S_MIN);
            // createElementNS is REQUIRED for SVG inside a Shadow DOM.
            // `document.createElement('circle')` would create an HTML
            // element, not an SVG one.
            const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            el.setAttribute('cx', cx.toFixed(2));
            el.setAttribute('cy', cy.toFixed(2));
            el.setAttribute('r', r.toFixed(2));
            el.setAttribute('fill', BUBBLE_FILL_COLOR);
            el.setAttribute('pointer-events', 'none');
            el.setAttribute('opacity', '0.9');
            bubbleLayer.appendChild(el);
            this._bubbles.push({el, cx, cy, r, vy, life: 0, maxLife: BUBBLE_MAX_LIFE_S});
            this._nextBubbleAt += 1 / BUBBLE_SPAWN_RATE_HZ;
          }
          // Clamp to 0 to avoid a spawn-backlog burst when running
          // re-engages after a cap-hit window.
          if (this._nextBubbleAt < 0) this._nextBubbleAt = 0;
        }

        // Advance + retire active bubbles regardless of running state
        // (graceful exit: existing bubbles continue to rise until they
        // retire naturally; no abrupt vanish on running → false).
        // N5: drop the `?? 0` fallback — _render() always sets
        // _waterBaseY to a finite number before any RAF frame runs,
        // and the bubble-retire loop only fires when bubbles exist
        // (= at least one spawn happened, = _render() ran).
        const surfaceY = this._waterBaseY + 4;
        const alive = [];
        for (const b of this._bubbles) {
          b.life += dt;
          b.cy -= b.vy * dt;
          b.r = Math.min(b.r + 0.5 * dt, BUBBLE_RADIUS_MAX + 2);
          if (b.cy < surfaceY || b.life >= b.maxLife) {
            b.el.remove();
            continue;
          }
          const lifeT = b.life / b.maxLife;
          const opacity = Math.max(0, 1 - lifeT) * this._currentColorMix;
          b.el.setAttribute('cy', b.cy.toFixed(2));
          b.el.setAttribute('r', b.r.toFixed(2));
          b.el.setAttribute('opacity', opacity.toFixed(3));
          alive.push(b);
        }
        this._bubbles = alive;
      }

      // === QS-211 steam system: dynamic spawn, soft-capped at MAX_CONCURRENT_STEAM.
      // Mirrors the bubble subsystem above. Spawn gated on `boiling`;
      // advance/retire runs unconditionally (graceful exit on running→false
      // via `_currentColorMix` opacity multiplier — see D15 in QS-211 story).
      const steamLayer = this._steamLayerEl
        ?? (this._steamLayerEl = this._steamLayerId
              ? (this._root?.getElementById(this._steamLayerId) ?? null)
              : null);
      if (steamLayer) {
        if (boiling) {
          this._nextSteamAt -= dt;
          while (this._nextSteamAt <= 0 && this._steamPuffs.length < MAX_CONCURRENT_STEAM) {
            // Spawn at the water surface, within the chord of the clip
            // circle at that Y (so puffs originate inside the visible
            // water surface — degenerate chord → cluster near CENTER_CX,
            // visually inconspicuous for an empty tank).
            const jitter = Math.random() * 4 - 2;
            const cySpawn = this._waterBaseY + jitter;
            const dy = cySpawn - CENTER_CY;
            const chordHalf = Math.sqrt(Math.max(0, CLIP_R * CLIP_R - dy * dy));
            const cxSpawn = CENTER_CX + (Math.random() * 2 - 1) * chordHalf * 0.85;
            const r = STEAM_RADIUS_MIN + Math.random() * (STEAM_RADIUS_MAX - STEAM_RADIUS_MIN);
            const vy = STEAM_RISE_PX_PER_S_MIN + Math.random() * (STEAM_RISE_PX_PER_S_MAX - STEAM_RISE_PX_PER_S_MIN);
            const phase = Math.random() * Math.PI * 2;
            // createElementNS is REQUIRED for SVG inside a Shadow DOM.
            const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            el.setAttribute('cx', cxSpawn.toFixed(2));
            el.setAttribute('cy', cySpawn.toFixed(2));
            el.setAttribute('r', r.toFixed(2));
            el.setAttribute('fill', STEAM_FILL_COLOR);
            el.setAttribute('pointer-events', 'none');
            el.setAttribute('opacity', '0');
            steamLayer.appendChild(el);
            this._steamPuffs.push({el, cx: cxSpawn, cy: cySpawn, r, vy, phase, life: 0, maxLife: STEAM_MAX_LIFE_S});
            this._nextSteamAt += 1 / STEAM_SPAWN_RATE_HZ;
          }
          // Clamp to 0 to avoid a spawn-backlog burst when capacity recovers.
          if (this._nextSteamAt < 0) this._nextSteamAt = 0;
        }

        // Advance + retire active puffs regardless of boiling state.
        // Graceful exit: in-flight puffs keep rising; opacity fades to
        // 0 via `_currentColorMix` over ~1.5s when running→false.
        const topY = CENTER_CY - CLIP_R + STEAM_TOP_MARGIN_PX;
        const aliveSteam = [];
        for (const p of this._steamPuffs) {
          p.life += dt;
          p.cy -= p.vy * dt;
          p.cx += STEAM_DRIFT_PX_PER_S * Math.sin(2 * Math.PI * STEAM_DRIFT_FREQ_HZ * p.life + p.phase) * dt;
          p.r = Math.min(p.r + STEAM_RADIUS_GROW_PX_PER_S * dt, STEAM_RADIUS_MAX + 4);
          if (p.cy < topY || p.life >= p.maxLife) {
            p.el.remove();
            continue;
          }
          // Life curve: piecewise linear — fade-in [0, 0.15], hold
          // [0.15, 0.7], fade-out [0.7, 1].
          const lifeT = p.life / p.maxLife;
          let lifeOpacity;
          if (lifeT < 0.15) lifeOpacity = lifeT / 0.15;
          else if (lifeT < 0.7) lifeOpacity = 1;
          else lifeOpacity = Math.max(0, 1 - (lifeT - 0.7) / 0.3);
          const opacity = lifeOpacity * this._currentColorMix;
          p.el.setAttribute('cx', p.cx.toFixed(2));
          p.el.setAttribute('cy', p.cy.toFixed(2));
          p.el.setAttribute('r', p.r.toFixed(2));
          p.el.setAttribute('opacity', opacity.toFixed(3));
          aliveSteam.push(p);
        }
        this._steamPuffs = aliveSteam;
      }

      this._animRaf = requestAnimationFrame(step);
    };
    this._animRaf = requestAnimationFrame(step);
  }

  _stopAnimation() {
    if (this._animRaf != null) cancelAnimationFrame(this._animRaf);
    this._animRaf = null;
    this._lastAnimTs = null;
    // Review-fix #02 N12: capture the running state at stop time so
    // a re-attach after the boiler flipped state can detect the
    // mismatch and force `_needsAnimationPrime = true` on the next
    // `_render`. Without this, the wave would lerp from the
    // pre-disconnect colorMix (~1 if boiling at detach) toward the
    // new target over ~1.5s — visually wrong if the boiler was off
    // for the entire detach window.
    this._runningAtStop = this._running;
  }

  connectedCallback() {
    // QS-200: continuous RAF while connected, mirroring qs-pool-card.
    // The wave animation is intrinsically visible at all times (cool
    // blue when not running, near-white boiling when running) so RAF
    // is no longer gated on `showAnimation`. See also
    // docs/agents/concepts/dashboard-and-cards.md.
    this._startAnimation();
    // Review-fix #04 M1: arm the one-shot `_pendingReattachCheck`
    // flag so the next `_render()` consumes the `_runningAtStop`
    // stash exactly once. Setting it here (NOT in `_stopAnimation`)
    // is what distinguishes "first post-reattach render" from "any
    // mid-detach hass-push render" — `set hass` doesn't gate on
    // `this.isConnected`, so renders DO fire during the detached
    // window, and those must NOT consume the stash.
    this._pendingReattachCheck = true;
  }

  disconnectedCallback() {
    this._stopAnimation();
    // S7: reset interaction flags so a re-attach after mid-interaction
    // (e.g. dragging the ring or processing a mode change when the
    // dashboard rearranges) doesn't silently short-circuit `set hass`
    // on stale flags.
    this._isInteractingMode = false;
    this._isInteractingTarget = false;
    this._isProcessingModeChange = false;
    this._modalOpen = false;
    // QS-200: eagerly tear down bubble DOM nodes. Without this they
    // would be GC'd along with the shadow root, but explicit cleanup is
    // cheap and avoids dangling SVG nodes during a rapid detach/attach.
    this._bubbles?.forEach(b => b.el?.remove?.());
    this._bubbles = [];
    // QS-211: same eager teardown for the steam <circle> nodes.
    // Optional-chaining matches the bubble shape so a partially-
    // constructed puff (e.g. from a mid-spawn throw) doesn't crash
    // teardown.
    this._steamPuffs?.forEach(p => p.el?.remove?.());
    this._steamPuffs = [];
  }

  static getStubConfig() {
    return { name: "QS Water Boiler", entities: {} };
  }

  setConfig(config) {
    if (!config || !config.entities) throw new Error("entities is required");
    this._config = config;
    this._root = this.attachShadow({ mode: "open" });
    // Review-fix #04 M1: explicitly initialise the reconnect-consume
    // flag to false so the very first `_render` after mount doesn't
    // trigger a phantom consume. `connectedCallback` flips it to true
    // before the next render, and `_render` clears it after the
    // one-shot consume.
    this._pendingReattachCheck = false;
    this._render();
  }

  // S6: defence-in-depth HTML escaping for user-/3rd-party-controlled
  // strings interpolated into innerHTML (card title, entity unit, etc.).
  // HA entity-id validation makes most paths unreachable in practice,
  // but treat as untrusted.
  _escapeHtml(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // S8: safe numeric coercion. `Number(s?.state || N)` short-circuits to
  // the truthy string when `state === "unknown" | "unavailable"`, then
  // `Number("unknown")` is `NaN` and propagates into SVG cx/cy/d
  // attributes. Filter degenerate states BEFORE conversion.
  _safeNumber(sensor, defaultValue) {
    if (!sensor || sensor.state == null) return defaultValue;
    const s = sensor.state;
    if (s === '' || s === 'unknown' || s === 'unavailable') return defaultValue;
    const n = Number(s);
    return Number.isNaN(n) ? defaultValue : n;
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._root) return;
    // Avoid re-rendering while user is interacting with selects or a modal is open
    if (this._isInteractingMode || this._modalOpen || this._isInteractingTarget) return;
    this._render();
  }

  getCardSize() { return 5; }

  _entity(id) { return id ? this._hass?.states?.[id] : undefined; }

  _call(domain, service, data) {
    return this._hass.callService(domain, service, data);
  }

  _press(entity_id) { return this._call('button', 'press', { entity_id }); }
  _turnOn(entity_id) { return this._call('switch', 'turn_on', { entity_id }); }
  _turnOff(entity_id) { return this._call('switch', 'turn_off', { entity_id }); }
  _select(entity_id, option) { return this._call('select', 'select_option', { entity_id, option }); }
  _setNumber(entity_id, value) { return this._call('number', 'set_value', { entity_id, value }); }
  _setTime(entity_id, value) { return this._call('time', 'set_value', { entity_id, time: value }); }

  _fmt(num, round = true) {
    const n = Number(num);
    if (num == null || Number.isNaN(n)) return '--';
    return round ? Math.round(n) : n.toFixed(1);
  }

  _render() {
      const cfg = this._config || {};
      const e = cfg.entities || {};

      const sDurationLimit = this._entity(e.duration_limit);
      const sCurrentDuration = this._entity(e.current_duration);
      const sDefaultOnDuration = this._entity(e.default_on_duration);
      const sDefaultOnFinishTime = this._entity(e.default_on_finish_time);
      const sCommand = this._entity(e.command);
      const selBistateMode = this._entity(e.bistate_mode);
      const swGreenOnly = this._entity(e.green_only);
      const swEnableDevice = this._entity(e.enable_device);
      const sOverrideState = this._entity(e.override_state);
      const sStartTime = this._entity(e.start_time);
      const sEndTime = this._entity(e.end_time);
      const sIsOffGrid = this._entity(e.is_off_grid);
      // QS-194: optional water tank temperature sensor (plumbing-only —
      // the underlying QSWaterBoiler doesn't yet act on this value, but
      // the card surfaces it so users can see it at a glance).
      const sTemperatureSensor = this._entity(e.temperature_sensor);

      const title = (cfg.title || cfg.name) || "Device";
      
      // Check if device is enabled
      const isEnabled = swEnableDevice?.state === 'on';
      
      // Check if system is off-grid
      const isOffGrid = sIsOffGrid?.state === 'on';
      
      // Get target hours and current run hours directly (in hours)
      // S8: use safeNumber so an `unknown`/`unavailable` sensor doesn't
      // propagate `NaN` into the SVG path attributes downstream.
      const targetHours = this._safeNumber(sDurationLimit, 12);
      const hoursRun = this._safeNumber(sCurrentDuration, 0);
      const defaultDuration = this._safeNumber(sDefaultOnDuration, 6);
      
      // Get bistate mode
      const bistateMode = selBistateMode?.state || 'bistate_mode_default';
      const isDefaultMode = bistateMode === 'bistate_mode_default';
      
      // Get override state
      const overrideState = sOverrideState?.state || 'NO OVERRIDE';
      const isOverridden = overrideState !== 'NO OVERRIDE';
      const isResettingOverride = overrideState === 'ASKED FOR RESET OVERRIDE';
      
      // Determine if device is running (command state must be "on")
      const commandState = sCommand?.state || '';
      const running = commandState.toLowerCase() === 'on';
      
      // Determine max hours and what to display
      let maxHours, displayTargetHours;
      if (isDefaultMode) {
        // N3: configurable upper bound for the default-mode ring.
        // Boilers with significant thermal storage may legitimately
        // need >12h runs; set `max_default_hours: 24` (or similar)
        // on the card config to expand the visual range.
        maxHours = Number(cfg.max_default_hours) || 12;
        displayTargetHours = defaultDuration;
      } else {
        // BH (user-reported NaN bug): a brand-new water boiler with no
        // live constraint sensor reports targetHours == 0, which made
        // `hoursToPct` divide by zero and propagate NaN into the SVG
        // arc path (`A 130 130 0 0 1 NaN NaN`). Clamp to a sensible
        // positive fallback so the ring always renders.
        maxHours = targetHours > 0 ? targetHours : (Number(cfg.max_default_hours) || 12);
        displayTargetHours = targetHours;
      }
      
      // Determine if we should show from/to times
      const showFromTo = !isDefaultMode || isOverridden;
      
      // Helper to check if state is valid
      const isValidState = (state) => {
        if (!state) return false;
        const stateLower = String(state).toLowerCase();
        return !['unavailable', 'unknown', 'none', ''].includes(stateLower);
      };
      
      // Get from/to times
      const startTime = (sStartTime && isValidState(sStartTime.state)) ? sStartTime.state : '--:--';
      const endTime = (sEndTime && isValidState(sEndTime.state)) ? sEndTime.state : '--:--';
      
      // QS-200: heat palette (mirrors `qs-climate-card.js` `colorSchemes.heat`).
      // Hard-coded — boilers always render in the warm scheme. The cool blue
      // values previously here (`#2196F3` / `#00bcd4` / `#8bc34a` /
      // `#00e1ff` / `#0066ff`) were swapped to the climate-card heat hex
      // codes. Cool blue may still appear ELSEWHERE in the file — e.g.
      // `.power-btn.on` uses HA-blue `#2196F3` as a semantic "power"
      // anchor, which is intentional and unrelated to this palette.
      const colors = {
        primary:    '#FF5722',
        gradStart:  '#FF5722',
        gradEnd:    '#D32F2F',
        animStart:  '#FF6E40',
        animEnd:    '#E64A19',
      };
      
      const css = `
      :host { --pad: 18px; display:block; }
      .card { padding: var(--pad); position: relative; }
      .card.off-grid { background: rgba(244, 67, 54, 0.08); }
      .card-title { text-align:center; font-weight:800; font-size: 1.6rem; margin: 0px 0 0px; }
      /* QS-194: optional water-tank temperature row — only shown when
         the user configured a temperature sensor on the boiler. */
      .tank-temp { display:flex; align-items:center; justify-content:center; gap:8px;
                   margin: 4px auto 0; color: var(--secondary-text-color); font-size: 0.95rem; }
      .tank-temp .temp-value { font-weight: 700; color: var(--primary-text-color); }
      .top { display:flex; gap:12px; flex-wrap:wrap; }
      .below { display:flex; align-items:center; justify-content:center; margin-top: 8px; width:260px; margin-left:auto; margin-right:auto; }
      .below .pill { width:100%; }
      .below-line { width:260px; margin: 8px auto 0; display:grid; grid-template-columns: 1fr auto; align-items:center; column-gap:12px; }
      .below-line.full { display:block; }
      .below-line.full > button { width: 100%; justify-content: center; position: relative; }
      .pill { display:flex; align-items:center; gap:8px; border-radius: 28px; height:40px; min-height:40px; padding:0 12px; border:1px solid var(--divider-color);
              background: var(--ha-card-background, var(--card-background-color)); box-sizing: border-box; cursor: pointer; touch-action: manipulation; }
      .pill .dot { width:12px; height:12px; border-radius:50%; background: var(--divider-color); box-shadow: 0 0 8px rgba(0,0,0,.25) inset; }
      .pill.on { background: rgba(56,142,60,0.15); border-color: rgba(56,142,60,.35); }
      .pill.on .dot { background: #2ecc71; box-shadow: 0 0 12px #2ecc71aa; }
      .pill { position: relative; }
      .pill select { appearance:none; background: transparent; color: var(--primary-text-color); border: none; font-weight:700; position: absolute; left:0; top:0; width:100%; height:100%; text-align:center; text-align-last:center; padding: 0 12px 0 40px; border-radius: 28px; cursor: pointer; z-index:1; box-sizing: border-box; }

      .hero { margin-top: 0px; display:flex; align-items:center; justify-content:center; gap: 12px; }
      .ring { position: relative; width:300px; height:300px; margin: 0 auto; }
      /* Mobile touch fix: touch-action:none on the SVG (not the inner <circle>) prevents the
         browser from initiating scroll/pan gestures when dragging the ring handle. SVG child
         elements like <circle> don't reliably honor touch-action on iOS Safari / HA Companion. */
      .ring svg { touch-action: none; }
      /* Mobile touch fix: touch-action:manipulation removes the 300ms tap delay that mobile
         browsers impose for double-tap detection, making button taps register immediately.
         Without this, a hass re-render can destroy the DOM node before the synthetic click fires. */
      .ring .green-btn { width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; left: 50%; top: 50%; transform: translate(97px, -137px); z-index: 10; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .green-btn ha-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); display:block; line-height:1; }
      .ring .green-btn.on { border-color: rgba(56,142,60,.45); background: rgba(46,204,113,.14); box-shadow: 0 0 0 3px rgba(46,204,113,.20), 0 0 16px #4CAF50; }
      .ring .green-btn.on ha-icon { color: #4CAF50; }
      .ring .power-btn { width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; left: 50%; top: 50%; transform: translate(-137px, -137px); z-index: 10; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .power-btn ha-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); display:block; line-height:1; }
      .ring .power-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .power-btn.on ha-icon { color: #2196F3; }
      .card.disabled { opacity: 0.5; pointer-events: none; filter: grayscale(0.8); }
      .card.disabled .power-btn { pointer-events: auto; opacity: 1; filter: grayscale(0); }
      .ring .center { position:absolute; inset:0; display:grid; place-items:center; text-align:center; pointer-events: none; transform: translateY(-5px); }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .target-value { font-weight:800; font-size: 2.5rem; line-height: 1.1; }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; text-align:center; width: 220px; margin: 0 auto; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:6px; }
      .ring .from-to-row { display:flex; justify-content:space-between; width:140px; margin-top: 8px; gap:20px; }
      .ring .from-to-item { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .from-to-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .from-to-value { color: var(--primary-text-color); font-weight:800; font-size: 1.4rem; }
      .ring .center-controls { display:flex; align-items:center; justify-content:center; margin-top: 6px; }
      .ring .override-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .ring .override-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; }
      .ring .override-btn.disabled { cursor: not-allowed; opacity: 0.6; }
      .ring .override-btn.active { border-color: rgba(255,152,0,.45); background: rgba(255,152,0,.14); box-shadow: 0 0 0 3px rgba(255,152,0,.20), 0 0 16px #FF9800; }
      .ring .override-btn.active ha-icon { color: #FF9800; }
      .ring .override-btn.resetting { border-color: rgba(76,175,80,.45); background: rgba(76,175,80,.14); }
      .ring .override-btn.resetting ha-icon { color: #4CAF50; }
      .time-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; font-size: 0.99rem; font-weight: 800; line-height: 1; margin-top: 6px; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
      .time-btn:hover { border-color: ${colors.primary}; background: rgba(255,255,255,.08); }
      .time-btn { color: ${colors.primary}; }
      .time-btn.on { border-color: ${colors.primary}; background: color-mix(in srgb, ${colors.primary} 14%, transparent); box-shadow: 0 0 0 3px color-mix(in srgb, ${colors.primary} 20%, transparent), 0 0 16px ${colors.primary}; color: ${colors.primary}; }

      select {
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
      .below select { width: 100%; height: 40px; min-height: 40px; }
      select:focus { border-color: var(--primary-color); box-shadow: 0 0 0 2px rgba(0,0,0,0), 0 0 0 3px color-mix(in srgb, var(--primary-color) 30%, transparent); }
      button { border:none; border-radius:18px; padding:14px 16px; font-weight:700; cursor:pointer; font-size: .95rem; }
      button.pill { height: 40px; min-height: 40px; display:flex; align-items:center; }
      .danger { background: var(--error-color); color: #fff; }
      button.outline { background: transparent !important; border-width: 2px; }
      .danger.outline { color: var(--error-color) !important; border-color: var(--error-color) !important; }
      
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

      const ringCirc = 130;
      const gapDeg = 60;
      const rangeDeg = 360 - gapDeg;
      const startDeg = gapDeg / 2;
      const endDeg = startDeg + rangeDeg;

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
          // N8: a zero-length arc (a0 == a1, e.g. progress == 0) would
          // produce a single-point SVG path that renders as nothing or
          // a stray dot. Return empty string so the consumer can decide
          // to omit the <path> element entirely.
          if (Math.abs(a1 - a0) < 0.01) return '';
          const p0 = polar(cx, cy, r, a0);
          const p1 = polar(cx, cy, r, a1);
          let delta = a1 - a0;
          if (delta < 0) delta += 360;
          const laf = delta > 180 ? 1 : 0;
          return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${laf} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
      };

      // Convert hours to percentage for arc calculation
      const hoursToPct = (hours) => (hours / maxHours) * 100;
      const pctToHours = (pct) => (pct / 100) * maxHours;
      
      const pctToDeg = (p) => startDeg + (Math.max(0, Math.min(100, p)) / 100) * rangeDeg;

      // Progress: hours run as percentage of max hours
      const progressPct = hoursToPct(hoursRun);
      const progressEndDeg = pctToDeg(progressPct);
      
      // Handle: target hours (or default duration for default mode)
      const handlePct = this._targetDragPct != null ? this._targetDragPct : 
                        (this._localTargetPct != null ? this._localTargetPct : hoursToPct(displayTargetHours));
      const handleDeg = pctToDeg(handlePct);
      
      // Review-fix #02 S1: align the arc / handle / progress-ring
      // center with the water clip circle's CENTER_CX/CENTER_CY. A
      // future tweak to either constant must move every part of the
      // ring (waves AND ring/handle/arc) together — using the named
      // constants here closes that gap.
      const center = {cx: CENTER_CX, cy: CENTER_CY};
      const arcLen = 2 * Math.PI * ringCirc * (rangeDeg / 360);
      const segLen = arcLen * (Math.max(0, Math.min(100, progressPct)) / 100);
      let dashLen = Math.round(segLen * 0.22);
      let gapLen = Math.round(segLen * 0.28);
      dashLen = Math.max(6, dashLen);
      gapLen = Math.max(6, gapLen);
      const patternLen = dashLen + gapLen;
      if (patternLen >= segLen - 4) {
          const scale = (segLen - 4) / patternLen;
          dashLen = Math.max(4, Math.round(dashLen * scale));
          gapLen = Math.max(4, Math.round(gapLen * scale));
      }
      this._animPatternLen = Math.max(8, dashLen + gapLen);
      
      const handlePos = polar(center.cx, center.cy, ringCirc, handleDeg);
      const bgPath = arcPath(center.cx, center.cy, ringCirc, startDeg, endDeg);
      const progressPath = arcPath(center.cx, center.cy, ringCirc, startDeg, progressEndDeg);
      
      const gradGreenId = `gradG-${Math.floor(Math.random() * 1e6)}`;
      const gradRunningId = `gradR-${Math.floor(Math.random() * 1e6)}`;
      const activeGradId = running ? gradRunningId : gradGreenId;
      // `showAnimation` is now a render-time switch for the dashed-arc
      // <path id="running_anim"> element only — the RAF loop itself is
      // continuous (started from connectedCallback per QS-200).
      const showAnimation = (running && segLen > 6);

      // QS-200: stash the running state for the RAF loop (drives the
      // calm ↔ boiling lerp and the bubble spawn cadence).
      //
      // Reconnect re-prime — third revision (review-fix #04 M1).
      // The N12 / S1 / M1 chain threads the needle on a subtle
      // lifecycle invariant: `_runningAtStop` must be consumed
      // EXACTLY ONCE on the first post-reattach `_render`,
      // regardless of inner-guard outcome.
      //
      // - Plan #02 N12 introduced `_runningAtStop` (stashed in
      //   `_stopAnimation`, consumed here, cleared unconditionally
      //   after the guard).
      // - Plan #03 S1 moved the clear INSIDE the guard's if-body
      //   to fix "mid-detach hass-push consumes stash too early".
      // - Plan #04 M1 gates the entire consume on
      //   `_pendingReattachCheck` (set in `connectedCallback`),
      //   because the pass-3 form leaked the stash across renders
      //   when reattach happened with `running` unchanged — and
      //   the next in-place state flip (hours later, no detach
      //   involved) would falsely fire the prime → snap instead
      //   of lerp.
      //
      // The flag-gated block below handles all three paths:
      // * mid-detach hass-pushes → flag is false (only set in
      //   `connectedCallback`) → consume skipped → stash preserved.
      // * reattach with `running` unchanged → flag true → consume →
      //   inner guard fails → no prime → stash + flag cleared.
      // * reattach with `running` flipped → flag true → consume →
      //   inner guard fires → prime queued → stash + flag cleared.
      if (this._pendingReattachCheck) {
        if (this._runningAtStop !== undefined && this._runningAtStop !== running) {
          this._needsAnimationPrime = true;
        }
        this._runningAtStop = undefined;
        this._pendingReattachCheck = false;
      }
      this._running = running;

      // QS-200: per-instance unique SVG ids so two boiler cards on the
      // same dashboard don't collide on `id="surface_glow"` /
      // `clipPath` / bubble-layer attributes (mirrors the pool card's
      // _nextClipId pattern).
      if (!this._waterClipId) {
        QsWaterBoilerCard._nextClipId = (QsWaterBoilerCard._nextClipId || 0) + 1;
        const uid = QsWaterBoilerCard._nextClipId;
        this._waterClipId = `wb_wClip_${uid}`;
        this._surfaceGlowFilterId = `wb_surfGlow_${uid}`;
        this._bubbleLayerId = `wb_bubbleLayer_${uid}`;
        // QS-211: per-instance unique IDs for the steam layer + filter.
        // Derived from the same `uid` source as the other IDs so two
        // boiler cards on the same dashboard never collide.
        this._steamLayerId = `wb_steamLayer_${uid}`;
        this._steamFilterId = `wb_steamFilter_${uid}`;
      }
      const waterClipId = this._waterClipId;
      const surfaceGlowFilterId = this._surfaceGlowFilterId;
      const bubbleLayerId = this._bubbleLayerId;
      const steamLayerId = this._steamLayerId;
      const steamFilterId = this._steamFilterId;

      // QS-200: water level from progress fill (same mapping as pool).
      // Treat null / NaN / negative / zero `displayTargetHours` as "no
      // progress yet" so a brand-new device renders an empty-ish tank
      // rather than a NaN-propagating ring.
      const progressRatio = (Number.isFinite(displayTargetHours) && displayTargetHours > 0)
        ? Math.max(0, Math.min(1, hoursRun / displayTargetHours))
        : 0;
      const rawWaterBaseY = CENTER_CY + CLIP_R - (0.2 + progressRatio * 0.6) * 2 * CLIP_R;
      this._waterBaseY = Number.isFinite(rawWaterBaseY)
        ? rawWaterBaseY
        : (CENTER_CY + CLIP_R - 0.2 * 2 * CLIP_R);

      // QS-200: prime the wave animation state to the actual running
      // targets on the first render after connect, avoiding the
      // ~1.5s boot transient. Without this, a card mounted while the
      // boiler is already boiling would visibly lerp up from CALM.
      if (this._needsAnimationPrime) {
        this._currentAmplitude = running ? BOIL_AMP : CALM_AMP;
        this._currentSpeed = running ? BOIL_SPEED : CALM_SPEED;
        this._currentColorMix = running ? 1 : 0;
        this._needsAnimationPrime = false;
      }

      // QS-200: pre-generate the 3 initial wave path `d` strings so the
      // SVG renders with water immediately, avoiding the empty-`d=""`
      // flash between the innerHTML rewrite and the first RAF tick. The
      // cool/boil siblings of each layer share the same `d`. Memo keys
      // are synced post-innerHTML so the next frame skips a redundant
      // regen.
      const initialAmp = this._currentAmplitude ?? CALM_AMP;
      const initialBaseY = this._waterBaseY;
      const initialColorMix = this._currentColorMix ?? 0;
      const initialWavePaths = [0, 1, 2].map(i => {
        const freq = 2 + i;
        const phaseOffset = i * LAYER_PHASE_OFFSET;
        return this._generateWavePath(WAVE_WIDTH, initialAmp, freq, phaseOffset, initialBaseY);
      });
      const initialCoolOpacity = (1 - initialColorMix).toFixed(3);
      const initialBoilOpacity = initialColorMix.toFixed(3);

      // Bistate mode selector options with translations
      const modeOptions = selBistateMode?.attributes?.options || [];
      const modeState = (selBistateMode?.state || '').trim();
      
      // Helper to translate bistate mode options.
      // S10: use the water_boiler_mode namespace so future boiler-specific
      // labels (Anti-Legionella, etc.) can diverge from on_off_mode without
      // touching this card.
      const translateBistateMode = (value) => {
          const key = `component.quiet_solar.entity.select.water_boiler_mode.state.${value}`;
          const translated = this._hass?.localize?.(key);
          // If translation not found or returns the key itself, fall back to the raw value
          return (translated && translated !== key) ? translated : value;
      };
      
      const modeOptionsHtml = modeOptions.map(o => 
          `<option value="${this._escapeHtml(o)}" ${o === modeState ? 'selected' : ''}>${this._escapeHtml(translateBistateMode(o))}</option>`
      ).join('');

      // Parse override command from override state
      const parseOverrideCommand = (overrideStateStr) => {
        if (!overrideStateStr || overrideStateStr === 'NO OVERRIDE') return null;
        const match = String(overrideStateStr).match(/Override:\s*(.+)/i);
        return match ? match[1].trim() : null;
      };
      
      const overrideCommand = parseOverrideCommand(overrideState);
      const overrideCommandLower = overrideCommand ? overrideCommand.toLowerCase() : '';
      
      // Check if override is for "off" state (ends with "off")
      const isOverrideOff = overrideCommand && overrideCommandLower.endsWith('off');

      // Override button classes
      let overrideBtnClass = 'override-btn';
      let overrideBtnIcon = 'mdi:hand-back-right-off';
      let overrideBtnClickable = false;
      if (isResettingOverride) {
        overrideBtnClass += ' resetting disabled';
        overrideBtnIcon = isOverrideOff ? 'mdi:hand-back-right-off' : 'mdi:hand-back-right';
      } else if (isOverridden) {
        overrideBtnClass += ' active';
        overrideBtnIcon = isOverrideOff ? 'mdi:hand-back-right-off' : 'mdi:hand-back-right';
        overrideBtnClickable = true;
      }

      // Format time for display (remove seconds if present)
      const formatTime = (timeStr) => {
        if (!timeStr || timeStr === '--:--') return '--:--';
        const stateLower = String(timeStr).toLowerCase();
        if (['unavailable', 'unknown', 'none'].includes(stateLower)) return '--:--';
        const parts = String(timeStr).split(':');
        if (parts.length < 2) return '--:--';
        return `${parts[0]}:${parts[1]}`;
      };

      // Determine if we can drag the handle (only in default mode, enabled, and not overridden)
      const canDragHandle = isEnabled && isDefaultMode && !isOverridden && displayTargetHours > 0;

      // Parse time to minutes for time picker
      const parseTimeToMinutes = (txt) => {
          if (!txt) return 420; // 07:00
          const parts = String(txt).split(':').map(Number);
          const h = parts[0] || 0, m = parts[1] || 0;
          return h * 60 + m;
      };
      
      const formatHm = (mins) => {
          if (mins == null) return '';
          const h = String(Math.floor(mins / 60)).padStart(2, '0');
          const m = String(mins % 60).padStart(2, '0');
          return `${h}:${m}`;
      };

      const finishTimeStr = sDefaultOnFinishTime?.state || '07:00:00';
      const finishTimeMins = this._localFinishTimeMins != null ? this._localFinishTimeMins : parseTimeToMinutes(finishTimeStr);

      // QS-194: precompute temperature row when sensor is configured
      // and currently reporting a valid numeric value. Falsy / `unknown` /
      // `unavailable` states are filtered out so we don't render `--°C`
      // for sensors that are temporarily offline.
      let tempRowHtml = '';
      if (sTemperatureSensor) {
        const rawTempState = sTemperatureSensor.state;
        const rawTempNum = Number(rawTempState);
        const tempUnit = sTemperatureSensor.attributes?.unit_of_measurement || '°C';
        // S3: exclude empty string — Number("") === 0 would otherwise
        // render an empty-state sensor as `0.0 °C`.
        if (
          rawTempState != null
          && rawTempState !== ''
          && rawTempState !== 'unknown'
          && rawTempState !== 'unavailable'
          && !Number.isNaN(rawTempNum)
        ) {
          tempRowHtml = `
        <div class="tank-temp">
          <ha-icon icon="mdi:thermometer-water"></ha-icon>
          <span>Water:</span>
          <span class="temp-value">${rawTempNum.toFixed(1)} ${this._escapeHtml(tempUnit)}</span>
        </div>`;
        }
      }

      this._root.innerHTML = `
      <ha-card class="card ${!isEnabled ? 'disabled' : ''} ${isOffGrid ? 'off-grid' : ''}">
        <style>${css}</style>
        <div class="card-title">${this._escapeHtml(title)}</div>
        ${tempRowHtml}
        <div class="top"></div>

        <div class="hero">
          <div class="ring">
            ${swEnableDevice ? `<div id="power_btn" class="power-btn ${isEnabled ? 'on' : ''}"><ha-icon icon="mdi:power"></ha-icon></div>` : ''}
            <svg viewBox="0 0 320 320" width="300" height="300" style="touch-action: none;" aria-hidden="true">
              <defs>
                <linearGradient id="${gradGreenId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="${colors.gradStart}"/>
                  <stop offset="100%" stop-color="${colors.gradEnd}"/>
                </linearGradient>
                <linearGradient id="${gradRunningId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="${colors.animStart}"/>
                  <stop offset="100%" stop-color="${colors.animEnd}"/>
                </linearGradient>
                <filter id="runningGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <clipPath id="${waterClipId}">
                  <circle cx="${CENTER_CX}" cy="${CENTER_CY}" r="${CLIP_R}" />
                </clipPath>
                <filter id="${surfaceGlowFilterId}" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="${SURFACE_GLOW_BLUR_STDDEV}" result="blur" />
                  <feFlood flood-color="${SURFACE_GLOW_COLOR}" flood-opacity="1" />
                  <feComposite in2="blur" operator="in" result="glow" />
                  <feMerge>
                    <feMergeNode in="glow" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="${steamFilterId}" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="${STEAM_BLUR_STDDEV}" />
                </filter>
              </defs>
              <g clip-path="url(#${waterClipId})">
                <path id="wave0_cool" d="${initialWavePaths[0]}" fill="${COOL_WATER_COLORS[0]}" opacity="${initialCoolOpacity}" pointer-events="none" style="will-change: transform; mix-blend-mode: normal;" />
                <path id="wave0_boil" d="${initialWavePaths[0]}" fill="${BOIL_WATER_COLORS[0]}" opacity="${initialBoilOpacity}" pointer-events="none" style="will-change: transform;" />
                <path id="wave1_cool" d="${initialWavePaths[1]}" fill="${COOL_WATER_COLORS[1]}" opacity="${initialCoolOpacity}" pointer-events="none" style="will-change: transform; mix-blend-mode: normal;" />
                <path id="wave1_boil" d="${initialWavePaths[1]}" fill="${BOIL_WATER_COLORS[1]}" opacity="${initialBoilOpacity}" pointer-events="none" style="will-change: transform;" />
                <path id="wave2_cool" d="${initialWavePaths[2]}" fill="${COOL_WATER_COLORS[2]}" opacity="${initialCoolOpacity}" pointer-events="none" style="will-change: transform; mix-blend-mode: normal;" />
                <path id="wave2_boil" d="${initialWavePaths[2]}" fill="${BOIL_WATER_COLORS[2]}" opacity="${initialBoilOpacity}" pointer-events="none" style="will-change: transform;" />
                <g id="${bubbleLayerId}"></g>
                <!-- review-fix #02 N4: surface_glow.d MUST equal wave 0's d on every frame.
                     First-frame parity is anchored by the shared initialWavePaths[0]
                     below and on wave0_cool/wave0_boil above; the RAF regen block then
                     resyncs d only when wave 0's geometry changes. -->
                <path id="surface_glow" d="${initialWavePaths[0]}" stroke="${SURFACE_GLOW_COLOR}" stroke-width="${SURFACE_GLOW_STROKE_WIDTH}" fill="none" filter="url(#${surfaceGlowFilterId})" opacity="${initialBoilOpacity}" pointer-events="none" style="mix-blend-mode: screen; will-change: transform, opacity, d;" />
                <g id="${steamLayerId}" filter="url(#${steamFilterId})" pointer-events="none"></g>
              </g>
              <path d="${bgPath}" stroke="var(--divider-color)" stroke-width="14" fill="none" stroke-linecap="round" />
              <path d="${progressPath}" stroke="url(#${activeGradId})" stroke-width="14" fill="none" stroke-linecap="round" ${showAnimation ? 'stroke-opacity="0.35"' : ''} />
              ${showAnimation ? `
              <path id="running_anim"
                    d="${progressPath}"
                    stroke="url(#${gradRunningId})"
                    stroke-width="16"
                    fill="none"
                    stroke-linecap="round"
                    stroke-dasharray="${dashLen} ${gapLen}"
                    stroke-opacity="1"
                    filter="url(#runningGlow)"
                    style="mix-blend-mode:screen; will-change: stroke-dashoffset"
              />
              ` : ''}
              ${canDragHandle ? `
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="${colors.primary}" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="${colors.primary}" font-size="13" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${this._fmt(pctToHours(handlePct))}</text>
              ` : ''}
            </svg>
            <div class="center">
              <div class="stack">
                <div class="target-block">
                  <div class="target-label">Actual / Target Hours</div>
                  <div class="target-value">
                    <span style="color: var(--primary-text-color);">${this._fmt(hoursRun, false)}h</span>
                    <span style="color: var(--primary-text-color);"> / </span>
                    <span style="color: ${colors.primary};">${this._fmt(displayTargetHours)}h</span>
                  </div>
                </div>
                ${showFromTo ? `
                <div class="from-to-row">
                  <div class="from-to-item">
                    <div class="from-to-label">From:</div>
                    <div class="from-to-value">${isDefaultMode && !isOverridden ? '--:--' : formatTime(startTime)}</div>
                  </div>
                  <div class="from-to-item">
                    <div class="from-to-label">To:</div>
                    <div class="from-to-value">${isDefaultMode && !isOverridden ? (sDefaultOnFinishTime ? formatTime(finishTimeStr) : '--:--') : formatTime(endTime)}</div>
                  </div>
                </div>
                ` : ''}
                ${isDefaultMode && !isOverridden && sDefaultOnFinishTime ? `
                <div class="center-controls" style="flex-direction:column; gap:4px;">
                  <div style="color: var(--secondary-text-color); font-weight:700; font-size: .75rem;">Change Finish Time</div>
                  <div id="time_btn" class="time-btn ${finishTimeStr && finishTimeStr !== '--:--' ? 'on' : ''}">${formatTime(finishTimeStr)}</div>
                </div>
                ` : ''}
              </div>
            </div>
            ${e.override_reset ? `<div id="override_btn" class="${overrideBtnClass}"><ha-icon icon="${overrideBtnIcon}"></ha-icon></div>` : ''}
            ${swGreenOnly ? `<div id="green_btn" class="green-btn ${swGreenOnly.state === 'on' ? 'on' : ''}"><ha-icon icon="mdi:leaf"></ha-icon></div>` : ''}
          </div>
        </div>

        ${selBistateMode ? `
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:toggle-switch"></ha-icon>
            <select id="bistate_mode">
              ${modeOptionsHtml}
            </select>
          </div>
        </div>
        ` : ''}
        ${e.reset ? `
        <div class="below-line full">
           <button id="reset" class="danger pill outline">Reset</button>
        </div>
        ` : ''}
      </ha-card>
    `;

      // QS-200: innerHTML rewrite just replaced the 6 wave <path> nodes,
      // the surface_glow, and the bubble-layer <g>. The new wave nodes
      // already carry the freshly-generated `d` (initialWavePaths) and
      // the correct `fill` (COOL_/BOIL_WATER_COLORS), so we sync RAF's
      // memo keys to the rendered state — prevents a redundant regen on
      // the next frame. Cached DOM refs are nulled so RAF re-resolves
      // them lazily. The logical `_bubbles` array now points to detached
      // elements; reset it so subsequent spawns don't try to advance
      // dead nodes (the next spawn fires within ~167ms — a barely-
      // perceptible blip on a hass push, accepted per pool precedent).
      // Sync the memo keys to the rendered state (prevents a redundant
      // regen on the next frame). DOM refs + logical bubble array are
      // reset via `_resetDomRefs()` so adding a new memo key in one
      // place automatically lands in both.
      this._lastWaterBaseY = this._waterBaseY;
      this._lastAmplitude = initialAmp;
      this._resetDomRefs();

      // Wire events
      const root = this._root;
      const ids = (k) => root.getElementById(k);

      const showDialog = (opts) => {
          const {title, message, buttons, customContent} = opts;
          // N12: an empty `buttons` array would render a modal with no
          // dismiss path. Always append a "Close" fallback so the user
          // can never get locked out (and `_modalOpen` doesn't wedge
          // re-renders).
          const safeButtons = (Array.isArray(buttons) && buttons.length > 0)
              ? buttons
              : [{ text: 'Close' }];
          const wrap = document.createElement('div');
          wrap.className = 'modal';
          // S6: escape user-controlled `title` and `message`; customContent
          // is provided by the caller as already-rendered HTML.
          const contentHtml = customContent || `<p>${this._escapeHtml(message)}</p>`;
          wrap.innerHTML = `<div class="dialog"><h3>${this._escapeHtml(title)}</h3>${contentHtml}<div class="actions"></div></div>`;
          const actions = wrap.querySelector('.actions');
          this._modalOpen = true;
          safeButtons.forEach(b => {
              const el = document.createElement('button');
              el.className = `btn ${b.variant || 'secondary'}`;
              el.textContent = b.text;
              let activated = false;
              const activate = () => {
                  if (activated) return;
                  activated = true;
                  // N13: wrap onClick in try/finally so a synchronous
                  // throw doesn't leave the modal locked open with
                  // `_modalOpen = true` blocking subsequent renders.
                  try {
                      if (b.onClick) b.onClick();
                  } finally {
                      wrap.remove();
                      this._modalOpen = false;
                      this._render();
                  }
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

      // Bistate mode selector
      if (selBistateMode) {
          const modeSel = ids('bistate_mode');
          const startM = () => {
              this._isInteractingMode = true;
          };
          const endM = () => {
              // Don't clear flag on blur during change processing
              if (!this._isProcessingModeChange) {
                  this._isInteractingMode = false;
                  this._render();
              }
          };
          modeSel?.addEventListener('focus', startM);
          modeSel?.addEventListener('blur', endM);
          modeSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return;

              this._isProcessingModeChange = true;
              // M2: wrap in try/finally so the cleanup setTimeout ALWAYS
              // runs — otherwise a rejected `_select` (HA service failure,
              // network drop) would leave `_isProcessingModeChange = true`
              // forever and silently lock out subsequent re-renders.
              try {
                  // Call the service and wait for it to complete
                  await this._select(e.bistate_mode, option);
              } catch (_) {
                  // swallow — HA state will resync on the next push
              } finally {
                  // Wait a bit for HA state to propagate, then allow re-render
                  setTimeout(() => {
                      this._isProcessingModeChange = false;
                      this._isInteractingMode = false;
                      this._render();
                  }, 300);
              }
          });
          const modePill = modeSel?.closest('.pill');
          if (modePill && modeSel) {
              modePill.addEventListener('click', (ev) => {
                  if (ev.target.tagName === 'SELECT') return;
                  try { modeSel.showPicker(); } catch (_) { modeSel.focus(); }
              });
          }
      }

      // Mobile touch fix: every button below uses a dual click + touchend pattern.
      // On mobile, the browser synthesizes "click" from touchstart/touchend with up to a
      // 300ms delay. If a hass re-render (innerHTML replacement) occurs in that window, the
      // DOM node is destroyed before the synthetic click fires, so the tap is lost. The
      // touchend handler fires immediately, calls preventDefault() to suppress the delayed
      // synthetic click (avoiding double-fire on desktop), and invokes the action directly.

      // S16 — keyboard activation helper: registers Enter/Space handlers
      // on a `role="button" tabindex="0"` div so keyboard-only users can
      // trigger the same action as click/touchend. Stops the default
      // Space-scroll behaviour and the synthetic click that would
      // double-fire otherwise.
      const _registerKeyActivation = (el, action) => {
          if (!el) return;
          el.addEventListener('keydown', (ev) => {
              if (ev.key === 'Enter' || ev.key === ' ') {
                  ev.preventDefault();
                  action();
              }
          });
      };

      // Green-only toggle button
      if (swGreenOnly) {
          const toggleGreen = async () => {
              const btn = ids('green_btn');
              try {
                  if (swGreenOnly.state === 'on') {
                      await this._turnOff(e.green_only);
                      btn?.classList.remove('on');
                  } else {
                      await this._turnOn(e.green_only);
                      btn?.classList.add('on');
                  }
              } catch (_) {
                  // ignore errors; HA state will resync UI on next render
              }
          };
          const gbtn = ids('green_btn');
          if (gbtn) {
              gbtn.style.pointerEvents = 'auto';
              gbtn.addEventListener('click', toggleGreen);
              gbtn.addEventListener('touchend', (ev) => { ev.preventDefault(); toggleGreen(); });
              _registerKeyActivation(gbtn, toggleGreen);
          }
      }

      // Power/Enable device toggle button
      if (swEnableDevice) {
          const togglePower = async () => {
              const btn = ids('power_btn');
              try {
                  if (swEnableDevice.state === 'on') {
                      await this._turnOff(e.enable_device);
                      btn?.classList.remove('on');
                  } else {
                      await this._turnOn(e.enable_device);
                      btn?.classList.add('on');
                  }
              } catch (_) {
                  // ignore errors; HA state will resync UI on next render
              }
          };
          const pbtn = ids('power_btn');
          if (pbtn) {
              pbtn.style.pointerEvents = 'auto';
              pbtn.addEventListener('click', togglePower);
              pbtn.addEventListener('touchend', (ev) => { ev.preventDefault(); togglePower(); });
              _registerKeyActivation(pbtn, togglePower);
          }
      }

      // Override button
      if (e.override_reset && overrideBtnClickable) {
          const obtn = ids('override_btn');
          if (obtn) {
              obtn.style.pointerEvents = 'auto';
              const obtnAction = async () => {
                  showDialog({
                      title: 'Reset override',
                      message: 'This will reset the manual override and return to automatic mode.\nProceed?',
                      buttons: [
                          {text: 'Cancel', variant: 'secondary'},
                          {
                              text: 'Reset', variant: 'primary', onClick: async () => {
                                  await this._press(e.override_reset);
                              }
                          },
                      ]
                  });
              };
              obtn.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); obtnAction(); });
              obtn.addEventListener('touchend', (ev) => { ev.preventDefault(); obtnAction(); });
              _registerKeyActivation(obtn, obtnAction);
          }
      }

      // Time button for finish time (default mode only, hidden during override)
      if (isDefaultMode && !isOverridden && sDefaultOnFinishTime) {
          const timeAction = async () => {
              const defaultHour = Math.floor(finishTimeMins / 60);
              const defaultMin = finishTimeMins % 60;

              const customContent = `
            <p>Select the time the device should finish by:</p>
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
                  title: 'Finish Time',
                  customContent: customContent,
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
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
                              this._localFinishTimeMins = mins;
                              // S9: clear the local override after a
                              // grace period so out-of-band backend
                              // updates aren't masked indefinitely.
                              // Mirrors the _localTargetPct timeout.
                              if (this._localFinishTimeClearTimer) {
                                  clearTimeout(this._localFinishTimeClearTimer);
                              }
                              this._localFinishTimeClearTimer = setTimeout(() => {
                                  this._localFinishTimeMins = null;
                                  this._render();
                              }, 5000);
                              await this._setTime(e.default_on_finish_time, val);
                          }
                      },
                  ]
              });
          };

          const tbtn = ids('time_btn');
          if (tbtn) {
              tbtn.style.pointerEvents = 'auto';
              tbtn.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); timeAction(); });
              tbtn.addEventListener('touchend', (ev) => { ev.preventDefault(); timeAction(); });
              _registerKeyActivation(tbtn, timeAction);
          }
      }

      // Reset button
      if (e.reset) {
          const resetBtn = ids('reset');
          const resetAction = async () => {
              showDialog({
                  title: 'Reset device state',
                  message: 'This will reset internal state for the device and cannot be undone.\nProceed?',
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

      // Drag target handle on ring (only in default mode)
      if (canDragHandle) {
          const svg = this._root.querySelector('.ring svg');
          const handle = this._root.getElementById('target_handle');
          if (svg && handle) {
              const pt = svg.createSVGPoint();
              
              // Allowed hours (snap points) - 0.5 hour increments from 0 to 12
              const allowedHours = [];
              for (let i = 0; i <= 12; i += 0.5) {
                  allowedHours.push(i);
              }
              
              const onMove = (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  const e2 = ev.touches ? ev.touches[0] : ev;
                  pt.x = e2.clientX;
                  pt.y = e2.clientY;
                  const cursor = pt.matrixTransform(svg.getScreenCTM().inverse());
                  const dx = cursor.x - center.cx;
                  const dy = cursor.y - center.cy;
                  let ang = rad2deg(Math.atan2(-dy, dx));
                  let a = ang;
                  if (a < startDeg) a = startDeg;
                  if (a > endDeg) a = endDeg;
                  const rawPct = ((a - startDeg) / rangeDeg) * 100;
                  const rawHours = pctToHours(rawPct);
                  
                  // Snap to nearest allowed hour
                  const snapHours = allowedHours.reduce((best, v) => 
                      Math.abs(v - rawHours) < Math.abs(best - rawHours) ? v : best, 
                      allowedHours[0]
                  );

                  const displayPct = hoursToPct(snapHours);
                  this._targetDragPct = displayPct;
                  this._targetDragValue = snapHours;
                  this._isInteractingTarget = true;

                  const angSnap = startDeg + (displayPct / 100) * rangeDeg;
                  const pos = polar(center.cx, center.cy, ringCirc, angSnap);
                  handle.setAttribute('cx', pos.x.toFixed(2));
                  handle.setAttribute('cy', pos.y.toFixed(2));
                  const handleText = this._root.getElementById('target_handle_text');
                  if (handleText) {
                      handleText.setAttribute('x', pos.x.toFixed(2));
                      handleText.setAttribute('y', pos.y.toFixed(2));
                      handleText.textContent = this._fmt(snapHours, false);
                  }
                  const tv = this._root.querySelector('.target-value');
                  if (tv) {
                      tv.innerHTML = `<span style="color: var(--primary-text-color);">${this._fmt(hoursRun, false)}h</span><span style="color: var(--primary-text-color);"> / </span><span style="color: ${colors.primary};">${this._fmt(snapHours, false)}h</span>`;
                  }
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

                  // S17 — wrap the service call so the drag-release
                  // guards always clear, even if `_setNumber` throws.
                  try {
                      if (dragValue != null && e.default_on_duration) {
                          await this._setNumber(e.default_on_duration, dragValue);
                          this._localTargetPct = dragPct;
                          this._pendingClearLocalTarget && clearTimeout(this._pendingClearLocalTarget);
                          this._pendingClearLocalTarget = setTimeout(() => {
                              this._localTargetPct = null;
                              this._pendingClearLocalTarget = null;
                              this._render();
                          }, 5000);
                      }
                  } catch (_) {
                      // swallow — HA state will resync on the next push
                  } finally {
                      this._targetDragPct = null;
                      this._targetDragValue = null;
                      this._isInteractingTarget = false;
                      this._upInProgress = false;
                      handle.style.cursor = 'grab';
                  }
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
}

if (!customElements.get('qs-water-boiler-card')) {
    customElements.define('qs-water-boiler-card', QsWaterBoilerCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-water-boiler-card')) {
    window.customCards.push({
        type: 'qs-water-boiler-card',
        name: 'QS Water Boiler Card',
        description: 'Quiet Solar water boiler (cumulus / thermodynamic) control card',
    });
}
