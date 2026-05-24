/*
  QS Climate Card - custom:qs-climate-card
  Zero-build single-file Lit-style web component compatible with Home Assistant

  QS-210 (issue #210): the card now selects one of four backdrop visuals
  for the central ring based on the configured HVAC ON mode plus, for
  AUTO / HEAT_COOL, the backing climate entity's current temperature vs.
  resolved setpoint:

    - HEAT  → "flame" (3-layer peaked-teeth flame engine, copy of
              qs-radiator-card.js QS-204)
    - COOL  → "snow"  (3-layer wave-pile + falling snowflakes, adapted
              from qs-pool-card.js waves and qs-water-boiler-card.js
              bubble particle system — INVERTED: spawn at top, fall down,
              retire when they hit the pile)
    - AUTO / HEAT_COOL → "flame" or "snow" based on target > current,
              or "wind" (3 stroked sinusoidal wisps, linear scroll) when
              the climate entity exposes no resolvable setpoint AND the
              device is running, or "none" when not running.
    - everything else (fan_only / dry / off / unrecognised / null) →
              "none" (no backdrop) — short-circuits BEFORE any
              climate-entity attribute read.

  All four engines duplicate code already present in sibling cards
  (flame / wave / bubble) rather than extracting a shared base. The
  refactor is tracked at issue #199 and will collapse all four cards
  (radiator + pool + boiler + climate) in one PR. The radiator card's
  QS-195 review-fix #01 sets the precedent.
*/

// --- Geometry (must match the SVG <clipPath> circle attributes below) ---
const CENTER_CY = 160;              // SVG y-centre of the ring / clip circle
// QS-210 review-fix S5 — explicit X-centre constant. The viewBox is
// square (320×320) so `CENTER_X === CENTER_CY === 160` numerically,
// but a separate name reads correctly and survives a future
// non-square viewBox.
const CENTER_X = 160;               // SVG x-centre of the ring / clip circle
const CLIP_R = 120;                 // backdrop clip circle radius

// QS-217 — Override-button cover overlay geometry.
// Review-fix #03 abandons the earlier clipPath carve approach
// (which created a geometric lens-shape hole — intersection of the
// carve disc and the outer clip disc). Instead, a simple `<circle>`
// cover with `fill="var(--card-background-color)"` is drawn ON TOP
// of the clipped animation group, visually erasing the animation
// in a clean circular patch behind the button.
// Anchored to the CSS `.override-btn` at `position: absolute;
// bottom: 15px; left: 50%; transform: translateX(-50%); width: 50px;
// height: 50px;` inside `.ring` (300×300 CSS px). The SVG viewBox
// (320×320) renders at 300×300 → scale 320/300 ≈ 1.0667.
// Button centre CSS px (within .ring): (150, 260). Button centre
// SVG units: (160, 277.33) → rounded to (CENTER_X, 277). Cover
// radius R = 35 SVG ≈ 33 CSS px ⇒ ~8 CSS px padding around the
// 25 CSS-px-radius button outline. User-tunable.
// The cover applies uniformly across all backdrops (flame / snow /
// wind / none) — same DN-3 rationale as the original carve.
const OVERRIDE_BTN_CARVE_CY = 277;
const OVERRIDE_BTN_CARVE_R  = 35;

// --- Flame engine constants (QS-204 verbatim from qs-radiator-card.js) ---
const FLAME_WIDTH = 480;            // single layer width in SVG px (2× clip diameter)
const FLAME_BOTTOM_Y = 400;         // ≥ SVG viewBox max-y (320) so the closing rect is clipped
const LAYER_TEETH_COUNTS = [3, 4, 5];
const LAYER_TIP_FLICKER_HZ = [8, 7, 9];
const LAYER_BASE_HEIGHTS = [150, 120, 90];
const LAYER_TIP_AMP_MULTS = [1.2, 1.0, 0.8];
const STILL_AMP = 0;
const DANCE_AMP = 8;
const STATIC_PEAK_HEIGHT = 30;
const FLAME_BASE_MIN_PCT = 0.2;
const FLAME_BASE_MAX_PCT = 0.8;
const FLAME_FILLS = [
    'rgba(255, 87, 34, 0.55)',
    'rgba(255, 110, 64, 0.45)',
    'rgba(255, 193,  7, 0.35)',
];
const FLAME_GREY_FILLS = [
    'rgba(160, 160, 160, 0.40)',
    'rgba(140, 140, 140, 0.30)',
    'rgba(120, 120, 120, 0.22)',
];

// QS-204 review-fix #02 G7 — length-equality guard for the per-layer
// arrays. A future PR extending one without the others would silently
// produce NaN paths (out-of-bounds reads → undefined → arithmetic →
// NaN). `console.assert` is the browser-side equivalent of a Python
// `assert`.
console.assert(
    LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length &&
    LAYER_TEETH_COUNTS.length === LAYER_BASE_HEIGHTS.length &&
    LAYER_TEETH_COUNTS.length === LAYER_TIP_AMP_MULTS.length,
    "qs-climate-card: LAYER_* constants must be the same length"
);

// --- Animation tuning (shared across flame / snow / wind) ---
const LERP_RATE = 2;                // exp time-constant; ~95% of lerp in ~1.5s
const LERP_DT_CEIL = 0.1;           // s; clamp lerp dt against hidden-tab return
const AMP_REGEN_THRESHOLD = 0.25;   // amplitude delta threshold for path regen
const LEVEL_REGEN_THRESHOLD = 0.01; // px; base-Y threshold for path regen
const PHASE_REGEN_MIN_DT = 0.20;    // s; running-mode regen throttle (flame)
const PHASE_WRAP = 1e6;             // wrap phase accumulator to preserve float precision
const PHASE_TO_PX = 60;             // scroll px per phase unit (snow / wind)

// --- Snow engine constants (waves adapted from qs-pool-card.js, particles
//     inverted from qs-water-boiler-card.js bubbles) ---
const WAVE_WIDTH = 480;
const WAVE_BOTTOM_Y = 400;
const LAYER_PHASE_OFFSET = 2.1;
const LAYER_SCROLL_OFFSET = 1.2;
const SNOW_BACK_COLOR  = 'hsla(220, 60%, 82%, 0.55)';
const SNOW_MID_COLOR   = 'hsla(210, 50%, 88%, 0.45)';
const SNOW_FRONT_COLOR = 'hsla(210, 30%, 92%, 0.45)';
const SNOW_PILE_COLORS = [SNOW_BACK_COLOR, SNOW_MID_COLOR, SNOW_FRONT_COLOR];
// Snow doesn't agitate like boiling water — smaller deltas than pool's
// CALM_AMP=2 / PUMP_AMP=6.
const CALM_SNOW_AMP = 1.5;
const RUN_SNOW_AMP = 2.5;
const CALM_SNOW_SPEED = 0.2;
const RUN_SNOW_SPEED = 0.5;
// Snowflake particle system — INVERTS the boiler bubble system: spawn at
// top of clip (`CENTER_CY - CLIP_R + 8`), positive vy (falling), retire
// on `cy >= surfaceY` (hit the pile) or `life >= maxLife`.
const MAX_CONCURRENT_SNOWFLAKES = 14;
const SNOW_SPAWN_RATE_HZ = 5;
const SNOW_RADIUS_MIN = 1.5;
const SNOW_RADIUS_MAX = 4;
const SNOW_SPEED_PX_PER_S_MIN = 30;
const SNOW_SPEED_PX_PER_S_MAX = 60;
const SNOW_MAX_LIFE_S = 3.0;
const SNOW_FILL_COLOR = 'hsla(210, 30%, 92%, 0.85)';

// --- Wind engine constants (minimal — stroked sinusoidal wisps with a
//     linear scroll, no lerp envelope, no path regen) ---
const WIND_WISP_WIDTH = 480;
const WIND_SPEED_PX_PER_S = 60;
const WIND_LAYER_SCROLL_OFFSET = 1.2;
const WIND_WISP_Y_OFFSETS = [-30, 0, 30];   // relative to CENTER_CY
const WIND_WISP_COLORS = [
    'hsla(200, 30%, 85%, 0.45)',
    'hsla(200, 35%, 80%, 0.35)',
    'hsla(200, 40%, 75%, 0.25)',
];
const WIND_WAVE_AMP = 8;
const WIND_WAVE_FREQS = [2, 3, 4];          // cycles per WIND_WISP_WIDTH (integers → seamless wrap)

// QS-210 review-fix S4 — backdrop hysteresis. Strict
// `target > currentTemp ? 'flame' : 'snow'` flips on every ±0.1°C
// thermostat jitter at equilibrium. Treat `|target - current| <
// BACKDROP_DEADBAND_C` as "stay where we are" — keep the previous
// resolved backdrop ('flame' or 'snow'); if there is none yet, default
// to 'flame' (heating is the more common AUTO use case).
const BACKDROP_DEADBAND_C = 0.2;

class QsClimateCard extends HTMLElement {
  // M4: gate the requestAnimationFrame loop on `showAnimation` (the
  // dashed-arc) OR `this._backdrop !== 'none'` (any backdrop visual
  // needs the loop). The loop is started lazily by `_render()` and
  // stopped in `disconnectedCallback` and whenever both conditions
  // become false. Idle cards consume zero per-frame work.
  //
  // QS-210: the step function dispatches on `this._backdrop` so the
  // four visuals share one RAF loop. Backdrop-specific state is lazily
  // initialised on first use; the cache-invalidate helpers below clear
  // DOM refs (NOT animation accumulators) after each innerHTML rewrite
  // so re-renders don't visibly snap.
  _startAnimation() {
    // Pass-#2 N4 — early-return BEFORE the lazy-init so the init
    // blocks only execute when actually starting a fresh RAF loop.
    // (`_startAnimation` is called on every `_render` via the
    // umbrella `showAnimation` gate; re-entries while RAF is running
    // shouldn't pay the lazy-init cost.)
    if (this._animRaf != null) return;

    // Pass-#2 M1 — per-field lazy-init guards. The legacy single
    // guard `if (this._currentSnowAmp == null) { … five fields … }`
    // was fragile: a render-side write to `_currentSnowAmp` (e.g.,
    // the N5 transition reset) bypassed initialisation of the other
    // four snow fields, leaving `_snowflakes` / `_snowWavePhase` /
    // `_nextSnowflakeAt` as `undefined` and crashing the first RAF
    // tick of `_stepSnow` with `TypeError: undefined is not
    // iterable`. Per-field guards make the init robust regardless
    // of which field is primed first.
    if (this._currentFlameAmp == null) this._currentFlameAmp = STILL_AMP;
    if (this._tipPhases == null) {
      this._tipPhases = LAYER_TEETH_COUNTS.map((count) => new Array(count).fill(0));
    }
    if (this._needsFlamePrime == null) this._needsFlamePrime = true;
    if (this._currentSnowAmp == null) this._currentSnowAmp = CALM_SNOW_AMP;
    if (this._currentSnowSpeed == null) this._currentSnowSpeed = CALM_SNOW_SPEED;
    if (this._snowWavePhase == null) this._snowWavePhase = 0;
    if (this._snowflakes == null) this._snowflakes = [];
    if (this._nextSnowflakeAt == null) this._nextSnowflakeAt = 0;
    if (this._windPhase == null) this._windPhase = 0;

    // Pass-#2 N2 — drop the three `_invalidate*Cache()` calls that
    // used to live here. The post-innerHTML M1 block (at the tail of
    // `_render`) now invalidates all three caches unconditionally, so
    // the same calls here were redundant CPU work; worse,
    // `_invalidateSnowCache`'s `.remove()` was detaching live nodes
    // that the imminent rewrite was about to replace anyway.

    this._lastAnimTs = null;

    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      let dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      // QS-200 S6: cap dt against hidden-tab return.
      dt = Math.min(dt, LERP_DT_CEIL);
      this._lastAnimTs = ts;
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const speed = 80; // dash units per second
      this._animOffset = ((this._animOffset || 0) + speed * dt) % patternLen;
      const p = this._root?.getElementById('running_anim');
      if (p) {
        p.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }

      switch (this._backdrop) {
        case 'flame':
          this._stepFlame(ts, dt);
          break;
        case 'snow':
          this._stepSnow(ts, dt);
          break;
        case 'wind':
          this._stepWind(ts, dt);
          break;
        // 'none' → only dashed-arc runs above.
      }

      this._animRaf = requestAnimationFrame(step);
    };
    this._animRaf = requestAnimationFrame(step);
  }

  // Review-fix N8 — animation accumulators (`_currentFlameAmp`,
  // `_currentSnowAmp`, `_currentSnowSpeed`, `_snowWavePhase`,
  // `_windPhase`, `_tipPhases`) deliberately survive `_stopAnimation`
  // so the loop resumes seamlessly when `showAnimation` flips back to
  // true (e.g. user toggles the device on after a long off-period).
  // The DOM-ref caches DO get cleared in the post-innerHTML M1 block
  // and on `disconnectedCallback`; only RAF bookkeeping is reset here.
  _stopAnimation() {
    if (this._animRaf != null) cancelAnimationFrame(this._animRaf);
    this._animRaf = null;
    this._lastAnimTs = null;
  }

  // --- Flame engine (copy of qs-radiator-card.js QS-204 step block) ---
  _stepFlame(ts, dt) {
    const fireOn = this._running === true;
    const targetAmp = fireOn ? DANCE_AMP : STILL_AMP;
    const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
    this._currentFlameAmp += (targetAmp - this._currentFlameAmp) * lerpFactor;
    if (!fireOn && Math.abs(this._currentFlameAmp) < 0.05) {
      this._currentFlameAmp = 0;
    }
    if (fireOn && Math.abs(this._currentFlameAmp - DANCE_AMP) < 0.05) {
      this._currentFlameAmp = DANCE_AMP;
    }

    // Advance per-layer, per-tooth tip phases ONLY when running.
    if (fireOn && this._tipPhases) {
      for (let i = 0; i < LAYER_TEETH_COUNTS.length; i++) {
        const phasesForLayer = this._tipPhases[i];
        const phaseStep = 2 * Math.PI * LAYER_TIP_FLICKER_HZ[i] * dt;
        for (let j = 0; j < phasesForLayer.length; j++) {
          phasesForLayer[j] = (phasesForLayer[j] + phaseStep * (1 + 0.07 * j)) % (2 * Math.PI);
        }
      }
    }

    if (!this._flameEls) {
      this._flameEls = Array.from(
          {length: LAYER_TEETH_COUNTS.length},
          (_, i) => this._root?.getElementById(`flame${i}`) ?? null,
      );
    }

    const flameBaseY = this._flameBaseY;
    const hasValidBase = flameBaseY != null && !Number.isNaN(flameBaseY);
    const ampDelta = this._lastFlameAmp == null
        ? Infinity
        : Math.abs(this._currentFlameAmp - this._lastFlameAmp);
    const levelChanged = hasValidBase &&
        Math.abs(flameBaseY - (this._lastFlameBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
    const sinceLastRegen = (ts - (this._lastFlameRegenTs ?? -Infinity)) / 1000;
    const phaseChanged = fireOn && sinceLastRegen >= PHASE_REGEN_MIN_DT;
    const shouldRegen = hasValidBase && (
        phaseChanged
        || levelChanged
        || ampDelta > AMP_REGEN_THRESHOLD
    );
    if (shouldRegen) {
      this._lastFlameBaseY = flameBaseY;
      this._lastFlameAmp = this._currentFlameAmp;
      this._lastFlameRegenTs = ts;
      const flameColors = this._flameColors || FLAME_GREY_FILLS;
      for (let i = 0; i < LAYER_TEETH_COUNTS.length; i++) {
        const fEl = this._flameEls[i];
        if (fEl) {
          const d = this._generateFlameTeethPath(
              FLAME_WIDTH,
              flameBaseY,
              LAYER_BASE_HEIGHTS[i],
              this._currentFlameAmp * LAYER_TIP_AMP_MULTS[i],
              LAYER_TEETH_COUNTS[i],
              this._tipPhases ? this._tipPhases[i] : null,
              !fireOn,
          );
          fEl.setAttribute('d', d);
          fEl.setAttribute('fill', flameColors[i]);
        }
      }
    }
  }

  // --- Snow engine (waves adapted from pool, particles inverted from boiler bubbles) ---
  _stepSnow(ts, dt) {
    const snowing = this._running === true;
    const targetAmp = snowing ? RUN_SNOW_AMP : CALM_SNOW_AMP;
    const targetSpeed = snowing ? RUN_SNOW_SPEED : CALM_SNOW_SPEED;
    const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
    this._currentSnowAmp += (targetAmp - this._currentSnowAmp) * lerpFactor;
    this._currentSnowSpeed += (targetSpeed - this._currentSnowSpeed) * lerpFactor;
    this._snowWavePhase = ((this._snowWavePhase + this._currentSnowSpeed * dt) % PHASE_WRAP + PHASE_WRAP) % PHASE_WRAP;

    if (!this._snowWaveEls) {
      this._snowWaveEls = [
          this._root?.getElementById('snowWave0') ?? null,
          this._root?.getElementById('snowWave1') ?? null,
          this._root?.getElementById('snowWave2') ?? null,
      ];
    }
    const snowLayer = this._snowLayerEl
        ?? (this._snowLayerEl = this._snowLayerId
              ? (this._root?.getElementById(this._snowLayerId) ?? null)
              : null);

    // Per-layer translateX scroll.
    for (let i = 0; i < 3; i++) {
      const wEl = this._snowWaveEls[i];
      if (wEl) {
        const phaseOffset = i * LAYER_SCROLL_OFFSET;
        const raw = (this._snowWavePhase + phaseOffset) * PHASE_TO_PX;
        const scrollOffset = ((raw % WAVE_WIDTH) + WAVE_WIDTH) % WAVE_WIDTH;
        const tx = -CLIP_R - scrollOffset;
        wEl.style.transform = `translateX(${tx.toFixed(1)}px)`;
      }
    }

    // Path regen throttled by amp / base-Y delta.
    const snowBaseY = this._snowBaseY;
    const hasValidBase = snowBaseY != null && !Number.isNaN(snowBaseY);
    const ampDelta = this._lastSnowAmp == null
        ? Infinity
        : Math.abs(this._currentSnowAmp - this._lastSnowAmp);
    const levelChanged = hasValidBase &&
        Math.abs(snowBaseY - (this._lastSnowBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
    if (hasValidBase && (levelChanged || ampDelta > AMP_REGEN_THRESHOLD)) {
      this._lastSnowBaseY = snowBaseY;
      this._lastSnowAmp = this._currentSnowAmp;
      for (let i = 0; i < 3; i++) {
        const wEl = this._snowWaveEls[i];
        if (wEl) {
          const phaseOffset = i * LAYER_PHASE_OFFSET;
          const freq = 2 + i;
          const d = this._generateWavePath(WAVE_WIDTH, this._currentSnowAmp, freq, phaseOffset, snowBaseY);
          wEl.setAttribute('d', d);
          wEl.setAttribute('fill', SNOW_PILE_COLORS[i]);
        }
      }
    }

    // === Snowflake particle system (three-way inverted from boiler bubbles) ===
    if (snowLayer) {
      // Spawn cadence (only while running, capped at MAX_CONCURRENT_SNOWFLAKES).
      // Snowflakes spawn at the TOP of the clip (inverted from bubbles).
      if (snowing) {
        this._nextSnowflakeAt -= dt;
        while (this._nextSnowflakeAt <= 0 && this._snowflakes.length < MAX_CONCURRENT_SNOWFLAKES) {
          // Review-fix S5 — bias spawn `cx` toward the actual visible
          // chord at the spawn-y (the clip is a CIRCLE, not the
          // bounding box). Without this, ~62% of spawns at spawn-y =
          // CENTER_CY - CLIP_R + 8 land outside the clipped region
          // and are invisible; the effective concurrent-flake count
          // drops to ~5 of MAX_CONCURRENT_SNOWFLAKES = 14.
          const spawnYOffset = 8;
          const dyFromCenter = (CENTER_CY - CLIP_R + spawnYOffset) - CENTER_CY;
          const halfChord = Math.sqrt(Math.max(0, CLIP_R * CLIP_R - dyFromCenter * dyFromCenter));
          const cx = CENTER_X + (Math.random() * 2 - 1) * Math.max(8, halfChord - 4);
          const cy = CENTER_CY - CLIP_R + spawnYOffset;
          const r = SNOW_RADIUS_MIN + Math.random() * (SNOW_RADIUS_MAX - SNOW_RADIUS_MIN);
          const vy = SNOW_SPEED_PX_PER_S_MIN + Math.random() * (SNOW_SPEED_PX_PER_S_MAX - SNOW_SPEED_PX_PER_S_MIN);
          const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          el.setAttribute('cx', cx.toFixed(2));
          el.setAttribute('cy', cy.toFixed(2));
          el.setAttribute('r', r.toFixed(2));
          el.setAttribute('fill', SNOW_FILL_COLOR);
          el.setAttribute('pointer-events', 'none');
          el.setAttribute('opacity', '0.9');
          snowLayer.appendChild(el);
          this._snowflakes.push({el, cx, cy, r, vy, life: 0, maxLife: SNOW_MAX_LIFE_S});
          this._nextSnowflakeAt += 1 / SNOW_SPAWN_RATE_HZ;
        }
        if (this._nextSnowflakeAt < 0) this._nextSnowflakeAt = 0;
      }

      // Advance + retire active snowflakes regardless of running state.
      //
      // QS-216 — cross-render flakes NOW survive same-backdrop
      // re-renders: `_render()` snapshots `_snowflakes` BEFORE the
      // innerHTML rewrite and re-attaches each `<circle>` to the
      // fresh snow layer AFTER `_invalidateSnowCache()` (see AC-2).
      // Flakes are still cleared on `disconnectedCallback` and on
      // real backdrop transitions away from 'snow' (restore-block
      // null-layer branch).
      // Pass-#3 N2 — when `_snowBaseY` is null (no valid pile base
      // computed yet, e.g. cold-start before the first render finishes
      // populating snow geometry), fall back to the BOTTOM of the
      // clip (`CENTER_CY + CLIP_R`) rather than the ring centre.
      // With a centre fallback, flakes spawned at the top
      // (`CENTER_CY - CLIP_R + 8 ≈ 48 px`) had to traverse only
      // ~108 px before retiring on the centre "surface", which at
      // 30-60 px/s puts them at ~2.4s of life — close to
      // `SNOW_MAX_LIFE_S = 3.0` and visibly truncated. Falling to the
      // bottom of the clip lets them traverse the full visible chord
      // so the cosmetic effect lands.
      const surfaceY = (this._snowBaseY ?? (CENTER_CY + CLIP_R)) - 4;
      const alive = [];
      for (const b of this._snowflakes) {
        b.life += dt;
        // INVERTED vs boiler: positive vy → cy increases (falling).
        b.cy += b.vy * dt;
        if (b.cy >= surfaceY || b.life >= b.maxLife) {
          b.el.remove();
          continue;
        }
        const lifeT = b.life / b.maxLife;
        const opacity = Math.max(0, 1 - lifeT * 0.5);
        b.el.setAttribute('cy', b.cy.toFixed(2));
        b.el.setAttribute('opacity', opacity.toFixed(3));
        alive.push(b);
      }
      this._snowflakes = alive;
    }
  }

  // --- Wind engine (minimal — 3 stroked wisps, linear translateX scroll) ---
  _stepWind(ts, dt) {
    // QS-210 review-fix C-IMP-2 — modulo wrap against WIND_WISP_WIDTH
    // prevents float drift over long sessions. WIND_SPEED_PX_PER_S is
    // the visible scroll speed; convert to phase units via PHASE_TO_PX.
    this._windPhase += (WIND_SPEED_PX_PER_S / PHASE_TO_PX) * dt;
    this._windPhase = ((this._windPhase % PHASE_WRAP) + PHASE_WRAP) % PHASE_WRAP;

    if (!this._windWispEls) {
      this._windWispEls = [
          this._root?.getElementById('windWisp0') ?? null,
          this._root?.getElementById('windWisp1') ?? null,
          this._root?.getElementById('windWisp2') ?? null,
      ];
    }

    for (let i = 0; i < 3; i++) {
      const wispEl = this._windWispEls[i];
      if (wispEl) {
        const phaseOffset = i * WIND_LAYER_SCROLL_OFFSET;
        const raw = (this._windPhase + phaseOffset) * PHASE_TO_PX;
        const scrollOffset = ((raw % WIND_WISP_WIDTH) + WIND_WISP_WIDTH) % WIND_WISP_WIDTH;
        const tx = -CLIP_R - scrollOffset;
        wispEl.style.transform = `translateX(${tx.toFixed(1)}px)`;
      }
    }
  }

  // --- Path generators (verbatim from sibling cards) ---

  // QS-204 verbatim: SVG path for a single flame layer (peaked teeth).
  _generateFlameTeethPath(width, baseY, peakHeight, tipAmp, numTeeth, tipPhases, isIdle) {
    const teethCount = Math.max(1, numTeeth | 0);
    const toothWidth = width / teethCount;
    const idlePeakBoost = isIdle ? STATIC_PEAK_HEIGHT : 0;
    let d = `M 0 ${baseY.toFixed(2)}`;
    for (let i = 0; i < teethCount; i++) {
      const phase = tipPhases && tipPhases[i] != null ? tipPhases[i] : 0;
      const tipWobble = isIdle ? 0 : tipAmp * Math.sin(phase);
      const peakY = baseY - peakHeight - idlePeakBoost - tipWobble;
      const startX = i * toothWidth;
      const midX = startX + toothWidth / 2;
      const endX = startX + toothWidth;
      const ctrlUpX = startX + toothWidth / 3;
      const ctrlDownX = startX + (2 * toothWidth) / 3;
      const ctrlY = peakY - 8;
      d += ` Q ${ctrlUpX.toFixed(2)} ${ctrlY.toFixed(2)} ${midX.toFixed(2)} ${peakY.toFixed(2)}`;
      d += ` Q ${ctrlDownX.toFixed(2)} ${ctrlY.toFixed(2)} ${endX.toFixed(2)} ${baseY.toFixed(2)}`;
    }
    d += ` L ${width.toFixed(2)} ${FLAME_BOTTOM_Y} L 0 ${FLAME_BOTTOM_Y} Z`;
    return d;
  }

  // Pool-card verbatim: closed wave-shape path (two repetitions wide).
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

  // QS-210 new: OPEN sinusoidal path (no closing L → L → Z). Stroked
  // line, not filled polygon. Two repetitions wide so the translated
  // path always covers the clip.
  _generateWispPath(width, amplitude, frequency, phase, yOffset) {
    const points = [];
    const stepsPerPeriod = 60;
    const totalSteps = stepsPerPeriod * 2;
    for (let i = 0; i <= totalSteps; i++) {
      const x = (i / stepsPerPeriod) * width;
      const y = yOffset + amplitude * Math.sin((x / width) * frequency * 2 * Math.PI + phase);
      points.push(`${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    return `M ${points[0]} ` + points.slice(1).map(p => `L ${p}`).join(' ');
  }

  // --- Cache invalidators (cleared after each innerHTML rewrite) ---
  _invalidateFlameCache() {
    this._flameEls = null;
    this._lastFlameBaseY = null;
    this._lastFlameAmp = null;
    this._lastFlameRegenTs = -Infinity;
  }

  _invalidateSnowCache() {
    this._snowWaveEls = null;
    this._snowLayerEl = null;
    this._lastSnowBaseY = null;
    this._lastSnowAmp = null;
    // QS-216 — same-backdrop re-renders preserve flakes via
    // _render()'s snapshot/restore block (AC-2); this reset only
    // governs `disconnectedCallback` and real backdrop transitions
    // away from 'snow'. Critical invariant (AC-3): do NOT null
    // `b.el` — the truthy-branch restore filter `(b => b?.el)`
    // would silently drop anything we null here.
    if (this._snowflakes) {
      for (const b of this._snowflakes) {
        b?.el?.remove?.();
      }
      this._snowflakes = [];
    }
    this._nextSnowflakeAt = 0;
  }

  // Review-fix N6 — wind has no path-regen state (constant scroll, no
  // amp envelope, no per-layer geometry to memoize), so only the DOM
  // ref needs clearing. Animation accumulator (`_windPhase`) survives
  // so re-attach picks up where it left off.
  _invalidateWindCache() {
    this._windWispEls = null;
  }

  connectedCallback() {
    // RAF intentionally NOT started here — _render() will call
    // _startAnimation() when needed.
  }

  disconnectedCallback() {
    this._stopAnimation();
    // QS-210: clear backdrop DOM caches and tear down snowflake nodes.
    // Animation accumulators (`_currentFlameAmp`, `_currentSnowAmp`,
    // `_currentSnowSpeed`, `_snowWavePhase`, `_windPhase`, `_tipPhases`)
    // deliberately survive disconnect so re-attach resumes without a
    // visible snap.
    this._invalidateFlameCache();
    this._invalidateSnowCache();
    this._invalidateWindCache();
    // S7: reset interaction flags so a re-attach after mid-interaction
    // (e.g. dragging the ring or processing a mode change when the
    // dashboard rearranges) doesn't silently short-circuit `set hass`
    // on stale flags.
    this._isInteractingMode = false;
    this._isInteractingStateOn = false;
    this._isInteractingTarget = false;
    this._isProcessingModeChange = false;
    this._isProcessingStateOnChange = false;
    this._modalOpen = false;
  }

  static getStubConfig() {
    return { name: "QS Climate", entities: {} };
  }

  setConfig(config) {
    if (!config || !config.entities) throw new Error("entities is required");
    this._config = config;
    this._root = this.attachShadow({ mode: "open" });
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
  //
  // QS-210 review-fix S6 / S7:
  //   - trim whitespace-only string state so `_safeNumber({state: " "})`
  //     doesn't coerce via `Number(" ") === 0`;
  //   - return-side `Number.isFinite(n)` guard so `±Infinity` (which
  //     `Number.isNaN` does NOT catch) cannot propagate downstream into
  //     the `target > currentTemp` ternary.
  _safeNumber(sensor, defaultValue) {
    if (!sensor || sensor.state == null) return defaultValue;
    const raw = sensor.state;
    // Pass-#3 N3 — `trimmed` can only be (a) an empty/whitespace
    // string after `.trim()` collapses to `""`, or (b) the original
    // non-null `raw` for non-string types (the outer guard already
    // rejected null/undefined). So `trimmed == null` was dead code.
    // Keep the empty-string check; the `Number.isFinite` final guard
    // catches NaN / ±Infinity that survive the string filter.
    const trimmed = (typeof raw === 'string') ? raw.trim() : raw;
    if (trimmed === '') return defaultValue;
    if (trimmed === 'unknown' || trimmed === 'unavailable') return defaultValue;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : defaultValue;
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._root) return;
    // Avoid re-rendering while user is interacting with selects or a modal is open
    if (this._isInteractingMode || this._isInteractingStateOn || this._modalOpen || this._isInteractingTarget) return;
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
      const selClimateMode = this._entity(e.climate_mode);
      const selClimateStateOn = this._entity(e.climate_state_on);
      const selClimateStateOff = this._entity(e.climate_state_off);
      const swGreenOnly = this._entity(e.green_only);
      const swEnableDevice = this._entity(e.enable_device);
      const sOverrideState = this._entity(e.override_state);
      const sStartTime = this._entity(e.start_time);
      const sEndTime = this._entity(e.end_time);
      const sIsOffGrid = this._entity(e.is_off_grid);

      const title = (cfg.title || cfg.name) || "Climate";
      
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
      
      // Get climate mode
      const climateMode = selClimateMode?.state || 'bistate_mode_default';
      const isDefaultMode = climateMode === 'bistate_mode_default';
      
      // Get override state
      const overrideState = sOverrideState?.state || 'NO OVERRIDE';
      const isOverridden = overrideState !== 'NO OVERRIDE';
      const isResettingOverride = overrideState === 'ASKED FOR RESET OVERRIDE';
      
      // Determine if climate is running (command state must be "on")
      const commandState = sCommand?.state || '';
      const running = commandState.toLowerCase() === 'on';
      // Review-fix N3 — stash `running` immediately so the RAF step
      // closures see a consistent value (single-threaded JS makes the
      // race impossible, but the proximity is clearer for readers).
      this._running = running;

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
        // BH (user-reported NaN bug): a brand-new device with no live
        // constraint sensor reports targetHours == 0, which made
        // `hoursToPct` divide by zero and propagate NaN into the SVG
        // arc path (`A 130 130 0 0 1 NaN NaN`). Clamp to a sensible
        // positive fallback so the ring always renders.
        maxHours = targetHours > 0 ? targetHours : (Number(cfg.max_default_hours) || 12);
        displayTargetHours = targetHours;
      }

      // QS-210: progressRatio drives the flame and snow base-Y formulas
      // (mirror of qs-radiator-card.js line 519-521 and
      // qs-pool-card.js line 339-341).
      const progressRatio = maxHours > 0
          ? Math.max(0, Math.min(1, hoursRun / maxHours))
          : 0;

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
      
      // Color schemes based on climate_state_on - MUST BE DEFINED BEFORE CSS
      // Review-fix N4 — default to `''` (not `'cool'`) so the boot
      // race condition (SELECT entity not yet populated) defers to the
      // catch-all `'none'` branch instead of flashing snow until the
      // real state arrives. The downstream `colorSchemes` lookup falls
      // back to `colorSchemes.cool` for `''` (preserving existing ring
      // chrome behaviour).
      const climateStateOn = (selClimateStateOn?.state || '').toLowerCase();

      // QS-210: read backing climate entity for AUTO / HEAT_COOL temp
      // comparison. The dashboard template emits `climate_entity:`
      // (T3.1); when it's missing or the entity has dropped from the
      // state machine, all four reads return null and the resolved-
      // target algorithm falls through to the "ambiguous" branch
      // (wind / none) without crashing.
      //
      // Review-fix S8 — gate the reads on `needsTemps`. AC5 requires
      // "no climate-entity attribute reads when they cannot influence
      // the outcome" (fan_only / dry / off short-circuit). Only the
      // AUTO / HEAT_COOL branch consults the temps; everything else
      // gets `null`s that the resolved-target algorithm never sees.
      const needsTemps = climateStateOn === 'auto' || climateStateOn === 'heat_cool';
      const climateEntity = (needsTemps && e.climate_entity) ? this._hass?.states?.[e.climate_entity] : null;
      const attrs = climateEntity?.attributes;
      const currentTemp = needsTemps ? this._safeNumber({state: attrs?.current_temperature}, null) : null;
      const singleTarget = needsTemps ? this._safeNumber({state: attrs?.temperature}, null) : null;
      const lowTarget = needsTemps ? this._safeNumber({state: attrs?.target_temp_low}, null) : null;
      const highTarget = needsTemps ? this._safeNumber({state: attrs?.target_temp_high}, null) : null;

      // Resolved-target algorithm (AC1):
      //   T finite           → target = T
      //   L finite AND H finite → target = (L + H) / 2  (midpoint)
      //   L finite           → target = L
      //   H finite           → target = H
      //   otherwise          → target = null  (genuinely ambiguous)
      const resolveTarget = () => {
        if (Number.isFinite(singleTarget)) return singleTarget;
        if (Number.isFinite(lowTarget) && Number.isFinite(highTarget)) {
          return (lowTarget + highTarget) / 2;
        }
        if (Number.isFinite(lowTarget)) return lowTarget;
        if (Number.isFinite(highTarget)) return highTarget;
        return null;
      };

      // Backdrop decision (AC1 — resolved-target algorithm):
      //   'heat'  → 'flame'
      //   'cool'  → 'snow'
      //   'auto' / 'heat_cool' →
      //      target & current finite →
      //          |target - current| < BACKDROP_DEADBAND_C → hold the
      //              previous resolved backdrop (review-fix S4 —
      //              avoid flame↔snow flips on thermostat jitter at
      //              equilibrium); default to 'flame' if there's no
      //              previous resolved value yet (cold start).
      //          else → target > current ? 'flame' : 'snow'
      //      else → running ? 'wind' : 'none'
      //   anything else (fan_only / dry / off / unrecognised / '' /
      //      null) → 'none'. The four climate-entity attribute reads
      //      above short-circuit to `null` via the `needsTemps` guard
      //      (review-fix S8 — matches AC5's "no reads when they
      //      cannot influence the outcome").
      // Pass-#3 N1 — small helper to deduplicate the two sign-based
      // ternary call sites (the non-deadband arm and the deadband
      // sign-based fallback). Both pick flame vs snow purely from
      // the temperature sign — no hysteresis, no `_lastBackdrop`
      // dependency.
      const _signBackdrop = (target, currentTemp) =>
          target > currentTemp ? 'flame' : 'snow';

      const deriveBackdrop = () => {
        if (climateStateOn === 'heat') return 'flame';
        if (climateStateOn === 'cool') return 'snow';
        if (climateStateOn === 'auto' || climateStateOn === 'heat_cool') {
          const target = resolveTarget();
          if (Number.isFinite(currentTemp) && target != null) {
            if (Math.abs(target - currentTemp) < BACKDROP_DEADBAND_C) {
              // Hysteresis: keep the previous resolved backdrop so
              // ±0.1°C sensor jitter doesn't flip the visual every
              // hass push.
              if (this._lastBackdrop === 'flame' || this._lastBackdrop === 'snow') {
                return this._lastBackdrop;
              }
              // Pass-#2 N5 — sign-based fallback when there's no
              // previous resolved backdrop (e.g., transitioning from
              // 'wind' or 'none' straight into a deadband-AUTO state).
              // The pass-#1 unconditional `return 'flame';` ignored
              // the temp sign and showed flame even on a slight
              // cooling delta — pick based on the sign so the very
              // first deadband render respects the user's intent.
              // (Subsequent renders fall into the hold-previous arm
              // above.)
              return _signBackdrop(target, currentTemp);
            }
            return _signBackdrop(target, currentTemp);
          }
          return running ? 'wind' : 'none';
        }
        return 'none';
      };
      this._backdrop = deriveBackdrop();

      const colorSchemes = {
        cool: {
          primary: '#2196F3',
          gradStart: '#00bcd4',
          gradEnd: '#8bc34a',
          animStart: '#00e1ff',
          animEnd: '#0066ff'
        },
        heat: {
          primary: '#FF5722',
          gradStart: '#FF5722',
          gradEnd: '#D32F2F',
          animStart: '#FF6E40',
          animEnd: '#E64A19'
        },
        heat_cool: {
          primary: '#9C27B0',
          gradStart: '#9C27B0',
          gradEnd: '#BA68C8',
          animStart: '#AB47BC',
          animEnd: '#8E24AA'
        },
        auto: {
          primary: '#9C27B0',
          gradStart: '#9C27B0',
          gradEnd: '#BA68C8',
          animStart: '#AB47BC',
          animEnd: '#8E24AA'
        },
        fan_only: {
          primary: '#00BCD4',
          gradStart: '#00BCD4',
          gradEnd: '#4DD0E1',
          animStart: '#00E5FF',
          animEnd: '#00B8D4'
        },
        dry: {
          primary: '#D4A574',
          gradStart: '#D4A574',
          gradEnd: '#E6C9A8',
          animStart: '#E0B589',
          animEnd: '#C9935D'
        }
      };
      
      const colors = colorSchemes[climateStateOn] || colorSchemes.cool;
      
      const css = `
      :host { --pad: 18px; --ring-text-shadow: 0 0 12px rgba(0,0,0,0.8), 0 2px 4px rgba(0,0,0,0.5); display:block; }
      .card { padding: var(--pad); position: relative; }
      .card.off-grid { background: rgba(244, 67, 54, 0.08); }
      .card-title { text-align:center; font-weight:800; font-size: 1.6rem; margin: 0px 0 0px; }
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
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; text-shadow: var(--ring-text-shadow); }
      .ring .target-value { font-weight:800; font-size: 2.5rem; line-height: 1.1; text-shadow: var(--ring-text-shadow); }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; text-align:center; width: 220px; margin: 0 auto; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:6px; }
      .ring .from-to-row { display:flex; justify-content:space-between; width:140px; margin-top: 8px; gap:20px; }
      .ring .from-to-item { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .from-to-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; text-shadow: var(--ring-text-shadow); }
      .ring .from-to-value { color: var(--primary-text-color); font-weight:800; font-size: 1.4rem; text-shadow: var(--ring-text-shadow); }
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
      
      const center = {cx: 160, cy: 160};
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
      // QS-210: gate split. The dashed-progress arc only needs RAF
      // when there's enough progress to dash through. The flame / snow
      // / wind backdrop wants RAF as soon as it's active. The umbrella
      // `showAnimation` is the OR — drives both the
      // <path id="running_anim"> emission AND the _startAnimation()
      // start/stop dance.
      const ringDashActive = running && segLen > 6;
      const backdropActive = this._backdrop !== 'none';
      const showAnimation = ringDashActive || backdropActive;

      // Review-fix S1 — initialise `_needsFlamePrime` to `true` here so
      // the first paint primes `_currentFlameAmp` directly to its
      // target instead of lerping STILL_AMP→DANCE_AMP over ~1.5s. The
      // lazy-init in `_startAnimation` only fires AFTER this point
      // (and only when `_currentFlameAmp == null`), so without this
      // line the flag would be `undefined` → falsy → skip on the
      // first paint. `== null` (loose equality) catches both
      // `undefined` and `null` so the prime fires exactly once.
      if (this._needsFlamePrime == null) this._needsFlamePrime = true;

      // QS-210: flame backdrop — base-Y formula and colour palette.
      // Both reuse the radiator card's progress envelope.
      const flameBaseY = CENTER_CY + CLIP_R
          - (FLAME_BASE_MIN_PCT + progressRatio * (FLAME_BASE_MAX_PCT - FLAME_BASE_MIN_PCT))
            * 2 * CLIP_R;
      this._flameBaseY = Number.isNaN(flameBaseY) ? null : flameBaseY;
      this._flameColors = running ? FLAME_FILLS : FLAME_GREY_FILLS;
      // Honour first-connect prime: skip the 1.5s boot lerp.
      // Review-fix S3 — only prime when the actual backdrop is flame.
      // The state is otherwise unused for snow/wind/none, so mutating
      // it would be semantically wrong (no visible bug today, but
      // confusing to a reader).
      if (this._backdrop === 'flame' && this._needsFlamePrime) {
        this._currentFlameAmp = running ? DANCE_AMP : STILL_AMP;
        this._needsFlamePrime = false;
      }

      // QS-210: snow backdrop — base-Y mirrors pool's envelope so the
      // pile height tracks runtime progress (same 0.2..0.8 of clip
      // diameter the flame uses, kept symmetric for readability).
      const snowBaseY = CENTER_CY + CLIP_R - (0.2 + progressRatio * 0.6) * 2 * CLIP_R;
      this._snowBaseY = Number.isNaN(snowBaseY) ? null : snowBaseY;

      // QS-210: per-instance clip / snow-layer / wind-layer ids so
      // multiple climate cards on one dashboard don't collide. Shadow
      // DOM scopes the ids anyway, but a stable per-instance id makes
      // diff/inspection easier (mirror of radiator's `_flameClipId`
      // pattern; renamed to `_climateClipId` since it's shared across
      // all three backdrops).
      if (!this._climateClipId) {
        QsClimateCard._nextClipId = (QsClimateCard._nextClipId || 0) + 1;
        const uid = QsClimateCard._nextClipId;
        this._climateClipId = `cClip-${uid}`;
        this._snowLayerId = `cSnowLayer-${uid}`;
        this._windLayerId = `cWindLayer-${uid}`;
      }
      const climateClipId = this._climateClipId;
      const snowLayerId = this._snowLayerId;
      const windLayerId = this._windLayerId;

      // QS-210: backdrop-change side-effects block.
      //
      // Review-fix N5 — when transitioning INTO snow from a non-snow
      // backdrop, reset the snow amp/speed accumulators to their calm
      // defaults so the pile doesn't briefly inherit RUN_SNOW_AMP /
      // RUN_SNOW_SPEED from a previous off-transient. The phase
      // accumulator stays so the scroll position is continuous.
      //
      // Pass-#2 M1 — additionally defend `_snowflakes`,
      // `_snowWavePhase`, and `_nextSnowflakeAt`: on a cold start
      // straight into `'snow'` mode (e.g., `climate_state_on === 'cool'`
      // on first paint), the render-side write to `_currentSnowAmp`
      // ran BEFORE `_startAnimation`'s legacy single-guard lazy-init,
      // bypassing initialisation of the other four snow fields. The
      // first RAF tick of `_stepSnow` then crashed on
      // `for (const b of this._snowflakes)` (`undefined is not
      // iterable`). The defensive `if (X == null) X = …;` guards here
      // — combined with the per-field guards in `_startAnimation`
      // (also added in pass-#2 M1) — make the init robust regardless
      // of which field is primed first.
      //
      // Pass-#2 N1 — the three `_invalidate*Cache()` calls that used
      // to live here have been removed. The post-innerHTML M1 block
      // (at the tail of `_render`) now invalidates all three caches
      // unconditionally, so calling them here was redundant CPU work.
      if (this._backdrop !== this._lastBackdrop) {
        if (this._backdrop === 'snow' && this._lastBackdrop !== 'snow') {
          this._currentSnowAmp = CALM_SNOW_AMP;
          this._currentSnowSpeed = CALM_SNOW_SPEED;
          if (this._snowWavePhase == null) this._snowWavePhase = 0;
          if (this._snowflakes == null) this._snowflakes = [];
          if (this._nextSnowflakeAt == null) this._nextSnowflakeAt = 0;
        }
        this._lastBackdrop = this._backdrop;
      }

      // QS-210: pre-generate the initial flame / snow / wind paths so
      // the SVG renders with the backdrop immediately, avoiding an
      // empty-d="" flash between the innerHTML rewrite and the first
      // RAF tick.
      //
      // Review-fix S2 — gate each pre-generation block on the active
      // backdrop. The SVG fragment below only references the matching
      // initial-path variable for the active backdrop, so the other
      // two arrays of generator calls would be pure waste (~9 paths
      // generated per render, none consumed).
      let initialFlamePaths = null;
      let initialFlameColors = null;
      if (this._backdrop === 'flame') {
        const initialFlameAmp = this._currentFlameAmp ?? STILL_AMP;
        const initialFlameBaseY = this._flameBaseY ?? CENTER_CY;
        initialFlameColors = this._flameColors ?? FLAME_GREY_FILLS;
        if (!this._tipPhases) {
          this._tipPhases = LAYER_TEETH_COUNTS.map((count) => new Array(count).fill(0));
        }
        initialFlamePaths = Array.from(
            {length: LAYER_TEETH_COUNTS.length},
            (_, i) => this._generateFlameTeethPath(
                FLAME_WIDTH,
                initialFlameBaseY,
                LAYER_BASE_HEIGHTS[i],
                initialFlameAmp * LAYER_TIP_AMP_MULTS[i],
                LAYER_TEETH_COUNTS[i],
                this._tipPhases[i],
                !running,
            ),
        );
      }
      let initialSnowWavePaths = null;
      if (this._backdrop === 'snow') {
        const initialSnowAmp = this._currentSnowAmp ?? CALM_SNOW_AMP;
        const initialSnowBaseY = this._snowBaseY ?? CENTER_CY;
        initialSnowWavePaths = [0, 1, 2].map((i) => {
          const freq = 2 + i;
          const phaseOffset = i * LAYER_PHASE_OFFSET;
          return this._generateWavePath(WAVE_WIDTH, initialSnowAmp, freq, phaseOffset, initialSnowBaseY);
        });
      }
      let initialWindPaths = null;
      if (this._backdrop === 'wind') {
        initialWindPaths = [0, 1, 2].map((i) => this._generateWispPath(
            WIND_WISP_WIDTH,
            WIND_WAVE_AMP,
            WIND_WAVE_FREQS[i],
            i * Math.PI / 3,
            CENTER_CY + WIND_WISP_Y_OFFSETS[i],
        ));
      }

      // M4: start/stop the RAF loop based on whether any visible
      // animation needs to run.
      if (showAnimation) {
        this._startAnimation();
      } else {
        this._stopAnimation();
      }

      // Climate mode selector options with translations
      const modeOptions = selClimateMode?.attributes?.options || [];
      const modeState = (selClimateMode?.state || '').trim();
      
      // Helper to translate climate mode options
      const translateClimateMode = (value) => {
          const key = `component.quiet_solar.entity.select.climate_mode.state.${value}`;
          const translated = this._hass?.localize?.(key);
          // If translation not found or returns the key itself, fall back to the raw value
          return (translated && translated !== key) ? translated : value;
      };
      
      const modeOptionsHtml = modeOptions.map(o => 
          `<option value="${o}" ${o === modeState ? 'selected' : ''}>${translateClimateMode(o)}</option>`
      ).join('');

      // Climate state on selector options (filtered to exclude the state selected in state_off)
      const stateOnOptions = selClimateStateOn?.attributes?.options || [];
      const stateOnState = (selClimateStateOn?.state || '').trim();
      const stateOffCurrent = (selClimateStateOff?.state || '').trim();
      
      // Helper to translate climate state options
      const translateClimateState = (value) => {
          const key = `component.quiet_solar.entity.select.climate_state_on.state.${value}`;
          const translated = this._hass?.localize?.(key);
          // If translation not found or returns the key itself, fall back to the raw value
          return (translated && translated !== key) ? translated : value;
      };
      
      // Filter out the current state_off value and "off" from state_on options
      const filteredStateOnOptions = stateOnOptions.filter(o => o !== stateOffCurrent && o.toLowerCase() !== 'off');
      
      const stateOnOptionsHtml = filteredStateOnOptions.map(o => 
          `<option value="${o}" ${o === stateOnState ? 'selected' : ''}>${translateClimateState(o)}</option>`
      ).join('');

      // Override button - parse and determine icon
      // Parse override command from override state
      const parseOverrideCommand = (overrideStateStr) => {
        if (!overrideStateStr || overrideStateStr === 'NO OVERRIDE') return null;
        const match = String(overrideStateStr).match(/Override:\s*(.+)/i);
        return match ? match[1].trim() : null;
      };
      
      const overrideCommand = parseOverrideCommand(overrideState);
      const overrideCommandLower = overrideCommand ? overrideCommand.toLowerCase() : '';
      
      // Check if override is for "off" state (ends with "off" OR matches climate_state_off)
      const isOverrideOff = overrideCommand && (
        overrideCommandLower.endsWith('off') || 
        overrideCommand === stateOffCurrent
      );

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

      // QS-216 AC-1 — snapshot in-flight snowflakes before the
      // innerHTML rewrite (mirror of QS-214 boiler precedent). Both
      // locals may capture `undefined` on a cold render; the restore
      // block's `?.length` guard handles that. See AC-2 + AC-4 for
      // the restore + ordering invariant.
      const preservedSnowflakes = this._snowflakes;
      const preservedNextSnowflakeAt = this._nextSnowflakeAt;

      // QS-217 review-fix #03 — clipPath is just the outer disc;
      // the override-button area is hidden by a separate `<circle>`
      // cover drawn AFTER the clipped animation group (see the SVG
      // markup below). This replaces the earlier carve+cancel
      // clipPath approach, which produced a geometric lens-shape
      // hole. The cover applies uniformly to all four backdrops
      // (flame / snow / wind / none) per DN-3.
      const clipPathD =
        `M ${CENTER_X - CLIP_R},${CENTER_CY}` +
        ` a ${CLIP_R},${CLIP_R} 0 1,0 ${2 * CLIP_R},0` +
        ` a ${CLIP_R},${CLIP_R} 0 1,0 ${-2 * CLIP_R},0`;

      this._root.innerHTML = `
      <ha-card class="card ${!isEnabled ? 'disabled' : ''} ${isOffGrid ? 'off-grid' : ''}">
        <style>${css}</style>
        <div class="card-title">${this._escapeHtml(title)}</div>
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
                <clipPath id="${climateClipId}">
                  <path clip-rule="evenodd" d="${clipPathD}" />
                </clipPath>
              </defs>
              <g clip-path="url(#${climateClipId})">
                ${this._backdrop === 'flame' ? initialFlamePaths.map((d, i) =>
                  `<path id="flame${i}" d="${d}" fill="${initialFlameColors[i]}" opacity="1" pointer-events="none" style="will-change: transform;" />`
                ).join("\n                ") : ''}
                ${this._backdrop === 'snow' ? `
                <path id="snowWave0" d="${initialSnowWavePaths[0]}" fill="${SNOW_BACK_COLOR}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <path id="snowWave1" d="${initialSnowWavePaths[1]}" fill="${SNOW_MID_COLOR}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <path id="snowWave2" d="${initialSnowWavePaths[2]}" fill="${SNOW_FRONT_COLOR}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <g id="${snowLayerId}"></g>
                ` : ''}
                ${this._backdrop === 'wind' ? `
                <g id="${windLayerId}">
                  <path id="windWisp0" d="${initialWindPaths[0]}" stroke="${WIND_WISP_COLORS[0]}" fill="none" stroke-width="2" pointer-events="none" style="will-change: transform;" />
                  <path id="windWisp1" d="${initialWindPaths[1]}" stroke="${WIND_WISP_COLORS[1]}" fill="none" stroke-width="2" pointer-events="none" style="will-change: transform;" />
                  <path id="windWisp2" d="${initialWindPaths[2]}" stroke="${WIND_WISP_COLORS[2]}" fill="none" stroke-width="2" pointer-events="none" style="will-change: transform;" />
                </g>
                ` : ''}
              </g>
              ${e.override_reset ? `<circle id="override_btn_cover" cx="${CENTER_X}" cy="${OVERRIDE_BTN_CARVE_CY}" r="${OVERRIDE_BTN_CARVE_R}" fill="var(--card-background-color)" pointer-events="none" />` : ''}
              <path d="${bgPath}" stroke="var(--divider-color)" stroke-width="14" fill="none" stroke-linecap="round" />
              <path d="${progressPath}" stroke="url(#${activeGradId})" stroke-width="14" fill="none" stroke-linecap="round" ${ringDashActive ? 'stroke-opacity="0.35"' : ''} />
              ${ringDashActive ? `
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

        ${selClimateMode ? `
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:sun-thermometer-outline"></ha-icon>
            <select id="climate_mode">
              ${modeOptionsHtml}
            </select>
          </div>
        </div>
        ` : ''}
        ${selClimateStateOn ? `
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:state-machine"></ha-icon>
            <select id="climate_state_on">
              ${stateOnOptionsHtml}
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

      // Review-fix M1 — invalidate the three backdrop DOM caches
      // AFTER every innerHTML rewrite (not only when the backdrop
      // type changes, which the block higher up handles). The
      // rewrite detaches every cached element (`_flameEls`,
      // `_snowWaveEls`, `_snowLayerEl`, `_windWispEls`); without
      // this unconditional reset, same-backdrop re-renders (the
      // common case — every hass push, ~once per second) leave RAF
      // ticking on orphan nodes and the visual freezes. Mirror of
      // `qs-radiator-card.js:967-969`. `_invalidateSnowCache` also
      // clears `_snowflakes[]`, so the spawn-cap stops drifting up
      // toward MAX_CONCURRENT_SNOWFLAKES across renders.
      this._invalidateFlameCache();
      this._invalidateSnowCache();
      this._invalidateWindCache();

      // QS-216 AC-2/AC-4 — restore snapshot into the fresh snow
      // layer AFTER the _invalidate*Cache() triplet (mirror of QS-214
      // boiler). The null-layer else is the real backdrop-transition
      // path (snow `<g>` is emitted only when `_backdrop === 'snow'`),
      // not just defensive belt-and-braces.
      if (preservedSnowflakes?.length) {
        const newSnowLayer = this._snowLayerId
          ? this._root.getElementById(this._snowLayerId)
          : null;
        if (newSnowLayer) {
          for (const b of preservedSnowflakes) {
            if (b?.el) newSnowLayer.appendChild(b.el);
          }
          // N8 — `.filter` builds a fresh array. All hot-path code
          // (`_stepSnow`, `_invalidateSnowCache`) reads `this._snowflakes`
          // each call, so the new reference is safe; do not pass the
          // original `preservedSnowflakes` const to any closure.
          this._snowflakes = preservedSnowflakes.filter(b => b?.el);
          this._snowLayerEl = newSnowLayer;
          this._nextSnowflakeAt = preservedNextSnowflakeAt;
        } else {
          for (const b of preservedSnowflakes) {
            b?.el?.remove();
          }
          this._snowflakes = [];
        }
      }

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

      // Climate mode selector
      if (selClimateMode) {
          const modeSel = ids('climate_mode');
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
                  await this._select(e.climate_mode, option);
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

      // Climate state on selector
      if (selClimateStateOn) {
          const stateOnSel = ids('climate_state_on');
          const startS = () => {
              this._isInteractingStateOn = true;
          };
          const endS = () => {
              // Don't clear flag on blur during change processing
              if (!this._isProcessingStateOnChange) {
                  this._isInteractingStateOn = false;
                  this._render();
              }
          };
          stateOnSel?.addEventListener('focus', startS);
          stateOnSel?.addEventListener('blur', endS);
          stateOnSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return;

              this._isProcessingStateOnChange = true;
              // M2: see above comment in the mode-change handler.
              try {
                  // Call the service and wait for it to complete
                  await this._select(e.climate_state_on, option);
              } finally {
                  // Wait a bit for HA state to propagate, then allow re-render
                  setTimeout(() => {
                      this._isProcessingStateOnChange = false;
                      this._isInteractingStateOn = false;
                      this._render();
                  }, 300);
              }
          });
          const stateOnPill = stateOnSel?.closest('.pill');
          if (stateOnPill && stateOnSel) {
              stateOnPill.addEventListener('click', (ev) => {
                  if (ev.target.tagName === 'SELECT') return;
                  try { stateOnSel.showPicker(); } catch (_) { stateOnSel.focus(); }
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
            <p>Select the time the climate should finish by:</p>
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
                  title: 'Climate Finish Time',
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
                  title: 'Reset climate state',
                  message: 'This will reset internal state for the climate and cannot be undone.\nProceed?',
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

if (!customElements.get('qs-climate-card')) {
    customElements.define('qs-climate-card', QsClimateCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-climate-card')) {
    window.customCards.push({
        type: 'qs-climate-card',
        name: 'QS Climate Card',
        description: 'Quiet Solar climate control card',
    });
}
