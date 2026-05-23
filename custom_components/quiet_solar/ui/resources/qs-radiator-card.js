/*
  QS Radiator Card - custom:qs-radiator-card
  Zero-build single-file Lit-style web component compatible with Home Assistant.

  Provenance and divergence (S4 review-fix note):
  This file started as a clone of qs-on-off-duration-card.js with the
  custom-element renamed. Per QS-195 review-fix #01 it has since picked up:
    - S14 — `_safeNumber` coercion (NaN-safe state reads)
    - S15 — `_escapeHtml` for innerHTML-bound strings (title + mode labels)
    - S16 — keyboard-accessible action buttons (role="button" + tabindex
            + Enter/Space activation)
    - S17 — try/finally around async service calls so interaction
            guards never leak on transient failures
    - N7  — cold-start `running` derivation OR-s in the underlying
            backing entity state when published
  The two cards are intentionally not yet collapsed into a shared base
  module — the on/off card is older and broadly deployed; a clean
  shared-base extraction is a larger refactor scheduled as a follow-up.
  When that lands, the on/off card SHOULD adopt the same S14-S17/N7
  safety improvements that this card already has.

  TODO: shared base, tracked at https://github.com/tmenguy/quiet-solar/issues/199
  (A1 review-fix #02).
*/

// QS-204: 3-layer peaked-teeth flame backdrop.
// The QS-201 sine-wave path was indistinguishable from qs-pool-card.js's
// water waves (both at running orange and idle grey). QS-204 swaps the
// path generator for a piecewise-quadratic tooth shape (sharper tips,
// taller peaks) and drops the global translateX scroll in favour of
// per-tooth tip-flicker — the silhouette reads as flames in both states.
// Constants stay duplicated in-file rather than imported from the pool
// card so a future pool-card refactor cannot silently break this card.

// --- Geometry (must match the <clipPath> circle attributes below) ---
const CENTER_CY = 160;              // SVG y-centre of the ring / clip circle
const CLIP_R = 120;                 // flame clip circle radius
const FLAME_WIDTH = 480;            // single layer width in SVG px (2× clip diameter)
const FLAME_BOTTOM_Y = 400;         // ≥ SVG viewBox max-y (320) so the closing rect is clipped

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
// SVG units: (160, 277.33) → rounded to (CENTER_CY, 277). Cover
// radius R = 35 SVG ≈ 33 CSS px ⇒ ~8 CSS px padding around the
// 25 CSS-px-radius button outline. User-tunable.
const OVERRIDE_BTN_CARVE_CY = 277;
const OVERRIDE_BTN_CARVE_R  = 35;

// --- Animation tuning ---
const LERP_RATE = 2;                // exp time-constant; ~95% of lerp in ~1.5s
const LERP_DT_CEIL = 0.1;           // s; clamp lerp dt to avoid snap-to after tab hidden
const AMP_REGEN_THRESHOLD = 0.25;   // amplitude delta threshold for path regen (throttle)
const LEVEL_REGEN_THRESHOLD = 0.01; // px; threshold for base-Y path regen (jitter-proof)
// QS-204 review-fix #03 H2 — running-mode regen throttle. Time-based
// (NOT phase-delta) because phase advances ~0.84 rad/frame at 60 FPS
// (2π × 8 Hz × 1/60), which would clear any reasonable phase threshold
// every frame and defeat the throttle. PHASE_REGEN_MIN_DT = 0.20 s
// caps running-mode regen at ~5 fps — well below the 7-9 Hz visual
// flicker rate, and a 12× perf win over the unthrottled fireOn path.
const PHASE_REGEN_MIN_DT = 0.20;

// --- Flame dynamics targets (still vs dancing) ---
// STILL_AMP = 0 — idle silhouette is fully frozen (per-tooth tip phases
// never advance), pinning AC4's "no horizontal scroll, no tip wobble"
// constraint. STATIC_PEAK_HEIGHT keeps the silhouette visibly peaked
// while STILL_AMP === 0 so the idle frame still reads as a flame.
const STILL_AMP = 0;                // tip-flicker amplitude when off — frozen
const DANCE_AMP = 8;                // tip-flicker amplitude when on — visibly dancing
const STATIC_PEAK_HEIGHT = 30;      // extra peak height when STILL_AMP=0 so idle has visible peaks

// --- Per-layer tuning ---
// LAYER_BASE_HEIGHTS — back layer reaches ≈ half clip radius (~100 px)
// so the back-flame tip approaches the top of the clip circle, which
// is what makes the silhouette read as flames rather than ripples.
const LAYER_BASE_HEIGHTS = [150, 120, 90];
const LAYER_TIP_AMP_MULTS = [1.2, 1.0, 0.8];  // multiplies _currentFlameAmp per layer
const LAYER_TEETH_COUNTS = [3, 4, 5];          // fewer wider teeth read more like flames
// LAYER_TIP_FLICKER_HZ — independent per-layer flicker frequencies so
// the three layers desynchronise and read as turbulent fire rather than
// a single global scroll. Frequencies are in Hz (tooth-phase cycles/sec).
const LAYER_TIP_FLICKER_HZ = [8, 7, 9];

// QS-204 review-fix #02 G7 — length-equality guard. All four per-layer
// arrays MUST stay the same length; a future PR that extends one without
// the others would silently produce `NaN` paths (out-of-bounds reads
// yield undefined → arithmetic → NaN). `console.assert` is the
// browser-side equivalent of a Python `assert`.
console.assert(
    LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length &&
    LAYER_TEETH_COUNTS.length === LAYER_BASE_HEIGHTS.length &&
    LAYER_TEETH_COUNTS.length === LAYER_TIP_AMP_MULTS.length,
    "qs-radiator-card: LAYER_* constants must be the same length"
);

// --- Flame height envelope (1/5..4/5 of clip diameter; mirrors pool) ---
const FLAME_BASE_MIN_PCT = 0.2;     // hoursRun=0 → flame base low
const FLAME_BASE_MAX_PCT = 0.8;     // hoursRun=maxHours → flame base high

// --- Fills (warm when running, greys when off) ---
const FLAME_FILLS = [
    'rgba(255, 87, 34, 0.55)',      // back layer (tallest tongues) — primary
    'rgba(255, 110, 64, 0.45)',     // mid layer — animStart
    'rgba(255, 193,  7, 0.35)',     // front layer (shortest tips) — amber
];
const FLAME_GREY_FILLS = [
    'rgba(160, 160, 160, 0.40)',    // back layer
    'rgba(140, 140, 140, 0.30)',    // mid layer
    'rgba(120, 120, 120, 0.22)',    // front layer
];

class QsRadiatorCard extends HTMLElement {
  // M4: gate the requestAnimationFrame loop on `showAnimation`.
    // condition (`showAnimation`). The loop is started lazily by
  // `_render()` only when the running-progress dash needs to advance,
  // and stopped in `disconnectedCallback` AND whenever `showAnimation`
  // becomes false. Avoids constant per-card repaint overhead when no
  // visible animation is in progress.
  _startAnimation() {
    // QS-201 / QS-204: first-connect prime — initialize flame anim state
    // if null so the next _render() can read sane defaults. Set
    // _needsFlamePrime so _render() skips the 1.5s lerp from defaults at
    // boot. The prime block must run BEFORE the early-return so the very
    // first _render() after construction sees primed amp values; it's
    // idempotent on every subsequent call (guarded by the null check).
    if (this._currentFlameAmp == null) {
      this._currentFlameAmp = STILL_AMP;
      // QS-204 — per-layer, per-tooth tip phases. Independent flicker
      // across layers reads as fire turbulence rather than scroll.
      this._tipPhases = LAYER_TEETH_COUNTS.map((count) => new Array(count).fill(0));
      this._needsFlamePrime = true;
    }

    // QS-201 review fix S1: the cache-invalidate MUST live behind the
    // early-return guard. `_startAnimation()` is called on every _render()
    // via the umbrella RAF gate, but nulling `_lastFlameAmp` /
    // `_lastFlameBaseY` on every render would defeat the path-regen
    // throttle (ampDelta becomes Infinity, forcing unconditional path
    // regeneration on the next RAF tick). The cache only needs clearing
    // when RAF is actually being (re)started; the post-innerHTML reset in
    // `_render()` already handles the rewrite case.
    if (this._animRaf != null) return;
    this._invalidateFlameCache();
    this._lastAnimTs = null;
    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      const dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      this._lastAnimTs = ts;
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const speed = 80; // dash units per second
      this._animOffset = ((this._animOffset || 0) + speed * dt) % patternLen;
      const p = this._root?.getElementById('running_anim');
      if (p) {
        p.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }

      // --- QS-204: flame animation update ---
      const fireOn = this._fireRunning === true;
      const targetAmp = fireOn ? DANCE_AMP : STILL_AMP;
      // Clamp the lerp dt: after a tab is hidden for several seconds, the
      // first frame back has huge dt and lerpFactor ≈ 1, which would snap
      // amp to target.
      const lerpDt = Math.min(dt, LERP_DT_CEIL);
      const lerpFactor = 1 - Math.exp(-LERP_RATE * lerpDt);
      this._currentFlameAmp += (targetAmp - this._currentFlameAmp) * lerpFactor;
      // QS-204 review-fix #01 F5 / #02 G6 — symmetric snap-to-target
      // once the asymptotic lerp brings amp within an epsilon of its
      // target. The float64 lerp factor never settles exactly on the
      // target, so without these snaps the downstream throttle would
      // pay regen frames forever; they're also what makes the running
      // amp land cleanly on DANCE_AMP (avoids 7.99…px tip-amplitude
      // jitter on the steady-state running silhouette).
      if (!fireOn && Math.abs(this._currentFlameAmp) < 0.05) {
        this._currentFlameAmp = 0;
      }
      if (fireOn && Math.abs(this._currentFlameAmp - DANCE_AMP) < 0.05) {
        this._currentFlameAmp = DANCE_AMP;
      }

      // Advance per-layer, per-tooth tip phases ONLY when running. Idle
      // keeps the phases frozen at their last value, which combined with
      // STILL_AMP === 0 produces a fully motionless silhouette (AC4).
      //
      // QS-204 review-fix #03 H4 — use `lerpDt` (clamped to
      // `LERP_DT_CEIL`) instead of raw `dt`. After a long hidden-tab
      // pause the browser may report dt ≈ 60 s, which would advance
      // phase by `2π × 8 × 60 ≈ 3000 rad` and wrap to an essentially-
      // arbitrary value — visible to the user as a sudden tip-pattern
      // snap on re-show. Mirrors the amp-lerp clamp's purpose.
      if (fireOn && this._tipPhases) {
        for (let i = 0; i < LAYER_TEETH_COUNTS.length; i++) {
          const phasesForLayer = this._tipPhases[i];
          const phaseStep = 2 * Math.PI * LAYER_TIP_FLICKER_HZ[i] * lerpDt;
          for (let j = 0; j < phasesForLayer.length; j++) {
            // Scale phase advance by `(1 + 0.07 * j)` so each tooth
            // flickers at a slightly different rate — desynchronises
            // the per-tooth flickers within a layer.
            phasesForLayer[j] = (phasesForLayer[j] + phaseStep * (1 + 0.07 * j)) % (2 * Math.PI);
          }
        }
      }

      // Lazy-resolve flame DOM refs once per innerHTML rewrite.
      // QS-204 review-fix #03 H3 — sized off LAYER_TEETH_COUNTS so a
      // future PR extending the per-layer arrays automatically renders
      // the new layers (G7's length-equality assertion + this dynamic
      // sizing form a complete parameterisation of the layer count).
      if (!this._flameEls) {
        this._flameEls = Array.from(
            {length: LAYER_TEETH_COUNTS.length},
            (_, i) => this._root?.getElementById(`flame${i}`) ?? null,
        );
      }

      // QS-204 — no more translateX scroll. The horizontal-scroll
      // transform is what made the QS-201 layer look like a wave rolling
      // past; removed so each flame stays anchored in place while only
      // its tips flicker.

      // Regenerate paths only when base-Y or amp or per-tooth phases
      // change appreciably (throttle). Guard against undefined/NaN base
      // — the RAF loop can fire before _render() populates _flameBaseY
      // on cold start.
      const flameBaseY = this._flameBaseY;
      const hasValidBase = flameBaseY != null && !Number.isNaN(flameBaseY);
      const ampDelta = this._lastFlameAmp == null
          ? Infinity
          : Math.abs(this._currentFlameAmp - this._lastFlameAmp);
      // Threshold compare for float-jitter robustness vs strict !==.
      const levelChanged = hasValidBase &&
          Math.abs(flameBaseY - (this._lastFlameBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
      // QS-204 review-fix #03 H2 — time-based throttle for the fireOn
      // regen path. The previous per-tooth phase-delta throttle
      // (review-fix #02 G4) was a no-op: phase advances ~0.84 rad per
      // frame at 60 FPS with LAYER_TIP_FLICKER_HZ ≈ 8, far above any
      // reasonable phase threshold. Time-based gating
      // (≥ PHASE_REGEN_MIN_DT seconds since last regen) caps regen at
      // ~5 fps — well below the visual flicker rate (7-9 Hz) and a
      // 12× perf win over per-frame regen.
      const sinceLastRegen = (ts - (this._lastRegenTs ?? -Infinity)) / 1000;
      const phaseChanged = fireOn && sinceLastRegen >= PHASE_REGEN_MIN_DT;
      const shouldRegen = hasValidBase && (
          phaseChanged
          || levelChanged
          || ampDelta > AMP_REGEN_THRESHOLD
      );
      if (shouldRegen) {
        this._lastFlameBaseY = flameBaseY;
        this._lastFlameAmp = this._currentFlameAmp;
        this._lastRegenTs = ts;
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
    // RAF intentionally NOT started here — _render() will call
    // _startAnimation() when needed.
  }

  disconnectedCallback() {
    this._stopAnimation();
    // QS-204: clear flame DOM cache and memoization keys. _currentFlameAmp
    // and _tipPhases deliberately survive so re-attach resumes the dance
    // without a visible jump.
    this._invalidateFlameCache();
    // S7: reset interaction flags so a re-attach after mid-interaction
    // (e.g. dragging the ring or processing a mode change when the
    // dashboard rearranges) doesn't silently short-circuit `set hass`
    // on stale flags.
    this._isInteractingMode = false;
    this._isInteractingTarget = false;
    this._isProcessingModeChange = false;
    this._modalOpen = false;
  }

  static getStubConfig() {
    return { name: "QS Radiator", entities: {} };
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
  _safeNumber(sensor, defaultValue) {
    if (!sensor || sensor.state == null) return defaultValue;
    const s = sensor.state;
    if (s === '' || s === 'unknown' || s === 'unavailable') return defaultValue;
    const n = Number(s);
    return Number.isNaN(n) ? defaultValue : n;
  }

  // QS-204: SVG path for a single flame layer rendered as peaked teeth.
  // Each tooth is a piecewise-quadratic ascend → peak → descend, drawn
  // sharper than a sine bump (tip angle ≲ 60°) so the silhouette reads
  // as a flame. Adjacent teeth share their base point. Per-tooth tip
  // phases let layers flicker independently of one another.
  //
  //   width      — total layer width in SVG px (FLAME_WIDTH)
  //   baseY      — y-coord of the flame base (lower y = higher up the canvas)
  //   peakHeight — base peak height above baseY (positive number)
  //   tipAmp     — extra tip oscillation magnitude (px)
  //   numTeeth   — number of teeth across `width`
  //   tipPhases  — array of per-tooth phases (radians); length == numTeeth.
  //                When STILL_AMP === 0 + frozen tipPhases, the silhouette
  //                is static (idle).
  //   isIdle     — true when the card is in the OFF / idle state. Drives
  //                the STATIC_PEAK_HEIGHT boost so the idle silhouette
  //                still shows visible peaks. QS-204 review-fix #02 G5
  //                gates the boost on isIdle (not tipAmp < epsilon) so a
  //                running→idle transition doesn't briefly "pop" the
  //                flames taller during the lerp ramp.
  _generateFlameTeethPath(width, baseY, peakHeight, tipAmp, numTeeth, tipPhases, isIdle) {
    const teethCount = Math.max(1, numTeeth | 0);
    const toothWidth = width / teethCount;
    // Idle keeps a visible peak via STATIC_PEAK_HEIGHT so the silhouette
    // doesn't collapse onto the baseline. Gated on the explicit isIdle
    // flag (review-fix #02 G5) — using `tipAmp < epsilon` would briefly
    // fire the boost during the running→idle and idle→running lerps,
    // producing a momentary flame-pop.
    const idlePeakBoost = isIdle ? STATIC_PEAK_HEIGHT : 0;
    let d = `M 0 ${baseY.toFixed(2)}`;
    for (let i = 0; i < teethCount; i++) {
      const phase = tipPhases && tipPhases[i] != null ? tipPhases[i] : 0;
      // QS-204 review-fix #03 H1 — gate the per-tooth wobble on
      // `!isIdle`. The RAF lerp is cancelled by `_stopAnimation()`
      // when the radiator transitions running→idle, freezing
      // `_currentFlameAmp` at ~DANCE_AMP rather than at STILL_AMP=0.
      // Without this gate the idle silhouette inherits that frozen
      // amplitude and shows an asymmetric mid-wobble per tooth,
      // visually inconsistent with the cold-boot idle (which starts
      // at amp=0). Mirrors the `idlePeakBoost` gate above.
      const tipWobble = isIdle ? 0 : tipAmp * Math.sin(phase);
      const peakY = baseY - peakHeight - idlePeakBoost - tipWobble;
      const startX = i * toothWidth;
      const midX = startX + toothWidth / 2;
      const endX = startX + toothWidth;
      // Quadratic-Bezier ascend and descend; the control point sits
      // slightly below the peak (`peakY - 8`) so the tip pinches into
      // a sharp angle rather than a rounded curve.
      const ctrlUpX = startX + toothWidth / 3;
      const ctrlDownX = startX + (2 * toothWidth) / 3;
      const ctrlY = peakY - 8;
      d += ` Q ${ctrlUpX.toFixed(2)} ${ctrlY.toFixed(2)} ${midX.toFixed(2)} ${peakY.toFixed(2)}`;
      d += ` Q ${ctrlDownX.toFixed(2)} ${ctrlY.toFixed(2)} ${endX.toFixed(2)} ${baseY.toFixed(2)}`;
    }
    d += ` L ${width.toFixed(2)} ${FLAME_BOTTOM_Y} L 0 ${FLAME_BOTTOM_Y} Z`;
    return d;
  }

  // QS-204: clear the memoization keys and cached DOM refs after each
  // _render() innerHTML rewrite. Does NOT touch _currentFlameAmp or
  // _tipPhases — those survive disconnect so re-attach resumes the
  // dance without a visible jump.
  //
  // QS-204 review-fix #03 H5 — also reset the regen-throttle memo
  // (`_lastRegenTs`). Without this, a re-render that lands within
  // `PHASE_REGEN_MIN_DT` of the prior regen would skip the first
  // running-mode regen against the new DOM nodes (the old `ts` is
  // still cached). Symmetric cleanup with the other `_last*` fields.
  _invalidateFlameCache() {
    this._flameEls = null;
    this._lastFlameBaseY = null;
    this._lastFlameAmp = null;
    this._lastRegenTs = -Infinity;
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
      
      // Determine if device is running (command state must be "on").
      // N7 + BH10 cold-start fallback: when the QS command sensor hasn't
      // been published yet, also recognise the backing entity's state if
      // the dashboard template passes it through `entities.backing_entity`.
      // The configured HVAC ON mode (`entities.climate_hvac_mode_on`,
      // typically "heat" or "auto") is also recognised as "on-like" so
      // users who configured a non-default HVAC mode aren't shown an
      // incorrect "off" state during the cold-start grace window.
      const commandState = sCommand?.state || '';
      const commandReportsOn = commandState.toLowerCase() === 'on';
      const backingEntityId = e.backing_entity;
      const liveBackingState = backingEntityId
          ? (this._hass?.states?.[backingEntityId]?.state || '').toLowerCase()
          : '';
      // BH defense-in-depth: wrap in `String(...)` so a YAML-coerced
      // boolean (HA parses bare `on` as `true` if the template ever
      // regresses and emits an unquoted value) doesn't crash the card
      // on `true.toLowerCase is not a function`.
      const configuredHvacOn = String(e.climate_hvac_mode_on || 'heat').toLowerCase();
      const liveBackingOn = liveBackingState === 'on' || liveBackingState === configuredHvacOn;
      const running = commandReportsOn || liveBackingOn;
      
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
        // BH (user-reported NaN bug): a brand-new switch-backed
        // radiator with no live constraint sensor reports
        // targetHours == 0, which made `hoursToPct` divide by zero
        // and propagate NaN into the SVG arc path
        // (`A 130 130 0 0 1 NaN NaN`). Clamp to a sensible positive
        // fallback so the ring always renders.
        maxHours = targetHours > 0 ? targetHours : (Number(cfg.max_default_hours) || 12);
        displayTargetHours = targetHours;
      }
      
      // QS-201: flame backdrop — height tracks hoursRun/maxHours (mirrors pool).
      const progressRatio = maxHours > 0
          ? Math.max(0, Math.min(1, hoursRun / maxHours))
          : 0;
      const flameBaseY = CENTER_CY + CLIP_R - (FLAME_BASE_MIN_PCT + progressRatio * (FLAME_BASE_MAX_PCT - FLAME_BASE_MIN_PCT)) * 2 * CLIP_R;
      this._fireRunning = running;          // consumed by _startAnimation's step() closure
      this._flameBaseY = Number.isNaN(flameBaseY) ? null : flameBaseY;
      this._flameColors = running ? FLAME_FILLS : FLAME_GREY_FILLS;

      // Honor first-connect prime: skip the 1.5s boot lerp.
      if (this._needsFlamePrime) {
        this._currentFlameAmp = running ? DANCE_AMP : STILL_AMP;
        this._needsFlamePrime = false;
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
      
      // QS-201: heat-mode palette — borrowed verbatim from qs-climate-card.js's
      // `heat` colorScheme block. The radiator is heating-only so a fixed warm
      // palette is appropriate (no per-state branching like the climate card).
      const colors = {
        primary: '#FF5722',
        gradStart: '#FF5722',
        gradEnd: '#D32F2F',
        animStart: '#FF6E40',
        animEnd: '#E64A19'
      };

      // QS-201: per-instance clip id so multiple radiator cards on one
      // dashboard don't collide. Shadow DOM scopes the id anyway, but a
      // stable per-instance id makes diff/inspection easier.
      if (!this._flameClipId) {
        QsRadiatorCard._nextClipId = (QsRadiatorCard._nextClipId || 0) + 1;
        this._flameClipId = 'fClip-' + QsRadiatorCard._nextClipId;
      }
      const flameClipId = this._flameClipId;

      // QS-204: pre-generate paths so the SVG renders with flames immediately,
      // avoiding the empty-d="" flash between innerHTML rewrite and the first
      // RAF tick. The RAF loop continues animating from these initial shapes
      // seamlessly (memo keys synced after the innerHTML write).
      const initialAmp = this._currentFlameAmp ?? STILL_AMP;
      const initialBaseY = this._flameBaseY ?? CENTER_CY;
      const initialFlameColors = this._flameColors ?? FLAME_GREY_FILLS;
      // Initial tip phases — zero-filled if first paint, otherwise the
      // surviving per-tooth phases (re-attach after disconnect keeps the
      // flicker positions so the very first frame matches the last frame
      // before detach).
      if (!this._tipPhases) {
        this._tipPhases = LAYER_TEETH_COUNTS.map((count) => new Array(count).fill(0));
      }
      // QS-204 review-fix #03 H3 — parameterised by LAYER_TEETH_COUNTS
      // so the initial paint adapts to any layer count.
      const initialFlamePaths = Array.from(
          {length: LAYER_TEETH_COUNTS.length},
          (_, i) => this._generateFlameTeethPath(
              FLAME_WIDTH,
              initialBaseY,
              LAYER_BASE_HEIGHTS[i],
              initialAmp * LAYER_TIP_AMP_MULTS[i],
              LAYER_TEETH_COUNTS[i],
              this._tipPhases[i],
              !running,
          ),
      );

      const css = `
      :host {
        --pad: 18px;
        --ring-text-shadow: 0 0 12px rgba(0,0,0,0.8), 0 2px 4px rgba(0,0,0,0.5);
        display: block;
      }
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
      // QS-201: gate split. The ring-dash animation only needs RAF when
      // there is enough progress to dash through (segLen > 6). The flame
      // backdrop wants RAF as soon as the radiator is running, even with
      // zero progress yet. The umbrella `showAnimation` is the OR.
      const ringDashActive = running && segLen > 6;
      const fireActive = running;
      const showAnimation = ringDashActive || fireActive;

      // M4: start/stop the RAF loop based on whether the dash animation
      // is actually needed. Idle cards consume zero per-frame work.
      if (showAnimation) {
        this._startAnimation();
      } else {
        this._stopAnimation();
      }

      // Bistate mode selector options with translations
      const modeOptions = selBistateMode?.attributes?.options || [];
      const modeState = (selBistateMode?.state || '').trim();

      // Helper to translate bistate mode options
      const translateBistateMode = (value) => {
          const key = `component.quiet_solar.entity.select.radiator_mode.state.${value}`;
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

      // S15 — sanitise the title (a user-supplied entity / device name)
      // before it lands in innerHTML.
      // S16 — primary action `<div>`s get `role="button"` + `tabindex="0"`
      // so keyboard-only users can tab to and activate them. The Enter/
      // Space handlers are wired in the event-binding loops below.

      // QS-217 review-fix #03 — clipPath is just the outer disc;
      // the override-button area is hidden by a separate `<circle>`
      // cover drawn AFTER the clipped animation group (see the SVG
      // markup below). This replaces the earlier carve+cancel
      // clipPath approach, which produced a geometric lens-shape
      // hole.
      const clipPathD =
        `M ${CENTER_CY - CLIP_R},${CENTER_CY}` +
        ` a ${CLIP_R},${CLIP_R} 0 1,0 ${2 * CLIP_R},0` +
        ` a ${CLIP_R},${CLIP_R} 0 1,0 ${-2 * CLIP_R},0`;

      this._root.innerHTML = `
      <ha-card class="card ${!isEnabled ? 'disabled' : ''} ${isOffGrid ? 'off-grid' : ''}">
        <style>${css}</style>
        <div class="card-title">${this._escapeHtml(title)}</div>
        <div class="top"></div>

        <div class="hero">
          <div class="ring">
            ${swEnableDevice ? `<div id="power_btn" class="power-btn ${isEnabled ? 'on' : ''}" role="button" tabindex="0" aria-label="Toggle device"><ha-icon icon="mdi:power"></ha-icon></div>` : ''}
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
                <clipPath id="${flameClipId}">
                  <path clip-rule="evenodd" d="${clipPathD}" />
                </clipPath>
              </defs>
              <g clip-path="url(#${flameClipId})">
                ${initialFlamePaths.map((d, i) =>
                  `<path id="flame${i}" d="${d}" fill="${initialFlameColors[i]}" opacity="1" pointer-events="none" style="will-change: transform;" />`
                ).join("\n                ")}
              </g>
              ${e.override_reset ? `<circle id="override_btn_cover" cx="${CENTER_CY}" cy="${OVERRIDE_BTN_CARVE_CY}" r="${OVERRIDE_BTN_CARVE_R}" fill="var(--card-background-color)" pointer-events="none" />` : ''}
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
                  <div id="time_btn" class="time-btn ${finishTimeStr && finishTimeStr !== '--:--' ? 'on' : ''}" role="button" tabindex="0" aria-label="Change finish time">${formatTime(finishTimeStr)}</div>
                </div>
                ` : ''}
              </div>
            </div>
            ${e.override_reset ? `<div id="override_btn" class="${overrideBtnClass}" role="button" tabindex="0" aria-label="Reset override"><ha-icon icon="${overrideBtnIcon}"></ha-icon></div>` : ''}
            ${swGreenOnly ? `<div id="green_btn" class="green-btn ${swGreenOnly.state === 'on' ? 'on' : ''}" role="button" tabindex="0" aria-label="Toggle solar-only mode"><ha-icon icon="mdi:leaf"></ha-icon></div>` : ''}
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

      // QS-201: invalidate cached flame DOM refs after innerHTML rewrite so
      // the next RAF tick re-queries fresh elements.
      this._invalidateFlameCache();

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

if (!customElements.get('qs-radiator-card')) {
    customElements.define('qs-radiator-card', QsRadiatorCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-radiator-card')) {
    window.customCards.push({
        type: 'qs-radiator-card',
        name: 'QS Radiator Card',
        description: 'Quiet Solar radiator control card',
    });
}
