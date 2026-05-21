/*
  QS Pool Card - custom:qs-pool-card
  Zero-build single-file Lit-style web component compatible with Home Assistant
*/

// --- Geometry (must match the SVG <clipPath> circle attributes below) ---
const CENTER_CY = 160;            // SVG y-center of the ring / clip circle
const CLIP_R = 120;               // water clip circle radius
const WAVE_WIDTH = 480;           // single wave period in SVG px (2× clip diameter)
// Must be ≥ SVG viewBox max-y (320) so the closing rectangle is clipped.
const WAVE_BOTTOM_Y = 400;

// --- Per-layer wave offsets (different magic numbers, different purposes) ---
const LAYER_SCROLL_OFFSET = 1.2;  // per-layer extra scroll phase (visual depth)
const LAYER_PHASE_OFFSET = 2.1;   // per-layer static sine phase (shape variety)

// --- Water color (HSL endpoints) ---
const COOL_HUE = 195, WARM_HUE = 175;     // deeper blue-teal → warmer cyan
const COOL_SAT = 70,  WARM_SAT = 55;
const COOL_LIGHT = 18, WARM_LIGHT = 28;
const COOL_TEMP_C = 15;           // °C at which tint is coolest
const WARM_TEMP_C = 30;           // °C at which tint is warmest

// --- Fallback water colors (no temp sensor / NaN / non-number input) ---
const DEFAULT_WATER_COLORS = [
    'hsla(185, 60%, 22%, 0.55)',
    'hsla(185, 60%, 20%, 0.45)',
    'hsla(185, 60%, 18%, 0.35)',
];

// --- Animation tuning ---
const LERP_RATE = 2;              // exp time-constant; ~95% of lerp in ~1.5s
const LERP_DT_CEIL = 0.1;         // s; clamp lerp dt to avoid snap-to after tab hidden
const AMP_REGEN_THRESHOLD = 0.25; // amplitude delta threshold for path regen (throttle)
const LEVEL_REGEN_THRESHOLD = 0.01; // px; threshold for water-level path regen (jitter-proof)
const PHASE_WRAP = 1e6;           // wrap _wavePhase to preserve float precision
const PHASE_TO_PX = 60;           // scroll px per phase-unit

// --- Wave dynamics targets (calm vs pumping) ---
const CALM_AMP = 2,  PUMP_AMP = 6;
const CALM_SPEED = 0.3, PUMP_SPEED = 1.2;

class QsPoolCard extends HTMLElement {

  // Generate an SVG path for a sine-wave-based closed shape.
  // Emits TWO repetitions of the wave (path extent = [0, 2*width]) so the
  // translated path always covers the clip region regardless of scroll offset.
  // `frequency` is in cycles per `width`, so each period is identical.
  _generateWavePath(width, amplitude, frequency, phase, yOffset) {
    const points = [];
    const stepsPerPeriod = 60;
    const totalSteps = stepsPerPeriod * 2;
    const totalWidth = 2 * width;
    for (let i = 0; i <= totalSteps; i++) {
      const x = (i / stepsPerPeriod) * width; // 0 .. 2*width
      const y = yOffset + amplitude * Math.sin((x / width) * frequency * 2 * Math.PI + phase);
      points.push(`${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    return `M ${points[0]} ` +
           points.slice(1).map(p => `L ${p}`).join(' ') +
           ` L ${totalWidth.toFixed(2)} ${WAVE_BOTTOM_Y} L 0 ${WAVE_BOTTOM_Y} Z`;
  }

  // Map pool temperature (°C, finite number) to per-layer water HSL colors.
  // Cool end → deeper blue-teal; warm end → warmer cyan-turquoise.
  // Rejects null, undefined, NaN, Infinity, and any non-number input —
  // callers must coerce strings via `Number(...)` before calling.
  _tempToColor(tempC) {
    if (typeof tempC !== 'number' || !Number.isFinite(tempC)) {
      return DEFAULT_WATER_COLORS;
    }
    const span = WARM_TEMP_C - COOL_TEMP_C;
    const t = Math.max(0, Math.min(1, (tempC - COOL_TEMP_C) / span));
    const h = COOL_HUE + t * (WARM_HUE - COOL_HUE);
    const s = COOL_SAT + t * (WARM_SAT - COOL_SAT);
    const l = COOL_LIGHT + t * (WARM_LIGHT - COOL_LIGHT);
    return [
      `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${(l + 2).toFixed(0)}%, 0.55)`,
      `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${l.toFixed(0)}%, 0.45)`,
      `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${(l - 2).toFixed(0)}%, 0.35)`,
    ];
  }

  // Clear the wave-path memoization keys and cached DOM refs. Called on
  // every (re-)connect and after each _render() innerHTML rewrite.
  _invalidateWaveCache() {
    this._lastWaterBaseY = null;
    this._lastAmplitude = null;
    this._waveEls = null;
  }

  connectedCallback() {
    this._startAnimation();
  }

  // M4: refactor to start/stop helpers for cross-card consistency.
  // Unlike the other cards, the pool wave is intentionally continuous
  // while connected — the visual is the wave; calm vs. pump is the
  // amplitude. Idle-gating is therefore _stopAnimation in
  // disconnectedCallback only; no in-render gating.
  _startAnimation() {
    if (this._animRaf != null) return;
    // Initialize wave animation state ONLY on the first-ever connect, so
    // that detach/re-attach (HA dashboard rearrangement, tab navigation)
    // preserves _currentAmplitude/_currentSpeed/_wavePhase. Without this
    // guard, a pump-on wave would visibly snap back to CALM on each
    // reconnect and re-lerp up over ~1.5s.
    if (this._currentAmplitude == null) {
      this._currentAmplitude = CALM_AMP;
      this._currentSpeed = CALM_SPEED;
      this._wavePhase = 0;
      // Tell the next _render() to prime amp/speed directly to the actual
      // pump state, skipping the 1.5s lerp envelope at boot.
      this._needsAnimationPrime = true;
    }
    this._lastAnimTs = null;
    this._invalidateWaveCache();

    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      // QS-200 S6: cap `dt` against hidden-tab return. Without this,
      // the first frame after a multi-second tab-hidden window
      // produces a huge `dt`, snapping wave phase by hundreds of pixels
      // in one frame. The cap matches `LERP_DT_CEIL` so all step-loop
      // subsystems are bounded by the same envelope.
      // Review-fix #02 N3 trade-off note: the prior pool-card behavior
      // intentionally let phase accumulate raw `dt` so scroll "caught
      // up" after a hidden-tab return. Capping at 100ms trades that
      // catch-up for a single-frame visual freeze on return — defensible
      // either way; we picked the cap for cross-card consistency with
      // qs-water-boiler-card.js (which also drives bubble life off `dt`,
      // making the cap necessary there).
      let dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      dt = Math.min(dt, LERP_DT_CEIL);
      this._lastAnimTs = ts;
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const speed = 80; // dash units per second
      this._animOffset = ((this._animOffset || 0) + speed * dt) % patternLen;
      const p = this._root?.getElementById('running_anim');
      if (p) {
        p.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }

      // --- Wave animation update ---
      // `dt` is already clamped at LERP_DT_CEIL above (QS-200 S6), so
      // the lerpFactor envelope and phase advance share the same upper
      // bound. The lerp itself reads from the clamped `dt` directly —
      // no local `lerpDt` variable needed.
      const pumpOn = this._pumpRunning === true;
      const targetAmplitude = pumpOn ? PUMP_AMP : CALM_AMP;
      const targetSpeed = pumpOn ? PUMP_SPEED : CALM_SPEED;
      const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
      this._currentAmplitude += (targetAmplitude - this._currentAmplitude) * lerpFactor;
      this._currentSpeed += (targetSpeed - this._currentSpeed) * lerpFactor;
      this._wavePhase += this._currentSpeed * dt;
      // Modulo wrap is robust to any sign / magnitude of accumulated phase.
      this._wavePhase = this._wavePhase % PHASE_WRAP;

      // Lazy-resolve wave DOM refs once per innerHTML rewrite. This collapses
      // 6 getElementById calls/frame (3 in transform loop + 3 in regen loop)
      // down to 3 calls per render.
      if (!this._waveEls) {
        this._waveEls = [
          this._root?.getElementById('wave0') ?? null,
          this._root?.getElementById('wave1') ?? null,
          this._root?.getElementById('wave2') ?? null,
        ];
      }

      // Update wave transforms (CSS translateX for GPU compositing).
      // Path extent is [0, 2*WAVE_WIDTH] so the translated path always
      // covers the clip region. We offset by -CLIP_R to align the path
      // start with the clip's left edge, then scroll within one period.
      for (let i = 0; i < 3; i++) {
        const wEl = this._waveEls[i];
        if (wEl) {
          const phaseOffset = i * LAYER_SCROLL_OFFSET;
          const raw = (this._wavePhase + phaseOffset) * PHASE_TO_PX;
          const scrollOffset = ((raw % WAVE_WIDTH) + WAVE_WIDTH) % WAVE_WIDTH;
          const tx = -CLIP_R - scrollOffset;
          wEl.style.transform = `translateX(${tx.toFixed(1)}px)`;
        }
      }

      // Regenerate wave paths when water level or amplitude changes.
      // Guard against undefined/NaN base — the RAF loop can fire before
      // _render() populates _waterBaseY on cold start.
      const waterBaseY = this._waterBaseY;
      const hasValidBase = waterBaseY != null && !Number.isNaN(waterBaseY);
      const ampDelta = this._lastAmplitude == null
          ? Infinity
          : Math.abs(this._currentAmplitude - this._lastAmplitude);
      // Threshold compare for float-jitter robustness vs strict !==.
      const levelChanged = hasValidBase &&
          Math.abs(waterBaseY - (this._lastWaterBaseY ?? Number.NEGATIVE_INFINITY)) > LEVEL_REGEN_THRESHOLD;
      if (hasValidBase && (levelChanged || ampDelta > AMP_REGEN_THRESHOLD)) {
        this._lastWaterBaseY = waterBaseY;
        this._lastAmplitude = this._currentAmplitude;
        const colors = this._waterColors || DEFAULT_WATER_COLORS;
        for (let i = 0; i < 3; i++) {
          const wEl = this._waveEls[i];
          if (wEl) {
            const phaseOffset = i * LAYER_PHASE_OFFSET;
            const freq = 2 + i; // integer freq → seamless wrap at WAVE_WIDTH
            const d = this._generateWavePath(WAVE_WIDTH, this._currentAmplitude, freq, phaseOffset, waterBaseY);
            wEl.setAttribute('d', d);
            wEl.setAttribute('fill', colors[i]);
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

  disconnectedCallback() {
    this._stopAnimation();
    // S7: reset interaction flags so a re-attach after mid-interaction
    // (e.g. dragging the ring when the dashboard rearranges) doesn't
    // silently short-circuit `set hass` on stale flags.
    this._isInteractingMode = false;
    this._isProcessingModeChange = false;
    this._isInteractingTarget = false;
    this._modalOpen = false;
  }

  static getStubConfig() {
    return { name: "QS Pool", entities: {} };
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

  _fmt(num, round = true) {
    const n = Number(num);
    if (num == null || Number.isNaN(n)) return '--';
    return round ? Math.round(n) : n.toFixed(1);
  }

  _render() {
      const cfg = this._config || {};
      const e = cfg.entities || {};

      const sTemperature = this._entity(e.temperature_sensor);
      const sDurationLimit = this._entity(e.duration_limit);
      const sCurrentDailyRunDuration = this._entity(e.current_daily_run_duration);
      const sDefaultOnDuration = this._entity(e.default_on_duration);
      const sCommand = this._entity(e.command);
      const selPoolMode = this._entity(e.pool_mode);
      const swGreenOnly = this._entity(e.green_only);
      const swEnableDevice = this._entity(e.enable_device);
      const sIsOffGrid = this._entity(e.is_off_grid);

      const title = (cfg.title || cfg.name) || "Pool";
      
      // Check if device is enabled
      const isEnabled = swEnableDevice?.state === 'on';
      
      // Check if system is off-grid
      const isOffGrid = sIsOffGrid?.state === 'on';
      
      // Get pool temperature
      const temp = sTemperature?.state;
      const tempNum = Number(temp);
      const tempDisplay = temp != null && !Number.isNaN(tempNum) ? `${Math.round(tempNum)}°C` : '--';
      
      // Get target hours and current run hours directly (in hours)
      const maxHours = 24;
      const rawTarget = sDurationLimit ? Number(sDurationLimit.state) : null;
      const targetHours = (rawTarget != null && !Number.isNaN(rawTarget)) ? rawTarget : 0;
      const hoursRun = Number(sCurrentDailyRunDuration?.state) || 0;
      
      // Determine if pool is running (command state must be "on")
      const commandState = sCommand?.state || '';
      const running = commandState.toLowerCase() === 'on';

      // Store pump state for RAF loop
      this._pumpRunning = running;

      // Prime animation state to the actual pump targets on the first
      // render after connect — avoids a ~1.5s boot transient where the
      // wave lerps up from CALM_AMP to PUMP_AMP if the pump was already
      // running when the card mounted.
      if (this._needsAnimationPrime) {
        this._currentAmplitude = running ? PUMP_AMP : CALM_AMP;
        this._currentSpeed = running ? PUMP_SPEED : CALM_SPEED;
        this._needsAnimationPrime = false;
      }

      // Water level from runtime progress. Clamp on both sides: a negative
      // hoursRun (sensor reset glitch / device reporting -1) would otherwise
      // push waterBaseY past WAVE_BOTTOM_Y and self-intersect the polygon.
      const progressRatio = targetHours > 0
          ? Math.max(0, Math.min(1, hoursRun / targetHours))
          : 0;
      // Map 0..1 progress → 1/5..4/5 fill of the clip circle.
      // CENTER_CY / CLIP_R MUST match the <clipPath> circle attributes below.
      const waterBaseY = CENTER_CY + CLIP_R - (0.2 + progressRatio * 0.6) * 2 * CLIP_R;
      this._waterBaseY = Number.isNaN(waterBaseY) ? null : waterBaseY;

      // Temperature-based water colors
      this._waterColors = this._tempToColor(tempNum);

      // Pre-generate wave paths so the SVG renders with water immediately,
      // avoiding the empty-d="" flash between the innerHTML rewrite and the
      // next RAF tick. The RAF loop continues animating from these initial
      // shapes seamlessly (memo keys synced below after the innerHTML write).
      const initialAmp = this._currentAmplitude ?? CALM_AMP;
      const initialBaseY = this._waterBaseY ?? CENTER_CY;
      const initialColors = this._waterColors ?? DEFAULT_WATER_COLORS;
      const initialWavePaths = [0, 1, 2].map(i => {
        const freq = 2 + i;
        const phaseOffset = i * LAYER_PHASE_OFFSET;
        return this._generateWavePath(WAVE_WIDTH, initialAmp, freq, phaseOffset, initialBaseY);
      });

      // Instance-unique clip ID (stable across re-renders)
      if (!this._waterClipId) {
          QsPoolCard._nextClipId = (QsPoolCard._nextClipId || 0) + 1;
          this._waterClipId = 'wClip-' + QsPoolCard._nextClipId;
      }
      const waterClipId = this._waterClipId;

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

      .hero { margin-top: 0px; display:flex; align-items:center; justify-content:center; }
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
      .ring .center { position:absolute; inset:0; display:grid; place-items:center; text-align:center; pointer-events: none; transform: translateY(16px); }
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; text-shadow: var(--ring-text-shadow); }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; text-shadow: var(--ring-text-shadow); }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; text-shadow: var(--ring-text-shadow); }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; text-align:center; width: 180px; margin: 0 auto; }
      .ring .temp-block { display:flex; flex-direction:column; align-items:center; gap:2px; margin-top:4px; margin-bottom:8px; }
      .ring .stack > * { text-align:center; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:4px; margin-top:8px; }

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
      .dialog .actions { display:flex; gap:10px; justify-content:flex-end; margin-top: 6px; }
      .btn { border:none; border-radius:12px; padding:10px 14px; font-weight:700; cursor:pointer; touch-action: manipulation; min-height: 44px; -webkit-tap-highlight-color: transparent; }
      .btn.secondary { background: rgba(255,255,255,.06); color: var(--primary-text-color); border:1px solid var(--divider-color); }
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
          const p0 = polar(cx, cy, r, a0);
          const p1 = polar(cx, cy, r, a1);
          let delta = a1 - a0;
          if (delta < 0) delta += 360;
          const laf = delta > 180 ? 1 : 0;
          return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${laf} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
      };

      // Convert hours (0-24) to percentage (0-100) for arc calculation
      const hoursToPct = (hours) => (hours / maxHours) * 100;
      const pctToHours = (pct) => (pct / 100) * maxHours;
      
      const pctToDeg = (p) => startDeg + (Math.max(0, Math.min(100, p)) / 100) * rangeDeg;

      // Progress: hours run as percentage of max hours
      const progressPct = hoursToPct(hoursRun);
      const progressEndDeg = pctToDeg(progressPct);
      
      // Handle: target hours (always show when enabled so user can drag from 0)
      const hasValidTarget = isEnabled;
      const handleTargetHours = targetHours > 0 ? targetHours
          : (sDefaultOnDuration ? Number(sDefaultOnDuration.state) || 0 : 0);
      const handlePct = this._targetDragPct != null ? this._targetDragPct :
                        (this._localTargetPct != null ? this._localTargetPct : hoursToPct(handleTargetHours));
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
      const showAnimation = (running && segLen > 6);

      // Pool mode selector options with translations
      const modeOptions = selPoolMode?.attributes?.options || [];
      const modeState = (selPoolMode?.state || '').trim();
      
      // Helper to translate pool mode options
      const translatePoolMode = (value) => {
          const key = `component.quiet_solar.entity.select.pool_mode.state.${value}`;
          const translated = this._hass?.localize?.(key);
          // If translation not found or returns the key itself, fall back to the raw value
          return (translated && translated !== key) ? translated : value;
      };
      
      const modeOptionsHtml = modeOptions.map(o => 
          `<option value="${o}" ${o === modeState ? 'selected' : ''}>${translatePoolMode(o)}</option>`
      ).join('');

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
                  <stop offset="0%" stop-color="#00bcd4"/>
                  <stop offset="100%" stop-color="#8bc34a"/>
                </linearGradient>
                <linearGradient id="${gradRunningId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#00e1ff"/>
                  <stop offset="100%" stop-color="#0066ff"/>
                </linearGradient>
                <filter id="runningGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <clipPath id="${waterClipId}">
                  <circle cx="160" cy="${CENTER_CY}" r="${CLIP_R}" />
                </clipPath>
              </defs>
              <g clip-path="url(#${waterClipId})">
                <path id="wave0" d="${initialWavePaths[0]}" fill="${initialColors[0]}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <path id="wave1" d="${initialWavePaths[1]}" fill="${initialColors[1]}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <path id="wave2" d="${initialWavePaths[2]}" fill="${initialColors[2]}" opacity="1" pointer-events="none" style="will-change: transform;" />
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
              ${hasValidTarget ? `
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="var(--primary-color)" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="var(--primary-color)" font-size="13" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${this._fmt(pctToHours(handlePct))}</text>
              ` : ''}
            </svg>
            <div class="center">
              <div class="stack">
                <div class="temp-block">
                  <div class="pct" style="margin-bottom:0;">${tempDisplay}</div>
                </div>
                <div class="target-block">
                  <div class="target-label">Actual / Target Hours</div>
                  <div class="target-value">
                    <span style="color: var(--primary-text-color);">${this._fmt(hoursRun, false)}h</span>
                    <span style="color: var(--primary-text-color);"> / </span>
                    <span style="color: var(--primary-color);">${this._fmt(targetHours)}h</span>
                  </div>
                </div>
              </div>
            </div>
            <div id="green_btn" class="green-btn ${swGreenOnly?.state === 'on' ? 'on' : ''}"><ha-icon icon="mdi:leaf"></ha-icon></div>
          </div>
        </div>

        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:pool"></ha-icon>
            <select id="pool_mode">
              ${modeOptionsHtml}
            </select>
          </div>
        </div>
        <div class="below-line full">
           <button id="reset" class="danger pill outline">Reset</button>
        </div>
      </ha-card>
    `;

      // Wire events
      const root = this._root;
      const ids = (k) => root.getElementById(k);

      // innerHTML rewrite just replaced the wave <path> nodes. The new nodes
      // already carry the freshly-generated `d` and `fill` (initialWavePaths
      // / initialColors), so we sync RAF's memo keys to the rendered state
      // — this prevents a redundant regen on the next frame. We also clear
      // the cached wave element refs so RAF re-resolves them lazily.
      this._lastWaterBaseY = this._waterBaseY;
      this._lastAmplitude = initialAmp;
      this._waveEls = null;

      // Pool mode selector
      if (selPoolMode) {
          const modeSel = ids('pool_mode');
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
              // network drop) would leave the flag wedged forever.
              try {
                  // Call the service and wait for it to complete
                  await this._select(e.pool_mode, option);
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
          }
      }

      // Reset button
      if (e.reset) {
          const resetBtn = ids('reset');
          const resetAction = async () => {
              // Clear any stale drag/local target state before reset
              this._localTargetPct = null;
              this._targetDragPct = null;
              this._targetDragValue = null;
              if (this._pendingClearLocalTarget) {
                  clearTimeout(this._pendingClearLocalTarget);
                  this._pendingClearLocalTarget = null;
              }
              showDialog({
                  title: 'Reset pool state',
                  message: 'This will reset internal state for the pool and cannot be undone.\nProceed?',
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

      const showDialog = (opts) => {
          const {title, message, buttons} = opts;
          const wrap = document.createElement('div');
          wrap.className = 'modal';
          // S6: escape user-controlled `title` and `message`.
          wrap.innerHTML = `<div class="dialog"><h3>${this._escapeHtml(title)}</h3><p>${this._escapeHtml(message)}</p><div class="actions"></div></div>`;
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

      // Drag target handle on ring
      const svg = this._root.querySelector('.ring svg');
      const handle = this._root.getElementById('target_handle');
      if (svg && handle) {
          const pt = svg.createSVGPoint();

          const allowedHours = Array.from({length: 25}, (_, i) => i);

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
                  handleText.textContent = this._fmt(snapHours);
              }
              const tv = this._root.querySelector('.target-value');
              if (tv) {
                  const currentRunHours = Number(sCurrentDailyRunDuration?.state || 0);
                  tv.innerHTML = `<span style="color: var(--primary-text-color);">${this._fmt(currentRunHours, false)}h</span><span style="color: var(--primary-text-color);"> / </span><span style="color: var(--primary-color);">${this._fmt(snapHours)}h</span>`;
              }
          };

          const onUp = async (ev) => {
              if (this._upInProgress) return;
              this._upInProgress = true;

              if (ev) {
                  ev.stopPropagation();
                  ev.preventDefault();
              }

              // Capture drag values before any async work so they survive concurrent calls
              const dragPct = this._targetDragPct;
              const dragValue = this._targetDragValue;

              if (dragValue != null && e.default_on_duration && e.pool_mode) {
                  await this._select(e.pool_mode, 'bistate_mode_default');
                  await this._setNumber(e.default_on_duration, dragValue);
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
              // Re-render to pick up any hass updates skipped during interaction
              // (don't touch handle.style — the DOM node may be detached after render)
              this._render();
          };

          if (window.PointerEvent) {
              // Prefer PointerEvent: captures all input types and supports pointer capture
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
              // Fallback for browsers without PointerEvent
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

if (!customElements.get('qs-pool-card')) {
    customElements.define('qs-pool-card', QsPoolCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-pool-card')) {
    window.customCards.push({
        type: 'qs-pool-card',
        name: 'QS Pool Card',
        description: 'Quiet Solar pool control card',
    });
}
