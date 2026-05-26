/*
  QS Pool Card - custom:qs-pool-card

  Zero-build single-file Lit-style web component compatible with Home
  Assistant. Extends `QsRingDurationCardBase` and composes
  `generateWavePath` from `shared/qs-anim-wave.js` for the 3-layer
  water wave backdrop. Retains all pool-specific tuning: temperature
  → HSL palette mapping, CALM/PUMP amplitude lerp, 3-layer scroll
  offsets, and the always-visible RAF model (the wave is the visual,
  calm vs pump is the amplitude).
*/

import { baseCardCSS } from './shared/qs-card-styles.js';
import { QsRingDurationCardBase } from './shared/qs-ring-duration-base.js';
import { arcPath, polar, pctToDeg } from './shared/qs-card-base.js';
// QS-199 review-fix S1 — pool keeps its own `_generateWavePath` (emits
// 2× width for the GPU translateX scroll), so it does NOT import the
// shared single-period `generateWavePath`.

// --- Geometry (must match the SVG <clipPath> circle attributes below) ---
const CENTER_CY = 160;
const CLIP_R = 120;
const WAVE_WIDTH = 480;
const WAVE_BOTTOM_Y = 400;

// --- Per-layer wave offsets (kept local — pool-specific values) ---
const LAYER_SCROLL_OFFSET = 1.2;
const LAYER_PHASE_OFFSET = 2.1;

// --- Water color (HSL endpoints) ---
const COOL_HUE = 210, WARM_HUE = 200;
const COOL_SAT = 65,  WARM_SAT = 55;
const COOL_LIGHT = 32, WARM_LIGHT = 42;
const COOL_TEMP_C = 15;
const WARM_TEMP_C = 30;

// --- Fallback water colors (no temp sensor / NaN / non-number input) ---
const DEFAULT_WATER_COLORS = [
    'hsla(210, 65%, 35%, 0.30)',
    'hsla(210, 65%, 33%, 0.20)',
    'hsla(210, 65%, 31%, 0.12)',
];

// --- Animation tuning ---
const LERP_RATE = 2;
const LERP_DT_CEIL = 0.1;
const AMP_REGEN_THRESHOLD = 0.25;
const LEVEL_REGEN_THRESHOLD = 0.01;
const PHASE_WRAP = 1e6;
const PHASE_TO_PX = 60;

// --- Wave dynamics targets (calm vs pumping) ---
const CALM_AMP = 2,  PUMP_AMP = 6;
const CALM_SPEED = 0.3, PUMP_SPEED = 1.2;


class QsPoolCard extends QsRingDurationCardBase {
    static getStubConfig() {
        return { name: "QS Pool", entities: {} };
    }

    getCardSize() { return 5; }

    connectedCallback() {
        super.connectedCallback();
        this._startAnimation();
    }

    // Pool wave is intrinsically continuous while connected — the visual
    // IS the wave; calm vs pump is the amplitude. Override _startAnimation
    // to inject pool's per-frame work via the _onAnimationTick hook.
    _startAnimation() {
        if (this._animRaf != null) return;
        if (this._currentAmplitude == null) {
            this._currentAmplitude = CALM_AMP;
            this._currentSpeed = CALM_SPEED;
            this._wavePhase = 0;
            this._needsAnimationPrime = true;
        }
        this._invalidateWaveCache();
        super._startAnimation();
    }

    _onAnimationTick(_ts, dt) {
        // QS-200 S6: cap RAF dt against hidden-tab return. Without this,
        // the first frame after a multi-second tab-hidden window
        // produces a huge `dt`, snapping wave phase by hundreds of pixels
        // in one frame. The cap matches LERP_DT_CEIL so all step-loop
        // subsystems are bounded by the same envelope.
        dt = Math.min(dt, LERP_DT_CEIL);

        const pumpOn = this._pumpRunning === true;
        const targetAmplitude = pumpOn ? PUMP_AMP : CALM_AMP;
        const targetSpeed = pumpOn ? PUMP_SPEED : CALM_SPEED;
        const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);
        this._currentAmplitude += (targetAmplitude - this._currentAmplitude) * lerpFactor;
        this._currentSpeed += (targetSpeed - this._currentSpeed) * lerpFactor;
        this._wavePhase += this._currentSpeed * dt;
        this._wavePhase = this._wavePhase % PHASE_WRAP;

        // Lazy-resolve wave DOM refs once per innerHTML rewrite.
        if (!this._waveEls) {
            this._waveEls = [
                this._root?.getElementById('wave0') ?? null,
                this._root?.getElementById('wave1') ?? null,
                this._root?.getElementById('wave2') ?? null,
            ];
        }

        // Update wave transforms (CSS translateX for GPU compositing).
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

        // Regenerate wave paths when water level or amplitude changes appreciably.
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
            const colors = this._waterColors || DEFAULT_WATER_COLORS;
            for (let i = 0; i < 3; i++) {
                const wEl = this._waveEls[i];
                if (wEl) {
                    const phaseOffset = i * LAYER_PHASE_OFFSET;
                    const freq = 2 + i;
                    const d = this._generateWavePath(WAVE_WIDTH, this._currentAmplitude, freq, phaseOffset, waterBaseY);
                    wEl.setAttribute('d', d);
                    wEl.setAttribute('fill', colors[i]);
                }
            }
        }
    }

    /*
      _generateWavePath — pool-specific wave path. Pool emits TWO
      repetitions of the wave (path extent = [0, 2*width]) so the
      translated path always covers the clip region regardless of scroll
      offset. The shared `shared/qs-anim-wave.js#generateWavePath` only
      emits one period, so pool keeps its own widened version rather than
      importing it (QS-199 review-fix M1/S1).
    */
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
            points.slice(1).map((p) => `L ${p}`).join(' ') +
            ` L ${totalWidth.toFixed(2)} ${WAVE_BOTTOM_Y} L 0 ${WAVE_BOTTOM_Y} Z`;
    }

    // Map pool temperature (°C, finite number) to per-layer water HSL colors.
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
            `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${(l + 2).toFixed(0)}%, 0.30)`,
            `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${l.toFixed(0)}%, 0.20)`,
            `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${(l - 2).toFixed(0)}%, 0.12)`,
        ];
    }

    _invalidateWaveCache() {
        this._lastWaterBaseY = null;
        this._lastAmplitude = null;
        this._waveEls = null;
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this._invalidateWaveCache();
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

        const isEnabled = swEnableDevice?.state === 'on';
        const isOffGrid = sIsOffGrid?.state === 'on';

        const temp = sTemperature?.state;
        const tempNum = Number(temp);
        const tempDisplay = temp != null && !Number.isNaN(tempNum) ? `${Math.round(tempNum)}°C` : '--';

        const maxHours = 24;
        const rawTarget = sDurationLimit ? Number(sDurationLimit.state) : null;
        const targetHours = (rawTarget != null && !Number.isNaN(rawTarget)) ? rawTarget : 0;
        const hoursRun = Number(sCurrentDailyRunDuration?.state) || 0;

        const commandState = sCommand?.state || '';
        const running = commandState.toLowerCase() === 'on';

        this._pumpRunning = running;

        if (this._needsAnimationPrime) {
            this._currentAmplitude = running ? PUMP_AMP : CALM_AMP;
            this._currentSpeed = running ? PUMP_SPEED : CALM_SPEED;
            this._needsAnimationPrime = false;
        }

        const progressRatio = targetHours > 0
            ? Math.max(0, Math.min(1, hoursRun / targetHours))
            : 0;
        const waterBaseY = CENTER_CY + CLIP_R - (0.2 + progressRatio * 0.6) * 2 * CLIP_R;
        this._waterBaseY = Number.isNaN(waterBaseY) ? null : waterBaseY;

        this._waterColors = this._tempToColor(tempNum);

        // Pre-generate wave paths so the SVG renders with water immediately.
        const initialAmp = this._currentAmplitude ?? CALM_AMP;
        const initialBaseY = this._waterBaseY ?? CENTER_CY;
        const initialColors = this._waterColors ?? DEFAULT_WATER_COLORS;
        const initialWavePaths = [0, 1, 2].map((i) => {
            const freq = 2 + i;
            const phaseOffset = i * LAYER_PHASE_OFFSET;
            return this._generateWavePath(WAVE_WIDTH, initialAmp, freq, phaseOffset, initialBaseY);
        });

        if (!this._waterClipId) this._waterClipId = this._instanceId('wClip');
        const waterClipId = this._waterClipId;

        // Pool palette (passes through to baseCardCSS).
        const colors = {
            primary: '#2196F3',
            gradStart: '#00bcd4',
            gradEnd: '#8bc34a',
            animStart: '#00e1ff',
            animEnd: '#0066ff',
        };

        const css = baseCardCSS(colors) + `
      .ring .center { transform: translateY(16px); }
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; text-shadow: var(--ring-text-shadow); }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; text-shadow: var(--ring-text-shadow); }
      .ring .stack { width: 180px; gap: 4px; }
      .ring .temp-block { display:flex; flex-direction:column; align-items:center; gap:2px; margin-top:4px; margin-bottom:8px; }
      .ring .stack > * { text-align:center; }
      .ring .target-block { margin-top:8px; gap:4px; }
    `;

        const ringCirc = 130;
        const gapDeg = 60;
        const rangeDeg = 360 - gapDeg;
        const startDeg = gapDeg / 2;
        const endDeg = startDeg + rangeDeg;

        const hoursToPct = (hours) => (hours / maxHours) * 100;
        const pctToHours = (pct) => (pct / 100) * maxHours;

        const progressPct = hoursToPct(hoursRun);
        const progressEndDeg = pctToDeg(progressPct, startDeg, rangeDeg);

        const hasValidTarget = isEnabled;
        const handleTargetHours = targetHours > 0 ? targetHours
            : (sDefaultOnDuration ? Number(sDefaultOnDuration.state) || 0 : 0);
        const handlePct = this._targetDragPct != null ? this._targetDragPct :
            (this._localTargetPct != null ? this._localTargetPct : hoursToPct(handleTargetHours));
        const handleDeg = pctToDeg(handlePct, startDeg, rangeDeg);

        const center = { cx: 160, cy: 160 };
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

        const gradGreenId = this._instanceId('gradG');
        const gradRunningId = this._instanceId('gradR');
        const activeGradId = running ? gradRunningId : gradGreenId;
        const showAnimation = (running && segLen > 6);

        // Pool mode selector options with translations
        const modeOptions = selPoolMode?.attributes?.options || [];
        const modeState = (selPoolMode?.state || '').trim();
        const translatePoolMode = (value) => {
            const key = `component.quiet_solar.entity.select.pool_mode.state.${value}`;
            const translated = this._hass?.localize?.(key);
            return (translated && translated !== key) ? translated : value;
        };
        // S4: escape the entity-derived option value + label before
        // interpolating into innerHTML (matches the radiator pattern).
        const modeOptionsHtml = modeOptions.map((o) =>
            `<option value="${this._escapeHtml(o)}" ${o === modeState ? 'selected' : ''}>${this._escapeHtml(translatePoolMode(o))}</option>`,
        ).join('');

        // Build the ring SVG markup via the sub-base helper (NOTE: pool's
        // backdrop is the wave clip + paths, NOT layered onto _buildRingHTML
        // — pool renders the wave clip group BEFORE the ring arcs).
        const ringMarkup = this._buildRingHTML({
            palette: colors, ringCirc, center,
            startDeg, endDeg, rangeDeg, gapDeg,
            progressPath, bgPath,
            handlePos, handlePct, displayTargetHours: handleTargetHours, hoursRun,
            showAnimation, canDragHandle: hasValidTarget,
            gradGreenId, gradRunningId, activeGradId,
            dashLen, gapLen,
            pctToHours,
        });

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
                <clipPath id="${waterClipId}">
                  <circle cx="160" cy="${CENTER_CY}" r="${CLIP_R}" />
                </clipPath>
              </defs>
              <g clip-path="url(#${waterClipId})">
                <path id="wave0" d="${initialWavePaths[0]}" fill="${initialColors[0]}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <path id="wave1" d="${initialWavePaths[1]}" fill="${initialColors[1]}" opacity="1" pointer-events="none" style="will-change: transform;" />
                <path id="wave2" d="${initialWavePaths[2]}" fill="${initialColors[2]}" opacity="1" pointer-events="none" style="will-change: transform;" />
              </g>
              ${ringMarkup}
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
            <div id="green_btn" class="green-btn ${swGreenOnly?.state === 'on' ? 'on' : ''}" role="button" tabindex="0" aria-label="Toggle solar-only mode"><ha-icon icon="mdi:leaf"></ha-icon></div>
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

        // Sync RAF memo keys to rendered state.
        this._lastWaterBaseY = this._waterBaseY;
        this._lastAmplitude = initialAmp;
        this._waveEls = null;

        // Wire events
        const root = this._root;
        const ids = (k) => root.getElementById(k);

        if (selPoolMode) {
            this._wireBistateMode({
                selectEl: ids('pool_mode'),
                entityId: e.pool_mode,
                translationNamespace: 'pool_mode',
            });
        }

        if (swGreenOnly) {
            this._wireGreenButton({
                buttonEl: ids('green_btn'),
                swEntity: swGreenOnly,
                entityId: e.green_only,
            });
        }

        if (swEnableDevice) {
            this._wirePowerButton({
                buttonEl: ids('power_btn'),
                swEntity: swEnableDevice,
                entityId: e.enable_device,
            });
        }

        if (e.reset) {
            this._wireResetButton({
                buttonEl: ids('reset'),
                entityId: e.reset,
            });
        }

        if (hasValidTarget) {
            const allowedHours = Array.from({ length: 25 }, (_, i) => i);

            this._wireTargetHandle({
                ringSvg: this._root.querySelector('.ring svg'),
                handle: this._root.getElementById('target_handle'),
                center, ringCirc,
                startDeg, endDeg, rangeDeg,
                hoursToPct, pctToHours, allowedHours,
                entityId: e.default_on_duration,
                getHoursRun: () => hoursRun,
                colors,
                onCommit: async () => {
                    // Pool's commit pipeline also sets the bistate mode to default
                    // so the drag ALWAYS lands the device in default-mode operation.
                    if (e.pool_mode) {
                        try { await this._select(e.pool_mode, 'bistate_mode_default'); } catch (_) { /* ignore */ }
                    }
                },
            });
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
