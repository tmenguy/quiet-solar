/*
  QS Radiator Card - custom:qs-radiator-card

  Zero-build single-file Lit-style web component compatible with Home
  Assistant. Extends `QsRingDurationCardBase` and composes a
  `QsFlameEngine` for the heat-mode flame backdrop.

  QS-199 refactor: the lifecycle, service callers, defensive utilities,
  ring HTML builder, dialog system, drag-handle, time picker, reset/
  power/green buttons, override button, and bistate-mode handler all
  live in the shared base (see shared/qs-card-base.js and
  shared/qs-ring-duration-base.js). The flame state-machine + path
  generator lives in shared/qs-anim-flame.js. This file retains:
    - per-card palette (`FLAME_FILLS`, `FLAME_GREY_FILLS`, `colors`)
    - per-card per-layer config (LAYER_*)
    - the geometry constants for the clip circle and override-btn cover
    - the umbrella RAF tick override (`_onAnimationTick`) that advances
      the flame engine alongside the dashed-arc loop

  History preserved as inline comments — the redesign markers
  (`_generateFlameTeethPath`, `LAYER_BASE_HEIGHTS`, `STILL_AMP`,
  `STATIC_PEAK_HEIGHT`, `PHASE_REGEN_MIN_DT`, `sinceLastRegen`, `isIdle`)
  are referenced by `tests/test_radiator_card_smoke.py` via
  `card_source_union('qs-radiator-card.js')`.
*/

import { baseCardCSS } from './shared/qs-card-styles.js';
import { QsRingDurationCardBase } from './shared/qs-ring-duration-base.js';
import { arcPath, polar, pctToDeg } from './shared/qs-card-base.js';
import { FLAME_CONSTANTS, QsFlameEngine } from './shared/qs-anim-flame.js';

// Local aliases for the engine constants — keeps the radiator-card
// formula readable (mirrors pool-card) and pins the smoke-test literals
// (`FLAME_BASE_MIN_PCT`, `FLAME_BASE_MAX_PCT`) in this file.
const { FLAME_BASE_MIN_PCT, FLAME_BASE_MAX_PCT } = FLAME_CONSTANTS;

// --- Geometry (must match the <clipPath> circle attributes below) ---
const CENTER_CY = 160;              // SVG y-centre of the ring / clip circle
const CLIP_R = 120;                 // flame clip circle radius

// QS-217 — Override-button cover overlay geometry.
const OVERRIDE_BTN_CARVE_CY = 277;
const OVERRIDE_BTN_CARVE_R  = 35;

// --- Per-layer tuning ---
// LAYER_BASE_HEIGHTS — back layer reaches ≈ half clip radius (~100 px)
// so the back-flame tip approaches the top of the clip circle, which
// is what makes the silhouette read as flames rather than ripples.
const LAYER_BASE_HEIGHTS = [150, 120, 90];
const LAYER_TIP_AMP_MULTS = [1.2, 1.0, 0.8];  // multiplies _currentFlameAmp per layer
const LAYER_TEETH_COUNTS = [3, 4, 5];          // fewer wider teeth read more like flames
const LAYER_TIP_FLICKER_HZ = [8, 7, 9];

// QS-204 review-fix #02 G7 — length-equality guard. All four per-layer
// arrays MUST stay the same length; a future PR that extends one without
// the others would silently produce `NaN` paths (out-of-bounds reads
// yield undefined → arithmetic → NaN). `console.assert` is the
// browser-side equivalent of a Python `assert`. Reflected by both the
// per-card guard here AND the engine constructor's identical check.
console.assert(
    LAYER_TEETH_COUNTS.length === LAYER_TIP_FLICKER_HZ.length &&
    LAYER_TEETH_COUNTS.length === LAYER_BASE_HEIGHTS.length &&
    LAYER_TEETH_COUNTS.length === LAYER_TIP_AMP_MULTS.length,
    "qs-radiator-card: LAYER_* constants must be the same length"
);

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


class QsRadiatorCard extends QsRingDurationCardBase {
    constructor() {
        super();
        this._flameEngine = new QsFlameEngine({
            layerTeethCounts: LAYER_TEETH_COUNTS,
            layerTipFlickerHz: LAYER_TIP_FLICKER_HZ,
            layerBaseHeights: LAYER_BASE_HEIGHTS,
            layerTipAmpMults: LAYER_TIP_AMP_MULTS,
            runningFills: FLAME_FILLS,
            greyFills: FLAME_GREY_FILLS,
        });
    }

    static getStubConfig() {
        return { name: "QS Radiator", entities: {} };
    }

    getCardSize() { return 5; }

    disconnectedCallback() {
        super.disconnectedCallback();
        // QS-204: clear flame DOM cache and memoization keys. The engine's
        // `_currentFlameAmp` and `_tipPhases` deliberately survive so
        // re-attach resumes the dance without a visible jump.
        this._invalidateFlameCache();
    }

    // QS-204 — clear cached flame DOM refs after each _render() innerHTML
    // rewrite, and delegate the engine memo-key reset to the engine.
    // QS-199 review-fix #03 N5 — the card no longer mirrors
    // `_lastFlameBaseY` / `_lastFlameAmp`; they were dead state
    // (`_onAnimationTick` reads `shouldRegen` straight off
    // `_flameEngine.step(...)` and never consults card mirrors — the
    // engine owns them). The engine's `_currentFlameAmp` / `_tipPhases`
    // deliberately survive disconnect so re-attach resumes the dance
    // without a visible jump.
    _invalidateFlameCache() {
        this._flameEls = null;
        this._flameEngine.invalidate();
    }

    // Subclass hook — advances the flame engine on each RAF tick alongside
    // the umbrella dashed-arc loop driven by the base.
    _onAnimationTick(ts, dt) {
        const fireOn = this._fireRunning === true;
        const baseY = this._flameBaseY;

        const { shouldRegen } = this._flameEngine.step(dt, fireOn, baseY, ts);

        // Lazy-resolve flame DOM refs once per innerHTML rewrite.
        if (!this._flameEls) {
            this._flameEls = Array.from(
                { length: LAYER_TEETH_COUNTS.length },
                (_, i) => this._root?.getElementById(`flame${i}`) ?? null,
            );
        }

        if (shouldRegen) {
            const flameColors = this._flameColors || FLAME_GREY_FILLS;
            const paths = this._flameEngine.generatePaths(baseY, !fireOn);
            for (let i = 0; i < LAYER_TEETH_COUNTS.length; i++) {
                const fEl = this._flameEls[i];
                if (fEl) {
                    fEl.setAttribute('d', paths[i]);
                    fEl.setAttribute('fill', flameColors[i]);
                }
            }
        }

        // QS-199 review-fix #02 S7 — once the radiator is off AND the
        // flame has fully decayed to its still amplitude, stop the RAF.
        // (When `!fireOn` the dashed-arc isn't running either, so this is
        // safe.) The base step honours the null `_animRaf` and won't
        // reschedule. The last regen above already painted the settled
        // idle silhouette, so the final frame is a clean still flame
        // rather than a frozen mid-flicker shape.
        if (!fireOn && this._flameEngine.isIdle()) {
            this._stopAnimation();
        }
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

        const isEnabled = swEnableDevice?.state === 'on';
        const isOffGrid = sIsOffGrid?.state === 'on';

        const targetHours = this._safeNumber(sDurationLimit, 12);
        const hoursRun = this._safeNumber(sCurrentDuration, 0);
        const defaultDuration = this._safeNumber(sDefaultOnDuration, 6);

        const bistateMode = selBistateMode?.state || 'bistate_mode_default';
        const isDefaultMode = bistateMode === 'bistate_mode_default';

        const overrideState = sOverrideState?.state || 'NO OVERRIDE';
        const isOverridden = overrideState !== 'NO OVERRIDE';
        const isResettingOverride = overrideState === 'ASKED FOR RESET OVERRIDE';

        // N7 + BH10 cold-start fallback: backing-entity OR-in when QS command
        // sensor hasn't been published yet.
        const commandState = sCommand?.state || '';
        const commandReportsOn = commandState.toLowerCase() === 'on';
        const backingEntityId = e.backing_entity;
        const liveBackingState = backingEntityId
            ? (this._hass?.states?.[backingEntityId]?.state || '').toLowerCase()
            : '';
        const configuredHvacOn = String(e.climate_hvac_mode_on || 'heat').toLowerCase();
        const liveBackingOn = liveBackingState === 'on' || liveBackingState === configuredHvacOn;
        const running = commandReportsOn || liveBackingOn;

        let maxHours;
        let displayTargetHours;
        if (isDefaultMode) {
            maxHours = this._clampMaxHours(cfg.max_default_hours);
            displayTargetHours = defaultDuration;
        } else {
            maxHours = this._clampMaxHours(targetHours > 0 ? targetHours : cfg.max_default_hours);
            displayTargetHours = targetHours;
        }

        // QS-201: flame backdrop — height tracks hoursRun/maxHours (mirrors pool).
        const progressRatio = maxHours > 0
            ? Math.max(0, Math.min(1, hoursRun / maxHours))
            : 0;
        const flameBaseY = CENTER_CY + CLIP_R - (FLAME_BASE_MIN_PCT + progressRatio * (FLAME_BASE_MAX_PCT - FLAME_BASE_MIN_PCT)) * 2 * CLIP_R;
        this._fireRunning = running;
        // QS-199 review-fix #03 N3 — `Number.isFinite` (not `!isNaN`) so an
        // Infinity baseY can't reach `getInitialPaths(...)` →
        // `baseY.toFixed(2)` → `"Infinity"` in the SVG `d` (consistent with
        // the engine's S11 regen gate). Latent today (inputs are finite).
        this._flameBaseY = Number.isFinite(flameBaseY) ? flameBaseY : null;
        this._flameColors = running ? FLAME_FILLS : FLAME_GREY_FILLS;

        // Honor first-connect prime: skip the 1.5s boot lerp.
        this._flameEngine.primeForCurrentState(running);

        const showFromTo = !isDefaultMode || isOverridden;

        const startTime = (sStartTime && this._isValidState(sStartTime.state)) ? sStartTime.state : '--:--';
        const endTime = (sEndTime && this._isValidState(sEndTime.state)) ? sEndTime.state : '--:--';

        // Heat palette — warm orange/red.
        const colors = {
            primary: '#FF5722',
            gradStart: '#FF5722',
            gradEnd: '#D32F2F',
            animStart: '#FF6E40',
            animEnd: '#E64A19',
        };

        // QS-201: per-instance clip id so multiple radiator cards on one
        // dashboard don't collide.
        if (!this._flameClipId) this._flameClipId = this._instanceId('fClip');
        const flameClipId = this._flameClipId;

        // QS-204: pre-generate paths so the SVG renders with flames
        // immediately, avoiding the empty-d="" flash between the
        // innerHTML rewrite and the first RAF tick. The engine owns its
        // amp + tip-phase state; we go through the public
        // `getInitialPaths(baseY, isIdle)` accessor (QS-199 review-fix
        // M2 — cards must not dot into engine private fields).
        const initialBaseY = this._flameBaseY ?? CENTER_CY;
        const initialFlameColors = this._flameColors ?? FLAME_GREY_FILLS;
        const initialFlamePaths = this._flameEngine.getInitialPaths(initialBaseY, !running);

        const css = baseCardCSS(colors);

        const ringCirc = 130;
        const gapDeg = 60;
        const rangeDeg = 360 - gapDeg;
        const startDeg = gapDeg / 2;
        const endDeg = startDeg + rangeDeg;

        const hoursToPct = (hours) => (hours / maxHours) * 100;
        const pctToHours = (pct) => (pct / 100) * maxHours;

        const progressPct = hoursToPct(hoursRun);
        const progressEndDeg = pctToDeg(progressPct, startDeg, rangeDeg);

        const handlePct = this._targetDragPct != null ? this._targetDragPct :
            (this._localTargetPct != null ? this._localTargetPct : hoursToPct(displayTargetHours));
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
        // QS-201: gate split. The ring-dash animation only needs RAF when
        // there is enough progress to dash through. The flame backdrop
        // wants RAF as soon as the radiator is running. The umbrella
        // showAnimation is the OR.
        // QS-199 review-fix #02 S7 — also keep the loop alive while the
        // flame is still decaying to idle after an on→off transition, so
        // it settles to a clean still silhouette instead of freezing
        // mid-flicker. `_onAnimationTick` stops the loop once the engine
        // reports `isIdle()`.
        const ringDashActive = running && segLen > 6;
        const fireActive = running;
        const flameSettling = !this._flameEngine.isIdle();
        const showAnimation = ringDashActive || fireActive || flameSettling;

        if (showAnimation) {
            this._startAnimation();
        } else {
            this._stopAnimation();
        }

        const modeOptions = selBistateMode?.attributes?.options || [];
        const modeState = (selBistateMode?.state || '').trim();
        const translateBistateMode = (value) => {
            const key = `component.quiet_solar.entity.select.radiator_mode.state.${value}`;
            const translated = this._hass?.localize?.(key);
            return (translated && translated !== key) ? translated : value;
        };
        const modeOptionsHtml = modeOptions.map((o) =>
            `<option value="${this._escapeHtml(o)}" ${o === modeState ? 'selected' : ''}>${this._escapeHtml(translateBistateMode(o))}</option>`,
        ).join('');

        const parseOverrideCommand = (overrideStateStr) => {
            if (!overrideStateStr || overrideStateStr === 'NO OVERRIDE') return null;
            const match = String(overrideStateStr).match(/Override:\s*(.+)/i);
            return match ? match[1].trim() : null;
        };
        const overrideCommand = parseOverrideCommand(overrideState);
        const overrideCommandLower = overrideCommand ? overrideCommand.toLowerCase() : '';
        const isOverrideOff = overrideCommand && overrideCommandLower.endsWith('off');

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

        const canDragHandle = isEnabled && isDefaultMode && !isOverridden && displayTargetHours > 0;

        const finishTimeStr = sDefaultOnFinishTime?.state || '07:00:00';
        const finishTimeMins = this._localFinishTimeMins != null
            ? this._localFinishTimeMins
            : this._parseTimeToMinutes(finishTimeStr);

        // QS-217 review-fix #03 — clipPath is just the outer disc.
        const clipPathD =
            `M ${CENTER_CY - CLIP_R},${CENTER_CY}` +
            ` a ${CLIP_R},${CLIP_R} 0 1,0 ${2 * CLIP_R},0` +
            ` a ${CLIP_R},${CLIP_R} 0 1,0 ${-2 * CLIP_R},0`;

        // Build the ring SVG markup via the sub-base helper.
        const ringMarkup = this._buildRingHTML({
            palette: colors, ringCirc, center,
            startDeg, endDeg, rangeDeg, gapDeg,
            progressPath, bgPath,
            handlePos, handlePct, displayTargetHours, hoursRun,
            showAnimation: ringDashActive, canDragHandle,
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
                <clipPath id="${flameClipId}">
                  <path clip-rule="evenodd" d="${clipPathD}" />
                </clipPath>
              </defs>
              <g clip-path="url(#${flameClipId})">
                ${initialFlamePaths.map((d, i) =>
                    `<path id="flame${i}" d="${d}" fill="${initialFlameColors[i]}" opacity="1" pointer-events="none" style="will-change: transform;" />`,
                ).join("\n                ")}
              </g>
              ${e.override_reset ? `<circle id="override_btn_cover" cx="${CENTER_CY}" cy="${OVERRIDE_BTN_CARVE_CY}" r="${OVERRIDE_BTN_CARVE_R}" fill="var(--card-background-color)" pointer-events="none" />` : ''}
              ${ringMarkup}
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
                    <div class="from-to-value">${isDefaultMode && !isOverridden ? '--:--' : this._formatTime(startTime)}</div>
                  </div>
                  <div class="from-to-item">
                    <div class="from-to-label">To:</div>
                    <div class="from-to-value">${isDefaultMode && !isOverridden ? (sDefaultOnFinishTime ? this._formatTime(finishTimeStr) : '--:--') : this._formatTime(endTime)}</div>
                  </div>
                </div>
                ` : ''}
                ${isDefaultMode && !isOverridden && sDefaultOnFinishTime ? `
                <div class="center-controls" style="flex-direction:column; gap:4px;">
                  <div style="color: var(--secondary-text-color); font-weight:700; font-size: .75rem;">Change Finish Time</div>
                  <div id="time_btn" class="time-btn ${finishTimeStr && finishTimeStr !== '--:--' ? 'on' : ''}" role="button" tabindex="0" aria-label="Change finish time">${this._formatTime(finishTimeStr)}</div>
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

        if (selBistateMode) {
            this._wireBistateMode({
                selectEl: ids('bistate_mode'),
                entityId: e.bistate_mode,
                translationNamespace: 'radiator_mode',
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

        if (e.override_reset) {
            this._wireOverrideButton({
                buttonEl: ids('override_btn'),
                entityId: e.override_reset,
                overrideBtnClickable,
            });
        }

        if (isDefaultMode && !isOverridden && sDefaultOnFinishTime) {
            this._wireTimePicker({
                buttonEl: ids('time_btn'),
                entityId: e.default_on_finish_time,
                currentMins: finishTimeMins,
                localStateKey: '_localFinishTimeMins',
                clearTimerKey: '_localFinishTimeClearTimer',
            });
        }

        if (e.reset) {
            this._wireResetButton({
                buttonEl: ids('reset'),
                entityId: e.reset,
            });
        }

        if (canDragHandle) {
            // S8: snap points derived from the configured max (default 12).
            const allowedHours = this._allowedHalfHours(maxHours);

            this._wireTargetHandle({
                ringSvg: this._root.querySelector('.ring svg'),
                handle: this._root.getElementById('target_handle'),
                center, ringCirc,
                startDeg, endDeg, rangeDeg,
                hoursToPct, pctToHours, allowedHours,
                entityId: e.default_on_duration,
                getHoursRun: () => hoursRun,
                colors,
            });
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
