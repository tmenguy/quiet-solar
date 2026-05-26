/*
  QS On/Off Duration Card - custom:qs-on-off-duration-card

  Zero-build single-file Lit-style web component compatible with Home
  Assistant. Extends `QsRingDurationCardBase` (which itself extends
  `QsCardBase`) — see `shared/qs-card-base.js` and
  `shared/qs-ring-duration-base.js` for the common lifecycle, service
  callers, defensive utilities, wire-helpers, and ring HTML builder.

  QS-199 refactor (extract shared base): per-card responsibilities
  reduced to:
    - per-state computations (running, isDefaultMode, override parsing)
    - the palette + card-specific HTML template (no backdrop animation)
    - calling shared wire-helpers with the right config
*/

import { baseCardCSS } from './shared/qs-card-styles.js';
import {
    QsRingDurationCardBase,
} from './shared/qs-ring-duration-base.js';
import { arcPath, polar, pctToDeg } from './shared/qs-card-base.js';

class QsOnOffDurationCard extends QsRingDurationCardBase {
    static getStubConfig() {
        return { name: "QS On/Off Duration", entities: {} };
    }

    getCardSize() { return 5; }

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

        // S8: NaN-safe state coercion.
        const targetHours = this._safeNumber(sDurationLimit, 12);
        const hoursRun = this._safeNumber(sCurrentDuration, 0);
        const defaultDuration = this._safeNumber(sDefaultOnDuration, 6);

        const bistateMode = selBistateMode?.state || 'bistate_mode_default';
        const isDefaultMode = bistateMode === 'bistate_mode_default';

        const overrideState = sOverrideState?.state || 'NO OVERRIDE';
        const isOverridden = overrideState !== 'NO OVERRIDE';
        const isResettingOverride = overrideState === 'ASKED FOR RESET OVERRIDE';

        const commandState = sCommand?.state || '';
        const running = commandState.toLowerCase() === 'on';

        // Determine max hours and display target
        let maxHours;
        let displayTargetHours;
        if (isDefaultMode) {
            // N3: configurable upper bound for the default-mode ring.
            maxHours = this._clampMaxHours(cfg.max_default_hours);
            displayTargetHours = defaultDuration;
        } else {
            // BH: zero-clamp against div-by-zero in hoursToPct.
            maxHours = targetHours > 0 ? targetHours : this._clampMaxHours(cfg.max_default_hours);
            displayTargetHours = targetHours;
        }

        const showFromTo = !isDefaultMode || isOverridden;

        const startTime = (sStartTime && this._isValidState(sStartTime.state)) ? sStartTime.state : '--:--';
        const endTime = (sEndTime && this._isValidState(sEndTime.state)) ? sEndTime.state : '--:--';

        // Card palette (passes through to baseCardCSS for theming).
        const colors = {
            primary: '#2196F3',
            gradStart: '#00bcd4',
            gradEnd: '#8bc34a',
            animStart: '#00e1ff',
            animEnd: '#0066ff',
        };

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
        const showAnimation = (running && segLen > 6);

        // M4: gate the RAF loop on showAnimation.
        if (showAnimation) {
            this._startAnimation();
        } else {
            this._stopAnimation();
        }

        // Bistate mode <option> list
        const modeOptions = selBistateMode?.attributes?.options || [];
        const modeState = (selBistateMode?.state || '').trim();
        const translateBistateMode = (value) => {
            const key = `component.quiet_solar.entity.select.on_off_mode.state.${value}`;
            const translated = this._hass?.localize?.(key);
            return (translated && translated !== key) ? translated : value;
        };
        const modeOptionsHtml = modeOptions.map((o) =>
            `<option value="${this._escapeHtml(o)}" ${o === modeState ? 'selected' : ''}>${this._escapeHtml(translateBistateMode(o))}</option>`,
        ).join('');

        // Override state parsing
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

        // Drag handle is gated on default mode + enabled + not overridden + positive target
        const canDragHandle = isEnabled && isDefaultMode && !isOverridden && displayTargetHours > 0;

        const finishTimeStr = sDefaultOnFinishTime?.state || '07:00:00';
        const finishTimeMins = this._localFinishTimeMins != null
            ? this._localFinishTimeMins
            : this._parseTimeToMinutes(finishTimeStr);

        const ringMarkup = this._buildRingHTML({
            palette: colors, ringCirc, center,
            startDeg, endDeg, rangeDeg, gapDeg,
            progressPath, bgPath,
            handlePos, handlePct, displayTargetHours, hoursRun,
            showAnimation, canDragHandle,
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
            ${swEnableDevice ? `<div id="power_btn" class="power-btn ${isEnabled ? 'on' : ''}"><ha-icon icon="mdi:power"></ha-icon></div>` : ''}
            <svg viewBox="0 0 320 320" width="300" height="300" style="touch-action: none;" aria-hidden="true">
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
                  <div id="time_btn" class="time-btn ${finishTimeStr && finishTimeStr !== '--:--' ? 'on' : ''}">${this._formatTime(finishTimeStr)}</div>
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

        // Wire events
        const root = this._root;
        const ids = (k) => root.getElementById(k);

        // Bistate mode selector
        if (selBistateMode) {
            this._wireBistateMode({
                selectEl: ids('bistate_mode'),
                entityId: e.bistate_mode,
                translationNamespace: 'on_off_mode',
            });
        }

        // Green-only toggle
        if (swGreenOnly) {
            this._wireGreenButton({
                buttonEl: ids('green_btn'),
                swEntity: swGreenOnly,
                entityId: e.green_only,
            });
        }

        // Power/Enable toggle
        if (swEnableDevice) {
            this._wirePowerButton({
                buttonEl: ids('power_btn'),
                swEntity: swEnableDevice,
                entityId: e.enable_device,
            });
        }

        // Override button
        if (e.override_reset) {
            this._wireOverrideButton({
                buttonEl: ids('override_btn'),
                entityId: e.override_reset,
                overrideBtnClickable,
            });
        }

        // Time picker (finish time)
        if (isDefaultMode && !isOverridden && sDefaultOnFinishTime) {
            this._wireTimePicker({
                buttonEl: ids('time_btn'),
                entityId: e.default_on_finish_time,
                currentMins: finishTimeMins,
                localStateKey: '_localFinishTimeMins',
                clearTimerKey: '_localFinishTimeClearTimer',
            });
        }

        // Reset button
        if (e.reset) {
            this._wireResetButton({
                buttonEl: ids('reset'),
                entityId: e.reset,
            });
        }

        // Drag target handle
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

if (!customElements.get('qs-on-off-duration-card')) {
    customElements.define('qs-on-off-duration-card', QsOnOffDurationCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-on-off-duration-card')) {
    window.customCards.push({
        type: 'qs-on-off-duration-card',
        name: 'QS On/Off Duration Card',
        description: 'Quiet Solar on/off duration control card',
    });
}
