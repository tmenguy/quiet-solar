/*
  QS-199 — Ring + duration sub-base.

  Used by every "duration-style" card (on-off-duration, radiator, pool,
  water-boiler, climate). NOT used by car (different ring geometry +
  3 inside-disc buttons instead of one override-btn).

  Adds:
    - _buildRingHTML({...}) — produces the outer .ring SVG markup
      (linear-gradient defs, background-arc + progress-arc, override-btn
      cover circle, target-handle circle + text, power-btn + green-btn +
      override-btn DOM elements). Cards layer their backdrop-specific
      SVG inside the returned template.
    - _wireOverrideButton({buttonEl, entityId, overrideBtnClickable}) —
      confirmation dialog + state-aware class assignment.
    - _wireBistateMode({selectEl, entityId, translationNamespace}) —
      focus/blur/change handlers; M2 try/finally + 300ms setTimeout
      cleanup.
*/

import { QsCardBase, arcPath, polar } from './qs-card-base.js';

export class QsRingDurationCardBase extends QsCardBase {
    /*
      _buildRingHTML — returns an HTML string for the outer .ring SVG
      markup. Caller injects this string into innerHTML between any
      card-specific backdrop SVG and the .center / override-btn /
      green-btn DOM nodes.

      params (long but mechanical — every duration card needs them):
        palette                      — {primary, gradStart, gradEnd, animStart, animEnd}
        ringCirc                     — ring radius in SVG units (typically 130)
        center                       — {cx, cy} centre of the ring
        startDeg, endDeg, rangeDeg, gapDeg
        progressPath, bgPath         — pre-computed SVG `d` attribute strings
        handlePos                    — {x, y} handle centre
        handlePct                    — handle percentage (for label)
        displayTargetHours           — value to display at the handle text
        hoursRun                     — current run hours
        showAnimation                — boolean — whether to render running_anim path
        canDragHandle                — boolean — whether to render handle circle
        gradGreenId, gradRunningId, activeGradId
        dashLen, gapLen              — stroke-dasharray
        title                        — escaped card title
        running                      — boolean — for active-state colouring
        pctToHours                   — function used in label render
      Returns string.
    */
    _buildRingHTML(params) {
        const {
            palette, ringCirc, center,
            progressPath, bgPath,
            handlePos, handlePct, displayTargetHours, hoursRun,
            showAnimation, canDragHandle,
            gradGreenId, gradRunningId, activeGradId,
            dashLen, gapLen,
            pctToHours,
        } = params;

        const escapedHandleLabel = this._fmt(pctToHours(handlePct));

        // QS-199 review-fix N6 — per-instance glow filter id (parity with
        // gradGreenId / gradRunningId) so two cards on one page can't
        // collide if shadow-DOM id scoping ever changes.
        const glowId = this._instanceId('runningGlow');

        return `
              <defs>
                <linearGradient id="${gradGreenId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="${palette.gradStart}"/>
                  <stop offset="100%" stop-color="${palette.gradEnd}"/>
                </linearGradient>
                <linearGradient id="${gradRunningId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="${palette.animStart}"/>
                  <stop offset="100%" stop-color="${palette.animEnd}"/>
                </linearGradient>
                <filter id="${glowId}" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
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
                    filter="url(#${glowId})"
                    style="mix-blend-mode:screen; will-change: stroke-dashoffset"
              />
              ` : ''}
              ${canDragHandle ? `
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="${palette.primary}" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="${palette.primary}" font-size="13" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${escapedHandleLabel}</text>
              ` : ''}
        `;
    }

    /*
      _wireOverrideButton — wires the override-btn DOM node:
        - confirmation dialog → _press(entityId)
        - state-aware class assignment (active / resetting / disabled)
      params: {buttonEl, entityId, overrideBtnClickable}
    */
    _wireOverrideButton(params) {
        const { buttonEl, entityId, overrideBtnClickable } = params;
        if (!buttonEl || !entityId || !overrideBtnClickable) return;

        buttonEl.style.pointerEvents = 'auto';
        const obtnAction = async () => {
            this._showDialog({
                title: 'Reset override',
                message: 'This will reset the manual override and return to automatic mode.\nProceed?',
                buttons: [
                    { text: 'Cancel', variant: 'secondary' },
                    {
                        text: 'Reset', variant: 'primary',
                        onClick: async () => { await this._press(entityId); },
                    },
                ],
            });
        };
        buttonEl.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); obtnAction(); });
        buttonEl.addEventListener('touchend', (ev) => { ev.preventDefault(); obtnAction(); });
        this._registerKeyActivation(buttonEl, obtnAction);
    }

    /*
      _wireBistateMode — focus/blur/change handlers for the bistate-mode
      <select>. M2: try/finally + 300ms setTimeout cleanup so a rejected
      _select doesn't wedge `_isProcessingModeChange` forever.

      params: {
          selectEl,                — the <select> DOM element
          entityId,                — service target for _select
          translationNamespace,    — e.g. 'on_off_mode' or 'climate_mode'
                                     (currently unused for routing; reserved
                                     for future translation work)
      }
    */
    _wireBistateMode(params) {
        const { selectEl, entityId } = params;
        if (!selectEl || !entityId) return;

        const startM = () => { this._isInteractingMode = true; };
        const endM = () => {
            if (!this._isProcessingModeChange) {
                this._isInteractingMode = false;
                this._render();
            }
        };
        selectEl.addEventListener('focus', startM);
        selectEl.addEventListener('blur', endM);
        selectEl.addEventListener('change', async (ev) => {
            const option = ev.target.value;
            if (!option) return;

            this._isProcessingModeChange = true;
            // M2: try/finally so the cleanup setTimeout ALWAYS runs.
            try {
                await this._select(entityId, option);
            } catch (_) {
                // swallow — HA state will resync on the next push
            } finally {
                setTimeout(() => {
                    this._isProcessingModeChange = false;
                    this._isInteractingMode = false;
                    this._render();
                }, 300);
            }
        });
        const modePill = selectEl.closest('.pill');
        if (modePill) {
            modePill.addEventListener('click', (ev) => {
                if (ev.target.tagName === 'SELECT') return;
                try { selectEl.showPicker(); } catch (_) { selectEl.focus(); }
            });
        }
    }
}

// Re-export for convenience (cards can import everything from the duration sub-base).
export { QsCardBase, arcPath, polar };
