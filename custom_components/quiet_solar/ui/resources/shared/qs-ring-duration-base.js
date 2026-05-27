/*
  QS-199 — Ring + duration sub-base.

  Used by every "duration-style" card (on-off-duration, radiator, pool,
  water-boiler, climate). NOT used by car (different ring geometry +
  3 inside-disc buttons instead of one override-btn).

  Adds:
    - _wireOverrideButton({buttonEl, entityId, overrideBtnClickable}) —
      confirmation dialog + state-aware class assignment.
    - _wireBistateMode({selectEl, entityId, translationNamespace}) —
      focus/blur/change handlers; M2 try/finally + 300ms setTimeout
      cleanup.

  QS-235 — `_buildRingHTML` was moved UP to `QsCardBase` (single source
  of truth) so the car card, which extends `QsCardBase` directly, can
  consume it too. The duration cards inherit it unchanged through this
  sub-base. The hours-/override-specific helpers (`_clampMaxHours`,
  `_allowedHalfHours`, `_wireOverrideButton`, `_wireBistateMode`) stay
  here — the car needs none of them.
*/

import { QsCardBase, arcPath, polar } from './qs-card-base.js';

// QS-199 review-fix #05 N1 — single source for the drag-range constants
// shared by `_clampMaxHours` + `_allowedHalfHours`, so the gauge max and
// the snap-list max stay equal by construction and the `12` default can't
// drift between the two helpers.
const MAX_HOURS_DEFAULT = 12;   // fallback for missing / non-finite / non-positive config
const SNAP_STEP_HOURS = 0.5;    // draggable snap granularity (gauge max grid-aligns to this)
const MAX_HOURS_CEILING = 168;  // one week — bounds the gauge scale AND the snap-list loop

export class QsRingDurationCardBase extends QsCardBase {
    /*
      _clampMaxHours(raw) — the SINGLE source-of-truth normalizer for a
      card's draggable-range maximum. QS-199 review-fix #04 ES1 + #05 S1
      (root-cause): every duration card derives `maxHours` through this
      helper on EVERY branch (default `max_default_hours` AND non-default
      runtime `targetHours`), so the SAME value feeds BOTH the gauge math
      (`hoursToPct` / arcs / handle) AND the snap list
      (`_allowedHalfHours`). They are then equal BY CONSTRUCTION.

      The historical edge cases this closes (one class, four rounds):
        - S8/M1: a non-finite config (`Number("1e999") === Infinity`)
          hung the snap loop → `Number.isFinite` guard.
        - ES1: clamping in only ONE place (gauge vs snap) desynced them
          for `>168` → bound BOTH via this single source.
        - #05 S1: a fractional cap (13.3) left the snap list ending at
          13.0 while the gauge drew 13.3 → a ~0.5h top-of-ring dead zone.
          Grid-aligning to the 0.5 snap step makes the gauge's 100% land
          EXACTLY on a draggable value.

      Normalization (in order): reject non-finite / non-positive → default
      (`MAX_HOURS_DEFAULT`); grid-align to the `SNAP_STEP_HOURS` (0.5) grid
      via `Math.round`; floor at one step (so the list is never just
      `[0]`); ceiling at `MAX_HOURS_CEILING` (168 = one week — bounds the
      gauge scale AND the snap-list loop on every branch). 168 is itself a
      0.5-multiple, so the final clamp stays grid-aligned.
    */
    _clampMaxHours(raw) {
        const n = Number(raw);
        if (!Number.isFinite(n) || n <= 0) return MAX_HOURS_DEFAULT;
        const aligned = Math.round(n / SNAP_STEP_HOURS) * SNAP_STEP_HOURS;
        return Math.min(Math.max(aligned, SNAP_STEP_HOURS), MAX_HOURS_CEILING);
    }

    /*
      _allowedHalfHours(maxHours) — the drag-ring snap points: 0 ..
      maxHours in `SNAP_STEP_HOURS` (0.5) steps. Callers pass the
      `_clampMaxHours` output (grid-aligned + bounded), so the last
      element equals `maxHours` exactly → snap-max == gauge-max.

      QS-199 review-fix #04 ES1 — no own `Math.min` clamp (that desynced
      it from the gauge); keeps only the `Number.isFinite` Infinity
      defense (+ the shared `MAX_HOURS_DEFAULT` fallback, #05 N1) so a
      future un-clamped caller still can't hang the loop.
    */
    _allowedHalfHours(maxHours) {
        const n = Number(maxHours);
        const cap = Number.isFinite(n) && n > 0 ? n : MAX_HOURS_DEFAULT;
        const out = [];
        for (let i = 0; i <= cap; i += SNAP_STEP_HOURS) out.push(i);
        return out;
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
