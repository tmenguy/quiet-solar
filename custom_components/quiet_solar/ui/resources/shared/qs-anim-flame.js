/*
  QS-199 — Shared flame engine (used by radiator + climate heat mode).

  Encapsulates flame animation state (amp lerp, per-tooth tip phases,
  regen throttle) and path generation. The card owns DOM refs and the
  outer RAF loop; this engine is a pure state-machine + path generator.

  Exports:
    FLAME_CONSTANTS  — tuning constants (kept here as the canonical
                       source; cards may pass overrides via constructor)
    QsFlameEngine    — class with step(dt, fireOn) + generatePaths(baseY)

  Per-card palettes (FLAME_FILLS / FLAME_GREY_FILLS) and per-layer
  configuration arrays (LAYER_TEETH_COUNTS, etc.) remain in each card
  file as the source of truth — passed to the engine as constructor args.
  This module contains NO branded colour literals.
*/

export const FLAME_CONSTANTS = {
    FLAME_WIDTH: 480,
    FLAME_BOTTOM_Y: 400,
    STILL_AMP: 0,
    DANCE_AMP: 8,
    STATIC_PEAK_HEIGHT: 30,
    LERP_RATE: 2,
    LERP_DT_CEIL: 0.1,
    AMP_REGEN_THRESHOLD: 0.25,
    LEVEL_REGEN_THRESHOLD: 0.01,
    PHASE_REGEN_MIN_DT: 0.20,
    FLAME_BASE_MIN_PCT: 0.2,
    FLAME_BASE_MAX_PCT: 0.8,
};

/*
  QsFlameEngine — state machine + path generator for a 3-layer (or N-layer)
  flame silhouette. Each layer's tip flickers independently of the others
  (read as turbulence).

  Constructor params (concrete-planner R5):
    layerTeethCounts    — array of teeth counts per layer (e.g. [3, 4, 5])
    layerTipFlickerHz   — array of flicker frequencies (Hz) per layer
    layerBaseHeights    — array of base heights per layer
    layerTipAmpMults    — array of tip-amplitude multipliers per layer
    runningFills        — array of fill colors when running
    greyFills           — array of fill colors when idle/cold
*/
export class QsFlameEngine {
    constructor(params) {
        const {
            layerTeethCounts,
            layerTipFlickerHz,
            layerBaseHeights,
            layerTipAmpMults,
            runningFills,
            greyFills,
        } = params;

        this.layerTeethCounts = layerTeethCounts;
        this.layerTipFlickerHz = layerTipFlickerHz;
        this.layerBaseHeights = layerBaseHeights;
        this.layerTipAmpMults = layerTipAmpMults;
        this.runningFills = runningFills;
        this.greyFills = greyFills;

        // Length-equality guard (QS-204 review-fix #02 G7 + #03 H7):
        console.assert(
            layerTeethCounts.length === layerTipFlickerHz.length &&
            layerTeethCounts.length === layerBaseHeights.length &&
            layerTeethCounts.length === layerTipAmpMults.length,
            'QsFlameEngine: layerTeethCounts / layerTipFlickerHz / layerBaseHeights / layerTipAmpMults must be the same length',
        );

        this._currentFlameAmp = FLAME_CONSTANTS.STILL_AMP;
        this._tipPhases = layerTeethCounts.map((count) => new Array(count).fill(0));
        this._lastFlameAmp = null;
        this._lastFlameBaseY = null;
        this._lastRegenTs = -Infinity;
        this._needsFlamePrime = true;
    }

    /*
      step(dt, fireOn) — advance the lerp + tip phases.
      Returns {shouldRegen} — whether the path should be regenerated.

      ts is optional; if provided used for the time-throttle. Default to
      performance.now() / 1000 ≈ epoch-ish for the throttle.
    */
    step(dt, fireOn, baseY, ts) {
        const C = FLAME_CONSTANTS;
        const tsSec = ts != null ? ts : performance.now();

        const targetAmp = fireOn ? C.DANCE_AMP : C.STILL_AMP;
        const lerpDt = Math.min(dt, C.LERP_DT_CEIL);
        const lerpFactor = 1 - Math.exp(-C.LERP_RATE * lerpDt);
        this._currentFlameAmp += (targetAmp - this._currentFlameAmp) * lerpFactor;

        // QS-204 review-fix #01 F5 / #02 G6 — symmetric snap-to-target.
        if (!fireOn && Math.abs(this._currentFlameAmp) < 0.05) {
            this._currentFlameAmp = 0;
        }
        if (fireOn && Math.abs(this._currentFlameAmp - C.DANCE_AMP) < 0.05) {
            this._currentFlameAmp = C.DANCE_AMP;
        }

        // QS-204 review-fix #03 H4 — use clamped lerpDt, not raw dt.
        if (fireOn && this._tipPhases) {
            for (let i = 0; i < this.layerTeethCounts.length; i++) {
                const phasesForLayer = this._tipPhases[i];
                const phaseStep = 2 * Math.PI * this.layerTipFlickerHz[i] * lerpDt;
                for (let j = 0; j < phasesForLayer.length; j++) {
                    phasesForLayer[j] = (phasesForLayer[j] + phaseStep * (1 + 0.07 * j)) % (2 * Math.PI);
                }
            }
        }

        const hasValidBase = baseY != null && !Number.isNaN(baseY);
        const ampDelta = this._lastFlameAmp == null
            ? Infinity
            : Math.abs(this._currentFlameAmp - this._lastFlameAmp);
        const levelChanged = hasValidBase &&
            Math.abs(baseY - (this._lastFlameBaseY ?? Number.NEGATIVE_INFINITY)) > C.LEVEL_REGEN_THRESHOLD;
        const sinceLastRegen = (tsSec - (this._lastRegenTs ?? -Infinity)) / 1000;
        const phaseChanged = fireOn && sinceLastRegen >= C.PHASE_REGEN_MIN_DT;
        const shouldRegen = hasValidBase && (phaseChanged || levelChanged || ampDelta > C.AMP_REGEN_THRESHOLD);

        if (shouldRegen) {
            this._lastFlameBaseY = baseY;
            this._lastFlameAmp = this._currentFlameAmp;
            this._lastRegenTs = tsSec;
        }
        return { shouldRegen };
    }

    /*
      generatePaths(baseY, isIdle) → string[]
      Returns one SVG `d` path per layer.
    */
    generatePaths(baseY, isIdle) {
        const C = FLAME_CONSTANTS;
        return this.layerTeethCounts.map((count, i) => this._generateFlameTeethPath(
            C.FLAME_WIDTH,
            baseY,
            this.layerBaseHeights[i],
            this._currentFlameAmp * this.layerTipAmpMults[i],
            count,
            this._tipPhases ? this._tipPhases[i] : null,
            isIdle,
        ));
    }

    /*
      _generateFlameTeethPath — SVG path for a single peaked-teeth layer.
      Each tooth is a piecewise-quadratic ascend → peak → descend.
      Mirrors qs-radiator-card.js's pre-extraction implementation
      (verbatim) so the rendered silhouette is byte-identical.

      Uses local `STATIC_PEAK_HEIGHT` / `FLAME_BOTTOM_Y` aliases so the
      gate predicates (`isIdle ? STATIC_PEAK_HEIGHT : 0`,
      `isIdle ? 0 : tipAmp * Math.sin(phase)`) appear verbatim — pinned
      by `tests/test_radiator_card_smoke.py` via `card_source_union`.
    */
    _generateFlameTeethPath(width, baseY, peakHeight, tipAmp, numTeeth, tipPhases, isIdle) {
        const { STATIC_PEAK_HEIGHT, FLAME_BOTTOM_Y } = FLAME_CONSTANTS;
        const teethCount = Math.max(1, numTeeth | 0);
        const toothWidth = width / teethCount;
        const idlePeakBoost = isIdle ? STATIC_PEAK_HEIGHT : 0;
        let d = `M 0 ${baseY.toFixed(2)}`;
        for (let i = 0; i < teethCount; i++) {
            const phase = tipPhases && tipPhases[i] != null ? tipPhases[i] : 0;
            // Per-tooth wobble gated on `!isIdle` (QS-204 review-fix #03 H1).
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

    /*
      invalidate() — clear the memo keys after the parent card's innerHTML
      rewrite. Does NOT touch _currentFlameAmp or _tipPhases (those survive
      so re-attach resumes the dance without a visible jump).
    */
    invalidate() {
        this._lastFlameBaseY = null;
        this._lastFlameAmp = null;
        this._lastRegenTs = -Infinity;
    }

    /*
      reset() — full reset for tests / explicit re-prime.
    */
    reset() {
        this._currentFlameAmp = FLAME_CONSTANTS.STILL_AMP;
        this._tipPhases = this.layerTeethCounts.map((count) => new Array(count).fill(0));
        this._lastFlameAmp = null;
        this._lastFlameBaseY = null;
        this._lastRegenTs = -Infinity;
        this._needsFlamePrime = true;
    }

    /*
      primeForCurrentState(running) — skip the 1.5s boot lerp so the
      card's first render shows the steady-state silhouette.
    */
    primeForCurrentState(running) {
        if (this._needsFlamePrime) {
            this._currentFlameAmp = running ? FLAME_CONSTANTS.DANCE_AMP : FLAME_CONSTANTS.STILL_AMP;
            this._needsFlamePrime = false;
        }
    }
}
