/*
  QS-199 — Shared wave path helper.

  Per concrete-planner finding R4: NO class. Just constants + pure helper.
  The boilerplate around the helper (per-card RAF step body that owns
  _currentAmplitude / _currentSpeed / _wavePhase) is ~30 LOC and diverges
  per card (pool palette, climate-cool palette, water-boiler bubbles/steam,
  car single-layer); the helper itself is ~15 LOC of pure path generation.
  An engine class would add comparable LOC of feature-flagging — keeping
  it as constants + helper is the right granularity.

  Exports:
    WAVE_CONSTANTS  — tuning constants
    generateWavePath — pure function (width, amplitude, frequency, phase, yOffset) → string
*/

export const WAVE_CONSTANTS = {
    WAVE_WIDTH: 480,
    WAVE_BOTTOM_Y: 400,
    LAYER_SCROLL_OFFSET: 1.2,
    LAYER_PHASE_OFFSET: 2.1,
    LERP_RATE: 2,
    LERP_DT_CEIL: 0.1,
    AMP_REGEN_THRESHOLD: 0.25,
    LEVEL_REGEN_THRESHOLD: 0.01,
    PHASE_WRAP: 1e6,
    PHASE_TO_PX: 60,
};

/*
  generateWavePath — SVG path for a single sine-wave layer.

  width       — total layer width in SVG px (typically WAVE_WIDTH)
  amplitude   — wave height in px
  frequency   — number of full periods across `width`
  phase       — radians offset (drives the scroll)
  yOffset     — y-coord of the wave baseline (lower y = higher up canvas)

  Returns a `d` attribute string that closes back to the bottom of the
  canvas — the consumer typically applies a `<clipPath>` circle to keep
  the wave shape inside the ring.
*/
export function generateWavePath(width, amplitude, frequency, phase, yOffset) {
    const STEP = 8; // px between sample points along x
    const samples = Math.ceil(width / STEP) + 1;
    const k = (2 * Math.PI * frequency) / width;
    let d = `M 0 ${(yOffset + amplitude * Math.sin(phase)).toFixed(2)}`;
    for (let i = 1; i < samples; i++) {
        const x = i * STEP;
        const y = yOffset + amplitude * Math.sin(k * x + phase);
        d += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    d += ` L ${width.toFixed(2)} ${WAVE_CONSTANTS.WAVE_BOTTOM_Y} L 0 ${WAVE_CONSTANTS.WAVE_BOTTOM_Y} Z`;
    return d;
}
