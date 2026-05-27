/*
  QS-199 shared CSS template — baseCardCSS(palette, options)
  Returns the common CSS string used by every qs-*-card.js. Cards append
  their own card-specific extras (e.g. car's .sun-btn / .rabbit-btn,
  water-boiler's .temp-row, climate's nothing-card-specific).

  Colour literal policy (plan-critic CL3 / review-fix #02 S10):
    Allowed in this shared template:
      - neutral rgba(0,0,0,*) / rgba(255,255,255,*)
      - HA theme variables (var(--*-color))
      - semantic-anchor colours that are identical across all 6 cards
        today and are NOT card-branded:
          * power-blue    #2196F3 (.power-btn.on)
          * solar-green   #4CAF50 (.green-btn.on icon)
          * override-orange #FF9800 (.override-btn.active)
          * pill-on green #2ecc71 / #2ecc71aa (.pill.on .dot)
    NOT allowed:
      - branded palette literals — flow through `palette` argument
        ({primary, gradStart, gradEnd, animStart, animEnd}). The
        `palette || {…}` fallback uses ONLY var(--primary-color) so the
        shared module contains zero branded hexes
        (pinned by test_shared_css_has_no_branded_colour_literals).
*/

// QS-199 review-fix #02 S10 — neutral, AC6-compliant palette defaults
// (theme variables only — zero branded colour literals). Pinned by
// test_shared_css_has_no_branded_colour_literals.
const PALETTE_DEFAULTS = {
    primary: 'var(--primary-color)',
    gradStart: 'var(--primary-color)',
    gradEnd: 'var(--primary-color)',
    animStart: 'var(--primary-color)',
    animEnd: 'var(--primary-color)',
};

export function baseCardCSS(palette, _options) {
    // QS-199 review-fix #04 CR3 — merge defaults PER KEY, not only when
    // `palette` is entirely falsy. The old `palette || {…}` emitted
    // `undefined` into the generated CSS (breaking gradients / color-mix)
    // for a partial palette like `{ primary: '#FF5722' }`. Spreading the
    // defaults first guarantees every key resolves to a real value.
    const colors = { ...PALETTE_DEFAULTS, ...(palette || {}) };

    return `
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
}
