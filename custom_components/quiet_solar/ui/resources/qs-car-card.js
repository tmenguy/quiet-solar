/*
  QS Car Card - custom:qs-car-card
  Zero-build single-file Lit-style web component compatible with Home Assistant
*/

class QsCarCard extends HTMLElement {
  connectedCallback() {
    if (this._animRaf != null) return;
    const step = (ts) => {
      if (!this.isConnected) { this._animRaf = null; return; }
      if (this._lastAnimTs == null) this._lastAnimTs = ts;
      const dt = Math.max(0, (ts - this._lastAnimTs) / 1000);
      this._lastAnimTs = ts;
      const patternLen = Math.max(8, this._animPatternLen || 64);
      const speed = 80; // dash units per second
      this._animOffset = ((this._animOffset || 0) + speed * dt) % patternLen;
      const p = this._root?.getElementById('charge_anim');
      if (p) {
        p.setAttribute('stroke-dashoffset', String(-this._animOffset));
      }
      this._animRaf = requestAnimationFrame(step);
    };
    this._animRaf = requestAnimationFrame(step);
  }

  disconnectedCallback() {
    if (this._animRaf != null) cancelAnimationFrame(this._animRaf);
    this._animRaf = null;
    this._lastAnimTs = null;
  }
  static getStubConfig() {
    return { name: "QS Car", entities: {} };
  }

  setConfig(config) {
    if (!config || !config.entities) throw new Error("entities is required");
    this._config = config;
    this._root = this.attachShadow({ mode: "open" });
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._root) return;
    // Avoid re-rendering while user is interacting with selects or a modal is open
    if (this._isInteracting || this._isInteractingCharger || this._isInteractingPerson || this._modalOpen || this._isInteractingTarget) return;
    this._render();
  }

  getCardSize() { return 6; }

  _entity(id) { return id ? this._hass?.states?.[id] : undefined; }

  _call(domain, service, data) {
    return this._hass.callService(domain, service, data);
  }

  _press(entity_id) { return this._call('button', 'press', { entity_id }); }
  _turnOn(entity_id) { return this._call('switch', 'turn_on', { entity_id }); }
  _turnOff(entity_id) { return this._call('switch', 'turn_off', { entity_id }); }
  _select(entity_id, option) { return this._call('select', 'select_option', { entity_id, option }); }
  _setTime(entity_id, value) { return this._call('time', 'set_value', { entity_id, time: value }); }

  _percent(num) {
    const n = Number(num);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(100, n));
  }

  _render() {
      const cfg = this._config || {};
      const e = cfg.entities || {};

      const sSoc = this._entity(e.soc);
      const sPower = this._entity(e.power);
      const selCharger = this._entity(e.charger_select);
      const selPerson = this._entity(e.attached_person);

      const selLimit = this._entity(e.next_limit);
      const swPriority = this._entity(e.bump_priority);
      const tNext = this._entity(e.next_time);
      const sChargeType = this._entity(e.charge_type);
      const sChargeTime = this._entity(e.charge_time);
      const sRangeNow = this._entity(e.range_now);
      const sRangeTarget = this._entity(e.range_target);
      const sPersonForecast = this._entity(e.person_forecast);

      const title = (cfg.title || cfg.name) || (sSoc ? (sSoc.attributes.friendly_name || sSoc.entity_id) : "Car");
      let soc = this._percent(sSoc?.state);
      const power = sPower?.state || "0";
      const target = selLimit?.state || "";
      const charging = (Number(power) > 50);
      const carChargeTypeIcons = {
          "Unknown": "mdi:help-circle-outline",
          "Not Plugged": "mdi:power-plug-off",
          "Faulted": "mdi:emoticon-dead",
          "No Power To Car": "mdi:flash-off",
          "Not Charging": "mdi:battery-off",
          "Target Met": "mdi:battery-high",
          "As Fast As Possible": "mdi:rabbit",
          "Scheduled": "mdi:clock-outline",
          "Solar Priority": "mdi:solar-power",
          "Solar": "mdi:white-balance-sunny",
          "Person Automated": "mdi:auto-fix",
      };
      const iconForChargeType = (str) => carChargeTypeIcons[str];
      const chargeIcon = iconForChargeType(sChargeType?.state);
      const chargeTime = sChargeTime?.state || '';
      const chargeIconLabel = 'Mode';
      const chargeTimeLabel = 'Finish';

      const isNumberLike = (v) => v != null && v !== '' && !Number.isNaN(Number(v));
      const normState = (s) => String(s || '').toLowerCase();
      const validState = (s) => s != null && !['unavailable', 'unknown', 'none'].includes(normState(s));
      const rangeNowStr = (sRangeNow && validState(sRangeNow.state) && isNumberLike(sRangeNow.state)) ? `${Math.round(Number(sRangeNow.state))} km` : '';
      const rangeTargetStr = (sRangeTarget && validState(sRangeTarget.state) && isNumberLike(sRangeTarget.state)) ? `${Math.round(Number(sRangeTarget.state))} km` : '';

      const parseTargetPercent = (txt) => {
          if (!txt) return undefined;
          const m = String(txt).match(/(\d+)\s*%?/);
          return m ? Number(m[1]) : undefined;
      };

      const parseTargetEnergy = (txt) => {
          if (!txt) return undefined;
          const m = String(txt).match(/([\d.]+)\s*kWh/i);
          return m ? Number(m[1]) : undefined;
      };

      // Check if we're in energy mode
      const useEnergyMode = e.use_energy_mode === "on";

      // Font size for energy unit (kWh) relative to the number
      const energyUnitFontSize = 0.4; // 40% of the number size

      // Get target value based on mode
      let targetPct, displayTargetValue, maxCircleValue, displaySocValue;
      if (useEnergyMode) {
          const targetEnergy = parseTargetEnergy(target);
          // Get max energy from last option
          const limitOptions = selLimit?.attributes?.options || [];
          const lastOption = limitOptions.length > 0 ? limitOptions[limitOptions.length - 1] : "100kWh";
          maxCircleValue = parseTargetEnergy(lastOption) || 100;

          // Convert SOC from Wh to kWh and round
          const socKwh = Math.round(Number(sSoc?.state || 0) / 1000);

          // Calculate percentage for circle (0 to maxCircleValue)
          const socPct = maxCircleValue > 0 ? (socKwh / maxCircleValue) * 100 : 0;
          targetPct = targetEnergy != null && maxCircleValue > 0 ? (targetEnergy / maxCircleValue) * 100 : socPct;

          displaySocValue = `${socKwh}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>`;

          let toBeDisplayedvalue = Math.round(targetEnergy || socKwh)
          displayTargetValue = `${toBeDisplayedvalue}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>`;

          // Override soc for display
          soc = Math.max(0, Math.min(100, socPct));
      } else {
          targetPct = parseTargetPercent(target);
          maxCircleValue = 100;
          displayTargetValue = `${Math.round(targetPct ?? soc)}%`;
          displaySocValue = `${soc}%`;
      }

      const formatHm = (mins) => {
          if (mins == null) return '';
          const h = String(Math.floor(mins / 60)).padStart(2, '0');
          const m = String(mins % 60).padStart(2, '0');
          return `${h}:${m}`;
      };

      const parseTimeToMinutes = (txt) => {
          if (!txt) return 420; // 07:00
          const parts = String(txt).split(':').map(Number);
          const h = parts[0] || 0, m = parts[1] || 0;
          return h * 60 + m;
      };
      const nextTimeStr = tNext?.state || '07:00:00';
      const nextTimeMins = this._localNextTimeMins != null ? this._localNextTimeMins : parseTimeToMinutes(nextTimeStr);

      const css = `
      :host { --pad: 18px; display:block; }
      .card { padding: var(--pad); }
      .card-title { text-align:center; font-weight:800; font-size: 1.6rem; margin: 0px 0 0px; }
      .top { display:flex; gap:12px; flex-wrap:wrap; }
      .below { display:flex; align-items:center; justify-content:center; margin-top: 0px; width:300px; margin-left:auto; margin-right:auto; }
      .below .pill { width:100%; justify-content:center; }
      .forecast-row { text-align:center; width:300px; margin: 4px auto 0; color: var(--secondary-text-color); font-weight:600; font-size: .85rem; }
      .below-line { width:300px; margin: 8px auto 0; display:grid; grid-template-columns: 1fr auto; align-items:center; column-gap:12px; }
      .below-line.full { display:block; }
      .below-line.full > button { width: 100%; justify-content: center; position: relative; }
      .below-line.full > button.align-left { justify-content: flex-start; }
      .below-line.full > button .btn-center { position: absolute; left: 50%; transform: translateX(-50%); }
      .below-line .time-row { justify-self: end; margin-top: 0; }
      .btn-clock { display:flex; align-items:center; gap:8px; }
      .pill { display:flex; align-items:center; gap:8px; border-radius: 28px; height:40px; min-height:40px; padding:0 12px; border:1px solid var(--divider-color);
              background: var(--ha-card-background, var(--card-background-color)); box-sizing: border-box; }
      .pill .dot { width:12px; height:12px; border-radius:50%; background: var(--divider-color); box-shadow: 0 0 8px rgba(0,0,0,.25) inset; }
      .pill.on { background: rgba(56,142,60,0.15); border-color: rgba(56,142,60,.35); }
      .pill.on .dot { background: #2ecc71; box-shadow: 0 0 12px #2ecc71aa; }
      .pill { position: relative; }
      .pill select { appearance:none; background: transparent; color: var(--primary-text-color); border: none; font-weight:700; flex:1; width:auto; text-align:center; text-align-last:center; padding-left: 8px; }

      .hero { margin-top: 0px; display:flex; align-items:center; justify-content:center; }
      .hero .side { text-align:center; color: var(--secondary-text-color); font-weight:600; }
      .hero .side .value { display:block; font-size:1.2rem; color: var(--primary-text-color); }
      .ring { position: relative; width:300px; height:300px; margin: 0 auto; }
      .ring .center { position:absolute; inset:0; display:grid; place-items:center; text-align:center; pointer-events: none; transform: translateY(16px); }
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; }
      .ring ha-icon { --mdc-icon-size: 32px; color: var(--secondary-text-color); margin-bottom: 6px; }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; text-align:center; width: 180px; margin: 0 auto; }
      .ring .soc-block { display:flex; flex-direction:column; align-items:center; gap:2px; margin-top:4px; margin-bottom:8px; }
      .ring .soc-block .charge-type-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); margin-bottom: 2px; }
      .ring .stack > * { text-align:center; }
      .ring .mini-grid { display:grid; grid-template-columns: repeat(3, 60px); grid-auto-rows: auto; width:180px; margin: 0 auto; justify-items:center; align-items:center; row-gap:4px; column-gap:0; }
      .ring .mini-grid.extra { row-gap:0; margin-top:2px; margin-bottom:6px; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:0; }
      .ring .target-cell { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .mini-title { color: var(--secondary-text-color); font-weight:700; font-size: .7rem; letter-spacing:.2px; white-space: nowrap; }
      .ring .mini-value { color: var(--primary-text-color); font-weight:800; font-size: .95rem; line-height: 1.1; white-space: pre-line; }
      .ring .mini-icon { --mdc-icon-size: 18px; color: var(--primary-text-color); }
      .ring .mini-range { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .mini-range-now { color: var(--primary-text-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; transform: translateY(-8px); }
      .ring .mini-range-target { color: var(--primary-color); font-weight:700; font-size: .95rem; line-height: 1; margin-top:0; margin-bottom:0; }
      .disabled .ring .mini-range-target { color: var(--secondary-text-color); }
      .ring .mini-range:empty, .ring .mini-range-now:empty, .ring .mini-range-target:empty { display:none; }
      .ring .center-controls { display:flex; align-items:center; justify-content:center; margin-top: 6px; }
      .ring .sun-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; }
      .ring .sun-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; transform: translateY(3px); }
      .ring .sun-btn.on { border-color: rgba(56,142,60,.45); background: rgba(46,204,113,.14); box-shadow: 0 0 0 3px rgba(46,204,113,.20), 0 0 16px #2ecc71; }
      .ring .sun-btn.on ha-icon { color: #2ecc71; }
      .ring .rabbit-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; }
      .ring .rabbit-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; transform: translateY(3px); }
      .ring .rabbit-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .rabbit-btn.on ha-icon { color: #2196F3; }
      .ring .time-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; font-size: 0.99rem; font-weight: 800; color: var(--primary-color); line-height: 1; }
      .ring .time-btn:hover { border-color: var(--primary-color); background: rgba(255,255,255,.08); }
      .ring .time-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .time-btn.on { color: #2196F3; }

      .grid { display:grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap:16px; margin-top: 16px; }
      @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
      .section { border-radius: 20px; background: var(--card-background-color); padding:16px; border: 1px solid var(--divider-color); }
      .row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:8px 0; }
      select, input[type="time"] {
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
      /* Charger select full width */
      .below select { width: 100%; height: 40px; min-height: 40px; }
      /* Hour/minute compact width */
      .below-line .time-row select { width: auto; min-width: 64px; height: 40px; min-height: 40px; text-align: center; text-align-last: center; }
      /* Focus state for consistency */
      select:focus { border-color: var(--primary-color); box-shadow: 0 0 0 2px rgba(0,0,0,0), 0 0 0 3px color-mix(in srgb, var(--primary-color) 30%, transparent); }
      .actions { display:grid; grid-template-columns: repeat(2,minmax(180px,1fr)); gap:12px; margin-top:8px; }
      button { border:none; border-radius:18px; padding:14px 16px; font-weight:700; cursor:pointer; font-size: .95rem; }
      button.pill { height: 40px; min-height: 40px; display:flex; align-items:center; }
      .primary { background: var(--primary-color); color: var(--text-primary-color, #fff); }
      .secondary { background: rgba(255,255,255,.06); color: var(--primary-text-color); border:1px solid var(--divider-color); }
      .danger { background: var(--error-color); color: #fff; }
      /* Outline style variant */
      button.outline { background: transparent !important; border-width: 2px; }
      .danger.outline { color: var(--error-color) !important; border-color: var(--error-color) !important; }
      /* Disconnected visual mode (greyscale) */
      .disabled .ring .target-value { color: var(--secondary-text-color); }
      .disabled .primary { background: var(--divider-color); color: var(--primary-text-color); }
      .disabled .secondary { background: rgba(255,255,255,.04); border-color: var(--divider-color); color: var(--secondary-text-color); }
      /* Keep reset/danger buttons red even in disabled mode */
      .disabled .danger { background: var(--error-color); color: #fff; }
      .disabled .danger.outline { background: transparent !important; color: var(--error-color) !important; border-color: var(--error-color) !important; }
      .disabled .chip, .disabled .pill { border-color: var(--divider-color); }
      .disabled .sun-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .sun-btn ha-icon { color: var(--secondary-text-color); }
      .disabled .sun-btn.on { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .sun-btn.on ha-icon { color: var(--secondary-text-color); }
      .disabled .rabbit-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .rabbit-btn ha-icon { color: var(--secondary-text-color); }
      .disabled .rabbit-btn.on { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; }
      .disabled .rabbit-btn.on ha-icon { color: var(--secondary-text-color); }
      .disabled .time-btn { border-color: var(--divider-color); background: rgba(255,255,255,.03); box-shadow: none; cursor: not-allowed; color: var(--secondary-text-color); }
      .disabled .progress > div { background: var(--divider-color); }
      .disabled #force, .disabled #schedule_inline { pointer-events: none; cursor: not-allowed; }
      .live { display:grid; gap:10px; }
      .fault .card-title { color: var(--error-color); }
      .fault .time-btn { color: var(--secondary-text-color); }
      .progress { height:10px; border-radius:999px; background: var(--divider-color); overflow:hidden; }
      .progress > div { height:100%; background: var(--accent-color); width:${soc}% }
      .menu { text-align:right; }
      .quick { display:flex; gap:10px; flex-wrap:wrap; margin-top: 10px; }
      .chip { padding:10px 14px; border-radius: 999px; background: rgba(255,255,255,.06); border:1px solid var(--divider-color); font-weight:700; }
      .time-row { display:grid; grid-template-columns: auto auto auto; align-items:center; gap: 12px; margin-top: 6px; justify-content:center; }
      input[type="range"]{ width:100%; height:6px; border-radius:999px; background: var(--divider-color); outline:none; -webkit-appearance:none; appearance:none; }
      input[type="range"]::-webkit-slider-thumb{ -webkit-appearance:none; appearance:none; width:20px; height:20px; border-radius:50%; background: var(--accent-color); box-shadow: 0 0 0 4px rgba(0,0,0,.15); cursor:pointer; }

      /* Themed confirm dialog */
      #target_handle { touch-action: none; }
      .modal { position:absolute; inset:0; background: rgba(0,0,0,.45); display:flex; align-items:center; justify-content:center; z-index: 50; }
      .dialog { background: var(--card-background-color); color: var(--primary-text-color); border:1px solid var(--divider-color); border-radius: 16px; padding: 16px; width: min(92%, 360px); box-shadow: 0 10px 30px rgba(0,0,0,.3); }
      .dialog h3 { margin: 0 0 8px; font-size: 1.1rem; font-weight:800; text-align:left; }
      .dialog p { margin: 0 0 14px; line-height:1.4; color: var(--secondary-text-color); white-space: pre-line; }
      .dialog .time-picker { display:flex; align-items:center; justify-content:center; gap:12px; margin: 16px 0; }
      .dialog .time-picker select { width: auto; min-width: 64px; height: 40px; text-align: center; text-align-last: center; }
      .dialog .time-picker span { font-weight:700; font-size: 1.2rem; }
      .dialog .actions { display:flex; gap:10px; justify-content:flex-end; margin-top: 6px; }
      .btn { border:none; border-radius:12px; padding:10px 14px; font-weight:700; cursor:pointer; }
      .btn.secondary { background: rgba(255,255,255,.06); color: var(--primary-text-color); border:1px solid var(--divider-color); }
      .btn.primary { background: var(--primary-color); color:#fff; }
      .btn.danger { background: var(--error-color); color:#fff; }
    `;

      const ringCirc = 130; // bigger radius
      const gapDeg = 60;
      const rangeDeg = 360 - gapDeg;
      // Place the gap centered at bottom. Start at bottom-left, end at bottom-right.
      // use degree as 0 bottom, positive clockwise , contrary to radian beware in svg : y are downward
      // degree part of [0,360[
      const startDeg = gapDeg / 2;   // bottom-left
      const endDeg = startDeg + rangeDeg; // bottom-right (wraps past 360)

      const deg2rad = (d) => ((270 - d) * Math.PI) / 180;
      const rad2deg = (r) => {
          if (r < 0) r += 2 * Math.PI;
          return (((270 - ((r * 180) / Math.PI)) + 360) % 360);
      };
      const polar = (cx, cy, r, deg) => ({x: cx + r * Math.cos(deg2rad(deg)), y: cy - r * Math.sin(deg2rad(deg))});
      const arcPath = (cx, cy, r, a0, a1) => {
          const p0 = polar(cx, cy, r, a0);
          const p1 = polar(cx, cy, r, a1);
          let delta = a1 - a0;
          if (delta < 0) delta += 360;
          const laf = delta > 180 ? 1 : 0;
          return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${laf} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
      };

      const pctToDeg = (p) => startDeg + (Math.max(0, Math.min(100, p)) / 100) * rangeDeg;

      const socEndDeg = pctToDeg(soc);
      const handlePct =
          this._targetDragPct != null ? this._targetDragPct :
              (this._localTargetPct != null ? this._localTargetPct : (targetPct ?? soc));
      const handleDeg = pctToDeg(handlePct);
      // Enlarge gauge
      const center = {cx: 160, cy: 160};
      const arcLen = 2 * Math.PI * ringCirc * (rangeDeg / 360);
      const segLen = arcLen * (Math.max(0, Math.min(100, soc)) / 100);
      let dashLen = Math.round(segLen * 0.22);
      let gapLen = Math.round(segLen * 0.28);
      dashLen = Math.max(6, dashLen);
      gapLen = Math.max(6, gapLen);
      const patternLen = dashLen + gapLen;
      // Ensure pattern repeats within the segment for visible motion
      if (patternLen >= segLen - 4) {
          const scale = (segLen - 4) / patternLen;
          dashLen = Math.max(4, Math.round(dashLen * scale));
          gapLen = Math.max(4, Math.round(gapLen * scale));
      }
      // Persist animation pattern for smoothness across re-renders
      this._animPatternLen = Math.max(8, dashLen + gapLen);
      // showAnimation will be defined after connection state is resolved
      const handlePos = polar(center.cx, center.cy, ringCirc, handleDeg);
      const bgPath = arcPath(center.cx, center.cy, ringCirc, startDeg, endDeg);
      const socPath = arcPath(center.cx, center.cy, ringCirc, startDeg, socEndDeg);
      const gradGreenId = `gradG-${Math.floor(Math.random() * 1e6)}`;
      const gradChargeId = `gradC-${Math.floor(Math.random() * 1e6)}`;
      const gradDisabledId = `gradD-${Math.floor(Math.random() * 1e6)}`;
      const gradFaultId = `gradF-${Math.floor(Math.random() * 1e6)}`;
      const isDisconnected = (sChargeType?.state === 'Not Plugged' || sChargeType?.state === 'Unknown');
      const chargeTypeState = (sChargeType?.state || '').toLowerCase();
      const isFaulted = chargeTypeState === 'faulted' || chargeTypeState === 'unknown' || chargeTypeState === 'not charging' || chargeTypeState === 'no power to car';

      const chargerOptions = selCharger?.attributes?.options || [];
      const chargerState = (selCharger?.state || '').trim();
      const stateLc = chargerState.toLowerCase();
      const invalidStates = ['unavailable', 'unknown', 'none', 'not plugged', 'not_plugged', 'not connected', 'not_connected'];
      const shouldShowPlaceholder = isDisconnected || !chargerState || invalidStates.includes(stateLc) || !chargerOptions.includes(chargerState);
      const chargerOptionsHtml = shouldShowPlaceholder
          ? [`<option value="" selected>No connected Charger</option>`, ...chargerOptions.map(o => `<option>${o}</option>`)].join('')
          : chargerOptions.map(o => `<option ${o === chargerState ? 'selected' : ''}>${o}</option>`).join('');

      // Person selector options
      const personOptions = selPerson?.attributes?.options || [];
      const personState = (selPerson?.state || '').trim();
      const personStateLc = personState.toLowerCase();
      const personInvalidStates = ['unavailable', 'unknown', 'none'];
      const shouldShowPersonPlaceholder = !personState || personInvalidStates.includes(personStateLc) || !personOptions.includes(personState);
      const personOptionsHtml = shouldShowPersonPlaceholder
          ? [`<option value="" selected>No person attached</option>`, ...personOptions.map(o => `<option>${o}</option>`)].join('')
          : personOptions.map(o => `<option ${o === personState ? 'selected' : ''}>${o}</option>`).join('');

      // Person forecast string
      const personForecastStr = sPersonForecast?.state || '';
      const validPersonForecast = personForecastStr && personForecastStr.toLowerCase() !== 'none' && personForecastStr.toLowerCase() !== 'unknown' && personForecastStr.toLowerCase() !== 'unavailable' && personForecastStr.trim() !== '';
      const forecastDisplay = validPersonForecast ? personForecastStr : 'None';

      const activeGradId = isFaulted ? gradFaultId : (isDisconnected ? gradDisabledId : (charging ? gradChargeId : gradGreenId));
      const showAnimation = (charging && !shouldShowPlaceholder && segLen > 6);

      //const forecastedPersonStr = sForecastedPerson?.state;
      //const showForecastedPerson = forecastedPersonStr && forecastedPersonStr.toLowerCase() !== 'none' && forecastedPersonStr.toLowerCase() !== 'unknown' && forecastedPersonStr.toLowerCase() !== 'unavailable' && forecastedPersonStr.trim() !== '';
      //const displayTitle = showForecastedPerson ? `${title} (${forecastedPersonStr})` : title;
      const displayTitle = title;

      this._root.innerHTML = `
      <ha-card class="card ${isDisconnected ? 'disabled' : ''} ${isFaulted ? 'fault' : ''}">
        <style>${css}</style>
        <div class="card-title">${displayTitle}</div>
        <div class="top"></div>

        <div class="hero">
          <div class="ring" title="${soc}%">
            <svg viewBox="0 0 320 320" width="300" height="300" aria-hidden="true">
              <defs>
                <linearGradient id="${gradGreenId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#00bcd4"/>
                  <stop offset="100%" stop-color="#8bc34a"/>
                </linearGradient>
                <linearGradient id="${gradChargeId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#00e1ff"/>
                  <stop offset="100%" stop-color="#0066ff"/>
                </linearGradient>
                <linearGradient id="${gradDisabledId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#6b6b6b" stop-opacity="0.65"/>
                  <stop offset="100%" stop-color="#a0a0a0" stop-opacity="0.75"/>
                </linearGradient>
                <linearGradient id="${gradFaultId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#ff8a80"/>
                  <stop offset="100%" stop-color="#ff1744"/>
                </linearGradient>
                <filter id="chargeGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <path d="${bgPath}" stroke="${isFaulted ? 'rgba(244,67,54,0.35)' : 'var(--divider-color)'}" stroke-width="14" fill="none" stroke-linecap="round" />
              <path d="${socPath}" stroke="url(#${activeGradId})" stroke-width="14" fill="none" stroke-linecap="round" ${showAnimation ? 'stroke-opacity="0.35"' : ''} />
              ${showAnimation ? `
              <path id="charge_anim"
                    d="${socPath}"
                    stroke="url(#${isFaulted ? gradFaultId : gradChargeId})"
                    stroke-width="16"
                    fill="none"
                    stroke-linecap="round"
                    stroke-dasharray="${dashLen} ${gapLen}"
                    stroke-opacity="1"
                    filter="url(#chargeGlow)"
                    style="mix-blend-mode:screen; will-change: stroke-dashoffset"
              />
              ` : ''}
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="${isDisconnected ? 'var(--divider-color)' : 'var(--primary-color)'}" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="${isDisconnected ? 'var(--secondary-text-color)' : 'var(--primary-color)'}" font-size="${useEnergyMode ? '11' : '13'}" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${useEnergyMode ? Math.round(parseTargetEnergy(target) ?? (Number(sSoc?.state || 0) / 1000)) : Math.round(targetPct ?? soc)}</text>
            </svg>
            <div class="center">
              <div class="stack">
                <div class="soc-block">
                  ${chargeIcon ? `<ha-icon class="charge-type-icon" icon="${chargeIcon}"></ha-icon>` : ''}
                  <div class="pct" style="margin-bottom:0;">${displaySocValue}</div>
                  ${useEnergyMode ? '' : `<div class="mini-range-now" aria-label="current range">${rangeNowStr}</div>`}
                </div>
                <div class="target-block">
                <div class="mini-grid">
                  <div class="mini-title">Force Now</div>
                  <div class="mini-title">${useEnergyMode ? 'Target Energy' : 'Target SOC'}</div>
                  <div class="mini-title">${chargeTimeLabel}</div>

                  <div id="rabbit_btn" class="rabbit-btn ${sChargeType?.state === 'As Fast As Possible' ? 'on' : ''}"><ha-icon icon="mdi:rabbit"></ha-icon></div>
                  <div class="target-cell">
                    <div id="target_value" class="target-value">${displayTargetValue}</div>
                    ${useEnergyMode ? '' : `<div class="mini-range-target" aria-label="range at target">${rangeTargetStr}</div>`}
                  </div>
                  <div id="time_btn" class="time-btn ${chargeTime && chargeTime !== '--:--' ? 'on' : ''}">${chargeTime}</div>
                </div>
                </div>
                <div class="center-controls" style="flex-direction:column; gap:4px;">
                  <div class="mini-title">Solar priority</div>
                  <div id="sun_btn" class="sun-btn ${swPriority?.state === 'on' ? 'on' : ''}"><ha-icon icon="mdi:weather-sunny"></ha-icon></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:account"></ha-icon>
            <select id="person">
              ${personOptionsHtml}
            </select>
          </div>
        </div>
        <div class="forecast-row">Forecast: ${forecastDisplay}</div>
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:ev-station"></ha-icon>
            <select id="charger">
              ${chargerOptionsHtml}
            </select>
          </div>
        </div>
        <div class="below-line full">
           <button id="reset" class="danger pill outline">Reset</button>
        </div>
      </ha-card>
    `;

      // old buttons:
/*      <div className="below-line">
          <button id="schedule_inline" className="secondary btn-clock pill">
              <ha-icon icon="mdi:clock-outline"></ha-icon>
              <span>Charge at:</span></button>
          <div className="time-row" style="margin-top:0;">
              <select id="hour_select">
                  ${Array.from({length: 24}, (_, h) => `<option value="${h}" ${Math.floor(nextTimeMins / 60) === h ? 'selected' : ''}>${String(h).padStart(2, '0')}</option>`).join('')}
              </select>
              <span style="font-weight:700;">:</span>
              <select id="minute_select">
                  ${[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map(m => `<option value="${m}" ${(nextTimeMins % 60) === m ? 'selected' : ''}>${String(m).padStart(2, '0')}</option>`).join('')}
              </select>
          </div>
      </div>
      <div className="below-line full">
          <button id="force" className="primary pill align-left">
              <ha-icon icon="mdi:rabbit"></ha-icon>
              <span className="btn-center">Force charge now</span></button>
      </div>*/

      // Wire events
      const root = this._root;
      const ids = (k) => root.getElementById(k);
      const withEntityId = (id) => ({entity_id: id});

      if (selPerson) {
          const personSel = ids('person');
          const startP = () => {
              this._isInteractingPerson = true;
          };
          const endP = () => {
              this._isInteractingPerson = false;
              this._render();
          };
          personSel?.addEventListener('focus', startP);
          personSel?.addEventListener('blur', endP);
          personSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return; // ignore placeholder
              await this._select(e.attached_person, option);
              // Clear the interaction flag and force re-render to show the updated state
              this._isInteractingPerson = false;
              this._render();
          });
      }

      if (selCharger) {
          const chargerSel = ids('charger');
          const startC = () => {
              this._isInteractingCharger = true;
          };
          const endC = () => {
              this._isInteractingCharger = false;
              this._render();
          };
          chargerSel?.addEventListener('focus', startC);
          chargerSel?.addEventListener('blur', endC);
          chargerSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return; // ignore placeholder
              await this._select(e.charger_select, option);
              // Clear the interaction flag and force re-render to show the updated state
              this._isInteractingCharger = false;
              this._render();
          });
          // In disconnected mode, force placeholder selection and disable the control
          // Do not force selection index; placeholder is marked selected in markup and remains interactive
      }

      if (selLimit) {
          ids('limit')?.addEventListener('change', (ev) => {
              const option = ev.target.value;
              this._select(e.next_limit, option);
          });
      }

      if (tNext) {
          const hourSel = ids('hour_select');
          const minSel = ids('minute_select');
          const update = () => {
              const h = Number(hourSel?.value ?? 0);
              const m = Number(minSel?.value ?? 0);
              const mins = h * 60 + m;
              const hm = formatHm(mins);
              const val = hm + ':00';
              this._localNextTimeMins = mins; // keep local until HA push comes back to avoid select jumping
              this._setTime(e.next_time, val);
          };
          const startInteract = () => {
              this._isInteracting = true;
          };
          const endInteract = () => {
              this._isInteracting = false;
              this._render();
          };
          hourSel?.addEventListener('focus', startInteract);
          minSel?.addEventListener('focus', startInteract);
          hourSel?.addEventListener('blur', endInteract);
          minSel?.addEventListener('blur', endInteract);
          hourSel?.addEventListener('change', update);
          minSel?.addEventListener('change', update);
      }

      if (swPriority) {
          const togglePriority = async () => {
              const btn = ids('sun_btn');
              try {
                  if (swPriority.state === 'on') {
                      await this._turnOff(e.bump_priority);
                      btn?.classList.remove('on');
                  } else {
                      await this._turnOn(e.bump_priority);
                      btn?.classList.add('on');
                  }
              } catch (_) {
                  // ignore errors; HA state will resync UI on next render
              }
          };
          ids('priority')?.addEventListener('click', togglePriority);
          const sbtn = ids('sun_btn');
          if (sbtn) {
              sbtn.style.pointerEvents = 'auto';
              sbtn.addEventListener('click', togglePriority);
          }
      }

      // Rabbit button for force now
      if (e.force_now) {
          const rbtn = ids('rabbit_btn');
          if (rbtn) {
              rbtn.style.pointerEvents = 'auto';
              rbtn.addEventListener('click', async (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  if (this._root?.querySelector('.disabled')) return;

                  // Check if already in "As Fast As Possible" mode
                  const isAlreadyForcing = sChargeType?.state === 'As Fast As Possible';

                  if (isAlreadyForcing && e.clean_constraints) {
                      // Show reset dialog
                      showDialog({
                          title: 'Stop Force Charging',
                          message: 'This will stop the current charge.\nOk to proceed?',
                          buttons: [
                              {text: 'Cancel', variant: 'secondary'},
                              {
                                  text: 'Reset', variant: 'danger', onClick: async () => {
                                      await this._press(e.clean_constraints);
                                  }
                              },
                          ]
                      });
                  } else {
                      // Show force charge dialog
                      showDialog({
                          title: 'Force charge now',
                          message: 'Start full-speed charge immediately?\nThis will use maximum available power.',
                          buttons: [
                              {text: 'Cancel', variant: 'secondary'},
                              {
                                  text: 'Start', variant: 'primary', onClick: async () => {
                                      await this._press(e.force_now);
                                  }
                              },
                          ]
                      });
                  }
              });
          }
      }

      // Time button for finish time
      if (tNext && e.schedule) {
          const tbtn = ids('time_btn');
          if (tbtn) {
              tbtn.style.pointerEvents = 'auto';
              tbtn.addEventListener('click', async (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  if (this._root?.querySelector('.disabled')) return;

                  // Use chargeTime if it's valid (not "--:--"), otherwise fall back to nextTimeMins
                  let defaultHour, defaultMin;
                  if (chargeTime && chargeTime !== '--:--' && chargeTime.includes(':')) {
                      const chargeMins = parseTimeToMinutes(chargeTime);
                      defaultHour = Math.floor(chargeMins / 60);
                      defaultMin = chargeMins % 60;
                      // Round up to the next 5-minute multiple
                      defaultMin = Math.ceil(defaultMin / 5) * 5;
                      if (defaultMin === 60) {
                          defaultMin = 0;
                          defaultHour = (defaultHour + 1) % 24;
                      }
                  } else {
                      defaultHour = Math.floor(nextTimeMins / 60);
                      defaultMin = nextTimeMins % 60;
                      // Round up to the next 5-minute multiple
                      defaultMin = Math.ceil(defaultMin / 5) * 5;
                      if (defaultMin === 60) {
                          defaultMin = 0;
                          defaultHour = (defaultHour + 1) % 24;
                      }
                  }

                  const customContent = `
            <p>Select the next time the charge of the car should end:</p>
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
                      title: 'Charge Finish Time',
                      customContent: customContent,
                      buttons: [
                          {text: 'Cancel', variant: 'secondary'},
                          {
                              text: 'Reset',
                              variant: 'danger',
                              onClick: async () => {
                                  if (e.clean_constraints) await this._press(e.clean_constraints);
                              }
                          },
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
                                  this._localNextTimeMins = mins;
                                  await this._setTime(e.next_time, val);
                                  await this._press(e.schedule);
                              }
                          },
                      ]
                  });
              });
          }
      }
      const showDialog = (opts) => {
          const {title, message, buttons, customContent} = opts;
          const wrap = document.createElement('div');
          wrap.className = 'modal';
          const contentHtml = customContent || `<p>${message}</p>`;
          wrap.innerHTML = `<div class="dialog"><h3>${title}</h3>${contentHtml}<div class="actions"></div></div>`;
          const actions = wrap.querySelector('.actions');
          this._modalOpen = true;
          buttons.forEach(b => {
              const el = document.createElement('button');
              el.className = `btn ${b.variant || 'secondary'}`;
              el.textContent = b.text;
              el.addEventListener('click', () => {
                  if (b.onClick) b.onClick();
                  wrap.remove();
                  this._modalOpen = false;
                  this._render();
              });
              actions.appendChild(el);
          });
          this._root.appendChild(wrap);
          return wrap;
      };

      if (e.force_now) {
          ids('force')?.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              if (this._root?.querySelector('.disabled')) return;
              showDialog({
                  title: 'Force charge now',
                  message: 'Start full-speed charge immediately?\nThis will use maximum available power.',
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
                      {
                          text: 'Start', variant: 'primary', onClick: async () => {
                              await this._press(e.force_now);
                          }
                      },
                  ]
              });
          });
      }

      if (e.schedule) {
          ids('schedule_inline')?.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              if (this._root?.querySelector('.disabled')) return;
              const limit = selLimit?.state || '';
              const time = tNext?.state || '';
              showDialog({
                  title: 'Add scheduled charge',
                  message: `Schedule a charge for ${time} at target SOC: ${limit} .\nProceed?`,
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
                      {
                          text: 'Add', variant: 'primary', onClick: async () => {
                              await this._press(e.schedule);
                          }
                      },
                  ]
              });
          });
      }

      if (e.reset) {
          ids('reset')?.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              showDialog({
                  title: 'Reset car state',
                  message: 'This will reset internal state for this car and cannot be undone.\nProceed?',
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
                      {
                          text: 'Reset', variant: 'danger', onClick: async () => {
                              await this._press(e.reset);
                          }
                      },
                  ]
              });
          });
      }

      // Quick percent chips
      const findOptionByPercent = (percent) => {
          const options = selLimit?.attributes?.options || [];
          const p = String(percent);
          return options.find(o => o.startsWith(p) || o.startsWith(p + "%") || o.includes(`${p}%`));
      };

      const findOptionByEnergy = (energy) => {
          const options = selLimit?.attributes?.options || [];
          const e = String(Math.round(energy));
          return options.find(o => o.startsWith(e) || o.includes(`${e}kWh`) || o.includes(`${e} kWh`));
      };

      const parseEnergyFromOption = (opt) => {
          const m = String(opt).match(/([\d.]+)\s*kWh/i);
          return m ? Number(m[1]) : undefined;
      };

      const parsePctFromOption = (opt) => {
          const m = String(opt).match(/(\d+)\s*%/);
          return m ? Number(m[1]) : undefined;
      };

      const allowedPercents = (selLimit?.attributes?.options || [])
          .map(parsePctFromOption)
          .filter(v => v != null && !Number.isNaN(v))
          .sort((a, b) => a - b);

      const allowedEnergies = (selLimit?.attributes?.options || [])
          .map(parseEnergyFromOption)
          .filter(v => v != null && !Number.isNaN(v))
          .sort((a, b) => a - b);

      const allowedOrDefault = useEnergyMode
          ? (allowedEnergies.length ? allowedEnergies : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
          : (allowedPercents.length ? allowedPercents : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]);

      root.querySelectorAll('.chip[data-pct]')?.forEach(el => {
          el.addEventListener('click', () => {
              const pct = Number(el.getAttribute('data-pct'));
              const opt = findOptionByPercent(pct);
              if (opt) this._select(e.next_limit, opt);
          });
      });

      // Drag target handle on ring
      const svg = this._root.querySelector('.ring svg');
      const handle = this._root.getElementById('target_handle');
      if (svg && handle) {
          const pt = svg.createSVGPoint();
          const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
          const onMove = (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              const e2 = ev.touches ? ev.touches[0] : ev;
              pt.x = e2.clientX;
              pt.y = e2.clientY;
              const cursor = pt.matrixTransform(svg.getScreenCTM().inverse());
              const dx = cursor.x - center.cx;
              const dy = cursor.y - center.cy;
              // Convert to [0,360)
              // -dy because svg y are downward
              let ang = rad2deg(Math.atan2(-dy, dx));
              // Map to arc domain [startDeg, endDeg] allowing values > 360
              let a = ang;
              if (a < startDeg) a = startDeg;
              if (a > endDeg) a = endDeg;
              // Snap to available select options
              const rawPct = ((a - startDeg) / rangeDeg) * 100;
              const list = allowedOrDefault;
              const snapValue = list.reduce((best, v) => Math.abs(v - (useEnergyMode ? (rawPct / 100 * maxCircleValue) : rawPct)) < Math.abs(best - (useEnergyMode ? (rawPct / 100 * maxCircleValue) : rawPct)) ? v : best, list[0]);

              // Store the percentage for the circle (0-100)
              const displayPct = useEnergyMode ? (snapValue / maxCircleValue * 100) : snapValue;
              this._targetDragPct = displayPct;
              this._targetDragValue = snapValue; // Store actual value (percent or kWh)
              this._isInteractingTarget = true;

              // Update handle and target label without full render to keep it smooth
              const angSnap = startDeg + (displayPct / 100) * rangeDeg;
              const pos = polar(center.cx, center.cy, ringCirc, angSnap);
              handle.setAttribute('cx', pos.x.toFixed(2));
              handle.setAttribute('cy', pos.y.toFixed(2));
              const handleText = this._root.getElementById('target_handle_text');
              if (handleText) {
                  handleText.setAttribute('x', pos.x.toFixed(2));
                  handleText.setAttribute('y', pos.y.toFixed(2));
                  handleText.textContent = Math.round(snapValue);
              }
              const tv = this._root.getElementById('target_value');
              if (tv) tv.innerHTML = useEnergyMode ? `${Math.round(snapValue)}<span style="font-size: ${energyUnitFontSize}em;"> kWh</span>` : `${snapValue}%`;
          };
          const onUp = async (ev) => {
              if (ev) {
                  ev.stopPropagation();
                  ev.preventDefault();
              }
              document.removeEventListener('mousemove', onMove);
              document.removeEventListener('touchmove', onMove);
              document.removeEventListener('mouseup', onUp);
              document.removeEventListener('touchend', onUp);
              if (this._targetDragValue != null) {
                  const opt = useEnergyMode ? findOptionByEnergy(this._targetDragValue) : findOptionByPercent(this._targetDragValue);
                  if (opt) await this._select(e.next_limit, opt);
                  // optimistic: keep local until HA pushes the new state
                  this._localTargetPct = this._targetDragPct;
                  this._pendingClearLocalTarget && clearTimeout(this._pendingClearLocalTarget);
                  this._pendingClearLocalTarget = setTimeout(() => {
                      this._localTargetPct = null;
                      this._pendingClearLocalTarget = null;
                      this._render();
                  }, 2000);
              }
              this._targetDragPct = null;
              this._targetDragValue = null;
              // allow re-rendering now
              this._isInteractingTarget = false;
              handle.style.cursor = 'grab';
          };
          const onDown = (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              this._isInteractingTarget = true;
              document.addEventListener('mousemove', onMove);
              document.addEventListener('touchmove', onMove, {passive: false});
              document.addEventListener('mouseup', onUp);
              document.addEventListener('touchend', onUp);
              handle.style.cursor = 'grabbing';
          };
          handle.addEventListener('mousedown', onDown);
          handle.addEventListener('touchstart', onDown, {passive: false});

          // Pointer Events for more reliable drag on modern browsers
          if (window.PointerEvent) {
              const onPointerMove = (ev) => onMove(ev);
              const onPointerUp = async (ev) => {
                  try {
                      handle.releasePointerCapture(ev.pointerId);
                  } catch (_) {
                  }
                  await onUp(ev);
                  handle.removeEventListener('pointermove', onPointerMove);
                  handle.removeEventListener('pointerup', onPointerUp);
                  handle.removeEventListener('pointercancel', onPointerUp);
              };
              const onPointerDown = (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  this._isInteractingTarget = true;
                  try {
                      handle.setPointerCapture(ev.pointerId);
                  } catch (_) {
                  }
                  handle.addEventListener('pointermove', onPointerMove);
                  handle.addEventListener('pointerup', onPointerUp);
                  handle.addEventListener('pointercancel', onPointerUp);
                  handle.style.cursor = 'grabbing';
              };
              handle.addEventListener('pointerdown', onPointerDown);
          }
      }
  }
}

if (!customElements.get('qs-car-card')) {
    customElements.define('qs-car-card', QsCarCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-car-card')) {
    window.customCards.push({
        type: 'qs-car-card',
        name: 'QS Car Card',
        description: 'Quiet Solar car control card',
    });
}
