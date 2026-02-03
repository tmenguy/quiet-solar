/*
  QS Pool Card - custom:qs-pool-card
  Zero-build single-file Lit-style web component compatible with Home Assistant
*/

class QsPoolCard extends HTMLElement {
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
      const p = this._root?.getElementById('running_anim');
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
    return { name: "QS Pool", entities: {} };
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
      const targetHours = Number(sDurationLimit?.state || 12);
      const hoursRun = Number(sCurrentDailyRunDuration?.state || 0);
      
      // Determine if pool is running (command state must be "on")
      const commandState = sCommand?.state || '';
      const running = commandState.toLowerCase() === 'on';
      
      const css = `
      :host { --pad: 18px; display:block; }
      .card { padding: var(--pad); position: relative; }
      .card.off-grid { background: rgba(244, 67, 54, 0.08); }
      .card-title { text-align:center; font-weight:800; font-size: 1.6rem; margin: 0px 0 0px; }
      .top { display:flex; gap:12px; flex-wrap:wrap; }
      .below { display:flex; align-items:center; justify-content:center; margin-top: 8px; width:260px; margin-left:auto; margin-right:auto; }
      .below .pill { width:100%; justify-content:center; }
      .below-line { width:260px; margin: 8px auto 0; display:grid; grid-template-columns: 1fr auto; align-items:center; column-gap:12px; }
      .below-line.full { display:block; }
      .below-line.full > button { width: 100%; justify-content: center; position: relative; }
      .pill { display:flex; align-items:center; gap:8px; border-radius: 28px; height:40px; min-height:40px; padding:0 12px; border:1px solid var(--divider-color);
              background: var(--ha-card-background, var(--card-background-color)); box-sizing: border-box; }
      .pill .dot { width:12px; height:12px; border-radius:50%; background: var(--divider-color); box-shadow: 0 0 8px rgba(0,0,0,.25) inset; }
      .pill.on { background: rgba(56,142,60,0.15); border-color: rgba(56,142,60,.35); }
      .pill.on .dot { background: #2ecc71; box-shadow: 0 0 12px #2ecc71aa; }
      .pill { position: relative; }
      .pill select { appearance:none; background: transparent; color: var(--primary-text-color); border: none; font-weight:700; flex:1; width:auto; text-align:center; text-align-last:center; padding-left: 8px; }

      .hero { margin-top: 0px; display:flex; align-items:center; justify-content:center; }
      .ring { position: relative; width:300px; height:300px; margin: 0 auto; }
      .ring .green-btn { width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; left: 50%; top: 50%; transform: translate(97px, -137px); z-index: 10; }
      .ring .green-btn ha-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); display:block; line-height:1; }
      .ring .green-btn.on { border-color: rgba(56,142,60,.45); background: rgba(46,204,113,.14); box-shadow: 0 0 0 3px rgba(46,204,113,.20), 0 0 16px #4CAF50; }
      .ring .green-btn.on ha-icon { color: #4CAF50; }
      .ring .power-btn { width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; left: 50%; top: 50%; transform: translate(-137px, -137px); z-index: 10; }
      .ring .power-btn ha-icon { --mdc-icon-size: 20px; color: var(--secondary-text-color); display:block; line-height:1; }
      .ring .power-btn.on { border-color: rgba(33,150,243,.45); background: rgba(33,150,243,.14); box-shadow: 0 0 0 3px rgba(33,150,243,.20), 0 0 16px #2196F3; }
      .ring .power-btn.on ha-icon { color: #2196F3; }
      .card.disabled { opacity: 0.5; pointer-events: none; filter: grayscale(0.8); }
      .card.disabled .power-btn { pointer-events: auto; opacity: 1; filter: grayscale(0); }
      .ring .center { position:absolute; inset:0; display:grid; place-items:center; text-align:center; pointer-events: none; transform: translateY(16px); }
      .ring .pct { font-size: 4rem; font-weight:800; letter-spacing:-1px; line-height:1; }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .target-value { color: var(--primary-color); font-weight:800; font-size: 1.5rem; line-height: 1; }
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
      .modal { position:absolute; inset:0; background: rgba(0,0,0,.45); display:flex; align-items:center; justify-content:center; z-index: 50; }
      .dialog { background: var(--card-background-color); color: var(--primary-text-color); border:1px solid var(--divider-color); border-radius: 16px; padding: 16px; width: min(92%, 360px); box-shadow: 0 10px 30px rgba(0,0,0,.3); }
      .dialog h3 { margin: 0 0 8px; font-size: 1.1rem; font-weight:800; text-align:left; }
      .dialog p { margin: 0 0 14px; line-height:1.4; color: var(--secondary-text-color); white-space: pre-line; }
      .dialog .actions { display:flex; gap:10px; justify-content:flex-end; margin-top: 6px; }
      .btn { border:none; border-radius:12px; padding:10px 14px; font-weight:700; cursor:pointer; }
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
      
      // Handle: target hours (only show if enabled and valid)
      const hasValidTarget = isEnabled && targetHours > 0;
      const handlePct = this._targetDragPct != null ? this._targetDragPct : 
                        (this._localTargetPct != null ? this._localTargetPct : hoursToPct(targetHours));
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
        <div class="card-title">${title}</div>
        <div class="top"></div>

        <div class="hero">
          <div class="ring">
            ${swEnableDevice ? `<div id="power_btn" class="power-btn ${isEnabled ? 'on' : ''}"><ha-icon icon="mdi:power"></ha-icon></div>` : ''}
            <svg viewBox="0 0 320 320" width="300" height="300" aria-hidden="true">
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
              
              // Call the service and wait for it to complete
              await this._select(e.pool_mode, option);
              
              // Wait a bit for HA state to propagate, then allow re-render
              setTimeout(() => {
                  this._isProcessingModeChange = false;
                  this._isInteractingMode = false;
                  this._render();
              }, 300);
          });
      }

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
          }
      }

      // Reset button
      if (e.reset) {
          ids('reset')?.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
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
          });
      }

      const showDialog = (opts) => {
          const {title, message, buttons} = opts;
          const wrap = document.createElement('div');
          wrap.className = 'modal';
          wrap.innerHTML = `<div class="dialog"><h3>${title}</h3><p>${message}</p><div class="actions"></div></div>`;
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

      // Drag target handle on ring
      const svg = this._root.querySelector('.ring svg');
      const handle = this._root.getElementById('target_handle');
      if (svg && handle) {
          const pt = svg.createSVGPoint();
          const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
          
          // Allowed hours (snap points) - 1 hour increments from 0 to 24
          const allowedHours = Array.from({length: 25}, (_, i) => i); // [0, 1, 2, 3, ..., 24]
          
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
              
              // Snap to nearest allowed hour
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
                  // Get current run hours from entity
                  const currentRunHours = Number(sCurrentDailyRunDuration?.state || 0);
                  tv.innerHTML = `<span style="color: var(--primary-text-color);">${this._fmt(currentRunHours, false)}h</span><span style="color: var(--primary-text-color);"> / </span><span style="color: var(--primary-color);">${this._fmt(snapHours)}h</span>`;
              }
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
              
              // When handle is dragged, switch to "Use Default" mode and update default_on_duration
              if (this._targetDragValue != null && e.default_on_duration && e.pool_mode) {
                  // First, select "bistate_mode_default" in the pool mode
                  await this._select(e.pool_mode, 'bistate_mode_default');
                  // Then set the new default duration
                  await this._setNumber(e.default_on_duration, this._targetDragValue);
                  // Keep local state optimistic until HA updates
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

          // Pointer Events for more reliable drag
          if (window.PointerEvent) {
              const onPointerMove = (ev) => onMove(ev);
              const onPointerUp = async (ev) => {
                  try {
                      handle.releasePointerCapture(ev.pointerId);
                  } catch (_) {}
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
                  } catch (_) {}
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
