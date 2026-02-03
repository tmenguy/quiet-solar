/*
  QS Climate Card - custom:qs-climate-card
  Zero-build single-file Lit-style web component compatible with Home Assistant
*/

class QsClimateCard extends HTMLElement {
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
    return { name: "QS Climate", entities: {} };
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
    if (this._isInteractingMode || this._isInteractingStateOn || this._modalOpen || this._isInteractingTarget) return;
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
  _setTime(entity_id, value) { return this._call('time', 'set_value', { entity_id, time: value }); }

  _fmt(num, round = true) {
    const n = Number(num);
    if (num == null || Number.isNaN(n)) return '--';
    return round ? Math.round(n) : n.toFixed(1);
  }

  _render() {
      const cfg = this._config || {};
      const e = cfg.entities || {};

      const sDurationLimit = this._entity(e.duration_limit);
      const sCurrentDuration = this._entity(e.current_duration);
      const sDefaultOnDuration = this._entity(e.default_on_duration);
      const sDefaultOnFinishTime = this._entity(e.default_on_finish_time);
      const sCommand = this._entity(e.command);
      const selClimateMode = this._entity(e.climate_mode);
      const selClimateStateOn = this._entity(e.climate_state_on);
      const selClimateStateOff = this._entity(e.climate_state_off);
      const swGreenOnly = this._entity(e.green_only);
      const swEnableDevice = this._entity(e.enable_device);
      const sOverrideState = this._entity(e.override_state);
      const sStartTime = this._entity(e.start_time);
      const sEndTime = this._entity(e.end_time);
      const sIsOffGrid = this._entity(e.is_off_grid);

      const title = (cfg.title || cfg.name) || "Climate";
      
      // Check if device is enabled
      const isEnabled = swEnableDevice?.state === 'on';
      
      // Check if system is off-grid
      const isOffGrid = sIsOffGrid?.state === 'on';
      
      // Get target hours and current run hours directly (in hours)
      const targetHours = Number(sDurationLimit?.state || 12);
      const hoursRun = Number(sCurrentDuration?.state || 0);
      const defaultDuration = Number(sDefaultOnDuration?.state || 6);
      
      // Get climate mode
      const climateMode = selClimateMode?.state || 'bistate_mode_default';
      const isDefaultMode = climateMode === 'bistate_mode_default';
      
      // Get override state
      const overrideState = sOverrideState?.state || 'NO OVERRIDE';
      const isOverridden = overrideState !== 'NO OVERRIDE';
      const isResettingOverride = overrideState === 'ASKED FOR RESET OVERRIDE';
      
      // Determine if climate is running (command state must be "on")
      const commandState = sCommand?.state || '';
      const running = commandState.toLowerCase() === 'on';
      
      // Determine max hours and what to display
      let maxHours, displayTargetHours;
      if (isDefaultMode) {
        maxHours = 12; // Fixed for default mode
        displayTargetHours = defaultDuration;
      } else {
        maxHours = targetHours;
        displayTargetHours = targetHours;
      }
      
      // Determine if we should show from/to times
      const showFromTo = !isDefaultMode || isOverridden;
      
      // Helper to check if state is valid
      const isValidState = (state) => {
        if (!state) return false;
        const stateLower = String(state).toLowerCase();
        return !['unavailable', 'unknown', 'none', ''].includes(stateLower);
      };
      
      // Get from/to times
      const startTime = (sStartTime && isValidState(sStartTime.state)) ? sStartTime.state : '--:--';
      const endTime = (sEndTime && isValidState(sEndTime.state)) ? sEndTime.state : '--:--';
      
      // Color schemes based on climate_state_on - MUST BE DEFINED BEFORE CSS
      const climateStateOn = (selClimateStateOn?.state || 'cool').toLowerCase();
      const colorSchemes = {
        cool: {
          primary: '#2196F3',
          gradStart: '#00bcd4',
          gradEnd: '#8bc34a',
          animStart: '#00e1ff',
          animEnd: '#0066ff'
        },
        heat: {
          primary: '#FF5722',
          gradStart: '#FF5722',
          gradEnd: '#D32F2F',
          animStart: '#FF6E40',
          animEnd: '#E64A19'
        },
        heat_cool: {
          primary: '#9C27B0',
          gradStart: '#9C27B0',
          gradEnd: '#BA68C8',
          animStart: '#AB47BC',
          animEnd: '#8E24AA'
        },
        auto: {
          primary: '#9C27B0',
          gradStart: '#9C27B0',
          gradEnd: '#BA68C8',
          animStart: '#AB47BC',
          animEnd: '#8E24AA'
        },
        fan_only: {
          primary: '#00BCD4',
          gradStart: '#00BCD4',
          gradEnd: '#4DD0E1',
          animStart: '#00E5FF',
          animEnd: '#00B8D4'
        },
        dry: {
          primary: '#D4A574',
          gradStart: '#D4A574',
          gradEnd: '#E6C9A8',
          animStart: '#E0B589',
          animEnd: '#C9935D'
        }
      };
      
      const colors = colorSchemes[climateStateOn] || colorSchemes.cool;
      
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

      .hero { margin-top: 0px; display:flex; align-items:center; justify-content:center; gap: 12px; }
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
      .ring .center { position:absolute; inset:0; display:grid; place-items:center; text-align:center; pointer-events: none; transform: translateY(-5px); }
      .ring .target-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .target-value { font-weight:800; font-size: 2.5rem; line-height: 1.1; }
      .ring .stack { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; text-align:center; width: 220px; margin: 0 auto; }
      .ring .target-block { display:flex; flex-direction:column; align-items:center; gap:6px; }
      .ring .from-to-row { display:flex; justify-content:space-between; width:140px; margin-top: 8px; gap:20px; }
      .ring .from-to-item { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .ring .from-to-label { color: var(--secondary-text-color); font-weight:700; font-size: .95rem; }
      .ring .from-to-value { color: var(--primary-text-color); font-weight:800; font-size: 1.4rem; }
      .ring .center-controls { display:flex; align-items:center; justify-content:center; margin-top: 6px; }
      .ring .override-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); }
      .ring .override-btn ha-icon { --mdc-icon-size: 26px; color: var(--secondary-text-color); display:block; line-height:1; }
      .ring .override-btn.disabled { cursor: not-allowed; opacity: 0.6; }
      .ring .override-btn.active { border-color: rgba(255,152,0,.45); background: rgba(255,152,0,.14); box-shadow: 0 0 0 3px rgba(255,152,0,.20), 0 0 16px #FF9800; }
      .ring .override-btn.active ha-icon { color: #FF9800; }
      .ring .override-btn.resetting { border-color: rgba(76,175,80,.45); background: rgba(76,175,80,.14); }
      .ring .override-btn.resetting ha-icon { color: #4CAF50; }
      .time-btn { width: 50px; height: 50px; border-radius: 50%; border: 2px solid var(--divider-color); background: rgba(255,255,255,.04); display:grid; place-items:center; cursor:pointer; box-shadow: none; pointer-events: auto; box-sizing: border-box; font-size: 0.99rem; font-weight: 800; line-height: 1; margin-top: 6px; }
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

      // Convert hours to percentage for arc calculation
      const hoursToPct = (hours) => (hours / maxHours) * 100;
      const pctToHours = (pct) => (pct / 100) * maxHours;
      
      const pctToDeg = (p) => startDeg + (Math.max(0, Math.min(100, p)) / 100) * rangeDeg;

      // Progress: hours run as percentage of max hours
      const progressPct = hoursToPct(hoursRun);
      const progressEndDeg = pctToDeg(progressPct);
      
      // Handle: target hours (or default duration for default mode)
      const handlePct = this._targetDragPct != null ? this._targetDragPct : 
                        (this._localTargetPct != null ? this._localTargetPct : hoursToPct(displayTargetHours));
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

      // Climate mode selector options with translations
      const modeOptions = selClimateMode?.attributes?.options || [];
      const modeState = (selClimateMode?.state || '').trim();
      
      // Helper to translate climate mode options
      const translateClimateMode = (value) => {
          const key = `component.quiet_solar.entity.select.climate_mode.state.${value}`;
          const translated = this._hass?.localize?.(key);
          // If translation not found or returns the key itself, fall back to the raw value
          return (translated && translated !== key) ? translated : value;
      };
      
      const modeOptionsHtml = modeOptions.map(o => 
          `<option value="${o}" ${o === modeState ? 'selected' : ''}>${translateClimateMode(o)}</option>`
      ).join('');

      // Climate state on selector options (filtered to exclude the state selected in state_off)
      const stateOnOptions = selClimateStateOn?.attributes?.options || [];
      const stateOnState = (selClimateStateOn?.state || '').trim();
      const stateOffCurrent = (selClimateStateOff?.state || '').trim();
      
      // Helper to translate climate state options
      const translateClimateState = (value) => {
          const key = `component.quiet_solar.entity.select.climate_state_on.state.${value}`;
          const translated = this._hass?.localize?.(key);
          // If translation not found or returns the key itself, fall back to the raw value
          return (translated && translated !== key) ? translated : value;
      };
      
      // Filter out the current state_off value and "off" from state_on options
      const filteredStateOnOptions = stateOnOptions.filter(o => o !== stateOffCurrent && o.toLowerCase() !== 'off');
      
      const stateOnOptionsHtml = filteredStateOnOptions.map(o => 
          `<option value="${o}" ${o === stateOnState ? 'selected' : ''}>${translateClimateState(o)}</option>`
      ).join('');

      // Override button - parse and determine icon
      // Parse override command from override state
      const parseOverrideCommand = (overrideStateStr) => {
        if (!overrideStateStr || overrideStateStr === 'NO OVERRIDE') return null;
        const match = String(overrideStateStr).match(/Override:\s*(.+)/i);
        return match ? match[1].trim() : null;
      };
      
      const overrideCommand = parseOverrideCommand(overrideState);
      const overrideCommandLower = overrideCommand ? overrideCommand.toLowerCase() : '';
      
      // Check if override is for "off" state (ends with "off" OR matches climate_state_off)
      const isOverrideOff = overrideCommand && (
        overrideCommandLower.endsWith('off') || 
        overrideCommand === stateOffCurrent
      );

      // Override button classes
      let overrideBtnClass = 'override-btn';
      let overrideBtnIcon = 'mdi:hand-back-right-off';
      let overrideBtnClickable = false;
      if (isResettingOverride) {
        overrideBtnClass += ' resetting disabled';
        overrideBtnIcon = isOverrideOff ? 'mdi:power-off' : 'mdi:hand-back-right';
      } else if (isOverridden) {
        overrideBtnClass += ' active';
        overrideBtnIcon = isOverrideOff ? 'mdi:power-off' : 'mdi:hand-back-right';
        overrideBtnClickable = true;
      }

      // Format time for display (remove seconds if present)
      const formatTime = (timeStr) => {
        if (!timeStr || timeStr === '--:--') return '--:--';
        const stateLower = String(timeStr).toLowerCase();
        if (['unavailable', 'unknown', 'none'].includes(stateLower)) return '--:--';
        const parts = String(timeStr).split(':');
        if (parts.length < 2) return '--:--';
        return `${parts[0]}:${parts[1]}`;
      };

      // Determine if we can drag the handle (only in default mode, enabled, and not overridden)
      const canDragHandle = isEnabled && isDefaultMode && !isOverridden && displayTargetHours > 0;

      // Parse time to minutes for time picker
      const parseTimeToMinutes = (txt) => {
          if (!txt) return 420; // 07:00
          const parts = String(txt).split(':').map(Number);
          const h = parts[0] || 0, m = parts[1] || 0;
          return h * 60 + m;
      };
      
      const formatHm = (mins) => {
          if (mins == null) return '';
          const h = String(Math.floor(mins / 60)).padStart(2, '0');
          const m = String(mins % 60).padStart(2, '0');
          return `${h}:${m}`;
      };

      const finishTimeStr = sDefaultOnFinishTime?.state || '07:00:00';
      const finishTimeMins = this._localFinishTimeMins != null ? this._localFinishTimeMins : parseTimeToMinutes(finishTimeStr);

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
                  <stop offset="0%" stop-color="${colors.gradStart}"/>
                  <stop offset="100%" stop-color="${colors.gradEnd}"/>
                </linearGradient>
                <linearGradient id="${gradRunningId}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="${colors.animStart}"/>
                  <stop offset="100%" stop-color="${colors.animEnd}"/>
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
              ${canDragHandle ? `
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="${colors.primary}" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="${colors.primary}" font-size="13" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${this._fmt(pctToHours(handlePct))}</text>
              ` : ''}
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
                    <div class="from-to-value">${isDefaultMode ? '--:--' : formatTime(startTime)}</div>
                  </div>
                  <div class="from-to-item">
                    <div class="from-to-label">To:</div>
                    <div class="from-to-value">${isDefaultMode ? (sDefaultOnFinishTime ? formatTime(finishTimeStr) : '--:--') : formatTime(endTime)}</div>
                  </div>
                </div>
                ` : ''}
                ${isDefaultMode && sDefaultOnFinishTime ? `
                <div class="center-controls" style="flex-direction:column; gap:4px;">
                  <div style="color: var(--secondary-text-color); font-weight:700; font-size: .75rem;">Change Finish Time</div>
                  <div id="time_btn" class="time-btn ${finishTimeStr && finishTimeStr !== '--:--' ? 'on' : ''}">${formatTime(finishTimeStr)}</div>
                </div>
                ` : ''}
              </div>
            </div>
            ${e.override_reset ? `<div id="override_btn" class="${overrideBtnClass}"><ha-icon icon="${overrideBtnIcon}"></ha-icon></div>` : ''}
            ${swGreenOnly ? `<div id="green_btn" class="green-btn ${swGreenOnly.state === 'on' ? 'on' : ''}"><ha-icon icon="mdi:leaf"></ha-icon></div>` : ''}
          </div>
        </div>

        ${selClimateMode ? `
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:sun-thermometer-outline"></ha-icon>
            <select id="climate_mode">
              ${modeOptionsHtml}
            </select>
          </div>
        </div>
        ` : ''}
        ${selClimateStateOn ? `
        <div class="below">
          <div class="pill">
            <ha-icon icon="mdi:state-machine"></ha-icon>
            <select id="climate_state_on">
              ${stateOnOptionsHtml}
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

      // Climate mode selector
      if (selClimateMode) {
          const modeSel = ids('climate_mode');
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
              await this._select(e.climate_mode, option);
              
              // Wait a bit for HA state to propagate, then allow re-render
              setTimeout(() => {
                  this._isProcessingModeChange = false;
                  this._isInteractingMode = false;
                  this._render();
              }, 300);
          });
      }

      // Climate state on selector
      if (selClimateStateOn) {
          const stateOnSel = ids('climate_state_on');
          const startS = () => {
              this._isInteractingStateOn = true;
          };
          const endS = () => {
              // Don't clear flag on blur during change processing
              if (!this._isProcessingStateOnChange) {
                  this._isInteractingStateOn = false;
                  this._render();
              }
          };
          stateOnSel?.addEventListener('focus', startS);
          stateOnSel?.addEventListener('blur', endS);
          stateOnSel?.addEventListener('change', async (ev) => {
              const option = ev.target.value;
              if (!option) return;
              
              this._isProcessingStateOnChange = true;
              
              // Call the service and wait for it to complete
              await this._select(e.climate_state_on, option);
              
              // Wait a bit for HA state to propagate, then allow re-render
              setTimeout(() => {
                  this._isProcessingStateOnChange = false;
                  this._isInteractingStateOn = false;
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

      // Override button
      if (e.override_reset && overrideBtnClickable) {
          const obtn = ids('override_btn');
          if (obtn) {
              obtn.style.pointerEvents = 'auto';
              obtn.addEventListener('click', async (ev) => {
                  ev.stopPropagation();
                  ev.preventDefault();
                  showDialog({
                      title: 'Reset override',
                      message: 'This will reset the manual override and return to automatic mode.\nProceed?',
                      buttons: [
                          {text: 'Cancel', variant: 'secondary'},
                          {
                              text: 'Reset', variant: 'primary', onClick: async () => {
                                  await this._press(e.override_reset);
                              }
                          },
                      ]
                  });
              });
          }
      }

      // Time button for finish time (default mode only)
      if (isDefaultMode && sDefaultOnFinishTime) {
          const timeHandler = async (ev) => {
              ev.stopPropagation();
              ev.preventDefault();

              const defaultHour = Math.floor(finishTimeMins / 60);
              const defaultMin = finishTimeMins % 60;

              const customContent = `
            <p>Select the time the climate should finish by:</p>
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
                  title: 'Climate Finish Time',
                  customContent: customContent,
                  buttons: [
                      {text: 'Cancel', variant: 'secondary'},
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
                              this._localFinishTimeMins = mins;
                              await this._setTime(e.default_on_finish_time, val);
                          }
                      },
                  ]
              });
          };

          const tbtn = ids('time_btn');
          if (tbtn) {
              tbtn.style.pointerEvents = 'auto';
              tbtn.addEventListener('click', timeHandler);
          }
      }

      // Reset button
      if (e.reset) {
          ids('reset')?.addEventListener('click', async (ev) => {
              ev.stopPropagation();
              ev.preventDefault();
              showDialog({
                  title: 'Reset climate state',
                  message: 'This will reset internal state for the climate and cannot be undone.\nProceed?',
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

      // Drag target handle on ring (only in default mode)
      if (canDragHandle) {
          const svg = this._root.querySelector('.ring svg');
          const handle = this._root.getElementById('target_handle');
          if (svg && handle) {
              const pt = svg.createSVGPoint();
              
              // Allowed hours (snap points) - 0.5 hour increments from 0 to 12
              const allowedHours = [];
              for (let i = 0; i <= 12; i += 0.5) {
                  allowedHours.push(i);
              }
              
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
                      handleText.textContent = this._fmt(snapHours, false);
                  }
                  const tv = this._root.querySelector('.target-value');
                  if (tv) {
                      tv.innerHTML = `<span style="color: var(--primary-text-color);">${this._fmt(hoursRun, false)}h</span><span style="color: var(--primary-text-color);"> / </span><span style="color: ${colors.primary};">${this._fmt(snapHours, false)}h</span>`;
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
                  
                  // Update default duration
                  if (this._targetDragValue != null && e.default_on_duration) {
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
}

if (!customElements.get('qs-climate-card')) {
    customElements.define('qs-climate-card', QsClimateCard);
}

window.customCards = window.customCards || [];
if (!window.customCards.find((c) => c.type === 'qs-climate-card')) {
    window.customCards.push({
        type: 'qs-climate-card',
        name: 'QS Climate Card',
        description: 'Quiet Solar climate control card',
    });
}
