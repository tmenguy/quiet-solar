/*
  QS-199 — Shared base class for every qs-*-card.js.

  Provides:
    - HTMLElement lifecycle (setConfig / connectedCallback /
      disconnectedCallback / set hass / getCardSize abstract override).
    - Service callers (_entity, _call, _press, _turnOn, _turnOff,
      _select, _setNumber, _setTime).
    - Defensive utilities (_escapeHtml, _safeNumber — climate-card
      hardened variant with trim + Number.isFinite, _fmt).
    - State helpers (_isValidState, _formatTime, _parseTimeToMinutes,
      _formatHm).
    - Modal dialog (_showDialog with N12/N13 hardening).
    - Keyboard activation (_registerKeyActivation).
    - Per-instance ID counter (_instanceId).
    - Dashed-arc RAF helpers (_startArcDashAnimation / _stopArcDashAnimation).
    - Wire helpers (_wireTargetHandle, _wireTimePicker, _wireResetButton,
      _wirePowerButton, _wireGreenButton) hoisted from the duration sub-base
      so the car card (which does NOT extend `QsRingDurationCardBase`) can
      consume them too.
    - Ring builder (_buildRingHTML) — QS-235 moved it UP from
      `QsRingDurationCardBase` so the car card (which extends `QsCardBase`
      directly) consumes it too. The 5 duration cards inherit it unchanged.
      Backward-compatible optional params (handleLabel / bgStroke /
      handleFontSize / handleStroke / handleFill / animPathId / extraDefs)
      default to the duration-card output; the car overrides them.
    - Ring bottom-center carve cover (_ringCarveCover + shared
      RING_BOTTOM_CARVE_* constants) — QS-235 unified the duplicated
      bottom-center `<circle>` cover across the car (sun-btn) and the
      duration cards (override-btn). The car reuses the helper for its
      rabbit/time covers with car-local geometry.
    - Named exports of the geometry pure functions (deg2rad, rad2deg,
      polar, arcPath, pctToDeg) — concrete-planner finding R5.

  Defensive-hardening glossary (S6/S7/S8/S9/S14/S16/S17/M2/M4/N7/N12/
  N13/BH) lives in `docs/stories/QS-199.story.md` § "Defensive-hardening
  glossary". Inline comments below reference the codes.
*/

// --- Geometry pure functions (named exports) ---

// 270° offset puts 0° at the top of the ring (12 o'clock position).
export function deg2rad(d) {
    return ((270 - d) * Math.PI) / 180;
}

export function rad2deg(r) {
    if (r < 0) r += 2 * Math.PI;
    return (((270 - ((r * 180) / Math.PI)) + 360) % 360);
}

export function polar(cx, cy, r, deg) {
    return { x: cx + r * Math.cos(deg2rad(deg)), y: cy - r * Math.sin(deg2rad(deg)) };
}

/*
  arcPath — SVG `d` attribute for an arc from a0° to a1° on a circle.
  BH defense-in-depth:
    - non-finite angles (NaN / Infinity) would render as
      `A 130 130 0 0 1 NaN NaN` and trigger a browser "Configuration
      error". Returns empty string so the consumer omits the <path>.
    - N8: zero-length arc (a0 == a1) produces a single-point stray dot;
      return empty string so the consumer can omit the path.
*/
export function arcPath(cx, cy, r, a0, a1) {
    if (!Number.isFinite(a0) || !Number.isFinite(a1)) return '';
    if (Math.abs(a1 - a0) < 0.01) return '';
    const p0 = polar(cx, cy, r, a0);
    const p1 = polar(cx, cy, r, a1);
    let delta = a1 - a0;
    if (delta < 0) delta += 360;
    const laf = delta > 180 ? 1 : 0;
    return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${laf} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
}

/*
  pctToDeg — map a 0..100 percentage to the angle range
  [startDeg, startDeg + rangeDeg]. Clamps the input to [0, 100].
*/
export function pctToDeg(p, startDeg, rangeDeg) {
    return startDeg + (Math.max(0, Math.min(100, p)) / 100) * rangeDeg;
}

// QS-235 — shared bottom-center carve-cover geometry. The duration
// cards (radiator / water-boiler / climate) draw an `override_btn_cover`
// circle here; the car draws its `sun_btn_cover` at the same point. Both
// erase a clean circular patch of the inside-disc animation so the
// bottom-center button reads clearly (QS-217 motivation). Single source
// of truth so the three duration `OVERRIDE_BTN_CARVE_*` duplicates and
// the car's `SUN_BTN_CARVE_*` duplicate collapse to one constant set.
export const RING_BOTTOM_CARVE_CX = 160;   // SVG x-centre of the ring
export const RING_BOTTOM_CARVE_CY = 277;   // button-centre y (derived from CSS .*-btn position)
export const RING_BOTTOM_CARVE_R = 35;     // cover radius (user-tunable)


export class QsCardBase extends HTMLElement {
    // ----- Lifecycle -----

    connectedCallback() {
        // RAF intentionally NOT started here — _render() decides when to
        // start the dashed-arc animation via _startArcDashAnimation().
    }

    disconnectedCallback() {
        this._stopAnimation();
        // S7: reset interaction flags so a re-attach after mid-interaction
        // (e.g. dragging the ring or processing a mode change when the
        // dashboard rearranges) doesn't silently short-circuit `set hass`
        // on stale flags.
        this._isInteractingMode = false;
        this._isInteractingTarget = false;
        this._isProcessingModeChange = false;
        this._modalOpen = false;
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

    // Each card supplies its own `getCardSize()` (NOT defined on base —
    // car returns 6, others return 5).

    _render() {
        throw new Error("QsCardBase._render() must be overridden by subclass");
    }

    // ----- Service callers -----

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

    // ----- Defensive utilities -----

    // S6: defence-in-depth HTML escaping for user-/3rd-party-controlled
    // strings interpolated into innerHTML.
    _escapeHtml(s) {
        if (s == null) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // S8: safe numeric coercion (hardened climate-card variant).
    //   - filters degenerate states BEFORE conversion
    //   - QS-210 S6: trims whitespace-only string so `_safeNumber({state: " "})`
    //     doesn't coerce via `Number(" ") === 0`
    //   - QS-210 S7: `Number.isFinite` guard against ±Infinity (which
    //     `Number.isNaN` does NOT catch)
    _safeNumber(sensor, defaultValue) {
        if (!sensor || sensor.state == null) return defaultValue;
        const raw = sensor.state;
        const trimmed = (typeof raw === 'string') ? raw.trim() : raw;
        if (trimmed === '') return defaultValue;
        if (trimmed === 'unknown' || trimmed === 'unavailable') return defaultValue;
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : defaultValue;
    }

    _fmt(num, round = true) {
        const n = Number(num);
        if (num == null || Number.isNaN(n)) return '--';
        return round ? Math.round(n) : n.toFixed(1);
    }

    // ----- State helpers -----

    _isValidState(state) {
        if (!state) return false;
        const stateLower = String(state).toLowerCase();
        return !['unavailable', 'unknown', 'none', ''].includes(stateLower);
    }

    _formatTime(timeStr) {
        if (!timeStr || timeStr === '--:--') return '--:--';
        const stateLower = String(timeStr).toLowerCase();
        if (['unavailable', 'unknown', 'none'].includes(stateLower)) return '--:--';
        const parts = String(timeStr).split(':');
        if (parts.length < 2) return '--:--';
        return `${parts[0]}:${parts[1]}`;
    }

    _parseTimeToMinutes(txt) {
        // QS-199 review-fix S16 — fall back to the documented 07:00
        // default for empty / `unavailable` / `unknown` / non-time
        // states, instead of letting `Number('unavailable')` → NaN → 0
        // map them to midnight.
        if (!txt || !this._isValidState(txt)) return 420; // 07:00
        const parts = String(txt).split(':').map(Number);
        const h = parts[0] || 0;
        const m = parts[1] || 0;
        return h * 60 + m;
    }

    _formatHm(mins) {
        if (mins == null) return '';
        const h = String(Math.floor(mins / 60)).padStart(2, '0');
        const m = String(mins % 60).padStart(2, '0');
        return `${h}:${m}`;
    }

    // ----- Per-instance ID counter (per concrete-planner finding) -----
    // Each subclass gets its own monotonic counter on
    // `this.constructor._nextInstanceId`. Useful for SVG `<linearGradient>`
    // id collisions across multiple cards of the same type on one page.
    _instanceId(prefix) {
        const ctor = this.constructor;
        ctor._nextInstanceId = (ctor._nextInstanceId || 0) + 1;
        return `${prefix}-${ctor._nextInstanceId}`;
    }

    // ----- Dashed-arc RAF loop -----
    // The `running_anim` SVG path animates its `stroke-dashoffset` to
    // produce a moving-dash effect during the "running" state. RAF is
    // gated by the card via `_render()` calling start/stop — base
    // does not auto-start. Method names (_startAnimation /
    // _stopAnimation) are kept stable so existing structural tests
    // that grep for them keep passing without renaming.
    _startAnimation() {
        if (this._animRaf != null) return;
        this._lastAnimTs = null;
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
            // Subclass hook for additional RAF work (e.g. flame/wave engines).
            this._onAnimationTick?.(ts, dt);
            // QS-199 review-fix #02 S7 — a subclass tick may call
            // `_stopAnimation()` (e.g. radiator stopping once the flame
            // settles to idle), which nulls `_animRaf`. Respect that and
            // don't reschedule, otherwise the final line would restart the
            // loop the tick just asked to stop.
            if (this._animRaf == null) return;
            this._animRaf = requestAnimationFrame(step);
        };
        this._animRaf = requestAnimationFrame(step);
    }

    _stopAnimation() {
        if (this._animRaf != null) cancelAnimationFrame(this._animRaf);
        this._animRaf = null;
        this._lastAnimTs = null;
    }

    // ----- Modal dialog -----
    /*
      _showDialog({title, message, customContent?, buttons, onClose?})
        title          — dialog header (escaped)
        message        — body text (escaped); ignored when customContent provided
        customContent  — pre-rendered HTML for the body
        buttons        — array of {text, variant?, onClick?}
                         When empty/missing, base appends a "Close" fallback (N12)
      Returns the `wrap` element so callers can read dialog selects in onClick.

      N12: empty buttons → "Close" fallback so the modal always dismisses.
      N13: per-button `activate` wraps onClick in try/finally so a
           synchronous throw doesn't leave _modalOpen=true wedged.
    */
    _showDialog(opts) {
        const { title, message, buttons, customContent, onClose } = opts;
        const safeButtons = (Array.isArray(buttons) && buttons.length > 0)
            ? buttons
            : [{ text: 'Close' }];
        const wrap = document.createElement('div');
        wrap.className = 'modal';
        const contentHtml = customContent || `<p>${this._escapeHtml(message)}</p>`;
        wrap.innerHTML = `<div class="dialog"><h3>${this._escapeHtml(title)}</h3>${contentHtml}<div class="actions"></div></div>`;
        const actions = wrap.querySelector('.actions');
        this._modalOpen = true;
        safeButtons.forEach((b) => {
            const el = document.createElement('button');
            el.className = `btn ${b.variant || 'secondary'}`;
            el.textContent = b.text;
            let activated = false;
            const activate = async () => {
                if (activated) return;
                activated = true;
                // N13: try/finally so a throw (sync OR async rejection)
                // can't leave the modal locked.
                // QS-199 review-fix N1: `await b.onClick()` so the dialog
                // stays until the (almost always async) service call
                // resolves and rejections are captured here rather than
                // surfacing as unhandled promise rejections.
                try {
                    if (b.onClick) await b.onClick();
                } catch (_) {
                    // swallow — HA state resyncs on the next push
                } finally {
                    wrap.remove();
                    this._modalOpen = false;
                    if (onClose) {
                        try { onClose(); } catch (_) { /* ignore */ }
                    }
                    this._render();
                }
            };
            el.addEventListener('click', activate);
            el.addEventListener('touchend', (ev) => {
                ev.preventDefault();
                activate();
            });
            this._registerKeyActivation(el, activate);
            actions.appendChild(el);
        });
        this._root.appendChild(wrap);
        return wrap;
    }

    // ----- Keyboard activation -----
    // S16: registers Enter/Space handlers on a `role="button" tabindex="0"`
    // element so keyboard-only users can trigger the same action as
    // click/touchend.
    _registerKeyActivation(el, action) {
        if (!el) return;
        el.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') {
                ev.preventDefault();
                action();
            }
        });
    }

    // ----- Wire helpers -----
    /*
      _wireTargetHandle — wires pointer + mouse + touch drag on the ring
      target handle (SVG <circle id="target_handle">). Snaps the cursor
      angle to the nearest allowed value. S17: try/finally around the
      commit so the drag-release guards always clear, even on service
      failure.

      params: {
          ringSvg, handle,        // DOM refs
          center, ringCirc,        // {cx, cy} and ring radius
          startDeg, endDeg, rangeDeg,
          hoursToPct, pctToHours,  // duration value↔pct mapping (defaults below)
          allowedHours,            // duration snap list (alias of allowedValues)
          entityId,                // service target for the default _setNumber commit
          onBeforeCommit,          // optional async hook AWAITED before the commit
                                   //   (e.g. pool selects default mode first)
          getHoursRun,             // optional () => current hours for the default label update
          colors,                  // optional palette {primary} for the default label update
      }

      QS-235 — generalized so the car card (which commits via
      `_select`, maps %↔kWh, and updates a different live label) can
      consume it. All NEW params default to the duration behaviour so a
      caller that passes none of them is byte-equivalent:
          allowedValues  — snap list (default `allowedHours`)
          pctToValue     — pct → snap-domain value (default `pctToHours`)
          valueToPct     — snap-domain value → pct (default `hoursToPct`)
          onCommit       — async (value) => …; REPLACES the default
                           `_setNumber(entityId, value)` write when provided
                           (still AFTER `onBeforeCommit`). The car maps
                           value → select option then `_select(...)`.
          onDragMove     — (value) => …; REPLACES the default `.target-value`
                           live-label update (car updates `#target_value`).
          fmtHandleText  — (value) => string for the handle TEXT
                           (default `this._fmt(value, false)`).
    */
    _wireTargetHandle(params) {
        const {
            ringSvg, handle, center, ringCirc,
            startDeg, endDeg, rangeDeg,
            hoursToPct, pctToHours, allowedHours,
            allowedValues = allowedHours,
            pctToValue = pctToHours,
            valueToPct = hoursToPct,
            entityId, onBeforeCommit, getHoursRun, colors,
            onCommit, onDragMove,
            fmtHandleText = (v) => this._fmt(v, false),
        } = params;
        if (!ringSvg || !handle) return;
        // QS-199 review-fix S15 — guard against an empty snap list: the
        // `reduce` below would otherwise seed `best = undefined` and
        // `valueToPct(undefined)` → NaN would break the handle.
        if (!Array.isArray(allowedValues) || allowedValues.length === 0) return;

        const pt = ringSvg.createSVGPoint();

        // QS-235 — default live `.target-value` updater reproduces the
        // duration-card label byte-for-byte; the car overrides it via
        // `onDragMove` to update its `#target_value` (energy/percent).
        const dragMove = onDragMove || ((snapValue) => {
            const tv = this._root.querySelector('.target-value');
            if (tv && getHoursRun && colors) {
                const hoursRun = getHoursRun();
                tv.innerHTML = `<span style="color: var(--primary-text-color);">${this._fmt(hoursRun, false)}h</span><span style="color: var(--primary-text-color);"> / </span><span style="color: ${colors.primary};">${this._fmt(snapValue, false)}h</span>`;
            }
        });

        const onMove = (ev) => {
            ev.stopPropagation();
            ev.preventDefault();
            const e2 = ev.touches ? ev.touches[0] : ev;
            pt.x = e2.clientX;
            pt.y = e2.clientY;
            const cursor = pt.matrixTransform(ringSvg.getScreenCTM().inverse());
            const dx = cursor.x - center.cx;
            const dy = cursor.y - center.cy;
            let ang = rad2deg(Math.atan2(-dy, dx));
            let a = ang;
            if (a < startDeg) a = startDeg;
            if (a > endDeg) a = endDeg;
            const rawPct = ((a - startDeg) / rangeDeg) * 100;
            const rawValue = pctToValue(rawPct);

            const snapValue = allowedValues.reduce(
                (best, v) => (Math.abs(v - rawValue) < Math.abs(best - rawValue) ? v : best),
                allowedValues[0],
            );

            const displayPct = valueToPct(snapValue);
            this._targetDragPct = displayPct;
            this._targetDragValue = snapValue;
            this._isInteractingTarget = true;

            const angSnap = startDeg + (displayPct / 100) * rangeDeg;
            const pos = polar(center.cx, center.cy, ringCirc, angSnap);
            handle.setAttribute('cx', pos.x.toFixed(2));
            handle.setAttribute('cy', pos.y.toFixed(2));
            const handleText = this._root.getElementById('target_handle_text');
            if (handleText) {
                handleText.setAttribute('x', pos.x.toFixed(2));
                handleText.setAttribute('y', pos.y.toFixed(2));
                handleText.textContent = fmtHandleText(snapValue);
            }
            dragMove(snapValue);
        };

        const onUp = async (ev) => {
            if (this._upInProgress) return;
            this._upInProgress = true;

            if (ev) {
                ev.stopPropagation();
                ev.preventDefault();
            }

            const dragPct = this._targetDragPct;
            const dragValue = this._targetDragValue;

            // S17 — try/finally: drag-release guards always clear.
            try {
                if (dragValue != null && (entityId || onCommit)) {
                    // QS-199 review-fix #03 S1/S2 — await an optional
                    // pre-commit hook BEFORE the commit (the pool selects
                    // default mode first; writing while not yet in default
                    // mode can be rejected backend-side). Both hooks are
                    // awaited so neither races the local-target timeout /
                    // next render.
                    if (onBeforeCommit) await onBeforeCommit(dragValue);
                    // QS-235 — `onCommit` (car: map value → option then
                    // `_select`) replaces the default `_setNumber` write.
                    if (onCommit) {
                        await onCommit(dragValue);
                    } else {
                        await this._setNumber(entityId, dragValue);
                    }
                    this._localTargetPct = dragPct;
                    if (this._pendingClearLocalTarget) clearTimeout(this._pendingClearLocalTarget);
                    this._pendingClearLocalTarget = setTimeout(() => {
                        this._localTargetPct = null;
                        this._pendingClearLocalTarget = null;
                        // QS-199 review-fix S14: don't render a detached card.
                        if (this.isConnected && this._root) this._render();
                    }, 5000);
                }
            } catch (_) {
                // swallow — HA state will resync on the next push
            } finally {
                this._targetDragPct = null;
                this._targetDragValue = null;
                this._isInteractingTarget = false;
                this._upInProgress = false;
                handle.style.cursor = 'grab';
            }
        };

        if (window.PointerEvent) {
            const onPointerMove = (ev) => onMove(ev);
            const onPointerUp = async (ev) => {
                try { handle.releasePointerCapture(ev.pointerId); } catch (_) { /* ignore */ }
                handle.removeEventListener('pointermove', onPointerMove);
                handle.removeEventListener('pointerup', onPointerUp);
                handle.removeEventListener('pointercancel', onPointerUp);
                await onUp(ev);
            };
            const onPointerDown = (ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                this._isInteractingTarget = true;
                try { handle.setPointerCapture(ev.pointerId); } catch (_) { /* ignore */ }
                handle.addEventListener('pointermove', onPointerMove);
                handle.addEventListener('pointerup', onPointerUp);
                handle.addEventListener('pointercancel', onPointerUp);
                handle.style.cursor = 'grabbing';
            };
            handle.addEventListener('pointerdown', onPointerDown);
        } else {
            // QS-199 review-fix N2 — declare `onUpLegacy` BEFORE `onDown`
            // so there's no forward reference (it worked via call-time
            // closure lookup, but reads cleaner declared first).
            const onUpLegacy = async (ev) => {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('touchmove', onMove);
                document.removeEventListener('mouseup', onUpLegacy);
                document.removeEventListener('touchend', onUpLegacy);
                await onUp(ev);
            };
            const onDown = (ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                this._isInteractingTarget = true;
                document.addEventListener('mousemove', onMove);
                document.addEventListener('touchmove', onMove, { passive: false });
                document.addEventListener('mouseup', onUpLegacy);
                document.addEventListener('touchend', onUpLegacy);
                handle.style.cursor = 'grabbing';
            };
            handle.addEventListener('mousedown', onDown);
            handle.addEventListener('touchstart', onDown, { passive: false });
        }
    }

    /*
      _wireTimePicker — opens hour/minute picker dialog via _showDialog,
      applies new finish-time via _setTime. S9 local-state-cleanup: clears
      _localFinishTimeMins after 5 seconds so out-of-band backend updates
      aren't masked indefinitely.

      params: {
          buttonEl,                — the time-btn DOM element
          entityId,                — service target for _setTime
          currentMins,             — int minutes-since-midnight to pre-select
          localStateKey,           — e.g. '_localFinishTimeMins'
          clearTimerKey,           — e.g. '_localFinishTimeClearTimer'
      }

      QS-235 — generalized so the car card can adopt its bespoke
      finish-time flow. NEW optional params default to the duration
      2-button dialog so a caller that passes none of them is
      byte-equivalent:
          onAfterCommit  — async () => …; AWAITED after `_setTime`
                           (car: `_press(schedule)`).
          resetButton    — {text, variant?, onClick}; inserted as a 3rd
                           dialog button between Cancel and Apply
                           (car: Reset → `_press(clean_constraints)`).
                           Only spread when both `text` and `onClick` are
                           present (NTH2 shape guard).
          title          — dialog title (default `'Finish Time'`).
          bodyText       — body prompt (default `'Select the time the
                           device should finish by:'`).
    */
    _wireTimePicker(params) {
        const {
            buttonEl, entityId, currentMins,
            localStateKey = '_localFinishTimeMins',
            clearTimerKey = '_localFinishTimeClearTimer',
            onAfterCommit, resetButton,
            title = 'Finish Time',
            bodyText = 'Select the time the device should finish by:',
        } = params;
        if (!buttonEl || !entityId) return;

        const timeAction = async () => {
            const defaultHour = Math.floor(currentMins / 60);
            const defaultMin = currentMins % 60;

            const customContent = `
            <p>${this._escapeHtml(bodyText)}</p>
            <div class="time-picker">
              <select id="dialog_hour_select">
                ${Array.from({ length: 24 }, (_, h) => `<option value="${h}" ${defaultHour === h ? 'selected' : ''}>${String(h).padStart(2, '0')}</option>`).join('')}
              </select>
              <span>:</span>
              <select id="dialog_minute_select">
                ${[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map((m) => `<option value="${m}" ${defaultMin === m ? 'selected' : ''}>${String(m).padStart(2, '0')}</option>`).join('')}
              </select>
            </div>
          `;

            const dialog = this._showDialog({
                title,
                customContent,
                buttons: [
                    { text: 'Cancel', variant: 'secondary' },
                    // QS-235 — optional reset button (car: clean_constraints).
                    // review-fix #01 NTH2 — only spread a well-shaped
                    // `{text, onClick}` so a malformed object can't render a
                    // silent broken button.
                    ...(resetButton && resetButton.text && resetButton.onClick ? [resetButton] : []),
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
                            const hm = this._formatHm(mins);
                            const val = hm + ':00';
                            this[localStateKey] = mins;
                            // S9: clear local override after 5s so out-of-band updates aren't masked indefinitely.
                            // QS-199 review-fix S14: guard the deferred
                            // _render against a card that was detached
                            // (dashboard rearranged) after Apply — writing
                            // to a torn-down shadow root throws.
                            if (this[clearTimerKey]) clearTimeout(this[clearTimerKey]);
                            this[clearTimerKey] = setTimeout(() => {
                                this[localStateKey] = null;
                                if (this.isConnected && this._root) this._render();
                            }, 5000);
                            await this._setTime(entityId, val);
                            // QS-235 — car presses `schedule` after the write.
                            if (onAfterCommit) await onAfterCommit();
                        },
                    },
                ],
            });
        };

        buttonEl.style.pointerEvents = 'auto';
        buttonEl.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); timeAction(); });
        buttonEl.addEventListener('touchend', (ev) => { ev.preventDefault(); timeAction(); });
        this._registerKeyActivation(buttonEl, timeAction);
    }

    /*
      _wireResetButton — confirmation dialog → _press(entityId).
      params: {buttonEl, entityId}

      QS-199 review-fix #03 N6 — intentionally does NOT call
      `_registerKeyActivation`: the reset control is a native
      `<button id="reset">` in every card (keyboard-native — Enter/Space
      already activate it), unlike the other controls which are custom
      `div`s. No `role`/`tabindex` needed.
    */
    _wireResetButton(params) {
        const { buttonEl, entityId } = params;
        if (!buttonEl || !entityId) return;

        const resetAction = async () => {
            this._showDialog({
                title: 'Reset device state',
                message: 'This will reset internal state for the device and cannot be undone.\nProceed?',
                buttons: [
                    { text: 'Cancel', variant: 'secondary' },
                    {
                        text: 'Reset', variant: 'danger',
                        onClick: async () => { await this._press(entityId); },
                    },
                ],
            });
        };
        buttonEl.addEventListener('click', (ev) => { ev.stopPropagation(); ev.preventDefault(); resetAction(); });
        buttonEl.addEventListener('touchend', (ev) => { ev.preventDefault(); resetAction(); });
    }

    /*
      _wirePowerButton — toggle switch via _turnOn/_turnOff.
      params: {buttonEl, swEntity, entityId}
        swEntity — the existing HA state entity (for current-state probe)
        entityId — the entity_id to invoke the service on
    */
    _wirePowerButton(params) {
        const { buttonEl, swEntity, entityId } = params;
        if (!buttonEl || !swEntity || !entityId) return;

        const togglePower = async () => {
            try {
                if (swEntity.state === 'on') {
                    await this._turnOff(entityId);
                    buttonEl.classList.remove('on');
                } else {
                    await this._turnOn(entityId);
                    buttonEl.classList.add('on');
                }
            } catch (_) {
                // ignore; HA state will resync UI on next render
            }
        };
        buttonEl.style.pointerEvents = 'auto';
        buttonEl.addEventListener('click', togglePower);
        buttonEl.addEventListener('touchend', (ev) => { ev.preventDefault(); togglePower(); });
        this._registerKeyActivation(buttonEl, togglePower);
    }

    /*
      _wireGreenButton — toggle the "green only" switch (solar-priority).
      params: {buttonEl, swEntity, entityId}
    */
    _wireGreenButton(params) {
        const { buttonEl, swEntity, entityId } = params;
        if (!buttonEl || !swEntity || !entityId) return;

        const toggleGreen = async () => {
            try {
                if (swEntity.state === 'on') {
                    await this._turnOff(entityId);
                    buttonEl.classList.remove('on');
                } else {
                    await this._turnOn(entityId);
                    buttonEl.classList.add('on');
                }
            } catch (_) {
                // ignore; HA state will resync UI on next render
            }
        };
        buttonEl.style.pointerEvents = 'auto';
        buttonEl.addEventListener('click', toggleGreen);
        buttonEl.addEventListener('touchend', (ev) => { ev.preventDefault(); toggleGreen(); });
        this._registerKeyActivation(buttonEl, toggleGreen);
    }

    // ----- Ring builder -----
    /*
      _buildRingHTML — returns an HTML string for the outer .ring SVG
      markup. Caller injects this string into innerHTML between any
      card-specific backdrop SVG and the .center / override-btn /
      green-btn DOM nodes.

      QS-235 — moved UP here from `QsRingDurationCardBase` so the car
      card (which extends `QsCardBase` directly) can consume it. The 5
      duration cards inherit it unchanged: every NEW param defaults to
      the duration-card behaviour, so a caller that passes none of them
      gets byte-equivalent output (the empty `extraDefs` adds only
      insignificant SVG whitespace).

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
        showAnimation                — boolean — whether to render the dash anim path
        canDragHandle                — boolean — whether to render handle circle
        gradGreenId, gradRunningId, activeGradId
        dashLen, gapLen              — stroke-dasharray
        pctToHours                   — function used in the DEFAULT label render

      QS-235 backward-compatible optional params (car overrides):
        handleLabel                  — handle TEXT label (default
                                       `this._fmt(pctToHours(handlePct))`)
        bgStroke                     — background-arc stroke (default
                                       `'var(--divider-color)'`)
        handleFontSize               — handle text font-size (default `13`)
        handleStroke                 — handle circle stroke (default `palette.primary`)
        handleFill                   — handle text fill (default `palette.primary`)
        animPathId                   — dash anim path id (default `'running_anim'`)
        animGradId                   — dash anim stroke gradient id
                                       (default `gradRunningId`; SF1 — the car
                                       passes a fault-aware gradient)
        extraDefs                    — extra `<defs>` children (default `''`)
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
            bgStroke = 'var(--divider-color)',
            handleFontSize = 13,
            animPathId = 'running_anim',
            // QS-235 review-fix #01 SF1 — the dash anim stroke gradient,
            // defaulting to `gradRunningId` (so the duration callers are
            // byte-identical). The car passes a fault-aware gradient so
            // the moving dash matches its static fault arc.
            animGradId = gradRunningId,
            extraDefs = '',
        } = params;

        // QS-199 review-fix #07 N1 — the handle's TEXT LABEL round-trips
        // the target back from `handlePct`: `pctToHours(handlePct)`. On the
        // non-default branch each card sets
        // `handlePct = hoursToPct(displayTargetHours)` with
        // `displayTargetHours = targetHours` left UNCLAMPED on purpose, so
        // this label shows the TRUE target even when it exceeds `maxHours`.
        // Only the handle POSITION is clamped (the card's `pctToDeg`
        // clamps to [0,100] → the handle pins at the ring top for an
        // out-of-range target while the number stays correct).
        // DO NOT "fix" the >100% `handlePct` by clamping `handlePct` or
        // `displayTargetHours` — that would make this label render
        // `maxHours` instead of the real value (a regression across all
        // duration cards). The pin-at-top-but-number-correct behaviour is
        // intentional and graceful.
        // QS-235 — `handleLabel` lets the car supply its own
        // energy/percent TRUE-target string (still unclamped); the
        // duration default reproduces the round-trip above byte-for-byte.
        // review-fix #01 NTH3 — named `handleLabelText` (NOT
        // `escaped…`): a supplied `handleLabel` is used verbatim, with no
        // escaping (SVG <text> content renders metacharacters as text, so
        // this is safe; the old name wrongly implied sanitization).
        const handleLabelText = params.handleLabel != null
            ? params.handleLabel
            : this._fmt(pctToHours(handlePct));
        // QS-235 — handle stroke/fill default to `palette.primary` (the
        // duration behaviour); the car passes connection-state colours.
        const handleStroke = params.handleStroke != null ? params.handleStroke : palette.primary;
        const handleFill = params.handleFill != null ? params.handleFill : palette.primary;

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
                ${extraDefs}
              </defs>
              <path d="${bgPath}" stroke="${bgStroke}" stroke-width="14" fill="none" stroke-linecap="round" />
              <path d="${progressPath}" stroke="url(#${activeGradId})" stroke-width="14" fill="none" stroke-linecap="round" ${showAnimation ? 'stroke-opacity="0.35"' : ''} />
              ${showAnimation ? `
              <path id="${animPathId}"
                    d="${progressPath}"
                    stroke="url(#${animGradId})"
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
              <circle id="target_handle" cx="${handlePos.x.toFixed(2)}" cy="${handlePos.y.toFixed(2)}" r="13" fill="var(--card-background-color)" stroke="${handleStroke}" stroke-width="3" style="cursor: grab; pointer-events: all;" />
              <text id="target_handle_text" x="${handlePos.x.toFixed(2)}" y="${handlePos.y.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" fill="${handleFill}" font-size="${handleFontSize}" font-weight="800" style="cursor: grab; pointer-events: none; user-select: none;">${handleLabelText}</text>
              ` : ''}
        `;
    }

    // ----- Ring bottom-center carve cover -----
    /*
      _ringCarveCover({cx, cy, r, id, show}) — QS-235. Returns a single
      `<circle>` cover overlay string (or '' when `show` is falsy) that
      erases a clean circular patch of the inside-disc animation so a
      button drawn there reads clearly. Replaces the per-card inline
      `<circle id="…_cover" fill="var(--card-background-color)"
      pointer-events="none" />` literals (QS-217 + QS-232). The shared
      `RING_BOTTOM_CARVE_*` constants supply the bottom-center geometry;
      the car passes its own car-local geometry for the rabbit/time
      covers.
    */
    _ringCarveCover(params) {
        const { cx, cy, r, id, show } = params;
        if (!show) return '';
        return `<circle id="${id}" cx="${cx}" cy="${cy}" r="${r}" fill="var(--card-background-color)" pointer-events="none" />`;
    }
}
