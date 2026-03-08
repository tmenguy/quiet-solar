# 2026-03-08: Mobile touch fixes, override layout, and override start time bug

This document covers all fixes applied on 2026-03-08.

---

## 1. Mobile touch: ring handle drag scrolls the page instead of dragging

On mobile (iOS Safari, HA Companion WebView), touching the circular drag handle on any card
would scroll the entire page. The existing CSS rule `#target_handle { touch-action: none; }`
targeted an SVG `<circle>` element, which mobile browsers do not honor because SVG child
elements are not HTML elements participating in the CSS box model.

### Changes (all 4 card JS files)

- Added CSS rule `.ring svg { touch-action: none; }` targeting the `<svg>` HTML element.
- Added inline `style="touch-action: none;"` on the `<svg>` element in the HTML template.
- Kept the existing `#target_handle` CSS rule for defense-in-depth.

---

## 2. Mobile touch: buttons require multiple taps

On mobile, buttons (solar priority, green only, rabbit/force, time, reset, override, power)
required multiple tap attempts. Two factors: (A) the full `innerHTML` re-render on every
`set hass()` can destroy DOM nodes during the 300ms touch-to-click synthesis window, and
(B) interactive ring buttons lacked `touch-action: manipulation` to remove that delay.

### Changes (all 4 card JS files)

- Added `touch-action: manipulation; -webkit-tap-highlight-color: transparent;` to all
  interactive ring button CSS classes (`.sun-btn`, `.rabbit-btn`, `.time-btn`, `.green-btn`,
  `.power-btn`, `.override-btn`).
- Added dual `click` + `touchend` event handlers on every interactive button. The `touchend`
  fires immediately and calls `preventDefault()` to suppress the delayed synthetic `click`.
- Added code comments in all 4 cards explaining the mobile touch rationale.

---

## 3. Override button overlapping time button in End+Duration mode

When a climate or on-off-duration card was in "Default: End+Duration" mode with an active
override, the override hand icon overlapped the "Change Finish Time" button because too
many elements stacked vertically inside the 300px ring.

### Changes (`qs-climate-card.js` and `qs-on-off-duration-card.js`)

- **Hidden "Change Finish Time" during override.** Changed the template and JS handler
  condition from `isDefaultMode && sDefaultOnFinishTime` to
  `isDefaultMode && !isOverridden && sDefaultOnFinishTime`. The user cannot change the
  finish time during override anyway.
- **Show actual From/To times during override in default mode.** Changed the From/To value
  logic so that when `isDefaultMode && isOverridden`, both display the actual entity values
  (`startTime`, `endTime`) from the override constraint instead of hardcoded `--:--` or the
  static default finish time.

---

## 4. Override From time shows "--:--" (backend bug)

When overriding a bistate device in default mode, "From:" always showed `--:--`.

In `bistate_duration.py`, the override constraint was created with `start_time=time` but
the `LoadConstraint` constructor expects `start_of_constraint`. The wrong parameter name
was silently absorbed by `**kwargs` and discarded, causing `start_of_constraint` to default
to `DATETIME_MIN_UTC`, which the sensor renders as `"--:--"`.

### Change (`bistate_duration.py`)

- Renamed `start_time=time` to `start_of_constraint=time` on the override constraint
  creation (line 328).

---

## 5. Minor fixes

- Changed comment `// Convert to [0,360)` to `// Convert to [0,360]` in `qs-car-card.js`
  to avoid a closing parenthesis in a comment confusing naive syntax checkers.
- Created symlink `venv -> venv314` in the project root.

---

## Files changed

| File | Fixes |
| ---- | ----- |
| `custom_components/quiet_solar/ui/resources/qs-car-card.js` | 1, 2, 5 |
| `custom_components/quiet_solar/ui/resources/qs-climate-card.js` | 1, 2, 3 |
| `custom_components/quiet_solar/ui/resources/qs-on-off-duration-card.js` | 1, 2, 3 |
| `custom_components/quiet_solar/ui/resources/qs-pool-card.js` | 1, 2 |
| `custom_components/quiet_solar/ha_model/bistate_duration.py` | 4 |

## Test results

All 3223 tests passed after all changes.
