"""Tests for dashboard template rendering using real device objects.

These tests catch bugs where:
- Entity descriptions use translation_key with no matching entry in strings.json,
  causing ha_entity.name to return UNDEFINED and crashing the Jinja2 template.
- Dashboard templates reference entity keys that don't exist or produce invalid YAML.
- The LOAD_TYPE_DASHBOARD_DEFAULT_SECTION mapping fails to assign devices to sections.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from homeassistant.helpers.template import Template
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    CONF_POOL_TEMPERATURE_SENSOR,
    CONF_POWER,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    CONF_SWITCH,
    CONF_WATER_BOILER_TEMPERATURE_SENSOR,
    DASHBOARD_DEFAULT_SECTIONS,
    DEVICE_TYPE,
    DOMAIN,
    LOAD_TYPE_DASHBOARD_DEFAULT_SECTION,
    SOLCAST_SOLAR_DOMAIN,
    CONF_TYPE_NAME_QSPool,
)
from tests.ha_tests.const import (
    MOCK_BATTERY_CONFIG,
    MOCK_CAR_CONFIG,
    MOCK_CHARGER_CONFIG,
    MOCK_CLIMATE_DURATION_CONFIG,
    MOCK_HEAT_PUMP_CONFIG,
    MOCK_HOME_CONFIG,
    MOCK_ON_OFF_DURATION_CONFIG,
    MOCK_PERSON_CONFIG,
    MOCK_RADIATOR_CLIMATE_CONFIG,
    MOCK_RADIATOR_SWITCH_CONFIG,
    MOCK_SENSOR_STATES,
    MOCK_SOLAR_CONFIG,
    MOCK_WATER_BOILER_CONFIG,
    MOCK_WATER_BOILER_CONFIG_NO_TEMP,
)

COMPONENT_ROOT = Path(__file__).parent.parent / "custom_components" / "quiet_solar"


def _strip_js_comments(source: str) -> str:
    """Strip ``/* ... */`` block comments and ``// ...`` line comments
    from a JS source string.

    **Review-fix #02 N2 — known limitation:** the ``//`` line-comment
    pattern eats anything from ``//`` to end-of-line, INCLUDING ``//``
    inside string literals — most notably URL literals like
    ``"http://www.w3.org/2000/svg"`` become ``"http:`` after stripping.
    No current test asserts an http URL literal, but any future regex
    that does (e.g. pinning ``createElementNS('http://...')``) must
    operate on the pre-stripped ``source`` directly, not the stripped
    output of this helper.
    """
    import re

    no_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    no_line = re.sub(r"//[^\n]*", "", no_block)
    return no_line


def _extract_js_function_body(source: str, signature_regex: str) -> str | None:
    """Return the body between matching braces for a JS function whose
    signature matches `signature_regex`. Walks balanced braces so nested
    object literals, arrow function bodies, and most template-literal
    interpolations are handled cleanly. Returns ``None`` if the
    signature isn't found or the braces don't balance.

    This avoids the brittle ``[^{}]*?`` pattern used by earlier
    per-card body checks — that pattern silently mis-matches the moment
    any nested brace lands in the body, producing misleading "X must
    do Y" failures even when the code is correct.

    **Review-fix #02 N1 — known limitation:** the walker doesn't
    understand string-literal quoting. A ``const s = "{ ... }"`` or
    ```const s = `${...}` ``` containing a stray ``{`` / ``}`` inside
    the quoted span will mis-count the balanced-brace depth. Today's
    `connectedCallback` body is plain code with no string literals, so
    the walker is safe — but a future addition of a template literal
    with stray braces inside its quoted span could throw the walker
    off. If that happens, harden the walker to skip braces inside
    ``'…'`` / ``"…"`` / `` `…` `` spans (respecting escapes and
    ``${…}`` nesting) or extract the body manually for that test.
    """
    import re

    m = re.search(signature_regex, source)
    if m is None:
        return None
    brace_idx = source.find("{", m.end() - 1)
    if brace_idx == -1:
        return None
    depth = 1
    i = brace_idx + 1
    while i < len(source) and depth > 0:
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return source[brace_idx + 1 : i - 1]


# Pool config (not in ha_tests/const.py)
MOCK_POOL_CONFIG = {
    "name": "Test Pool",
    DEVICE_TYPE: CONF_TYPE_NAME_QSPool,
    CONF_SWITCH: "switch.test_pool_pump",
    CONF_POWER: 1100,
    CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temperature",
}

# Solar config with providers (so dynamic entities are created)
MOCK_SOLAR_WITH_PROVIDERS_CONFIG = {
    **MOCK_SOLAR_CONFIG,
    CONF_SOLAR_FORECAST_PROVIDERS: [
        {
            CONF_SOLAR_PROVIDER_DOMAIN: SOLCAST_SOLAR_DOMAIN,
            CONF_SOLAR_PROVIDER_NAME: "Solcast",
        },
    ],
}

# Extra mock states needed for pool + water boiler
EXTRA_MOCK_STATES = {
    "switch.test_pool_pump": {"state": "off", "attributes": {}},
    "sensor.pool_temperature": {
        "state": "22",
        "attributes": {"unit_of_measurement": "°C"},
    },
    "switch.test_water_boiler": {"state": "off", "attributes": {}},
    "sensor.test_water_boiler_temperature": {
        "state": "55",
        "attributes": {"unit_of_measurement": "°C"},
    },
    "switch.test_water_boiler_no_temp": {"state": "off", "attributes": {}},
}


def _apply_all_mock_states(hass) -> None:
    """Seed HA state machine with all mock sensor states."""
    for entity_id, data in {**MOCK_SENSOR_STATES, **EXTRA_MOCK_STATES}.items():
        hass.states.async_set(entity_id, data["state"], data.get("attributes", {}))


def _load_strings_json() -> dict:
    """Load strings.json and return the entity translation keys by platform."""
    with open(COMPONENT_ROOT / "strings.json", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entity", {})


# ============================================================================
# Fixture: Full home with all device types
# ============================================================================


@pytest.fixture
async def full_dashboard_home(hass):
    """Set up home with all device types for dashboard rendering tests.

    Creates real QSHome, QSSolar, QSCar, QSCharger, QSPerson, QSBattery,
    QSOnOffDuration, QSClimateDuration, QSHeatPump, and QSPool devices
    through HA config entries. All entity platforms are set up, so
    device.ha_entities is populated with real registered entities.
    """
    _apply_all_mock_states(hass)

    # Device configs in setup order (home must be first)
    device_configs = [
        ("home", MOCK_HOME_CONFIG),
        ("solar", MOCK_SOLAR_WITH_PROVIDERS_CONFIG),
        ("charger", MOCK_CHARGER_CONFIG),
        ("car", MOCK_CAR_CONFIG),
        ("person", MOCK_PERSON_CONFIG),
        ("battery", MOCK_BATTERY_CONFIG),
        ("on_off", MOCK_ON_OFF_DURATION_CONFIG),
        ("climate", MOCK_CLIMATE_DURATION_CONFIG),
        ("heat_pump", MOCK_HEAT_PUMP_CONFIG),
        ("pool", MOCK_POOL_CONFIG),
        ("radiator_switch", MOCK_RADIATOR_SWITCH_CONFIG),
        ("radiator_climate", MOCK_RADIATOR_CLIMATE_CONFIG),
        ("water_boiler", MOCK_WATER_BOILER_CONFIG),
    ]

    entries = {}
    with patch(
        "custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers",
        new_callable=AsyncMock,
    ):
        for label, config in device_configs:
            uid = uuid.uuid4().hex[:8]
            entry = MockConfigEntry(
                domain=DOMAIN,
                data={**config},
                entry_id=f"dashboard_test_{label}_{uid}",
                title=f"{label}: Test {label.replace('_', ' ').title()}",
                unique_id=f"dashboard_test_{label}_{uid}",
            )
            entry.add_to_hass(hass)
            result = await hass.config_entries.async_setup(entry.entry_id)
            await hass.async_block_till_done()
            assert result is True, f"Failed to set up {label} config entry"
            entries[label] = entry

    home = hass.data[DOMAIN].get(entries["home"].entry_id)
    assert home is not None, "Home device not found after setup"
    return home


# ============================================================================
# Test: Dashboard template renders without errors for all device types
# ============================================================================


class TestDashboardTemplateRendering:
    """Verify that dashboard templates render without crashes using real objects."""

    @pytest.mark.asyncio
    async def test_custom_template_renders_all_device_types(self, hass, full_dashboard_home):
        """Custom card template renders with all real device types present."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        parsed = yaml.safe_load(rendered)
        assert parsed is not None
        assert "views" in parsed

    @pytest.mark.asyncio
    async def test_standard_template_renders_all_device_types(self, hass, full_dashboard_home):
        """Standard HA template renders with all real device types present."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template_standard_ha.yaml.j2"
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        parsed = yaml.safe_load(rendered)
        assert parsed is not None
        assert "views" in parsed

    @pytest.mark.asyncio
    async def test_radiator_devices_use_dedicated_card(self, hass, full_dashboard_home):
        """AC-14 — radiator devices dispatch to `custom:qs-radiator-card`.

        Both switch-backed and climate-backed radiators must render with
        the dedicated card type, never the on/off or climate card.
        """
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        # CR2 — offload file I/O so the event loop doesn't block.
        template_content = await hass.async_add_executor_job(template_path.read_text)

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        # N14 — the fixture installs TWO radiators (switch + climate);
        # both must dispatch to the dedicated card. `>= 2` rather than
        # `>= 1` so a regression that drops one of them is caught.
        assert rendered.count("custom:qs-radiator-card") >= 2
        # Sanity: the section name surfaces in the rendered YAML.
        assert "radiators" in rendered

    def test_radiator_card_resource_file_present(self):
        """AC-11 — `qs-radiator-card.js` ships in the resources directory.

        CR2 — plain `def` (no `async`) because this test does pure
        file I/O + string analysis. Sync tests avoid blocking the
        event loop on `Path.read_text()`.
        """
        card_path = COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js"
        assert card_path.is_file(), f"Missing radiator card resource: {card_path}"

        content = card_path.read_text()
        # Custom-element registration must reference the renamed element.
        assert "customElements.define('qs-radiator-card'" in content
        # Class identifier must be renamed (no stale `QsOnOffDurationCard`).
        assert "QsRadiatorCard" in content
        assert "QsOnOffDurationCard" not in content
        # N10 — `customCards` registration uses the user-facing card name.
        assert "'QS Radiator Card'" in content
        # Stub config also reads the renamed display name.
        assert "QS Radiator" in content

    def test_radiator_card_s14_safe_number_guards_against_nan(self):  # CR2 — sync (no hass)
        """A2 — pin S14 via regex, not a bare substring.

        We assert that:
          1. A `_safeNumber` helper is present (the wrapper around
             `Number()` that returns the fallback on degenerate input).
          2. The helper gates against NaN (`Number.isNaN` OR
             `Number.isFinite` — both are acceptable defensive styles).
          3. NO raw `Number(<expr> || <fallback>)` pattern lurks in the
             card's executable code — that's exactly the NaN footgun
             S14 fixed. (Inline `//` comments are stripped before the
             regex scan so the test prose isn't false-positive.)
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # `_safeNumber` is the canonical helper (either as a closure
        # local or a class method via `this._safeNumber(...)`).
        assert "_safeNumber" in content
        # The helper gates against NaN. Accept either positive
        # (`Number.isFinite`) or negative (`Number.isNaN`) form.
        assert "Number.isFinite" in content or "Number.isNaN" in content

        # Strip `//` line comments AND `/* ... */` block comments before
        # the regex scan so we only match executable code.
        executable = _strip_js_comments(content)

        # Raw `Number(state || fallback)` is the anti-pattern.
        raw_pattern = re.compile(r"Number\(\s*[^()]+\|\|[^()]+\)")
        assert raw_pattern.search(executable) is None, (
            "Found a raw `Number(... || ...)` pattern in executable code — should use `_safeNumber(...)`"
        )

    def test_radiator_card_s15_escape_html_before_innerHTML(self):  # CR2 — sync (no hass)
        """A2 — pin S15 via regex: every `_safeNumber` is fine; every
        `innerHTML = ` write must reference an escaped value or constant
        template. Helper may be invoked as a closure-local `_escapeHtml`
        or as a class method `this._escapeHtml`.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # The escape helper is defined.
        assert "_escapeHtml" in content
        # The card-title interpolation (a primary injection vector) is
        # routed through `_escapeHtml` (closure-local or method form).
        assert re.search(r'class="card-title">\s*\$\{(?:this\.)?_escapeHtml\(title\)\}', content) is not None, (
            "Card title is not escaped before innerHTML"
        )
        # Mode labels in the bistate-mode select go through `_escapeHtml`.
        assert re.search(r"\$\{(?:this\.)?_escapeHtml\(translateBistateMode\(o\)\)\}", content) is not None, (
            "Bistate mode labels are not escaped before innerHTML"
        )

    def test_radiator_card_s16_keyboard_accessibility(self):  # CR2 — sync (no hass)
        """A2 — pin S16 via regex: each primary action div has
        `role="button"`, `tabindex="0"`, AND a keyboard activation hook.

        QS-199 — the `role/tabindex` attributes remain in the card's HTML
        template; the `_registerKeyActivation` helper moved to
        `shared/qs-card-base.js` and is now called from each wire-helper
        inside `shared/qs-card-base.js` and `shared/qs-ring-duration-base.js`.
        Uses `card_source_union` for the count assertion (which now spans
        the union of the card + every shared module).
        """
        import re

        from tests.utils.card_sources import card_source_union

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # Every action `<div>` carries the role/tabindex attribute pair.
        for button_id in ("power_btn", "green_btn", "override_btn", "time_btn"):
            pattern = re.compile(rf'id="{button_id}"[^>]*role="button"[^>]*tabindex="0"', re.DOTALL)
            assert pattern.search(content) is not None, f"Missing role/tabindex on `{button_id}`"

        # The keyboard helper exists across the card + shared modules,
        # and is wired into every action via the shared wire-helpers.
        union = card_source_union("qs-radiator-card.js")
        assert "_registerKeyActivation(" in union
        # QS-199 review-fix #02 S12 — count CALL-SITES only. The method
        # DEFINITION `_registerKeyActivation(el, action) {` in
        # qs-card-base.js would otherwise inflate the count, letting the
        # test pass with fewer wired actions than expected. `this.` and
        # bare-call forms are call-sites; the `(el, action)` signature is
        # the definition.
        call_sites = re.findall(r"_registerKeyActivation\(", union)
        definitions = re.findall(r"_registerKeyActivation\(el,\s*action\)", union)
        wired = len(call_sites) - len(definitions)
        assert wired >= 4, (
            f"expected ≥4 _registerKeyActivation call-sites across the union, "
            f"got {wired} (calls={len(call_sites)}, defs={len(definitions)})"
        )

    def test_radiator_card_s17_async_calls_wrapped_in_try_finally(self):  # CR2 — sync (no hass)
        """A2 — pin S17: each `_select` / `_setNumber` await is wrapped.

        Verifies that for both `await this._select(...)` and
        `await this._setNumber(...)`:
          - A `try {` opens within the preceding ~30 lines (i.e. the
            await is inside a try block, not bare).
          - A `finally {` clause appears within the next ~30 lines
            after the await (the cleanup-on-failure path).

        Without this, a transient service-call failure would leak
        interaction guards and the card would get stuck.

        QS-199 — both awaits moved to shared modules
        (`shared/qs-ring-duration-base.js` for `_select`,
        `shared/qs-card-base.js` for `_setNumber`). Uses
        `card_source_union` so the assertion still finds them.
        """
        from tests.utils.card_sources import card_source_union

        content = card_source_union("qs-radiator-card.js")

        def _try_finally_brackets(call_signature: str) -> bool:
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if call_signature in line:
                    # Look backwards up to 30 lines for `try {`.
                    has_try_before = any(
                        "try {" in lines[j] or "try{" in lines[j] for j in range(max(0, idx - 30), idx)
                    )
                    # Look forwards up to 30 lines for `} finally {`.
                    has_finally_after = any("finally" in lines[j] for j in range(idx, min(len(lines), idx + 30)))
                    if has_try_before and has_finally_after:
                        return True
            return False

        assert _try_finally_brackets("await this._select("), (
            "`await this._select(...)` is not wrapped in try / finally — interaction guards may leak"
        )
        assert _try_finally_brackets("await this._setNumber("), (
            "`await this._setNumber(...)` is not wrapped in try / finally — interaction guards may leak"
        )

    def test_radiator_card_b5_show_dialog_escapes_interpolations(self):  # CR2 — sync (no hass)
        """AA1 — pin B5: every interpolation inside `_showDialog` is escaped.

        Verifies that the showDialog implementation body does NOT contain
        raw `${title}` / `${message}` interpolations against innerHTML
        — they must all route through `_escapeHtml(...)`. Without this
        a future entity-derived title/message would silently
        reintroduce the S15 injection vector.

        QS-199 — the closure `const showDialog = (opts) => {...}` was
        lifted to a method `_showDialog(opts) {...}` on `QsCardBase`
        (in `shared/qs-card-base.js`). Uses `card_source_union` and
        extracts the new method body via brace-counting from the
        method-declaration token (after stripping JS comments to avoid
        matching the doc-string examples).
        """
        import re

        from tests.utils.card_sources import card_source_union

        raw = card_source_union("qs-radiator-card.js")
        content = _strip_js_comments(raw)

        # Locate either the legacy closure (`const showDialog = ...`)
        # or the post-QS-199 method form (`_showDialog(opts) {`).
        m = re.search(r"_showDialog\s*\(", content)
        legacy = content.find("const showDialog")
        assert m is not None or legacy != -1, (
            "Missing `_showDialog(...)` method (or legacy `const showDialog`) declaration"
        )
        start = m.start() if m is not None else legacy
        # Find the opening `{` after the parameter list.
        body_start = content.find("{", start)
        assert body_start != -1
        # Brace-count to the matching `}`.
        depth = 0
        body_end = -1
        for i in range(body_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    body_end = i
                    break
        assert body_end != -1, "Could not find closing brace of `_showDialog`"

        body = content[body_start : body_end + 1]

        # Both `${title}` and `${message}` interpolations must go through
        # `_escapeHtml`. Accept either the closure-local form
        # `${_escapeHtml(...)}` or the class-method form
        # `${this._escapeHtml(...)}` — both styles exist across the
        # bundled cards.
        raw_title = re.search(r"\$\{title\}", body)
        raw_message = re.search(r"\$\{message\}", body)
        escaped_title = re.search(r"\$\{(?:this\.)?_escapeHtml\(title\)\}", body)
        escaped_message = re.search(r"\$\{(?:this\.)?_escapeHtml\(message\)\}", body)

        assert raw_title is None, "`${title}` is not escaped in showDialog"
        assert raw_message is None, "`${message}` is not escaped in showDialog"
        assert escaped_title is not None, "`_escapeHtml(title)` missing in showDialog"
        assert escaped_message is not None, "`_escapeHtml(message)` missing in showDialog"

    def test_radiator_card_n7_running_includes_live_backing_state(self):  # CR2 — sync (no hass)
        """A2 — pin N7: the `running` derivation OR-s in the backing state.

        Cold-start fallback so a freshly-restarted integration doesn't
        show an actively-heating radiator as off.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # The card reads the backing entity id from `e.backing_entity`.
        assert "e.backing_entity" in content
        # The `running` constant must be an OR over command state + live backing.
        # We allow surrounding whitespace and `liveBackingOn`/similar locals.
        pattern = re.compile(r"const\s+running\s*=\s*commandReportsOn\s*\|\|\s*liveBackingOn")
        assert pattern.search(content) is not None, (
            "Cold-start `running` fallback is missing — radiator may show as off during the cold-start grace window"
        )

    def test_radiator_card_bh10_configured_hvac_on_compared(self):  # CR2 — sync (no hass)
        """BH10 — `liveBackingOn` compares against the configured HVAC mode
        (e.g. `auto`), not just the hard-coded `"heat"`.

        Without this, a user who set `CONF_CLIMATE_HVAC_MODE_ON = "auto"`
        sees `running=false` during the cold-start grace window even
        though the climate entity reports `state="auto"`.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # The card reads the configured HVAC mode from
        # `e.climate_hvac_mode_on` with a `"heat"` default for backward
        # compatibility.
        assert "e.climate_hvac_mode_on" in content
        # The comparison against `configuredHvacOn` MUST be present.
        pattern = re.compile(r"liveBackingState\s*===\s*configuredHvacOn")
        assert pattern.search(content) is not None, (
            "Cold-start `running` derivation no longer compares against the "
            "configured HVAC ON mode — non-default HVAC modes won't be recognised"
        )

    def test_radiator_card_heat_palette(self):  # CR2 — sync (no hass)
        """QS-201 AC-1 — heat palette is applied verbatim, cool palette gone."""
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()
        # Strip comments first.
        no_block = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        no_comments = re.sub(r"//[^\n]*", "", no_block)

        # Required heat-palette literals (all five keys).
        for literal in ("'#FF5722'", "'#D32F2F'", "'#FF6E40'", "'#E64A19'"):
            assert literal in no_comments, f"Missing heat-palette literal {literal}"

        # The `colors` const must use these literals (not just appear somewhere).
        assert re.search(r"const\s+colors\s*=\s*\{[^}]*primary:\s*'#FF5722'", no_comments) is not None

        # Cool-palette literals must NOT appear in executable code.
        for forbidden in ("'#2196F3'", "'#00bcd4'", "'#8bc34a'", "'#00e1ff'", "'#0066ff'"):
            assert forbidden not in no_comments, f"Stale cool-palette literal {forbidden} still present"

    def test_radiator_card_flame_layers_present(self):  # CR2 — sync (no hass)
        """QS-201 AC-2 + AC-7 — flame paths, circular clip, cache-clear whitelist.

        QS-204 review-fix #03 H3 parameterised the per-layer loops by
        ``LAYER_TEETH_COUNTS.length``, so the three flame ``<path>``
        elements are now emitted via a ``map(... id="flame${i}" ...)``
        template-literal rather than hard-coded ``id="flame0/1/2"``
        attributes. The test asserts the dynamic pattern is present
        (plus the constants-level confirmation that the layer count is
        still 3).
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # AC-2: the dynamic template emits per-layer paths whose ids
        # are computed from the array index — `id="flame${i}"`.
        assert 'id="flame${i}"' in content, (
            'Missing dynamic `id="flame${i}"` template in the flame '
            "<path> emission — H3 parameterised the layer loops but "
            "the template must still emit per-layer path ids."
        )
        # Constants confirm the layer count is still 3 (the assertion
        # would still pass for any forward-compatible extension to N).
        assert (
            re.search(
                r"const\s+LAYER_TEETH_COUNTS\s*=\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
                content,
            )
            is not None
        ), (
            "LAYER_TEETH_COUNTS must declare exactly 3 entries (current "
            "layer count); extending the array is supported but a smoke-"
            "test bump should follow."
        )

        # AC-2 / QS-217 AC-6: clipPath uses `<path clip-rule="evenodd"
        # d="${clipPathD}" />`. The bare `<circle>` form was replaced
        # by QS-217 to enable the override-button hole — review-fix
        # #03 simplifies clipPathD back to just an outer-disc circle
        # path (the lens-shape carve was dropped in favour of a cover
        # overlay drawn ON TOP of the animation; see the
        # test_radiator_card_override_btn_carve_out test for the cover
        # invariants).
        assert (
            re.search(
                r"<clipPath\s+id=\"\$\{flameClipId\}\">\s*<path\s+clip-rule=\"evenodd\"\s+d=\"\$\{clipPathD\}\"\s*/>\s*</clipPath>",
                content,
                re.DOTALL,
            )
            is not None
        ), 'Missing clipPath <path clip-rule="evenodd" d="${clipPathD}" /> form.'
        # The clipPathD builder must still reference `CENTER_CY` and
        # `CLIP_R` (the outer-disc geometry). QS-217 review-fix #03
        # drops the carve+cancel constants (no longer in the builder).
        builder_match = re.search(
            r"const\s+clipPathD\s*=([\s\S]*?);",
            content,
        )
        assert builder_match is not None, (
            "Missing `const clipPathD = …;` builder declaration — must be present above the innerHTML template literal."
        )
        builder_block = builder_match.group(1)
        for ident in ("CENTER_CY", "CLIP_R"):
            assert re.search(rf"\b{ident}\b", builder_block) is not None, (
                f"qs-radiator-card.js: clipPathD builder must reference `{ident}` by name (not hard-coded)."
            )
        # The constants themselves carry the correct geometric values.
        assert re.search(r"const\s+CENTER_CY\s*=\s*160\b", content) is not None
        assert re.search(r"const\s+CLIP_R\s*=\s*120\b", content) is not None

        # AC-2: <g clip-path="url(#${flameClipId})"> wraps the three paths.
        assert 'clip-path="url(#${flameClipId})"' in content

        # AC-7: _invalidateFlameCache clears the card's DOM ref cache and
        # delegates the memo-key reset to the engine.
        # QS-199 review-fix #03 N5 — the card no longer mirrors
        # `_lastFlameBaseY` / `_lastFlameAmp` (dead state owned by the
        # engine). The method now clears `_flameEls` and calls
        # `this._flameEngine.invalidate()`.
        inv_match = re.search(
            r"_invalidateFlameCache\s*\(\s*\)\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        assert inv_match is not None, "Missing _invalidateFlameCache method"
        body = inv_match.group(1)
        assert "_flameEls" in body, "_invalidateFlameCache must clear `this._flameEls`"
        assert "_flameEngine.invalidate()" in body, (
            "_invalidateFlameCache must delegate memo-key reset to "
            "`this._flameEngine.invalidate()` (N5: engine owns the memo keys)"
        )
        # Forbidden: animation state that must SURVIVE disconnect, AND the
        # removed dead card mirrors (N5).
        for forbidden in ("_currentFlameAmp", "_currentFlameSpeed", "_flamePhase", "_lastFlameBaseY", "_lastFlameAmp"):
            assert forbidden not in body, (
                f"_invalidateFlameCache must NOT touch `this.{forbidden}` "
                f"(animation state survives disconnect; dead mirrors removed in N5)"
            )

        # AC-7: disconnectedCallback calls _invalidateFlameCache.
        disc_match = re.search(
            r"disconnectedCallback\s*\(\s*\)\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        assert disc_match is not None
        assert "_invalidateFlameCache" in disc_match.group(1)

    def test_radiator_card_flame_height_envelope_uses_progress(self):  # CR2 — sync (no hass)
        """QS-201 AC-3 + AC-8 — flame base tracks progress; gate split into named sub-conditions."""
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()
        no_block = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        no_comments = re.sub(r"//[^\n]*", "", no_block)

        # AC-3: progressRatio is clamped against maxHours.
        assert (
            re.search(
                r"progressRatio\s*=\s*maxHours\s*>\s*0\s*\?\s*Math\.max\(\s*0\s*,\s*Math\.min\(\s*1\s*,\s*hoursRun\s*/\s*maxHours",
                no_comments,
            )
            is not None
        ), "Missing progressRatio clamp"

        # AC-3: flameBaseY formula uses the 1/5..4/5 envelope.
        assert "FLAME_BASE_MIN_PCT" in no_comments
        assert "FLAME_BASE_MAX_PCT" in no_comments
        assert (
            re.search(
                r"flameBaseY\s*=\s*CENTER_CY\s*\+\s*CLIP_R\s*-\s*\(\s*FLAME_BASE_MIN_PCT\s*\+\s*progressRatio\s*\*\s*\(\s*FLAME_BASE_MAX_PCT\s*-\s*FLAME_BASE_MIN_PCT\s*\)\s*\)\s*\*\s*2\s*\*\s*CLIP_R",
                no_comments,
            )
            is not None
        ), "flameBaseY formula does not mirror pool's water-level envelope"

        # AC-8: gate split.
        assert re.search(r"const\s+ringDashActive\s*=\s*running\s*&&\s*segLen\s*>\s*6", no_comments) is not None
        assert re.search(r"const\s+fireActive\s*=\s*running", no_comments) is not None
        assert re.search(r"const\s+showAnimation\s*=\s*ringDashActive\s*\|\|\s*fireActive", no_comments) is not None

        # AC-8: <path id="running_anim"> emission is gated on ringDashActive,
        # not the umbrella showAnimation.
        #
        # QS-199 — the running_anim path is now emitted by the shared
        # `_buildRingHTML` template (in shared/qs-ring-duration-base.js)
        # which gates on its `showAnimation` parameter. The radiator
        # card MUST pass `showAnimation: ringDashActive` (NOT the OR
        # umbrella `showAnimation`) so the dash animation only renders
        # when there's enough progress to dash through. Accepts either
        # the legacy literal template form OR the new call-site form.
        legacy_form = re.search(
            r"\$\{\s*ringDashActive\s*\?\s*`[^`]*?id=\"running_anim\"",
            content,
            re.DOTALL,
        )
        call_form = re.search(
            r"showAnimation:\s*ringDashActive\b",
            content,
        )
        assert legacy_form is not None or call_form is not None, (
            '`<path id="running_anim">` must be gated on `ringDashActive` '
            "(either via legacy `${ringDashActive ? ...}` inline template "
            "or via `showAnimation: ringDashActive` call to `_buildRingHTML`)."
        )

    def test_radiator_card_flame_off_grey(self):  # CR2 — sync (no hass)
        """QS-201 AC-5 — grey fills when !running; FLAME_GREY_FILLS constant exists."""
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()
        no_block = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        no_comments = re.sub(r"//[^\n]*", "", no_block)

        # FLAME_GREY_FILLS constant present.
        assert "FLAME_GREY_FILLS" in no_comments
        # Constant is an array of at least three grey values (rgba with all
        # three channels equal or near-equal — accept the simple form).
        grey_array = re.search(r"FLAME_GREY_FILLS\s*=\s*\[([^\]]+)\]", no_comments)
        assert grey_array is not None
        # FLAME_FILLS (warm) also present — both constants required.
        assert "FLAME_FILLS" in no_comments

        # Selection between the two constants is gated on `running`.
        assert re.search(r"running\s*\?\s*FLAME_FILLS\s*:\s*FLAME_GREY_FILLS", no_comments) is not None, (
            "Fill-selection must branch on `running` — `running ? FLAME_FILLS : FLAME_GREY_FILLS`"
        )

    def test_radiator_card_text_shadow_for_flame_readability(self):  # CR2 — sync (no hass)
        """QS-201 AC-9 — --ring-text-shadow on :host + applied to 4 ring text classes.

        Without the text-shadow, the centre text collapses in contrast over the
        warm orange flame backdrop. AC-9 is required-for-correctness — pin it
        so a CSS-only regression that drops the shadow can't slip past review.

        QS-199 — the shared CSS template (`shared/qs-card-styles.js`)
        owns these rules now. Uses `card_source_union` so the assertion
        searches the union.
        """
        import re

        from tests.utils.card_sources import card_source_union

        content = card_source_union("qs-radiator-card.js")

        # CSS variable declaration on :host (whitespace-tolerant).
        assert (
            re.search(
                r"--ring-text-shadow:\s*0 0 12px rgba\(0,0,0,0\.8\),\s*0 2px 4px rgba\(0,0,0,0\.5\)",
                content,
            )
            is not None
        ), "AC-9: --ring-text-shadow variable must be declared on :host"

        # Each of the four ring text classes applies the variable.
        for cls in ("target-label", "target-value", "from-to-label", "from-to-value"):
            assert (
                re.search(
                    rf"\.ring\s+\.{cls}\s*\{{[^}}]*text-shadow:\s*var\(--ring-text-shadow\)",
                    content,
                )
                is not None
            ), f"AC-9: .ring .{cls} must apply text-shadow: var(--ring-text-shadow)"

    # QS-228 — uniform ring text shadow across all 6 QS Lovelace cards.
    # Map each card to its enumerated ring-text classes. The test method
    # below loops via Python `for` (matches the radiator AC-9 idiom — the
    # surrounding class lacks parametrize precedent).
    CARDS_TO_RING_TEXT_CLASSES = {
        "qs-pool-card.js": ("pct", "target-label", "target-value"),
        "qs-radiator-card.js": (
            "target-label",
            "target-value",
            "from-to-label",
            "from-to-value",
        ),
        "qs-car-card.js": (
            "pct",
            "target-label",
            "target-value",
            "mini-title",
            "mini-value",
            "mini-range",
            "mini-range-now",
            "mini-range-target",
        ),
        "qs-climate-card.js": (
            "target-label",
            "target-value",
            "from-to-label",
            "from-to-value",
        ),
        "qs-on-off-duration-card.js": (
            "target-label",
            "target-value",
            "from-to-label",
            "from-to-value",
        ),
        "qs-water-boiler-card.js": (
            "target-label",
            "target-value",
            "from-to-label",
            "from-to-value",
        ),
    }

    def test_all_qs_cards_apply_ring_text_shadow_for_readability(self):  # CR2 — sync (no hass)
        """QS-228 — uniform ring text shadow across all 6 QS Lovelace cards.

        The pool and radiator cards already shipped this in QS-201 to keep
        inside-the-ring text legible over their animated backdrops. QS-228
        extends the same `--ring-text-shadow` variable and `text-shadow:
        var(--ring-text-shadow)` per-rule application to the four remaining
        cards (car, climate, on-off-duration, water-boiler). Some of these
        cards have animated backdrops (climate flame/snow/wind,
        water-boiler bubble/steam/wave) and some don't (car, on-off-duration)
        — the maintainer asked for visual consistency across the set.

        Block-scoped regex (`\\{[^}]*`) prevents a commented-out CSS rule
        from satisfying the assertion. Pattern matches the existing
        `test_radiator_card_text_shadow_for_flame_readability` idiom
        verbatim — the radiator test is kept intact and the two tests
        overlap on radiator by design (AC-9's own pin is its contract).

        QS-199 — uses ``card_source_union`` so the shared CSS in
        ``shared/qs-card-styles.js`` is searched alongside each card.
        """
        import re

        from tests.utils.card_sources import card_source_union

        for card_filename, classes in self.CARDS_TO_RING_TEXT_CLASSES.items():
            content = card_source_union(card_filename)

            # AC-1: --ring-text-shadow declared on :host (whitespace-tolerant).
            assert (
                re.search(
                    r"--ring-text-shadow:\s*0 0 12px rgba\(0,0,0,0\.8\),\s*0 2px 4px rgba\(0,0,0,0\.5\)",
                    content,
                )
                is not None
            ), (
                f"AC-1 ({card_filename}): --ring-text-shadow variable must "
                "be declared on :host with the verbatim value "
                "`0 0 12px rgba(0,0,0,0.8), 0 2px 4px rgba(0,0,0,0.5)`"
            )

            # AC-2: every enumerated ring-text class applies the variable.
            for cls in classes:
                assert (
                    re.search(
                        rf"\.ring\s+\.{cls}\s*\{{[^}}]*text-shadow:\s*var\(--ring-text-shadow\)",
                        content,
                    )
                    is not None
                ), f"AC-2 ({card_filename}): .ring .{cls} must apply text-shadow: var(--ring-text-shadow)"

    def test_radiator_card_flame_dancing_dynamic_proxy(self):  # CR2 — sync (no hass)
        """QS-204 AC-4 — structural proxy for "flames flicker when running".

        QS-201 implemented "dancing flames" via a sine-wave path plus a
        global `translateX` scroll driven by `_flamePhase`. The QS-201
        proxy test asserted that pattern. QS-204 redesigned the
        backdrop into peaked teeth with per-tooth tip flicker (no
        scroll), so the structural markers change. Updated assertions:

        * `_generateFlameTeethPath` — the new path generator;
        * `_tipPhases` — per-layer, per-tooth phase array driving the
          flicker (replaces the global `_flamePhase` accumulator);
        * `LAYER_TIP_FLICKER_HZ` — per-layer flicker frequency table;
        * the obsolete `LAYER_SCROLL_OFFSET` / `_flamePhase` markers
          should be ABSENT so the redesign cannot regress silently.

        QS-199 — `_generateFlameTeethPath` / `_tipPhases` /
        `LAYER_TIP_FLICKER_HZ` now live in the shared `QsFlameEngine`
        (`shared/qs-anim-flame.js`); `card_source_union` includes it.
        The ABSENCE checks (`LAYER_SCROLL_OFFSET` / `this._flamePhase`)
        still target the radiator card file directly — those obsolete
        markers must not reappear anywhere in the card-or-engine union.
        """
        import re

        from tests.utils.card_sources import card_source_union

        content = card_source_union("qs-radiator-card.js")

        # QS-204 review-fix #02 G12 — strip comments before scanning
        # for legacy markers. Without the strip, a future "// removed
        # LAYER_SCROLL_OFFSET in QS-204" comment would silently make
        # the absence-checks pass even if the symbol were reintroduced
        # in executable code. Mirrors the comment-strip pattern used
        # by `test_radiator_card_flame_height_envelope_uses_progress`.
        no_block = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        executable = re.sub(r"//[^\n]*", "", no_block)

        # New path generator must be present.
        assert "_generateFlameTeethPath" in executable, (
            "QS-204 AC-4: `_generateFlameTeethPath` (the peaked-teeth path generator) must be declared"
        )

        # Per-tooth tip phase array must drive the flicker.
        assert "_tipPhases" in executable, (
            "QS-204 AC-4: `_tipPhases` (per-layer, per-tooth phase array) must drive the per-frame flicker"
        )

        # Per-layer flicker frequencies in Hz.
        assert "LAYER_TIP_FLICKER_HZ" in executable, (
            "QS-204 AC-4: `LAYER_TIP_FLICKER_HZ` (per-layer flicker rate) must declare the per-layer turbulence rates"
        )

        # Obsolete QS-201 markers must be gone from the executable
        # source — they reintroduce the wave-scroll behaviour the user
        # reported as visually wrong. Stripped of comments so a future
        # commit message referencing the symbol cannot mask a regression.
        assert "LAYER_SCROLL_OFFSET" not in executable, (
            "QS-204 AC-4 regression: `LAYER_SCROLL_OFFSET` (parallax-tongue "
            "scroll constant from QS-201) re-appeared — the redesign drops "
            "the global scroll in favour of per-tooth tip-flicker"
        )
        assert not re.search(r"this\._flamePhase\b", executable), (
            "QS-204 AC-4 regression: `this._flamePhase` accumulator "
            "re-appeared — replaced by per-tooth `_tipPhases` arrays"
        )

    @pytest.mark.asyncio
    async def test_radiator_dashboard_passes_climate_hvac_mode_on(self, hass, full_dashboard_home):
        """BH10 — the dashboard template plumbs `climate_hvac_mode_on` through.

        The card reads it from `entities.climate_hvac_mode_on`. The
        radiator section must therefore emit a `climate_hvac_mode_on:`
        key whose value matches the transport's `state_on`.
        """
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        # CR2 — offload file I/O so the event loop doesn't block.
        template_content = await hass.async_add_executor_job(template_path.read_text)

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        # The MOCK_RADIATOR_CLIMATE_CONFIG fixture installs a
        # climate-backed radiator with `CONF_CLIMATE_HVAC_MODE_ON="heat"`,
        # so the rendered YAML must include the entry. Post-bugfix the
        # value is quoted so YAML 1.1 doesn't coerce bare `on` / `off`
        # to booleans (switch-backed radiators emit `on` literal).
        assert "climate_hvac_mode_on: 'heat'" in rendered

    @pytest.mark.asyncio
    async def test_solar_entities_appear_in_rendered_output(self, hass, full_dashboard_home):
        """Solar device entities including dynamic ones are present in the rendered YAML."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        # Entity IDs are generated by HA from platform domain + device name + translated
        # entity name. Check that solar forecast entities appear in the rendered output
        # using sufficiently specific substrings of the HA-generated entity IDs.
        assert "solar_test_solar_solar_forecast_ok" in rendered
        assert "solar_test_solar_solar_forecast_age" in rendered
        # Dynamic entities created from Solcast provider
        assert "forecast_raw_score_solcast" in rendered
        assert "active_solar_provider" in rendered

    @pytest.mark.asyncio
    async def test_home_forecast_entities_absent_from_rendered_output(self, hass, full_dashboard_home):
        """Home forecast sensors are NOT present in the rendered home dashboard YAML."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        assert "qs_no_control_forecast" not in rendered
        # qs_solar_forecast may still appear in the solar device section
        # but must not appear in the home device section
        for line in rendered.splitlines():
            if "qs_no_control_forecast" in line:
                pytest.fail(f"qs_no_control_forecast found in rendered output: {line}")


# ============================================================================
# Test: Entity names are proper strings (catches translation_key bugs)
# ============================================================================


class TestEntityNamesAreStrings:
    """Verify all entity names in ha_entities are proper strings.

    This catches the exact bug where translation_key is used for dynamic
    entities without a matching entry in strings.json, causing entity.name
    to return UNDEFINED (not a string).
    """

    @pytest.mark.asyncio
    async def test_all_entity_names_are_strings(self, full_dashboard_home):
        """Every entity attached to dashboard devices must have a string name."""
        home = full_dashboard_home
        failures = []

        for section_name, _ in home.dashboard_sections:
            for device in home.get_devices_for_dashboard_section(section_name):
                for key, entity in device.ha_entities.items():
                    name = entity.name
                    if not isinstance(name, str):
                        failures.append(
                            f"{device.name} ({device.device_type}): entity '{key}' has non-string name: {name!r}"
                        )

        assert failures == [], "Entities with non-string names (likely missing translation):\n" + "\n".join(
            f"  - {f}" for f in failures
        )

    @pytest.mark.asyncio
    async def test_all_entity_ids_are_strings(self, full_dashboard_home):
        """Every entity attached to dashboard devices must have a string entity_id."""
        home = full_dashboard_home
        failures = []

        for section_name, _ in home.dashboard_sections:
            for device in home.get_devices_for_dashboard_section(section_name):
                for key, entity in device.ha_entities.items():
                    eid = entity.entity_id
                    if not isinstance(eid, str):
                        failures.append(
                            f"{device.name} ({device.device_type}): entity '{key}' has non-string entity_id: {eid!r}"
                        )

        assert failures == [], "Entities with non-string entity_ids:\n" + "\n".join(f"  - {f}" for f in failures)


# ============================================================================
# Test: Entity translation_key validation against strings.json
# ============================================================================


class TestEntityTranslationKeys:
    """Verify all translation_key values have matching entries in strings.json.

    Uses real entity objects created through the full HA setup to check
    that every entity description with a translation_key has a corresponding
    entry in the strings.json translation file.
    """

    @pytest.mark.asyncio
    async def test_all_translation_keys_exist_in_strings_json(self, full_dashboard_home):
        """Every entity with translation_key must have a matching translation."""
        from homeassistant.helpers.entity import UNDEFINED

        home = full_dashboard_home
        translations = _load_strings_json()
        missing = []

        for section_name, _ in home.dashboard_sections:
            for device in home.get_devices_for_dashboard_section(section_name):
                for key, entity in device.ha_entities.items():
                    desc = entity.entity_description
                    tk = getattr(desc, "translation_key", None)
                    if tk is None or tk is UNDEFINED:
                        continue

                    # Determine which platform this entity belongs to
                    platform = entity.entity_id.split(".")[0] if entity.entity_id else None
                    if platform is None:
                        continue

                    platform_translations = translations.get(platform, {})
                    if tk not in platform_translations:
                        missing.append(
                            f"{device.name} ({device.device_type}): "
                            f"entity '{key}' has translation_key='{tk}' "
                            f"not found in strings.json[entity][{platform}]"
                        )

        assert missing == [], "Missing translations:\n" + "\n".join(f"  - {m}" for m in missing)


# ============================================================================
# Test: Default dashboard section assignment
# ============================================================================


class TestConfirmationPopups:
    """Verify confirmation popups appear for all home and solar action buttons."""

    EXPECTED_CONFIRMATIONS = {
        "qs_home_serialize_for_debug": "This will export home data to a debug file.",
        "qs_home_light_reset_history": "This will clear recent home history (keeps older data).",
        "qs_home_recompute_people_historical_data": "This will recompute all people historical data.",
        "qs_home_reset_history": "'WARNING: This will permanently delete ALL home history data.'",
        "qs_home_generate_yaml_dashboard": "This will regenerate the YAML dashboard.",
        "qs_solar_recompute_forecast_scores": "This will recompute all solar forecast scores.",
        "qs_solar_compute_dampening_1day": "This will compute solar dampening from 1 day of data.",
        "qs_solar_compute_dampening_7day": "This will compute solar dampening from 7 days of data.",
        "qs_solar_reset_dampening": "'WARNING: This will reset ALL solar dampening values.'",
    }

    @pytest.mark.parametrize(
        "template_name",
        [
            "quiet_solar_dashboard_template.yaml.j2",
            "quiet_solar_dashboard_template_standard_ha.yaml.j2",
        ],
    )
    @pytest.mark.asyncio
    async def test_confirmation_text_present_for_all_action_buttons(self, hass, full_dashboard_home, template_name):
        """All 9 home/solar action buttons must have confirmation text in rendered output."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / template_name
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        missing = []
        for entity_key, expected_text in self.EXPECTED_CONFIRMATIONS.items():
            if f"text: {expected_text}" not in rendered:
                missing.append(entity_key)

        assert missing == [], f"Template {template_name} missing confirmation text for: {', '.join(missing)}"


class TestDashboardSectionMapping:
    """Verify LOAD_TYPE_DASHBOARD_DEFAULT_SECTION maps all device types correctly."""

    @pytest.mark.parametrize(
        "device_type,expected_section",
        [
            ("home", "settings"),
            ("battery", "settings"),
            ("solar", "settings"),
            ("person", "settings"),
            ("car", "cars"),
            ("pool", "pools"),
            ("water_boiler", "water_boilers"),
            ("on_off_duration", "others"),
            ("climate", "climates"),
            ("heat_pump", "climates"),
            ("radiator", "radiators"),
        ],
    )
    def test_device_type_maps_to_section(self, device_type, expected_section):
        """Each device type must map to an existing dashboard section."""
        section = LOAD_TYPE_DASHBOARD_DEFAULT_SECTION.get(device_type)
        assert section == expected_section, (
            f"Device type '{device_type}' maps to '{section}', expected '{expected_section}'"
        )

        section_names = [s[0] for s in DASHBOARD_DEFAULT_SECTIONS]
        assert section in section_names, (
            f"Section '{section}' for device type '{device_type}' not in DASHBOARD_DEFAULT_SECTIONS"
        )

    def test_dashboard_device_section_translations_match_const_order(self):
        """The `dashboard_device_section` selector translation keys
        MUST match the `#N - <name>` positions computed by
        `map_section_selected_name_in_section_list` from
        `DASHBOARD_DEFAULT_SECTIONS`. A drift between the two surfaces
        as untranslated dropdown labels in the user's config flow
        (option present but rendered as raw `"#N - foo"`).
        """
        import json

        strings_path = COMPONENT_ROOT / "strings.json"
        strings = json.loads(strings_path.read_text())
        translation_options = strings["selector"]["dashboard_device_section"]["options"]

        # Build the expected `#N - <name>` keys from the const ordering.
        expected_keys = {"Not in dashboard"}
        for i, (name, _icon) in enumerate(DASHBOARD_DEFAULT_SECTIONS):
            expected_keys.add(f"#{i + 1} - {name}")

        translation_keys = set(translation_options.keys())

        missing = expected_keys - translation_keys
        extra = translation_keys - expected_keys
        assert not missing, (
            f"`dashboard_device_section` translation is MISSING keys: "
            f"{sorted(missing)}. Add them to strings.json with the "
            f"matching index from `DASHBOARD_DEFAULT_SECTIONS`."
        )
        assert not extra, (
            f"`dashboard_device_section` translation has STALE keys: "
            f"{sorted(extra)}. Remove them — the index probably shifted "
            f"after a section was inserted / removed."
        )

    def test_dashboard_default_sections_translation_alignment(self):
        """M2 (rev. 2) — `DASHBOARD_DEFAULT_SECTIONS` ordering must
        agree with the `dashboard_device_section` translation block.

        The earlier "radiators MUST be appended last" invariant was
        relaxed in favour of co-locating heating-related sections
        (`water_boilers` and `radiators` next to each other). The new
        invariant is just "translation positions match const order"
        — that's what `test_dashboard_device_section_translations_match_const_order`
        enforces; this one pins the contents independently.
        """
        names = [s[0] for s in DASHBOARD_DEFAULT_SECTIONS]
        # Every bundled section that user-visible code references must
        # exist somewhere in the list.
        for expected in ("cars", "climates", "pools", "water_boilers", "radiators", "others", "settings"):
            assert expected in names, f"Missing default section: {expected}"

    def test_dashboard_default_section_uses_device_type_not_builtin_type(self):
        """Regression: load.py must use device_type variable, not Python builtin type."""
        assert type not in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION, (
            "LOAD_TYPE_DASHBOARD_DEFAULT_SECTION should not have Python's builtin `type` as a key"
        )

        for key in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION:
            assert isinstance(key, str), f"Key {key!r} in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION is not a string"

    def test_normalize_dashboard_sections_order_reorders_bundled_to_const(self):
        """BH (post-QS-195 user bug): the `_normalize_dashboard_sections_order`
        helper must rewrite a list of bundled defaults that arrived in the
        wrong order (e.g. because an older append-only migration tacked
        sections onto the end) so the result follows
        `DASHBOARD_DEFAULT_SECTIONS` order. Without this, a user who
        upgraded across QS-194/QS-195 sees their radiator section
        rendered AFTER `settings` on the dashboard.
        """
        from custom_components.quiet_solar.ha_model.home import (
            _normalize_dashboard_sections_order,
        )

        # Simulate the broken append-only migration state: bundled
        # defaults present but out of const order.
        input_sections = [
            ("cars", "mdi:car"),
            ("climates", "mdi:home-thermometer"),
            ("pools", "mdi:pool"),
            ("others", "mdi:home"),
            ("settings", "mdi:cog-outline"),
            ("water_boilers", "mdi:water-boiler"),
            ("radiators", "mdi:radiator"),
        ]

        result = _normalize_dashboard_sections_order(input_sections)

        expected = [
            ("cars", "mdi:car"),
            ("climates", "mdi:home-thermometer"),
            ("pools", "mdi:pool"),
            ("water_boilers", "mdi:water-boiler"),
            ("radiators", "mdi:radiator"),
            ("others", "mdi:home"),
            ("settings", "mdi:cog-outline"),
        ]
        assert result == expected, (
            f"_normalize_dashboard_sections_order must reorder bundled "
            f"defaults to match DASHBOARD_DEFAULT_SECTIONS. Expected "
            f"{expected}, got {result}"
        )

    def test_normalize_dashboard_sections_order_preserves_custom_sections(self):
        """BH companion: user-defined custom sections (names NOT in
        `DASHBOARD_DEFAULT_SECTIONS_DICT`) must be preserved at the end
        of the list in their original relative order. Reordering bundled
        defaults must NOT delete or shuffle the user's custom entries.
        """
        from custom_components.quiet_solar.ha_model.home import (
            _normalize_dashboard_sections_order,
        )

        input_sections = [
            ("cars", "mdi:car"),
            ("my_workshop", "mdi:hammer"),  # custom
            ("pools", "mdi:pool"),
            ("garden_lights", "mdi:lamp"),  # custom
            ("settings", "mdi:cog-outline"),
        ]

        result = _normalize_dashboard_sections_order(input_sections)

        # Bundled defaults sorted by const order, then customs in their
        # original relative order.
        expected = [
            ("cars", "mdi:car"),
            ("pools", "mdi:pool"),
            ("settings", "mdi:cog-outline"),
            ("my_workshop", "mdi:hammer"),
            ("garden_lights", "mdi:lamp"),
        ]
        assert result == expected, (
            f"_normalize_dashboard_sections_order must preserve custom "
            f"sections in original order at the end. Expected {expected}, "
            f"got {result}"
        )

    def test_normalize_dashboard_sections_order_dedups_repeats(self):
        """Defensive: if a section name is repeated (shouldn't happen in
        practice, but the migration code path could theoretically add a
        duplicate), the helper keeps only the FIRST occurrence so the
        bundled-default ordering stays deterministic.
        """
        from custom_components.quiet_solar.ha_model.home import (
            _normalize_dashboard_sections_order,
        )

        input_sections = [
            ("cars", "mdi:car"),
            ("cars", "mdi:car-electric"),  # duplicate, different icon
            ("pools", "mdi:pool"),
        ]

        result = _normalize_dashboard_sections_order(input_sections)

        assert result == [("cars", "mdi:car"), ("pools", "mdi:pool")], (
            f"Duplicate section names must be deduped (first wins). Got {result}"
        )

    def test_normalize_dashboard_sections_order_empty_input(self):
        """Edge case: an empty input returns an empty list without error."""
        from custom_components.quiet_solar.ha_model.home import (
            _normalize_dashboard_sections_order,
        )

        assert _normalize_dashboard_sections_order([]) == []

    def test_radiator_and_water_boiler_cards_guard_against_zero_max_hours(self):
        """User-reported bug: a brand-new switch-backed radiator with no
        active constraint sensor (or one that reports 0) produces
        ``maxHours = targetHours = 0`` in the non-default-mode branch,
        causing every arc-path calc to divide by zero. The SVG paths
        end up as ``M ... A 130 130 0 0 1 NaN NaN`` and the dashboard
        shows a "Configuration error".

        Fix invariant: in the non-default branch, both cards must
        clamp ``maxHours`` to a positive finite value before using
        it in arc-path math. We pin this via regex so a future edit
        that drops the clamp regresses the test.
        """
        import re

        for card_filename in ("qs-radiator-card.js", "qs-water-boiler-card.js"):
            content = (COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text()
            # Strip comments so the regex only matches executable code.
            executable = _strip_js_comments(content)

            # The non-default branch MUST guard against zero (or NaN)
            # `targetHours` before assigning to `maxHours`. Acceptable
            # patterns include:
            #   maxHours = targetHours > 0 ? targetHours : <fallback>;
            #   maxHours = Math.max(<fallback>, targetHours);
            # Detect ANY clamp-style guard around `targetHours` near the
            # `maxHours = targetHours` assignment.
            unguarded_pat = re.compile(r"maxHours\s*=\s*targetHours\s*;", re.MULTILINE)
            assert unguarded_pat.search(executable) is None, (
                f"{card_filename}: a bare `maxHours = targetHours;` assignment "
                f"would divide by zero when the constraint sensor reports 0 "
                f"(brand-new device, no live constraint). Use a clamp like "
                f"`maxHours = targetHours > 0 ? targetHours : 12;` instead."
            )

    def test_radiator_template_quotes_climate_hvac_mode_on(self):
        """User-reported bug: a switch-backed radiator emits
        ``climate_hvac_mode_on: on`` in the dashboard YAML. YAML 1.1
        parses bare ``on`` / ``off`` as booleans, so HA hands the JS
        card ``true`` instead of the string ``"on"``. The card then
        crashes inside ``_render()`` with
        ``TypeError: true.toLowerCase is not a function`` and Lovelace
        renders a "Configuration error" card.

        Fix invariant: the radiator's `climate_hvac_mode_on` value MUST
        be quoted in BOTH dashboard templates so YAML never coerces it.
        We pin the quoted form via a regex on the template source.
        """
        import re

        for template_filename in (
            "quiet_solar_dashboard_template.yaml.j2",
            "quiet_solar_dashboard_template_standard_ha.yaml.j2",
        ):
            template_path = COMPONENT_ROOT / "ui" / template_filename
            if not template_path.exists():
                continue
            content = template_path.read_text()
            # The Jinja line we care about lives only in the custom
            # template (the standard one doesn't pass `climate_hvac_mode_on`
            # to the card — it uses entity rows). Skip if absent.
            if "climate_hvac_mode_on" not in content:
                continue
            # The value must be wrapped in quotes — single or double.
            # Anything else (bare interpolation) lets YAML 1.1 coerce
            # `on` → True / `off` → False / `yes` → True / etc.
            unquoted_pat = re.compile(
                r"climate_hvac_mode_on:\s*\{\{\s*device\.hvac_state_on\s*\}\}",
                re.MULTILINE,
            )
            assert unquoted_pat.search(content) is None, (
                f"{template_filename}: `climate_hvac_mode_on: "
                f"{{{{ device.hvac_state_on }}}}` is unquoted — YAML 1.1 "
                f"will parse bare `on`/`off` as booleans and the JS card "
                f"will crash with `true.toLowerCase is not a function`. "
                f"Quote the interpolation: "
                f"`climate_hvac_mode_on: '{{{{ device.hvac_state_on }}}}'`."
            )
            # And positively assert a quoted form is present.
            quoted_pat = re.compile(
                r"climate_hvac_mode_on:\s*['\"]\{\{\s*device\.hvac_state_on\s*\}\}['\"]",
                re.MULTILINE,
            )
            assert quoted_pat.search(content) is not None, (
                f"{template_filename}: expected a quoted "
                f"`climate_hvac_mode_on: '{{{{ device.hvac_state_on }}}}'` "
                f"emit. The unquoted variant trips YAML's bool coercion."
            )

    def test_radiator_card_tolerates_non_string_hvac_mode_on(self):
        """Defense-in-depth for the same user-reported bug: even if a
        future template regression emits an unquoted ``on`` (parsed as
        Python ``True``) into ``climate_hvac_mode_on``, the JS card
        must NOT crash on ``true.toLowerCase()`` — wrap the value in
        ``String(...)`` before the call.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()
        executable = _strip_js_comments(content)

        # Every reference to `e.climate_hvac_mode_on` that feeds into a
        # chained `.toLowerCase()` MUST be wrapped in `String(...)`
        # first. Walk every `e.climate_hvac_mode_on` occurrence and
        # check the preceding token is `String(`.
        ref_pat = re.compile(r"e\.climate_hvac_mode_on")
        unsafe_uses: list[str] = []
        for m in ref_pat.finditer(executable):
            # Search backwards from `m.start()` for the nearest
            # non-whitespace token. Accept `String(`, `String  (`,
            # or any usage that doesn't lead to `.toLowerCase()`.
            tail = executable[m.end() : m.end() + 200]
            if ".toLowerCase" not in tail:
                continue  # not a stringy use, irrelevant
            preceding = executable[max(0, m.start() - 60) : m.start()]
            if not re.search(r"String\s*\(\s*$", preceding):
                unsafe_uses.append(f"...{preceding[-50:]}<HERE>{executable[m.start() : m.end() + 40]}...")
        assert not unsafe_uses, (
            "qs-radiator-card.js: every `e.climate_hvac_mode_on` use that "
            "leads to `.toLowerCase()` must be wrapped in `String(...)`. "
            "Without the wrap, a YAML-coerced boolean (`on` → True) "
            "crashes the card. Offending uses:\n  " + "\n  ".join(unsafe_uses)
        )

    def test_radiator_and_water_boiler_cards_arc_path_guards_nan(self):
        """Defense-in-depth for the same user-reported bug: the
        ``arcPath`` helper must reject non-finite inputs so a stray
        NaN from upstream math (e.g. a 0-divisor in hoursToPct) never
        reaches the rendered SVG attribute. Without this guard, the
        browser shows ``Error: <path> attribute d: Expected number,
        "...A 130 130 0 0 1 NaN NaN"``.

        QS-199 — ``arcPath`` is a named export in
        ``shared/qs-card-base.js`` (`export function arcPath(...) {...}`).
        Accepts both the legacy arrow form and the new function-declaration
        form via the union.
        """
        import re

        from tests.utils.card_sources import card_source_union

        for card_filename in ("qs-radiator-card.js", "qs-water-boiler-card.js"):
            content = card_source_union(card_filename)
            executable = _strip_js_comments(content)

            # The arcPath helper definition must include a finiteness
            # check on its angle inputs. Accept either:
            #   - Legacy form: `arcPath = (...) => { ... };`
            #   - QS-199 form: `function arcPath(...) { ... }` (named export)
            arc_pat_arrow = re.compile(
                r"arcPath\s*=\s*\([^)]*\)\s*=>\s*\{(?P<body>[^}]*(?:\}[^}]*)*?)\};",
                re.DOTALL,
            )
            arc_pat_fn = re.compile(
                r"function\s+arcPath\s*\([^)]*\)\s*\{(?P<body>[^}]*(?:\}[^}]*)*?)\}",
                re.DOTALL,
            )
            m = arc_pat_arrow.search(executable) or arc_pat_fn.search(executable)
            assert m is not None, (
                f"{card_filename}: could not find an `arcPath` arrow OR function "
                f"declaration in the card+shared union — check the helper layout."
            )
            body = m.group("body")
            has_finite = "Number.isFinite" in body or "Number.isNaN" in body
            assert has_finite, (
                f"{card_filename}: `arcPath` must guard against non-finite "
                f"angle inputs (use Number.isFinite / Number.isNaN). "
                f"Body was: {body[:300]}..."
            )

    def test_device_dashboard_section_translation_present_for_each_step(self):
        """User-reported bug: the `device_dashboard_section` form field
        renders as the raw key in the UI for every step except `person`
        because no translation was defined elsewhere.

        Every config-step (and its corresponding options-step) whose
        `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION` mapping is non-None
        includes the `CONF_DEVICE_DASHBOARD_SECTION` field in its
        schema via `get_common_schema`, so each MUST have a
        translation entry in `strings.json` under `data` (either a
        literal or a `[%key:...%]` reference).
        """
        import json

        from custom_components.quiet_solar.const import LOAD_TYPE_DASHBOARD_DEFAULT_SECTION

        strings_path = COMPONENT_ROOT / "strings.json"
        strings = json.loads(strings_path.read_text())

        # Map device-type constants to their config-flow step name.
        # Step name = the value of `conf_type_name` on the device
        # class, which matches the device-type constant for everything
        # except chargers (we skip those — their LOAD_TYPE_... is
        # None so they don't get the dashboard-section field).
        type_to_step = {
            "home": "home",
            "battery": "battery",
            "solar": "solar",
            "person": "person",
            "car": "car",
            "pool": "pool",
            "water_boiler": "water_boiler",
            "on_off_duration": "on_off_duration",
            "climate": "climate",
            "heat_pump": "heat_pump",
            "radiator": "radiator",
        }

        # All step names that should have the translation.
        required_steps = [
            type_to_step[type_name]
            for type_name, section in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION.items()
            if section is not None and type_name in type_to_step
        ]
        # Sanity: must cover at least the post-QS-195 set.
        for must_have in (
            "home",
            "battery",
            "solar",
            "person",
            "car",
            "pool",
            "water_boiler",
            "on_off_duration",
            "climate",
            "heat_pump",
            "radiator",
        ):
            assert must_have in required_steps, (
                f"Step {must_have!r} expected to be required but isn't "
                f"derived from LOAD_TYPE_DASHBOARD_DEFAULT_SECTION; "
                f"got {required_steps}"
            )

        # Verify both config.step.<X>.data.device_dashboard_section
        # and options.step.<X>.data.device_dashboard_section exist.
        missing: list[str] = []
        for namespace in ("config", "options"):
            for step in required_steps:
                try:
                    data_block = strings[namespace]["step"][step]["data"]
                except KeyError:
                    missing.append(f"{namespace}.step.{step}.data (missing)")
                    continue
                if "device_dashboard_section" not in data_block:
                    missing.append(f"{namespace}.step.{step}.data.device_dashboard_section")

        assert not missing, (
            f"Every step that includes the CONF_DEVICE_DASHBOARD_SECTION "
            f"field MUST have a translation in strings.json. Missing: "
            f"{missing}"
        )

    @pytest.mark.asyncio
    async def test_all_devices_assigned_to_sections(self, full_dashboard_home):
        """Regression: all devices must appear in at least one dashboard section."""
        home = full_dashboard_home
        all_section_devices = set()

        for section_name, _ in home.dashboard_sections:
            for device in home.get_devices_for_dashboard_section(section_name):
                all_section_devices.add(device.device_id)

        # Home itself plus all devices that were set up
        assert len(all_section_devices) > 0, "No devices found in any dashboard section"
        # At minimum: home, solar, charger, car, person, battery, on_off,
        # climate, heat_pump, pool, water_boiler
        assert len(all_section_devices) >= 11, (
            f"Expected at least 11 devices in dashboard sections, found {len(all_section_devices)}"
        )


# ============================================================================
# Water-boiler-specific dashboard rendering
# ============================================================================


async def _build_water_boiler_home(hass, water_boiler_config: dict):
    """Build a minimal home with a single water_boiler device.

    Helper for the two parametrised water-boiler dashboard tests below.
    Seeds all mock states and sets up home + water_boiler config entries.
    Uses the same add/setup-per-entry pattern as `full_dashboard_home`.
    """
    _apply_all_mock_states(hass)

    uid = uuid.uuid4().hex[:8]

    with patch(
        "custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers",
        new_callable=AsyncMock,
    ):
        home_entry = MockConfigEntry(
            domain=DOMAIN,
            data=MOCK_HOME_CONFIG,
            entry_id=f"wb_home_{uid}",
            title="home: Test Home",
            unique_id=f"wb_home_{uid}",
        )
        home_entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(home_entry.entry_id) is True
        await hass.async_block_till_done()

        boiler_entry = MockConfigEntry(
            domain=DOMAIN,
            data=water_boiler_config,
            entry_id=f"wb_boiler_{uid}",
            title="water_boiler: Test Water Boiler",
            unique_id=f"wb_boiler_{uid}",
        )
        boiler_entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(boiler_entry.entry_id) is True
        await hass.async_block_till_done()

    home = hass.data[DOMAIN].get(home_entry.entry_id)
    assert home is not None
    return home


def _collect_water_boiler_entity_rows(parsed: dict) -> list[str]:
    """Walk the parsed dashboard YAML and return the entity ids in the
    boiler card's `entities:` block.

    The custom template emits a `key: value` mapping (JS-card input
    contract) while the standard template emits a list of `- entity:`
    rows. Both shapes are normalised to a flat list of entity-id
    strings here.
    """
    entity_ids: list[str] = []
    for view in parsed.get("views", []):
        if view.get("path") != "water_boilers":
            continue
        for grid in view.get("sections", []):
            for card in grid.get("cards", []):
                entities = card.get("entities")
                if isinstance(entities, dict):
                    # Custom template: { key: entity_id, ... }
                    entity_ids.extend(v for v in entities.values() if isinstance(v, str))
                elif isinstance(entities, list):
                    # Standard template: [ {entity: id, name: ...}, ... ]
                    for row in entities:
                        if isinstance(row, str):
                            entity_ids.append(row)
                        elif isinstance(row, dict) and "entity" in row:
                            entity_ids.append(row["entity"])
    return entity_ids


@pytest.mark.parametrize(
    "template_name",
    [
        "quiet_solar_dashboard_template.yaml.j2",
        "quiet_solar_dashboard_template_standard_ha.yaml.j2",
    ],
)
@pytest.mark.asyncio
async def test_water_boiler_temperature_sensor_row_present_when_configured(hass, template_name):
    """When the temp sensor is configured, its entity id is in the card's entities."""
    home = await _build_water_boiler_home(hass, MOCK_WATER_BOILER_CONFIG)
    template_path = COMPONENT_ROOT / "ui" / template_name
    template_content = template_path.read_text()

    tpl = Template(template_content, hass)
    rendered = tpl.async_render(variables={"home": home})

    parsed = yaml.safe_load(rendered)
    assert parsed is not None
    entity_ids = _collect_water_boiler_entity_rows(parsed)
    assert "sensor.test_water_boiler_temperature" in entity_ids, (
        f"Expected the configured temperature sensor in the boiler card; got rows: {entity_ids}"
    )


@pytest.mark.asyncio
async def test_water_boiler_renders_dedicated_js_card_in_custom_template(hass):
    """Water boiler renders as `custom:qs-water-boiler-card` in the custom template.

    Per the QS-194 contract: water_boiler MUST NOT reuse
    `qs-on-off-duration-card`; it gets its own dedicated card so future
    boiler-specific UI (temperature, water usage, anti-legionella) has
    a place to land without churning every on/off-duration user. The
    JS file lives at `ui/resources/qs-water-boiler-card.js` and is
    auto-registered through the listdir loop in `dashboard.py:342`.
    """
    home = await _build_water_boiler_home(hass, MOCK_WATER_BOILER_CONFIG)
    template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
    template_content = template_path.read_text()

    tpl = Template(template_content, hass)
    rendered = tpl.async_render(variables={"home": home})

    parsed = yaml.safe_load(rendered)
    assert parsed is not None

    water_boiler_section = None
    for view in parsed.get("views", []):
        if view.get("path") == "water_boilers":
            water_boiler_section = view
            break
    assert water_boiler_section is not None, "water_boilers view not rendered"

    sections = water_boiler_section.get("sections", [])
    assert len(sections) > 0, "water_boilers section has no device grids"
    card_types: list[str] = []
    for grid in sections:
        for card in grid.get("cards", []):
            card_types.append(card.get("type", ""))

    assert "custom:qs-water-boiler-card" in card_types, (
        f"water_boiler must render as custom:qs-water-boiler-card; got {card_types}"
    )
    assert "custom:qs-on-off-duration-card" not in card_types, (
        f"water_boiler must NOT reuse qs-on-off-duration-card; got {card_types}"
    )


@pytest.mark.asyncio
async def test_water_boiler_card_js_resource_present(hass):
    """`qs-water-boiler-card.js` is present in `ui/resources/`.

    The auto-registration loop in `dashboard.py:342` walks every file
    under `ui/resources/` and registers each as a Lovelace resource at
    `/local/quiet_solar/<filename>`. If the JS file is missing the
    `custom:qs-water-boiler-card` dispatcher entry above silently
    breaks for any user. This test guards against the JS file being
    deleted or renamed.
    """
    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    assert js_path.is_file(), (
        f"Expected the dedicated JS card at {js_path} — the dispatcher "
        "in quiet_solar_dashboard_template.yaml.j2 references "
        "`custom:qs-water-boiler-card`, which requires this file to "
        "exist and be auto-registered via dashboard.py:342."
    )
    content = js_path.read_text(encoding="utf-8")
    # N17: regex tolerates either quote style + optional whitespace.
    import re

    assert re.search(
        r"customElements\.define\(\s*['\"]qs-water-boiler-card['\"]",
        content,
    ), (
        "qs-water-boiler-card.js must register the `qs-water-boiler-card` "
        "custom element so HA Lovelace can instantiate the card."
    )


@pytest.mark.asyncio
async def test_water_boiler_card_filters_empty_temperature_state(hass):
    """S3 regression: empty-string temperature state must not render as `0.0`.

    `Number("") === 0`, so without an explicit empty-string filter the
    temperature row would render `0.0 °C` for a sensor whose state is
    transiently empty. Source-level guard since the card runs in a
    browser, not in Python tests.
    """
    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    content = js_path.read_text(encoding="utf-8")
    # The filter chain must include `!== ''` for the temperature state.
    # We require it to appear in the same block as the existing
    # `unknown`/`unavailable` filters so a future edit can't silently
    # drop it.
    import re

    pattern = re.compile(
        r"rawTempState[^;]*?!==\s*['\"]\s*['\"][^;]*?unknown[^;]*?unavailable",
        re.DOTALL,
    )
    assert pattern.search(content), (
        "S3: qs-water-boiler-card.js must filter empty-string temperature "
        "state alongside `unknown`/`unavailable` to avoid rendering `0.0 °C`."
    )


@pytest.mark.parametrize(
    "card_filename",
    [
        "qs-water-boiler-card.js",
        "qs-on-off-duration-card.js",
        "qs-climate-card.js",
        "qs-pool-card.js",
    ],
)
def test_card_mode_change_wrapped_in_try_finally(card_filename):
    """M2: every card with a `_isProcessing*` flag wraps the awaited
    `_select` call in `try { ... } finally { ... }` so a rejected
    service call can't wedge the flag forever.

    Accepts both `try {} finally {}` and `try {} catch {} finally {}`
    structures (the radiator card uses the latter — a defensive
    `catch` swallows the error before falling through to the
    `finally` cleanup, which the other cards now share).

    QS-199 — uses ``card_source_union`` so the shared
    ``_wireBistateMode`` implementation in
    ``shared/qs-ring-duration-base.js`` is included in the search.
    """
    import re

    from tests.utils.card_sources import card_source_union

    content = card_source_union(card_filename)
    # Must find at least one `_isProcessing... = true` set followed by
    # a `try { ... await this._select(... } [catch (...) { ... }] finally { ... }`
    # block. The `(?:catch\b[^{}]*\{[^{}]*\}\s*)?` group permits the
    # defensive catch block introduced when QS-195's radiator card
    # style was ported across all cards.
    pattern = re.compile(
        r"_isProcessing\w+\s*=\s*true\s*;[^;]*?"
        r"try\s*\{[^}]*?await\s+this\._select\([^)]*\)[^}]*?\}"
        r"\s*(?:catch\s*\([^)]*\)\s*\{[^}]*\}\s*)?finally",
        re.DOTALL,
    )
    assert pattern.search(content), (
        f"M2: {card_filename} must wrap the `_isProcessing... = true; "
        f"await this._select(...)` block in `try {'{'}...{'}'} finally "
        f"{'{'}...{'}'}` so a rejected service call doesn't leave "
        f"the flag wedged."
    )


# Cards covered by `test_card_raf_idle_gated` — the idle-gated set.
#
# QS-200 (review-fix #02 N9): `qs-water-boiler-card.js` was previously
# in this list with gate ``showAnimation``. The boiler now mirrors the
# pool's intrinsically-continuous RAF model (water always visible —
# cool when off, boiling when on), so it's covered separately by
# `test_water_boiler_card_has_start_stop_helpers` below. Keeping this
# history above the decorator (instead of inline in the parametrize
# list) means a future reviewer scanning the parametrize entries can
# read them as a clean enumeration without losing the removed-card
# rationale.
#
# QS-232 (this commit): `qs-car-card.js` was previously in this list
# with gate `charging`. The car card now mirrors the continuous-RAF
# model adopted by the pool and (per QS-200) the water-boiler cards
# — RAF runs continuously while connected because the electron-soup
# wave is intrinsically visible (idle green or degraded grey).
# Covered separately by the new
# `test_car_card_continuous_raf_from_connected_callback` below.
@pytest.mark.parametrize(
    "card_filename,gate",
    [
        ("qs-on-off-duration-card.js", "showAnimation"),
        ("qs-climate-card.js", "showAnimation"),
        ("qs-radiator-card.js", "showAnimation"),
    ],
)
def test_card_raf_idle_gated(card_filename, gate):
    """M4: every card except pool / water boiler gates `_startAnimation`
    on a runtime condition (`showAnimation` for on-off-duration/
    climate, `charging` for car). The pool card's wave is intrinsically
    continuous-while-connected and uses `_startAnimation` from
    `connectedCallback` directly — that file is covered by the
    presence-of-helper check below. QS-200 moved
    `qs-water-boiler-card.js` to the same continuous-RAF model, also
    covered separately.

    QS-199 — ``_startAnimation`` / ``_stopAnimation`` now live in
    ``shared/qs-card-base.js``. Uses ``card_source_union`` so the
    helpers are seen by the presence assertion.
    """
    import re

    from tests.utils.card_sources import card_source_union

    content = card_source_union(card_filename)
    assert "_startAnimation" in content and "_stopAnimation" in content, (
        f"M4: {card_filename} must define both _startAnimation and _stopAnimation helpers."
    )
    # The render path must call _startAnimation conditionally on `gate`.
    gate_call_pattern = re.compile(
        rf"if\s*\(\s*{re.escape(gate)}\s*\)\s*\{{[^}}]*?_startAnimation",
        re.DOTALL,
    )
    assert gate_call_pattern.search(content), (
        f"M4: {card_filename} must call `_startAnimation()` conditionally on `{gate}` from within `_render()`."
    )


def test_pool_card_has_start_stop_helpers():
    """M4 pool-variant: pool's RAF is intrinsically continuous, but the
    `_startAnimation` / `_stopAnimation` helper-naming is still present
    for cross-card consistency.

    QS-199 — the helpers now live in `shared/qs-card-base.js`. Uses
    `card_source_union` so the assertion still finds them.
    """
    from tests.utils.card_sources import card_source_union

    content = card_source_union("qs-pool-card.js")
    assert "_startAnimation" in content and "_stopAnimation" in content, (
        "M4: qs-pool-card.js (or its shared base) must define "
        "_startAnimation/_stopAnimation helpers for cross-card consistency."
    )


def test_water_boiler_card_uses_safe_number_helper():
    """S8: water-boiler card uses `_safeNumber` instead of
    `Number(s?.state || N)` so `unknown`/`unavailable`/`""` don't
    propagate NaN into SVG path attributes.
    """
    import re

    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    content = js_path.read_text(encoding="utf-8")
    assert "_safeNumber" in content, "S8: qs-water-boiler-card.js must define `_safeNumber(...)` helper."
    # The duration-related reads must use _safeNumber, not raw `Number(.state || N)`.
    # Forbidden pattern: `Number(s<X>?.state || N)` where <X> is one of the duration sensors.
    forbidden = re.compile(
        r"Number\(s(DurationLimit|CurrentDuration|DefaultOnDuration)\?\.state\s*\|\|",
    )
    assert not forbidden.search(content), (
        "S8: replace `Number(s*?.state || N)` for duration sensors with "
        "`this._safeNumber(s*, N)` to avoid NaN propagation."
    )


@pytest.mark.parametrize(
    "card_filename",
    [
        "qs-water-boiler-card.js",
        "qs-pool-card.js",
        "qs-car-card.js",
    ],
)
def test_card_caps_raf_dt_against_hidden_tab(card_filename):
    """QS-200 review-fix S6: cap RAF `dt` against hidden-tab return.

    Without a cap, the first frame after a tab returns from a hidden
    state can produce a `dt` of many seconds (or more). Effects on
    the pool-pattern step loop:

    - `_wavePhase += _currentSpeed * dt` advances by a huge amount in
      one frame → visible wave jump (modulo wrap saves correctness,
      but the visual snaps).
    - Bubble `b.life += dt` can exceed `BUBBLE_MAX_LIFE_S` in a single
      frame → entire bubble layer retires at once on tab return.

    The lerp factor already has its own `lerpDt = Math.min(dt,
    LERP_DT_CEIL)` clamp; this test pins a TOP-level `dt` clamp so the
    phase advance and bubble life increment are also bounded.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text()
    executable = _strip_js_comments(content)
    # Accept either the literal `0.1` or the named `LERP_DT_CEIL` form
    # (both cards expose the same constant; pool's value is the same).
    pattern = re.compile(r"dt\s*=\s*Math\.min\s*\(\s*dt\s*,\s*(?:0\.1|LERP_DT_CEIL)\s*\)")
    assert pattern.search(executable), (
        f"S6: {card_filename} must clamp the RAF step `dt` with "
        f"`dt = Math.min(dt, LERP_DT_CEIL)` (or `Math.min(dt, 0.1)`) "
        f"after the initial computation, so a hidden-tab return "
        f"doesn't trigger a one-frame wave-scroll burst or mass "
        f"bubble-life expiry."
    )


def test_water_boiler_card_has_start_stop_helpers():
    """QS-200: water-boiler card mirrors the pool card — continuous RAF
    while connected, no `showAnimation` gate.

    Asserts:
    - both `_startAnimation()` and `_stopAnimation()` helpers are defined.
    - `connectedCallback()` calls `this._startAnimation()` DIRECTLY, with
      no `if` statement between the brace and the call (continuous-RAF
      architecture; the calm-vs-boiling distinction is amplitude / speed
      / color-mix lerp, not RAF on/off).

    Review-fix #01 S2: the `connectedCallback` body is extracted via a
    balanced-brace walk instead of a `[^{}]*?` regex, so a future
    template literal or arrow function inside the body doesn't trip
    a misleading false negative.

    Review-fix #01 N1: the `if`-gate guard is a tokenized
    ``\\bif\\s*\\(`` regex, not a substring search — identifiers like
    ``verify`` / ``modifier`` / ``lifecycle`` can't trigger a false
    positive.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    # Strip comments first so the regex only matches executable code.
    executable = _strip_js_comments(content)
    assert re.search(r"_startAnimation\s*\(\s*\)\s*\{", executable), (
        "qs-water-boiler-card.js: missing _startAnimation method"
    )
    assert re.search(r"_stopAnimation\s*\(\s*\)\s*\{", executable), (
        "qs-water-boiler-card.js: missing _stopAnimation method"
    )
    cb_body = _extract_js_function_body(executable, r"connectedCallback\s*\(\s*\)\s*")
    assert cb_body is not None, "qs-water-boiler-card.js: connectedCallback() not found"
    assert re.search(r"this\._startAnimation\s*\(\s*\)", cb_body), (
        "qs-water-boiler-card.js: connectedCallback must call "
        "_startAnimation directly (continuous-RAF model, mirror pool)"
    )
    assert re.search(r"\bif\s*\(", cb_body) is None, (
        "qs-water-boiler-card.js: connectedCallback must call "
        "_startAnimation() unconditionally — no `if (...)` gate "
        "(continuous-RAF model, mirror pool)"
    )


def test_water_boiler_card_uses_heat_palette():
    """QS-200: boiler card adopts the climate-card heat palette.

    The `const colors = { ... };` block at the top of `_render()`
    MUST contain the heat hex codes and MUST NOT contain any of the
    legacy cool-blue values. Cool-blue may still appear ELSEWHERE in
    the file (e.g. `.power-btn.on` legitimately uses HA-blue
    `#2196F3` as a semantic anchor) — the assertion is scoped to the
    `colors` const only.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    block_match = re.search(
        r"const\s+colors\s*=\s*\{(?P<body>.*?)\};",
        content,
        flags=re.DOTALL,
    )
    assert block_match is not None, "qs-water-boiler-card.js: expected `const colors = { ... };` block"
    body = block_match.group("body")
    for heat_hex in ("#FF5722", "#D32F2F", "#FF6E40", "#E64A19"):
        assert heat_hex in body, f"qs-water-boiler-card.js: heat-palette color {heat_hex} missing from `const colors`"
    for cool_hex in ("#2196F3", "#00bcd4", "#8bc34a", "#00e1ff", "#0066ff"):
        assert cool_hex not in body, (
            f"qs-water-boiler-card.js: legacy cool-blue color {cool_hex} still present in `const colors`"
        )


def test_water_boiler_card_renders_water_layer():
    """QS-200: boiler card renders a circular clipPath wrapping the
    six wave paths (3 cool + 3 boil cross-fade layers).

    Review-fix #01 N9: also pins the DOM z-order of the clipped water
    group — it MUST appear before ``<path d="${bgPath}">`` in source
    order so the ring, progress arc, and handle render ON TOP of the
    water (not under it).
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    assert re.search(r"<clipPath\s+id=", executable), "qs-water-boiler-card.js: missing <clipPath> definition"
    for layer in (
        "wave0_cool",
        "wave0_boil",
        "wave1_cool",
        "wave1_boil",
        "wave2_cool",
        "wave2_boil",
    ):
        assert re.search(rf'id="{layer}"', executable), f'qs-water-boiler-card.js: missing <path id="{layer}">'
    assert re.search(r"_generateWavePath\s*\(", executable), "qs-water-boiler-card.js: missing _generateWavePath method"
    # N9: z-order — the clipped water <g> must precede the bgPath <path>
    # in source order. Pin via index comparison on stable anchor strings
    # that appear exactly once in `_render()`'s SVG template literal.
    g_anchor = '<g clip-path="url(#${waterClipId})">'
    bg_anchor = '<path d="${bgPath}"'
    g_idx = executable.find(g_anchor)
    bg_idx = executable.find(bg_anchor)
    assert g_idx != -1, f"qs-water-boiler-card.js: missing clipped water group anchor {g_anchor!r}"
    assert bg_idx != -1, f"qs-water-boiler-card.js: missing bgPath anchor {bg_anchor!r}"
    assert g_idx < bg_idx, (
        "qs-water-boiler-card.js: the clipped water <g> must appear "
        "BEFORE the bgPath <path> in DOM order so the dashed ring, "
        "progress arc, and target handle render on top of the water. "
        f"Got g_idx={g_idx}, bg_idx={bg_idx}."
    )


def test_water_boiler_card_has_bubble_system():
    """QS-200: boiler card has a dynamic bubble system with a soft cap."""
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    assert re.search(r"MAX_CONCURRENT_BUBBLES\s*=\s*12", executable), (
        "qs-water-boiler-card.js: missing `MAX_CONCURRENT_BUBBLES = 12` constant"
    )
    assert re.search(r"BUBBLE_SPAWN_RATE_HZ\s*=", executable), (
        "qs-water-boiler-card.js: missing BUBBLE_SPAWN_RATE_HZ constant"
    )
    assert re.search(r"this\._bubbles\s*=", executable), (
        "qs-water-boiler-card.js: missing `this._bubbles` instance array"
    )


def test_water_boiler_card_has_surface_glow():
    """QS-200: boiler card renders a red Gaussian-blurred surface glow
    with mix-blend-mode: screen, locked to wave 0.

    Review-fix #01 N2: pins the exact ``SURFACE_GLOW_COLOR`` literal
    (``'#FF3D00'`` — the canonical Material "Deep Orange A400"). The
    prior ``#FF[0-9A-Fa-f]{4}`` regex was inconsistently restrictive
    (rejected 3-digit / 8-digit forms, accepted unrelated reds).
    Pinning the literal makes any drift instantly visible.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    assert re.search(r"<filter\s+id=", executable) and ("feGaussianBlur" in executable), (
        "qs-water-boiler-card.js: missing <filter>/<feGaussianBlur> in defs"
    )
    assert re.search(r'id="surface_glow"', executable), 'qs-water-boiler-card.js: missing <path id="surface_glow">'
    assert re.search(r"mix-blend-mode\s*:\s*screen", executable), (
        "qs-water-boiler-card.js: surface_glow must use `mix-blend-mode: screen`"
    )
    assert re.search(r"SURFACE_GLOW_COLOR\s*=\s*['\"]#FF3D00['\"]", executable), (
        "qs-water-boiler-card.js: SURFACE_GLOW_COLOR must be the "
        "canonical `'#FF3D00'` (Material Deep Orange A400). Any drift "
        "should be deliberate — update this assertion in lock-step."
    )


def test_water_boiler_card_pins_geometry_constants():
    """Review-fix #01 S5: pin the module-level geometry constants
    (AC-2). The SVG layout depends on `CLIP_R`, `CENTER_CX`, `CENTER_CY`,
    and `WAVE_WIDTH` matching specific values that the ``<clipPath>``
    markup interpolates via the `clipPathD` builder (QS-217 swapped the
    bare ``<circle cx cy r>`` form for a ``<path>`` with evenodd
    fill-rule so the override button can be carved out as a hole). A
    future tweak that changes any of them without re-checking the SVG
    would silently misalign the water animation, so we pin each here.

    `CENTER_CX` is review-fix #01 N4's new constant (was inlined as
    `160` in two places). `CENTER_CY` and `CLIP_R` and `WAVE_WIDTH`
    pre-date this fix but the original test suite never pinned them.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    expected = {
        "CENTER_CX": "160",
        "CENTER_CY": "160",
        "CLIP_R": "120",
        "WAVE_WIDTH": "480",
    }
    for name, value in expected.items():
        pat = re.compile(rf"const\s+{name}\s*=\s*{re.escape(value)}\b")
        assert pat.search(content), (
            f"qs-water-boiler-card.js: expected module-level `const {name} = {value};` declaration"
        )
    # N5: pin BUBBLE_FILL_COLOR as a named constant too (was inlined
    # as `'rgba(255,255,255,0.85)'` in the bubble createElementNS call).
    assert re.search(
        r"const\s+BUBBLE_FILL_COLOR\s*=\s*['\"]rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*0\.85\s*\)['\"]",
        content,
    ), "qs-water-boiler-card.js: expected module-level `const BUBBLE_FILL_COLOR = 'rgba(255,255,255,0.85)';`"


def test_water_boiler_card_pins_water_level_formula():
    """Review-fix #01 S3: pin the AC-4 water-level mapping literals so
    a refactor can't silently change ``progressRatio`` clamping or the
    1/5..4/5 fill window.

    AC-4 requires:
    - guarded predicate ``Number.isFinite(displayTargetHours) && displayTargetHours > 0``
    - fill window literal ``0.2 + progressRatio * 0.6`` (= 1/5..4/5 of clip)
    - geometry literal ``* 2 * CLIP_R`` in the ``rawWaterBaseY`` expression

    Each is asserted independently so a failure message tells you
    which knob was touched.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    assert re.search(
        r"Number\.isFinite\s*\(\s*displayTargetHours\s*\)\s*&&\s*displayTargetHours\s*>\s*0",
        executable,
    ), (
        "qs-water-boiler-card.js (AC-4): the progress-ratio guard must "
        "be `Number.isFinite(displayTargetHours) && displayTargetHours "
        "> 0` so a null / NaN / negative / zero target falls cleanly "
        "to a 0 ratio."
    )
    assert re.search(r"0\.2\s*\+\s*progressRatio\s*\*\s*0\.6", executable), (
        "qs-water-boiler-card.js (AC-4): the fill-window literal "
        "`0.2 + progressRatio * 0.6` (= 1/5..4/5 of clip diameter) "
        "must appear in the rawWaterBaseY expression."
    )
    # Scope the `* 2 * CLIP_R` literal to the rawWaterBaseY expression
    # so we don't match unrelated `2 * CLIP_R` arithmetic elsewhere.
    assert re.search(
        r"rawWaterBaseY\s*=\s*CENTER_CY\s*\+\s*CLIP_R\s*-\s*\([^)]*\)\s*\*\s*2\s*\*\s*CLIP_R",
        executable,
    ), (
        "qs-water-boiler-card.js (AC-4): the rawWaterBaseY expression "
        "must match `CENTER_CY + CLIP_R - (...) * 2 * CLIP_R` — any "
        "deviation breaks the 1/5..4/5 fill mapping."
    )


def test_water_boiler_card_pins_opacity_cross_fade():
    """Review-fix #01 S4: pin the AC-5/AC-6 dual-layer opacity cross-fade
    formulae so a refactor can't accidentally introduce an HSL hue lerp
    (which would pass through yellow-green at the midpoint) or a
    staircase regen threshold.

    AC-5/AC-6 require:
    - cool-layer opacity = ``1 - this._currentColorMix`` (with
      ``.toFixed(3)`` rounding before ``setAttribute``)
    - boil-layer opacity = ``this._currentColorMix`` (same rounding)
    - lerp target form ``running ? 1 : 0`` (or equivalent ternary on
      ``boiling`` / ``this._running``)

    These are scoped to ``executable`` (comments stripped) so a
    comment quoting the formula doesn't accidentally pass.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    # Cool-layer opacity literal — allow either `(1 - this._currentColorMix).toFixed(N)`
    # or `1 - this._currentColorMix` direct.
    assert re.search(
        r"\(\s*1\s*-\s*this\._currentColorMix\s*\)\s*\.toFixed\s*\(",
        executable,
    ), (
        "qs-water-boiler-card.js (AC-5): expected the cool-layer "
        "opacity literal `(1 - this._currentColorMix).toFixed(N)` in "
        "the RAF step body."
    )
    # Boil-layer opacity literal.
    assert re.search(
        r"this\._currentColorMix\s*\.toFixed\s*\(",
        executable,
    ), "qs-water-boiler-card.js (AC-6): expected the boil-layer opacity literal `this._currentColorMix.toFixed(N)`."
    # Lerp target: `targetColorMix = ... ? 1 : 0` — accept any condition
    # token (e.g. `boiling`, `this._running === true`, etc.) so future
    # readability tweaks don't break the test, but pin the `: 0` tail
    # AND the `1` numerator.
    assert re.search(
        r"targetColorMix\s*=\s*[^;]*\?\s*1\s*:\s*0",
        executable,
    ), (
        "qs-water-boiler-card.js (AC-6): expected the colorMix lerp "
        "target to be `targetColorMix = <cond> ? 1 : 0`, mirroring "
        "amplitude/speed."
    )


def test_water_boiler_card_pins_boil_water_palette():
    """QS-220 + QS-225 AC-4: BOIL_WATER_COLORS is in the true-blue
    family, not the legacy near-white triplet. QS-220 originally
    pinned `hue == 185` (cyan-teal); QS-225's post-PR amendment
    widened the band to `[200, 230]` (true blue, matching the pool's
    direction) and added two further direction pins: saturation
    `[20, 45]` (paler than QS-220's sat 60) and alpha `[0.05, 0.30]`
    (more transparent than QS-220's 0.40/0.32/0.24). The intent of the
    QS-220 sentinel is preserved — BOIL stays in the blue-water
    family, not the legacy near-white triplet — only the band shape
    changes."""
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    # Extract the array body so the regex isn't tricked by
    # unrelated literals elsewhere in the file.
    match = re.search(
        r"const\s+BOIL_WATER_COLORS\s*=\s*\[(?P<body>[^\]]+)\]",
        content,
    )
    assert match, "qs-water-boiler-card.js: BOIL_WATER_COLORS declaration must remain a literal array."
    body = match.group("body")
    entries = re.findall(
        r"hsla\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%\s*,\s*(0\.\d+)\s*\)",
        body,
    )
    assert len(entries) == 3, f"QS-225 AC-4: BOIL_WATER_COLORS must contain exactly 3 hsla entries; got {len(entries)}."
    for hue_s, sat_s, _light_s, alpha_s in entries:
        hue, sat, alpha = int(hue_s), int(sat_s), float(alpha_s)
        assert 200 <= hue <= 230, (
            f"QS-225 AC-4: BOIL_WATER_COLORS hue {hue} outside [200, 230] (true-blue band; legacy QS-220 was 185)."
        )
        assert 20 <= sat <= 45, (
            f"QS-225 AC-4: BOIL_WATER_COLORS saturation {sat} outside [20, 45] (paler direction; legacy QS-220 was 60)."
        )
        assert 0.05 <= alpha <= 0.30, (
            f"QS-225 AC-4: BOIL_WATER_COLORS alpha {alpha} outside "
            "[0.05, 0.30] (more transparent than legacy 0.24-0.40)."
        )
    # Sentinel: legacy near-white pattern is gone (preserved from QS-220).
    assert "hsla(0, 0%" not in body, (
        "QS-220 AC-1: the legacy near-white pattern `hsla(0, 0%, …)` must not appear in BOIL_WATER_COLORS."
    )
    # Sentinel: legacy QS-220 cyan-teal pattern is gone.
    assert "hsla(185, 60%" not in body, (
        "QS-225 AC-4: the legacy QS-220 cyan-teal pattern `hsla(185, 60%, …)` must not appear in BOIL_WATER_COLORS."
    )


def test_pool_card_pins_default_water_palette_direction():
    """QS-225 AC-1 (fallback): DEFAULT_WATER_COLORS is in the true-blue
    family (hue 200-230) and substantially more transparent than the
    legacy 0.55/0.45/0.35 alphas. Hue/sat/light are direction-only —
    the implementer can iterate inside the bands without amending the
    story."""
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-pool-card.js").read_text()
    match = re.search(
        r"const\s+DEFAULT_WATER_COLORS\s*=\s*\[(?P<body>[^\]]+)\]",
        content,
    )
    assert match, "qs-pool-card.js: DEFAULT_WATER_COLORS declaration must remain a literal array."
    body = match.group("body")
    entries = re.findall(
        r"hsla\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%\s*,\s*(0\.\d+)\s*\)",
        body,
    )
    assert len(entries) == 3, (
        f"QS-225 AC-1: DEFAULT_WATER_COLORS must contain exactly 3 hsla entries; got {len(entries)}."
    )
    for hue_s, _sat_s, _light_s, alpha_s in entries:
        hue = int(hue_s)
        alpha = float(alpha_s)
        assert 200 <= hue <= 230, f"QS-225 AC-1: DEFAULT_WATER_COLORS hue {hue} outside [200, 230] (true-blue band)."
        assert 0.05 <= alpha <= 0.40, (
            f"QS-225 AC-1: DEFAULT_WATER_COLORS alpha {alpha} outside "
            "[0.05, 0.40] (substantially more transparent than legacy)."
        )
    assert "hsla(185, 60%, 22%, 0.55)" not in content, (
        "QS-225 AC-1: legacy literal `hsla(185, 60%, 22%, 0.55)` must be removed from qs-pool-card.js."
    )


def test_pool_card_pins_temp_envelope_direction():
    """QS-225 AC-1 (runtime): _tempToColor's HSL envelope shifted to
    the true-blue band and the embedded alphas dropped substantially.
    This is the runtime path used whenever the pool has a temp sensor —
    DEFAULT_WATER_COLORS is only the no-sensor fallback."""
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-pool-card.js").read_text()
    executable = _strip_js_comments(content)

    # Envelope constants.
    env = re.search(
        r"const\s+COOL_HUE\s*=\s*(\d+)\s*,\s*WARM_HUE\s*=\s*(\d+)\s*;",
        executable,
    )
    assert env, (
        "qs-pool-card.js: COOL_HUE / WARM_HUE declaration must remain "
        "in the canonical `const COOL_HUE = N, WARM_HUE = M;` shape."
    )
    cool_hue, warm_hue = int(env.group(1)), int(env.group(2))
    assert 200 <= cool_hue <= 230, f"QS-225 AC-1: COOL_HUE {cool_hue} outside [200, 230]."
    assert 195 <= warm_hue <= 225, f"QS-225 AC-1: WARM_HUE {warm_hue} outside [195, 225]."
    assert warm_hue <= cool_hue, (
        f"QS-225 AC-1: WARM_HUE ({warm_hue}) must be ≤ COOL_HUE ({cool_hue}) to preserve cool>warm hue ordering."
    )
    assert "const COOL_HUE = 195, WARM_HUE = 175" not in executable, (
        "QS-225 AC-1: legacy declaration `const COOL_HUE = 195, WARM_HUE = 175;` must be removed."
    )

    # Embedded alphas inside _tempToColor's body only — must scope via
    # the brace-walker helper to avoid catching CSS `rgba(0,0,0,0.8)`
    # at the bottom of the file.
    body = _extract_js_function_body(executable, r"_tempToColor\s*\(")
    assert body is not None, "qs-pool-card.js: _tempToColor function body must be extractable (signature unchanged)."
    # Alpha literals look like `, 0.NN)` at the end of an hsla(...)
    # template — exclude `toFixed(N)` which has the form `(N)`.
    alphas = [float(a) for a in re.findall(r",\s*(0\.\d+)\s*\)", body)]
    assert len(alphas) >= 3, (
        f"QS-225 AC-1: _tempToColor body must contain ≥3 alpha literals; got {len(alphas)} ({alphas})."
    )
    for alpha in alphas:
        assert 0.05 <= alpha <= 0.40, f"QS-225 AC-1: _tempToColor alpha {alpha} outside [0.05, 0.40]."


def test_water_boiler_card_pins_cool_water_palette_direction():
    """QS-225 AC-2: COOL_WATER_COLORS (boiler "not running") shifted
    less-blue + greyish + more transparent. Sat ≤ 30 makes it read
    grey-ish instead of saturated blue."""
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    # Scope by symbol name (mirrors the QS-220 BOIL_WATER_COLORS pin)
    # so the BOIL palette is not walked into.
    match = re.search(
        r"const\s+COOL_WATER_COLORS\s*=\s*\[(?P<body>[^\]]+)\]",
        content,
    )
    assert match, "qs-water-boiler-card.js: COOL_WATER_COLORS declaration must remain a literal array."
    body = match.group("body")
    entries = re.findall(
        r"hsla\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%\s*,\s*(0\.\d+)\s*\)",
        body,
    )
    assert len(entries) == 3, f"QS-225 AC-2: COOL_WATER_COLORS must contain exactly 3 hsla entries; got {len(entries)}."
    for hue_s, sat_s, _light_s, alpha_s in entries:
        hue, sat, alpha = int(hue_s), int(sat_s), float(alpha_s)
        assert 195 <= hue <= 215, (
            f"QS-225 AC-2: COOL_WATER_COLORS hue {hue} outside [195, 215] (still cool, but not the legacy 185 teal)."
        )
        assert 10 <= sat <= 30, (
            f"QS-225 AC-2: COOL_WATER_COLORS saturation {sat} outside [10, 30] (greyish direction; legacy was 60)."
        )
        assert 0.05 <= alpha <= 0.40, f"QS-225 AC-2: COOL_WATER_COLORS alpha {alpha} outside [0.05, 0.40]."
    assert "hsla(185, 60%, 22%, 0.55)" not in body, (
        "QS-225 AC-2: legacy literal `hsla(185, 60%, 22%, 0.55)` must be removed from COOL_WATER_COLORS."
    )


def test_climate_card_snowflake_matches_pile_tint_more_solid():
    """QS-225 AC-3: SNOW_FILL_COLOR (falling snowflakes) shares the
    hue/sat/light of SNOW_FRONT_COLOR (pile front) so the in-flight
    flake reads as the same material as the snow below it, but its
    alpha is higher so flakes look more solid than the translucent
    pile."""
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
    executable = _strip_js_comments(content)

    hsla_re = r"'hsla\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%\s*,\s*(0\.\d+)\s*\)'"
    front = re.search(
        r"const\s+SNOW_FRONT_COLOR\s*=\s*" + hsla_re,
        executable,
    )
    fill = re.search(
        r"const\s+SNOW_FILL_COLOR\s*=\s*" + hsla_re,
        executable,
    )
    assert front, "qs-climate-card.js: SNOW_FRONT_COLOR must remain a module-level const assigned to an hsla literal."
    assert fill, (
        "QS-225 AC-3: SNOW_FILL_COLOR must be a module-level const "
        "assigned to an hsla literal (no longer the legacy rgba)."
    )

    fh, fs, fl, fa = int(front.group(1)), int(front.group(2)), int(front.group(3)), float(front.group(4))
    xh, xs, xl, xa = int(fill.group(1)), int(fill.group(2)), int(fill.group(3)), float(fill.group(4))

    assert (xh, xs, xl) == (fh, fs, fl), (
        f"QS-225 AC-3: SNOW_FILL_COLOR hue/sat/light ({xh},{xs}%,{xl}%) "
        f"must equal SNOW_FRONT_COLOR ({fh},{fs}%,{fl}%) so the "
        "snowflake tint matches the pile below."
    )
    assert xa > fa, (
        f"QS-225 AC-3: SNOW_FILL_COLOR alpha {xa} must be strictly "
        f"greater than SNOW_FRONT_COLOR alpha {fa} so flakes read as "
        "more solid than the translucent pile."
    )
    assert 0.65 <= xa <= 0.95, (
        f"QS-225 AC-3: SNOW_FILL_COLOR alpha {xa} outside [0.65, 0.95] "
        "(solid enough to read in flight, not pure-opaque)."
    )
    assert "rgba(255, 255, 255, 0.9)" not in content, (
        "QS-225 AC-3: legacy literal `rgba(255, 255, 255, 0.9)` must be removed from qs-climate-card.js."
    )


def test_water_boiler_card_center_uses_named_constants():
    """Review-fix #02 S1: the arc / handle / progress-ring `center`
    object in `_render()` must use the named `CENTER_CX` / `CENTER_CY`
    constants, not inlined `160` literals. The water clip circle
    moved to `CENTER_CX` / `CENTER_CY` with fix #01 N4; the ring
    geometry must follow so a future tweak to either constant keeps
    the ring, handle, and progress arc aligned with the water layer.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    positive = re.compile(r"const\s+center\s*=\s*\{\s*cx\s*:\s*CENTER_CX\s*,\s*cy\s*:\s*CENTER_CY\s*\}")
    assert positive.search(executable), (
        "qs-water-boiler-card.js: the `const center = {...}` literal "
        "must use `CENTER_CX` / `CENTER_CY`, not inlined `160` values. "
        "A future tweak to either constant must keep the ring, handle, "
        "and progress arc aligned with the water clip circle."
    )
    forbidden = re.compile(r"center\s*=\s*\{\s*cx\s*:\s*160\b")
    assert forbidden.search(executable) is None, (
        "qs-water-boiler-card.js: inlined `center = {cx: 160, ...}` "
        "literal found; replace the raw 160 values with "
        "`CENTER_CX` / `CENTER_CY`."
    )


def test_water_boiler_card_factors_reset_dom_refs_helper():
    """Review-fix #02 S3: a `_resetDomRefs()` helper is defined and
    called from BOTH `_invalidateWaveCache()` AND the
    post-`innerHTML` cleanup block — so a future memo-key addition
    inside the helper is automatically picked up at both call sites.

    Without the shared helper, the two cleanup paths drift: a new
    `this._fooEl = null` line added to `_invalidateWaveCache()` is
    silently missed by the post-innerHTML block, and the next frame
    sees stale memo state until the field happens to mutate.
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    helper_def = re.search(r"_resetDomRefs\s*\(\s*\)\s*\{", executable)
    assert helper_def, (
        "qs-water-boiler-card.js: missing `_resetDomRefs()` helper "
        "method. Extract the shared DOM-ref / bubble-array reset from "
        "`_invalidateWaveCache()` and the post-innerHTML cleanup block "
        "so a future memo-key addition is picked up at both sites."
    )
    calls = re.findall(r"this\._resetDomRefs\s*\(\s*\)", executable)
    assert len(calls) >= 2, (
        "qs-water-boiler-card.js: `_resetDomRefs()` must be called "
        "from BOTH `_invalidateWaveCache()` AND the post-innerHTML "
        f"cleanup block; found {len(calls)} call site(s)."
    )


def test_water_boiler_card_steam_layer_after_surface_glow_in_clip():
    """QS-211 AC-1: the steam `<g>` is the new last child of the water
    clip-path group, sitting AFTER `<path id="surface_glow">` in source
    order. It carries the steam filter and `pointer-events="none"`."""
    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    clip_open = source.find('<g clip-path="url(#${waterClipId})">')
    assert clip_open != -1, (
        "qs-water-boiler-card.js: missing the water clip-path group "
        'opener `<g clip-path="url(#${waterClipId})">` — has the '
        "QS-200 markup been removed?"
    )
    surface_idx = source.find('id="surface_glow"', clip_open)
    steam_idx = source.find('id="${steamLayerId}"', clip_open)
    assert surface_idx != -1, (
        'qs-water-boiler-card.js: missing `id="surface_glow"` inside the water clip group (QS-200 anchor).'
    )
    assert steam_idx != -1, (
        "qs-water-boiler-card.js: missing the QS-211 steam layer "
        '`<g id="${steamLayerId}" ...>` inside the water clip group.'
    )
    assert surface_idx < steam_idx, (
        "qs-water-boiler-card.js: the steam `<g>` must appear AFTER "
        '`<path id="surface_glow">` in source order so it stacks '
        "above the surface glow inside the clip group."
    )
    steam_group = '<g id="${steamLayerId}" filter="url(#${steamFilterId})" pointer-events="none"></g>'
    assert steam_group in source, (
        "qs-water-boiler-card.js: steam group markup must be exactly "
        f"`{steam_group}` (filter + pointer-events both required)."
    )
    # Review-fix #01 finding #1: the previous assertion
    # `source.find("</g>", steam_idx)` resolved to the steam group's OWN
    # self-closing tag — trivially `!= -1` even if the steam layer were
    # OUTSIDE the clip group. To prove containment, advance the search
    # past the steam group's own `</g>` so the next `</g>` we find must
    # be a sibling/ancestor closer — specifically the outer water-clip
    # group's `</g>`. AC-1: "both indices live within the clipped
    # group".
    steam_group_idx = source.find(steam_group, clip_open)
    assert steam_group_idx != -1, (
        "qs-water-boiler-card.js: literal steam group block must appear inside the water clip group."
    )
    outer_close = source.find("</g>", steam_group_idx + len(steam_group))
    assert outer_close != -1, (
        "qs-water-boiler-card.js: no `</g>` AFTER the steam group's "
        "self-close — the steam layer must be nested inside the outer "
        "water-clip `<g>` so puffs are clipped at the ring."
    )


def test_water_boiler_card_steam_spawn_is_boiling_gated():
    """QS-211 AC-2: the steam spawn `while` loop is wrapped in
    `if (boiling) { ... }`, while the advance/retire `for` loop sits
    OUTSIDE that guard (graceful-exit per design decision D15)."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    spawn_gate = re.search(
        r"if \(boiling\)\s*\{[^}]*while\s*\(\s*this\._nextSteamAt\s*<=\s*0"
        r"[^}]*MAX_CONCURRENT_STEAM",
        executable,
        re.DOTALL,
    )
    assert spawn_gate, (
        "qs-water-boiler-card.js: the steam spawn `while` must live "
        "inside an `if (boiling) { ... }` guard (the same `boiling` "
        "const reused from the bubble block)."
    )
    advance_match = re.search(
        r"for\s*\(\s*const\s+p\s+of\s+this\._steamPuffs\s*\)",
        executable,
    )
    assert advance_match, "qs-water-boiler-card.js: missing the steam advance loop `for (const p of this._steamPuffs)`."
    cap_clamp = re.search(
        r"if\s*\(\s*this\._nextSteamAt\s*<\s*0\s*\)\s*"
        r"this\._nextSteamAt\s*=\s*0\s*;",
        executable,
    )
    assert cap_clamp, (
        "qs-water-boiler-card.js: missing the cap-recovery clamp `if (this._nextSteamAt < 0) this._nextSteamAt = 0;`."
    )
    # The advance loop must come AFTER the cap-clamp line (which is the
    # last statement inside the `if (boiling)` block), proving the
    # advance loop is outside the boiling-gate.
    assert advance_match.start() > cap_clamp.start(), (
        "qs-water-boiler-card.js: the steam advance/retire loop must "
        "appear AFTER the cap-clamp line — that is, OUTSIDE the "
        "`if (boiling)` guard — so in-flight puffs continue to rise "
        "and fade gracefully when boiling flips off."
    )


def test_water_boiler_card_steam_retire_both_branches():
    """QS-214 rim-fade iteration: the steam retire predicate covers
    BOTH branches — `p.cy < localTopY` (reached the local clip-circle
    top) AND `p.life >= p.maxLife` (hard upper-bound). Either retires.
    The pin-at-rim experiment was reverted because it left puffs
    statically pinned for several seconds, which read as "stuck above
    the surface" rather than "rising and dissipating at the rim". The
    rim-fade (re-added to the opacity test) makes the geometric retire
    smooth — by the time the puff's cy reaches localTopY, its opacity
    is already 0, so the remove is invisible."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    advance_start = executable.find("for (const p of this._steamPuffs)")
    assert advance_start != -1, "qs-water-boiler-card.js: missing steam advance loop."
    advance_block = executable[advance_start : advance_start + 1500]
    # Per-puff geometry (unchanged).
    assert re.search(
        r"const\s+dx\s*=\s*p\.cx\s*-\s*CENTER_CX",
        advance_block,
    ), "qs-water-boiler-card.js: the per-puff geometry must derive `dx = p.cx - CENTER_CX` inside the advance loop."
    assert re.search(
        r"const\s+localChordHalf\s*=\s*Math\.sqrt\(\s*Math\.max\(\s*0\s*,\s*"
        r"CLIP_R\s*\*\s*CLIP_R\s*-\s*dx\s*\*\s*dx\s*\)\s*\)",
        advance_block,
    ), (
        "qs-water-boiler-card.js: the per-puff geometry must derive "
        "`localChordHalf = Math.sqrt(Math.max(0, CLIP_R * CLIP_R - dx * "
        "dx))` inside the advance loop (mirrors the spawn-side chord-"
        "half formula)."
    )
    assert re.search(
        r"const\s+localTopY\s*=\s*CENTER_CY\s*-\s*localChordHalf\s*\+\s*"
        r"STEAM_TOP_MARGIN_PX",
        advance_block,
    ), (
        "qs-water-boiler-card.js: the per-puff geometry must derive "
        "`localTopY = CENTER_CY - localChordHalf + STEAM_TOP_MARGIN_PX`"
        " inside the advance loop."
    )
    # Disjunctive retire (geometric + time). With the rim-fade in
    # place the puff is already at opacity 0 by the time `p.cy <
    # localTopY` fires, so the remove is invisible — no abrupt blink.
    assert re.search(
        r"p\.cy\s*<\s*localTopY\s*\|\|\s*p\.life\s*>=\s*p\.maxLife",
        advance_block,
    ), (
        "qs-water-boiler-card.js: the steam retire predicate must be "
        "exactly `p.cy < localTopY || p.life >= p.maxLife` — both "
        "branches are required (local clip-top + maxLife)."
    )
    # Negative guard: the pin-at-rim assignment must NOT appear (it
    # caused the "stuck above the surface" perception and has been
    # superseded by the rim-fade design).
    assert not re.search(r"p\.cy\s*=\s*localTopY", advance_block), (
        "qs-water-boiler-card.js: the pin-at-rim assignment "
        "`p.cy = localTopY` must NOT appear — it was reverted in "
        "favour of the rim-fade approach."
    )


def test_water_boiler_card_steam_spawn_cx_narrowed_to_central_band():
    """QS-214 rim-fade iteration: spawn cx is constrained to a narrow
    central band (CENTER_CX ± STEAM_SPAWN_CX_HALF_PX) regardless of
    water level, so every spawned puff has a substantial vertical
    rise budget before hitting the local rim. Previously the spawn
    range was `chordHalf * 0.85` of the water-surface chord, which
    let edge-spawned puffs land within 12 px of the local rim for
    partial water levels — the "stops just above the surface"
    complaint. The narrow central band guarantees ≥ 110 px of rise
    for any water level above ≈ y=160 (mid-tank or lower).
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    assert re.search(
        r"const\s+STEAM_SPAWN_CX_HALF_PX\s*=\s*\d+",
        executable,
    ), (
        "qs-water-boiler-card.js: missing the `STEAM_SPAWN_CX_HALF_PX` "
        "constant that bounds the spawn cx band around CENTER_CX."
    )
    # The spawn formula must clamp the chord-half by
    # STEAM_SPAWN_CX_HALF_PX so every puff lands in the narrow central
    # band even when the water-surface chord is wider.
    assert re.search(
        r"Math\.min\(\s*[^,]*chordHalf[^,]*,\s*STEAM_SPAWN_CX_HALF_PX\s*\)",
        executable,
    ), (
        "qs-water-boiler-card.js: the steam spawn must clamp the "
        "spawn-cx half-width to STEAM_SPAWN_CX_HALF_PX, e.g. "
        "`Math.min(chordHalf * 0.85, STEAM_SPAWN_CX_HALF_PX)`."
    )


def test_water_boiler_card_render_preserves_steam_puffs_across_innerhtml():
    """`_render()` rewrites `this._root.innerHTML` on every HA state
    push, which previously wiped the entire SVG including every
    in-flight steam puff. That caused the user-visible "3-4 puffs
    rising and all disappearing at the exact same time" symptom —
    puffs were being nuked by `set hass → _render → innerHTML`
    every few seconds (whenever any watched entity state changed),
    long before their individual lifecycles ended.

    This test pins the snapshot-and-restore preservation pattern:
    `_render()` must capture `_steamPuffs` / `_steamLayerEl` /
    `_nextSteamAt` BEFORE the innerHTML rewrite, then re-attach the
    preserved DOM nodes to the new steam layer AFTER `_resetDomRefs()`
    and restore the array + cadence counter. Without this pattern, no
    other steam-visual tuning can be perceived correctly — puffs get
    wiped before the user sees the lifecycle.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)

    # Pre-innerHTML snapshot.
    assert re.search(
        r"const\s+preservedSteamPuffs\s*=\s*this\._steamPuffs",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must snapshot "
        "`this._steamPuffs` into `preservedSteamPuffs` BEFORE the "
        "innerHTML rewrite, so the puff DOM nodes survive."
    )
    assert re.search(
        r"const\s+preservedNextSteamAt\s*=\s*this\._nextSteamAt",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must snapshot "
        "`this._nextSteamAt` so the spawn cadence counter doesn't "
        "reset to 0 on every render (which would cause a spawn-burst "
        "after every HA state push)."
    )

    # Post-innerHTML restore: re-attach each preserved puff's DOM node
    # to the new steam layer, then restore the array + counter.
    assert re.search(
        r"newSteamLayer\.appendChild\s*\(\s*p\.el\s*\)",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must re-attach each "
        "preserved puff's DOM node to the new steam layer via "
        "`newSteamLayer.appendChild(p.el)`."
    )
    assert re.search(
        r"this\._steamPuffs\s*=\s*preservedSteamPuffs",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must restore "
        "`this._steamPuffs = preservedSteamPuffs` after the innerHTML "
        "rewrite + _resetDomRefs() cleanup."
    )
    assert re.search(
        r"this\._nextSteamAt\s*=\s*preservedNextSteamAt",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must restore "
        "`this._nextSteamAt = preservedNextSteamAt` so the spawn "
        "cadence picks up where it left off."
    )
    # Truthy-branch array semantics: `_steamPuffs` is assigned the
    # filtered set (drop entries whose `el` was missing) so the array
    # is the canonical "preserved AND truthy" view (review fix #01
    # finding #6, alignment with the null-layer drop).
    assert re.search(
        r"this\._steamPuffs\s*=\s*preservedSteamPuffs\.filter\(\s*p\s*=>\s*p\?\.el\s*\)",
        executable,
    ), (
        "qs-water-boiler-card.js: the steam puff preserve truthy "
        "branch must align array semantics with the null-layer drop "
        "via `this._steamPuffs = preservedSteamPuffs.filter(p => p?.el)`."
    )
    # Null-layer else branch: explicit DOM removal of preserved puffs
    # so detached nodes don't leak (honours the docstring contract;
    # review fix #01 finding #6).
    assert re.search(
        r"for\s*\(\s*const\s+p\s+of\s+preservedSteamPuffs\s*\)\s*\{\s*"
        r"p\?\.el\?\.remove\s*\(\s*\)\s*;?\s*\}",
        executable,
    ), (
        "qs-water-boiler-card.js: the steam puff preserve null-layer "
        "else branch must iterate `preservedSteamPuffs` and explicitly "
        "call `p?.el?.remove()` so detached DOM nodes don't leak."
    )


def test_water_boiler_card_render_preserves_bubbles_across_innerhtml():
    """Symmetric to the steam-puff preservation: `_render()` must also
    snapshot `_bubbles` and `_nextBubbleAt` BEFORE the innerHTML
    rewrite and restore them AFTER `_resetDomRefs()`, re-attaching
    each preserved bubble's detached DOM node to the freshly-rendered
    bubble layer.

    The bubble blip on hass push was previously documented as
    "barely-perceptible (167 ms respawn)" — accepted at the time
    because bubbles live ~1.5 s. Now that we have the snapshot/
    restore pattern in place for steam puffs, applying it
    symmetrically to bubbles eliminates the blip entirely and removes
    a per-push DOM-thrash spike."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)

    assert re.search(
        r"const\s+preservedBubbles\s*=\s*this\._bubbles",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must snapshot "
        "`this._bubbles` into `preservedBubbles` BEFORE the innerHTML "
        "rewrite."
    )
    assert re.search(
        r"const\s+preservedNextBubbleAt\s*=\s*this\._nextBubbleAt",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must snapshot "
        "`this._nextBubbleAt` so the bubble spawn cadence doesn't "
        "reset to 0 (which would burst-spawn on every push)."
    )
    assert re.search(
        r"newBubbleLayer\.appendChild\s*\(\s*b\.el\s*\)",
        executable,
    ), (
        "qs-water-boiler-card.js: `_render()` must re-attach each "
        "preserved bubble's DOM node to the new bubble layer via "
        "`newBubbleLayer.appendChild(b.el)`."
    )
    assert re.search(
        r"this\._bubbles\s*=\s*preservedBubbles",
        executable,
    ), "qs-water-boiler-card.js: `_render()` must restore `this._bubbles = preservedBubbles` after _resetDomRefs()."
    assert re.search(
        r"this\._nextBubbleAt\s*=\s*preservedNextBubbleAt",
        executable,
    ), "qs-water-boiler-card.js: `_render()` must restore `this._nextBubbleAt = preservedNextBubbleAt`."
    # Truthy-branch filter for symmetry with the steam path.
    assert re.search(
        r"this\._bubbles\s*=\s*preservedBubbles\.filter\(\s*b\s*=>\s*b\?\.el\s*\)",
        executable,
    ), (
        "qs-water-boiler-card.js: the bubble preserve truthy branch "
        "must align array semantics with the null-layer drop via "
        "`this._bubbles = preservedBubbles.filter(b => b?.el)`."
    )
    # Null-layer else branch: explicit DOM removal for bubbles.
    assert re.search(
        r"for\s*\(\s*const\s+b\s+of\s+preservedBubbles\s*\)\s*\{\s*"
        r"b\?\.el\?\.remove\s*\(\s*\)\s*;?\s*\}",
        executable,
    ), (
        "qs-water-boiler-card.js: the bubble preserve null-layer else "
        "branch must iterate `preservedBubbles` and explicitly call "
        "`b?.el?.remove()` so detached DOM nodes don't leak."
    )


def test_climate_card_render_preserves_snowflakes_across_innerhtml():
    """Mirror of the QS-214 boiler steam-puff and bubble preservation,
    applied to climate snowflakes per QS-216.

    The climate card's `_render()` rewrites `this._root.innerHTML` on
    every HA state push (`set hass → _render → innerHTML`). Before
    QS-216 the rewrite + subsequent `_invalidateSnowCache()` call wiped
    every in-flight snowflake simultaneously — the same systemic wipe
    pathology QS-214 fixed for steam puffs and bubbles. This test pins
    the snapshot-and-restore pattern: `_render()` must capture
    `_snowflakes` / `_nextSnowflakeAt` BEFORE the innerHTML rewrite,
    then re-attach each preserved `<circle>` to the freshly-rendered
    snow layer AFTER the three `_invalidate*Cache()` calls and restore
    the array + cadence counter.

    Six content regexes pin the presence of each clause; a follow-on
    block (review fix #01 S1) extracts the `_render()` body via
    `_extract_js_function_body` and pins the AC-4 ordering invariant
    via character-offset comparisons.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
    executable = _strip_js_comments(source)

    # Pre-innerHTML snapshot.
    assert re.search(
        r"const\s+preservedSnowflakes\s*=\s*this\._snowflakes",
        executable,
    ), (
        "qs-climate-card.js: `_render()` must snapshot "
        "`this._snowflakes` into `preservedSnowflakes` BEFORE the "
        "innerHTML rewrite, so the snowflake DOM nodes survive."
    )
    assert re.search(
        r"const\s+preservedNextSnowflakeAt\s*=\s*this\._nextSnowflakeAt",
        executable,
    ), (
        "qs-climate-card.js: `_render()` must snapshot "
        "`this._nextSnowflakeAt` so the spawn cadence counter doesn't "
        "reset to 0 on every render (which would cause a spawn-burst "
        "after every HA state push)."
    )

    # Post-innerHTML restore: re-attach each preserved snowflake's DOM
    # node to the new snow layer, then restore the array + counter.
    assert re.search(
        r"newSnowLayer\.appendChild\s*\(\s*b\.el\s*\)",
        executable,
    ), (
        "qs-climate-card.js: `_render()` must re-attach each "
        "preserved snowflake's DOM node to the new snow layer via "
        "`newSnowLayer.appendChild(b.el)`."
    )
    assert re.search(
        r"this\._nextSnowflakeAt\s*=\s*preservedNextSnowflakeAt",
        executable,
    ), (
        "qs-climate-card.js: `_render()` must restore "
        "`this._nextSnowflakeAt = preservedNextSnowflakeAt` so the "
        "spawn cadence picks up where it left off."
    )
    # Truthy-branch array semantics: `_snowflakes` is assigned the
    # filtered set (drop entries whose `el` was missing) so the array
    # is the canonical "preserved AND truthy" view (mirror of boiler
    # steam-puff pattern). Review fix #01 N2: this filter assertion
    # supersedes a previously-redundant bare-assignment regex; the
    # filter form is a strict subset, so a single pin is sufficient.
    assert re.search(
        r"this\._snowflakes\s*=\s*preservedSnowflakes\.filter\(\s*b\s*=>\s*b\?\.el\s*\)",
        executable,
    ), (
        "qs-climate-card.js: the snowflake preserve truthy branch "
        "must align array semantics with the null-layer drop via "
        "`this._snowflakes = preservedSnowflakes.filter(b => b?.el)`."
    )
    # Null-layer else branch: explicit DOM removal of preserved
    # snowflakes so detached nodes don't leak (mirror of boiler).
    assert re.search(
        r"for\s*\(\s*const\s+b\s+of\s+preservedSnowflakes\s*\)\s*\{\s*"
        r"b\?\.el\?\.remove\s*\(\s*\)\s*;?\s*\}",
        executable,
    ), (
        "qs-climate-card.js: the snowflake preserve null-layer "
        "else branch must iterate `preservedSnowflakes` and explicitly "
        "call `b?.el?.remove()` so detached DOM nodes don't leak."
    )

    # Review fix #01 S1 — AC-4 ordering invariant. Presence regexes
    # above don't enforce *position*; a refactor that moved the
    # snapshot AFTER innerHTML, or the restore BEFORE the
    # _invalidate*Cache() triplet, would silently re-introduce the
    # original wipe bug. Pin the ordering via character offsets on
    # the stripped source.
    #
    # We deliberately don't use `_extract_js_function_body` here:
    # (a) `r"_render\s*\(\s*\)"` would match a call site before the
    #     declaration; and (b) the giant `this._root.innerHTML = \`…\``
    #     template literal contains many `${…}` interpolation braces
    #     that throw off the helper's balanced-brace walker (this is
    #     the known limitation called out in the helper's docstring).
    #
    # Anchoring from `snap_idx` is sufficient: the snapshot string is
    # unique, and every "after snap" needle below either occurs only
    # in `_render()` (innerHTML, newSnowLayer.appendChild) or has its
    # *other* occurrences strictly before `snap_idx` (the
    # `_invalidate*Cache()` declarations + the `disconnectedCallback`
    # call site, all earlier in the file).
    snap_idx = executable.find("const preservedSnowflakes")
    next_idx = executable.find("const preservedNextSnowflakeAt")
    innerhtml_idx = executable.find("this._root.innerHTML")
    assert snap_idx >= 0, "qs-climate-card.js: file must contain `const preservedSnowflakes` (review fix #01 S1)."
    assert next_idx >= 0, "qs-climate-card.js: file must contain `const preservedNextSnowflakeAt` (review fix #01 S1)."
    assert innerhtml_idx >= 0, (
        "qs-climate-card.js: file must contain `this._root.innerHTML` rewrite (review fix #01 S1)."
    )
    # All `_invalidate*Cache()` and `newSnowLayer.appendChild` lookups
    # start AT `snap_idx` to skip the pre-_render() declaration and
    # disconnectedCallback call-site occurrences.
    inv_flame_idx = executable.find("_invalidateFlameCache()", snap_idx)
    inv_snow_idx = executable.find("_invalidateSnowCache()", snap_idx)
    inv_wind_idx = executable.find("_invalidateWindCache()", snap_idx)
    attach_idx = executable.find("newSnowLayer.appendChild(b.el)", snap_idx)
    assert inv_flame_idx >= 0, (
        "qs-climate-card.js: `_render()` (after the snapshot) must call `_invalidateFlameCache()` (review fix #01 S1)."
    )
    assert inv_snow_idx >= 0, (
        "qs-climate-card.js: `_render()` (after the snapshot) must call `_invalidateSnowCache()` (review fix #01 S1)."
    )
    assert inv_wind_idx >= 0, (
        "qs-climate-card.js: `_render()` (after the snapshot) must call `_invalidateWindCache()` (review fix #01 S1)."
    )
    assert attach_idx >= 0, (
        "qs-climate-card.js: `_render()` (after the snapshot) must "
        "call `newSnowLayer.appendChild(b.el)` (review fix #01 S1)."
    )
    assert snap_idx < innerhtml_idx, (
        "qs-climate-card.js: AC-4 ordering invariant — "
        "`const preservedSnowflakes` snapshot MUST appear BEFORE the "
        "`this._root.innerHTML` rewrite in `_render()`. The snapshot "
        "captures the array reference that the rewrite would otherwise "
        "wipe (review fix #01 S1)."
    )
    assert next_idx < innerhtml_idx, (
        "qs-climate-card.js: AC-4 ordering invariant — "
        "`const preservedNextSnowflakeAt` snapshot MUST appear BEFORE "
        "the `this._root.innerHTML` rewrite in `_render()` "
        "(review fix #01 S1)."
    )
    assert attach_idx > inv_flame_idx, (
        "qs-climate-card.js: AC-4 ordering invariant — the restore "
        "block's `newSnowLayer.appendChild(b.el)` MUST appear AFTER "
        "`_invalidateFlameCache()` in `_render()` (review fix #01 S1)."
    )
    assert attach_idx > inv_snow_idx, (
        "qs-climate-card.js: AC-4 ordering invariant — the restore "
        "block's `newSnowLayer.appendChild(b.el)` MUST appear AFTER "
        "`_invalidateSnowCache()` in `_render()`. Calling it before "
        "would let the invalidate-cache wipe the array we just "
        "restored (review fix #01 S1)."
    )
    assert attach_idx > inv_wind_idx, (
        "qs-climate-card.js: AC-4 ordering invariant — the restore "
        "block's `newSnowLayer.appendChild(b.el)` MUST appear AFTER "
        "`_invalidateWindCache()` in `_render()` (review fix #01 S1)."
    )


def test_invalidate_snow_cache_does_not_null_el():
    """Review fix #01 S2 — pin the AC-3 invariant that
    `_invalidateSnowCache()` calls `b.el.remove()` (or any safe-nav
    variant thereof) but MUST NOT null `b.el`. If a future refactor
    adds `b.el = null` (or any morally equivalent reassignment), the
    truthy-branch `.filter(b => b?.el)` in `_render()`'s restore block
    silently drops every preserved flake — the wipe bug returns
    invisibly and the content regexes in
    `test_climate_card_render_preserves_snowflakes_across_innerhtml`
    all still pass.

    This test extracts the `_invalidateSnowCache()` body and asserts:
    1. A `.remove()` call on `b.el` is present (positive pin).
    2. No `b.el = …` assignment is present (negative pin).
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
    executable = _strip_js_comments(source)
    body = _extract_js_function_body(executable, r"_invalidateSnowCache\s*\(\s*\)")
    assert body is not None, (
        "qs-climate-card.js: `_invalidateSnowCache()` body must be "
        "extractable for the AC-3 invariant pin (review fix #01 S2)."
    )
    # Positive pin: some `b.el.remove()` or `b?.el?.remove?.()` form
    # must be present so the disconnect path still tears down DOM.
    # `(?:\?\.)?` permits the JS optional-chained call syntax
    # `remove?.()` as well as the plain `remove()` form.
    assert re.search(r"b\??\.el\??\.remove(?:\?\.)?\(\s*\)", body), (
        "qs-climate-card.js: AC-3 invariant — `_invalidateSnowCache()` "
        "must call `.remove()` on each flake's `el` (e.g. "
        "`b?.el?.remove?.()`) so the disconnectedCallback / real "
        "backdrop-transition paths leave no orphaned DOM "
        "(review fix #01 S2)."
    )
    # Negative pin: no `b.el = …` assignment. The boiler's symmetric
    # helper avoids nulling `p.el`; matching that contract is what
    # makes the truthy-branch filter `(b => b?.el)` safe.
    assert not re.search(r"b\.el\s*=\s*", body), (
        "qs-climate-card.js: AC-3 invariant — `_invalidateSnowCache()` "
        "must NOT assign to `b.el` (e.g. `b.el = null`). The "
        "truthy-branch restore filter `(b => b?.el)` in `_render()` "
        "silently drops any entry whose `el` was nulled, which would "
        "silently re-introduce the wipe pathology QS-216 fixed "
        "(review fix #01 S2)."
    )


def test_water_boiler_card_steam_spawn_defers_on_zero_budget():
    """Review fix #01 finding #5: the spawn-loop `riseBudget <= 0`
    branch must genuinely defer the spawn slot by advancing the
    cadence counter (`_nextSteamAt += 1 / STEAM_SPAWN_RATE_HZ`) and
    `continue`-ing the while loop, NOT clamp `_nextSteamAt` to 0 and
    `break`.

    The old form (`_nextSteamAt = Math.max(_nextSteamAt, 0); break;`)
    was a no-op: the while condition `_nextSteamAt <= 0` already
    guarantees `_nextSteamAt <= 0` on entry, so `Math.max(_, 0)`
    collapses to `0`. Next frame `_nextSteamAt -= dt` immediately
    re-enters the loop, runs the random `cxSpawn` + two `Math.sqrt`
    computations, hits the same branch, breaks — a per-frame CPU spin
    when the tank is too full for any spawn cx to have positive
    riseBudget.

    The fix advances the cadence by a real spawn-slot duration so the
    next attempt waits a full slot."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    # Locate the steam spawn while-loop.
    spawn_idx = executable.find("while (this._nextSteamAt <= 0")
    assert spawn_idx != -1, "qs-water-boiler-card.js: missing the steam spawn while loop."
    spawn_block = executable[spawn_idx : spawn_idx + 3000]
    # Pin the riseBudget guard exists in the spawn loop.
    assert re.search(
        r"if\s*\(\s*riseBudget\s*<=\s*0\s*\)",
        spawn_block,
    ), "qs-water-boiler-card.js: missing the `if (riseBudget <= 0)` guard in the steam spawn loop."
    # Pin the proper cadence advance.
    assert re.search(
        r"this\._nextSteamAt\s*\+=\s*1\s*/\s*STEAM_SPAWN_RATE_HZ",
        spawn_block,
    ), (
        "qs-water-boiler-card.js: the `if (riseBudget <= 0)` branch "
        "must advance the cadence with "
        "`this._nextSteamAt += 1 / STEAM_SPAWN_RATE_HZ` so the spawn "
        "is genuinely deferred (not no-op-clamped)."
    )
    # Pin the `continue` (NOT `break`).
    assert re.search(
        r"if\s*\(\s*riseBudget\s*<=\s*0\s*\)\s*\{[^}]*continue\s*;",
        spawn_block,
    ), (
        "qs-water-boiler-card.js: the `if (riseBudget <= 0)` branch "
        "must use `continue` (re-check the while condition) rather "
        "than `break` (which would exit the loop with no retry)."
    )
    # Negative guard: the old no-op clamp form must NOT appear inside
    # the riseBudget block.
    assert not re.search(
        r"if\s*\(\s*riseBudget\s*<=\s*0\s*\)\s*\{[^}]*Math\.max\s*\(\s*this\._nextSteamAt",
        spawn_block,
    ), (
        "qs-water-boiler-card.js: the no-op clamp "
        "`Math.max(this._nextSteamAt, 0)` must NOT appear inside the "
        "riseBudget guard (was the dead-code form before the review "
        "fix)."
    )


def test_water_boiler_card_steam_opacity_formula():
    """QS-214 per-puff proportional rim-fade iteration: per-frame
    opacity is `lifeOpacity * rimOpacity * this._currentColorMix`,
    where `rimOpacity` is the puff's distance below its local rim
    divided by its STORED `p.fadeBand` (computed at spawn time as
    `STEAM_RIM_FADE_FRACTION` of the rise budget at that spawn cx).
    This makes the fade band geometry-aware: center puffs (long rise)
    get a long fade band; side puffs (short local rise) get a
    proportionally shorter one. The life-curve breakpoints `0.15`
    (fade-in) and `0.85` (fade-out — safety branch for puffs that
    stall via maxLife rather than geometric retire) remain literal in
    the advance block. The `<circle>` fill is
    `STEAM_FILL_COLOR = 'rgba(195,215,235,0.45)'`."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    assert re.search(
        r"STEAM_FILL_COLOR\s*=\s*['\"]rgba\(195,\s*215,\s*235,\s*0\.45\)['\"]",
        executable,
    ), (
        "qs-water-boiler-card.js: STEAM_FILL_COLOR must be the literal "
        "`'rgba(195,215,235,0.45)'` (cool blue-gray, alpha 0.45 baked in)."
    )
    # Review fix #01 finding #2: pin the uniform vy. The asymmetric
    # [18, 32] band initially planned was retracted in favour of
    # `MIN == MAX == 24` (see ARN-D1); without these pins a future
    # edit could silently drift either constant and reintroduce the
    # "two-stream" speed-sorting near the rim.
    assert re.search(r"STEAM_RISE_PX_PER_S_MIN\s*=\s*24\b", executable), (
        "qs-water-boiler-card.js: STEAM_RISE_PX_PER_S_MIN must be `24` "
        "(uniform vy with STEAM_RISE_PX_PER_S_MAX — see ARN-D1)."
    )
    assert re.search(r"STEAM_RISE_PX_PER_S_MAX\s*=\s*24\b", executable), (
        "qs-water-boiler-card.js: STEAM_RISE_PX_PER_S_MAX must be `24` "
        "(uniform vy with STEAM_RISE_PX_PER_S_MIN — see ARN-D1)."
    )
    # User iteration on PR #215: the wave speed inherited from the
    # pool card (`BOIL_SPEED = 1.6`, `CALM_SPEED = 0.2`) read as a
    # pumped-flow pace — wrong for a heated boiler tank where the
    # visible motion should be a gentle bubbling drift. Slowed to
    # `CALM_SPEED = 0.1`, `BOIL_SPEED = 0.4` (~ 4× slower while
    # boiling). Pinned so a future pool-card sync doesn't silently
    # restore the pumped-flow speeds.
    assert re.search(r"CALM_SPEED\s*=\s*0\.1\b", executable), (
        "qs-water-boiler-card.js: CALM_SPEED must be `0.1` (slowed "
        "from the pool-card-inherited 0.2 — gentle drift when not "
        "boiling)."
    )
    assert re.search(r"BOIL_SPEED\s*=\s*0\.4\b", executable), (
        "qs-water-boiler-card.js: BOIL_SPEED must be `0.4` (slowed "
        "from the pool-card-inherited 1.6 — gentle bubbling pace, "
        "not pumped flow)."
    )
    assert re.search(
        r"const\s+opacity\s*=\s*lifeOpacity\s*\*\s*rimOpacity\s*\*\s*"
        r"this\._currentColorMix",
        executable,
    ), (
        "qs-water-boiler-card.js: the per-frame opacity must be the "
        "three-factor product `lifeOpacity * rimOpacity * "
        "this._currentColorMix` — the rim-fade factor is back, this "
        "time as a per-puff proportional fade band."
    )
    # Find the steam advance block and assert per-puff fade derivations.
    advance_start = executable.find("for (const p of this._steamPuffs)")
    assert advance_start != -1, "qs-water-boiler-card.js: missing steam advance loop."
    advance_block = executable[advance_start : advance_start + 1500]
    assert "0.15" in advance_block, (
        "qs-water-boiler-card.js: missing the `0.15` fade-in breakpoint in the steam life-curve."
    )
    assert re.search(r"lifeT\s*<\s*0\.85\b", advance_block), (
        "qs-water-boiler-card.js: missing the `lifeT < 0.85` fade-out "
        "guard in the steam life-curve (safety branch for puffs that "
        "stall via maxLife rather than geometric retire)."
    )
    assert re.search(r"\(\s*lifeT\s*-\s*0\.85\s*\)\s*/\s*0\.15", advance_block), (
        "qs-water-boiler-card.js: missing the `(lifeT - 0.85) / 0.15` fade-out ramp in the steam life-curve."
    )
    # Per-puff rim-fade derivation: rimDistance from p.cy to localTopY,
    # divided by the STORED p.fadeBand (set at spawn).
    assert re.search(
        r"const\s+rimDistance\s*=\s*p\.cy\s*-\s*localTopY",
        advance_block,
    ), "qs-water-boiler-card.js: missing `rimDistance = p.cy - localTopY` derivation in the advance loop."
    assert re.search(
        r"const\s+rimOpacity\s*=\s*Math\.max\(\s*0\s*,\s*Math\.min\(\s*1\s*,"
        r"\s*rimDistance\s*/\s*p\.fadeBand\s*\)\s*\)",
        advance_block,
    ), (
        "qs-water-boiler-card.js: missing `rimOpacity = Math.max(0, "
        "Math.min(1, rimDistance / p.fadeBand))` in the advance loop — "
        "the fade band is per-puff (stored on the puff at spawn time) "
        "rather than a global pixel constant."
    )
    # The fade band must be assigned at spawn time as a fraction of the
    # puff's spawn-side rise budget. Pin the constant and the spawn-side
    # derivation.
    assert re.search(
        r"const\s+STEAM_RIM_FADE_FRACTION\s*=\s*0\.\d+",
        executable,
    ), (
        "qs-water-boiler-card.js: missing the STEAM_RIM_FADE_FRACTION "
        "constant — controls the per-puff fade band as a fraction of "
        "the puff's rise budget."
    )
    assert re.search(
        r"fadeBand\s*[:=]\s*[^,;\n]*STEAM_RIM_FADE_FRACTION",
        executable,
    ), (
        "qs-water-boiler-card.js: the spawn loop must store a per-puff "
        "`fadeBand` computed from `STEAM_RIM_FADE_FRACTION` times the "
        "puff's rise budget."
    )
    # Negative guard: the global STEAM_RIM_FADE_PX constant must NOT
    # appear (replaced by the per-puff fadeBand).
    assert "STEAM_RIM_FADE_PX" not in executable, (
        "qs-water-boiler-card.js: `STEAM_RIM_FADE_PX` constant must "
        "NOT appear — replaced by the per-puff `STEAM_RIM_FADE_FRACTION` "
        "scaled by the per-puff rise budget."
    )


def test_water_boiler_card_steam_ids_derive_from_next_clip_id():
    """QS-211 AC-5: `_steamLayerId` and `_steamFilterId` are assigned
    inside the same `if (!this._waterClipId)` block that owns the other
    per-instance IDs, and derive from the same `uid` variable sourced
    from `QsWaterBoilerCard._nextClipId`. Naming convention:
    `wb_steamLayer_${uid}` / `wb_steamFilter_${uid}`."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    assert re.search(
        r"this\._steamLayerId\s*=\s*`wb_steamLayer_\$\{uid\}`",
        executable,
    ), "qs-water-boiler-card.js: missing `this._steamLayerId = \\`wb_steamLayer_${uid}\\`` assignment."
    assert re.search(
        r"this\._steamFilterId\s*=\s*`wb_steamFilter_\$\{uid\}`",
        executable,
    ), "qs-water-boiler-card.js: missing `this._steamFilterId = \\`wb_steamFilter_${uid}\\`` assignment."
    # Both must live inside the per-instance ID guard, i.e. after the
    # `_nextClipId` bump and before the next consumer (`const waterClipId
    # = this._waterClipId;`).
    guard_open = re.search(
        r"if\s*\(\s*!this\._waterClipId\s*\)\s*\{",
        executable,
    )
    assert guard_open, "qs-water-boiler-card.js: missing the `if (!this._waterClipId) { ... }` per-instance ID guard."
    consumer = executable.find(
        "const waterClipId = this._waterClipId;",
        guard_open.end(),
    )
    assert consumer != -1, (
        "qs-water-boiler-card.js: missing the `const waterClipId = this._waterClipId;` consumer line."
    )
    steam_layer_idx = executable.find("this._steamLayerId", guard_open.end())
    steam_filter_idx = executable.find("this._steamFilterId", guard_open.end())
    assert guard_open.end() < steam_layer_idx < consumer, (
        "qs-water-boiler-card.js: `this._steamLayerId` must be assigned "
        "inside the `if (!this._waterClipId)` block, alongside the "
        "other per-instance IDs."
    )
    assert guard_open.end() < steam_filter_idx < consumer, (
        "qs-water-boiler-card.js: `this._steamFilterId` must be "
        "assigned inside the `if (!this._waterClipId)` block, alongside "
        "the other per-instance IDs."
    )


def test_water_boiler_card_reset_dom_refs_includes_steam():
    """QS-211 AC-6: `_resetDomRefs()` body nulls `_steamLayerEl` AND
    clears `_steamPuffs`. Both call sites (`_invalidateWaveCache()`
    and the post-innerHTML cleanup block) inherit the steam reset
    automatically — no per-site duplication."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    body = _extract_js_function_body(executable, r"_resetDomRefs\s*\(\s*\)")
    assert body is not None, "qs-water-boiler-card.js: missing `_resetDomRefs()` method."
    assert re.search(r"this\._steamLayerEl\s*=\s*null", body), (
        "qs-water-boiler-card.js: `_resetDomRefs()` must null "
        "`this._steamLayerEl` so a stale ref doesn't survive into the "
        "next RAF tick after an innerHTML rewrite."
    )
    assert re.search(r"this\._steamPuffs\s*=\s*\[\s*\]", body), (
        "qs-water-boiler-card.js: `_resetDomRefs()` must clear "
        "`this._steamPuffs = []` so a re-render starts with a fresh "
        "particle array (matches the QS-200 bubble precedent)."
    )


def test_water_boiler_card_steam_cap_hit_clamps_next_steam_at():
    """QS-211 AC-7: after the steam spawn `while` loop,
    `if (this._nextSteamAt < 0) this._nextSteamAt = 0;` clamps the
    cadence counter so capacity-recovery doesn't fire a backlog
    burst. Mirrors the QS-200 bubble pattern (line 347)."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    assert re.search(
        r"if\s*\(\s*this\._nextSteamAt\s*<\s*0\s*\)\s*"
        r"this\._nextSteamAt\s*=\s*0\s*;",
        executable,
    ), (
        "qs-water-boiler-card.js: missing cap-recovery clamp "
        "`if (this._nextSteamAt < 0) this._nextSteamAt = 0;` after the "
        "steam spawn `while` loop."
    )


def test_water_boiler_card_disconnected_callback_clears_steam():
    """QS-211 AC-8: `disconnectedCallback()` eagerly removes steam
    `<circle>` DOM nodes and resets `_steamPuffs` to `[]`. Mirrors the
    bubble teardown shape (line ~426)."""
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(source)
    body = _extract_js_function_body(executable, r"disconnectedCallback\s*\(\s*\)")
    assert body is not None, "qs-water-boiler-card.js: missing `disconnectedCallback()`."
    assert re.search(
        r"this\._steamPuffs\?\.forEach\s*\(\s*p\s*=>\s*p\.el\?\.remove\?\.\(\)\s*\)",
        body,
    ), (
        "qs-water-boiler-card.js: `disconnectedCallback` must eagerly "
        "tear down steam DOM with "
        "`this._steamPuffs?.forEach(p => p.el?.remove?.())`."
    )
    assert re.search(r"this\._steamPuffs\s*=\s*\[\s*\]", body), (
        "qs-water-boiler-card.js: `disconnectedCallback` must clear `this._steamPuffs = []` after removing nodes."
    )


def test_water_boiler_card_steam_filter_region_attributes():
    """QS-211 AC-9: the steam `<filter>` is declared in `<defs>` with
    the safe-region attributes (`x="-50%" y="-50%" width="200%"
    height="200%"`) so the Gaussian blur isn't clipped at the puff
    bounding box. Mirrors the surface_glow filter precedent."""
    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    filter_decl = '<filter id="${steamFilterId}" x="-50%" y="-50%" width="200%" height="200%">'
    assert filter_decl in source, (
        "qs-water-boiler-card.js: steam filter declaration must include "
        "the safe-region attributes — expected literal substring "
        f"`{filter_decl}` (mirrors the surface_glow filter precedent)."
    )
    assert '<feGaussianBlur stdDeviation="${STEAM_BLUR_STDDEV}" />' in source, (
        'qs-water-boiler-card.js: steam filter must contain `<feGaussianBlur stdDeviation="${STEAM_BLUR_STDDEV}" />`.'
    )


# ============================================================================
# === QS-232 car-card electron-soup tests ===
# QS-232 transplants the QS-200 / QS-211 boiler architecture onto
# `qs-car-card.js`: a single sine wave inside a circular clipPath
# (idle vs charging cross-fade), lightning-blue "electron" sparkles
# that pop randomly inside the soup, lightning bolts flashing from the
# top of the clip circle to the soup surface only while charging, and
# a feTurbulence grain filter applied to the wave fill for fuzzy
# granularity. Degraded states (disconnected / faulted / stale)
# desaturate the entire clipped group via a CSS filter and suppress
# lightning. Three inside-disc buttons (sun / rabbit / time) get
# QS-217-style `<circle>` cover overlays. RAF is continuous while
# connected (no more `_charging`-gated start/stop).
# ============================================================================


def test_car_card_electron_soup_clip_group_and_two_waves():
    """QS-232 AC-1: the clipped `<g clip-path="url(#${electronClipId})">`
    group contains exactly two wave `<path>` elements sharing one `d`
    placeholder, differing only in fill (IDLE_SOUP_COLOR vs
    CHARGE_SOUP_COLOR) and opacity. Sparkle and lightning layers
    follow the waves in source order. The issue's
    "do not superpose 3 layer of sin" is pinned negatively: NO
    `wave1_*` / `wave2_*` ids.

    QS-235 AC7: the `feTurbulence` grain filter that previously
    decorated both wave fills has been removed; its dedicated negative
    guard is `test_car_card_grain_filter_removed`. This test no longer
    asserts the grain filter — only the surviving two-wave structure.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # The two wave paths share the clip group anchor.
    clip_g_anchor = '<g clip-path="url(#${electronClipId})"'
    assert clip_g_anchor in source, (
        f"qs-car-card.js: missing the electron-soup clipped group `{clip_g_anchor}…>` (AC-1)."
    )

    assert 'id="electron_wave_idle"' in source, 'qs-car-card.js: missing `<path id="electron_wave_idle">` (AC-1 #2).'
    assert 'id="electron_wave_charge"' in source, (
        'qs-car-card.js: missing `<path id="electron_wave_charge">` (AC-1 #2).'
    )

    # Both waves share the same `d` placeholder (initialWavePath),
    # differing only in fill + opacity. QS-235 AC7 dropped the trailing
    # `filter="url(#${grainFilterId})"` token from each.
    idle_wave_re = re.compile(
        r'<path\s+id="electron_wave_idle"\s+'
        r'd="\$\{initialWavePath\}"\s+'
        r'fill="\$\{IDLE_SOUP_COLOR\}"\s+'
        r'opacity="\$\{initialIdleOpacity\}"\s+'
        r'pointer-events="none"',
    )
    assert idle_wave_re.search(source), (
        "qs-car-card.js: `electron_wave_idle` <path> must carry "
        '`d="${initialWavePath}" fill="${IDLE_SOUP_COLOR}" '
        'opacity="${initialIdleOpacity}" pointer-events="none"` '
        "(AC-1 #2; grain filter removed per QS-235 AC7)."
    )
    charge_wave_re = re.compile(
        r'<path\s+id="electron_wave_charge"\s+'
        r'd="\$\{initialWavePath\}"\s+'
        r'fill="\$\{CHARGE_SOUP_COLOR\}"\s+'
        r'opacity="\$\{initialChargeOpacity\}"\s+'
        r'pointer-events="none"',
    )
    assert charge_wave_re.search(source), (
        "qs-car-card.js: `electron_wave_charge` <path> must carry "
        '`d="${initialWavePath}" fill="${CHARGE_SOUP_COLOR}" '
        'opacity="${initialChargeOpacity}" pointer-events="none"` '
        "(AC-1 #2; grain filter removed per QS-235 AC7)."
    )

    # Sparkle layer follows waves; lightning layer follows sparkles.
    idle_idx = source.find('id="electron_wave_idle"')
    charge_idx = source.find('id="electron_wave_charge"')
    sparkle_idx = source.find('id="${sparkleLayerId}"')
    lightning_idx = source.find('id="${lightningLayerId}"')
    assert idle_idx != -1 and charge_idx != -1, "qs-car-card.js: missing wave anchors for source-order check."
    assert sparkle_idx != -1, 'qs-car-card.js: missing `<g id="${sparkleLayerId}">` after the waves (AC-1 #4).'
    assert lightning_idx != -1, (
        'qs-car-card.js: missing `<g id="${lightningLayerId}" …>` after the sparkle layer (AC-1 #5).'
    )
    assert idle_idx < charge_idx < sparkle_idx < lightning_idx, (
        "qs-car-card.js: source order must be wave_idle → wave_charge "
        f"→ sparkleLayer → lightningLayer. Got idle={idle_idx}, "
        f"charge={charge_idx}, sparkle={sparkle_idx}, "
        f"lightning={lightning_idx}."
    )

    # Negative pins — single sine, not three layers.
    for forbidden in ("wave1_idle", "wave1_charge", "wave2_idle", "wave2_charge"):
        assert forbidden not in executable, (
            f"qs-car-card.js: forbidden id `{forbidden}` present — the "
            'issue explicitly says "do not superpose 3 layer of sin" '
            "(AC-1 #6)."
        )


def test_car_card_cross_fade_lerp_is_binary_idle_charge():
    """QS-232 AC-2: the wave cross-fade is a binary idle↔charge lerp.
    `targetColorMix = charging ? 1 : 0` exists; per-frame opacity uses
    `(1 - _currentColorMix)` and `_currentColorMix` with `.toFixed(3)`.
    Critically, `_currentColorMix` is NEVER mutated inside the
    `degraded` branch — the third (degraded) palette is handled via a
    CSS filter on the clip group, not via the lerp envelope.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # `targetColorMix = charging ? 1 : 0` (the car-card analogue of
    # boiler `boiling ? 1 : 0`).
    assert re.search(
        r"targetColorMix\s*=\s*charging\s*\?\s*1\s*:\s*0",
        executable,
    ), "qs-car-card.js (AC-2): missing `targetColorMix = charging ? 1 : 0` lerp target (mirrors boiler precedent)."

    # Per-frame opacity: `(1 - <colorMix>).toFixed(3)` and
    # `<colorMix>.toFixed(3)`. The `<colorMix>` token may be either
    # `this._currentColorMix` directly OR a clamped local derived
    # from it (review-fix #01 #18 introduced a `mix = Math.max(0,
    # Math.min(1, this._currentColorMix))` clamp). Either form
    # satisfies the semantic AC-2 contract.
    idle_re = re.compile(
        r"\(\s*1\s*-\s*(?:this\._currentColorMix|mix)\s*\)"
        r"\s*\.toFixed\s*\(\s*3\s*\)",
    )
    assert idle_re.search(executable), (
        "qs-car-card.js (AC-2): missing idle-opacity expression "
        "`(1 - <colorMix>).toFixed(3)` where <colorMix> is either "
        "`this._currentColorMix` or a clamped local derived from it."
    )
    charge_re = re.compile(
        r"(?:this\._currentColorMix|mix)\s*\.toFixed\s*\(\s*3\s*\)",
    )
    assert charge_re.search(executable), (
        "qs-car-card.js (AC-2): missing charge-opacity expression "
        "`<colorMix>.toFixed(3)` where <colorMix> is either "
        "`this._currentColorMix` or a clamped local derived from it."
    )
    # If a `mix` local is used for opacity, it must be derived from
    # `this._currentColorMix` (NOT a fresh independent variable
    # that bypasses the binary lerp).
    if re.search(r"\(\s*1\s*-\s*mix\s*\)\s*\.toFixed", executable):
        assert re.search(
            r"const\s+mix\s*=\s*[^;]*this\._currentColorMix",
            executable,
        ), (
            "qs-car-card.js (AC-2): the local `mix` used for opacity "
            "must be derived from `this._currentColorMix` "
            "(typically `const mix = Math.max(0, Math.min(1, "
            "this._currentColorMix));`)."
        )

    # LERP_RATE constant.
    assert re.search(r"const\s+LERP_RATE\s*=\s*2\b", executable), (
        "qs-car-card.js (AC-2): missing `const LERP_RATE = 2;` (mirror boiler envelope)."
    )

    # Negative: `_currentColorMix` is NOT mutated under a `degraded`
    # branch. Pin: no `if (degraded)` block that touches
    # `_currentColorMix`.
    forbidden = re.compile(
        r"if\s*\(\s*degraded\s*\)\s*\{[^}]*_currentColorMix",
        re.DOTALL,
    )
    assert not forbidden.search(executable), (
        "qs-car-card.js (AC-2): `_currentColorMix` must not be mutated "
        "inside a `if (degraded)` branch — the degraded palette is a "
        "runtime CSS filter on the clip group (D8), independent of "
        "the binary idle↔charge lerp."
    )


def test_car_card_sparkle_power_scaling_constants():
    """QS-232 AC-3: sparkle density / spawn-rate / radius scale
    linearly with charge power on `[SPARKLE_POWER_MIN_W,
    SPARKLE_POWER_MAX_W] = [1500W, 22000W]`. Below MIN, values clamp
    to the MIN-power endpoint. Pin all six scaling constants + the
    clamping + linear-interp formula.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    expected = {
        "SPARKLE_POWER_MIN_W": "1500",
        "SPARKLE_POWER_MAX_W": "22000",
        "SPARKLE_CHARGE_MAX_AT_MIN_POWER": "12",
        "SPARKLE_CHARGE_MAX_AT_MAX_POWER": "28",
        "SPARKLE_CHARGE_RATE_HZ_AT_MIN_POWER": "3",
        "SPARKLE_CHARGE_RATE_HZ_AT_MAX_POWER": "10",
    }
    for name, value in expected.items():
        pat = re.compile(rf"const\s+{name}\s*=\s*{re.escape(value)}\b")
        assert pat.search(executable), f"qs-car-card.js (AC-3): expected module-level `const {name} = {value};`"

    # Clamping formula.
    assert re.search(
        r"Math\.max\s*\(\s*SPARKLE_POWER_MIN_W\s*,\s*"
        r"Math\.min\s*\(\s*SPARKLE_POWER_MAX_W\s*,",
        executable,
    ), (
        "qs-car-card.js (AC-3): missing the clamping expression "
        "`Math.max(SPARKLE_POWER_MIN_W, Math.min(SPARKLE_POWER_MAX_W, "
        "power))`."
    )

    # Linear interp denominator.
    assert re.search(
        r"\(\s*SPARKLE_POWER_MAX_W\s*-\s*SPARKLE_POWER_MIN_W\s*\)",
        executable,
    ), (
        "qs-car-card.js (AC-3): missing the linear-interp denominator "
        "`(SPARKLE_POWER_MAX_W - SPARKLE_POWER_MIN_W)` in the "
        "power-scaling formula."
    )


def test_car_card_sparkle_color_decided_at_spawn():
    """QS-232 AC-4: each sparkle's `fill` attribute is decided at
    spawn time via a ternary on `_charging`, resolving to
    `SPARKLE_IDLE_COLOR` (greenish) or `SPARKLE_CHARGE_COLOR`
    (electric lightning blue, `#00E5FF`). The fill is set on the
    `<circle>` element only at spawn — it is not re-assigned per
    frame in the advance loop.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Palette constants. `SPARKLE_IDLE_COLOR` is an `hsla(...)`
    # expression with a hue in the yellow-green-to-green family
    # `[60, 160]`. The exact hue is user-tunable per the QS-217
    # visual-iteration precedent:
    # - QS-232 initial:   hue 140 (pure green, light)
    # - Review-fix #03:   hue 140, near-white mint (saturation drop)
    # - Review-fix #02
    #   user follow-up:   hue 80 (vivid yellow-green / lime) — user
    #                     said "they have to be green/yellow … not a
    #                     disc … little explosion".
    idle_color_re = re.compile(
        r"const\s+SPARKLE_IDLE_COLOR\s*=\s*"
        r"['\"]hsla\(\s*(\d+)\s*,",
    )
    idle_color_m = idle_color_re.search(executable)
    assert idle_color_m is not None, (
        "qs-car-card.js (AC-4): expected `const SPARKLE_IDLE_COLOR = 'hsla(<hue>, …, …, …)';` — value user-tunable."
    )
    idle_hue = int(idle_color_m.group(1))
    assert 60 <= idle_hue <= 160, (
        f"qs-car-card.js (AC-4): `SPARKLE_IDLE_COLOR` hue must be in "
        f"the yellow-green-to-green family `[60, 160]` so the idle "
        f'palette reads as "electron green / yellow". Got hue '
        f"{idle_hue}."
    )
    # SPARKLE_CHARGE_COLOR — pinned by name only (value is the
    # electric-blue family but the user is free to retune the exact
    # hex code).
    assert re.search(
        r"const\s+SPARKLE_CHARGE_COLOR\s*=\s*['\"]#[0-9A-Fa-f]{3,8}['\"]",
        executable,
    ), (
        "qs-car-card.js (AC-4): expected `const SPARKLE_CHARGE_COLOR = "
        "'#<hex>';` (electric lightning blue — value user-tunable)."
    )

    # Ternary selecting fill at spawn (mention both constants in a
    # ternary on `_charging` or `charging`).
    spawn_fill_re = re.compile(
        r"(?:this\._charging|charging)\s*\?\s*"
        r"SPARKLE_CHARGE_COLOR\s*:\s*SPARKLE_IDLE_COLOR",
    )
    assert spawn_fill_re.search(executable), (
        "qs-car-card.js (AC-4): expected a `_charging ? "
        "SPARKLE_CHARGE_COLOR : SPARKLE_IDLE_COLOR` ternary at the "
        "spawn site so the fill is decided once and never re-assigned."
    )


def test_car_card_lightning_gated_and_spawn_interval():
    """QS-232 AC-5: lightning bolts only spawn when `charging &&
    !degraded`. Spawn interval is randomised on
    `[LIGHTNING_SPAWN_MIN_S, LIGHTNING_SPAWN_MAX_S] = [1.5, 3.0]`s.
    `MAX_CONCURRENT_LIGHTNING = 3`. The advance/retire loop runs
    OUTSIDE the gate (graceful exit for in-flight bolts).
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Constants.
    assert re.search(r"const\s+LIGHTNING_SPAWN_MIN_S\s*=\s*1\.5\b", executable), (
        "qs-car-card.js (AC-5): missing `const LIGHTNING_SPAWN_MIN_S = 1.5;`"
    )
    assert re.search(r"const\s+LIGHTNING_SPAWN_MAX_S\s*=\s*3(\.0)?\b", executable), (
        "qs-car-card.js (AC-5): missing `const LIGHTNING_SPAWN_MAX_S = 3.0;`"
    )
    assert re.search(r"const\s+MAX_CONCURRENT_LIGHTNING\s*=\s*3\b", executable), (
        "qs-car-card.js (AC-5): missing `const MAX_CONCURRENT_LIGHTNING = 3;`"
    )

    # Spawn gate: `if (charging && !degraded)`.
    gate_re = re.compile(
        r"if\s*\(\s*charging\s*&&\s*!\s*degraded\s*\)",
    )
    assert gate_re.search(executable), (
        "qs-car-card.js (AC-5): missing the lightning spawn gate `if (charging && !degraded) { … }`."
    )

    # Re-arm formula uses MIN_S + Math.random() * (MAX_S - MIN_S).
    rearm_re = re.compile(
        r"this\._nextLightningAt\s*=\s*LIGHTNING_SPAWN_MIN_S\s*\+\s*"
        r"Math\.random\s*\(\s*\)\s*\*\s*"
        r"\(\s*LIGHTNING_SPAWN_MAX_S\s*-\s*LIGHTNING_SPAWN_MIN_S\s*\)",
    )
    assert rearm_re.search(executable), (
        "qs-car-card.js (AC-5): missing the lightning re-arm formula "
        "`this._nextLightningAt = LIGHTNING_SPAWN_MIN_S + "
        "Math.random() * (LIGHTNING_SPAWN_MAX_S - "
        "LIGHTNING_SPAWN_MIN_S);`"
    )


def test_car_card_lightning_life_and_glow_filter():
    """QS-232 AC-6: lightning life is `LIGHTNING_LIFE_S = 0.25`s.
    Three-phase opacity curve breakpoints `0.10` and `0.70` appear in
    the advance loop. The `<filter id="${lightningFilterId}" …>`
    carries safe-region attributes and a `<feGaussianBlur
    stdDeviation="${LIGHTNING_GLOW_STDDEV}" />`. The lightning layer
    `<g>` carries `filter="url(#${lightningFilterId})"` AND
    `mix-blend-mode: screen`. Lightning elements are `<path>`s, not
    `<polyline>`s.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Life constant.
    assert re.search(r"const\s+LIGHTNING_LIFE_S\s*=\s*0\.25\b", executable), (
        "qs-car-card.js (AC-6): missing `const LIGHTNING_LIFE_S = 0.25;`"
    )
    # Three-phase opacity breakpoints. Review-fix #01 #10: scope the
    # `0.10` check to the `lifeT < …` lightning-fade context so a
    # regression on the actual breakpoint can't be masked by an
    # unrelated `0.1` literal elsewhere in the file (e.g.,
    # `LERP_DT_CEIL = 0.1`).
    assert re.search(r"lifeT\s*<\s*0\.10\b", executable), (
        "qs-car-card.js (AC-6): missing fade-in breakpoint `lifeT < 0.10` in the lightning advance loop."
    )
    assert re.search(r"lifeT\s*<\s*0\.70\b", executable), (
        "qs-car-card.js (AC-6): missing fade-out breakpoint `lifeT < 0.70` in the lightning advance loop."
    )

    # Review-fix #02 user follow-up #2: the Gaussian-blur glow filter
    # was removed per user "no need of the glow effect for the
    # lightning bolts". Both the filter declaration AND its
    # application on the lightning layer must now be absent. The
    # `mix-blend-mode: screen` is kept for the electric "Screen"
    # composite against the soup.
    # Scope these "must-not-exist" checks to EXECUTABLE code only —
    # historical comments referencing the removed identifiers are
    # allowed (helpful for `git log` archaeology).
    assert "${lightningFilterId}" not in executable, (
        "qs-car-card.js (AC-6): the lightning Gaussian-blur glow "
        "filter has been removed; no `${lightningFilterId}` "
        'reference should remain in executable code. User: "no need '
        'of the glow effect for the lightning bolts".'
    )
    assert "LIGHTNING_GLOW_STDDEV" not in executable, (
        "qs-car-card.js (AC-6): the `LIGHTNING_GLOW_STDDEV` "
        "constant was removed alongside the glow filter — "
        "no reference should remain in executable code."
    )
    # Layer keeps `mix-blend-mode: screen` (composite, not blur).
    assert "mix-blend-mode: screen" in source, (
        "qs-car-card.js (AC-6): lightning layer must still use "
        "`mix-blend-mode: screen` so the bolt composites brightly "
        "against the soup even without the glow filter."
    )

    # Lightning element type: <path>, not <polyline>. Pin negatively:
    # the spawn code must construct a `path` SVG element (not
    # `polyline`).
    assert re.search(
        r"createElementNS\(\s*['\"]http://www\.w3\.org/2000/svg['\"]"
        r"\s*,\s*['\"]path['\"]",
        source,
    ), (
        "qs-car-card.js (AC-6): lightning bolts must be `<path>` "
        "elements created via `createElementNS(SVG_NS, 'path')` — "
        "not `polyline`."
    )
    assert "createElementNS('http://www.w3.org/2000/svg', 'polyline'" not in source, (
        "qs-car-card.js (AC-6): forbidden `<polyline>` creation — "
        "lightning must be `<path>` per boiler-precedent shape."
    )

    # Lightning palette + shape constants.
    # `LIGHTNING_STROKE_COLOR` is the bolt's fill colour. The value
    # is user-tunable per the QS-217 visual-iteration precedent
    # (initial `#E0F7FF` pale-white → review-fix #02 user follow-up
    # `#33B5FF` electric blue, per user "they should be light and
    # bright blue, like Zeus thunderbolts").
    assert re.search(
        r"const\s+LIGHTNING_STROKE_COLOR\s*=\s*['\"]#[0-9A-Fa-f]{3,8}['\"]",
        executable,
    ), "qs-car-card.js (AC-6): expected `const LIGHTNING_STROKE_COLOR = '#<hex>';` — value user-tunable."
    # `LIGHTNING_TOP_WIDTH` is the width at the top of the tapered
    # bolt outline (pixel value user-tunable; pinned by NAME only).
    # The previous `LIGHTNING_STROKE_WIDTH` was removed when the
    # shape changed from uniform-stroke polyline to filled-polygon
    # taper (user feedback: "wider from the top and reducing to the
    # bottom").
    assert re.search(
        r"const\s+LIGHTNING_TOP_WIDTH\s*=\s*\d+(?:\.\d+)?\b",
        executable,
    ), (
        "qs-car-card.js (AC-6): expected `const LIGHTNING_TOP_WIDTH "
        "= <number>;` (value user-tunable; pinned by NAME only)."
    )


def test_car_card_grain_filter_removed():
    """QS-235 AC7 — the `feTurbulence` grain filter is removed from the
    car card. The `<filter id="${grainFilterId}">` def, both
    `filter="url(#${grainFilterId})"` wave references, the
    `this._grainFilterId` allocation, AND the stale grain comments are
    all gone. The negative guard is scoped to the car source ONLY
    (no other card ever carried grain). The sparkle / lightning / wave
    motion layers are untouched — those positive pins live in their own
    tests (`test_car_card_electron_soup_clip_group_and_two_waves`,
    `test_car_card_sparkle_*`, `test_car_card_lightning_*`).
    """
    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()

    # AC7: every concrete grain-filter artifact — the filter def, its
    # `<feTurbulence>` / `<feComposite>` steps, both wave-fill
    # references, and the per-instance id allocation — is gone.
    #
    # QS-235 review-fix #01 SF2 — scoped to these specific tokens rather
    # than a blanket `"grain" not in source` ban: the blanket form would
    # fail CI for any future unrelated identifier or comment that merely
    # contains the letters "grain" (e.g. "granular", "fine-grained"),
    # even with the grain filter correctly removed.
    for forbidden in (
        "feTurbulence",
        "fractalNoise",
        "feComposite",
        "grainFilterId",
        "_grainFilterId",
        "grainFilter",
    ):
        assert forbidden not in source, (
            f"qs-car-card.js (QS-235 AC7): `{forbidden}` must be removed "
            f"along with the feTurbulence grain filter (def + both wave "
            f"refs + per-instance id allocation + stale comments)."
        )

    # The two wave paths survive — just without the grain filter.
    assert 'id="electron_wave_idle"' in source, (
        "qs-car-card.js (QS-235 AC7): the idle wave must survive the grain removal."
    )
    assert 'id="electron_wave_charge"' in source, (
        "qs-car-card.js (QS-235 AC7): the charge wave must survive the grain removal."
    )


def test_car_card_degraded_state_css_filter_and_lightning_suppression():
    """QS-232 AC-8: `degraded = isDisconnected || isFaulted ||
    isStale` is computed in `_render()`. The clipped `<g>` carries a
    conditional inline style `style="${degraded ? 'filter:
    saturate(0.3) brightness(0.7);' : ''}"`. Lightning is suppressed
    via the `if (charging && !degraded)` gate. `isOffGrid` is
    explicitly NOT part of `degraded`.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # `degraded` boolean.
    degraded_re = re.compile(
        r"const\s+degraded\s*=\s*isDisconnected\s*\|\|\s*"
        r"isFaulted\s*\|\|\s*isStale\b",
    )
    m = degraded_re.search(executable)
    assert m, "qs-car-card.js (AC-8): missing `const degraded = isDisconnected || isFaulted || isStale;`"
    # Negative: `isOffGrid` is NOT part of `degraded`.
    matched_line = m.group(0)
    assert "isOffGrid" not in matched_line, (
        "qs-car-card.js (AC-8): the `degraded =` expression must NOT "
        "mention `isOffGrid` — off-grid keeps the card-level tint via "
        "the `.off-grid` CSS class; the soup is unaffected (D8)."
    )

    # Inline style ternary on the clip group.
    assert "${degraded ? 'filter: saturate(0.3) brightness(0.7);' : ''}" in source, (
        "qs-car-card.js (AC-8): the clipped `<g>` must carry the "
        "inline style ternary "
        "`style=\"${degraded ? 'filter: saturate(0.3) "
        "brightness(0.7);' : ''}\"`."
    )

    # Lightning gate already pinned in AC-5 — re-assert here for
    # locality.
    assert re.search(r"if\s*\(\s*charging\s*&&\s*!\s*degraded\s*\)", executable), (
        "qs-car-card.js (AC-8): lightning spawn must be gated on `charging && !degraded`."
    )


def test_car_card_continuous_raf_from_connected_callback():
    """QS-232 AC-9: `connectedCallback()` calls `this._startAnimation()`
    directly (continuous-RAF model). `_render()` does NOT contain a
    conditional `_startAnimation()` gated on `charging`, nor any
    `_stopAnimation()` call (the only `_stopAnimation` call site is
    `disconnectedCallback`). The car card has been removed from the
    `test_card_raf_idle_gated` parametrize list (and added to the
    `test_card_caps_raf_dt_against_hidden_tab` list) — both side-
    effects are pinned here by re-reading this test file.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # connectedCallback calls _startAnimation directly, no `if` gate.
    cb_body = _extract_js_function_body(executable, r"connectedCallback\s*\(\s*\)\s*")
    assert cb_body is not None, "qs-car-card.js (AC-9): `connectedCallback()` not found."
    assert re.search(r"this\._startAnimation\s*\(\s*\)", cb_body), (
        "qs-car-card.js (AC-9): `connectedCallback` must call `this._startAnimation()` directly (continuous-RAF model)."
    )
    # Review-fix #01 #13: the contract is "`_startAnimation()` is
    # unconditional", not "no `if` at all". Future harmless guards
    # unrelated to RAF startup (e.g., the QS-232 #7 fix that
    # re-primes on state-flip after reconnect) are legitimate. Pin
    # only `if`-blocks that wrap or precede `_startAnimation`.
    raf_gate_re = re.compile(
        r"if\s*\([^)]*\)\s*\{[^}]*this\._startAnimation",
        re.DOTALL,
    )
    assert raf_gate_re.search(cb_body) is None, (
        "qs-car-card.js (AC-9): `connectedCallback` must call "
        "`_startAnimation()` unconditionally — no `if (...)` block "
        "wrapping or preceding the `_startAnimation` call "
        "(continuous-RAF model, mirror boiler/pool)."
    )

    # `_render()` body must NOT contain `_stopAnimation` or a
    # conditional `_startAnimation` gated on charging.
    # NB: the signature regex includes the opening `{` so that
    # `_extract_js_function_body` matches the METHOD DEFINITION
    # (not the various `this._render()` CALL sites in
    # `connectedCallback` / `setConfig` / event handlers, which
    # appear before the definition in source order).
    render_body = _extract_js_function_body(executable, r"_render\s*\(\s*\)\s*\{")
    assert render_body is not None, "qs-car-card.js (AC-9): `_render()` body not found."
    assert "_stopAnimation" not in render_body, (
        "qs-car-card.js (AC-9): `_render()` must not call "
        "`_stopAnimation()` — the only call site is "
        "`disconnectedCallback` (continuous-RAF model)."
    )
    forbidden_gated = re.compile(
        r"if\s*\(\s*(?:this\._)?charging\s*\)\s*\{[^}]*_startAnimation",
        re.DOTALL,
    )
    assert not forbidden_gated.search(render_body), (
        "qs-car-card.js (AC-9): `_render()` must not start RAF "
        "conditionally on `charging` — `connectedCallback` starts it "
        "continuously."
    )

    # Verify the parametrize migration by inspecting the parametrize
    # decorator of `test_card_raf_idle_gated` — the car-card tuple
    # must be absent from its list.
    #
    # NB: we deliberately avoid hard-coding the forbidden tuple as a
    # bare literal in this file (that would itself be a substring
    # match and trip the assertion). Instead we extract the parametrize
    # body via regex and inspect its contents structurally.
    test_file = Path(__file__).read_text(encoding="utf-8")
    idle_gated_block = re.search(
        r"@pytest\.mark\.parametrize\s*\(\s*\"card_filename\s*,\s*gate\""
        r".*?def\s+test_card_raf_idle_gated",
        test_file,
        re.DOTALL,
    )
    assert idle_gated_block is not None, (
        "AC-9: `test_card_raf_idle_gated` parametrize decorator not found — the test file structure has drifted."
    )
    block_text = idle_gated_block.group(0)
    assert "qs-car-card.js" not in block_text, (
        "tests/test_dashboard_rendering.py (AC-9): the car-card entry "
        "must be removed from the `test_card_raf_idle_gated` "
        "parametrize list — the car card now uses continuous RAF."
    )
    # The dt-cap list must include qs-car-card.js (AC-12 side-effect).
    cap_list_block = re.search(
        r"@pytest\.mark\.parametrize\s*\(\s*\"card_filename\"\s*,\s*\["
        r"(.*?)\]\s*,?\s*\)\s*def\s+test_card_caps_raf_dt_against_hidden_tab",
        test_file,
        re.DOTALL,
    )
    assert cap_list_block is not None, (
        "AC-9: `test_card_caps_raf_dt_against_hidden_tab` parametrize "
        "decorator not found — the test file structure has drifted."
    )
    assert "qs-car-card.js" in cap_list_block.group(1), (
        "AC-9 side-effect: `qs-car-card.js` must be present in the "
        "`test_card_caps_raf_dt_against_hidden_tab` parametrize list."
    )
    # We can't easily check the parametrize list contents from inside
    # one of its tests without circular complexity; this test merely
    # pins the removal from the idle-gated list. The dt-cap test
    # itself (test_car_card_caps_raf_dt_at_lerp_dt_ceil below) covers
    # the additive side.


def test_car_card_reset_dom_refs_helper_and_disconnect_teardown():
    """QS-232 AC-10: `_resetDomRefs()` instance method exists. It
    nulls `_waveEls`, `_sparkleLayerEl`, `_lightningLayerEl` AND clears
    `_sparkles = []` and `_lightningBolts = []`. It is called from
    `_invalidateAnimCache()` AND from the post-`innerHTML` cleanup in
    `_render()` (≥ 2 call sites). `disconnectedCallback()` tears down
    sparkle AND lightning DOM with optional-chaining shape.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    body = _extract_js_function_body(executable, r"_resetDomRefs\s*\(\s*\)\s*")
    assert body is not None, "qs-car-card.js (AC-10): `_resetDomRefs()` method not found."

    # Review-fix #01 #14: `_grainFilterEl` was a dead field — no code
    # path ever read or lazily-resolved it (and QS-235 AC7 removed the
    # grain filter entirely). `_resetDomRefs()` tracks only the live
    # DOM-ref surface area.
    for assignment_re in (
        r"this\._waveEls\s*=\s*null",
        r"this\._sparkleLayerEl\s*=\s*null",
        r"this\._lightningLayerEl\s*=\s*null",
        r"this\._sparkles\s*=\s*\[\s*\]",
        r"this\._lightningBolts\s*=\s*\[\s*\]",
    ):
        assert re.search(assignment_re, body), (
            f"qs-car-card.js (AC-10): `_resetDomRefs()` must assign "
            f"`{assignment_re}` so all DOM-ref memos and particle "
            f"arrays reset together."
        )

    # ≥ 2 `this._resetDomRefs()` call sites in the file.
    call_sites = re.findall(r"this\._resetDomRefs\s*\(\s*\)", executable)
    assert len(call_sites) >= 2, (
        f"qs-car-card.js (AC-10): expected ≥ 2 `this._resetDomRefs()` "
        f"call sites (one from `_invalidateAnimCache`, one post-"
        f"`innerHTML` in `_render`). Found {len(call_sites)}."
    )

    # `_invalidateAnimCache` exists and calls `_resetDomRefs`.
    inv_body = _extract_js_function_body(executable, r"_invalidateAnimCache\s*\(\s*\)\s*")
    assert inv_body is not None, "qs-car-card.js (AC-10): missing `_invalidateAnimCache()` method."
    assert "this._resetDomRefs()" in inv_body, (
        "qs-car-card.js (AC-10): `_invalidateAnimCache()` must call `this._resetDomRefs()`."
    )

    # disconnectedCallback teardown.
    dc_body = _extract_js_function_body(executable, r"disconnectedCallback\s*\(\s*\)\s*")
    assert dc_body is not None, "qs-car-card.js (AC-10): `disconnectedCallback()` not found."
    assert re.search(
        r"this\._sparkles\?\.forEach\s*\(\s*s\s*=>\s*s\.el\?\.remove\?\.\(\)\s*\)",
        dc_body,
    ), (
        "qs-car-card.js (AC-10): `disconnectedCallback` must eagerly "
        "tear down sparkle DOM with "
        "`this._sparkles?.forEach(s => s.el?.remove?.())`."
    )
    assert re.search(r"this\._sparkles\s*=\s*\[\s*\]", dc_body), (
        "qs-car-card.js (AC-10): `disconnectedCallback` must clear `this._sparkles = []` after the teardown."
    )
    assert re.search(
        r"this\._lightningBolts\?\.forEach\s*\(\s*b\s*=>\s*b\.el\?\.remove\?\.\(\)\s*\)",
        dc_body,
    ), (
        "qs-car-card.js (AC-10): `disconnectedCallback` must eagerly "
        "tear down lightning DOM with "
        "`this._lightningBolts?.forEach(b => b.el?.remove?.())`."
    )
    assert re.search(r"this\._lightningBolts\s*=\s*\[\s*\]", dc_body), (
        "qs-car-card.js (AC-10): `disconnectedCallback` must clear `this._lightningBolts = []` after the teardown."
    )


def test_car_card_per_instance_unique_svg_ids():
    """QS-232 AC-11: `QsCarCard._nextClipId` static counter is bumped
    inside `if (!this._electronClipId)`. Three per-instance ID fields
    are derived from the same `uid`: `_electronClipId`,
    `_sparkleLayerId`, `_lightningLayerId`. All follow the
    `car_<role>_${uid}` naming convention.

    Review-fix #02 user follow-up #2: `_lightningFilterId` was
    removed alongside the lightning glow filter — the new sharp
    purple bolt has no `<filter>` element, so the per-instance
    filter ID is unused.

    QS-235 AC7: `_grainFilterId` was removed alongside the
    `feTurbulence` grain filter, dropping the field count from 4 → 3.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Static counter bump.
    assert re.search(
        r"QsCarCard\._nextClipId\s*=\s*\(\s*QsCarCard\._nextClipId\s*\|\|\s*0\s*\)\s*\+\s*1",
        executable,
    ), (
        "qs-car-card.js (AC-11): missing the `QsCarCard._nextClipId "
        "= (QsCarCard._nextClipId || 0) + 1` static-counter bump."
    )
    # Guard.
    assert re.search(r"if\s*\(\s*!\s*this\._electronClipId\s*\)", executable), (
        "qs-car-card.js (AC-11): the counter bump must be wrapped in "
        "`if (!this._electronClipId) { … }` so the IDs are assigned "
        "once."
    )

    # Three ID fields, all `car_<role>_${uid}`-shaped. The
    # `_lightningFilterId` was removed in review-fix #02 follow-up
    # alongside the lightning glow filter; `_grainFilterId` was
    # removed in QS-235 AC7 alongside the feTurbulence grain filter.
    id_assignments = {
        "_electronClipId": "car_eClip_",
        "_sparkleLayerId": "car_sparkLayer_",
        "_lightningLayerId": "car_lightningLayer_",
    }
    for field, prefix in id_assignments.items():
        pat = re.compile(
            rf"this\.{field}\s*=\s*`{re.escape(prefix)}\$\{{uid\}}`",
        )
        assert pat.search(executable), f"qs-car-card.js (AC-11): expected `this.{field} = \\`{prefix}${{uid}}\\``"


def test_car_card_caps_raf_dt_at_lerp_dt_ceil():
    """QS-232 AC-12: module constant `const LERP_DT_CEIL = 0.1;` and
    the RAF step closure clamps `dt = Math.min(dt, LERP_DT_CEIL)` at
    the top. The car card joins
    `test_card_caps_raf_dt_against_hidden_tab` for the same hidden-
    tab guard the boiler / pool cards carry.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    assert re.search(r"const\s+LERP_DT_CEIL\s*=\s*0\.1\b", executable), (
        "qs-car-card.js (AC-12): missing `const LERP_DT_CEIL = 0.1;` module constant."
    )
    assert re.search(
        r"dt\s*=\s*Math\.min\s*\(\s*dt\s*,\s*LERP_DT_CEIL\s*\)",
        executable,
    ), "qs-car-card.js (AC-12): expected `dt = Math.min(dt, LERP_DT_CEIL);` at the top of the RAF step closure."


def test_dashboard_and_cards_doc_pins_qs_232_paragraph():
    """QS-232 AC-13: a new H3 section
    `### QS-232 — Car-card electron-soup animation` exists in
    `docs/agents/concepts/dashboard-and-cards.md` AFTER the existing
    `### QS-217` section and BEFORE the `## Hardened JS-card patterns`
    H2 header. The section body mentions the key concepts:
    single-layer wave, dual-opacity cross-fade, sparkle palette switch
    + [1500W, 22000W] power-scaling, lightning bolt 3-segment path +
    glow + spawn interval, degraded CSS filter, continuous RAF, three
    carve covers (D17). The `last_verified` field is present and
    well-formed.

    QS-235 AC7: the `feTurbulence` grain filter was removed, so the
    `feturbulence` / `grain` key-concept pins are dropped here (the
    grain's absence is pinned by `test_car_card_grain_filter_removed`).
    """
    doc_path = Path(__file__).parent.parent / "docs" / "agents" / "concepts" / "dashboard-and-cards.md"
    doc = doc_path.read_text(encoding="utf-8")

    headline = "### QS-232 — Car-card electron-soup animation"
    headline_idx = doc.find(headline)
    assert headline_idx != -1, f"dashboard-and-cards.md: missing the literal QS-232 H3 headline `{headline}` (AC-13)."

    qs217_idx = doc.find("### QS-217")
    assert qs217_idx != -1, "dashboard-and-cards.md: missing `### QS-217` headline — AC-13 anchors QS-232 AFTER it."
    hardened_idx = doc.find("## Hardened JS-card patterns")
    assert hardened_idx != -1, (
        "dashboard-and-cards.md: missing `## Hardened JS-card patterns` H2 — AC-13 anchors QS-232 BEFORE it."
    )
    assert qs217_idx < headline_idx < hardened_idx, (
        f"dashboard-and-cards.md: QS-232 headline must appear AFTER "
        f"`### QS-217` AND BEFORE `## Hardened JS-card patterns`. "
        f"Current order: QS-217 at {qs217_idx}, QS-232 at "
        f"{headline_idx}, Hardened at {hardened_idx}."
    )

    # Front-matter `last_verified` field shape.
    front_matter_match = re.search(
        r"^last_verified:\s*(\d{4}-\d{2}-\d{2})\s*$",
        doc,
        re.MULTILINE,
    )
    assert front_matter_match, (
        "dashboard-and-cards.md: front-matter must contain a `last_verified: YYYY-MM-DD` field (AC-13)."
    )

    # Slice the section body — from the QS-232 headline to the next
    # H2 (or H3) boundary.
    section = doc[headline_idx:hardened_idx]
    section_lower = section.lower()

    # AC-13 key-concept pins.
    expected_substrings = [
        ("electron", "soup electron-soup concept"),
        ("sparkle", "sparkle palette switch"),
        ("lightning", "lightning bolt subsystem"),
        ("degraded", "degraded CSS filter"),
        ("continuous", "continuous-RAF model"),
        ("1500", "[1500W, 22000W] power-scaling range (lower)"),
        ("22000", "[1500W, 22000W] power-scaling range (upper)"),
        ("3-segment", "3-segment lightning path"),
        ("0.25", "lightning life 0.25 s"),
        ("[1.5", "lightning spawn interval lower bound"),
        ("3.0", "lightning spawn interval upper bound"),
        ("d17", "D17 explicit reference"),
        ("sun-btn", "sun-btn carve cover"),
        ("rabbit-btn", "rabbit-btn carve cover"),
        ("time-btn", "time-btn carve cover"),
    ]
    for token, description in expected_substrings:
        assert token in section_lower, (
            f"dashboard-and-cards.md (QS-232 section): missing `{token}` — {description} (AC-13)."
        )

    # All three carve-cover element ids named in the section.
    for cover_id in (
        "sun_btn_cover",
        "rabbit_btn_cover",
        "time_btn_cover",
    ):
        assert cover_id in section, (
            f"dashboard-and-cards.md (QS-232 section): missing the carve-cover element id `{cover_id}` (AC-13/AC-14)."
        )


def test_car_card_inside_disc_button_carve_covers():
    """QS-235 AC3 — the car's three inside-disc carve covers (sun /
    rabbit / time) are emitted via the shared `_ringCarveCover` helper
    (replacing the QS-232 inline `<circle id="…_cover">` literals):

    - the bottom-center `sun_btn_cover` uses the SHARED
      `RING_BOTTOM_CARVE_CX/CY/R` constants (the same set the duration
      cards use for `override_btn_cover`); the per-card `SUN_BTN_CARVE_*`
      duplicate is REMOVED (negative pin);
    - `rabbit_btn_cover` / `time_btn_cover` reuse the helper with the
      car-local `RABBIT_BTN_CARVE_*` / `TIME_BTN_CARVE_*` geometry
      (those constants survive — NAMES pinned, values user-tunable);
    - each call carries its render predicate via `show:` (`swPriority`,
      `e.force_now`, `(tNext && e.schedule)`), and the rendered cover
      ids are preserved;
    - the three calls appear AFTER the clipped `</g>` and BEFORE
      `${ringMarkup}` (so the covers paint on top of the soup, under
      the ring built by `_buildRingHTML`).
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # The car imports the shared bottom-center carve constants.
    assert (
        re.search(
            r"import\s*\{[^}]*RING_BOTTOM_CARVE_CX[^}]*RING_BOTTOM_CARVE_CY[^}]*RING_BOTTOM_CARVE_R[^}]*\}\s*from\s*['\"]\./shared/qs-card-base\.js['\"]",
            source,
            re.DOTALL,
        )
        is not None
    ), (
        "qs-car-card.js (QS-235 AC3): must import `RING_BOTTOM_CARVE_CX, "
        "RING_BOTTOM_CARVE_CY, RING_BOTTOM_CARVE_R` from "
        "`./shared/qs-card-base.js` (shared bottom-center carve geometry)."
    )

    # The per-card SUN_BTN_CARVE_* duplicate is removed.
    for legacy_const in ("SUN_BTN_CARVE_CX", "SUN_BTN_CARVE_CY", "SUN_BTN_CARVE_R"):
        assert re.search(rf"const\s+{legacy_const}\s*=", executable) is None, (
            f"qs-car-card.js (QS-235 AC3): the per-card `const "
            f"{legacy_const} = …` must be removed — the bottom-center "
            f"sun cover now uses the shared `RING_BOTTOM_CARVE_*` set."
        )

    # The car-local rabbit/time carve constants survive (NAMES only).
    for const_name in (
        "RABBIT_BTN_CARVE_CX",
        "RABBIT_BTN_CARVE_CY",
        "RABBIT_BTN_CARVE_R",
        "TIME_BTN_CARVE_CX",
        "TIME_BTN_CARVE_CY",
        "TIME_BTN_CARVE_R",
    ):
        assert re.search(rf"const\s+{const_name}\s*=\s*\d+\b", executable), (
            f"qs-car-card.js (QS-235 AC3): missing module-level "
            f"`const {const_name} = <integer>;` (car-local rabbit/time "
            f"carve geometry; only the NAME is pinned)."
        )

    # Three `_ringCarveCover` calls, each with the right geometry + id +
    # render predicate.
    covers = {
        "sun_btn_cover": ("RING_BOTTOM_CARVE_CX", "RING_BOTTOM_CARVE_CY", "RING_BOTTOM_CARVE_R", "swPriority"),
        "rabbit_btn_cover": ("RABBIT_BTN_CARVE_CX", "RABBIT_BTN_CARVE_CY", "RABBIT_BTN_CARVE_R", "e.force_now"),
        "time_btn_cover": ("TIME_BTN_CARVE_CX", "TIME_BTN_CARVE_CY", "TIME_BTN_CARVE_R", "(tNext && e.schedule)"),
    }
    for cover_id, (cx, cy, r, show) in covers.items():
        call_m = re.search(
            rf"this\._ringCarveCover\(\s*\{{([^}}]*id:\s*'{re.escape(cover_id)}'[^}}]*)\}}\s*\)",
            source,
        )
        assert call_m is not None, (
            f"qs-car-card.js (QS-235 AC3): missing a "
            f"`this._ringCarveCover({{ … id: '{cover_id}' … }})` call."
        )
        norm = re.sub(r"\s+", " ", call_m.group(1))
        for token in (f"cx: {cx}", f"cy: {cy}", f"r: {r}", f"show: {show}"):
            assert token in norm, (
                f"qs-car-card.js (QS-235 AC3): the `_ringCarveCover` "
                f"call for `{cover_id}` must pass `{token}`. Got: {norm.strip()}"
            )

    # Source order: cover calls must appear AFTER the clipped </g> AND
    # BEFORE `${ringMarkup}` (so they paint on top of the soup and under
    # the ring emitted by `_buildRingHTML`).
    sun_idx = source.find("id: 'sun_btn_cover'")
    rabbit_idx = source.find("id: 'rabbit_btn_cover'")
    time_idx = source.find("id: 'time_btn_cover'")
    # `${ringMarkup}` also appears in a code comment above the template;
    # `rfind` selects the actual SVG injection point (the last one).
    ringmarkup_idx = source.rfind("${ringMarkup}")
    assert ringmarkup_idx != -1, "qs-car-card.js (QS-235 AC2): missing the `${ringMarkup}` injection point."
    # Locate the OUTER closing </g> of the clipped group via `rfind`
    # scoped to the [clip_g_idx, ringmarkup_idx] window (so a nested
    # sparkle/lightning `</g>` can't falsely satisfy the order check).
    clip_g_idx = source.find('<g clip-path="url(#${electronClipId})"')
    assert clip_g_idx != -1, "qs-car-card.js (QS-235 AC2): missing the electron-soup clipped group anchor."
    clip_g_close_idx = source.rfind("</g>", clip_g_idx, ringmarkup_idx)
    assert clip_g_close_idx != -1, (
        "qs-car-card.js (QS-235): cannot find the outer closing `</g>` "
        "of the electron-soup clipped group within the expected "
        "[clip_g_idx, ringmarkup_idx] window."
    )

    for cover_name, cover_idx in (
        ("sun_btn_cover", sun_idx),
        ("rabbit_btn_cover", rabbit_idx),
        ("time_btn_cover", time_idx),
    ):
        assert cover_idx != -1, f"qs-car-card.js (QS-235 AC3): missing the `{cover_name}` cover call in source."
        assert clip_g_close_idx < cover_idx < ringmarkup_idx, (
            f"qs-car-card.js (QS-235 AC3): the `{cover_name}` cover call "
            f"must appear AFTER the clipped `</g>` (idx {clip_g_close_idx}) "
            f"AND BEFORE `${{ringMarkup}}` (idx {ringmarkup_idx}). Got "
            f"cover idx {cover_idx}."
        )


def test_car_card_adopts_shared_structural_helpers():
    """QS-235 AC9 — the car card consumes the shared structural helpers
    (the commonalization) instead of reimplementing them inline:

    - `_buildRingHTML` (AC2 ring builder; car passes `animPathId:
      'charge_anim'` + `extraDefs`),
    - `_wireTargetHandle` (AC1 drag; car passes `onCommit` /
      `pctToValue` / `valueToPct` / `onDragMove`),
    - `_ringCarveCover` (AC3 covers; geometry pinned in
      `test_car_card_inside_disc_button_carve_covers`),
    - `_wireGreenButton` (AC4 solar-priority),
    - `_wireResetButton` (AC4 reset),
    - `_wireTimePicker` (AC5 finish-time; car passes `onAfterCommit` +
      `resetButton`),
    - `_safeNumber` (AC6 guard-shaped sensor reads).

    The ~120-LOC inline pointer-drag block is gone — pinned negatively
    by the absence of its `createSVGPoint()` / `_targetDragValue` (the
    drag state now lives in the shared `_wireTargetHandle`). The
    single-definition rule (`test_no_duplicate_card_method_definitions`)
    proves none of the helpers are re-defined here.
    """
    car = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8")
    )

    for call in (
        "this._buildRingHTML(",
        "this._wireTargetHandle(",
        "this._ringCarveCover(",
        "this._wireGreenButton(",
        "this._wireResetButton(",
        "this._wireTimePicker(",
        "this._safeNumber(",
    ):
        assert call in car, (
            f"qs-car-card.js (QS-235 AC9): must call `{call}…)` — the car "
            f"adopts the shared helper rather than reimplementing it inline."
        )

    # AC2 — car-specific wiring params on the `_buildRingHTML` call.
    assert "animPathId: 'charge_anim'" in car, (
        "qs-car-card.js (QS-235 AC2): `_buildRingHTML` call must pass "
        "`animPathId: 'charge_anim'` (the car's dashed-arc id)."
    )
    assert "extraDefs" in car, (
        "qs-car-card.js (QS-235 AC2): `_buildRingHTML` call must pass "
        "`extraDefs` (the car's disabled/fault/stale gradients + clipPath)."
    )

    # AC1 — the four drag callbacks on the `_wireTargetHandle` call.
    for param in ("onCommit:", "pctToValue:", "valueToPct:", "onDragMove:"):
        assert param in car, (
            f"qs-car-card.js (QS-235 AC1): `_wireTargetHandle` call must "
            f"pass `{param}` (the car's select-commit / %↔kWh mapping / "
            f"live-label callbacks)."
        )

    # AC5 — the finish-time generalization params.
    for param in ("onAfterCommit:", "resetButton:"):
        assert param in car, (
            f"qs-car-card.js (QS-235 AC5): `_wireTimePicker` call must "
            f"pass `{param}` (the car's schedule-press / reset-button)."
        )

    # AC1 — the inline pointer-drag block is gone (the helper owns it).
    assert "createSVGPoint()" not in car, (
        "qs-car-card.js (QS-235 AC1): the inline pointer-drag block must "
        "be removed — `createSVGPoint()` belongs to the shared "
        "`_wireTargetHandle` now."
    )
    assert "_targetDragValue" not in car, (
        "qs-car-card.js (QS-235 AC1): the inline drag state "
        "`_targetDragValue` must be gone — it lives in the shared "
        "`_wireTargetHandle`."
    )


def test_car_card_ac6_leaves_non_sensor_reads_untouched():
    """QS-235 review-fix #01 SF4 — AC6's NEGATIVE half. AC6 swaps ONLY
    the guard-shaped sensor reads (`sPower`, `sCurrentInputedEnergy`) to
    `_safeNumber`. The config read (`Number(e.car_battery_capacity_kwh)`),
    the regex captures (`Number(m[1])`), and the DOM reads
    (`Number(hourSel?.value …)`, `Number(el.getAttribute('data-pct'))`)
    must survive as raw `Number(...)` coercions — `_safeNumber` expects a
    sensor object `{state}`, so wrapping a scalar/regex/DOM read would be
    a real bug (and silently coerce wrong). The positive half is pinned
    by `test_car_card_adopts_shared_structural_helpers`.
    """
    import re

    car = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8")
    )

    # The non-sensor reads survive as raw `Number(...)` coercions.
    for survivor in (
        "Number(e.car_battery_capacity_kwh)",  # config scalar
        "Number(m[1])",                        # regex capture
        "Number(hourSel?.value",               # DOM value read
        "Number(el.getAttribute('data-pct'))",  # DOM attribute read
    ):
        assert survivor in car, (
            f"qs-car-card.js (QS-235 AC6 negative half): the non-sensor "
            f"read `{survivor}` must survive untouched — AC6 only swaps "
            f"guard-shaped SENSOR reads to `_safeNumber`."
        )

    # And none of them is wrapped in `_safeNumber` (which takes a sensor
    # object, not a scalar / regex match / DOM value).
    for forbidden in (
        "_safeNumber(e.car_battery_capacity_kwh",
        "_safeNumber(configBatteryCapacity",
        "_safeNumber(m[1]",
        "_safeNumber(hourSel",
        "_safeNumber(minSel",
        "_safeNumber(el.getAttribute",
    ):
        assert forbidden not in car, (
            f"qs-car-card.js (QS-235 AC6 negative half): `{forbidden}…)` "
            f"must NOT exist — `_safeNumber` expects a sensor `{{state}}` "
            f"object, not a scalar/regex/DOM read."
        )

    # Positive corroboration: every `_safeNumber` call in the car targets
    # a sensor object (`sPower` / `sCurrentInputedEnergy`), nothing else.
    safe_args = re.findall(r"_safeNumber\(\s*([A-Za-z_][\w.]*)", car)
    assert safe_args, "qs-car-card.js (QS-235 AC6): expected `_safeNumber` calls in the car."
    for arg in safe_args:
        assert arg in ("sPower", "sCurrentInputedEnergy"), (
            f"qs-car-card.js (QS-235 AC6): `_safeNumber` should only wrap "
            f"sensor objects (`sPower` / `sCurrentInputedEnergy`); got "
            f"`{arg}`."
        )


def test_car_card_handle_label_round_trip_invariant():
    """QS-235 AC9 — the car's ring handle shows the TRUE (unclamped)
    target via `_buildRingHTML`'s `handleLabel`, while the handle
    POSITION is clamped via `pctToDeg`. So an out-of-range target pins
    the handle at the ring top but the label still reads the real
    number (the same pin-at-top-but-number-correct invariant the
    duration cards get from `pctToHours(handlePct)`). The car supplies
    energy-/percent-mode label text instead of hours.
    """
    import re

    car = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8")
    )

    # The handle TEXT label is the true target (mode-switched), passed
    # via `_buildRingHTML`'s `handleLabel` param — NOT a clamped %.
    assert re.search(r"handleLabel:\s*useEnergyMode\s*\?", car), (
        "qs-car-card.js (QS-235 AC9): `_buildRingHTML` must receive a "
        "mode-switched `handleLabel: useEnergyMode ? … : …`."
    )
    assert "parseTargetEnergy(target)" in car, (
        "qs-car-card.js (QS-235 AC9): the energy-mode handle label must "
        "show the parsed true target energy (`parseTargetEnergy(target)`)."
    )
    assert "this._fmt(targetPct ?? soc)" in car, (
        "qs-car-card.js (QS-235 AC9): the percent-mode handle label must "
        "show the true target percent (`this._fmt(targetPct ?? soc)`)."
    )

    # The handle POSITION is clamped via `pctToDeg` (which clamps the
    # percentage to [0,100]); the position derives from that angle.
    assert re.search(r"handleDeg\s*=\s*pctToDeg\(\s*handlePct", car), (
        "qs-car-card.js (QS-235 AC9): the handle angle must be "
        "`handleDeg = pctToDeg(handlePct, …)` (position clamped to [0,100])."
    )
    assert re.search(r"handlePos\s*=\s*polar\([^)]*handleDeg", car), (
        "qs-car-card.js (QS-235 AC9): `handlePos` must derive from the "
        "clamped `handleDeg` via `polar(…, handleDeg)`."
    )


def test_car_card_faulted_dash_uses_fault_gradient():
    """QS-235 review-fix #01 SF1 — a car that is faulted WHILE still
    drawing current must render the moving dashed arc with the SAME
    fault gradient as the static progress arc (pre-refactor behavior:
    `stroke="url(#${isFaulted ? gradFaultId : gradChargeId})"`).

    The shared `_buildRingHTML` dash stroke is driven by an `animGradId`
    param that defaults to `gradRunningId` (so the four duration callers
    stay byte-identical). The car passes the fault-aware
    `animGradId: isFaulted ? gradFaultId : gradChargeId`, which restores
    the dash↔static-arc agreement AND keeps `gradFaultId` referenced by
    a stroke (not just emitted as a dead `<linearGradient>` in
    `extraDefs`).
    """
    import re

    base = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "shared" / "qs-card-base.js").read_text(encoding="utf-8")
    )
    car = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8")
    )

    # The shared dash anim stroke is parameterised by `animGradId`,
    # defaulting to `gradRunningId` (duration cards unchanged).
    assert 'stroke="url(#${animGradId})"' in base, (
        "qs-card-base.js (SF1): `_buildRingHTML`'s dash anim `<path>` "
        'must use `stroke="url(#${animGradId})"` so the dash gradient '
        "is caller-selectable."
    )
    assert re.search(r"animGradId\s*=\s*gradRunningId", base), (
        "qs-card-base.js (SF1): `animGradId` must default to "
        "`gradRunningId` so the duration callers stay byte-identical."
    )

    # The car drives `animGradId` off the fault state, matching the
    # static arc's `activeGradId` selection (which also uses gradFaultId
    # when faulted).
    assert re.search(
        r"animGradId:\s*isFaulted\s*\?\s*gradFaultId\s*:\s*gradChargeId",
        car,
    ), (
        "qs-car-card.js (SF1): the `_buildRingHTML` call must pass "
        "`animGradId: isFaulted ? gradFaultId : gradChargeId` so the "
        "faulted-while-charging dash matches the static fault arc."
    )
    # `gradFaultId` is referenced by a stroke selector, not orphaned.
    assert "gradFaultId" in car, (
        "qs-car-card.js (SF1): `gradFaultId` must remain referenced "
        "(static arc + fault-aware dash), not a dead gradient."
    )


# ============================================================================
# === QS-232 review-fix #01 tests ===
# Additional regression pins from the post-PR review pass. Each
# guards a specific edge case or bugfix from
# `docs/stories/QS-232.story_review_fix_#01.md`.
# ============================================================================


def test_car_card_render_guards_water_base_y_against_nan():
    """QS-232 review-fix #01 #3: `_waterBaseY` must NOT become NaN
    if `soc` is NaN (a non-numeric `current_inputed_energy` state
    in `useEnergyMode`, or a corrupted percent reading). Without a
    guard, the initial `_generateWavePath()` call at `_render()` time
    emits literal `"NaN"` tokens in the `d` attribute and no soup
    paints until the next valid hass push. Pinned via a structural
    `Number.isFinite` check on either `soc` or `progressRatio`
    before the `_waterBaseY` assignment.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # The `_waterBaseY` assignment must be preceded by (or wrap with)
    # a `Number.isFinite` guard against NaN soc. Accept either:
    # - `Number.isFinite(soc)` ternary in the progressRatio
    #   computation, OR
    # - `Number.isFinite(progressRatio)` guard, OR
    # - `Number.isFinite(this._waterBaseY)` post-assignment guard.
    waterbase_guard_re = re.compile(
        r"Number\.isFinite\s*\(\s*"
        r"(?:soc|progressRatio|this\._waterBaseY|rawWaterBaseY)\s*\)",
    )
    assert waterbase_guard_re.search(executable), (
        "qs-car-card.js (review-fix #01 #3): `_waterBaseY` "
        "assignment must be guarded by `Number.isFinite(...)` against "
        "NaN `soc`. Without this, a non-numeric energy/percent state "
        "poisons the initial wave path's `d` attribute with literal "
        "`NaN` tokens."
    )


def test_car_card_initial_wave_path_skips_nan_water_base_y():
    """QS-232 review-fix #01 #5: belt-and-braces guard at the
    `initialWavePath` call site. Even with the upstream
    `_waterBaseY` guard, the call-site itself must check
    `Number.isFinite(this._waterBaseY)` before invoking
    `_generateWavePath(...)`, so a future regression to the upstream
    guard can't silently poison the emitted SVG `d` attribute.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # The `initialWavePath = ...` line must use a ternary on
    # `Number.isFinite(this._waterBaseY)` so the empty-string
    # fallback path exists.
    call_site_guard_re = re.compile(
        r"const\s+initialWavePath\s*=\s*"
        r"Number\.isFinite\s*\(\s*this\._waterBaseY\s*\)\s*\?\s*"
        r"this\._generateWavePath\s*\(",
        re.DOTALL,
    )
    assert call_site_guard_re.search(executable), (
        "qs-car-card.js (review-fix #01 #5): the `initialWavePath` "
        "call site must wrap `_generateWavePath(...)` in "
        "`Number.isFinite(this._waterBaseY) ? ... : ''` so the "
        "emitted `d` attribute is never poisoned by a regression "
        "upstream."
    )


def test_car_card_setConfig_primes_animation_state():
    """QS-232 review-fix #01 #4: `_needsAnimationPrime` must be set
    in `setConfig()` so the FIRST `_render()` (which runs from
    `setConfig`, BEFORE `connectedCallback`) primes
    `_currentAmplitude` / `_currentSpeed` / `_currentColorMix`
    against the current `_charging` state. Without this, the first
    paint always uses the idle palette regardless of `_charging`,
    and a subsequent RAF lerp visibly transitions over ~1.5 s
    (the boot-transient avoidance is moot).
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    set_config_body = _extract_js_function_body(executable, r"setConfig\s*\(\s*config\s*\)\s*")
    assert set_config_body is not None, (
        "qs-car-card.js (review-fix #01 #4): `setConfig(config)` "
        "method not found — the test file structure has drifted."
    )
    assert re.search(
        r"this\._needsAnimationPrime\s*=\s*true",
        set_config_body,
    ), (
        "qs-car-card.js (review-fix #01 #4): `setConfig()` must set "
        "`this._needsAnimationPrime = true` so the first render "
        "(which fires from `setConfig` BEFORE `connectedCallback`) "
        "primes the wave animation state against `_charging`."
    )


def test_car_card_time_btn_render_and_cover_gates_match():
    """QS-232 review-fix #01 #6 (QS-235 form): `time_btn` and
    `time_btn_cover` must share the same render gate. If the button
    rendered under a different predicate than its cover, the button
    face could draw over un-carved soup. QS-235 moved the cover to a
    `_ringCarveCover({ … show: <gate> })` call, so the cover gate is
    now the `show:` value; this pin extracts the button ternary
    predicate and the cover `show:` predicate and asserts equality.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()

    # The `time_btn` <div> render gate (the ternary predicate).
    btn_m = re.search(r"(\([^)]*\))\s*\?\s*`?\s*<div\s+id=\"time_btn\"", source)
    assert btn_m is not None, (
        "qs-car-card.js (review-fix #01 #6): the `time_btn` <div> "
        'must be wrapped in a `(predicate) ? `<div id="time_btn" …>` '
        ": ''` ternary so its render gate aligns with the "
        "`time_btn_cover` carve gate."
    )

    # The `time_btn_cover` carve gate (the `_ringCarveCover` `show:`).
    cover_m = re.search(
        r"this\._ringCarveCover\(\s*\{[^}]*id:\s*'time_btn_cover'[^}]*show:\s*([^}]+?)\s*\}\s*\)",
        source,
    )
    assert cover_m is not None, (
        "qs-car-card.js (QS-235 AC3): missing the `time_btn_cover` "
        "`_ringCarveCover({ … show: <gate> })` call."
    )

    def _normalise(s: str) -> str:
        return re.sub(r"\s+", "", s)

    btn_predicate = _normalise(btn_m.group(1))
    cover_predicate = _normalise(cover_m.group(1))
    assert btn_predicate == cover_predicate, (
        f"qs-car-card.js (review-fix #01 #6): the `time_btn` render "
        f"gate `{btn_m.group(1)}` and the `time_btn_cover` carve "
        f"`show:` gate `{cover_m.group(1)}` must be equal "
        f"(whitespace-normalised). Asymmetric gates allow the button "
        f"face to draw over un-covered soup."
    )


def test_car_card_connectedCallback_reprimes_on_state_flip():
    """QS-232 review-fix #01 #7: `connectedCallback()` must
    re-prime `_needsAnimationPrime` when the `_charging` state
    flipped during a detached interval. Otherwise, the
    lazy-init guard in `_startAnimation()` (which preserves
    `_currentAmplitude` etc. across detach/re-attach) skips the
    prime block and the soup visibly lerps on reconnect rather
    than snapping. Pinned structurally: the `connectedCallback`
    body must reference both `_charging` and `_currentColorMix`
    inside a re-prime guard.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    cb_body = _extract_js_function_body(executable, r"connectedCallback\s*\(\s*\)\s*")
    assert cb_body is not None, "qs-car-card.js (review-fix #01 #7): `connectedCallback()` not found."
    # The re-prime guard must reference `this._charging` AND
    # `this._currentColorMix` (the implied state derivation).
    assert re.search(r"this\._charging", cb_body), (
        "qs-car-card.js (review-fix #01 #7): `connectedCallback` "
        "must reference `this._charging` to compute the implied "
        "state for re-prime."
    )
    assert re.search(r"this\._currentColorMix", cb_body), (
        "qs-car-card.js (review-fix #01 #7): `connectedCallback` "
        "must reference `this._currentColorMix` to detect a "
        "state-flip during the detached interval."
    )
    assert re.search(
        r"this\._needsAnimationPrime\s*=\s*true",
        cb_body,
    ), (
        "qs-car-card.js (review-fix #01 #7): `connectedCallback` "
        "must conditionally set `this._needsAnimationPrime = true` "
        "when `_charging` flipped during detach."
    )


def test_car_card_last_amplitude_uses_finite_guard_not_nullish():
    """QS-232 review-fix #01 #8: `this._lastAmplitude` must be
    assigned via `Number.isFinite(this._currentAmplitude) ? ... :
    CALM_AMP` instead of `this._currentAmplitude ?? CALM_AMP`. The
    nullish coalescing operator does NOT trap NaN: if
    `_currentAmplitude` ever becomes NaN, `_lastAmplitude` becomes
    NaN, the regen check `ampDelta = Math.abs(_ - NaN) = NaN` is
    always `< threshold`, and the wave-path attribute is never
    regenerated for the lifetime of the card.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Positive pin: the explicit finite-check form.
    finite_form_re = re.compile(
        r"this\._lastAmplitude\s*=\s*"
        r"Number\.isFinite\s*\(\s*this\._currentAmplitude\s*\)\s*\?\s*"
        r"this\._currentAmplitude\s*:\s*CALM_AMP",
    )
    assert finite_form_re.search(executable), (
        "qs-car-card.js (review-fix #01 #8): `_lastAmplitude` must "
        "be assigned via `Number.isFinite(this._currentAmplitude) ? "
        "this._currentAmplitude : CALM_AMP` so a NaN amplitude can't "
        "permanently freeze the wave-path regen."
    )

    # Negative pin: the old `??` form must not be present for
    # `_lastAmplitude`. Pinned narrowly so unrelated `??` usages
    # (e.g., `_currentColorMix ?? 0`) are not flagged.
    forbidden_re = re.compile(
        r"this\._lastAmplitude\s*=\s*this\._currentAmplitude\s*\?\?",
    )
    assert not forbidden_re.search(executable), (
        "qs-car-card.js (review-fix #01 #8): the old "
        "`this._lastAmplitude = this._currentAmplitude ?? CALM_AMP` "
        "form must be replaced — `??` does not trap NaN."
    )


def test_car_card_lightning_segments_constant_removed():
    """QS-232 review-fix #01 #16: `LIGHTNING_SEGMENTS = 3` was
    declared at module scope but never referenced; the lightning
    path hardcoded 3 `L` segments rather than looping driven by
    the constant. Dead code; removed. The pin is scoped to
    EXECUTABLE code (comments stripped) so a `// removed …`
    historical comment is allowed.
    """
    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)
    assert "LIGHTNING_SEGMENTS" not in executable, (
        "qs-car-card.js (review-fix #01 #16): `LIGHTNING_SEGMENTS` "
        "must be removed from EXECUTABLE code — declared at module "
        "scope but never referenced (the lightning path hardcodes "
        "its 3 `L` segments). Historical references in comments "
        "are allowed."
    )


def test_car_card_dashed_arc_comment_no_stale_m4_reference():
    """QS-232 review-fix #01 #17: the dashed-arc inline comment
    used to reference "M4" (the deleted `_charging`-gated
    milestone). Should be updated to reflect the QS-232
    continuous-RAF model. Pinned narrowly to the dashed-arc block,
    not to all M4 references in the file.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()

    # Find the dashed-arc comment block (it precedes the
    # `_animOffset = ...` / `charge_anim` block).
    dashed_arc_re = re.compile(
        r"^\s*//[^\n]*PRESERVED\s+from\s+M4[^\n]*$",
        re.MULTILINE,
    )
    assert dashed_arc_re.search(source) is None, (
        "qs-car-card.js (review-fix #01 #17): the stale "
        '"PRESERVED from M4" comment in the dashed-arc block '
        "must be rewritten to reflect the QS-232 continuous-RAF "
        "model. The M4 milestone is no longer the controlling "
        "lifecycle anchor."
    )


# ============================================================================
# === QS-232 review-fix #02 tests ===
# Round-2 follow-ups. The headline regression: review-fix #01 #4 armed
# `_needsAnimationPrime = true` in `setConfig`, then `_render`
# consumed the flag (setting `_currentAmplitude`/`_currentSpeed`/
# `_currentColorMix`) BEFORE `connectedCallback` fired. The lazy-init
# guard in `_startAnimation()` is `if (this._currentAmplitude == null)`
# — which is now FALSE — so the rest of the lazy-init block
# (`_wavePhase`, `_nextSparkleAt`, `_nextLightningAt`, `_sparkles`,
# `_lightningBolts`) is skipped, leaving them undefined. First RAF
# tick poisons `_wavePhase` with NaN; the sparkle / lightning spawn
# loops never enter. User-visible symptom: "no sparkles at all".
# Fix: each RAF-state field guards itself individually.
# ============================================================================


def test_car_card_first_raf_tick_initialises_wave_phase_unconditionally():
    """QS-232 review-fix #02 #1: `_wavePhase`, `_nextSparkleAt`,
    `_nextLightningAt`, `_sparkles`, `_lightningBolts` must each be
    seeded by `_startAnimation()` regardless of whether
    `_currentAmplitude` has already been primed by `_render()`.
    The previous form bundled all five inside
    `if (this._currentAmplitude == null) { ... }`, which the
    prime-block-in-`_render` consumed — leaving the other five
    undefined and poisoning the first RAF tick with NaN.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    body = _extract_js_function_body(executable, r"_startAnimation\s*\(\s*\)\s*")
    assert body is not None, "qs-car-card.js (review-fix #02 #1): `_startAnimation()` body not found."

    # The five RAF-state fields must each have their OWN
    # `if (this.X == null) this.X = ...` guard so that consumption
    # of `_currentAmplitude` by the prime block doesn't strand them.
    for field, default_pattern in [
        ("_wavePhase", r"=\s*0"),
        ("_nextSparkleAt", r"=\s*0"),
        ("_nextLightningAt", r"=\s*LIGHTNING_SPAWN_MIN_S"),
        ("_sparkles", r"=\s*\[\s*\]"),
        ("_lightningBolts", r"=\s*\[\s*\]"),
    ]:
        per_field_guard = re.compile(
            rf"if\s*\(\s*this\.{field}\s*==\s*null\s*\)\s*"
            rf"this\.{field}\s*{default_pattern}",
        )
        assert per_field_guard.search(body), (
            f"qs-car-card.js (review-fix #02 #1): `_startAnimation` "
            f"must seed `this.{field}` under its OWN `== null` guard "
            f"(not bundled inside the `_currentAmplitude == null` "
            f"block) so the prime path can't strand it."
        )


def test_car_card_initial_paint_uses_finite_guards_not_nullish():
    """QS-232 review-fix #02 #2: `initialAmp` / `initialColorMix`
    must be derived via `Number.isFinite(...)` checks rather than
    `??`. `??` only traps null/undefined, so a runtime NaN (from a
    poisoned lerp accumulator) propagates into the initial wave path
    `d` attribute and opacity attribute strings. Mirrors the fix-#01-#8
    finite-guard shape for `_lastAmplitude`.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Positive: explicit finite check for initialAmp.
    initial_amp_re = re.compile(
        r"const\s+initialAmp\s*=\s*"
        r"Number\.isFinite\s*\(\s*this\._currentAmplitude\s*\)\s*\?\s*"
        r"this\._currentAmplitude\s*:\s*CALM_AMP",
    )
    assert initial_amp_re.search(executable), (
        "qs-car-card.js (review-fix #02 #2): `initialAmp` must be "
        "computed via `Number.isFinite(this._currentAmplitude) ? "
        "this._currentAmplitude : CALM_AMP` — `??` does not trap "
        "NaN."
    )

    # Positive: explicit finite check for initialColorMix.
    initial_mix_re = re.compile(
        r"const\s+initialColorMix\s*=\s*"
        r"Number\.isFinite\s*\(\s*this\._currentColorMix\s*\)\s*\?\s*"
        r"this\._currentColorMix\s*:\s*0",
    )
    assert initial_mix_re.search(executable), (
        "qs-car-card.js (review-fix #02 #2): `initialColorMix` must "
        "be computed via `Number.isFinite(this._currentColorMix) ? "
        "this._currentColorMix : 0` — `??` does not trap NaN."
    )

    # Negative pins: the old `??` shapes for the same identifiers
    # must not be present.
    for old_form in (
        r"const\s+initialAmp\s*=\s*this\._currentAmplitude\s*\?\?",
        r"const\s+initialColorMix\s*=\s*this\._currentColorMix\s*\?\?",
    ):
        assert not re.search(old_form, executable), (
            "qs-car-card.js (review-fix #02 #2): the old `??` form "
            f"matching /{old_form}/ must be replaced with a "
            "`Number.isFinite(...)` ternary."
        )


def test_car_card_setConfig_restarts_animation_when_connected():
    """QS-232 review-fix #02 #3: `setConfig()` must re-invoke
    `_startAnimation()` at its tail if the element is already
    connected and RAF hasn't started yet. Otherwise — in the
    (uncommon) lifecycle where `connectedCallback` fires before
    `setConfig` — RAF never starts because `_startAnimation` bails
    early on `!_root` and nothing re-calls it once `_root` is set.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    set_config_body = _extract_js_function_body(executable, r"setConfig\s*\(\s*config\s*\)\s*")
    assert set_config_body is not None, "qs-car-card.js (review-fix #02 #3): `setConfig(config)` method not found."
    # Re-entry guard: must reference both `isConnected` and call
    # `_startAnimation()` inside a guarded block.
    reentry_re = re.compile(
        r"if\s*\(\s*this\.isConnected[^)]*_animRaf[^)]*\)\s*\{[^}]*"
        r"this\._startAnimation\s*\(\s*\)",
        re.DOTALL,
    )
    assert reentry_re.search(set_config_body), (
        "qs-car-card.js (review-fix #02 #3): `setConfig` must "
        "contain a tail guard "
        "`if (this.isConnected && this._animRaf == null) "
        "this._startAnimation();` so RAF starts even when "
        "`connectedCallback` fired before `setConfig`."
    )


def test_car_card_connectedCallback_reprime_calls_render():
    """QS-232 review-fix #02 #4: the reprime branch in
    `connectedCallback()` (which sets `_needsAnimationPrime = true`
    when `_charging` flipped during a detached interval) must also
    call `this._render()` so the flag is consumed immediately. The
    previous form left the flag set until the next `set hass` push,
    during which the RAF lerps visibly from the stale `_currentColorMix`
    toward the new target.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    cb_body = _extract_js_function_body(executable, r"connectedCallback\s*\(\s*\)\s*")
    assert cb_body is not None, "qs-car-card.js (review-fix #02 #4): `connectedCallback()` not found."
    # The reprime block must contain BOTH the flag set AND the
    # `_render()` call, adjacent or with whitespace between them.
    # NB: we don't use an `if (... _charging ...)` regex anchor here
    # because `_charging` may live in a `const targetMix = ...`
    # statement BEFORE the `if`, not inside the `if` predicate.
    # Adjacency check is sufficient: prime + render in the same
    # block.
    adjacent_re = re.compile(
        r"this\._needsAnimationPrime\s*=\s*true\s*;\s*"
        r"this\._render\s*\(\s*\)",
    )
    assert adjacent_re.search(cb_body), (
        "qs-car-card.js (review-fix #02 #4): `connectedCallback` "
        "must call `this._render()` immediately after setting "
        "`this._needsAnimationPrime = true` (adjacent statements) "
        "so the flag is consumed on the next paint, not on the "
        "next hass push."
    )


def test_car_card_last_amplitude_in_raf_uses_finite_guard():
    """QS-232 review-fix #02 #5: review-fix #01 #8 patched only the
    POST-innerHTML write to `_lastAmplitude` (at the end of
    `_render`). The IN-RAF write at the wave-regen block was left
    unguarded — if `_currentAmplitude` ever becomes NaN mid-frame,
    NaN propagates into `_lastAmplitude`, the regen check freezes,
    and the wave path is never regenerated. Both the in-RAF read
    AND the in-RAF write must guard against NaN.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    raf_body = _extract_js_function_body(executable, r"_startAnimation\s*\(\s*\)\s*")
    assert raf_body is not None, "qs-car-card.js (review-fix #02 #5): `_startAnimation()` body not found."

    # In-RAF read: the `ampDelta = ...` ternary must treat both null
    # AND non-finite as "first frame". The source uses a parenthesised
    # disjunction `(this._lastAmplitude == null || !Number.isFinite(
    # this._lastAmplitude))` so the regex allows an optional leading
    # `(`.
    ampdelta_finite_re = re.compile(
        r"ampDelta\s*=\s*\(?\s*"
        r"(?:!?\s*Number\.isFinite\s*\(\s*this\._lastAmplitude\s*\)"
        r"|this\._lastAmplitude\s*==\s*null\s*\|\|\s*!\s*Number\.isFinite)",
    )
    assert ampdelta_finite_re.search(raf_body), (
        "qs-car-card.js (review-fix #02 #5): the in-RAF "
        "`ampDelta = …` computation must guard against non-finite "
        "`_lastAmplitude` (use `Number.isFinite(...)` rather than "
        "`== null` alone, since NaN can be stored)."
    )

    # In-RAF write: count `_lastAmplitude = ` occurrences. The shape
    # with the finite guard must appear AT LEAST TWICE (in-RAF write
    # at line ~345, post-innerHTML write at the end of _render).
    finite_writes = re.findall(
        r"this\._lastAmplitude\s*=\s*"
        r"Number\.isFinite\s*\(\s*this\._currentAmplitude\s*\)\s*\?\s*"
        r"this\._currentAmplitude\s*:\s*CALM_AMP",
        executable,
    )
    assert len(finite_writes) >= 2, (
        f"qs-car-card.js (review-fix #02 #5): expected at least 2 "
        f"`_lastAmplitude = Number.isFinite(this._currentAmplitude) ? "
        f"this._currentAmplitude : CALM_AMP` assignments (one in-RAF "
        f"at the wave-regen block, one post-innerHTML in `_render`). "
        f"Found {len(finite_writes)}."
    )


def test_car_card_initial_paint_opacity_is_clamped():
    """QS-232 review-fix #02 #6: review-fix #01 #18 clamped the
    RAF-step opacity emission via a local `mix` variable, but the
    INITIAL-PAINT opacity emission (`initialIdleOpacity` /
    `initialChargeOpacity`) was left unclamped. After RAF has run
    long enough for `_currentColorMix` to drift slightly outside
    `[0, 1]`, the next `_render` emits `"-0.000"` opacity tokens.
    The initial-paint sites must clamp too.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # Either form is acceptable:
    # (a) a local `mix` / `initialMix` = `Math.max(0, Math.min(1,
    #     initialColorMix))` used in both opacity emissions, OR
    # (b) the Math.max/min inline at both emission sites.
    has_local_clamp = re.search(
        r"const\s+(?:mix|initialMix)\s*=\s*Math\.max\s*\(\s*0\s*,\s*"
        r"Math\.min\s*\(\s*1\s*,\s*initialColorMix",
        executable,
    )
    has_inline_clamp_idle = re.search(
        r"const\s+initialIdleOpacity\s*=\s*"
        r"\(\s*1\s*-\s*Math\.max\s*\(\s*0\s*,\s*Math\.min\s*\(\s*1\s*,\s*initialColorMix",
        executable,
    )
    has_inline_clamp_charge = re.search(
        r"const\s+initialChargeOpacity\s*=\s*"
        r"Math\.max\s*\(\s*0\s*,\s*Math\.min\s*\(\s*1\s*,\s*initialColorMix",
        executable,
    )
    assert has_local_clamp or (has_inline_clamp_idle and has_inline_clamp_charge), (
        "qs-car-card.js (review-fix #02 #6): the initial-paint "
        "opacity emission must clamp `initialColorMix` to `[0, 1]` "
        "before `.toFixed(3)` — either via a `const mix = "
        "Math.max(0, Math.min(1, initialColorMix));` (or `initialMix`) "
        "local OR via the Math.max/min inline at both "
        "`initialIdleOpacity` and `initialChargeOpacity` sites."
    )


def test_car_card_rabbit_btn_render_and_cover_gates_match():
    """QS-232 review-fix #02 #7 (QS-235 form): `rabbit_btn` and
    `rabbit_btn_cover` must share the same render gate (symmetric to
    the `time_btn` pin) — otherwise the button draws over un-carved
    soup. QS-235 moved the cover to a `_ringCarveCover({ … show:
    <gate> })` call, so the cover gate is the `show:` value.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()

    # The `rabbit_btn` <div> render gate. Walk BACKWARD from the marker
    # to the enclosing `${` template opener and read up to the `?` (the
    # rabbit gate `e.force_now` has no parens, so a parens-only regex
    # wouldn't match it).
    def _extract_template_predicate(target_marker: str) -> str | None:
        target_idx = source.find(target_marker)
        if target_idx == -1:
            return None
        dollar_idx = source.rfind("${", 0, target_idx)
        if dollar_idx == -1:
            return None
        question_idx = source.find("?", dollar_idx, target_idx)
        if question_idx == -1:
            return None
        return source[dollar_idx + 2 : question_idx].strip()

    btn_predicate = _extract_template_predicate('<div id="rabbit_btn"')
    assert btn_predicate is not None, (
        "qs-car-card.js (review-fix #02 #7): the `rabbit_btn` <div> "
        'must be wrapped in a `${predicate ? `<div id="rabbit_btn" …>` '
        ": ''}` ternary so its render gate aligns with the "
        "`rabbit_btn_cover` carve gate."
    )

    # The `rabbit_btn_cover` carve gate (the `_ringCarveCover` `show:`).
    cover_m = re.search(
        r"this\._ringCarveCover\(\s*\{[^}]*id:\s*'rabbit_btn_cover'[^}]*show:\s*([^}]+?)\s*\}\s*\)",
        source,
    )
    assert cover_m is not None, (
        "qs-car-card.js (QS-235 AC3): missing the `rabbit_btn_cover` "
        "`_ringCarveCover({ … show: <gate> })` call."
    )

    def _normalise(s: str) -> str:
        return re.sub(r"\s+", "", s)

    assert _normalise(btn_predicate) == _normalise(cover_m.group(1)), (
        f"qs-car-card.js (review-fix #02 #7): `rabbit_btn` render "
        f"gate `{btn_predicate}` and `rabbit_btn_cover` carve `show:` "
        f"gate `{cover_m.group(1)}` must be equal (whitespace-normalised)."
    )


def test_car_card_mix_clamp_traps_nan():
    """QS-232 review-fix #02 #9: the RAF-step `const mix = Math.max(0,
    Math.min(1, this._currentColorMix));` clamp doesn't trap NaN
    (since `Math.max(0, Math.min(1, NaN)) === NaN`). Wrap with a
    `Number.isFinite` ternary so NaN falls back to 0.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    mix_finite_re = re.compile(
        r"const\s+mix\s*=\s*"
        r"Number\.isFinite\s*\(\s*this\._currentColorMix\s*\)\s*\?\s*"
        r"Math\.max\s*\(\s*0\s*,\s*Math\.min\s*\(\s*1\s*,\s*this\._currentColorMix\s*\)\s*\)\s*:\s*0",
    )
    assert mix_finite_re.search(executable), (
        "qs-car-card.js (review-fix #02 #9): the RAF-step "
        "`const mix = ...` clamp must be wrapped in a "
        "`Number.isFinite(this._currentColorMix)` ternary so NaN "
        "falls back to 0 (Math.max/min don't trap NaN)."
    )


def test_car_card_wave_els_retry_on_null():
    """QS-232 review-fix #02 #10: the `_waveEls` lazy-resolve must
    retry if EITHER slot is null. The previous form
    `if (!this._waveEls) { this._waveEls = [idle, charge]; }` caches
    `[null, null]` (which is truthy) permanently if both lookups
    miss during a transient detach.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    retry_re = re.compile(
        r"if\s*\(\s*"
        r"!\s*this\._waveEls"
        r"\s*\|\|\s*!\s*this\._waveEls\s*\[\s*0\s*\]"
        r"\s*\|\|\s*!\s*this\._waveEls\s*\[\s*1\s*\]"
        r"\s*\)",
    )
    assert retry_re.search(executable), (
        "qs-car-card.js (review-fix #02 #10): the `_waveEls` "
        "lazy-resolve guard must be "
        "`if (!this._waveEls || !this._waveEls[0] || "
        "!this._waveEls[1])` so a transient `[null, null]` cache is "
        "retried on subsequent ticks."
    )


def test_car_card_has_valid_base_uses_number_is_finite():
    """QS-232 review-fix #02 #11: `hasValidBase` must use
    `Number.isFinite(waterBaseY)` (not `!Number.isNaN(...)`), so a
    spurious `Infinity` doesn't slip through the guard. Asymmetric
    with fix-#01-#3 / fix-#01-#5 which already use `Number.isFinite`.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    finite_re = re.compile(
        r"const\s+hasValidBase\s*=\s*"
        r"waterBaseY\s*!=\s*null\s*&&\s*"
        r"Number\.isFinite\s*\(\s*waterBaseY\s*\)",
    )
    assert finite_re.search(executable), (
        "qs-car-card.js (review-fix #02 #11): `const hasValidBase = "
        "waterBaseY != null && Number.isFinite(waterBaseY);` — "
        "`!Number.isNaN(...)` is asymmetric with the other guards "
        "in the file and lets `Infinity` through."
    )


def test_car_card_render_snapshot_defensively_coerces_undefined_sparkles():
    """QS-232 review-fix #02 #18: the `_render` snapshot block at the
    start of `_render` reads `this._sparkles` / `this._lightningBolts`
    BEFORE the first call to `_startAnimation` (which lazy-inits
    them). On the very first `_render` (from `setConfig`), both are
    `undefined`. Currently safe because the snapshot just captures
    `undefined` and the restore branches skip. But if a downstream
    consumer ever dereferences `.length`, this becomes a runtime
    crash. Defensive `?? []` coercion eliminates the coupling.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    sparkle_snap_re = re.compile(
        r"const\s+preservedSparkles\s*=\s*"
        r"this\._sparkles\s*\?\?\s*\[\s*\]",
    )
    lightning_snap_re = re.compile(
        r"const\s+preservedLightningBolts\s*=\s*"
        r"this\._lightningBolts\s*\?\?\s*\[\s*\]",
    )
    assert sparkle_snap_re.search(executable), (
        "qs-car-card.js (review-fix #02 #18): the `preservedSparkles` "
        "snapshot must defensively coerce undefined to `[]` via "
        "`this._sparkles ?? []`."
    )
    assert lightning_snap_re.search(executable), (
        "qs-car-card.js (review-fix #02 #18): the "
        "`preservedLightningBolts` snapshot must defensively coerce "
        "undefined to `[]` via `this._lightningBolts ?? []`."
    )


def test_car_card_restore_drops_sparkles_above_new_water_base_y():
    """QS-232 review-fix #02 #16: a sparkle spawned at
    `cy ≈ _waterBaseY + 2` is preserved across `_render`; if the SOC
    changed enough that the new `_waterBaseY` is now BELOW the
    sparkle's frozen `cy` (i.e., new water surface is HIGHER on the
    SVG, meaning lower y value), the sparkle would visually appear
    in the air ABOVE the soup. Drop those sparkles at restore time.
    """
    import re

    source = (COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text()
    executable = _strip_js_comments(source)

    # The restore loop must filter sparkles whose `cy` is above
    # (less than) the new `_waterBaseY`. Accept either an inline
    # filter on the preserved array OR an explicit `if (s.cy < ...)`
    # branch inside the restore loop.
    inline_filter_re = re.compile(
        r"preservedSparkles\s*\.\s*filter\s*\([^)]*"
        r"\.cy\s*>=\s*this\._waterBaseY",
        re.DOTALL,
    )
    body_filter_re = re.compile(
        r"if\s*\(\s*[a-zA-Z_]+\.cy\s*<\s*this\._waterBaseY\s*\)",
    )
    assert inline_filter_re.search(executable) or body_filter_re.search(executable), (
        "qs-car-card.js (review-fix #02 #16): the sparkle restore "
        "block must drop preserved sparkles whose `cy` is above the "
        "NEW `_waterBaseY` (i.e., `cy < this._waterBaseY`) — "
        "otherwise they appear floating above the soup after a "
        "large SOC change."
    )


# ============================================================================
# QS-217 — Override-button carve-out across radiator / water-boiler / climate
# cards. Each card now wraps a `<path clip-rule="evenodd" d="${clipPathD}" />`
# in its `<clipPath>` so the semi-transparent bottom-center override "hand"
# button is no longer painted over by the animation (orange flames, white
# snow, blue water). The carve is gated on `e.override_reset`, identical to
# the existing button-render gate, and pins the constant *names* — not
# values — so the user can iterate visually on the radius.
# ============================================================================


def _qs217_assert_card_carve_out(card_filename: str) -> None:
    """Shared override-button carve-cover invariant check.

    QS-217 introduced a per-card COVER OVERLAY (a single ``<circle>``
    with ``fill="var(--card-background-color)"`` drawn ON TOP of the
    clipped animation group) to keep the bottom-center override button
    legible. QS-235 UNIFIED that cover across the car + the three
    duration cards: the geometry now lives in the shared
    ``RING_BOTTOM_CARVE_*`` constants (`shared/qs-card-base.js`) and
    the markup is emitted by the shared ``_ringCarveCover({…})`` helper
    — there is genuinely no per-card duplication left.

    Pinned invariants (QS-235):

    - (a) the ``<clipPath>`` is just the outer disc (no carve, no
      cancel subpath in ``clipPathD``) — unchanged.
    - (b) the per-card ``OVERRIDE_BTN_CARVE_*`` constants are GONE
      (negative pin) — the geometry moved to the shared
      ``RING_BOTTOM_CARVE_*`` constants.
    - (c) the card imports the shared ``RING_BOTTOM_CARVE_CX/CY/R``
      constants from ``./shared/qs-card-base.js``.
    - (d) the card emits its bottom-center cover via
      ``this._ringCarveCover({ cx: RING_BOTTOM_CARVE_CX, cy:
      RING_BOTTOM_CARVE_CY, r: RING_BOTTOM_CARVE_R, id:
      'override_btn_cover', show: e.override_reset })`` — same
      ``override_btn_cover`` id + ``e.override_reset`` gate as before.
    - (e) ``<g clip-path="url(#${…ClipId})">`` wrapper unchanged.
    - (f) the obsolete carve-out artifacts (carveSubpath, cancelSubpath,
      OVERRIDE_BTN_CARVE_INT_X, OVERRIDE_BTN_CARVE_INT_Y) MUST be
      absent — pinned as negative assertions so a regression can't
      silently reintroduce the lens-shape approach.
    """
    content = (COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text()

    # (a) clipPath has the simple outer disc (no carve, no cancel).
    assert (
        re.search(
            r"<clipPath\s+id=\"\$\{[^}]+\}\">\s*<path\s+clip-rule=\"evenodd\"\s+d=\"\$\{clipPathD\}\"\s*/>\s*</clipPath>",
            content,
            re.DOTALL,
        )
        is not None
    ), (
        f"{card_filename}: missing `<clipPath …><path "
        f'clip-rule="evenodd" d="${{clipPathD}}" /></clipPath>` '
        f"form. The clipPath wraps the simple outer-disc circle."
    )

    # (b) QS-235 — the per-card OVERRIDE_BTN_CARVE_* constants are gone.
    for legacy_const in (
        "OVERRIDE_BTN_CARVE_CX",
        "OVERRIDE_BTN_CARVE_CY",
        "OVERRIDE_BTN_CARVE_R",
    ):
        assert re.search(rf"const\s+{legacy_const}\s*=", content) is None, (
            f"{card_filename} (QS-235 AC3): the per-card `const "
            f"{legacy_const} = …` declaration must be removed — the "
            f"geometry moved to the shared `RING_BOTTOM_CARVE_*` "
            f"constants in `shared/qs-card-base.js`."
        )

    # (c) QS-235 — the card imports the shared carve constants.
    assert (
        re.search(
            r"import\s*\{[^}]*RING_BOTTOM_CARVE_CX[^}]*RING_BOTTOM_CARVE_CY[^}]*RING_BOTTOM_CARVE_R[^}]*\}\s*from\s*['\"]\./shared/qs-card-base\.js['\"]",
            content,
            re.DOTALL,
        )
        is not None
    ), (
        f"{card_filename} (QS-235 AC3): must import "
        f"`RING_BOTTOM_CARVE_CX, RING_BOTTOM_CARVE_CY, "
        f"RING_BOTTOM_CARVE_R` from `./shared/qs-card-base.js`."
    )

    # (d) QS-235 — the cover is emitted via the shared `_ringCarveCover`
    # helper with the shared constants + the `override_btn_cover` id +
    # the `e.override_reset` gate.
    call_re = re.compile(
        r"this\._ringCarveCover\(\s*\{([^}]*id:\s*'override_btn_cover'[^}]*)\}\s*\)",
        re.DOTALL,
    )
    call_m = call_re.search(content)
    assert call_m is not None, (
        f"{card_filename} (QS-235 AC3): missing a "
        f"`this._ringCarveCover({{ … id: 'override_btn_cover' … }})` "
        f"call. Review-fix #03's inline `<circle id=\"override_btn_cover\">` "
        f"literal is replaced by the shared helper."
    )
    call_body = call_m.group(1)
    for token_re, label in (
        (r"cx:\s*RING_BOTTOM_CARVE_CX", "cx: RING_BOTTOM_CARVE_CX"),
        (r"cy:\s*RING_BOTTOM_CARVE_CY", "cy: RING_BOTTOM_CARVE_CY"),
        (r"r:\s*RING_BOTTOM_CARVE_R", "r: RING_BOTTOM_CARVE_R"),
        (r"show:\s*e\.override_reset", "show: e.override_reset"),
    ):
        assert re.search(token_re, call_body) is not None, (
            f"{card_filename} (QS-235 AC3): the `_ringCarveCover` call "
            f"for `override_btn_cover` must pass `{label}`."
        )

    # (e) `<g clip-path="url(#${…ClipId})">` wrapper unchanged.
    assert (
        re.search(
            r'<g\s+clip-path="url\(#\$\{[a-zA-Z]+ClipId\}\)">',
            content,
        )
        is not None
    ), f'{card_filename}: expected unchanged `<g clip-path="url(#${{…ClipId}})">` wrapper.'

    # (f) Negative pins — the obsolete carve+cancel artifacts MUST be
    # absent so a regression cannot silently reintroduce the lens-
    # shape approach.
    for artifact in (
        "carveSubpath",
        "cancelSubpath",
        "OVERRIDE_BTN_CARVE_INT_X",
        "OVERRIDE_BTN_CARVE_INT_Y",
    ):
        assert artifact not in content, (
            f"{card_filename}: the obsolete carve+cancel clipPath "
            f"approach, but `{artifact}` is still present in the "
            f"source. Removing the cover-overlay approach requires an "
            f"explicit design discussion — don't silently reintroduce "
            f"the lens-shape geometry."
        )


def test_radiator_card_override_btn_carve_out():
    """QS-235 AC3 — radiator card: bottom-center override cover emitted
    via the shared `_ringCarveCover` helper + the shared
    `RING_BOTTOM_CARVE_*` constants (no per-card `OVERRIDE_BTN_CARVE_*`).

    See the helper docstring for invariant (a)–(f) details.
    """
    _qs217_assert_card_carve_out("qs-radiator-card.js")


def test_water_boiler_card_override_btn_carve_out():
    """QS-235 AC3 — water-boiler card: shared `_ringCarveCover` +
    `RING_BOTTOM_CARVE_*` carve cover (helper invariants (a)–(f))."""
    _qs217_assert_card_carve_out("qs-water-boiler-card.js")


def test_climate_card_override_btn_carve_out():
    """QS-235 AC3 — climate card: shared `_ringCarveCover` +
    `RING_BOTTOM_CARVE_*` carve cover (helper invariants (a)–(f))."""
    _qs217_assert_card_carve_out("qs-climate-card.js")


def test_dashboard_and_cards_doc_pins_qs_217_carve_paragraph():
    """QS-217 review-fix #01 should-fix #2: pin the QS-217 doc
    paragraph in `docs/agents/concepts/dashboard-and-cards.md` so a
    future edit can't silently strip the AC-8-required content.

    AC-8 mandates the new "QS-217 — Override-button carve-out" section
    mentions five specific items (a–e: legibility motivation, the
    three affected cards, the `<path clip-rule="evenodd">` idiom, the
    `OVERRIDE_BTN_CARVE_*` constants plus the CSS-to-SVG derivation,
    the `e.override_reset` gate). The QS-211 precedent
    (`test_dashboard_and_cards_doc_pins_qs_211_steam_paragraph`)
    pins QS-211 the same way; QS-217 needs the analogous test.

    Pinned invariants (mirror of the QS-211 test pattern):

    - The literal H3 headline `### QS-217 — Override-button carve-out`
      is present.
    - Headline position is GREATER than the `### QS-210` headline
      (which immediately precedes it per the in-doc order) AND LESS
      than the `## Hardened JS-card patterns` H2 (which immediately
      follows). This anchors the section between climate-backdrops
      and the hardened-patterns history.
    - Front-matter contains a `last_verified: YYYY-MM-DD` field — the
      regex deliberately does NOT pin a specific date so the test
      doesn't go red every day.
    - The section body satisfies AC-8 (a)–(e):
      (a) "legibility" OR "legible" AND "override" AND "button" —
          the human-readable motivation;
      (b) all three card filenames (`qs-radiator-card.js`,
          `qs-water-boiler-card.js`, `qs-climate-card.js`);
      (c) the literal string `clip-rule="evenodd"` (with quotes);
      (d) both `OVERRIDE_BTN_CARVE_CY` and `OVERRIDE_BTN_CARVE_R`,
          AND a CSS-to-SVG scale reference (any of `320/300`,
          `320×320`, `300×300`, `1.0667`);
      (e) the literal `e.override_reset`.

    Review-fix #03 dropped the original `clip-rule="evenodd"` pin
    in (c) — the implementation no longer uses the evenodd carve
    approach. The (c) pin now anchors on `override_btn_cover` and
    `var(--card-background-color)`, the two new identifiers for
    the cover-overlay approach.
    """
    doc_path = Path(__file__).parent.parent / "docs" / "agents" / "concepts" / "dashboard-and-cards.md"
    doc = doc_path.read_text(encoding="utf-8")

    headline = "### QS-217 — Override-button cover overlay"
    headline_idx = doc.find(headline)
    assert headline_idx != -1, (
        f"dashboard-and-cards.md: missing the literal QS-217 H3 "
        f"headline `{headline}`. Review-fix #03 changed the wording "
        f"from `carve-out` to `cover overlay` to match the simpler "
        f"implementation (cover overlay instead of clipPath carve)."
    )

    # Anchor the section between the QS-210 H3 above and the next H2
    # below (`## Hardened JS-card patterns`). The next-H2 search uses
    # `\n## ` (with trailing space + capital H) so it doesn't match
    # arbitrary `##` substrings or the front-matter delimiters.
    qs210_idx = doc.find("### QS-210")
    assert qs210_idx != -1, (
        "dashboard-and-cards.md: missing the `### QS-210` headline — "
        "AC-8 + review-fix #01 should-fix #2 anchor QS-217 AFTER "
        "the climate-card backdrop section."
    )
    hardened_idx = doc.find("## Hardened JS-card patterns")
    assert hardened_idx != -1, (
        "dashboard-and-cards.md: missing `## Hardened JS-card patterns` "
        "H2 — review-fix #01 should-fix #2 anchors QS-217 BEFORE that "
        "section."
    )
    assert qs210_idx < headline_idx < hardened_idx, (
        f"dashboard-and-cards.md: the QS-217 headline must appear "
        f"AFTER `### QS-210` AND BEFORE `## Hardened JS-card "
        f"patterns`. Current order: QS-210 at {qs210_idx}, QS-217 "
        f"at {headline_idx}, Hardened at {hardened_idx}."
    )

    # Front-matter `last_verified: YYYY-MM-DD` shape — same deliberate-
    # vagueness as the QS-211 test.
    front_matter_match = re.search(
        r"^last_verified:\s*(\d{4}-\d{2}-\d{2})\s*$",
        doc,
        re.MULTILINE,
    )
    assert front_matter_match, (
        "dashboard-and-cards.md: front-matter must contain a `last_verified: YYYY-MM-DD` field per AC-8."
    )

    # Slice the section body — from the QS-217 headline to the next H2
    # (or H3) boundary. Use `\n## ` for the H2 sentinel (more reliable
    # than `\n\n` because the body itself contains blank lines between
    # paragraphs). Fall back to `hardened_idx` since we already
    # asserted that exists above.
    section = doc[headline_idx:hardened_idx]
    section_lower = section.lower()

    # (a) Legibility motivation: "legibility" or "legible" + "override"
    # + "button".
    assert "legibility" in section_lower or "legible" in section_lower, (
        "dashboard-and-cards.md (QS-217 section): missing legibility-"
        "motivation language. AC-8 (a) requires a human-readable "
        "explanation of why the carve-out exists."
    )
    assert "override" in section_lower, (
        "dashboard-and-cards.md (QS-217 section): missing `override` "
        "reference — AC-8 (a) requires mentioning the override "
        "button as the motivating element."
    )
    assert "button" in section_lower, (
        "dashboard-and-cards.md (QS-217 section): missing `button` "
        "reference — AC-8 (a) requires mentioning the override "
        "button as the motivating element."
    )

    # (b) All three card filenames mentioned in the section.
    for card_filename in (
        "qs-radiator-card.js",
        "qs-water-boiler-card.js",
        "qs-climate-card.js",
    ):
        assert card_filename in section, (
            f"dashboard-and-cards.md (QS-217 section): missing card "
            f"filename `{card_filename}` — AC-8 (b) requires all "
            f"three affected cards be named explicitly."
        )

    # (c) Literal `<circle ... fill="var(--card-background-color)"`
    # describing the cover-overlay approach (review-fix #03 replaced
    # the earlier `clip-rule="evenodd"` clipPath idiom). The cover
    # element name `override_btn_cover` is also pinned so a doc
    # rewrite can't silently drop the implementation anchor.
    assert "override_btn_cover" in section, (
        "dashboard-and-cards.md (QS-217 section): missing the literal "
        "`override_btn_cover` element id — pin requires the doc "
        "name the cover element so a reader can grep it back to the "
        "source."
    )
    assert "var(--card-background-color)" in section, (
        "dashboard-and-cards.md (QS-217 section): missing the literal "
        "`var(--card-background-color)` — the cover overlay's fill "
        "value must be documented so the visual behaviour is "
        "reproducible from the doc alone."
    )

    # (d) Both shared constant names + a CSS-to-SVG scale reference.
    # QS-235 unified the per-card `OVERRIDE_BTN_CARVE_*` duplicates into
    # the shared `RING_BOTTOM_CARVE_*` constants, so the doc now names
    # those (and the `_ringCarveCover` helper).
    for const_name in ("RING_BOTTOM_CARVE_CY", "RING_BOTTOM_CARVE_R"):
        assert const_name in section, (
            f"dashboard-and-cards.md (QS-217 section): missing "
            f"`{const_name}` — AC-8 (d) / QS-235 AC3 require both shared "
            f"carve constants be named (so the doc reader can grep them "
            f"back to the source)."
        )
    assert "_ringCarveCover" in section, (
        "dashboard-and-cards.md (QS-217 section): missing the shared "
        "`_ringCarveCover` helper name — QS-235 AC3 unified the carve "
        "cover into this helper, which the doc must name."
    )
    scale_patterns = ("320/300", "320×320", "300×300", "1.0667")
    assert any(pat in section for pat in scale_patterns), (
        f"dashboard-and-cards.md (QS-217 section): missing a CSS-to-"
        f"SVG scale reference. AC-8 (d) requires one of "
        f"{scale_patterns!r} so the derivation chain is reproducible "
        f"from the doc alone."
    )

    # (e) Literal `e.override_reset`.
    assert "e.override_reset" in section, (
        "dashboard-and-cards.md (QS-217 section): missing the literal "
        "`e.override_reset` — AC-8 (e) requires the gate idiom be "
        "named so the reader knows which truthy check controls the "
        "carve."
    )


def test_dashboard_and_cards_doc_pins_qs_211_steam_paragraph():
    """Review-fix #01 finding #3: pin the QS-211 doc paragraph in
    `docs/agents/concepts/dashboard-and-cards.md` so a future edit
    can't silently strip the headline, move it, or stale the
    `last_verified` field without flipping this test red.

    AC-10 already routes verification through
    `python scripts/qs/check_doc_drift.py`, but drift detection only
    fires on co-modification mismatches — it does not enforce
    placement, headline literal, or front-matter field shape. This
    regression-test complements the drift checker on those three
    dimensions.

    Pinned invariants:

    - The literal headline `**QS-211 — boiling steam puffs.**` is
      present.
    - Headline index is GREATER than the QS-200 block marker
      (`**QS-200`) AND LESS than the first `Review-fix #01` marker —
      i.e. the paragraph sits between the QS-200 block and the
      review-fix history, per AC-10.
    - The front-matter contains a `last_verified:` field with a
      `YYYY-MM-DD` value. The regex deliberately does NOT pin a
      specific date so the test doesn't go red every day; the goal
      is that the field exists and is well-formed.
    - The paragraph body mentions the four QS-211 anchors
      (`MAX_CONCURRENT_STEAM`, `STEAM_FILL_COLOR`,
      `_currentColorMix`, `disconnectedCallback`) so a future doc
      rewrite can't silently strip the substantive content.
    """
    import re

    doc_path = Path(__file__).parent.parent / "docs" / "agents" / "concepts" / "dashboard-and-cards.md"
    doc = doc_path.read_text(encoding="utf-8")

    headline = "**QS-211 — boiling steam puffs.**"
    headline_idx = doc.find(headline)
    assert headline_idx != -1, (
        f"dashboard-and-cards.md: missing the literal QS-211 headline "
        f"`{headline}`. AC-10 requires this exact wording so the "
        "paragraph is discoverable from a substring search."
    )

    qs200_idx = doc.find("**QS-200")
    assert qs200_idx != -1, (
        "dashboard-and-cards.md: missing the `**QS-200` block marker — "
        "AC-10 anchors the QS-211 paragraph relative to that block."
    )
    review_fix_idx = doc.find("Review-fix #01")
    assert review_fix_idx != -1, (
        "dashboard-and-cards.md: missing the `Review-fix #01` marker — "
        "AC-10 anchors the QS-211 paragraph BEFORE that marker."
    )
    assert qs200_idx < headline_idx < review_fix_idx, (
        "dashboard-and-cards.md: the QS-211 headline must appear AFTER "
        "the `**QS-200` block AND BEFORE `Review-fix #01`. Current "
        f"order: QS-200 at {qs200_idx}, QS-211 at {headline_idx}, "
        f"Review-fix #01 at {review_fix_idx}."
    )

    # Front-matter `last_verified:` field must be present and shaped
    # YYYY-MM-DD. We deliberately do NOT pin a specific date so this
    # test doesn't go red every day — the field-exists + well-formed
    # check is the right level for a regression test.
    front_matter_match = re.search(
        r"^last_verified:\s*(\d{4}-\d{2}-\d{2})\s*$",
        doc,
        re.MULTILINE,
    )
    assert front_matter_match, (
        "dashboard-and-cards.md: front-matter must contain a `last_verified: YYYY-MM-DD` field per AC-10."
    )

    # Body content guards: a future doc rewrite that strips these
    # anchors leaves a paragraph without substance, defeating the
    # AC-10 documentation intent. We pin the four most-load-bearing
    # tokens (cap constant, fill-color constant, the colorMix
    # cross-fade reference, the disconnectedCallback teardown
    # reference).
    #
    # Review-fix #02 finding #1: the slice is bounded EXACTLY to the
    # paragraph (terminated by the next blank line) — no widening. A
    # `+ 2000` reach-ahead would let any of the four anchors live in
    # a NEIGHBOURING section (e.g. Review-fix #01/#02 below) and
    # still satisfy this test, defeating the "pin the substantive
    # content of the QS-211 paragraph" intent of fix-plan #01 #3.
    paragraph_end = doc.find("\n\n", headline_idx)
    if paragraph_end == -1:
        paragraph_end = len(doc)
    paragraph = doc[headline_idx:paragraph_end]
    for anchor in (
        "MAX_CONCURRENT_STEAM",
        "STEAM_FILL_COLOR",
        "_currentColorMix",
        "disconnectedCallback",
    ):
        assert anchor in paragraph, (
            f"dashboard-and-cards.md: the QS-211 paragraph must "
            f"mention `{anchor}` — it's one of the four load-bearing "
            "anchors the AC-10 paragraph is required to cover. A "
            "rewrite that drops it leaves the paragraph as a stub."
        )


def test_water_boiler_card_running_at_stop_consume_is_flag_gated():
    """Review-fix #04 M1: the `_runningAtStop` stash must be consumed
    EXACTLY ONCE on the first post-reattach `_render`, regardless of
    whether the running-state guard fires.

    This is the third revision of the N12 (fix-plan #02) reconnect
    re-prime logic. The history:

    - Plan #02 N12 introduced `_runningAtStop` set in
      `_stopAnimation` and consumed in `_render` (cleared
      unconditionally after the guard).
    - Plan #03 S1 moved the clear INSIDE the guard's if-body to fix
      a "mid-detach hass-pushes consume the stash too early" hole.
    - Plan #04 M1 (this) gates the entire consume on a
      `_pendingReattachCheck` flag set in `connectedCallback`,
      because the pass-3 fix introduced a complementary hole: a
      stash that's only cleared when the inner guard fires LEAKS
      across renders when reattach happens with `running` unchanged,
      and the next in-place state flip (hours later, no detach
      involved) then falsely triggers the prime → wave snaps to
      target in one frame instead of lerping.

    5-step regression scenario this test pins against (the pass-3
    inside-if structure alone misses it):

    1. Boiler running, card attached: `_running = true`,
       `_runningAtStop = undefined`.
    2. Detach (HA dashboard rearrange) → `_stopAnimation()` sets
       `_runningAtStop = true`.
    3. Reattach with `running` still `true` → guard
       `_runningAtStop(true) !== running(true)` is false → with
       pass-3 inside-if, stash NOT cleared → `_runningAtStop = true`
       indefinitely.
    4. Card stays attached for hours. Boiler finishes naturally;
       hass-push delivers `running = false`.
    5. Next `_render`: guard `_runningAtStop(true) !== running(false)`
       is true → prime fires → wave snaps. ← BUG.

    The fix shape that handles all three patterns:

    - Mid-detach hass-pushes: flag is false (only set in
      `connectedCallback`), consume block is skipped, stash
      preserved.
    - Reattach with `running` unchanged: flag is true → consume
      runs → guard fails → no prime → stash cleared → flag cleared
      → subsequent in-place flips lerp normally.
    - Reattach with `running` flipped: flag is true → consume runs
      → guard fires → prime → stash cleared → flag cleared → wave
      snaps to new target on reattach (the original N12 intent).
    """
    import re

    content = (COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js").read_text()
    executable = _strip_js_comments(content)
    # The flag must be set somewhere — typically in connectedCallback
    # after `_startAnimation()`.
    assert re.search(
        r"this\._pendingReattachCheck\s*=\s*true",
        executable,
    ), (
        "qs-water-boiler-card.js (review-fix #04 M1): expected "
        "`this._pendingReattachCheck = true` (set in "
        "`connectedCallback` after `_startAnimation()`) so the next "
        "`_render` knows to consume the `_runningAtStop` stash "
        "exactly once per reattach."
    )
    # The flag-gated consume block must exist in `_render` and contain
    # the inner guard, the stash clear, AND the flag's own clear.
    # Use a balanced-brace walk (not a regex) so the inner `if (...)`
    # nested block doesn't truncate the captured body at the inner `}`.
    body = _extract_js_function_body(executable, r"if\s*\(\s*this\._pendingReattachCheck\s*\)\s*")
    assert body is not None, (
        "qs-water-boiler-card.js (review-fix #04 M1): missing "
        "`if (this._pendingReattachCheck) { ... }` block in "
        "`_render()`. The stash must be consumed inside this flag-"
        "gated block so the consume fires exactly once per reattach "
        "— independent of the inner running-state guard outcome."
    )
    assert "this._runningAtStop !== undefined" in body, (
        "qs-water-boiler-card.js (review-fix #04 M1): the flag-"
        "gated block must contain the inner guard "
        "`if (this._runningAtStop !== undefined && ...)` that "
        "decides whether to set `_needsAnimationPrime`."
    )
    assert re.search(r"this\._runningAtStop\s*=\s*undefined", body), (
        "qs-water-boiler-card.js (review-fix #04 M1): the flag-"
        "gated block must clear `this._runningAtStop = undefined` "
        "UNCONDITIONALLY (i.e. outside the inner guard's if-body, "
        "but inside the outer flag-gated block) — runs regardless "
        "of inner-guard outcome."
    )
    assert re.search(r"this\._pendingReattachCheck\s*=\s*false", body), (
        "qs-water-boiler-card.js (review-fix #04 M1): the flag-"
        "gated block must clear `this._pendingReattachCheck = "
        "false` so the consume fires only on the first post-"
        "reattach `_render`, not on every subsequent render."
    )
    # Negative: `_runningAtStop = undefined` must NOT appear outside
    # the flag-gated block — exactly ONE total occurrence in the file.
    clear_count = len(re.findall(r"this\._runningAtStop\s*=\s*undefined", executable))
    assert clear_count == 1, (
        f"qs-water-boiler-card.js (review-fix #04 M1): expected "
        f"exactly ONE `this._runningAtStop = undefined` clear "
        f"(inside the flag-gated block); found {clear_count}. Any "
        f"clear outside the flag gate would reintroduce either the "
        f"pass-3 'mid-detach hass push consumes stash too early' "
        f"hole OR the pass-4 'leaked stash → false prime on later "
        f"in-place flip' hole."
    )


def test_water_boiler_card_translates_via_water_boiler_mode_namespace():
    """S10: the card resolves bistate mode labels under the
    `water_boiler_mode` translation namespace, not `on_off_mode`, so
    future boiler-specific labels can diverge without touching this card.
    """
    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    content = js_path.read_text(encoding="utf-8")
    assert "entity.select.water_boiler_mode.state" in content, (
        "S10: qs-water-boiler-card.js must resolve bistate labels under "
        "`component.quiet_solar.entity.select.water_boiler_mode.state`."
    )
    # Defensive: the on_off_mode key must NOT appear here — we want a
    # clean namespace separation so a future label change for boilers
    # can't accidentally bleed into the on_off card or vice versa.
    assert "entity.select.on_off_mode.state" not in content, (
        "S10: qs-water-boiler-card.js must not reference the on_off_mode "
        "translation namespace; use water_boiler_mode instead."
    )


@pytest.mark.asyncio
async def test_water_boiler_card_input_contract_keys(hass):
    """N4: the JS card's input keys match what the template emits.

    Extract the destructured key set from the JS card source (via the
    `this._entity(e.<key>)` accesses) and assert it's a SUPERSET of the
    keys the dashboard template emits in the `entities:` mapping. The
    JS card may read additional keys that are emitted only conditionally
    (e.g. `temperature_sensor`); the rule is "every emitted key is
    consumed by the card" plus "no spurious keys in the template".
    """
    import re

    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    js_content = js_path.read_text(encoding="utf-8")
    js_keys = set(re.findall(r"this\._entity\(\s*e\.([A-Za-z_][A-Za-z_0-9]*)", js_content))

    tpl_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
    tpl_content = tpl_path.read_text(encoding="utf-8")
    # Find the water_boiler branch and extract `key: ...` lines.
    branch_match = re.search(
        r'elif\s+device\.device_type\s*==\s*"water_boiler"\s*-?%}(.*?)(?=\{%\s*elif|\{%\s*endif)',
        tpl_content,
        re.DOTALL,
    )
    assert branch_match, "Could not locate the water_boiler branch in the custom template"
    branch = branch_match.group(1)
    tpl_keys = set(re.findall(r"(?:^|\n)\s*\{%[^%]*%\}\s*([a-z_][a-z_0-9]*)\s*:", branch))
    # The temperature_sensor key is emitted via a literal `{{ "temperature_sensor: " + ... }}`
    if '"temperature_sensor: "' in branch:
        tpl_keys.add("temperature_sensor")
    # Card-level keys (siblings of `entities:`) are not part of the
    # `entities` contract — drop them from the comparison.
    tpl_keys.discard("title")

    # Every template-emitted key must be a key the JS card reads.
    unconsumed = tpl_keys - js_keys
    assert not unconsumed, f"Template emits keys not consumed by qs-water-boiler-card.js: {sorted(unconsumed)}"


@pytest.mark.asyncio
async def test_water_boiler_card_input_contract_emits_temperature_sensor_key(hass):
    """Custom template emits `temperature_sensor: <id>` for the JS card.

    The dedicated `qs-water-boiler-card` reads `entities.temperature_sensor`
    to render the optional water-tank temperature row. The template
    must wire the configured `water_boiler_temperature_sensor` to that
    key so the JS card can display the value.
    """
    home = await _build_water_boiler_home(hass, MOCK_WATER_BOILER_CONFIG)
    template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
    template_content = template_path.read_text()

    tpl = Template(template_content, hass)
    rendered = tpl.async_render(variables={"home": home})

    parsed = yaml.safe_load(rendered)
    assert parsed is not None

    # Find the boiler card and inspect its `entities` mapping
    water_boiler_section = None
    for view in parsed.get("views", []):
        if view.get("path") == "water_boilers":
            water_boiler_section = view
            break
    assert water_boiler_section is not None
    expected_id = MOCK_WATER_BOILER_CONFIG[CONF_WATER_BOILER_TEMPERATURE_SENSOR]
    found = False
    for grid in water_boiler_section.get("sections", []):
        for card in grid.get("cards", []):
            entities = card.get("entities")
            if isinstance(entities, dict) and entities.get("temperature_sensor") == expected_id:
                found = True
                break
    assert found, (
        f"Expected `temperature_sensor: {expected_id}` in the boiler card's "
        f"entities mapping; got: {water_boiler_section}"
    )


@pytest.mark.parametrize(
    "template_name",
    [
        "quiet_solar_dashboard_template.yaml.j2",
        "quiet_solar_dashboard_template_standard_ha.yaml.j2",
    ],
)
@pytest.mark.asyncio
async def test_water_boiler_temperature_sensor_row_absent_when_unset(hass, template_name):
    """When no temp sensor is configured, the configured entity id is NOT rendered.

    Structural check using the configured entity id (not a prefix
    heuristic) so a future regression that emitted a *different* sensor
    id couldn't sneak past this assertion.
    """
    expected_temp_id = MOCK_WATER_BOILER_CONFIG[CONF_WATER_BOILER_TEMPERATURE_SENSOR]
    home = await _build_water_boiler_home(hass, MOCK_WATER_BOILER_CONFIG_NO_TEMP)
    template_path = COMPONENT_ROOT / "ui" / template_name
    template_content = template_path.read_text()

    tpl = Template(template_content, hass)
    rendered = tpl.async_render(variables={"home": home})

    parsed = yaml.safe_load(rendered)
    assert parsed is not None
    entity_ids = _collect_water_boiler_entity_rows(parsed)
    assert expected_temp_id not in entity_ids, (
        f"Configured temp sensor {expected_temp_id!r} leaked into the no-temp boiler card; got rows: {entity_ids}"
    )


# ============================================================================
# QS-210 — Climate card backdrop modes (HEAT flame / COOL snow / AUTO temp /
# WIND fallback) and the dashboard jinja plumbing for the backing climate
# entity id. Mirrors the regex-on-source pattern used by the radiator-card
# tests above (`test_radiator_card_flame_layers_present` etc.).
# ============================================================================


class TestClimateCardBackdropDerivation:
    """QS-210 AC1 — single resolved-target backdrop algorithm."""

    def test_climate_card_backdrop_heat_cool_branches_present(self):
        """HEAT → 'flame' and COOL → 'snow' must both appear, gated on
        the configured HVAC-on state (typically `climate_state_on`).
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # 'heat' literal must drive a return-of-'flame' branch.
        # Accept either `climateStateOn === 'heat'` directly or via the
        # `deriveBackdrop` helper.
        assert (
            re.search(
                r"===\s*'heat'.{0,200}return\s+'flame'",
                executable,
                re.DOTALL,
            )
            is not None
        ), "QS-210 AC1: HEAT branch must short-circuit to `return 'flame'` when the climate-state-on string is 'heat'."

        # 'cool' literal must drive a return-of-'snow' branch.
        assert (
            re.search(
                r"===\s*'cool'.{0,200}return\s+'snow'",
                executable,
                re.DOTALL,
            )
            is not None
        ), "QS-210 AC1: COOL branch must short-circuit to `return 'snow'` when the climate-state-on string is 'cool'."

    def test_climate_card_backdrop_auto_resolves_single_target(self):
        """AUTO / HEAT_COOL must read `temperature`, `target_temp_low`,
        and `target_temp_high` attributes, blend dual setpoints via a
        midpoint, and fall through to a `running ? 'wind' : 'none'`
        branch when nothing resolves.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # All four climate attributes must be read.
        for attr in (
            "current_temperature",
            "temperature",
            "target_temp_low",
            "target_temp_high",
        ):
            assert re.search(rf"attrs\??\.{attr}\b", executable) is not None, (
                f"QS-210 AC1: AUTO branch must read climate entity attribute `{attr}`."
            )

        # Midpoint blend for dual setpoints: (low + high) / 2.
        # Accept identifier names (`lowTarget`, `highTarget`, `low`, `high`, `L`, `H`).
        assert (
            re.search(
                r"\(\s*\w+Target\s*\+\s*\w+Target\s*\)\s*/\s*2",
                executable,
            )
            is not None
        ), (
            "QS-210 AC1: AUTO branch must blend `target_temp_low` and "
            "`target_temp_high` via a midpoint expression `(low + high) / 2`."
        )

        # 'auto' or 'heat_cool' literal entering the resolve branch.
        assert "'auto'" in executable or "'heat_cool'" in executable, (
            "QS-210 AC1: AUTO branch must trigger on `climate_state_on` equalling 'auto' or 'heat_cool'."
        )

        # 'wind' fallback gated on `running`.
        assert (
            re.search(
                r"running\s*\?\s*'wind'\s*:\s*'none'",
                executable,
            )
            is not None
        ), "QS-210 AC1: AUTO branch must fall through to `running ? 'wind' : 'none'` when no setpoint resolves."

        # Review-fix S11: the `'wind'` literal must pair with `'none'`
        # inside the SAME `running ? … : …` ternary so off-with-no-
        # setpoint deterministically maps to `'none'`, not `'wind'`.
        # Pin the ternary as a single semantic unit via a strict regex.
        wind_none_pair = re.compile(r"return\s+running\s*\?\s*'wind'\s*:\s*'none'\s*;")
        assert wind_none_pair.search(executable) is not None, (
            "QS-210 AC1 / review-fix S11: the `'wind'` literal must "
            "appear in a `return running ? 'wind' : 'none';` ternary so "
            "off-with-no-setpoint maps to 'none' (not 'wind')."
        )


class TestClimateCardFlameBackdrop:
    """QS-210 AC2 — HEAT backdrop uses the radiator-card flame engine."""

    def test_climate_card_flame_engine_markers_present(self):
        """The radiator's flame engine is copied verbatim: path
        generator, per-layer paths, fill branch, and the progress-tracked
        base-Y formula.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # The peaked-teeth path generator.
        assert "_generateFlameTeethPath" in executable, (
            "QS-210 AC2: `_generateFlameTeethPath` (peaked-teeth path "
            "generator copied from qs-radiator-card.js) must be declared."
        )

        # Per-layer flame template-literal id pattern.
        assert 'id="flame${i}"' in content, (
            'QS-210 AC2: per-layer flame path id must be emitted as the template literal `id="flame${i}"`.'
        )

        # Running ? FLAME_FILLS : FLAME_GREY_FILLS branch.
        assert (
            re.search(
                r"running\s*\?\s*FLAME_FILLS\s*:\s*FLAME_GREY_FILLS",
                executable,
            )
            is not None
        ), "QS-210 AC2: flame fill must branch on `running ? FLAME_FILLS : FLAME_GREY_FILLS` (mirror radiator card)."

        # The full flameBaseY formula matching the radiator's envelope.
        assert (
            re.search(
                r"flameBaseY\s*=\s*CENTER_CY\s*\+\s*CLIP_R\s*-\s*\(\s*"
                r"FLAME_BASE_MIN_PCT\s*\+\s*progressRatio\s*\*\s*\(\s*"
                r"FLAME_BASE_MAX_PCT\s*-\s*FLAME_BASE_MIN_PCT\s*\)\s*\)\s*"
                r"\*\s*2\s*\*\s*CLIP_R",
                executable,
            )
            is not None
        ), "QS-210 AC2: flameBaseY formula must mirror the radiator's progress envelope verbatim."


class TestClimateCardSnowBackdrop:
    """QS-210 AC3 — COOL backdrop renders snow pile + falling snowflakes."""

    def test_climate_card_snow_pile_three_waves_with_palette(self):
        """3 snow-pile wave paths plus the documented palette (front
        white-ish, back blue-ish).
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # 3 wave paths — either template-literal (`id="snowWave${i}"`)
        # or three explicit ids. Accept both.
        if 'id="snowWave${i}"' not in content:
            for i in range(3):
                assert re.search(rf'id="snowWave{i}"', executable) is not None, (
                    f'QS-210 AC3: missing snow-pile wave path `id="snowWave{i}"`.'
                )

        # QS-220 AC-2: the front layer is no longer pure white.
        assert "hsla(0, 0%, 95%, 0.65)" not in executable, (
            "QS-220 AC-2: the legacy pure-white SNOW_FRONT_COLOR "
            "literal `hsla(0, 0%, 95%, 0.65)` must be replaced "
            "with a translucent pale-blue value."
        )
        # SNOW_FRONT_COLOR constant must still exist and reference an
        # hsla literal — direction-only, no hue/sat/light/alpha pin.
        assert re.search(
            r"const\s+SNOW_FRONT_COLOR\s*=\s*'hsla\([^']+\)'",
            executable,
        ), "QS-220 AC-2: SNOW_FRONT_COLOR must remain a module-level const assigned to an hsla literal."

        # At least one blue-ish layer (hue around 200-230) — after
        # QS-220 the front layer also matches; back/mid still do.
        assert (
            re.search(
                r"hsla\(\s*2[0-3]\d\s*,",
                executable,
            )
            is not None
        ), "QS-210 AC3: ≥1 blue-ish pile layer (back/mid/front) must use a blue-ish hue (regex on `hsla(2[0-3]X, …)`)."

    def test_climate_card_snowflakes_fall_down(self):
        """Snowflake particle loop INVERTS the boiler-bubble system:
        spawn near the top of the clip and increment cy (fall down).
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # The fall direction: cy += b.vy * dt (or equivalent positive
        # accumulation). Accept either `b.cy += b.vy * dt` or
        # `b.cy = b.cy + b.vy * dt`.
        cy_increments = bool(
            re.search(r"b\.cy\s*\+=\s*b\.vy\s*\*\s*dt", executable)
            or re.search(r"b\.cy\s*=\s*b\.cy\s*\+\s*b\.vy\s*\*\s*dt", executable)
        )
        assert cy_increments, (
            "QS-210 AC3: snowflake fall must INCREMENT `cy` per frame "
            "(`b.cy += b.vy * dt`) — boiler bubbles decrement, snow falls."
        )

        # Top-of-clip spawn — `CENTER_CY - CLIP_R + …` (contrast with
        # the boiler's bottom-of-clip `CENTER_CY + CLIP_R - 8`).
        assert (
            re.search(
                r"CENTER_CY\s*-\s*CLIP_R",
                executable,
            )
            is not None
        ), (
            "QS-210 AC3: snowflake spawn `cy` must use "
            "`CENTER_CY - CLIP_R` (top of clip) — contrasts with the "
            "boiler's bottom-of-clip spawn formula."
        )


class TestClimateCardWindBackdrop:
    """QS-210 AC4 — WIND backdrop renders 3 stroked sinusoidal wisps."""

    def test_climate_card_wind_three_stroked_wisps(self):
        """3 wind wisp paths each with `stroke=` and `fill="none"`,
        plus the WIND_SPEED_PX_PER_S linear-scroll constant.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # 3 wind wisp paths — template-literal or three explicit ids.
        if 'id="windWisp${i}"' not in content:
            for i in range(3):
                assert re.search(rf'id="windWisp{i}"', executable) is not None, (
                    f'QS-210 AC4: missing wisp path `id="windWisp{i}"`.'
                )

        # `stroke=` attribute on a wisp path.
        # Accept stroke="..." or stroke="${...}".
        assert (
            re.search(
                r'<path[^>]*id="windWisp[^"]*"[^>]*stroke=',
                content,
                re.DOTALL,
            )
            is not None
        ), "QS-210 AC4: each wisp path must carry a `stroke=` attribute (open sinusoidal line, not a filled polygon)."

        # `fill="none"` on the wisp path.
        assert (
            re.search(
                r'<path[^>]*id="windWisp[^"]*"[^>]*fill="none"',
                content,
                re.DOTALL,
            )
            is not None
        ), (
            'QS-210 AC4: each wisp path must carry `fill="none"` so it '
            "renders as a stroked open line, not a filled wave polygon."
        )

        # WIND_SPEED_PX_PER_S constant.
        assert re.search(r"const\s+WIND_SPEED_PX_PER_S\s*=", executable) is not None, (
            "QS-210 AC4: `WIND_SPEED_PX_PER_S` constant must drive the linear-scroll translateX accumulator."
        )

        # Review-fix S10: `_generateWispPath` must emit an OPEN
        # sinusoidal path — NOT closed with `L width WAVE_BOTTOM_Y L 0
        # WAVE_BOTTOM_Y Z` like the wave generator. Pin the difference
        # so a refactor that copy-pastes the wave generator (closed
        # polygon) is caught.
        wisp_body = _extract_js_function_body(
            executable,
            r"_generateWispPath\s*\([^)]*\)\s*",
        )
        assert wisp_body is not None, (
            "Review-fix S10: `_generateWispPath` method must be defined in qs-climate-card.js."
        )
        assert "WAVE_BOTTOM_Y" not in wisp_body, (
            "Review-fix S10: `_generateWispPath` must NOT reference "
            "`WAVE_BOTTOM_Y` (that would close the path into a polygon — "
            "the wisp is an OPEN stroked line)."
        )
        # The body's last meaningful character must be `;` from the
        # return statement; no trailing ` Z` closing the path.
        assert not re.search(r"\bZ\b", wisp_body), (
            "Review-fix S10: `_generateWispPath` body must NOT contain "
            "the SVG path-closing `Z` operator (the wisp is OPEN, not a "
            "filled polygon)."
        )


class TestClimateCardNoneBackdrop:
    """QS-210 AC5 — fan_only/dry/off/null short-circuit to 'none'."""

    def test_climate_card_fan_only_dry_off_short_circuit_to_none(self):
        """The backdrop derivation must return 'none' for non-HEAT,
        non-COOL, non-AUTO/HEAT_COOL values BEFORE any climate-entity
        attribute read. Pin the structural shape: a `return 'none'`
        exists inside the `deriveBackdrop` function body, at the
        function's tail (catch-all branch).
        """

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()

        # Find the deriveBackdrop function body (handle balanced braces).
        deriveBody = _extract_js_function_body(
            content,
            r"const\s+deriveBackdrop\s*=\s*\(\s*\)\s*=>\s*",
        )
        assert deriveBody is not None, (
            "QS-210 AC5: a `deriveBackdrop = () => { ... }` arrow "
            "function must be declared (the backdrop decision body)."
        )

        # Strip comments inside the body.
        executable_body = _strip_js_comments(deriveBody)

        # 'none' is returned somewhere in the body.
        assert "return 'none'" in executable_body, (
            "QS-210 AC5: `deriveBackdrop` body must contain a `return 'none'` branch for the everything-else case."
        )

        # The catch-all `return 'none'` is at the tail of the body — i.e.
        # AFTER the 'auto'/'heat_cool' branch, so values like
        # 'fan_only'/'dry'/'off' never enter the attribute-read path.
        # Approximate this with an order check: the last `return 'none'`
        # comes AFTER the last `return 'snow'` AND the last `return 'flame'`.
        last_none = executable_body.rfind("return 'none'")
        last_snow = executable_body.rfind("return 'snow'")
        last_flame = executable_body.rfind("return 'flame'")
        assert last_none > last_snow and last_none > last_flame, (
            "QS-210 AC5: the catch-all `return 'none'` must sit AFTER "
            "all HEAT/COOL/AUTO branches so unrecognised "
            "`climate_state_on` values short-circuit without attribute "
            "reads."
        )


class TestClimateCardAutoTempComparison:
    """QS-210 AC6 — `_safeNumber` wraps every climate-attribute read."""

    def test_climate_card_inline_safe_number_at_four_attribute_reads(self):
        """Each of the four climate-entity attributes is read via
        `_safeNumber({state: ...}, null)`. No `_safeAttr` helper exists.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # Forbidden: no `_safeAttr` helper.
        assert re.search(r"_safeAttr\s*\(", executable) is None, (
            "QS-210 AC6: no `_safeAttr(...)` helper — adversarial review "
            "rejected this as YAGNI; inline four `_safeNumber({state: …})` "
            "calls instead."
        )

        # Four `_safeNumber({state: attrs?.<X>}, null)` calls, one per
        # climate attribute, in close proximity.
        attrs_in_order = [
            "current_temperature",
            "temperature",
            "target_temp_low",
            "target_temp_high",
        ]
        positions = []
        for attr in attrs_in_order:
            pattern = re.compile(rf"_safeNumber\s*\(\s*\{{\s*state\s*:\s*attrs\??\.{attr}\s*\}}\s*,\s*null\s*\)")
            m = pattern.search(executable)
            assert m is not None, (
                f"QS-210 AC6: missing `_safeNumber({{state: attrs?.{attr}}}, null)` "
                f"wrapper for the `{attr}` attribute read."
            )
            positions.append(m.start())

        # All four reads sit within ~600 characters of each other
        # (~10 lines at typical formatting) so they read as a tight
        # block, not scattered through the file.
        assert (max(positions) - min(positions)) < 600, (
            "QS-210 AC6: the four `_safeNumber` calls must sit close "
            "together (within ~10 lines / 600 characters) so the reads "
            "read as one block of code, not scattered."
        )


class TestClimateCardJinjaClimateEntity:
    """QS-210 AC7 — dashboard jinja exposes the backing climate entity id."""

    @pytest.mark.asyncio
    async def test_dashboard_template_emits_climate_entity_for_climate_device(self, hass, full_dashboard_home):
        """The climate device block in the custom-card dashboard must
        emit a `climate_entity: climate.X` mapping inside the
        `qs-climate-card`'s `entities:` block. Regex `climate\\.\\w+`
        so a fixture rename doesn't break the test.
        """
        import re

        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = await hass.async_add_executor_job(template_path.read_text)

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        assert re.search(r"climate_entity:\s*climate\.\w+", rendered) is not None, (
            "QS-210 AC7: the rendered dashboard YAML must contain a "
            "`climate_entity: climate.<id>` line inside the climate "
            "device's `entities:` mapping (mirror of `backing_entity` "
            "on the radiator card)."
        )

    @pytest.mark.asyncio
    async def test_dashboard_template_omits_climate_entity_when_device_attr_missing(self, hass):
        """Review-fix S9 — the `{% if device.climate_entity %}` guard
        must omit the line entirely when the underlying device has no
        backing climate entity (defensive AC7 path). Renders the climate
        block against a tiny stub home with `device.climate_entity =
        None` and asserts no `climate_entity:` line appears.
        """
        import re

        # Minimal duck-typed stub for the climate block in the template.
        # The template touches `device.device_type`, `device.name`,
        # `device.ha_entities`, `device.climate_entity`, and the home's
        # dashboard plumbing — supply only what's strictly required.
        class _StubEntity:
            def __init__(self, entity_id):
                self.entity_id = entity_id

        class _StubHaEntities(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        class _StubDevice:
            device_type = "climate"
            name = "Stub Climate"
            calendar = None
            ha_entities = _StubHaEntities()
            climate_entity = None  # ← the falsy attribute under test
            # The template also touches `device.home.ha_entities` for
            # `qs_home_is_off_grid`. An empty dict short-circuits.
            home = type("_StubHome", (), {"ha_entities": _StubHaEntities()})()

        class _StubHome:
            dashboard_sections = [("climate", None)]
            ha_entities = _StubHaEntities()

            def get_devices_for_dashboard_section(self, name):
                return [_StubDevice()]

        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = await hass.async_add_executor_job(template_path.read_text)

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": _StubHome()})

        # The negative path: no `climate_entity:` mapping at all.
        assert re.search(r"\bclimate_entity:", rendered) is None, (
            "Review-fix S9: when `device.climate_entity` is falsy, the "
            "`{% if device.climate_entity %}` guard must omit the "
            "`climate_entity:` line entirely so the JS card's defensive "
            "`e.climate_entity ? … : null` path is exercised."
        )


class TestClimateCardReviewFix01Hardening:
    """QS-210 review-fix #01 — hardening for cache staleness, prime
    correctness, hysteresis, snowflake geometry, _safeNumber rigor,
    and AC5 short-circuit semantics.
    """

    def test_climate_card_invalidates_caches_after_innerhtml_rewrite(self):
        """Review-fix M1 — every `_render()` rewrites
        `this._root.innerHTML`. The cache invalidators MUST run
        unconditionally AFTER the rewrite (mirror of
        `qs-radiator-card.js:967-969`), not only when the backdrop type
        changes. Without this, same-backdrop re-renders (the common
        case, every hass push) leave RAF holding stale DOM refs.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # Locate the innerHTML assignment.
        innerhtml_match = re.search(
            r"this\._root\.innerHTML\s*=\s*`",
            executable,
        )
        assert innerhtml_match is not None, (
            "Review-fix M1: `this._root.innerHTML = `…`` template assignment must exist in qs-climate-card.js."
        )
        # The outer template literal contains nested `${... `…` ...}`
        # template literals (e.g. for the power-btn fragment), so a
        # naive `find('`', ...)` would land on a nested closing
        # backtick rather than the outer one. Instead, look for the
        # idiomatic outer-close pattern: a backtick immediately
        # followed by `;` on its own line (`    `;`) — that's the
        # template-literal end + statement terminator. Scan the
        # whole file to find it.
        outer_close = re.search(
            r"^\s*`\s*;\s*$",
            executable[innerhtml_match.end() :],
            re.MULTILINE,
        )
        assert outer_close is not None, (
            "Review-fix M1: outer closing ```;`` for the "
            "`this._root.innerHTML = `…`` template literal not found — "
            "file refactored away from the established `<innerHTML> = "
            "`…`;` shape?"
        )
        close_idx = innerhtml_match.end() + outer_close.end()
        # The post-rewrite tail (next ~600 chars) must contain the three
        # invalidator calls. This pins them as UNCONDITIONAL — they
        # cannot be hidden behind an `if (this._backdrop !== _lastBackdrop)`
        # guard (the pre-existing block at the top of `_render` stays;
        # this is the post-rewrite mirror of the radiator pattern).
        tail = executable[close_idx : close_idx + 600]
        for invalidator in (
            "_invalidateFlameCache",
            "_invalidateSnowCache",
            "_invalidateWindCache",
        ):
            assert f"this.{invalidator}()" in tail, (
                f"Review-fix M1: `this.{invalidator}()` must be called "
                f"unconditionally after the `this._root.innerHTML = `…`` "
                f"rewrite (within ~600 chars of the closing backtick). "
                f"Mirror of qs-radiator-card.js:967-969."
            )

    def test_climate_card_needs_flame_prime_initialized_in_render(self):
        """Review-fix S1 — `_needsFlamePrime` MUST be initialised to
        `true` somewhere in `_render()` (e.g.
        `if (this._needsFlamePrime == null) this._needsFlamePrime =
        true;`) before the prime block consumes it. Otherwise the
        first-paint flame opens at STILL_AMP and lerps up over ~1.5s
        instead of priming directly to DANCE_AMP.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # The initialiser pattern: any `_needsFlamePrime = true` or
        # `_needsFlamePrime == null` guard that lives in `_render`
        # (not just in `_startAnimation`'s lazy-init block).
        # We approximate "in _render" by scanning for ALL occurrences
        # and asserting that at least TWO exist (one in _startAnimation
        # already; one new in _render).
        occurrences = list(re.finditer(r"_needsFlamePrime\s*(?:==|=)\s*(?:null|true)", executable))
        assert len(occurrences) >= 2, (
            "Review-fix S1: `_needsFlamePrime` must be initialised in "
            "`_render()` (in addition to the lazy-init inside "
            "`_startAnimation`) so the first paint primes directly to "
            "DANCE_AMP. Expected at least 2 sites; found "
            f"{len(occurrences)}."
        )

        # The render-side initialiser uses `== null` (so the prime fires
        # exactly once on first paint).
        assert re.search(r"this\._needsFlamePrime\s*==\s*null", executable) is not None, (
            "Review-fix S1: the render-side `_needsFlamePrime` "
            "initialiser must use `== null` so the prime fires exactly "
            "once on first paint."
        )

        # Pass-#2 S4 — the prime CONSUMER must be gated on
        # `this._backdrop === 'flame'`. Without this gate, the prime
        # mutates `_currentFlameAmp` even when the active backdrop is
        # snow / wind / none (no visible bug today since flame state
        # is unused for those, but semantically wrong — a regression
        # that drops the gate could mask future flame-mode bugs).
        assert (
            re.search(
                r"this\._backdrop\s*===\s*'flame'\s*&&\s*this\._needsFlamePrime",
                executable,
            )
            is not None
        ), (
            "Pass-#2 S4: the `_needsFlamePrime` consumer must be gated "
            "on `this._backdrop === 'flame' && this._needsFlamePrime` "
            "so the prime only fires when the active backdrop is flame."
        )

    def test_climate_card_backdrop_hysteresis_at_setpoint(self):
        """Review-fix S4 — `|target - currentTemp| < BACKDROP_DEADBAND_C`
        must use a deadband to avoid flipping flame↔snow on ±0.1°C
        thermostat jitter at equilibrium.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # Constant declaration.
        assert re.search(r"const\s+BACKDROP_DEADBAND_C\s*=", executable) is not None, (
            "Review-fix S4: `BACKDROP_DEADBAND_C` constant must be "
            "declared at the file top (alongside other tuning constants)."
        )

        # The deadband comparison must appear inside the AUTO/HEAT_COOL
        # branch — match `Math.abs(target - currentTemp) <
        # BACKDROP_DEADBAND_C`.
        assert (
            re.search(
                r"Math\.abs\(\s*target\s*-\s*currentTemp\s*\)\s*<\s*"
                r"BACKDROP_DEADBAND_C",
                executable,
            )
            is not None
        ), (
            "Review-fix S4: the AUTO/HEAT_COOL branch must guard the "
            "flame/snow flip with "
            "`Math.abs(target - currentTemp) < BACKDROP_DEADBAND_C`."
        )

        # Pass-#2 S5 — pin the two-arm structure of the deadband block:
        # (a) the "hold previous resolved backdrop" branch checks
        #     `_lastBackdrop === 'flame' || _lastBackdrop === 'snow'`,
        # (b) the no-prior fallback returns based on the temp sign
        #     (`target > currentTemp ? 'flame' : 'snow'`), NOT an
        #     unconditional `return 'flame'`. The "fallback to flame"
        #     of pass-#1 was too aggressive — pass-#2 N5 refines.
        derive_body = _extract_js_function_body(executable, r"const\s+deriveBackdrop\s*=\s*\(\s*\)\s*=>\s*(?=\{)")
        assert derive_body is not None, (
            "Pass-#2 S5: `deriveBackdrop` arrow-function body must be "
            "extractable to inspect the deadband two-arm structure."
        )
        # (a) hold-previous arm.
        assert (
            re.search(
                r"this\._lastBackdrop\s*===\s*'flame'\s*\|\|\s*"
                r"this\._lastBackdrop\s*===\s*'snow'",
                derive_body,
            )
            is not None
        ), (
            "Pass-#2 S5: deadband block must check "
            "`_lastBackdrop === 'flame' || _lastBackdrop === 'snow'` "
            "before holding the previous resolved backdrop."
        )
        # (b) sign-based fallback (pass-#2 N5 — no unconditional flame).
        # Extract the deadband sub-block body (inside
        # `if (Math.abs(target - currentTemp) < BACKDROP_DEADBAND_C)
        # { ... }`) and assert the sign-based ternary is the fallback,
        # NOT an unconditional `return 'flame';`. The non-deadband
        # branch also uses the sign-based ternary, so we must scope
        # this check to the deadband body specifically.
        deadband_body = _extract_js_function_body(
            derive_body,
            r"if\s*\(\s*Math\.abs\(\s*target\s*-\s*currentTemp\s*\)\s*<\s*"
            r"BACKDROP_DEADBAND_C\s*\)\s*(?=\{)",
        )
        assert deadband_body is not None, (
            "Pass-#2 N5: deadband sub-block `if (Math.abs(target - "
            "currentTemp) < BACKDROP_DEADBAND_C) { ... }` must be "
            "extractable inside `deriveBackdrop`."
        )
        # Pass-#2 N5 fallback may be the inline ternary
        # `return target > currentTemp ? 'flame' : 'snow';` OR the
        # pass-#3 N1 helper call `return _signBackdrop(target,
        # currentTemp);`. Accept either form so the test stays robust
        # across the helper-extraction refactor.
        sign_inline = re.search(
            r"return\s+target\s*>\s*currentTemp\s*\?\s*'flame'\s*:\s*'snow'",
            deadband_body,
        )
        sign_helper = re.search(
            r"return\s+_signBackdrop\s*\(\s*target\s*,\s*currentTemp\s*\)",
            deadband_body,
        )
        assert sign_inline is not None or sign_helper is not None, (
            "Pass-#2 N5 / pass-#3 N1: the deadband fallback must be "
            "sign-based — either the inline ternary "
            "`return target > currentTemp ? 'flame' : 'snow';` or the "
            "extracted helper `return _signBackdrop(target, "
            "currentTemp);`. The bare-flame fallback (`return 'flame';`) "
            "ignored the temperature sign on first transition into a "
            "deadband-AUTO state."
        )
        # The bare-flame fallback must NOT exist inside the deadband
        # body (the only remaining `return 'flame'` strings live OUTSIDE
        # the deadband block — in the early HEAT branch and the
        # non-deadband sign-based site, both of which sit outside this
        # extracted sub-block).
        assert not re.search(r"return\s+'flame'\s*;", deadband_body), (
            "Pass-#2 N5: the deadband block must NOT contain an "
            "unconditional `return 'flame';` — use the sign-based "
            "ternary / helper instead so wind/none → deadband-AUTO "
            "transitions respect the temp sign."
        )

    def test_climate_card_snowflake_spawn_uses_halfchord_geometry(self):
        """Review-fix S5 — snowflake spawn `cx` must be biased toward
        the actual visible chord at the spawn-y, not uniformly across
        the bounding box. Pin the `halfChord` formula structurally.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # `CENTER_X` constant introduced alongside `CENTER_CY`.
        assert re.search(r"const\s+CENTER_X\s*=\s*160", executable) is not None, (
            "Review-fix S5: `CENTER_X = 160` must be declared so the "
            "snowflake spawn `cx` and clip `<circle cx=...>` use a "
            "semantically-correct X-centre constant (not `CENTER_CY`)."
        )

        # The halfChord-bounded spawn formula.
        assert re.search(r"halfChord\s*=\s*Math\.sqrt", executable) is not None, (
            "Review-fix S5: snowflake spawn `cx` must compute "
            "`halfChord = Math.sqrt(…)` so it stays inside the visible "
            "chord at the spawn-y."
        )

        # Pass-#2 S1 / N3 — the clipPath x-coordinate must use
        # `CENTER_X`, not `CENTER_CY`. Numerically equivalent today
        # (square viewBox) but the `CENTER_X` comment block claims
        # future-proofing for a non-square viewBox; that claim only
        # holds if EVERY x-coordinate reference uses `CENTER_X`.
        # Review-fix #03 dropped the carve+cancel clipPath approach;
        # the clipPathD is now just the outer-disc circle. The
        # invariant carries over: outer-disc subpath must use
        # `CENTER_X - CLIP_R`. The cover overlay (drawn after the
        # clip group) ALSO uses `CENTER_X` for its `cx` (verified by
        # test_climate_card_override_btn_carve_out via the shared
        # helper).
        builder_match = re.search(
            r"const\s+clipPathD\s*=([\s\S]*?);",
            content,
        )
        assert builder_match is not None, (
            "`clipPathD` builder declaration missing — must be present above the innerHTML template literal."
        )
        builder_block = builder_match.group(1)
        # Outer disc subpath uses `CENTER_X - CLIP_R`.
        assert (
            re.search(
                r"\$\{\s*CENTER_X\s*-\s*CLIP_R\s*\}",
                builder_block,
            )
            is not None
        ), (
            "Pass-#2 S1: the climate card's clipPathD outer-disc "
            "subpath must use `${CENTER_X - CLIP_R}` (not "
            "`${CENTER_CY - CLIP_R}`) so the `CENTER_X` constant's "
            "future-proofing comment is accurate."
        )
        # And the builder must NOT use `CENTER_CY - CLIP_R` for the
        # x-coordinate (the pass-#2 anti-pattern). `CENTER_CY` still
        # appears legitimately as a y-coordinate.
        assert "CENTER_CY - CLIP_R" not in builder_block, (
            "Pass-#2 S1: the climate card's clipPathD must NOT use "
            "`CENTER_CY - CLIP_R` as an x-coordinate. That was the "
            "pre-pass-#2 anti-pattern; use `CENTER_X - CLIP_R`."
        )

    def test_climate_card_safe_number_filters_whitespace_and_infinity(self):
        """Review-fix S6 + S7 — `_safeNumber` must trim string state
        and use `Number.isFinite(n)` (which also excludes ±Infinity).

        QS-199 — `_safeNumber` moved to `shared/qs-card-base.js`. Uses
        `card_source_union` to find the (now-shared) hardened body.
        """

        from tests.utils.card_sources import card_source_union

        content = card_source_union("qs-climate-card.js")
        executable = _strip_js_comments(content)

        # Locate the _safeNumber body. The lookahead `(?=\{)` ensures
        # we match the method DEFINITION, not the various call sites
        # which are followed by `;` or `,`.
        body = _extract_js_function_body(executable, r"_safeNumber\s*\([^)]*\)\s*(?=\{)")
        assert body is not None, "Review-fix S6/S7: `_safeNumber` method body must be extractable for inspection."

        # Trim of string state.
        assert ".trim()" in body, (
            "Review-fix S6: `_safeNumber` must call `.trim()` on the "
            'raw string state so a whitespace-only state (e.g. " ") '
            "doesn't coerce to 0."
        )

        # Number.isFinite return guard.
        assert "Number.isFinite" in body, (
            "Review-fix S6/S7: `_safeNumber` must guard the return "
            "with `Number.isFinite(n)` so `±Infinity` is rejected."
        )

    def test_climate_card_skips_temp_reads_for_fan_only_dry_off(self):
        """Review-fix S8 — when `climateStateOn` is anything other than
        `'auto'` / `'heat_cool'`, the four climate-entity attribute
        reads must be gated so they short-circuit to `null`. The AC5
        wording requires "no attribute reads when they cannot
        influence the outcome".
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # `needsTemps` guard declared.
        assert (
            re.search(
                r"const\s+needsTemps\s*=\s*"
                r"climateStateOn\s*===\s*'auto'\s*\|\|\s*"
                r"climateStateOn\s*===\s*'heat_cool'",
                executable,
            )
            is not None
        ), (
            "Review-fix S8: `const needsTemps = climateStateOn === "
            "'auto' || climateStateOn === 'heat_cool';` must gate the "
            "four climate-entity attribute reads."
        )

        # Each attribute read uses the `needsTemps ? … : null` ternary.
        for attr in (
            "current_temperature",
            "temperature",
            "target_temp_low",
            "target_temp_high",
        ):
            pattern = re.compile(
                rf"needsTemps\s*\?\s*this\._safeNumber\s*\(\s*"
                rf"\{{\s*state\s*:\s*attrs\??\.{attr}\s*\}}\s*,\s*"
                rf"null\s*\)\s*:\s*null"
            )
            assert pattern.search(executable) is not None, (
                f"Review-fix S8: the `{attr}` read must be gated as "
                f"`needsTemps ? this._safeNumber({{state: attrs?.{attr}}}, null) "
                f": null`."
            )

    def test_climate_card_snow_off_state_pile_keeps_scrolling(self):
        """Review-fix S12 — in `_stepSnow`, the snowflake-spawn block
        must be gated on `if (snowing)`, but the wave-scroll loop and
        path-regen block must run unconditionally so the pile keeps
        scrolling at calm rates when the device is off (per AC3's
        "snow could be represented like the pool water when off,
        moving very slowly" contract).
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # Extract _stepSnow body via balanced-brace walk. The signature
        # regex requires an immediately-following `{` (method
        # definition form) so it doesn't match the call site inside
        # `_startAnimation`'s switch — `_stepSnow(ts, dt);` (semicolon)
        # would otherwise hit first and the walker would pick up the
        # wrong body.
        snow_body = _extract_js_function_body(executable, r"_stepSnow\s*\([^)]*\)\s*(?=\{)")
        assert snow_body is not None, "Review-fix S12: `_stepSnow(ts, dt) { … }` method body must be extractable."

        # The per-layer translateX scroll loop must live OUTSIDE the
        # `if (snowing)` guard. Locate the first
        # `.style.transform = `translateX` site (the wave scroll) and
        # the `if (snowing)` guard, and assert the scroll site comes
        # FIRST in the body. That keeps the wave-pile scrolling at
        # calm rates even when the device is off (AC3 contract).
        scroll_site = snow_body.find(".style.transform = `translateX")
        assert scroll_site != -1, (
            "Review-fix S12: `_stepSnow` body must contain a "
            "`<el>.style.transform = `translateX(…)`` assignment "
            "driving the per-layer wave scroll."
        )

        # The `if (snowing)` guard exists.
        snowing_guard = re.search(r"if\s*\(\s*snowing\s*\)", snow_body)
        assert snowing_guard is not None, (
            "Review-fix S12: `_stepSnow` must contain an `if (snowing)` guard for the spawn block."
        )

        # The scroll site must appear BEFORE the `if (snowing)` guard
        # so it runs every frame regardless of running state. The
        # spawn guard sits later in the body.
        assert scroll_site < snowing_guard.start(), (
            "Review-fix S12: the per-layer wave-scroll site must come "
            "BEFORE the `if (snowing)` spawn-gate so the snow pile "
            "keeps scrolling at calm rates when the device is off."
        )

        # Also sanity-check the for-loop driving the scroll is the
        # canonical `for (let i = 0; i < 3; i++)`.
        assert re.search(r"for\s*\(\s*let\s+i\s*=\s*0\s*;\s*i\s*<\s*3\s*;", snow_body) is not None, (
            "Review-fix S12: snow scroll loop must iterate over the 3 wave layers via `for (let i = 0; i < 3; i++)`."
        )


class TestClimateCardReviewFix02Hardening:
    """QS-210 review-fix #02 — defensive snow init, clip CENTER_X,
    initial-paths gating regex test, redundant invalidations removed.
    """

    def test_climate_card_snow_n5_reset_initialises_array_state(self):
        """Pass-#2 M1 — the N5 backdrop-change snow block sets
        `_currentSnowAmp = CALM_SNOW_AMP` BEFORE `_startAnimation()`
        runs. That bypasses the `if (this._currentSnowAmp == null)`
        lazy-init guard in `_startAnimation`, so `_snowflakes` /
        `_snowWavePhase` / `_nextSnowflakeAt` stay undefined on the
        first cool-mode render — first RAF tick of `_stepSnow` then
        crashes with `TypeError: undefined is not iterable` on the
        `for (const b of this._snowflakes)` loop.

        Defence: the N5 block must defensively initialise the three
        missing fields. ADDITIONALLY, `_startAnimation`'s lazy-init
        should use per-field guards so the fix is robust in either
        order.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # The N5 transition block: `if (this._backdrop === 'snow' &&
        # this._lastBackdrop !== 'snow') { ... }`. Extract its body
        # via a balanced-brace walk.
        n5_body = _extract_js_function_body(
            executable,
            r"if\s*\(\s*this\._backdrop\s*===\s*'snow'\s*&&\s*"
            r"this\._lastBackdrop\s*!==\s*'snow'\s*\)\s*(?=\{)",
        )
        assert n5_body is not None, (
            "Pass-#2 M1: the `if (this._backdrop === 'snow' && "
            "this._lastBackdrop !== 'snow') { ... }` N5 reset block "
            "must exist for the defensive initialisation."
        )

        # The N5 block must reset amp/speed AND defensively initialise
        # the three array/scalar fields the snow RAF step depends on.
        assert "this._currentSnowAmp = CALM_SNOW_AMP" in n5_body, (
            "Pass-#2 M1: the N5 block must reset `this._currentSnowAmp = CALM_SNOW_AMP`."
        )
        assert "this._currentSnowSpeed = CALM_SNOW_SPEED" in n5_body, (
            "Pass-#2 M1: the N5 block must reset `this._currentSnowSpeed = CALM_SNOW_SPEED`."
        )
        for guard, field, init in (
            ("this._snowWavePhase == null", "_snowWavePhase", "this._snowWavePhase = 0"),
            ("this._snowflakes == null", "_snowflakes", "this._snowflakes = []"),
            ("this._nextSnowflakeAt == null", "_nextSnowflakeAt", "this._nextSnowflakeAt = 0"),
        ):
            assert guard in n5_body and init in n5_body, (
                f"Pass-#2 M1: the N5 block must defensively initialise "
                f"`{field}` (`if ({guard}) {init};`) so the first RAF "
                f"tick of `_stepSnow` doesn't crash on `undefined`."
            )

        # And the `_startAnimation` lazy-init must use per-field
        # guards (so the fix is robust in either order). The legacy
        # single-guard `if (this._currentSnowAmp == null) { ... all
        # five fields ... }` is fragile: any earlier write to
        # `_currentSnowAmp` skips ALL five inits.
        start_body = _extract_js_function_body(executable, r"_startAnimation\s*\(\s*\)\s*(?=\{)")
        assert start_body is not None, "Pass-#2 M1: `_startAnimation()` body must be extractable."
        for field, init in (
            ("_currentSnowAmp", "this._currentSnowAmp = CALM_SNOW_AMP"),
            ("_currentSnowSpeed", "this._currentSnowSpeed = CALM_SNOW_SPEED"),
            ("_snowWavePhase", "this._snowWavePhase = 0"),
            ("_snowflakes", "this._snowflakes = []"),
            ("_nextSnowflakeAt", "this._nextSnowflakeAt = 0"),
        ):
            # Per-field guard `if (this.<field> == null) <init>;`
            # (allow optional braces around the body).
            pattern = re.compile(
                rf"if\s*\(\s*this\.{field}\s*==\s*null\s*\)\s*"
                rf"(?:\{{\s*)?{re.escape(init)}",
            )
            assert pattern.search(start_body) is not None, (
                f"Pass-#2 M1: `_startAnimation` must use a per-field "
                f"lazy-init for `{field}` "
                f"(`if (this.{field} == null) {init};`) so the snow "
                f"state initialisation is robust to any field being "
                f"primed independently."
            )

    def test_climate_card_initial_paths_gated_on_active_backdrop(self):
        """Pass-#2 S3 — the three pre-gen blocks must each be enclosed
        by `if (this._backdrop === '<x>') { ... }`. Pass-#1 S2 added
        the gates; this test pins them so a future copy-paste can't
        silently lift them back out (the existing AC2/AC3/AC4 tests
        check the emitted SVG, not the JS structure).
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # Each of the three initial-paths variables must be ASSIGNED
        # inside an `if (this._backdrop === '<x>') { ... }` block, NOT
        # at the top level of `_render`. Approximate this by scanning
        # for the pattern `if (this._backdrop === '<x>') { … <var> = `
        # within ~700 chars.
        gating_pairs = [
            ("flame", "initialFlamePaths"),
            ("snow", "initialSnowWavePaths"),
            ("wind", "initialWindPaths"),
        ]
        for backdrop, var_name in gating_pairs:
            # Find the `if (this._backdrop === '<x>')` guard.
            guard_match = re.search(
                rf"if\s*\(\s*this\._backdrop\s*===\s*'{backdrop}'\s*\)\s*\{{",
                executable,
            )
            assert guard_match is not None, (
                f"Pass-#2 S3: missing `if (this._backdrop === "
                f"'{backdrop}') {{ ... }}` guard before the "
                f"`{var_name}` assignment."
            )
            # Walk balanced braces to find the guard's body close.
            depth = 1
            i = guard_match.end()
            while i < len(executable) and depth > 0:
                if executable[i] == "{":
                    depth += 1
                elif executable[i] == "}":
                    depth -= 1
                i += 1
            guard_body = executable[guard_match.end() : i - 1]
            assert var_name in guard_body, (
                f"Pass-#2 S3: `{var_name}` must be assigned INSIDE the "
                f"`if (this._backdrop === '{backdrop}') {{ ... }}` "
                f"block (currently assigned outside — pre-gen runs "
                f"unconditionally, wasting ~3 path-generator calls)."
            )

    def test_climate_card_backdrop_change_block_drops_redundant_invalidators(self):
        """Pass-#2 N1 — the backdrop-change block at the top of
        `_render` no longer calls `_invalidateFlameCache/Snow/Wind()`.
        Those invalidators run unconditionally post-rewrite (M1 from
        pass-#1), so calling them BEFORE the rewrite was redundant
        CPU work. Only the N5 accumulator reset has unique work; that
        stays.
        """

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        # Locate the backdrop-change block:
        # `if (this._backdrop !== this._lastBackdrop) { ... }`.
        bc_body = _extract_js_function_body(
            executable,
            r"if\s*\(\s*this\._backdrop\s*!==\s*this\._lastBackdrop\s*\)\s*(?=\{)",
        )
        assert bc_body is not None, (
            "Pass-#2 N1: the backdrop-change block "
            "`if (this._backdrop !== this._lastBackdrop) { ... }` "
            "must still exist (it carries the N5 accumulator reset)."
        )
        for invalidator in (
            "_invalidateFlameCache",
            "_invalidateSnowCache",
            "_invalidateWindCache",
        ):
            assert f"this.{invalidator}()" not in bc_body, (
                f"Pass-#2 N1: the backdrop-change block must NOT call "
                f"`this.{invalidator}()` — the post-innerHTML M1 block "
                f"now runs the same invalidator unconditionally, so the "
                f"pre-rewrite call is redundant CPU work."
            )

    def test_climate_card_start_animation_drops_redundant_invalidators(self):
        """Pass-#2 N2 — `_startAnimation` no longer calls the three
        `_invalidate*Cache()` helpers before starting RAF. The
        post-innerHTML M1 block has already cleared the caches; the
        pre-RAF call was redundant and `_invalidateSnowCache`'s
        `.remove()` was also detaching live nodes that the imminent
        rewrite was about to replace anyway.
        """

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        start_body = _extract_js_function_body(executable, r"_startAnimation\s*\(\s*\)\s*(?=\{)")
        assert start_body is not None
        for invalidator in (
            "_invalidateFlameCache",
            "_invalidateSnowCache",
            "_invalidateWindCache",
        ):
            assert f"this.{invalidator}()" not in start_body, (
                f"Pass-#2 N2: `_startAnimation` must NOT call "
                f"`this.{invalidator}()` — the post-innerHTML M1 block "
                f"already handles invalidation."
            )

    def test_climate_card_start_animation_lazy_init_below_early_return(self):
        """Pass-#2 N4 — the lazy-init blocks must sit BELOW the
        `if (this._animRaf != null) return;` early-return so they only
        fire when actually starting a fresh RAF loop.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-climate-card.js").read_text()
        executable = _strip_js_comments(content)

        start_body = _extract_js_function_body(executable, r"_startAnimation\s*\(\s*\)\s*(?=\{)")
        assert start_body is not None

        early_return = re.search(
            r"if\s*\(\s*this\._animRaf\s*!=\s*null\s*\)\s*return\s*;",
            start_body,
        )
        assert early_return is not None, (
            "Pass-#2 N4: `_startAnimation` must contain the `if (this._animRaf != null) return;` early-return guard."
        )

        # The lazy-init for at least one of the snow / wind / flame
        # state fields must appear AFTER the early-return.
        for field in ("_currentFlameAmp", "_currentSnowAmp", "_windPhase"):
            lazy_init = re.search(
                rf"if\s*\(\s*this\.{field}\s*==\s*null\s*\)",
                start_body,
            )
            assert lazy_init is not None, f"Pass-#2 N4: `_startAnimation` must lazy-init `{field}`."
            assert lazy_init.start() > early_return.end(), (
                f"Pass-#2 N4: the lazy-init guard for `{field}` must "
                f"appear AFTER the `if (this._animRaf != null) "
                f"return;` early-return so it only fires when starting "
                f"a fresh RAF loop."
            )


# =============================================================================
# QS-199 — Structural tests pinning the shared-base extraction (Phase D2).
# =============================================================================


# Canonical list of methods that — after QS-199 — must appear EXACTLY ONCE
# in the union of `ui/resources/shared/*.js` files and ZERO TIMES in any
# `ui/resources/qs-*-card.js` top-level card file. The single-definition
# rule is what AC1 actually pins (the original LOC-threshold AC was
# replaced per scope-guardian finding #4).
_QS_199_CANONICAL_SHARED_METHODS = (
    "_escapeHtml",
    "_safeNumber",
    "_fmt",
    "_entity",
    "_call",
    "_press",
    "_turnOn",
    "_turnOff",
    "_select",
    "_setNumber",
    "_setTime",
    "_showDialog",
    "_registerKeyActivation",
    "_wireTargetHandle",
    "_wireTimePicker",
    "_wireResetButton",
    "_wirePowerButton",
    "_wireGreenButton",
    "_wireOverrideButton",
    "_wireBistateMode",
    "_buildRingHTML",
    # QS-235 — the bottom-center carve cover helper, defined once on
    # `QsCardBase` and consumed by the car + the three duration cards.
    "_ringCarveCover",
)


def test_no_duplicate_card_method_definitions():
    """QS-199 AC1 — each previously-duplicated block appears in exactly one place.

    For every method in `_QS_199_CANONICAL_SHARED_METHODS`, the grep
    pattern `^\\s*<method>\\s*\\([^)]*\\)\\s*\\{` (a method definition,
    NOT a call site or closure-assignment) should match exactly ONCE
    across all files under `ui/resources/shared/` and ZERO times across
    all `ui/resources/qs-*-card.js` top-level files. Card-side
    `const _foo = (...) => this._foo(...)` aliases are NOT counted
    as definitions (they delegate to the inherited method).

    JS comments are stripped before matching so a doc-comment example
    of the method signature can't accidentally count as a definition.
    """
    import re

    shared_dir = COMPONENT_ROOT / "ui" / "resources" / "shared"
    cards_dir = COMPONENT_ROOT / "ui" / "resources"

    shared_text = _strip_js_comments(
        "\n".join(shared_path.read_text(encoding="utf-8") for shared_path in shared_dir.glob("*.js"))
    )
    cards_text_per_file = {
        card_path.name: _strip_js_comments(card_path.read_text(encoding="utf-8"))
        for card_path in cards_dir.glob("qs-*-card.js")
    }
    assert len(cards_text_per_file) == 6, (
        f"Expected exactly 6 top-level qs-*-card.js files; got {sorted(cards_text_per_file.keys())}"
    )

    duplicates: list[str] = []
    for method in _QS_199_CANONICAL_SHARED_METHODS:
        # Match a method DEFINITION: `<method>(...) {` at line start
        # (modulo indentation). Excludes call sites (which end in `;`
        # or `,`) and closure assignments (which are preceded by `const`).
        pat = re.compile(rf"^\s*{re.escape(method)}\s*\([^)]*\)\s*\{{", re.MULTILINE)
        shared_hits = len(pat.findall(shared_text))
        if shared_hits != 1:
            duplicates.append(f"shared/* defines `{method}` {shared_hits} times (expected 1)")
        for card_name, card_text in cards_text_per_file.items():
            card_hits = len(pat.findall(card_text))
            if card_hits > 0:
                duplicates.append(f"{card_name} still defines `{method}` ({card_hits} times); AC1 requires zero")

    assert not duplicates, "QS-199 AC1 violation — duplicated method definitions found:\n  " + "\n  ".join(duplicates)


@pytest.mark.asyncio
async def test_shared_modules_are_not_lovelace_registered(tmp_path):
    """QS-199 AC3 — files under `ui/resources/shared/` are copied to
    `www/quiet_solar/shared/` but are NOT registered as Lovelace
    resources. Only top-level `qs-*-card.js` files register.
    """
    from custom_components.quiet_solar.ui.dashboard import (
        _async_copy_and_register_resources,
    )
    from tests.test_ui_dashboard import MockResourceStorageCollection

    src = COMPONENT_ROOT / "ui" / "resources"
    dst = tmp_path / "dst"

    resources = MockResourceStorageCollection()

    await _async_copy_and_register_resources(
        str(src),
        str(dst),
        "/local/quiet_solar",
        "TESTTAG",
        resources,
    )

    # No Lovelace registration whose URL contains `/shared/`.
    shared_urls = [item["url"] for item in resources.created_items if "/shared/" in item["url"]]
    assert not shared_urls, f"QS-199 AC3 violation — shared modules registered as Lovelace resources: {shared_urls}"

    # But the shared files WERE copied to the destination.
    for shared_name in (
        "qs-card-styles.js",
        "qs-card-base.js",
        "qs-ring-duration-base.js",
        "qs-anim-flame.js",
        "qs-anim-wave.js",
    ):
        assert (dst / "shared" / shared_name).exists(), (
            f"QS-199 AC3 violation — shared module `shared/{shared_name}` was not copied to destination"
        )

    # QS-199 review-fix #02 S9 — assert the EXACT set of registered card
    # URLs (the six qs-*-card.js), not just the count. A count-only check
    # would still pass if a card URL disappeared and an unrelated
    # top-level JS file took its place, so AC3 wouldn't actually be pinned.
    top_level_urls = [item["url"] for item in resources.created_items if "/shared/" not in item["url"]]
    registered_paths = {url.split("?", 1)[0] for url in top_level_urls}
    expected_paths = {
        "/local/quiet_solar/qs-car-card.js",
        "/local/quiet_solar/qs-climate-card.js",
        "/local/quiet_solar/qs-on-off-duration-card.js",
        "/local/quiet_solar/qs-pool-card.js",
        "/local/quiet_solar/qs-radiator-card.js",
        "/local/quiet_solar/qs-water-boiler-card.js",
    }
    assert registered_paths == expected_paths, (
        f"QS-199 AC3 — registered card set mismatch.\n"
        f"  expected: {sorted(expected_paths)}\n"
        f"  got:      {sorted(registered_paths)}"
    )


@pytest.mark.asyncio
async def test_card_imports_use_cache_busted_url(tmp_path):
    """QS-199 AC2a / AC8 — every copied card file's `from './shared/*.js'`
    imports are rewritten with a `?qs_tag=<tag>` cache-buster, using
    the same `tag` value as the Lovelace registration URL.
    """
    import re

    from custom_components.quiet_solar.ui.dashboard import (
        _async_copy_and_register_resources,
    )
    from tests.test_ui_dashboard import MockResourceStorageCollection

    src = COMPONENT_ROOT / "ui" / "resources"
    dst = tmp_path / "dst"

    resources = MockResourceStorageCollection()

    await _async_copy_and_register_resources(
        str(src),
        str(dst),
        "/local/quiet_solar",
        "STAMPED",
        resources,
    )

    # Every migrated card imports from `./shared/*.js` and the rewrite
    # must inject the same `qs_tag=STAMPED` cache-buster used for
    # the top-level Lovelace registration.
    import_pat = re.compile(r"""from\s+['"]\./shared/[^'"]+\.js\?qs_tag=STAMPED['"]""")

    cards_with_imports = (
        "qs-on-off-duration-card.js",
        "qs-radiator-card.js",
        "qs-pool-card.js",
        "qs-water-boiler-card.js",
        "qs-climate-card.js",
        "qs-car-card.js",
    )
    for card_name in cards_with_imports:
        dst_text = (dst / card_name).read_text(encoding="utf-8")
        matches = import_pat.findall(dst_text)
        assert matches, (
            f"QS-199 AC2a violation — `{card_name}` has no rewritten "
            f"`from './shared/*.js?qs_tag=STAMPED'` imports after copy"
        )

    # Top-level Lovelace URLs also carry the same tag.
    for item in resources.created_items:
        assert "?qs_tag=STAMPED" in item["url"], (
            f"QS-199 AC8 violation — Lovelace registration URL "
            f"{item['url']!r} does not carry the expected `qs_tag=STAMPED` "
            f"cache-buster"
        )


def test_no_closure_form_duplicate_helpers_in_cards():
    """QS-199 review-fix N4 — closure-form duplicates of the shared
    helpers must not reappear in any top-level card.

    The original `test_no_duplicate_card_method_definitions` only catches
    method-FORM definitions (`_foo(...) {`), so the closure-FORM
    duplicates (`const showDialog = (...) => {...}`,
    `const arcPath = (...) => {...}`) that M3 fixed slipped through. This
    second pass greps for `const <name> =` assignments of the canonical
    helper names and asserts zero across all 6 cards.

    Allowed exception: a one-line delegating alias
    `const _registerKeyActivation = (el, action) => this._registerKeyActivation(...)`
    is NOT allowed either — M3/N3 removed those too.
    """
    import re

    cards_dir = COMPONENT_ROOT / "ui" / "resources"
    forbidden_names = (
        "showDialog",
        "_showDialog",
        "_registerKeyActivation",
        "deg2rad",
        "rad2deg",
        "polar",
        "arcPath",
        "pctToDeg",
    )
    pat = re.compile(
        r"const\s+(" + "|".join(re.escape(n) for n in forbidden_names) + r")\s*=",
    )

    offenders: list[str] = []
    for card_path in cards_dir.glob("qs-*-card.js"):
        executable = _strip_js_comments(card_path.read_text(encoding="utf-8"))
        for m in pat.finditer(executable):
            offenders.append(f"{card_path.name}: `const {m.group(1)} =`")

    assert not offenders, (
        "QS-199 review-fix N4 violation — closure-form duplicate helpers "
        "found (should use the inherited shared method / imported helper):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "card_filename",
    [
        "qs-on-off-duration-card.js",
        "qs-radiator-card.js",
        "qs-pool-card.js",
        "qs-water-boiler-card.js",
        "qs-climate-card.js",
        "qs-car-card.js",
    ],
)
def test_all_cards_inherit_hardened_dialog(card_filename):
    """QS-199 review-fix S6 — every card's dialog path exposes the
    N12 (close-fallback) / N13 (try-finally around onClick) / S16
    (keyboard activation) guarantees.

    After M3/M5 all six cards route through the shared `_showDialog` in
    `shared/qs-card-base.js`, so the union of the card + its shared
    modules must contain the hardened implementation. This pins the AC5
    "automatic inheritance" contract across the whole set (previously
    only the radiator card was pinned, which is why the car/climate
    inline-dialog regressions went uncaught).
    """
    from tests.utils.card_sources import card_source_union

    union = card_source_union(card_filename)

    # N12 — empty `buttons` falls back to a "Close" button.
    assert "[{ text: 'Close' }]" in union or "[{text: 'Close'}]" in union, (
        f"{card_filename}: dialog path missing N12 close-fallback (empty buttons → undismissable modal)"
    )
    # N13 — onClick wrapped in try/finally that resets `_modalOpen`.
    assert "_modalOpen = false" in union, (
        f"{card_filename}: dialog path missing N13 `_modalOpen = false` reset in a finally block"
    )
    # S16 — dialog buttons get keyboard activation.
    assert "_registerKeyActivation" in union, f"{card_filename}: dialog/buttons missing S16 keyboard activation"

    # QS-199 review-fix #02 N6 — POSITIVE assertion that the card actually
    # routes its dialogs through the shared base. The single-definition
    # tests prove no inline `_showDialog`/`showDialog` closure exists, but
    # a future regression that builds modal DOM inline (no closure, no
    # method def) would slip past them — so require the card to either
    # call `this._showDialog(` directly OR go through a wire-helper that
    # does (override/reset/time pickers all open dialogs via the base).
    card_text = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text(encoding="utf-8"))
    routes_through_base = "this._showDialog(" in card_text or any(
        f"this.{w}(" in card_text for w in ("_wireOverrideButton", "_wireResetButton", "_wireTimePicker")
    )
    assert routes_through_base, (
        f"{card_filename}: no `this._showDialog(` call and no dialog-opening "
        f"wire-helper — the card may be building modal DOM inline, bypassing "
        f"the hardened base dialog (N6)."
    )


def test_card_does_not_reach_into_flame_engine_privates():
    """QS-199 review-fix M2 — the radiator (and climate) card must NOT
    dot into `QsFlameEngine`'s `_`-prefixed private fields/methods.
    Public accessors (`getCurrentAmp`, `getInitialPaths`, `generatePaths`,
    `step`, `invalidate`, `reset`, `primeForCurrentState`) are the only
    sanctioned surface.
    """
    import re

    cards_dir = COMPONENT_ROOT / "ui" / "resources"
    private_access = re.compile(r"_flameEngine\._[A-Za-z]")

    offenders: list[str] = []
    for card_name in ("qs-radiator-card.js", "qs-climate-card.js"):
        executable = _strip_js_comments((cards_dir / card_name).read_text(encoding="utf-8"))
        if private_access.search(executable):
            offenders.append(card_name)

    assert not offenders, (
        "QS-199 review-fix M2 violation — card reaches into "
        f"`_flameEngine._<private>`: {offenders}. Use the public accessors."
    )


def test_card_source_union_entries_all_exist():
    """QS-199 review-fix N7 — every shared file referenced by
    `CARD_TO_SHARED_FILES` must resolve to an existing file, so a
    rename/typo fails loudly instead of silently dropping that file's
    patterns from the cross-cutting tests.
    """
    from tests.utils.card_sources import CARD_TO_SHARED_FILES

    resources = COMPONENT_ROOT / "ui" / "resources"
    missing: list[str] = []
    for card_name, shared_files in CARD_TO_SHARED_FILES.items():
        if not (resources / card_name).exists():
            missing.append(card_name)
        for shared_name in shared_files:
            if not (resources / "shared" / shared_name).exists():
                missing.append(f"shared/{shared_name} (for {card_name})")

    assert not missing, (
        f"QS-199 review-fix N7 violation — CARD_TO_SHARED_FILES references files that don't exist: {missing}"
    )


def test_shared_css_has_no_branded_colour_literals():
    """QS-199 review-fix #02 S10 — the shared CSS module must contain ZERO
    branded colour hex literals (AC6 policy). Only neutral
    `rgba(0,0,0,*)` / `rgba(255,255,255,*)`, `var(--*)` theme variables,
    and the four documented semantic-anchor colours (power-blue #2196F3,
    solar-green #4CAF50, override-orange #FF9800, pill-on green #2ecc71)
    are allowed. The `palette || {…}` fallback must use only
    `var(--primary-color)` — no pre-refactor branded palette hexes.
    """
    import re

    css_src = (COMPONENT_ROOT / "ui" / "resources" / "shared" / "qs-card-styles.js").read_text(encoding="utf-8")
    # Strip JS comments so the policy doc-comment (which names the
    # allowed anchors) doesn't count as a violation.
    executable = _strip_js_comments(css_src)

    # Semantic anchors (not card-branded) + their alpha (8-digit) variants.
    allowed_anchors = {"#2196F3", "#4CAF50", "#FF9800", "#2ecc71", "#2ecc71aa"}
    # Neutral white / black hexes are allowed (equivalent to the neutral
    # rgba(255,255,255,*) / rgba(0,0,0,*) the policy already permits).
    neutral_hexes = {"#fff", "#ffffff", "#000", "#000000"}
    allowed = {h.lower() for h in (allowed_anchors | neutral_hexes)}
    hexes = set(re.findall(r"#[0-9a-fA-F]{3,8}\b", executable))
    branded = {h for h in hexes if h.lower() not in allowed}

    assert not branded, (
        "QS-199 review-fix #02 S10 — shared/qs-card-styles.js contains "
        f"non-anchor branded colour hex literal(s): {sorted(branded)}. "
        "Branded colours must flow through the `palette` argument; the "
        "fallback must use var(--primary-color)."
    )


def test_base_card_css_merges_palette_defaults_per_key():
    """QS-199 review-fix #04 CR3 — `baseCardCSS` must merge palette
    defaults PER KEY (`{ ...PALETTE_DEFAULTS, ...(palette || {}) }`), not
    only substitute when `palette` is entirely falsy. A partial palette
    like `{ primary: '#FF5722' }` would otherwise emit `undefined` into
    the generated CSS (breaking gradients / `color-mix`). The defaults
    must stay AC6-compliant (all five keys → `var(--primary-color)`).
    """
    import re

    src = (COMPONENT_ROOT / "ui" / "resources" / "shared" / "qs-card-styles.js").read_text(encoding="utf-8")

    # Per-key merge (spread defaults first, then the caller's palette).
    assert re.search(
        r"\{\s*\.\.\.\s*PALETTE_DEFAULTS\s*,\s*\.\.\.\s*\(\s*palette\s*\|\|\s*\{\}\s*\)\s*\}",
        src,
    ), "CR3: `baseCardCSS` must merge per key — `const colors = { ...PALETTE_DEFAULTS, ...(palette || {}) }`."
    # The old whole-object fallback (`palette || { ... }`) must be gone.
    assert "const colors = palette ||" not in src, (
        "CR3: the whole-object `const colors = palette || {…}` fallback "
        "(which leaves a partial palette's missing keys undefined) must "
        "be replaced by the per-key merge."
    )
    # Defaults block declares all five palette keys, AC6-compliant.
    defaults = re.search(r"PALETTE_DEFAULTS\s*=\s*\{(?P<body>[^}]*)\}", src)
    assert defaults is not None, "CR3: PALETTE_DEFAULTS const must exist"
    dbody = defaults.group("body")
    for key in ("primary", "gradStart", "gradEnd", "animStart", "animEnd"):
        assert key in dbody, f"CR3: PALETTE_DEFAULTS missing `{key}`"
    # No branded hex in the defaults (AC6).
    assert not re.search(r"#[0-9a-fA-F]{3,8}", dbody), (
        "CR3: PALETTE_DEFAULTS must use `var(--*)` only — no branded hexes (AC6)."
    )


def test_car_uses_inherited_parse_time_helper():
    """QS-199 review-fix #02 S2 — the car card must use the inherited,
    hardened `this._parseTimeToMinutes` (07:00 fallback for invalid
    states) rather than a local closure that maps `unavailable`/`unknown`
    to midnight.
    """
    import re

    src = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8"))
    assert not re.search(r"const\s+parseTimeToMinutes\s*=", src), (
        "qs-car-card.js still defines a local `parseTimeToMinutes` closure; "
        "use the inherited `this._parseTimeToMinutes` (S2)."
    )
    assert "this._parseTimeToMinutes(" in src, "qs-car-card.js must call the inherited `this._parseTimeToMinutes` (S2)."


def test_cards_escape_entity_derived_option_text():
    """QS-199 review-fix #02 S3/S4 — entity-derived strings interpolated
    into innerHTML (car charger/person options + forecast, pool mode
    options) must route through `this._escapeHtml(...)`.
    """
    import re

    car = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8"))
    # Car option maps must escape the option value.
    assert "this._escapeHtml(o)" in car, "qs-car-card.js: charger/person `<option>` text must be escaped (S3)."
    assert "this._escapeHtml(personForecastStr)" in car, "qs-car-card.js: person forecast string must be escaped (S3)."
    # A raw `<option>${o}</option>` (no escape) must NOT remain.
    assert not re.search(r"<option[^>]*>\$\{o\}</option>", car), (
        "qs-car-card.js: found an unescaped `<option>${o}</option>` (S3)."
    )

    pool = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / "qs-pool-card.js").read_text(encoding="utf-8"))
    assert "this._escapeHtml(o)" in pool and "this._escapeHtml(translatePoolMode(o))" in pool, (
        "qs-pool-card.js: pool-mode option value + label must be escaped (S4)."
    )


def test_in_ring_controls_are_keyboard_reachable():
    """QS-199 review-fix #02 S5/S6 — custom `div` controls that carry
    click/keyboard handlers must be focusable: `role="button"` +
    `tabindex`. Pins the car's sun/rabbit/time controls and the pool's
    power/green toggles.

    QS-199 review-fix #04 CR1 — the car's controls now set tabindex
    CONDITIONALLY (`${ctrlTabAttrs}` → `tabindex="0"` when connected,
    `tabindex="-1" aria-disabled="true"` when disconnected), so the car
    side accepts the interpolated form; the pool toggles keep the literal
    `tabindex="0"`.
    """
    import re

    # Pool: literal tabindex="0"; Car: conditional ${ctrlTabAttrs}.
    checks = {
        "qs-pool-card.js": (("power_btn", "green_btn"), r'role="button"[^>]*tabindex="0"'),
        "qs-car-card.js": (("sun_btn", "rabbit_btn", "time_btn"), r'role="button"[^>]*\$\{ctrlTabAttrs\}'),
    }
    failures: list[str] = []
    for card_name, (ids, attr_pat) in checks.items():
        src = (COMPONENT_ROOT / "ui" / "resources" / card_name).read_text(encoding="utf-8")
        for el_id in ids:
            pat = re.compile(rf'id="{el_id}"[^>]*{attr_pat}', re.DOTALL)
            if not pat.search(src):
                failures.append(f"{card_name}:#{el_id}")

    assert not failures, f"QS-199 review-fix #02 S5/S6 — in-ring controls not keyboard-focusable: {failures}"


@pytest.mark.parametrize(
    "card_filename",
    [
        "qs-on-off-duration-card.js",
        "qs-radiator-card.js",
        "qs-climate-card.js",
        "qs-water-boiler-card.js",
    ],
)
def test_drag_snap_range_derived_from_max_hours(card_filename):
    """QS-199 review-fix #02 S8 — the drag snap points are derived from
    the configured max (`this._allowedHalfHours(maxHours)`) rather than a
    hard-coded `for (let i = 0; i <= 12; ...)` loop, so targets above 12h
    are selectable.
    """
    import re

    src = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text(encoding="utf-8"))
    assert "this._allowedHalfHours(" in src, (
        f"{card_filename}: must derive drag snap points via `this._allowedHalfHours(maxHours)` (S8)."
    )
    assert not re.search(r"for\s*\(\s*let\s+i\s*=\s*0;\s*i\s*<=\s*12;\s*i\s*\+=\s*0\.5\s*\)", src), (
        f"{card_filename}: stale hard-coded `0..12` half-hour loop for the "
        f"drag snap range (S8) — use `_allowedHalfHours`."
    )


def test_flame_engine_keeps_raf_until_idle():
    """QS-199 review-fix #02 S7 — the radiator keeps its RAF loop alive
    until the flame engine reports `isIdle()` (so an on→off transition
    settles to a clean still silhouette), and the engine exposes the
    `isIdle()` accessor.
    """
    from tests.utils.card_sources import card_source_union

    union = card_source_union("qs-radiator-card.js")
    assert "isIdle()" in union, "QsFlameEngine must expose isIdle() (S7)"
    radiator = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text(encoding="utf-8")
    )
    assert "this._flameEngine.isIdle()" in radiator, (
        "qs-radiator-card.js must gate RAF teardown on `_flameEngine.isIdle()` (S7)."
    )


def test_max_hours_clamp_helper_bounds_at_source():
    """QS-199 review-fix #03 M1/N1 + #04 ES1 + #05 S1/N1 (root-cause) — the
    clamp lives at the SOURCE (`_clampMaxHours`) so the gauge and the snap
    list consume the SAME normalized value and can never diverge.

    `_clampMaxHours` must:
      - guard with `Number.isFinite` (prevents the Infinity hang:
        `Number("1e999") === Infinity`),
      - grid-align to the 0.5 snap step (`Math.round(...)`) so the gauge's
        100% lands EXACTLY on a draggable snap value (#05 S1 — no
        top-of-ring dead zone for fractional configs),
      - bound to the shared ceiling (168).
    `_allowedHalfHours` keeps only the `Number.isFinite` defense (no second
    `Math.min` — the snap-only clamp is what desynced it from the gauge in
    #04 ES1; the source value is already grid-aligned + bounded). The `12`
    default (#05 N1) is a single shared module constant.
    """
    import re

    src = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "shared" / "qs-ring-duration-base.js").read_text(encoding="utf-8")
    )

    clamp = re.search(r"_clampMaxHours\s*\([^)]*\)\s*\{(?P<body>.*?)\n    \}", src, re.DOTALL)
    assert clamp is not None, "ES1: `_clampMaxHours` source-clamp helper must exist"
    cbody = clamp.group("body")
    assert "Number.isFinite" in cbody, "ES1: `_clampMaxHours` must guard with Number.isFinite"
    assert "Math.round(" in cbody, (
        "#05 S1: `_clampMaxHours` must grid-align (Math.round to the 0.5 step) "
        "so the gauge max lands on a draggable snap value."
    )

    snap = re.search(r"_allowedHalfHours\s*\([^)]*\)\s*\{(?P<body>.*?)\n    \}", src, re.DOTALL)
    assert snap is not None, "could not locate _allowedHalfHours body"
    sbody = snap.group("body")
    assert "Number.isFinite" in sbody, "M1: `_allowedHalfHours` keeps the `Number.isFinite` Infinity defense."
    assert "Math.min(" not in sbody, (
        "ES1: `_allowedHalfHours` must NOT apply its own `Math.min` clamp — "
        "the clamp moved to `_clampMaxHours` at the source so the snap list "
        "can't diverge from the (also-clamped) gauge."
    )

    # #05 N1 — the `12` default is a single shared constant referenced by
    # both helpers, not two independent magic literals.
    assert re.search(r"const\s+MAX_HOURS_DEFAULT\s*=\s*12\b", src), (
        "#05 N1: define the `12` default once as `MAX_HOURS_DEFAULT`."
    )
    assert "MAX_HOURS_DEFAULT" in cbody and "MAX_HOURS_DEFAULT" in sbody, (
        "#05 N1: both `_clampMaxHours` and `_allowedHalfHours` must reference the shared `MAX_HOURS_DEFAULT` constant."
    )


def test_gauge_max_equals_snap_max_by_construction():
    """QS-199 review-fix #05 S1 — `gauge max == max(snap_list)` BY
    CONSTRUCTION for the representative configs called out in the plan
    (`13.3`, `167.8`, `0.3`, `1e6`, `Infinity`, `12`, `24`, `168`), plus
    non-positive / non-finite fallbacks. The gauge's 100% is `maxHours =
    _clampMaxHours(raw)`; the max draggable target is the last element of
    `_allowedHalfHours(maxHours)`. They must be EQUAL, the snap list must
    never be just `[0]`, and the gauge max must be bounded (≤168).

    Behavioral check via node (the helpers are pure — no `this`). Skipped
    if node isn't installed; the structural pins above guarantee the shape
    regardless.
    """
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available — structural pins cover the shape")

    module_path = COMPONENT_ROOT / "ui" / "resources" / "shared" / "qs-ring-duration-base.js"
    harness = f"""
        globalThis.HTMLElement = class {{}};
        const mod = await import({json.dumps(module_path.as_uri())});
        const proto = mod.QsRingDurationCardBase.prototype;
        const cases = [13.3, 167.8, 0.3, 1e6, Infinity, 12, 24, 168, "1e999", 0, -5, "abc"];
        const out = [];
        for (const v of cases) {{
            const gaugeMax = proto._clampMaxHours(v);
            const list = proto._allowedHalfHours(gaugeMax);
            out.push({{ v: String(v), gaugeMax, snapMax: list[list.length - 1], len: list.length }});
        }}
        console.log(JSON.stringify(out));
    """
    proc = subprocess.run(
        [node, "--input-type=module", "-e", harness],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    for r in results:
        assert r["gaugeMax"] == r["snapMax"], (
            f"#05 S1: gauge/snap divergence for max_default_hours={r['v']}: "
            f"gauge={r['gaugeMax']} snapMax={r['snapMax']}"
        )
        assert r["len"] >= 2, f"#05 S1: snap list degenerate ([0]) for {r['v']}"
        assert r["gaugeMax"] <= 168, f"#05 S1: gauge max unbounded for {r['v']}"


@pytest.mark.parametrize(
    "card_filename",
    [
        "qs-on-off-duration-card.js",
        "qs-radiator-card.js",
        "qs-climate-card.js",
        "qs-water-boiler-card.js",
    ],
)
def test_card_derives_max_hours_via_clamp_helper(card_filename):
    """QS-199 review-fix #04 ES1 + #05 S1 — each duration card derives
    `maxHours` via the shared `this._clampMaxHours(...)` on BOTH the
    default-mode AND the non-default `targetHours` branch, so the SAME
    normalized value feeds the gauge math (hoursToPct / arcs / handle) and
    the snap list (`_allowedHalfHours(maxHours)`). The stale bare
    `Number(cfg.max_default_hours) || 12` (gauge-side, unclamped) must be
    gone, and the raw runtime `targetHours` sensor must NOT feed `maxHours`
    unclamped (#05 S1 face 2: fractional / pathological sensor values).
    """
    import re

    src = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text(encoding="utf-8"))
    assert "this._clampMaxHours(" in src, f"{card_filename}: must derive maxHours via `this._clampMaxHours(...)` (ES1)."
    assert not re.search(r"Number\(cfg\.max_default_hours\)\s*\|\|\s*12", src), (
        f"{card_filename}: stale unclamped `Number(cfg.max_default_hours) || 12` "
        f"(gauge-side) — route it through `this._clampMaxHours` so the gauge and "
        f"snap range agree (ES1)."
    )
    # The snap list consumes the same `maxHours` the gauge uses.
    assert "this._allowedHalfHours(maxHours)" in src, (
        f"{card_filename}: snap list must be `_allowedHalfHours(maxHours)` so it "
        f"shares the clamped source value with the gauge (ES1)."
    )
    # #05 S1 — the non-default branch must NOT assign the raw `targetHours`
    # straight to `maxHours`; it must route through `_clampMaxHours`
    # (grid-align + bound) so a fractional/huge runtime sensor can't
    # desync the snap range or balloon the gauge scale.
    assert not re.search(r"maxHours\s*=\s*targetHours\s*>\s*0\s*\?\s*targetHours\s*:", src), (
        f"{card_filename}: the non-default branch assigns raw `targetHours` to "
        f"`maxHours` unclamped — wrap it in `this._clampMaxHours(...)` (#05 S1)."
    )


def test_wire_target_handle_awaits_commit_hook_before_setnumber():
    """QS-199 review-fix #03 S1 + S2 — the shared `_wireTargetHandle`
    must run an optional `onBeforeCommit` hook (awaited) BEFORE the
    commit, so the pool's `_select(pool_mode, 'bistate_mode_default')`
    lands first (the old pre-migration ordering — writing the duration
    before default mode is active can be rejected/clamped backend-side).

    QS-235 AC1 — re-introduces `onCommit` as a commit OVERRIDE (the car
    maps the dragged value → a select option then `_select(...)` instead
    of `_setNumber`). Unlike the QS-199-NH1 footgun (a dead post-write
    `onCommit`), this `onCommit` REPLACES the `_setNumber` write and is
    awaited in the same position, AFTER `onBeforeCommit`, BEFORE the
    local-target clear timer is scheduled. The default branch (no
    `onCommit`) still calls `_setNumber`, so the duration cards are
    byte-equivalent.
    """
    src = _strip_js_comments(
        (COMPONENT_ROOT / "ui" / "resources" / "shared" / "qs-card-base.js").read_text(encoding="utf-8")
    )
    # QS-235 review-fix #01 SF3 — scope the membership + ordering checks
    # to the `_wireTargetHandle` BODY (not the whole file), so a matching
    # token elsewhere in `qs-card-base.js` can't turn this into a false
    # pass/fail and the ordering pin actually tracks this method.
    body = _extract_js_function_body(src, r"_wireTargetHandle\s*\([^)]*\)\s*\{")
    assert body is not None, (
        "qs-card-base.js: `_wireTargetHandle` method body must be extractable for SF3 inspection."
    )

    assert "await onBeforeCommit(" in body, (
        "S1: `_wireTargetHandle` must `await onBeforeCommit(...)` (pool mode-select)."
    )
    # QS-235 AC1 — the commit-override hook is awaited.
    assert "await onCommit(" in body, (
        "QS-235 AC1: `_wireTargetHandle` must support an `await "
        "onCommit(...)` commit override (the car commits via `_select`)."
    )
    # The default `_setNumber` write survives for the duration cards.
    assert "await this._setNumber(" in body, (
        "QS-235 AC1: the default (no-`onCommit`) branch must still "
        "`await this._setNumber(entityId, ...)` so the duration cards "
        "are unchanged."
    )
    # Ordering: onBeforeCommit fires before BOTH the override commit and
    # the default `_setNumber` write.
    before_idx = body.index("await onBeforeCommit(")
    commit_idx = body.index("await onCommit(")
    setnum_idx = body.index("await this._setNumber(")
    assert before_idx < commit_idx and before_idx < setnum_idx, (
        "S1: `onBeforeCommit` (mode-select) must run BEFORE the commit "
        "(override or the default `_setNumber` write) to preserve the "
        "pool's service-call ordering."
    )


def test_pool_drag_commit_selects_mode_before_duration():
    """QS-199 review-fix #03 S1 — the pool card wires the mode-select into
    the `onBeforeCommit` (pre-duration) hook, not the post-write
    `onCommit`, so default-mode is active before the duration write.
    """
    pool = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / "qs-pool-card.js").read_text(encoding="utf-8"))
    assert "onBeforeCommit:" in pool, (
        "qs-pool-card.js: the drag-commit mode-select must be wired via "
        "`onBeforeCommit:` so it runs before the duration write (S1)."
    )
    assert "bistate_mode_default" in pool, "pool drag-commit must still select default mode"


def test_pool_green_btn_guarded_on_backing_entity():
    """QS-199 review-fix #04 CR2 — the pool green toggle only RENDERS when
    its `green_only` backing entity exists (wiring was already guarded on
    `swGreenOnly`); otherwise it was a visible non-functional control.
    """
    import re

    pool = (COMPONENT_ROOT / "ui" / "resources" / "qs-pool-card.js").read_text(encoding="utf-8")
    # The green_btn markup is gated on `swGreenOnly` (mirrors the wiring).
    assert re.search(r"\$\{swGreenOnly\s*\?\s*`<div id=\"green_btn\"", pool), (
        "CR2: pool `green_btn` must render only when `swGreenOnly` is present."
    )


def test_car_disconnected_controls_leave_tab_order():
    """QS-199 review-fix #04 CR1 — when the car is disconnected the
    pointer-disabled custom controls (sun/rabbit/time) must drop out of
    the keyboard tab order (`tabindex="-1"` + `aria-disabled="true"`)
    instead of being dead tab stops whose action just bails.
    """
    car = _strip_js_comments((COMPONENT_ROOT / "ui" / "resources" / "qs-car-card.js").read_text(encoding="utf-8"))
    assert 'isDisconnected ? \'tabindex="-1" aria-disabled="true"\' : \'tabindex="0"\'' in car, (
        "CR1: car must compute a conditional `ctrlTabAttrs` so disconnected controls leave the tab order."
    )
    # The three in-ring custom controls use the conditional attrs.
    for el_id in ("sun_btn", "rabbit_btn", "time_btn"):
        assert re.search(rf'id="{el_id}"[^>]*\$\{{ctrlTabAttrs\}}', car), (
            f"CR1: car `{el_id}` must use `${{ctrlTabAttrs}}` (conditional tab order)."
        )
