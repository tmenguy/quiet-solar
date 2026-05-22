"""Tests for dashboard template rendering using real device objects.

These tests catch bugs where:
- Entity descriptions use translation_key with no matching entry in strings.json,
  causing ha_entity.name to return UNDEFINED and crashing the Jinja2 template.
- Dashboard templates reference entity keys that don't exist or produce invalid YAML.
- The LOAD_TYPE_DASHBOARD_DEFAULT_SECTION mapping fails to assign devices to sections.
"""

from __future__ import annotations

import json
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
            "Found a raw `Number(... || ...)` pattern in executable code "
            "— should use `_safeNumber(...)`"
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
        assert re.search(
            r'class="card-title">\s*\$\{(?:this\.)?_escapeHtml\(title\)\}', content
        ) is not None, "Card title is not escaped before innerHTML"
        # Mode labels in the bistate-mode select go through `_escapeHtml`.
        assert re.search(
            r"\$\{(?:this\.)?_escapeHtml\(translateBistateMode\(o\)\)\}", content
        ) is not None, "Bistate mode labels are not escaped before innerHTML"

    def test_radiator_card_s16_keyboard_accessibility(self):  # CR2 — sync (no hass)
        """A2 — pin S16 via regex: each primary action div has
        `role="button"`, `tabindex="0"`, AND a keyboard activation hook.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # Every action `<div>` carries the role/tabindex attribute pair.
        for button_id in ("power_btn", "green_btn", "override_btn", "time_btn"):
            pattern = re.compile(
                rf'id="{button_id}"[^>]*role="button"[^>]*tabindex="0"', re.DOTALL
            )
            assert pattern.search(content) is not None, (
                f"Missing role/tabindex on `{button_id}`"
            )

        # The keyboard helper exists and is wired into every action.
        assert "_registerKeyActivation(" in content
        # At least four calls (one per action div).
        assert content.count("_registerKeyActivation(") >= 4

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
        """
        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        def _try_finally_brackets(call_signature: str) -> bool:
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if call_signature in line:
                    # Look backwards up to 30 lines for `try {`.
                    has_try_before = any(
                        "try {" in lines[j] or "try{" in lines[j]
                        for j in range(max(0, idx - 30), idx)
                    )
                    # Look forwards up to 30 lines for `} finally {`.
                    has_finally_after = any(
                        "finally" in lines[j]
                        for j in range(idx, min(len(lines), idx + 30))
                    )
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
        """AA1 — pin B5: every interpolation inside `showDialog` is escaped.

        Verifies that the `showDialog` function body does NOT contain
        raw `${title}` / `${message}` interpolations against innerHTML
        — they must all route through `_escapeHtml(...)`. Without this
        a future entity-derived title/message would silently
        reintroduce the S15 injection vector.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # Extract `showDialog` body via brace-counting.
        start = content.find("const showDialog")
        assert start != -1, "Missing `showDialog` declaration"
        # Find the `=>` then the opening `{`.
        arrow = content.find("=>", start)
        assert arrow != -1
        body_start = content.find("{", arrow)
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
        assert body_end != -1, "Could not find closing brace of `showDialog`"

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
        pattern = re.compile(
            r"const\s+running\s*=\s*commandReportsOn\s*\|\|\s*liveBackingOn"
        )
        assert pattern.search(content) is not None, (
            "Cold-start `running` fallback is missing — radiator may show as off "
            "during the cold-start grace window"
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
        pattern = re.compile(
            r"liveBackingState\s*===\s*configuredHvacOn"
        )
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
        assert re.search(
            r"const\s+colors\s*=\s*\{[^}]*primary:\s*'#FF5722'", no_comments
        ) is not None

        # Cool-palette literals must NOT appear in executable code.
        for forbidden in ("'#2196F3'", "'#00bcd4'", "'#8bc34a'", "'#00e1ff'", "'#0066ff'"):
            assert forbidden not in no_comments, (
                f"Stale cool-palette literal {forbidden} still present"
            )

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
            "Missing dynamic `id=\"flame${i}\"` template in the flame "
            "<path> emission — H3 parameterised the layer loops but "
            "the template must still emit per-layer path ids."
        )
        # Constants confirm the layer count is still 3 (the assertion
        # would still pass for any forward-compatible extension to N).
        assert re.search(
            r"const\s+LAYER_TEETH_COUNTS\s*=\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
            content,
        ) is not None, (
            "LAYER_TEETH_COUNTS must declare exactly 3 entries (current "
            "layer count); extending the array is supported but a smoke-"
            "test bump should follow."
        )

        # AC-2: clipPath circle at the ring centre. Geometry must be wired
        # through the module-top `CENTER_CY` / `CLIP_R` constants so the
        # SVG cannot silently drift if those constants are tweaked
        # (review fix S2).
        assert re.search(
            r"<clipPath\s+id=\"\$\{flameClipId\}\">\s*<circle\s+cx=\"\$\{CENTER_CY\}\"\s+cy=\"\$\{CENTER_CY\}\"\s+r=\"\$\{CLIP_R\}\"",
            content,
            re.DOTALL,
        ) is not None, (
            "Missing clipPath circle interpolating CENTER_CY / CLIP_R "
            "(must use ${CENTER_CY} / ${CLIP_R}, not hard-coded 160 / 120)"
        )
        # The constants themselves carry the correct geometric values.
        assert re.search(r"const\s+CENTER_CY\s*=\s*160\b", content) is not None
        assert re.search(r"const\s+CLIP_R\s*=\s*120\b", content) is not None

        # AC-2: <g clip-path="url(#${flameClipId})"> wraps the three paths.
        assert 'clip-path="url(#${flameClipId})"' in content

        # AC-7: _invalidateFlameCache body whitelist — EXACTLY three fields cleared.
        inv_match = re.search(
            r"_invalidateFlameCache\s*\(\s*\)\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        assert inv_match is not None, "Missing _invalidateFlameCache method"
        body = inv_match.group(1)
        # Required positive whitelist.
        for required in ("_flameEls", "_lastFlameBaseY", "_lastFlameAmp"):
            assert required in body, (
                f"_invalidateFlameCache must clear `this.{required}`"
            )
        # Forbidden: fields that must SURVIVE disconnect.
        for forbidden in ("_currentFlameAmp", "_currentFlameSpeed", "_flamePhase"):
            assert forbidden not in body, (
                f"_invalidateFlameCache must NOT touch `this.{forbidden}` "
                f"(animation state survives disconnect — mirror pool)"
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
        assert re.search(
            r"progressRatio\s*=\s*maxHours\s*>\s*0\s*\?\s*Math\.max\(\s*0\s*,\s*Math\.min\(\s*1\s*,\s*hoursRun\s*/\s*maxHours",
            no_comments,
        ) is not None, "Missing progressRatio clamp"

        # AC-3: flameBaseY formula uses the 1/5..4/5 envelope.
        assert "FLAME_BASE_MIN_PCT" in no_comments
        assert "FLAME_BASE_MAX_PCT" in no_comments
        assert re.search(
            r"flameBaseY\s*=\s*CENTER_CY\s*\+\s*CLIP_R\s*-\s*\(\s*FLAME_BASE_MIN_PCT\s*\+\s*progressRatio\s*\*\s*\(\s*FLAME_BASE_MAX_PCT\s*-\s*FLAME_BASE_MIN_PCT\s*\)\s*\)\s*\*\s*2\s*\*\s*CLIP_R",
            no_comments,
        ) is not None, "flameBaseY formula does not mirror pool's water-level envelope"

        # AC-8: gate split.
        assert re.search(
            r"const\s+ringDashActive\s*=\s*running\s*&&\s*segLen\s*>\s*6", no_comments
        ) is not None
        assert re.search(
            r"const\s+fireActive\s*=\s*running", no_comments
        ) is not None
        assert re.search(
            r"const\s+showAnimation\s*=\s*ringDashActive\s*\|\|\s*fireActive", no_comments
        ) is not None

        # AC-8: <path id="running_anim"> emission is gated on ringDashActive,
        # not the umbrella showAnimation.
        assert re.search(
            r"\$\{\s*ringDashActive\s*\?\s*`[^`]*?id=\"running_anim\"",
            content,
            re.DOTALL,
        ) is not None, (
            "<path id=\"running_anim\"> must be gated on ringDashActive, not showAnimation"
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
        grey_array = re.search(
            r"FLAME_GREY_FILLS\s*=\s*\[([^\]]+)\]", no_comments
        )
        assert grey_array is not None
        # FLAME_FILLS (warm) also present — both constants required.
        assert "FLAME_FILLS" in no_comments

        # Selection between the two constants is gated on `running`.
        assert re.search(
            r"running\s*\?\s*FLAME_FILLS\s*:\s*FLAME_GREY_FILLS", no_comments
        ) is not None, (
            "Fill-selection must branch on `running` — `running ? FLAME_FILLS : FLAME_GREY_FILLS`"
        )

    def test_radiator_card_text_shadow_for_flame_readability(self):  # CR2 — sync (no hass)
        """QS-201 AC-9 — --ring-text-shadow on :host + applied to 4 ring text classes.

        Without the text-shadow, the centre text collapses in contrast over the
        warm orange flame backdrop. AC-9 is required-for-correctness — pin it
        so a CSS-only regression that drops the shadow can't slip past review.
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

        # CSS variable declaration on :host (whitespace-tolerant).
        assert re.search(
            r"--ring-text-shadow:\s*0 0 12px rgba\(0,0,0,0\.8\),\s*0 2px 4px rgba\(0,0,0,0\.5\)",
            content,
        ) is not None, "AC-9: --ring-text-shadow variable must be declared on :host"

        # Each of the four ring text classes applies the variable.
        for cls in ("target-label", "target-value", "from-to-label", "from-to-value"):
            assert re.search(
                rf"\.ring\s+\.{cls}\s*\{{[^}}]*text-shadow:\s*var\(--ring-text-shadow\)",
                content,
            ) is not None, (
                f"AC-9: .ring .{cls} must apply text-shadow: var(--ring-text-shadow)"
            )

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
        """
        import re

        content = (COMPONENT_ROOT / "ui" / "resources" / "qs-radiator-card.js").read_text()

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
            "QS-204 AC-4: `_generateFlameTeethPath` (the peaked-teeth "
            "path generator) must be declared"
        )

        # Per-tooth tip phase array must drive the flicker.
        assert "_tipPhases" in executable, (
            "QS-204 AC-4: `_tipPhases` (per-layer, per-tooth phase array) "
            "must drive the per-frame flicker"
        )

        # Per-layer flicker frequencies in Hz.
        assert "LAYER_TIP_FLICKER_HZ" in executable, (
            "QS-204 AC-4: `LAYER_TIP_FLICKER_HZ` (per-layer flicker rate) "
            "must declare the per-layer turbulence rates"
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
    async def test_radiator_dashboard_passes_climate_hvac_mode_on(
        self, hass, full_dashboard_home
    ):
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
    async def test_confirmation_text_present_for_all_action_buttons(
        self, hass, full_dashboard_home, template_name
    ):
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

        assert missing == [], (
            f"Template {template_name} missing confirmation text for: {', '.join(missing)}"
        )


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
            ("my_workshop", "mdi:hammer"),   # custom
            ("pools", "mdi:pool"),
            ("garden_lights", "mdi:lamp"),    # custom
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
            f"Duplicate section names must be deduped (first wins). "
            f"Got {result}"
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
            unguarded_pat = re.compile(
                r"maxHours\s*=\s*targetHours\s*;", re.MULTILINE
            )
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
            tail = executable[m.end():m.end() + 200]
            if ".toLowerCase" not in tail:
                continue  # not a stringy use, irrelevant
            preceding = executable[max(0, m.start() - 60) : m.start()]
            if not re.search(r"String\s*\(\s*$", preceding):
                unsafe_uses.append(
                    f"...{preceding[-50:]}<HERE>{executable[m.start() : m.end() + 40]}..."
                )
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
        """
        import re

        for card_filename in ("qs-radiator-card.js", "qs-water-boiler-card.js"):
            content = (COMPONENT_ROOT / "ui" / "resources" / card_filename).read_text()
            executable = _strip_js_comments(content)

            # The arcPath helper definition must include a finiteness
            # check on its angle inputs. Accept either `Number.isFinite`
            # OR `Number.isNaN` (negative form).
            arc_pat = re.compile(
                r"arcPath\s*=\s*\([^)]*\)\s*=>\s*\{(?P<body>[^}]*(?:\}[^}]*)*?)\};",
                re.DOTALL,
            )
            m = arc_pat.search(executable)
            assert m is not None, (
                f"{card_filename}: could not find an `arcPath = (...) => "
                f"{{...}};` arrow definition — check the helper layout."
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
            "home", "battery", "solar", "person", "car", "pool",
            "water_boiler", "on_off_duration", "climate", "heat_pump",
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
                    missing.append(
                        f"{namespace}.step.{step}.data.device_dashboard_section"
                    )

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
async def test_water_boiler_temperature_sensor_row_present_when_configured(
    hass, template_name
):
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
        f"Expected the configured temperature sensor in the boiler card; "
        f"got rows: {entity_ids}"
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
    """
    import re

    js_path = COMPONENT_ROOT / "ui" / "resources" / card_filename
    content = js_path.read_text(encoding="utf-8")
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
        f"await this._select(...)` block in `try { '{' }...{ '}' } finally "
        f"{ '{' }...{ '}' }` so a rejected service call doesn't leave "
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
@pytest.mark.parametrize(
    "card_filename,gate",
    [
        ("qs-on-off-duration-card.js", "showAnimation"),
        ("qs-climate-card.js", "showAnimation"),
        ("qs-car-card.js", "charging"),
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
    """
    import re

    js_path = COMPONENT_ROOT / "ui" / "resources" / card_filename
    content = js_path.read_text(encoding="utf-8")
    assert "_startAnimation" in content and "_stopAnimation" in content, (
        f"M4: {card_filename} must define both _startAnimation and "
        f"_stopAnimation helpers."
    )
    # The render path must call _startAnimation conditionally on `gate`.
    gate_call_pattern = re.compile(
        rf"if\s*\(\s*{re.escape(gate)}\s*\)\s*\{{[^}}]*?_startAnimation",
        re.DOTALL,
    )
    assert gate_call_pattern.search(content), (
        f"M4: {card_filename} must call `_startAnimation()` conditionally "
        f"on `{gate}` from within `_render()`."
    )


def test_pool_card_has_start_stop_helpers():
    """M4 pool-variant: pool's RAF is intrinsically continuous, but the
    `_startAnimation` / `_stopAnimation` helper-naming is still present
    for cross-card consistency.
    """
    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-pool-card.js"
    content = js_path.read_text(encoding="utf-8")
    assert "_startAnimation" in content and "_stopAnimation" in content, (
        "M4: qs-pool-card.js must define _startAnimation/_stopAnimation "
        "helpers for cross-card consistency."
    )


def test_water_boiler_card_uses_safe_number_helper():
    """S8: water-boiler card uses `_safeNumber` instead of
    `Number(s?.state || N)` so `unknown`/`unavailable`/`""` don't
    propagate NaN into SVG path attributes.
    """
    import re

    js_path = COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    content = js_path.read_text(encoding="utf-8")
    assert "_safeNumber" in content, (
        "S8: qs-water-boiler-card.js must define `_safeNumber(...)` helper."
    )
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / card_filename
    ).read_text()
    executable = _strip_js_comments(content)
    # Accept either the literal `0.1` or the named `LERP_DT_CEIL` form
    # (both cards expose the same constant; pool's value is the same).
    pattern = re.compile(
        r"dt\s*=\s*Math\.min\s*\(\s*dt\s*,\s*(?:0\.1|LERP_DT_CEIL)\s*\)"
    )
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    # Strip comments first so the regex only matches executable code.
    executable = _strip_js_comments(content)
    assert re.search(r"_startAnimation\s*\(\s*\)\s*\{", executable), (
        "qs-water-boiler-card.js: missing _startAnimation method"
    )
    assert re.search(r"_stopAnimation\s*\(\s*\)\s*\{", executable), (
        "qs-water-boiler-card.js: missing _stopAnimation method"
    )
    cb_body = _extract_js_function_body(
        executable, r"connectedCallback\s*\(\s*\)\s*"
    )
    assert cb_body is not None, (
        "qs-water-boiler-card.js: connectedCallback() not found"
    )
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    block_match = re.search(
        r"const\s+colors\s*=\s*\{(?P<body>.*?)\};",
        content,
        flags=re.DOTALL,
    )
    assert block_match is not None, (
        "qs-water-boiler-card.js: expected `const colors = { ... };` block"
    )
    body = block_match.group("body")
    for heat_hex in ("#FF5722", "#D32F2F", "#FF6E40", "#E64A19"):
        assert heat_hex in body, (
            f"qs-water-boiler-card.js: heat-palette color {heat_hex} "
            f"missing from `const colors`"
        )
    for cool_hex in ("#2196F3", "#00bcd4", "#8bc34a", "#00e1ff", "#0066ff"):
        assert cool_hex not in body, (
            f"qs-water-boiler-card.js: legacy cool-blue color {cool_hex} "
            f"still present in `const colors`"
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(content)
    assert re.search(r"<clipPath\s+id=", executable), (
        "qs-water-boiler-card.js: missing <clipPath> definition"
    )
    for layer in (
        "wave0_cool",
        "wave0_boil",
        "wave1_cool",
        "wave1_boil",
        "wave2_cool",
        "wave2_boil",
    ):
        assert re.search(rf'id="{layer}"', executable), (
            f'qs-water-boiler-card.js: missing <path id="{layer}">'
        )
    assert re.search(r"_generateWavePath\s*\(", executable), (
        "qs-water-boiler-card.js: missing _generateWavePath method"
    )
    # N9: z-order — the clipped water <g> must precede the bgPath <path>
    # in source order. Pin via index comparison on stable anchor strings
    # that appear exactly once in `_render()`'s SVG template literal.
    g_anchor = '<g clip-path="url(#${waterClipId})">'
    bg_anchor = '<path d="${bgPath}"'
    g_idx = executable.find(g_anchor)
    bg_idx = executable.find(bg_anchor)
    assert g_idx != -1, (
        f"qs-water-boiler-card.js: missing clipped water group anchor "
        f"{g_anchor!r}"
    )
    assert bg_idx != -1, (
        f"qs-water-boiler-card.js: missing bgPath anchor {bg_anchor!r}"
    )
    assert g_idx < bg_idx, (
        "qs-water-boiler-card.js: the clipped water <g> must appear "
        "BEFORE the bgPath <path> in DOM order so the dashed ring, "
        "progress arc, and target handle render on top of the water. "
        f"Got g_idx={g_idx}, bg_idx={bg_idx}."
    )


def test_water_boiler_card_has_bubble_system():
    """QS-200: boiler card has a dynamic bubble system with a soft cap."""
    import re

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(content)
    assert re.search(r"MAX_CONCURRENT_BUBBLES\s*=\s*12", executable), (
        "qs-water-boiler-card.js: missing `MAX_CONCURRENT_BUBBLES = 12` "
        "constant"
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(content)
    assert re.search(r"<filter\s+id=", executable) and (
        "feGaussianBlur" in executable
    ), (
        "qs-water-boiler-card.js: missing <filter>/<feGaussianBlur> in defs"
    )
    assert re.search(r'id="surface_glow"', executable), (
        'qs-water-boiler-card.js: missing <path id="surface_glow">'
    )
    assert re.search(r"mix-blend-mode\s*:\s*screen", executable), (
        "qs-water-boiler-card.js: surface_glow must use "
        "`mix-blend-mode: screen`"
    )
    assert re.search(
        r"SURFACE_GLOW_COLOR\s*=\s*['\"]#FF3D00['\"]", executable
    ), (
        "qs-water-boiler-card.js: SURFACE_GLOW_COLOR must be the "
        "canonical `'#FF3D00'` (Material Deep Orange A400). Any drift "
        "should be deliberate — update this assertion in lock-step."
    )


def test_water_boiler_card_pins_geometry_constants():
    """Review-fix #01 S5: pin the module-level geometry constants
    (AC-2). The SVG layout depends on `CLIP_R`, `CENTER_CX`, `CENTER_CY`,
    and `WAVE_WIDTH` matching specific values that the clipPath
    ``<circle cx cy r>`` markup interpolates. A future tweak that
    changes any of them without re-checking the SVG would silently
    misalign the water animation, so we pin each here.

    `CENTER_CX` is review-fix #01 N4's new constant (was inlined as
    `160` in two places). `CENTER_CY` and `CLIP_R` and `WAVE_WIDTH`
    pre-date this fix but the original test suite never pinned them.
    """
    import re

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    expected = {
        "CENTER_CX": "160",
        "CENTER_CY": "160",
        "CLIP_R": "120",
        "WAVE_WIDTH": "480",
    }
    for name, value in expected.items():
        pat = re.compile(rf"const\s+{name}\s*=\s*{re.escape(value)}\b")
        assert pat.search(content), (
            f"qs-water-boiler-card.js: expected module-level "
            f"`const {name} = {value};` declaration"
        )
    # N5: pin BUBBLE_FILL_COLOR as a named constant too (was inlined
    # as `'rgba(255,255,255,0.85)'` in the bubble createElementNS call).
    assert re.search(
        r"const\s+BUBBLE_FILL_COLOR\s*=\s*['\"]rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*0\.85\s*\)['\"]",
        content,
    ), (
        "qs-water-boiler-card.js: expected module-level "
        "`const BUBBLE_FILL_COLOR = 'rgba(255,255,255,0.85)';`"
    )


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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
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
    ), (
        "qs-water-boiler-card.js (AC-6): expected the boil-layer "
        "opacity literal `this._currentColorMix.toFixed(N)`."
    )
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


def test_water_boiler_card_center_uses_named_constants():
    """Review-fix #02 S1: the arc / handle / progress-ring `center`
    object in `_render()` must use the named `CENTER_CX` / `CENTER_CY`
    constants, not inlined `160` literals. The water clip circle
    moved to `CENTER_CX` / `CENTER_CY` with fix #01 N4; the ring
    geometry must follow so a future tweak to either constant keeps
    the ring, handle, and progress arc aligned with the water layer.
    """
    import re

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(content)
    positive = re.compile(
        r"const\s+center\s*=\s*\{\s*cx\s*:\s*CENTER_CX\s*,\s*cy\s*:\s*CENTER_CY\s*\}"
    )
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
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
    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    clip_open = source.find('<g clip-path="url(#${waterClipId})">')
    assert clip_open != -1, (
        "qs-water-boiler-card.js: missing the water clip-path group "
        "opener `<g clip-path=\"url(#${waterClipId})\">` — has the "
        "QS-200 markup been removed?"
    )
    surface_idx = source.find('id="surface_glow"', clip_open)
    steam_idx = source.find('id="${steamLayerId}"', clip_open)
    assert surface_idx != -1, (
        "qs-water-boiler-card.js: missing `id=\"surface_glow\"` inside "
        "the water clip group (QS-200 anchor)."
    )
    assert steam_idx != -1, (
        "qs-water-boiler-card.js: missing the QS-211 steam layer "
        "`<g id=\"${steamLayerId}\" ...>` inside the water clip group."
    )
    assert surface_idx < steam_idx, (
        "qs-water-boiler-card.js: the steam `<g>` must appear AFTER "
        "`<path id=\"surface_glow\">` in source order so it stacks "
        "above the surface glow inside the clip group."
    )
    steam_group = (
        '<g id="${steamLayerId}" filter="url(#${steamFilterId})" '
        'pointer-events="none"></g>'
    )
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
        "qs-water-boiler-card.js: literal steam group block must appear "
        "inside the water clip group."
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

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
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
    assert advance_match, (
        "qs-water-boiler-card.js: missing the steam advance loop "
        "`for (const p of this._steamPuffs)`."
    )
    cap_clamp = re.search(
        r"if\s*\(\s*this\._nextSteamAt\s*<\s*0\s*\)\s*"
        r"this\._nextSteamAt\s*=\s*0\s*;",
        executable,
    )
    assert cap_clamp, (
        "qs-water-boiler-card.js: missing the cap-recovery clamp "
        "`if (this._nextSteamAt < 0) this._nextSteamAt = 0;`."
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


def test_water_boiler_card_steam_pin_at_rim_and_life_retire():
    """QS-214 pin-at-rim iteration: the per-puff geometry (`dx`,
    `localChordHalf`, `localTopY`) is unchanged from the AC-4 design,
    but the abrupt geometric retire `if (p.cy < localTopY) {...remove}`
    has been replaced by a position pin
    `if (p.cy < localTopY) p.cy = localTopY` so the puff sits at the
    local clip-circle top once it arrives. The sole `remove()` branch
    is now the time-based `p.life >= p.maxLife`, allowing the
    life-curve fade-out to dissolve the puff in place over the last
    30% of its life budget. This addresses the user feedback that
    puffs were "disappearing too abruptly" instead of fading at the
    rim."""
    import re

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(source)
    advance_start = executable.find("for (const p of this._steamPuffs)")
    assert advance_start != -1, (
        "qs-water-boiler-card.js: missing steam advance loop."
    )
    advance_block = executable[advance_start : advance_start + 1500]
    # Per-puff geometry (unchanged from AC-4).
    assert re.search(
        r"const\s+dx\s*=\s*p\.cx\s*-\s*CENTER_CX",
        advance_block,
    ), (
        "qs-water-boiler-card.js: the per-puff geometry must derive "
        "`dx = p.cx - CENTER_CX` inside the advance loop."
    )
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
    # Pin-at-rim: when the puff rises past localTopY, clamp its cy back
    # down so it sits at the rim instead of being removed.
    assert re.search(
        r"if\s*\(\s*p\.cy\s*<\s*localTopY\s*\)\s*p\.cy\s*=\s*localTopY",
        advance_block,
    ), (
        "qs-water-boiler-card.js: the steam advance loop must pin the "
        "puff at the local clip-circle top with "
        "`if (p.cy < localTopY) p.cy = localTopY` — the geometric "
        "retire was replaced by a position pin so the life-curve fade "
        "dissolves the puff in place at the rim rather than blinking "
        "it out."
    )
    # Sole remove() branch: time-based retire only. The negative
    # assertion guards against a future regression that re-introduces
    # the abrupt `p.cy < localTopY || ...` retire.
    assert re.search(
        r"if\s*\(\s*p\.life\s*>=\s*p\.maxLife\s*\)",
        advance_block,
    ), (
        "qs-water-boiler-card.js: the steam advance loop must retire "
        "on `if (p.life >= p.maxLife)` as the sole time-based remove "
        "branch."
    )
    assert not re.search(
        r"p\.cy\s*<\s*localTopY\s*\|\|", advance_block
    ), (
        "qs-water-boiler-card.js: the abrupt geometric retire "
        "`p.cy < localTopY || ...` must NOT appear — it was replaced "
        "by the position pin so the puff fades gracefully at the rim."
    )


def test_water_boiler_card_steam_opacity_formula():
    """QS-214 pin-at-rim iteration: per-frame opacity is the assignment
    `lifeOpacity * this._currentColorMix` (not a compound `*=`), and
    the life-curve breakpoints `0.15` (fade-in) and `0.7` (fade-out)
    appear literally in the steam advance block. The `<circle>` fill
    is the literal `STEAM_FILL_COLOR = 'rgba(195,215,235,0.45)'` (cool
    blue-gray tint, alpha 0.45 baked into the SVG fill literal). The
    rim-fade factor has been removed — the puff is now pinned at
    `localTopY` and the wider `[0.7, 1.0]` life-curve fade-out band
    (30% of life) dissolves it gracefully in place at the rim."""
    import re

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(source)
    assert re.search(
        r"STEAM_FILL_COLOR\s*=\s*['\"]rgba\(195,\s*215,\s*235,\s*0\.45\)['\"]",
        executable,
    ), (
        "qs-water-boiler-card.js: STEAM_FILL_COLOR must be the literal "
        "`'rgba(195,215,235,0.45)'` (cool blue-gray, alpha 0.45 baked in)."
    )
    assert re.search(
        r"const\s+opacity\s*=\s*lifeOpacity\s*\*\s*this\._currentColorMix",
        executable,
    ), (
        "qs-water-boiler-card.js: the per-frame opacity must be an "
        "ASSIGNMENT `const opacity = lifeOpacity * this._currentColorMix` "
        "— the rim-fade factor was removed in favour of pin-at-rim + "
        "wider life-curve fade-out. Compound `*=` would decay "
        "exponentially."
    )
    # Find the steam advance block and assert breakpoints appear inside it.
    advance_start = executable.find("for (const p of this._steamPuffs)")
    assert advance_start != -1, (
        "qs-water-boiler-card.js: missing steam advance loop."
    )
    advance_block = executable[advance_start : advance_start + 1500]
    assert "0.15" in advance_block, (
        "qs-water-boiler-card.js: missing the `0.15` fade-in breakpoint "
        "in the steam life-curve."
    )
    assert re.search(r"lifeT\s*<\s*0\.7\b", advance_block), (
        "qs-water-boiler-card.js: missing the `lifeT < 0.7` fade-out "
        "guard in the steam life-curve (widened from 0.85 so the "
        "fade-out band spans 30% of life)."
    )
    assert re.search(
        r"\(\s*lifeT\s*-\s*0\.7\s*\)\s*/\s*0\.3", advance_block
    ), (
        "qs-water-boiler-card.js: missing the `(lifeT - 0.7) / 0.3` "
        "fade-out ramp in the steam life-curve."
    )
    # Negative guard: the rim-fade factor (rimDistance / rimOpacity /
    # STEAM_RIM_FADE_PX) must NOT appear — it was removed in favour of
    # the pin-at-rim design.
    assert "rimOpacity" not in advance_block, (
        "qs-water-boiler-card.js: `rimOpacity` must NOT appear — the "
        "rim-proximity fade was removed in favour of pin-at-rim + "
        "life-curve fade-out."
    )
    assert "STEAM_RIM_FADE_PX" not in executable, (
        "qs-water-boiler-card.js: `STEAM_RIM_FADE_PX` constant must "
        "NOT appear — removed in the pin-at-rim iteration."
    )


def test_water_boiler_card_steam_ids_derive_from_next_clip_id():
    """QS-211 AC-5: `_steamLayerId` and `_steamFilterId` are assigned
    inside the same `if (!this._waterClipId)` block that owns the other
    per-instance IDs, and derive from the same `uid` variable sourced
    from `QsWaterBoilerCard._nextClipId`. Naming convention:
    `wb_steamLayer_${uid}` / `wb_steamFilter_${uid}`."""
    import re

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(source)
    assert re.search(
        r"this\._steamLayerId\s*=\s*`wb_steamLayer_\$\{uid\}`",
        executable,
    ), (
        "qs-water-boiler-card.js: missing "
        "`this._steamLayerId = \\`wb_steamLayer_${uid}\\`` assignment."
    )
    assert re.search(
        r"this\._steamFilterId\s*=\s*`wb_steamFilter_\$\{uid\}`",
        executable,
    ), (
        "qs-water-boiler-card.js: missing "
        "`this._steamFilterId = \\`wb_steamFilter_${uid}\\`` assignment."
    )
    # Both must live inside the per-instance ID guard, i.e. after the
    # `_nextClipId` bump and before the next consumer (`const waterClipId
    # = this._waterClipId;`).
    guard_open = re.search(
        r"if\s*\(\s*!this\._waterClipId\s*\)\s*\{",
        executable,
    )
    assert guard_open, (
        "qs-water-boiler-card.js: missing the "
        "`if (!this._waterClipId) { ... }` per-instance ID guard."
    )
    consumer = executable.find(
        "const waterClipId = this._waterClipId;",
        guard_open.end(),
    )
    assert consumer != -1, (
        "qs-water-boiler-card.js: missing the "
        "`const waterClipId = this._waterClipId;` consumer line."
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

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(source)
    body = _extract_js_function_body(executable, r"_resetDomRefs\s*\(\s*\)")
    assert body is not None, (
        "qs-water-boiler-card.js: missing `_resetDomRefs()` method."
    )
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

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
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

    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    executable = _strip_js_comments(source)
    body = _extract_js_function_body(
        executable, r"disconnectedCallback\s*\(\s*\)"
    )
    assert body is not None, (
        "qs-water-boiler-card.js: missing `disconnectedCallback()`."
    )
    assert re.search(
        r"this\._steamPuffs\?\.forEach\s*\(\s*p\s*=>\s*p\.el\?\.remove\?\.\(\)\s*\)",
        body,
    ), (
        "qs-water-boiler-card.js: `disconnectedCallback` must eagerly "
        "tear down steam DOM with "
        "`this._steamPuffs?.forEach(p => p.el?.remove?.())`."
    )
    assert re.search(r"this\._steamPuffs\s*=\s*\[\s*\]", body), (
        "qs-water-boiler-card.js: `disconnectedCallback` must clear "
        "`this._steamPuffs = []` after removing nodes."
    )


def test_water_boiler_card_steam_filter_region_attributes():
    """QS-211 AC-9: the steam `<filter>` is declared in `<defs>` with
    the safe-region attributes (`x="-50%" y="-50%" width="200%"
    height="200%"`) so the Gaussian blur isn't clipped at the puff
    bounding box. Mirrors the surface_glow filter precedent."""
    source = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
    filter_decl = (
        '<filter id="${steamFilterId}" x="-50%" y="-50%" '
        'width="200%" height="200%">'
    )
    assert filter_decl in source, (
        "qs-water-boiler-card.js: steam filter declaration must include "
        "the safe-region attributes — expected literal substring "
        f"`{filter_decl}` (mirrors the surface_glow filter precedent)."
    )
    assert (
        '<feGaussianBlur stdDeviation="${STEAM_BLUR_STDDEV}" />' in source
    ), (
        "qs-water-boiler-card.js: steam filter must contain "
        "`<feGaussianBlur stdDeviation=\"${STEAM_BLUR_STDDEV}\" />`."
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

    doc_path = (
        Path(__file__).parent.parent
        / "docs"
        / "agents"
        / "concepts"
        / "dashboard-and-cards.md"
    )
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
        "dashboard-and-cards.md: front-matter must contain a "
        "`last_verified: YYYY-MM-DD` field per AC-10."
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

    content = (
        COMPONENT_ROOT / "ui" / "resources" / "qs-water-boiler-card.js"
    ).read_text()
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
    body = _extract_js_function_body(
        executable, r"if\s*\(\s*this\._pendingReattachCheck\s*\)\s*"
    )
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
    clear_count = len(
        re.findall(r"this\._runningAtStop\s*=\s*undefined", executable)
    )
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
    assert not unconsumed, (
        f"Template emits keys not consumed by qs-water-boiler-card.js: "
        f"{sorted(unconsumed)}"
    )


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
async def test_water_boiler_temperature_sensor_row_absent_when_unset(
    hass, template_name
):
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
        f"Configured temp sensor {expected_temp_id!r} leaked into the "
        f"no-temp boiler card; got rows: {entity_ids}"
    )


