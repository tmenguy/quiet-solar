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
        no_block_comments = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        no_line_comments = re.sub(r"//[^\n]*", "", no_block_comments)

        # Raw `Number(state || fallback)` is the anti-pattern.
        raw_pattern = re.compile(r"Number\(\s*[^()]+\|\|[^()]+\)")
        assert raw_pattern.search(no_line_comments) is None, (
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
        # so the rendered YAML must include the entry.
        assert "climate_hvac_mode_on: heat" in rendered

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


@pytest.mark.parametrize(
    "card_filename,gate",
    [
        ("qs-water-boiler-card.js", "showAnimation"),
        ("qs-on-off-duration-card.js", "showAnimation"),
        ("qs-climate-card.js", "showAnimation"),
        ("qs-car-card.js", "charging"),
    ],
)
def test_card_raf_idle_gated(card_filename, gate):
    """M4: every card except pool gates `_startAnimation` on a
    runtime condition (`showAnimation` for boiler/on-off-duration/
    climate, `charging` for car). Pool's wave is intrinsically
    continuous-while-connected and uses `_startAnimation` from
    `connectedCallback` directly — that file is covered by the
    presence-of-helper check below.
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


