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

from custom_components.quiet_solar.const import (
    CONF_POOL_TEMPERATURE_SENSOR,
    CONF_SOLAR_FORECAST_PROVIDERS,
    CONF_SOLAR_PROVIDER_DOMAIN,
    CONF_SOLAR_PROVIDER_NAME,
    CONF_SWITCH,
    CONF_TYPE_NAME_QSPool,
    DASHBOARD_DEFAULT_SECTIONS,
    DEVICE_TYPE,
    DOMAIN,
    LOAD_TYPE_DASHBOARD_DEFAULT_SECTION,
    CONF_POWER,
    SOLCAST_SOLAR_DOMAIN,
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
    MOCK_SENSOR_STATES,
    MOCK_SOLAR_CONFIG,
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

# Extra mock states needed for pool
EXTRA_MOCK_STATES = {
    "switch.test_pool_pump": {"state": "off", "attributes": {}},
    "sensor.pool_temperature": {
        "state": "22",
        "attributes": {"unit_of_measurement": "°C"},
    },
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
    from pytest_homeassistant_custom_component.common import MockConfigEntry

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
    async def test_solar_entities_appear_in_rendered_output(self, hass, full_dashboard_home):
        """Solar device entities including dynamic ones are present in the rendered YAML."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        assert "qs_solar_forecast_ok" in rendered
        assert "qs_solar_forecast_age" in rendered
        # Dynamic entities created from Solcast provider
        assert "qs_solar_dampening_solcast" in rendered
        assert "qs_solar_forecast_score_solcast" in rendered

    @pytest.mark.asyncio
    async def test_home_forecast_entities_appear_in_rendered_output(self, hass, full_dashboard_home):
        """Home forecast sensors are present in the rendered dashboard YAML."""
        home = full_dashboard_home
        template_path = COMPONENT_ROOT / "ui" / "quiet_solar_dashboard_template.yaml.j2"
        template_content = template_path.read_text()

        tpl = Template(template_content, hass)
        rendered = tpl.async_render(variables={"home": home})

        assert "qs_no_control_forecast" in rendered
        assert "qs_solar_forecast" in rendered


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
                            f"{device.name} ({device.device_type}): "
                            f"entity '{key}' has non-string name: {name!r}"
                        )

        assert failures == [], (
            "Entities with non-string names (likely missing translation):\n"
            + "\n".join(f"  - {f}" for f in failures)
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
                            f"{device.name} ({device.device_type}): "
                            f"entity '{key}' has non-string entity_id: {eid!r}"
                        )

        assert failures == [], (
            "Entities with non-string entity_ids:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )


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

        assert missing == [], (
            "Missing translations:\n" + "\n".join(f"  - {m}" for m in missing)
        )


# ============================================================================
# Test: Default dashboard section assignment
# ============================================================================


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
            ("on_off_duration", "others"),
            ("climate", "climates"),
            ("heat_pump", "climates"),
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

    def test_dashboard_default_section_uses_device_type_not_builtin_type(self):
        """Regression: load.py must use device_type variable, not Python builtin type."""
        assert type not in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION, (
            "LOAD_TYPE_DASHBOARD_DEFAULT_SECTION should not have Python's builtin `type` as a key"
        )

        for key in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION:
            assert isinstance(key, str), (
                f"Key {key!r} in LOAD_TYPE_DASHBOARD_DEFAULT_SECTION is not a string"
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
        # At minimum: home, solar, charger, car, person, battery, on_off, climate, heat_pump, pool
        assert len(all_section_devices) >= 10, (
            f"Expected at least 10 devices in dashboard sections, found {len(all_section_devices)}"
        )
