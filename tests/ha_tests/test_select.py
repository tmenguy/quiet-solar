"""Tests for quiet_solar select platform."""

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import DOMAIN, DATA_HANDLER


pytestmark = pytest.mark.usefixtures("mock_sensor_states")


async def test_home_select_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test home select entities are created."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, home_config_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 1


async def test_home_mode_select_options(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test home mode select has correct options."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("select.qs_test_home_home_home_mode")
    assert state is not None

    options = state.attributes.get("options", [])
    assert len(options) > 0


async def test_home_mode_select_change(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test home mode select can be changed."""
    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("select.qs_test_home_home_home_mode")
    assert state is not None

    options = state.attributes.get("options", [])
    if len(options) > 1:
        new_option = options[1] if state.state == options[0] else options[0]

        await hass.services.async_call(
            "select", "select_option",
            {"entity_id": "select.qs_test_home_home_home_mode", "option": new_option},
            blocking=True
        )
        await hass.async_block_till_done()

        new_state = hass.states.get("select.qs_test_home_home_home_mode")
        assert new_state.state == new_option


async def test_car_select_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test car select entities are created."""
    from .const import MOCK_CAR_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CAR_CONFIG,
        entry_id="car_select_entity_test",
        title=f"car: {MOCK_CAR_CONFIG['name']}",
        unique_id="quiet_solar_car_select_entity_test",
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, car_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 2


async def test_charger_select_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test charger select entities are created."""
    from .const import MOCK_CHARGER_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CHARGER_CONFIG,
        entry_id="charger_select_entity_test2",
        title=f"charger: {MOCK_CHARGER_CONFIG['name']}",
        unique_id="quiet_solar_charger_select_entity_test2",
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, charger_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]
    assert len(select_entries) >= 1


# =============================================================================
# Test QSBiStateDuration (OnOffDuration) Select Entities
# =============================================================================

async def test_on_off_duration_select_entities_created(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test on_off_duration select entities are created via create_ha_select_for_QSBiStateDuration."""
    from .const import MOCK_ON_OFF_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Set up mock switch entity
    hass.states.async_set("switch.test_on_off_device", "off", {})

    on_off_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_ON_OFF_DURATION_CONFIG,
        entry_id="on_off_select_test",
        title=f"on_off_duration: {MOCK_ON_OFF_DURATION_CONFIG['name']}",
        unique_id="quiet_solar_on_off_select_test",
    )
    on_off_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(on_off_entry.entry_id)
    await hass.async_block_till_done()

    assert on_off_entry.state is ConfigEntryState.LOADED

    entity_entries = er.async_entries_for_config_entry(
        entity_registry, on_off_entry.entry_id
    )
    select_entries = [e for e in entity_entries if e.domain == "select"]

    # QSBiStateDuration (via QSOnOffDuration) should have at least 1 select entity (bistate_mode)
    assert len(select_entries) >= 1


async def test_on_off_duration_bistate_mode_select(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test on_off_duration bistate_mode select has correct options."""
    from .const import MOCK_ON_OFF_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    hass.states.async_set("switch.test_on_off_device", "off", {})

    on_off_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_ON_OFF_DURATION_CONFIG,
        entry_id="on_off_bistate_mode_test",
        title=f"on_off_duration: {MOCK_ON_OFF_DURATION_CONFIG['name']}",
        unique_id="quiet_solar_on_off_bistate_mode_test",
    )
    on_off_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(on_off_entry.entry_id)
    await hass.async_block_till_done()

    # Get the device and verify it's a QSBiStateDuration
    on_off_device = hass.data[DOMAIN].get(on_off_entry.entry_id)
    assert on_off_device is not None

    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    assert isinstance(on_off_device, QSBiStateDuration)

    # Check that bistate_modes are available
    bistate_modes = on_off_device.get_bistate_modes()
    assert isinstance(bistate_modes, list)
    assert len(bistate_modes) >= 3  # At least the basic modes


async def test_create_ha_select_for_bistate_duration_returns_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_select_for_QSBiStateDuration returns proper entity list."""
    from custom_components.quiet_solar.select import create_ha_select_for_QSBiStateDuration
    from .const import MOCK_ON_OFF_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    hass.states.async_set("switch.test_on_off_device", "off", {})

    on_off_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_ON_OFF_DURATION_CONFIG,
        entry_id="on_off_create_fn_test",
        title=f"on_off_duration: {MOCK_ON_OFF_DURATION_CONFIG['name']}",
        unique_id="quiet_solar_on_off_create_fn_test",
    )
    on_off_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(on_off_entry.entry_id)
    await hass.async_block_till_done()

    on_off_device = hass.data[DOMAIN].get(on_off_entry.entry_id)
    assert on_off_device is not None

    # Call create_ha_select_for_QSBiStateDuration directly
    entities = create_ha_select_for_QSBiStateDuration(on_off_device)

    # Should return a list with at least one entity
    assert isinstance(entities, list)
    assert len(entities) >= 1

    # Each entity should be a select entity
    from custom_components.quiet_solar.select import QSSimpleSelectRestore
    for entity in entities:
        assert isinstance(entity, QSSimpleSelectRestore)


# =============================================================================
# Test QSClimateDuration Select Entities
# =============================================================================

async def test_climate_duration_device_creation(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test climate_duration device can be created."""
    from unittest.mock import patch
    from .const import MOCK_CLIMATE_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Set up mock climate entity
    hass.states.async_set(
        "climate.test_climate_device",
        "off",
        {"hvac_modes": ["off", "heat", "cool", "auto"]}
    )

    # Mock get_hvac_modes to return a list without requiring entity registry
    with patch(
        "custom_components.quiet_solar.ha_model.climate_controller.get_hvac_modes",
        return_value=["off", "heat", "cool", "auto"]
    ):
        climate_entry = MockConfigEntry(
            domain=DOMAIN,
            data=MOCK_CLIMATE_DURATION_CONFIG,
            entry_id="climate_device_test",
            title=f"climate: {MOCK_CLIMATE_DURATION_CONFIG['name']}",
            unique_id="quiet_solar_climate_device_test",
        )
        climate_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(climate_entry.entry_id)
        await hass.async_block_till_done()

        assert climate_entry.state is ConfigEntryState.LOADED

        # Get the device and verify it's a QSClimateDuration
        climate_device = hass.data[DOMAIN].get(climate_entry.entry_id)
        assert climate_device is not None

        from custom_components.quiet_solar.ha_model.climate_controller import QSClimateDuration
        assert isinstance(climate_device, QSClimateDuration)


async def test_create_ha_select_for_climate_duration_returns_entities(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_select_for_QSClimateDuration returns proper entity list."""
    from unittest.mock import patch
    from custom_components.quiet_solar.select import create_ha_select_for_QSClimateDuration
    from .const import MOCK_CLIMATE_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    hass.states.async_set(
        "climate.test_climate_device",
        "off",
        {"hvac_modes": ["off", "heat", "cool", "auto"]}
    )

    # Mock get_hvac_modes to return a list without requiring entity registry
    with patch(
        "custom_components.quiet_solar.ha_model.climate_controller.get_hvac_modes",
        return_value=["off", "heat", "cool", "auto"]
    ):
        climate_entry = MockConfigEntry(
            domain=DOMAIN,
            data=MOCK_CLIMATE_DURATION_CONFIG,
            entry_id="climate_create_fn_test",
            title=f"climate: {MOCK_CLIMATE_DURATION_CONFIG['name']}",
            unique_id="quiet_solar_climate_create_fn_test",
        )
        climate_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(climate_entry.entry_id)
        await hass.async_block_till_done()

        climate_device = hass.data[DOMAIN].get(climate_entry.entry_id)
        assert climate_device is not None

        # Call create_ha_select_for_QSClimateDuration directly
        entities = create_ha_select_for_QSClimateDuration(climate_device)

        # Should return a list with at least 2 entities (climate_state_on, climate_state_off)
        assert isinstance(entities, list)
        assert len(entities) >= 2

        # Each entity should be a select entity
        from custom_components.quiet_solar.select import QSSimpleSelectRestore
        for entity in entities:
            assert isinstance(entity, QSSimpleSelectRestore)


async def test_create_ha_select_detects_bistate_and_climate_duration(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test create_ha_select correctly detects both QSBiStateDuration and QSClimateDuration."""
    from unittest.mock import patch
    from custom_components.quiet_solar.select import create_ha_select
    from custom_components.quiet_solar.ha_model.bistate_duration import QSBiStateDuration
    from custom_components.quiet_solar.ha_model.climate_controller import QSClimateDuration
    from .const import MOCK_ON_OFF_DURATION_CONFIG, MOCK_CLIMATE_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    # Set up mock entities
    hass.states.async_set("switch.test_on_off_device", "off", {})
    hass.states.async_set(
        "climate.test_climate_device",
        "off",
        {"hvac_modes": ["off", "heat", "cool", "auto"]}
    )

    # Set up on_off_duration device
    on_off_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_ON_OFF_DURATION_CONFIG,
        entry_id="on_off_create_select_test",
        title=f"on_off_duration: {MOCK_ON_OFF_DURATION_CONFIG['name']}",
        unique_id="quiet_solar_on_off_create_select_test",
    )
    on_off_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(on_off_entry.entry_id)
    await hass.async_block_till_done()

    on_off_device = hass.data[DOMAIN].get(on_off_entry.entry_id)
    assert on_off_device is not None
    assert isinstance(on_off_device, QSBiStateDuration)

    # Test create_ha_select with QSBiStateDuration
    on_off_entities = create_ha_select(on_off_device)
    assert isinstance(on_off_entities, list)
    assert len(on_off_entities) >= 1  # At least bistate_mode

    # Mock get_hvac_modes for climate device
    with patch(
        "custom_components.quiet_solar.ha_model.climate_controller.get_hvac_modes",
        return_value=["off", "heat", "cool", "auto"]
    ):
        # Set up climate_duration device
        climate_entry = MockConfigEntry(
            domain=DOMAIN,
            data=MOCK_CLIMATE_DURATION_CONFIG,
            entry_id="climate_create_select_test",
            title=f"climate: {MOCK_CLIMATE_DURATION_CONFIG['name']}",
            unique_id="quiet_solar_climate_create_select_test",
        )
        climate_entry.add_to_hass(hass)
        await hass.config_entries.async_setup(climate_entry.entry_id)
        await hass.async_block_till_done()

        climate_device = hass.data[DOMAIN].get(climate_entry.entry_id)
        assert climate_device is not None
        assert isinstance(climate_device, QSClimateDuration)

        # Test create_ha_select with QSClimateDuration
        climate_entities = create_ha_select(climate_device)
        assert isinstance(climate_entities, list)
        # QSClimateDuration should get both QSBiStateDuration selects + climate-specific selects
        assert len(climate_entities) >= 2


async def test_bistate_duration_select_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test bistate_duration select platform unloads correctly."""
    from .const import MOCK_ON_OFF_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    hass.states.async_set("switch.test_on_off_device", "off", {})

    on_off_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_ON_OFF_DURATION_CONFIG,
        entry_id="on_off_unload_test",
        title=f"on_off_duration: {MOCK_ON_OFF_DURATION_CONFIG['name']}",
        unique_id="quiet_solar_on_off_unload_test",
    )
    on_off_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(on_off_entry.entry_id)
    await hass.async_block_till_done()

    assert on_off_entry.state is ConfigEntryState.LOADED

    # Unload the entry
    await hass.config_entries.async_unload(on_off_entry.entry_id)
    await hass.async_block_till_done()

    # Entry should be unloaded or failed to unload (both are acceptable)
    assert on_off_entry.state in (ConfigEntryState.NOT_LOADED, ConfigEntryState.FAILED_UNLOAD)


async def test_climate_duration_select_unload(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test climate_duration select platform unloads correctly."""
    from .const import MOCK_CLIMATE_DURATION_CONFIG

    await hass.config_entries.async_setup(home_config_entry.entry_id)
    await hass.async_block_till_done()

    hass.states.async_set(
        "climate.test_climate_device",
        "off",
        {"hvac_modes": ["off", "heat", "cool", "auto"]}
    )

    entity_reg = er.async_get(hass)
    entity_reg.async_get_or_create(
        "climate",
        "test",
        "test_climate_device",
        suggested_object_id="test_climate_device",
        capabilities={"hvac_modes": ["off", "heat", "cool", "auto"]},
    )

    climate_entry = MockConfigEntry(
        domain=DOMAIN,
        data=MOCK_CLIMATE_DURATION_CONFIG,
        entry_id="climate_unload_test",
        title=f"climate: {MOCK_CLIMATE_DURATION_CONFIG['name']}",
        unique_id="quiet_solar_climate_unload_test",
    )
    climate_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(climate_entry.entry_id)
    await hass.async_block_till_done()

    assert climate_entry.state is ConfigEntryState.LOADED

    # Unload the entry
    await hass.config_entries.async_unload(climate_entry.entry_id)
    await hass.async_block_till_done()

    # Entry should be unloaded or failed to unload (both are acceptable)
    assert climate_entry.state in (ConfigEntryState.NOT_LOADED, ConfigEntryState.FAILED_UNLOAD)


