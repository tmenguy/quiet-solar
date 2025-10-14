from unittest.mock import MagicMock

from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    PERCENTAGE,
    UnitOfElectricCurrent,
    UnitOfPower,
)

from custom_components.quiet_solar.config_flow import (
    selectable_amps_entities,
    selectable_percent_number_entities,
    selectable_percent_sensor_entities,
    selectable_power_entities,
)
from custom_components.quiet_solar.const import DOMAIN


class DummyState:
    def __init__(self, entity_id: str, unit: str | None):
        self.entity_id = entity_id
        self.domain = entity_id.split(".")[0]
        attrs = {}
        if unit is not None:
            attrs[ATTR_UNIT_OF_MEASUREMENT] = unit
        self.attributes = attrs


class DummyStates:
    def __init__(self, *states: DummyState):
        self._states = list(states)

    def async_all(self, domains):
        return [state for state in self._states if state.domain in domains]


class DummyEntityEntry:
    def __init__(self, entity_id: str, platform: str):
        self.entity_id = entity_id
        self.platform = platform


class DummyEntityRegistry:
    def __init__(self, entries: dict[str, DummyEntityEntry]):
        self._entries = entries

    def async_get(self, entity_id: str):
        return self._entries.get(entity_id)


def _build_hass_with_states(*states: DummyState, quiet_solar_entities: list[str] | None = None):
    """Build a mock hass object with states and entity registry.
    
    Args:
        *states: DummyState objects to include
        quiet_solar_entities: List of entity IDs that should be marked as quiet_solar entities
    """
    if quiet_solar_entities is None:
        quiet_solar_entities = []
    
    # Create entity registry entries for all entities
    registry_entries = {}
    for state in states:
        platform = DOMAIN if state.entity_id in quiet_solar_entities else "other_integration"
        registry_entries[state.entity_id] = DummyEntityEntry(state.entity_id, platform)
    
    class Hass:
        def __init__(self, states, entity_registry):
            self.states = states
            self._entity_registry = entity_registry

    hass = Hass(DummyStates(*states), DummyEntityRegistry(registry_entries))
    
    # Mock the entity_registry.async_get function to return our dummy registry
    import homeassistant.helpers.entity_registry as er
    original_async_get = er.async_get
    
    def mock_async_get(hass_instance):
        if hass_instance == hass:
            return hass._entity_registry
        return original_async_get(hass_instance)
    
    er.async_get = mock_async_get
    
    return hass


def test_selectable_power_entities_filters_units():
    hass = _build_hass_with_states(
        DummyState("sensor.valid_watt", UnitOfPower.WATT),
        DummyState("sensor.invalid_unit", UnitOfElectricCurrent.AMPERE),
        DummyState("switch.not_sensor", UnitOfPower.WATT),
        DummyState("sensor.no_unit", None),
    )

    result = selectable_power_entities(hass)

    assert result == ["sensor.valid_watt"]


def test_selectable_amps_entities_accepts_current_units():
    hass = _build_hass_with_states(
        DummyState("sensor.phase_current", UnitOfElectricCurrent.AMPERE),
        DummyState("sensor.power", UnitOfPower.WATT),
    )

    result = selectable_amps_entities(hass)

    assert result == ["sensor.phase_current"]


def test_selectable_percent_sensor_entities_includes_percentage():
    hass = _build_hass_with_states(
        DummyState("sensor.charge_percent", PERCENTAGE),
        DummyState("sensor.non_percentage", UnitOfPower.WATT),
    )

    result = selectable_percent_sensor_entities(hass)

    assert result == ["sensor.charge_percent"]


def test_selectable_percent_number_entities_includes_percentage_numbers():
    hass = _build_hass_with_states(
        DummyState("number.charge_limit", PERCENTAGE),
        DummyState("number.power_limit", UnitOfPower.WATT),
        DummyState("sensor.percent_sensor", PERCENTAGE),
    )

    result = selectable_percent_number_entities(hass)

    assert result == ["number.charge_limit"]

