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


def _build_hass_with_states(*states: DummyState):
    class Hass:
        def __init__(self, states):
            self.states = states

    return Hass(DummyStates(*states))


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

