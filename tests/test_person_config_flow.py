"""Tests for QSPerson config flow."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from tests.factories import create_minimal_home_model
from custom_components.quiet_solar.config_flow import QSFlowHandler
from custom_components.quiet_solar.const import (
    CONF_PERSON_PERSON_ENTITY,
    CONF_PERSON_AUTHORIZED_CARS,
    CONF_PERSON_PREFERRED_CAR,
    CONF_PERSON_NOTIFICATION_TIME,
    DEVICE_TYPE,
    DOMAIN,
    DATA_HANDLER,
    CONF_TYPE_NAME_QSPerson,
)


@pytest.fixture
def person_flow_handler(hass: HomeAssistant) -> QSFlowHandler:
    """Flow handler for person config flow tests."""
    flow = QSFlowHandler()
    flow.hass = hass
    flow.config_entry = MockConfigEntry(domain=DOMAIN, entry_id="test_entry", data={})
    return flow


@pytest.mark.asyncio
async def test_person_step_with_person_entities(
    hass: HomeAssistant, person_flow_handler
):
    """Test people config step when person entities exist."""
    hass.states.async_set("person.john_doe", "home", {})
    hass.states.async_set("person.jane_smith", "away", {})

    mock_car1 = MagicMock()
    mock_car1.name = "Tesla Model 3"
    mock_car1.car_is_invited = False
    mock_car2 = MagicMock()
    mock_car2.name = "Nissan Leaf"
    mock_car2.car_is_invited = False
    mock_car3 = MagicMock()
    mock_car3.name = "Guest Car"
    mock_car3.car_is_invited = True

    mock_home = create_minimal_home_model()
    mock_home._cars = [mock_car1, mock_car2, mock_car3]
    mock_data_handler = MagicMock()
    mock_data_handler.home = mock_home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = mock_data_handler

    result = await person_flow_handler.async_step_person()

    assert result["type"] == "form"
    assert result["step_id"] == CONF_TYPE_NAME_QSPerson


@pytest.mark.asyncio
async def test_person_step_no_person_entities(
    hass: HomeAssistant, person_flow_handler
):
    """Test people config step when no person entities exist."""
    result = await person_flow_handler.async_step_person()

    assert result["type"] == "abort"
    assert result["reason"] == "no_person_entities"


@pytest.mark.asyncio
async def test_person_step_filters_invited_cars(
    hass: HomeAssistant, person_flow_handler
):
    """Test that invited cars are filtered out from car options."""
    hass.states.async_set("person.john_doe", "home", {})

    mock_car1 = MagicMock()
    mock_car1.name = "My Tesla"
    mock_car1.car_is_invited = False
    mock_car2 = MagicMock()
    mock_car2.name = "Guest Nissan"
    mock_car2.car_is_invited = True
    mock_car3 = MagicMock()
    mock_car3.name = "My BMW"
    mock_car3.car_is_invited = False
    mock_car4 = MagicMock()
    mock_car4.name = "Visitor Renault"
    mock_car4.car_is_invited = True

    mock_home = create_minimal_home_model()
    mock_home._cars = [mock_car1, mock_car2, mock_car3, mock_car4]
    mock_data_handler = MagicMock()
    mock_data_handler.home = mock_home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = mock_data_handler

    result = await person_flow_handler.async_step_person()

    assert result["type"] == "form"


@pytest.mark.asyncio
async def test_person_step_with_user_input(
    hass: HomeAssistant, person_flow_handler
):
    """Test people config step with user input."""
    hass.states.async_set("person.john_doe", "home", {})

    mock_home = create_minimal_home_model()
    mock_data_handler = MagicMock()
    mock_data_handler.home = mock_home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = mock_data_handler

    user_input = {
        CONF_NAME: "John Doe",
        CONF_PERSON_PERSON_ENTITY: "person.john_doe",
        CONF_PERSON_AUTHORIZED_CARS: ["Tesla Model 3"],
        CONF_PERSON_PREFERRED_CAR: "Tesla Model 3",
        CONF_PERSON_NOTIFICATION_TIME: "08:00:00",
    }

    with patch.object(
        person_flow_handler,
        "async_entry_next",
        return_value={"type": "create_entry"},
    ) as mock_next:
        result = await person_flow_handler.async_step_person(user_input)

        assert mock_next.called
        call_args = mock_next.call_args[0]
        assert call_args[0][CONF_NAME] == "John Doe"
        assert call_args[0][CONF_PERSON_PERSON_ENTITY] == "person.john_doe"
        assert call_args[0][DEVICE_TYPE] == CONF_TYPE_NAME_QSPerson


@pytest.mark.asyncio
async def test_person_creation_from_config_entry(hass: HomeAssistant):
    """Test creating QSPerson from config entry."""
    from custom_components.quiet_solar.ha_model.person import QSPerson
    from custom_components.quiet_solar.ha_model.home import QSHome

    hass.data.setdefault(DOMAIN, {})
    home = QSHome(hass=hass, config_entry=None, name="test home")

    config_entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_person_entry",
        data={
            CONF_NAME: "John Doe",
            CONF_PERSON_PERSON_ENTITY: "person.john_doe",
            CONF_PERSON_AUTHORIZED_CARS: ["Tesla Model 3", "Nissan Leaf"],
            CONF_PERSON_PREFERRED_CAR: "Tesla Model 3",
            CONF_PERSON_NOTIFICATION_TIME: "08:00:00",
        },
    )

    person = QSPerson(
        hass=hass,
        home=home,
        config_entry=config_entry,
        **config_entry.data
    )

    assert person.name == "John Doe"
    assert person.person_entity_id == "person.john_doe"
    assert person.authorized_cars == ["Tesla Model 3", "Nissan Leaf"]
    assert person.preferred_car == "Tesla Model 3"
    assert person.notification_time == "08:00:00"


@pytest.mark.asyncio
async def test_person_creation_direct_params(hass: HomeAssistant):
    """Test backward compatibility with direct parameters."""
    from custom_components.quiet_solar.ha_model.person import QSPerson
    from custom_components.quiet_solar.ha_model.home import QSHome

    hass.data.setdefault(DOMAIN, {})
    home = QSHome(hass=hass, config_entry=None, name="test home")

    person = QSPerson(
        hass=hass,
        home=home,
        config_entry=None,
        name="Jane Smith",
        person_person_entity="person.jane_smith",
    )

    assert person.name == "Jane Smith"
    assert person.person_entity_id == "person.jane_smith"
    assert person.authorized_cars == []
    assert person.preferred_car is None
    assert person.notification_time is None
