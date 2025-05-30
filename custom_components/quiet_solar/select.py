import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import pytz
from homeassistant.components.select import SelectEntityDescription, SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from .entity import QSDeviceEntity
from .ha_model.bistate_duration import QSBiStateDuration
from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric
from .ha_model.home import QSHome, QSHomeMode

from .home_model.load import AbstractDevice
from .const import (
    DOMAIN, CHARGER_NO_CAR_CONNECTED,
)

_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class QSSelectEntityDescription(SelectEntityDescription):
    qs_default_option:str | None  = None


def create_ha_select_for_QSCar(device: QSCar):
    entities = []
    return entities

def create_ha_select_for_QSCharger(device: QSChargerGeneric):
    entities = []

    selected_car_description = QSSelectEntityDescription(
        key="selected_car_for_charger",
        translation_key="selected_car_for_charger",
    )
    entities.append(QSChargerCarSelect(data_handler=device.data_handler, device=device, description=selected_car_description))
    return entities



def create_ha_select_for_QSBiStateDuration(device: QSBiStateDuration):
    entities = []

    bistate_mode_description = QSSelectEntityDescription(
        key="bistate_mode",
        translation_key=device.get_select_translation_key(),
        options= device.get_bistate_modes(),
        qs_default_option="bistate_mode_default"
    )
    entities.append(QSBaseSelectRestore(data_handler=device.data_handler, device=device, description=bistate_mode_description))

    return entities


def create_ha_select_for_QSHome(device: QSHome):
    entities = []

    home_mode_description = QSSelectEntityDescription(
        key="home_mode",
        translation_key="home_mode",
        options= list(map(str, QSHomeMode)),
        qs_default_option=str(QSHomeMode.HOME_MODE_SENSORS_ONLY.value)
    )
    entities.append(QSBaseSelectRestore(data_handler=device.data_handler, device=device, description=home_mode_description))

    return entities


def create_ha_select(device: AbstractDevice):
    ret = []
    if isinstance(device, QSCar):
        ret.extend(create_ha_select_for_QSCar(device))

    if isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_select_for_QSCharger(device))

    if isinstance(device, QSBiStateDuration):
        ret.extend(create_ha_select_for_QSBiStateDuration(device))

    if isinstance(device, QSHome):
        ret.extend(create_ha_select_for_QSHome(device))

    return ret

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_select(device)

        if entities:
            async_add_entities(entities)

    return


class QSBaseSelect(QSDeviceEntity, SelectEntity):
    """Implementation of a Qs Select sensor."""
    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSelectEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self._set_availabiltiy()
        self._do_restore_default = True

    async def async_select_option(self, option: str) -> None:
        """Select an option."""
        self._attr_current_option = option
        try:
            setattr(self.device, self.entity_description.key, option)
        except:
            _LOGGER.info(f"can't set select option {option} on {self.device.name} for {self.entity_description.key}")
        self._set_availabiltiy()
        self.async_write_ha_state()

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        if self.hass is None:
            return

        self._set_availabiltiy()

        #if self.entity_description.value_fn is None:
        if (option := getattr(self.device, self.entity_description.key)) is None:
            return

        self._attr_current_option = option
        self.async_write_ha_state()
    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        self._set_availabiltiy()

        if self._do_restore_default:
            if self.entity_description.qs_default_option:
                new_option = self.entity_description.qs_default_option.value
                await self.async_select_option(new_option)

class QSBaseSelectRestore(QSBaseSelect, RestoreEntity):
    """Entity to represent VAD sensitivity."""

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSelectEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self._set_availabiltiy()
        self._do_restore_default = False


    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        state = await self.async_get_last_state()

        if state is not None and state.state in self.options:
            new_option = state.state
        else:
            new_option = getattr(self.device, self.entity_description.key, None)


        if new_option is None and self.entity_description.qs_default_option:
            new_option = self.entity_description.qs_default_option

        if new_option is None:
            new_option = self.options[0]

        await self.async_select_option(new_option)


@dataclass
class QSExtraStoredDataCarSelect(ExtraStoredData):
    """Object to hold extra stored data."""
    user_selected_car: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the text data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, restored: dict[str, Any]):
        """Initialize a stored text state from a dict."""
        try:
            return cls(
                restored["user_selected_car"],
            )
        except KeyError:
            return None

class QSChargerCarSelect(QSBaseSelect, RestoreEntity):

    device: QSChargerGeneric
    user_selected_car: str | None = None

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSelectEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        if isinstance(device, QSChargerGeneric):
            self._attr_options = device.get_car_options(datetime.now(pytz.UTC))
            self._attr_current_option = device.get_current_selected_car_option()

        super().__init__(data_handler=data_handler, device=device, description=description)

    @property
    def extra_restore_state_data(self) -> QSExtraStoredDataCarSelect:
        """Return sensor specific state data to be restored."""
        return QSExtraStoredDataCarSelect(self.user_selected_car)

    async def async_get_last_select_data(self) -> QSExtraStoredDataCarSelect | None:
        """Restore native_value and native_unit_of_measurement."""
        if (restored_last_extra_data := await self.async_get_last_extra_data()) is None:
            return None
        return QSExtraStoredDataCarSelect.from_dict(restored_last_extra_data.as_dict())


    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        self._attr_options = self.device.get_car_options(datetime.now(pytz.UTC))
        await super().async_added_to_hass()

        last_sensor_state = await self.async_get_last_select_data()
        if (
                not last_sensor_state
        ):
            return
        self.user_selected_car = last_sensor_state.user_selected_car
        self._attr_current_option = self.user_selected_car

        if self.user_selected_car is not None:
            await self.device.set_user_selected_car_by_name(datetime.now(pytz.UTC), self.user_selected_car)

        self.async_write_ha_state()


    async def async_select_option(self, option: str) -> None:
        """Select an option."""
        self._attr_current_option = option
        self.user_selected_car = option
        await self.device.set_user_selected_car_by_name(datetime.now(pytz.UTC), option)
        self.async_write_ha_state()

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        if self.hass is None:
            return

        #if self.entity_description.value_fn is None:
        self._attr_options = self.device.get_car_options(time)
        self._attr_current_option = self.device.get_current_selected_car_option()

        if self._attr_current_option == CHARGER_NO_CAR_CONNECTED:
            self.user_selected_car = None

        self._set_availabiltiy()

        self.async_write_ha_state()



