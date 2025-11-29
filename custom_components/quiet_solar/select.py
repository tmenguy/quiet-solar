import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable


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
from .const import DOMAIN


_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class QSSelectEntityDescription(SelectEntityDescription):
    qs_default_option:str | None  = None
    get_available_options_fn: Callable[[AbstractDevice, str], list[str] | None] | None = None
    get_current_option_fn   : Callable[[AbstractDevice, str], str|None] | None = None
    async_set_current_option_fn   : Callable[[AbstractDevice, str, str], Any] | None = None




def create_ha_select_for_QSCharger(device: QSChargerGeneric):
    entities = []

    selected_car_description = QSSelectEntityDescription(
        key="selected_car_for_charger",
        translation_key="selected_car_for_charger",
        get_available_options_fn=lambda device, key: device.get_car_options(),
        get_current_option_fn=lambda device, key: device.get_current_selected_car_option(),
        async_set_current_option_fn=lambda device, key, option: device.set_user_selected_car_by_name(option),

    )
    # use QSBaseSelect as it needs to be recomputed every time, the information is stored on the charger constraint load infos
    entities.append(QSBaseSelect(data_handler=device.data_handler, device=device, description=selected_car_description))
    return entities

def create_ha_select_for_QSCar(device: QSCar):
    entities = []

    selected_car_description = QSSelectEntityDescription(
        key="selected_charger_for_car",
        translation_key="selected_charger_for_car",
        get_available_options_fn=lambda device, key: device.get_charger_options(),
        get_current_option_fn=lambda device, key: device.get_current_selected_charger_option(),
        async_set_current_option_fn=lambda device, key, option: device.set_user_selected_charger_by_name(option),

    )
    # use QSBaseSelect as it needs to be recomputed every time, the information is stored on the charger constraint load infos
    entities.append(QSBaseSelect(data_handler=device.data_handler, device=device, description=selected_car_description))


    # the selector will automatically be in kwh or % depending on the car capabilities
    selected_car_description = QSSelectEntityDescription(
        key="selected_next_charge_limit_for_car",
        translation_key="selected_next_charge_limit_for_car",
        get_available_options_fn=lambda device, key: device.get_car_next_charge_values_options(),
        get_current_option_fn=lambda device, key: device.get_car_target_charge_option(),
        async_set_current_option_fn=lambda device, key, option: device.user_set_next_charge_target(option),
    )
    entities.append(QSSimpleSelectRestore(data_handler=device.data_handler, device=device, description=selected_car_description))


    # the person force selector
    selected_car_description = QSSelectEntityDescription(
        key="selected_person_for_car",
        translation_key="selected_person_for_car",
        get_available_options_fn=lambda device, key: device.get_car_persons_options(),
        get_current_option_fn=lambda device, key: device.get_car_person_option(),
        async_set_current_option_fn=lambda device, key, option: device.set_user_person_for_car(option),
    )
    # use QSBaseSelect as it needs to be recomputed every time, the information is stored on the car device infos
    entities.append(QSBaseSelect(data_handler=device.data_handler, device=device, description=selected_car_description))

    return entities


def create_ha_select_for_QSBiStateDuration(device: QSBiStateDuration):
    entities = []

    bistate_mode_description = QSSelectEntityDescription(
        key="bistate_mode",
        translation_key=device.get_select_translation_key(),
        options= device.get_bistate_modes(),
        qs_default_option="bistate_mode_default"
    )
    entities.append(QSSimpleSelectRestore(data_handler=device.data_handler, device=device, description=bistate_mode_description))

    return entities


def create_ha_select_for_QSHome(device: QSHome):
    entities = []

    home_mode_description = QSSelectEntityDescription(
        key="home_mode",
        translation_key="home_mode",
        options= list(map(str, QSHomeMode)),
        qs_default_option=str(QSHomeMode.HOME_MODE_SENSORS_ONLY.value)
    )
    entities.append(QSSimpleSelectRestore(data_handler=device.data_handler, device=device, description=home_mode_description))

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
    """Set up the Quiet Solar select platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_select(device)
        for attached_device in device.get_attached_virtual_devices():
            entities.extend(create_ha_select(attached_device))

        if entities:
            async_add_entities(entities)

    return

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    device = hass.data[DOMAIN].get(entry.entry_id)
    if device:
        try:
            if device.home:
                device.home.remove_device(device)
        except Exception as e:
            _LOGGER.error("async_unload_entry select: exception for device %s %s", device.name, e, exc_info=True, stack_info=True)


    return True

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
        self._attr_options = self.get_available_options(description, device)
        self._set_availabiltiy()
        self._do_restore_default = False

    def get_available_options(self, description:QSSelectEntityDescription|None = None, device:AbstractDevice|None=None) -> list[str] | None:
        """Return the available options."""
        if description is None:
            description = self.entity_description
        if device is None:
            device = self.device

        if description.get_available_options_fn is None:
            return description.options
        else:
            return description.get_available_options_fn(device, description.key)

    def get_current_option(self) -> str | None:
        """Return the current option."""
        if self.entity_description.get_current_option_fn is None:
            return getattr(self.device, self.entity_description.key)
        else:
            return self.entity_description.get_current_option_fn(self.device, self.entity_description.key)

    async def set_current_option(self, option):
        """Return the current option."""
        if self.entity_description.async_set_current_option_fn is None:
            try:
                setattr(self.device, self.entity_description.key, option)
            except:
                _LOGGER.info(
                    f"can't set select option {option} on {self.device.name} for {self.entity_description.key}")
        else:
            await self.entity_description.async_set_current_option_fn(self.device, self.entity_description.key, option)
        if self.device.home:
            await self.device.home.force_update_all()

    async def async_select_option(self, option: str | None) -> None:
        """Select an option."""
        _LOGGER.info(f"QSBaseSelect:async_select_option: {option} {self.entity_description.key} on {self.device.name}")

        self._attr_current_option = option
        await self.set_current_option(option)
        self._set_availabiltiy()
        self.async_write_ha_state()

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        if self.hass is None:
            return

        self._set_availabiltiy()

        self._attr_options = self.get_available_options()
        self._attr_current_option = self.get_current_option()

        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        self._attr_options = self.get_available_options()
        self._attr_current_option = self.get_current_option()

        await super().async_added_to_hass()

        self._set_availabiltiy()


class QSBaseSelectRestore(QSBaseSelect, RestoreEntity):
    """Entity."""

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSelectEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)


class QSSimpleSelectRestore(QSBaseSelectRestore):
    """Entity."""

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        state = await self.async_get_last_state()

        if state is not None and state.state in self.options:
            new_option = state.state
        else:
            new_option = self.get_current_option()

        if new_option is None and self.entity_description.qs_default_option:
            new_option = self.entity_description.qs_default_option

        if new_option is None:
            new_option = self.options[0]

        await self.async_select_option(new_option)


@dataclass
class QSExtraStoredDataSelect(ExtraStoredData):
    """Object to hold extra stored data."""
    user_selected_option: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the text data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, restored: dict[str, Any]):
        """Initialize a stored text state from a dict."""
        try:
            return cls(
                restored["user_selected_option"],
            )
        except Exception as e:
            _LOGGER.error("QSExtraStoredDataSelect.from_dict exception %s %s", restored,e, exc_info=True, stack_info=True)
            return None

class QSUserOverrideSelectRestore(QSBaseSelectRestore):

    user_selected_option: str | None = None

    @property
    def extra_restore_state_data(self) -> QSExtraStoredDataSelect:
        """Return sensor specific state data to be restored."""
        return QSExtraStoredDataSelect(self.user_selected_option)

    async def async_get_last_select_data(self) -> QSExtraStoredDataSelect | None:
        """Restore native_value and native_unit_of_measurement."""
        if (restored_last_extra_data := await self.async_get_last_extra_data()) is None:
            return None
        return QSExtraStoredDataSelect.from_dict(restored_last_extra_data.as_dict())

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()

        last_sensor_state = await self.async_get_last_select_data()

        if not last_sensor_state:
            user_option = None
        else:
            user_option = last_sensor_state.user_selected_option

        self.user_selected_option = user_option
        self._attr_current_option = self.user_selected_option

        await self.async_select_option(self.user_selected_option)

    async def async_select_option(self, option: str) -> None:
        """Select an option."""
        self.user_selected_option = option
        await super().async_select_option(option)


