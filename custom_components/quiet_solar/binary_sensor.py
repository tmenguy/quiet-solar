import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from homeassistant.components.binary_sensor import BinarySensorEntity, BinarySensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED, BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS
from .entity import QSDeviceEntity
from .ha_model.car import QSCar
from .home_model.load import AbstractDevice, PilotedDevice

_LOGGER = logging.getLogger(__name__)


def create_ha_binary_sensor_for_PilotedDevice(device: PilotedDevice):
    """Create binary sensors for a PilotedDevice."""
    entities = []

    piloted_activated = QSBinarySensorEntityDescription(
        key="is_piloted_device_activated",
        translation_key=BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED,
        value_fn=lambda d, key: d.is_piloted_device_activated,
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=piloted_activated))

    return entities


def create_ha_binary_sensor_for_QSCar(device: QSCar):
    """Create binary sensors for a PilotedDevice."""
    entities = []

    piloted_activated = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS,
        translation_key=BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS,
        value_fn=lambda d, key: d.can_use_charge_percent_constraints(),
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=piloted_activated))

    return entities


def create_ha_binary_sensor(device: AbstractDevice):
    """Create binary sensors for a device."""
    ret = []

    if isinstance(device, PilotedDevice):
        ret.extend(create_ha_binary_sensor_for_PilotedDevice(device))

    if isinstance(device, QSCar):
        ret.extend(create_ha_binary_sensor_for_QSCar(device))

    return ret


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Quiet Solar binary sensor platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:
        entities = create_ha_binary_sensor(device)
        for attached_device in device.get_attached_virtual_devices():
            entities.extend(create_ha_binary_sensor(attached_device))

        if entities:
            async_add_entities(entities)

    return


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""
    device = hass.data[DOMAIN].get(entry.entry_id)
    if device:
        try:
            if device.home:
                device.home.remove_device(device)
        except Exception as e:
            _LOGGER.error("async_unload_entry binary_sensor: exception for device %s %s", device.name, e, exc_info=True, stack_info=True)

    return True


@dataclass(frozen=True, kw_only=True)
class QSBinarySensorEntityDescription(BinarySensorEntityDescription):
    """Describes Quiet Solar binary sensor entity."""
    value_fn: Callable[[AbstractDevice, str], bool] | None = None


class QSBaseBinarySensor(QSDeviceEntity, BinarySensorEntity):
    """Implementation of a Quiet Solar binary sensor."""

    entity_description: QSBinarySensorEntityDescription

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSBinarySensorEntityDescription,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)

    @callback
    def async_update_callback(self, time: datetime) -> None:
        """Update the entity's state."""
        self._set_availabiltiy()

        if self.entity_description.value_fn is not None:
            state = self.entity_description.value_fn(self.device, self.entity_description.key)
        else:
            state = getattr(self.device, self.entity_description.key, False)

        self._attr_is_on = state
        self.async_write_ha_state()
