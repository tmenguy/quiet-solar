import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    BINARY_SENSOR_CAR_API_OK,
    BINARY_SENSOR_CAR_IS_STALE,
    BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS,
    BINARY_SENSOR_HOME_IS_OFF_GRID,
    BINARY_SENSOR_HOME_PERSISTENCE_HEALTH,
    BINARY_SENSOR_HOME_REAL_OFF_GRID,
    BINARY_SENSOR_PILOTED_DEVICE_ACTIVATED,
    BINARY_SENSOR_SOLAR_FORECAST_OK,
    DOMAIN,
)
from .entity import QSDeviceEntity
from .ha_model.car import QSCar
from .ha_model.home import QSHome
from .ha_model.solar import QSSolar
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


def create_ha_binary_sensor_for_QSHome(device: QSHome):
    """Create binary sensors for a QSHome."""
    entities = []

    qs_off_grid = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_HOME_IS_OFF_GRID, translation_key=BINARY_SENSOR_HOME_IS_OFF_GRID
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=qs_off_grid))

    real_off_grid = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_HOME_REAL_OFF_GRID,
        translation_key=BINARY_SENSOR_HOME_REAL_OFF_GRID,
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=real_off_grid))

    persistence_health = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_HOME_PERSISTENCE_HEALTH,
        translation_key=BINARY_SENSOR_HOME_PERSISTENCE_HEALTH,
        device_class=BinarySensorDeviceClass.PROBLEM,
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=persistence_health))

    return entities


def create_ha_binary_sensor_for_QSCar(device: QSCar):
    """Create binary sensors for a QSCar."""
    entities = []

    piloted_activated = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS,
        translation_key=BINARY_SENSOR_CAR_USE_CHARGE_PERCENT_CONSTRAINTS,
        value_fn=lambda d, key: d.can_use_charge_percent_constraints(),
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=piloted_activated))

    # Raw API health (ignores select override)
    api_ok = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_CAR_API_OK,
        translation_key=BINARY_SENSOR_CAR_API_OK,
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
        value_fn=lambda d, key: d.is_car_api_ok(d._get_time_for_sensor()),
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=api_ok))

    # Effective stale status (combines detection + select override)
    is_stale = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_CAR_IS_STALE,
        translation_key=BINARY_SENSOR_CAR_IS_STALE,
        value_fn=lambda d, key: d.is_car_effectively_stale(d._get_time_for_sensor()),
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=is_stale))

    return entities


def create_ha_binary_sensor_for_QSSolar(device: QSSolar):
    """Create binary sensors for a QSSolar."""
    entities = []
    if not device.solar_forecast_providers:
        return entities

    forecast_ok = QSBinarySensorEntityDescription(
        key=BINARY_SENSOR_SOLAR_FORECAST_OK,
        translation_key=BINARY_SENSOR_SOLAR_FORECAST_OK,
        value_fn=lambda d, key: d.is_forecast_ok(),
    )
    entities.append(QSBaseBinarySensor(data_handler=device.data_handler, device=device, description=forecast_ok))

    return entities


def create_ha_binary_sensor(device: AbstractDevice):
    """Create binary sensors for a device."""
    ret = []

    if isinstance(device, PilotedDevice):
        ret.extend(create_ha_binary_sensor_for_PilotedDevice(device))

    if isinstance(device, QSCar):
        ret.extend(create_ha_binary_sensor_for_QSCar(device))

    if isinstance(device, QSHome):
        ret.extend(create_ha_binary_sensor_for_QSHome(device))

    if isinstance(device, QSSolar):
        ret.extend(create_ha_binary_sensor_for_QSSolar(device))

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
            _LOGGER.error(
                "async_unload_entry binary_sensor: exception for device %s %s",
                device.name,
                e,
                exc_info=True,
                stack_info=True,
            )

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
