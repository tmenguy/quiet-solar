from abc import abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, Any

from homeassistant.components.sensor import SensorEntityDescription, SensorEntity, RestoreSensor, SensorExtraStoredData
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData
from homeassistant.helpers.typing import StateType

from . import QSDataHandler
from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    DATA_HANDLER, DEVICE_TYPE
)
from .entity import QSDeviceEntity
from .home_model.load import AbstractDevice


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = device.create_ha_entities(Platform.SENSOR)

        if entities:
            async_add_entities(entities)

    return


@dataclass(frozen=True, kw_only=True)
class QSSensorEntityDescription(SensorEntityDescription):
    """Describes Netatmo sensor entity."""
    qs_name: str | None = None


class QSBaseSensor(QSDeviceEntity, SensorEntity):
    """Implementation of a Netatmo sensor."""

    entity_description: QSSensorEntityDescription

    def __init__(
        self,
        data_handler : QSDataHandler,
        qs_device: AbstractDevice,
        description: QSSensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=qs_device)
        self.entity_description = description

        self._attr_unique_id = (
            f"{self.device.device_id}-{description.key}"
        )
    @callback
    def async_update_callback(self) -> None:
        """Update the entity's state."""

        #if self.entity_description.value_fn is None:
        if (state := getattr(self.device, self.entity_description.key)) is None:
            return

        self._attr_available = True
        self._attr_native_value = state

        if (attrs := getattr(self.device, "_qs_attributes")) is not None:
            self._attr_extra_state_attributes = attrs

        self.async_write_ha_state()


@dataclass
class QSExtraStoredData(ExtraStoredData):
    """Object to hold extra stored data."""

    native_value: str | None
    native_attr: dict


    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the text data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, restored: dict[str, Any]):
        """Initialize a stored text state from a dict."""
        try:
            return cls(
                restored["native_value"],
                restored["native_attr"],
            )
        except KeyError:
            return None

class QSBaseSensorRestore(QSBaseSensor, RestoreEntity):

    @property
    def extra_restore_state_data(self) -> QSExtraStoredData:
        """Return sensor specific state data to be restored."""
        return QSExtraStoredData(self._attr_native_value, self._attr_extra_state_attributes)

    async def async_get_last_sensor_data(self) -> QSExtraStoredData | None:
        """Restore native_value and native_unit_of_measurement."""
        if (restored_last_extra_data := await self.async_get_last_extra_data()) is None:
            return None
        return QSExtraStoredData.from_dict(restored_last_extra_data.as_dict())

    async def async_added_to_hass(self) -> None:
        """Restore ATTR_CHANGED_BY on startup since it is likely no longer in the activity log."""
        await super().async_added_to_hass()

        last_sensor_state = await self.async_get_last_sensor_data()
        if (
            not last_sensor_state
        ):
            return

        self._attr_native_value = last_sensor_state.native_value
        self._attr_extra_state_attributes = last_sensor_state.native_attr

    # @property
    # def extra_state_attributes(self) -> dict[str, Any]:
    #    """Return the device specific state attributes."""

