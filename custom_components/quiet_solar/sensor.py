
from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING

from homeassistant.components.sensor import SensorEntityDescription, SensorEntity, RestoreSensor, SensorExtraStoredData, \
    SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, Platform, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric

if TYPE_CHECKING:
    from . import QSDataHandler

from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    DATA_HANDLER, DEVICE_TYPE
)
from .entity import QSDeviceEntity
from .home_model.load import AbstractDevice


def create_ha_sensor_for_QSCar(device: QSCar):
    entities = []

    store_sensor = QSSensorEntityDescription(
        key="charge_state",
        device_class=SensorDeviceClass.ENUM,
        options=[
            "not_in_charge",
            "waiting_for_a_planned_charge",
            "charge_ended",
            "waiting_for_current_charge",
            "energy_flap_opened",
            "charge_in_progress",
            "charge_error"
        ],
    )

    entities.append(QSBaseSensorRestore(data_handler=device.data_handler, qs_device=device, description=store_sensor))

    return entities

def create_ha_sensor_for_QSCharger(device: QSChargerGeneric):
    entities = []


    charge_sensor = QSSensorEntityDescription(
        key="charge_state",
        device_class=SensorDeviceClass.ENUM,
        options=[
            "not_in_charge",
            "waiting_for_a_planned_charge",
            "charge_ended",
            "waiting_for_current_charge",
            "energy_flap_opened",
            "charge_in_progress",
            "charge_error",
            STATE_UNAVAILABLE,
            STATE_UNKNOWN,
        ],
    )

    entities.append(QSBaseSensor(data_handler=device.data_handler, qs_device=device, description=charge_sensor))

    entities.extend(create_ha_sensor_for_QSCar(device._default_generic_car))

    return entities


def create_ha_sensor(self, device: AbstractDevice):
    device.home = self

    if isinstance(device, QSCar):
        return create_ha_sensor_for_QSCar(device)
    elif isinstance(device, QSChargerGeneric):
        return create_ha_sensor_for_QSCharger(device)

    return []





async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Netatmo switch platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_sensor(device)

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

