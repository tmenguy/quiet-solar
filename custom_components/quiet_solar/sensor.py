
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import pytz
from homeassistant.components.sensor import SensorEntityDescription, SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric


from .const import (
    DOMAIN, SENSOR_HOME_AVAILABLE_EXTRA_POWER, SENSOR_HOME_CONSUMPTION_POWER,
    SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, HA_CONSTRAINT_SENSOR_HISTORY,
)
from .entity import QSDeviceEntity
from .ha_model.device import HADeviceMixin
from .ha_model.home import QSHome
from .home_model.load import AbstractDevice, AbstractLoad


def create_ha_sensor_for_QSCar(device: QSCar):
    entities = []
    return entities

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

    entities.append(QSBaseSensorRestore(data_handler=device.data_handler, device=device, description=store_sensor))

    return entities

def create_ha_sensor_for_QSCharger(device: QSChargerGeneric):
    entities = []
    return entities


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

    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=charge_sensor))

    entities.extend(create_ha_sensor_for_QSCar(device._default_generic_car))

    return entities

def create_ha_sensor_for_Load(device: AbstractLoad):
    entities = []

    if isinstance(device, HADeviceMixin) and isinstance(device, AbstractLoad):

        load_power_sensor = QSSensorEntityDescription(
            key="best_power_value",
            name=device.get_virtual_load_HA_power_entity_name(),
            native_unit_of_measurement=UnitOfPower.WATT,
            state_class=SensorStateClass.MEASUREMENT,
            device_class=SensorDeviceClass.POWER,
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_power_sensor))


        constraints_sensor = QSSensorEntityDescription(
            key="current_constraint",
            name=device.get_virtual_current_constraint_entity_name()
        )
        entities.append(QSLoadSensorCurrentConstraints(data_handler=device.data_handler, device=device, description=constraints_sensor))

    return entities


def create_ha_sensor_for_QSHome(device: QSHome):
    entities = []


    home_non_controlled_consumption_sensor = QSSensorEntityDescription(
        key="home_non_controlled_consumption",
        name=SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT,
        device_class=SensorDeviceClass.POWER,
    )

    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=home_non_controlled_consumption_sensor))

    home_consumption_sensor = QSSensorEntityDescription(
        key="home_consumption",
        name=SENSOR_HOME_CONSUMPTION_POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT,
        device_class=SensorDeviceClass.POWER,
    )

    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=home_consumption_sensor))

    home_available_power = QSSensorEntityDescription(
        key="home_available_power",
        name=SENSOR_HOME_AVAILABLE_EXTRA_POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT,
        device_class=SensorDeviceClass.POWER,
    )

    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=home_available_power))

    return entities

def create_ha_sensor(device: AbstractDevice):

    ret = []
    if isinstance(device, QSCar):
        ret.extend(create_ha_sensor_for_QSCar(device))
    elif isinstance(device, QSChargerGeneric):
        ret.extend(create_ha_sensor_for_QSCharger(device))
    elif isinstance(device, QSHome):
        ret.extend(create_ha_sensor_for_QSHome(device))

    if isinstance(device, AbstractLoad):
        ret.extend(create_ha_sensor_for_Load(device))

    return ret

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
        data_handler,
        device: AbstractDevice,
        description: QSSensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(data_handler=data_handler, device=device, description=description)
        self.entity_description = description

        self._attr_unique_id = (
            f"{self.device.device_id}-{description.key}"
        )
    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""

        #if self.entity_description.value_fn is None:
        if (state := getattr(self.device, self.entity_description.key)) is None:
            return

        self._attr_available = True
        self._attr_native_value = state
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

        self._attr_native_value = None
        self._attr_extra_state_attributes = {}

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


class QSLoadSensorCurrentConstraints(QSBaseSensorRestore):

    device: AbstractLoad
    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""
        do_save = False
        self._attr_available = True

        current_constraint = self.device.get_current_active_constraint(time)

        if current_constraint is None:
            new_val = "NO TARGET"
        else:
            new_val = current_constraint.get_readable_name_for_load()

        if self._attr_native_value != new_val:
            do_save = True
            self._attr_native_value = new_val

        loads = self.device.get_active_constraints_for_storage(time)
        serialized_loads = [ l.to_dict() for l in loads]

        old = self._attr_extra_state_attributes.get(HA_CONSTRAINT_SENSOR_HISTORY, [])

        if old != serialized_loads:
            do_save = True
            self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_HISTORY] = serialized_loads

        if do_save:
            self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """add back the stored constraints."""
        await super().async_added_to_hass()
        stored_cs = self._attr_extra_state_attributes.get(HA_CONSTRAINT_SENSOR_HISTORY, [])
        await self.hass.async_add_executor_job(
            self.device.load_constraints_from_storage, datetime.now(pytz.UTC), stored_cs
        )
