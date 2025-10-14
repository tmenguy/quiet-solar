import importlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable

import pytz
from homeassistant.components.sensor import SensorEntityDescription, SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfPower, EntityCategory, UnitOfEnergy, PERCENTAGE, UnitOfLength
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity, ExtraStoredData

from .ha_model.car import QSCar
from .ha_model.charger import QSChargerGeneric


from .const import (
    DOMAIN, SENSOR_HOME_AVAILABLE_EXTRA_POWER, SENSOR_HOME_CONSUMPTION_POWER,
    SENSOR_HOME_NON_CONTROLLED_CONSUMPTION_POWER, HA_CONSTRAINT_SENSOR_HISTORY,
    HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT,
    QSForecastHomeNonControlledSensors, QSForecastSolarSensors, SENSOR_LOAD_CURRENT_COMMAND,
    SENSOR_LOAD_BEST_POWER_VALUE, SENSOR_CONSTRAINT_SENSOR_VALUE, SENSOR_CONSTRAINT_SENSOR_ENERGY,
    HA_CONSTRAINT_SENSOR_LOAD_INFO, SENSOR_CONSTRAINT_SENSOR_COMPLETION, SENSOR_LOAD_OVERRIDE_STATE,
    SENSOR_CONSTRAINT_SENSOR_CHARGE, SENSOR_CAR_SOC_PERCENT, HA_CONSTRAINT_SENSOR_FROM_AGENDA_CONSTRAINT,
    SENSOR_CAR_CHARGE_TYPE, SENSOR_CAR_CHARGE_TIME,
    SENSOR_CAR_ESTIMATED_RANGE_KM, SENSOR_CAR_AUTONOMY_TO_TARGET_SOC_KM
)
from .entity import QSDeviceEntity
from .ha_model.device import HADeviceMixin
from .ha_model.home import QSHome
from .home_model.load import AbstractDevice, AbstractLoad


def create_ha_sensor_for_QSCar(device: QSCar):
    entities = []

    load_current_command = QSSensorEntityDescription(
        key="current_constraint_car",
        translation_key=SENSOR_CONSTRAINT_SENSOR_CHARGE,
        value_fn=lambda device, key: "NO CHARGE" if device.charger is None else device.charger.get_active_readable_name(),
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))

    load_current_command = QSSensorEntityDescription(
        key="best_power_value",
        translation_key=SENSOR_LOAD_BEST_POWER_VALUE,
        native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT,
        device_class=SensorDeviceClass.POWER,
        value_fn=lambda device, key: 0.0 if device.charger is None else device.charger.best_power_value,
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))

    load_current_command = QSSensorEntityDescription(
        key="car_soc_percentage",
        translation_key=SENSOR_CAR_SOC_PERCENT,
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.BATTERY,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device, key: device.get_car_charge_percent(),
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))


    load_current_command = QSSensorEntityDescription(
        key="car_charge_type",
        translation_key=SENSOR_CAR_CHARGE_TYPE,
        value_fn=lambda device, key: device.get_car_charge_type(),
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))


    load_current_command = QSSensorEntityDescription(
        key="car_charge_time",
        translation_key=SENSOR_CAR_CHARGE_TIME,
        value_fn=lambda device, key: device.get_car_charge_time_readable_name(),
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))

    # Estimated remaining range now
    load_current_command = QSSensorEntityDescription(
        key="car_estimated_range_km",
        translation_key=SENSOR_CAR_ESTIMATED_RANGE_KM,
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=UnitOfLength.KILOMETERS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device, key: device.get_estimated_range_km(),
        qs_is_none_unavailable=True,
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))

    # Autonomy to selected target SOC
    load_current_command = QSSensorEntityDescription(
        key="car_autonomy_to_target_soc_km",
        translation_key=SENSOR_CAR_AUTONOMY_TO_TARGET_SOC_KM,
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=UnitOfLength.KILOMETERS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda device, key: device.get_autonomy_to_target_soc_km(),
        qs_is_none_unavailable=True,
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))

    return entities


def create_ha_sensor_for_QSCharger(device: QSChargerGeneric):
    entities = []
    return entities


def create_ha_sensor_for_Load(device: AbstractLoad):
    entities = []

    load_current_command = QSSensorEntityDescription(
        key="load_current_command",
        translation_key=SENSOR_LOAD_CURRENT_COMMAND,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda device, key: "NO CMD" if device.current_command is None else device.current_command.command,
    )
    entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_current_command))


    if device.support_user_override():
        # add hare a sensor to know if the load has been ovreriden externally
        load_override_state = QSSensorEntityDescription(
            key="load_override_state",
            translation_key=SENSOR_LOAD_OVERRIDE_STATE,
            entity_category=EntityCategory.DIAGNOSTIC,
            value_fn=lambda device, key: device.get_override_state(),
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_override_state))


    if isinstance(device, HADeviceMixin) and isinstance(device, AbstractLoad):

        load_power_sensor = QSSensorEntityDescription(
            key="best_power_value",
            translation_key=SENSOR_LOAD_BEST_POWER_VALUE,
            native_unit_of_measurement=UnitOfPower.WATT,
            state_class=SensorStateClass.MEASUREMENT,
            device_class=SensorDeviceClass.POWER,
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=load_power_sensor))


        constraints_sensor = QSSensorEntityDescription(
            key="current_constraint",
            translation_key=device.get_virtual_current_constraint_translation_key()
        )
        entities.append(QSLoadSensorCurrentConstraints(data_handler=device.data_handler, device=device, description=constraints_sensor))

        constraints_sensor = QSSensorEntityDescription(
            key="current_constraint_current_value",
            translation_key=SENSOR_CONSTRAINT_SENSOR_VALUE,
            state_class=SensorStateClass.MEASUREMENT,
            entity_category=EntityCategory.DIAGNOSTIC,
            qs_is_none_unavailable=True
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=constraints_sensor))

        constraints_sensor = QSSensorEntityDescription(
            key="current_constraint_current_energy",
            translation_key=SENSOR_CONSTRAINT_SENSOR_ENERGY,
            device_class=SensorDeviceClass.ENERGY,
            native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
            entity_category=EntityCategory.DIAGNOSTIC,
            qs_is_none_unavailable=True
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=constraints_sensor))

        constraints_sensor = QSSensorEntityDescription(
            key="current_constraint_current_percent_completion",
            translation_key=SENSOR_CONSTRAINT_SENSOR_COMPLETION,
            state_class=SensorStateClass.MEASUREMENT,
            native_unit_of_measurement=PERCENTAGE,
            entity_category=EntityCategory.DIAGNOSTIC,
            qs_is_none_unavailable=True
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=constraints_sensor))


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

    for name in QSForecastHomeNonControlledSensors:
        home_forecast_power = QSSensorEntityDescription(
            key=name,
            name=name,
            native_unit_of_measurement=UnitOfPower.WATT,
            state_class=SensorStateClass.MEASUREMENT,
            device_class=SensorDeviceClass.POWER,
            entity_category=EntityCategory.DIAGNOSTIC,
            value_fn = lambda device, key: device.home_non_controlled_power_forecast_sensor_values.get(key, None),
            qs_is_none_unavailable=True
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=home_forecast_power))

    for name in QSForecastSolarSensors:
        home_forecast_power = QSSensorEntityDescription(
            key=name,
            name=name,
            native_unit_of_measurement=UnitOfPower.WATT,
            state_class=SensorStateClass.MEASUREMENT,
            device_class=SensorDeviceClass.POWER,
            entity_category=EntityCategory.DIAGNOSTIC,
            value_fn = lambda device, key: device.home_solar_forecast_sensor_values.get(key, None),
            qs_is_none_unavailable=True
        )
        entities.append(QSBaseSensor(data_handler=device.data_handler, device=device, description=home_forecast_power))




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
    """Set up the Quiet Solar sensor platform."""
    device = hass.data[DOMAIN].get(entry.entry_id)

    if device:

        entities = create_ha_sensor(device)

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
            pass


    return True


@dataclass(frozen=True, kw_only=True)
class QSSensorEntityDescription(SensorEntityDescription):
    """Describes Quiet Solar sensor entity."""
    qs_is_none_unavailable: bool  = False
    value_fn: Callable[[AbstractDevice, str], Any] | None = None


class QSBaseSensor(QSDeviceEntity, SensorEntity):
    """Implementation of a Quiet Solar sensor."""

    entity_description: QSSensorEntityDescription

    def __init__(
        self,
        data_handler,
        device: AbstractDevice,
        description: QSSensorEntityDescription,
    ) -> None:
        super().__init__(data_handler=data_handler, device=device, description=description)

    @callback
    def async_update_callback(self, time:datetime) -> None:
        """Update the entity's state."""

        if self.entity_description.value_fn is None:
            state = getattr(self.device, self.entity_description.key)
        else:
            state = self.entity_description.value_fn(self.device, self.entity_description.key)

        self._set_availabiltiy()

        if state is None:
            if self.entity_description.qs_is_none_unavailable:
                self._attr_available = False
                self._attr_native_value = STATE_UNAVAILABLE
                self.async_write_ha_state()
                return
            else:
                self.async_write_ha_state()
                return

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

        self._set_availabiltiy()

        new_val = self.device.get_active_readable_name(time)

        if self._attr_native_value != new_val:
            self._attr_native_value = new_val

        constraints = self.device.get_active_constraints(time)
        serialized_constraints = [ l.to_dict() for l in constraints]

        self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_HISTORY] = serialized_constraints

        if self.device._last_completed_constraint is None:
            self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT] = None
        else:
            serialized_constraint = self.device._last_completed_constraint.to_dict()
            self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT] = serialized_constraint

        if self.device._last_pushed_end_constraint_from_agenda is None:
            self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_FROM_AGENDA_CONSTRAINT] = None
        else:
            serialized_constraint = self.device._last_pushed_end_constraint_from_agenda.to_dict()
            self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_FROM_AGENDA_CONSTRAINT] = serialized_constraint

        self._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_LOAD_INFO] = self.device.get_to_be_saved_info()

        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """add back the stored constraints."""
        await super().async_added_to_hass()
        stored_cs = self._attr_extra_state_attributes.get(HA_CONSTRAINT_SENSOR_HISTORY, [])
        stored_executed = self._attr_extra_state_attributes.get(HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT, None)
        stored_load_info = self._attr_extra_state_attributes.get(HA_CONSTRAINT_SENSOR_LOAD_INFO, None)
        stored_from_agenda = self._attr_extra_state_attributes.get(HA_CONSTRAINT_SENSOR_FROM_AGENDA_CONSTRAINT, None)
        await self.hass.async_add_executor_job(
            self.device.load_constraints_from_storage, datetime.now(pytz.UTC), stored_cs, stored_executed, stored_load_info, stored_from_agenda
        )
