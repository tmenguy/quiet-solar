from typing import Any

from homeassistant.core import callback, Event, EventStateChangedData
from homeassistant.helpers.event import async_track_state_change_event

from quiet_solar.const import CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, \
    CONF_BATTERY_MAX_CHARGE_POWER_NUMBER
from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.home_model.battery import Battery
from homeassistant.const import Platform, STATE_UNKNOWN, STATE_UNAVAILABLE, ATTR_ENTITY_ID
from homeassistant.components import number, homeassistant

class QSBattery(HADeviceMixin, Battery):


    def __init__(self, **kwargs) -> None:
        self.charge_discharge_sensor = kwargs.pop(CONF_BATTERY_CHARGE_DISCHARGE_SENSOR, None)
        self.max_discharge_number = kwargs.pop(CONF_BATTERY_MAX_DISCHARGE_POWER_NUMBER, None)
        self.max_charge_number = kwargs.pop(CONF_BATTERY_MAX_CHARGE_POWER_NUMBER, None)

        super().__init__(**kwargs)

        self.attach_ha_state_to_probe(self.charge_discharge_sensor,
                                      is_numerical=True)

        self.max_discharging_power_current = self.max_discharging_power
        self.max_charging_power_current = self.max_charging_power

        @callback
        def async_threshold_sensor_state_listener(
                event: Event[EventStateChangedData],
        ) -> None:
            """Handle sensor state changes."""
            new_state = event.data["new_state"]
            if new_state is None or new_state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                return

            value = float(new_state.state)

            time = new_state.last_updated
            self.add_to_history(new_state.entity_id, time, value)


        if self.charge_discharge_sensor is not None:

                self._unsub = async_track_state_change_event(
                    self.hass,
                    list(self.charge_discharge_sensor,),
                    async_threshold_sensor_state_listener,
            )

    async def set_max_discharging_power(self, power: float | None, blocking: bool = False):

        if power is None:
            power = self.max_discharging_power

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_discharge_number}
        range_value = float(power)
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_discharging_power)
        max_value = float(self.max_discharging_power)

        val = int(min(max_value, max(min_value, range_value)))

        if val == self.max_discharging_power_current:
            return

        data[number.ATTR_VALUE] = val
        domain = number.DOMAIN


        await self.hass.services.async_call(
            domain, service, data, blocking=blocking
        )

        self.max_discharging_power_current = power

    async def set_max_charging_power(self, power: float | None, blocking: bool = False):

        if power is None:
            power = self.max_charging_power

        data: dict[str, Any] = {ATTR_ENTITY_ID: self.max_charge_number}
        range_value = float(power)
        service = number.SERVICE_SET_VALUE
        min_value = float(self.min_charging_power)
        max_value = float(self.max_charging_power)

        val = int(min(max_value, max(min_value, range_value)))

        if val == self.max_charging_power_current:
            return

        data[number.ATTR_VALUE] = val
        domain = number.DOMAIN

        await self.hass.services.async_call(
            domain, service, data, blocking=blocking
        )

        self.max_charging_power_current = power





    def get_platforms(self):
        return [ Platform.SENSOR]
