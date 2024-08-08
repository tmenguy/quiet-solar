from ..const import CONF_INVERTER_ACTIVE_POWER_SENSOR, CONF_INVERTER_INPUT_POWER_SENSOR
from ..ha_model.device import HADeviceMixin
from ..home_model.load import AbstractDevice





class QSSolar(HADeviceMixin, AbstractDevice):

    def __init__(self, **kwargs) -> None:
        self.solar_inverter_active_power = kwargs.pop(CONF_INVERTER_ACTIVE_POWER_SENSOR, None)
        self.solar_inverter_input_active_power = kwargs.pop(CONF_INVERTER_INPUT_POWER_SENSOR, None)

        super().__init__(**kwargs)

        self.attach_power_to_probe(self.solar_inverter_active_power)
        self.attach_power_to_probe(self.solar_inverter_input_active_power)


