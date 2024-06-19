from abc import abstractmethod
from homeassistant.core import HomeAssistant

class HADeviceMixin:

    def __init__(self, hass:HomeAssistant, **kwargs ):
        super().__init__(**kwargs)
        self.hass = hass


    @abstractmethod
    def get_platforms(self):
        """ returns associated platforms for this device """


