from homeassistant.core import HomeAssistant

from home_model.load import AbstractLoad
from abc import ABC


class HALoad(ABC):

    hass : HomeAssistant
    def __init__(self, hass, name:str, **kwargs):
        super().__init__(name,  **kwargs)
        self.hass = hass



