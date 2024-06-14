from home_model.commands import LoadCommand

from home_model.load import AbstractLoad


class SwitchLoad(AbstractLoad):

    def __init__(self, power:float, **kwargs):
        super().__init__(**kwargs)
        self._power = power