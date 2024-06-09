from dataclasses import dataclass


@dataclass
class LoadCommand:
    command: str
    value: float
    param: str


CMD_ON = LoadCommand("on", 0.0, "on")
CMD_AUTO_GREEN_ONLY = LoadCommand("auto", 0.0, "green_only")
CMD_AUTO_ECO = LoadCommand("auto", 0.0, "eco")
CMD_OFF = LoadCommand("off", 0.0, "off")
