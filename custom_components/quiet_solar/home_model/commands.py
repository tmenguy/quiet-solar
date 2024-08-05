from dataclasses import dataclass
from typing import Any

@dataclass
class LoadCommand:
    command: str
    power_consign: float
    param: Any
    def __eq__(self, other):
        return other is not None and self.command == other.command and self.param == other.param

def copy_command(cmd:LoadCommand, power_consign=None) -> LoadCommand:
    if power_consign is None:
        return LoadCommand(command=cmd.command, param=cmd.param, power_consign=cmd.power_consign)
    return LoadCommand(command=cmd.command, param=cmd.param, power_consign=power_consign)



CMD_ON = LoadCommand(command="on", power_consign=0.0, param="on")
CMD_AUTO_GREEN_ONLY = LoadCommand(command="auto", power_consign=0.0, param="green_only")
CMD_AUTO_ECO = LoadCommand(command="auto", power_consign=0.0, param="eco")
CMD_OFF = LoadCommand(command="off", power_consign=0.0, param=0)
