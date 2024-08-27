from dataclasses import dataclass, asdict



CMD_CST_AUTO_GREEN = "auto_green"
CMD_CST_AUTO_CONSIGN = "auto_consign"
@dataclass
class LoadCommand:
    command: str
    power_consign: float
    def __eq__(self, other):
        return other is not None and self.command == other.command and self.power_consign == other.power_consign

    def to_dict(self) -> dict:
        return asdict(self) #self.__dict_

    def is_auto(self) -> bool:
        return self.command == CMD_CST_AUTO_GREEN or self.command == CMD_CST_AUTO_CONSIGN

    def is_like(self, other) -> bool:
        if other is None:
            return False
        return self.command == other.command


def copy_command(cmd:LoadCommand, power_consign=None) -> LoadCommand:
    if power_consign is None:
        return LoadCommand(command=cmd.command, power_consign=cmd.power_consign)
    return LoadCommand(command=cmd.command, power_consign=power_consign)



CMD_ON = LoadCommand(command="on", power_consign=0.0)
CMD_AUTO_GREEN_ONLY = LoadCommand(command=CMD_CST_AUTO_GREEN, power_consign=0.0)
CMD_AUTO_FROM_CONSIGN = LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=0.0)
CMD_OFF = LoadCommand(command="off", power_consign=0.0)
CMD_IDLE = LoadCommand(command="idle", power_consign=0.0)