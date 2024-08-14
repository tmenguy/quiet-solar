from dataclasses import dataclass

@dataclass
class LoadCommand:
    command: str
    power_consign: float
    def __eq__(self, other):
        return other is not None and self.command == other.command

def copy_command(cmd:LoadCommand, power_consign=None) -> LoadCommand:
    if power_consign is None:
        return LoadCommand(command=cmd.command, power_consign=cmd.power_consign)
    return LoadCommand(command=cmd.command, power_consign=power_consign)



CMD_ON = LoadCommand(command="on", power_consign=0.0)
CMD_AUTO_GREEN_ONLY = LoadCommand(command="auto_green", power_consign=0.0)
CMD_AUTO_FROM_CONSIGN = LoadCommand(command="auto_consign", power_consign=0.0)
CMD_OFF = LoadCommand(command="off", power_consign=0.0)
CMD_IDLE = LoadCommand(command="idle", power_consign=0.0)