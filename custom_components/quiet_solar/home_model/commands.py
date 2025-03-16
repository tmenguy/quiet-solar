from dataclasses import dataclass, asdict



CMD_CST_AUTO_GREEN = "auto_green"
CMD__CST_GREEN_CHARGE_ONLY = "green_charge-ONLY"
CMD_CST_AUTO_CONSIGN = "auto_consign"
CMD_CST_AUTO_PRICE = "auto_price"

CMD_CST_FORCE_CHARGE = "force_charge"
@dataclass
class LoadCommand:
    command: str
    power_consign: float
    phase_current: float | None = None

    def __eq__(self, other):
        return other is not None and self.command == other.command and self.power_consign == other.power_consign

    def to_dict(self) -> dict:
        return asdict(self) #self.__dict_

    def is_auto(self) -> bool:
        return self.command in [CMD_CST_AUTO_GREEN, CMD_CST_AUTO_CONSIGN, CMD_CST_AUTO_PRICE]

    def is_like(self, other) -> bool:
        if other is None:
            return False
        return self.command == other.command

    def is_like_one_of_cmds(self, cmds) -> bool:
        return any([self.is_like(cmd) for cmd in cmds])

    def is_off_or_idle(self) -> bool:
        return self.command == "off" or self.command == "idle"


def copy_command(cmd:LoadCommand, power_consign=None, phase_current=None) -> LoadCommand:
    if power_consign is None:
        power_consign=cmd.power_consign
    if phase_current is None:
        phase_current=cmd.phase_current
    return LoadCommand(command=cmd.command, power_consign=power_consign, phase_current=phase_current)

def copy_command_and_change_type(cmd:LoadCommand, new_type:str) -> LoadCommand:
    return LoadCommand(command=new_type, power_consign=cmd.power_consign, phase_current=cmd.phase_current)

CMD_ON = LoadCommand(command="on", power_consign=0.0)
CMD_AUTO_GREEN_ONLY = LoadCommand(command=CMD_CST_AUTO_GREEN, power_consign=0.0)
CMD_AUTO_FROM_CONSIGN = LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=0.0)
CMD_AUTO_PRICE = LoadCommand(command=CMD_CST_AUTO_PRICE, power_consign=0.0)
CMD_OFF = LoadCommand(command="off", power_consign=0.0)
CMD_IDLE = LoadCommand(command="idle", power_consign=0.0)

CMD_GREEN_CHARGE_AND_DISCHARGE = LoadCommand(command=CMD_CST_AUTO_GREEN, power_consign=0.0)
CMD_GREEN_CHARGE_ONLY = LoadCommand(command=CMD__CST_GREEN_CHARGE_ONLY, power_consign=0.0)
CMD_FORCE_CHARGE = LoadCommand(command=CMD_CST_FORCE_CHARGE, power_consign=0.0) #can be on grid