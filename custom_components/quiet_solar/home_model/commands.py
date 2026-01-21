from dataclasses import dataclass, asdict


CMD_CST_ON = "on"
CMD_CST_OFF = "off"
CMD_CST_IDLE = "idle"
CMD_CST_AUTO_GREEN = "auto_green"
CMD_CST_AUTO_GREEN_CAP = "auto_green_cap"
CMD_CST_GREEN_CHARGE_ONLY = "green_charge-ONLY"
CMD_CST_AUTO_GREEN_CONSIGN = "auto_green_consign"
CMD_CST_AUTO_CONSIGN = "auto_consign"
CMD_CST_AUTO_PRICE = "auto_price"
CMD_CST_FORCE_CHARGE = "force_charge"

commands_scores = {
    CMD_CST_ON: 100,
    CMD_CST_FORCE_CHARGE: 90,
    CMD_CST_AUTO_CONSIGN: 80,
    CMD_CST_AUTO_PRICE: 70,
    CMD_CST_AUTO_GREEN_CONSIGN: 60,
    CMD_CST_GREEN_CHARGE_ONLY: 50,
    CMD_CST_AUTO_GREEN: 40,
    CMD_CST_AUTO_GREEN_CAP: 30, # this one is less "free style" and is more constrained than the AUTO GREEN
    CMD_CST_IDLE: 20,
    CMD_CST_OFF: 10,
}

@dataclass
class LoadCommand:
    command: str
    power_consign: float

    def __eq__(self, other):
        return other is not None and self.command == other.command and self.power_consign == other.power_consign

    def to_dict(self) -> dict:
        return asdict(self) #self.__dict_

    def is_auto(self) -> bool:
        return self.command in [CMD_CST_AUTO_GREEN, CMD_CST_AUTO_CONSIGN, CMD_CST_AUTO_PRICE, CMD_CST_AUTO_GREEN_CAP, CMD_CST_AUTO_GREEN_CONSIGN]

    def is_like(self, other) -> bool:
        if other is None:
            return False
        return self.command == other.command

    def is_like_one_of_cmds(self, cmds) -> bool:
        return any([self.is_like(cmd) for cmd in cmds])

    def is_off_or_idle(self) -> bool:
        return self.command == "off" or self.command == "idle"


def merge_commands(cmd1:LoadCommand, cmd2:LoadCommand) -> LoadCommand:
    """Merge two LoadCommand objects"""
    if cmd1 is None:
        return cmd2
    if cmd2 is None:
        return cmd1

    if commands_scores.get(cmd1.command, 0) >= commands_scores.get(cmd2.command, 0):
        command = cmd1.command
    else:
        command = cmd2.command

    power_consign = max(cmd1.power_consign, cmd2.power_consign)

    return LoadCommand(command=command, power_consign=power_consign)




def copy_command(cmd:LoadCommand|None, power_consign=None) -> LoadCommand |None:
    if cmd is None:
        return None
    if power_consign is None:
        power_consign=cmd.power_consign
    return LoadCommand(command=cmd.command, power_consign=power_consign)

def copy_command_and_change_type(cmd:LoadCommand, new_type:str) -> LoadCommand:
    return LoadCommand(command=new_type, power_consign=cmd.power_consign)

CMD_ON = LoadCommand(command=CMD_CST_ON, power_consign=0.0)
CMD_AUTO_GREEN_ONLY = LoadCommand(command=CMD_CST_AUTO_GREEN, power_consign=0.0)
CMD_AUTO_GREEN_CAP = LoadCommand(command=CMD_CST_AUTO_GREEN_CAP, power_consign=0.0)
CMD_AUTO_GREEN_CONSIGN = LoadCommand(command=CMD_CST_AUTO_GREEN_CONSIGN, power_consign=0.0)
CMD_AUTO_FROM_CONSIGN = LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=0.0)
CMD_AUTO_PRICE = LoadCommand(command=CMD_CST_AUTO_PRICE, power_consign=0.0)
CMD_OFF = LoadCommand(command=CMD_CST_OFF, power_consign=0.0)
CMD_IDLE = LoadCommand(command=CMD_CST_IDLE, power_consign=0.0)

CMD_GREEN_CHARGE_AND_DISCHARGE = LoadCommand(command=CMD_CST_AUTO_GREEN, power_consign=0.0)
CMD_GREEN_CHARGE_ONLY = LoadCommand(command=CMD_CST_GREEN_CHARGE_ONLY, power_consign=0.0)
CMD_FORCE_CHARGE = LoadCommand(command=CMD_CST_FORCE_CHARGE, power_consign=0.0) #can be on grid