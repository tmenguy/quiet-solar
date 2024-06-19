from quiet_solar.ha_model.battery import QSBattery
from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.charger import QSCharger
from quiet_solar.ha_model.device import HADeviceMixin
from quiet_solar.ha_model.solar import QSSolar
from quiet_solar.home_model.battery import Battery
from quiet_solar.home_model.commands import LoadCommand
from quiet_solar.home_model.load import AbstractLoad, AbstractDevice
from datetime import datetime, timedelta
from quiet_solar.home_model.solver import PeriodSolver
from homeassistant.const import Platform

class QSHome(HADeviceMixin, AbstractDevice):

    _battery: QSBattery = None

    _chargers : list[QSCharger] = []
    _cars: list[QSCar] = []


    _devices : list[AbstractDevice] = []
    _solar_plant: QSSolar | None = None
    _all_loads : list[AbstractLoad] = []

    _active_loads: list[AbstractLoad] = []

    _period : timedelta = timedelta(days=1)
    _commands : list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]] = []
    _solver_step_s : timedelta = timedelta(seconds=900)
    _update_step_s : timedelta = timedelta(seconds=5)
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def add_device(self, device: AbstractDevice):
        if isinstance(device, QSBattery):
            self._battery = device
        elif isinstance(device, QSCar):
            self._cars.append(device)
        elif isinstance(device, QSCharger):
            self._chargers.append(device)
        elif isinstance(device, QSSolar):
            self._solar_plant = device

        if isinstance(device, AbstractLoad):
            self._all_loads.append(device)

    async def update_load_status(self, time: datetime):
        pass
        # c'ets la qu'on va check que les etats d'une voiture il y a bien branche et tout, qu'on va mettre si besoin des contraintes par default
        # une load est active ici si elle a une contrainte , mais imagine ca vien de l('exterieur comme unde demande de run de la voiture '
        #                                                                              'sans que ce soit quiet_solar qui l'a lance
        # ca il faut pouvoir check si la load est en fait controllee par l('exterieur ou pas si oui : il ne faut surtout pas en prendre le control'
        #                                                                  ''
        #                                                         et donc la sortir des loads a calculer et de toutes les commandes quie je pourrais lui lancer
        #
        # pour check ca le check command peut etre ok, mais surtout comment savoir que la commande vient de quiet soar ou pas ? ... il faut donc se souvenir
        # si ion a lance une commande ou pas (a running et current commande peuvent aider ... mais il faut persister ca aussi en cas de reboot!!!!
        # sinon on ne reprendra jamais la main ..... )



    async def update(self, time: datetime):

        for load in self._active_loads:
            await load.check_commands()

        do_force_solve = False
        for load in self._active_loads:
            if (await load.update_live_constraints(time, self._period)) :
                do_force_solve = True



        if do_force_solve:
            solver = PeriodSolver(
                start_time = time,
                end_time = time + self._period,
                tariffs = None,
                actionable_loads = None,
                battery = self._battery,
                pv_forecast  = None,
                unavoidable_consumption_forecast = None,
                step_s = self._solver_step_s
            )

            self._commands = solver.solve()

        for load, commands in self._commands:
            while len(commands) > 0 and commands[0][0] < time + self._update_step_s:
                cmd_time, command = commands.pop(0)
                await load.launch_command(command)











