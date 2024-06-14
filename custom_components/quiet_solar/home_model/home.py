from home_model.battery import Battery
from home_model.commands import LoadCommand
from home_model.load import AbstractLoad
from datetime import datetime, timedelta
from home_model.solver import PeriodSolver


class Home:

    _battery: Battery = None
    _active_loads : list[AbstractLoad] = []
    _all_loads : list[AbstractLoad] = []
    _period : timedelta = timedelta(days=1)
    _commands : list[tuple[AbstractLoad, list[tuple[datetime, LoadCommand]]]] = []
    _solver_step_s : timedelta = timedelta(seconds=900)
    _update_step_s : timedelta = timedelta(seconds=5)
    def __init__(self) -> None:
        pass

    def add_load(self, load: AbstractLoad):
        self._all_loads.append(load)

    def set_battery(self, battery: Battery):
        self._battery = battery

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











