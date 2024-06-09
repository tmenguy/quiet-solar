from home_model.battery import Battery
from home_model.load import AbstractLoad
from datetime import datetime, timedelta

from home_model.solver import PeriodSolver


class Home:

    _battery: Battery = None
    _loads : list[AbstractLoad] = []
    _period : timedelta = timedelta(days=1)
    def __init__(self) -> None:
        pass

    def add_load(self, load: AbstractLoad):
        self._loads.append(load)

    def set_battery(self, battery: Battery):
        self._battery = battery

    async def update(self, time: datetime):

        do_force_solve = False
        for load in self._loads:
            if (await load.update_live_constraints(time, self._period)) :
                do_force_solve = True


        if do_force_solve:
            solver = PeriodSolver(
                start_time = time,
                end_time = time + self._period,
                tariffs: list[tuple[float, datetime]] = None,
                actionable_loads: list[AbstractLoad] = None,
                battery: Battery | None = None,
                pv_forecast: list[float, datetime] = None,
                unavoidable_consumption_forecast: list[float, datetime] = None,
            )

            solver.solve()

        faire le tour de tou les loads .... et s pas de command ni de contrainte : mettre la contrainte par default
        des loads








