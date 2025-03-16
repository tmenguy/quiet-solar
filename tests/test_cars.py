import unittest
import datetime

from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.home import QSHome


class TestCars(unittest.TestCase):
    def test_dampening(self):
        home = QSHome(hass=None, config_entry=None, name="test home")
        car = QSCar(hass=None, home=home, config_entry=None,
                                          name=f"test car")

        car.update_dampening_value(amperage_transition=(8,10), power_value_or_delta=1000, for_3p=True, time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=8, to_amp=10, for_3p=True)
        assert val == 1000

        val = car.get_delta_dampened_power(from_amp=10, to_amp=8, for_3p=True)
        assert val == -1000

        car.update_dampening_value(amperage_transition=(9, 10), power_value_or_delta=501, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage_transition=(9, 12), power_value_or_delta=10000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=8, to_amp=12, for_3p=True)
        assert val == 10499

        val = car.get_delta_dampened_power(from_amp=8, to_amp=13, for_3p=True)
        assert val is None

        car.update_dampening_value(amperage_transition=(6, 14), power_value_or_delta=10000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage_transition=(13, 14), power_value_or_delta=10000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage_transition=(7, 8), power_value_or_delta=10000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage_transition=(11, 12), power_value_or_delta=10000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage_transition=(12, 13), power_value_or_delta=10000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)


        val = car.get_delta_dampened_power(from_amp=14, to_amp=7, for_3p=True)
        assert val == -40499


        car.update_dampening_value(amperage_transition=7, power_value_or_delta=100000, for_3p=True,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=14, to_amp=0, for_3p=True)
        assert val == -140499



if __name__ == '__main__':
    unittest.main()
