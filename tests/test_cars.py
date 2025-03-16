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




if __name__ == '__main__':
    unittest.main()
