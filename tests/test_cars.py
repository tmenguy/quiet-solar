import unittest
import datetime

from quiet_solar.ha_model.car import QSCar
from quiet_solar.ha_model.home import QSHome


class TestCars(unittest.TestCase):
    def test_dampening(self):
        home = QSHome(hass=None, config_entry=None, name="test home")
        car = QSCar(hass=None, home=home, config_entry=None,
                                          name=f"test car")

        car.update_dampening_value(amperage=None, amperage_transition=((8, 3),(10, 3)), power_value_or_delta=1000, time=datetime.datetime.now(), can_be_saved=False)

        val, _, _ = car.get_delta_dampened_power(from_amp=8, from_num_phase=3, to_amp=10, to_num_phase=3)
        assert val == 1000

        val, _, _ = car.get_delta_dampened_power(from_amp=10, from_num_phase=3, to_amp=8, to_num_phase=3)
        assert val == -1000

        car.update_dampening_value(amperage=None,amperage_transition=((9,3), (10,3)), power_value_or_delta=501, time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None,amperage_transition=((9,3), (12,3)), power_value_or_delta=10000, time=datetime.datetime.now(), can_be_saved=False)

        val, _, _ = car.get_delta_dampened_power(from_amp=8, to_amp=12, from_num_phase=3, to_num_phase=3)
        assert val == 10499

        val, _, _ = car.get_delta_dampened_power(from_amp=8, to_amp=13, from_num_phase=3, to_num_phase=3)
        assert val is None

        car.update_dampening_value(amperage=None, amperage_transition=((6,3), (14,3)), power_value_or_delta=10000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((13,3), (14,3)), power_value_or_delta=10000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((7,3), (8,3)), power_value_or_delta=10000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((11,3), (12,3)), power_value_or_delta=10000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((12,3), (13,3)), power_value_or_delta=10000,
                                   time=datetime.datetime.now(), can_be_saved=False)


        val, _, _ = car.get_delta_dampened_power(from_amp=14, to_amp=7, from_num_phase=3, to_num_phase=3)
        assert val == -40499


        car.update_dampening_value(amperage=(7,3), amperage_transition=None, power_value_or_delta=100000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val, _, _ = car.get_delta_dampened_power(from_amp=14, to_amp=0, from_num_phase=3, to_num_phase=3)
        assert val == -140499

    def test_dampening_phase_changes(self):
        home = QSHome(hass=None, config_entry=None, name="test home")
        car = QSCar(hass=None, home=home, config_entry=None,
                                          name=f"test car")

        car.update_dampening_value(amperage=None, amperage_transition=((8, 1),(10, 3)), power_value_or_delta=1000, time=datetime.datetime.now(), can_be_saved=False)

        val, _, _ = car.get_delta_dampened_power(from_amp=8, from_num_phase=1, to_amp=10, to_num_phase=3)
        assert val == 1000

        val, _, _ = car.get_delta_dampened_power(from_amp=10, from_num_phase=3, to_amp=8, to_num_phase=1)
        assert val == -1000

        car.update_dampening_value(amperage=None,amperage_transition=((16,1), (10,3)), power_value_or_delta=501, time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None,amperage_transition=((16,1), (12,3)), power_value_or_delta=10000, time=datetime.datetime.now(), can_be_saved=False)

        val, _, _ = car.get_delta_dampened_power(from_amp=8, to_amp=12, from_num_phase=1, to_num_phase=3)
        assert val == 10499

        val, _, _ = car.get_delta_dampened_power(from_amp=8, to_amp=13, from_num_phase=1, to_num_phase=3)
        assert val is None

        car.update_dampening_value(amperage=None, amperage_transition=((18,1), (14,3)), power_value_or_delta=20000,
                                   time=datetime.datetime.now(), can_be_saved=False)


        car.update_dampening_value(amperage=None, amperage_transition=((21,1), (18,1)), power_value_or_delta=-10000,
                                   time=datetime.datetime.now(), can_be_saved=False)



        val, _, _ = car.get_delta_dampened_power(from_amp=14, to_amp=7, from_num_phase=3, to_num_phase=3)
        assert val == -10000


        car.update_dampening_value(amperage=(7,3), amperage_transition=None, power_value_or_delta=100000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val, _, _ = car.get_delta_dampened_power(from_amp=14, to_amp=0, from_num_phase=3, to_num_phase=3)
        assert val == -110000



if __name__ == '__main__':
    unittest.main()
