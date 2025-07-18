import unittest
import datetime

from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.home import QSHome

from custom_components.quiet_solar.const import CONF_CAR_CHARGER_MIN_CHARGE, CONF_CAR_CUSTOM_POWER_CHARGE_VALUES, \
    CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P


class TestCars(unittest.TestCase):

    def test_cars_custom_data(self):
        home = QSHome(hass=None, config_entry=None, name="test home")
        car_conf = {
            "hass": None,
            "home": home,
            "config_entry": None,
            "name": "test car",
            CONF_CAR_CHARGER_MIN_CHARGE:7,
            CONF_CAR_CUSTOM_POWER_CHARGE_VALUES: True,
            CONF_CAR_IS_CUSTOM_POWER_CHARGE_VALUES_3P: True,
            "charge_7": 700,
            "charge_8": 800,
            "charge_9": 900,
            "charge_10": 1000,
        }
        car = QSCar(**car_conf)

        assert car.car_charger_min_charge == 7

        car.can_dampen_strongly_dynamically = True

        val = car.get_delta_dampened_power(from_amp=7, to_amp=8, from_num_phase=3, to_num_phase=3)

        assert val == 100

        car.update_dampening_value(amperage=(7,3), amperage_transition=None, power_value_or_delta=600,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=7, to_amp=8, from_num_phase=3, to_num_phase=3)

        assert val == 200

        car.update_dampening_value(amperage=(7,3), amperage_transition=None, power_value_or_delta=20,
                                   time=datetime.datetime.now(), can_be_saved=False)
        assert car.car_charger_min_charge == 8


    def test_dampening(self):
        home = QSHome(hass=None, config_entry=None, name="test home")
        car = QSCar(hass=None, home=home, config_entry=None,
                                          name=f"test car")
        car.can_dampen_strongly_dynamically = True

        car.update_dampening_value(amperage=None, amperage_transition=((8, 3),(10, 3)), power_value_or_delta=1000, time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=8, from_num_phase=3, to_amp=10, to_num_phase=3)
        assert val == 1000

        val = car.get_delta_dampened_power(from_amp=10, from_num_phase=3, to_amp=8, to_num_phase=3)
        assert val == -1000

        car.update_dampening_value(amperage=None,amperage_transition=((9,3), (10,3)), power_value_or_delta=501, time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None,amperage_transition=((9,3), (12,3)), power_value_or_delta=2000, time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=8, to_amp=12, from_num_phase=3, to_num_phase=3)
        assert val == 2499

        val = car.get_delta_dampened_power(from_amp=8, to_amp=13, from_num_phase=3, to_num_phase=3)
        assert val == 3450

        car.update_dampening_value(amperage=None, amperage_transition=((6,3), (14,3)), power_value_or_delta=2000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((13,3), (14,3)), power_value_or_delta=500,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((7,3), (8,3)), power_value_or_delta=500,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((11,3), (12,3)), power_value_or_delta=500,
                                   time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None, amperage_transition=((12,3), (13,3)), power_value_or_delta=500,
                                   time=datetime.datetime.now(), can_be_saved=False)


        val = car.get_delta_dampened_power(from_amp=14, to_amp=7, from_num_phase=3, to_num_phase=3)
        assert val == -3999


        car.update_dampening_value(amperage=(7,3), amperage_transition=None, power_value_or_delta=1211,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=14, to_amp=0, from_num_phase=3, to_num_phase=3)
        assert val == -5210

    def test_dampening_phase_changes(self):
        home = QSHome(hass=None, config_entry=None, name="test home")
        car = QSCar(hass=None, home=home, config_entry=None,
                                          name=f"test car")
        car.can_dampen_strongly_dynamically = True

        car.update_dampening_value(amperage=None, amperage_transition=((8, 1),(10, 3)), power_value_or_delta=4000, time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=8, from_num_phase=1, to_amp=10, to_num_phase=3)
        assert val == 4000

        val = car.get_delta_dampened_power(from_amp=10, from_num_phase=3, to_amp=8, to_num_phase=1)
        assert val == -4000

        car.update_dampening_value(amperage=None,amperage_transition=((16,1), (10,3)), power_value_or_delta=2001, time=datetime.datetime.now(), can_be_saved=False)

        car.update_dampening_value(amperage=None,amperage_transition=((16,1), (12,3)), power_value_or_delta=3000, time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=8, to_amp=12, from_num_phase=1, to_num_phase=3)
        assert val == 4999

        val = car.get_delta_dampened_power(from_amp=8, to_amp=13, from_num_phase=1, to_num_phase=3)
        assert val == 7130.0

        car.update_dampening_value(amperage=None, amperage_transition=((18,1), (14,3)), power_value_or_delta=5000,
                                   time=datetime.datetime.now(), can_be_saved=False)


        car.update_dampening_value(amperage=None, amperage_transition=((21,1), (18,1)), power_value_or_delta=-750,
                                   time=datetime.datetime.now(), can_be_saved=False)



        val = car.get_delta_dampened_power(from_amp=14, to_amp=7, from_num_phase=3, to_num_phase=3)
        assert val == -4250


        car.update_dampening_value(amperage=(7,3), amperage_transition=None, power_value_or_delta=4000,
                                   time=datetime.datetime.now(), can_be_saved=False)

        val = car.get_delta_dampened_power(from_amp=14, to_amp=0, from_num_phase=3, to_num_phase=3)
        assert val == -8250



if __name__ == '__main__':
    unittest.main()
