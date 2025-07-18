import unittest
import datetime

from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.home import QSHome


class TestCarDampeningValues(unittest.TestCase):
    def setUp(self):
        self.home = QSHome(hass=None, config_entry=None, name="test home")
        self.car = QSCar(hass=None, home=self.home, config_entry=None, name="test car")
        self.car.can_dampen_strongly_dynamically = True
        self.now = datetime.datetime.now()

    def test_update_dampening_value_validation(self):
        # create initial dampening delta
        updated = self.car.update_dampening_value(
            amperage=None,
            amperage_transition=((6, 3), (10, 3)),
            power_value_or_delta=1000,
            time=self.now,
            can_be_saved=False,
        )
        self.assertTrue(updated)
        self.assertEqual(
            self.car.get_delta_dampened_power(6, 3, 10, 3),
            1000,
        )

        # reject sign change
        self.assertFalse(
            self.car.update_dampening_value(
                amperage=None,
                amperage_transition=((6, 3), (10, 3)),
                power_value_or_delta=-500,
                time=self.now,
                can_be_saved=False,
            )
        )
        self.assertEqual(
            self.car.get_delta_dampened_power(6, 3, 10, 3),
            1000,
        )

        # reject ratio too high (>1.1)
        self.assertFalse(
            self.car.update_dampening_value(
                amperage=None,
                amperage_transition=((6, 3), (10, 3)),
                power_value_or_delta=1500,
                time=self.now,
                can_be_saved=False,
            )
        )
        self.assertEqual(
            self.car.get_delta_dampened_power(6, 3, 10, 3),
            1000,
        )

        # reject ratio too low (<0.2)
        self.assertFalse(
            self.car.update_dampening_value(
                amperage=None,
                amperage_transition=((6, 3), (10, 3)),
                power_value_or_delta=150,
                time=self.now,
                can_be_saved=False,
            )
        )
        self.assertEqual(
            self.car.get_delta_dampened_power(6, 3, 10, 3),
            1000,
        )

        # accept reasonable change
        self.assertTrue(
            self.car.update_dampening_value(
                amperage=None,
                amperage_transition=((6, 3), (10, 3)),
                power_value_or_delta=950,
                time=self.now,
                can_be_saved=False,
            )
        )
        self.assertEqual(
            self.car.get_delta_dampened_power(6, 3, 10, 3),
            950,
        )


if __name__ == "__main__":
    unittest.main()
