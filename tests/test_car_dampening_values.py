import datetime
import unittest
from unittest.mock import MagicMock, patch

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


class TestCarDampeningSave(unittest.TestCase):
    """Test that measured dampening values are persisted to config entry."""

    def _make_car_with_hass(self):
        """Create a car wired to a fake hass and config entry."""
        mock_hass = MagicMock()
        mock_config_entry = MagicMock()
        mock_config_entry.data = {}

        home = QSHome(hass=mock_hass, config_entry=None, name="test home")
        car = QSCar(
            hass=mock_hass,
            home=home,
            config_entry=mock_config_entry,
            name="test car",
        )
        car.can_dampen_strongly_dynamically = True
        return car, mock_hass, mock_config_entry

    def test_salvable_dampening_stored_on_valid_update(self):
        """Measured value is saved when all conditions are met."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        self.assertIn("measured_charge_10", car._salvable_dampening)
        self.assertEqual(car._salvable_dampening["measured_charge_10"], 2200.0)

    def test_async_update_entry_called_on_first_save(self):
        """Config entry is updated on first measured dampening save."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        mock_hass.config_entries.async_update_entry.assert_called_once()
        call_args = mock_hass.config_entries.async_update_entry.call_args
        self.assertIs(call_args[0][0], mock_config_entry)
        self.assertIn("measured_charge_10", call_args[1]["data"])
        self.assertEqual(call_args[1]["data"]["measured_charge_10"], 2200.0)

    def test_async_update_entry_throttled_within_300s(self):
        """Config entry is NOT updated again within 300 seconds."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        self.assertEqual(mock_hass.config_entries.async_update_entry.call_count, 1)

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(12, 3),
                amperage_transition=None,
                power_value_or_delta=2700.0,
                time=now + datetime.timedelta(seconds=100),
                can_be_saved=True,
            )

        self.assertIn("measured_charge_12", car._salvable_dampening)
        self.assertEqual(mock_hass.config_entries.async_update_entry.call_count, 1)

    def test_async_update_entry_called_again_after_300s(self):
        """Config entry IS updated again after 300 seconds have passed."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        self.assertEqual(mock_hass.config_entries.async_update_entry.call_count, 1)

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(12, 3),
                amperage_transition=None,
                power_value_or_delta=2700.0,
                time=now + datetime.timedelta(seconds=301),
                can_be_saved=True,
            )

        self.assertEqual(mock_hass.config_entries.async_update_entry.call_count, 2)
        last_data = mock_hass.config_entries.async_update_entry.call_args[1]["data"]
        self.assertIn("measured_charge_10", last_data)
        self.assertIn("measured_charge_12", last_data)

    def test_no_save_when_can_be_saved_is_false(self):
        """No save when can_be_saved is False."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=False,
            )

        self.assertEqual(len(car._salvable_dampening), 0)
        mock_hass.config_entries.async_update_entry.assert_not_called()

    def test_no_save_when_config_entry_is_none(self):
        """No save when config_entry is None."""
        home = QSHome(hass=None, config_entry=None, name="test home")
        car = QSCar(hass=None, home=home, config_entry=None, name="test car")
        car.can_dampen_strongly_dynamically = True
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        self.assertEqual(len(car._salvable_dampening), 0)

    def test_no_save_when_car_percent_out_of_range(self):
        """No save when car_percent <= 10 or >= 70."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        for percent in [None, 5.0, 10.0, 70.0, 95.0]:
            car._salvable_dampening = {}
            car._last_dampening_update = None
            mock_hass.config_entries.async_update_entry.reset_mock()

            with patch.object(car, "get_car_charge_percent", return_value=percent):
                car.update_dampening_value(
                    amperage=(10, 3),
                    amperage_transition=None,
                    power_value_or_delta=2200.0,
                    time=now,
                    can_be_saved=True,
                )

            self.assertEqual(
                len(car._salvable_dampening),
                0,
                f"Should not save for car_percent={percent}",
            )
            mock_hass.config_entries.async_update_entry.assert_not_called()

    def test_no_save_when_phase_mismatch(self):
        """No save when for_3p does not match car_is_custom_power_charge_values_3p."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        car.car_is_custom_power_charge_values_3p = True

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 1),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        self.assertEqual(len(car._salvable_dampening), 0)
        mock_hass.config_entries.async_update_entry.assert_not_called()

    def test_phase_auto_set_on_first_save(self):
        """car_is_custom_power_charge_values_3p is auto-set on first save if None."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        self.assertIsNone(car.car_is_custom_power_charge_values_3p)

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 1),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        self.assertFalse(car.car_is_custom_power_charge_values_3p)
        self.assertIn("measured_charge_10", car._salvable_dampening)

    def test_multiple_amps_accumulated_in_salvable(self):
        """Multiple amp values accumulate in _salvable_dampening dict."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(8, 3),
                amperage_transition=None,
                power_value_or_delta=1800.0,
                time=now,
                can_be_saved=True,
            )
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now + datetime.timedelta(seconds=10),
                can_be_saved=True,
            )
            car.update_dampening_value(
                amperage=(16, 3),
                amperage_transition=None,
                power_value_or_delta=3600.0,
                time=now + datetime.timedelta(seconds=20),
                can_be_saved=True,
            )

        self.assertEqual(car._salvable_dampening["measured_charge_8"], 1800.0)
        self.assertEqual(car._salvable_dampening["measured_charge_10"], 2200.0)
        self.assertEqual(car._salvable_dampening["measured_charge_16"], 3600.0)

    def test_saved_data_merges_with_existing_entry_data(self):
        """Saved dampening data is merged with existing config entry data."""
        car, mock_hass, mock_config_entry = self._make_car_with_hass()
        mock_config_entry.data = {"existing_key": "existing_value"}
        now = datetime.datetime.now()

        with patch.object(car, "get_car_charge_percent", return_value=50.0):
            car.update_dampening_value(
                amperage=(10, 3),
                amperage_transition=None,
                power_value_or_delta=2200.0,
                time=now,
                can_be_saved=True,
            )

        saved_data = mock_hass.config_entries.async_update_entry.call_args[1]["data"]
        self.assertEqual(saved_data["existing_key"], "existing_value")
        self.assertEqual(saved_data["measured_charge_10"], 2200.0)


if __name__ == "__main__":
    unittest.main()
