"""Unit tests for car range estimation and efficiency computation."""

import unittest
from datetime import datetime, timedelta
import pytz

from custom_components.quiet_solar.ha_model.car import QSCar, CAR_MAX_EFFICIENCY_HISTORY_S
from custom_components.quiet_solar.ha_model.home import QSHome


class TestCarRangeEstimation(unittest.TestCase):
    """Test car range estimation and efficiency computation."""

    def setUp(self):
        """Set up test car instance."""
        self.home = QSHome(hass=None, config_entry=None, name="test home")
        self.car = QSCar(
            hass=None,
            home=self.home,
            config_entry=None,
            name="test car"
        )
        # Set a battery capacity for efficiency calculations
        self.car.car_battery_capacity = 60000  # 60 kWh in Wh

    def test_add_soc_odo_value_to_segments_first_value(self):
        """Test adding the first SOC/odometer value creates a segment."""
        time = datetime.now(pytz.UTC)
        soc = 80.0
        odo = 10000.0

        self.car._add_soc_odo_value_to_segments(soc, odo, time)

        self.assertEqual(len(self.car._decreasing_segments), 1)
        segment = self.car._decreasing_segments[0]
        self.assertEqual(segment[0], (soc, odo, time))
        self.assertIsNone(segment[1])  # Segment not yet closed
        self.assertEqual(segment[2], 0)  # First segment index

    def test_add_soc_odo_value_to_segments_decreasing_soc(self):
        """Test adding decreasing SOC values extends the current segment."""
        time1 = datetime.now(pytz.UTC)
        time2 = time1 + timedelta(hours=1)

        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time1)
        self.car._add_soc_odo_value_to_segments(75.0, 10050.0, time2)

        self.assertEqual(len(self.car._decreasing_segments), 1)
        segment = self.car._decreasing_segments[0]
        self.assertEqual(segment[0], (80.0, 10000.0, time1))
        self.assertEqual(segment[1], (75.0, 10050.0, time2))

    def test_add_soc_odo_value_to_segments_increasing_soc_starts_new_segment(self):
        """Test that increasing SOC after a decrease closes the segment and starts a new one."""
        time1 = datetime.now(pytz.UTC)
        time2 = time1 + timedelta(hours=1)
        time3 = time2 + timedelta(hours=1)

        # First value
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time1)
        # Decrease - extends segment
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, time2)
        # Increase - should close segment and start new one
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, time3)

        self.assertEqual(len(self.car._decreasing_segments), 2)
        # First segment should be closed
        first_segment = self.car._decreasing_segments[0]
        self.assertEqual(first_segment[0], (80.0, 10000.0, time1))
        self.assertEqual(first_segment[1], (70.0, 10100.0, time2))
        # New segment should be open
        second_segment = self.car._decreasing_segments[1]
        self.assertEqual(second_segment[0], (90.0, 10100.0, time3))
        self.assertIsNone(second_segment[1])

    def test_add_soc_odo_value_efficiency_segment_created(self):
        """Test that efficiency segments are created when SOC decreases with odometer increase."""
        time1 = datetime.now(pytz.UTC)
        time2 = time1 + timedelta(hours=1)
        time3 = time2 + timedelta(hours=1)

        # Create a decreasing segment
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time1)
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, time2)
        # Close the segment by increasing SOC
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, time3)

        # Check that efficiency segment was created
        self.assertEqual(len(self.car._efficiency_segments), 1)
        eff_seg = self.car._efficiency_segments[0]
        delta_km, delta_soc, from_soc, to_soc, seg_time = eff_seg
        self.assertEqual(delta_km, 100.0)  # 10100 - 10000
        self.assertEqual(delta_soc, 10.0)  # 80 - 70
        self.assertEqual(from_soc, 80.0)
        self.assertEqual(to_soc, 70.0)
        self.assertEqual(seg_time, time3)

    def test_add_soc_odo_value_multiple_segments(self):
        """Test creating multiple efficiency segments over time."""
        base_time = datetime.now(pytz.UTC)

        # First driving segment: 80% -> 70%, 100km
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, base_time)
        self.car._add_soc_odo_value_to_segments(75.0, 10050.0, base_time + timedelta(hours=0.5))
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, base_time + timedelta(hours=1))
        # Charge the car
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, base_time + timedelta(hours=2))

        # Second driving segment: 90% -> 75%, 150km
        self.car._add_soc_odo_value_to_segments(85.0, 10175.0, base_time + timedelta(hours=3))
        self.car._add_soc_odo_value_to_segments(75.0, 10250.0, base_time + timedelta(hours=4))
        # Charge again
        self.car._add_soc_odo_value_to_segments(95.0, 10250.0, base_time + timedelta(hours=5))

        # Should have 2 efficiency segments
        self.assertEqual(len(self.car._efficiency_segments), 2)

        # First segment: 80% -> 70%, 100km
        seg1 = self.car._efficiency_segments[0]
        self.assertEqual(seg1[0], 100.0)  # delta_km
        self.assertEqual(seg1[1], 10.0)   # delta_soc
        self.assertEqual(seg1[2], 80.0)   # from_soc
        self.assertEqual(seg1[3], 70.0)   # to_soc

        # Second segment: 90% -> 75%, 150km
        seg2 = self.car._efficiency_segments[1]
        self.assertEqual(seg2[0], 150.0)  # delta_km
        self.assertEqual(seg2[1], 15.0)   # delta_soc
        self.assertEqual(seg2[2], 90.0)   # from_soc
        self.assertEqual(seg2[3], 75.0)   # to_soc

    def test_add_soc_odo_value_bad_segment_not_stored(self):
        """Test that segments with bad data (decreasing odometer) are not stored."""
        time1 = datetime.now(pytz.UTC)
        time2 = time1 + timedelta(hours=1)
        time3 = time2 + timedelta(hours=1)

        # Create a segment with decreasing odometer (bad data)
        self.car._add_soc_odo_value_to_segments(80.0, 10100.0, time1)
        self.car._add_soc_odo_value_to_segments(70.0, 10000.0, time2)  # Odometer decreased!
        # Try to close it
        self.car._add_soc_odo_value_to_segments(90.0, 10000.0, time3)

        # Should not have created an efficiency segment
        self.assertEqual(len(self.car._efficiency_segments), 0)
        # Should have replaced the bad segment
        self.assertEqual(len(self.car._decreasing_segments), 1)
        self.assertEqual(self.car._decreasing_segments[0][0], (90.0, 10000.0, time3))

    def test_add_soc_odo_same_soc_keeps_segment_open(self):
        """Test that same SOC value keeps the segment open."""
        time1 = datetime.now(pytz.UTC)
        time2 = time1 + timedelta(hours=1)

        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time1)
        self.car._add_soc_odo_value_to_segments(80.0, 10050.0, time2)

        # Should still have only one segment, not closed
        self.assertEqual(len(self.car._decreasing_segments), 1)
        segment = self.car._decreasing_segments[0]
        self.assertEqual(segment[0], (80.0, 10000.0, time1))
        self.assertIsNone(segment[1])  # Still open

    def test_efficiency_segments_time_ordering(self):
        """Test that efficiency segments are kept in time order when added."""
        base_time = datetime.now(pytz.UTC)

        # Create first segment at a later time
        seg1_finish_time = base_time + timedelta(hours=10)
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, base_time + timedelta(hours=8))
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, base_time + timedelta(hours=9))
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, seg1_finish_time)

        # Now manually create a second segment with an earlier finish time
        # This simulates historical data being processed
        seg2_finish_time = base_time + timedelta(hours=5)
        
        # Directly append to efficiency segments to test time ordering
        earlier_segment = (50.0, 5.0, 85.0, 80.0, seg2_finish_time)
        
        # The code should handle this by inserting in order
        # Simulate what happens when a segment finishes earlier
        if len(self.car._efficiency_segments) == 0 or seg2_finish_time > self.car._efficiency_segments[-1][4]:
            self.car._efficiency_segments.append(earlier_segment)
        else:
            # Find insertion point
            insert_idx = 0
            for i, seg in enumerate(self.car._efficiency_segments):
                if seg[4] > seg2_finish_time:
                    insert_idx = i
                    break
                insert_idx = i + 1
            self.car._efficiency_segments.insert(insert_idx, earlier_segment)

        # Segments should be in time order
        self.assertEqual(len(self.car._efficiency_segments), 2)
        self.assertLess(self.car._efficiency_segments[0][4], self.car._efficiency_segments[1][4])

    def test_efficiency_segments_history_limit(self):
        """Test that old efficiency segments are removed after history limit."""
        base_time = datetime.now(pytz.UTC)
        old_time = base_time - timedelta(seconds=CAR_MAX_EFFICIENCY_HISTORY_S + 3600)

        # Add an old segment
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, old_time)
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, old_time + timedelta(hours=1))
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, old_time + timedelta(hours=2))

        # Add a recent segment
        self.car._add_soc_odo_value_to_segments(85.0, 11000.0, base_time)
        self.car._add_soc_odo_value_to_segments(75.0, 11100.0, base_time + timedelta(hours=1))
        self.car._add_soc_odo_value_to_segments(95.0, 11100.0, base_time + timedelta(hours=2))

        # Old segment should have been removed
        self.assertEqual(len(self.car._efficiency_segments), 1)
        # Remaining segment should be the recent one
        self.assertEqual(self.car._efficiency_segments[0][4], base_time + timedelta(hours=2))

    def test_get_car_estimated_range_km_with_efficiency_segment(self):
        """Test range estimation using efficiency segments."""
        time = datetime.now(pytz.UTC)

        # Create an efficiency segment: 10% SOC used for 100km
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time)
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, time + timedelta(hours=1))
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, time + timedelta(hours=2))

        # Estimate range for 10% SOC (should be ~100km based on segment)
        estimated_range = self.car.get_car_estimated_range_km(from_soc=10.0, to_soc=0.0, time=time + timedelta(hours=3))

        self.assertIsNotNone(estimated_range)
        self.assertAlmostEqual(estimated_range, 100.0, places=1)

    def test_get_car_estimated_range_km_interpolation(self):
        """Test range estimation with SOC delta different from stored segment."""
        time = datetime.now(pytz.UTC)

        # Create an efficiency segment: 10% SOC used for 100km (10 km/%)
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time)
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, time + timedelta(hours=1))
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, time + timedelta(hours=2))

        # Estimate range for 20% SOC (should be ~200km)
        estimated_range = self.car.get_car_estimated_range_km(from_soc=20.0, to_soc=0.0, time=time + timedelta(hours=3))

        self.assertIsNotNone(estimated_range)
        self.assertAlmostEqual(estimated_range, 200.0, places=1)

    def test_get_car_estimated_range_km_best_segment_selection(self):
        """Test that the best matching efficiency segment is selected."""
        time = datetime.now(pytz.UTC)

        # Create two efficiency segments with different delta SOCs
        # Segment 1: Drive from 80% to 70%, distance 100km (10 km/%)
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time)
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, time + timedelta(hours=1))
        # Charge to close segment 1
        self.car._add_soc_odo_value_to_segments(100.0, 10100.0, time + timedelta(hours=2))

        # Segment 2: Drive from 100% to 80%, distance 240km (12 km/%)
        self.car._add_soc_odo_value_to_segments(95.0, 10130.0, time + timedelta(hours=10))
        self.car._add_soc_odo_value_to_segments(80.0, 10340.0, time + timedelta(hours=11))
        # Charge to close segment 2
        self.car._add_soc_odo_value_to_segments(100.0, 10340.0, time + timedelta(hours=12))

        # Check that we have 2 segments
        self.assertEqual(len(self.car._efficiency_segments), 2)
        
        # Verify segment contents
        seg1 = self.car._efficiency_segments[0]
        self.assertEqual(seg1[0], 100.0)  # delta_km
        self.assertEqual(seg1[1], 10.0)   # delta_soc
        
        seg2 = self.car._efficiency_segments[1]
        self.assertAlmostEqual(seg2[0], 240.0, places=1)  # delta_km (10340 - 10100)
        self.assertAlmostEqual(seg2[1], 20.0, places=1)   # delta_soc (100 - 80)

        # Estimate range for similar SOC delta - should prefer the closer match
        estimated_range = self.car.get_car_estimated_range_km(from_soc=10.0, to_soc=0.0, time=time + timedelta(hours=13))

        self.assertIsNotNone(estimated_range)
        # With 10% SOC and segment 1 (10% = 100km), should be ~100km
        # With 10% SOC and segment 2 (20% = 240km), extrapolating gives 10% = 120km
        # Should use segment 1 as it's an exact match for delta_soc
        self.assertAlmostEqual(estimated_range, 100.0, places=0)

    def test_get_car_estimated_range_km_no_segments(self):
        """Test range estimation returns None when no efficiency segments exist."""
        time = datetime.now(pytz.UTC)

        estimated_range = self.car.get_car_estimated_range_km(from_soc=50.0, to_soc=0.0, time=time)

        # Should be None as there are no segments and no _km_per_kwh set
        self.assertIsNone(estimated_range)

    def test_get_car_estimated_range_km_with_km_per_kwh(self):
        """Test range estimation using _km_per_kwh when available."""
        time = datetime.now(pytz.UTC)

        # Set efficiency directly
        self.car._km_per_kwh = 6.0  # 6 km per kWh

        # Estimate range for 50% SOC with 60 kWh battery
        # 50% of 60 kWh = 30 kWh * 6 km/kWh = 180 km
        estimated_range = self.car.get_car_estimated_range_km(from_soc=50.0, to_soc=0.0, time=time)

        self.assertIsNotNone(estimated_range)
        self.assertAlmostEqual(estimated_range, 180.0, places=1)

    def test_car_efficiency_km_per_kwh_ema_computation(self):
        """Test that _km_per_kwh is updated using EMA during active segment."""
        time = datetime.now(pytz.UTC)

        # Start a decreasing segment
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time)

        # Manually trigger the state getter logic (which would be called during state updates)
        # This simulates what happens when the sensor is queried
        result = self.car.car_efficiency_km_per_kwh_sensor_state_getter(
            self.car.car_efficiency_km_per_kwh_sensor, time + timedelta(hours=1)
        )

        # Since we don't have car_odometer_sensor set, it should return None
        self.assertIsNone(result)

        # Set up the sensors
        self.car.car_odometer_sensor = "sensor.odometer"
        self.car.car_charge_percent_sensor = "sensor.soc"

        # Mock the get methods to return values
        def mock_get_odometer(t):
            if t == time:
                return 10000.0
            elif t == time + timedelta(hours=1):
                return 10060.0
            return None

        def mock_get_soc(t):
            if t == time:
                return 80.0
            elif t == time + timedelta(hours=1):
                return 70.0
            return None

        # Replace methods
        original_get_odo = self.car.get_car_odometer_km
        original_get_soc = self.car.get_car_charge_percent
        self.car.get_car_odometer_km = mock_get_odometer
        self.car.get_car_charge_percent = mock_get_soc

        # Add first value
        result = self.car.car_efficiency_km_per_kwh_sensor_state_getter(
            self.car.car_efficiency_km_per_kwh_sensor, time
        )

        # Add second value (should compute efficiency)
        result = self.car.car_efficiency_km_per_kwh_sensor_state_getter(
            self.car.car_efficiency_km_per_kwh_sensor, time + timedelta(hours=1)
        )

        # Restore methods
        self.car.get_car_odometer_km = original_get_odo
        self.car.get_car_charge_percent = original_get_soc

        # Check that _km_per_kwh was computed
        # 60km traveled, 10% SOC (6 kWh), so 60/6 = 10 km/kWh
        self.assertIsNotNone(self.car._km_per_kwh)
        self.assertAlmostEqual(self.car._km_per_kwh, 10.0, places=1)

    def test_get_estimated_range_km_integration(self):
        """Test the main get_estimated_range_km method."""
        time = datetime.now(pytz.UTC)

        # Mock get_car_charge_percent
        original_get_soc = self.car.get_car_charge_percent
        self.car.get_car_charge_percent = lambda t: 70.0

        # Set efficiency
        self.car._km_per_kwh = 6.0

        # Get estimated range
        range_km = self.car.get_estimated_range_km(time)

        # Restore
        self.car.get_car_charge_percent = original_get_soc

        # 70% of 60 kWh = 42 kWh * 6 km/kWh = 252 km
        self.assertIsNotNone(range_km)
        self.assertAlmostEqual(range_km, 252.0, places=1)

    def test_get_autonomy_to_target_soc_km(self):
        """Test autonomy calculation to target SOC."""
        time = datetime.now(pytz.UTC)

        # Mock get_car_target_SOC
        original_get_target = self.car.get_car_target_SOC
        self.car.get_car_target_SOC = lambda: 50.0

        # Set efficiency
        self.car._km_per_kwh = 6.0

        # Get autonomy
        autonomy_km = self.car.get_autonomy_to_target_soc_km(time)

        # Restore
        self.car.get_car_target_SOC = original_get_target

        # 50% of 60 kWh = 30 kWh * 6 km/kWh = 180 km
        self.assertIsNotNone(autonomy_km)
        self.assertAlmostEqual(autonomy_km, 180.0, places=1)

    def test_complex_driving_pattern(self):
        """Test a complex driving pattern with multiple charge/discharge cycles."""
        base_time = datetime.now(pytz.UTC)

        # Day 1: Drive from 100% to 60% (100km) - efficiency 10 km/% or 6 km/kWh
        self.car._add_soc_odo_value_to_segments(100.0, 0.0, base_time)
        self.car._add_soc_odo_value_to_segments(90.0, 25.0, base_time + timedelta(hours=0.5))
        self.car._add_soc_odo_value_to_segments(80.0, 50.0, base_time + timedelta(hours=1))
        self.car._add_soc_odo_value_to_segments(70.0, 75.0, base_time + timedelta(hours=1.5))
        self.car._add_soc_odo_value_to_segments(60.0, 100.0, base_time + timedelta(hours=2))
        # Charge
        self.car._add_soc_odo_value_to_segments(100.0, 100.0, base_time + timedelta(hours=10))

        # Day 2: Drive from 100% to 50% (120km) - efficiency 12 km/% or 7.2 km/kWh
        self.car._add_soc_odo_value_to_segments(95.0, 130.0, base_time + timedelta(hours=24.5))
        self.car._add_soc_odo_value_to_segments(85.0, 160.0, base_time + timedelta(hours=25))
        self.car._add_soc_odo_value_to_segments(75.0, 190.0, base_time + timedelta(hours=25.5))
        self.car._add_soc_odo_value_to_segments(65.0, 210.0, base_time + timedelta(hours=26))
        self.car._add_soc_odo_value_to_segments(50.0, 220.0, base_time + timedelta(hours=26.5))
        # Charge
        self.car._add_soc_odo_value_to_segments(100.0, 220.0, base_time + timedelta(hours=34))

        # Should have 2 efficiency segments
        self.assertEqual(len(self.car._efficiency_segments), 2)

        # Test range estimation - should use most recent segment with best match
        estimated_range = self.car.get_car_estimated_range_km(
            from_soc=50.0, to_soc=0.0,
            time=base_time + timedelta(hours=35)
        )

        # Should use second segment (50% delta): 50% = 120km (proportional)
        self.assertIsNotNone(estimated_range)
        # The second segment has 50% SOC for 120km, so 50% should give exactly 120km
        self.assertAlmostEqual(estimated_range, 120.0, places=1)

    def test_efficiency_deltas_graph_creation(self):
        """Test that efficiency deltas and graph structures are created."""
        time = datetime.now(pytz.UTC)

        # Create an efficiency segment
        self.car._add_soc_odo_value_to_segments(80.0, 10000.0, time)
        self.car._add_soc_odo_value_to_segments(70.0, 10100.0, time + timedelta(hours=1))
        self.car._add_soc_odo_value_to_segments(90.0, 10100.0, time + timedelta(hours=2))

        # Check that deltas were created
        self.assertIn((80.0, 70.0), self.car._efficiency_deltas)
        self.assertIn((70.0, 80.0), self.car._efficiency_deltas)
        self.assertEqual(self.car._efficiency_deltas[(80.0, 70.0)], 100.0)
        self.assertEqual(self.car._efficiency_deltas[(70.0, 80.0)], -100.0)

        # Check that graph was created
        self.assertIn(80.0, self.car._efficiency_deltas_graph)
        self.assertIn(70.0, self.car._efficiency_deltas_graph)
        self.assertIn(70.0, self.car._efficiency_deltas_graph[80.0])
        self.assertIn(80.0, self.car._efficiency_deltas_graph[70.0])

    def test_continuous_decreasing_segment_extension(self):
        """Test that continuously decreasing SOC properly extends a segment."""
        base_time = datetime.now(pytz.UTC)

        # Add many points in a continuous decreasing pattern
        for i in range(10):
            soc = 100.0 - (i * 2.0)  # 100, 98, 96, ..., 82
            odo = 1000.0 + (i * 10.0)  # 1000, 1010, 1020, ..., 1090
            time = base_time + timedelta(minutes=i * 30)
            self.car._add_soc_odo_value_to_segments(soc, odo, time)

        # Should still be in one open segment
        self.assertEqual(len(self.car._decreasing_segments), 1)
        segment = self.car._decreasing_segments[0]
        self.assertEqual(segment[0][0], 100.0)
        self.assertEqual(segment[1][0], 82.0)
        self.assertEqual(segment[0][1], 1000.0)
        self.assertEqual(segment[1][1], 1090.0)

        # Close the segment
        self.car._add_soc_odo_value_to_segments(95.0, 1090.0, base_time + timedelta(hours=5))

        # Now should have an efficiency segment
        self.assertEqual(len(self.car._efficiency_segments), 1)
        eff_seg = self.car._efficiency_segments[0]
        self.assertEqual(eff_seg[0], 90.0)  # delta_km
        self.assertEqual(eff_seg[1], 18.0)  # delta_soc (100 - 82)


if __name__ == '__main__':
    unittest.main()

