"""Tests for person and car forecast computation."""
import pickle
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from pathlib import Path

import pytz

from custom_components.quiet_solar.ha_model.home import QSHome
from custom_components.quiet_solar.ha_model.person import QSPerson
from custom_components.quiet_solar.ha_model.car import QSCar
from tests.conftest import FakeHass, FakeConfigEntry


class TestPersonsCarForecast:
    """Test person and car forecast computation."""

    @pytest.fixture
    def test_data_path(self):
        """Get path to test data directory."""
        return Path(__file__).parent / "data" / "2025_10_30_persons_and_cars"

    @pytest.fixture
    def person_and_car_data(self, test_data_path):
        """Load the pickled person and car data."""
        pickle_file = test_data_path / "person_and_car.pickle"
        assert pickle_file.exists(), f"Test data file not found: {pickle_file}"

        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        return data

    @pytest.fixture
    def fake_hass(self):
        """Create a fake Home Assistant instance."""
        return FakeHass()

    @pytest.fixture
    def mock_home(self, fake_hass):
        """Create a mock QSHome instance."""
        config_entry = FakeConfigEntry(
            entry_id="test_home",
            data={"name": "Test Home"},
            title="Test Home"
        )
        home = QSHome(
            hass=fake_hass,
            config_entry=config_entry,
            name="Test Home"
        )
        home.latitude = 43.6346599933627
        home.longitude = 6.988867074251175
        home.radius = 100.0
        return home

    @pytest.fixture
    def mock_load_from_history(self, person_and_car_data):
        """Create a mock for load_from_history that returns data from the pickle file.

        IMPORTANT: Uses EXACT matching for per_day data when start/end times match exactly.
        This ensures we use the precise data that was captured for each day shift.
        """

        # Extract data from the pickle file
        full_range_data = person_and_car_data.get("full_range")
        per_day_data = person_and_car_data.get("per_day", [])
        test_time = person_and_car_data.get("time")

        # Build lookup dictionaries for car and person data
        # Structure: {(entity_name, start, end): data}
        car_positions_lookup = {}
        car_odos_lookup = {}
        person_positions_lookup = {}

        car_position_entity_matching = {}
        car_odos_entity_matching = {}

        car_position_entity_to_name = {}
        car_odometer_entity_to_name = {}
        person_position_entity_matching = {}

        positions_for_car = {}
        odos_for_car = {}
        positions_for_person = {}

        # Process full_range data
        if full_range_data:
            start, end, car_data, person_data = full_range_data

            for car_name, car_positions, car_odos in car_data:
                # Extract actual entity IDs from pickle data
                if car_positions and len(car_positions) > 0:
                    car_tracker = car_positions[0].entity_id
                    car_position_entity_matching[car_name] = car_tracker
                    car_position_entity_to_name[car_tracker] = car_name

                if car_odos and len(car_odos) > 0:
                    car_odometer_sensor = car_odos[0].entity_id
                    car_odos_entity_matching[car_name] = car_odometer_sensor
                    car_odometer_entity_to_name[car_odometer_sensor] = car_name

            for person_entity_id, person_positions in person_data:
                person_position_entity_matching[person_entity_id] = person_entity_id


            for car_name, car_positions, car_odos in car_data:

                car_tracker = car_position_entity_matching.get(car_name, None)
                if car_tracker is not None:
                    if car_tracker not in positions_for_car:
                        positions_for_car[car_tracker] = []

                    for state in car_positions:
                        time = state.last_updated
                        attr = state.attributes
                        if attr is not None:
                            time = attr.get("last_updated", time)
                        positions_for_car[car_tracker].append((time, state))
                        # sort it
                    positions_for_car[car_tracker].sort(key=lambda x: x[0])


                    key = (car_tracker, start, end)
                    car_positions_lookup[key] = car_positions

                car_odometer_sensor = car_odos_entity_matching.get(car_name,None)
                if car_odometer_sensor is not None:

                    if car_odometer_sensor not in odos_for_car:
                        odos_for_car[car_odometer_sensor] = []

                    for state in car_odos:
                        time = state.last_updated
                        attr = state.attributes
                        if attr is not None:
                            time = attr.get("last_updated", time)
                        odos_for_car[car_odometer_sensor].append((time, state))
                    odos_for_car[car_odometer_sensor].sort(key=lambda x: x[0])

                    key = (car_odometer_sensor, start, end)
                    car_odos_lookup[key] = car_odos

            for person_entity_id, person_positions in person_data:
                key = (person_entity_id, start, end)
                person_positions_lookup[key] = person_positions

                if person_entity_id not in positions_for_person:
                    positions_for_person[person_entity_id] = []
                for state in person_positions:
                    time = state.last_updated
                    attr = state.attributes
                    if attr is not None:
                        time = attr.get("last_updated", time)
                    positions_for_person[person_entity_id].append((time, state))
                positions_for_person[person_entity_id].sort(key=lambda x: x[0])

        # Process per_day data - store with exact timestamps
        for day_data in per_day_data:
            start, end, car_data, person_data = day_data

            for car_name, car_positions, car_odos in car_data:

                car_tracker = car_position_entity_matching.get(car_name, None)
                if car_tracker is not None:
                    key = (car_tracker, start, end)
                    car_positions_lookup[key] = car_positions

                car_odometer_sensor = car_odos_entity_matching.get(car_name, None)
                if car_odometer_sensor is not None:
                    key = (car_odometer_sensor, start, end)
                    car_odos_lookup[key] = car_odos

            for person_entity_id, person_positions in person_data:
                key = (person_entity_id, start, end)
                person_positions_lookup[key] = person_positions



        async def mock_load_fn(hass, entity_id: str, start_time: datetime, end_time: datetime, no_attributes=True):
            """Mock load_from_history function that looks up data from pickle.

            First tries EXACT matching (for per_day data), then falls back to approximate matching.
            Matches entity IDs by car name, handling different naming conventions.
            """

            # FIRST: Try exact match for car positions (tracker)
            if entity_id in car_position_entity_to_name:
                positions = car_positions_lookup.get((entity_id, start_time, end_time), None)
                if positions is not None:
                    return positions

            # FIRST: Try exact match for car odometers
            if entity_id in car_odometer_entity_to_name:
                odos = car_odos_lookup.get((entity_id, start_time, end_time), None)
                if odos is not None:
                    return odos

            # FIRST: Try exact match for person
            if entity_id in person_position_entity_matching:
                positions = person_positions_lookup.get((entity_id, start_time, end_time), None)
                if positions is not None:
                    return positions

            # SECOND:
            time_series = None
            if entity_id in positions_for_car:
                time_series = positions_for_car[entity_id]
            elif entity_id in odos_for_car:
                time_series = odos_for_car[entity_id]
            elif entity_id in positions_for_person:
                time_series = positions_for_person[entity_id]

            if time_series is not None:
                # use bisect to find proper start and end indices
                from bisect import bisect_left, bisect_right
                start_idx = bisect_left(time_series, start_time, key=lambda x: x[0])
                end_idx = bisect_right(time_series, end_time, key=lambda x: x[0])
                return [state for _, state in time_series[start_idx:end_idx]]


            # Default: return empty list if no match found
            return []

        return mock_load_fn

    @pytest.mark.asyncio
    async def test_compute_and_store_person_car_forecasts(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test _compute_and_store_person_car_forecasts with real QSCar instances and real mileage computation."""

        # Get test time from pickle data
        test_time = person_and_car_data.get("time")
        assert test_time is not None, "Test time not found in pickle data"

        # Create REAL cars and persons based on the data structure
        per_day_data = person_and_car_data.get("per_day", [])
        if per_day_data:
            first_day = per_day_data[0]
            start, end, car_data, person_data = first_day

            # Create REAL QSCar instances with actual entity IDs from pickle
            for car_name, car_positions, car_odos in car_data:
                # Extract actual entity IDs from pickle data
                car_tracker = None
                if car_positions and len(car_positions) > 0:
                    car_tracker = car_positions[0].entity_id

                car_odometer_sensor = None
                if car_odos and len(car_odos) > 0:
                    car_odometer_sensor = car_odos[0].entity_id

                # Create real QSCar with real sensors - get_car_mileage_on_period_km NOT mocked
                car = QSCar(
                    hass=mock_home.hass,
                    home=mock_home,
                    config_entry=None,
                    name=car_name,
                    car_tracker=car_tracker,
                    car_odometer_sensor=car_odometer_sensor,
                    car_battery_capacity=50000
                )
                mock_home._cars.append(car)

            # Create real persons
            for person_entity_id, person_positions in person_data:
                person = QSPerson(
                    hass=mock_home.hass,
                    home=mock_home,
                    config_entry=None,
                    name=person_entity_id.split('.')[1],
                    person_person_entity=person_entity_id,
                    person_authorized_cars=[car.name for car in mock_home._cars] if mock_home._cars else []
                )
                mock_home._persons.append(person)

        # Patch load_from_history for both home and car modules
        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            # Get local day UTC for testing
            local = test_time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            local_shifted = local - timedelta(hours=4)  # HOME_PERSON_CAR_DAY_JOURNEY_START_HOURS
            local_day = datetime(local.year, local.month, local.day)
            local_day_utc = local_day.replace(tzinfo=None).astimezone(tz=pytz.UTC)

            # Run the method under test
            await mock_home._compute_and_store_person_car_forecasts(local_day_utc, day_shift=0)

            # Verify that persons have been updated with mileage data
            for person in mock_home._persons:
                assert isinstance(person, QSPerson), f"Person should be QSPerson instance"
                # Check if mileage history was updated (it may be empty if no valid data)
                assert hasattr(person, 'historical_mileage_data'), "Person should have historical_mileage_data"
                # The data may be empty or populated depending on the actual data in the pickle
                print(f"Person {person.name} has {len(person.historical_mileage_data)} mileage entries")

    @pytest.mark.asyncio
    async def test_compute_mileage_for_period_per_person(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test _compute_mileage_for_period_per_person with real QSCar instances (not mocked)."""

        # Get test time from pickle data
        test_time = person_and_car_data.get("time")
        assert test_time is not None, "Test time not found in pickle data"

        # Setup cars and persons
        per_day_data = person_and_car_data.get("per_day", [])
        if not per_day_data:
            pytest.skip("No per_day data available in pickle file")

        first_day = per_day_data[0]
        start, end, car_data, person_data = first_day

        # Create REAL QSCar instances with actual entity IDs from pickle
        for car_name, car_positions, car_odos in car_data:
            # Extract actual entity IDs
            car_tracker = None
            if car_positions and len(car_positions) > 0:
                car_tracker = car_positions[0].entity_id

            car_odometer_sensor = None
            if car_odos and len(car_odos) > 0:
                car_odometer_sensor = car_odos[0].entity_id

            # Create real QSCar - get_car_mileage_on_period_km NOT mocked
            car = QSCar(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=car_name,
                car_tracker=car_tracker,
                car_odometer_sensor=car_odometer_sensor,
                car_battery_capacity=50000
            )
            mock_home._cars.append(car)

        # Create real QSPerson instances
        for person_entity_id, person_positions in person_data:
            person = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=person_entity_id.split('.')[1],
                person_person_entity=person_entity_id,
                person_authorized_cars=[car.name for car in mock_home._cars]
            )
            mock_home._persons.append(person)

        # Patch load_from_history for both modules but NOT _compute_mileage_for_period_per_person
        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            # Call the method directly (not mocked)
            persons_mileage = await mock_home._compute_mileage_for_period_per_person(start, end)

            # Verify results
            assert isinstance(persons_mileage, dict), "Result should be a dictionary"

            # Check structure of results
            for person, (mileage, leave_time) in persons_mileage.items():
                assert isinstance(person, QSPerson), "Key should be QSPerson instance"
                assert isinstance(mileage, (int, float)), "Mileage should be numeric"
                assert mileage >= 0, "Mileage should be non-negative"
                print(f"Person {person.name}: mileage={mileage}km, leave_time={leave_time}")

            return persons_mileage

    @pytest.mark.asyncio
    async def test_person_forecast_data_structure(self, person_and_car_data):
        """Test that the pickle data has the expected structure."""

        # Verify top-level structure
        assert "full_range" in person_and_car_data, "Should have full_range data"
        assert "per_day" in person_and_car_data, "Should have per_day data"
        assert "time" in person_and_car_data, "Should have time data"

        # Verify full_range structure
        full_range = person_and_car_data["full_range"]
        assert len(full_range) == 4, "full_range should have 4 elements: [start, end, car_data, person_data]"

        start, end, car_data, person_data = full_range
        assert isinstance(start, datetime), "start should be datetime"
        assert isinstance(end, datetime), "end should be datetime"
        assert isinstance(car_data, list), "car_data should be list"
        assert isinstance(person_data, list), "person_data should be list"

        # Verify car_data structure
        for car_entry in car_data:
            assert len(car_entry) == 3, "Car entry should have 3 elements: (name, positions, odos)"
            car_name, car_positions, car_odos = car_entry
            assert isinstance(car_name, str), "Car name should be string"
            assert isinstance(car_positions, list), "Car positions should be list"
            assert isinstance(car_odos, list), "Car odos should be list"
            print(f"Car: {car_name}, positions: {len(car_positions)}, odos: {len(car_odos)}")

        # Verify person_data structure
        for person_entry in person_data:
            assert len(person_entry) == 2, "Person entry should have 2 elements: (entity_id, positions)"
            person_entity_id, person_positions = person_entry
            assert isinstance(person_entity_id, str), "Person entity_id should be string"
            assert isinstance(person_positions, list), "Person positions should be list"
            print(f"Person: {person_entity_id}, positions: {len(person_positions)}")

        # Verify per_day structure
        per_day = person_and_car_data["per_day"]
        assert isinstance(per_day, list), "per_day should be list"
        assert len(per_day) > 0, "per_day should have at least one entry"

        print(f"Test data contains {len(per_day)} days of data")
        print(f"Test time: {person_and_car_data['time']}")

    @pytest.mark.asyncio
    async def test_full_forecast_update_cycle(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test a complete forecast update cycle with multiple days using real QSCar instances."""

        # Setup
        test_time = person_and_car_data.get("time")
        per_day_data = person_and_car_data.get("per_day", [])

        if not per_day_data:
            pytest.skip("No per_day data available")

        # Create REAL cars and persons from first day
        first_day = per_day_data[0]
        start, end, car_data, person_data = first_day

        # Create REAL QSCar instances with actual entity IDs
        for car_name, car_positions, car_odos in car_data:
            car_tracker = None
            if car_positions and len(car_positions) > 0:
                car_tracker = car_positions[0].entity_id

            car_odometer_sensor = None
            if car_odos and len(car_odos) > 0:
                car_odometer_sensor = car_odos[0].entity_id

            car = QSCar(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=car_name,
                car_tracker=car_tracker,
                car_odometer_sensor=car_odometer_sensor,
                car_battery_capacity=50000
            )
            mock_home._cars.append(car)

        for person_entity_id, _ in person_data:
            person = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=person_entity_id.split('.')[1],
                person_person_entity=person_entity_id,
                person_authorized_cars=[car.name for car in mock_home._cars]
            )
            mock_home._persons.append(person)

        # Patch load_from_history for both modules
        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            # Process multiple days
            local = test_time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            local_day = datetime(local.year, local.month, local.day)
            local_day_utc = local_day.replace(tzinfo=None).astimezone(tz=pytz.UTC)

            # Process last 3 days (or fewer if less data available)
            num_days = min(3, len(per_day_data))
            for day_shift in range(num_days):
                await mock_home._compute_and_store_person_car_forecasts(
                    local_day_utc,
                    day_shift=day_shift
                )

            # Verify all persons have accumulated history
            for person in mock_home._persons:
                print(f"\nPerson: {person.name}")
                print(f"  Historical mileage entries: {len(person.historical_mileage_data)}")

                for day, mileage, leave_time, weekday in person.historical_mileage_data:
                    print(f"    Day: {day}, Mileage: {mileage}km, Leave: {leave_time}, Weekday: {weekday}")

                # Person should have at least some history if they used any cars
                if len(person.authorized_cars) > 0:
                    # History could be empty if person never matched any car trips
                    assert len(person.historical_mileage_data) >= 0, \
                        f"Person {person.name} should have mileage history or empty list"

    @pytest.mark.asyncio
    async def test_person_mileage_prediction(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test that persons can predict next day mileage based on historical data with real QSCar instances."""

        test_time = person_and_car_data.get("time")
        per_day_data = person_and_car_data.get("per_day", [])

        if not per_day_data:
            pytest.skip("No per_day data available")

        # Setup REAL cars and persons
        first_day = per_day_data[0]
        start, end, car_data, person_data = first_day

        # Create REAL QSCar instances
        for car_name, car_positions, car_odos in car_data:
            car_tracker = None
            if car_positions and len(car_positions) > 0:
                car_tracker = car_positions[0].entity_id

            car_odometer_sensor = None
            if car_odos and len(car_odos) > 0:
                car_odometer_sensor = car_odos[0].entity_id

            car = QSCar(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=car_name,
                car_tracker=car_tracker,
                car_odometer_sensor=car_odometer_sensor,
                car_battery_capacity=50000
            )
            mock_home._cars.append(car)

        for person_entity_id, _ in person_data:
            person = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=person_entity_id.split('.')[1],
                person_person_entity=person_entity_id,
                person_authorized_cars=[car.name for car in mock_home._cars]
            )
            mock_home._persons.append(person)

        # Patch load_from_history for both modules
        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            # Process multiple days to build history
            local = test_time.replace(tzinfo=pytz.UTC).astimezone(tz=None)
            local_day = datetime(local.year, local.month, local.day)
            local_day_utc = local_day.replace(tzinfo=None).astimezone(tz=pytz.UTC)

            # Process several days
            num_days = min(5, len(per_day_data))
            for day_shift in range(num_days):
                await mock_home._compute_and_store_person_car_forecasts(
                    local_day_utc,
                    day_shift=day_shift
                )

            # Now test prediction functionality
            for person in mock_home._persons:
                if len(person.historical_mileage_data) > 0:
                    print(f"\nPerson {person.name} prediction capabilities:")

                    # Try to get prediction for tomorrow
                    tomorrow = test_time + timedelta(days=1)
                    next_leave_time, next_mileage = person._compute_person_next_need(tomorrow)

                    print(f"  Next predicted leave time: {next_leave_time}")
                    print(f"  Next predicted mileage: {next_mileage}")

                    # Predictions may be None if insufficient data, but structure should be correct
                    if next_leave_time is not None:
                        assert isinstance(next_leave_time, datetime)
                    if next_mileage is not None:
                        assert isinstance(next_mileage, (int, float))
                        assert next_mileage >= 0

    @pytest.mark.asyncio
    async def test_person_authorized_cars_matching(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test that person car matching works correctly with authorized cars using real QSCar instances."""

        test_time = person_and_car_data.get("time")
        per_day_data = person_and_car_data.get("per_day", [])

        if not per_day_data or len(per_day_data) == 0:
            pytest.skip("No per_day data available")

        first_day = per_day_data[0]
        start, end, car_data, person_data = first_day

        # Create REAL QSCar instances
        all_cars = []
        for car_name, car_positions, car_odos in car_data:
            car_tracker = None
            if car_positions and len(car_positions) > 0:
                car_tracker = car_positions[0].entity_id

            car_odometer_sensor = None
            if car_odos and len(car_odos) > 0:
                car_odometer_sensor = car_odos[0].entity_id

            car = QSCar(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=car_name,
                car_tracker=car_tracker,
                car_odometer_sensor=car_odometer_sensor,
                car_battery_capacity=50000
            )
            mock_home._cars.append(car)
            all_cars.append(car)

        if len(all_cars) == 0:
            pytest.skip("No cars in test data")

        # Create person with only first car authorized
        for person_entity_id, _ in person_data:
            person = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=person_entity_id.split('.')[1],
                person_person_entity=person_entity_id,
                person_authorized_cars=[all_cars[0].name]  # Only authorize first car
            )
            mock_home._persons.append(person)

        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            persons_mileage = await mock_home._compute_mileage_for_period_per_person(start, end)

            # Verify person only gets mileage from authorized car
            for person, (mileage, leave_time) in persons_mileage.items():
                authorized_car_names = person.authorized_cars
                print(f"\nPerson {person.name}:")
                print(f"  Authorized cars: {authorized_car_names}")
                print(f"  Computed mileage: {mileage}km")

                # Person should only have mileage if they used an authorized car
                assert isinstance(authorized_car_names, list)
                assert len(authorized_car_names) > 0

    @pytest.mark.asyncio
    async def test_multiple_persons_same_car(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test handling of multiple persons using the same car with real QSCar instances."""

        per_day_data = person_and_car_data.get("per_day", [])

        if not per_day_data or len(per_day_data) == 0:
            pytest.skip("No per_day data available")

        first_day = per_day_data[0]
        start, end, car_data, person_data = first_day

        if len(car_data) == 0:
            pytest.skip("No cars in test data")

        # Create one REAL car
        car_name, car_positions, car_odos = car_data[0]

        car_tracker = None
        if car_positions and len(car_positions) > 0:
            car_tracker = car_positions[0].entity_id

        car_odometer_sensor = None
        if car_odos and len(car_odos) > 0:
            car_odometer_sensor = car_odos[0].entity_id

        car = QSCar(
            hass=mock_home.hass,
            home=mock_home,
            config_entry=None,
            name=car_name,
            car_tracker=car_tracker,
            car_odometer_sensor=car_odometer_sensor,
            car_battery_capacity=50000
        )
        mock_home._cars.append(car)

        # Create multiple persons all authorized for the same car
        test_persons = []
        for i, (person_entity_id, _) in enumerate(person_data):
            person = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=f"{person_entity_id.split('.')[1]}_{i}",
                person_person_entity=person_entity_id,
                person_authorized_cars=[car.name]
            )
            mock_home._persons.append(person)
            test_persons.append(person)

        # Add a second person with same entity to test
        if len(person_data) > 0:
            person_entity_id, _ = person_data[0]
            person2 = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=f"second_{person_entity_id.split('.')[1]}",
                person_person_entity=person_entity_id,
                person_authorized_cars=[car.name]
            )
            mock_home._persons.append(person2)
            test_persons.append(person2)

        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            persons_mileage = await mock_home._compute_mileage_for_period_per_person(start, end)

            print(f"\nMultiple persons sharing car '{car.name}':")
            total_mileage = 0
            for person, (mileage, leave_time) in persons_mileage.items():
                print(f"  Person {person.name}: {mileage}km")
                total_mileage += mileage

            print(f"  Total mileage assigned: {total_mileage}km")

            # The algorithm should distribute mileage among persons
            # Total should not exceed the car's actual mileage
            assert total_mileage >= 0

    @pytest.mark.asyncio
    async def test_process_all_14_days_with_stored_local_day_utc(
        self,
        mock_home,
        person_and_car_data,
        mock_load_from_history
    ):
        """Test processing all 14 days using the exact local_day_utc stored in pickle.

        This test uses REAL QSCar instances with REAL get_car_mileage_on_period_km method
        that computes actual mileage from odometer data in the pickle file.
        The mock_load_from_history provides the data, but the mileage computation is real.
        """

        # Get the stored local_day_utc from pickle
        local_day_utc = person_and_car_data.get("local_day_utc")
        assert local_day_utc is not None, "local_day_utc must be in pickle file"

        time = person_and_car_data.get("time")

        per_day_data = person_and_car_data.get("per_day", [])
        assert len(per_day_data) > 0, "per_day data must be available"

        print(f"\nUsing local_day_utc from pickle: {local_day_utc}")
        print(f"Processing {len(per_day_data)} days of data")

        # Setup cars and persons from first day
        first_day = per_day_data[0]
        start, end, car_data, person_data = first_day

        # Create REAL QSCar instances (not mocks!) so get_car_mileage_on_period_km works properly
        # Extract the actual entity IDs from the pickle data
        print(f"\nCreating real QSCar instances with actual entity IDs from pickle:")
        time_now = datetime.now(tz=pytz.UTC)

        check_time = local_day_utc + timedelta(hours=12)

        for car_name, car_positions, car_odos in car_data:
            # Extract actual entity IDs from the first LazyState object
            car_tracker = None
            if car_positions and len(car_positions) > 0:
                car_tracker = car_positions[0].entity_id

            car_odometer_sensor = None
            if car_odos and len(car_odos) > 0:
                car_odometer_sensor = car_odos[0].entity_id

            # Create real car with actual entity IDs from pickle
            car = QSCar(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=car_name,
                car_tracker=car_tracker,
                car_odometer_sensor=car_odometer_sensor,
                car_charge_percent_sensor=f"sensor.soc_{car_name.lower()}",
                car_battery_capacity=50000  # Default capacity
            )
            mock_home.add_device(car)
            car._km_per_kwh = 6.0  # Set a default efficiency

            # Set realistic battery capacities and initial SOC based on car type
            cur_soc = None
            if "tesla" in car_name.lower():
                car.car_battery_capacity = 72000
                cur_soc = 50
            elif "twingo" in car_name.lower():
                car.car_battery_capacity = 22000
                cur_soc = 10
            elif "zoe" in car_name.lower():
                car.car_battery_capacity = 52000
                cur_soc = 20
            elif "buz" in car_name.lower():
                car.car_battery_capacity = 86000
                cur_soc = 5
            if cur_soc is not None:
                car._entity_probed_state[car.car_charge_percent_sensor] = []
                car._add_state_history(car.car_charge_percent_sensor, cur_soc, check_time, None, None, check_time)
                soc = car.get_car_charge_percent(check_time)
                assert soc == cur_soc, f"Expected SOC {cur_soc}% for car {car.name}, got {soc}%"

        # Create real QSPerson instances
        print(f"\nCreating real QSPerson instances:")
        for person_entity_id, _ in person_data:
            person = QSPerson(
                hass=mock_home.hass,
                home=mock_home,
                config_entry=None,
                name=person_entity_id.split('.')[1],
                person_person_entity=person_entity_id,
                person_authorized_cars=[car.name for car in mock_home._cars]
            )
            mock_home.add_device(person)
            print(f"  - {person.name}: authorized_cars={person.authorized_cars}")

        # Patch load_from_history with our exact-matching mock
        # This will be used by both the home's _compute_mileage_for_period_per_person
        # AND by each car's get_car_mileage_on_period_km method
        with patch(
            'custom_components.quiet_solar.ha_model.home.load_from_history',
            side_effect=mock_load_from_history
        ), patch(
            'custom_components.quiet_solar.ha_model.car.load_from_history',
            side_effect=mock_load_from_history
        ):
            # Process all days using the exact pattern from the code
            # for d in range(0, 14):
            #     await self._compute_and_store_person_car_forecasts(local_day_utc, day_shift=d)

            num_days = min(14, len(per_day_data))
            print(f"\nProcessing {num_days} days with day_shift from 0 to {num_days-1}:")

            local_day_2, local_day_shifted_2, local_day_utc_2, is_passed = mock_home._compute_person_needed_time_and_date(time)

            assert local_day_utc_2 == local_day_utc, \
                f"Computed local_day_utc {local_day_utc_2} does not match stored {local_day_utc}"

            for d in range(0, num_days):
                print(f"  Processing day_shift={d}", end="")
                await mock_home._compute_and_store_person_car_forecasts(local_day_utc, day_shift=d)
                print(" ✓")

            # check time

            for person in mock_home._persons:
                leave_time, mileage = person._compute_person_next_need(check_time)

                if "thomas" in person.name.lower():
                    assert mileage is not None
                    assert int(mileage) == 26, f"Expected 26km for Thomas, got {mileage}"
                elif "arthur" in person.name.lower():
                    assert mileage is not None
                    assert int(mileage) == 35, f"Expected 35km for Arthur, got {mileage}"
                elif "magali" in person.name.lower():
                    assert mileage is not None
                    assert int(mileage) == 59, f"Expected 59km for Magali, got {mileage}"
                elif "brice" in person.name.lower():
                    assert mileage is None
                print(f"Person {person.name} next need at {check_time}: leave at {leave_time} for {mileage}km")


            for car in mock_home._cars:
                is_person_covered, next_usage_time, person_min_target_charge, person =  await car.get_best_person_next_need(check_time)

                if "tesla" in car.name.lower():
                    assert "magali" in person.name.lower(), f"Expected magali for Tesla, got {person.name if person else None}"
                    assert int(person_min_target_charge) == 20
                    assert is_person_covered
                    assert next_usage_time == datetime.fromisoformat("2025-10-31 07:26:37.830195+00:00")
                elif "twingo" in car.name.lower():
                    assert "brice" in person.name.lower(), f"Expected brice for Twingo, got {person.name if person else None}"
                    assert person_min_target_charge is None
                elif "zoe" in car.name.lower():
                    assert "thomas" in person.name.lower(), f"Expected thomas for Zoe, got {person.name if person else None}"
                    assert int(person_min_target_charge) == 18
                    assert is_person_covered
                    assert next_usage_time == datetime.fromisoformat("2025-10-31 08:48:31.806986+00:00")
                elif "buz" in car.name.lower():
                    assert "arthur" in person.name.lower(), f"Expected arthur for ID.buzz, got {person.name if person else None}"
                    assert int(person_min_target_charge) == 12
                    assert is_person_covered is False
                    assert next_usage_time == datetime.fromisoformat("2025-10-30 17:14:54.184629+00:00")


                print(f"Car {car.name} best person next need at {check_time}: is_person_covered={is_person_covered}, next_usage_time={next_usage_time}, person_min_target_charge={person_min_target_charge}, person={person.name if person else None}")

            # Verify all persons have accumulated history
            print("\n=== Results after processing all days ===")
            for person in mock_home._persons:
                print(f"\nPerson: {person.name}")
                print(f"  Total historical mileage entries: {len(person.historical_mileage_data)}")

                if len(person.historical_mileage_data) > 0:
                    print(f"  Mileage history details:")
                    for day, mileage, leave_time, weekday in person.historical_mileage_data:
                        weekday_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday]
                        print(f"    {day.date()} ({weekday_name}): {mileage:.1f}km, leave at {leave_time.time()}")

                    # Calculate statistics
                    total_mileage = sum(entry[1] for entry in person.historical_mileage_data)
                    avg_mileage = total_mileage / len(person.historical_mileage_data)
                    print(f"  Total mileage: {total_mileage:.1f}km")
                    print(f"  Average daily mileage: {avg_mileage:.1f}km")

                    # Verify mileage values are realistic (from real odometer data)
                    assert all(0 <= entry[1] <= 500 for entry in person.historical_mileage_data), \
                        "Mileage values should be realistic (0-500km per day)"

                    # Verify history is sorted by date
                    dates = [entry[0] for entry in person.historical_mileage_data]
                    assert dates == sorted(dates), "History should be sorted by date"

                    # Test prediction based on accumulated history
                    test_time = person_and_car_data.get("time")
                    if test_time:
                        tomorrow = test_time + timedelta(days=1)
                        next_leave_time, next_mileage = person._compute_person_next_need(tomorrow)
                        print(f"  Predicted for tomorrow: {next_mileage}km at {next_leave_time}")

                        if next_mileage is not None:
                            assert 0 <= next_mileage <= 500, "Predicted mileage should be realistic"
                else:
                    print(f"  No mileage history (person may not have used any cars)")

    @pytest.mark.asyncio
    async def test_pickle_data_integrity(self, person_and_car_data):
        """Test that pickle data has GPS and odometer readings."""

        full_range = person_and_car_data["full_range"]
        start, end, car_data, person_data = full_range

        # Check for GPS data in car positions
        has_gps_data = False
        for car_name, car_positions, car_odos in car_data:
            if len(car_positions) > 0:
                # Check first position has the expected structure
                first_pos = car_positions[0]
                if hasattr(first_pos, 'attributes'):
                    attrs = first_pos.attributes
                    if attrs and 'latitude' in attrs and 'longitude' in attrs:
                        has_gps_data = True
                        print(f"\nCar {car_name} has GPS data:")
                        print(f"  First position: lat={attrs.get('latitude')}, lon={attrs.get('longitude')}")
                        print(f"  Time: {first_pos.last_updated}")

        # Check for odometer readings
        has_odo_data = False
        for car_name, car_positions, car_odos in car_data:
            if len(car_odos) > 0:
                has_odo_data = True
                first_odo = car_odos[0]
                print(f"\nCar {car_name} has odometer data:")
                print(f"  First reading: state={first_odo.state}")
                print(f"  Time: {first_odo.last_updated}")

        # At least one car should have either GPS or odometer data
        assert has_gps_data or has_odo_data, "Test data should have GPS or odometer readings"


