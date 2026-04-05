"""Tests for bug #48: charger person-constraint / best-effort oscillation.

Verifies fixes for:
- Task 1: realized_charge_target set when existing ASAP constraint found
- Task 2: Person ASAP constraint target preserved (not overwritten by car target)
- Task 4: push_live_constraint duplicate detection uses initial_end_of_constraint
- Task 5: Normal charging flow regression tests
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from custom_components.quiet_solar.const import (
    CONF_CHARGER_CONSUMPTION,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGeneric,
)
from custom_components.quiet_solar.home_model.commands import LoadCommand
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraintChargePercent,
)
from tests.factories import (
    MinimalTestLoad,
    create_charge_percent_constraint,
    create_minimal_home_model,
)

# =============================================================================
# Test Helpers
# =============================================================================


def create_mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.data = {DOMAIN: {DATA_HANDLER: MagicMock()}}
    return hass


def create_mock_home(hass):
    """Create a mock QSHome instance."""
    home = create_minimal_home_model()
    home.hass = hass
    home.battery = None
    home.is_off_grid = MagicMock(return_value=False)
    home.get_available_power_values = MagicMock(return_value=None)
    home.get_grid_consumption_power_values = MagicMock(return_value=None)
    home.battery_can_discharge = MagicMock(return_value=True)
    home.get_tariff = MagicMock(return_value=0.15)
    home.get_best_tariff = MagicMock(return_value=0.10)
    home.force_next_solve = MagicMock()
    home.get_car_by_name = MagicMock(return_value=None)
    return home


def create_charger_generic(hass, home, name="TestCharger", **extra_config):
    """Create a QSChargerGeneric instance for testing."""
    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    config = {
        "name": name,
        "hass": hass,
        "home": home,
        "config_entry": config_entry,
        CONF_CHARGER_MIN_CHARGE: 6,
        CONF_CHARGER_MAX_CHARGE: 32,
        CONF_CHARGER_CONSUMPTION: 70,
        CONF_IS_3P: True,
        CONF_MONO_PHASE: 1,
        CONF_CHARGER_STATUS_SENSOR: f"sensor.{name}_status",
        CONF_CHARGER_PLUGGED: f"sensor.{name}_plugged",
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: f"number.{name}_max_current",
    }
    config.update(extra_config)

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry"):
        charger = QSChargerGeneric(**config)

    return charger


def create_mock_car(
    name="TestCar",
    target_charge=80.0,
    car_default_charge=100.0,
    car_battery_capacity=50000.0,
    current_charge=70.0,
    min_ok_soc=20.0,
):
    """Create a mock car with configurable charge parameters."""
    mock_car = MagicMock()
    mock_car.name = name
    mock_car.car_battery_capacity = car_battery_capacity
    mock_car.car_default_charge = car_default_charge
    mock_car.efficiency_factor = 1.0
    mock_car.do_force_next_charge = False
    mock_car.do_next_charge_time = None
    mock_car.can_use_charge_percent_constraints.return_value = True
    mock_car.setup_car_charge_target_if_needed = AsyncMock(return_value=target_charge)
    mock_car.get_car_charge_percent.return_value = current_charge
    mock_car.get_car_target_SOC.return_value = target_charge
    mock_car.get_car_minimum_ok_SOC.return_value = min_ok_soc
    mock_car.get_next_scheduled_event = AsyncMock(return_value=(None, None))
    mock_car.get_best_person_next_need = AsyncMock(return_value=(None, None, None, None))
    mock_car.set_next_charge_target_percent = AsyncMock()

    # Default: no user-originated values
    user_values = {}

    def has_user_originated(key):
        return key in user_values

    def get_user_originated(key):
        return user_values.get(key)

    def set_user_originated(key, value):
        user_values[key] = value

    def clear_user_originated(key):
        user_values.pop(key, None)

    def clear_all_user_originated():
        user_values.clear()

    mock_car.has_user_originated = MagicMock(side_effect=has_user_originated)
    mock_car.get_user_originated = MagicMock(side_effect=get_user_originated)
    mock_car.set_user_originated = MagicMock(side_effect=set_user_originated)
    mock_car.clear_user_originated = MagicMock(side_effect=clear_user_originated)
    mock_car.clear_all_user_originated = MagicMock(side_effect=clear_all_user_originated)

    return mock_car


def setup_charger_with_plugged_car(charger, mock_car, time):
    """Set up charger to be plugged in with a car, past boot time."""
    charger.car = mock_car
    charger._boot_time = None
    charger._boot_time_adjusted = None
    charger._power_steps = [LoadCommand(command="on", power_consign=7000.0)]


# =============================================================================
# Task 1: realized_charge_target set when existing ASAP constraint found
# =============================================================================


class TestAsapConstraintRealizedChargeTarget(unittest.IsolatedAsyncioTestCase):
    """Test that finding an existing ASAP constraint sets realized_charge_target."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)
        self.time = datetime(2026, 3, 27, 8, 0, 0, tzinfo=pytz.UTC)

    async def test_asap_constraint_prevents_unnecessary_best_effort(self):
        """When ASAP exists with target >= car_default_charge, no best-effort should be pushed."""
        mock_car = create_mock_car(
            target_charge=100.0,
            car_default_charge=100.0,
            current_charge=70.0,
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        # Create an existing ASAP constraint with target = 100%, current=20 so it stays active
        asap_constraint = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=50000.0,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=self.time - timedelta(minutes=30),
            load=self.charger,
            load_param=mock_car.name,
            from_user=False,
            initial_value=10.0,
            current_value=20.0,
            target_value=100.0,
            power_steps=self.charger._power_steps,
            support_auto=True,
        )
        self.charger._constraints = [asap_constraint]

        push_calls = []
        original_push = self.charger.push_live_constraint

        def tracking_push(time, constraint):
            push_calls.append(constraint)
            return original_push(time, constraint)

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", side_effect=tracking_push),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        # No best-effort (filler) constraint should be pushed when ASAP target >= car_default_charge
        filler_pushes = [
            c for c in push_calls if c._type in (CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN)
        ]
        assert filler_pushes == [], (
            f"Best-effort constraint should NOT be pushed when ASAP target >= car_default_charge, "
            f"but got {len(filler_pushes)} filler push(es)"
        )

    async def test_asap_constraint_allows_best_effort_topping_up(self):
        """When ASAP target < car_default_charge, best-effort SHOULD run to top up."""
        mock_car = create_mock_car(
            target_charge=80.0,
            car_default_charge=100.0,
            current_charge=70.0,
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        # Create an existing person ASAP constraint with target = 75%, current=20 so it stays active
        asap_constraint = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=50000.0,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=self.time - timedelta(minutes=30),
            load=self.charger,
            load_param=mock_car.name,
            load_info={"person": "Magali"},
            from_user=False,
            initial_value=10.0,
            current_value=20.0,
            target_value=75.0,
            power_steps=self.charger._power_steps,
            support_auto=True,
        )
        self.charger._constraints = [asap_constraint]

        push_calls = []
        original_push = self.charger.push_live_constraint

        def tracking_push(time, constraint):
            push_calls.append(constraint)
            return original_push(time, constraint)

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", side_effect=tracking_push),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        # Best-effort SHOULD be pushed to top up from 75% to 100%
        filler_pushes = [
            c for c in push_calls if c._type in (CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN)
        ]
        assert len(filler_pushes) > 0, "Best-effort should be pushed when ASAP target < car_default_charge"
        # The filler should start from the ASAP target (75%), not from car_initial_value (70%)
        filler = filler_pushes[0]
        assert filler.initial_value == pytest.approx(75.0), (
            f"Best-effort initial_value should be ASAP target (75%), got {filler.initial_value}"
        )


# =============================================================================
# Task 2: Person ASAP constraint target preservation
# =============================================================================


class TestPersonAsapTargetPreservation(unittest.IsolatedAsyncioTestCase):
    """Test that person-originated ASAP constraints keep their target."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)
        self.time = datetime(2026, 3, 27, 8, 0, 0, tzinfo=pytz.UTC)

    async def test_person_asap_target_not_overwritten_by_car_target(self):
        """Person ASAP at 75% should NOT be overwritten by car target of 80%."""
        mock_car = create_mock_car(
            target_charge=80.0,
            car_default_charge=100.0,
            current_charge=70.0,
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        # Create person ASAP constraint at 75%, current=20 so it stays active
        person_asap = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=50000.0,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=self.time - timedelta(minutes=30),
            load=self.charger,
            load_param=mock_car.name,
            load_info={"person": "Magali"},
            from_user=False,
            initial_value=10.0,
            current_value=20.0,
            target_value=75.0,
            power_steps=self.charger._power_steps,
            support_auto=True,
        )
        self.charger._constraints = [person_asap]

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", return_value=(False, False)),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        # Person constraint target should be preserved at 75%, NOT overwritten to 80%
        assert person_asap.target_value == pytest.approx(75.0), (
            f"Person ASAP target should remain 75%, but was overwritten to {person_asap.target_value}"
        )

    async def test_non_person_asap_target_updated_normally(self):
        """Non-person ASAP constraint should still have its target updated."""
        mock_car = create_mock_car(
            target_charge=80.0,
            car_default_charge=100.0,
            current_charge=70.0,
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        # Create non-person ASAP constraint at 75%, current=20 so it stays active
        user_asap = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=50000.0,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=self.time - timedelta(minutes=30),
            load=self.charger,
            load_param=mock_car.name,
            from_user=True,
            initial_value=10.0,
            current_value=20.0,
            target_value=75.0,
            power_steps=self.charger._power_steps,
            support_auto=True,
        )
        self.charger._constraints = [user_asap]

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", return_value=(False, False)),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        # Non-person ASAP target SHOULD be updated to car's target
        assert user_asap.target_value == pytest.approx(80.0), (
            f"Non-person ASAP target should be updated to 80%, but is {user_asap.target_value}"
        )

    async def test_person_asap_target_updated_when_user_overrides(self):
        """Person ASAP should be updated when user explicitly sets charge target."""
        mock_car = create_mock_car(
            target_charge=80.0,
            car_default_charge=100.0,
            current_charge=70.0,
        )
        # Simulate user explicitly setting charge_target_percent to 90%
        mock_car.has_user_originated.side_effect = lambda key: key == "charge_target_percent"
        mock_car.get_user_originated.side_effect = lambda key: 90.0 if key == "charge_target_percent" else None
        mock_car.setup_car_charge_target_if_needed = AsyncMock(return_value=90.0)

        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        # Create person ASAP constraint at 75%
        person_asap = MultiStepsPowerLoadConstraintChargePercent(
            total_capacity_wh=50000.0,
            type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            degraded_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=self.time - timedelta(minutes=30),
            load=self.charger,
            load_param=mock_car.name,
            load_info={"person": "Magali"},
            from_user=False,
            initial_value=10.0,
            current_value=20.0,
            target_value=75.0,
            power_steps=self.charger._power_steps,
            support_auto=True,
        )
        self.charger._constraints = [person_asap]

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", return_value=(False, False)),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        # User override should update even person ASAP target from 75% to 90%
        assert person_asap.target_value == pytest.approx(90.0), (
            f"Person ASAP target should be updated to 90% (user override), but is {person_asap.target_value}"
        )


# =============================================================================
# Task 4: push_live_constraint duplicate detection with initial_end_of_constraint
# =============================================================================


class TestPushLiveConstraintDuplicateDetection:
    """Test improved duplicate detection using initial_end_of_constraint."""

    def test_blocks_duplicate_when_initial_end_matches(self):
        """Completed constraint with extended end should block new constraint matching original end."""
        load = MinimalTestLoad(name="TestCharger")
        time = datetime(2026, 3, 27, 7, 30, 0, tzinfo=pytz.UTC)
        original_deadline = datetime(2026, 3, 27, 7, 30, 0, tzinfo=pytz.UTC)
        extended_deadline = datetime(2026, 3, 27, 7, 50, 0, tzinfo=pytz.UTC)

        # Simulate a completed constraint that was extended from 07:30 to 07:50
        completed = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time - timedelta(hours=8),
            end_of_constraint=original_deadline,
            initial_value=50.0,
            target_value=75.0,
            load_param="TestCar",
        )
        # Simulate the extension that happened during update_live_constraints
        completed.end_of_constraint = extended_deadline
        # initial_end_of_constraint should still be the original deadline
        assert completed.initial_end_of_constraint == original_deadline

        load._last_completed_constraint = completed

        # Try to push a new constraint with the original deadline
        new_constraint = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time,
            end_of_constraint=original_deadline,
            initial_value=50.0,
            target_value=75.0,
            load_param="TestCar",
        )

        result, _ = load.push_live_constraint(time, new_constraint)

        assert result is False, (
            "push_live_constraint should block duplicate when initial_end_of_constraint "
            "of completed constraint matches new constraint's end_of_constraint"
        )

    def test_allows_different_target_even_with_matching_initial_end(self):
        """Different target value should allow push even if initial_end matches."""
        load = MinimalTestLoad(name="TestCharger")
        time = datetime(2026, 3, 27, 7, 30, 0, tzinfo=pytz.UTC)
        original_deadline = datetime(2026, 3, 27, 7, 30, 0, tzinfo=pytz.UTC)
        extended_deadline = datetime(2026, 3, 27, 7, 50, 0, tzinfo=pytz.UTC)

        completed = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time - timedelta(hours=8),
            end_of_constraint=original_deadline,
            initial_value=50.0,
            target_value=75.0,
            load_param="TestCar",
        )
        completed.end_of_constraint = extended_deadline

        load._last_completed_constraint = completed

        # New constraint with DIFFERENT target
        new_constraint = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time,
            end_of_constraint=original_deadline,
            initial_value=50.0,
            target_value=90.0,
            load_param="TestCar",
        )

        result, _ = load.push_live_constraint(time, new_constraint)

        assert result is True, "Different target value should allow push even if initial_end matches"

    def test_existing_end_of_constraint_match_still_works(self):
        """Original duplicate detection (exact end_of_constraint match) should still work."""
        load = MinimalTestLoad(name="TestCharger")
        time = datetime(2026, 3, 27, 7, 30, 0, tzinfo=pytz.UTC)
        deadline = datetime(2026, 3, 27, 7, 50, 0, tzinfo=pytz.UTC)

        completed = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time - timedelta(hours=8),
            end_of_constraint=deadline,
            initial_value=50.0,
            target_value=75.0,
            load_param="TestCar",
        )

        load._last_completed_constraint = completed

        # Same end_of_constraint and same target
        new_constraint = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time,
            end_of_constraint=deadline,
            initial_value=50.0,
            target_value=75.0,
            load_param="TestCar",
        )

        result, _ = load.push_live_constraint(time, new_constraint)

        assert result is False, "Exact end_of_constraint match should still block duplicate"


# =============================================================================
# Task 5: Normal flow regression tests
# =============================================================================


class TestNormalChargingFlowRegression(unittest.IsolatedAsyncioTestCase):
    """Verify that normal charging scenarios still work correctly after fixes."""

    def setUp(self):
        self.hass = create_mock_hass()
        self.home = create_mock_home(self.hass)
        self.charger = create_charger_generic(self.hass, self.home)
        self.time = datetime(2026, 3, 27, 8, 0, 0, tzinfo=pytz.UTC)

    async def test_force_charge_still_creates_asap_constraint(self):
        """User force charge should still create ASAP constraint normally."""
        mock_car = create_mock_car(
            target_charge=80.0,
            car_default_charge=100.0,
            current_charge=50.0,
        )
        mock_car.do_force_next_charge = True
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)

        push_calls = []

        def tracking_push(time, constraint):
            push_calls.append(constraint)
            return True, False

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", side_effect=tracking_push),
            patch.object(self.charger, "constraint_reset_and_reset_commands_if_needed"),
        ):
            result = await self.charger.check_load_activity_and_constraints(self.time)

        # Force charge should push an ASAP constraint
        asap_pushes = [c for c in push_calls if c._type == CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE]
        assert len(asap_pushes) >= 1, "Force charge should create ASAP constraint"
        assert asap_pushes[0].target_value == pytest.approx(80.0)

    async def test_best_effort_created_when_no_constraint_exists(self):
        """Best-effort should be created when no active constraint exists."""
        mock_car = create_mock_car(
            target_charge=80.0,
            car_default_charge=100.0,
            current_charge=50.0,
        )
        setup_charger_with_plugged_car(self.charger, mock_car, self.time)
        self.charger._constraints = []

        push_calls = []

        def tracking_push(time, constraint):
            push_calls.append(constraint)
            return True, False

        with (
            patch.object(self.charger, "is_not_plugged", return_value=False),
            patch.object(self.charger, "is_plugged", return_value=True),
            patch.object(self.charger, "is_charger_unavailable", return_value=False),
            patch.object(self.charger, "probe_for_possible_needed_reboot", return_value=False),
            patch.object(self.charger, "get_best_car", return_value=mock_car),
            patch.object(
                self.charger, "clean_constraints_for_load_param_and_if_same_key_same_value_info", return_value=False
            ),
            patch.object(self.charger, "push_live_constraint", side_effect=tracking_push),
        ):
            await self.charger.check_load_activity_and_constraints(self.time)

        # Should have a filler/best-effort push
        filler_pushes = [
            c for c in push_calls if c._type in (CONSTRAINT_TYPE_FILLER, CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN)
        ]
        assert len(filler_pushes) > 0, "Best-effort should be created when no constraint exists"

    async def test_push_constraint_with_no_completed_still_works(self):
        """Pushing constraint with no last completed should always succeed."""
        load = MinimalTestLoad(name="TestCharger")
        time = datetime(2026, 3, 27, 8, 0, 0, tzinfo=pytz.UTC)
        load._last_completed_constraint = None

        constraint = create_charge_percent_constraint(
            load=load,
            constraint_type=CONSTRAINT_TYPE_MANDATORY_END_TIME,
            time=time,
            initial_value=50.0,
            target_value=75.0,
            load_param="TestCar",
        )

        result, _ = load.push_live_constraint(time, constraint)

        assert result is True, "Push should succeed when no last completed constraint exists"
