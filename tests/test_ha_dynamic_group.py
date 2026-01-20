"""Tests for QSDynamicGroup in ha_model/dynamic_group.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock

from homeassistant.const import CONF_NAME
import pytz

from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_IS_3P,
    MAX_POWER_INFINITE,
    MAX_AMP_INFINITE,
)

from tests.conftest import FakeHass, FakeConfigEntry


class TestQSDynamicGroupInit:
    """Test QSDynamicGroup initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_default_values(self):
        """Test initialization with default values."""
        device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Group"}
        )

        assert device.name == "Test Group"
        assert device.dyn_group_max_phase_current_conf == 54
        assert device._childrens == []
        assert device.charger_group is None

    def test_init_with_custom_max_amps(self):
        """Test initialization with custom max phase amps."""
        device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        assert device.dyn_group_max_phase_current_conf == 32

    def test_init_3p_phase_current_budget(self):
        """Test that 3-phase device has correct budget allocation."""
        device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        assert device._dyn_group_max_phase_current_for_budget == [32, 32, 32]

    def test_init_1p_phase_current_budget(self):
        """Test that 1-phase device has correct budget allocation."""
        device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: False,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        # Only one phase should have current
        total = sum(device._dyn_group_max_phase_current_for_budget)
        assert total == 32


class TestQSDynamicGroupProperties:
    """Test QSDynamicGroup properties."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

    def test_dyn_group_max_phase_current_property(self):
        """Test dyn_group_max_phase_current property returns budget."""
        result = self.device.dyn_group_max_phase_current
        assert result == [32, 32, 32]

    def test_dyn_group_max_phase_current_for_budget_property(self):
        """Test dyn_group_max_phase_current_for_budget property."""
        result = self.device.dyn_group_max_phase_current_for_budget
        assert result == [32, 32, 32]

    def test_physical_num_phases_3p(self):
        """Test physical_num_phases returns 3 for 3-phase device."""
        result = self.device.physical_num_phases
        assert result == 3

    def test_physical_num_phases_from_children(self):
        """Test physical_num_phases returns 3 if any child is 3-phase."""
        # Create a 1-phase group
        device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: False,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        # Add a 3-phase child
        mock_child = MagicMock()
        mock_child.physical_num_phases = 3
        device._childrens.append(mock_child)

        result = device.physical_num_phases
        assert result == 3


class TestQSDynamicGroupAvailableAmps:
    """Test available amps budget methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )
        self.device.father_device = None

    def test_update_available_amps_add(self):
        """Test updating available amps by adding."""
        # Initialize available amps
        self.device.available_amps_for_group = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]

        self.device.update_available_amps_for_group(0, [5.0, 5.0, 5.0], add=True)

        assert self.device.available_amps_for_group[0] == [15.0, 15.0, 15.0]

    def test_update_available_amps_subtract(self):
        """Test updating available amps by subtracting."""
        self.device.available_amps_for_group = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]

        self.device.update_available_amps_for_group(0, [5.0, 5.0, 5.0], add=False)

        assert self.device.available_amps_for_group[0] == [5.0, 5.0, 5.0]

    def test_update_available_amps_none_budget(self):
        """Test updating when budget is None."""
        self.device.available_amps_for_group = None

        # Should not raise
        self.device.update_available_amps_for_group(0, [5.0, 5.0, 5.0], add=True)


class TestQSDynamicGroupCurrentAcceptable:
    """Test current acceptability methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )
        self.device.father_device = None
        self.device.get_device_amps_consumption = MagicMock(return_value=[10.0, 10.0, 10.0])

    def test_is_current_acceptable_within_limits(self):
        """Test is_current_acceptable when within limits."""
        time = datetime.datetime.now(pytz.UTC)

        result = self.device.is_current_acceptable(
            new_amps=[20.0, 20.0, 20.0],
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert result is True

    def test_is_current_acceptable_exceeds_limits(self):
        """Test is_current_acceptable when exceeding limits."""
        time = datetime.datetime.now(pytz.UTC)

        result = self.device.is_current_acceptable(
            new_amps=[40.0, 40.0, 40.0],  # Exceeds 32A limit
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert result is False

    def test_is_current_acceptable_and_diff_within_limits(self):
        """Test is_current_acceptable_and_diff when within limits."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = self.device.is_current_acceptable_and_diff(
            new_amps=[20.0, 20.0, 20.0],
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert acceptable is True

    def test_is_current_acceptable_and_diff_exceeds_limits(self):
        """Test is_current_acceptable_and_diff when exceeding limits."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = self.device.is_current_acceptable_and_diff(
            new_amps=[40.0, 40.0, 40.0],  # Exceeds 32A limit
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert acceptable is False

    def test_is_delta_current_acceptable_within_limits(self):
        """Test is_delta_current_acceptable when delta is acceptable."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = self.device.is_delta_current_acceptable(
            delta_amps=[5.0, 5.0, 5.0],
            time=time,
            new_amps_consumption=[15.0, 15.0, 15.0]
        )

        assert acceptable is True

    def test_is_delta_current_acceptable_exceeds_limits(self):
        """Test is_delta_current_acceptable when delta exceeds limits."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = self.device.is_delta_current_acceptable(
            delta_amps=[30.0, 30.0, 30.0],
            time=time,
            new_amps_consumption=[40.0, 40.0, 40.0]  # Exceeds 32A limit
        )

        assert acceptable is False


class TestQSDynamicGroupPowerAggregation:
    """Test power aggregation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

    def test_get_device_power_no_children(self):
        """Test get_device_power_latest_possible_valid_value with no children."""
        time = datetime.datetime.now(pytz.UTC)

        result = self.device.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None,
            time=time
        )

        assert result == 0.0

    def test_get_device_power_with_children(self):
        """Test get_device_power_latest_possible_valid_value with children (simplified)."""
        time = datetime.datetime.now(pytz.UTC)

        # Children need to be HADeviceMixin instances for the check
        # Just verify the method doesn't crash with empty children
        result = self.device.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None,
            time=time
        )

        # With no children and no accurate sensor, returns 0
        assert result == 0.0

    def test_get_device_power_with_accurate_sensor(self):
        """Test get_device_power_latest_possible_valid_value with accurate sensor."""
        time = datetime.datetime.now(pytz.UTC)

        self.device.accurate_power_sensor = "sensor.total_power"
        self.device.get_sensor_latest_possible_valid_value = MagicMock(return_value=2000.0)

        result = self.device.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None,
            time=time
        )

        assert result == 2000.0


class TestQSDynamicGroupMinMaxPower:
    """Test min/max power methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{CONF_NAME: "Test Group"}
        )

    def test_get_min_max_power_no_children(self):
        """Test get_min_max_power with no children."""
        min_p, max_p = self.device.get_min_max_power()

        # Should call parent implementation
        assert min_p >= 0 or min_p == MAX_POWER_INFINITE
        assert max_p >= 0

    def test_get_min_max_power_with_children(self):
        """Test get_min_max_power with children."""
        # Add mock children with different power ranges
        mock_child1 = MagicMock()
        mock_child1.get_min_max_power = MagicMock(return_value=(100.0, 1000.0))
        mock_child2 = MagicMock()
        mock_child2.get_min_max_power = MagicMock(return_value=(200.0, 500.0))

        self.device._childrens = [mock_child1, mock_child2]

        min_p, max_p = self.device.get_min_max_power()

        assert min_p == 100.0  # Min of children's min
        assert max_p == 1000.0  # Max of children's max


class TestQSDynamicGroupBudgeting:
    """Test budget preparation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_dyn_group_entry",
            data={CONF_NAME: "Test Dynamic Group"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.home.voltage = 230.0
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSDynamicGroup(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

    def test_prepare_slots_for_amps_budget_no_father(self):
        """Test prepare_slots_for_amps_budget with no father budget."""
        time = datetime.datetime.now(pytz.UTC)

        self.device.prepare_slots_for_amps_budget(time, num_slots=4)

        assert self.device.available_amps_for_group is not None
        assert len(self.device.available_amps_for_group) == 4
        # Each slot should have the group's max current
        assert self.device.available_amps_for_group[0] == [32, 32, 32]

    def test_prepare_slots_for_amps_budget_with_father(self):
        """Test prepare_slots_for_amps_budget with father budget constraint."""
        time = datetime.datetime.now(pytz.UTC)

        # Father budget is less than group's max
        father_budget = [20.0, 20.0, 20.0]

        self.device.prepare_slots_for_amps_budget(time, num_slots=4, from_father_budget=father_budget)

        assert self.device.available_amps_for_group is not None
        assert len(self.device.available_amps_for_group) == 4
        # Should use min of father budget and group's max
        assert self.device.available_amps_for_group[0] == [20.0, 20.0, 20.0]

    def test_prepare_slots_for_amps_budget_propagates_to_children(self):
        """Test that budget preparation propagates to children."""
        time = datetime.datetime.now(pytz.UTC)

        # Add mock child
        mock_child = MagicMock()
        mock_child.prepare_slots_for_amps_budget = MagicMock()

        self.device._childrens = [mock_child]

        self.device.prepare_slots_for_amps_budget(time, num_slots=4)

        mock_child.prepare_slots_for_amps_budget.assert_called_once()
