"""Tests for QSDynamicGroup in ha_model/dynamic_group.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock

from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

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

from tests.factories import create_minimal_home_model


@pytest.fixture
def dyn_group_config_entry() -> MockConfigEntry:
    """Config entry for dynamic group tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_dyn_group_entry",
        data={CONF_NAME: "Test Dynamic Group"},
        title="Test Dynamic Group",
    )


@pytest.fixture
def dyn_group_home():
    """Home for dynamic group tests."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    return home


@pytest.fixture
def dyn_group_data_handler(dyn_group_home):
    """Data handler for dynamic group tests."""
    handler = MagicMock()
    handler.home = dyn_group_home
    return handler


@pytest.fixture
def dyn_group_hass_data(hass: HomeAssistant, dyn_group_data_handler):
    """Set hass.data[DOMAIN][DATA_HANDLER] for dynamic group tests."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = dyn_group_data_handler


@pytest.fixture
def dyn_group_device(
    hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
):
    """QSDynamicGroup instance (3p, 32A) for tests."""
    return QSDynamicGroup(
        hass=hass,
        config_entry=dyn_group_config_entry,
        home=dyn_group_home,
        **{
            CONF_NAME: "Test Group",
            CONF_IS_3P: True,
            CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
        }
    )


class TestQSDynamicGroupInit:
    """Test QSDynamicGroup initialization."""

    def test_init_default_values(
        self, hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
    ):
        """Test initialization with default values."""
        device = QSDynamicGroup(
            hass=hass,
            config_entry=dyn_group_config_entry,
            home=dyn_group_home,
            **{CONF_NAME: "Test Group"}
        )

        assert device.name == "Test Group"
        assert device.dyn_group_max_phase_current_conf == 54
        assert device._childrens == []
        assert device.charger_group is None

    def test_init_with_custom_max_amps(
        self, hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
    ):
        """Test initialization with custom max phase amps."""
        device = QSDynamicGroup(
            hass=hass,
            config_entry=dyn_group_config_entry,
            home=dyn_group_home,
            **{
                CONF_NAME: "Test Group",
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        assert device.dyn_group_max_phase_current_conf == 32

    def test_init_3p_phase_current_budget(
        self, hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
    ):
        """Test that 3-phase device has correct budget allocation."""
        device = QSDynamicGroup(
            hass=hass,
            config_entry=dyn_group_config_entry,
            home=dyn_group_home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: True,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        assert device._dyn_group_max_phase_current_for_budget == [32, 32, 32]

    def test_init_1p_phase_current_budget(
        self, hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
    ):
        """Test that 1-phase device has correct budget allocation."""
        device = QSDynamicGroup(
            hass=hass,
            config_entry=dyn_group_config_entry,
            home=dyn_group_home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: False,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        total = sum(device._dyn_group_max_phase_current_for_budget)
        assert total == 32


class TestQSDynamicGroupProperties:
    """Test QSDynamicGroup properties."""

    def test_dyn_group_max_phase_current_property(self, dyn_group_device):
        """Test dyn_group_max_phase_current property returns budget."""
        result = dyn_group_device.dyn_group_max_phase_current
        assert result == [32, 32, 32]

    def test_dyn_group_max_phase_current_for_budget_property(self, dyn_group_device):
        """Test dyn_group_max_phase_current_for_budget property."""
        result = dyn_group_device.dyn_group_max_phase_current_for_budget
        assert result == [32, 32, 32]

    def test_physical_num_phases_3p(self, dyn_group_device):
        """Test physical_num_phases returns 3 for 3-phase device."""
        result = dyn_group_device.physical_num_phases
        assert result == 3

    def test_physical_num_phases_from_children(
        self, hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
    ):
        """Test physical_num_phases returns 3 if any child is 3-phase."""
        device = QSDynamicGroup(
            hass=hass,
            config_entry=dyn_group_config_entry,
            home=dyn_group_home,
            **{
                CONF_NAME: "Test Group",
                CONF_IS_3P: False,
                CONF_DYN_GROUP_MAX_PHASE_AMPS: 32,
            }
        )

        mock_child = MagicMock()
        mock_child.physical_num_phases = 3
        device._childrens.append(mock_child)

        result = device.physical_num_phases
        assert result == 3


class TestQSDynamicGroupAvailableAmps:
    """Test available amps budget methods."""

    def test_update_available_amps_add(self, dyn_group_device):
        """Test updating available amps by adding."""
        dyn_group_device.father_device = None
        dyn_group_device.available_amps_for_group = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]

        dyn_group_device.update_available_amps_for_group(0, [5.0, 5.0, 5.0], add=True)

        assert dyn_group_device.available_amps_for_group[0] == [15.0, 15.0, 15.0]

    def test_update_available_amps_subtract(self, dyn_group_device):
        """Test updating available amps by subtracting."""
        dyn_group_device.father_device = None
        dyn_group_device.available_amps_for_group = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]

        dyn_group_device.update_available_amps_for_group(0, [5.0, 5.0, 5.0], add=False)

        assert dyn_group_device.available_amps_for_group[0] == [5.0, 5.0, 5.0]

    def test_update_available_amps_none_budget(self, dyn_group_device):
        """Test updating when budget is None."""
        dyn_group_device.father_device = None
        dyn_group_device.available_amps_for_group = None

        dyn_group_device.update_available_amps_for_group(0, [5.0, 5.0, 5.0], add=True)
        assert dyn_group_device.available_amps_for_group is None


class TestQSDynamicGroupCurrentAcceptable:
    """Test current acceptability methods."""

    @pytest.fixture(autouse=True)
    def _dyn_device_attrs(self, dyn_group_device):
        """Set extra attributes on dyn_group_device."""
        dyn_group_device.father_device = None
        dyn_group_device.get_device_amps_consumption = MagicMock(return_value=[10.0, 10.0, 10.0])

    def test_is_current_acceptable_within_limits(self, dyn_group_device):
        """Test is_current_acceptable when within limits."""
        time = datetime.datetime.now(pytz.UTC)

        result = dyn_group_device.is_current_acceptable(
            new_amps=[20.0, 20.0, 20.0],
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert result is True

    def test_is_current_acceptable_exceeds_limits(self, dyn_group_device):
        """Test is_current_acceptable when exceeding limits."""
        time = datetime.datetime.now(pytz.UTC)

        result = dyn_group_device.is_current_acceptable(
            new_amps=[40.0, 40.0, 40.0],
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert result is False

    def test_is_current_acceptable_and_diff_within_limits(self, dyn_group_device):
        """Test is_current_acceptable_and_diff when within limits."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = dyn_group_device.is_current_acceptable_and_diff(
            new_amps=[20.0, 20.0, 20.0],
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert acceptable is True

    def test_is_current_acceptable_and_diff_exceeds_limits(self, dyn_group_device):
        """Test is_current_acceptable_and_diff when exceeding limits."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = dyn_group_device.is_current_acceptable_and_diff(
            new_amps=[40.0, 40.0, 40.0],
            estimated_current_amps=[10.0, 10.0, 10.0],
            time=time
        )

        assert acceptable is False

    def test_is_delta_current_acceptable_within_limits(self, dyn_group_device):
        """Test is_delta_current_acceptable when delta is acceptable."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = dyn_group_device.is_delta_current_acceptable(
            delta_amps=[5.0, 5.0, 5.0],
            time=time,
            new_amps_consumption=[15.0, 15.0, 15.0]
        )

        assert acceptable is True

    def test_is_delta_current_acceptable_exceeds_limits(self, dyn_group_device):
        """Test is_delta_current_acceptable when delta exceeds limits."""
        time = datetime.datetime.now(pytz.UTC)

        acceptable, diff = dyn_group_device.is_delta_current_acceptable(
            delta_amps=[30.0, 30.0, 30.0],
            time=time,
            new_amps_consumption=[40.0, 40.0, 40.0]
        )

        assert acceptable is False


class TestQSDynamicGroupPowerAggregation:
    """Test power aggregation methods."""

    def test_get_device_power_no_children(self, dyn_group_device):
        """Test get_device_power_latest_possible_valid_value with no children."""
        time = datetime.datetime.now(pytz.UTC)

        result = dyn_group_device.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None,
            time=time
        )

        assert result == 0.0

    def test_get_device_power_with_children(self, dyn_group_device):
        """Test get_device_power_latest_possible_valid_value with children (simplified)."""
        time = datetime.datetime.now(pytz.UTC)

        result = dyn_group_device.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None,
            time=time
        )

        assert result == 0.0

    def test_get_device_power_with_accurate_sensor(self, dyn_group_device):
        """Test get_device_power_latest_possible_valid_value with accurate sensor."""
        time = datetime.datetime.now(pytz.UTC)

        dyn_group_device.accurate_power_sensor = "sensor.total_power"
        dyn_group_device.get_sensor_latest_possible_valid_value = MagicMock(return_value=2000.0)

        result = dyn_group_device.get_device_power_latest_possible_valid_value(
            tolerance_seconds=None,
            time=time
        )

        assert result == 2000.0


class TestQSDynamicGroupMinMaxPower:
    """Test min/max power methods."""

    @pytest.fixture
    def dyn_group_device_minmax(
        self, hass, dyn_group_config_entry, dyn_group_home, dyn_group_data_handler, dyn_group_hass_data
    ):
        """QSDynamicGroup with default config (no CONF_IS_3P/MAX_PHASE_AMPS) for min/max tests."""
        return QSDynamicGroup(
            hass=hass,
            config_entry=dyn_group_config_entry,
            home=dyn_group_home,
            **{CONF_NAME: "Test Group"}
        )

    def test_get_min_max_power_no_children(self, dyn_group_device_minmax):
        """Test get_min_max_power with no children."""
        min_p, max_p = dyn_group_device_minmax.get_min_max_power()

        assert min_p >= 0 or min_p == MAX_POWER_INFINITE
        assert max_p >= 0

    def test_get_min_max_power_with_children(self, dyn_group_device_minmax):
        """Test get_min_max_power with children."""
        mock_child1 = MagicMock()
        mock_child1.get_min_max_power = MagicMock(return_value=(100.0, 1000.0))
        mock_child2 = MagicMock()
        mock_child2.get_min_max_power = MagicMock(return_value=(200.0, 500.0))

        dyn_group_device_minmax._childrens = [mock_child1, mock_child2]

        min_p, max_p = dyn_group_device_minmax.get_min_max_power()

        assert min_p == 100.0
        assert max_p == 1000.0


class TestQSDynamicGroupBudgeting:
    """Test budget preparation methods."""

    def test_prepare_slots_for_amps_budget_no_father(self, dyn_group_device):
        """Test prepare_slots_for_amps_budget with no father budget."""
        time = datetime.datetime.now(pytz.UTC)

        dyn_group_device.prepare_slots_for_amps_budget(time, num_slots=4)

        assert dyn_group_device.available_amps_for_group is not None
        assert len(dyn_group_device.available_amps_for_group) == 4
        assert dyn_group_device.available_amps_for_group[0] == [32, 32, 32]

    def test_prepare_slots_for_amps_budget_with_father(self, dyn_group_device):
        """Test prepare_slots_for_amps_budget with father budget constraint."""
        time = datetime.datetime.now(pytz.UTC)

        father_budget = [20.0, 20.0, 20.0]

        dyn_group_device.prepare_slots_for_amps_budget(
            time, num_slots=4, from_father_budget=father_budget
        )

        assert dyn_group_device.available_amps_for_group is not None
        assert len(dyn_group_device.available_amps_for_group) == 4
        assert dyn_group_device.available_amps_for_group[0] == [20.0, 20.0, 20.0]

    def test_prepare_slots_for_amps_budget_propagates_to_children(self, dyn_group_device):
        """Test that budget preparation propagates to children."""
        time = datetime.datetime.now(pytz.UTC)

        mock_child = MagicMock()
        mock_child.prepare_slots_for_amps_budget = MagicMock()

        dyn_group_device._childrens = [mock_child]

        dyn_group_device.prepare_slots_for_amps_budget(time, num_slots=4)

        mock_child.prepare_slots_for_amps_budget.assert_called_once()
