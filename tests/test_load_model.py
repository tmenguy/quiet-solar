"""Tests for home_model/load.py - Load device functionality."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytz

from custom_components.quiet_solar.home_model.load import (
    AbstractDevice,
    AbstractLoad,
    extract_name_and_index_from_dashboard_section_option,
    map_section_selected_name_in_section_list,
)
from custom_components.quiet_solar.home_model.commands import LoadCommand, CMD_OFF, CMD_ON, CMD_IDLE
from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_DEVICE_EFFICIENCY,
    CONF_IS_3P,
    DASHBOARD_NO_SECTION,
    CONF_NUM_MAX_ON_OFF,
)


# =============================================================================
# Test Helper Functions
# =============================================================================

def test_extract_name_and_index_simple_name():
    """Test extracting name without index."""
    name, idx = extract_name_and_index_from_dashboard_section_option("Section A")
    assert name == "Section A"
    assert idx is None


def test_extract_name_and_index_with_index():
    """Test extracting name with index prefix."""
    name, idx = extract_name_and_index_from_dashboard_section_option("#1 - Section A")
    assert name == "Section A"
    assert idx == 0


def test_extract_name_and_index_high_index():
    """Test extracting name with high index."""
    name, idx = extract_name_and_index_from_dashboard_section_option("#5 - Another Section")
    assert name == "Another Section"
    assert idx == 4


def test_extract_name_and_index_invalid_index():
    """Test extracting name with invalid index format."""
    name, idx = extract_name_and_index_from_dashboard_section_option("#abc - Section")
    assert name == "#abc - Section"
    assert idx is None


def test_extract_name_and_index_zero_index():
    """Test extracting name with index 0."""
    name, idx = extract_name_and_index_from_dashboard_section_option("#0 - Section")
    assert idx is None


def test_map_section_no_section():
    """Test mapping when section is NO_SECTION."""
    section_list = [("Section A", "type_a"), ("Section B", "type_b")]
    result_idx, options = map_section_selected_name_in_section_list(DASHBOARD_NO_SECTION, section_list)

    assert result_idx is None


def test_map_section_compute_options():
    """Test mapping with compute_options=True."""
    section_list = [("Section A", "type_a"), ("Section B", "type_b")]
    result_idx, options = map_section_selected_name_in_section_list(
        DASHBOARD_NO_SECTION, section_list, compute_options=True
    )

    assert options is not None
    assert DASHBOARD_NO_SECTION in options
    assert "#1 - Section A" in options
    assert "#2 - Section B" in options


def test_map_section_by_name():
    """Test mapping finds section by name."""
    section_list = [("Section A", "type_a"), ("Section B", "type_b")]
    result_idx, _ = map_section_selected_name_in_section_list("Section B", section_list)

    assert result_idx == 1


def test_map_section_by_index():
    """Test mapping finds section by index."""
    section_list = [("Section A", "type_a"), ("Section B", "type_b")]
    result_idx, _ = map_section_selected_name_in_section_list("#2 - Unknown", section_list)

    assert result_idx == 1


def test_map_section_name_priority_over_index():
    """Test that name match takes priority over index."""
    section_list = [("Section A", "type_a"), ("Section B", "type_b")]
    result_idx, _ = map_section_selected_name_in_section_list("#1 - Section B", section_list)

    assert result_idx == 1


# =============================================================================
# Test AbstractDevice
# =============================================================================

class TestAbstractDevice:
    """Test AbstractDevice class."""

    def create_device(self, **kwargs):
        """Create a test device with default values."""
        defaults = {
            "name": "Test Device",
            "device_type": "test",
            CONF_POWER: 1000,
        }
        defaults.update(kwargs)

        mock_home = MagicMock()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractDevice(**defaults)

    def test_device_init(self):
        """Test device initialization."""
        device = self.create_device(name="My Device")

        assert device.name == "My Device"
        assert device.power_use == 1000
        assert device._enabled is True

    def test_device_id_generation(self):
        """Test device_id is generated from name."""
        device = self.create_device(name="Test Device")

        assert "test_device" in device.device_id.lower()
        assert device.device_id.startswith("qs_")

    def test_device_efficiency(self):
        """Test device efficiency setting."""
        device = self.create_device(**{CONF_DEVICE_EFFICIENCY: 95.0})

        assert device.efficiency == 95.0

    def test_device_efficiency_max_100(self):
        """Test device efficiency is capped at 100."""
        device = self.create_device(**{CONF_DEVICE_EFFICIENCY: 150.0})

        assert device.efficiency == 100.0

    def test_device_3phase_config(self):
        """Test 3-phase configuration."""
        device = self.create_device(**{CONF_IS_3P: True})

        assert device._device_is_3p_conf is True

    def test_device_voltage_from_home(self):
        """Test voltage comes from home."""
        device = self.create_device()

        assert device.voltage == 230.0

    def test_device_voltage_default_no_home(self):
        """Test default voltage when no home."""
        device = AbstractDevice(name="Test", device_type="test", home=None)

        assert device.voltage == 230.0

    def test_device_is_off_grid(self):
        """Test is_off_grid delegates to home."""
        device = self.create_device()
        device.home.is_off_grid.return_value = True

        assert device.is_off_grid() is True

    def test_device_num_max_on_off(self):
        """Test num_max_on_off configuration."""
        device = self.create_device(**{CONF_NUM_MAX_ON_OFF: 4})

        assert device.num_max_on_off == 4

    def test_device_num_max_on_off_odd_rounds_up(self):
        """Test odd num_max_on_off is rounded up to even."""
        device = self.create_device(**{CONF_NUM_MAX_ON_OFF: 3})

        assert device.num_max_on_off == 4

    def test_dashboard_section_no_home_sections(self):
        """Test dashboard_section returns None when home has no sections."""
        device = self.create_device()
        device.home.dashboard_sections = None

        assert device.dashboard_section is None

    def test_dashboard_section_with_sections(self):
        """Test dashboard_section returns computed section."""
        device = self.create_device()
        device.home.dashboard_sections = [("Section A", "type_a")]
        device._conf_dashboard_section_option = "Section A"

        section = device.dashboard_section
        assert section == "Section A"

    def test_dashboard_sort_string_in_type(self):
        """Test default sort string."""
        device = self.create_device()

        assert device.dashboard_sort_string_in_type == "ZZZ"

    def test_devices_to_pilot_empty(self):
        """Test devices_to_pilot is empty by default."""
        device = self.create_device()

        assert device.devices_to_pilot == []


# =============================================================================
# Test AbstractLoad
# =============================================================================

class TestAbstractLoad:
    """Test AbstractLoad class."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = MagicMock()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def test_load_init(self):
        """Test load initialization."""
        load = self.create_load(name="My Load")

        assert load.name == "My Load"
        assert load.power_use == 1500

    def test_load_get_min_max_power(self):
        """Test get_min_max_power returns power_use."""
        load = self.create_load()

        min_p, max_p = load.get_min_max_power()
        assert min_p == 1500
        assert max_p == 1500

    def test_load_constraint_storage(self):
        """Test load can store constraints."""
        load = self.create_load()

        assert hasattr(load, '_constraints')


# =============================================================================
# Test LoadCommand
# =============================================================================

def test_cmd_off_is_off():
    """Test CMD_OFF is considered off."""
    assert CMD_OFF.is_off_or_idle() is True


def test_cmd_on_is_not_off():
    """Test CMD_ON is not considered off."""
    assert CMD_ON.is_off_or_idle() is False


def test_cmd_idle_is_off():
    """Test CMD_IDLE is considered off/idle."""
    assert CMD_IDLE.is_off_or_idle() is True


def test_load_command_equality():
    """Test LoadCommand equality."""
    cmd1 = LoadCommand(command="test", power_consign=1000)
    cmd2 = LoadCommand(command="test", power_consign=1000)

    assert cmd1.command == cmd2.command
    assert cmd1.power_consign == cmd2.power_consign


def test_load_command_power_consign():
    """Test LoadCommand power_consign."""
    cmd = LoadCommand(command="on", power_consign=2000)

    assert cmd.power_consign == 2000
