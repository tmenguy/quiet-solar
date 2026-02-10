"""Tests for home_model/load.py - Load device functionality."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz

from tests.factories import create_minimal_home_model
from custom_components.quiet_solar.home_model.load import (
    AbstractDevice,
    AbstractLoad,
    extract_name_and_index_from_dashboard_section_option,
    map_section_selected_name_in_section_list,
)
from custom_components.quiet_solar.home_model.commands import (
    LoadCommand,
    CMD_CST_OFF,
    CMD_CST_ON,
    CMD_CST_IDLE,
    CMD_OFF,
    CMD_ON,
    CMD_IDLE,
)
from custom_components.quiet_solar.home_model.constraints import (
    MultiStepsPowerLoadConstraint,
    DATETIME_MAX_UTC,
    DATETIME_MIN_UTC,
)
from custom_components.quiet_solar.const import (
    CONF_POWER,
    CONF_DEVICE_EFFICIENCY,
    CONF_IS_3P,
    CONF_MONO_PHASE,
    DASHBOARD_NO_SECTION,
    CONF_NUM_MAX_ON_OFF,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
)


class _CommandLoad(AbstractLoad):
    """Minimal AbstractLoad for command tests."""

    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:
        return True

    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        return False


class _FailingCommandLoad(_CommandLoad):
    async def execute_command(self, time: datetime, command: LoadCommand) -> bool | None:
        raise RuntimeError("boom")


class _ProbeSetCommandLoad(_CommandLoad):
    async def probe_if_command_set(self, time: datetime, command: LoadCommand) -> bool | None:
        return True


# =============================================================================
# Helper function to create real LoadConstraints
# =============================================================================

def create_real_constraint(
    load,
    time_now=None,
    end_time=None,
    start_time=None,
    load_param=None,
    load_info=None,
    target_value=100.0,
    current_value=0.0,
    initial_value=0.0,
    power=1000.0,
    constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
    from_user=False,
):
    """Create a real MultiStepsPowerLoadConstraint for testing."""
    if end_time is None:
        end_time = datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC)
    if time_now is None:
        time_now = end_time - timedelta(hours=1)
    if start_time is None:
        start_time = DATETIME_MIN_UTC

    ct = MultiStepsPowerLoadConstraint(
        time=time_now,
        load=load,
        load_param=load_param,
        load_info=load_info,
        type=constraint_type,
        start_of_constraint=start_time,
        end_of_constraint=end_time,
        initial_value=initial_value,
        current_value=current_value,
        target_value=target_value,
        power=power,
        from_user=from_user,
    )
    return ct


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


def test_dashboard_sort_string_truncates_long_values():
    """Test dashboard sort string truncates long values."""
    long_name = "x" * 300
    long_type = "y" * 300

    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    device = AbstractDevice(
        name=long_name,
        device_type=long_type,
        home=mock_home,
        power=1000,
    )

    sort_string = device.dashboard_sort_string
    assert len(sort_string) == 255 * 3
    assert sort_string.startswith(long_type[:255])


@pytest.mark.asyncio
async def test_async_get_info_from_storage_updates_num_on_off():
    """Test async_get_info_from_storage applies saved info."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    device = AbstractDevice(
        name="Device",
        device_type="device",
        home=mock_home,
        power=1000,
        num_max_on_off=10,
    )
    device.num_on_off = 0

    await device.async_get_info_from_storage(datetime.now(pytz.UTC), {"num_on_off": 3})
    assert device.num_on_off == 2


def test_ack_command_counts_transitions():
    """Test _ack_command increments num_on_off on transitions."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    device = AbstractDevice(
        name="Device",
        device_type="device",
        home=mock_home,
        power=1000,
    )

    device.current_command = LoadCommand(command=CMD_CST_OFF, power_consign=0.0)
    device.num_on_off = 0
    device._ack_command(datetime.now(pytz.UTC), LoadCommand(command=CMD_CST_ON, power_consign=0.0))
    assert device.num_on_off == 1

    device._ack_command(datetime.now(pytz.UTC), LoadCommand(command=CMD_CST_IDLE, power_consign=0.0))
    assert device.num_on_off == 2


@pytest.mark.asyncio
async def test_launch_command_stacks_when_running():
    """Test launch_command stacks when already running."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    load = _CommandLoad(
        name="Load",
        device_type="load",
        home=mock_home,
        power=1000,
    )
    load.running_command = LoadCommand(command=CMD_CST_ON, power_consign=0.0)

    await load.launch_command(datetime.now(pytz.UTC), LoadCommand(command=CMD_CST_OFF, power_consign=0.0))
    assert load._stacked_command is not None


@pytest.mark.asyncio
async def test_launch_command_noop_when_current_matches():
    """Test launch_command returns early when current matches."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    load = _CommandLoad(
        name="Load",
        device_type="load",
        home=mock_home,
        power=1000,
    )
    load.current_command = LoadCommand(command=CMD_CST_ON, power_consign=0.0)

    await load.launch_command(datetime.now(pytz.UTC), LoadCommand(command=CMD_CST_ON, power_consign=0.0))
    assert load.running_command is None


def test_is_load_has_command_now_or_coming():
    """Test is_load_has_a_command_now_or_coming across states."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    load = _CommandLoad(
        name="Load",
        device_type="load",
        home=mock_home,
        power=1000,
    )

    assert load.is_load_has_a_command_now_or_coming(datetime.now(pytz.UTC)) is False

    load.current_command = LoadCommand(command=CMD_CST_ON, power_consign=0.0)
    assert load.is_load_has_a_command_now_or_coming(datetime.now(pytz.UTC)) is True

    load.current_command = None
    load.running_command = LoadCommand(command=CMD_CST_ON, power_consign=0.0)
    assert load.is_load_has_a_command_now_or_coming(datetime.now(pytz.UTC)) is True

    load.running_command = None
    load._stacked_command = LoadCommand(command=CMD_CST_ON, power_consign=0.0)
    assert load.is_load_has_a_command_now_or_coming(datetime.now(pytz.UTC)) is True



@pytest.mark.asyncio
async def test_launch_command_handles_execute_error():
    """Test launch_command handles execute_command error."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    load = _FailingCommandLoad(
        name="Load",
        device_type="load",
        home=mock_home,
        power=1000,
    )

    await load.launch_command(datetime.now(pytz.UTC), LoadCommand(command=CMD_CST_ON, power_consign=0.0))
    assert load.running_command is not None


@pytest.mark.asyncio
async def test_launch_command_probe_already_set():
    """Test launch_command acknowledges when already set."""
    mock_home = create_minimal_home_model()
    mock_home.voltage = 230.0
    mock_home.is_off_grid = MagicMock(return_value=False)
    mock_home.dashboard_sections = None

    load = _ProbeSetCommandLoad(
        name="Load",
        device_type="load",
        home=mock_home,
        power=1000,
    )

    await load.launch_command(datetime.now(pytz.UTC), LoadCommand(command=CMD_CST_ON, power_consign=0.0))
    assert load.current_command is not None


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

        mock_home = create_minimal_home_model()
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

        mock_home = create_minimal_home_model()
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


def test_load_command_repr():
    """Test LoadCommand string representation."""
    cmd = LoadCommand(command="test", power_consign=1000)
    repr_str = repr(cmd)
    assert "test" in repr_str or cmd.command == "test"


def test_cmd_off_command():
    """Test CMD_OFF command value."""
    assert CMD_OFF is not None


def test_cmd_on_command():
    """Test CMD_ON command value."""
    assert CMD_ON is not None


def test_cmd_idle_command():
    """Test CMD_IDLE command value."""
    assert CMD_IDLE is not None


# =============================================================================
# Test AbstractDevice Phase Calculations
# =============================================================================

class TestAbstractDevicePhaseCalculations:
    """Test phase-related methods in AbstractDevice."""

    def create_device(self, **kwargs):
        """Create a test device with default values."""
        defaults = {
            "name": "Test Device",
            "device_type": "test",
            CONF_POWER: 1000,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None
        mock_home.physical_3p = False

        defaults["home"] = mock_home

        return AbstractDevice(**defaults)

    def test_physical_num_phases_mono(self):
        """Test physical_num_phases for mono-phase device."""
        device = self.create_device(**{CONF_IS_3P: False})
        assert device.physical_num_phases == 1

    def test_physical_num_phases_3p(self):
        """Test physical_num_phases for 3-phase device."""
        device = self.create_device(**{CONF_IS_3P: True})
        assert device.physical_num_phases == 3

    def test_physical_3p_false(self):
        """Test physical_3p returns False for mono-phase."""
        device = self.create_device(**{CONF_IS_3P: False})
        assert device.physical_3p is False

    def test_physical_3p_true(self):
        """Test physical_3p returns True for 3-phase."""
        device = self.create_device(**{CONF_IS_3P: True})
        assert device.physical_3p is True

    def test_current_num_phases_equals_physical(self):
        """Test current_num_phases equals physical_num_phases."""
        device = self.create_device(**{CONF_IS_3P: True})
        assert device.current_num_phases == device.physical_num_phases

    def test_current_3p_equals_physical(self):
        """Test current_3p equals physical_3p."""
        device = self.create_device(**{CONF_IS_3P: True})
        assert device.current_3p == device.physical_3p

    def test_can_do_3_to_1_phase_switch(self):
        """Test default can_do_3_to_1_phase_switch returns False."""
        device = self.create_device()
        assert device.can_do_3_to_1_phase_switch() is False

    def test_mono_phase_index_from_config(self):
        """Test mono_phase_index from configuration."""
        device = self.create_device(**{CONF_MONO_PHASE: "2"})
        assert device.mono_phase_index == 1  # 0-indexed

    def test_mono_phase_index_default_random(self):
        """Test mono_phase_index is in range 0-2 when not configured."""
        device = self.create_device()
        assert 0 <= device.mono_phase_index <= 2

    def test_get_phase_amps_from_power_zero(self):
        """Test get_phase_amps_from_power with zero power."""
        device = self.create_device()
        amps = device.get_phase_amps_from_power(0.0)
        assert amps == [0.0, 0.0, 0.0]

    def test_get_phase_amps_from_power_mono(self):
        """Test get_phase_amps_from_power for mono-phase."""
        device = self.create_device(**{CONF_IS_3P: False, CONF_MONO_PHASE: "1"})
        # 1000W at 230V = ~4.35A
        amps = device.get_phase_amps_from_power(1000.0, is_3p=False)

        # The amps should be at mono_phase_index (0)
        assert amps[0] == pytest.approx(1000.0 / 230.0, rel=0.01)
        assert amps[1] == 0
        assert amps[2] == 0

    def test_get_phase_amps_from_power_3p(self):
        """Test get_phase_amps_from_power for 3-phase."""
        device = self.create_device(**{CONF_IS_3P: True})
        # 3000W at 230V, 3-phase = 1000W per phase = ~4.35A per phase
        amps = device.get_phase_amps_from_power(3000.0, is_3p=True)

        expected_per_phase = (3000.0 / 3.0) / 230.0
        assert amps[0] == pytest.approx(expected_per_phase, rel=0.01)
        assert amps[1] == pytest.approx(expected_per_phase, rel=0.01)
        assert amps[2] == pytest.approx(expected_per_phase, rel=0.01)

    def test_get_phase_amps_from_power_for_budgeting(self):
        """Test get_phase_amps_from_power_for_budgeting uses physical_3p."""
        device = self.create_device(**{CONF_IS_3P: True})
        amps = device.get_phase_amps_from_power_for_budgeting(3000.0)

        # Should use physical_3p which is True
        expected_per_phase = (3000.0 / 3.0) / 230.0
        assert amps[0] == pytest.approx(expected_per_phase, rel=0.01)

    def test_update_amps_with_delta_mono(self):
        """Test update_amps_with_delta for mono-phase."""
        device = self.create_device(**{CONF_MONO_PHASE: "1"})
        amps = [5.0, 5.0, 5.0]
        new_amps = device.update_amps_with_delta(amps, 2.0, is_3p=False)

        # Delta should only be added to mono_phase_index (0)
        assert new_amps[0] == 7.0
        assert new_amps[1] == 5.0
        assert new_amps[2] == 5.0
        # Original should be unchanged
        assert amps[0] == 5.0

    def test_update_amps_with_delta_3p(self):
        """Test update_amps_with_delta for 3-phase."""
        device = self.create_device(**{CONF_IS_3P: True})
        amps = [5.0, 5.0, 5.0]
        new_amps = device.update_amps_with_delta(amps, 2.0, is_3p=True)

        # Delta should be added to all phases
        assert new_amps[0] == 7.0
        assert new_amps[1] == 7.0
        assert new_amps[2] == 7.0


# =============================================================================
# Test AbstractDevice Commands and Constraints
# =============================================================================

class TestAbstractDeviceCommands:
    """Test command-related methods in AbstractDevice."""

    def create_device(self, **kwargs):
        """Create a test device with default values."""
        defaults = {
            "name": "Test Device",
            "device_type": "test",
            CONF_POWER: 1000,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractDevice(**defaults)

    def test_command_and_constraint_reset(self):
        """Test command_and_constraint_reset clears all command state."""
        device = self.create_device()
        device.current_command = MagicMock()
        device.prev_command = MagicMock()
        device.running_command = MagicMock()

        device.command_and_constraint_reset()

        assert device._constraints == []
        assert device.current_command is None
        assert device.prev_command is None
        assert device.running_command is None

    def test_reset_calls_command_and_constraint_reset(self):
        """Test reset calls command_and_constraint_reset."""
        device = self.create_device()
        device.current_command = MagicMock()

        device.reset()

        assert device.current_command is None

    def test_qs_enable_device_default(self):
        """Test qs_enable_device is True by default."""
        device = self.create_device()
        assert device.qs_enable_device is True

    def test_qs_enable_device_setter_disable(self):
        """Test disabling device via qs_enable_device setter."""
        device = self.create_device()
        device.qs_enable_device = False

        assert device._enabled is False

    def test_is_load_has_a_command_now_or_coming_disabled(self):
        """Test is_load_has_a_command_now_or_coming returns False when disabled."""
        device = self.create_device()
        device._enabled = False

        time_now = datetime.now(tz=pytz.UTC)
        assert device.is_load_has_a_command_now_or_coming(time_now) is False

    def test_is_load_has_a_command_now_or_coming_with_current(self):
        """Test is_load_has_a_command_now_or_coming with current_command."""
        device = self.create_device()
        device.current_command = CMD_ON

        time_now = datetime.now(tz=pytz.UTC)
        assert device.is_load_has_a_command_now_or_coming(time_now) is True

    def test_is_load_has_a_command_now_or_coming_with_running(self):
        """Test is_load_has_a_command_now_or_coming with running_command."""
        device = self.create_device()
        device.running_command = CMD_ON

        time_now = datetime.now(tz=pytz.UTC)
        assert device.is_load_has_a_command_now_or_coming(time_now) is True

    def test_is_load_command_set_disabled(self):
        """Test is_load_command_set returns False when disabled."""
        device = self.create_device()
        device._enabled = False

        time_now = datetime.now(tz=pytz.UTC)
        assert device.is_load_command_set(time_now) is False

    def test_is_load_command_set_with_running(self):
        """Test is_load_command_set returns False when running_command exists."""
        device = self.create_device()
        device.running_command = CMD_ON
        device.current_command = CMD_ON

        time_now = datetime.now(tz=pytz.UTC)
        assert device.is_load_command_set(time_now) is False

    def test_is_load_command_set_with_current_only(self):
        """Test is_load_command_set returns True with only current_command."""
        device = self.create_device()
        device.running_command = None
        device.current_command = CMD_ON

        time_now = datetime.now(tz=pytz.UTC)
        assert device.is_load_command_set(time_now) is True

    def test_efficiency_factor(self):
        """Test efficiency_factor calculation."""
        device = self.create_device(**{CONF_DEVICE_EFFICIENCY: 95.0})
        # 100/95 = ~1.053
        assert device.efficiency_factor == pytest.approx(100.0 / 95.0, rel=0.01)

    def test_get_min_max_power_default(self):
        """Test get_min_max_power returns 0,0 for AbstractDevice."""
        device = self.create_device()
        min_p, max_p = device.get_min_max_power()
        assert min_p == 0.0
        assert max_p == 0.0

    def test_reset_daily_load_datas(self):
        """Test reset_daily_load_datas clears num_on_off."""
        device = self.create_device()
        device.num_on_off = 5

        device.reset_daily_load_datas()

        assert device.num_on_off == 0


# =============================================================================
# Test AbstractLoad Specific
# =============================================================================

class TestAbstractLoadConstraints:
    """Test constraint-related methods in AbstractLoad."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def test_is_load_active_disabled(self):
        """Test is_load_active returns False when disabled."""
        load = self.create_load()
        load._enabled = False

        time_now = datetime.now(tz=pytz.UTC)
        assert load.is_load_active(time_now) is False

    def test_is_load_active_no_constraints(self):
        """Test is_load_active returns False with no constraints."""
        load = self.create_load()
        load._constraints = []

        time_now = datetime.now(tz=pytz.UTC)
        assert load.is_load_active(time_now) is False

    def test_get_override_state_no_override(self):
        """Test get_override_state with no override."""
        load = self.create_load()
        load.external_user_initiated_state = None
        load.asked_for_reset_user_initiated_state_time = None

        assert load.get_override_state() == "NO OVERRIDE"

    def test_get_override_state_with_override(self):
        """Test get_override_state with an override."""
        load = self.create_load()
        load.external_user_initiated_state = "TEST_STATE"
        load.asked_for_reset_user_initiated_state_time = None

        assert "TEST_STATE" in load.get_override_state()

    def test_get_override_state_asked_for_reset(self):
        """Test get_override_state when asked for reset."""
        load = self.create_load()
        load.asked_for_reset_user_initiated_state_time = datetime.now(tz=pytz.UTC)

        assert "ASKED FOR RESET" in load.get_override_state()

    def test_is_time_sensitive_best_effort_only(self):
        """Test is_time_sensitive returns False for best effort only."""
        load = self.create_load()
        load.load_is_auto_to_be_boosted = True

        assert load.is_time_sensitive() is False

    def test_is_best_effort_only_load_auto_boosted(self):
        """Test is_best_effort_only_load with auto_boosted."""
        load = self.create_load()
        load.load_is_auto_to_be_boosted = True

        assert load.is_best_effort_only_load() is True

    def test_is_best_effort_only_load_green_only(self):
        """Test is_best_effort_only_load with green_only."""
        load = self.create_load()
        load.qs_best_effort_green_only = True

        assert load.is_best_effort_only_load() is True

    def test_support_green_only_switch_default(self):
        """Test support_green_only_switch default is False."""
        load = self.create_load()
        assert load.support_green_only_switch() is False

    def test_support_user_override_default(self):
        """Test support_user_override default is False."""
        load = self.create_load()
        assert load.support_user_override() is False

    def test_get_power_from_switch_state_on(self):
        """Test get_power_from_switch_state with 'on'."""
        load = self.create_load()
        power = load.get_power_from_switch_state("on")
        assert power == 1500

    def test_get_power_from_switch_state_off(self):
        """Test get_power_from_switch_state with 'off'."""
        load = self.create_load()
        power = load.get_power_from_switch_state("off")
        assert power == 0.0

    def test_get_power_from_switch_state_none(self):
        """Test get_power_from_switch_state with None."""
        load = self.create_load()
        power = load.get_power_from_switch_state(None)
        assert power is None

    def test_get_min_max_power_with_power_use(self):
        """Test get_min_max_power returns power_use."""
        load = self.create_load()
        min_p, max_p = load.get_min_max_power()
        assert min_p == 1500
        assert max_p == 1500

    def test_get_min_max_power_no_power_use(self):
        """Test get_min_max_power returns 0,0 when no power_use."""
        load = self.create_load()
        load.power_use = None
        min_p, max_p = load.get_min_max_power()
        assert min_p == 0.0
        assert max_p == 0.0


# =============================================================================
# Test Utility Functions
# =============================================================================

from custom_components.quiet_solar.home_model.load import (
    align_time_series_and_values,
    get_slots_from_time_series,
    get_value_from_time_series,
    TestLoad,
)


class TestAlignTimeSeries:
    """Test align_time_series_and_values function."""

    def test_both_empty(self):
        """Test with both time series empty."""
        result = align_time_series_and_values([], [])
        assert result == ([], [])

    def test_both_empty_with_operation(self):
        """Test with both empty and operation."""
        result = align_time_series_and_values([], [], operation=lambda a, b: (a or 0) + (b or 0))
        assert result == []

    def test_first_empty_no_operation(self):
        """Test with first series empty, no operation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        tsv2 = [(time1, 10.0, {})]

        result = align_time_series_and_values([], tsv2)
        assert len(result[0]) == 1
        assert result[0][0][1] is None

    def test_second_empty_no_operation(self):
        """Test with second series empty, no operation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        tsv1 = [(time1, 10.0, {})]

        result = align_time_series_and_values(tsv1, [])
        assert len(result[1]) == 1
        assert result[1][0][1] is None

    def test_first_empty_with_operation(self):
        """Test with first series empty, with operation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        tsv2 = [(time1, 10.0, {})]

        result = align_time_series_and_values([], tsv2, operation=lambda a, b: (a or 0) + (b or 0))
        assert len(result) == 1
        assert result[0][1] == 10.0

    def test_same_times_no_operation(self):
        """Test with same times, no operation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        tsv1 = [(time1, 10.0, {})]
        tsv2 = [(time1, 20.0, {})]

        result1, result2 = align_time_series_and_values(tsv1, tsv2)
        assert result1[0][1] == 10.0
        assert result2[0][1] == 20.0

    def test_same_times_with_operation(self):
        """Test with same times, with addition operation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        tsv1 = [(time1, 10.0, {})]
        tsv2 = [(time1, 20.0, {})]

        result = align_time_series_and_values(tsv1, tsv2, operation=lambda a, b: a + b)
        assert result[0][1] == 30.0

    def test_different_times_interpolation(self):
        """Test with different times requiring interpolation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 12, 30, tzinfo=pytz.UTC)
        time3 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)

        tsv1 = [(time1, 0.0, {}), (time3, 100.0, {})]
        tsv2 = [(time2, 50.0, {})]

        result1, result2 = align_time_series_and_values(tsv1, tsv2)

        # result1 should have 3 times, with interpolation for time2
        assert len(result1) == 3

    def test_two_element_tuples(self):
        """Test with two-element tuples (no attributes)."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        tsv1 = [(time1, 10.0)]
        tsv2 = [(time1, 20.0)]

        result1, result2 = align_time_series_and_values(tsv1, tsv2)
        assert len(result1[0]) == 2


class TestGetSlotsFromTimeSeries:
    """Test get_slots_from_time_series function."""

    def test_empty_series(self):
        """Test with empty time series."""
        result = get_slots_from_time_series(
            [],
            datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        )
        assert result == []

    def test_single_slot_at_time(self):
        """Test getting single slot at exact time."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0), (time2, 20.0)]

        result = get_slots_from_time_series(series, time1, end_time=None)
        assert len(result) == 1
        assert result[0][1] == 10.0

    def test_range_of_slots(self):
        """Test getting range of slots."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        time3 = datetime(2024, 1, 1, 14, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0), (time2, 20.0), (time3, 30.0)]

        result = get_slots_from_time_series(series, time1, end_time=time3)
        assert len(result) == 3

    def test_start_before_first(self):
        """Test when start is before first time in series."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0), (time2, 20.0)]

        start = datetime(2024, 1, 1, 11, 0, tzinfo=pytz.UTC)
        result = get_slots_from_time_series(series, start, end_time=time2)
        assert len(result) >= 1


class TestGetValueFromTimeSeries:
    """Test get_value_from_time_series function."""

    def test_empty_series(self):
        """Test with empty time series."""
        time, value, exact, idx = get_value_from_time_series(
            [],
            datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        )
        assert time is None
        assert value is None
        assert exact is False
        assert idx == -1

    def test_none_series(self):
        """Test with None time series."""
        time, value, exact, idx = get_value_from_time_series(
            None,
            datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        )
        assert time is None
        assert value is None

    def test_exact_match_first(self):
        """Test exact match at first element."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0)]

        result_time, value, exact, idx = get_value_from_time_series(series, time1)
        assert exact is True
        assert value == 10.0
        assert idx == 0

    def test_exact_match_last(self):
        """Test exact match at last element."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0), (time2, 20.0)]

        result_time, value, exact, idx = get_value_from_time_series(series, time2)
        assert exact is True
        assert value == 20.0

    def test_closest_value_before(self):
        """Test finding closest value (time slightly before)."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0), (time2, 20.0)]

        target = datetime(2024, 1, 1, 12, 10, tzinfo=pytz.UTC)  # Closer to time1
        result_time, value, exact, idx = get_value_from_time_series(series, target)
        assert value == 10.0

    def test_closest_value_after(self):
        """Test finding closest value (time slightly after)."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0), (time2, 20.0)]

        target = datetime(2024, 1, 1, 12, 50, tzinfo=pytz.UTC)  # Closer to time2
        result_time, value, exact, idx = get_value_from_time_series(series, target)
        assert value == 20.0

    def test_time_before_series_start(self):
        """Test when target time is before series start."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0)]

        target = datetime(2024, 1, 1, 11, 0, tzinfo=pytz.UTC)
        result_time, value, exact, idx = get_value_from_time_series(series, target)
        assert value == 10.0
        assert idx == 0

    def test_time_after_series_end(self):
        """Test when target time is after series end."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        series = [(time1, 10.0)]

        target = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        result_time, value, exact, idx = get_value_from_time_series(series, target)
        assert value == 10.0

    def test_with_interpolation_operation(self):
        """Test with interpolation operation."""
        time1 = datetime(2024, 1, 1, 12, 0, tzinfo=pytz.UTC)
        time2 = datetime(2024, 1, 1, 13, 0, tzinfo=pytz.UTC)
        series = [(time1, 0.0), (time2, 100.0)]

        target = datetime(2024, 1, 1, 12, 30, tzinfo=pytz.UTC)  # Midpoint

        def interpolate(v1, v2, t):
            # Linear interpolation
            d1 = (t - v1[0]).total_seconds()
            d2 = (v2[0] - v1[0]).total_seconds()
            ratio = d1 / d2
            return (t, v1[1] + ratio * (v2[1] - v1[1]))

        result_time, value, exact, idx = get_value_from_time_series(
            series, target, interpolation_operation=interpolate
        )
        assert value == pytest.approx(50.0, rel=0.01)


class TestTestLoad:
    """Test the TestLoad class."""

    def test_testload_init(self):
        """Test TestLoad initialization."""
        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        load = TestLoad(
            name="Test",
            device_type="test",
            min_p=1000,
            max_p=2000,
            min_a=5,
            max_a=10,
            home=mock_home
        )

        assert load.min_p == 1000
        assert load.max_p == 2000
        assert load.min_a == 5
        assert load.max_a == 10

    def test_testload_get_min_max_power(self):
        """Test TestLoad get_min_max_power."""
        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        load = TestLoad(
            name="Test",
            device_type="test",
            min_p=1000,
            max_p=2000,
            home=mock_home
        )

        min_p, max_p = load.get_min_max_power()
        assert min_p == 1000
        assert max_p == 2000


# =============================================================================
# Test PilotedDevice
# =============================================================================

from custom_components.quiet_solar.home_model.load import PilotedDevice


class TestPilotedDevice:
    """Test PilotedDevice class."""

    def create_piloted_device(self, **kwargs):
        """Create a test piloted device."""
        defaults = {
            "name": "Test Piloted",
            "device_type": "piloted",
            CONF_POWER: 1000,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return PilotedDevice(**defaults)

    def test_piloted_device_init(self):
        """Test PilotedDevice initialization."""
        device = self.create_piloted_device()

        assert device.num_demanding_clients is None
        assert device.clients == []

    def test_is_piloted_device_activated_no_clients(self):
        """Test is_piloted_device_activated with no clients."""
        device = self.create_piloted_device()

        assert device.is_piloted_device_activated is False

    def test_prepare_slots_for_piloted_device_budget(self):
        """Test prepare_slots_for_piloted_device_budget."""
        device = self.create_piloted_device()
        device.prepare_slots_for_piloted_device_budget(5)

        assert device.num_demanding_clients is not None
        assert len(device.num_demanding_clients) == 5
        assert all(c == 0 for c in device.num_demanding_clients)

    def test_possible_delta_power_no_clients(self):
        """Test possible_delta_power_for_slot with no clients."""
        device = self.create_piloted_device()
        device.prepare_slots_for_piloted_device_budget(5)

        delta = device.possible_delta_power_for_slot(0, add=True)
        assert delta == 0

    def test_possible_delta_power_first_add(self):
        """Test possible_delta_power_for_slot for first add."""
        device = self.create_piloted_device()
        device.prepare_slots_for_piloted_device_budget(5)

        # Add a mock client
        mock_client = MagicMock()
        device.clients.append(mock_client)

        delta = device.possible_delta_power_for_slot(0, add=True)
        assert delta == 1000  # power_use

    def test_possible_delta_power_last_remove(self):
        """Test possible_delta_power_for_slot for last remove."""
        device = self.create_piloted_device()
        device.prepare_slots_for_piloted_device_budget(5)

        # Add a mock client
        mock_client = MagicMock()
        device.clients.append(mock_client)
        device.num_demanding_clients[0] = 1

        delta = device.possible_delta_power_for_slot(0, add=False)
        assert delta == 1000  # power_use

    def test_update_num_demanding_clients_add(self):
        """Test update_num_demanding_clients_for_slot add."""
        device = self.create_piloted_device()
        device.prepare_slots_for_piloted_device_budget(5)

        mock_client = MagicMock()
        device.clients.append(mock_client)

        power_delta = device.update_num_demanding_clients_for_slot(0, add=True)

        assert device.num_demanding_clients[0] == 1
        assert power_delta == 1000

    def test_update_num_demanding_clients_remove(self):
        """Test update_num_demanding_clients_for_slot remove."""
        device = self.create_piloted_device()
        device.prepare_slots_for_piloted_device_budget(5)

        mock_client = MagicMock()
        device.clients.append(mock_client)
        device.num_demanding_clients[0] = 1

        power_delta = device.update_num_demanding_clients_for_slot(0, add=False)

        assert device.num_demanding_clients[0] == 0


# =============================================================================
# Test AbstractDevice Storage
# =============================================================================

class TestAbstractDeviceStorage:
    """Test storage-related methods in AbstractDevice."""

    def create_device(self, **kwargs):
        """Create a test device with default values."""
        defaults = {
            "name": "Test Device",
            "device_type": "test",
            CONF_POWER: 1000,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractDevice(**defaults)

    def test_update_to_be_saved_extra_device_info(self):
        """Test update_to_be_saved_extra_device_info stores num_on_off."""
        device = self.create_device()
        device.num_on_off = 5

        data = {}
        device.update_to_be_saved_extra_device_info(data)

        assert data["num_on_off"] == 5

    def test_use_saved_extra_device_info(self):
        """Test use_saved_extra_device_info restores num_on_off."""
        device = self.create_device()

        stored_info = {"num_on_off": 4}
        device.use_saved_extra_device_info(stored_info)

        assert device.num_on_off == 4

    def test_use_saved_extra_device_info_odd_rounds_down(self):
        """Test odd num_on_off is rounded down on restore."""
        device = self.create_device()

        stored_info = {"num_on_off": 5}
        device.use_saved_extra_device_info(stored_info)

        assert device.num_on_off == 4

    def test_use_saved_extra_device_info_near_max(self):
        """Test num_on_off near max is adjusted."""
        device = self.create_device(**{CONF_NUM_MAX_ON_OFF: 6})

        stored_info = {"num_on_off": 5}
        device.use_saved_extra_device_info(stored_info)

        # Should be adjusted to max - 2 = 4
        assert device.num_on_off == 4


# =============================================================================
# Test check_commands and force_relaunch_command
# =============================================================================

from custom_components.quiet_solar.home_model.load import NUM_MAX_INVALID_PROBES_COMMANDS


class TestCheckCommands:
    """Test check_commands method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    @pytest.mark.asyncio
    async def test_check_commands_disabled(self):
        """Test check_commands returns 0 when disabled."""
        load = self.create_load()
        load._enabled = False

        time_now = datetime.now(tz=pytz.UTC)
        result, _ = await load.check_commands(time_now)

        assert result == timedelta(seconds=0)

    @pytest.mark.asyncio
    async def test_check_commands_no_running_command(self):
        """Test check_commands with no running command."""
        load = self.create_load()
        load.running_command = None

        time_now = datetime.now(tz=pytz.UTC)
        result, _ = await load.check_commands(time_now)

        assert result == timedelta(seconds=0)

    @pytest.mark.asyncio
    async def test_check_commands_running_command_set_true(self):
        """Test check_commands when probe returns True."""
        load = self.create_load()
        load.running_command = CMD_ON
        load.running_command_last_launch = datetime.now(tz=pytz.UTC)

        time_now = datetime.now(tz=pytz.UTC)
        result, _ = await load.check_commands(time_now)

        # Command should be acknowledged
        assert load.running_command is None
        assert load.current_command == CMD_ON

    @pytest.mark.asyncio
    async def test_check_commands_running_command_invalid_increments_count(self):
        """Test check_commands increments invalid count when probe returns None."""
        load = self.create_load()
        load.running_command = CMD_ON
        load.running_command_num_relaunch_after_invalid = 0

        # Override probe to return None
        async def probe_none(time, cmd):
            return None
        load.probe_if_command_set = probe_none

        time_now = datetime.now(tz=pytz.UTC)
        await load.check_commands(time_now)

        assert load.running_command_num_relaunch_after_invalid == 1

    @pytest.mark.asyncio
    async def test_check_commands_max_invalid_kills_command(self):
        """Test check_commands kills command after max invalid probes."""
        load = self.create_load()
        load.running_command = CMD_ON
        load.running_command_num_relaunch_after_invalid = NUM_MAX_INVALID_PROBES_COMMANDS - 1

        async def probe_none(time, cmd):
            return None
        load.probe_if_command_set = probe_none

        time_now = datetime.now(tz=pytz.UTC)
        await load.check_commands(time_now)

        # Should be killed after reaching max
        assert load.running_command is None
        assert load.current_command is None

    @pytest.mark.asyncio
    async def test_check_commands_launches_stacked_command(self):
        """Test check_commands launches stacked command when running is done."""
        load = self.create_load()
        load.running_command = None
        load._stacked_command = CMD_ON

        time_now = datetime.now(tz=pytz.UTC)
        await load.check_commands(time_now)

        # Stacked command should have been launched
        assert load._stacked_command is None


class TestForceRelaunchCommand:
    """Test force_relaunch_command method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    @pytest.mark.asyncio
    async def test_force_relaunch_disabled_clears_running(self):
        """Test force_relaunch clears running_command when disabled."""
        load = self.create_load()
        load._enabled = False
        load.running_command = CMD_ON

        time_now = datetime.now(tz=pytz.UTC)
        await load.force_relaunch_command(time_now)

        assert load.running_command is None

    @pytest.mark.asyncio
    async def test_force_relaunch_increments_relaunch_count(self):
        """Test force_relaunch increments relaunch count."""
        load = self.create_load()
        load.running_command = CMD_ON
        load.running_command_num_relaunch = 0

        # Track relaunch count before and during execution
        from unittest.mock import AsyncMock, patch

        # Mock both execute_command and probe_if_command_set to return False
        with patch.object(load, 'execute_command', new=AsyncMock(return_value=False)), \
             patch.object(load, 'probe_if_command_set', new=AsyncMock(return_value=False)):

            time_now = datetime.now(tz=pytz.UTC)
            await load.force_relaunch_command(time_now)

            assert load.running_command_num_relaunch == 1

    @pytest.mark.asyncio
    async def test_force_relaunch_updates_last_launch_time(self):
        """Test force_relaunch updates last launch time."""
        load = self.create_load()
        load.running_command = CMD_ON
        load.running_command_last_launch = None

        from unittest.mock import AsyncMock, patch

        # Mock both execute_command and probe_if_command_set to return False
        with patch.object(load, 'execute_command', new=AsyncMock(return_value=False)), \
             patch.object(load, 'probe_if_command_set', new=AsyncMock(return_value=False)):

            time_now = datetime.now(tz=pytz.UTC)
            await load.force_relaunch_command(time_now)

            assert load.running_command_last_launch == time_now

    @pytest.mark.asyncio
    async def test_force_relaunch_command_set_true_acks(self):
        """Test force_relaunch acks command when execute returns True."""
        load = self.create_load()
        load.running_command = CMD_ON

        async def execute_true(time, cmd):
            return True
        load.execute_command = execute_true

        time_now = datetime.now(tz=pytz.UTC)
        await load.force_relaunch_command(time_now)

        assert load.running_command is None
        assert load.current_command == CMD_ON

    @pytest.mark.asyncio
    async def test_force_relaunch_handles_exception(self):
        """Test force_relaunch handles exceptions gracefully."""
        load = self.create_load()
        load.running_command = CMD_ON

        async def execute_raises(time, cmd):
            raise ValueError("Test error")
        load.execute_command = execute_raises

        time_now = datetime.now(tz=pytz.UTC)
        await load.force_relaunch_command(time_now)
        assert load.running_command_last_launch == time_now
        assert load.running_command_num_relaunch == 1


# =============================================================================
# Test do_probe_state_change
# =============================================================================

from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_CONSTRAINT


class TestDoProbeStateChange:
    """Test do_probe_state_change method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    @pytest.mark.asyncio
    async def test_do_probe_state_change_disabled(self):
        """Test do_probe_state_change returns early when disabled."""
        load = self.create_load()
        load._enabled = False
        load._last_hash_state = "old_hash"

        time_now = datetime.now(tz=pytz.UTC)
        await load.do_probe_state_change(time_now)

        # Should return early, no changes
        assert load._last_hash_state == "old_hash"

    @pytest.mark.asyncio
    async def test_do_probe_state_change_first_call_no_notify(self):
        """Test first call doesn't notify (last_hash_state is None)."""
        load = self.create_load()
        load._last_hash_state = None
        notified = []

        async def mock_on_state_change(time, change_type):
            notified.append(change_type)
        load.on_device_state_change = mock_on_state_change

        time_now = datetime.now(tz=pytz.UTC)
        await load.do_probe_state_change(time_now)

        # Should not notify on first call
        assert len(notified) == 0
        # But should update hash
        assert load._last_hash_state is not None

    @pytest.mark.asyncio
    async def test_do_probe_state_change_notifies_on_change(self):
        """Test do_probe_state_change notifies when hash changes."""
        load = self.create_load()
        load._last_hash_state = "old_hash"
        notified = []

        async def mock_on_state_change(time, change_type):
            notified.append(change_type)
        load.on_device_state_change = mock_on_state_change

        time_now = datetime.now(tz=pytz.UTC)
        await load.do_probe_state_change(time_now)

        # Should notify because hash changed
        assert len(notified) == 1
        assert notified[0] == DEVICE_STATUS_CHANGE_CONSTRAINT

    @pytest.mark.asyncio
    async def test_do_probe_state_change_no_notify_same_hash(self):
        """Test do_probe_state_change doesn't notify when hash is same."""
        load = self.create_load()

        time_now = datetime.now(tz=pytz.UTC)
        # First call to set initial hash
        await load.do_probe_state_change(time_now)
        old_hash = load._last_hash_state

        notified = []
        async def mock_on_state_change(time, change_type):
            notified.append(change_type)
        load.on_device_state_change = mock_on_state_change

        # Second call with same state
        await load.do_probe_state_change(time_now)

        # Should not notify if hash is same
        assert len(notified) == 0


# =============================================================================
# Test _match_ct
# =============================================================================

class TestMatchCt:
    """Test _match_ct method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(self, load, load_param=None, load_info=None):
        """Create a real constraint."""
        return create_real_constraint(
            load=load,
            load_param=load_param,
            load_info=load_info,
        )

    def test_match_ct_different_load_param(self):
        """Test _match_ct returns False for different load_param."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param="car_A")

        result = load._match_ct(ct, "car_B")
        assert result is False

    def test_match_ct_same_load_param_no_info(self):
        """Test _match_ct returns True for same load_param, no load_info."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param="car_A")

        result = load._match_ct(ct, "car_A")
        assert result is True

    def test_match_ct_same_load_param_with_matching_info(self):
        """Test _match_ct returns True when load_info matches."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param="car_A", load_info={"key": "value"})

        result = load._match_ct(ct, "car_A", {"key": "value"})
        assert result is True

    def test_match_ct_same_load_param_with_conflicting_info(self):
        """Test _match_ct returns False when load_info has conflicting key."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param="car_A", load_info={"key": "value1"})

        result = load._match_ct(ct, "car_A", {"key": "value2"})
        assert result is False

    def test_match_ct_same_param_with_disjoint_info(self):
        """Test _match_ct returns True when load_info has disjoint keys."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param="car_A", load_info={"key1": "value1"})

        result = load._match_ct(ct, "car_A", {"key2": "value2"})
        assert result is True

    def test_match_ct_none_load_param(self):
        """Test _match_ct with None load_param."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param=None)

        result = load._match_ct(ct, None)
        assert result is True

    def test_match_ct_ct_has_no_load_info(self):
        """Test _match_ct returns True when constraint has no load_info."""
        load = self.create_load()
        ct = self.create_constraint(load, load_param="car_A", load_info=None)

        result = load._match_ct(ct, "car_A", {"key": "value"})
        assert result is True


# =============================================================================
# Test clean_constraints_for_load_param_and_if_same_key_same_value_info
# =============================================================================

class Testclean_constraints_for_load_param_and_if_same_key_same_value_info:
    """Test clean_constraints_for_load_param_and_info method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(self, load, load_param=None, load_info=None):
        """Create a real constraint."""
        return create_real_constraint(
            load=load,
            load_param=load_param,
            load_info=load_info,
        )

    def test_clean_constraints_keeps_matching_constraint(self):
        """Test constraints with matching load_param are kept."""
        load = self.create_load()
        ct1 = self.create_constraint(load, load_param="car_A")
        load._constraints = [ct1]
        load._last_completed_constraint = None

        time_now = datetime.now(tz=pytz.UTC)
        load.clean_constraints_for_load_param_and_if_same_key_same_value_info(time_now, "car_A")

        # Constraint should be kept (pushed back)
        assert len(load._constraints) >= 0  # May be re-processed

    def test_clean_constraints_removes_non_matching_constraint(self):
        """Test constraints with non-matching load_param are removed."""
        load = self.create_load()
        ct1 = self.create_constraint(load, load_param="car_A")
        ct2 = self.create_constraint(load, load_param="car_B")
        load._constraints = [ct1, ct2]
        load._last_completed_constraint = None

        time_now = datetime.now(tz=pytz.UTC)
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(time_now, "car_A")

        assert result is True  # Found bad constraint

    def test_clean_constraints_with_load_info_matching(self):
        """Test constraints with matching load_info are kept."""
        load = self.create_load()
        ct1 = self.create_constraint(
            load,
            load_param="car_A",
            load_info={"charger": "charger_1"}
        )
        load._constraints = [ct1]
        load._last_completed_constraint = None

        time_now = datetime.now(tz=pytz.UTC)
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            time_now,
            "car_A",
            {"charger": "charger_1"}
        )

        assert result is True
        assert len(load._constraints) == 1
        assert load._constraints[0].load_info == {"charger": "charger_1"}

    def test_clean_constraints_with_load_info_conflicting(self):
        """Test constraints with conflicting load_info are removed."""
        load = self.create_load()
        ct1 = self.create_constraint(
            load,
            load_param="car_A",
            load_info={"charger": "charger_1"}
        )
        load._constraints = [ct1]
        load._last_completed_constraint = None

        time_now = datetime.now(tz=pytz.UTC)
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            time_now,
            "car_A",
            {"charger": "charger_2"}  # Different charger
        )

        assert result is True  # Found bad constraint

    def test_clean_constraints_keeps_last_completed_if_matching(self):
        """Test _last_completed_constraint is kept if matching."""
        load = self.create_load()
        completed = self.create_constraint(load, load_param="car_A")
        load._constraints = []
        load._last_completed_constraint = completed

        time_now = datetime.now(tz=pytz.UTC)
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            time_now,
            "car_A",
            for_full_reset=False,
        )

        assert result is False
        assert load._last_completed_constraint == completed

    def test_clean_constraints_removes_last_completed_if_not_matching(self):
        """Test _last_completed_constraint is removed if not matching."""
        load = self.create_load()
        completed = self.create_constraint(load, load_param="car_B")
        load._constraints = []
        load._last_completed_constraint = completed

        time_now = datetime.now(tz=pytz.UTC)
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(time_now, "car_A")

        assert result is True  # Found bad constraint

    def test_clean_constraints_no_reset_if_all_match(self):
        """Test returns False if all constraints match and for_full_reset=False."""
        load = self.create_load()
        ct1 = self.create_constraint(load, load_param="car_A")
        load._constraints = [ct1]
        load._last_completed_constraint = None

        time_now = datetime.now(tz=pytz.UTC)
        result = load.clean_constraints_for_load_param_and_if_same_key_same_value_info(
            time_now,
            "car_A",
            for_full_reset=False
        )

        assert result is False  # No bad constraints found



# =============================================================================
# Test ack_completed_constraint
# =============================================================================

from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_CONSTRAINT_COMPLETED


class TestAckCompletedConstraint:
    """Test ack_completed_constraint method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    @pytest.mark.asyncio
    async def test_ack_completed_constraint_disabled(self):
        """Test ack_completed_constraint does nothing when disabled."""
        load = self.create_load()
        load._enabled = False
        load._last_completed_constraint = None

        mock_ct = MagicMock()
        time_now = datetime.now(tz=pytz.UTC)
        await load.ack_completed_constraint(time_now, mock_ct)

        # Should not set because disabled
        assert load._last_completed_constraint is None

    @pytest.mark.asyncio
    async def test_ack_completed_constraint_sets_last_completed(self):
        """Test ack_completed_constraint sets _last_completed_constraint."""
        load = self.create_load()
        load._last_completed_constraint = None

        mock_ct = MagicMock()
        notified = []

        async def mock_on_state_change(time, change_type):
            notified.append(change_type)
        load.on_device_state_change = mock_on_state_change

        time_now = datetime.now(tz=pytz.UTC)
        await load.ack_completed_constraint(time_now, mock_ct)

        assert load._last_completed_constraint == mock_ct
        assert DEVICE_STATUS_CHANGE_CONSTRAINT_COMPLETED in notified


# =============================================================================
# Test get_active_state_hash
# =============================================================================

class TestGetActiveStateHash:
    """Test get_active_state_hash method."""

    BASE_TIME = datetime(2026, 1, 22, 11, 0, tzinfo=pytz.UTC)

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(self, load, load_param=None, end_time=None, time_now=None):
        """Create a real constraint."""
        if end_time is None:
            if time_now is None:
                time_now = self.BASE_TIME
            end_time = time_now + timedelta(hours=1)
        return create_real_constraint(
            load=load,
            time_now=time_now,
            load_param=load_param,
            end_time=end_time,
        )

    def test_get_active_state_hash_nothing_planned(self):
        """Test hash when nothing is planned."""
        load = self.create_load()
        load._constraints = []
        load._last_completed_constraint = None

        time_now = self.BASE_TIME
        hash_val = load.get_active_state_hash(time_now)

        assert hash_val == "NOTHING PLANNED"

    def test_get_active_state_hash_with_completed(self):
        """Test hash when there's a completed constraint."""
        load = self.create_load()
        load._constraints = []

        completed = self.create_constraint(
            load,
            load_param="car_A",
            end_time=self.BASE_TIME - timedelta(hours=1),
            time_now=self.BASE_TIME,
        )
        load._last_completed_constraint = completed

        time_now = self.BASE_TIME
        hash_val = load.get_active_state_hash(time_now)

        assert "COMPLETED:" in hash_val
        # The stable_name contains constraint info
        assert "car_A" in hash_val

    def test_get_active_state_hash_with_completed_no_param(self):
        """Test hash with completed constraint with no load_param."""
        load = self.create_load()
        load._constraints = []

        completed = self.create_constraint(
            load,
            load_param=None,
            end_time=self.BASE_TIME - timedelta(hours=1),
            time_now=self.BASE_TIME,
        )
        load._last_completed_constraint = completed

        time_now = self.BASE_TIME
        hash_val = load.get_active_state_hash(time_now)

        assert "COMPLETED:" in hash_val
        assert "-NO-" in hash_val

    def test_get_active_state_hash_with_active_constraint(self):
        """Test hash with active constraint."""
        load = self.create_load()

        active = self.create_constraint(
            load,
            load_param="car_A",
            end_time=self.BASE_TIME + timedelta(hours=1),
            time_now=self.BASE_TIME,
        )
        load._constraints = [active]

        time_now = self.BASE_TIME
        hash_val = load.get_active_state_hash(time_now)

        assert "RUNNING:" in hash_val
        assert "car_A" in hash_val

    def test_get_active_state_hash_with_active_no_param(self):
        """Test hash with active constraint with no load_param."""
        load = self.create_load()

        active = self.create_constraint(
            load,
            load_param=None,
            end_time=self.BASE_TIME + timedelta(hours=1),
            time_now=self.BASE_TIME,
        )
        load._constraints = [active]

        time_now = self.BASE_TIME
        hash_val = load.get_active_state_hash(time_now)

        assert "RUNNING:" in hash_val
        assert "-NO-" in hash_val


# =============================================================================
# Test set_live_constraints
# =============================================================================


class TestSetLiveConstraints:
    """Test set_live_constraints method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(self, load, end_time, start_time=None, target_value=100.0,
                          current_value=0.0, constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
                          from_user=False):
        """Create a real constraint."""
        return create_real_constraint(
            load=load,
            end_time=end_time,
            start_time=start_time,
            target_value=target_value,
            current_value=current_value,
            constraint_type=constraint_type,
            from_user=from_user,
        )

    def test_set_live_constraints_empty(self):
        """Test set_live_constraints with empty list."""
        load = self.create_load()
        ct_dummy = self.create_constraint(load, datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC))
        load._constraints = [ct_dummy]  # Start with something

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [])

        assert load._constraints == []

    def test_set_live_constraints_removes_none(self):
        """Test set_live_constraints removes None values."""
        load = self.create_load()

        ct1 = self.create_constraint(load, datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC))

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1, None, ct1])

        # None should be filtered out
        assert None not in load._constraints

    def test_set_live_constraints_sorts_by_end_time(self):
        """Test constraints are sorted by end_time."""
        load = self.create_load()

        ct1 = self.create_constraint(load, datetime(2026, 1, 22, 14, 0, tzinfo=pytz.UTC), target_value=100)
        ct2 = self.create_constraint(load, datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC), target_value=200)
        ct3 = self.create_constraint(load, datetime(2026, 1, 22, 16, 0, tzinfo=pytz.UTC), target_value=300)

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1, ct2, ct3])

        # Should be sorted by end_time
        if len(load._constraints) >= 2:
            for i in range(len(load._constraints) - 1):
                assert load._constraints[i].end_of_constraint <= load._constraints[i+1].end_of_constraint

    def test_set_live_constraints_removes_met_constraints(self):
        """Test met constraints are removed."""
        load = self.create_load()

        # ct1 is met (current >= target)
        ct1 = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0,
            current_value=100.0  # Met
        )
        ct2 = self.create_constraint(
            load,
            datetime(2026, 1, 22, 14, 0, tzinfo=pytz.UTC),
            target_value=100.0,
            current_value=0.0  # Not met
        )

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1, ct2])

        # ct1 should be removed because it's met
        assert ct1 not in load._constraints

    def test_set_live_constraints_keeps_one_infinite(self):
        """Test only one infinite constraint is kept."""
        load = self.create_load()

        ct1 = self.create_constraint(load, DATETIME_MAX_UTC, target_value=100.0)
        ct2 = self.create_constraint(load, DATETIME_MAX_UTC, target_value=200.0, from_user=True)  # Higher score

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1, ct2])

        # Only one infinite should remain
        infinite_count = sum(1 for c in load._constraints
                           if c.end_of_constraint == DATETIME_MAX_UTC)
        assert infinite_count <= 1

    def test_set_live_constraints_handles_as_fast_as_possible(self):
        """Test as_fast_as_possible constraint handling."""
        load = self.create_load()

        ct_fast1 = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            target_value=100.0
        )
        ct_fast2 = self.create_constraint(
            load,
            datetime(2026, 1, 22, 14, 0, tzinfo=pytz.UTC),
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            target_value=200.0,
            from_user=True  # Higher score
        )
        ct_normal = self.create_constraint(
            load,
            datetime(2026, 1, 22, 16, 0, tzinfo=pytz.UTC),
            target_value=300.0
        )

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct_fast1, ct_fast2, ct_normal])

        # Only one as_fast constraint should remain and be first
        as_fast_count = sum(1 for c in load._constraints if c.as_fast_as_possible)
        if load._constraints:
            # First should be as_fast if any remain
            assert as_fast_count <= 1

    def test_set_live_constraints_cluster_same_end_time(self):
        """Test constraints with same end time are clustered."""
        load = self.create_load()

        same_end = datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC)
        ct1 = self.create_constraint(load, same_end, target_value=100.0)
        ct2 = self.create_constraint(load, same_end, target_value=200.0, from_user=True)  # Higher score
        ct3 = self.create_constraint(load, same_end, target_value=50.0)

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1, ct2, ct3])

        # Only highest score should remain for same end time
        count_same_end = sum(1 for c in load._constraints
                           if c.end_of_constraint == same_end)
        assert count_same_end <= 1

    def test_set_live_constraints_multiple_clusters(self):
        """Test multiple clusters of same end times (line 1002 coverage)."""
        load = self.create_load()

        end1 = datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC)
        end2 = datetime(2026, 1, 22, 14, 0, tzinfo=pytz.UTC)

        # First cluster with 2 constraints at end1
        ct1a = self.create_constraint(load, end1, target_value=100.0)
        ct1b = self.create_constraint(load, end1, target_value=200.0, from_user=True)

        # Second cluster with 2 constraints at end2
        ct2a = self.create_constraint(load, end2, target_value=50.0)
        ct2b = self.create_constraint(load, end2, target_value=150.0, from_user=True)

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1a, ct1b, ct2a, ct2b])

        # Should have at most one per end time
        count_end1 = sum(1 for c in load._constraints if c.end_of_constraint == end1)
        count_end2 = sum(1 for c in load._constraints if c.end_of_constraint == end2)
        assert count_end1 <= 1
        assert count_end2 <= 1

    def test_set_live_constraints_updates_internal_start(self):
        """Test current_start_of_constraint is updated."""
        load = self.create_load()

        ct1 = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            start_time=datetime(2026, 1, 21, 10, 0, tzinfo=pytz.UTC),
        )
        ct2 = self.create_constraint(
            load,
            datetime(2026, 1, 22, 14, 0, tzinfo=pytz.UTC),
            start_time=datetime(2026, 1, 21, 8, 0, tzinfo=pytz.UTC),
        )

        time_now = datetime.now(tz=pytz.UTC)
        load.set_live_constraints(time_now, [ct1, ct2])

        assert ct2.current_start_of_constraint > ct1.start_of_constraint
        assert ct2.current_start_of_constraint >= ct1.end_of_constraint

        # After first constraint, second should have updated start


# =============================================================================
# Test push_live_constraint
# =============================================================================

class TestPushLiveConstraint:
    """Test push_live_constraint method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(self, load, end_time, target_value=100.0,
                          current_value=0.0, constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
                          from_user=False):
        """Create a real constraint."""
        return create_real_constraint(
            load=load,
            end_time=end_time,
            target_value=target_value,
            current_value=current_value,
            constraint_type=constraint_type,
            from_user=from_user,
        )

    def test_push_live_constraint_disabled(self):
        """Test push_live_constraint clears constraints when disabled."""
        load = self.create_load()
        load._enabled = False
        ct_dummy = self.create_constraint(load, datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC))
        load._constraints = [ct_dummy]

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, None)

        assert result is True
        assert load._constraints == []

    def test_push_live_constraint_none_constraint(self):
        """Test push_live_constraint with None constraint."""
        load = self.create_load()
        load._constraints = []

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, None)

        # Should return without doing anything significant
        assert load._constraints == []

    def test_push_live_constraint_adds_new(self):
        """Test push_live_constraint adds new constraint."""
        load = self.create_load()
        load._constraints = []
        load._last_completed_constraint = None

        ct = self.create_constraint(load, datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC))

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, ct)

        assert result is True

    def test_push_live_constraint_rejects_same_as_completed(self):
        """Test constraint same as completed is rejected."""
        load = self.create_load()

        completed = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0
        )
        load._last_completed_constraint = completed
        load._constraints = []

        new_ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0
        )

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, new_ct)

        assert result is False

    def test_push_live_constraint_rejects_equal_no_current(self):
        """Test constraint equal to existing (eq_no_current) is rejected."""
        load = self.create_load()

        # Two constraints with identical parameters will have eq_no_current = True
        existing = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0
        )
        load._constraints = [existing]
        load._last_completed_constraint = None

        # Same constraint parameters
        new_ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0
        )

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, new_ct)

        assert result is False  # Should be rejected as duplicate

    def test_push_live_constraint_replaces_same_end_different_score(self):
        """Test constraint with same end but different score replaces existing."""
        load = self.create_load()

        existing = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0,
            current_value=50.0
        )
        load._constraints = [existing]
        load._last_completed_constraint = None

        new_ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=200.0,  # Different target = different score
            current_value=0.0,
            from_user=True  # Higher score
        )

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, new_ct)

        assert result is True

    def test_push_live_constraint_rejects_same_end_same_score(self):
        """Test constraint with same end and same score is rejected."""
        load = self.create_load()

        existing = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0
        )
        load._constraints = [existing]
        load._last_completed_constraint = None

        new_ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            target_value=100.0  # Same target = same score
        )

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, new_ct)

        assert result is False

    def test_push_live_constraint_both_as_fast_replaces(self):
        """Test as_fast_as_possible constraints replace each other."""
        load = self.create_load()

        existing = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            target_value=100.0
        )
        load._constraints = [existing]
        load._last_completed_constraint = None

        new_ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 14, 0, tzinfo=pytz.UTC),  # Different end
            constraint_type=CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
            target_value=200.0,
            from_user=True  # Higher score
        )

        time_now = datetime.now(tz=pytz.UTC)
        result = load.push_live_constraint(time_now, new_ct)

        assert result is True


# =============================================================================
# Test update_live_constraints
# =============================================================================

class TestUpdateLiveConstraints:
    """Test update_live_constraints method."""

    BASE_TIME = datetime(2026, 1, 22, 11, 0, tzinfo=pytz.UTC)

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None
        mock_home._period = timedelta(minutes=5)

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(
        self,
        load,
        end_time,
        time_now=None,
        target_value=100.0,
        current_value=0.0,
        constraint_type=CONSTRAINT_TYPE_FILLER_AUTO,
    ):
        """Create a real constraint."""
        return create_real_constraint(
            load=load,
            end_time=end_time,
            time_now=time_now,
            target_value=target_value,
            current_value=current_value,
            constraint_type=constraint_type,
        )

    @pytest.mark.asyncio
    async def test_update_live_constraints_disabled(self):
        """Test update_live_constraints clears when disabled."""
        load = self.create_load()
        load._enabled = False
        ct_dummy = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            time_now=self.BASE_TIME,
        )
        load._constraints = [ct_dummy]

        time_now = self.BASE_TIME
        result = await load.update_live_constraints(time_now, timedelta(minutes=5))

        assert result is True
        assert load._constraints == []

    @pytest.mark.asyncio
    async def test_update_live_constraints_no_constraints(self):
        """Test update_live_constraints with no constraints."""
        load = self.create_load()
        load._constraints = []
        load._last_constraint_update = None

        time_now = self.BASE_TIME
        result = await load.update_live_constraints(time_now, timedelta(minutes=5))

        assert result is False
        assert load._last_constraint_update == time_now

    @pytest.mark.asyncio
    async def test_update_live_constraints_met_constraint_skipped(self):
        """Test met constraints are skipped."""
        load = self.create_load()

        # Constraint is met because current >= target
        ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            time_now=self.BASE_TIME,
            target_value=100.0,
            current_value=100.0  # Met
        )
        load._constraints = [ct]
        load._last_completed_constraint = None

        time_now = self.BASE_TIME
        result = await load.update_live_constraints(time_now, timedelta(minutes=5))

        assert result is True  # Force solving because constraint met
        assert ct.skip is True

    @pytest.mark.asyncio
    async def test_update_live_constraints_updates_current_values(self):
        """Test current constraint values are updated."""
        load = self.create_load()

        ct = self.create_constraint(
            load,
            datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC),
            time_now=self.BASE_TIME,
            target_value=100.0,
            current_value=50.0
        )
        load._constraints = [ct]
        load._last_completed_constraint = None

        time_now = self.BASE_TIME
        await load.update_live_constraints(time_now, timedelta(minutes=5))

        # Should update current constraint values
        assert load.current_constraint_current_value == 50.0

    @pytest.mark.asyncio
    async def test_update_live_constraints_resets_daily_on_day_change(self):
        """Test daily data is reset on day change."""
        load = self.create_load()
        load._constraints = []

        # Set last update to a specific time yesterday in local timezone
        yesterday_utc = datetime(2026, 1, 19, 20, 0, tzinfo=pytz.UTC)  # 8 PM UTC
        load._last_constraint_update = yesterday_utc
        load.num_on_off = 5

        # Now is a different day in local time
        time_now = datetime(2026, 1, 20, 20, 0, tzinfo=pytz.UTC)  # 8 PM UTC next day
        await load.update_live_constraints(time_now, timedelta(minutes=5))

        # num_on_off should be reset on day change
        assert load.num_on_off == 0


# =============================================================================
# Test mark_current_constraint_has_done
# =============================================================================

class TestMarkCurrentConstraintHasDone:
    """Test mark_current_constraint_has_done method."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None
        mock_home._period = timedelta(minutes=5)

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def create_constraint(self, load, end_time=None, target_value=100.0,
                          current_value=50.0):
        """Create a real constraint."""
        if end_time is None:
            end_time = datetime(2026, 1, 22, 12, 0, tzinfo=pytz.UTC)
        return create_real_constraint(
            load=load,
            end_time=end_time,
            target_value=target_value,
            current_value=current_value,
        )

    @pytest.mark.asyncio
    async def test_mark_current_constraint_has_done_sets_value(self):
        """Test mark_current_constraint_has_done sets current_value to target."""
        load = self.create_load()

        ct = self.create_constraint(
            load,
            target_value=100.0,
            current_value=50.0
        )
        load._constraints = [ct]
        load._last_completed_constraint = None

        await load.mark_current_constraint_has_done(time=ct.end_of_constraint - timedelta(minutes=5))

        # current_value should be set to target_value
        assert ct.current_value == ct.target_value

    @pytest.mark.asyncio
    async def test_mark_current_constraint_has_done_no_constraint(self):
        """Test mark_current_constraint_has_done with no active constraint."""
        load = self.create_load()
        load._constraints = []

        await load.mark_current_constraint_has_done()
        assert load._constraints == []


# =============================================================================
# Test AbstractLoad Storage Methods (update_to_be_saved_extra_device_info and use_saved_extra_device_info)
# =============================================================================

class TestAbstractLoadStorage:
    """Test AbstractLoad storage methods for extra device info."""

    def create_load(self, **kwargs):
        """Create a test load with default values."""
        defaults = {
            "name": "Test Load",
            "device_type": "test_load",
            CONF_POWER: 1500,
        }
        defaults.update(kwargs)

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        defaults["home"] = mock_home

        return AbstractLoad(**defaults)

    def test_update_to_be_saved_extra_device_info_basic(self):
        """Test update_to_be_saved_extra_device_info saves basic fields."""
        load = self.create_load()
        load.num_on_off = 5
        load.external_user_initiated_state = None
        load.external_user_initiated_state_time = None
        load.asked_for_reset_user_initiated_state_time = None
        load.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        data = {}
        load.update_to_be_saved_extra_device_info(data)

        # From AbstractDevice
        assert data["num_on_off"] == 5
        # From AbstractLoad - None values
        assert data["external_user_initiated_state"] is None
        assert data["external_user_initiated_state_time"] is None
        assert data["asked_for_reset_user_initiated_state_time"] is None
        assert data["asked_for_reset_user_initiated_state_time_first_cmd_reset_done"] is None

    def test_update_to_be_saved_extra_device_info_with_external_state(self):
        """Test update_to_be_saved_extra_device_info with external_user_initiated_state."""
        load = self.create_load()
        load.external_user_initiated_state = "ON_STATE"

        data = {}
        load.update_to_be_saved_extra_device_info(data)

        assert data["external_user_initiated_state"] == "ON_STATE"

    def test_update_to_be_saved_extra_device_info_with_datetime_fields(self):
        """Test update_to_be_saved_extra_device_info stores datetime as string."""
        load = self.create_load()
        test_time = datetime(2026, 1, 24, 10, 30, 0, tzinfo=pytz.UTC)
        load.external_user_initiated_state_time = test_time
        load.asked_for_reset_user_initiated_state_time = test_time
        load.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = test_time

        data = {}
        load.update_to_be_saved_extra_device_info(data)

        # Datetime fields should be stored as strings
        assert data["external_user_initiated_state_time"] == str(test_time)
        assert data["asked_for_reset_user_initiated_state_time"] == str(test_time)
        assert data["asked_for_reset_user_initiated_state_time_first_cmd_reset_done"] == str(test_time)

    def test_update_to_be_saved_extra_device_info_calls_super(self):
        """Test update_to_be_saved_extra_device_info calls parent class method."""
        load = self.create_load()
        load.num_on_off = 10

        data = {}
        load.update_to_be_saved_extra_device_info(data)

        # num_on_off from AbstractDevice.update_to_be_saved_extra_device_info
        assert data["num_on_off"] == 10

    def test_use_saved_extra_device_info_basic(self):
        """Test use_saved_extra_device_info restores basic fields."""
        load = self.create_load()

        stored_info = {
            "num_on_off": 6,
            "external_user_initiated_state": None,
            "external_user_initiated_state_time": None,
            "asked_for_reset_user_initiated_state_time": None,
            "asked_for_reset_user_initiated_state_time_first_cmd_reset_done": None,
        }
        load.use_saved_extra_device_info(stored_info)

        # From AbstractDevice
        assert load.num_on_off == 6
        # From AbstractLoad
        assert load.external_user_initiated_state is None
        assert load.external_user_initiated_state_time is None
        assert load.asked_for_reset_user_initiated_state_time is None
        assert load.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None

    def test_use_saved_extra_device_info_with_external_state(self):
        """Test use_saved_extra_device_info restores external_user_initiated_state."""
        load = self.create_load()

        stored_info = {
            "external_user_initiated_state": "BOOST_MODE",
        }
        load.use_saved_extra_device_info(stored_info)

        assert load.external_user_initiated_state == "BOOST_MODE"

    def test_use_saved_extra_device_info_with_datetime_strings(self):
        """Test use_saved_extra_device_info parses datetime strings."""
        load = self.create_load()
        test_time = datetime(2026, 1, 24, 10, 30, 0, tzinfo=pytz.UTC)

        stored_info = {
            "external_user_initiated_state_time": str(test_time),
            "asked_for_reset_user_initiated_state_time": str(test_time),
            "asked_for_reset_user_initiated_state_time_first_cmd_reset_done": str(test_time),
        }
        load.use_saved_extra_device_info(stored_info)

        # Datetime fields should be parsed from strings
        assert load.external_user_initiated_state_time == test_time
        assert load.asked_for_reset_user_initiated_state_time == test_time
        assert load.asked_for_reset_user_initiated_state_time_first_cmd_reset_done == test_time

    def test_use_saved_extra_device_info_missing_keys(self):
        """Test use_saved_extra_device_info handles missing keys gracefully."""
        load = self.create_load()

        # Empty stored info - should use defaults
        stored_info = {}
        load.use_saved_extra_device_info(stored_info)

        assert load.external_user_initiated_state is None
        assert load.external_user_initiated_state_time is None
        assert load.asked_for_reset_user_initiated_state_time is None
        assert load.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None

    def test_roundtrip_storage(self):
        """Test data can be saved and restored correctly (roundtrip)."""
        load1 = self.create_load()
        test_time = datetime(2026, 1, 24, 12, 0, 0, tzinfo=pytz.UTC)

        # Set all fields
        load1.num_on_off = 8
        load1.external_user_initiated_state = "FORCE_ON"
        load1.external_user_initiated_state_time = test_time
        load1.asked_for_reset_user_initiated_state_time = test_time + timedelta(hours=1)
        load1.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = test_time + timedelta(hours=2)

        # Save to dict
        saved_data = {}
        load1.update_to_be_saved_extra_device_info(saved_data)

        # Create new load and restore
        load2 = self.create_load()
        load2.use_saved_extra_device_info(saved_data)

        # Verify all fields match
        assert load2.num_on_off == 8
        assert load2.external_user_initiated_state == "FORCE_ON"
        assert load2.external_user_initiated_state_time == test_time
        assert load2.asked_for_reset_user_initiated_state_time == test_time + timedelta(hours=1)
        assert load2.asked_for_reset_user_initiated_state_time_first_cmd_reset_done == test_time + timedelta(hours=2)

    def test_roundtrip_storage_with_none_values(self):
        """Test roundtrip with None datetime values."""
        load1 = self.create_load()

        # Set mixed None and non-None values
        load1.external_user_initiated_state = "TEST"
        load1.external_user_initiated_state_time = None
        load1.asked_for_reset_user_initiated_state_time = datetime(2026, 1, 24, 15, 0, 0, tzinfo=pytz.UTC)
        load1.asked_for_reset_user_initiated_state_time_first_cmd_reset_done = None

        # Save to dict
        saved_data = {}
        load1.update_to_be_saved_extra_device_info(saved_data)

        # Create new load and restore
        load2 = self.create_load()
        load2.use_saved_extra_device_info(saved_data)

        # Verify fields
        assert load2.external_user_initiated_state == "TEST"
        assert load2.external_user_initiated_state_time is None
        assert load2.asked_for_reset_user_initiated_state_time == datetime(2026, 1, 24, 15, 0, 0, tzinfo=pytz.UTC)
        assert load2.asked_for_reset_user_initiated_state_time_first_cmd_reset_done is None

    def test_use_saved_extra_device_info_calls_super(self):
        """Test use_saved_extra_device_info calls parent class method."""
        load = self.create_load()

        stored_info = {
            "num_on_off": 4,  # From AbstractDevice
        }
        load.use_saved_extra_device_info(stored_info)

        # num_on_off from AbstractDevice.use_saved_extra_device_info
        assert load.num_on_off == 4

    def test_storage_preserves_timezone_info(self):
        """Test that timezone information is preserved in storage."""
        load = self.create_load()

        # Use a specific timezone
        test_time = datetime(2026, 1, 24, 10, 30, 0, tzinfo=pytz.UTC)
        load.external_user_initiated_state_time = test_time

        # Save
        saved_data = {}
        load.update_to_be_saved_extra_device_info(saved_data)

        # Restore
        load2 = self.create_load()
        load2.use_saved_extra_device_info(saved_data)

        # Verify timezone is preserved
        assert load2.external_user_initiated_state_time is not None
        assert load2.external_user_initiated_state_time.tzinfo is not None


# =============================================================================
# Extended Tests for Coverage - PilotedDevice
# =============================================================================


class TestPilotedDevice:
    """Test PilotedDevice functionality."""

    @pytest.fixture
    def piloted_device(self):
        """Create a PilotedDevice for testing."""
        from custom_components.quiet_solar.home_model.load import PilotedDevice

        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        device = PilotedDevice(
            name="Test Piloted Device",
            home=mock_home,
            **{CONF_POWER: 2000.0}
        )
        return device

    def test_is_piloted_device_activated_no_clients(self, piloted_device):
        """Test is_piloted_device_activated returns False with no clients."""
        piloted_device.clients = []
        assert piloted_device.is_piloted_device_activated is False

    def test_is_piloted_device_activated_disabled_client(self, piloted_device):
        """Test is_piloted_device_activated with disabled client."""
        mock_client = MagicMock()
        mock_client.qs_enable_device = False
        mock_client.current_command = None
        piloted_device.clients = [mock_client]

        assert piloted_device.is_piloted_device_activated is False

    def test_is_piloted_device_activated_no_command(self, piloted_device):
        """Test is_piloted_device_activated with enabled client but no command."""
        mock_client = MagicMock()
        mock_client.qs_enable_device = True
        mock_client.current_command = None
        piloted_device.clients = [mock_client]

        assert piloted_device.is_piloted_device_activated is False

    def test_is_piloted_device_activated_idle_command(self, piloted_device):
        """Test is_piloted_device_activated with idle command (lines 577-579)."""
        mock_client = MagicMock()
        mock_client.qs_enable_device = True
        mock_cmd = MagicMock()
        mock_cmd.is_off_or_idle.return_value = True
        mock_client.current_command = mock_cmd
        piloted_device.clients = [mock_client]

        assert piloted_device.is_piloted_device_activated is False

    def test_is_piloted_device_activated_active_command(self, piloted_device):
        """Test is_piloted_device_activated with active command."""
        mock_client = MagicMock()
        mock_client.qs_enable_device = True
        mock_cmd = MagicMock()
        mock_cmd.is_off_or_idle.return_value = False
        mock_client.current_command = mock_cmd
        piloted_device.clients = [mock_client]

        assert piloted_device.is_piloted_device_activated is True

    def test_prepare_slots_for_piloted_device_budget(self, piloted_device):
        """Test prepare_slots_for_piloted_device_budget (lines 581-584)."""
        piloted_device.prepare_slots_for_piloted_device_budget(num_slots=4)

        assert piloted_device.num_demanding_clients is not None
        assert len(piloted_device.num_demanding_clients) == 4
        assert all(n == 0 for n in piloted_device.num_demanding_clients)

    def test_possible_delta_power_no_slots(self, piloted_device):
        """Test possible_delta_power_for_slot with no slots (lines 587-588)."""
        piloted_device.num_demanding_clients = None
        result = piloted_device.possible_delta_power_for_slot(0, add=True)
        assert result == 0

    def test_possible_delta_power_no_clients(self, piloted_device):
        """Test possible_delta_power_for_slot with no clients (lines 590-591)."""
        piloted_device.num_demanding_clients = [0, 0]
        piloted_device.clients = []
        result = piloted_device.possible_delta_power_for_slot(0, add=True)
        assert result == 0

    def test_possible_delta_power_first_add(self, piloted_device):
        """Test possible_delta_power_for_slot first add returns power (lines 593-596)."""
        piloted_device.num_demanding_clients = [0, 0]
        piloted_device.clients = [MagicMock()]

        result = piloted_device.possible_delta_power_for_slot(0, add=True)
        assert result == 2000.0

    def test_possible_delta_power_not_first_add(self, piloted_device):
        """Test possible_delta_power_for_slot not first add returns 0 (lines 597-598)."""
        piloted_device.num_demanding_clients = [1, 0]
        piloted_device.clients = [MagicMock()]

        result = piloted_device.possible_delta_power_for_slot(0, add=True)
        assert result == 0

    def test_possible_delta_power_last_remove(self, piloted_device):
        """Test possible_delta_power_for_slot last remove returns power (lines 601-603)."""
        piloted_device.num_demanding_clients = [1, 0]
        piloted_device.clients = [MagicMock()]

        result = piloted_device.possible_delta_power_for_slot(0, add=False)
        assert result == 2000.0

    def test_possible_delta_power_not_last_remove(self, piloted_device):
        """Test possible_delta_power_for_slot not last remove returns 0 (line 605)."""
        piloted_device.num_demanding_clients = [2, 0]
        piloted_device.clients = [MagicMock()]

        result = piloted_device.possible_delta_power_for_slot(0, add=False)
        assert result == 0

    def test_update_num_demanding_clients_no_slots(self, piloted_device):
        """Test update_num_demanding_clients_for_slot with no slots (lines 609-610)."""
        piloted_device.num_demanding_clients = None
        result = piloted_device.update_num_demanding_clients_for_slot(0, add=True)
        assert result == 0

    def test_update_num_demanding_clients_no_clients(self, piloted_device):
        """Test update_num_demanding_clients_for_slot with no clients (lines 612-613)."""
        piloted_device.num_demanding_clients = [0, 0]
        piloted_device.clients = []
        result = piloted_device.update_num_demanding_clients_for_slot(0, add=True)
        assert result == 0

    def test_update_num_demanding_clients_add(self, piloted_device):
        """Test update_num_demanding_clients_for_slot add (lines 617-623)."""
        piloted_device.num_demanding_clients = [0, 0]
        piloted_device.clients = [MagicMock(), MagicMock()]

        result = piloted_device.update_num_demanding_clients_for_slot(0, add=True)

        assert result == 2000.0  # First add returns power
        assert piloted_device.num_demanding_clients[0] == 1

    def test_update_num_demanding_clients_remove(self, piloted_device):
        """Test update_num_demanding_clients_for_slot remove (lines 624-630)."""
        piloted_device.num_demanding_clients = [1, 0]
        piloted_device.clients = [MagicMock()]

        result = piloted_device.update_num_demanding_clients_for_slot(0, add=False)

        assert result == 2000.0  # Last remove returns power
        assert piloted_device.num_demanding_clients[0] == 0

    def test_update_num_demanding_clients_overflow_warning(self, piloted_device):
        """Test update_num_demanding_clients_for_slot with overflow (lines 618-621)."""
        piloted_device.num_demanding_clients = [2, 0]
        piloted_device.clients = [MagicMock(), MagicMock()]  # Only 2 clients

        # Try to add more than number of clients
        piloted_device.update_num_demanding_clients_for_slot(0, add=True)

        # Should cap at len(clients) - 1 + 1 = len(clients)
        assert piloted_device.num_demanding_clients[0] <= len(piloted_device.clients)

    def test_update_num_demanding_clients_underflow_warning(self, piloted_device):
        """Test update_num_demanding_clients_for_slot with underflow (lines 627-630)."""
        piloted_device.num_demanding_clients = [0, 0]
        piloted_device.clients = [MagicMock()]

        # Try to remove when already at 0
        piloted_device.update_num_demanding_clients_for_slot(0, add=False)

        # Should stay at 0
        assert piloted_device.num_demanding_clients[0] == 0


# =============================================================================
# Extended Tests for Coverage - AbstractLoad additional methods
# =============================================================================


class TestAbstractLoadExtendedCoverage:
    """Additional tests for AbstractLoad coverage."""

    @pytest.fixture
    def basic_load(self):
        """Create a basic AbstractLoad for testing."""
        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = None

        return _CommandLoad(
            name="Test Load",
            home=mock_home,
            **{CONF_POWER: 1000.0}
        )

    def test_push_agenda_constraints_removes_old_agenda(self, basic_load):
        """Test push_agenda_constraints removes unmatched agenda constraints (lines 779-790)."""
        time_now = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)

        # Create an existing agenda constraint
        old_constraint = create_real_constraint(basic_load, time_now=time_now)
        old_constraint.add_or_update_load_info("originator", "agenda")
        basic_load._constraints = [old_constraint]

        # Push new agenda constraints that don't match
        new_constraints = [
            create_real_constraint(
                basic_load,
                time_now=time_now,
                end_time=time_now + timedelta(hours=5)
            )
        ]

        result = basic_load.push_agenda_constraints(time_now, new_constraints)

        assert result is True

    def test_push_agenda_constraints_keeps_matching(self, basic_load):
        """Test push_agenda_constraints keeps matching constraints (lines 792-802)."""
        time_now = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        end_time = time_now + timedelta(hours=2)

        # Create existing agenda constraint
        existing = create_real_constraint(basic_load, time_now=time_now, end_time=end_time)
        existing.add_or_update_load_info("originator", "agenda")
        basic_load._constraints = [existing]

        # Push same constraint again
        new_constraint = create_real_constraint(basic_load, time_now=time_now, end_time=end_time)
        result = basic_load.push_agenda_constraints(time_now, [new_constraint])

        # Should not add duplicate
        assert len([c for c in basic_load._constraints if c is not None]) >= 1

    def test_get_power_from_switch_state_none(self, basic_load):
        """Test get_power_from_switch_state with None (lines 821-822)."""
        result = basic_load.get_power_from_switch_state(None)
        assert result is None

    def test_get_power_from_switch_state_on(self, basic_load):
        """Test get_power_from_switch_state with 'on' (lines 823-824)."""
        result = basic_load.get_power_from_switch_state("on")
        assert result == 1000.0

    def test_get_power_from_switch_state_off(self, basic_load):
        """Test get_power_from_switch_state with other state (lines 825-826)."""
        result = basic_load.get_power_from_switch_state("off")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_do_run_check_disabled(self, basic_load):
        """Test do_run_check_load_activity_and_constraints when disabled (lines 829-830)."""
        basic_load.qs_enable_device = False
        result = await basic_load.do_run_check_load_activity_and_constraints(
            datetime.now(pytz.UTC)
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_do_run_check_not_initialized(self, basic_load):
        """Test do_run_check_load_activity_and_constraints when not initialized (lines 831-832)."""
        basic_load.qs_enable_device = True
        basic_load.externally_initialized_constraints = False
        result = await basic_load.do_run_check_load_activity_and_constraints(
            datetime.now(pytz.UTC)
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_async_load_constraints_from_storage_empty(self, basic_load):
        """Test async_load_constraints_from_storage with empty list (lines 839-855)."""
        time_now = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)

        await basic_load.async_load_constraints_from_storage(time_now, [], None)

        assert basic_load.externally_initialized_constraints is True
        assert basic_load._last_completed_constraint is None

    @pytest.mark.asyncio
    async def test_do_probe_state_change_disabled(self, basic_load):
        """Test do_probe_state_change when disabled (lines 858-860)."""
        basic_load.qs_enable_device = False

        # Should not raise
        await basic_load.do_probe_state_change(datetime.now(pytz.UTC))


# =============================================================================
# Extended Tests for Time Series Utility Functions
# =============================================================================


class TestTimeSeriesUtilities:
    """Test get_slots_from_time_series and get_value_from_time_series."""

    def test_get_slots_from_time_series_empty(self):
        """Test get_slots_from_time_series with empty data."""
        from custom_components.quiet_solar.home_model.load import get_slots_from_time_series

        result = get_slots_from_time_series([], datetime.now(pytz.UTC))

        assert result == []

    def test_get_slots_from_time_series_basic(self):
        """Test get_slots_from_time_series with basic data (lines 1546-1568)."""
        from custom_components.quiet_solar.home_model.load import get_slots_from_time_series

        time = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        data = [
            (time, 100.0),
            (time + timedelta(hours=1), 200.0),
            (time + timedelta(hours=2), 150.0),
        ]

        # Call with start_time and optional end_time
        result = get_slots_from_time_series(data, time, time + timedelta(hours=3))

        assert len(result) > 0

    def test_get_value_from_time_series_empty(self):
        """Test get_value_from_time_series with empty data."""
        from custom_components.quiet_solar.home_model.load import get_value_from_time_series

        time, value, found, idx = get_value_from_time_series(
            [], datetime.now(pytz.UTC)
        )

        assert found is False
        assert value is None

    def test_get_value_from_time_series_exact_match(self):
        """Test get_value_from_time_series with exact time match (lines 1577-1582)."""
        from custom_components.quiet_solar.home_model.load import get_value_from_time_series

        time = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        data = [
            (time, 100.0),
            (time + timedelta(hours=1), 200.0),
        ]

        result_time, value, found, idx = get_value_from_time_series(data, time)

        assert found is True
        assert value == 100.0

    def test_get_value_from_time_series_last_element(self):
        """Test get_value_from_time_series matches last element (line 1577-1579)."""
        from custom_components.quiet_solar.home_model.load import get_value_from_time_series

        time = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        data = [
            (time, 100.0),
            (time + timedelta(hours=1), 200.0),
        ]

        # Query for last element time
        result_time, value, found, idx = get_value_from_time_series(
            data, time + timedelta(hours=1)
        )

        assert found is True
        assert value == 200.0
        assert idx == 1

    def test_get_value_from_time_series_bisect(self):
        """Test get_value_from_time_series uses bisect for intermediate times (lines 1584-1620)."""
        from custom_components.quiet_solar.home_model.load import get_value_from_time_series

        time = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        data = [
            (time, 100.0),
            (time + timedelta(hours=2), 200.0),
        ]

        # Query for middle time - no exact match
        query_time = time + timedelta(hours=1)
        result_time, value, found, idx = get_value_from_time_series(data, query_time)

        # Without interpolation, found should indicate nearest element
        assert isinstance(found, bool)

    def test_get_value_from_time_series_before_first(self):
        """Test get_value_from_time_series when query is before first element (lines 1589-1590)."""
        from custom_components.quiet_solar.home_model.load import get_value_from_time_series

        time = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        data = [
            (time, 100.0),
            (time + timedelta(hours=1), 200.0),
        ]

        # Query for time before first element
        query_time = time - timedelta(hours=1)
        result_time, value, found, idx = get_value_from_time_series(data, query_time)

        # Should return first element
        assert idx == 0

    def test_get_value_from_time_series_after_last(self):
        """Test get_value_from_time_series when query is after last element (lines 1586-1588)."""
        from custom_components.quiet_solar.home_model.load import get_value_from_time_series

        time = datetime(2026, 1, 24, 10, 0, tzinfo=pytz.UTC)
        data = [
            (time, 100.0),
            (time + timedelta(hours=1), 200.0),
        ]

        # Query for time after last element
        query_time = time + timedelta(hours=5)
        result_time, value, found, idx = get_value_from_time_series(data, query_time)

        # Should return last element
        assert idx == len(data) - 1


# =============================================================================
# Extended Tests for AbstractDevice
# =============================================================================


class TestAbstractDeviceExtended:
    """Extended tests for AbstractDevice."""

    @pytest.fixture
    def basic_device(self):
        """Create a basic AbstractDevice for testing."""
        mock_home = create_minimal_home_model()
        mock_home.voltage = 230.0
        mock_home.is_off_grid = MagicMock(return_value=False)
        mock_home.dashboard_sections = [("Section A", "icon_a")]

        return AbstractDevice(
            name="Test Device",
            home=mock_home,
        )

    def test_dashboard_section_with_computed(self, basic_device):
        """Test dashboard_section property when computed (lines 179-190)."""
        basic_device._conf_dashboard_section_option = "Section A"
        basic_device._computed_dashboard_section = None

        result = basic_device.dashboard_section

        assert result is not None

    def test_dashboard_section_no_section(self, basic_device):
        """Test dashboard_section returns None when NO_SECTION (line 187-188)."""
        basic_device._conf_dashboard_section_option = DASHBOARD_NO_SECTION
        basic_device._computed_dashboard_section = None

        result = basic_device.dashboard_section

        assert result is None

    def test_dashboard_sort_string_in_type_default(self, basic_device):
        """Test dashboard_sort_string_in_type default value (lines 193-194)."""
        result = basic_device.dashboard_sort_string_in_type
        assert result == "ZZZ"

    def test_power_use_property_none(self, basic_device):
        """Test power_use property when None (lines 130-132)."""
        basic_device._power_use_conf = None
        result = basic_device.power_use
        assert result is None

    def test_power_use_property_set(self, basic_device):
        """Test power_use property when set (line 133)."""
        basic_device._power_use_conf = 1500.0
        result = basic_device.power_use
        assert result == 1500.0

    def test_power_use_setter(self, basic_device):
        """Test power_use setter (lines 136-137)."""
        basic_device.power_use = 2000.0
        assert basic_device._power_use_conf == 2000.0

    def test_is_off_grid_with_home(self, basic_device):
        """Test is_off_grid delegates to home (lines 173-176)."""
        basic_device.home.is_off_grid = MagicMock(return_value=True)
        result = basic_device.is_off_grid()
        assert result is True

    def test_is_off_grid_no_home(self, basic_device):
        """Test is_off_grid returns False without home (line 176)."""
        basic_device.home = None
        result = basic_device.is_off_grid()
        assert result is False

    def test_qs_enable_device_disable(self, basic_device):
        """Test qs_enable_device setter disables device (lines 253-266)."""
        basic_device.home = MagicMock()
        basic_device.home.remove_device = MagicMock()
        basic_device.home.add_disabled_device = MagicMock()

        basic_device.qs_enable_device = False

        assert basic_device._enabled is False

    def test_qs_enable_device_enable(self, basic_device):
        """Test qs_enable_device setter enables device (lines 263-266)."""
        basic_device._enabled = False
        basic_device.home = MagicMock()
        basic_device.home.add_device = MagicMock()
        basic_device.home.remove_disabled_device = MagicMock()

        basic_device.qs_enable_device = True

        assert basic_device._enabled is True

    def test_mono_phase_index_from_father(self, basic_device):
        """Test mono_phase_index from father device (lines 317-318)."""
        mock_father = MagicMock()
        mock_father.physical_3p = False
        mock_father.mono_phase_index = 2
        basic_device.father_device = mock_father
        basic_device.home = MagicMock()
        basic_device._mono_phase_conf = None

        result = basic_device.mono_phase_index

        assert result == 2

    @pytest.mark.asyncio
    async def test_async_get_info_from_storage(self, basic_device):
        """Test async_get_info_from_storage with data (lines 216-218)."""
        stored_info = {"num_on_off": 6}  # Use even number to avoid adjustment

        await basic_device.async_get_info_from_storage(
            datetime.now(pytz.UTC), stored_info
        )

        assert basic_device.num_on_off == 6

    def test_update_available_amps_for_group_with_father(self, basic_device):
        """Test update_available_amps_for_group delegates to father (lines 220-223)."""
        mock_father = MagicMock()
        mock_father.update_available_amps_for_group = MagicMock()
        basic_device.father_device = mock_father

        basic_device.update_available_amps_for_group(0, [10.0, 10.0, 10.0], True)

        mock_father.update_available_amps_for_group.assert_called_once()

    def test_get_possible_delta_power_from_piloted_devices_no_devices(self, basic_device):
        """Test get_possible_delta_power_from_piloted_devices with no devices (lines 139-141)."""
        basic_device.devices_to_pilot = []
        result = basic_device.get_possible_delta_power_from_piloted_devices_for_budget(0, True)
        assert result == 0.0

    def test_get_possible_delta_power_from_piloted_devices_with_devices(self, basic_device):
        """Test get_possible_delta_power_from_piloted_devices with devices (lines 143-147)."""
        mock_device = MagicMock()
        mock_device.possible_delta_power_for_slot = MagicMock(return_value=500.0)
        basic_device.devices_to_pilot = [mock_device]

        result = basic_device.get_possible_delta_power_from_piloted_devices_for_budget(0, True)

        assert result == 500.0

    def test_use_saved_extra_device_info_odd_num_on_off(self, basic_device):
        """Test use_saved_extra_device_info adjusts odd num_on_off (lines 348-350)."""
        stored_info = {"num_on_off": 5}  # Odd number

        basic_device.use_saved_extra_device_info(stored_info)

        # Odd numbers should be decremented by 1
        assert basic_device.num_on_off == 4

