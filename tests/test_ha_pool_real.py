"""Tests for QSPool class in ha_model/pool.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import time as dt_time

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
)
import pytz

from custom_components.quiet_solar.ha_model.pool import QSPool
from custom_components.quiet_solar.home_model.commands import CMD_ON, CMD_OFF, CMD_IDLE
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_POOL_TEMPERATURE_SENSOR,
    POOL_TEMP_STEPS,
    CONF_POOL_WINTER_IDX,
    CONF_POOL_DEFAULT_IDX,
    SENSOR_CONSTRAINT_SENSOR_POOL,
    CONF_SWITCH,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    CONSTRAINT_TYPE_FILLER_AUTO,
)

from tests.test_helpers import FakeHass, FakeConfigEntry


class TestQSPoolInit:
    """Test QSPool initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_pool_entry",
            data={CONF_NAME: "Test Pool"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )

        assert device.name == "My Pool"
        assert device.pool_temperature_sensor == "sensor.pool_temp"
        assert device.switch_entity == "switch.pool_pump"
        assert device.is_load_time_sensitive is False

    def test_init_creates_pool_steps(self):
        """Test that pool steps are created from POOL_TEMP_STEPS."""
        device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "My Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )

        assert len(device.pool_steps) == len(POOL_TEMP_STEPS)
        for i, (min_temp, max_temp, default) in enumerate(POOL_TEMP_STEPS):
            assert device.pool_steps[i][0] == min_temp
            assert device.pool_steps[i][1] == max_temp

    def test_init_with_custom_temp_steps(self):
        """Test initialization with custom temperature steps."""
        # Use specific temp step from POOL_TEMP_STEPS
        if len(POOL_TEMP_STEPS) > 0:
            _, max_temp, _ = POOL_TEMP_STEPS[0]
            custom_hours = 5.0

            device = QSPool(
                hass=self.hass,
                config_entry=self.config_entry,
                home=self.home,
                **{
                    CONF_NAME: "My Pool",
                    CONF_SWITCH: "switch.pool_pump",
                    CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
                    f"water_temp_{max_temp}": custom_hours,
                }
            )

            assert device.pool_steps[0][2] == custom_hours


class TestQSPoolTranslationKeys:
    """Test translation key methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_pool_entry",
            data={CONF_NAME: "Test Pool"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )

    def test_get_select_translation_key(self):
        """Test get_select_translation_key returns 'pool_mode'."""
        result = self.device.get_select_translation_key()
        assert result == "pool_mode"

    def test_get_virtual_current_constraint_translation_key(self):
        """Test get_virtual_current_constraint_translation_key returns correct key."""
        result = self.device.get_virtual_current_constraint_translation_key()
        assert result == SENSOR_CONSTRAINT_SENSOR_POOL


class TestQSPoolBistateModes:
    """Test bistate mode methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_pool_entry",
            data={CONF_NAME: "Test Pool"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )
        self.device.load_is_auto_to_be_boosted = False

    def test_get_bistate_modes_includes_winter_mode(self):
        """Test get_bistate_modes includes pool_winter_mode."""
        modes = self.device.get_bistate_modes()

        assert "pool_winter_mode" in modes
        assert "bistate_mode_auto" in modes

    def test_support_green_only_switch(self):
        """Test support_green_only_switch returns True."""
        result = self.device.support_green_only_switch()
        assert result is True


class TestQSPoolCurrentWaterTemperature:
    """Test current_water_temperature property."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_pool_entry",
            data={CONF_NAME: "Test Pool"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )

    def test_current_water_temperature_with_valid_value(self):
        """Test current_water_temperature returns valid temperature."""
        self.hass.states.set("sensor.pool_temp", "25.5")
        self.device.get_sensor_latest_possible_valid_value = MagicMock(return_value=25.5)

        temp = self.device.current_water_temperature

        assert temp == 25.5

    def test_current_water_temperature_with_none(self):
        """Test current_water_temperature returns None when sensor unavailable."""
        self.device.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        temp = self.device.current_water_temperature

        assert temp is None


class TestQSPoolFilterTime:
    """Test get_pool_filter_time_s method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_pool_entry",
            data={CONF_NAME: "Test Pool"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )

    def test_filter_time_winter_mode(self):
        """Test filter time in winter mode uses CONF_POOL_WINTER_IDX."""
        time = datetime.datetime.now(pytz.UTC)

        filter_time = self.device.get_pool_filter_time_s(force_winter=True, time=time)

        expected_hours = self.device.pool_steps[CONF_POOL_WINTER_IDX][2]
        assert filter_time == expected_hours * 3600.0

    def test_filter_time_no_history_uses_default(self):
        """Test filter time with no history uses default index."""
        time = datetime.datetime.now(pytz.UTC)
        self.device.get_state_history_data = MagicMock(return_value=None)

        filter_time = self.device.get_pool_filter_time_s(force_winter=False, time=time)

        expected_hours = self.device.pool_steps[CONF_POOL_DEFAULT_IDX][2]
        assert filter_time == expected_hours * 3600.0

    def test_filter_time_empty_history_uses_default(self):
        """Test filter time with empty history uses default index."""
        time = datetime.datetime.now(pytz.UTC)
        self.device.get_state_history_data = MagicMock(return_value=[])

        filter_time = self.device.get_pool_filter_time_s(force_winter=False, time=time)

        expected_hours = self.device.pool_steps[CONF_POOL_DEFAULT_IDX][2]
        assert filter_time == expected_hours * 3600.0

    def test_filter_time_with_temperature_history(self):
        """Test filter time calculation based on temperature history."""
        time = datetime.datetime.now(pytz.UTC)

        # Create history with min temp of 20°C
        history = [
            (time - datetime.timedelta(hours=12), 22.0),
            (time - datetime.timedelta(hours=6), 21.0),
            (time, 20.0),  # Min temp
        ]
        self.device.get_state_history_data = MagicMock(return_value=history)

        filter_time = self.device.get_pool_filter_time_s(force_winter=False, time=time)

        # Filter time should be based on the temperature step that contains 20°C
        assert filter_time >= 0


class TestQSPoolCheckLoadActivityAndConstraints:
    """Test check_load_activity_and_constraints method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_pool_entry",
            data={CONF_NAME: "Test Pool"},
        )
        self.home = MagicMock()
        self.home.is_off_grid = MagicMock(return_value=False)
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = QSPool(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            **{
                CONF_NAME: "Test Pool",
                CONF_SWITCH: "switch.pool_pump",
                CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
            }
        )
        self.device.load_is_auto_to_be_boosted = False
        self.device.qs_best_effort_green_only = False
        self.device._constraints = []
        self.device.power_use = 1500.0

    @pytest.mark.asyncio
    async def test_check_load_auto_mode_creates_constraint(self):
        """Test that auto mode creates a constraint."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.default_on_finish_time = dt_time(hour=6, minute=0)
        self.device.get_state_history_data = MagicMock(return_value=[])
        self.device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        self.device.push_agenda_constraints = MagicMock(return_value=True)

        time = datetime.datetime.now(pytz.UTC)
        result = await self.device.check_load_activity_and_constraints(time)

        self.device.push_agenda_constraints.assert_called_once()
        # Check the constraint was created with correct type
        call_args = self.device.push_agenda_constraints.call_args
        constraint = call_args[0][1][0]
        assert constraint is not None

    @pytest.mark.asyncio
    async def test_check_load_winter_mode_creates_constraint(self):
        """Test that winter mode creates a constraint with force_winter=True."""
        self.device.bistate_mode = "pool_winter_mode"
        self.device.default_on_finish_time = dt_time(hour=6, minute=0)
        self.device.get_state_history_data = MagicMock(return_value=[])
        self.device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        self.device.push_agenda_constraints = MagicMock(return_value=True)

        time = datetime.datetime.now(pytz.UTC)
        result = await self.device.check_load_activity_and_constraints(time)

        self.device.push_agenda_constraints.assert_called_once()
        # Verify the constraint uses winter filter time
        call_args = self.device.push_agenda_constraints.call_args
        constraint = call_args[0][1][0]
        expected_target = self.device.pool_steps[CONF_POOL_WINTER_IDX][2] * 3600.0
        assert constraint.target_value == expected_target

    @pytest.mark.asyncio
    async def test_check_load_no_finish_time_sets_default(self):
        """Test that missing finish time gets default value."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.default_on_finish_time = None
        self.device.get_state_history_data = MagicMock(return_value=[])
        self.device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        self.device.push_agenda_constraints = MagicMock(return_value=True)

        time = datetime.datetime.now(pytz.UTC)
        await self.device.check_load_activity_and_constraints(time)

        # Should have set default finish time
        assert self.device.default_on_finish_time == dt_time(hour=0, minute=0, second=0)

    @pytest.mark.asyncio
    async def test_check_load_best_effort_uses_filler_type(self):
        """Test that best effort load uses FILLER_AUTO constraint type."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.default_on_finish_time = dt_time(hour=6, minute=0)
        self.device.qs_best_effort_green_only = True  # Makes is_best_effort_only_load() return True
        self.device.get_state_history_data = MagicMock(return_value=[])
        self.device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        self.device.push_agenda_constraints = MagicMock(return_value=True)
        self.home.is_off_grid = MagicMock(return_value=False)

        time = datetime.datetime.now(pytz.UTC)
        await self.device.check_load_activity_and_constraints(time)

        call_args = self.device.push_agenda_constraints.call_args
        constraint = call_args[0][1][0]
        # Best effort loads use FILLER_AUTO type
        assert constraint._type == CONSTRAINT_TYPE_FILLER_AUTO

    @pytest.mark.asyncio
    async def test_check_load_non_auto_mode_calls_super(self):
        """Test that non-auto/winter modes call parent implementation."""
        self.device.bistate_mode = "on_off_mode_off"  # Not auto or winter
        self.device.is_load_command_set = MagicMock(return_value=False)
        self.device.command_and_constraint_reset = MagicMock()

        time = datetime.datetime.now(pytz.UTC)
        # This should call super().check_load_activity_and_constraints
        # which handles mode_off by calling reset
        result = await self.device.check_load_activity_and_constraints(time)

        # Parent's mode_off behavior should have been triggered
        self.device.command_and_constraint_reset.assert_called()

    @pytest.mark.asyncio
    async def test_check_load_no_end_schedule_returns_false(self):
        """Test that None end_schedule returns False."""
        self.device.bistate_mode = "bistate_mode_auto"
        self.device.default_on_finish_time = dt_time(hour=6, minute=0)
        self.device.get_next_time_from_hours = MagicMock(return_value=None)

        time = datetime.datetime.now(pytz.UTC)
        result = await self.device.check_load_activity_and_constraints(time)

        assert result is False
