"""Tests for QSPool class in ha_model/pool.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock

from tests.factories import create_minimal_home_model
from datetime import time as dt_time

from homeassistant.const import (
    Platform,
    STATE_UNKNOWN,
    STATE_UNAVAILABLE,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

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


@pytest.fixture
def pool_config_entry() -> MockConfigEntry:
    """Config entry for pool tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_pool_entry",
        data={CONF_NAME: "Test Pool"},
        title="Test Pool",
    )


@pytest.fixture
def pool_home():
    """Mock home for pool tests."""
    home = create_minimal_home_model()
    home.is_off_grid = MagicMock(return_value=False)
    return home


@pytest.fixture
def pool_data_handler(pool_home):
    """Data handler for pool tests."""
    handler = MagicMock()
    handler.home = pool_home
    return handler


@pytest.fixture
def pool_hass_data(hass: HomeAssistant, pool_data_handler):
    """Set hass.data[DOMAIN][DATA_HANDLER] for pool tests."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_HANDLER] = pool_data_handler


@pytest.fixture
def pool_device(hass, pool_config_entry, pool_home, pool_data_handler, pool_hass_data):
    """QSPool instance for tests."""
    return QSPool(
        hass=hass,
        config_entry=pool_config_entry,
        home=pool_home,
        **{
            CONF_NAME: "Test Pool",
            CONF_SWITCH: "switch.pool_pump",
            CONF_POOL_TEMPERATURE_SENSOR: "sensor.pool_temp",
        }
    )


class TestQSPoolInit:
    """Test QSPool initialization."""

    def test_init_with_required_params(
        self, hass, pool_config_entry, pool_home, pool_data_handler, pool_hass_data
    ):
        """Test initialization with required parameters."""
        device = QSPool(
            hass=hass,
            config_entry=pool_config_entry,
            home=pool_home,
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

    def test_init_creates_pool_steps(
        self, hass, pool_config_entry, pool_home, pool_data_handler, pool_hass_data
    ):
        """Test that pool steps are created from POOL_TEMP_STEPS."""
        device = QSPool(
            hass=hass,
            config_entry=pool_config_entry,
            home=pool_home,
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

    def test_init_with_custom_temp_steps(
        self, hass, pool_config_entry, pool_home, pool_data_handler, pool_hass_data
    ):
        """Test initialization with custom temperature steps."""
        if len(POOL_TEMP_STEPS) > 0:
            _, max_temp, _ = POOL_TEMP_STEPS[0]
            custom_hours = 5.0

            device = QSPool(
                hass=hass,
                config_entry=pool_config_entry,
                home=pool_home,
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

    def test_get_select_translation_key(self, pool_device):
        """Test get_select_translation_key returns 'pool_mode'."""
        result = pool_device.get_select_translation_key()
        assert result == "pool_mode"

    def test_get_virtual_current_constraint_translation_key(self, pool_device):
        """Test get_virtual_current_constraint_translation_key returns correct key."""
        result = pool_device.get_virtual_current_constraint_translation_key()
        assert result == SENSOR_CONSTRAINT_SENSOR_POOL


class TestQSPoolBistateModes:
    """Test bistate mode methods."""

    def test_get_bistate_modes_includes_winter_mode(self, pool_device):
        """Test get_bistate_modes includes pool_winter_mode."""
        pool_device.load_is_auto_to_be_boosted = False
        modes = pool_device.get_bistate_modes()

        assert "pool_winter_mode" in modes
        assert "bistate_mode_auto" in modes

    def test_support_green_only_switch(self, pool_device):
        """Test support_green_only_switch returns True."""
        result = pool_device.support_green_only_switch()
        assert result is True


class TestQSPoolCurrentWaterTemperature:
    """Test current_water_temperature property."""

    def test_current_water_temperature_with_valid_value(self, pool_device, hass):
        """Test current_water_temperature returns valid temperature."""
        hass.states.async_set("sensor.pool_temp", "25.5")
        pool_device.get_sensor_latest_possible_valid_value = MagicMock(return_value=25.5)

        temp = pool_device.current_water_temperature

        assert temp == 25.5

    def test_current_water_temperature_with_none(self, pool_device):
        """Test current_water_temperature returns None when sensor unavailable."""
        pool_device.get_sensor_latest_possible_valid_value = MagicMock(return_value=None)

        temp = pool_device.current_water_temperature

        assert temp is None


class TestQSPoolFilterTime:
    """Test get_pool_filter_time_s method."""

    def test_filter_time_winter_mode(self, pool_device):
        """Test filter time in winter mode uses CONF_POOL_WINTER_IDX."""
        time = datetime.datetime.now(pytz.UTC)

        filter_time = pool_device.get_pool_filter_time_s(force_winter=True, time=time)

        expected_hours = pool_device.pool_steps[CONF_POOL_WINTER_IDX][2]
        assert filter_time == expected_hours * 3600.0

    def test_filter_time_no_history_uses_default(self, pool_device):
        """Test filter time with no history uses default index."""
        time = datetime.datetime.now(pytz.UTC)
        pool_device.get_state_history_data = MagicMock(return_value=None)

        filter_time = pool_device.get_pool_filter_time_s(force_winter=False, time=time)

        expected_hours = pool_device.pool_steps[CONF_POOL_DEFAULT_IDX][2]
        assert filter_time == expected_hours * 3600.0

    def test_filter_time_empty_history_uses_default(self, pool_device):
        """Test filter time with empty history uses default index."""
        time = datetime.datetime.now(pytz.UTC)
        pool_device.get_state_history_data = MagicMock(return_value=[])

        filter_time = pool_device.get_pool_filter_time_s(force_winter=False, time=time)

        expected_hours = pool_device.pool_steps[CONF_POOL_DEFAULT_IDX][2]
        assert filter_time == expected_hours * 3600.0

    def test_filter_time_with_temperature_history(self, pool_device):
        """Test filter time calculation based on temperature history."""
        time = datetime.datetime.now(pytz.UTC)

        history = [
            (time - datetime.timedelta(hours=12), 22.0),
            (time - datetime.timedelta(hours=6), 21.0),
            (time, 20.0),
        ]
        pool_device.get_state_history_data = MagicMock(return_value=history)

        filter_time = pool_device.get_pool_filter_time_s(force_winter=False, time=time)

        assert filter_time >= 0


class TestQSPoolCheckLoadActivityAndConstraints:
    """Test check_load_activity_and_constraints method."""

    @pytest.fixture(autouse=True)
    def _pool_device_attrs(self, pool_device, pool_home):
        """Set extra attributes on pool_device for check_load tests."""
        pool_device.load_is_auto_to_be_boosted = False
        pool_device.qs_best_effort_green_only = False
        pool_device._constraints = []
        pool_device.power_use = 1500.0

    @pytest.mark.asyncio
    async def test_check_load_auto_mode_creates_constraint(self, pool_device):
        """Test that auto mode creates a constraint."""
        pool_device.bistate_mode = "bistate_mode_auto"
        pool_device.default_on_finish_time = dt_time(hour=6, minute=0)
        pool_device.get_state_history_data = MagicMock(return_value=[])
        pool_device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        pool_device.push_live_constraint = MagicMock(return_value=True)

        time = datetime.datetime.now(pytz.UTC)
        result = await pool_device.check_load_activity_and_constraints(time)

        pool_device.push_live_constraint.assert_called_once()
        call_args = pool_device.push_live_constraint.call_args
        constraint = call_args[0][1]
        assert constraint is not None

    @pytest.mark.asyncio
    async def test_check_load_winter_mode_creates_constraint(self, pool_device):
        """Test that winter mode creates a constraint with force_winter=True."""
        pool_device.bistate_mode = "pool_winter_mode"
        pool_device.default_on_finish_time = dt_time(hour=6, minute=0)
        pool_device.get_state_history_data = MagicMock(return_value=[])
        pool_device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        pool_device.push_live_constraint = MagicMock(return_value=True)

        time = datetime.datetime.now(pytz.UTC)
        result = await pool_device.check_load_activity_and_constraints(time)

        pool_device.push_live_constraint.assert_called_once()
        call_args = pool_device.push_live_constraint.call_args
        constraint = call_args[0][1]
        expected_target = pool_device.pool_steps[CONF_POOL_WINTER_IDX][2] * 3600.0
        assert constraint.target_value == expected_target

    @pytest.mark.asyncio
    async def test_check_load_no_finish_time_sets_default(self, pool_device):
        """Test that missing finish time gets default value."""
        pool_device.bistate_mode = "bistate_mode_auto"
        pool_device.default_on_finish_time = None
        pool_device.get_state_history_data = MagicMock(return_value=[])
        pool_device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        pool_device.push_agenda_constraints = MagicMock(return_value=True)

        time = datetime.datetime.now(pytz.UTC)
        await pool_device.check_load_activity_and_constraints(time)

        assert pool_device.default_on_finish_time == dt_time(hour=0, minute=0, second=0)

    @pytest.mark.asyncio
    async def test_check_load_best_effort_uses_filler_type(self, pool_device, pool_home):
        """Test that best effort load uses FILLER_AUTO constraint type."""
        pool_device.bistate_mode = "bistate_mode_auto"
        pool_device.default_on_finish_time = dt_time(hour=6, minute=0)
        pool_device.qs_best_effort_green_only = True
        pool_device.get_state_history_data = MagicMock(return_value=[])
        pool_device.get_next_time_from_hours = MagicMock(
            return_value=datetime.datetime.now(pytz.UTC) + datetime.timedelta(hours=6)
        )
        pool_device.push_live_constraint = MagicMock(return_value=True)
        pool_home.is_off_grid = MagicMock(return_value=False)

        time = datetime.datetime.now(pytz.UTC)
        await pool_device.check_load_activity_and_constraints(time)

        call_args = pool_device.push_live_constraint.call_args
        constraint = call_args[0][1]
        assert constraint._type == CONSTRAINT_TYPE_FILLER_AUTO

    @pytest.mark.asyncio
    async def test_check_load_non_auto_mode_calls_super(self, pool_device):
        """Test that non-auto/winter modes call parent implementation."""
        pool_device.bistate_mode = "on_off_mode_off"
        pool_device.is_load_command_set = MagicMock(return_value=False)
        pool_device.constraint_reset_and_reset_commands_if_needed = MagicMock()

        time = datetime.datetime.now(pytz.UTC)
        result = await pool_device.check_load_activity_and_constraints(time)

        pool_device.constraint_reset_and_reset_commands_if_needed.assert_called()

    @pytest.mark.asyncio
    async def test_check_load_no_end_schedule_returns_false(self, pool_device):
        """Test that None end_schedule returns False."""
        pool_device.bistate_mode = "bistate_mode_auto"
        pool_device.default_on_finish_time = dt_time(hour=6, minute=0)
        pool_device.get_next_time_from_hours = MagicMock(return_value=None)

        time = datetime.datetime.now(pytz.UTC)
        result = await pool_device.check_load_activity_and_constraints(time)

        assert result is False
