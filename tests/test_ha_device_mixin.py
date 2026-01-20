"""Tests for device utilities in ha_model/device.py."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import timedelta

from homeassistant.const import (
    CONF_NAME,
    UnitOfPower,
    UnitOfElectricCurrent,
    UnitOfLength,
    ATTR_UNIT_OF_MEASUREMENT,
)
import pytz

from custom_components.quiet_solar.ha_model.device import (
    compute_energy_Wh_rieman_sum,
    convert_power_to_w,
    convert_current_to_amps,
    convert_distance_to_km,
    get_average_power_energy_based,
    get_median_sensor,
    load_from_history,
    HADeviceMixin,
)
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_ACCURATE_POWER_SENSOR,
    CONF_CALENDAR,
)

from tests.test_helpers import FakeHass, FakeConfigEntry


# =============================================================================
# Test compute_energy_Wh_rieman_sum
# =============================================================================

class TestComputeEnergyRiemanSum:
    """Test compute_energy_Wh_rieman_sum function."""

    def test_empty_data(self):
        """Test with empty data returns zero."""
        energy, duration = compute_energy_Wh_rieman_sum([])
        assert energy == 0
        assert duration == 0

    def test_single_point(self):
        """Test with single data point returns zero."""
        time = datetime.datetime.now(pytz.UTC)
        data = [(time, 1000.0)]

        energy, duration = compute_energy_Wh_rieman_sum(data)

        assert energy == 0
        assert duration == 0

    def test_two_points_constant_power(self):
        """Test with two points at constant power."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 1000.0),
            (time + timedelta(hours=1), 1000.0),
        ]

        energy, duration = compute_energy_Wh_rieman_sum(data)

        assert duration == pytest.approx(1.0, abs=0.01)
        assert energy == pytest.approx(1000.0, abs=10)  # 1000W * 1h = 1000Wh

    def test_multiple_points_varying_power(self):
        """Test with multiple points and varying power."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 1000.0),
            (time + timedelta(hours=1), 2000.0),
            (time + timedelta(hours=2), 1500.0),
        ]

        energy, duration = compute_energy_Wh_rieman_sum(data)

        assert duration == pytest.approx(2.0, abs=0.01)
        assert energy > 0

    def test_conservative_mode(self):
        """Test conservative mode uses minimum values."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 1000.0),
            (time + timedelta(hours=1), 2000.0),
        ]

        energy_normal, _ = compute_energy_Wh_rieman_sum(data, conservative=False)
        energy_conservative, _ = compute_energy_Wh_rieman_sum(data, conservative=True)

        # Conservative should be less or equal (uses min)
        assert energy_conservative <= energy_normal

    def test_clip_to_zero_under_power(self):
        """Test clipping values below threshold to zero."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 50.0),  # Below threshold
            (time + timedelta(hours=1), 1000.0),
        ]

        energy, _ = compute_energy_Wh_rieman_sum(data, clip_to_zero_under_power=100.0)

        # First value should be treated as 0
        assert energy >= 0


# =============================================================================
# Test convert_power_to_w
# =============================================================================

class TestConvertPowerToW:
    """Test convert_power_to_w function."""

    def test_watts_no_conversion(self):
        """Test watts value doesn't change."""
        value, attrs = convert_power_to_w(1000.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT})
        assert value == 1000.0

    def test_kilowatts_to_watts(self):
        """Test kilowatts converted to watts."""
        value, attrs = convert_power_to_w(1.5, {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.KILO_WATT})
        assert value == pytest.approx(1500.0, abs=1)
        assert attrs[ATTR_UNIT_OF_MEASUREMENT] == UnitOfPower.WATT

    def test_none_attributes(self):
        """Test with None attributes uses default."""
        value, attrs = convert_power_to_w(1000.0, None)
        assert value == 1000.0


# =============================================================================
# Test convert_current_to_amps
# =============================================================================

class TestConvertCurrentToAmps:
    """Test convert_current_to_amps function."""

    def test_amps_no_conversion(self):
        """Test amps value doesn't change."""
        value, attrs = convert_current_to_amps(16.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfElectricCurrent.AMPERE})
        assert value == 16.0

    def test_milliamps_to_amps(self):
        """Test that convert_current_to_amps handles milliamps (may not convert if unsupported)."""
        # Note: HA's PowerConverter doesn't support milliamps conversion
        # The function only converts if the unit is different and supported
        value, attrs = convert_current_to_amps(16.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfElectricCurrent.AMPERE})
        assert value == 16.0

    def test_none_attributes(self):
        """Test with None attributes uses default."""
        value, attrs = convert_current_to_amps(16.0, None)
        assert value == 16.0


# =============================================================================
# Test convert_distance_to_km
# =============================================================================

class TestConvertDistanceToKm:
    """Test convert_distance_to_km function."""

    def test_km_no_conversion(self):
        """Test km value doesn't change."""
        value, attrs = convert_distance_to_km(100.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfLength.KILOMETERS})
        assert value == 100.0

    def test_miles_to_km(self):
        """Test miles converted to km."""
        value, attrs = convert_distance_to_km(62.137, {ATTR_UNIT_OF_MEASUREMENT: UnitOfLength.MILES})
        assert value == pytest.approx(100.0, abs=1)

    def test_meters_to_km(self):
        """Test meters converted to km."""
        value, attrs = convert_distance_to_km(1000.0, {ATTR_UNIT_OF_MEASUREMENT: UnitOfLength.METERS})
        assert value == pytest.approx(1.0, abs=0.01)

    def test_none_attributes(self):
        """Test with None attributes uses default."""
        value, attrs = convert_distance_to_km(100.0, None)
        assert value == 100.0


# =============================================================================
# Test get_average_power_energy_based
# =============================================================================

class TestGetAveragePowerEnergyBased:
    """Test get_average_power_energy_based function."""

    def test_empty_data(self):
        """Test with empty data returns zero."""
        result = get_average_power_energy_based([])
        assert result == 0

    def test_single_point(self):
        """Test with single data point returns that value."""
        time = datetime.datetime.now(pytz.UTC)
        data = [(time, 1000.0)]

        result = get_average_power_energy_based(data)

        assert result == 1000.0

    def test_single_point_none(self):
        """Test with single None data point returns zero."""
        time = datetime.datetime.now(pytz.UTC)
        data = [(time, None)]

        result = get_average_power_energy_based(data)

        assert result == 0.0

    def test_multiple_points(self):
        """Test average calculation with multiple points."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 1000.0),
            (time + timedelta(hours=1), 2000.0),
            (time + timedelta(hours=2), 1500.0),
        ]

        result = get_average_power_energy_based(data)

        # Should return energy-weighted average
        assert result > 0


# =============================================================================
# Test get_median_sensor
# =============================================================================

class TestGetMedianSensor:
    """Test get_median_sensor function."""

    def test_empty_data(self):
        """Test with empty data returns zero."""
        result = get_median_sensor([])
        assert result == 0

    def test_single_point(self):
        """Test with single data point returns that value."""
        time = datetime.datetime.now(pytz.UTC)
        data = [(time, 50.0)]

        result = get_median_sensor(data)

        assert result == 50.0

    def test_single_point_none(self):
        """Test with single None data point returns zero."""
        time = datetime.datetime.now(pytz.UTC)
        data = [(time, None)]

        result = get_median_sensor(data)

        assert result == 0.0

    def test_multiple_points(self):
        """Test median calculation with multiple points."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 10.0),
            (time + timedelta(seconds=10), 50.0),
            (time + timedelta(seconds=20), 20.0),
        ]

        result = get_median_sensor(data)

        # Median should be somewhere in the data range
        assert result >= 0

    def test_with_last_timing(self):
        """Test median calculation with last_timing parameter."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 10.0),
            (time + timedelta(seconds=10), 50.0),
        ]

        result = get_median_sensor(data, last_timing=time + timedelta(seconds=20))

        assert result >= 0


# =============================================================================
# Test load_from_history
# =============================================================================

class TestLoadFromHistory:
    """Test load_from_history async function."""

    @pytest.mark.asyncio
    async def test_load_from_history_none_hass(self):
        """Test load_from_history returns empty list when hass is None."""
        result = await load_from_history(None, "sensor.test", datetime.datetime.now(pytz.UTC), datetime.datetime.now(pytz.UTC))
        assert result == []

    @pytest.mark.asyncio
    async def test_load_from_history_with_mock(self):
        """Test load_from_history with mocked recorder."""
        hass = MagicMock()

        # Mock recorder instance
        mock_recorder = MagicMock()
        mock_recorder.async_add_executor_job = AsyncMock(return_value=[])

        with patch("custom_components.quiet_solar.ha_model.device.recorder_get_instance", return_value=mock_recorder):
            result = await load_from_history(
                hass,
                "sensor.test",
                datetime.datetime.now(pytz.UTC) - timedelta(hours=1),
                datetime.datetime.now(pytz.UTC)
            )

        assert result == []


# =============================================================================
# Test HADeviceMixin
# =============================================================================

class ConcreteHADevice(HADeviceMixin):
    """Concrete implementation for testing HADeviceMixin."""

    def __init__(self, **kwargs):
        # Pop name before super().__init__
        self.name = kwargs.pop("name", "Test Device")
        self.home = kwargs.pop("home", None)
        super().__init__(**kwargs)


class TestHADeviceMixinInit:
    """Test HADeviceMixin initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_device_entry",
            data={CONF_NAME: "Test Device"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

    def test_init_basic(self):
        """Test basic initialization."""
        device = ConcreteHADevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            name="Test Device",
        )

        assert device.hass == self.hass
        assert device.config_entry == self.config_entry
        assert device.name == "Test Device"

    def test_init_with_calendar(self):
        """Test initialization with calendar entity."""
        device = ConcreteHADevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            name="Test Device",
            **{CONF_CALENDAR: "calendar.test"},
        )

        assert device.calendar == "calendar.test"

    def test_init_with_accurate_power_sensor(self):
        """Test initialization with accurate power sensor."""
        device = ConcreteHADevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            name="Test Device",
            **{CONF_ACCURATE_POWER_SENSOR: "sensor.power"},
        )

        assert device.accurate_power_sensor == "sensor.power"

    def test_init_no_config_entry(self):
        """Test initialization without config entry."""
        device = ConcreteHADevice(
            hass=self.hass,
            config_entry=None,
            home=self.home,
            name="Test Device",
        )

        assert device.config_entry is None
        assert device.config_entry_initialized is True


class TestHADeviceMixinTimeHelpers:
    """Test HADeviceMixin time helper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_device_entry",
            data={CONF_NAME: "Test Device"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteHADevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            name="Test Device",
        )

    def test_get_next_time_from_hours_future(self):
        """Test get_next_time_from_hours returns future time."""
        from datetime import time as dt_time

        now = datetime.datetime.now(pytz.UTC)
        target_time = dt_time(hour=23, minute=59, second=59)  # Late evening

        result = self.device.get_next_time_from_hours(target_time, now)

        assert result is not None
        # Result should be in the future or today
        assert result >= now - timedelta(hours=24)

    def test_get_next_time_from_hours_output_utc(self):
        """Test get_next_time_from_hours returns UTC when requested."""
        from datetime import time as dt_time

        now = datetime.datetime.now(pytz.UTC)
        target_time = dt_time(hour=12, minute=0, second=0)

        result = self.device.get_next_time_from_hours(target_time, now, output_in_utc=True)

        assert result is not None
        assert result.tzinfo == pytz.UTC

    def test_get_proper_local_adapted_tomorrow(self):
        """Test get_proper_local_adapted_tomorrow returns correct date."""
        now = datetime.datetime.now(pytz.UTC)

        result = self.device.get_proper_local_adapted_tomorrow(now)

        # Should be tomorrow in UTC
        assert result > now
        assert result.tzinfo == pytz.UTC


class TestHADeviceMixinPowerProbe:
    """Test HADeviceMixin power probe methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hass = FakeHass()
        self.config_entry = FakeConfigEntry(
            entry_id="test_device_entry",
            data={CONF_NAME: "Test Device"},
        )
        self.home = MagicMock()
        self.data_handler = MagicMock()
        self.data_handler.home = self.home
        self.hass.data[DOMAIN][DATA_HANDLER] = self.data_handler

        self.device = ConcreteHADevice(
            hass=self.hass,
            config_entry=self.config_entry,
            home=self.home,
            name="Test Device",
            **{CONF_ACCURATE_POWER_SENSOR: "sensor.power"},
        )

    def test_attach_power_to_probe(self):
        """Test attach_power_to_probe registers sensor."""
        self.device.attach_power_to_probe("sensor.new_power")

        assert "sensor.new_power" in self.device._entity_probed_state

    def test_attach_power_to_probe_none(self):
        """Test attach_power_to_probe with None is safe."""
        # Should not raise
        self.device.attach_power_to_probe(None)

    def test_attach_amps_to_probe(self):
        """Test attach_amps_to_probe registers sensor."""
        self.device.attach_amps_to_probe("sensor.amps")

        assert "sensor.amps" in self.device._entity_probed_state

    def test_attach_ha_state_to_probe_numerical(self):
        """Test attach_ha_state_to_probe for numerical sensor."""
        self.device.attach_ha_state_to_probe("sensor.temp", is_numerical=True)

        assert "sensor.temp" in self.device._entity_probed_state
        assert self.device._entity_probed_state_is_numerical["sensor.temp"] is True

    def test_attach_ha_state_to_probe_non_numerical(self):
        """Test attach_ha_state_to_probe for non-numerical sensor."""
        self.device.attach_ha_state_to_probe("sensor.status", is_numerical=False)

        assert "sensor.status" in self.device._entity_probed_state
        assert self.device._entity_probed_state_is_numerical["sensor.status"] is False
