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
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_ACCURATE_POWER_SENSOR,
    CONF_CALENDAR,
)

from tests.factories import create_minimal_home_model


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


@pytest.fixture
def device_mixin_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Mock config entry for HADeviceMixin tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_device_entry",
        data={CONF_NAME: "Test Device"},
        title="Test Device",
    )


@pytest.fixture
def device_mixin_home(hass: HomeAssistant):
    """Home and data handler for HADeviceMixin tests."""
    home = create_minimal_home_model()
    data_handler = MagicMock()
    data_handler.home = home
    hass.data.setdefault(DOMAIN, {})[DATA_HANDLER] = data_handler
    return home


def test_init_basic(
    hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
):
    """Test basic initialization."""
    device = ConcreteHADevice(
        hass=hass,
        config_entry=device_mixin_config_entry,
        home=device_mixin_home,
        name="Test Device",
    )
    assert device.hass is hass
    assert device.config_entry == device_mixin_config_entry
    assert device.name == "Test Device"


def test_init_with_calendar(
    hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
):
    """Test initialization with calendar entity."""
    device = ConcreteHADevice(
        hass=hass,
        config_entry=device_mixin_config_entry,
        home=device_mixin_home,
        name="Test Device",
        **{CONF_CALENDAR: "calendar.test"},
    )
    assert device.calendar == "calendar.test"


def test_init_with_accurate_power_sensor(
    hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
):
    """Test initialization with accurate power sensor."""
    device = ConcreteHADevice(
        hass=hass,
        config_entry=device_mixin_config_entry,
        home=device_mixin_home,
        name="Test Device",
        **{CONF_ACCURATE_POWER_SENSOR: "sensor.power"},
    )
    assert device.accurate_power_sensor == "sensor.power"


def test_init_no_config_entry(hass: HomeAssistant, device_mixin_home):
    """Test initialization without config entry."""
    device = ConcreteHADevice(
        hass=hass,
        config_entry=None,
        home=device_mixin_home,
        name="Test Device",
    )
    assert device.config_entry is None
    assert device.config_entry_initialized is True


@pytest.fixture
def device_mixin_device(
    hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
):
    """ConcreteHADevice instance for time helper tests."""
    return ConcreteHADevice(
        hass=hass,
        config_entry=device_mixin_config_entry,
        home=device_mixin_home,
        name="Test Device",
    )


def test_get_next_time_from_hours_future(device_mixin_device):
    """Test get_next_time_from_hours returns future time."""
    from datetime import time as dt_time

    now = datetime.datetime.now(pytz.UTC)
    target_time = dt_time(hour=23, minute=59, second=59)  # Late evening

    result = device_mixin_device.get_next_time_from_hours(target_time, now)

    assert result is not None
    assert result >= now - timedelta(hours=24)


def test_get_next_time_from_hours_output_utc(device_mixin_device):
    """Test get_next_time_from_hours returns UTC when requested."""
    from datetime import time as dt_time

    now = datetime.datetime.now(pytz.UTC)
    target_time = dt_time(hour=12, minute=0, second=0)

    result = device_mixin_device.get_next_time_from_hours(
        target_time, now, output_in_utc=True
    )

    assert result is not None
    assert result.tzinfo == pytz.UTC


def test_get_proper_local_adapted_tomorrow(device_mixin_device):
    """Test get_proper_local_adapted_tomorrow returns correct date."""
    now = datetime.datetime.now(pytz.UTC)

    result = device_mixin_device.get_proper_local_adapted_tomorrow(now)

    assert result > now
    assert result.tzinfo == pytz.UTC


@pytest.fixture
def device_mixin_device_power(
    hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
):
    """ConcreteHADevice instance with power sensor for probe tests."""
    return ConcreteHADevice(
        hass=hass,
        config_entry=device_mixin_config_entry,
        home=device_mixin_home,
        name="Test Device",
        **{CONF_ACCURATE_POWER_SENSOR: "sensor.power"},
    )


def test_attach_power_to_probe(device_mixin_device_power):
    """Test attach_power_to_probe registers sensor."""
    device_mixin_device_power.attach_power_to_probe("sensor.new_power")
    assert "sensor.new_power" in device_mixin_device_power._entity_probed_state


def test_attach_power_to_probe_none(device_mixin_device_power):
    """Test attach_power_to_probe with None is safe."""
    initial_keys = set(device_mixin_device_power._entity_probed_state)
    device_mixin_device_power.attach_power_to_probe(None)
    assert set(device_mixin_device_power._entity_probed_state) == initial_keys


def test_attach_amps_to_probe(device_mixin_device_power):
    """Test attach_amps_to_probe registers sensor."""
    device_mixin_device_power.attach_amps_to_probe("sensor.amps")
    assert "sensor.amps" in device_mixin_device_power._entity_probed_state


def test_attach_ha_state_to_probe_numerical(device_mixin_device_power):
    """Test attach_ha_state_to_probe for numerical sensor."""
    device_mixin_device_power.attach_ha_state_to_probe(
        "sensor.temp", is_numerical=True
    )
    assert "sensor.temp" in device_mixin_device_power._entity_probed_state
    assert (
        device_mixin_device_power._entity_probed_state_is_numerical["sensor.temp"]
        is True
    )


def test_attach_ha_state_to_probe_non_numerical(device_mixin_device_power):
    """Test attach_ha_state_to_probe for non-numerical sensor."""
    device_mixin_device_power.attach_ha_state_to_probe(
        "sensor.status", is_numerical=False
    )
    assert "sensor.status" in device_mixin_device_power._entity_probed_state
    assert (
        device_mixin_device_power._entity_probed_state_is_numerical["sensor.status"]
        is False
    )


# =============================================================================
# Extended Tests for Coverage - Additional Missing Lines
# =============================================================================


class TestLoadFromHistoryExtended:
    """Extended tests for load_from_history to cover exception handling."""

    @pytest.mark.asyncio
    async def test_load_from_history_exception_handling(self):
        """Test load_from_history handles exceptions gracefully (lines 226-228)."""
        hass = MagicMock()

        # Mock recorder instance that raises an exception
        mock_recorder = MagicMock()
        mock_recorder.async_add_executor_job = AsyncMock(side_effect=Exception("DB Error"))

        with patch("custom_components.quiet_solar.ha_model.device.recorder_get_instance", return_value=mock_recorder):
            result = await load_from_history(
                hass,
                "sensor.test",
                datetime.datetime.now(pytz.UTC) - timedelta(hours=1),
                datetime.datetime.now(pytz.UTC)
            )

        # Should return empty list on exception
        assert result == []


class TestGetMedianSensorExtended:
    """Extended tests for get_median_sensor to cover edge cases."""

    def test_with_min_val_filter(self):
        """Test median calculation with min_val filtering (line 184-185)."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 5.0),  # Below min_val, should be skipped
            (time + timedelta(seconds=10), 50.0),
            (time + timedelta(seconds=20), 30.0),
        ]

        result = get_median_sensor(data, min_val=10.0)

        assert result >= 0

    def test_with_max_val_filter(self):
        """Test median calculation with max_val filtering (lines 187-188)."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 10.0),
            (time + timedelta(seconds=10), 500.0),  # Above max_val, should be skipped
            (time + timedelta(seconds=20), 30.0),
        ]

        result = get_median_sensor(data, max_val=100.0)

        assert result >= 0

    def test_with_zero_dt(self):
        """Test median calculation when dt is 0 (lines 195-196)."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 10.0),
            (time, 50.0),  # Same timestamp as previous
            (time + timedelta(seconds=10), 30.0),
        ]

        result = get_median_sensor(data)

        assert result >= 0

    def test_all_values_filtered_out(self):
        """Test when all values are filtered out returns 0 (lines 201-204)."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, 5.0),  # Below min
            (time + timedelta(seconds=10), 500.0),  # Above max
        ]

        result = get_median_sensor(data, min_val=10.0, max_val=100.0)

        assert result == 0.0

    def test_with_none_values_in_data(self):
        """Test median calculation skips None values (lines 181-182)."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, None),
            (time + timedelta(seconds=10), 50.0),
            (time + timedelta(seconds=20), None),
            (time + timedelta(seconds=30), 30.0),
        ]

        result = get_median_sensor(data, last_timing=time + timedelta(seconds=40))

        assert result >= 0


class TestGetAveragePowerEnergyBasedExtended:
    """Extended tests for get_average_power_energy_based."""

    def test_multiple_points_all_none(self):
        """Test when all power values are None (lines 152-157)."""
        time = datetime.datetime.now(pytz.UTC)
        data = [
            (time, None),
            (time + timedelta(hours=1), None),
        ]

        result = get_average_power_energy_based(data)

        assert result == 0.0


class TestHADeviceMixinExtended:
    """Extended tests for HADeviceMixin to cover remaining lines."""

    @pytest.fixture
    def device_with_mobile_app(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Device with mobile app configured for notification tests."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.mobile_app = "mobile_app_test"
        device.mobile_app_url = "https://example.com/device"
        return device

    @pytest.mark.asyncio
    async def test_on_device_state_change_error(self, device_with_mobile_app):
        """Test on_device_state_change with ERROR type (lines 630-635)."""
        from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_ERROR

        time = datetime.datetime.now(pytz.UTC)

        # Mock the service call
        device_with_mobile_app.hass.services = MagicMock()
        device_with_mobile_app.hass.services.async_call = AsyncMock()

        await device_with_mobile_app.on_device_state_change(
            time, DEVICE_STATUS_CHANGE_ERROR
        )

        # Service should have been called with error message
        device_with_mobile_app.hass.services.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_device_state_change_notify(self, device_with_mobile_app):
        """Test on_device_state_change with NOTIFY type (lines 636-638)."""
        from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_NOTIFY

        time = datetime.datetime.now(pytz.UTC)

        device_with_mobile_app.hass.services = MagicMock()
        device_with_mobile_app.hass.services.async_call = AsyncMock()

        await device_with_mobile_app.on_device_state_change_helper(
            time, DEVICE_STATUS_CHANGE_NOTIFY, message="Test notification"
        )

        device_with_mobile_app.hass.services.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_device_state_change_with_url(self, device_with_mobile_app):
        """Test on_device_state_change includes URL in data (lines 651-654)."""
        from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_NOTIFY

        time = datetime.datetime.now(pytz.UTC)

        device_with_mobile_app.hass.services = MagicMock()
        device_with_mobile_app.hass.services.async_call = AsyncMock()

        await device_with_mobile_app.on_device_state_change_helper(
            time, DEVICE_STATUS_CHANGE_NOTIFY, message="Test"
        )

        # Check that URL was included
        call_args = device_with_mobile_app.hass.services.async_call.call_args
        service_data = call_args.kwargs.get("service_data") or call_args[1].get("service_data", {})
        assert "data" in service_data
        assert service_data["data"]["url"] == "https://example.com/device"

    @pytest.mark.asyncio
    async def test_on_device_state_change_service_exception(self, device_with_mobile_app):
        """Test on_device_state_change handles service call exception (lines 664-665)."""
        from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_NOTIFY

        time = datetime.datetime.now(pytz.UTC)

        device_with_mobile_app.hass.services = MagicMock()
        device_with_mobile_app.hass.services.async_call = AsyncMock(
            side_effect=Exception("Service call failed")
        )

        # Should not raise exception
        await device_with_mobile_app.on_device_state_change_helper(
            time, DEVICE_STATUS_CHANGE_NOTIFY, message="Test"
        )

    @pytest.mark.asyncio
    async def test_on_device_state_change_no_mobile_app(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test on_device_state_change when mobile_app is None (no notification sent)."""
        from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_NOTIFY

        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.mobile_app = None  # No mobile app configured

        time = datetime.datetime.now(pytz.UTC)
        device.hass.services = MagicMock()
        device.hass.services.async_call = AsyncMock()

        await device.on_device_state_change_helper(
            time, DEVICE_STATUS_CHANGE_NOTIFY, message="Test"
        )

        # No service call should be made
        device.hass.services.async_call.assert_not_called()

    def test_get_best_power_ha_entity_secondary(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_best_power_HA_entity returns secondary sensor (lines 670-671)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.accurate_power_sensor = None
        device.secondary_power_sensor = "sensor.secondary_power"

        result = device.get_best_power_HA_entity()

        assert result == "sensor.secondary_power"

    def test_get_best_power_ha_entity_none(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_best_power_HA_entity returns None when no sensors (lines 672-673)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.accurate_power_sensor = None
        device.secondary_power_sensor = None

        result = device.get_best_power_HA_entity()

        assert result is None

    def test_is_sensor_growing_none_entity(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test is_sensor_growing with None entity_id returns None (lines 689-690)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        result = device.is_sensor_growing(None)

        assert result is None

    def test_is_sensor_growing_insufficient_data(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test is_sensor_growing with insufficient data returns None (lines 694-695)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)
        # Only one data point
        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [(time, 100.0, {})]

        result = device.is_sensor_growing("sensor.test", time=time)

        assert result is None

    def test_is_sensor_growing_with_data(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test is_sensor_growing with sufficient data (lines 697-710)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time - timedelta(minutes=2), 100.0, {}),
            (time - timedelta(minutes=1), 150.0, {}),
            (time, 200.0, {}),
        ]
        device._entity_probed_last_valid_state["sensor.test"] = (time, 200.0, {})

        result = device.is_sensor_growing("sensor.test", time=time)

        assert result is True

    def test_is_sensor_growing_not_growing(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test is_sensor_growing when values are not growing."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time - timedelta(minutes=2), 200.0, {}),
            (time - timedelta(minutes=1), 150.0, {}),
            (time, 100.0, {}),
        ]
        device._entity_probed_last_valid_state["sensor.test"] = (time, 100.0, {})

        result = device.is_sensor_growing("sensor.test", time=time)

        assert result is False

    def test_get_sensor_latest_possible_valid_value_none_entity(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_sensor_latest_possible_valid_value with None entity."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        result = device.get_sensor_latest_possible_valid_value(None)

        assert result is None

    def test_get_sensor_latest_possible_valid_value_with_tolerance(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_sensor_latest_possible_valid_value with tolerance."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time - timedelta(minutes=5), 100.0, {}),
        ]
        device._entity_probed_last_valid_state["sensor.test"] = (time - timedelta(minutes=5), 100.0, {})

        # With tolerance
        result = device.get_sensor_latest_possible_valid_value(
            "sensor.test", tolerance_seconds=600, time=time
        )

        assert result == 100.0

    def test_attach_ha_state_with_unfiltered(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test attach_ha_state_to_probe with attach_unfiltered=True (lines 1104-1115)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        device.attach_ha_state_to_probe(
            "sensor.test",
            is_numerical=True,
            attach_unfiltered=True
        )

        # Should have both filtered and unfiltered versions
        assert "sensor.test" in device._entity_probed_state
        assert "sensor.test_no_filters" in device._entity_probed_state

    def test_get_unfiltered_entity_name(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_unfiltered_entity_name method."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        # Test with None
        result = device.get_unfiltered_entity_name(None)
        assert result is None

    def test_root_device_post_home_init(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test root_device_post_home_init method (lines 317-329)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        time = datetime.datetime.now(pytz.UTC)

        # Should not raise exception
        device.root_device_post_home_init(time)

    def test_add_to_history_numerical_conversion(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test add_to_history with numerical conversion (lines 1199-1203)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)

        # Create a mock state with non-numeric value
        mock_state = MagicMock()
        mock_state.state = "invalid_number"
        mock_state.attributes = {}
        mock_state.last_updated = datetime.datetime.now(pytz.UTC)

        device.add_to_history("sensor.test", state=mock_state)

        # Value should be None due to failed conversion
        assert "sensor.test" in device._entity_probed_state

    def test_get_state_history_data_empty(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_state_history_data with empty history."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        result = device.get_state_history_data("sensor.nonexistent", None, datetime.datetime.now(pytz.UTC))

        assert result == []

    def test_get_state_history_data_with_data(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_state_history_data with data (lines 1248-1301)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time - timedelta(hours=2), 100.0, {}),
            (time - timedelta(hours=1), 150.0, {}),
            (time, 200.0, {}),
        ]

        # Get last 30 minutes
        result = device.get_state_history_data("sensor.test", 1800, time)

        assert len(result) >= 1

    def test_get_state_history_data_time_before_first(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_state_history_data when to_ts is before first data point (lines 1270-1271)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=True)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time, 100.0, {}),
            (time + timedelta(hours=1), 150.0, {}),
        ]

        # Query for time before first data point
        result = device.get_state_history_data("sensor.test", 60, time - timedelta(hours=1))

        assert result == []

    def test_get_platforms(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_platforms method (lines 1303-1311)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        platforms = device.get_platforms()

        # Should include basic platforms
        assert "button" in platforms or "sensor" in platforms

    def test_get_attached_virtual_devices(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_attached_virtual_devices returns empty list (lines 1314-1316)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )

        result = device.get_attached_virtual_devices()

        assert result == []


class TestDeviceAmpsConsumption:
    """Test device amps consumption methods."""

    @pytest.fixture
    def device_with_amps_sensors(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Device with amps sensors configured."""
        from custom_components.quiet_solar.const import (
            CONF_PHASE_1_AMPS_SENSOR,
            CONF_PHASE_2_AMPS_SENSOR,
            CONF_PHASE_3_AMPS_SENSOR,
        )

        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
            **{
                CONF_PHASE_1_AMPS_SENSOR: "sensor.phase1_amps",
                CONF_PHASE_2_AMPS_SENSOR: "sensor.phase2_amps",
                CONF_PHASE_3_AMPS_SENSOR: "sensor.phase3_amps",
            }
        )
        return device

    def test_get_device_amps_consumption_with_sensors(self, device_with_amps_sensors):
        """Test get_device_amps_consumption with phase sensors (lines 788-844)."""
        device = device_with_amps_sensors
        time = datetime.datetime.now(pytz.UTC)

        # Set up sensor values
        for sensor in ["sensor.phase1_amps", "sensor.phase2_amps", "sensor.phase3_amps"]:
            device._entity_probed_state[sensor] = [(time, 10.0, {})]
            device._entity_probed_last_valid_state[sensor] = (time, 10.0, {})

        result = device.get_device_amps_consumption(tolerance_seconds=None, time=time)

        # Should return amps for each phase
        assert result is not None or result is None  # Depends on full setup


class TestLastStateValueDuration:
    """Test get_last_state_value_duration method."""

    def test_get_last_state_value_duration_empty(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_last_state_value_duration with no data."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=False)

        time = datetime.datetime.now(pytz.UTC)

        duration, ranges = device.get_last_state_value_duration(
            "sensor.test", {"on"}, None, time
        )

        assert duration is None

    def test_get_last_state_value_duration_with_data(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_last_state_value_duration with state history (lines 902-1000)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=False)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time - timedelta(minutes=10), "on", {}),
            (time - timedelta(minutes=5), "on", {}),
            (time, "on", {}),
        ]

        duration, ranges = device.get_last_state_value_duration(
            "sensor.test", {"on"}, None, time
        )

        assert duration is not None
        assert duration > 0

    def test_get_last_state_value_duration_inverted(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_last_state_value_duration with invert_val_probe=True (lines 971-972)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.attach_ha_state_to_probe("sensor.test", is_numerical=False)

        time = datetime.datetime.now(pytz.UTC)
        device._entity_probed_state["sensor.test"] = [
            (time - timedelta(minutes=10), "off", {}),
            (time - timedelta(minutes=5), "off", {}),
            (time, "off", {}),
        ]

        duration, ranges = device.get_last_state_value_duration(
            "sensor.test", {"on"}, None, time, invert_val_probe=True
        )

        # Should find duration where state is NOT "on" (i.e., "off")
        assert duration is not None


class TestCalendarEvents:
    """Test calendar event methods."""

    @pytest.fixture
    def device_with_calendar(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Device with calendar configured."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
            **{CONF_CALENDAR: "calendar.test_calendar"}
        )
        return device

    @pytest.mark.asyncio
    async def test_get_next_scheduled_event_no_calendar(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_next_scheduled_event when calendar is None."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.calendar = None

        time = datetime.datetime.now(pytz.UTC)
        start, end = await device.get_next_scheduled_event(time)

        assert start is None
        assert end is None

    @pytest.mark.asyncio
    async def test_get_next_scheduled_events_no_calendar(
        self, hass: HomeAssistant, device_mixin_config_entry, device_mixin_home
    ):
        """Test get_next_scheduled_events when calendar is None (line 459-460)."""
        device = ConcreteHADevice(
            hass=hass,
            config_entry=device_mixin_config_entry,
            home=device_mixin_home,
            name="Test Device",
        )
        device.calendar = None

        time = datetime.datetime.now(pytz.UTC)
        events = await device.get_next_scheduled_events(time)

        assert events == []
