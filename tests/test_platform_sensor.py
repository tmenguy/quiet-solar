"""Tests for sensor platform."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN

from custom_components.quiet_solar.const import DOMAIN
from custom_components.quiet_solar.sensor import (
    QSBaseSensor,
    QSBaseSensorForecastRestore,
    QSBaseSensorRestore,
    QSBaseSensorSolarActiveProviderRestore,
    QSBaseSensorSolarScoreRestore,
    QSLoadSensorCurrentConstraints,
    async_setup_entry,
    async_unload_entry,
    create_ha_sensor_for_Load,
    create_ha_sensor_for_QSCar,
    create_ha_sensor_for_QSHome,
    create_ha_sensor_for_QSSolar,
)
from tests.factories import create_minimal_home_model
from tests.test_helpers import create_mock_device


def test_create_ha_sensor_for_home():
    """Test creating sensors for home device."""
    mock_home = create_mock_device("home", name="Test Home")
    mock_home.data_handler = MagicMock()
    mock_home.home_non_controlled_power_forecast_sensor_values = {}
    mock_home.home_solar_forecast_sensor_values = {}

    entities = create_ha_sensor_for_QSHome(mock_home)

    assert len(entities) > 0
    # Should create home consumption, available power, and forecast sensors
    assert any("home_non_controlled_consumption" in e.entity_description.key for e in entities)
    assert any("home_consumption" in e.entity_description.key for e in entities)
    assert any("home_available_power" in e.entity_description.key for e in entities)


def test_create_ha_sensor_for_car():
    """Test creating sensors for car device."""
    mock_car = create_mock_device("car", name="Test Car")
    mock_car.data_handler = MagicMock()
    mock_car.charger = None
    mock_car.get_car_charge_percent = MagicMock(return_value=80)
    mock_car.get_car_charge_type = MagicMock(return_value="Solar")
    mock_car.get_car_charge_time_readable_name = MagicMock(return_value="2 hours")

    mock_car.get_estimated_range_km = MagicMock(return_value=260)
    mock_car.get_autonomy_to_target_soc_km = MagicMock(return_value=50)

    entities = create_ha_sensor_for_QSCar(mock_car)

    assert len(entities) > 0
    # Should create SOC, charge type, charge time, and range sensors
    assert any("car_soc_percentage" in e.entity_description.key for e in entities)
    assert any("car_charge_type" in e.entity_description.key for e in entities)
    assert any("car_estimated_range_km" in e.entity_description.key for e in entities)
    assert any("car_autonomy_to_target_soc_km" in e.entity_description.key for e in entities)


def test_create_ha_sensor_for_load():
    """Test creating sensors for load device."""
    from custom_components.quiet_solar.ha_model.device import HADeviceMixin
    from custom_components.quiet_solar.home_model.load import AbstractLoad

    mock_load = MagicMock(spec=[AbstractLoad, HADeviceMixin])
    mock_load.data_handler = MagicMock()
    mock_load.device_id = "test_load_123"
    mock_load.device_type = "load"
    mock_load.name = "Test Load"
    mock_load.qs_enable_device = True
    mock_load.current_command = None
    mock_load.support_user_override = MagicMock(return_value=True)
    mock_load.get_override_state = MagicMock(return_value="off")
    mock_load.get_virtual_current_constraint_translation_key = MagicMock(return_value="constraint")

    entities = create_ha_sensor_for_Load(mock_load)

    assert len(entities) > 0
    # Should create  override state sensor
    assert any("load_override_state" in e.entity_description.key for e in entities)


def test_create_ha_sensor_car_without_charger():
    """Test car sensor with no charger attached."""
    mock_car = create_mock_device("car", name="Test Car")
    mock_car.data_handler = MagicMock()
    mock_car.charger = None
    mock_car.get_car_charge_percent = MagicMock(return_value=50)
    mock_car.get_car_charge_type = MagicMock(return_value="Not Charging")
    mock_car.get_car_charge_time_readable_name = MagicMock(return_value="N/A")

    entities = create_ha_sensor_for_QSCar(mock_car)

    # Should still create entities even without charger
    assert len(entities) > 0


def test_qs_base_sensor_init():
    """Test QSBaseSensor initialization."""
    mock_handler = MagicMock()
    mock_device = create_mock_device("test")

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_sensor",
        translation_key="test",
        qs_is_none_unavailable=False,
    )

    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)

    assert sensor.device == mock_device
    assert sensor.entity_description == mock_description


def test_qs_base_sensor_update_with_value():
    """Test sensor update with valid value."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_value = 42.5

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_value", translation_key="test", qs_is_none_unavailable=False, value_fn=None
    )

    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    assert sensor._attr_native_value == 42.5
    sensor.async_write_ha_state.assert_called_once()


def test_qs_base_sensor_update_with_none_not_unavailable():
    """Test sensor update with None value when not marked unavailable."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_value = None

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_value", translation_key="test", qs_is_none_unavailable=False, value_fn=None
    )

    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    sensor.async_write_ha_state.assert_called_once()


def test_qs_base_sensor_update_with_none_unavailable():
    """Test sensor becomes unavailable when value is None and marked unavailable."""
    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")
    mock_device.test_value = None

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_description = QSSensorEntityDescription(
        key="test_value", translation_key="test", qs_is_none_unavailable=True, value_fn=None
    )

    sensor = QSBaseSensor(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    assert sensor._attr_available is False
    assert sensor._attr_native_value == STATE_UNAVAILABLE


def test_qs_base_sensor_update_with_value_fn():
    """Test sensor update using value function."""

    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")

    description = QSSensorEntityDescription(
        key="computed_value",
        translation_key="test",
        qs_is_none_unavailable=False,
        value_fn=lambda device, key: device.name.upper(),
    )

    sensor = QSBaseSensor(mock_handler, mock_device, description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    assert sensor._attr_native_value == "MOCK DEVICE"


@pytest.mark.asyncio
async def test_async_setup_entry(fake_hass, mock_config_entry):
    """Test sensor platform setup."""
    mock_device = create_mock_device("home")
    mock_device.data_handler = MagicMock()
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    mock_add_entities = MagicMock()

    with patch("custom_components.quiet_solar.sensor.create_ha_sensor", return_value=[MagicMock()]):
        await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

        mock_add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry_no_device(fake_hass, mock_config_entry):
    """Test sensor platform setup with no device."""
    mock_add_entities = MagicMock()

    await async_setup_entry(fake_hass, mock_config_entry, mock_add_entities)

    mock_add_entities.assert_not_called()


@pytest.mark.asyncio
async def test_async_unload_entry(fake_hass, mock_config_entry):
    """Test sensor platform unload."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_device.home = mock_home
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    assert result is True
    mock_home.remove_device.assert_called_once_with(mock_device)


@pytest.mark.asyncio
async def test_async_unload_entry_handles_exception(fake_hass, mock_config_entry):
    """Test sensor platform unload handles exceptions gracefully."""
    mock_device = create_mock_device("test")
    mock_home = create_minimal_home_model()
    mock_home.remove_device = MagicMock(side_effect=Exception("Test error"))
    mock_device.home = mock_home
    fake_hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device

    result = await async_unload_entry(fake_hass, mock_config_entry)

    # Should still return True even with exception
    assert result is True


def test_qs_base_sensor_update_with_value_fn_and_attr():
    """Line 419: async_update_callback uses value_fn_and_attr when set."""
    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = create_mock_device("test")

    description = QSSensorEntityDescription(
        key="computed_value",
        translation_key="test",
        qs_is_none_unavailable=False,
        value_fn_and_attr=lambda device, key: ("sensor_val", {"extra": 42}),
    )

    sensor = QSBaseSensor(mock_handler, mock_device, description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    assert sensor._attr_native_value == "sensor_val"
    assert sensor._attr_extra_state_attributes == {"extra": 42}
    sensor.async_write_ha_state.assert_called_once()


def test_qs_extra_stored_data_from_dict_invalid():
    """Lines 464-467: QSExtraStoredData.from_dict returns None on invalid dict."""
    from custom_components.quiet_solar.sensor import QSExtraStoredData

    result = QSExtraStoredData.from_dict({"bad_key": "value"})

    assert result is None


def test_qs_extra_stored_data_from_dict_valid():
    """Verify QSExtraStoredData.from_dict works with valid dict."""
    from custom_components.quiet_solar.sensor import QSExtraStoredData

    result = QSExtraStoredData.from_dict(
        {
            "native_value": "hello",
            "native_attr": {"foo": "bar"},
        }
    )

    assert result is not None
    assert result.native_value == "hello"
    assert result.native_attr == {"foo": "bar"}


@pytest.mark.asyncio
async def test_qs_load_sensor_current_constraints_async_added_to_hass():
    """Lines 555-556: async_added_to_hass loads stored constraints via device."""
    from custom_components.quiet_solar.const import (
        HA_CONSTRAINT_SENSOR_HISTORY,
        HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT,
    )
    from custom_components.quiet_solar.sensor import (
        QSExtraStoredData,
        QSLoadSensorCurrentConstraints,
        QSSensorEntityDescription,
    )

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()

    mock_device = MagicMock()
    mock_device.device_id = "test_load"
    mock_device.device_type = "load"
    mock_device.name = "Test Load"
    mock_device.qs_enable_device = True
    mock_device.async_load_constraints_from_storage = AsyncMock()

    description = QSSensorEntityDescription(
        key="current_constraint",
        translation_key="test",
    )

    sensor = QSLoadSensorCurrentConstraints(mock_handler, mock_device, description)

    stored_constraints = [{"type": "filler", "value": 100}]
    stored_executed = {"type": "mandatory", "value": 200}
    stored_data = QSExtraStoredData(
        native_value="Active",
        native_attr={
            HA_CONSTRAINT_SENSOR_HISTORY: stored_constraints,
            HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT: stored_executed,
        },
    )

    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = MagicMock()
        mock_extra.return_value.as_dict.return_value = stored_data.as_dict()
        await sensor.async_added_to_hass()

    mock_device.async_load_constraints_from_storage.assert_called_once()
    call_args = mock_device.async_load_constraints_from_storage.call_args
    assert call_args[0][1] == stored_constraints
    assert call_args[0][2] == stored_executed


def test_qs_load_sensor_current_constraints_update():
    """Test QSLoadSensorCurrentConstraints update callback."""
    from custom_components.quiet_solar.home_model.load import AbstractLoad

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=AbstractLoad)
    mock_device.device_id = "test_load"
    mock_device.device_type = "load"
    mock_device.name = "Test Load"
    mock_device.qs_enable_device = True
    mock_device.get_active_readable_name = MagicMock(return_value="Active Constraint")
    mock_device.get_active_constraints = MagicMock(return_value=[])
    mock_device._last_completed_constraint = None
    mock_device.update_to_be_saved_info = MagicMock(return_value={})

    mock_description = MagicMock()
    mock_description.key = "current_constraint"
    mock_description.name = None
    mock_description.translation_key = "test"

    sensor = QSLoadSensorCurrentConstraints(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    assert sensor._attr_native_value == "Active Constraint"
    sensor.async_write_ha_state.assert_called_once()


def test_qs_load_sensor_current_constraints_update_with_last_completed():
    """Test update callback stores last completed constraint (lines 555-556)."""
    from custom_components.quiet_solar.const import (
        HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT,
    )
    from custom_components.quiet_solar.home_model.load import AbstractLoad

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()
    mock_device = MagicMock(spec=AbstractLoad)
    mock_device.device_id = "test_load"
    mock_device.device_type = "load"
    mock_device.name = "Test Load"
    mock_device.qs_enable_device = True
    mock_device.get_active_readable_name = MagicMock(return_value="Active")
    mock_device.get_active_constraints = MagicMock(return_value=[])
    mock_device.update_to_be_saved_info = MagicMock(return_value={})

    mock_completed = MagicMock()
    mock_completed.to_dict = MagicMock(return_value={"type": "mandatory", "value": 42})
    mock_device._last_completed_constraint = mock_completed

    mock_description = MagicMock()
    mock_description.key = "current_constraint"
    mock_description.name = None
    mock_description.translation_key = "test"

    sensor = QSLoadSensorCurrentConstraints(mock_handler, mock_device, mock_description)
    sensor.async_write_ha_state = MagicMock()

    test_time = datetime.now(pytz.UTC)
    sensor.async_update_callback(test_time)

    stored = sensor._attr_extra_state_attributes[HA_CONSTRAINT_SENSOR_LAST_EXECUTED_CONSTRAINT]
    assert stored == {"type": "mandatory", "value": 42}
    mock_completed.to_dict.assert_called_once()


def test_solar_sensors_use_restore_class():
    """Test that forecast age, score, and active provider sensors use QSBaseSensorRestore."""
    from custom_components.quiet_solar.const import (
        SENSOR_SOLAR_ACTIVE_PROVIDER,
        SENSOR_SOLAR_FORECAST_AGE,
        SENSOR_SOLAR_FORECAST_SCORE_PREFIX,
    )

    mock_provider = MagicMock()
    mock_provider.score = 100.0

    mock_device = create_mock_device("solar", name="Test Solar")
    mock_device.data_handler = MagicMock()
    mock_device.solar_forecast_providers = {"TestProvider": mock_provider}
    mock_device.get_forecast_age_hours = MagicMock(return_value=1.5)
    mock_device.active_provider_name = "TestProvider"
    mock_device.solar_forecast_sensor_values = {}
    mock_device.solar_forecast_sensor_values_probers = {}
    mock_device.solar_forecast_sensor_values_per_provider = {}
    mock_device.solar_forecast_sensor_values_per_provider_probers = {}

    entities = create_ha_sensor_for_QSSolar(mock_device)

    # Find the three sensors by key
    forecast_age = [e for e in entities if e.entity_description.key == SENSOR_SOLAR_FORECAST_AGE]
    scores = [e for e in entities if e.entity_description.key.startswith(SENSOR_SOLAR_FORECAST_SCORE_PREFIX)]
    active_provider = [e for e in entities if e.entity_description.key == SENSOR_SOLAR_ACTIVE_PROVIDER]

    assert len(forecast_age) == 1
    assert len(scores) == 1
    assert len(active_provider) == 1

    assert isinstance(forecast_age[0], QSBaseSensorRestore), "Forecast age sensor must use QSBaseSensorRestore"
    assert isinstance(scores[0], QSBaseSensorSolarScoreRestore), "Score sensor must use QSBaseSensorSolarScoreRestore"
    assert isinstance(active_provider[0], QSBaseSensorSolarActiveProviderRestore), "Active provider sensor must use QSBaseSensorSolarActiveProviderRestore"


# --- Tests for QSBaseSensorRestore filtering unavailable/unknown on restore ---


def _make_restore_sensor(cls=QSBaseSensorRestore):
    """Create a minimal sensor instance for restore testing."""
    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()

    mock_device = MagicMock()
    mock_device.device_id = "test_restore"
    mock_device.device_type = "home"
    mock_device.name = "Test Restore"
    mock_device.qs_enable_device = True

    description = QSSensorEntityDescription(
        key="test_restore_key",
        translation_key="test",
    )

    return cls(mock_handler, mock_device, description)


async def _restore_with_value(native_value, cls=QSBaseSensorRestore):
    """Helper: restore a sensor with given native_value, return the sensor."""
    from custom_components.quiet_solar.sensor import QSExtraStoredData

    sensor = _make_restore_sensor(cls)
    stored_data = QSExtraStoredData(
        native_value=native_value,
        native_attr={"some_attr": "val"},
    )

    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = MagicMock()
        mock_extra.return_value.as_dict.return_value = stored_data.as_dict()
        await sensor.async_added_to_hass()

    return sensor


@pytest.mark.asyncio
async def test_restore_sensor_unavailable_becomes_none():
    """Restoring 'unavailable' native_value must result in None, not the string."""
    sensor = await _restore_with_value(STATE_UNAVAILABLE)
    assert sensor._attr_native_value is None


@pytest.mark.asyncio
async def test_restore_sensor_unknown_becomes_none():
    """Restoring 'unknown' native_value must result in None, not the string."""
    sensor = await _restore_with_value(STATE_UNKNOWN)
    assert sensor._attr_native_value is None


@pytest.mark.asyncio
async def test_restore_sensor_normal_value_unchanged():
    """Restoring a normal numeric value passes through unchanged."""
    sensor = await _restore_with_value("42.5")
    assert sensor._attr_native_value == "42.5"


@pytest.mark.asyncio
async def test_restore_sensor_none_value_unchanged():
    """Restoring None native_value stays None."""
    sensor = await _restore_with_value(None)
    assert sensor._attr_native_value is None


@pytest.mark.asyncio
async def test_restore_sensor_extra_attrs_preserved():
    """Extra state attributes are always restored regardless of native_value filtering."""
    sensor = await _restore_with_value(STATE_UNAVAILABLE)
    assert sensor._attr_extra_state_attributes == {"some_attr": "val"}


@pytest.mark.asyncio
async def test_forecast_restore_sensor_filters_unavailable():
    """QSBaseSensorForecastRestore inherits the unavailable filtering via super()."""
    sensor = await _restore_with_value(STATE_UNAVAILABLE, cls=QSBaseSensorForecastRestore)
    assert sensor._attr_native_value is None


# --- Tests for QSBaseSensorSolarScoreRestore (Task 5) ---


def _make_score_restore_sensor(provider=None):
    """Create a score restore sensor for testing."""
    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()

    mock_device = MagicMock()
    mock_device.device_id = "test_solar"
    mock_device.device_type = "solar"
    mock_device.name = "Test Solar"
    mock_device.qs_enable_device = True

    description = QSSensorEntityDescription(
        key="test_score_key",
        translation_key="test",
    )

    if provider is None:
        provider = MagicMock()
        provider.score = None

    return QSBaseSensorSolarScoreRestore(
        mock_handler, mock_device, description, provider=provider
    )


async def _restore_score_with_value(native_value, provider=None):
    """Helper: restore a score sensor with given native_value."""
    from custom_components.quiet_solar.sensor import QSExtraStoredData

    sensor = _make_score_restore_sensor(provider=provider)
    stored_data = QSExtraStoredData(
        native_value=native_value,
        native_attr={"some_attr": "val"},
    )

    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = MagicMock()
        mock_extra.return_value.as_dict.return_value = stored_data.as_dict()
        await sensor.async_added_to_hass()

    return sensor


@pytest.mark.asyncio
async def test_score_restore_hydrates_provider_score():
    """Numeric restore value hydrates provider.score when score is None."""
    provider = MagicMock()
    provider.score = None

    sensor = await _restore_score_with_value("123.45", provider=provider)

    assert provider.score == 123.45
    assert sensor._attr_native_value == "123.45"


@pytest.mark.asyncio
async def test_score_restore_skips_when_score_already_set():
    """Skip hydration when provider.score is already set."""
    provider = MagicMock()
    provider.score = 50.0

    sensor = await _restore_score_with_value("999.0", provider=provider)

    assert provider.score == 50.0


@pytest.mark.asyncio
async def test_score_restore_skips_unavailable():
    """Skip hydration when restored value is unavailable."""
    provider = MagicMock()
    provider.score = None

    await _restore_score_with_value(STATE_UNAVAILABLE, provider=provider)

    assert provider.score is None


@pytest.mark.asyncio
async def test_score_restore_skips_unknown():
    """Skip hydration when restored value is unknown."""
    provider = MagicMock()
    provider.score = None

    await _restore_score_with_value(STATE_UNKNOWN, provider=provider)

    assert provider.score is None


@pytest.mark.asyncio
async def test_score_restore_skips_non_numeric():
    """Skip hydration when restored value is non-numeric string."""
    provider = MagicMock()
    provider.score = None

    await _restore_score_with_value("not_a_number", provider=provider)

    assert provider.score is None


@pytest.mark.asyncio
async def test_score_restore_skips_no_extra_data():
    """Skip hydration when no last extra data available."""
    provider = MagicMock()
    provider.score = None

    sensor = _make_score_restore_sensor(provider=provider)

    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = None
        await sensor.async_added_to_hass()

    assert provider.score is None


# --- Tests for QSBaseSensorSolarActiveProviderRestore (Task 6) ---


def _make_active_provider_restore_sensor(providers=None):
    """Create an active provider restore sensor for testing."""
    from custom_components.quiet_solar.sensor import QSSensorEntityDescription

    mock_handler = MagicMock()
    mock_handler.hass = MagicMock()

    mock_device = MagicMock()
    mock_device.device_id = "test_solar"
    mock_device.device_type = "solar"
    mock_device.name = "Test Solar"
    mock_device.qs_enable_device = True

    if providers is None:
        providers = {"Solcast": MagicMock(), "OpenMeteo": MagicMock()}
    mock_device.solar_forecast_providers = providers
    mock_device._set_active_provider = MagicMock()
    mock_device._provider_mode = "auto"

    description = QSSensorEntityDescription(
        key="test_active_provider_key",
        translation_key="test",
    )

    return QSBaseSensorSolarActiveProviderRestore(
        mock_handler, mock_device, description
    )


async def _restore_active_provider_with_value(native_value, providers=None):
    """Helper: restore an active provider sensor with given native_value."""
    from custom_components.quiet_solar.sensor import QSExtraStoredData

    sensor = _make_active_provider_restore_sensor(providers=providers)
    stored_data = QSExtraStoredData(
        native_value=native_value,
        native_attr={},
    )

    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = MagicMock()
        mock_extra.return_value.as_dict.return_value = stored_data.as_dict()
        await sensor.async_added_to_hass()

    return sensor


@pytest.mark.asyncio
async def test_active_provider_restore_valid_name():
    """Valid provider name from restore updates device active provider."""
    sensor = await _restore_active_provider_with_value("Solcast")

    sensor.device._set_active_provider.assert_called_once_with("Solcast")


@pytest.mark.asyncio
async def test_active_provider_restore_unknown_name_skipped():
    """Unknown provider name (not in dict) is skipped."""
    sensor = await _restore_active_provider_with_value("NonExistent")

    sensor.device._set_active_provider.assert_not_called()


@pytest.mark.asyncio
async def test_active_provider_restore_unavailable_skipped():
    """Unavailable restored value is skipped."""
    sensor = await _restore_active_provider_with_value(STATE_UNAVAILABLE)

    sensor.device._set_active_provider.assert_not_called()


@pytest.mark.asyncio
async def test_active_provider_restore_unknown_string_skipped():
    """Unknown state string restored value is skipped."""
    sensor = await _restore_active_provider_with_value(STATE_UNKNOWN)

    sensor.device._set_active_provider.assert_not_called()


@pytest.mark.asyncio
async def test_active_provider_restore_empty_string_skipped():
    """Empty string restored value is skipped."""
    sensor = await _restore_active_provider_with_value("")

    sensor.device._set_active_provider.assert_not_called()


@pytest.mark.asyncio
async def test_active_provider_restore_does_not_change_mode():
    """Hydration does NOT set _provider_mode."""
    sensor = await _restore_active_provider_with_value("Solcast")

    # _provider_mode should not be touched — only _set_active_provider is called
    assert sensor.device._provider_mode == "auto"


@pytest.mark.asyncio
async def test_active_provider_restore_skips_no_extra_data():
    """Skip hydration when no last extra data available."""
    sensor = _make_active_provider_restore_sensor()

    with patch.object(sensor, "async_get_last_extra_data", new_callable=AsyncMock) as mock_extra:
        mock_extra.return_value = None
        await sensor.async_added_to_hass()

    sensor.device._set_active_provider.assert_not_called()
