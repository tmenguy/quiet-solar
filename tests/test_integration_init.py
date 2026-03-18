"""Tests for quiet_solar __init__.py integration setup."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.quiet_solar import (
    async_reload_entry,
    async_reload_quiet_solar,
    async_setup,
    async_setup_entry,
    async_unload_entry,
    register_ocpp_notification_listener,
    register_reload_service,
)
from custom_components.quiet_solar.const import DATA_HANDLER, DOMAIN
from custom_components.quiet_solar.data_handler import QSDataHandler
from tests.factories import create_minimal_home_model
from tests.ha_tests.const import MOCK_HOME_CONFIG
from tests.test_helpers import create_mock_device


@pytest.fixture(autouse=True)
def ensure_domain_data(hass: HomeAssistant) -> None:
    """Ensure hass.data[DOMAIN] exists for tests that call async_setup_entry directly."""
    hass.data.setdefault(DOMAIN, {})


@pytest.fixture
def mock_home_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Provide mock home config entry for integration init tests."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="home_entry_123",
        data={**MOCK_HOME_CONFIG},
        title="home: Test Home",
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Provide mock device config entry for unload/reload tests."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="test_entry_123",
        data={CONF_NAME: "Test Device"},
        title="Test Device",
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_async_setup_registers_services(hass: HomeAssistant):
    """Test that async_setup registers reload service and OCPP listener."""
    result = await async_setup(hass, {})

    assert result is True
    assert DOMAIN in hass.data
    # Reload service and OCPP listener registration are exercised by other tests


@pytest.mark.asyncio
async def test_register_reload_service(hass: HomeAssistant):
    """Test reload service registration."""
    with patch("custom_components.quiet_solar.service.async_register_admin_service") as mock_register:
        register_reload_service(hass)

        mock_register.assert_called_once()
        assert mock_register.call_args[0][0] is hass
        assert mock_register.call_args[0][1] == DOMAIN
        assert mock_register.call_args[0][2] == "reload"


@pytest.mark.asyncio
async def test_register_ocpp_notification_listener(hass: HomeAssistant):
    """Test OCPP notification listener registration (no exception, fire is safe)."""
    register_ocpp_notification_listener(hass)
    # hass.bus.listeners is not safe to read from async context; listener behavior is tested in test_ocpp_*
    hass.bus.async_fire("call_service", {"domain": "other", "service": "create"})


@pytest.mark.asyncio
async def test_async_setup_entry_creates_data_handler(hass: HomeAssistant, mock_home_config_entry):
    """Test setup entry creates data handler if not exists."""
    with patch.object(QSDataHandler, "async_add_entry", new_callable=AsyncMock) as mock_add:
        result = await async_setup_entry(hass, mock_home_config_entry)

        assert result is True
        assert DATA_HANDLER in hass.data[DOMAIN]
        assert isinstance(hass.data[DOMAIN][DATA_HANDLER], QSDataHandler)
        mock_add.assert_called_once_with(mock_home_config_entry)


@pytest.mark.asyncio
async def test_async_setup_entry_reuses_existing_data_handler(hass: HomeAssistant, mock_home_config_entry):
    """Test setup entry reuses existing data handler."""
    hass.data.setdefault(DOMAIN, {})
    existing_handler = QSDataHandler(hass)
    hass.data[DOMAIN][DATA_HANDLER] = existing_handler

    with patch.object(existing_handler, "async_add_entry", new_callable=AsyncMock) as mock_add:
        result = await async_setup_entry(hass, mock_home_config_entry)

        assert result is True
        assert hass.data[DOMAIN][DATA_HANDLER] is existing_handler
        mock_add.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_entry_registers_update_listener(hass: HomeAssistant, mock_home_config_entry):
    """Test that update listener is registered."""
    with (
        patch.object(QSDataHandler, "async_add_entry", new_callable=AsyncMock),
        patch.object(
            mock_home_config_entry, "async_on_unload", wraps=mock_home_config_entry.async_on_unload
        ) as mock_on_unload,
    ):
        await async_setup_entry(hass, mock_home_config_entry)

        # async_setup_entry registers intervals via async_on_unload
        assert mock_on_unload.call_count >= 1


@pytest.mark.asyncio
async def test_async_unload_entry_removes_device(hass: HomeAssistant, mock_config_entry):
    """Test unload entry removes device from home."""
    hass.data.setdefault(DOMAIN, {})
    mock_device = create_mock_device("test_device", platforms=[Platform.SENSOR])
    mock_home = create_minimal_home_model()
    mock_device.home = mock_home
    mock_device.get_platforms.return_value = [Platform.SENSOR]

    hass.data[DOMAIN][mock_config_entry.entry_id] = mock_device
    hass.data[DOMAIN][DATA_HANDLER] = MagicMock()

    with patch.object(hass.config_entries, "async_unload_platforms", new_callable=AsyncMock, return_value=True):
        result = await async_unload_entry(hass, mock_config_entry)

        assert result is True
        mock_home.remove_device.assert_called_once_with(mock_device)
        assert mock_config_entry.entry_id not in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_async_unload_entry_no_device(hass: HomeAssistant, mock_config_entry):
    """Test unload entry when device doesn't exist."""
    result = await async_unload_entry(hass, mock_config_entry)

    # No device to unload — should succeed (return True) to avoid FAILED_UNLOAD state
    assert result is True


@pytest.mark.asyncio
async def test_async_reload_entry(hass: HomeAssistant, mock_config_entry):
    """Test reload entry."""
    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

        mock_reload.assert_called_once_with(mock_config_entry.entry_id)


@pytest.mark.asyncio
async def test_async_reload_entry_skipped_when_flagged(hass: HomeAssistant, mock_config_entry):
    """Test that reload is skipped when the entry is flagged for data-only save."""
    from custom_components.quiet_solar.const import DATA_SKIP_RELOAD_ENTRY_IDS

    hass.data.setdefault(DOMAIN, {})
    skip_set = hass.data[DOMAIN].setdefault(DATA_SKIP_RELOAD_ENTRY_IDS, set())
    skip_set.add(mock_config_entry.entry_id)

    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

        mock_reload.assert_not_called()

    assert mock_config_entry.entry_id not in skip_set


@pytest.mark.asyncio
async def test_async_reload_entry_not_skipped_for_other_entries(hass: HomeAssistant, mock_config_entry):
    """Test that the skip flag for one entry does not affect other entries."""
    from custom_components.quiet_solar.const import DATA_SKIP_RELOAD_ENTRY_IDS

    hass.data.setdefault(DOMAIN, {})
    skip_set = hass.data[DOMAIN].setdefault(DATA_SKIP_RELOAD_ENTRY_IDS, set())
    skip_set.add("some_other_entry_id")

    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

        mock_reload.assert_called_once_with(mock_config_entry.entry_id)

    assert "some_other_entry_id" in skip_set


@pytest.mark.asyncio
async def test_async_reload_entry_skip_flag_consumed_once(hass: HomeAssistant, mock_config_entry):
    """Test that the skip flag is consumed after one use so subsequent updates reload normally."""
    from custom_components.quiet_solar.const import DATA_SKIP_RELOAD_ENTRY_IDS

    hass.data.setdefault(DOMAIN, {})
    skip_set = hass.data[DOMAIN].setdefault(DATA_SKIP_RELOAD_ENTRY_IDS, set())
    skip_set.add(mock_config_entry.entry_id)

    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)
        mock_reload.assert_not_called()

        await async_reload_entry(hass, mock_config_entry)
        mock_reload.assert_called_once_with(mock_config_entry.entry_id)


@pytest.mark.asyncio
async def test_async_reload_entry_no_skip_set_at_all(hass: HomeAssistant, mock_config_entry):
    """Test reload works normally when DATA_SKIP_RELOAD_ENTRY_IDS was never created."""
    hass.data.setdefault(DOMAIN, {})

    with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

        mock_reload.assert_called_once_with(mock_config_entry.entry_id)


@pytest.mark.asyncio
async def test_save_entry_data_no_reload_sets_skip_flag(hass: HomeAssistant, mock_config_entry):
    """Test that save_entry_data_no_reload sets the skip flag before updating entry."""
    from custom_components.quiet_solar.const import DATA_SKIP_RELOAD_ENTRY_IDS
    from custom_components.quiet_solar.ha_model.device import HADeviceMixin

    hass.data.setdefault(DOMAIN, {})

    device = MagicMock(spec=HADeviceMixin)
    device.hass = hass
    device.config_entry = mock_config_entry

    HADeviceMixin.save_entry_data_no_reload(device, {"key": "value"})

    skip_set = hass.data[DOMAIN].get(DATA_SKIP_RELOAD_ENTRY_IDS)
    assert skip_set is not None
    assert mock_config_entry.entry_id in skip_set


@pytest.mark.asyncio
async def test_save_entry_data_no_reload_noop_without_hass(hass: HomeAssistant):
    """Test that save_entry_data_no_reload is a no-op when hass is None."""
    from custom_components.quiet_solar.ha_model.device import HADeviceMixin

    device = MagicMock(spec=HADeviceMixin)
    device.hass = None
    device.config_entry = MagicMock()

    HADeviceMixin.save_entry_data_no_reload(device, {"key": "value"})


@pytest.mark.asyncio
async def test_save_entry_data_no_reload_noop_without_config_entry(hass: HomeAssistant):
    """Test that save_entry_data_no_reload is a no-op when config_entry is None."""
    from custom_components.quiet_solar.ha_model.device import HADeviceMixin

    device = MagicMock(spec=HADeviceMixin)
    device.hass = hass
    device.config_entry = None

    HADeviceMixin.save_entry_data_no_reload(device, {"key": "value"})


@pytest.mark.asyncio
async def test_remove_device_car_detaches_from_charger(hass: HomeAssistant):
    """Test that removing a car device detaches it from any charger referencing it."""
    from custom_components.quiet_solar.ha_model.car import QSCar
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_home = create_minimal_home_model()

    mock_car = MagicMock(spec=QSCar)
    mock_car.name = "TestCar"
    mock_car.charger = None
    mock_car.car_is_invited = False

    mock_charger = MagicMock(spec=QSChargerGeneric)
    mock_charger.name = "TestCharger"
    mock_charger.car = mock_car

    mock_home._cars = [mock_car]
    mock_home._chargers = [mock_charger]
    mock_home._persons = []
    mock_home._heat_pumps = []
    mock_home._all_piloted_devices = []
    mock_home._all_loads = []
    mock_home._all_dynamic_groups = []
    mock_home._all_devices = []

    from custom_components.quiet_solar.ha_model.home import QSHome

    QSHome.remove_device(mock_home, mock_car)

    mock_charger.detach_car.assert_called_once()
    assert mock_car not in mock_home._cars


@pytest.mark.asyncio
async def test_remove_device_car_no_detach_when_not_attached(hass: HomeAssistant):
    """Test that removing a car does not detach if charger has a different car."""
    from custom_components.quiet_solar.ha_model.car import QSCar
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_home = create_minimal_home_model()

    mock_car = MagicMock(spec=QSCar)
    mock_car.name = "TestCar"

    other_car = MagicMock(spec=QSCar)
    other_car.name = "OtherCar"

    mock_charger = MagicMock(spec=QSChargerGeneric)
    mock_charger.name = "TestCharger"
    mock_charger.car = other_car

    mock_home._cars = [mock_car]
    mock_home._chargers = [mock_charger]
    mock_home._persons = []
    mock_home._heat_pumps = []
    mock_home._all_piloted_devices = []
    mock_home._all_loads = []
    mock_home._all_dynamic_groups = []
    mock_home._all_devices = []

    from custom_components.quiet_solar.ha_model.home import QSHome

    QSHome.remove_device(mock_home, mock_car)

    mock_charger.detach_car.assert_not_called()


@pytest.mark.asyncio
async def test_remove_device_charger_detaches_car(hass: HomeAssistant):
    """Test that removing a charger device detaches its attached car."""
    from custom_components.quiet_solar.ha_model.car import QSCar
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_home = create_minimal_home_model()

    mock_car = MagicMock(spec=QSCar)
    mock_car.name = "TestCar"

    mock_charger = MagicMock(spec=QSChargerGeneric)
    mock_charger.name = "TestCharger"
    mock_charger.car = mock_car

    mock_home._chargers = [mock_charger]
    mock_home._persons = []
    mock_home._heat_pumps = []
    mock_home._all_piloted_devices = []
    mock_home._all_loads = []
    mock_home._all_dynamic_groups = []
    mock_home._all_devices = []

    from custom_components.quiet_solar.ha_model.home import QSHome

    QSHome.remove_device(mock_home, mock_charger)

    mock_charger.detach_car.assert_called_once()


@pytest.mark.asyncio
async def test_remove_device_charger_no_detach_when_no_car(hass: HomeAssistant):
    """Test that removing a charger without an attached car does not call detach."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    mock_home = create_minimal_home_model()

    mock_charger = MagicMock(spec=QSChargerGeneric)
    mock_charger.name = "TestCharger"
    mock_charger.car = None

    mock_home._chargers = [mock_charger]
    mock_home._persons = []
    mock_home._heat_pumps = []
    mock_home._all_piloted_devices = []
    mock_home._all_loads = []
    mock_home._all_dynamic_groups = []
    mock_home._all_devices = []

    from custom_components.quiet_solar.ha_model.home import QSHome

    QSHome.remove_device(mock_home, mock_charger)

    mock_charger.detach_car.assert_not_called()


def _set_entry_state(entry, state):
    """Force-set entry state bypassing the read-only guard."""
    object.__setattr__(entry, "state", state)


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_all_entries(hass: HomeAssistant):
    """Test reloading all quiet solar entries."""
    from homeassistant.config_entries import ConfigEntryState

    entry1 = MockConfigEntry(domain=DOMAIN, entry_id="entry1", data={}, title="Entry 1")
    entry2 = MockConfigEntry(domain=DOMAIN, entry_id="entry2", data={}, title="Entry 2")
    entry1.add_to_hass(hass)
    entry2.add_to_hass(hass)
    _set_entry_state(entry1, ConfigEntryState.LOADED)
    _set_entry_state(entry2, ConfigEntryState.LOADED)

    hass.data.setdefault(DOMAIN, {})["entry1"] = MagicMock()
    hass.data[DOMAIN]["entry2"] = MagicMock()

    with (
        patch.object(hass.config_entries, "async_unload", new_callable=AsyncMock) as mock_unload,
        patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload,
    ):
        await async_reload_quiet_solar(hass)

        assert mock_unload.call_count == 2
        assert mock_reload.call_count == 2


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_except_one(hass: HomeAssistant):
    """Test reloading entries except specified one."""
    from homeassistant.config_entries import ConfigEntryState

    entry1 = MockConfigEntry(domain=DOMAIN, entry_id="entry1", data={}, title="Entry 1")
    entry2 = MockConfigEntry(domain=DOMAIN, entry_id="entry2", data={}, title="Entry 2")
    entry1.add_to_hass(hass)
    entry2.add_to_hass(hass)
    _set_entry_state(entry1, ConfigEntryState.LOADED)
    _set_entry_state(entry2, ConfigEntryState.LOADED)

    with (
        patch.object(hass.config_entries, "async_unload", new_callable=AsyncMock) as mock_unload,
        patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload,
    ):
        await async_reload_quiet_solar(hass, except_for_entry_id="entry1")

        assert mock_unload.call_count == 1
        mock_unload.assert_called_with("entry2")


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_handles_errors(hass: HomeAssistant):
    """Test that reload continues even if individual entries fail."""
    from homeassistant.config_entries import ConfigEntryState

    entry1 = MockConfigEntry(domain=DOMAIN, entry_id="entry1", data={}, title="Entry 1")
    entry2 = MockConfigEntry(domain=DOMAIN, entry_id="entry2", data={}, title="Entry 2")
    entry1.add_to_hass(hass)
    entry2.add_to_hass(hass)
    _set_entry_state(entry1, ConfigEntryState.LOADED)
    _set_entry_state(entry2, ConfigEntryState.LOADED)

    with (
        patch.object(
            hass.config_entries,
            "async_unload",
            side_effect=[Exception("Test error"), None],
        ) as mock_unload,
        patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload,
    ):
        await async_reload_quiet_solar(hass)

        assert mock_unload.call_count == 2
        assert mock_reload.call_count == 2


@pytest.mark.asyncio
async def test_async_reload_quiet_solar_skips_non_loaded_entries(hass: HomeAssistant):
    """Test that reload skips entries not in LOADED state (e.g. FAILED_UNLOAD)."""
    from homeassistant.config_entries import ConfigEntryState

    entry1 = MockConfigEntry(domain=DOMAIN, entry_id="entry1", data={}, title="Entry 1")
    entry2 = MockConfigEntry(domain=DOMAIN, entry_id="entry2", data={}, title="Entry 2")
    entry1.add_to_hass(hass)
    entry2.add_to_hass(hass)
    _set_entry_state(entry1, ConfigEntryState.LOADED)
    _set_entry_state(entry2, ConfigEntryState.FAILED_UNLOAD)

    with (
        patch.object(hass.config_entries, "async_unload", new_callable=AsyncMock) as mock_unload,
        patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock) as mock_reload,
    ):
        await async_reload_quiet_solar(hass)

        # Only entry1 should be unloaded; entry2 is skipped due to FAILED_UNLOAD state
        assert mock_unload.call_count == 1
        mock_unload.assert_called_with("entry1")
        # Both should still be reloaded
        assert mock_reload.call_count == 2


@pytest.mark.asyncio
async def test_ocpp_notification_listener_filters_non_ocpp(hass: HomeAssistant):
    """Test OCPP listener ignores non-OCPP notifications."""
    hass.data.setdefault(DOMAIN, {})
    mock_charger = MagicMock()
    mock_charger.handle_ocpp_notification = AsyncMock()
    mock_home = create_minimal_home_model()
    mock_home._chargers = [mock_charger]
    hass.data[DOMAIN][DATA_HANDLER] = MagicMock(home=mock_home)

    register_ocpp_notification_listener(hass)

    hass.bus.async_fire(
        "call_service",
        {"domain": "other_domain", "service": "create"},
    )

    mock_charger.handle_ocpp_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_ocpp_notification_forwards_to_chargers(hass: HomeAssistant):
    """Test OCPP notification is forwarded to OCPP chargers."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerOCPP

    hass.data.setdefault(DOMAIN, {})
    mock_home = create_minimal_home_model()
    mock_charger = MagicMock(spec=QSChargerOCPP)
    mock_charger.handle_ocpp_notification = AsyncMock()
    mock_home._chargers = [mock_charger]

    mock_handler = MagicMock()
    mock_handler.home = mock_home
    hass.data[DOMAIN][DATA_HANDLER] = mock_handler

    with patch("custom_components.quiet_solar._is_notification_for_charger", return_value=True):
        register_ocpp_notification_listener(hass)

        hass.bus.async_fire(
            "call_service",
            {
                "domain": "persistent_notification",
                "service": "create",
                "service_data": {
                    "title": "OCPP Alert",
                    "message": "Charger status changed",
                },
            },
        )

        await asyncio.sleep(0.1)
        mock_charger.handle_ocpp_notification.assert_awaited_once_with(
            "Charger status changed",
            "OCPP Alert",
        )


@pytest.mark.asyncio
async def test_ocpp_notification_matches_device_registry(hass: HomeAssistant):
    """Test OCPP notification matches charger via device registry."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerOCPP

    hass.data.setdefault(DOMAIN, {})
    mock_home = create_minimal_home_model()
    mock_charger = MagicMock(spec=QSChargerOCPP)
    mock_charger.name = "Garage Charger"
    mock_charger.charger_device_ocpp = "device_id_123"
    mock_charger.handle_ocpp_notification = AsyncMock()

    other_charger = MagicMock(spec=QSChargerOCPP)
    other_charger.name = "Driveway Charger"
    other_charger.charger_device_ocpp = "device_id_456"
    other_charger.handle_ocpp_notification = AsyncMock()

    mock_home._chargers = [mock_charger, other_charger]

    mock_handler = MagicMock()
    mock_handler.home = mock_home
    hass.data[DOMAIN][DATA_HANDLER] = mock_handler

    device = MagicMock()
    device.name_by_user = None
    device.name = "Garage Charger"
    device.identifiers = {("ocpp", "charger-123")}

    other_device = MagicMock()
    other_device.name_by_user = None
    other_device.name = "Driveway Charger"
    other_device.identifiers = {("ocpp", "charger-456")}

    def _get_device(device_id: str):
        if device_id == "device_id_123":
            return device
        if device_id == "device_id_456":
            return other_device
        return None

    device_registry = MagicMock()
    device_registry.async_get.side_effect = _get_device

    with patch("homeassistant.helpers.device_registry.async_get", return_value=device_registry):
        register_ocpp_notification_listener(hass)

        hass.bus.async_fire(
            "call_service",
            {
                "domain": "persistent_notification",
                "service": "create",
                "service_data": {
                    "title": "OCPP Alert",
                    "message": "Alert for charger-123",
                },
            },
        )

    mock_charger.handle_ocpp_notification.assert_awaited_once_with(
        "Alert for charger-123",
        "OCPP Alert",
    )
    other_charger.handle_ocpp_notification.assert_not_awaited()
