"""Test fixtures and helpers for quiet_solar tests."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

from homeassistant.const import CONF_NAME, Platform
from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CAR_BATTERY_CAPACITY,
)
from tests.factories import create_minimal_home_model
from tests.ha_tests.const import (
    MOCK_CAR_CONFIG,
    MOCK_CHARGER_CONFIG,
    MOCK_HOME_CONFIG,
    MOCK_SENSOR_STATES,
)


@dataclass
class FakeConfigEntry:
    """Fake config entry for testing."""
    entry_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    _update_listener: Callable | None = None
    _on_unload_callbacks: List[Callable] = field(default_factory=list)

    def add_update_listener(self, listener: Callable) -> Callable[[], None]:
        """Add update listener."""
        self._update_listener = listener
        return lambda: None

    def async_on_unload(self, callback: Callable) -> None:
        """Register unload callback."""
        self._on_unload_callbacks.append(callback)


class FakeState:
    """Fake Home Assistant state."""
    def __init__(self, entity_id: str, state: str, attributes: Dict[str, Any] | None = None):
        self.entity_id = entity_id
        self.domain = entity_id.split(".")[0]
        self.state = state
        self.attributes = attributes or {}


class FakeStates:
    """Fake states manager."""
    def __init__(self):
        self._states: Dict[str, FakeState] = {}

    def get(self, entity_id: str) -> FakeState | None:
        """Get state by entity ID."""
        return self._states.get(entity_id)

    def async_all(self, domains: List[str] | None = None) -> List[FakeState]:
        """Get all states, optionally filtered by domain."""
        if domains is None:
            return list(self._states.values())
        return [s for s in self._states.values() if s.entity_id.split('.')[0] in domains]

    def set(self, entity_id: str, state: str, attributes: Dict[str, Any] | None = None):
        """Set a state."""
        self._states[entity_id] = FakeState(entity_id, state, attributes)

    def async_available(self, entity_id: str) -> bool:
        """Check if an entity_id is available (not already used)."""
        return entity_id not in self._states


class FakeServices:
    """Fake services manager."""
    def __init__(self) -> None:
        self.calls: List[tuple[str, str, Dict[str, Any], bool]] = []
        self.registered: List[tuple[str, str]] = []

    async def async_call(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any] | None = None,
        blocking: bool = False,
        return_response: bool = False,
        target: Dict[str, Any] | None = None
    ):
        """Record service call."""
        # Merge target into data if provided (Home Assistant style)
        call_data = data.copy() if data else {}
        if target:
            call_data.update(target)
        self.calls.append((domain, service, call_data, blocking))
        if return_response:
            return {}
    
    def async_register(self, domain: str, service: str, service_func: Callable, schema: Any = None, supports_response: Any = None, *, description_placeholders: Any = None):
        """Register a service."""
        self.registered.append((domain, service))
    
    def has_service(self, domain: str, service: str) -> bool:
        """Check if service is registered."""
        return (domain, service) in self.registered

    def async_services_for_domain(self, domain: str) -> List[str]:
        """Get all service names for a domain."""
        return [service for d, service in self.registered if d == domain]


class FakeBus:
    """Fake event bus."""
    def __init__(self) -> None:
        self.listeners: Dict[str, List[Callable]] = {}

    def async_listen(self, event_type: str, callback: Callable, event_filter: Callable | None = None) -> Callable[[], None]:
        """Listen to events."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
        return lambda: self.listeners[event_type].remove(callback)

    async def async_fire(self, event_type: str, event_data: Dict[str, Any] | None = None):
        """Fire an event."""
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                event_obj = MagicMock()
                event_obj.data = event_data or {}
                await listener(event_obj)


class FakeConfigEntries:
    """Fake config entries manager."""
    def __init__(self, hass: "FakeHass") -> None:
        self.hass = hass
        self.forwarded: List[tuple[FakeConfigEntry, List[str]]] = []
        self.unloaded: List[tuple[FakeConfigEntry, List[str]]] = []
        self.reload_calls: List[str] = []
        self.update_calls: List[tuple[FakeConfigEntry, Dict[str, Any]]] = []

    async def async_forward_entry_setups(self, entry: FakeConfigEntry, platforms: List[str]) -> None:
        """Forward entry setup to platforms."""
        self.forwarded.append((entry, platforms))

    async def async_unload_platforms(self, entry: FakeConfigEntry, platforms: List[str]) -> bool:
        """Unload platforms."""
        self.unloaded.append((entry, platforms))
        return True

    async def async_reload(self, entry_id: str) -> None:
        """Reload entry."""
        self.reload_calls.append(entry_id)

    async def async_unload(self, entry_id: str) -> bool:
        """Unload entry."""
        return True

    def async_entries(self, domain: str) -> List[FakeConfigEntry]:
        """Get entries for domain."""
        return [
            entry for entry in self.hass.data.get(domain, {}).values()
            if isinstance(entry, FakeConfigEntry)
        ]

    def async_get_entry(self, entry_id: str) -> FakeConfigEntry | None:
        """Get entry by ID."""
        for entries in self.hass.data.values():
            if isinstance(entries, dict) and entry_id in entries:
                entry = entries[entry_id]
                if isinstance(entry, FakeConfigEntry):
                    return entry
        return None
    
    def async_get_known_entry(self, entry_id: str) -> FakeConfigEntry | None:
        """Get entry by ID (alias for async_get_entry)."""
        return self.async_get_entry(entry_id)

    def async_update_entry(
        self,
        entry: FakeConfigEntry,
        data: Dict[str, Any] | None = None,
        options: Dict[str, Any] | None = None,
        title: str | None = None
    ) -> None:
        """Update entry."""
        if data is not None:
            entry.data = data
        if options is not None:
            entry.options = options
        if title is not None:
            entry.title = title
        self.update_calls.append((entry, data or {}))


class FakeDeviceRegistry:
    """Fake device registry."""
    def __init__(self, devices: Dict[str, Any] | None = None) -> None:
        self._devices = devices or {}

    def async_get(self, device_id: str):
        """Get device by ID."""
        return self._devices.get(device_id)


class FakeEntityRegistry:
    """Fake entity registry."""
    def __init__(self):
        self._entities: Dict[str, Any] = {}

    def async_get(self, entity_id: str):
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def async_entries_for_device(self, device_id: str, include_disabled_entities: bool = False):
        """Get entities for device."""
        return []


class FakeConfig:
    """Fake Home Assistant config."""
    def __init__(self, config_dir: str = "/tmp/test_config"):
        self.config_dir = config_dir
    
    def path(self, *args) -> str:
        """Get path in config directory."""
        if args:
            return f"{self.config_dir}/{'/'.join(args)}"
        return self.config_dir


class FakeHass:
    """Fake Home Assistant instance for testing."""
    def __init__(self) -> None:
        # Create a new event loop for each test instance
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        except:
            self.loop = asyncio.get_event_loop()
        
        self.data: Dict[str, Dict[str, Any]] = {DOMAIN: {}}
        self.services = FakeServices()
        self.bus = FakeBus()
        self.config_entries = FakeConfigEntries(self)
        self.states = FakeStates()
        self.helpers = MagicMock()
        self.config = FakeConfig()

    def add_domain_entry(self, entry_id: str, value: Any) -> None:
        """Add entry to domain data."""
        self.data[DOMAIN][entry_id] = value

    async def async_add_executor_job(self, func: Callable, *args):
        """Run job in executor (just run it directly in tests)."""
        return func(*args)

    def create_task(self, coro, name: str | None = None):
        """Create a task (just schedules but doesn't run in tests)."""
        # In tests, we just discard the coroutine to avoid warnings
        # The actual task execution isn't needed for most tests
        import asyncio
        try:
            # Close the coroutine to avoid "coroutine was never awaited" warning
            coro.close()
        except:
            pass
        return None


@pytest.fixture
def fake_hass():
    """Provide fake hass instance."""
    return FakeHass()


@pytest.fixture
def mock_config_entry():
    """Provide mock config entry."""
    return FakeConfigEntry(
        entry_id="test_entry_123",
        data={CONF_NAME: "Test Device"},
        title="Test Device"
    )


@pytest.fixture
def mock_home_config_entry():
    """Provide mock home config entry."""
    return FakeConfigEntry(
        entry_id="home_entry_123",
        data={
            CONF_NAME: "Test Home",
            CONF_HOME_VOLTAGE: 230,
            CONF_IS_3P: True,
        },
        title="home: Test Home"
    )


@pytest.fixture
def mock_charger_config_entry():
    """Provide mock charger config entry."""
    return FakeConfigEntry(
        entry_id="charger_entry_123",
        data={
            CONF_NAME: "Test Charger",
            CONF_CHARGER_MIN_CHARGE: 6,
            CONF_CHARGER_MAX_CHARGE: 16,
            CONF_IS_3P: False,
        },
        title="charger: Test Charger"
    )


@pytest.fixture
def mock_car_config_entry():
    """Provide mock car config entry."""
    return FakeConfigEntry(
        entry_id="car_entry_123",
        data={
            CONF_NAME: "Test Car",
            CONF_CAR_BATTERY_CAPACITY: 50000,
        },
        title="car: Test Car"
    )


@pytest.fixture
def current_time():
    """Provide consistent test time."""
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)


@pytest.fixture
def mock_data_handler(fake_hass):
    """Provide mock data handler."""
    handler = MagicMock()
    handler.hass = fake_hass
    handler.home = None
    fake_hass.data[DOMAIN][DATA_HANDLER] = handler
    return handler


def create_mock_device(device_type: str, name: str = "Mock Device", **kwargs):
    """Create a mock device for testing."""
    device = MagicMock()
    device.device_type = device_type
    device.name = name
    device.device_id = f"mock_{device_type}_{name}"
    device.qs_enable_device = kwargs.get("qs_enable_device", True)
    device.get_platforms = MagicMock(return_value=[Platform.SENSOR, Platform.SWITCH])
    device.home = kwargs.get("home", None)
    
    for key, value in kwargs.items():
        setattr(device, key, value)
    
    return device


# ============================================================================
# Real Home Assistant Fixtures (for integration tests)
# ============================================================================
# These fixtures are for integration tests that use a real Home Assistant
# instance. They are separate from the unit test fixtures above.


@pytest.fixture
def hass_storage():
    """Fixture to provide storage dictionary for Home Assistant tests.

    This fixture overrides the default mock storage to prevent teardown errors
    when the storage tries to write data.
    """
    return {}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable loading custom integrations in all tests using real HA.
    
    This fixture is from pytest-homeassistant-custom-component and allows
    the test to discover and load integrations from the custom_components directory.
    """
    yield


@pytest.fixture
def real_home_config_entry(hass):
    """Create and register real mock config entry for home device."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    import uuid
    
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_HOME_CONFIG},
        entry_id=f"test_home_real_{uuid.uuid4().hex[:8]}",
        title="home: Test Home",
        unique_id=f"home_test_home_real_{uuid.uuid4().hex[:8]}"
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def real_charger_config_entry(hass):
    """Create and register real mock config entry for charger device."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    import uuid
    
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CHARGER_CONFIG},
        entry_id=f"test_charger_real_{uuid.uuid4().hex[:8]}",
        title="charger: Test Charger",
        unique_id=f"charger_test_charger_real_{uuid.uuid4().hex[:8]}"
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def real_car_config_entry(hass):
    """Create and register real mock config entry for car device."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    import uuid
    
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG},
        entry_id=f"test_car_real_{uuid.uuid4().hex[:8]}",
        title="car: Test Car",
        unique_id=f"car_test_car_real_{uuid.uuid4().hex[:8]}"
    )
    entry.add_to_hass(hass)
    return entry


def _apply_mock_states(hass) -> None:
    """Seed HA state machine with mock sensor states."""
    for entity_id, data in MOCK_SENSOR_STATES.items():
        hass.states.async_set(entity_id, data["state"], data.get("attributes", {}))


@pytest.fixture
def mock_solcast_data():
    """Mock Solcast solar forecast data."""
    return {
        "forecasts": [
            {"period_end": "2024-06-15T13:00:00Z", "pv_estimate": 2.5},
            {"period_end": "2024-06-15T14:00:00Z", "pv_estimate": 3.0},
            {"period_end": "2024-06-15T15:00:00Z", "pv_estimate": 3.5},
        ]
    }


@pytest.fixture
def mock_external_apis():
    """Mock all external API calls for integration tests."""
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock) as mock_forecast:
        yield {
            "forecast": mock_forecast,
        }


@pytest.fixture
async def setup_home_first(hass, real_home_config_entry):
    """Setup home device first and wait for it to complete.
    
    This fixture ensures home is fully initialized before other devices can be added.
    """
    _apply_mock_states(hass)
    with patch("custom_components.quiet_solar.ha_model.home.QSHome.update_forecast_probers", new_callable=AsyncMock):
        result = await hass.config_entries.async_setup(real_home_config_entry.entry_id)
        await hass.async_block_till_done()
        assert result is True
    return real_home_config_entry


@pytest.fixture
async def home_and_charger(hass, setup_home_first):
    """Setup home and charger together in correct order.
    
    Home is set up first via setup_home_first fixture, then charger is added.
    """
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    import uuid
    
    # Home is already set up from setup_home_first
    home_entry = setup_home_first
    
    # Create charger entry with unique ID (using generic charger type)
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CHARGER_CONFIG},
        entry_id=f"test_charger_composite_{uuid.uuid4().hex[:8]}",
        title="charger: Test Charger",
        unique_id=f"charger_test_composite_{uuid.uuid4().hex[:8]}"
    )
    charger_entry.add_to_hass(hass)
    
    # Now setup charger (home is already initialized)
    result = await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()
    
    return home_entry, charger_entry


@pytest.fixture
async def home_and_car(hass, setup_home_first):
    """Setup home and car together in correct order.
    
    Home is set up first via setup_home_first fixture, then car is added.
    """
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    import uuid
    
    # Home is already set up from setup_home_first
    home_entry = setup_home_first
    
    # Create car entry with unique ID
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG},
        entry_id=f"test_car_composite_{uuid.uuid4().hex[:8]}",
        title="car: Test Car",
        unique_id=f"car_test_composite_{uuid.uuid4().hex[:8]}"
    )
    car_entry.add_to_hass(hass)
    
    # Now setup car (home is already initialized)
    result = await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()
    
    return home_entry, car_entry


@pytest.fixture
async def home_charger_and_car(hass, setup_home_first):
    """Setup home, charger, and car together in correct order.
    
    Home is set up first, then charger, then car.
    """
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    import uuid
    
    # Home is already set up
    home_entry = setup_home_first
    
    # Create charger entry (using generic charger type)
    charger_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CHARGER_CONFIG},
        entry_id=f"test_charger_all_{uuid.uuid4().hex[:8]}",
        title="charger: Test Charger",
        unique_id=f"charger_test_all_{uuid.uuid4().hex[:8]}"
    )
    charger_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(charger_entry.entry_id)
    await hass.async_block_till_done()
    
    # Create car entry
    car_entry = MockConfigEntry(
        domain=DOMAIN,
        data={**MOCK_CAR_CONFIG},
        entry_id=f"test_car_all_{uuid.uuid4().hex[:8]}",
        title="car: Test Car",
        unique_id=f"car_test_all_{uuid.uuid4().hex[:8]}"
    )
    car_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(car_entry.entry_id)
    await hass.async_block_till_done()
    
    return home_entry, charger_entry, car_entry


# ============================================================================
# Additional Test Fixtures for Comprehensive Testing
# ============================================================================


class MockLazyState:
    """Mock LazyState for testing history-based functions."""
    def __init__(self, entity_id: str, state: str, attributes: dict | None = None,
                 last_updated: "datetime" | None = None):
        import pytz
        from datetime import datetime
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes or {}
        self.last_updated = last_updated or datetime.now(pytz.UTC)


@pytest.fixture
def mock_lazy_state_factory():
    """Factory fixture for creating MockLazyState objects."""
    def factory(entity_id: str, state: str, attributes: dict | None = None,
                last_updated: "datetime" | None = None):
        return MockLazyState(entity_id, state, attributes, last_updated)
    return factory


@pytest.fixture
def mock_recorder_history():
    """Mock recorder history for testing history-based methods.

    Returns a function that can be used to configure load_from_history return values.
    """
    histories = {}

    def set_history(entity_id: str, states: list):
        """Set the history for an entity."""
        histories[entity_id] = states

    async def mock_load_from_history(hass, entity_id, start, end, **kwargs):
        """Mock implementation of load_from_history."""
        return histories.get(entity_id, [])

    return {
        "set_history": set_history,
        "mock_fn": mock_load_from_history,
    }


@pytest.fixture
def mock_gps_states(mock_lazy_state_factory):
    """Fixture for creating GPS location state sequences."""
    import pytz
    from datetime import datetime, timedelta

    def create_gps_path(entity_id: str, locations: list, start_time: datetime | None = None):
        """Create a list of states representing a GPS path.

        Args:
            entity_id: The entity ID for the tracker
            locations: List of tuples (state_name, lat, lon, minutes_offset)
            start_time: Base time for the path

        Returns:
            List of MockLazyState objects
        """
        if start_time is None:
            start_time = datetime.now(pytz.UTC) - timedelta(hours=1)

        states = []
        for state_name, lat, lon, minutes in locations:
            time = start_time + timedelta(minutes=minutes)
            state = mock_lazy_state_factory(
                entity_id,
                state_name,
                {"latitude": lat, "longitude": lon},
                time
            )
            states.append(state)
        return states

    return create_gps_path


@pytest.fixture
def mock_charger_group_factory(fake_hass):
    """Factory fixture for creating mock charger groups."""
    from custom_components.quiet_solar.ha_model.charger import QSChargerGroup
    from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup

    def factory(chargers: list | None = None, name: str = "Test Group"):
        """Create a mock charger group.

        Args:
            chargers: List of charger mocks to include
            name: Name for the group
        """
        mock_dyn_group = MagicMock(spec=QSDynamicGroup)
        mock_dyn_group.name = name
        mock_dyn_group._childrens = chargers or []
        mock_dyn_group.home = create_minimal_home_model()

        return QSChargerGroup(mock_dyn_group)

    return factory


@pytest.fixture
def mock_person_with_history(fake_hass):
    """Factory fixture for creating persons with pre-populated mileage history."""
    from custom_components.quiet_solar.ha_model.person import QSPerson
    import pytz
    from datetime import datetime, timedelta

    def factory(name: str = "Test Person", days_of_history: int = 7):
        """Create a person with history.

        Args:
            name: Person name
            days_of_history: Number of days of mileage history to generate
        """
        config_entry = FakeConfigEntry(
            entry_id=f"person_{name.lower().replace(' ', '_')}",
            data={CONF_NAME: name},
        )

        home = create_minimal_home_model()

        data_handler = MagicMock()
        data_handler.home = home
        fake_hass.data[DOMAIN][DATA_HANDLER] = data_handler

        person = QSPerson(
            hass=fake_hass,
            config_entry=config_entry,
            home=home,
            name=name,
            person_entity_id=f"person.{name.lower().replace(' ', '_')}"
        )

        # Add historical data
        base_date = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(days_of_history):
            day = base_date - timedelta(days=days_of_history - i)
            leave_time = day.replace(hour=8, minute=30)
            mileage = 40.0 + (i * 5)  # Varying mileage
            person.add_to_mileage_history(day, mileage, leave_time)

        person.has_been_initialized = True

        return person

    return factory
