"""Test helper classes for quiet_solar tests.

This module contains FakeHass, FakeConfigEntry, FakeState and other helper classes
that are used across multiple test files. These are imported directly rather than
through conftest.py to avoid import conflicts with Home Assistant core's tests.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock, AsyncMock

import pytz

from homeassistant.const import CONF_NAME
from custom_components.quiet_solar.const import DOMAIN


@dataclass
class FakeConfigEntry:
    """Fake config entry for testing."""
    entry_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    _update_listener: Callable | None = None

    @property
    def unique_id(self) -> str | None:
        return self.data.get("unique_id", self.entry_id)

    def add_update_listener(self, listener: Callable) -> Callable[[], None]:
        """Add update listener."""
        self._update_listener = listener
        return lambda: None


class FakeState:
    """Fake Home Assistant state object."""
    def __init__(self, entity_id: str, state: str, attributes: Dict[str, Any] | None = None):
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes or {}
        self.last_changed = datetime.now(pytz.UTC)
        self.last_updated = datetime.now(pytz.UTC)


class FakeStates:
    """Fake Home Assistant states object."""
    def __init__(self):
        self._states: Dict[str, FakeState] = {}

    def get(self, entity_id: str) -> FakeState | None:
        """Get state by entity_id."""
        return self._states.get(entity_id)

    def async_all(self, domains: List[str] | None = None) -> List[FakeState]:
        """Get all states, optionally filtered by domain."""
        if domains is None:
            return list(self._states.values())
        return [s for s in self._states.values() if s.entity_id.split('.')[0] in domains]

    def set(self, entity_id: str, state: str, attributes: Dict[str, Any] | None = None):
        """Set state."""
        self._states[entity_id] = FakeState(entity_id, state, attributes)


class FakeServiceCall:
    """Fake Home Assistant service call."""
    def __init__(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any] | None = None,
        context: Any = None,
        target: Dict[str, Any] | None = None
    ):
        self.domain = domain
        self.service = service
        self.data = data or {}
        self.context = context
        self.target = target


class FakeServices:
    """Fake Home Assistant services registry."""
    def __init__(self):
        self._services: Dict[str, Dict[str, Callable]] = {}
        # calls stores tuples of (domain, service, data, blocking) for compatibility
        self.calls: List[tuple[str, str, Dict[str, Any], bool]] = []
        self.registered: List[tuple[str, str]] = []

    def has_service(self, domain: str, service: str) -> bool:
        """Check if service exists."""
        return (domain, service) in self.registered or (domain in self._services and service in self._services[domain])

    async def async_call(
        self,
        domain: str,
        service: str,
        data: Dict[str, Any] | None = None,
        blocking: bool = False,
        return_response: bool = False,
        target: Dict[str, Any] | None = None,
        **kwargs
    ) -> None:
        """Call a service."""
        # Merge target into data if provided (Home Assistant style)
        call_data = data.copy() if data else {}
        if target:
            call_data.update(target)
        self.calls.append((domain, service, call_data, blocking))
        if return_response:
            return {}
        if domain in self._services and service in self._services[domain]:
            await self._services[domain][service](FakeServiceCall(domain, service, call_data))

    def async_register(self, domain: str, service: str, service_func: Callable, schema: Any = None, supports_response: Any = None, *, description_placeholders: Any = None):
        """Register a service."""
        self.registered.append((domain, service))
        if domain not in self._services:
            self._services[domain] = {}
        self._services[domain][service] = service_func

    def async_services_for_domain(self, domain: str) -> List[str]:
        """Get all service names for a domain."""
        return [service for d, service in self.registered if d == domain]


class FakeBus:
    """Fake Home Assistant event bus."""
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def async_listen(self, event_type: str, callback: Callable, event_filter: Callable | None = None) -> Callable[[], None]:
        """Listen for events."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        return lambda: self._listeners[event_type].remove(callback) if callback in self._listeners[event_type] else None

    async def async_fire(self, event_type: str, event_data: Dict[str, Any] | None = None):
        """Fire an event."""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)


class FakeEntity:
    """Fake Home Assistant entity."""
    def __init__(self, entity_id: str = None):
        self.entity_id = entity_id
        self._attr_native_value = None
        self._attr_available = True
        self.async_write_ha_state = MagicMock()


class FakeEntityRegistry:
    """Fake entity registry."""
    def __init__(self):
        self._entities: Dict[str, FakeEntity] = {}

    def async_get(self, entity_id: str) -> FakeEntity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)


class FakeConfigEntries:
    """Fake config entries manager."""
    def __init__(self, hass):
        self._entries: Dict[str, FakeConfigEntry] = {}
        self._hass = hass

    def async_get_entry(self, entry_id: str) -> FakeConfigEntry | None:
        """Get config entry by ID."""
        return self._entries.get(entry_id)

    async def async_setup(self, entry_id: str) -> bool:
        """Set up config entry."""
        return True

    def async_get_known_entry(self, entry_id: str) -> FakeConfigEntry | None:
        """Get known config entry."""
        return self._entries.get(entry_id)

    async def async_add_entry(
        self,
        entry_id: str,
        data: Dict[str, Any] | None = None,
        options: Dict[str, Any] | None = None,
        title: str | None = None
    ):
        """Add a config entry."""
        entry = FakeConfigEntry(entry_id, data or {}, options or {}, title)
        self._entries[entry_id] = entry
        return entry


class FakeDeviceRegistry:
    """Fake device registry."""
    def __init__(self, devices: Dict[str, Any] | None = None) -> None:
        self._devices = devices or {}

    def async_get_device(self, identifiers: set | None = None, connections: set | None = None):
        """Get device by identifiers or connections."""
        if identifiers:
            for device_id, device in self._devices.items():
                device_identifiers = device.get("identifiers", set())
                if identifiers & device_identifiers:
                    return SimpleNamespace(**device, id=device_id)
        return None


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
        self.helpers = SimpleNamespace()
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
        try:
            # Close the coroutine to avoid "coroutine was never awaited" warning
            coro.close()
        except:
            pass
        return None


def create_mock_device(device_type: str, name: str = "Mock Device", **kwargs):
    """Create a mock device for testing."""
    from homeassistant.const import Platform

    mock_device = MagicMock()
    mock_device.name = name
    mock_device.device_type = device_type
    mock_device.device_id = f"mock_{device_type}_{name}"
    mock_device.entity_id = f"sensor.{name.lower().replace(' ', '_')}"
    mock_device.unique_id = f"{device_type}_{name.lower().replace(' ', '_')}"
    mock_device.qs_enable_device = kwargs.get("qs_enable_device", True)
    mock_device._constraints = []
    mock_device.ha_entities = {}
    mock_device.get_platforms = MagicMock(return_value=[Platform.SENSOR, Platform.SWITCH])
    mock_device.home = kwargs.get("home", None)

    for key, value in kwargs.items():
        setattr(mock_device, key, value)

    return mock_device
