"""Tests for calendar event methods in HADeviceMixin.

This module tests:
- get_next_scheduled_event
- get_next_scheduled_events

Uses a mock calendar service to properly test the calendar integration.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any
from collections.abc import Callable

import pytz
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse

from custom_components.quiet_solar.const import (
    DOMAIN,
    DATA_HANDLER,
    FLOATING_PERIOD_S,
)

pytestmark = pytest.mark.usefixtures("mock_sensor_states")


# =============================================================================
# Fixtures for calendar events
# =============================================================================

def _to_local_iso(dt: datetime) -> str:
    """Convert datetime to ISO format string without timezone info.

    This matches how Home Assistant calendar stores datetime in state attributes.
    The datetime is first converted to local time, then made naive and formatted.
    """
    local_dt = dt.astimezone()  # Convert to local timezone
    return local_dt.replace(tzinfo=None).isoformat()


@pytest.fixture
def time_now() -> datetime:
    """Return a fixed 'now' time for tests."""
    return datetime(2026, 1, 22, 10, 0, 0, tzinfo=pytz.UTC)


@pytest.fixture
def calendar_entity_id() -> str:
    """Return the test calendar entity ID."""
    return "calendar.test_calendar"


@pytest.fixture
def future_event(time_now: datetime) -> dict:
    """Return a future event starting in 1 hour."""
    return {
        "start": _to_local_iso(time_now + timedelta(hours=1)),
        "end": _to_local_iso(time_now + timedelta(hours=2)),
    }


@pytest.fixture
def second_future_event(time_now: datetime) -> dict:
    """Return a second future event starting in 3 hours."""
    return {
        "start": _to_local_iso(time_now + timedelta(hours=3)),
        "end": _to_local_iso(time_now + timedelta(hours=4)),
    }


@pytest.fixture
def third_future_event(time_now: datetime) -> dict:
    """Return a third future event starting in 5 hours."""
    return {
        "start": _to_local_iso(time_now + timedelta(hours=5)),
        "end": _to_local_iso(time_now + timedelta(hours=6)),
    }


@pytest.fixture
def currently_running_event(time_now: datetime) -> dict:
    """Return an event that is currently running (started 30 min ago, ends in 30 min)."""
    return {
        "start": _to_local_iso(time_now - timedelta(minutes=30)),
        "end": _to_local_iso(time_now + timedelta(minutes=30)),
    }


@pytest.fixture
def past_event(time_now: datetime) -> dict:
    """Return a past event that has already ended."""
    return {
        "start": _to_local_iso(time_now - timedelta(hours=2)),
        "end": _to_local_iso(time_now - timedelta(hours=1)),
    }


@pytest.fixture
async def mock_calendar_service(hass: HomeAssistant, calendar_entity_id: str):
    """Register a mock calendar.get_events service that can be configured per test.

    Returns a function to set the events that will be returned by the service.
    """
    events_to_return: list[dict] = []

    async def mock_get_events(call: ServiceCall) -> ServiceResponse:
        """Mock implementation of calendar.get_events service."""
        return {
            calendar_entity_id: {
                "events": events_to_return
            }
        }

    # Register the mock service
    hass.services.async_register(
        "calendar",
        "get_events",
        mock_get_events,
        supports_response=SupportsResponse.ONLY,
    )

    def set_events(events: list[dict]) -> None:
        """Set the events that will be returned by the mock service."""
        nonlocal events_to_return
        events_to_return = events

    yield set_events

    # Cleanup
    hass.services.async_remove("calendar", "get_events")


# =============================================================================
# Tests for get_next_scheduled_event
# =============================================================================

class TestGetNextScheduledEvent:
    """Tests for the get_next_scheduled_event method."""

    async def test_returns_none_when_no_calendar_configured(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
    ) -> None:
        """Test returns (None, None) when no calendar is configured."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = None

        start_time, end_time = await home.get_next_scheduled_event(time_now)

        assert start_time is None
        assert end_time is None

    async def test_returns_none_when_calendar_unavailable(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
    ) -> None:
        """Test returns (None, None) when calendar state is unavailable."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, STATE_UNAVAILABLE, {})

        start_time, end_time = await home.get_next_scheduled_event(time_now)

        assert start_time is None
        assert end_time is None

    async def test_returns_none_when_calendar_unknown(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
    ) -> None:
        """Test returns (None, None) when calendar state is unknown."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, STATE_UNKNOWN, {})

        start_time, end_time = await home.get_next_scheduled_event(time_now)

        assert start_time is None
        assert end_time is None

    async def test_returns_future_event_from_state(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
    ) -> None:
        """Test returns future event from calendar state attributes."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": future_event["start"],
                "end_time": future_event["end"],
            },
        )

        start_time, end_time = await home.get_next_scheduled_event(time_now)

        assert start_time is not None
        assert end_time is not None
        expected_start = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert start_time == expected_start
        assert end_time == expected_end

    async def test_returns_currently_running_event_when_flag_true(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        currently_running_event: dict,
    ) -> None:
        """Test returns currently running event when give_currently_running_event=True."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": currently_running_event["start"],
                "end_time": currently_running_event["end"],
            },
        )

        start_time, end_time = await home.get_next_scheduled_event(
            time_now, give_currently_running_event=True
        )

        assert start_time is not None
        assert end_time is not None
        expected_start = datetime.fromisoformat(currently_running_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(currently_running_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert start_time == expected_start
        assert end_time == expected_end

    async def test_skips_running_event_and_returns_next_from_service(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        currently_running_event: dict,
        future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test with flag=False skips running event and returns next from service."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        # Set state to running event
        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": currently_running_event["start"],
                "end_time": currently_running_event["end"],
            },
        )

        # Mock service returns the running event and a future event
        mock_calendar_service([currently_running_event, future_event])

        # With flag=False, should skip running event and return future event
        start_time, end_time = await home.get_next_scheduled_event(
            time_now, give_currently_running_event=False
        )

        assert start_time is not None
        assert end_time is not None
        expected_start = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert start_time == expected_start
        assert end_time == expected_end

    async def test_returns_none_for_past_event(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        past_event: dict,
    ) -> None:
        """Test returns (None, None) when calendar shows a past event."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": past_event["start"],
                "end_time": past_event["end"],
            },
        )

        start_time, end_time = await home.get_next_scheduled_event(time_now)

        assert start_time is None
        assert end_time is None

    async def test_handles_missing_state_attributes(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
    ) -> None:
        """Test handles calendar state without start_time/end_time attributes."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        start_time, end_time = await home.get_next_scheduled_event(time_now)

        assert start_time is None
        assert end_time is None


# =============================================================================
# Tests for get_next_scheduled_events
# =============================================================================

class TestGetNextScheduledEvents:
    """Tests for the get_next_scheduled_events method."""

    async def test_returns_empty_list_when_no_calendar_configured(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
    ) -> None:
        """Test returns empty list when no calendar is configured."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = None

        events = await home.get_next_scheduled_events(time_now)

        assert events == []

    async def test_returns_empty_list_when_calendar_unavailable_with_max_1(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
    ) -> None:
        """Test returns empty list when calendar state is unavailable (optimization path)."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, STATE_UNAVAILABLE, {})

        events = await home.get_next_scheduled_events(time_now, max_number_of_events=1)

        assert events == []

    async def test_optimization_returns_future_event_from_state(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
    ) -> None:
        """Test max_number_of_events=1 uses calendar state for optimization."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": future_event["start"],
                "end_time": future_event["end"],
            },
        )

        events = await home.get_next_scheduled_events(time_now, max_number_of_events=1)

        assert len(events) == 1
        expected_start = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start, expected_end)

    async def test_optimization_returns_running_event_when_flag_true(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        currently_running_event: dict,
    ) -> None:
        """Test optimization returns running event when give_currently_running_event=True."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": currently_running_event["start"],
                "end_time": currently_running_event["end"],
            },
        )

        events = await home.get_next_scheduled_events(
            time_now, give_currently_running_event=True, max_number_of_events=1
        )

        assert len(events) == 1
        expected_start = datetime.fromisoformat(currently_running_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(currently_running_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start, expected_end)

    async def test_returns_multiple_events_from_service(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
        second_future_event: dict,
        third_future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test returns multiple events from calendar service."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Set up mock to return 3 events
        mock_calendar_service([future_event, second_future_event, third_future_event])

        events = await home.get_next_scheduled_events(time_now)

        assert len(events) == 3

        # Verify events are in order
        expected_start_1 = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end_1 = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start_1, expected_end_1)

        expected_start_2 = datetime.fromisoformat(second_future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end_2 = datetime.fromisoformat(second_future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[1] == (expected_start_2, expected_end_2)

        expected_start_3 = datetime.fromisoformat(third_future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end_3 = datetime.fromisoformat(third_future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[2] == (expected_start_3, expected_end_3)

    async def test_respects_max_number_of_events_limit(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
        second_future_event: dict,
        third_future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test respects max_number_of_events parameter."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Service returns 3 events
        mock_calendar_service([future_event, second_future_event, third_future_event])

        # But we only want 2
        events = await home.get_next_scheduled_events(time_now, max_number_of_events=2)

        assert len(events) == 2

        expected_start_1 = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end_1 = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start_1, expected_end_1)

        expected_start_2 = datetime.fromisoformat(second_future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end_2 = datetime.fromisoformat(second_future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[1] == (expected_start_2, expected_end_2)

    async def test_excludes_currently_running_event_when_flag_false(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        currently_running_event: dict,
        future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test excludes currently running event when give_currently_running_event=False."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Service returns running event and future event
        mock_calendar_service([currently_running_event, future_event])

        events = await home.get_next_scheduled_events(
            time_now, give_currently_running_event=False
        )

        # Should only return the future event
        assert len(events) == 1
        expected_start = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start, expected_end)

    async def test_includes_currently_running_event_when_flag_true(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        currently_running_event: dict,
        future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test includes currently running event when give_currently_running_event=True."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Service returns running event and future event
        mock_calendar_service([currently_running_event, future_event])

        events = await home.get_next_scheduled_events(
            time_now, give_currently_running_event=True
        )

        # Should return both events
        assert len(events) == 2
        expected_start_1 = datetime.fromisoformat(currently_running_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end_1 = datetime.fromisoformat(currently_running_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start_1, expected_end_1)

    async def test_excludes_past_events(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        past_event: dict,
        future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test excludes events that have already ended."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Service returns past event and future event
        mock_calendar_service([past_event, future_event])

        events = await home.get_next_scheduled_events(time_now)

        # Should only return the future event
        assert len(events) == 1
        expected_start = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start, expected_end)

    async def test_excludes_events_beyond_search_period(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test excludes events that start after the search period."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Create event beyond FLOATING_PERIOD_S
        end_time = time_now + timedelta(seconds=FLOATING_PERIOD_S)
        far_future_event = {
            "start": _to_local_iso(end_time + timedelta(hours=1)),
            "end": _to_local_iso(end_time + timedelta(hours=2)),
        }

        # Service returns one valid event and one beyond search period
        mock_calendar_service([future_event, far_future_event])

        events = await home.get_next_scheduled_events(time_now)

        # Should only return the event within the search period
        assert len(events) == 1
        expected_start = datetime.fromisoformat(future_event["start"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        expected_end = datetime.fromisoformat(future_event["end"]).replace(tzinfo=None).astimezone(tz=pytz.UTC)
        assert events[0] == (expected_start, expected_end)

    async def test_returns_events_sorted_by_start_time(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
        second_future_event: dict,
        third_future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test events are returned sorted by start time."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Return events in reverse order
        mock_calendar_service([third_future_event, future_event, second_future_event])

        events = await home.get_next_scheduled_events(time_now)

        # Should be sorted by start time
        assert len(events) == 3
        assert events[0][0] < events[1][0] < events[2][0]

    async def test_handles_empty_service_response(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test handles empty calendar response gracefully."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        # Service returns no events
        mock_calendar_service([])

        events = await home.get_next_scheduled_events(time_now)

        assert events == []

    async def test_no_limit_when_max_is_zero(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
        future_event: dict,
        second_future_event: dict,
        third_future_event: dict,
        mock_calendar_service: Callable[[list[dict]], None],
    ) -> None:
        """Test no limit when max_number_of_events is 0."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        hass.states.async_set(calendar_entity_id, "on", {})

        mock_calendar_service([future_event, second_future_event, third_future_event])

        # max_number_of_events=0 means no limit
        events = await home.get_next_scheduled_events(time_now, max_number_of_events=0)

        assert len(events) == 3

    async def test_event_beyond_search_period_excluded_in_optimization(
        self,
        hass: HomeAssistant,
        home_config_entry: ConfigEntry,
        time_now: datetime,
        calendar_entity_id: str,
    ) -> None:
        """Test event starting beyond FLOATING_PERIOD_S is excluded in optimization path."""
        await hass.config_entries.async_setup(home_config_entry.entry_id)
        await hass.async_block_till_done()

        data_handler = hass.data[DOMAIN][DATA_HANDLER]
        home = data_handler.home
        home.calendar = calendar_entity_id

        # Create event that starts after the search period
        end_time = time_now + timedelta(seconds=FLOATING_PERIOD_S)
        far_future_start = end_time + timedelta(hours=1)
        far_future_end = end_time + timedelta(hours=2)

        hass.states.async_set(
            calendar_entity_id,
            "on",
            {
                "start_time": _to_local_iso(far_future_start),
                "end_time": _to_local_iso(far_future_end),
            },
        )

        events = await home.get_next_scheduled_events(time_now, max_number_of_events=1)

        # Event beyond search period should be excluded
        assert events == []
