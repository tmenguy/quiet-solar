"""QS-359 — OCPP integration v0.11.4 -> v0.12.0 compatibility and hardening.

Covers the three product changes in `QSChargerOCPP` (plus the two `QSChargerGeneric`
hooks): the task-hop amp command (Block 1), the `ocpp.set_charge_rate` fallback mode for
chargers that reject `ChargePointMaxProfile` (Item 2), and the notify-only stack-level
clip detector (Item 1).

Reuses the real-object test infrastructure from `tests.test_charger_coverage_deep`
(precedent: `tests/test_bug_99_idle_command.py`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytz
from homeassistant.components import number
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.exceptions import HomeAssistantError

from custom_components.quiet_solar.const import DEVICE_STATUS_CHANGE_ERROR
from custom_components.quiet_solar.ha_model.charger import (
    OCPP_CLIP_DETECT_WINDOW_S,
    OCPP_FALLBACK_CONN_ID,
    OCPP_STATION_PROFILE_REJECTION_MARKER,
    QSOCPPv16v201ChargePointStatus,
)
from tests.test_charger_coverage_deep import (
    _create_charger,
    _create_ocpp_charger,
    _init_charger_states,
    _make_hass,
    _make_home,
)

_T0 = pytz.UTC.localize(datetime(2026, 9, 21, 12, 0, 0))


# =============================================================================
# Module-local test infrastructure
# =============================================================================


def _capture_task(hass) -> list:
    """Replace `async_create_task`'s side effect with one that captures the coroutine.

    Returns the list the test awaits (or closes) before it ends.
    """
    captured: list = []
    hass.async_create_task = MagicMock(side_effect=lambda coro: captured.append(coro) or MagicMock())
    return captured


def _raise_for_number(hass, exc, calls) -> None:
    """Record every `hass.services.async_call` and raise `exc` for the number domain."""

    async def fn(domain, service, data=None, **kwargs):
        calls.append((domain, service, data))
        if domain == number.DOMAIN:
            raise exc

    hass.services.async_call = AsyncMock(side_effect=fn)


def _record_calls(hass, calls) -> None:
    """Record every `hass.services.async_call` without raising."""

    async def fn(domain, service, data=None, **kwargs):
        calls.append((domain, service, data))

    hass.services.async_call = AsyncMock(side_effect=fn)


def _seed(
    ch,
    *,
    status=QSOCPPv16v201ChargePointStatus.charging,
    offered="14.0",
    number_state="16",
    charge_enabled=True,
    plugged=True,
) -> None:
    """Seed the clip detector's gates. The gates read the probe cache, not `hass.states`,
    except the number ack which reads `hass.states.get`."""
    ch.is_charge_enabled = MagicMock(return_value=charge_enabled)
    ch.is_not_plugged = MagicMock(return_value=not plugged)
    ch.get_sensor_latest_possible_valid_value = MagicMock(
        side_effect=lambda eid, **kw: {
            ch.charger_status_sensor_unfiltered: status,
            ch.charger_ocpp_current_offered: offered,
        }.get(eid)
    )
    ch.hass.states.get.side_effect = lambda eid=None: {
        ch.charger_max_charging_current_number: SimpleNamespace(state=number_state, attributes={})
    }.get(eid)


def _detector_charger(name="OcppCharger"):
    """Build an OCPP charger with the optional `current_offered` sensor bound."""
    hass = _make_hass()
    home = _make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    devname = name.lower().replace(" ", "_")
    ch = _create_ocpp_charger(hass, home, name=name, extra_entity_ids=(f"sensor.{devname}_current_offered",))
    _init_charger_states(ch)
    return hass, home, ch


# =============================================================================
# AC1 — accepted path unchanged
# =============================================================================


@pytest.mark.asyncio
async def test_ac1_accepted_path_non_blocking(caplog):
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    calls: list = []
    _record_calls(hass, calls)
    captured = _capture_task(hass)
    hass.states.get.side_effect = lambda eid=None: {
        ch.charger_max_charging_current_number: SimpleNamespace(state="16", attributes={})
    }.get(eid)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = await ch.low_level_set_max_charging_current(16, _T0)
        assert result is True
        assert len(captured) == 1
        await captured[0]

    # number.set_value once, blocking=True, expected payload
    number_calls = [c for c in calls if c[0] == number.DOMAIN]
    assert len(number_calls) == 1
    assert number_calls[0][2] == {ATTR_ENTITY_ID: ch.charger_max_charging_current_number, number.ATTR_VALUE: 16}
    assert hass.services.async_call.await_args[1]["blocking"] is True
    # no set_charge_rate; ack reads the number
    assert not [c for c in calls if c[0] == "ocpp"]
    assert ch.get_max_charging_amp_per_phase() == 16.0
    assert ch._ocpp_station_profile_rejected is False
    assert ch._ocpp_station_profile_rejection_streak == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_ac1_accepted_path_blocking_caller_returns_real_outcome():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    calls: list = []
    _record_calls(hass, calls)
    hass.async_create_task = MagicMock()  # should NOT be called

    result = await ch.low_level_set_max_charging_current(16, _T0, blocking=True)
    assert result is True
    hass.async_create_task.assert_not_called()
    assert [c for c in calls if c[0] == number.DOMAIN]


# =============================================================================
# AC2 — rejection streak
# =============================================================================


@pytest.mark.parametrize("word", ["Rejected", "NotSupported"])
@pytest.mark.asyncio
async def test_ac2_rejection_streak(caplog, word):
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    marker = HomeAssistantError(f"Failed to set variable: {OCPP_STATION_PROFILE_REJECTION_MARKER}: {word}")
    calls: list = []
    _raise_for_number(hass, marker, calls)

    # First marker rejection -> streak 1, mode off, no WARNING, one DEBUG note.
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        res = await ch.low_level_set_max_charging_current(16, _T0, blocking=True)
    assert res is False
    assert ch._ocpp_station_profile_rejection_streak == 1
    assert ch._ocpp_station_profile_rejected is False
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any(
        "station profile rejected once by" in r.getMessage() and r.levelno == logging.DEBUG
        for r in caplog.records
    )

    # Accepted write in between resets the streak to 0.
    _record_calls(hass, calls)
    hass.states.get.side_effect = lambda eid=None: {
        ch.charger_max_charging_current_number: SimpleNamespace(state="16", attributes={})
    }.get(eid)
    res = await ch.low_level_set_max_charging_current(16, _T0, blocking=True)
    assert res is True
    assert ch._ocpp_station_profile_rejection_streak == 0

    # Two consecutive marker rejections -> latch with exactly one warning.
    _raise_for_number(hass, marker, calls)
    await ch.low_level_set_max_charging_current(16, _T0, blocking=True)
    assert ch._ocpp_station_profile_rejected is False
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await ch.low_level_set_max_charging_current(16, _T0, blocking=True)
    assert ch._ocpp_station_profile_rejected is True
    assert ch._ocpp_station_profile_rejection_streak == 2
    station_warnings = [r for r in caplog.records if "station profile rejected" in r.getMessage()]
    assert len(station_warnings) == 1

    # A third rejection whose task was created before the flip (hook called directly while
    # already latched) logs nothing and makes no service call itself (no replay).
    caplog.clear()
    calls.clear()
    with caplog.at_level(logging.DEBUG):
        handled = await ch._on_amp_command_error(marker, 16, _T0)
    assert handled is True
    assert ch._ocpp_station_profile_rejection_streak == 3
    assert caplog.records == []
    assert calls == []


# =============================================================================
# AC3 — fallback amp command after the latch
# =============================================================================


@pytest.mark.asyncio
async def test_ac3_fallback_calls_set_charge_rate_and_clamps():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home, min_charge=6, max_charge=32)
    _init_charger_states(ch)
    ch._ocpp_station_profile_rejected = True
    calls: list = []
    _record_calls(hass, calls)

    # Out-of-range high value -> clamped to max_charge (32).
    res = await ch.low_level_set_max_charging_current(999, _T0)
    assert res is True
    ocpp_calls = [c for c in calls if c[0] == "ocpp"]
    assert len(ocpp_calls) == 1
    assert ocpp_calls[0][1] == "set_charge_rate"
    assert ocpp_calls[0][2] == {"devid": ch.devid, "limit_amps": 32, "conn_id": OCPP_FALLBACK_CONN_ID}
    assert not [c for c in calls if c[0] == number.DOMAIN]

    # Out-of-range low value -> clamped to min_charge (6).
    calls.clear()
    await ch.low_level_set_max_charging_current(1, _T0)
    ocpp_calls = [c for c in calls if c[0] == "ocpp"]
    assert ocpp_calls[0][2]["limit_amps"] == 6


@pytest.mark.asyncio
async def test_ac3_fallback_blocking_caller_gets_real_outcome():
    """SF-1: a blocking caller on the fallback path forwards blocking=True and sees the
    real outcome — True on a non-raising call, False when the service raises."""
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    ch._ocpp_station_profile_rejected = True

    seen_blocking = []

    async def record_blocking(domain, service, data=None, **kwargs):
        seen_blocking.append(kwargs.get("blocking"))

    hass.services.async_call = AsyncMock(side_effect=record_blocking)
    assert await ch.low_level_set_max_charging_current(16, _T0, blocking=True) is True
    assert seen_blocking == [True]

    async def raise_ocpp(domain, service, data=None, **kwargs):
        raise HomeAssistantError("rejected")

    hass.services.async_call = AsyncMock(side_effect=raise_ocpp)
    assert await ch.low_level_set_max_charging_current(16, _T0, blocking=True) is False


# =============================================================================
# AC4 — non-marker error / custom-profile branch
# =============================================================================


@pytest.mark.asyncio
async def test_ac4_non_marker_error_logs_generic_and_does_not_count(caplog):
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    calls: list = []
    _raise_for_number(hass, HomeAssistantError("timeout waiting for response"), calls)

    with caplog.at_level(logging.WARNING):
        res = await ch.low_level_set_max_charging_current(16, _T0, blocking=True)
    assert res is False
    assert ch._ocpp_station_profile_rejection_streak == 0
    assert ch._ocpp_station_profile_rejected is False
    assert any("low_level_set_max_charging_current: Error" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_ac4_custom_profile_never_counts_even_with_marker():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    ch.use_ocpp_custom_charging_profile = True
    marker = HomeAssistantError(f"{OCPP_STATION_PROFILE_REJECTION_MARKER}: Rejected")
    handled = await ch._on_amp_command_error(marker, 16, _T0)
    assert handled is False
    assert ch._ocpp_station_profile_rejection_streak == 0


# =============================================================================
# AC5 — fallback ack and failure
# =============================================================================


@pytest.mark.asyncio
async def test_ac5_fallback_ack_and_failure(caplog):
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    ch._ocpp_station_profile_rejected = True

    # Before any successful fallback write -> None.
    assert ch.get_max_charging_amp_per_phase() is None

    calls: list = []
    _record_calls(hass, calls)
    assert await ch._ocpp_set_charge_rate_fallback(16, _T0) is True
    assert ch.get_max_charging_amp_per_phase() == 16.0

    # A HomeAssistantError -> False, previous ack untouched, warning first then debug.
    async def raise_ocpp(domain, service, data=None, **kwargs):
        raise HomeAssistantError("service not found")

    hass.services.async_call = AsyncMock(side_effect=raise_ocpp)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert await ch._ocpp_set_charge_rate_fallback(20, _T0) is False
    assert ch.get_max_charging_amp_per_phase() == 16.0  # untouched
    assert sum("_ocpp_set_charge_rate_fallback: Error" in r.getMessage() for r in caplog.records) == 1

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert await ch._ocpp_set_charge_rate_fallback(20, _T0) is False
    # second time: warning suppressed, logged at DEBUG only
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "_ocpp_set_charge_rate_fallback: Error" in r.getMessage() and r.levelno == logging.DEBUG
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_ac5_deviceless_fallback_returns_false_without_calling():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    ch._ocpp_station_profile_rejected = True
    ch.devid = None
    calls: list = []
    _record_calls(hass, calls)
    assert await ch._ocpp_set_charge_rate_fallback(16, _T0) is False
    assert calls == []


@pytest.mark.asyncio
async def test_nh4_bad_current_returns_false_without_propagating(caplog):
    """NH-4: a non-numeric `current` in the fallback path returns False, not a raise."""
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    ch._ocpp_station_profile_rejected = True
    calls: list = []
    _record_calls(hass, calls)
    with caplog.at_level(logging.WARNING):
        assert await ch._ocpp_set_charge_rate_fallback(None, _T0) is False
    assert calls == []  # clamp raised before the service call
    assert any("bad current" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_ac5_outside_mode_reads_number_state():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    _init_charger_states(ch)
    hass.states.get.side_effect = lambda eid=None: {
        ch.charger_max_charging_current_number: SimpleNamespace(state="18", attributes={})
    }.get(eid)
    assert ch._ocpp_station_profile_rejected is False
    assert ch.get_max_charging_amp_per_phase() == 18.0


# =============================================================================
# AC6 — optional discovery / binding / device-less
# =============================================================================


def test_ac6_optional_find_returns_none_no_warning(caplog):
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    device = MagicMock()
    device.id = "dev1"
    device.name_by_user = None
    device.name = "Dev"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        res = ch._find_charger_entity_id(device, [], "sensor.", "_x", optional=True)
    assert res is None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_ac6_optional_false_keeps_computed_fallback():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home)
    device = MagicMock()
    device.id = "dev1"
    device.name_by_user = None
    device.name = "Dev"
    res = ch._find_charger_entity_id(device, [], "sensor.", "_x", optional=False)
    assert res == "sensor.dev_x"


def test_ac6_current_offered_bound_and_in_probable_entities():
    hass = _make_hass()
    home = _make_home()
    ch = _create_ocpp_charger(hass, home, extra_entity_ids=("sensor.ocppcharger_current_offered",))
    assert ch.charger_ocpp_current_offered == "sensor.ocppcharger_current_offered"
    assert ch.charger_ocpp_current_offered in ch.get_probable_entities()


def test_ac6_deviceless_construction_has_none_devid_and_offered():
    from custom_components.quiet_solar.const import CONF_IS_3P, CONF_MONO_PHASE
    from custom_components.quiet_solar.ha_model.charger import QSChargerOCPP

    hass = _make_hass()
    home = _make_home()
    config_entry = MagicMock()
    config_entry.entry_id = "test_entry_noocpp"
    config_entry.data = {}
    from unittest.mock import patch

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry"):
        ch = QSChargerOCPP(
            name="NoDeviceOcpp",
            hass=hass,
            home=home,
            config_entry=config_entry,
            **{CONF_IS_3P: False, CONF_MONO_PHASE: 1},
        )
    assert ch.devid is None
    assert ch.charger_ocpp_current_offered is None
    assert ch.charger_ocpp_current_offered not in ch.get_probable_entities()


# =============================================================================
# AC7 / AC7b — clip detector positive + channel isolation
# =============================================================================


@pytest.mark.asyncio
async def test_ac7_clip_detector_positive_both_channels():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(
        ch,
        status=QSOCPPv16v201ChargePointStatus.charging,
        offered="14.0",
        number_state="16",
        charge_enabled=True,
        plugged=True,
    )
    ch.on_device_state_change = AsyncMock()

    # One tick below the window, then exactly at it -> notify on the last tick only.
    for dt in (0, 60, OCPP_CLIP_DETECT_WINDOW_S - 1, OCPP_CLIP_DETECT_WINDOW_S):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))

    assert ch.on_device_state_change.await_count == 1
    assert home.async_notify_all_mobile_apps.await_count == 1
    args = ch.on_device_state_change.await_args
    assert args[0][0] == _T0 + timedelta(seconds=OCPP_CLIP_DETECT_WINDOW_S)
    assert args[0][1] == DEVICE_STATUS_CHANGE_ERROR
    message = args[1]["message"]
    assert ch.name in message
    assert "16 A" in message
    assert "14 A" in message
    assert "power-sharing" in message
    assert "`ocpp.clear_profile`" in message
    assert ch.devid in message
    title, msg = home.async_notify_all_mobile_apps.await_args[0]
    assert title == "Charger current limited — check needed"
    assert msg == message
    assert ch._ocpp_clip_notified is True


@pytest.mark.asyncio
async def test_ac7b_channel_isolation_device_state_raises():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch)
    ch.on_device_state_change = AsyncMock(side_effect=HomeAssistantError("boom"))

    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))

    assert home.async_notify_all_mobile_apps.await_count == 1
    assert ch._ocpp_clip_notified is True


@pytest.mark.asyncio
async def test_ac7b_channel_isolation_mobile_apps_raises():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch)
    ch.on_device_state_change = AsyncMock()
    home.async_notify_all_mobile_apps = AsyncMock(side_effect=HomeAssistantError("boom"))

    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))

    assert ch.on_device_state_change.await_count == 1
    assert ch._ocpp_clip_notified is True


@pytest.mark.asyncio
async def test_nh1_message_floors_offered_never_reads_equal():
    """NH-1: a 15.6 A offer must not round up to 16 A and read as offered == requested."""
    hass, home, ch = _detector_charger()
    ch.on_device_state_change = AsyncMock()
    await ch._notify_ocpp_clip(_T0, expected=16, offered=15.6)
    message = ch.on_device_state_change.await_args[1]["message"]
    assert "offers 15 A while 16 A were requested" in message


@pytest.mark.asyncio
async def test_nh5_message_omits_clear_profile_when_no_devid():
    """NH-5: a device-less charger must not tell the household to run the action for `None`."""
    hass, home, ch = _detector_charger()
    ch.devid = None
    ch.on_device_state_change = AsyncMock()
    await ch._notify_ocpp_clip(_T0, expected=16, offered=14.0)
    message = ch.on_device_state_change.await_args[1]["message"]
    assert "clear_profile" not in message
    assert "None" not in message
    assert home.async_notify_all_mobile_apps.await_count == 1


# =============================================================================
# AC8 — detector negatives (one per guard)
# =============================================================================


@pytest.mark.asyncio
async def test_ac8_current_offered_unbound():
    hass = _make_hass()
    home = _make_home()
    home.async_notify_all_mobile_apps = AsyncMock()
    ch = _create_ocpp_charger(hass, home)  # no extra_entity_ids -> current_offered None
    _init_charger_states(ch)
    ch._expected_amperage.set(16, _T0)
    ch.on_device_state_change = AsyncMock()
    assert ch.charger_ocpp_current_offered is None
    for dt in (0, 60, 180, 300):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0
    assert home.async_notify_all_mobile_apps.await_count == 0


@pytest.mark.asyncio
async def test_ac8_fallback_mode_on():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    ch._ocpp_station_profile_rejected = True
    _seed(ch)
    ch.on_device_state_change = AsyncMock()
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_within_tolerance():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch, offered="15.5")
    ch.on_device_state_change = AsyncMock()
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_suspended_ev_zero_offered():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch, status="SuspendedEV", offered="0")
    ch.on_device_state_change = AsyncMock()
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_charge_disabled():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch, charge_enabled=False)
    ch.on_device_state_change = AsyncMock()
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_expected_at_minimum():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(6, _T0)
    _seed(ch, number_state="6", offered="4.0")
    ch.on_device_state_change = AsyncMock()
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_offered_unavailable():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch, offered="unavailable")
    ch.on_device_state_change = AsyncMock()
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_nan_offered_is_not_a_clip():
    """SF-2: a `NaN` reading held across the window must not trigger a notification."""
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    _seed(ch, offered="nan")
    ch.on_device_state_change = AsyncMock()
    assert ch._ocpp_read_current_offered(_T0) is None
    for dt in (0, 60, OCPP_CLIP_DETECT_WINDOW_S, OCPP_CLIP_DETECT_WINDOW_S + 60):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))
    assert ch.on_device_state_change.await_count == 0


@pytest.mark.asyncio
async def test_ac8_lapse_resets_and_rearms():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    ch.on_device_state_change = AsyncMock()

    # Clipped at t0, t0+60.
    _seed(ch, offered="14.0")
    await ch.check_amps_delivery(_T0)
    await ch.check_amps_delivery(_T0 + timedelta(seconds=60))
    # Offered recovers at t0+120 (reset).
    _seed(ch, offered="16.0")
    await ch.check_amps_delivery(_T0 + timedelta(seconds=120))
    assert ch._ocpp_clip_since is None
    # Clipped again at t0+130, t0+300 (170 s < 180 s -> no notification).
    _seed(ch, offered="14.0")
    await ch.check_amps_delivery(_T0 + timedelta(seconds=130))
    await ch.check_amps_delivery(_T0 + timedelta(seconds=300))
    assert ch.on_device_state_change.await_count == 0
    # t0+310 -> 180 s elapsed -> notification.
    await ch.check_amps_delivery(_T0 + timedelta(seconds=310))
    assert ch.on_device_state_change.await_count == 1


# =============================================================================
# AC9 — ack-mismatch hold
# =============================================================================


@pytest.mark.asyncio
async def test_ac9_ack_mismatch_holds_window():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    ch.on_device_state_change = AsyncMock()

    _seed(ch, offered="14.0", number_state="16")
    await ch.check_amps_delivery(_T0)  # clip_since = t0
    assert ch._ocpp_clip_since == _T0

    # Setpoint change in flight: number reads 12 while expected is 16 -> hold.
    _seed(ch, offered="14.0", number_state="12")
    await ch.check_amps_delivery(_T0 + timedelta(seconds=60))
    await ch.check_amps_delivery(_T0 + timedelta(seconds=120))
    assert ch._ocpp_clip_since == _T0
    assert ch.on_device_state_change.await_count == 0

    # Number acks 16 again at t0+180 -> notification (window was held, not reset).
    _seed(ch, offered="14.0", number_state="16")
    await ch.check_amps_delivery(_T0 + timedelta(seconds=180))
    assert ch.on_device_state_change.await_count == 1


# =============================================================================
# AC10 — no re-notify within a session; unplug resets
# =============================================================================


@pytest.mark.asyncio
async def test_ac10_no_renotify_then_unplug_rearms():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    ch.on_device_state_change = AsyncMock()
    _seed(ch, offered="14.0")

    await ch.check_amps_delivery(_T0)
    await ch.check_amps_delivery(_T0 + timedelta(seconds=180))
    assert ch.on_device_state_change.await_count == 1
    # Continuing clip in the same session: no re-notify.
    await ch.check_amps_delivery(_T0 + timedelta(seconds=240))
    assert ch.on_device_state_change.await_count == 1

    # One unplugged cycle resets both fields.
    _seed(ch, offered="14.0", plugged=False)
    await ch.check_amps_delivery(_T0 + timedelta(seconds=300))
    assert ch._ocpp_clip_since is None
    assert ch._ocpp_clip_notified is False

    # A later clip notifies again.
    _seed(ch, offered="14.0", plugged=True)
    await ch.check_amps_delivery(_T0 + timedelta(seconds=360))
    await ch.check_amps_delivery(_T0 + timedelta(seconds=540))
    assert ch.on_device_state_change.await_count == 2


# =============================================================================
# AC11 — call site
# =============================================================================


@pytest.mark.asyncio
async def test_ac11_check_load_activity_awaits_check_amps_delivery_once():
    hass, home, ch = _detector_charger()
    ch.check_amps_delivery = AsyncMock()
    ch._asked_for_reboot_at_time = None
    ch._boot_time = None
    ch.is_charger_unavailable = MagicMock(return_value=False)
    ch.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    ch.is_charger_faulted = MagicMock(return_value=False)
    ch.is_not_plugged = MagicMock(return_value=False)
    ch.is_plugged = MagicMock(return_value=False)

    await ch.check_load_activity_and_constraints(_T0)
    ch.check_amps_delivery.assert_awaited_once()


@pytest.mark.asyncio
async def test_sf3_check_amps_delivery_raise_does_not_abort_cycle(caplog):
    """SF-3: a raise in the detector override must be swallowed so load management continues."""
    hass, home, ch = _detector_charger()
    ch.check_amps_delivery = AsyncMock(side_effect=RuntimeError("sensor blew up"))
    ch._asked_for_reboot_at_time = None
    ch._boot_time = None
    ch.is_charger_unavailable = MagicMock(return_value=False)
    ch.probe_for_possible_needed_reboot = MagicMock(return_value=False)
    ch.is_charger_faulted = MagicMock(return_value=False)
    ch.is_not_plugged = MagicMock(return_value=False)
    ch.is_plugged = MagicMock(return_value=False)

    with caplog.at_level(logging.ERROR):
        # Must not raise.
        result = await ch.check_load_activity_and_constraints(_T0)
    assert result is False
    assert any("check_amps_delivery raised" in r.getMessage() for r in caplog.records)


# =============================================================================
# AC12 — clear_profile is never called
# =============================================================================


@pytest.mark.asyncio
async def test_ac12_clear_profile_never_called():
    hass, home, ch = _detector_charger()
    ch._expected_amperage.set(16, _T0)
    ch.on_device_state_change = AsyncMock()
    calls: list = []
    _record_calls(hass, calls)

    # Detector positive scenario.
    _seed(ch, offered="14.0")
    for dt in (0, 180):
        await ch.check_amps_delivery(_T0 + timedelta(seconds=dt))

    # Fallback scenario.
    ch._ocpp_station_profile_rejected = True
    await ch.low_level_set_max_charging_current(16, _T0)

    assert not [c for c in calls if c[1] == "clear_profile"]


# =============================================================================
# AC13 — generic hook coverage
# =============================================================================


@pytest.mark.asyncio
async def test_ac13_generic_hook_noop_returns_false(caplog):
    hass = _make_hass()
    home = _make_home()
    ch = _create_charger(hass, home, name="GenericErr")
    _init_charger_states(ch)
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("bang"))

    with caplog.at_level(logging.WARNING):
        res = await ch.low_level_set_max_charging_current(16, _T0)
    assert res is False
    assert any("low_level_set_max_charging_current: Error" in r.getMessage() for r in caplog.records)
    # the base hook returns False (unhandled)
    assert await ch._on_amp_command_error(RuntimeError("x"), 16, _T0) is False


@pytest.mark.asyncio
async def test_ac13_generic_check_amps_delivery_is_noop():
    hass = _make_hass()
    home = _make_home()
    ch = _create_charger(hass, home, name="GenericNoop")
    _init_charger_states(ch)
    assert await ch.check_amps_delivery(_T0) is None
