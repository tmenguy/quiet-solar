"""Misc tests for quiet_solar home helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant

from custom_components.quiet_solar.const import DATA_HANDLER, DOMAIN
from custom_components.quiet_solar.ha_model.home import QSHomeMode
from custom_components.quiet_solar.home_model.commands import CMD_IDLE, LoadCommand


async def _setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    if entry.state is not ConfigEntryState.NOT_LOADED:
        return
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_home_force_update_and_resets(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test force update and forecast reset helpers."""
    await _setup_entry(hass, home_config_entry)

    home = hass.data[DOMAIN][DATA_HANDLER].home

    home.update_all_states = AsyncMock()
    home.update_loads = AsyncMock()
    time_now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    await home.force_update_all(time_now)

    home.update_all_states.assert_awaited_with(time_now)
    home.update_loads.assert_awaited_with(time_now)

    home._consumption_forecast = MagicMock()
    home._consumption_forecast.reset_forecasts = AsyncMock()

    await home.reset_forecasts(time_now)
    home._consumption_forecast.reset_forecasts.assert_awaited_with(time_now)

    await home.light_reset_forecasts(time_now)
    home._consumption_forecast.reset_forecasts.assert_awaited_with(
        time_now, light_reset=True
    )


@pytest.mark.asyncio
async def test_home_reset_forecasts_defaults(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test reset helpers default to current time."""
    await _setup_entry(hass, home_config_entry)

    home = hass.data[DOMAIN][DATA_HANDLER].home
    home._consumption_forecast = MagicMock()
    home._consumption_forecast.reset_forecasts = AsyncMock()

    await home.reset_forecasts()
    await home.light_reset_forecasts()

    assert home._consumption_forecast.reset_forecasts.await_count == 2


@pytest.mark.asyncio
async def test_home_dump_for_debug_and_dashboard_yaml(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    tmp_path,
) -> None:
    """Test dump_for_debug and dashboard YAML generation."""
    await _setup_entry(hass, home_config_entry)

    home = hass.data[DOMAIN][DATA_HANDLER].home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home._consumption_forecast = MagicMock()
    home._consumption_forecast.dump_for_debug = AsyncMock()

    home.physical_solar_plant = MagicMock()
    home.physical_solar_plant.dump_for_debug = AsyncMock()

    home.dump_person_car_data_for_debug = AsyncMock()

    async def run_job(func, *args):
        return func(*args)

    home.hass.async_add_executor_job = AsyncMock(side_effect=run_job)

    with patch.object(home.hass.config, "path", return_value=str(tmp_path)):
        await home.dump_for_debug()

    debug_file = tmp_path / DOMAIN / "debug" / "debug_conf.pickle"
    assert debug_file.exists()
    home._consumption_forecast.dump_for_debug.assert_awaited()
    home.physical_solar_plant.dump_for_debug.assert_awaited()
    home.dump_person_car_data_for_debug.assert_awaited()

    with patch(
        "custom_components.quiet_solar.ha_model.home.generate_dashboard_yaml",
        new=AsyncMock(),
    ) as mock_generate:
        await home.generate_yaml_for_dashboard()
    mock_generate.assert_awaited_with(home)


@pytest.mark.asyncio
async def test_home_piloted_devices_and_removals(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
    battery_config_entry: ConfigEntry,
    solar_config_entry: ConfigEntry,
    car_config_entry: ConfigEntry,
    charger_config_entry: ConfigEntry,
    person_config_entry: ConfigEntry,
    heat_pump_config_entry: ConfigEntry,
) -> None:
    """Test piloted devices and remove_device branches."""
    await _setup_entry(hass, home_config_entry)
    await _setup_entry(hass, battery_config_entry)
    await _setup_entry(hass, solar_config_entry)
    await _setup_entry(hass, car_config_entry)
    await _setup_entry(hass, charger_config_entry)
    await _setup_entry(hass, person_config_entry)
    await _setup_entry(hass, heat_pump_config_entry)

    home = hass.data[DOMAIN][DATA_HANDLER].home
    heat_pump = hass.data[DOMAIN].get(heat_pump_config_entry.entry_id)

    home.prepare_slots_for_piloted_device_budget(3)
    assert heat_pump.num_demanding_clients == [0, 0, 0]

    battery = hass.data[DOMAIN].get(battery_config_entry.entry_id)
    solar = hass.data[DOMAIN].get(solar_config_entry.entry_id)
    car = hass.data[DOMAIN].get(car_config_entry.entry_id)
    charger = hass.data[DOMAIN].get(charger_config_entry.entry_id)
    person = hass.data[DOMAIN].get(person_config_entry.entry_id)

    home.remove_device(car)
    home.remove_device(charger)
    home.remove_device(person)
    home.remove_device(battery)
    home.remove_device(solar)
    home.remove_device(heat_pump)

    home.add_disabled_device(car)
    home.remove_disabled_device(car)
    home.remove_disabled_device(car)


@pytest.mark.asyncio
async def test_home_power_helpers_without_sources(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test power helpers with no solar or battery."""
    await _setup_entry(hass, home_config_entry)

    home = hass.data[DOMAIN][DATA_HANDLER].home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.physical_solar_plant = None
    home.physical_battery = None

    assert home.get_current_over_clamp_production_power() == 0.0
    assert home.get_current_maximum_production_output_power() == 0.0
    assert home.battery_can_discharge() is False
    assert home.get_battery_charge_values(3600, datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)) == []


@pytest.mark.asyncio
async def test_home_update_loads_forbid_command(
    hass: HomeAssistant,
    home_config_entry: ConfigEntry,
) -> None:
    """Test update_loads forbid branch for delta current."""
    await _setup_entry(hass, home_config_entry)

    home = hass.data[DOMAIN][DATA_HANDLER].home
    home.home_mode = QSHomeMode.HOME_MODE_ON.value
    home.finish_setup = AsyncMock(return_value=True)
    home.update_loads_constraints = AsyncMock()
    home._last_solve_done = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    load = MagicMock()
    load.name = "forbid_load"
    load.check_commands = AsyncMock(return_value=timedelta(seconds=0))
    load.running_command_num_relaunch = 0
    load.force_relaunch_command = AsyncMock()
    load.is_load_active = MagicMock(return_value=True)
    load.update_live_constraints = AsyncMock(return_value=False)
    load.get_phase_amps_from_power_for_budgeting = MagicMock(return_value=[1, 1, 1])
    load.launch_command = AsyncMock()
    load.is_load_has_a_command_now_or_coming = MagicMock(return_value=True)
    load.get_current_active_constraint = MagicMock(return_value=True)
    load.do_probe_state_change = AsyncMock()
    load.current_command = CMD_IDLE
    load.father_device = SimpleNamespace(
        is_delta_current_acceptable=MagicMock(return_value=False)
    )

    command = LoadCommand(command="on", power_consign=100.0)
    time_now = datetime(2026, 1, 15, 9, 0, tzinfo=pytz.UTC)

    home._all_loads = [load]
    home._chargers = []
    home.physical_battery = None
    home._commands = [(load, [(time_now, command)])]
    home._battery_commands = []

    await home.update_loads(time_now)

    load.launch_command.assert_not_called()
