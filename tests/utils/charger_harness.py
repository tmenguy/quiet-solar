"""Shared charger-test harness: real `QSChargerGeneric` / `QSCar` / `QSChargerGroup`.

Promoted from the module-level helpers in `tests/test_charger_coverage_deep.py`
(QS-306) so more than one test module can drive a real charger. Only Home
Assistant I/O (sensor state reads, service calls, the dynamic group) is mocked;
the charger, car and charger group are real objects.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytz

from custom_components.quiet_solar.const import (
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_IS_INVITED,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_DEVICE_EFFICIENCY,
    CONF_IS_3P,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_MONO_PHASE,
)
from custom_components.quiet_solar.ha_model.car import QSCar
from custom_components.quiet_solar.ha_model.charger import (
    QSChargerGeneric,
    QSChargerGroup,
    QSStateCmd,
)


def make_hass() -> MagicMock:
    """Minimal hass mock: only for HA-level I/O (states, services, bus)."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.config = MagicMock()
    hass.config.config_dir = "/tmp/test"
    hass.bus = MagicMock()
    hass.bus.async_listen = MagicMock(return_value=lambda: None)
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))
    return hass


def make_home(battery=None, voltage=230.0, home_load_power=500.0, max_production_power=3000.0):
    """Create a mock home.  Home has no simple real constructor so we mock it."""
    home = MagicMock()
    home.name = "TestHome"
    home.voltage = voltage
    home.is_3p = True
    home._cars = []
    home._chargers = []
    home._loads = []
    home._persons = []
    home.available_amps_for_group = [[32.0, 32.0, 32.0]]
    home.battery = battery
    home.get_car_by_name = lambda n: next((c for c in home._cars if c.name == n), None)
    home.get_available_power_values = MagicMock(return_value=None)
    home.get_grid_consumption_power_values = MagicMock(return_value=None)
    home.get_best_tariff = MagicMock(return_value=0.15)
    home.get_tariff = MagicMock(return_value=0.20)
    home.battery_can_discharge = MagicMock(return_value=True)
    home.is_off_grid = MagicMock(return_value=False)
    home.dashboard_sections = None
    home.compute_and_set_best_persons_cars_allocations = AsyncMock()
    home.get_preferred_person_for_car = MagicMock(return_value=None)
    home._last_persons_car_allocation = {}
    home.force_next_person_allocation_compute_and_set = MagicMock()

    # Provide realistic power values for budget capping in
    # budgeting_algorithm_minimize_diffs when battery discharge is involved.
    _now = datetime.now(pytz.UTC)
    home.get_device_power_values = MagicMock(
        return_value=[
            (_now - timedelta(seconds=30), home_load_power, {}),
            (_now - timedelta(seconds=15), home_load_power, {}),
            (_now, home_load_power, {}),
        ]
    )
    home.get_home_max_available_production_power = MagicMock(return_value=max_production_power)
    home.get_current_maximum_production_output_power = MagicMock(return_value=max_production_power)
    home.solar_plant = None

    return home


def make_real_car(
    hass,
    home,
    name="TestCar",
    battery_capacity=60000,
    min_charge=6,
    max_charge=32,
    default_charge=80.0,
    minimum_ok_charge=20.0,
    is_invited=False,
    has_soc_sensor=True,
) -> QSCar:
    """Create a REAL QSCar with minimal HA mocking."""
    kwargs = {
        "name": name,
        "hass": hass,
        "home": home,
        "config_entry": None,
        CONF_CAR_BATTERY_CAPACITY: battery_capacity,
        CONF_CAR_CHARGER_MIN_CHARGE: min_charge,
        CONF_CAR_CHARGER_MAX_CHARGE: max_charge,
        CONF_DEFAULT_CAR_CHARGE: default_charge,
        CONF_MINIMUM_OK_CAR_CHARGE: minimum_ok_charge,
        CONF_CAR_IS_INVITED: is_invited,
        CONF_DEVICE_EFFICIENCY: 90.0,
    }
    if has_soc_sensor:
        kwargs[CONF_CAR_CHARGE_PERCENT_SENSOR] = f"sensor.{name.lower().replace(' ', '_')}_soc"
    car = QSCar(**kwargs)
    home._cars.append(car)
    return car


def make_charger_group(home, chargers, max_amps=None) -> QSChargerGroup:
    """Build QSChargerGroup around a mock dynamic-group (the group has no easy real ctor)."""
    from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup

    if max_amps is None:
        max_amps = [32.0, 32.0, 32.0]

    dg = MagicMock(spec=QSDynamicGroup)
    dg.name = "TestGroup"
    dg.home = home
    dg._childrens = chargers
    dg.available_amps_for_group = [max_amps]
    dg.dyn_group_max_phase_current = max(max_amps)
    dg.is_current_acceptable = MagicMock(return_value=True)
    dg.is_current_acceptable_and_diff = MagicMock(return_value=(True, [0.0, 0.0, 0.0]))
    dg.get_median_sensor = MagicMock(return_value=None)
    dg.accurate_power_sensor = "sensor.group_power"
    dg.secondary_power_sensor = None

    group = QSChargerGroup(dg)
    group.charger_consumption_W = 70
    return group


def create_charger(
    hass, home, name="TestCharger", is_3p=False, min_charge=6, max_charge=32, **extra
) -> QSChargerGeneric:
    """Create a REAL QSChargerGeneric."""
    config_entry = MagicMock()
    config_entry.entry_id = f"test_entry_{name}"
    config_entry.data = {}

    config = {
        "name": name,
        "hass": hass,
        "home": home,
        "config_entry": config_entry,
        CONF_CHARGER_MIN_CHARGE: min_charge,
        CONF_CHARGER_MAX_CHARGE: max_charge,
        CONF_IS_3P: is_3p,
        CONF_MONO_PHASE: 1,
        CONF_CHARGER_STATUS_SENSOR: f"sensor.{name}_status",
        CONF_CHARGER_PLUGGED: f"sensor.{name}_plugged",
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: f"number.{name}_max_current",
    }
    config.update(extra)

    with patch("custom_components.quiet_solar.ha_model.charger.entity_registry"):
        charger = QSChargerGeneric(**config)

    home._chargers.append(charger)

    if hasattr(charger, "father_device") and charger.father_device is not None:
        group = make_charger_group(home, [charger])
        charger.father_device.charger_group = group
    return charger


def init_charger_states(charger, charge_state=True, amperage=None, num_phases=1) -> None:
    """Set up inner state objects (the charger resets them to None on reset)."""
    if amperage is None:
        amperage = charger.min_charge
    charger._inner_expected_charge_state = QSStateCmd()
    charger._inner_expected_charge_state.value = charge_state
    charger._inner_amperage = QSStateCmd()
    charger._inner_amperage.value = amperage
    charger._inner_num_active_phases = QSStateCmd()
    charger._inner_num_active_phases.value = num_phases


def plug_car(charger, car, time) -> None:
    """Attach a real car to a real charger (calls update_power_steps)."""
    charger.attach_car(car, time)


def make_battery(asked_charge=0.0) -> MagicMock:
    """Mock battery reporting a fixed asked-charge value."""
    battery = MagicMock()
    battery.get_current_battery_asked_change_for_outside_production_system = MagicMock(return_value=asked_charge)
    return battery
