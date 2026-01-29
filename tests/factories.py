"""Factory functions for creating real quiet-solar objects in tests.

This module provides factory functions that create real instances of quiet-solar
objects (QSCar, QSChargerGeneric, constraints, etc.) for use in tests. These
factories should be preferred over SimpleNamespace or MagicMock when testing
actual behavior.

For HA-related tests, use these factories with the real `hass` fixture from
pytest_homeassistant_custom_component rather than FakeHass.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytz

from homeassistant.const import CONF_NAME

from custom_components.quiet_solar.const import (
    CONF_CAR_BATTERY_CAPACITY,
    CONF_CAR_CHARGE_PERCENT_SENSOR,
    CONF_CAR_CHARGER_MAX_CHARGE,
    CONF_CAR_CHARGER_MIN_CHARGE,
    CONF_CAR_PLUGGED,
    CONF_CAR_TRACKER,
    CONF_CHARGER_MAX_CHARGE,
    CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER,
    CONF_CHARGER_MIN_CHARGE,
    CONF_CHARGER_PAUSE_RESUME_SWITCH,
    CONF_CHARGER_PLUGGED,
    CONF_CHARGER_STATUS_SENSOR,
    CONF_DEFAULT_CAR_CHARGE,
    CONF_DYN_GROUP_MAX_PHASE_AMPS,
    CONF_GRID_POWER_SENSOR,
    CONF_HOME_VOLTAGE,
    CONF_IS_3P,
    CONF_MINIMUM_OK_CAR_CHARGE,
    CONF_MONO_PHASE,
    CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    CONSTRAINT_TYPE_FILLER,
    CONSTRAINT_TYPE_FILLER_AUTO,
    CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
    CONSTRAINT_TYPE_MANDATORY_END_TIME,
    DATA_HANDLER,
    DOMAIN,
)
from custom_components.quiet_solar.home_model.commands import (
    CMD_AUTO_GREEN_ONLY,
    CMD_ON,
    LoadCommand,
    copy_command,
)
from custom_components.quiet_solar.home_model.constraints import (
    DATETIME_MAX_UTC,
    LoadConstraint,
    MultiStepsPowerLoadConstraint,
    MultiStepsPowerLoadConstraintChargePercent,
)
from custom_components.quiet_solar.home_model.load import AbstractDevice, AbstractLoad

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

    from custom_components.quiet_solar.ha_model.car import QSCar
    from custom_components.quiet_solar.ha_model.charger import (
        QSChargerGeneric,
        QSChargerGroup,
        QSStateCmd,
    )
    from custom_components.quiet_solar.ha_model.home import QSHome


# =============================================================================
# Minimal Load for Constraint Testing
# =============================================================================


class MinimalTestHome:
    """Minimal home model for testing that provides voltage."""

    def __init__(self, voltage: float = 230.0, is_3p: bool = True):
        self.voltage = voltage
        self.is_3p = is_3p
        self._cars = []
        self._chargers = []


class MinimalTestLoad(AbstractLoad):
    """Minimal load implementation for testing constraints and solver.

    This class provides just enough implementation to test constraint logic
    without requiring a full HA setup.
    """

    def __init__(
        self,
        name: str = "TestLoad",
        power: float = 1000.0,
        voltage: float = 230.0,
        efficiency: float = 100.0,
        **kwargs,
    ):
        """Initialize minimal test load.

        Args:
            name: Load name
            power: Power consumption in watts
            voltage: Voltage (default 230V)
            efficiency: Efficiency percentage (default 100%)
        """
        # Create a minimal home to provide voltage if not provided
        home = kwargs.pop("home", None)
        if home is None:
            home = MinimalTestHome(voltage=voltage)

        super().__init__(
            name=name,
            device_type="test_load",
            home=home,
            **kwargs,
        )
        self._power = power
        self._efficiency = efficiency
        self._constraints: list[LoadConstraint] = []
        self.current_command: LoadCommand | None = None

    @property
    def power_use(self) -> float:
        return self._power

    @property
    def efficiency_factor(self) -> float:
        return self._efficiency / 100.0

    def get_update_value_callback_for_constraint_class(self, constraint):
        """Return callback for constraint value updates."""
        return None

    def is_off_grid(self) -> bool:
        """Check if load is off-grid."""
        return False


# =============================================================================
# Constraint Factories
# =============================================================================


def create_constraint(
    load: AbstractLoad | None = None,
    constraint_type: int = CONSTRAINT_TYPE_FILLER_AUTO,
    time: datetime | None = None,
    start_of_constraint: datetime | None = None,
    end_of_constraint: datetime | None = None,
    initial_value: float = 0.0,
    target_value: float = 100.0,
    power: float = 1000.0,
    **kwargs,
) -> MultiStepsPowerLoadConstraint:
    """Create a real MultiStepsPowerLoadConstraint for testing.

    Args:
        load: The load this constraint applies to (creates MinimalTestLoad if None)
        constraint_type: Type of constraint (CONSTRAINT_TYPE_*)
        time: Current time for constraint creation
        start_of_constraint: When constraint starts
        end_of_constraint: When constraint must be met
        initial_value: Starting value (Wh for energy constraints)
        target_value: Target value to reach
        power: Power in watts for the constraint command
        **kwargs: Additional constraint parameters

    Returns:
        A real MultiStepsPowerLoadConstraint instance
    """
    if time is None:
        time = datetime.now(tz=pytz.UTC)

    if load is None:
        load = MinimalTestLoad(name="TestLoad", power=power)

    return MultiStepsPowerLoadConstraint(
        time=time,
        load=load,
        type=constraint_type,
        start_of_constraint=start_of_constraint,
        end_of_constraint=end_of_constraint,
        initial_value=initial_value,
        target_value=target_value,
        power=power,
        **kwargs,
    )


def create_charge_percent_constraint(
    load: AbstractLoad | None = None,
    constraint_type: int = CONSTRAINT_TYPE_MANDATORY_END_TIME,
    time: datetime | None = None,
    end_of_constraint: datetime | None = None,
    initial_value: float = 50.0,
    target_value: float = 80.0,
    total_capacity_wh: float = 60000.0,
    power: float = 7000.0,
    **kwargs,
) -> MultiStepsPowerLoadConstraintChargePercent:
    """Create a real charge percent constraint for car charging tests.

    Args:
        load: The load this constraint applies to
        constraint_type: Type of constraint
        time: Current time
        end_of_constraint: When charging must complete
        initial_value: Starting charge percent
        target_value: Target charge percent
        total_capacity_wh: Battery capacity in Wh
        power: Charging power in watts
        **kwargs: Additional parameters

    Returns:
        A real MultiStepsPowerLoadConstraintChargePercent instance
    """
    if time is None:
        time = datetime.now(tz=pytz.UTC)

    if end_of_constraint is None:
        end_of_constraint = time + timedelta(hours=8)

    if load is None:
        load = MinimalTestLoad(name="TestCar", power=power)

    return MultiStepsPowerLoadConstraintChargePercent(
        time=time,
        load=load,
        type=constraint_type,
        end_of_constraint=end_of_constraint,
        initial_value=initial_value,
        target_value=target_value,
        total_capacity_wh=total_capacity_wh,
        power=power,
        **kwargs,
    )


# =============================================================================
# Command Factories
# =============================================================================


def create_load_command(
    command: int = CMD_ON,
    power_consign: float = 1000.0,
    **kwargs,
) -> LoadCommand:
    """Create a real LoadCommand for testing.

    Args:
        command: Command type (CMD_ON, CMD_AUTO_GREEN_ONLY, etc.)
        power_consign: Power consign in watts
        **kwargs: Additional command parameters

    Returns:
        A real LoadCommand instance
    """
    return copy_command(
        LoadCommand(command=command, power_consign=power_consign),
        **kwargs,
    )


# =============================================================================
# QSStateCmd Factory
# =============================================================================


def create_state_cmd(
    initial_value: Any = None,
    is_ok_to_set: bool = True,
    command_retries_s: float = 42.0,
) -> "QSStateCmd":
    """Create a real QSStateCmd for testing charger state management.

    Args:
        initial_value: Initial value for the state
        is_ok_to_set: Whether setting is allowed
        command_retries_s: Retry interval in seconds

    Returns:
        A real QSStateCmd instance
    """
    from custom_components.quiet_solar.ha_model.charger import QSStateCmd

    state_cmd = QSStateCmd(command_retries_s=command_retries_s)
    if initial_value is not None:
        state_cmd.value = initial_value
    return state_cmd


# =============================================================================
# Home Model Factories (for pure logic tests, no HA required)
# =============================================================================


def create_minimal_home_model(
    name: str = "TestHome",
    voltage: float = 230.0,
    is_3p: bool = True,
    max_phase_amps: float = 32.0,
) -> MagicMock:
    """Create a minimal home model mock for pure logic tests.

    This is for tests that don't need full HA integration but need
    a home object for device relationships.

    Args:
        name: Home name
        voltage: Voltage (default 230V)
        is_3p: Three-phase configuration
        max_phase_amps: Max amps per phase

    Returns:
        A MagicMock with home-like attributes
    """
    home = MagicMock()
    home.name = name
    home.voltage = voltage
    home.is_3p = is_3p
    home._cars = []
    home._chargers = []
    home._loads = []
    home.available_amps_for_group = [[max_phase_amps] * 3]

    def get_car_by_name(car_name: str):
        for car in home._cars:
            if car.name == car_name:
                return car
        return None

    home.get_car_by_name = get_car_by_name
    return home


# =============================================================================
# HA Device Factories (for tests using real HA fixtures)
# =============================================================================


def get_default_car_config(
    name: str = "Test Car",
    **overrides,
) -> dict[str, Any]:
    """Get default configuration for a QSCar.

    Args:
        name: Car name
        **overrides: Override any default values

    Returns:
        Configuration dict suitable for QSCar initialization
    """
    config = {
        CONF_NAME: name,
        CONF_CAR_TRACKER: f"device_tracker.{name.lower().replace(' ', '_')}",
        CONF_CAR_PLUGGED: f"binary_sensor.{name.lower().replace(' ', '_')}_plugged",
        CONF_CAR_CHARGE_PERCENT_SENSOR: f"sensor.{name.lower().replace(' ', '_')}_soc",
        CONF_CAR_BATTERY_CAPACITY: 60000,  # 60 kWh
        CONF_CAR_CHARGER_MIN_CHARGE: 6,
        CONF_CAR_CHARGER_MAX_CHARGE: 32,
        CONF_DEFAULT_CAR_CHARGE: 80.0,
        CONF_MINIMUM_OK_CAR_CHARGE: 20.0,
    }
    config.update(overrides)
    return config


def get_default_charger_config(
    name: str = "Test Charger",
    **overrides,
) -> dict[str, Any]:
    """Get default configuration for a QSChargerGeneric.

    Args:
        name: Charger name
        **overrides: Override any default values

    Returns:
        Configuration dict suitable for QSChargerGeneric initialization
    """
    slug = name.lower().replace(" ", "_")
    config = {
        CONF_NAME: name,
        CONF_CHARGER_MIN_CHARGE: 6,
        CONF_CHARGER_MAX_CHARGE: 32,
        CONF_IS_3P: False,
        CONF_MONO_PHASE: 1,
        CONF_CHARGER_MAX_CHARGING_CURRENT_NUMBER: f"number.{slug}_max_current",
        CONF_CHARGER_PAUSE_RESUME_SWITCH: f"switch.{slug}_pause_resume",
        CONF_CHARGER_STATUS_SENSOR: f"sensor.{slug}_status",
        CONF_CHARGER_PLUGGED: f"binary_sensor.{slug}_plugged",
    }
    config.update(overrides)
    return config


def get_default_home_config(
    name: str = "Test Home",
    **overrides,
) -> dict[str, Any]:
    """Get default configuration for a QSHome.

    Args:
        name: Home name
        **overrides: Override any default values

    Returns:
        Configuration dict suitable for QSHome initialization
    """
    config = {
        CONF_NAME: name,
        CONF_HOME_VOLTAGE: 230,
        CONF_IS_3P: True,
        CONF_GRID_POWER_SENSOR: "sensor.grid_power",
        CONF_DYN_GROUP_MAX_PHASE_AMPS: 63,
    }
    config.update(overrides)
    return config


# =============================================================================
# Async Device Factories (for use with real hass fixture)
# =============================================================================


async def create_real_car(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry",
    home: "QSHome",
    name: str = "Test Car",
    **config_overrides,
) -> "QSCar":
    """Create a real QSCar instance for HA integration tests.

    This factory creates a real QSCar using the actual class, suitable for
    tests that use the pytest_homeassistant_custom_component fixtures.

    Args:
        hass: Real HomeAssistant instance from fixture
        config_entry: MockConfigEntry for the car
        home: QSHome instance (the car's parent)
        name: Car name
        **config_overrides: Override default config values

    Returns:
        A real QSCar instance
    """
    from custom_components.quiet_solar.ha_model.car import QSCar

    config = get_default_car_config(name=name, **config_overrides)

    car = QSCar(
        hass=hass,
        config_entry=config_entry,
        home=home,
        **config,
    )

    # Register with home
    if hasattr(home, "_cars"):
        home._cars.append(car)

    return car


async def create_real_charger(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry",
    home: "QSHome",
    name: str = "Test Charger",
    **config_overrides,
) -> "QSChargerGeneric":
    """Create a real QSChargerGeneric instance for HA integration tests.

    Args:
        hass: Real HomeAssistant instance from fixture
        config_entry: MockConfigEntry for the charger
        home: QSHome instance
        name: Charger name
        **config_overrides: Override default config values

    Returns:
        A real QSChargerGeneric instance
    """
    from custom_components.quiet_solar.ha_model.charger import QSChargerGeneric

    config = get_default_charger_config(name=name, **config_overrides)

    charger = QSChargerGeneric(
        hass=hass,
        config_entry=config_entry,
        home=home,
        **config,
    )

    # Register with home
    if hasattr(home, "_chargers"):
        home._chargers.append(charger)

    return charger


# =============================================================================
# Charger Group Factory
# =============================================================================


def create_charger_group(
    home: Any,
    chargers: list[Any] | None = None,
    name: str = "Test Charger Group",
    max_amps: list[float] | None = None,
) -> "QSChargerGroup":
    """Create a real QSChargerGroup for testing group charging logic.

    Args:
        home: Home object (real or mock)
        chargers: List of chargers in the group
        name: Group name
        max_amps: Max amps per phase [phase1, phase2, phase3]

    Returns:
        A real QSChargerGroup instance
    """
    from custom_components.quiet_solar.ha_model.charger import QSChargerGroup
    from custom_components.quiet_solar.ha_model.dynamic_group import QSDynamicGroup

    if chargers is None:
        chargers = []

    if max_amps is None:
        max_amps = [32.0, 32.0, 32.0]

    # Create a dynamic group that the charger group wraps
    dyn_group = MagicMock(spec=QSDynamicGroup)
    dyn_group.name = name
    dyn_group.home = home
    dyn_group._childrens = chargers
    dyn_group.available_amps_for_group = [max_amps]

    return QSChargerGroup(dyn_group)


# =============================================================================
# Test Scenario Helpers
# =============================================================================


def setup_car_charging_scenario(
    car: Any,
    charger: Any,
    current_soc: float = 50.0,
    target_soc: float = 80.0,
    is_plugged: bool = True,
) -> dict[str, Any]:
    """Set up a car charging scenario for testing.

    Args:
        car: QSCar or mock car object
        charger: QSChargerGeneric or mock charger
        current_soc: Current state of charge (%)
        target_soc: Target state of charge (%)
        is_plugged: Whether car is plugged in

    Returns:
        Dict with scenario details for assertions
    """
    # Link car to charger
    car.charger = charger
    charger.car = car

    # Set up state
    if hasattr(car, "_current_charge_percent"):
        car._current_charge_percent = current_soc

    return {
        "car": car,
        "charger": charger,
        "current_soc": current_soc,
        "target_soc": target_soc,
        "is_plugged": is_plugged,
        "energy_needed_wh": (target_soc - current_soc) / 100.0 * car.car_battery_capacity
        if hasattr(car, "car_battery_capacity")
        else 0,
    }


# =============================================================================
# Charger Internal State Helpers
# =============================================================================


def create_inner_state(
    value: Any = None,
    is_ok_to_set_return: bool = True,
) -> "QSStateCmd":
    """Create a QSStateCmd for charger internal state testing.

    This replaces patterns like:
        SimpleNamespace(value=True, is_ok_to_set=MagicMock(return_value=True))

    Args:
        value: Initial value for the state
        is_ok_to_set_return: What is_ok_to_set() should return

    Returns:
        A real QSStateCmd instance with the value set
    """
    from custom_components.quiet_solar.ha_model.charger import QSStateCmd

    state = QSStateCmd()
    state.value = value
    # QSStateCmd.is_ok_to_set() takes time and min_change_time args
    # The real method will work correctly for most tests
    return state


def setup_charger_inner_states(
    charger: Any,
    charge_state_value: bool = True,
    amperage_value: int | None = None,
    num_phases_value: int = 3,
) -> None:
    """Set up charger internal state objects for testing.

    This replaces patterns like:
        charger._inner_expected_charge_state = SimpleNamespace(value=True, ...)
        charger._inner_amperage = SimpleNamespace(value=6)
        charger._inner_num_active_phases = SimpleNamespace(value=3, ...)

    Args:
        charger: The charger device to configure
        charge_state_value: Value for _inner_expected_charge_state
        amperage_value: Value for _inner_amperage (None = charger.min_charge)
        num_phases_value: Value for _inner_num_active_phases
    """
    from custom_components.quiet_solar.ha_model.charger import QSStateCmd

    if amperage_value is None:
        amperage_value = getattr(charger, "min_charge", 6)

    charger._inner_expected_charge_state = QSStateCmd()
    charger._inner_expected_charge_state.value = charge_state_value

    charger._inner_amperage = QSStateCmd()
    charger._inner_amperage.value = amperage_value

    charger._inner_num_active_phases = QSStateCmd()
    charger._inner_num_active_phases.value = num_phases_value


# =============================================================================
# Test Double Helpers (for unit tests that don't need full HA)
# =============================================================================


class TestCarDouble:
    """Test double for QSCar that provides minimal required interface.

    Use this instead of SimpleNamespace for car mocks in unit tests.
    """

    def __init__(
        self,
        name: str = "Test Car",
        car_charger_min_charge: int = 6,
        car_charger_max_charge: int = 32,
        car_battery_capacity: int = 60000,
        qs_bump_solar_charge_priority: bool = False,
        **kwargs,
    ):
        self.name = name
        self.car_charger_min_charge = car_charger_min_charge
        self.car_charger_max_charge = car_charger_max_charge
        self.car_battery_capacity = car_battery_capacity
        self.qs_bump_solar_charge_priority = qs_bump_solar_charge_priority
        self.charger = None
        self._current_charge_percent = kwargs.get("current_soc", 50.0)

        # Apply any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_charge_power_per_phase_A(
        self,
        target_amps: int,
        num_phases: int,
    ) -> tuple[list[float], int, int]:
        """Default implementation returning theoretical power."""
        voltage = 230.0
        if num_phases == 1:
            power = [0.0, 0.0, 0.0]
            power[0] = target_amps * voltage
        else:
            power = [target_amps * voltage] * 3
        return power, self.car_charger_min_charge, self.car_charger_max_charge


class TestChargerDouble:
    """Test double for QSChargerGeneric that provides minimal required interface.

    Use this instead of SimpleNamespace for charger mocks in unit tests.
    """

    def __init__(
        self,
        name: str = "Test Charger",
        mono_phase_index: int = 0,
        min_charge: int = 6,
        max_charge: int = 32,
        car: Any = None,
        **kwargs,
    ):
        from custom_components.quiet_solar.ha_model.charger import QSStateCmd

        self.name = name
        self.mono_phase_index = mono_phase_index
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.car = car
        self.charger_default_idle_charge = min_charge
        self.num_on_off = 0
        self.qs_enable_device = True

        # State commands
        self._expected_charge_state = QSStateCmd()
        self._expected_amperage = QSStateCmd()
        self._expected_num_active_phases = QSStateCmd()

        # Apply any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_amps_with_delta(
        self,
        current_amps: list[float],
        delta_amps: float,
        num_phases: int,
    ) -> list[float]:
        """Default implementation."""
        result = list(current_amps)
        if num_phases == 1:
            result[self.mono_phase_index] += delta_amps
        else:
            for i in range(3):
                result[i] += delta_amps
        return result

    def get_delta_dampened_power(
        self,
        from_amp: float,
        from_phases: int,
        to_amp: float,
        to_phases: int,
    ) -> float:
        """Default implementation returning theoretical power difference."""
        voltage = 230.0
        from_power = from_amp * voltage * from_phases
        to_power = to_amp * voltage * to_phases
        return to_power - from_power

    def _get_amps_from_power_steps(
        self,
        power: float,
        num_phases: int,
    ) -> int | None:
        """Default implementation."""
        voltage = 230.0
        amps = int(power / (voltage * num_phases))
        if amps < self.min_charge:
            return None
        return min(amps, self.max_charge)

    def can_do_3_to_1_phase_switch(self) -> bool:
        """Default implementation."""
        return True


class TestDynamicGroupDouble:
    """Test double for QSDynamicGroup that provides minimal required interface."""

    def __init__(
        self,
        name: str = "Test Group",
        home: Any = None,
        chargers: list | None = None,
        max_amps: list[float] | None = None,
        **kwargs,
    ):
        self.name = name
        self.home = home or MinimalTestHome()
        self._childrens = chargers or []
        self.available_amps_for_group = [max_amps or [32.0, 32.0, 32.0]]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def is_current_acceptable(self, amps: list[float]) -> bool:
        """Check if current is within limits."""
        max_amps = self.available_amps_for_group[0]
        for i, amp in enumerate(amps):
            if amp > max_amps[i]:
                return False
        return True


def create_test_car_double(**kwargs) -> TestCarDouble:
    """Create a TestCarDouble for unit testing.

    Example:
        car = create_test_car_double(name="My Car", car_battery_capacity=75000)
    """
    return TestCarDouble(**kwargs)


def create_test_charger_double(car: Any = None, **kwargs) -> TestChargerDouble:
    """Create a TestChargerDouble for unit testing.

    Example:
        car = create_test_car_double()
        charger = create_test_charger_double(car=car, name="My Charger")
    """
    return TestChargerDouble(car=car, **kwargs)


def create_test_dynamic_group_double(**kwargs) -> TestDynamicGroupDouble:
    """Create a TestDynamicGroupDouble for unit testing.

    Example:
        group = create_test_dynamic_group_double(name="Charger Group")
    """
    return TestDynamicGroupDouble(**kwargs)


# =============================================================================
# Constraint Type Constants (re-exported for convenience)
# =============================================================================

CONSTRAINT_TYPES = {
    "filler_auto": CONSTRAINT_TYPE_FILLER_AUTO,
    "filler": CONSTRAINT_TYPE_FILLER,
    "before_battery": CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN,
    "mandatory_end_time": CONSTRAINT_TYPE_MANDATORY_END_TIME,
    "mandatory_asap": CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE,
}
