"""
Test scenario builders for creating realistic test cases.

These builders create common test scenarios with realistic data patterns.
"""

from datetime import datetime, timedelta
from typing import Optional
import pytz

from custom_components.quiet_solar.home_model.battery import Battery
from custom_components.quiet_solar.home_model.load import TestLoad
from custom_components.quiet_solar.home_model.commands import LoadCommand, CMD_CST_AUTO_CONSIGN


def build_realistic_solar_forecast(
    start_time: datetime,
    num_hours: int,
    peak_power: float = 8000.0,
    sunrise_hour: int = 6,
    sunset_hour: int = 18
) -> list[tuple[datetime, float]]:
    """
    Build realistic solar production forecast with parabolic curve.
    
    Args:
        start_time: Start time for forecast
        num_hours: Number of hours to forecast
        peak_power: Peak solar power at noon (W)
        sunrise_hour: Hour of sunrise (0-23)
        sunset_hour: Hour of sunset (0-23)
    
    Returns:
        List of (datetime, power_W) tuples
    """
    forecast = []
    noon = (sunrise_hour + sunset_hour) / 2
    
    for h in range(num_hours):
        hour_time = start_time + timedelta(hours=h)
        hour = (start_time.hour + h) % 24
        
        if sunrise_hour <= hour <= sunset_hour:
            # Parabolic curve peaking at noon
            hour_from_noon = abs(hour - noon)
            max_deviation = (sunset_hour - sunrise_hour) / 2
            solar_power = peak_power * (1 - (hour_from_noon / max_deviation) ** 2)
        else:
            solar_power = 0.0
        
        forecast.append((hour_time, max(0.0, solar_power)))
    
    return forecast


def build_realistic_consumption_forecast(
    start_time: datetime,
    num_hours: int,
    base_load: float = 500.0,
    peak_evening_load: float = 1500.0,
    night_load: float = 300.0
) -> list[tuple[datetime, float]]:
    """
    Build realistic home consumption forecast.
    
    Args:
        start_time: Start time for forecast
        num_hours: Number of hours to forecast
        base_load: Daytime base load (W)
        peak_evening_load: Evening peak load (W)
        night_load: Night time load (W)
    
    Returns:
        List of (datetime, power_W) tuples
    """
    forecast = []
    
    for h in range(num_hours):
        hour_time = start_time + timedelta(hours=h)
        hour = (start_time.hour + h) % 24
        
        if 7 <= hour <= 17:
            # Daytime
            consumption = base_load
        elif 18 <= hour <= 22:
            # Evening peak
            consumption = peak_evening_load
        else:
            # Night
            consumption = night_load
        
        forecast.append((hour_time, consumption))
    
    return forecast


def build_variable_pricing(
    start_time: datetime,
    num_hours: int,
    cheap_price: float = 0.10 / 1000.0,
    normal_price: float = 0.20 / 1000.0,
    expensive_price: float = 0.40 / 1000.0,
    peak_hours: Optional[list[int]] = None
) -> list[tuple[datetime, float]]:
    """
    Build variable electricity pricing.
    
    Args:
        start_time: Start time for pricing
        num_hours: Number of hours
        cheap_price: Off-peak price (€/Wh)
        normal_price: Normal price (€/Wh)
        expensive_price: Peak price (€/Wh)
        peak_hours: Hours with expensive pricing (default: 18-22)
    
    Returns:
        List of (datetime, price_€/Wh) tuples
    """
    if peak_hours is None:
        peak_hours = list(range(18, 23))
    
    pricing = []
    
    for h in range(num_hours):
        hour_time = start_time + timedelta(hours=h)
        hour = (start_time.hour + h) % 24
        
        if hour in peak_hours:
            price = expensive_price
        elif 2 <= hour <= 6:
            # Night - cheapest
            price = cheap_price
        else:
            price = normal_price
        
        pricing.append((hour_time, price))
    
    return pricing


def create_test_battery(
    capacity_wh: float = 10000.0,
    initial_soc_percent: float = 50.0,
    min_soc_percent: float = 20.0,
    max_soc_percent: float = 90.0,
    max_charge_power: float = 3000.0,
    max_discharge_power: float = 3000.0,
    is_dc_coupled: bool = True,
    name: str = "test_battery"
) -> Battery:
    """
    Create a test battery with typical parameters.
    
    Args:
        capacity_wh: Battery capacity (Wh)
        initial_soc_percent: Initial state of charge (%)
        min_soc_percent: Minimum allowed SOC (%)
        max_soc_percent: Maximum allowed SOC (%)
        max_charge_power: Max charging power (W)
        max_discharge_power: Max discharging power (W)
        name: Battery name
    
    Returns:
        Configured Battery instance
    """
    battery = Battery(name=name)
    battery.capacity = capacity_wh
    battery.max_charging_power = max_charge_power
    battery.max_discharging_power = max_discharge_power
    battery._current_charge_value = capacity_wh * initial_soc_percent / 100.0
    battery.min_charge_SOC_percent = min_soc_percent
    battery.max_charge_SOC_percent = max_soc_percent
    battery.is_dc_coupled = is_dc_coupled
    return battery


def create_car_with_power_steps(
    min_amps: int = 7,
    max_amps: int = 32,
    num_phases: int = 3,
    voltage: float = 230.0,
    name: str = "car"
) -> tuple[TestLoad, list[LoadCommand]]:
    """
    Create a car load with typical power steps.
    
    Args:
        min_amps: Minimum charging current (A)
        max_amps: Maximum charging current (A)
        num_phases: Number of phases (1 or 3)
        voltage: Phase voltage (V)
        name: Load name
    
    Returns:
        (TestLoad, list of LoadCommand power steps)
    """
    car = TestLoad(name=name)
    power_steps = []
    
    for amps in range(min_amps, max_amps + 1):
        power = amps * num_phases * voltage
        power_steps.append(LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=power))
    
    return car, power_steps


def create_simple_heater_load(
    power: float = 2000.0,
    name: str = "test_heater"
) -> TestLoad:
    """
    Create a simple resistive heater load.
    
    Args:
        power: Heater power (W)
        name: Load name
    
    Returns:
        Configured TestLoad
    """
    heater = TestLoad(name=name)
    return heater


def build_alternating_solar_pattern(
    start_time: datetime,
    num_hours: int,
    high_power: float = 5000.0,
    low_power: float = 500.0,
    period_hours: int = 2
) -> list[tuple[datetime, float]]:
    """
    Build alternating solar pattern (for testing on/off cycling).
    
    Args:
        start_time: Start time
        num_hours: Number of hours
        high_power: High solar power (W)
        low_power: Low solar power (W)
        period_hours: Hours per period (high then low)
    
    Returns:
        List of (datetime, power_W) tuples
    """
    forecast = []
    
    for h in range(num_hours):
        hour_time = start_time + timedelta(hours=h)
        cycle_position = (h // period_hours) % 2
        power = high_power if cycle_position == 0 else low_power
        forecast.append((hour_time, power))
    
    return forecast


def calculate_total_available_energy(
    solar_forecast: list[tuple[datetime, float]],
    consumption_forecast: list[tuple[datetime, float]]
) -> float:
    """
    Calculate net available energy from forecasts.
    
    Args:
        solar_forecast: Solar production forecast
        consumption_forecast: Consumption forecast
    
    Returns:
        Net available energy (Wh), can be negative
    """
    total_solar_wh = 0.0
    for i in range(len(solar_forecast) - 1):
        power = solar_forecast[i][1]
        duration_h = (solar_forecast[i+1][0] - solar_forecast[i][0]).total_seconds() / 3600.0
        total_solar_wh += power * duration_h
    
    total_consumption_wh = 0.0
    for i in range(len(consumption_forecast) - 1):
        power = consumption_forecast[i][1]
        duration_h = (consumption_forecast[i+1][0] - consumption_forecast[i][0]).total_seconds() / 3600.0
        total_consumption_wh += power * duration_h
    
    return total_solar_wh - total_consumption_wh
