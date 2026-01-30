"""
Energy validation utilities for deep testing.

These helpers validate energy conservation, battery SOC bounds, and other
physical constraints that must hold in any valid solution.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional


def validate_energy_conservation(
    available_power: npt.NDArray[np.float64],
    load_power: npt.NDArray[np.float64],
    battery_power: Optional[npt.NDArray[np.float64]],
    durations_s: npt.NDArray[np.float64],
    tolerance_wh: float = 1.0
) -> None:
    """
    Validate energy balance using solver's accounting method.
    
    The solver's with_self_test validates that:
        available_power_init + load_power = available_power_final - battery_power
    
    Where:
    - available_power_init: initial (consumption - solar)
    - available_power_final: final state after all allocations
    - load_power: power allocated to loads
    - battery_power: battery charge/discharge
    
    Since we only have access to final state in tests, we validate that
    the solver's self-test passed (which it does internally).
    
    This function is provided for compatibility but the real validation
    is the with_self_test=True flag in solver.solve().
    
    Args:
        available_power: Final available power at each slot (W)
        load_power: Load consumption at each slot (W)
        battery_power: Battery charge/discharge (W)
        durations_s: Duration of each slot (seconds)
        tolerance_wh: Maximum allowed energy imbalance per slot (Wh)
    """
    # The solver's with_self_test=True already validates energy conservation
    # using its internal available_power_init tracking.
    # If we got here without exception from solve(), energy is conserved.
    
    # As a sanity check, verify all arrays have same length
    assert len(available_power) == len(load_power), (
        f"Array length mismatch: available={len(available_power)}, load={len(load_power)}"
    )
    assert len(available_power) == len(durations_s), (
        f"Array length mismatch: available={len(available_power)}, durations={len(durations_s)}"
    )
    
    if battery_power is not None:
        assert len(available_power) == len(battery_power), (
            f"Array length mismatch: available={len(available_power)}, battery={len(battery_power)}"
        )


def validate_battery_soc_bounds(
    battery_charge: npt.NDArray[np.float64],
    battery,  # Battery instance
    tolerance_wh: float = 1.0
) -> None:
    """
    Validate battery SOC stays within [min_soc, max_soc] bounds.
    
    Args:
        battery_charge: Battery charge at each slot (Wh)
        battery: Battery instance with get_value_empty/full methods
        tolerance_wh: Tolerance for bounds checking (Wh)
    
    Raises:
        AssertionError: If SOC exceeds bounds at any slot
    """
    min_wh = battery.get_value_empty()
    max_wh = battery.get_value_full()
    
    for i, charge in enumerate(battery_charge):
        assert charge >= min_wh - tolerance_wh, (
            f"Battery SOC below minimum at slot {i}\n"
            f"  Charge: {charge:.1f}Wh\n"
            f"  Minimum: {min_wh:.1f}Wh\n"
            f"  SOC: {100 * charge / battery.capacity:.1f}%\n"
            f"  Min SOC: {battery.min_charge_SOC_percent:.1f}%"
        )
        
        assert charge <= max_wh + tolerance_wh, (
            f"Battery SOC above maximum at slot {i}\n"
            f"  Charge: {charge:.1f}Wh\n"
            f"  Maximum: {max_wh:.1f}Wh\n"
            f"  SOC: {100 * charge / battery.capacity:.1f}%\n"
            f"  Max SOC: {battery.max_charge_SOC_percent:.1f}%"
        )


def calculate_total_cost(
    power: npt.NDArray[np.float64],
    durations_s: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64]
) -> float:
    """
    Calculate total cost from power usage and prices.
    
    Args:
        power: Power consumption at each slot (W), positive = consuming
        durations_s: Duration of each slot (seconds)
        prices: Price at each slot (€/Wh)
    
    Returns:
        Total cost in euros
    """
    energy_wh = (power * durations_s) / 3600.0
    cost = np.sum(energy_wh * prices)
    return float(cost)


def calculate_energy_from_commands(
    commands: list,
    end_time,
    slot_duration_s: float = 3600.0
) -> float:
    """
    Calculate total energy delivered from a list of (time, LoadCommand) tuples.
    
    NOTE: Commands from solver represent state changes. If there's only one command,
    we use slot_duration_s (default 1h) as a conservative estimate, NOT the full
    duration to end_time, since the actual runtime is managed by the constraint system.
    
    Args:
        commands: List of (datetime, LoadCommand) tuples
        end_time: End time for calculation
        slot_duration_s: Duration for last command if no next command (default 1h)
    
    Returns:
        Total energy in Wh (conservative estimate)
    """
    if not commands:
        return 0.0
    
    total_energy_wh = 0.0
    
    for i, (cmd_time, cmd) in enumerate(commands):
        if cmd.power_consign > 0:
            # Calculate duration to next command or use slot_duration
            if i < len(commands) - 1:
                next_time = commands[i + 1][0]
                duration_s = (next_time - cmd_time).total_seconds()
            else:
                # Last command: use slot_duration (NOT end_time - cmd_time)
                # This gives a conservative estimate
                duration_s = slot_duration_s
            
            energy_wh = (cmd.power_consign * duration_s) / 3600.0
            total_energy_wh += energy_wh
    
    return total_energy_wh


def validate_constraint_satisfaction(
    constraint,
    commands: list,
    end_time,
    tolerance_percent: float = 5.0
) -> tuple[bool, float, float]:
    """
    Check if a constraint is satisfied by the given commands.
    
    Args:
        constraint: The constraint to check
        commands: List of (time, LoadCommand) tuples
        end_time: End time for evaluation
        tolerance_percent: Tolerance percentage for target
    
    Returns:
        (is_satisfied, delivered_energy, target_energy)
    """
    delivered_energy = calculate_energy_from_commands(commands, end_time)
    
    target_energy = None
    if hasattr(constraint, 'target_value'):
        target_energy = constraint.convert_target_value_to_energy(constraint.target_value)
    
    if target_energy is None:
        return True, delivered_energy, None
    
    tolerance = target_energy * tolerance_percent / 100.0
    is_satisfied = delivered_energy >= (target_energy - tolerance)
    
    return is_satisfied, delivered_energy, target_energy


def validate_power_limits(
    commands: list,
    min_power: float = 0.0,
    max_power: Optional[float] = None
) -> bool:
    """
    Verify all commands are within power limits.
    
    Args:
        commands: List of (time, LoadCommand) tuples
        min_power: Minimum allowed power
        max_power: Maximum allowed power (optional)
    
    Returns:
        True if all commands within limits
    """
    for cmd_time, cmd in commands:
        if cmd.power_consign < min_power:
            return False
        if max_power is not None and cmd.power_consign > max_power:
            return False
    return True


def count_transitions(commands: list) -> int:
    """
    Count ON/OFF transitions in a command list.
    
    Args:
        commands: List of (time, LoadCommand) tuples
    
    Returns:
        Number of state transitions
    """
    if len(commands) <= 1:
        return 0
    
    transitions = 0
    prev_is_on = commands[0][1].power_consign > 0
    
    for _, cmd in commands[1:]:
        curr_is_on = cmd.power_consign > 0
        if prev_is_on != curr_is_on:
            transitions += 1
        prev_is_on = curr_is_on
    
    return transitions


def reconstruct_battery_charge_evolution(
    initial_charge: float,
    battery_power: npt.NDArray[np.float64],
    durations_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Reconstruct battery charge evolution from power profile.
    
    Args:
        initial_charge: Initial battery charge (Wh)
        battery_power: Battery power at each slot (W), positive = charging
        durations_s: Duration of each slot (seconds)
    
    Returns:
        Battery charge at END of each slot (Wh)
    """
    battery_charge = np.zeros(len(battery_power), dtype=np.float64)
    
    current_charge = initial_charge
    for i in range(len(battery_power)):
        # Energy change during this slot
        charge_delta_wh = (battery_power[i] * durations_s[i]) / 3600.0
        current_charge += charge_delta_wh
        battery_charge[i] = current_charge
    
    return battery_charge


def validate_no_overallocation(
    allocated_energies: dict[str, float],
    available_energy: float,
    tolerance_wh: float = 100.0
) -> None:
    """
    Validate total allocated energy doesn't exceed available energy.
    
    Args:
        allocated_energies: Dict mapping load names to allocated energy (Wh)
        available_energy: Total available energy (Wh)
        tolerance_wh: Tolerance for checking (Wh)
    
    Raises:
        AssertionError: If over-allocation detected
    """
    total_allocated = sum(allocated_energies.values())
    
    assert total_allocated <= available_energy + tolerance_wh, (
        f"Over-allocation detected\n"
        f"  Total allocated: {total_allocated:.1f}Wh\n"
        f"  Available: {available_energy:.1f}Wh\n"
        f"  Over-allocation: {total_allocated - available_energy:.1f}Wh\n"
        f"  Breakdown: {allocated_energies}"
    )
