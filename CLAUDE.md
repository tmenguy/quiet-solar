# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quiet Solar is a Home Assistant custom component that optimizes solar energy self-consumption. It manages EV charging, battery storage, pool heating, and other controllable loads using a constraint-based solver to minimize grid usage and energy costs.

## Commands

ALWAYS use the venv in the project to run all tests

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_solver.py

# Run a single test
pytest tests/test_solver.py::test_function_name -v

# Run with coverage
pytest tests/ --cov=custom_components/quiet_solar --cov-report=html

# Run only unit or integration tests
pytest tests/ -m unit
pytest tests/ -m integration
```

## Architecture

The codebase has a two-layer architecture:

### Domain Logic Layer (`custom_components/quiet_solar/home_model/`)
Pure Python, no Home Assistant dependency:
- **solver.py** — `PeriodSolver`: constraint-based optimizer that allocates power across loads in 15-minute steps (`SOLVER_STEP_S = 900`). Takes solar forecasts, tariffs, battery state, and load constraints as input.
- **constraints.py** — Constraint definitions (mandatory ASAP, deadline-based, green-only, filler)
- **load.py** — `AbstractLoad`/`AbstractDevice` base classes for all controllable devices
- **commands.py** — Load command enums

### HA Integration Layer (`custom_components/quiet_solar/ha_model/`)
Bridges Home Assistant state/entities with domain logic:
- **home.py** — `QSHome`: main orchestrator. Calls `update_states()` then `solve()` every ~4 seconds via `QSDataHandler`
- **device.py** — `HADeviceMixin`: base class for HA entity lifecycle and state tracking
- **charger.py** — EV charger implementations (OCPP, Wallbox, Generic) with phase switching, power ramping
- **car.py** — EV car models with SOC tracking and charge management
- **battery.py**, **solar.py**, **person.py**, **pool.py**, **heat_pump.py**, **on_off_duration.py** — Other device types

### Data Flow
```
HA State Changes → HADeviceMixin tracks states → QSDataHandler.async_update_all_states()
→ Home.update_states() → Home.solve() via PeriodSolver → Load.execute_command() → HA service calls
```

### Other Key Files
- **config_flow.py** — Configuration UI for all device types (home, battery, solar, charger, car, person, pool, etc.)
- **const.py** — All constants, configuration keys, device type names
- **sensor.py/switch.py/number.py/select.py/button.py** — HA platform implementations

## Testing

- **Framework**: pytest + pytest-asyncio (async mode: auto)
- **conftest.py** provides FakeHass, FakeConfigEntry, FakeState, FakeServices for testing without a real HA instance
- **factories.py** provides factory functions (`create_minimal_home_model()`, etc.) that create real domain objects with proper configuration — used instead of mocks for domain logic tests
- Test files are large; the project targets 100% coverage

## Key Domain Concepts

- **Constraint types**: `MANDATORY_AS_FAST_AS_POSSIBLE`, `MANDATORY_END_TIME`, `BEFORE_BATTERY_GREEN`, `FILLER` — control how loads are prioritized
- **Charger types**: OCPP, Wallbox, Generic — each with different control capabilities (phase switching, current control)
- **Person-car allocation**: People are linked to cars for mileage tracking and charge scheduling
- **Dynamic groups**: Power-limiting groups that cap total power across multiple loads
