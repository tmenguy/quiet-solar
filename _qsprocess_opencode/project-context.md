---
project_name: 'quiet-solar'
user_name: 'Thomas'
date: '2026-03-18'
sections_completed: ['technology_stack', 'architecture', 'python_ha_rules', 'testing', 'code_quality', 'critical_rules']
status: 'complete'
rule_count: 42
optimized_for_llm: true
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Development Lifecycle

The development pipeline uses skill commands that orchestrate phase-specific agents + Python scripts:
- **`/setup-task`** → **`/create-plan`** → **`/implement-story`** → **`/review-story`** → **`/finish-story`** → **`/release`**

Skills are defined in `_qsprocess_opencode/` and `.opencode/agents/`. They delegate creative work to per-task agents and mechanical work to Python scripts in `scripts/qs_opencode/`.

Quality gates: `python scripts/qs/quality_gate.py` (pytest 100% coverage + ruff + mypy + translations). This is the shared quality gate script — not forked for OpenCode.

Doc-sync is built into `/implement-story`, `/review-story`, and `/finish-story` — the agent detects documentation impacts inline and runs a compound sync at lifecycle boundaries.

For workflow routing and architecture constraints, see `_qsprocess_opencode/project-rules.md`.

---

## Technology Stack & Versions

- **Python**: 3.14+
- **Home Assistant**: 2026.2.1+
- **Core dependencies**: haversine 2.9.0, numpy >=1.24.0, scipy >=1.11.0, pytz >=2023.3, aiofiles
- **Test stack**: pytest >=7.0, pytest-asyncio >=0.21 (asyncio_mode=auto), pytest-cov >=4.0, freezegun >=1.4.0, syrupy >=4.6.0, pytest-homeassistant-custom-component >=0.13.0
- **Code quality**: Ruff (formatting + linting), MyPy (type checking)
- **HACS**: Distributed via HACS as custom component

---

## Architecture Rules

### Two-Layer Architecture (STRICT BOUNDARY)

- **Domain logic layer** (`custom_components/quiet_solar/home_model/`): Pure Python. NEVER import from `homeassistant.*`. Contains solver, constraints, load models, commands.
- **HA integration layer** (`custom_components/quiet_solar/ha_model/`): Bridges HA state/entities with domain logic. All HA API calls live here.
- **UI layer** (`custom_components/quiet_solar/ui/`): Dashboard generation and custom JS card registration. Uses `aiofiles` for async file I/O, `yaml` for dashboard config, and HA's lovelace API for programmatic dashboard registration. Agents should not modify JS card code without explicit instruction.
- **Violation of the domain/HA boundary is a critical error.** If domain logic needs HA data, it must be passed as parameters.

### Runtime Data Flow

```text
HA State Changes → HADeviceMixin tracks states → QSDataHandler.async_update_all_states()
→ Home.update_states() → Home.solve() via PeriodSolver → Load.execute_command() → HA service calls
```

QSDataHandler runs three async cycles: state polling (~4s), load management (~7s), and forecast refresh (~30s). The solver runs within the load management cycle but NOT every cycle — it triggers on constraint/state changes (which reset `_last_solve_done`) or after a 5-minute periodic fallback. It plans in 15-minute windows (`SOLVER_STEP_S = 900`). Do not confuse the load management cycle frequency with solver re-evaluation frequency, or with the solver's planning granularity.

### Configuration Flow (Adding a New Device)

```text
User clicks "Add" in HA UI → config_flow.py handles step-by-step UI
→ ConfigEntry created → __init__.py async_setup_entry()
→ Device registered via HADeviceMixin → Platforms (sensor/switch/number/select/button) set up entities
→ QSDataHandler picks up device in next update cycle
```

When adding a new device type, agents must touch:
1. `const.py` — add `CONF_TYPE_NAME_QS*` constant and all config keys
2. `home_model/` — create domain model if needed (pure Python)
3. `ha_model/` — create HA bridge class extending `HADeviceMixin`
4. `config_flow.py` — add configuration UI steps
5. Platform files (`sensor.py`, `switch.py`, etc.) — register entities
6. Tests — full coverage for all new code

### Constants

- ALL configuration keys and constants live in `const.py`. Never hardcode string keys.
- Device type names follow `CONF_TYPE_NAME_QS*` pattern.
- Solver step size: `SOLVER_STEP_S = 900` (15 minutes) — do not change without understanding full impact on all constraint and solver logic.

---

## Python & HA Implementation Rules

### Language Features

- `from __future__ import annotations` at top of every file
- Use newest Python features: pattern matching, type hints, f-strings, dataclasses, walrus operator
- Comprehensive type hints on all functions, methods, and variables
- American English for all code, comments, and documentation (sentence case)

### Async Programming

- All external I/O must be async. Never block the event loop.
- Use `hass.async_add_executor_job()` for blocking operations
- Use `asyncio.sleep()` not `time.sleep()`
- Avoid awaiting in loops — use `gather` instead
- Group executor jobs when possible — switching between event loop and executor is expensive
- Use `@callback` decorator for event-loop-safe functions

### Error Handling

- Choose most specific exception: `ServiceValidationError` for user input, `HomeAssistantError` for device failures
- `ConfigEntryNotReady` — temporary setup issues (e.g., solar forecast API temporarily down, charger offline)
- `ConfigEntryAuthFailed` — authentication problems with external services
- `ConfigEntryError` — permanent setup issues (e.g., invalid configuration)
- Keep try blocks minimal — process data AFTER the try/catch, not inside it
- No bare `except Exception` in regular code. Allowed only in config flows and background tasks.

### Logging

- `_LOGGER = logging.getLogger(__name__)` in every module that logs
- Lazy logging: `_LOGGER.debug("Message with %s", variable)` — no f-strings in log calls
- No periods at end of log messages
- No integration names/domains in messages (added automatically by HA)
- No sensitive data in logs
- Use debug level for non-user-facing messages
- Log unavailability once, log recovery once (don't spam logs)

### Documentation

- Short file header docstrings: `"""Integration for quiet solar."""`
- Docstrings required on all public functions/methods
- Comments explain "why" not "what"
- Use backticks for file paths, filenames, variable names in user-facing messages
- Sentence case for titles and messages

---

## Testing Rules

### Coverage

- **100% test coverage is MANDATORY and NON-NEGOTIABLE.** Every PR must maintain 100% coverage.
- Run: `source venv/bin/activate && pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing`

### Test Infrastructure

**Factories (`tests/factories.py`)** — Use these to create real domain objects, NOT MagicMock:
- `MinimalTestHome` / `MinimalTestLoad` — lightweight test doubles for constraint/solver tests
- `create_constraint()` — create LoadConstraint with configurable type, time, values
- `create_charge_percent_constraint()` — create charge-specific constraints
- `create_load_command()` — create LoadCommand instances
- `create_state_cmd()` — create QSStateCmd for charger testing
- `create_minimal_home_model()` — create full home model with devices for integration tests
- `create_charger_group()` — create charger group with cars
- `create_inner_state()` — create device inner state
- `create_test_car_double()` / `create_test_charger_double()` / `create_test_dynamic_group_double()` — lightweight test doubles

**Mock Configs (`tests/ha_tests/const.py`)** — Standard mock configurations that MUST be used for HA tests:
- `MOCK_HOME_CONFIG` — standard home configuration
- `MOCK_CAR_CONFIG` — standard car configuration
- `MOCK_CHARGER_CONFIG` — standard charger configuration
- `MOCK_SENSOR_STATES` — standard sensor state values

Do NOT invent new mock configs when standard ones exist. Extend them if needed.

**FakeHass Infrastructure (`tests/conftest.py`)** — FakeHass, FakeConfigEntry, FakeState, FakeStates, FakeServices for testing without a real HA instance.

### Test Organization

- `tests/` — domain logic tests and cross-cutting integration tests
- `tests/ha_tests/` — HA integration layer tests with dedicated conftest and const
- `tests/utils/` — shared test utilities (`energy_validation.py`, `scenario_builders.py`)
- Test files: `test_*.py` pattern, test functions: `test_*` pattern

### Test Patterns

- Use `freezegun` for time-dependent tests
- Prefer explicit assertions over snapshot tests. Syrupy snapshots available but use only for complex output comparisons or when explicitly requested.
- Use plain functions and fixtures (pytest style), not test classes
- asyncio_mode=auto — all async test functions auto-detected, no `@pytest.mark.asyncio` needed
- Always use the project venv: `source venv/bin/activate`

---

## Code Quality & Style

### Formatting & Linting

- Ruff for formatting and linting
- MyPy for type checking
- Always fix the underlying issue before adding `# type: ignore` or `noqa` — suppressions are last resort

### Writing Style (user-facing messages)

- Friendly, informative tone
- Second-person ("you" and "your") for user-facing messages
- Use backticks for: file paths, filenames, variable names, field entries
- Sentence case for titles and messages
- Avoid abbreviations, write for non-native English speakers

### Naming Conventions

- Configuration keys: `CONF_*` prefix in `const.py`
- Device type names: `CONF_TYPE_NAME_QS*` pattern
- Sensor constants: `*_SENSOR` suffix
- Dashboard constants: `DASHBOARD_*` prefix

---

## Critical Don't-Miss Rules

### Architecture Violations (WILL BREAK THE PROJECT)

- NEVER import `homeassistant.*` in `home_model/` files — this layer must stay pure Python
- NEVER hardcode configuration keys — always use constants from `const.py`
- NEVER modify `SOLVER_STEP_S` without understanding the full solver impact
- NEVER add a new device type without touching all required files (const, home_model, ha_model, config_flow, platforms, tests)

### HA Anti-Patterns

- NEVER use blocking calls (`requests.get`, `time.sleep`, file I/O) in async code
- NEVER use bare `except Exception` in regular code (only config flows and background tasks)
- NEVER use f-strings in logging calls — use lazy logging with `%s`
- NEVER put data processing inside try blocks — only wrap the call that can throw
- NEVER access `hass.data` directly in tests — use proper fixtures and setup

### Testing Anti-Patterns

- NEVER use MagicMock for domain logic objects when a factory exists in `factories.py`
- NEVER invent new mock configs when standard ones exist in `tests/ha_tests/const.py`
- NEVER skip coverage — 100% is mandatory, non-negotiable
- NEVER submit tests without running the full suite first

### Developer Experience (PRODUCT GOAL, NOT JUST ENGINEERING)

- Developer delight is an explicit product objective — CI/CD automation, expressive test infrastructure, and codebase navigability are features, not hygiene
- When choosing between approaches, prefer the one that makes development more enjoyable (e.g., expressive test names over terse ones, rich test scenarios over minimal stubs)
- Test writing should feel satisfying — if it feels tedious, the test infrastructure needs improvement

### UI/Dashboard Rules

- Dashboard generation is in `ui/dashboard.py` — uses programmatic HA lovelace API
- Custom JS card code should not be modified without explicit instruction
- Dashboard content is preserved across restarts (only resources are updated on startup)

---

## Usage Guidelines

**For AI Agents:**
- Read this file before implementing any code
- Follow ALL rules exactly as documented
- When in doubt, prefer the more restrictive option
- Check `factories.py` and `tests/ha_tests/const.py` before writing test infrastructure

**For Humans:**
- Keep this file lean and focused on agent needs
- Update when technology stack or patterns change
- Remove rules that become obvious over time

Last Updated: 2026-03-18
