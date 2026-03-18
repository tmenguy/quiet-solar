---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: 'complete'
completedAt: '2026-03-18'
inputDocuments:
  - product-brief-quiet-solar-2026-03-17.md
  - project-context.md
workflowType: 'architecture'
project_name: 'quiet-solar'
user_name: 'Thomas'
date: '2026-03-18'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
- Whole-house energy orchestration across 8+ device types (solar, battery, EV chargers, cars, persons, pool, heat pump, on/off duration loads, dynamic groups)
- Constraint-based solver with four priority tiers: MANDATORY_AS_FAST_AS_POSSIBLE, MANDATORY_END_TIME, BEFORE_BATTERY_GREEN, FILLER (+ FILLER_AUTO)
- People-aware automation: trip prediction based on GPS/mileage history, person-car allocation, presence-based scheduling
- Smart device handling: detects and adapts to external control (devices operated outside quiet-solar)
- Off-grid resilience: automatic load shedding during grid outages
- Tariff-aware scheduling: peak/off-peak optimization, architecture ready for dynamic tariffs
- UI layer: programmatic Jinja2-based dashboard generation with 4 custom JS Lovelace cards
- Notification and override flows for household members via HA mobile app
- Multi-protocol charger support: OCPP, Wallbox, Generic — each with different control capabilities
- Real-time charger dynamic budgeting: balances power across multiple chargers respecting per-phase circuit limits

**Non-Functional Requirements:**
- 100% test coverage — mandatory, non-negotiable
- Solver re-evaluation: event-driven (triggered by constraint/state changes) with 5-minute periodic fallback. Load management cycle runs every ~7 seconds; solver runs within it only when triggered or when 5 minutes have elapsed since last solve. State polling every ~4 seconds, forecast refresh every ~30 seconds
- Solver plans in 15-minute windows (SOLVER_STEP_S = 900)
- Pure Python domain layer — zero HA dependencies in home_model/
- Async-first: all external I/O async, no event loop blocking
- Genericity: works for any home with solar and controllable loads
- HACS distribution: self-contained custom component, no external server dependencies
- Rate-limited command execution: max 1 command per load per cycle

**Scale & Complexity:**
- Primary domain: IoT/home automation with embedded constrained optimization
- Complexity level: High
- Architectural layers: 3 (domain model, HA integration, UI)
- Device type implementations: 8+ with shared base classes
- Charger protocol variants: 3 (OCPP, Wallbox, Generic)
- Core engines: constraint-based solver (strategic) + charger dynamic budgeting (tactical)

### Conceptual Component Model

#### Domain Layer (`home_model/`)

**LoadCommand** (`commands.py`)
- Represents a discrete action at a specific power level
- 10 command types ordered by priority score (CMD_ON=100 down to CMD_OFF=10)
- Commands merge by taking the higher-scored type and max power
- Command scores determine which command wins when two constraints overlap on the same load. Higher score = more aggressive intervention. This is the semantic meaning of the command hierarchy — an AI agent implementing constraint logic must understand that score ordering drives conflict resolution.
- Key distinction: `is_auto()` (green/price-aware modes) vs forced modes vs `is_off_or_idle()`

**LoadConstraint** (`constraints.py`)
- Time-windowed demand on a load with progress tracking
- Five priority tiers determine solver allocation order:
  - MANDATORY_AS_FAST_AS_POSSIBLE (9) — highest priority, minimize time
  - MANDATORY_END_TIME (7) — must complete by deadline
  - BEFORE_BATTERY_GREEN (5) — before battery goes solar-only
  - FILLER (3) — uses surplus energy, non-mandatory
  - FILLER_AUTO (1) — lowest priority, auto-only
- Tracks: initial_value -> current_value -> target_value with percent completion
- Carries metadata (`load_info` dict) for origin tracking (user override, agenda, prediction)
- Subclasses: MultiStepsPowerLoadConstraint (power-based with step options), ChargePercent variant (SOC-based), TimeBasedSimplePower (duration-based)

**AbstractDevice** (`load.py`)
- Base for all controllable devices. Manages configuration, command lifecycle, and state tracking
- Command lifecycle: pending -> launched (running_command) -> acked (current_command), with stacking for busy devices
- Switching cost protection: `num_max_on_off` daily budget, hysteresis enforcement (CHANGE_ON_OFF_STATE_HYSTERESIS_S = 10 min minimum between changes), multi-pass adaptation that tries free transitions first
- 3-phase aware: tracks phase configuration, converts power to per-phase amps for budgeting
- **AbstractLoad behavioral contract**: AbstractLoad defines a behavioral contract that all device types must honor. Device-specific tests should validate deviations from that contract, not re-test the contract itself. This is a contract testing pattern — test the guarantees once at the base level, then test device-specific behavior in isolation.

**PilotedDevice** (`load.py`)
- Extends AbstractDevice for devices that pilot other devices (e.g., heat pump with auxiliary heater)
- Tracks client list and per-slot demand counts

**AbstractLoad** (`load.py`)
- Extends AbstractDevice. Adds constraint management and solver interface
- Provides `get_for_solver_constraints()` — the solver's entry point to read active constraints
- Supports green-only mode, user override state, external control detection
- Tracks constraint progress for UI display

**Battery** (`battery.py`)
- Charge/discharge model with SOC bounds, power limits, DC-coupled awareness
- Calculates safe charge/discharge power respecting inverter limits and capacity boundaries

**PeriodSolver** (`solver.py`)
- The strategic optimization engine. Creates 15-minute time slots aligned with constraint boundaries and tariff changes
- **Optimization hierarchy** (strict priority ordering): maximize solar self-consumption, then minimize energy cost, then maintain comfort commitments. This hierarchy governs all solver trade-off decisions.
- Allocation algorithm:
  1. Create time slots and power slots from PV forecast + unavoidable consumption
  2. Allocate mandatory constraints (highest priority first, by score)
  3. Optimize battery charge/discharge to minimize grid imports
  4. Allocate filler constraints with remaining surplus
  5. Return command timeline: list of (load, [(time, LoadCommand)]) tuples
- Constraint scoring determines allocation order within each tier
- Re-evaluation trigger: event-driven (constraint/state changes reset `_last_solve_done`) with 5-minute periodic fallback

#### HA Integration Layer (`ha_model/`)

**HADeviceMixin** (`device.py`)
- The bridge pattern. Every HA device inherits from this + a domain class
- State tracking: polls HA entities into time-series history (`_entity_probed_state`), auto-trims to 3 days
- Power measurement: attaches power sensors with unit conversion, computes energy via Riemann sum
- Entity probing: `attach_ha_state_to_probe()` with numerical conversion, custom getters, transform functions
- Maintains reverse index of HA entities by description key (`ha_entities` dict) for dashboard template queries

**QSHome** (`home.py`)
- The orchestrator. Extends QSDynamicGroup (which extends HADeviceMixin + AbstractDevice)
- Root of the dynamic group tree — enforces phase-by-phase amperage budgets
- Three async cycles driven by QSDataHandler:
  - State polling (~4s): updates all device states from HA, refreshes forecasts
  - Load management (~7s): updates constraints, checks command ACKs, triggers solver, launches commands
  - Forecast refresh (~30s): solar forecast API polling
- Command launch: rate-limited to max 1 per load per cycle, validates amp budget before each launch

**QSDataHandler** (`data_handler.py`)
- HA config entry lifecycle manager. Creates QSHome on first entry, queues subsequent device entries
- Sets up `async_track_time_interval` for the three timing cycles
- Uses `asyncio.Lock` to prevent re-entrance on all cycles

**Charger Dynamic Budgeting** (`charger.py`, `dynamic_group.py`) — TRUST-CRITICAL COMPONENT
- This is where quiet-solar's value proposition is most visible to users. When Magali plugs in her car and the system seamlessly redistributes power — that's the "magic moment." When it fails and a breaker trips at 2am — that's a household incident. Changes to this component require extra scrutiny because failures are physically visible and immediate.
- See dedicated section: "Hierarchical Control Architecture" below

**Device Implementations:**
- QSChargerGeneric / QSChargerOCPP / QSChargerWallbox (`charger.py`): EV charging with power ramping, phase switching, multi-protocol adaptation. OCPP adds transaction handling; Wallbox maps vendor status enums
- QSCar (`car.py`): SOC tracking, charger assignment, person allocation, custom power-to-amperage lookup tables per car
- QSBattery (`battery.py`): Charge/discharge control via HA number and switch entities
- QSSolar (`solar.py`): Production tracking + forecast provider integration (Solcast, OpenMeteo)
- QSPerson (`person.py`): Presence tracking, GPS-based trip detection, historical mileage prediction (31-day window)
- QSPool (`pool.py`): Temperature-dependent filter duration, extends QSOnOffDuration
- QSOnOffDuration (`on_off_duration.py`): Simple binary switch loads via HA switch service calls
- QSHeatPump (`heat_pump.py`): Wraps PilotedDevice for multi-client control
- QSDynamicGroup (`dynamic_group.py`): Tree topology for hierarchical amperage budgeting

#### UI & Configuration Layer

**Config Flow** (`config_flow.py`):
- Hierarchical menu -> device-type steps -> common schema builder pattern
- `get_common_schema()` provides reusable field composition (power, 3-phase, calendar, groups, max on/off)
- Entity selectors filter by unit of measurement, exclude quiet-solar's own entities

**Platform Files** (sensor.py, switch.py, number.py, select.py, button.py):
- Factory pattern: type-specific creator functions -> polymorphic dispatcher -> platform setup
- Custom EntityDescription dataclasses with callable value_fn, option_fn, set_fn for dynamic behavior

**Dashboard Generation** (`ui/dashboard.py`):
- Jinja2 template iterates devices, accesses `device.ha_entities.get(key)` for entity IDs
- 4 custom JS Lovelace cards (car, pool, climate, on-off-duration)
- Programmatic registration via HA's lovelace API, version-tagged for cache busting

### Hierarchical Control Architecture

The system operates as a **two-level hierarchical control architecture**:

**Strategic Layer: PeriodSolver**
- Plans in 15-minute discrete windows
- Produces optimal command timelines: "charger A should get 7kW for the next 2 hours"
- Optimizes across all loads simultaneously
- Re-evaluates when triggered by constraint/state changes, or every 5 minutes as fallback (runs within the ~7s load management cycle)

**Tactical Layer: Charger Dynamic Budgeting**
- Operates in continuous real-time with 45-second adaptation windows
- Manages the physical realities of power distribution within circuit constraints
- **Can override the strategic layer.** If the solver says "charge at 7kW" but the amp budget only allows 5kW, the tactical layer wins
- Has its own independent state machine (adaptation windows, staged transitions, dampening measurements) that persists between solver runs

**Why this distinction matters for AI agents:**
- Modifying the solver modifies only the strategic half. The tactical layer may constrain or override solver intentions.
- A solver bug produces a bad plan. A budgeting bug can trip a breaker. The blast radius is physical, not just computational.
- An agent implementing a "charging improvement" must understand both layers and their interaction.

#### Charger Dynamic Budgeting: Detailed Architecture

**QSDynamicGroup Tree** (`dynamic_group.py`):
- Tree topology where QSHome is root, dynamic groups are interior nodes, chargers are leaves
- Phase-aware budget: always represented as `[phase1, phase2, phase3]` arrays
- Per-phase operations: `add_amps()`, `diff_amps()`, `is_amps_greater()`, `min_amps()`, `max_amps()`
- Budget validation propagates recursively up the tree (child -> parent -> root)
- Two validation methods:
  - `is_delta_current_acceptable()`: planning-phase check — "will this change fit?"
  - `is_current_acceptable_and_diff()`: execution-phase check — takes worst-case max of actual + estimated consumption

**Charger Group State Machine** (`charger.py`):
- QSChargerGroup aggregates chargers on the same circuit
- QSChargerStatus tracks per-charger state (amps, phases, real power, adaptation state)
- Each charger has a `charge_score` — higher priority chargers get power first when budget conflicts arise

**Budgeting Algorithm** (`budgeting_algorithm_minimize_diffs`):
1. Priority check: if highest-priority charger isn't charging but a lower one is, trigger reset allocation
2. Prepare budgets: either keep current amps (minimize transitions) or reset to minimum (rebalance)
3. Shave mandatory: if minimum amps still exceed group limit, stop lowest-score chargers first
4. Shave current: try phase switching (1P->3P) for lower-score chargers before reducing amps
5. Smart allocation loop: iteratively adjust each charger's budget toward power target while respecting all constraints

**Phase Switching** (3P <-> 1P):
- Tactical power redistribution: 1P@32A -> 3P@11A (reduces per-phase load)
- Phase switch attempted before amp reduction (less disruptive)
- May require charger reboot (`do_reboot_on_phase_switch`)

**Staged Transitions** (`apply_budget_strategy`):
- Large budget changes split into two phases to prevent transient overages:
  - Phase 1: Reduce decreasing chargers first (frees up amps)
  - Phase 2: Increase other chargers in next cycle (already validated safe)
- `remaining_budget_to_apply` persists between cycles to complete Phase 2

**Adaptation Windows**:
- CHARGER_ADAPTATION_WINDOW_S = 45 seconds — chargers must be stable before rebalancing
- CHARGER_STATE_REFRESH_INTERVAL_S = 14 seconds — state polling frequency
- Dampening: real power measurements (not just targets) drive future estimates

### Key Architectural Patterns

**1. Hysteresis-as-Stabilizer**
The solver re-evaluates when triggered by constraint or state changes (with a 5-minute periodic fallback via `_last_solve_done`), but switching budgets (`num_max_on_off` daily limit) and hysteresis (10-minute minimum between state changes) prevent plan instability. This is an explicit architectural principle: event-driven re-evaluation for responsiveness, budget-limited actuation for stability.

**2. Switching Cost Protection** (reusable pattern)
Daily on/off budget + hysteresis + multi-pass constraint adaptation (try free transitions first, then spend budget). All on/off device types (pool, heat pump, on/off duration) inherit this pattern. New device types should use it.

**3. AbstractLoad Behavioral Contract**
AbstractLoad defines guarantees (constraint management, command lifecycle, solver interface). Device-specific tests validate deviations from the contract. This is a contract testing pattern.

**4. Bridge Pattern (HADeviceMixin)**
Every HA device inherits from HADeviceMixin + a domain class. All HA state flows through `add_to_history()`. All commands flow through `execute_command()` -> HA service calls. This is the single integration seam.

**5. Recursive Budget Validation**
Amp budget checks propagate up the QSDynamicGroup tree. No charger can exceed its parent group's limit, which in turn respects the home's total circuit capacity.

### Key Data Flows

**Runtime Solve Cycle:**
```
HA State Changes -> HADeviceMixin.add_to_history() [~4s polling]
-> QSHome.update_all_states() -> device.update_states() for all devices
-> QSHome.update_loads() [~7s]:
  -> update_loads_constraints() [live constraint updates]
  -> check_loads_commands() [ACK validation, relaunch if stuck]
  -> IF solve needed: PeriodSolver.solve()
    -> Returns (load, [(time, LoadCommand)]) timeline
  -> Launch commands [max 1/load/cycle, amp budget checked]
    -> device.execute_command() -> HA service calls
```

**Charger Dynamic Budgeting Cycle** (within update_loads):
```
All chargers stable for 45s? -> dampening update (measure real power)
-> Detect transitions (single charger change -> record power delta)
-> Budget reset opportunity? (20 min timeout or high-priority charger waiting)
-> budgeting_algorithm_minimize_diffs():
  -> Priority check -> prepare budgets -> shave mandatory -> shave current
  -> Smart allocation loop (iterative, respects phase limits)
-> apply_budget_strategy():
  -> IF large change: Phase 1 (reduce first) -> Phase 2 (increase next cycle)
  -> ELSE: apply directly
```

**Device Configuration Flow:**
```
HA UI -> config_flow.py step -> ConfigEntry created
-> __init__.py async_setup_entry() -> QSDataHandler.async_add_entry()
-> create_device_from_type() -> device constructor with **config_entry.data
-> home.add_device() -> device.get_platforms()
-> HA platform setup -> create_ha_<platform>(device) -> entities registered
```

### User Interaction Architecture

The system serves three user personas through distinct architectural paths:

**TheAdmin's Path (Home Energy Manager):**
```
Config flow -> device setup -> entity creation
-> Dashboard (Jinja2 template -> custom JS cards) -> monitoring sensors
-> Constraint tuning (number/select entities) -> solver re-plan
-> Troubleshooting: constraint progress sensors, command history, forecast accuracy
```

**Magali's Path (Household Member):**
```
Person model -> trip prediction (GPS/mileage history)
-> Constraint creation (load_info: {originator: "prediction"})
-> Solver allocates power -> charger executes
-> Notification (via HA mobile app)
-> Override needed? -> user_override state -> constraint update (load_info: {originator: "user_override"})
-> Solver re-plans -> charger budgeting rebalances -> confirmation notification
```

**TheDev's Path (Developer):**
```
Code change -> pytest (100% coverage gate) -> Ruff + MyPy
-> CI pipeline (PR quality gate -> merge gate -> release pipeline)
-> Test infrastructure: factories (domain) + FakeHass (lightweight) + real HA fixtures (integration)
-> Failure mode catalog -> scenario-based tests -> confident deployment
```

**Architectural implication**: An agent implementing a notification or override story must trace through the full path: person prediction -> constraint metadata -> solver -> charger budgeting -> notification service. These are not isolated components. An agent implementing developer workflow improvements must understand the dual test infrastructure (FakeHass vs real HA fixtures) and the risk-weighted CI strategy.

### Technical Constraints & Dependencies

- **Home Assistant platform**: All HA API access confined to ha_model/ layer. Platform dictates entity lifecycle, config flow patterns, service call mechanisms
- **Solar forecast dependency**: External API (Solcast/OpenMeteo) for production forecasts — must handle unavailability gracefully
- **Device protocol diversity**: OCPP, Wallbox, Generic charger protocols each with different control capabilities and state reporting
- **Python 3.14+**: Uses latest language features (pattern matching, type hints, dataclasses)
- **Real-time state tracking**: HADeviceMixin bridges HA entity state changes into domain model — must be reliable and low-latency
- **Rate-limited actuation**: Max 1 command per load per solve cycle, charger adaptation window of 45s
- **Physical circuit constraints**: Amp budgets are not just optimization targets — exceeding them trips breakers. The dynamic group tree enforces hard physical limits.

### Cross-Cutting Concerns

- **Domain/HA boundary enforcement**: Every feature must respect the strict two-layer separation. Domain logic receives data as parameters, never imports HA
- **Device lifecycle management**: Adding a new device type touches 6+ files (const, home_model, ha_model, config_flow, platforms, tests)
- **Async discipline**: Event loop safety across all layers — @callback decorators, executor jobs for blocking ops, gather over await-in-loops
- **State consistency**: Three async cycles (4s/7s/30s) share device state via asyncio.Lock guards. Solver re-evaluates frequently but hysteresis and switching budgets prevent plan instability. Charger budgeting has its own 45-second adaptation windows that interleave with the solver cycles.
- **Constraint propagation**: Constraints from multiple sources (predictions, calendar, manual overrides) must compose correctly. Metadata tracking (load_info dict) preserves origin for conflict resolution
- **Hierarchical amperage budgeting**: QSDynamicGroup tree (rooted at QSHome) enforces per-phase current limits during command acceptance. Budget checks are recursive and phase-aware.
- **Switching cost protection**: num_max_on_off daily budget + hysteresis + multi-pass constraint adaptation is a reusable pattern all on/off device types inherit
- **Testing infrastructure**: Domain factories for real objects (not mocks), FakeHass for HA layer — both must stay in sync with production code

### Decision Map for AI Agents

| If you're implementing... | You need to understand... | Key files |
|---|---|---|
| A new device type | AbstractLoad behavioral contract, config flow pattern, platform factories, const.py, 6-file checklist in project-context.md | load.py, config_flow.py, const.py, sensor.py/switch.py/etc |
| A constraint change | LoadConstraint hierarchy, solver allocation order, switching budget interaction, command score semantics | constraints.py, solver.py, load.py |
| A notification/override | Person model -> constraint metadata (load_info) -> QSHome command flow -> notification. Trace the full Magali path. | person.py, home.py, constraints.py, charger.py |
| A solver improvement | PeriodSolver algorithm, constraint scoring, battery optimization. Remember: solver is strategic only — tactical charger budgeting may override. | solver.py, constraints.py, battery.py |
| A charger budgeting change | TRUST-CRITICAL. Dynamic group tree, QSChargerGroup state machine, phase switching protocol, staged transitions, adaptation windows. Failures trip breakers. | charger.py, dynamic_group.py, home.py |
| A dashboard change | Jinja2 template, ha_entities dict lookup, JS card API, dashboard section organization | ui/dashboard.py, device.py, const.py |
| A config flow change | Common schema builder, entity selectors, data cleaning, config entry -> device creation flow | config_flow.py, const.py, data_handler.py |

### Architectural Gaps (Prioritized by Risk)

**1. Charger Dynamic Budgeting Testing Gaps — HIGH RISK**
The charger budgeting system is the most complex real-time component and the hardest to test correctly. Specific gaps:
- **Staged transition recovery**: If the system crashes between Phase 1 (reduce) and Phase 2 (increase), chargers may be stuck in reduced state. Is `remaining_budget_to_apply` recoverable?
- **Race conditions between cycles**: State polling at 4s and charger budgeting at 45s adaptation windows may conflict — a state poll mid-adaptation could trigger a solver re-plan that conflicts with in-progress rebalancing.
- **Phase switching under load**: Switching 1P->3P briefly frees amps. Could another charger claim them before the switch completes? Is there a reservation mechanism?
- **Priority inversion**: High-score charger triggers reset, stopping a low-priority charger that has a MANDATORY_END_TIME constraint. Which system wins — budget priority or constraint priority?
- **Dampening accuracy over charging curve**: Non-linear EV charging curves (slowdown above 80% SOC) may make early dampening data misleading for later estimates.
- **Needed**: Scenario-based multi-charger integration tests that simulate rebalancing sequences over time and verify no phase is exceeded at any intermediate state.

**2. Constraint Interaction Testing — HIGH RISK**
Boundary between constraint types and physical limits needs explicit test coverage:
- MANDATORY_END_TIME vs exhausted num_max_on_off switching budget — which wins?
- Mandatory constraint vs exhausted charger amp budget — does the constraint system override budget priority?
- Multiple MANDATORY constraints competing for insufficient power — how does the solver prioritize?

**3. Failure Mode / Resilience Architecture — HIGH RISK (partially mitigated)**
Off-grid mode exists, but no broader taxonomy for degraded operation:
- Solar forecast API down for 6 hours — what's the fallback strategy?
- Charger stops responding mid-charge — how long before detection and what's the recovery?
- Solver produces infeasible plan — is there a safe default?
- Each external dependency should have a documented failure signature and response.

**4. State Consistency Model Documentation — LOW RISK, HIGH VALUE**
The mechanism works, but an AI agent that doesn't understand hysteresis-as-stabilizer might "fix" it by removing the delay. Document explicitly: "The solver may re-evaluate frequently, but hysteresis and switching budgets prevent plan instability. Do not remove adaptation delays without understanding the full impact."

**5. Solver Quality Benchmarking — MEDIUM RISK (future)**
Tests verify correctness (plan satisfies constraints) but not optimality (plan maximizes self-consumption vs. naive strategy). Quality benchmarks would catch solver regressions. Becomes critical when refactoring the solver.

**6. Data/Learning Architecture — FUTURE SCOPE**
Person mileage history (31 days) exists. No explicit data persistence or analysis layer for consumption patterns, forecast accuracy tracking, or prediction confidence scoring. The product brief marks this as out of scope for now. Document as a known future architectural need.

## Technology Stack & CI/CD Architecture

### Technology Stack (Established — Brownfield)

No starter template evaluation needed — the project is mature with a working production deployment. The following decisions are already made and locked.

**Dependency Chain** (each layer constrains the next):
```
Python 3.14+ -> Home Assistant Core 2026.2.1+ -> HACS -> quiet-solar
```
CI must validate the full chain, not just quiet-solar in isolation. Pin to HA's bundled dependency versions, not just declared minimums.

**Language & Runtime:**
- Python 3.14+ with `from __future__ import annotations` in every file
- Home Assistant 2026.2.1+ as the host platform
- HACS as the distribution mechanism

**Core Dependencies:**
- numpy >= 1.24.0, scipy >= 1.11.0 (solver numerics) — WARNING: HA bundles specific numpy versions. If HA ships numpy 2.x and solver uses deprecated 1.x APIs, runtime failures may not be caught by tests using different versions. CI must pin to HA's bundled versions.
- haversine 2.9.0 (GPS distance calculations for person/car tracking)
- pytz >= 2023.3 (timezone handling)
- aiofiles (async file I/O for dashboard generation)

**Testing Framework:**
- pytest >= 7.0 with pytest-asyncio >= 0.21 (asyncio_mode=auto)
- pytest-cov >= 4.0 (100% coverage enforcement)
- freezegun >= 1.4.0 (time-dependent test control)
- syrupy >= 4.6.0 (snapshot testing, used sparingly)
- pytest-homeassistant-custom-component >= 0.13.0 (real HA fixtures)
- Factory-based test doubles (not mocks) for domain objects
- **Dual test infrastructure**: FakeHass (lightweight, fast) for domain/unit tests AND real HA fixtures (pytest-homeassistant-custom-component) for integration tests. Document which tests use which — mixing them creates confusion about what's being tested.

**Code Quality:**
- Ruff for formatting and linting
- MyPy for type checking
- Suppressions (type: ignore, noqa) are last resort — fix the underlying issue

**UI Stack (KNOWN GAP: outside quality pipeline):**
- 4 custom JS Lovelace cards (car, pool, climate, on-off-duration, ~150KB total)
- Jinja2 templates for programmatic dashboard generation
- HA's lovelace API for dashboard registration
- No JS linting, no JS tests, no JS build step. The product brief says "don't modify JS without explicit instruction." This is a known gap — the JS cards are untested artifacts that exist outside the quality pipeline.

**Existing Pytest Markers:**
- `@pytest.mark.unit` — pure domain logic tests
- `@pytest.mark.integration` — HA integration tests
- `@pytest.mark.slow` — long-running tests
- `asyncio_mode=auto` — all async tests auto-detected

### Existing Test Suite Inventory

**~3,800+ tests** across 100+ files. Two-layer organization:

| Area | Test files | ~Tests | What's tested | Gaps |
|---|---|---|---|---|
| **Solver** | 5 | ~400 | Power allocation, energy conservation, algorithm correctness | No optimality benchmarks (correctness only, not "is this better than naive?") |
| **Constraints** | 5 | ~300 | All constraint types, transitions, edge cases | Missing: constraint type interactions under resource exhaustion (MANDATORY vs exhausted switching budget) |
| **Chargers** | 13 | ~700 | OCPP/Wallbox/Generic variants, phase switching, current control | Missing: multi-charger rebalancing sequence tests with intermediate state verification; staged transition recovery; race condition between adaptation windows and solver cycles |
| **Cars** | 9 | ~400 | SOC tracking, range estimation, dampening, person allocation | Dampening accuracy over non-linear charging curves |
| **Persons** | 8 | ~250 | Mileage history, car allocation, forecasting | Person-car allocation override notification flow |
| **Battery** | 3 | ~150 | Charge/discharge logic, clamping, SOC bounds | Adequate for current scope |
| **Solar** | 2 | ~100 | Inverter integration, power tracking | Forecast API failure fallback behavior |
| **Home/Integration** | 15 | ~400 | Orchestration, state updates, data flow, device relationships | Off-grid mode edge cases |
| **HA Platforms** | 12 | ~300 | Entity creation, state management for all platform types | Adequate |
| **HA Devices** | 12 | ~250 | Device-specific HA integration, registry, services | Adequate |
| **Config Flow** | 3 | ~100 | Configuration UI, validation, device addition | Adequate |
| **Utilities/Other** | 15 | ~150 | Helpers, forecasts, commands, calendar, UI dashboard | Test infrastructure smoke tests (verify FakeHass behaves like real HA) |

**Test infrastructure concern**: FakeHass, factories, and conftest fixtures are all custom. If someone modifies `conftest.py` or `factories.py` and breaks the test infrastructure itself, every test might still pass while testing the wrong thing. A smoke test for the test infrastructure should verify FakeHass behaves like real HA for critical operations.

### Risk-Weighted CI Strategy

Not every file deserves the same CI scrutiny. Changes to different areas carry different risk levels:

| Change area | Risk level | Blast radius | CI response |
|---|---|---|---|
| `solver.py`, `constraints.py` | CRITICAL | Every user's electricity bill and comfort | Full test suite + solver quality benchmarks (when they exist). Future: solver regression benchmarks. |
| `charger.py`, `dynamic_group.py` | CRITICAL (physical) | Can trip breakers. Household incident. | Full test suite + charger scenario integration tests. Future: multi-charger rebalancing sequence tests. |
| `load.py` (AbstractDevice/AbstractLoad) | HIGH | Base contract affects all device types | Full test suite (contract changes propagate everywhere) |
| `const.py` | HIGH (widespread) | Constants touch every file | Full test suite |
| `home.py`, `data_handler.py` | HIGH | Orchestration affects all devices | Full test suite |
| `car.py`, `person.py`, `battery.py` | MEDIUM | Device-specific, contained blast radius | Targeted test suite (device + integration tests) |
| `config_flow.py` | MEDIUM | Setup experience only, no runtime impact | Standard test suite |
| `sensor.py`, `switch.py`, `number.py`, `select.py`, `button.py` | LOW | Entity display only | Standard test suite |
| `ui/dashboard.py`, `ui/*.js` | LOW | Dashboard visual only | Standard test suite (JS untested) |

As the test suite grows, support targeted execution via existing pytest markers (`unit`, `integration`, `slow`) and path-based triggers. Run charger integration tests only when charger code changes. Run everything when `const.py` or the solver changes.

### CI/CD Pipeline Architecture (GitHub Actions)

**Current state**: No CI/CD exists. This is the first infrastructure story per the product brief.

#### Tier 1 — PR Quality Gate (every push to PR branch)

**Workflow**: `.github/workflows/pr-quality.yml`
- **Trigger**: `on: pull_request` (branches: main)
- **Jobs**:
  - `lint`: Ruff check + Ruff format check (fail on violations)
  - `typecheck`: MyPy type checking (fail on errors)
  - `test`: `pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing` with coverage threshold = 100% (fail if below)
  - `hacs-validate`: HACS validation action (`hacs/action@main`) — validates manifest.json, directory structure, HACS compatibility. Critical for HACS distribution, often missed.
- **Target runtime**: 3-5 minutes
- **Python version**: Match HA's production Python version

#### Tier 2 — PR Merge Gate (on PR ready for merge)

**Workflow**: `.github/workflows/pr-merge.yml`
- **Trigger**: `on: pull_request` (types: ready_for_review, review_requested) or manual
- **Jobs**:
  - `test-matrix`: Full test suite on multiple Python versions (HA's production version + latest stable)
  - `coverage-report`: Coverage report posted as PR comment (shows delta vs main)
  - `dependency-check`: Verify dependency compatibility with HA's bundled versions
- **Target runtime**: 5-10 minutes

#### Tier 3 — Release Pipeline (on tag push)

**Workflow**: `.github/workflows/release.yml`
- **Trigger**: `on: push` (tags: 'v*')
- **Jobs**:
  - `test`: Full test suite (belt and suspenders)
  - `validate`: HACS validation
  - `release`: GitHub Release creation with auto-generated changelog
  - `version-check`: Verify tag matches manifest.json version

#### Automation Workflows

**Workflow**: `.github/workflows/auto-review.yml`
- **Trigger**: `on: pull_request` (opened, synchronize)
- **Jobs**:
  - Auto-assign reviewers based on changed file paths (CODEOWNERS)
  - Auto-label PRs by area (solver, charger, config, docs, CI)
  - Post risk assessment comment based on changed files and the risk-weighted table above

**Workflow**: `.github/workflows/issue-triage.yml`
- **Trigger**: `on: issues` (opened)
- **Jobs**:
  - Auto-label issues by keywords (bug, feature, solver, charger, documentation)
  - Auto-assign to project board

**Workflow**: `.github/workflows/stale.yml`
- **Trigger**: `on: schedule` (weekly)
- **Jobs**:
  - Mark stale issues/PRs after 30 days of inactivity
  - Close after 60 days with a polite message

**Supporting files needed:**
- `.github/CODEOWNERS` — map file paths to reviewers
- `.github/labeler.yml` — PR auto-labeling rules by path
- `.github/ISSUE_TEMPLATE/bug_report.yml` — structured bug reports
- `.github/ISSUE_TEMPLATE/feature_request.yml` — structured feature requests
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist (tests pass, coverage maintained, risk assessment)

#### PR Template Checklist

```markdown
## Checklist
- [ ] Tests pass locally (`pytest tests/`)
- [ ] 100% coverage maintained
- [ ] Ruff format + check pass
- [ ] MyPy passes
- [ ] No new `# type: ignore` or `noqa` without justification
- [ ] HACS manifest.json updated if version changed

## Risk Assessment
<!-- Which row in the risk-weighted table does this PR touch? -->
- [ ] CRITICAL (solver, constraints, charger budgeting)
- [ ] HIGH (load base, constants, orchestration)
- [ ] MEDIUM (device-specific)
- [ ] LOW (platforms, UI, docs)
```

**Note**: No starter template initialization story needed. CI/CD pipeline setup (GitHub Actions) is the first infrastructure story per the product brief. Test infrastructure smoke tests (verify FakeHass behaves like real HA) should ship as part of the CI/CD story, not as a separate item — the CI pipeline's own self-test.

## Core Architectural Decisions

### Decision Priority Analysis

**Already Decided (by platform, existing codebase, and prior steps):**
- Data architecture: HA state machine + numpy ring buffer persistence (560 days, 15-min intervals) + HA recorder + person mileage via sensor attribute restoration (30 days)
- Authentication & security: Entirely HA's responsibility
- API & communication: HA entities + service calls, no external API
- Frontend: Custom JS Lovelace cards + Jinja2 dashboard templates
- Notification: HA mobile app notifications with per-person configuration, daily forecasts, constraint change alerts, error notifications, off-grid emergency broadcasts
- Infrastructure: HACS distribution, GitHub Actions CI/CD (detailed in step 3)
- Historical data: Multi-year numpy array persistence on disk + HA sensor attribute restoration + HA recorder integration

**Critical Decisions (Made Now):**

#### Decision 1: Resilience & Degradation Strategy

**Decision**: Each external dependency and data quality concern gets an explicit failure mode with documented behavior.

| Dependency | Failure signature | Fallback behavior | Recovery |
|---|---|---|---|
| Solar forecast API (Solcast/OpenMeteo) | API timeout or HTTP error | Use last successful forecast. If stale > 6h, fall back to historical consumption patterns (numpy ring buffer has 560 days of data). | Auto-retry on next 30s forecast cycle. |
| Charger communication | Command not ACKed within CHARGER_STATE_REFRESH_INTERVAL_S (14s) | Retry up to 3 times, then mark charger as unavailable. Solver excludes unavailable chargers. Notify TheAdmin via error notification. | Auto-recover when charger responds. Re-include in next solver cycle. |
| HA state unavailability | Entity state = "unavailable" or "unknown" | HADeviceMixin filters invalid values from history. Device continues with last-known-good state. | Auto-recover when entity returns valid state. |
| Solver infeasibility | No valid plan satisfies all constraints | Fall back to safe defaults: mandatory constraints get priority, filler constraints dropped. Battery held at current SOC. | Re-solve on next triggered evaluation or 5-minute fallback. |
| Grid outage | Grid power sensor drops to zero | Off-grid mode: reduce consumption to solar + battery capacity. Emergency broadcast to all mobile apps (high priority, alarm channel). | Auto-detect grid restoration. Broadcast recovery. Resume normal operation. |
| Numpy persistence corruption | Corrupted `.npy` file (partial write during power outage, disk full, permission error) | `np.load()` defaults to None on failure. System operates without historical data — cold start within a running system. Fall back to conservative consumption estimates. Log warning. | Ring buffer rebuilds over time as new data accumulates. Full 560-day window takes ~1.5 years to restore. |
| Prediction confidence degradation | Insufficient history, holiday period, major life change (new job/commute) | When prediction history is sparse or pattern match score is low, fall back to conservative defaults: charge to higher SOC than predicted necessary. Better to over-charge than leave a car short. | Predictions self-correct as new patterns accumulate in the 30-day mileage window. |

**Rationale**: The system is already partially resilient (off-grid mode, charger retry logic, invalid state filtering). This decision makes the strategy explicit and complete so AI agents implement consistent fallback behavior across all failure paths.

#### Decision 2: Trust-Critical Component Testing Strategy

**Decision**: Charger dynamic budgeting and constraint interaction testing are formal architectural requirements, not optional test improvements.

**Requirements:**
- Scenario-based multi-charger integration tests that simulate rebalancing sequences over time and verify no phase is exceeded at any intermediate state
- Constraint interaction boundary tests: MANDATORY vs exhausted switching budget, mandatory vs exhausted amp budget, multiple MANDATORY constraints competing for insufficient power
- Test infrastructure smoke tests: verify FakeHass behaves like real HA for critical operations — ships with the CI/CD story as the pipeline's self-test
- All trust-critical tests must use the `@pytest.mark.integration` marker for targeted CI execution

**Rationale**: Charger budgeting bugs trip breakers (physical consequences). Constraint interaction bugs cause missed commitments (trust consequences). These are the highest-risk areas identified in the gap analysis and deserve formal testing commitments.

#### Decision 3: Solver Optimization Strategy (Near-Term)

**Decision**: Solver improvements happen inside `PeriodSolver.solve()` without changing the input/output contract. The solver interface is stable.

**Boundary**: The solver accepts (constraints, tariffs, PV forecast, battery state, loads) and returns (load, [(time, LoadCommand)]) timelines. Optimizations improve the allocation algorithm within this contract. No restructuring of the component model.

**Rationale**: The product brief lists "solver improvements for edge cases" as a near-term priority. This is distinct from solver evolution (LP/MILP paradigm shift, which is deferred). AI agents working on solver optimization have a clear boundary: improve decisions inside the box, don't redesign the box.

**Deferred Decisions (Future Scope):**

#### Deferred: Dynamic Tariff Architecture
- **What**: Support for real-time spot pricing beyond current peak/off-peak
- **Why deferred**: Product brief marks this as future scope. Current tariff model (list of (datetime, price_per_wh) tuples) is extensible — dynamic tariffs can feed into the same interface.
- **Architectural preparation**: The solver already accepts tariff tuples. Future work adds a tariff provider abstraction (similar to solar forecast providers) without changing the solver interface.

#### Deferred: Solver Evolution
- **What**: Explore LP/MILP formulations for provably optimal solutions
- **Why deferred**: Current constraint-based solver is in production and working. Near-term improvements happen within the existing approach (Decision 3).
- **Architectural preparation**: The solver's input/output interface is stable. Alternative solver implementations can be swapped without changing the rest of the system.

### Decision Impact Analysis

**Implementation Sequence:**
1. **CI/CD pipeline** (GitHub Actions) + test infrastructure smoke tests — enables all subsequent work
2. **Bug fix: person-car assignment** — first story through the pipeline, validates the BMad workflow end-to-end with a contained, lower-risk story
3. **Trust-critical component tests** — charger scenarios, constraint interactions — fills highest-risk test gaps, now proven through a validated pipeline
4. **Resilience strategy implementation** — explicit fallback for each dependency
5. **Solver optimization** — edge case improvements within stable interface

**Rationale for sequence**: The bug fix (step 2) comes before trust-critical tests (step 3) because the product brief explicitly says "assignment bug fix is the first feature delivered through the structured workflow — proving the process works." Validate the workflow with a simpler story before tackling complex test scenarios through it.

**Cross-Component Dependencies:**
- Resilience strategy touches: solar.py (forecast fallback), charger.py (communication retry), home.py (solver infeasibility, numpy persistence), device.py (state unavailability), person.py (prediction confidence)
- Trust-critical tests touch: charger.py, dynamic_group.py, constraints.py, solver.py, conftest.py (infrastructure smoke tests)
- Solver optimization touches: solver.py, constraints.py only (stable interface protects other components)
- Both testing and resilience decisions feed into the CI/CD risk-weighted strategy from step 3

**Documentation note**: This architecture document partially satisfies product brief success criterion #5 ("documentation started"). Combined with the project-context.md, the foundational documentation is in place.

## Implementation Patterns & Consistency Rules

### Relationship to project-context.md

project-context.md contains 42 rules covering code-level conventions (naming, async, logging, error handling, testing anti-patterns). This section covers **architectural-level patterns** — how to extend and modify the system without breaking its design. AI agents MUST read both documents before implementing any code.

### Pattern 1: Adding a New Device Type

This is the most common extension pattern and touches 7+ files. Follow this exact sequence:

0. **const.py** (dashboard): Map device to dashboard section in `LOAD_TYPE_DASHBOARD_DEFAULT_SECTION`. Without this, the device is functional but invisible on the dashboard.
1. **const.py** (config): Add `CONF_TYPE_NAME_QS<Name>` constant + all `CONF_<NAME>_*` config keys
2. **home_model/<name>.py**: Create domain class extending `AbstractLoad` or `AbstractDevice` (pure Python, zero HA imports)
3. **ha_model/<name>.py**: Create HA class extending `HADeviceMixin` + domain class. Implement:
   - `__init__()`: call `attach_ha_state_to_probe()` for each tracked entity
   - `execute_command()`: translate LoadCommand -> HA service calls
   - `probe_if_command_set()`: verify command took effect by reading entity state
4. **config_flow.py**: Add `async_step_<name>()` using `get_common_schema()` builder
5. **Platform files**: Add `create_ha_<platform>_for_<Name>()` factory in each relevant platform
6. **Tests**: Full coverage for domain logic (using factories in `tests/`) AND HA integration (using real fixtures in `tests/ha_tests/`)
7. **Verify**: Device appears on dashboard with correct section mapping

**Anti-pattern**: Adding a device only in ha_model/ without a domain model class. Even simple devices need a domain layer presence to participate in the solver. Also: forgetting the dashboard section mapping — creates a ghost device that works but nobody can see.

### Pattern 2: Creating and Managing Constraints

Constraints are the system's demand language. When implementing features that create demand:

- **Choose the right constraint type** by priority: MANDATORY_AS_FAST_AS_POSSIBLE (must happen now) > MANDATORY_END_TIME (must happen by deadline) > BEFORE_BATTERY_GREEN (before battery goes solar) > FILLER (use surplus) > FILLER_AUTO (lowest priority)
- **Always include load_info metadata**: `{originator: "user_override" | "agenda" | "prediction" | "system"}` — this traces where the constraint came from for notification and debugging
- **Push constraints through the correct API**: `push_live_constraint()` for dynamic runtime constraints, `push_agenda_constraints()` for calendar/schedule-based constraints
- **Create constraints in the right lifecycle phase**: Create and push constraints in `update_loads_constraints()` (called during the 7s load management cycle). Never in `__init__()` (too early, solver hasn't started) or `update_states()` (wrong cycle, creates timing issues with the solver).
- **Never modify constraints directly** — create new ones. The solver reads constraints via `get_for_solver_constraints()` each cycle.

**Anti-pattern**: Creating a constraint without load_info metadata. This breaks notification tracing — the system can't tell the user why a decision was made. Also: creating constraints in the wrong lifecycle phase causes solver timing issues.

### Pattern 3: Command Execution & Verification

All device commands follow a launch -> execute -> verify cycle:

- **execute_command(time, command)**: Translate the domain LoadCommand into HA service calls. Return False (async, not awaiting ACK).
- **probe_if_command_set(time, command)**: Read the actual device state from HA and compare to expected state. Return True only when the device has confirmed the command.
- **Respect rate limiting**: Max 1 command per load per solve cycle. The orchestrator handles timing — individual devices should not implement their own scheduling.
- **Handle external control**: If the device state doesn't match any command quiet-solar sent, someone controlled it externally. Detect this in `update_states()` and set `external_user_initiated_state` appropriately.

**Anti-pattern**: Awaiting inside execute_command(). The HA event loop must not be blocked. All verification happens asynchronously via probe_if_command_set().

### Pattern 4: State Tracking via HADeviceMixin

All HA entity state flows through HADeviceMixin:

- **Attach entities on init**: `attach_ha_state_to_probe(entity_id, is_numerical=True, conversion_fn=...)` for auto-polled entities
- **Use standard conversion functions**: `convert_power_to_w`, `convert_amps`, `convert_km` — don't invent new converters
- **Query state via the mixin API**: `get_sensor_latest_possible_valid_value()` for current state, `get_state_history_data()` for time-series
- **Polling vs event-driven**: Most entities go in `_entity_probed_auto` (polled every 4s). High-frequency or critical-change entities can use `_entity_on_change` (event-driven). When in doubt, use auto-polled — it's the safer default.
- **Never access hass.states directly** in ha_model code — always go through the mixin's history

**Anti-pattern**: Reading HA state directly via `hass.states.get()` in device code. This bypasses history tracking and conversion functions.

### Pattern 5: Charger Budgeting Interaction

When modifying charger behavior or the dynamic group tree:

- **Budget validation is recursive**: Always call `is_delta_current_acceptable()` or `is_current_acceptable_and_diff()` before changing charger amps. These propagate up the tree.
- **Phase switching before amp reduction**: Try 1P->3P conversion (reduces per-phase load) before reducing amperage. Less disruptive.
- **Staged transitions for large changes**: If changing multiple chargers, reduce first (Phase 1), then increase (Phase 2 in next cycle). Never increase and decrease simultaneously — transient overages trip breakers.
- **Respect the 45-second adaptation window**: No budget changes until all chargers in the group have been stable for CHARGER_ADAPTATION_WINDOW_S.
- **Per-car charging curves**: Amp-to-power conversion uses per-car lookup tables (`amp_to_power_1p[amps]`, `amp_to_power_3p[amps]`), not linear calculation. Different EVs have different charging curves. Always use the car's LUT for power budget calculations.

**Anti-pattern**: Changing charger amps without calling budget validation. This bypasses circuit protection and can cause physical damage. Also: using linear amp-to-power conversion instead of per-car lookup tables — gives wrong power budget for every car except the one you tested with.

### Pattern 6: Solver Extension

When modifying the solver (within the stable interface per Decision 3):

- **Input contract**: (constraints, tariffs, PV forecast, battery state, loads) — don't add new input types without explicit architectural decision
- **Output contract**: list of (load, [(time, LoadCommand)]) — don't change the output format
- **Constraint scoring drives allocation order**: Modify scoring to change priorities, don't hardcode allocation sequences
- **15-minute time slots are fixed**: SOLVER_STEP_S = 900. Don't change without understanding the full impact on all constraint and budgeting logic.
- **Optimization hierarchy**: maximize solar self-consumption → minimize energy cost → maintain comfort. All trade-off decisions follow this strict priority ordering.
- **Mandatory before filler**: The allocation sequence (mandatory constraints -> battery optimization -> filler constraints) is an architectural invariant, not an implementation detail.
- **Battery is the ONE exception**: Battery has dedicated solver phases (`_battery_get_charging_power`, `_prepare_battery_segmentation`). All OTHER device types are device-agnostic in the solver. Don't add new device-specific logic — express device behavior through constraints and commands.

**Anti-pattern**: Adding special-case logic for a specific device type inside the solver. The solver is device-agnostic (except battery). Device-specific behavior belongs in the constraint or load implementation.

### Pattern 7: Notification Triggers

When implementing features that should notify household members:

- **Use `on_device_state_change()`** with the appropriate status type: `DEVICE_STATUS_CHANGE_CONSTRAINT`, `DEVICE_STATUS_CHANGE_CONSTRAINT_COMPLETED`, `DEVICE_STATUS_CHANGE_ERROR`
- **Notification routing**: Charger notifications route to the car's forecasted person's mobile app. Person notifications use the person's configured mobile app. Device notifications use the device's own mobile_app config.
- **Off-grid broadcasts**: Only QSHome sends emergency broadcasts to ALL mobile apps. Individual devices should never broadcast.
- **Human-readable messages**: Use `get_active_readable_name(time, filter_for_human_notification=True)` for constraint descriptions. Never expose internal IDs or technical details in notifications.
- **Deduplication is automatic**: The system tracks `_last_notification_hash` to avoid sending duplicates. When using `on_device_state_change()`, deduplication is handled for you. If you bypass this and call notify directly, you will spam the user every load management cycle (~7 seconds).

**Anti-pattern**: Sending notifications directly via hass.services.async_call(notify, ...) from device code. Always go through on_device_state_change() for consistent routing, message formatting, and deduplication.

### Pattern 8: Config Entry Persistence

When a device needs to persist runtime data back to its configuration:

- **Use `save_entry_data_no_reload()`** to persist data without triggering a reload. This updates the config entry data in place.
- **Never call `hass.config_entries.async_update_entry()`** for runtime data updates — this triggers a full reload cycle: all devices re-initialize, solver restarts, chargers lose their adaptation state, adaptation windows reset. A full reload for a config update is destructive.

**Anti-pattern**: Using `async_update_entry()` for runtime persistence. This causes a cascade: reload -> all devices re-init -> solver cold start -> charger adaptation lost -> 45-second stabilization delay before budgeting resumes.

### Pattern 9: Test Layer Selection

Tests live in two distinct layers. Choosing the wrong layer creates slow, brittle, or misleading tests:

| What you're testing | Where to write tests | Infrastructure to use |
|---|---|---|
| Domain logic (solver, constraints, commands, load behavior) | `tests/` | FakeHass + factories from `factories.py` |
| HA integration (entity lifecycle, service calls, state tracking, config flow) | `tests/ha_tests/` | Real HA fixtures from `tests/ha_tests/conftest.py` |
| Cross-cutting integration (full device setup + solver + command execution) | `tests/` with composite fixtures | `home_and_charger`, `home_charger_and_car` fixtures |

- **Never test pure domain logic through real HA fixtures** — slow, brittle, tests the framework more than your code
- **Never test HA-specific behavior with FakeHass** — misses real HA interactions, gives false confidence
- **Use factories for domain objects, mock configs for HA objects**: Check `factories.py` and `tests/ha_tests/const.py` before writing new test infrastructure

**Anti-pattern**: Writing all tests in `tests/ha_tests/` because "it's more realistic." Domain logic tests should be fast and isolated. Reserve HA fixtures for testing the HA integration surface.

### Enforcement Guidelines

**All AI Agents MUST:**
1. Read project-context.md (code-level rules) AND this architecture document (architectural patterns) before implementing
2. Follow the Decision Map (step 2) to identify which components are affected
3. Check the Risk-Weighted CI table (step 3) to understand the blast radius of their changes
4. Use the correct extension pattern for the type of change (device, constraint, command, state, notification)
5. Never bypass budget validation, rate limiting, or the domain/HA boundary
6. Choose the correct test layer (Pattern 9) before writing any tests
7. Use `save_entry_data_no_reload()` for runtime persistence, never `async_update_entry()`

**Pattern Violations:**
- Ruff and MyPy catch code-level violations automatically
- Architectural violations (wrong layer, missing load_info, bypassed budget validation) must be caught in code review
- CI risk assessment comment (auto-review workflow) flags changes to trust-critical components

## Project Structure & Boundaries

### Quick Navigation for AI Agents

When implementing a story, use this decision tree to find your files. For deeper context on what you need to *understand* before modifying those files, see the Decision Map in the Conceptual Component Model section.

| If you're implementing... | Start here | Test location |
|---|---|---|
| New device type | Pattern 1 (7-file checklist) | `tests/` (domain) + `tests/ha_tests/` (HA) |
| New constraint behavior | `home_model/constraints.py` | `tests/test_constraints*.py` |
| New HA integration for existing device | `ha_model/<device>.py` | `tests/ha_tests/test_<device>.py` |
| New platform entity | `<platform>.py` (factory pattern) | `tests/test_platform_<platform>.py` |
| Notification or override flow | Trace full path: `person.py` → `home.py` → notify | Tests across domain + HA layers |
| CI/CD change | `.github/workflows/` | Validate by running pipeline on test PR |
| Before writing test helpers | Check `tests/utils/` first | `energy_validation.py`, `scenario_builders.py` |

**Marker legend**: `[PLANNED]` = committed in the implementation sequence. `[FUTURE]` = architectural preparation, no commitment yet.

### Complete Project Directory Structure

The project has two distinct structural zones: **deployable code** (what ships via HACS) and **development infrastructure** (what supports development but never deploys).

#### Deployable Code

```
quiet-solar/
├── hacs.json                          # HACS distribution metadata (render_readme, HA version minimum).
│                                      #   Update when changing HA version requirements.
├── requirements.txt                   # Production dependencies
├── custom_components/quiet_solar/     # === THE COMPONENT ===
│   ├── manifest.json                  # HA component manifest. Version is managed here —
│   │                                  #   must match git tag for release pipeline.
│   ├── strings.json                   # UI string definitions
│   ├── translations/en.json           # English translations
│   │
│   ├── __init__.py                    # Entry point: async_setup_entry(), platform forwarding
│   ├── const.py                       # ALL constants, config keys, device type names
│   ├── config_flow.py                 # Configuration UI: menu → device-type steps
│   ├── data_handler.py                # QSDataHandler: config entry lifecycle, timing cycles [NOTE 1]
│   ├── entity.py                      # Base entity classes
│   ├── time.py                        # Time entity implementations
│   │
│   │   # --- Platform files (HA boilerplate, all follow same factory pattern:
│   │   #     type-specific creator → polymorphic dispatcher → platform setup.
│   │   #     LOW risk per CI strategy.) ---
│   ├── sensor.py                      # Sensor platform
│   ├── binary_sensor.py               # Binary sensor platform
│   ├── switch.py                      # Switch platform
│   ├── number.py                      # Number platform
│   ├── select.py                      # Select platform
│   ├── button.py                      # Button platform
│   │
│   ├── home_model/                    # === DOMAIN LAYER (pure Python, zero HA imports) ===
│   │   ├── __init__.py
│   │   ├── commands.py                # LoadCommand enum (10 types, priority-scored)
│   │   ├── constraints.py             # LoadConstraint hierarchy (5 priority tiers)
│   │   ├── load.py                    # AbstractDevice, PilotedDevice, AbstractLoad
│   │   ├── solver.py                  # PeriodSolver: strategic 15-min optimization
│   │   ├── battery.py                 # Battery charge/discharge model
│   │   └── home_utils.py              # Shared domain utilities
│   │                                  # [FUTURE: providers/ — forecast & tariff provider abstractions]
│   │
│   ├── ha_model/                      # === HA INTEGRATION LAYER ===
│   │   ├── __init__.py
│   │   ├── device.py                  # HADeviceMixin: bridge pattern base
│   │   ├── home.py                    # QSHome: orchestrator (root of dynamic group tree)
│   │   ├── dynamic_group.py           # QSDynamicGroup: hierarchical amp budgeting
│   │   ├── charger.py                 # QSCharger variants (OCPP/Wallbox/Generic) + budgeting
│   │   ├── car.py                     # QSCar: SOC tracking, person allocation
│   │   ├── battery.py                 # QSBattery: HA charge/discharge control
│   │   ├── solar.py                   # QSSolar: production + forecast providers
│   │   ├── person.py                  # QSPerson: presence, trip prediction, mileage
│   │   ├── pool.py                    # QSPool: temperature-dependent filtering
│   │   ├── heat_pump.py               # QSHeatPump: multi-client via PilotedDevice
│   │   ├── on_off_duration.py         # QSOnOffDuration: simple binary switch loads
│   │   ├── climate_controller.py      # Climate controller integration [NOTE 2]
│   │   └── bistate_duration.py        # Bi-state duration device [NOTE 2]
│   │                                  # [FUTURE: providers/ — HA-specific provider bridges]
│   │
│   └── ui/                            # === UI LAYER ===
│       ├── __init__.py
│       ├── dashboard.py               # Programmatic dashboard generation (Jinja2 + HA API)
│       ├── quiet_solar_dashboard_template.yaml.j2
│       ├── quiet_solar_dashboard_template_standard_ha.yaml.j2
│       └── resources/                 # Custom Lovelace JS cards (outside quality pipeline)
│           ├── qs-car-card.js
│           ├── qs-climate-card.js
│           ├── qs-on-off-duration-card.js
│           └── qs-pool-card.js
```

#### Structural Notes

**[NOTE 1] — data_handler.py location anomaly**: `data_handler.py` is architecturally part of the HA integration layer (it manages config entries, creates QSHome, sets up timing cycles). It lives at the component root for import path stability — established before the layer separation. Do not move it to `ha_model/` without updating all import chains. (Also noted as HIGH risk in the CI strategy — see risk-weighted table.)

**[NOTE 2] — Thin wrapper devices**: `bistate_duration.py` and `climate_controller.py` in `ha_model/` extend existing domain classes (`AbstractLoad` / `PilotedDevice`) without dedicated domain model files. This is acceptable for devices that add only HA-specific behavior with no new domain logic. They are not boundary violations. If a thin wrapper device later needs domain-specific logic (custom constraints, solver-visible behavior), promote it to a full device type by creating a `home_model/<name>.py` file. This is a normal evolution, not a refactoring emergency.

**[FUTURE] — Provider abstractions**: When dynamic tariff support (deferred decision) or additional forecast providers are implemented, provider abstractions would live in `home_model/providers/` (domain interface) + `ha_model/providers/` (HA-specific bridges). Do not create these directories until needed.

**CLAUDE.md role**: CLAUDE.md is the Claude Code bootstrap file (auto-loaded at conversation start). It should contain only operational essentials (venv, commands) and point to `_bmad-output/project-context.md` and `_bmad-output/planning-artifacts/architecture.md` for the full rules. Do not duplicate content between CLAUDE.md and the BMAD files.

#### Runtime Data

```
{HA config dir}/quiet_solar/          # Numpy ring buffer storage (.npy files)
{HA config dir}/quiet_solar/debug/    # Debug dumps (pickle, forecast data)
```

Runtime data at `{HA config dir}/quiet_solar/` is distinct from the component source at `{HA config dir}/custom_components/quiet_solar/`. The former contains `.npy` data files; the latter contains Python source.

#### Development Infrastructure

```
quiet-solar/
├── CLAUDE.md                          # AI agent bootstrap (points to BMAD files for full rules)
├── requirements_test.txt              # Test dependencies
├── pytest.ini                         # Pytest configuration (asyncio_mode=auto, pythonpath=., markers)
├── setenv.sh                          # Development environment setup (venv + pip install)
│
├── docs/                              # [PLANNED — documentation story, does not exist yet]
│                                      # User guide, setup guide, contribution guide
│                                      # Do NOT create documentation files in _bmad-output/
│
├── .github/                           # [PLANNED — CI/CD story, does not exist yet]
│   ├── workflows/
│   │   ├── pr-quality.yml             # Tier 1: lint + typecheck + test + HACS
│   │   ├── pr-merge.yml               # Tier 2: matrix test + coverage report
│   │   ├── release.yml                # Tier 3: release pipeline
│   │   ├── auto-review.yml            # Auto-label, auto-assign, risk assessment
│   │   ├── issue-triage.yml           # Issue auto-labeling
│   │   └── stale.yml                  # Stale issue/PR management
│   ├── CODEOWNERS
│   ├── labeler.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   └── PULL_REQUEST_TEMPLATE.md
│
├── tests/                             # === TEST SUITE (~3,800+ tests) ===
│   ├── __init__.py
│   ├── conftest.py                    # ⚠ PROTECTED: FakeHass, FakeConfigEntry, FakeState
│   ├── factories.py                   # ⚠ PROTECTED: Factory functions for domain test doubles
│   │
│   ├── [domain + cross-cutting tests — see Test File Organization below]
│   │
│   ├── ha_tests/                      # === HA INTEGRATION TESTS (real HA fixtures) ===
│   │   ├── __init__.py
│   │   ├── conftest.py                # ⚠ PROTECTED: Real HA fixtures (hass, mock_config_entry)
│   │   ├── const.py                   # ⚠ PROTECTED: MOCK_HOME_CONFIG, MOCK_CAR_CONFIG, etc.
│   │   └── [test files using real HA fixtures]
│   │
│   └── utils/                         # === TEST UTILITIES ===
│       ├── __init__.py
│       ├── energy_validation.py       # Energy conservation assertion helpers
│       └── scenario_builders.py       # Scenario construction for multi-device tests
│
└── _bmad-output/                      # === BMAD WORKFLOW OUTPUT ===
    ├── project-context.md             # 42 AI agent rules
    ├── planning-artifacts/
    │   ├── architecture.md            # THIS DOCUMENT
    │   └── product-brief-*.md
    ├── implementation-artifacts/       # [PLANNED] Story specs
    └── test-artifacts/                # [PLANNED] Test plans
```

**Build artifacts** (`coverage.xml`, `*.pyc`, `__pycache__/`) live at project root and should be gitignored. Do not commit build artifacts.

### Architectural Boundaries

**Strict Boundaries (violations are critical errors):**

| Boundary | Rule | Enforcement |
|---|---|---|
| Domain/HA | `home_model/` NEVER imports `homeassistant.*` | Code review + grep in CI |
| Domain purity | Domain layer receives all HA data as parameters | Code review |
| Constant centralization | All config keys in `const.py`, never hardcoded | Ruff linting |

**Strategic/Tactical boundary:**

| Layer | Component | Scope | Override behavior |
|---|---|---|---|
| Strategic | PeriodSolver | 15-min windows, all loads | Plans optimal allocation |
| Tactical | Charger dynamic budgeting | Real-time, chargers only | Can override strategic layer when amp budget constrains |

**Component Communication Boundaries:**

| From | To | Mechanism | Boundary rule |
|---|---|---|---|
| HA state | Domain model | `HADeviceMixin.add_to_history()` | All state flows through mixin — never `hass.states.get()` in device code |
| Solver | Devices | `LoadCommand` timeline | Rate-limited to 1 cmd/load/cycle |
| Solver | Battery | Dedicated solver phases (`_battery_get_charging_power`, `_prepare_battery_segmentation`) | Only device with special solver treatment — all others are device-agnostic |
| Devices | HA | `execute_command()` → HA service calls | Async, non-blocking, no await inside execute |
| Charger budgeting | Chargers | Budget allocation via `apply_budget_strategy()` | Validates amp budget recursively up tree before applying |
| Constraints | Solver | `get_for_solver_constraints()` | Read-only access; never modify constraints directly |
| Notifications | Mobile apps | `on_device_state_change()` → HA notify service | Always through the routing/dedup layer, never direct |
| Config persistence | Config entry | `save_entry_data_no_reload()` | Never `async_update_entry()` for runtime data |

### External Integration Points

| External system | Integration point | Failure mode | Data path |
|---|---|---|---|
| Solcast / OpenMeteo | `ha_model/solar.py` → API polling every ~30s | Stale forecast → use last good → fall back to numpy history | Solar forecast → solver PV input |
| OCPP charger | `ha_model/charger.py` → OCPP transaction protocol | Command not ACKed → retry → mark unavailable | Charger state → budgeting → solver |
| Wallbox charger | `ha_model/charger.py` → Wallbox API | Same as OCPP | Same path |
| HA entity states | `ha_model/device.py` → entity state polling ~4s | "unavailable"/"unknown" filtered → last-known-good | Entity → mixin history → device state |
| HA mobile app | `ha_model/home.py` → notify service | Notification failure → logged, not retried | Device state change → notification service |
| Grid power sensor | `ha_model/home.py` → grid power entity | Zero reading → off-grid mode → emergency broadcast | Grid state → home → solver mode switch |

### Near-Term Work → Structure Mapping

| Work item (from Implementation Sequence) | Files to create | Files to modify | Structural notes |
|---|---|---|---|
| CI/CD pipeline | `.github/` (entire directory) | `pytest.ini` (markers) | First structural addition. Creates all `[PLANNED]` .github files. Implement `paths:` filters in workflow triggers to map to the risk-weighted CI strategy (step 3). |
| CLAUDE.md slim-down | None | `CLAUDE.md` | Reduce to bootstrap: commands + pointers to BMAD files. Remove duplicated content. |
| Legacy cleanup | None | Remove `setup.py` | Not needed — `pytest.ini` handles pythonpath, `manifest.json` handles distribution. |
| Bug fix: person-car assignment | None | `ha_model/person.py`, `ha_model/car.py`, `ha_model/home.py`, tests | First story through CI pipeline. Contained changes. |
| Trust-critical component tests | New test file(s) for scenario-based charger integration tests — exact structure TBD during test design story. Name should reflect test *type* (scenario sequences), not coverage target. | `tests/conftest.py` (new fixtures) | Add infrastructure smoke tests alongside. |
| Resilience strategy | None | `ha_model/solar.py`, `ha_model/charger.py`, `ha_model/home.py`, `ha_model/device.py`, `ha_model/person.py` | Touches many files but each change is contained (fallback logic per dependency). |
| Solver optimization | None | `home_model/solver.py`, `home_model/constraints.py` | Contained within stable interface (Decision 3). |
| Documentation | `docs/` directory at project root | None | Not in current implementation sequence. When stories are created, establish `docs/` — do NOT put user-facing docs in `_bmad-output/`. |

### Test File Organization

**File selection rule for agents**: When adding tests, extend existing files by test type rather than creating new files. If a file exceeds ~500 tests, split by sub-concern (e.g., phase switching, budgeting, dampening). Do not create `test_*_additional_coverage.py` or `test_*_deep.py` files — these names reflect organic growth, not intentional structure.

**Root `tests/` naming clarification**: Files prefixed `test_ha_*` in `tests/` root use **FakeHass infrastructure** (lightweight, fast). Files in `tests/ha_tests/` use **real HA fixtures** (full HA instance). The naming overlap is historical. When determining which infrastructure a test uses, **check the imports** (FakeHass from `conftest.py` vs real `hass` fixture from `ha_tests/conftest.py`), not the file name.

**Shared test utilities**: Before writing test helper functions, check `tests/utils/` for existing utilities. `energy_validation.py` provides energy conservation assertion helpers. `scenario_builders.py` provides scenario construction helpers for multi-device tests. Do not reinvent helpers that already exist.

### Protected Infrastructure Files

These files are load-bearing — changes silently affect the behavior of hundreds of tests. They require extra review scrutiny:

| File | Role | Risk if broken |
|---|---|---|
| `tests/conftest.py` | FakeHass, FakeConfigEntry, FakeState, FakeServices | Every domain test may pass while testing the wrong thing |
| `tests/factories.py` | Factory functions for all domain test doubles | Same — silently alters 100+ test files |
| `tests/ha_tests/conftest.py` | Real HA fixtures | Every HA integration test affected |
| `tests/ha_tests/const.py` | Standard mock configurations | HA tests use wrong config silently |

These files are covered by the CI/CD risk-weighted strategy (HIGH blast radius) and should be included in CODEOWNERS for mandatory review. When modifying these files, check for in-flight PRs that touch the same file to avoid merge conflicts on load-bearing test infrastructure.

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
All decisions work together without conflicts:
- Technology stack (Python 3.14+, HA 2026.2.1+, numpy/scipy, HACS) is internally consistent
- CI/CD strategy (3-tier GitHub Actions) aligns with the risk-weighted table and test infrastructure
- Resilience strategy (Decision 1) aligns with existing device architecture — each dependency maps to a specific file in `ha_model/`
- Trust-critical testing (Decision 2) feeds directly into the CI pipeline design and risk-weighted strategy
- Solver optimization (Decision 3) preserves the stable interface that the rest of the system depends on
- Deferred decisions (dynamic tariffs, solver evolution) have explicit architectural preparation so they won't disrupt current work
- No contradictory decisions found

**Pattern Consistency:**
- All 9 implementation patterns reference correct APIs, files, and lifecycle phases
- Naming conventions are consistent: `CONF_*` for config keys, `CONF_TYPE_NAME_QS*` for device types, `*_SENSOR` suffixes
- Pattern 9 (test layer selection) aligns with the test file organization in Step 6
- Pattern 1 references "7+ files" which correctly matches the 0-7 numbered sequence (including step 0: dashboard mapping)
- Anti-patterns complement the positive patterns — no gaps where an agent could follow the pattern correctly but still make a mistake

**Structure Alignment:**
- Directory structure supports all 3 architectural layers with clear separation
- Quick Navigation cross-references the Decision Map — agents have both "where" and "why"
- Structural notes explain anomalies (data_handler.py location, thin wrapper devices) so agents don't try to "fix" what isn't broken
- `[PLANNED]` vs `[FUTURE]` markers distinguish committed work from speculative preparation

### Requirements Coverage Validation ✅

**Product Brief Feature Coverage:**

| Product brief requirement | Architectural support | Covered in |
|---|---|---|
| Whole-house energy orchestration | PeriodSolver + constraint system + 8+ device types | Conceptual Component Model |
| Constraint-based solver (4 priority tiers) | PeriodSolver, LoadConstraint hierarchy | Component Model + Pattern 6 |
| People-aware automation | Person model, trip prediction, person-car allocation | Component Model + Pattern 2 |
| Smart device handling (external control) | `external_user_initiated_state` detection | Pattern 3 |
| Off-grid resilience | Grid outage row in resilience table | Decision 1 |
| Tariff-aware scheduling | Solver accepts tariff tuples; dynamic tariffs deferred | Decision 3 + Deferred |
| UI dashboards | Jinja2 + 4 custom JS cards + programmatic HA API | Component Model (UI layer) |
| Notification & override flows | `on_device_state_change()`, per-person routing, dedup | Pattern 7 |
| Multi-protocol charger support | OCPP/Wallbox/Generic variants | Component Model |
| Real-time charger budgeting | Tactical layer, QSDynamicGroup tree, staged transitions | Hierarchical Control Architecture |
| 100% test coverage | CI pipeline threshold = 100%, mandatory | CI/CD Pipeline Architecture |
| HACS distribution | manifest.json, hacs.json, HACS validation in CI | Technology Stack + CI |

**Near-Term Success Criteria (from product brief):**

| Criterion | Architectural support | Status |
|---|---|---|
| CI/CD operational | Full 3-tier pipeline + 3 automation workflows designed | ✅ Ready to implement |
| BMad workflow validated | Bug fix as first story through pipeline | ✅ Sequenced as step 2 |
| Solver improved | Stable interface boundary (Decision 3) | ✅ Boundary defined |
| 100% coverage maintained | CI enforces, risk-weighted strategy prioritizes | ✅ Pipeline designed |
| Documentation started | Architecture doc + project-context.md | ✅ This document |
| Ready for structured feature work | 9 patterns + structure mapping + decision map | ✅ Complete |

**Non-Functional Requirements:**

| NFR | Architectural support |
|---|---|
| Performance (solver timing, charger adaptation) | Documented: ~7s re-evaluation, 45s adaptation windows, 15-min solver steps |
| Security | Delegated to HA platform (noted "Already Decided") |
| Reliability | Resilience strategy with 7-row failure table (Decision 1) |
| Async discipline | project-context.md rules + architecture boundary enforcement |
| Genericity | Device-agnostic solver, pattern-based extension, config flow abstraction |

### Implementation Readiness Validation ✅

**Decision Completeness:**
- 3 critical decisions with rationale and implementation sequence ✅
- 2 deferred decisions with architectural preparation ✅
- 7 "Already Decided" items documented (not reinvented) ✅
- Implementation sequence justified with cross-component dependencies ✅

**Structure Completeness:**
- Full directory tree with per-file annotations ✅
- Structural notes for 4 anomalies/clarifications ✅
- Runtime data paths documented ✅
- Near-term work → structure mapping (8 work items) ✅
- Protected infrastructure files identified ✅

**Pattern Completeness:**
- 9 patterns covering all extension types ✅
- Every pattern has an anti-pattern ✅
- Decision Map (step 2) provides component-level guidance ✅
- Quick Navigation (step 6) provides file-level guidance ✅
- Enforcement guidelines with 7 mandatory agent rules ✅
- Patterns cover *extension* (add device, add constraint, add notification). *Modification* stories (bug fixes, behavior changes in existing code) require agents to read existing code and apply patterns contextually — this is expected and correct behavior.

**Agent Navigation Test** — can an agent find what it needs?
- "I need to add a new device" → Quick Nav → Pattern 1 → 7-file checklist → test layer (Pattern 9) ✅
- "I need to fix a charger bug" → Decision Map → charger files → Risk table (CRITICAL, physical) → trust-critical testing ✅
- "I need to set up CI/CD" → Quick Nav → `.github/` → CI/CD Pipeline Architecture → risk-weighted strategy → Near-Term Work mapping ✅
- "I need to understand why the solver made this decision" → Data Flows → PeriodSolver → constraint scoring → command timeline ✅
- "I need to add a notification for a new event" → Pattern 7 → notification routing → `on_device_state_change()` → trace Magali path (person → constraint → solver → charger → notification) → test in domain + HA layers ✅

### Gap Analysis Results

**Critical Gaps: None.** All blocking decisions are made. No undefined integration points or missing patterns that would prevent implementation.

**Known Important Gaps (documented and committed):**

| Gap | Risk | Resolution | Where documented |
|---|---|---|---|
| Charger dynamic budgeting test gaps | HIGH (physical) | Decision 2 commits to scenario-based tests | Architectural Gaps #1 + Decision 2 |
| Constraint interaction test gaps | HIGH (trust) | Decision 2 commits to boundary tests | Architectural Gaps #2 + Decision 2 |
| Resilience implementation | HIGH (availability) | Decision 1 provides failure table | Decision 1 |
| Test infrastructure smoke tests | MEDIUM | Ships with CI/CD story | Decision 2 + CI/CD Pipeline |

**Known Future Gaps (explicitly deferred):**

| Gap | Status | Architectural preparation |
|---|---|---|
| Dynamic tariff support | Deferred | Solver accepts tariff tuples; future provider abstraction path noted |
| Solver evolution (LP/MILP) | Deferred | Stable input/output interface enables swap |
| Solver quality benchmarks | Future | Correctness tested, optimality not |
| Data/learning architecture | Future | Numpy ring buffer exists; no pattern analysis layer |
| JS card quality pipeline | Known gap | Outside quality pipeline, documented |

### Architecture Completeness Checklist

**✅ Requirements Analysis (Step 2)**
- [x] Project context analyzed with codebase deep dive
- [x] Scale and complexity assessed (8+ device types, 2 optimization engines)
- [x] Technical constraints identified (HA platform, physical circuits, async discipline)
- [x] Cross-cutting concerns mapped (7 concerns with interaction points)
- [x] Decision Map created for agent guidance

**✅ Technology Stack & CI/CD (Step 3)**
- [x] Technology stack fully specified with version pinning concerns
- [x] Dual test infrastructure documented (FakeHass + real HA fixtures)
- [x] Risk-weighted CI strategy mapping change areas to blast radius
- [x] 3-tier CI/CD pipeline architecture with 3 automation workflows
- [x] Test suite inventory with gap analysis

**✅ Architectural Decisions (Step 4)**
- [x] 3 critical decisions with rationale and impact analysis
- [x] 2 deferred decisions with preparation notes
- [x] 7 "Already Decided" items documented
- [x] Implementation sequence with rationale
- [x] Cross-component dependency analysis

**✅ Implementation Patterns (Step 5)**
- [x] 9 patterns covering all extension types
- [x] Anti-patterns for each pattern
- [x] Enforcement guidelines with 7 mandatory rules
- [x] Relationship to project-context.md clarified

**✅ Project Structure (Step 6)**
- [x] Complete directory tree with annotations
- [x] Deployable code vs development infrastructure separation
- [x] Architectural boundaries (strict, strategic/tactical, component communication)
- [x] External integration points with failure modes
- [x] Near-term work → structure mapping
- [x] Test file organization with protected files
- [x] Quick Navigation for agent file discovery

**✅ Architecture Validation (Step 7)**
- [x] Coherence validation passed
- [x] Requirements coverage verified against product brief
- [x] Implementation readiness confirmed with agent navigation tests
- [x] Gap analysis complete — no critical gaps, important gaps have commitments

### Architecture Readiness Assessment

**Overall Status: READY FOR IMPLEMENTATION**

**Confidence Level: HIGH**

The architecture document covers a mature, production brownfield codebase with established patterns. Most decisions were "already decided" by the existing code — this document makes them explicit and navigable for AI agents.

**Key Strengths:**
- Two-level hierarchy (Decision Map + Quick Navigation) provides both understanding and file-level navigation
- Risk-weighted CI strategy connects code changes to their real-world impact
- Trust-critical components (charger budgeting, constraint interactions) are explicitly flagged with testing commitments
- 9 implementation patterns with anti-patterns cover all common extension types
- Protected infrastructure files identified to prevent silent test regression

**Areas for Future Enhancement:**
- Solver quality benchmarks (when refactoring begins)
- Dynamic tariff provider abstraction (when tariff story starts)
- Data/learning architecture (when prediction improvement stories start)
- JS card quality pipeline (when frontend stories start)
- Consolidate charger test files by test type (scenarios, unit, integration) when charger test coverage matures

### Implementation Handoff

**AI Agent Guidelines:**
1. Read `_bmad-output/project-context.md` (42 code-level rules) AND this architecture document before implementing any code
2. Use the Quick Navigation table (step 6) to find your files
3. Use the Decision Map (step 2) to understand what you need to know
4. Check the Risk-Weighted CI table (step 3) to understand your blast radius
5. Follow the correct implementation pattern (step 5) for your change type
6. Choose the correct test layer (Pattern 9) before writing tests
7. Never bypass budget validation, rate limiting, or the domain/HA boundary

**First Implementation Priority:**
CI/CD pipeline (`.github/` directory) — enables all subsequent work through the structured workflow. After CI/CD, the first *code* story is the person-car assignment bug fix — see implementation sequence in the Decision Impact Analysis section.
