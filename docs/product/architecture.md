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

> **See also**: [docs/agents/index.md](../agents/index.md) — the
> addressable documentation hierarchy for plan / implement / review
> agents. Per-concept files live under `docs/agents/concepts/`,
> `docs/agents/principles/`, `docs/agents/use-cases/`, and
> `docs/agents/personas/`. This architecture document keeps the
> **decisions log**, **CI/CD strategy**, **validation tables**, and
> **architectural boundaries** — everything *not* covered by the
> per-concept docs.
>
> QS-185 trimmed sections describing concepts, hierarchical control
> mechanics, user interaction architecture, and the 9 implementation
> patterns. Where content was relocated, this document now points to
> the new home rather than duplicating it.

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

The per-concept descriptions live under
[docs/agents/concepts/](../agents/concepts/). Use the
[docs/agents/index.md](../agents/index.md) "Lookup by source file"
table to find the right doc.

- Domain layer (`home_model/`) — see
  [commands.md](../agents/concepts/commands.md),
  [constraints.md](../agents/concepts/constraints.md),
  [solver.md](../agents/concepts/solver.md),
  [load-base.md](../agents/concepts/load-base.md),
  [home-model-battery.md](../agents/concepts/home-model-battery.md).
- HA integration layer (`ha_model/`) — see
  [ha-device-mixin.md](../agents/concepts/ha-device-mixin.md),
  [qs-home-orchestrator.md](../agents/concepts/qs-home-orchestrator.md),
  [charger-budgeting.md](../agents/concepts/charger-budgeting.md),
  [dynamic-group-tree.md](../agents/concepts/dynamic-group-tree.md),
  [ha-battery.md](../agents/concepts/ha-battery.md),
  [solar-providers.md](../agents/concepts/solar-providers.md),
  [person-trip-prediction.md](../agents/concepts/person-trip-prediction.md),
  [bistate-duration-devices.md](../agents/concepts/bistate-duration-devices.md),
  [piloted-device-and-heat-pump.md](../agents/concepts/piloted-device-and-heat-pump.md),
  [off-grid-mode.md](../agents/concepts/off-grid-mode.md),
  [external-control-detection.md](../agents/concepts/external-control-detection.md),
  [notification-routing.md](../agents/concepts/notification-routing.md),
  [user-override.md](../agents/concepts/user-override.md).
- UI & configuration layer — see
  [config-and-setup-flow.md](../agents/concepts/config-and-setup-flow.md).
- Testing layers — see
  [testing-layers.md](../agents/concepts/testing-layers.md).

### Hierarchical Control Architecture

The strategic (PeriodSolver) vs tactical (charger dynamic budgeting)
split is the central control architecture. Full description in
[docs/agents/principles/strategic-tactical-control.md](../agents/principles/strategic-tactical-control.md).
Charger-budgeting detail (algorithm, phase switching, staged
transitions, adaptation windows) lives in
[docs/agents/concepts/charger-budgeting.md](../agents/concepts/charger-budgeting.md);
the tree topology in
[docs/agents/concepts/dynamic-group-tree.md](../agents/concepts/dynamic-group-tree.md).

### Key Architectural Patterns

The reusable architectural patterns are now formalised as
**principles** under
[docs/agents/principles/](../agents/principles/):

- [observe-predict-optimize.md](../agents/principles/observe-predict-optimize.md)
- [two-layer-boundary.md](../agents/principles/two-layer-boundary.md)
- [hysteresis-and-switching-cost.md](../agents/principles/hysteresis-and-switching-cost.md)
- [strategic-tactical-control.md](../agents/principles/strategic-tactical-control.md)
- [event-driven-with-fallback.md](../agents/principles/event-driven-with-fallback.md)

### Key Data Flows

**Runtime solve cycle:**
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

**Charger dynamic budgeting cycle** (within update_loads):
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

**Device configuration flow:**
```
HA UI -> config_flow.py step -> ConfigEntry created
-> __init__.py async_setup_entry() -> QSDataHandler.async_add_entry()
-> create_device_from_type() -> device constructor with **config_entry.data
-> home.add_device() -> device.get_platforms()
-> HA platform setup -> create_ha_<platform>(device) -> entities registered
```

### User Interaction Architecture

Persona-by-persona interaction paths are extracted to
[docs/agents/personas/](../agents/personas/) (TheAdmin, TheDev,
Magali). The canonical end-to-end magic-moment scenario lives in
[docs/agents/use-cases/magali-plugs-in-car.md](../agents/use-cases/magali-plugs-in-car.md).

**Architectural implication**: An agent implementing a notification
or override story must trace through the full path: person
prediction → constraint metadata → solver → charger budgeting →
notification. These are not isolated components. An agent
implementing developer workflow improvements must understand the
dual test infrastructure (FakeHass vs real HA fixtures) and the
risk-weighted CI strategy.

### Technical Constraints & Dependencies

- **Home Assistant platform**: All HA API access confined to ha_model/ layer. Platform dictates entity lifecycle, config flow patterns, service call mechanisms
- **Solar forecast dependency**: External API (Solcast/OpenMeteo) for production forecasts — must handle unavailability gracefully
- **Device protocol diversity**: OCPP, Wallbox, Generic charger protocols each with different control capabilities and state reporting
- **Python 3.14+**: Uses latest language features (pattern matching, type hints, dataclasses)
- **Real-time state tracking**: HADeviceMixin bridges HA entity state changes into domain model — must be reliable and low-latency
- **Rate-limited actuation**: Max 1 command per load per solve cycle, charger adaptation window of 45s
- **Physical circuit constraints**: Amp budgets are not just optimization targets — exceeding them trips breakers. The dynamic group tree enforces hard physical limits.

### Cross-Cutting Concerns

Most cross-cutting concerns are now expressed as **principles**
under [docs/agents/principles/](../agents/principles/). The
remaining architecture-specific concerns:

- **State consistency**: Three async cycles (4s/7s/30s) share device
  state via asyncio.Lock guards. See
  [qs-home-orchestrator.md](../agents/concepts/qs-home-orchestrator.md).
- **Device lifecycle management**: Adding a new device type touches
  6+ files. See pattern reference below + the 7-file checklist in
  [docs/agents/concepts/config-and-setup-flow.md](../agents/concepts/config-and-setup-flow.md).
- **Async discipline**: Event loop safety across all layers — see
  the relevant code-level rules in
  [docs/workflow/project-context.md](../workflow/project-context.md).
- **Testing infrastructure**: Domain factories for real objects (not
  mocks), FakeHass for HA layer — see
  [testing-layers.md](../agents/concepts/testing-layers.md).

### Decision Map for AI Agents

The full Decision Map is now hosted in the
[docs/agents/index.md](../agents/index.md) "Lookup by source file"
table. Use it as the entry point.

### Architectural Gaps (Prioritized by Risk)

**1. Data/Learning Architecture — FUTURE SCOPE**
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
- **Dual test infrastructure**: see
  [testing-layers.md](../agents/concepts/testing-layers.md) for the
  full FakeHass vs real-HA layer rules.

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

## Checklist

(Pre-implementation checklist for AI agents — kept in place; the
content has been moved into the validation tables below.)

## Risk Assessment

See the Risk-Weighted CI Strategy table above.

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
2. **Bug fix: person-car assignment** — first story through the pipeline, validates the development workflow end-to-end with a contained, lower-risk story
3. **Trust-critical component tests** — charger scenarios, constraint interactions — fills highest-risk test gaps, now proven through a validated pipeline
4. **Resilience strategy implementation** — explicit fallback for each dependency
5. **Solver optimization** — edge case improvements within stable interface

**Rationale for sequence**: The bug fix (step 2) comes before trust-critical tests (step 3) because the product brief explicitly says "assignment bug fix is the first feature delivered through the structured workflow — proving the process works." Validate the workflow with a simpler story before tackling complex test scenarios through it.

**Cross-Component Dependencies:**
- Resilience strategy touches: solar.py (forecast fallback), charger.py (communication retry), home.py (solver infeasibility, numpy persistence), device.py (state unavailability), person.py (prediction confidence)
- Trust-critical tests touch: charger.py, dynamic_group.py, constraints.py, solver.py, conftest.py (infrastructure smoke tests)
- Solver optimization touches: solver.py, constraints.py only (stable interface protects other components)
- Both testing and resilience decisions feed into the CI/CD risk-weighted strategy from step 3

**Documentation note**: This architecture document partially satisfies product brief success criterion #5 ("documentation started"). Combined with the project-context.md and the new `docs/agents/` hierarchy, the foundational documentation is in place.

## Implementation Patterns & Consistency Rules

### Relationship to project-context.md and docs/agents/

[docs/workflow/project-context.md](../workflow/project-context.md)
contains 42 rules covering code-level conventions (naming, async,
logging, error handling, testing anti-patterns).
[docs/agents/principles/](../agents/principles/) covers the
**architectural-level patterns** — how to extend and modify the
system without breaking its design. AI agents MUST read both
project-context and the relevant concept/principle docs before
implementing any code.

The 9 extension patterns previously described here have been moved
to the addressable docs:

| Pattern | New home |
|---|---|
| Pattern 1 — Adding a New Device Type | [config-and-setup-flow.md](../agents/concepts/config-and-setup-flow.md) (6-file checklist) + [load-base.md](../agents/concepts/load-base.md) (`AbstractLoad` contract) |
| Pattern 2 — Creating and Managing Constraints | [constraints.md](../agents/concepts/constraints.md) |
| Pattern 3 — Command Execution & Verification | [commands.md](../agents/concepts/commands.md) + [ha-device-mixin.md](../agents/concepts/ha-device-mixin.md) |
| Pattern 4 — State Tracking via HADeviceMixin | [ha-device-mixin.md](../agents/concepts/ha-device-mixin.md) |
| Pattern 5 — Charger Budgeting Interaction | [charger-budgeting.md](../agents/concepts/charger-budgeting.md) + [dynamic-group-tree.md](../agents/concepts/dynamic-group-tree.md) |
| Pattern 6 — Solver Extension | [solver.md](../agents/concepts/solver.md) + [strategic-tactical-control.md](../agents/principles/strategic-tactical-control.md) |
| Pattern 7 — Notification Triggers | [notification-routing.md](../agents/concepts/notification-routing.md) |
| Pattern 8 — Config Entry Persistence | [config-and-setup-flow.md](../agents/concepts/config-and-setup-flow.md) |
| Pattern 9 — Test Layer Selection | [testing-layers.md](../agents/concepts/testing-layers.md) |

### Enforcement Guidelines

**All AI Agents MUST:**
1. Read [docs/workflow/project-context.md](../workflow/project-context.md) (code-level rules) AND the relevant `docs/agents/` files (architectural patterns) before implementing
2. Use the [docs/agents/index.md](../agents/index.md) "Lookup by source file" table to identify which concept docs apply to your change
3. Check the Risk-Weighted CI table (step 3) to understand the blast radius of their changes
4. Use the correct extension pattern for the type of change (device, constraint, command, state, notification)
5. Never bypass budget validation, rate limiting, or the domain/HA boundary
6. Choose the correct test layer ([testing-layers.md](../agents/concepts/testing-layers.md)) before writing any tests
7. Use `save_entry_data_no_reload()` for runtime persistence, never `async_update_entry()` (see [config-and-setup-flow.md](../agents/concepts/config-and-setup-flow.md))

**Pattern Violations:**
- Ruff and MyPy catch code-level violations automatically
- Architectural violations (wrong layer, missing load_info, bypassed budget validation) must be caught in code review
- CI risk assessment comment (auto-review workflow) flags changes to trust-critical components
- The drift checker (`scripts/qs/check_doc_drift.py`) catches docs falling out of sync — see
  [docs/workflow/project-rules.md](../workflow/project-rules.md) "Doc maintenance".

## Project Structure & Boundaries

### Quick Navigation for AI Agents

The Quick Navigation table has moved to
[docs/agents/index.md](../agents/index.md). Use it to find concept
docs by source file; come back here for the architectural
boundaries and the directory tree.

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

**CLAUDE.md role**: CLAUDE.md is the Claude Code bootstrap file (auto-loaded at conversation start). It should contain only operational essentials (venv, commands) and point to `docs/workflow/project-context.md`, this document, and `docs/agents/index.md` for the full rules. Do not duplicate content between CLAUDE.md and the project context files.

#### Runtime Data

```
{HA config dir}/quiet_solar/          # Numpy ring buffer storage (.npy files)
{HA config dir}/quiet_solar/debug/    # Debug dumps (pickle, forecast data)
```

Runtime data at `{HA config dir}/quiet_solar/` is distinct from the component source at `{HA config dir}/custom_components/quiet_solar/`. The former contains `.npy` data files; the latter contains Python source.

#### Development Infrastructure

```
quiet-solar/
├── CLAUDE.md                          # AI agent bootstrap (points to project context + docs/agents/)
├── requirements_test.txt              # Test dependencies
├── pytest.ini                         # Pytest configuration (asyncio_mode=auto, pythonpath=., markers)
├── setenv.sh                          # Development environment setup (venv + pip install)
│
├── docs/                              # Documentation hierarchy
│   ├── product/                       # product-brief.md, this document
│   ├── workflow/                      # process / pipeline rules
│   └── agents/                        # AI-agent addressable concept hierarchy (QS-185)
│
├── .github/                           # [PLANNED — CI/CD story]
│
├── tests/                             # === TEST SUITE (~3,800+ tests) ===
│   ├── conftest.py                    # ⚠ PROTECTED: FakeHass, FakeConfigEntry, FakeState
│   ├── factories.py                   # ⚠ PROTECTED: factory functions
│   ├── ha_tests/                      # === HA integration layer (real HA fixtures) ===
│   │   ├── conftest.py                # ⚠ PROTECTED
│   │   └── const.py                   # ⚠ PROTECTED
│   ├── qs/                            # tests for dev-tooling (scripts/qs/, agents)
│   └── utils/                         # test helpers (energy_validation, scenario_builders)
│
├── scripts/qs/                        # dev-pipeline scripts (quality_gate, context, ...)
├── .claude/, .cursor/, .opencode/     # per-harness agent definitions + slash commands
└── legacy/                            # === FROZEN HISTORICAL CODE ===
```

**Build artifacts** (`coverage.xml`, `*.pyc`, `__pycache__/`) live at project root and should be gitignored. Do not commit build artifacts.

### Architectural Boundaries

**Strict Boundaries (violations are critical errors):**

| Boundary | Rule | Enforcement |
|---|---|---|
| Domain/HA | `home_model/` NEVER imports `homeassistant.*` | Code review + grep in CI |
| Domain purity | Domain layer receives all HA data as parameters | Code review |
| Constant centralization | All config keys in `const.py`, never hardcoded | Ruff linting |

See [two-layer-boundary.md](../agents/principles/two-layer-boundary.md)
for the full rule and rationale.

**Strategic/Tactical boundary:**

| Layer | Component | Scope | Override behavior |
|---|---|---|---|
| Strategic | PeriodSolver | 15-min windows, all loads | Plans optimal allocation |
| Tactical | Charger dynamic budgeting | Real-time, chargers only | Can override strategic layer when amp budget constrains |

See [strategic-tactical-control.md](../agents/principles/strategic-tactical-control.md).

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
| CLAUDE.md slim-down | None | `CLAUDE.md` | Reduce to bootstrap: commands + pointers to project context files. Remove duplicated content. |
| Bug fix: person-car assignment | None | `ha_model/person.py`, `ha_model/car.py`, `ha_model/home.py`, tests | First story through CI pipeline. Contained changes. |
| Trust-critical component tests | New test file(s) for scenario-based charger integration tests — exact structure TBD during test design story. | `tests/conftest.py` (new fixtures) | Add infrastructure smoke tests alongside. |
| Resilience strategy | None | `ha_model/solar.py`, `ha_model/charger.py`, `ha_model/home.py`, `ha_model/device.py`, `ha_model/person.py` | Touches many files but each change is contained (fallback logic per dependency). |
| Solver optimization | None | `home_model/solver.py`, `home_model/constraints.py` | Contained within stable interface (Decision 3). |

### Test File Organization

**File selection rule for agents**: When adding tests, extend existing files by test type rather than creating new files. If a file exceeds ~500 tests, split by sub-concern. Do not create `test_*_additional_coverage.py` or `test_*_deep.py` files.

**Root `tests/` naming clarification**: Files prefixed `test_ha_*` in `tests/` root use **FakeHass infrastructure** (lightweight, fast). Files in `tests/ha_tests/` use **real HA fixtures**. When determining which infrastructure a test uses, **check the imports**, not the file name. See [testing-layers.md](../agents/concepts/testing-layers.md).

**Shared test utilities**: Before writing test helper functions, check `tests/utils/` for existing utilities. `energy_validation.py` provides energy conservation assertion helpers. `scenario_builders.py` provides scenario construction helpers for multi-device tests.

### Protected Infrastructure Files

These files are load-bearing — changes silently affect the behavior of hundreds of tests. They require extra review scrutiny:

| File | Role | Risk if broken |
|---|---|---|
| `tests/conftest.py` | FakeHass, FakeConfigEntry, FakeState, FakeServices | Every domain test may pass while testing the wrong thing |
| `tests/factories.py` | Factory functions for all domain test doubles | Same — silently alters 100+ test files |
| `tests/ha_tests/conftest.py` | Real HA fixtures | Every HA integration test affected |
| `tests/ha_tests/const.py` | Standard mock configurations | HA tests use wrong config silently |

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
- The 9 extension patterns have been relocated to the per-concept docs under [docs/agents/](../agents/) (table above). The patterns themselves remain consistent in their new locations.
- Naming conventions are consistent: `CONF_*` for config keys, `CONF_TYPE_NAME_QS*` for device types, `*_SENSOR` suffixes
- Anti-patterns are documented alongside their patterns in each concept doc

**Structure Alignment:**
- Directory structure supports all 3 architectural layers with clear separation
- Architectural boundaries (this document) + the per-source-file lookup ([docs/agents/index.md](../agents/index.md)) provide both "why" and "where"
- Structural notes explain anomalies (data_handler.py location, thin wrapper devices) so agents don't try to "fix" what isn't broken
- `[PLANNED]` vs `[FUTURE]` markers distinguish committed work from speculative preparation

### Requirements Coverage Validation ✅

**Product Brief Feature Coverage:**

| Product brief requirement | Architectural support | Covered in |
|---|---|---|
| Whole-house energy orchestration | PeriodSolver + constraint system + 8+ device types | [solver.md](../agents/concepts/solver.md), [constraints.md](../agents/concepts/constraints.md) |
| Constraint-based solver (4 priority tiers) | PeriodSolver, LoadConstraint hierarchy | [solver.md](../agents/concepts/solver.md), [constraints.md](../agents/concepts/constraints.md) |
| People-aware automation | Person model, trip prediction, person-car allocation | [person-trip-prediction.md](../agents/concepts/person-trip-prediction.md) |
| Smart device handling (external control) | `external_user_initiated_state` detection | [external-control-detection.md](../agents/concepts/external-control-detection.md) |
| Off-grid resilience | Grid outage row in resilience table | Decision 1 + [off-grid-mode.md](../agents/concepts/off-grid-mode.md) |
| Tariff-aware scheduling | Solver accepts tariff tuples; dynamic tariffs deferred | Decision 3 + [cheap-grid-charging.md](../agents/use-cases/cheap-grid-charging.md) |
| UI dashboards | Jinja2 + 4 custom JS cards + programmatic HA API | This document (UI layer) |
| Notification & override flows | `on_device_state_change()`, per-person routing, dedup | [notification-routing.md](../agents/concepts/notification-routing.md), [user-override.md](../agents/concepts/user-override.md) |
| Multi-protocol charger support | OCPP/Wallbox/Generic variants | [charger-budgeting.md](../agents/concepts/charger-budgeting.md) |
| Real-time charger budgeting | Tactical layer, QSDynamicGroup tree, staged transitions | [strategic-tactical-control.md](../agents/principles/strategic-tactical-control.md) |
| 100% test coverage | CI pipeline threshold = 100%, mandatory | CI/CD Pipeline Architecture |
| HACS distribution | manifest.json, hacs.json, HACS validation in CI | Technology Stack + CI |

**Near-Term Success Criteria (from product brief):**

| Criterion | Architectural support | Status |
|---|---|---|
| CI/CD operational | Full 3-tier pipeline + 3 automation workflows designed | ✅ Ready to implement |
| Development workflow validated | Bug fix as first story through pipeline | ✅ Sequenced as step 2 |
| Solver improved | Stable interface boundary (Decision 3) | ✅ Boundary defined |
| 100% coverage maintained | CI enforces, risk-weighted strategy prioritizes | ✅ Pipeline designed |
| Documentation started | Architecture doc + project-context.md + docs/agents/ hierarchy | ✅ This document + QS-185 |
| Ready for structured feature work | Patterns relocated to concept docs + structure mapping + drift checker | ✅ Complete |

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
- Near-term work → structure mapping ✅
- Protected infrastructure files identified ✅

**Pattern Completeness:**
- 9 patterns relocated to [docs/agents/](../agents/) with cross-references in this document ✅
- Every pattern has an anti-pattern in its new home ✅
- Lookup-by-source-file table provides component-level guidance ([docs/agents/index.md](../agents/index.md)) ✅
- Enforcement guidelines preserved ✅

**Agent Navigation Test** — can an agent find what it needs?
- "I need to add a new device" → [docs/agents/index.md](../agents/index.md) → concept docs for relevant device-base patterns + [config-and-setup-flow.md](../agents/concepts/config-and-setup-flow.md) ✅
- "I need to fix a charger bug" → [docs/agents/index.md](../agents/index.md) → [charger-budgeting.md](../agents/concepts/charger-budgeting.md) + risk table in this doc (CRITICAL, physical) ✅
- "I need to set up CI/CD" → CI/CD Pipeline Architecture (this doc) → risk-weighted strategy → Near-Term Work mapping ✅
- "I need to understand why the solver made this decision" → [solver.md](../agents/concepts/solver.md) → Data Flows (this doc) ✅
- "I need to add a notification for a new event" → [notification-routing.md](../agents/concepts/notification-routing.md) + [magali-plugs-in-car.md](../agents/use-cases/magali-plugs-in-car.md) ✅

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
- [x] Cross-cutting concerns mapped (now expressed as principles under [docs/agents/principles/](../agents/principles/))
- [x] Decision Map relocated to [docs/agents/index.md](../agents/index.md)

**✅ Technology Stack & CI/CD (Step 3)**
- [x] Technology stack fully specified with version pinning concerns
- [x] Dual test infrastructure documented (pointer to [testing-layers.md](../agents/concepts/testing-layers.md))
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
- [x] 9 patterns covering all extension types (relocated to [docs/agents/](../agents/))
- [x] Anti-patterns documented in each concept doc
- [x] Enforcement guidelines preserved here

**✅ Project Structure (Step 6)**
- [x] Complete directory tree with annotations
- [x] Deployable code vs development infrastructure separation
- [x] Architectural boundaries (strict, strategic/tactical, component communication)
- [x] External integration points with failure modes
- [x] Near-term work → structure mapping
- [x] Test file organization with protected files
- [x] Quick Navigation moved to [docs/agents/index.md](../agents/index.md)

**✅ Architecture Validation (Step 7)**
- [x] Coherence validation passed
- [x] Requirements coverage verified against product brief
- [x] Implementation readiness confirmed with agent navigation tests
- [x] Gap analysis complete — no critical gaps, important gaps have commitments

### Architecture Readiness Assessment

**Overall Status: READY FOR IMPLEMENTATION**

**Confidence Level: HIGH**

The architecture document covers a mature, production brownfield codebase with established patterns. Most decisions were "already decided" by the existing code — this document makes them explicit and navigable for AI agents. As of QS-185, per-concept documentation lives in [docs/agents/](../agents/); this document is the decisions / CI / structure layer above it.

**Key Strengths:**
- Two-level hierarchy ([docs/agents/index.md](../agents/index.md) Decision Map + Quick Navigation) provides both understanding and file-level navigation
- Risk-weighted CI strategy connects code changes to their real-world impact
- Trust-critical components (charger budgeting, constraint interactions) are explicitly flagged with testing commitments
- 9 implementation patterns relocated to per-concept docs — agents pull just the pattern they need
- Protected infrastructure files identified to prevent silent test regression
- Drift checker (`scripts/qs/check_doc_drift.py`) catches docs falling out of sync with their `covers:` source

**Areas for Future Enhancement:**
- Solver quality benchmarks (when refactoring begins)
- Dynamic tariff provider abstraction (when tariff story starts)
- Data/learning architecture (when prediction improvement stories start)
- JS card quality pipeline (when frontend stories start)
- Consolidate charger test files by test type (scenarios, unit, integration) when charger test coverage matures

### Implementation Handoff

**AI Agent Guidelines:**
1. Read [docs/workflow/project-context.md](../workflow/project-context.md) (42 code-level rules), the relevant `docs/agents/concepts/` files for the components you're touching, and this architecture document for decisions / CI / structure context
2. Use [docs/agents/index.md](../agents/index.md) "Lookup by source file" to find concept docs from a source path
3. Check the Risk-Weighted CI table (Technology Stack section) to understand your blast radius
4. Follow the correct extension pattern via the "Pattern N → new home" table in the Implementation Patterns section
5. Choose the correct test layer ([testing-layers.md](../agents/concepts/testing-layers.md)) before writing tests
6. Never bypass budget validation, rate limiting, or the domain/HA boundary
7. Run `python scripts/qs/check_doc_drift.py` before staging — see [docs/workflow/project-rules.md](../workflow/project-rules.md) "Doc maintenance"

**First Implementation Priority:**
CI/CD pipeline (`.github/` directory) — enables all subsequent work through the structured workflow. After CI/CD, the first *code* story is the person-car assignment bug fix — see implementation sequence in the Decision Impact Analysis section.
