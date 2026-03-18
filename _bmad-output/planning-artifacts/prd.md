---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-02b-vision', 'step-02c-executive-summary', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type', 'step-08-scoping', 'step-09-functional', 'step-10-nonfunctional', 'step-11-polish', 'step-e-01-discovery', 'step-e-02-review', 'step-e-03-edit']
lastEdited: '2026-03-18'
editHistory:
  - date: '2026-03-18'
    changes: 'Post-validation refinement: tightened 8 flagged FRs, added measurable targets to 9 NFRs, added FR5b Optimization Hierarchy'
inputDocuments:
  - product-brief-quiet-solar-2026-03-17.md
  - project-context.md
  - architecture.md
documentCounts:
  briefs: 1
  research: 0
  brainstorming: 0
  projectDocs: 2
classification:
  projectType: 'Intelligent Energy Optimization Platform (Home Assistant Integration)'
  domain: 'Residential Energy Management'
  complexity: 'High — real-time optimization under uncertainty, multi-device orchestration, behavioral prediction, platform constraints'
  projectContext: 'brownfield'
workflowType: 'prd'
project_name: 'quiet-solar'
user_name: 'Thomas'
date: '2026-03-18'
---

# Product Requirements Document - quiet-solar

**Author:** Thomas
**Date:** 2026-03-18

## Executive Summary

Quiet Solar is an intelligent energy optimization platform built as a Home Assistant custom component. It orchestrates all controllable loads in a household — EV chargers, battery storage, HVAC, pool heating, water boilers, and more — through a constraint-based solver that continuously allocates power based on solar forecasts, energy tariffs, and predicted household needs. The system serves three distinct user types: TheAdmin (Home Energy Manager) who configures, monitors, and fine-tunes the system, Household Members (Magali) who benefit from invisible automation with minimal interaction, and TheDev (Developer) who builds, maintains, and evolves the system with delight. The core problem: homeowners with solar installations waste significant energy because their devices operate independently, unaware of each other, of solar production, or of the household's actual needs. Quiet Solar treats the home as a single energy system and optimizes it holistically.

### What Makes This Special

Quiet Solar's north star is that **the household doesn't have to think about energy**. Where competitors like EVCC and Emhass require homeowners to think like energy engineers and optimize individual devices, Quiet Solar optimizes for the household — predicting behavior, minimizing user input, and making decisions that serve real people with real schedules. The product rests on three pillars:

1. **Invisible optimization** — the system makes the right energy allocation decision without being asked, across all devices simultaneously
2. **People-first intelligence** — it learns household patterns (trips, presence, routines) and asks less over time, reducing mental load for every family member
3. **Trustworthy resilience** — when external services fail (solar forecast APIs, EV vendor APIs, charger protocols), when the grid goes down, or when predictions miss, the system degrades gracefully, communicates honestly, and recovers automatically

The founding proof: during a real grid failure, the household was notified and the system handled everything seamlessly — the family barely noticed. That moment — invisible protection under crisis — defines the product promise.

## Project Classification

- **Project type:** Intelligent Energy Optimization Platform (Home Assistant Integration)
- **Domain:** Residential Energy Management
- **Complexity:** High — real-time optimization under uncertainty, multi-device orchestration, behavioral prediction, platform constraints
- **Project context:** Brownfield — production system with 100% mandatory test coverage, serving a real household, under active development. The solidification phase focuses on CI/CD automation, bug fixes, solver improvements, failure mode resilience, expanded test scenarios (happy path, degraded state, crisis), and community readiness through documentation and guided onboarding. Future vision includes proactive recommendations, gamification, dynamic tariffs, and broader community adoption.

## Success Criteria

### User Success

**TheAdmin (Home Energy Manager):**
- Solver edge cases that currently require manual awareness are handled automatically
- When an external service fails, TheAdmin gets a clear notification explaining what happened and what the system is doing about it — no log diving required
- Self-consumption rate shows measurable improvement over time (tracked via dashboard)
- Counterfactual savings ("what would this have cost without quiet-solar?") are visible and growing

**TheDev (Developer):**
- Development workflow is a delight: zero manual GitHub operations — linting, testing, coverage checks, releases, and release notes are fully automated
- Validated when: the first bug fix through the new CI/CD pipeline feels smooth and enjoyable end-to-end
- Working on the project is fun, educational, and energizing — TheDev is eager to maintain and evolve the codebase
- Test infrastructure is rich and expressive — writing tests is satisfying, not tedious

**Magali (Household Member):**
- Override frequency decreases over time — the system predicts better as it learns
- Notifications are helpful, not noisy — dismissed notifications signal a quality problem
- The "I didn't have to think about it" experience remains the norm, not the exception

**Future TheAdmin-like user (Community):**
- A technically capable HA user can discover, install, and configure quiet-solar for their home using documentation alone
- They can then contribute back to the project through a clear contribution path

### Business Success

- **Open-source project health:** CI/CD pipeline operational, structured development workflow validated, PRs follow a disciplined plan-build-test-ship cycle
- **Quality bar maintained:** 100% test coverage — mandatory, non-negotiable, enforced automatically
- **Community readiness:** At least one non-author user can install and configure the system independently
- **Solver maturity:** Edge case handling improved, failure modes analyzed and addressed systematically

### Technical Success

- **CI/CD:** Every PR runs full test suite, coverage gated at 100%, releases automated with generated release notes. Zero manual GitHub operations for standard workflows.
- **Failure resilience:** Failure mode catalog completed — a table documenting every external dependency weakness with columns: dependency, failure mode, detection method, system response, recovery behavior, user impact, test tier. Each row has an implemented response and passing tests at the appropriate tier.
- **Expanded test scenarios:** Charger scenarios (multi-car allocation, concurrent charging, protocol-specific behavior) and solver scenarios (competing priorities, prediction mismatches, complex orchestration) validating core capabilities independent of failure modes.
- **Solver improvements:** Measurable improvement in edge case handling, validated through new test scenarios

### Measurable Outcomes

| Metric | Target | Measurement |
|--------|--------|-------------|
| CI/CD manual operations | Zero | No manual GitHub ops for lint/test/release |
| Test coverage | 100% | Enforced by CI gate |
| Failure modes analyzed | All identified | Failure mode catalog complete |
| Failure modes addressed | All planned | Each catalog row has implemented response |
| Failure mode tests | All passing | Each catalog row has test at appropriate tier |
| First bug fix through pipeline | Smooth, delightful | TheDev's subjective validation |
| External user install | 1+ successful | TheAdmin-like user completes setup from docs |
| Self-consumption rate | Improving trend | Dashboard tracking over weeks |

## Product Scope

The product is already in production. Development is organized into four tiers — detailed breakdown, cut-line strategy, and risk analysis in [Project Scoping & Phased Development](#project-scoping--phased-development).

| Tier | Focus | Status |
|------|-------|--------|
| **MVP — Solidification** | CI/CD, bug fixes, solver improvements, expanded test scenarios, failure mode resilience | Active |
| **Product Enhancement** | Onboarding UX, prediction accuracy, notification/override UX, new features | Post-MVP, independent |
| **Community Readiness** | Documentation, setup guide, contribution path | Post-MVP, independent |
| **Vision** | Proactive recommendations, gamification, dynamic tariffs, broader user base | Future |

## User Journeys

### TheDev — The Developer

**Persona:** TheDev is a developer who finds joy in solving real family problems through code. He's drawn to new technologies (including agentic coding), novel algorithmic approaches, and the challenge of modeling human/system interactions. He's allergic to bureaucracy, manual processes, and fighting tooling to get things done. TheDev and TheAdmin are often the same person wearing different hats — the boundary is porous, and that's by design.

**Journey 1: The Delightful Bug Fix (MVP validation journey)**

TheDev notices a bug in the person-car assignment flow — Magali's car wasn't assigned correctly when TheAdmin changed the schedule. He opens the project, creates an issue, and starts a branch. The test infrastructure is rich: he can write a scenario test that reproduces the exact multi-car, multi-person situation. He fixes the domain logic, runs the full suite locally — 100% coverage maintained, all green. He pushes. The CI pipeline picks it up automatically: linting, type checking, full test suite, coverage gate. Everything passes. He merges. The release is cut automatically with generated release notes. He pulls the update into his Home Assistant instance, verifies the fix live. Start to finish: smooth, fast, satisfying. *That's the aha moment — "I fixed a real problem for my family and the tooling made it a joy."*

**Journey 2: The Solver Exploration**

TheDev has an idea for improving how the solver handles competing priorities on cloudy days. He starts by writing new solver test scenarios that capture the edge case — multiple cars needing charge, limited solar, shifting forecasts. The tests describe the *desired* behavior before he writes a line of solver code. He experiments with the algorithm, runs the scenarios, sees the solver making better decisions. He can spin up a test home environment to see how it plays out with realistic device configurations. The new tests become permanent fixtures — they validate the improvement and prevent regression. He learns something new about optimization under uncertainty, and the codebase is better for it.

**Journey 3: The Failure Mode Investigation**

TheDev decides to tackle the solar forecast API reliability issue. He consults the failure mode catalog, adds a new row: "Solcast API returns 503 for 2+ hours." He designs the system response: fall back to last-known-good forecast, notify TheAdmin, degrade solver confidence for affected time windows, auto-recover when the API returns. He writes a degraded-state test that simulates the outage, verifies the fallback, checks the notification fires. The catalog row is complete: dependency, failure mode, detection, response, recovery, user impact (TheAdmin sees degraded forecast warning, Magali unaffected unless charging plan changes significantly), test tier. Clean, systematic, satisfying.

**Journey 4: The Scenario Craftsman**

TheDev wants to improve the charger test scenarios — the current tests cover basic allocation but not the messy reality of three cars, two chargers, one on OCPP and one on Wallbox, with a cloudy afternoon and a guest EV arriving mid-charge. He builds a rich simulation scenario using the test infrastructure: realistic device configurations, time-varying solar profiles, dynamic household events. The scenario runs, reveals a subtle budget allocation edge case the solver handles suboptimally. He iterates — refining the scenario, adjusting the solver, running again. The test suite grows more expressive and the solver gets smarter. *The joy: the simulation IS the playground. Building better tests IS building a better product.*

**Journey 5: First Day Back**

TheDev returns to the project after six weeks away. He pulls the latest, runs the test suite — one command, all green, 100% coverage. The test names read like documentation: `test_solver_competing_priorities_cloudy_day`, `test_charger_multi_car_budget_split`, `test_forecast_api_down_graceful_fallback`. He scans the failure mode catalog — two new rows added since he was last here, both with passing tests. He picks up an open issue, reads the related test scenarios, and understands the context without digging through git history. The codebase welcomed him back. *The aha: "I can be productive again in 15 minutes, not 2 hours."*

**Journey 6: The Broken Build**

TheDev pushes a solver optimization that looks clean locally but breaks a charger budget allocation scenario in CI. The pipeline catches it: red build, clear failure message pointing to `test_charger_three_cars_limited_capacity` with a diff showing expected vs. actual power allocation. TheDev sees exactly what his change broke and why. He fixes the edge case, pushes again — green. The bad code never reached production, no manual review needed to catch it, no rollback required. *The CI safety net doing its job — catching mistakes before they become problems.*

**Journey 7: The Feature Explorer**

TheDev has an idea — what if the system could suggest optimal appliance timing based on solar surplus? He doesn't jump to code. He starts with research: reads about similar approaches, sketches the interaction model, considers how it fits the three pillars. He opens a discussion (issue, doc, or conversation with an agentic coding partner) — bouncing ideas back and forth, challenging assumptions, exploring edge cases. "What if the suggestion comes too late? What if the user ignores it? What if it conflicts with an existing plan?" The idea gets refined through iteration before a single line of production code is written. When he finally implements, the design is solid because the thinking was thorough. *The aha: "The best features come from playing with ideas before committing to code."*

**Journey 8: The Agentic Pairing Session**

TheDev sits down with a complex problem — improving trip prediction when household patterns shift (school holidays, new job, changed routines). He works with an AI coding partner, iterating rapidly: "What if we add a recency bias to the trip history?" — tests written, results reviewed, approach adjusted. "What about a confidence score that drops when predictions miss?" — another iteration, another test scenario. The back-and-forth is fast, exploratory, educational. TheDev learns something new about time-series prediction. The code is better because the conversation challenged his assumptions. *The joy: coding as dialogue, not monologue.*

---

### TheAdmin — The Home Energy Manager

**Persona:** TheAdmin is a homeowner with solar panels, comfortable with Home Assistant — he installs integrations, configures them through the UI, and understands sensors, switches, and entities. He's not a developer in this role. He set up all smart devices in the home and understands household habits, schedules, and energy patterns. He manages dashboards and the HA app for other family members.

**Journey 1: First Setup (Onboarding)**

TheAdmin discovers quiet-solar through HA community forums while searching for solar optimization. He installs via HACS, adds the integration through the HA UI. The setup flow guides him step by step: solar source, battery, chargers, cars, people mappings. At each stage, clear feedback confirms things are connected correctly. He maps Charger A to Car A to Magali, Charger B to Car B to himself. The system auto-generates dashboards — a simple load-control view for Magali's phone, a detailed monitoring view for his own. Within an hour, the solver is running and making its first decisions.

**Journey 2: The Proud Complex Day**

A cloudy morning, two cars needing charge, guests arriving. TheAdmin checks his dashboard — the solver has already adjusted: prioritized Magali's car (earlier departure), shifted pool heating to afternoon when solar improves, pre-heated hot water overnight at off-peak rates. The self-consumption rate stays high despite the challenging conditions. No intervention needed. TheAdmin checks the counterfactual savings — "this day would have cost 40% more without quiet-solar." Pride.

**Journey 3: The Troubleshooting Moment**

TheAdmin wakes up to Magali's car at 60% instead of the promised 80%. He opens the dashboard — the system provides a human-readable explanation: "Solar production was 35% below forecast yesterday. Car B was prioritized due to earlier departure time. An override was available at 8pm but not acted on." Clear, layered, no log diving. He understands what happened and why. He adjusts the priority for tomorrow. Trust reinforced through transparency, not perfection.

**Journey 4: Off-Grid Crisis**

A grid failure at 2am. TheAdmin's phone buzzes: "Grid outage detected. Quiet Solar is managing your home on solar and battery. Non-essential loads paused. Essential loads protected." He goes back to sleep. In the morning, the house is warm, the fridge ran all night, and the battery still has 30%. When the grid returns, another notification: "Grid restored. Resuming normal operation." The family barely noticed. *This is the founding story.*

**Journey 5: The Letting Go**

Three weeks in. TheAdmin realizes he hasn't opened the energy dashboard since Monday. A weekly summary notification arrives: "This week: 94% self-consumption, all charging commitments met, zero overrides, €12.30 saved vs. no optimization." He smiles and closes it. The system is earning its keep — and earning his trust. Over months, his interaction frequency drops from daily to weekly to "only when something changes." The system actively supports this by providing periodic confidence signals that say "everything is fine, you don't need to check." *The best admin experience is no admin experience.*

---

### Magali — The Household Member

**Persona:** Magali lives in the house but has zero interest in Home Assistant internals. She drives one of the EVs, uses the house appliances, and expects things to just work. She interacts with the system rarely — passive 95% of the time.

**Journey 1: The Invisible Charge**

Magali gets home, plugs in her car, goes about her evening. She doesn't set a target, doesn't open an app. The system knows her typical departure time from historical patterns, calculates the charge needed, and schedules it optimally — maximizing solar, filling gaps with off-peak grid. At 6:45am, a notification: "Your car is ready — 80% charged for your 7:15am departure." She didn't have to think about it. *This is the aha moment.*

**Journey 2: The Override**

Magali has an unexpected early meeting — she needs the car at 6am with 90% charge, not the usual 7:15am at 80%. She opens the HA app, taps her car card, sets "90% by 6am." Confirmation in seconds. The solver replans immediately, shifting other loads to accommodate. Done in under 5 seconds. If the override affects other household members, they're notified instantly.

**Journey 3: The Conflict**

TheAdmin changes the charging plan — his car takes priority tonight due to an early trip. Magali gets an immediate notification: "Charging plan changed. Your car is now scheduled for 8am instead of 7:15am — tap to adjust." She discusses with TheAdmin, they agree on a compromise, the system updates both. Clear, instant, no confusion.

**Journey 4: The Visible Override (Magali → TheAdmin)**

Magali overrides her car charging at 10pm — "90% by 6am" instead of the usual 80% by 7:15am. The system replans: it needs to pull from the grid at peak rates to honor the request. TheAdmin gets a notification: "Magali requested priority charging. Estimated additional grid cost: €2.40. Charging plan adjusted." He sees it's a reasonable request and does nothing. The next morning, both cars are ready, and TheAdmin understands exactly why the grid import was higher. *Visibility without control — the system explains household decisions, not just its own.*

**Journey 5: The Helpful Suggestion (future vision)**

A sunny afternoon, solar production exceeding household consumption. Magali's phone buzzes: "Lots of free solar right now — good time to run the washing machine or dishwasher." She starts the laundry. Later, another: "Dishwasher scheduled for 2am tonight — cheapest rates." She taps confirm. Small interactions, zero mental load, delightful timing. *The system as a helpful household companion, not just invisible infrastructure.*

---

### Guest — The Visitor

**Persona:** Guests visit the household and may plug in their EV or increase resource consumption. They never interact with quiet-solar directly.

**Journey:** A guest arrives with an EV. TheAdmin tells the system "guest EV on charger 3, give it what's available after our cars are handled." The system accommodates the guest car at lowest priority while recognizing that guest presence means higher overall consumption — more showers, cooking, hot water. Predictions adjust automatically. When the guest leaves, the system returns to normal patterns.

---

### Future TheAdmin-like User — Community Adopter

**Persona:** A technically capable HA user with solar panels, motivated by the same problems TheAdmin faced. They've never seen quiet-solar before.

**Journey:** They discover quiet-solar through HACS or a community forum post. They read the setup guide, install the integration, and follow the step-by-step configuration for their specific setup (different charger brand, different battery, different solar capacity). The documentation covers their configuration. They get the system running, see their first optimized day, and start building trust. Eventually, they want to contribute — they file a bug report with a scenario description, or submit a PR with a fix. The contribution path is clear and welcoming.

---

### Journey Requirements Summary

| Journey | Capabilities Revealed | Scope Tier |
|---------|----------------------|------------|
| TheDev: Delightful Bug Fix | CI/CD pipeline, test infrastructure, automated releases, coverage gates | MVP |
| TheDev: Solver Exploration | Rich test scenario framework, test home environment, solver experimentation | MVP |
| TheDev: Failure Mode Investigation | Failure mode catalog, degraded-state testing, notification system | MVP |
| TheDev: Scenario Craftsman | Simulation infrastructure, realistic test scenarios, expressive test DSL | MVP |
| TheDev: First Day Back | Developer documentation, code navigability, test expressiveness | MVP |
| TheDev: The Broken Build | CI safety net, clear error reporting, regression prevention | MVP |
| TheDev: Feature Explorer | Research workflow, feature definition process, idea iteration | Product Enhancement |
| TheDev: Agentic Pairing | AI-assisted development, rapid iteration, test-driven exploration | Product Enhancement |
| TheAdmin: First Setup | Guided config flow, device mapping UI, auto-generated dashboards | Existing |
| TheAdmin: Proud Complex Day | Solver multi-device orchestration, counterfactual savings, dashboard | Existing |
| TheAdmin: Troubleshooting | Human-readable explanations, layered diagnostics, transparency | MVP |
| TheAdmin: Off-Grid Crisis | Grid failure detection, load shedding, notifications, auto-recovery | Existing |
| TheAdmin: The Letting Go | Periodic summary notifications, trust-building metrics | Product Enhancement |
| Magali: Invisible Charge | Trip prediction, behavioral learning, proactive notifications | Existing |
| Magali: Override | Quick override UI, solver replanning, instant confirmation | Existing |
| Magali: Conflict | Cross-household notifications, plan change communication | Existing |
| Magali → TheAdmin: Visible Override | Cross-household visibility, cost attribution | Product Enhancement |
| Magali: Helpful Suggestion | Recommendation engine, notification intelligence | Vision |
| Guest | Low-priority load handling, consumption prediction adjustment | Existing |
| Community Adopter | Documentation, setup guide, contribution path | Community Readiness |

## Domain-Specific Requirements

### Safety & Electrical Constraints

- **Circuit limit management** — the system must never exceed per-phase amperage limits when multiple chargers are active simultaneously. The dynamic charger budgeting system balances power across chargers respecting physical circuit constraints.
- **Battery management** — respect manufacturer-specified charge/discharge limits, minimum state of charge thresholds, and cycle management. Never over-discharge during off-grid mode.
- **Off-grid mode safety** — when grid power is lost, the system must prioritize essential loads, shed non-critical loads, and prevent battery depletion below safe thresholds. Incorrect load shedding decisions have real household impact.

### Real-Time Operations

- **Solver timing** — the solver re-evaluates when triggered by state changes or constraint modifications, with a periodic fallback (currently 5 minutes) to ensure no stale state persists. It plans in configurable discrete time windows (currently 15 minutes, SOLVER_STEP_S = 900). Solver execution must not block the HA event loop or cause delays in device command execution.
- **Device communication latency** — OCPP, Wallbox, and Generic charger protocols have different response characteristics and reliability profiles. The system must tolerate communication delays and detect when a device becomes unresponsive.
- **Stale state handling** — when device communication stalls, the system must detect staleness and make conservative decisions rather than acting on outdated state.

### Integration Constraints

- **Multi-protocol charger support** — OCPP, Wallbox, and Generic chargers each have different control capabilities (some support precise power setting, others only on/off). The system must abstract these differences and optimize within each protocol's constraints.
- **Solar forecast API dependencies** — external services (Solcast, etc.) are rate-limited, sometimes unreliable, and may return inaccurate data. The system must cache forecasts, detect API failures, and degrade gracefully with stale or missing forecast data.
- **EV vendor APIs** — flaky, rate-limited, may change without notice. Vehicle state (SOC, range, connection status) depends on these APIs. The system must tolerate API outages and make conservative assumptions when data is unavailable.
- **Home Assistant platform constraints** — all code must respect the HA async event loop, entity lifecycle, and configuration flow patterns. The domain logic layer must remain pure Python with no HA imports (strict two-layer architecture boundary).

### Data & Privacy

- **All data stays local** — core operation requires no cloud dependency. Household behavioral data (trip patterns, presence, routines) is stored only on the local HA instance.
- **Sensitive data handling** — trip prediction data, household routines, and person-device mappings are sensitive. Never expose in logs, never transmit externally, never include in error reports.

## Innovation & Novel Patterns

### Detected Innovation Areas

**1. Unified Household Intelligence (Meta-Innovation)**
Quiet Solar's core innovation isn't any single feature — it's the emergent property of running everything through a single constraint-based solver. Energy optimization, behavioral prediction, crisis handling, and multi-device orchestration all flow through one decision engine. Competitors solve pieces: EVCC optimizes EV charging, Emhass optimizes energy flows. Quiet Solar optimizes *household life*. This unified approach is architecturally unique and extremely hard to replicate — it requires a ground-up design that treats the home as a single system, enabled by a strict two-layer architecture (pure Python domain logic + HA integration bridge) and a three-persona design philosophy (TheAdmin, Magali, TheDev) that ensures every decision serves the whole household.

**2. People-Aware Energy Management**
No competitor in the HA ecosystem models household behavior as an input to energy optimization. Quiet Solar innovates on two fronts:
- **Prediction side:** Trip forecasting from GPS/mileage history, person-car allocation, presence-based scheduling. The system learns household patterns and minimizes what it needs to ask — the goal is that the system understands the household's needs before they're expressed.
- **Interaction side:** When human input IS needed (overrides, conflict resolution, suggestions), the experience is designed to be elegant, instant, and fun. A 5-second override. Clear conflict notifications. Well-timed helpful suggestions. The innovation is that interacting with a home energy system can be *delightful*, not just functional.

**3. Developer Experience as Product Design**
Most open-source HA integrations are maintained by solo developers who eventually burn out because the codebase becomes a burden. Quiet Solar treats developer experience with the same design rigor as user-facing UX: 100% coverage as a joy not a chore, expressive test scenarios that double as documentation, agentic pairing as a first-class workflow, a codebase designed to be a pleasure to return to after weeks away. This isn't just good engineering practice — it's a strategic innovation that creates a community flywheel: a delightful development experience attracts contributors, which makes the product better, which attracts users.

**4. Integrated Resilience**
Most solar optimization tools don't address grid failure at all. Quiet Solar's off-grid mode isn't a separate emergency system bolted on — it's the *same constraint solver* operating under different parameters. When the grid fails, the solver shifts from "maximize self-consumption" to "protect essential loads within available capacity." This architectural choice means resilience improves automatically as the solver improves. It transforms an optimization tool into a protection tool — the "founding story" that defines the product promise.

### Market Context & Competitive Landscape

The residential solar optimization space in Home Assistant is served primarily by:
- **EVCC** — focused on EV charging optimization, single-device scope, technically capable but narrow
- **Emhass** — energy management with forecast integration, requires technical configuration, hardware-centric optimization, no people modeling

Quiet Solar's competitive moat is the combination of unified household intelligence, people-aware prediction, and multi-persona UX. No existing solution attempts all three simultaneously. The moat deepens over time: more household data = better predictions = fewer overrides = more trust.

The biggest competitor isn't another tool — it's *inertia and manual management*. Most homeowners with solar don't optimize at all. They let energy flow to the grid at poor rates because setting up optimization feels too complex or too risky. Quiet Solar's most important competitive advantage is making the leap from "doing nothing" to "intelligent automation" easy enough that people actually take it.

### Validation Approach

- **Unified solver:** Validated through production use in a real household + expanded test scenarios covering multi-device orchestration, competing priorities, and crisis conditions
- **People-aware prediction:** Validated through real trip prediction accuracy over time, measured by decreasing override frequency
- **Developer experience:** Validated through TheDev's subjective satisfaction + community contributor onboarding success
- **Integrated resilience:** Validated through real grid failure events + crisis-tier test scenarios

### Risk Mitigation (Strategic)

These are product-level risks to the innovation thesis. For execution risks (technical, resource, market), see [Project Scoping — Risk Mitigation Strategy](#risk-mitigation-strategy).

- **Solver complexity:** The unified solver is the hardest piece to maintain and debug. Mitigation: failure mode catalog, three-tier test scenarios, and expressive test infrastructure that makes solver behavior observable.
- **Prediction accuracy:** Behavioral prediction will sometimes be wrong. Today's mitigation: easy override flows and clear notifications when predictions miss — the system recovers gracefully through human input. Future mitigation: exposing confidence levels so the system is transparent about how certain it is, allowing TheAdmin and Magali to calibrate their trust. Confidence level exposure is not yet implemented and belongs in the Product Enhancement track.
- **Developer experience sustainability:** Maintaining delight requires discipline. Mitigation: CI/CD enforces quality gates automatically, 100% coverage is non-negotiable, test expressiveness is a design goal not an afterthought.
- **Moat defensibility:** The realistic threat is another project forking the approach, not a well-funded competitor. A fork gets the code but not the accumulated household-specific data, the test scenarios built from real usage patterns, or the contributor community. The moat is the *combination* of unified architecture + community + accumulated real-world validation.

## Home Assistant Integration Specific Requirements

### Project-Type Overview

Quiet Solar is distributed as a Home Assistant custom component via HACS. It operates entirely within the HA ecosystem — leveraging HA's entity model, configuration flow, service calls, and event system. The integration manages 8+ device types through a two-layer architecture: pure Python domain logic that never imports from `homeassistant.*`, and an HA bridge layer that translates between HA state and domain models.

### Device Type Architecture

**Current device types:** Solar source, battery storage, EV chargers (OCPP, Wallbox, Generic), cars, persons, pool pump, heat pump/climate, on/off duration loads, dynamic charger groups.

**Extensibility:** New device types are expected as the product evolves. Adding a device type requires touching: `const.py` (constants), `home_model/` (domain logic), `ha_model/` (HA bridge), `config_flow.py` (UI), platform files (entities), and tests (full coverage). The architecture supports this cleanly but it's a multi-file operation — the PRD should account for device type additions in Product Enhancement and Vision scope.

### Distribution & Update Mechanism

- **HACS distribution:** Standard HACS custom component. Users install and update through HACS UI. No custom update mechanism needed.
- **Migration handling:** When the data model changes between versions, HA's built-in config entry migration system handles upgrades. This must be tested as part of the CI pipeline.
- **Version policy:** Aggressive HA version targeting (currently 2026.2.1+). No backwards compatibility window — users are expected to keep HA reasonably current. This simplifies the codebase by allowing use of latest HA APIs and Python features.

### Dashboard & UI Layer

- **Current approach:** Programmatic Jinja2-based dashboard generation with 4 custom JS Lovelace cards. Dashboards are auto-generated during setup — simple views for household members, detailed views for TheAdmin.
- **Solidification phase:** UI layer stays as-is. No changes planned.
- **Future considerations:** Possibility for TheAdmin or TheDev to customize Jinja templates for personalized dashboards. More advanced custom dashboard designs may come with new features post-solidification. The UI layer should be designed to accommodate this evolution without breaking auto-generated defaults.

### HA Platform Integration Patterns

- **Configuration:** Multi-step config flow through HA UI. Each device type has its own configuration step with validation and feedback.
- **Entities:** Sensors (state monitoring), switches (device control), numbers (parameter adjustment), selects (mode selection), buttons (manual actions). Each device type registers appropriate entity types.
- **Runtime:** QSDataHandler coordinates state tracking. The solver re-evaluates on state changes and constraint modifications, plans in configurable discrete time windows (currently 15 minutes), and issues commands through HA service calls.
- **Async discipline:** All external I/O is async. Blocking operations use `hass.async_add_executor_job()`. No `time.sleep()`, no synchronous file I/O, no blocking in the event loop.

### Implementation Considerations

- **Pure Python domain layer:** The solver, constraints, load models, and commands must never import from `homeassistant.*`. This is the project's most important architectural boundary — it enables testing without a running HA instance and keeps the core logic portable.
- **Test without HA:** Domain logic tests run with lightweight test doubles (FakeHass, factory-built objects), not a real HA instance. This keeps the test suite fast and focused.
- **HACS compliance:** Manifest, version, and file structure must comply with HACS requirements. CI should validate HACS compatibility.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Solidification MVP — the product is already in production and serving a real household. The MVP isn't "build something new" but "make what exists bulletproof, well-tested, and delightful to develop." This is an engineering-quality MVP focused on two problems: (1) the system needs hardened resilience and expanded test coverage, and (2) TheDev needs a development workflow that makes working on the project a joy.

**Resource Model:** Solo developer (TheDev) with agentic coding assistance. The existing codebase was hand-coded and significantly larger than the solidification scope. Agentic pairing makes this scope achievable for one person.

**Cut Line Strategy:** If scope pressure requires reduction, cut *depth* not *breadth* — reduce the number of test scenarios and failure mode catalog rows rather than dropping entire work streams. All four MVP work streams (CI/CD, bug fixes + solver, expanded tests, failure resilience) are must-have at minimum viable depth.

### MVP Feature Set (Phase 1 — Solidification)

**Core Journeys Supported:**
- TheDev: Delightful Bug Fix (validates CI/CD end-to-end)
- TheDev: Solver Exploration (validates test scenario framework)
- TheDev: Failure Mode Investigation (validates failure catalog workflow)
- TheDev: Scenario Craftsman (validates simulation infrastructure)
- TheDev: First Day Back (validates code navigability and test expressiveness)
- TheDev: The Broken Build (validates CI safety net)
- TheAdmin: Troubleshooting (validates human-readable failure explanations)

**Must-Have Capabilities:**

| Capability | Must-Have | Cut-Line Depth |
|-----------|-----------|----------------|
| CI/CD pipeline (lint, test, coverage, release) | Full | No reduction — this is the foundation |
| Bug fixes (person-car assignment, known issues) | Full | No reduction — these are real user pain |
| Solver edge case improvements | Core cases | Can defer less common edge cases |
| Expanded charger test scenarios | Representative set | Can start with 3-5 key scenarios, expand later |
| Expanded solver test scenarios | Representative set | Can start with 3-5 key scenarios, expand later |
| Failure mode catalog | Complete analysis | Can implement responses for top-priority rows first |
| Failure-specific test scenarios | Top-priority rows | Full catalog coverage can follow |

### Post-MVP Features

**Product Enhancement (independent track):**
- Guided onboarding UX improvements
- Enhanced prediction accuracy
- Richer notification and override UX for household members
- New features as identified (new device types, dashboard customization)
- TheAdmin: The Letting Go (periodic summary notifications, trust-building metrics)
- Magali → TheAdmin: Visible Override (cross-household visibility, cost attribution)
- TheDev: Feature Explorer, Agentic Pairing (research and iteration workflows)

**Community Readiness (independent track):**
- Documentation: setup guide, architecture docs, contribution guide
- Designed for TheAdmin-like HA-savvy users
- Contribution path for community members
- Community Adopter journey: install, configure, contribute

**Vision (Future):**
- Proactive recommendations ("Good time for laundry")
- Gamification (daily household energy efficiency score)
- Dynamic tariff support (real-time spot pricing)
- New device types as community identifies needs
- Broader user base beyond TheAdmin-like users
- Magali: Helpful Suggestion journey

### Risk Mitigation Strategy (Execution)

For strategic product risks (solver complexity, prediction accuracy, moat defensibility), see [Innovation — Risk Mitigation](#risk-mitigation-strategic).

**Technical Risks:**
- *Failure mode catalog scope creep:* The analysis could expand indefinitely. Mitigation: time-box the analysis, prioritize by user impact (the catalog's user impact column guides priority), implement top rows first.
- *Solver regression during improvements:* Changing the solver risks breaking existing behavior. Mitigation: expanded test scenarios are written *before* solver changes, CI catches regressions automatically.
- *CI/CD setup complexity:* GitHub Actions configuration for HA custom components has quirks. Mitigation: this is well-trodden ground in the HA ecosystem — reference existing integrations' CI setups.

**Market Risks:** Not applicable for the MVP solidification phase. The product serves a single household today, and community adoption is a post-MVP concern. Market validation will happen naturally when Community Readiness begins — the founding story and measurable savings will be the proof points at that stage.

**Resource Risks:**
- *Solo developer sustainability:* Everything depends on TheDev staying energized. Mitigation: developer delight is an explicit product goal — the CI/CD, test infrastructure, and agentic workflow are designed to keep development fun.
- *Scope overrun:* The solidification phase could expand as new issues surface. Mitigation: cut-line strategy — reduce depth rather than adding scope. Ship the MVP, iterate.

## Functional Requirements

### Energy Optimization & Scheduling

- FR1: The system can allocate available power across all controllable loads simultaneously based on solar forecasts, energy tariffs, and household commitments
- FR2: The system can plan energy allocation in configurable discrete time windows while re-evaluating with real-time state every solver cycle
- FR3: The system can prioritize loads using configurable priority tiers (mandatory urgent, mandatory with deadline, before-battery green, filler)
- FR4: The system can shift deferrable loads to cheaper tariff windows (peak/off-peak scheduling)
- FR5: The system can calculate and display counterfactual savings ("what would this have cost without optimization?")
- FR5b: The system can resolve optimization trade-offs using a strict priority hierarchy: maximize solar self-consumption, then minimize energy cost, then maintain comfort commitments

### Device Orchestration

- FR6: The system can control EV chargers across multiple protocols (OCPP, Wallbox, Generic) with protocol-appropriate commands
- FR7: The system can dynamically balance power budgets across devices within a dynamic group, respecting the physical or virtual circuit's measured capacity limits (per-phase amperage, breaker ratings)
- FR8: The system can manage battery storage (charge/discharge decisions) within manufacturer-specified limits
- FR9a: The system can control pool pumps, water boilers, and on/off duration loads with scheduling and solar-aware timing
- FR9b: The system can manage piloted devices (e.g., heat pump with splits) — a piloted device is turned ON when any linked device is ON, and turned OFF when all linked devices are OFF
- FR9c: The system can manage dynamic groups — a virtual device representing a node in the electrical circuit (physical or virtual) with optional power/current measurement, enforcing configurable capacity limits across all devices in that group. Can be used to model physical breaker protection or to virtually limit current in any circuit controlled by quiet-solar
- FR10: The system can detect when a device's state changes without a solver-initiated command (external control) and exclude that device from the current planning cycle, re-including it in the next evaluation
- FR11: The system can restrict specific devices to use only free solar energy, never drawing from the grid
- FR12: The system can manage devices that support only boost mode (not full control)

### Household Intelligence

- FR13: The system can predict household members' trips based on GPS and mileage history
- FR14: The system can automatically assign the best car with minimum required charge for predicted trips
- FR15: The system can learn household presence patterns (home, away, asleep) and factor them into optimization
- FR16: The system can create constraints from calendar events or manual daily schedules
- FR17: The system can predict baseline home consumption per solver time window using rolling historical data, and factor these predictions into forward planning
- FR18: The system can detect guest presence (via manual toggle or occupancy signal) and increase baseline consumption predictions accordingly

### Resilience & Failure Handling

- FR19: The system can detect grid outages and automatically switch to off-grid mode
- FR20: The system can prioritize essential loads and shed non-critical loads during grid outages
- FR21: The system can detect when an external service is unavailable (solar forecast API, EV vendor API, charger protocol) and operate with degraded data
- FR22: The system can fall back to last-known-good data when a service fails
- FR23: The system can automatically recover normal operation when a failed service returns
- FR24: The system can detect stale device state and make conservative decisions rather than acting on outdated information
- FR25a: The system can explain why a commitment was missed (e.g., car not charged to target SOC) with specific causes and contributing factors
- FR25b: The system can provide a decision log showing what the solver decided and why for any given time period

### Notifications & User Interaction

- FR26: Magali can receive informational notifications about scheduled outcomes ("Your car will be ready by 7am")
- FR27: Magali can override the system's plan for her car with a target SOC and departure time in under 5 seconds
- FR28: Magali can receive immediate notification when another household member changes a departure time, SOC target, or car assignment that affects her commitments
- FR29: TheAdmin can receive notifications when external services fail, explaining what happened and what the system is doing
- FR30: TheAdmin can receive notifications when a household member's override shifts consumption from solar to grid, including the estimated cost impact
- FR31: The system can detect when household members' constraints compete for shared resources (circuit capacity, car availability, overlapping departure windows) and notify all affected members
- FR32: TheAdmin can configure notification preferences for each household member

### Configuration & Onboarding

- FR33: TheAdmin can add and configure devices through the HA UI with a step-by-step guided flow
- FR34: TheAdmin can map chargers to cars to people through the configuration UI
- FR35: TheAdmin can configure solar forecast sources
- FR36: TheAdmin can set and adjust priority levels and constraints for each device
- FR37: The system can auto-generate dashboards appropriate to each user type (simple for Magali, detailed for TheAdmin)
- FR38: TheAdmin can set up a guest EV session by assigning a charger and departure time, completable in minimal steps

### Monitoring & Analytics

- FR39: TheAdmin can monitor self-consumption rate, grid export, grid import, and energy cost through dashboards
- FR40: TheAdmin can view counterfactual savings over configurable time periods
- FR41: TheAdmin can view solver decisions and understand why specific allocations were made
- FR42: TheAdmin can view prediction accuracy trends (forecast vs. actual) over time

### Developer Workflow & Quality

- FR43: TheDev can run the full test suite with a single command and see 100% coverage results
- FR44: TheDev can push code and have CI automatically run linting, type checking, full test suite, and coverage gate
- FR45: TheDev can merge code and have releases with release notes generated automatically
- FR46: TheDev can write expressive test scenarios that describe real-world device configurations and household situations
- FR47: TheDev can consult and update a failure mode catalog that documents all external dependency weaknesses
- FR48: TheDev can run failure-specific test scenarios that simulate degraded and crisis conditions
- FR49: TheDev can return to the codebase after extended absence and understand context through test names and catalog

### System Operations

- FR50: The system can migrate configuration and state data seamlessly when updating between versions
- FR51: The system can identify periods where solar surplus exceeds a configurable threshold for a configurable minimum duration, making that energy available for discretionary use

## Non-Functional Requirements

### Performance

- NFR1: Solver execution time must not exceed the shortest configurable time window — the solver must complete within its planning granularity to avoid stale decisions
- NFR2: Solver execution must not block the HA event loop — long-running computations must yield or run in an executor
- NFR3: The solver re-evaluates when triggered by state changes or constraint modifications, with a periodic fallback to prevent stale state — each re-evaluation must complete within the configured solver time window to avoid stale decisions
- NFR4: Device command execution must complete within one solver evaluation cycle of condition detection — the system must react to real-time conditions (e.g., sudden solar drop, charger disconnect) without waiting for additional cycles

### Privacy

- NFR5: All household data (trip patterns, presence routines, person-device mappings, energy consumption) must remain on the local HA instance — no external transmission for core operation
- NFR6: Sensitive behavioral data must never appear in log output, error reports, or diagnostic messages
- NFR7: Solar forecast API calls must not expose household consumption patterns or behavioral data to the forecast provider

### Reliability

- NFR8: After an HA restart (power failure, software update, or crash), the system must recover previous state from persisted sensor data and resume optimization without manual intervention — recovery must complete within one solver evaluation cycle of HA becoming ready
- NFR9: No constraint, load state, or device configuration should be lost across restarts — all critical state must be persisted through HA's sensor and entity storage mechanisms
- NFR10: The system must operate continuously 24/7 — all data structures (history buffers, prediction caches, accumulated state) must be bounded by configurable retention windows. Unbounded growth patterns must be flagged during code review and testing.
- NFR11: When the system cannot make an optimal decision (insufficient data, conflicting constraints), it must fail safe — make a conservative choice rather than an aggressive one

### Integration Quality

- NFR12: Device commands that are not acknowledged must be retried with configurable retry count and exponential backoff — the system must not silently fail to control a device. Failed commands must be logged and surfaced to TheAdmin.
- NFR13: External API failures (solar forecast, EV vendor) must be detected within the current evaluation cycle — not after the fact
- NFR14: Protocol-specific communication quirks (OCPP session management, Wallbox API rate limits, Generic charger polling) must be abstracted so the solver operates on a uniform device interface
- NFR15: Integration with HA must follow all async patterns — no blocking calls, proper entity lifecycle management, correct use of `@callback` decorator

### Code Quality

- NFR16: 100% test coverage — mandatory, non-negotiable, enforced by CI gate on every PR
- NFR17: Domain logic layer must contain zero imports from `homeassistant.*` — the two-layer architecture boundary must never be violated
- NFR18: Test names must include the scenario context and expected outcome — a developer returning after extended absence should understand what each test validates from its name alone
- NFR19: All code must pass Ruff formatting and linting, and MyPy type checking, enforced by CI
- NFR20: Failure mode catalog must be updated whenever a new failure mode is discovered or an existing system response changes — every external dependency weakness must be cataloged with its system response and test coverage

### Observability

- NFR21: Every solver decision and device command must be logged at debug level; state changes and failures at info level. Follows project convention: log state changes once, log recovery once, no log spam.

### Device Protection

- NFR22: The system must limit on/off cycling for physical devices that are sensitive to rapid transitions (e.g., pool pumps) using configurable `num_max_on_off` limits and hysteresis. Devices with different sensitivity profiles (water boilers vs. pool pumps vs. chargers) must have appropriate transition constraints.

### Developer Experience

- NFR23: Common developer workflows (bug fix, feature addition, release) must each require minimal manual steps — measured by the number of manual steps per workflow and the absence of repeated friction points.
