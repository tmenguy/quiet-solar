---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments: []
date: 2026-03-17
author: Thomas
---

# Product Brief: quiet-solar

## Executive Summary

Quiet Solar is an open-source Home Assistant custom component that optimizes household solar energy self-consumption. It orchestrates all controllable loads in a home — EV charging, battery storage, HVAC, pool heating, hot water boilers, and more — using a constraint-based solver that makes intelligent real-time decisions about what should run and when. Quiet Solar takes a whole-house, people-aware approach: it observes household patterns, predicts upcoming needs, and optimizes energy allocation — all through a family-friendly interface that makes solar optimization accessible to entire households, not just the tech-savvy owner.

---

## Core Vision

### Problem Statement

Homeowners with solar installations waste significant energy because their household devices — EV chargers, heat pumps, pool pumps, boilers, batteries — operate independently without awareness of solar production, energy prices, or household needs. Without intelligent coordination, solar energy flows back to the grid at unfavorable rates, cars aren't charged when family members need them, and electricity bills remain unnecessarily high.

### Problem Impact

- **Financial**: Excess solar exported at low grid rates instead of being self-consumed; high electricity bills despite significant solar investment
- **Quality of life**: Cars uncharged when needed for trips; homes too cold in winter or too hot in summer due to poor timing of HVAC; constant mental load on the homeowner to manually manage device schedules
- **Wasted potential**: Solar panels underperforming their true value because consumption isn't aligned with production

### Why Home Energy Optimization Is Hard

- **Fragmentation**: A household has many controllable loads — EVs, battery, HVAC, pool, boilers — each with its own constraints, priorities, and schedules. Orchestrating them together is fundamentally more complex than optimizing any single device
- **Uncertainty**: Solar forecasts are imperfect, household routines change, trips get planned or cancelled, weather shifts. Any optimization approach must make the best possible decisions with imperfect information and adapt gracefully when reality differs from prediction
- **People are unpredictable**: Energy optimization must account for real human patterns — who needs which car, when, how much charge is required, when people are home, asleep, or away. Automation that ignores household life creates frustration instead of value
- **UX for a whole family**: The system must serve both the technically minded person who sets it up and the household members who just want things to work — requiring radically different interaction models for different users

### Design Philosophy

Quiet Solar follows an **observe-predict-optimize** approach. It continuously observes household state, predicts upcoming energy needs and production, and optimizes the allocation of power across all controllable devices. This is the right engineering approach because home energy is inherently unpredictable — simple rules and manual schedules break under real-world variability, while a constraint-based solver handles uncertainty gracefully and makes the best possible decision at every moment.

### Proposed Solution

Quiet Solar is a Home Assistant integration that acts as a whole-house energy orchestrator. Its constraint-based solver runs in 15-minute intervals, continuously allocating available power across all controllable devices. It ingests solar production forecasts, energy tariffs, battery state, and household patterns — including per-person trip predictions — to make optimal decisions. The system is designed for real households: family members interact through a simple, intuitive interface that requires minimal input while the solver handles the complexity behind the scenes.

### Design Principles

- **Whole-house orchestration**: Manages all controllable loads together — EVs, battery, HVAC, pool, boilers — optimizing across them simultaneously rather than in isolation
- **Observe-predict-optimize**: Learns household patterns, forecasts upcoming needs, and continuously makes the best possible allocation decisions with available information
- **Embrace uncertainty**: Accepts that predictions will sometimes be wrong. The system earns trust not by being perfect, but by being transparent about its decisions and making recovery graceful — clear notifications, easy overrides, and automatic adaptation
- **Family-friendly by design**: Built for entire households — balancing deep automation with the right amount of transparency and human interaction. Household members should delight in the experience, not merely tolerate it
- **Open-source and generic**: Designed to work for any home with solar and controllable loads, not tuned to a specific setup — crafted with care to be the most elegant and enjoyable solution to this problem

---

## Target Users

### Primary Users

#### The Home Energy Manager — "TheAdmin"

TheAdmin is a homeowner with solar panels who is motivated to maximize his solar investment. He's comfortable with Home Assistant — he knows how to install integrations, configure them through the UI, and understands concepts like sensors, switches, and entities — but he's not a developer. He set up all the smart devices in his home and understands the household's habits, schedules, and energy patterns. He sets up dashboards and the HA app for other family members.

**Two distinct modes:**

- **Setup mode**: TheAdmin is learning the system, configuring devices, mapping chargers to cars to people. He's cautious, needs guidance, and wants clear feedback that things are connected correctly. The system should hold his hand through initial configuration with a logical, step-by-step UI flow.
- **Operational mode**: TheAdmin is confident and wants efficiency. He monitors overall energy performance through dedicated dashboards (self-consumption rate, grid export, savings), tweaks constraints and priorities as household patterns evolve, and troubleshoots when things go wrong.

**Emotional journey:**
- **Satisfaction**: Watching the self-consumption rate climb over weeks — the system is visibly working
- **Pride**: A complex day (cloudy morning, two cars needing charge, guests arriving) handled perfectly without intervention
- **Trust**: Over time, checking the dashboard less because the system consistently makes good decisions

**Frustration journey (when things go wrong):**
TheAdmin wakes up and Magali's car isn't charged — she's upset. TheAdmin needs to quickly understand *why*: Was it a bad solar forecast? A competing priority that won? A configuration issue? The system must provide clear, layered diagnostic information — not raw HA logs, but a human-readable explanation: "Solar production was 40% below forecast yesterday. Car B was prioritized over Car A due to earlier departure time. Override was available but not acted on." This troubleshooting experience is the product for TheAdmin in those moments — it's where trust is either reinforced or broken.

**Motivations:** Lower energy bills, maximize solar self-consumption, the craft and satisfaction of an optimized home, providing a seamless experience for his family
**Frustrations today:** Wasted solar going to the grid, no unified orchestration, manual coordination of devices, opaque system behavior when things go wrong

---

#### The Developer — "TheDev"

TheDev is a developer who finds joy in solving real family problems through code. He's drawn to new technologies (including agentic coding), novel algorithmic approaches, and the challenge of modeling human/system interactions. He's allergic to bureaucracy, manual processes, and fighting tooling to get things done. TheDev and TheAdmin are often the same person wearing different hats — the boundary is porous, and that's by design.

**What makes it fun:**
- Solving real problems that affect the family — the motivation is tangible and personal
- Exploring new tech: agentic coding workflows, constraint solvers, behavioral prediction algorithms
- The craft of building something elegant — clean architecture, expressive tests, delightful developer experience

**Two distinct modes:**

- **Building mode**: TheDev is implementing features, fixing bugs, improving the solver. The development workflow must be smooth: push code, CI handles everything (lint, test, coverage, release), no manual GitHub operations. Writing tests should be satisfying — the test infrastructure is rich and expressive, test scenarios read like documentation.
- **Exploring mode**: TheDev is researching new approaches, bouncing ideas with an AI coding partner, experimenting with solver improvements. The codebase welcomes experimentation — fast test feedback, clear architecture boundaries, easy to understand after weeks away.

**Emotional journey:**
- **Delight**: The first bug fix through the automated pipeline — smooth, fast, zero friction
- **Flow**: Deep in a solver exploration session, writing test scenarios that reveal edge cases, iterating rapidly
- **Pride**: Returning after weeks away and being productive in 15 minutes — the test names and architecture make the codebase self-documenting
- **Energy**: Working on the project recharges rather than drains — the tooling, the tests, the agentic pairing all contribute to making development fun

**Motivations:** The craft of building elegant solutions, solving real family problems through code, learning new technologies, maintaining a codebase that's a joy to work in
**Frustrations today:** Manual GitHub operations, fighting tooling, tedious test writing, losing context after time away

---

#### The Household Member — "Magali"

Magali lives in the house but has zero interest in Home Assistant internals. She drives one of the EVs, uses the house appliances, and expects things to just work. She may have the HA app on her phone but interacts with the system rarely and only when needed.

**Role with quiet-solar:**
- **Passive 95% of the time**: The system predicts her needs based on past habits, calendar, presence detection, and routines (trips, wake/sleep patterns, home/away)
- **Receives notifications**: Informational ("Your car will be ready by 7am") and actionable suggestions ("Good time to start the laundry now" or "Dishwasher scheduled for 3am tonight")
- **Occasional override**: In exceptional cases, she overrides the system — e.g., "I need the car at 80% SOC by 7am tomorrow." This must be a **5-second interaction**, not a 2-minute hunt through dashboards. If override is not frictionless and instant, trust in the whole system erodes.
- **Receives conflict alerts**: If another household member changes a plan that affects her, she gets an immediate, clear notification — e.g., "TheAdmin changed the charging plan. Your car is now scheduled for 8am instead of 7am — tap to adjust."

**Emotional journey:**
- **Relief**: The first time she realizes her car is charged without having asked — "Oh, I didn't even have to think about it"
- **Absence of worry**: Over weeks, she stops thinking about charging entirely. The mental load disappears.
- **Delight**: Getting a well-timed suggestion ("Good time for laundry now — lots of free solar") that feels helpful, not intrusive
- **Trust**: When an override is needed, it works instantly and the system confirms clearly

**Motivations:** Comfort, convenience, car always ready when needed, warm/cool house, zero mental load
**Frustrations today:** Car not charged when needed, house too cold, being asked to think about energy management

---

### Secondary Users

#### The Guest

Guests visit the household and may plug in their EV or increase resource consumption (hot water, heating, electricity). They never interact with quiet-solar directly. Any accommodation for guests is handled by a household member through a simple interaction — e.g., "there's a guest EV on charger 3, give it what's available after our cars are handled." The system should also be aware that guests impact consumption predictions: more showers, more cooking, more hot water, more overall electricity usage. A lightweight "guest mode" or temporary adjustment helps the system maintain accurate predictions during visits.

---

### Conflict Resolution Between Household Members

The system does not arbitrate between competing household needs — it **surfaces them**. When multiple people's needs conflict (e.g., two cars need charging but capacity is limited), the system makes a decision based on its constraints and priorities, then notifies all affected members clearly. Any household member can override the decision, which triggers an immediate notification to others affected. The expectation is that people in the same home talk to each other — the system's job is to ensure everyone knows the current plan and can adjust it. Notification chains must be near-instant and crystal clear. In practice, overrides that trigger counter-overrides should be extremely rare.

---

### User Journeys

#### TheAdmin's Journey (Home Energy Manager)
- **Discovery**: Finds quiet-solar through Home Assistant community forums, HACS, or word of mouth while searching for solar optimization
- **Onboarding (Setup mode)**: Installs via HACS, adds the integration through the HA UI, configures devices step by step — guided through solar, battery, chargers, cars, people mappings with clear feedback at each stage
- **Core Usage (Operational mode)**: Checks energy dashboards periodically, reviews system efficiency, tweaks constraints and priorities as needed. Frequency decreases as trust builds.
- **Success Moment**: First month where self-consumption is noticeably higher and the electricity bill drops — "the system is actually working." Later: a complex day handled perfectly without intervention.
- **Frustration Moment**: A prediction misfire — car not charged, family member upset. The system provides a clear explanation of what happened and why, reinforcing trust through transparency rather than perfection.
- **Long-term**: Occasional monitoring and tweaking; adds new devices as household evolves; becomes an advocate in the community

#### TheDev's Journey (Developer)
- **Discovery**: TheAdmin decides to formalize the development workflow — the codebase deserves better tooling
- **First win**: Sets up CI/CD pipeline, pushes the first bug fix, watches it flow through lint → test → coverage → release automatically. Zero manual steps. *"This is how it should be."*
- **Core Usage**: Writes expressive test scenarios, explores solver improvements, pairs with an AI coding partner to iterate rapidly. The test infrastructure is the playground — building better tests IS building a better product.
- **Success Moment**: Returns after six weeks away, runs the test suite, reads the test names, understands the context in 15 minutes. The codebase welcomed him back.
- **Exploration**: Researches new approaches — trip prediction with recency bias, confidence scores, better cloudy-day handling. Ideas get refined through back-and-forth iteration before committing to code.
- **Long-term**: The development workflow stays delightful as the project grows. New contributors can onboard through the test suite and failure mode catalog.

#### Magali's Journey (Household Member)
- **Discovery**: TheAdmin sets it up and tells her "the car will charge itself smartly now"
- **Onboarding**: Almost none — TheAdmin configures her HA app with a simple dashboard; she starts receiving notifications
- **Core Usage**: Sees occasional notifications, rarely overrides. Gets helpful suggestions for energy-efficient appliance timing.
- **Success Moment**: The first time she gets a notification "Your car will be ready at 7am" without having asked — "Oh, I didn't even have to think about it." That's the trust-building moment.
- **Override Moment**: Needs the car earlier than predicted — opens the app, taps "I need 80% by 6am", gets confirmation in seconds. Done.
- **Conflict Moment**: Gets a notification "TheAdmin changed the charging plan, your car is now at 8am" — taps to see why, discusses with TheAdmin, they agree on a compromise. The system updates both.
- **Long-term**: Trusts the system completely, only interacts for exceptions. Occasionally delighted by well-timed suggestions.

---

## Success Metrics

### Optimization Hierarchy

Quiet Solar follows a clear, sequential optimization priority — there is no conflict between objectives:

1. **Maximize free solar self-consumption** — Solar energy is always free. Use it for as much household consumption as possible before anything else.
2. **Minimize grid energy cost** — For remaining needs that solar can't cover, schedule grid consumption at the cheapest available tariff windows (peak/off-peak today, dynamic tariffs in the future).
3. **Maintain comfort and reliability** — Throughout, honor all commitments to household members: cars charged on time, house at comfortable temperature, hot water available.

### TheAdmin's Dashboard Metrics (Outcomes)

These are the metrics TheAdmin monitors to know the system is working:

- **Self-consumption rate (%)**: Percentage of solar production consumed internally rather than exported — the primary indicator of optimization quality
- **Grid export (kWh)**: Solar energy wasted to the grid. Lower is better — means more solar was used internally.
- **Grid import (kWh)**: Energy purchased from the grid. Lower means less dependency on external energy.
- **Energy cost (EUR)**: Total energy cost, optimized by tariff-aware scheduling of grid consumption. The system shifts deferrable loads to cheaper periods.
- **Cost savings — counterfactual (EUR)**: "What would today/this week/this month have cost without quiet-solar's optimization?" This delta is the real proof of value — it answers "was this worth setting up?"

### System Health Metrics (Troubleshooting)

These help TheAdmin diagnose issues when something goes wrong:

- **Execution accuracy**: Commands sent to devices produce the expected results — no bugs, no surprises. Target: 100%.
- **Prediction accuracy**: Forecast vs. actual delta for solar production, household consumption, and trip predictions. Should show an improving trend over time.
- **Tariff awareness accuracy**: For peak/off-peak schedules today — were loads correctly shifted to cheaper periods? Architecture ready for dynamic tariff prediction when TheAdmin switches plans in the future.
- **Missed commitments**: Count of promised-but-not-delivered targets (e.g., car promised at 80% but delivered at 65%). Target: zero.

### UX Quality Metrics (User Experience)

These indicate whether the system is achieving its "invisible magic" goal:

- **User override frequency**: How often household members need to manually intervene. A decreasing trend means predictions are improving and trust is building.
- **Notification quality**: Notifications should be helpful, not noisy. If household members dismiss or ignore notifications, they're not adding value.
- **Magali's trust indicator**: The system is succeeding when household members stop thinking about energy entirely — fewer app opens, fewer overrides, zero complaints.

### Household Engagement (Gamification — Future)

A single, simple daily score the whole family can understand — combining self-consumption rate and cost efficiency into one number (e.g., "Today's energy efficiency: 87%"). No need to understand kWh or tariffs — just a clear indicator of how well the household is doing on its shared goal of energy independence. This turns optimization from TheAdmin's solo project into a fun household challenge.

### Project Quality Metrics (Later Phase)

- **Genericity**: The system works for any home with solar and controllable loads, not just one specific setup
- **Community adoption**: Number of installs, GitHub engagement, community contributions — a future concern, but the architecture supports it from the start

---

## Product Scope

### Core Product (Shipped — In Production)

Quiet Solar is a fully functional, production-ready Home Assistant custom component with 100% test coverage (mandatory, non-negotiable). It manages the full energy lifecycle of a household:

**Whole-House Energy Awareness:**
The system knows about every energy-relevant device in the home — solar panels, battery storage, EV chargers, cars, climate systems (AC/heat pumps), pool pumps, water boilers, and other controllable loads. It understands the home as a single energy system, not a collection of independent devices.

**Intelligent Scheduling & Optimization:**
Each device's needs are expressed as priorities — from "this car MUST be ready by 7am" to "heat the pool whenever free solar is available" to "boost the water heater only if needed." The system juggles all these priorities simultaneously, deciding what runs when based on available solar, energy prices, and household commitments. It integrates solar production forecasts and predicts baseline home consumption to plan ahead, not just react.

**People-Aware Automation:**
The system predicts household members' trips based on past habits and automatically ensures the right car has enough charge — assigning the best car with minimum required charge. Constraints can also be created from calendar events or manual daily schedules. Household members are notified of decisions and can override them through the HA mobile app.

**Smart Device Handling:**
The system recognizes when someone controls a device physically outside of quiet-solar (turning on a pool pump manually, switching on the AC directly) and adapts its plans accordingly. Some devices can be forced to use only free solar energy, never drawing from the grid. Devices that can only be boosted (not fully controlled) — like certain heat-pump water heaters — are supported natively.

**Resilience — Off-Grid Mode:**
When the system detects a grid outage (power failure), it automatically reduces household consumption to match available solar and battery capacity. This isn't just optimization — it's protection. Quiet Solar keeps the house running when the grid fails, prioritizing essential loads and shedding non-critical ones.

**Setup & Onboarding:**
TheAdmin configures the integration through the Home Assistant UI — adding devices, mapping chargers to cars to people, setting up solar forecast sources. The system auto-generates dashboards with custom JS cards designed for easy household use: simple load-control views for Magali, and detailed monitoring views for TheAdmin.

> **Note:** Technical details about the constraint-based solver architecture, device type implementations, and internal data flow are documented separately in the architecture documentation, not in this product brief.

---

### Needs Solidifying (Near-Term)

These are existing aspects that need improvement before adding new capabilities, in priority order:

**1. CI/CD & Development Workflow:**
- Automated testing pipeline on GitHub — every PR runs the full test suite
- Structured agentic workflow for feature development
- Release notes, PR/issue tracking, versioning discipline

**2. Bug Fixes (First Real PR Through the Pipeline):**
- Person-car assignment override — notification and override flow needs fixing
- Other minor bugs identified through focused fix sessions

**3. Solver Improvements:**
- Optimization opportunities for better decision-making in edge cases

**4. Documentation & Community Readiness:**
- Entirely missing today — user guide, setup guide, contribution guide
- Documentation is the gateway to community growth — without it, only TheAdmin can configure the system
- Architecture documentation to capture technical details (solver mechanics, constraint types, device implementations) that belong outside this brief

---

### Out of Scope (Future Vision)

Exciting directions that build on the solid core, to be prioritized after near-term solidification:

- **Proactive recommendations**: "Good time to start the laundry now" / "Schedule dishwasher for 3am tonight" — actionable suggestions based on current solar surplus and tariff windows
- **Gamification**: Daily household energy efficiency score, shared goals, making optimization fun for the whole family
- **Dynamic tariff support**: Move beyond peak/off-peak to support real-time spot pricing and dynamic tariff optimization
- **New device types**: Expand the range of controllable loads as the community identifies needs
- **Enhanced prediction**: Better trip forecasting, consumption pattern learning, deeper calendar integration
- **Richer notification UX**: Conflict notifications, override flows, suggestion interactions — making Magali's experience truly delightful

---

### Phase Success Criteria

Since the core product is already in production, success for the current phase means:

- **CI/CD operational**: Every PR runs the full test suite, releases are automated and versioned
- **Development workflow validated**: Assignment bug fix is the first feature delivered through the structured workflow — proving the process works
- **Solver improved**: Measurable improvement in edge case handling
- **100% coverage maintained**: No regression on test quality — non-negotiable
- **Documentation started**: At minimum, architecture doc and basic setup guide exist
- **Ready for structured feature work**: New features follow a disciplined plan-build-test-ship cycle
