---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - prd.md
  - architecture.md
  - product-brief-quiet-solar-2026-03-17.md
workflowType: 'epics'
project_name: 'quiet-solar'
user_name: 'Thomas'
date: '2026-03-18'
---

# quiet-solar - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for quiet-solar, decomposing the requirements from the PRD, Architecture, and Product Brief into implementable stories.

## Requirements Inventory

### Functional Requirements

**Energy Optimization & Scheduling**
- FR1: The system can allocate available power across all controllable loads simultaneously based on solar forecasts, energy tariffs, and household commitments
- FR2: The system can plan energy allocation in configurable discrete time windows while re-evaluating with real-time state every solver cycle
- FR3: The system can prioritize loads using configurable priority tiers (mandatory urgent, mandatory with deadline, before-battery green, filler)
- FR4: The system can shift deferrable loads to cheaper tariff windows (peak/off-peak scheduling)
- FR5: The system can calculate and display counterfactual savings ("what would this have cost without optimization?")
- FR5b: The system can resolve optimization trade-offs using a strict priority hierarchy: maximize solar self-consumption, then minimize energy cost, then maintain comfort commitments

**Device Orchestration**
- FR6: The system can control EV chargers across multiple protocols (OCPP, Wallbox, Generic) with protocol-appropriate commands
- FR7: The system can dynamically balance power budgets across devices within a dynamic group, respecting the physical or virtual circuit's measured capacity limits (per-phase amperage, breaker ratings)
- FR8: The system can manage battery storage (charge/discharge decisions) within manufacturer-specified limits
- FR9a: The system can control pool pumps, water boilers, and on/off duration loads with scheduling and solar-aware timing
- FR9b: The system can manage piloted devices (e.g., heat pump with splits)
- FR9c: The system can manage dynamic groups — enforcing configurable capacity limits across all devices in that group
- FR10: The system can detect when a device's state changes without a solver-initiated command (external control) and exclude that device from the current planning cycle, re-including it in the next evaluation
- FR11: The system can restrict specific devices to use only free solar energy, never drawing from the grid
- FR12: The system can manage devices that support only boost mode (not full control)

**Household Intelligence**
- FR13: The system can predict household members' trips based on GPS and mileage history
- FR14: The system can automatically assign the best car with minimum required charge for predicted trips
- FR15: The system can learn household presence patterns (home, away, asleep) and factor them into optimization
- FR16: The system can create constraints from calendar events or manual daily schedules
- FR17: The system can predict baseline home consumption per solver time window using rolling historical data
- FR18: The system can detect guest presence (via manual toggle or occupancy signal) and increase baseline consumption predictions accordingly

**Resilience & Failure Handling**
- FR19: The system can detect grid outages and automatically switch to off-grid mode
- FR20: The system can prioritize essential loads and shed non-critical loads during grid outages
- FR21: The system can detect when an external service is unavailable and operate with degraded data
- FR22: The system can fall back to last-known-good data when a service fails
- FR23: The system can automatically recover normal operation when a failed service returns
- FR24: The system can detect stale device state and make conservative decisions rather than acting on outdated information
- FR25a: The system can explain why a commitment was missed with specific causes and contributing factors
- FR25b: The system can provide a decision log showing what the solver decided and why for any given time period

**Notifications & User Interaction**
- FR26: Magali can receive informational notifications about scheduled outcomes
- FR27: Magali can override the system's plan for her car with a target SOC and departure time in under 5 seconds
- FR28: Magali can receive immediate notification when another household member changes a departure time, SOC target, or car assignment that affects her commitments
- FR29: TheAdmin can receive notifications when external services fail
- FR30: TheAdmin can receive notifications when a household member's override shifts consumption from solar to grid, including the estimated cost impact
- FR31: The system can detect when household members' constraints compete for shared resources and notify all affected members
- FR32: TheAdmin can configure notification preferences for each household member

**Configuration & Onboarding**
- FR33: TheAdmin can add and configure devices through the HA UI with a step-by-step guided flow
- FR34: TheAdmin can map chargers to cars to people through the configuration UI
- FR35: TheAdmin can configure solar forecast sources
- FR36: TheAdmin can set and adjust priority levels and constraints for each device
- FR37: The system can auto-generate dashboards appropriate to each user type
- FR38: TheAdmin can set up a guest EV session by assigning a charger and departure time, completable in minimal steps

**Monitoring & Analytics**
- FR39: TheAdmin can monitor self-consumption rate, grid export, grid import, and energy cost through dashboards
- FR40: TheAdmin can view counterfactual savings over configurable time periods
- FR41: TheAdmin can view solver decisions and understand why specific allocations were made
- FR42: TheAdmin can view prediction accuracy trends (forecast vs. actual) over time

**Developer Workflow & Quality**
- FR43: TheDev can run the full test suite with a single command and see 100% coverage results
- FR44: TheDev can push code and have CI automatically run linting, type checking, full test suite, and coverage gate
- FR45: TheDev can merge code and have releases with release notes generated automatically
- FR46: TheDev can write expressive test scenarios that describe real-world device configurations and household situations
- FR47: TheDev can consult and update a failure mode catalog that documents all external dependency weaknesses
- FR48: TheDev can run failure-specific test scenarios that simulate degraded and crisis conditions
- FR49: TheDev can return to the codebase after extended absence and understand context through test names and catalog

**System Operations**
- FR50: The system can migrate configuration and state data seamlessly when updating between versions
- FR51: The system can identify periods where solar surplus exceeds a configurable threshold for a configurable minimum duration, making that energy available for discretionary use

### NonFunctional Requirements

**Performance**
- NFR1: Solver execution time must not exceed the shortest configurable time window
- NFR2: Solver execution must not block the HA event loop
- NFR3: The solver re-evaluates when triggered by state changes or constraint modifications, with a periodic fallback to prevent stale state
- NFR4: Device command execution must complete within one solver evaluation cycle of condition detection

**Privacy**
- NFR5: All household data must remain on the local HA instance
- NFR6: Sensitive behavioral data must never appear in log output, error reports, or diagnostic messages
- NFR7: Solar forecast API calls must not expose household consumption patterns

**Reliability**
- NFR8: After an HA restart, the system must recover previous state and resume optimization within one solver evaluation cycle of HA becoming ready
- NFR9: No constraint, load state, or device configuration should be lost across restarts
- NFR10: The system must operate continuously 24/7 — all data structures must be bounded by configurable retention windows
- NFR11: When the system cannot make an optimal decision, it must fail safe

**Integration Quality**
- NFR12: Device commands that are not acknowledged must be retried with configurable retry count and exponential backoff
- NFR13: External API failures must be detected within the current evaluation cycle
- NFR14: Protocol-specific communication quirks must be abstracted so the solver operates on a uniform device interface
- NFR15: Integration with HA must follow all async patterns

**Code Quality**
- NFR16: 100% test coverage — mandatory, non-negotiable, enforced by CI gate
- NFR17: Domain logic layer must contain zero imports from homeassistant.*
- NFR18: Test names must include the scenario context and expected outcome
- NFR19: All code must pass Ruff formatting and linting, and MyPy type checking, enforced by CI
- NFR20: Failure mode catalog must be updated whenever a new failure mode is discovered

**Observability**
- NFR21: Every solver decision and device command must be logged at debug level; state changes and failures at info level

**Device Protection**
- NFR22: The system must limit on/off cycling for physical devices using configurable limits and hysteresis

**Developer Experience**
- NFR23: Common developer workflows must each require minimal manual steps

### Additional Requirements

From Architecture document:

- AR1: CI/CD pipeline must include HACS validation (`hacs/action@main`) on every PR
- AR2: CI must pin to HA's bundled dependency versions (especially numpy) — version mismatch between CI and production can mask runtime failures
- AR3: Trust-critical component tests (multi-charger rebalancing scenarios, constraint interaction boundary tests) are formal architectural requirements, not optional test improvements
- AR4: Test infrastructure smoke tests (verify FakeHass behaves like real HA for critical operations) ship as part of the CI/CD story — the pipeline's self-test
- AR5: Each external dependency must have a documented failure signature, fallback behavior, and recovery path (resilience fallback table in Architecture Decision 1)
- AR6: Solver improvements happen within the stable input/output contract — no interface changes without explicit architectural decision
- AR7: Implementation sequence: CI/CD → bug fix (validates pipeline) → trust-critical tests → resilience strategy → solver optimization
- AR8: Risk-weighted CI strategy: changes to solver/charger are CRITICAL (full suite + benchmarks), changes to platforms are LOW (standard suite)
- AR9: PR template with checklist (tests, coverage, Ruff, MyPy, HACS manifest) and risk assessment by change area

### UX Design Requirements

No UX Design document exists. UX requirements are captured through User Journeys in the PRD and Product Brief persona definitions.

### FR Coverage Map

| FR | Epic | Context |
|----|------|---------|
| FR1-FR5b | Epic 4 | Solver edge case improvements (within stable interface) |
| FR6-FR12 | Epic 2 | Verified via trust-critical test scenarios |
| FR13-FR18 | Epic 5 | Prediction refinement (post-MVP) |
| FR19-FR20 | Epic 2/3 | Verified via tests + resilience implementation |
| FR21-FR25b | Epic 3 | Resilience + transparency implementation |
| FR26-FR32 | Epic 5 | Notification refinement (post-MVP) |
| FR33-FR38 | Epic 6 | Onboarding for community (post-MVP) |
| FR39-FR42 | Epic 5 | Monitoring enhancement (post-MVP) |
| FR43-FR45 | Epic 1 | CI/CD pipeline |
| FR46-FR49 | Epic 2 | Test scenario infrastructure + implemented scenarios |
| FR50 | Epic 1 | Version migration |
| FR51 | Epic 7 | Solar surplus identification (vision) |

**All 52 FRs covered. Zero orphans.**

## Epic List

### Epic 1: Automated Development Pipeline [IN PROGRESS]
TheDev can push code through a fully automated CI/CD pipeline with quality gates, automated releases, and HACS validation — and proves it works by fixing the person-car assignment bug (with targeted regression tests using existing patterns) as the first real PR. Extended with AI-assisted PR review and fully autonomous mobile-driven GitHub flow.
**FRs:** FR43, FR44, FR45, FR50
**ARs:** AR1, AR2, AR4, AR8, AR9
**NFRs:** NFR16, NFR19, NFR23
**Scope:** MVP | **Dependencies:** None (foundation)
**Status:** Core pipeline stories (1.1–1.7) complete. Stories 1.8–1.9 added for PR review automation and mobile-first autonomous flow.

### Epic 2: Test Scenarios & Failure Catalog [DONE]
The system provides a significant volume of implemented trust-critical test scenarios. TheDev can define charger budgeting, constraint interaction, and failure mode scenarios — and can return after extended absence understanding the codebase through test names and the failure catalog.
**FRs:** FR46, FR47, FR48, FR49
**ARs:** AR3
**NFRs:** NFR18, NFR20
**Also verifies:** FR6-FR12, FR19-FR20, NFR14, NFR22
**Scope:** MVP | **Dependencies:** Builds on Epic 1
**Note:** Defines a "done threshold" — minimum scenario set that unlocks Epic 3. Full catalog continues growing. Story 2.1 (Scenario Builder Framework) removed — test infrastructure emerges from implementing the scenarios directly.
**Status:** All stories complete. Story 2.2 (Charger Budgeting Scenario Tests), Story 2.3 (Constraint Interaction Boundary Tests), and Story 2.4 (Solver & Device Orchestration Scenario Tests) done.

### Epic 3: Failure Resilience & Transparency [IN PROGRESS]
TheAdmin can trust the system to handle failures gracefully and understand exactly what happened when things go wrong. Every external dependency has a documented failure path with fallback and recovery. Fed directly by Epic 2's failure catalog — Epic 2 documents what can fail, Epic 3 implements how the system responds. Story 3.2 broken into per-FM stories (one per failure mode) with CC-001/CC-002/CC-003 addressed per FM.
**FRs:** FR21, FR22, FR23, FR24, FR25a, FR25b
**ARs:** AR5
**NFRs:** NFR8, NFR9, NFR10, NFR11, NFR12, NFR13, NFR21
**Scope:** MVP | **Dependencies:** Builds on Epic 1, 2
**Status:** Stories 3.1–3.2 complete. Stories 3.3–3.12 defined (7 FM stories + dashboard + tests + transparency).

### Epic 4: Solver Edge Case Improvements [IN PROGRESS]
**FRs:** FR1-FR5b, FR14 | **ARs:** AR6 | **NFRs:** NFR1-NFR4 | **Scope:** MVP | **Depends on:** Epic 1, 2
**Status:** Story 4.1 added from GitHub Issue #30 (car default charge when no person assigned).

### Epic 5: Household Experience Enhancement *(stories deferred — post-MVP)*
**FRs:** FR13-FR18, FR26-FR32, FR39-FR42 | **NFRs:** NFR5-NFR7 | **Scope:** Product Enhancement

### Epic 6: Community Readiness *(stories deferred — post-MVP)*
**FRs:** FR33-FR38 | **Scope:** Community Readiness

### Epic 7: Vision Features *(stories deferred — post-MVP)*
**FRs:** FR51, future features | **Scope:** Vision

### Dependency Chain (load-bearing)

```
Epic 1 (pipeline) → Epic 2 (confidence) → Epic 3 (resilience) → Epic 4 (optimization)
                                            Epics 5, 6, 7 (independent, post-MVP)
```

## Epic 1: Automated Development Pipeline

TheDev can push code through a fully automated CI/CD pipeline with quality gates, automated releases, and HACS validation — and proves it works by fixing the person-car assignment bug as the first real PR through the system. Extended with AI-assisted PR review and a fully autonomous mobile-driven GitHub flow.

### Story 1.1: Agentic Development Workflow [DONE]

As TheDev,
I want to say "fix this bug" and have the system create a GitHub issue, create a branch, assist me through the fix (with tests added/updated for every change), run the test suite and linters locally, iterate until 100% coverage is maintained, and only then create the PR — all from within Cursor or Claude Code,
So that the entire bug-fix workflow requires zero manual GitHub operations and zero context switching.

**Acceptance Criteria:**

**Given** TheDev initiates a bug fix workflow (via BMad skill or command)
**When** the workflow starts
**Then** a GitHub issue is created with structured bug description
**And** a branch is created from main, named after the issue

**Given** TheDev is developing the fix with agentic assistance
**When** code changes are made
**Then** corresponding tests are added or updated to cover the changes
**And** TheDev can run the full test suite locally with a single command
**And** TheDev can run Ruff and MyPy locally with a single command
**And** the system reports coverage status and identifies uncovered lines
**And** the system plans test additions to cover uncovered lines and executes that plan automatically
**And** the cycle repeats until 100% coverage is achieved

**Given** development is complete and tests pass at 100% coverage
**When** TheDev asks to create the PR
**Then** a PR is created linking to the issue, with proper template filled
**And** the PR is only created after local quality checks pass (tests, lint, type check, coverage)

### Story 1.2: PR Quality Gate Pipeline [DONE]

As TheDev,
I want CI to automatically run Ruff linting, MyPy type checking, the full test suite with 100% coverage gate, and HACS validation on every PR push,
So that code quality is enforced on the remote as a safety net mirroring the local workflow.

**Acceptance Criteria:**

**Given** a PR is opened or pushed to against main
**When** the PR quality gate workflow triggers
**Then** Ruff format check and lint run and fail the pipeline on violations
**And** MyPy type checking runs and fails the pipeline on errors
**And** pytest runs with `--cov=custom_components/quiet_solar` and fails if coverage drops below 100%
**And** HACS validation (`hacs/action@main`) runs and fails on manifest/structure issues
**And** Python version matches HA's production version
**And** numpy and scipy are pinned to HA's bundled versions (AR2)

### Story 1.3: Release Pipeline & Version Migration [DONE]

As TheDev,
I want merging to main with a version tag to automatically create a GitHub Release with auto-generated changelog, and version consistency validated between tag and manifest,
So that shipping a release requires zero manual steps.

**Acceptance Criteria:**

**Given** a version tag (v*) is pushed
**When** the release pipeline triggers
**Then** the full test suite runs as a safety gate
**And** HACS validation passes
**And** tag version matches `manifest.json` version
**And** a GitHub Release is created with auto-generated changelog
**And** configuration migration support (FR50) is validated as part of the release checklist

### Story 1.4: PR Templates & Developer Workflow [DONE]

As TheDev,
I want PR templates with quality checklists and risk assessment categories, issue templates for bugs and features, and CODEOWNERS for auto-assignment,
So that every contribution follows consistent quality standards with minimal friction.

**Acceptance Criteria:**

**Given** a new PR is created
**When** the PR template is loaded
**Then** it includes a checklist (tests, coverage, Ruff, MyPy, HACS manifest)
**And** it includes risk assessment categories (CRITICAL/HIGH/MEDIUM/LOW per architecture risk table)
**And** auto-labeling assigns area labels based on changed file paths
**And** CODEOWNERS maps file paths to reviewers

**Given** a new issue is opened
**When** the issue template is loaded
**Then** bug reports have structured fields (steps to reproduce, expected behavior, device type)
**And** feature requests have structured fields (use case, persona, scope tier)

### Story 1.5: Test Infrastructure Smoke Tests [DISMISSED]

Dismissed — scope deferred to Epic 2 where FakeHass validation will be addressed as part of the scenario builder framework (Story 2.1).

~~As TheDev,
I want the CI pipeline to include smoke tests verifying FakeHass behaves like real HA for critical operations,
So that I can trust the test infrastructure itself isn't giving false confidence.~~

### Story 1.6: Person-Car Assignment Bug Fix (Pipeline Validation) [DONE]

As TheAdmin,
I want the person-car assignment notification and override flow to work correctly,
So that household members are properly notified when car assignments change and overrides are respected.

**Acceptance Criteria:**

**Given** a person's car assignment is changed (by prediction or manual override)
**When** the assignment update is processed
**Then** the affected person receives a notification with the new assignment
**And** the constraint system creates correct charger constraints for the new assignment
**And** the solver replans with the updated constraints
**And** targeted regression tests cover the fix using existing test patterns
**And** the PR passes through the full CI quality gate from Story 1.2
**And** this is the first real code change delivered through the agentic workflow from Story 1.1

### Story 1.7: Parallel Story Development with Git Worktrees [DONE]

As TheDev,
I want every story and bugfix to automatically use a git worktree (with shared venv and HA runtime config via symlinks) by default, with the option to opt out by saying "no worktree",
So that I can work on multiple stories in parallel without branch switching, stashing, or duplicating large dependencies.

**Acceptance Criteria:**

**Given** TheDev starts any story or bugfix
**When** Phase 1b (branch creation) executes
**Then** a worktree is created by default with its own branch
**And** venv, config/, and non-git custom_components are symlinked from the main worktree
**And** the main worktree stays on main, untouched

**Given** TheDev says "no worktree"
**When** Phase 1b executes
**Then** the old behavior is used: `git checkout -b QS_<N>` in the main directory

**Given** a story in a worktree is merged
**When** TheDev cleans up
**Then** the worktree is removed (symlinks go with it, originals untouched)
**And** the main worktree's main branch is updated

### Story 1.8: AI-Assisted PR Review with Interactive Feedback Loop [DONE]

As TheDev,
I want PRs to be reviewed by an AI reviewer (or human) on GitHub, with the system pulling review comments back into the local workflow so I can discuss, fix, or reject feedback interactively — a true PR back-and-forth,
So that code review is integrated into the agentic workflow without requiring manual GitHub context-switching.

**Acceptance Criteria:**

**Given** a PR is open on GitHub
**When** TheDev initiates a review cycle
**Then** an AI reviewer (configuration TBD) or human posts review comments on the PR

**Given** review comments exist on the PR
**When** TheDev asks to process review feedback
**Then** the system pulls all unresolved PR comments from GitHub into the local workflow
**And** each comment is presented with its diff context
**And** TheDev can choose per comment: fix (implement the suggestion), discuss (reply on the PR), or reject (dismiss with rationale)

**Given** TheDev chooses to fix a comment
**When** the fix is implemented
**Then** the fix is committed, pushed, and a reply is posted on the PR resolving the comment

**Given** TheDev chooses to discuss a comment
**When** TheDev provides a response
**Then** the response is posted as a PR reply and the comment remains open for further discussion

**Given** TheDev chooses to reject a comment
**When** TheDev provides a rationale
**Then** the rationale is posted as a PR reply and the comment is resolved

**Given** all comments are processed
**When** the review cycle completes
**Then** the system reports a summary of fixes, discussions, and rejections
**And** quality gates are re-run if any fixes were made

**Open questions:**
- Which AI reviewer to use (GitHub Copilot review, custom GitHub Action with Claude, third-party)?
- How to configure reviewer selection per-PR or per-repo?

### Story 1.9: Mobile-First Autonomous GitHub Flow [DONE]

As TheDev,
I want to create a tagged GitHub issue from my phone, have a cloud-based automation pick it up, run the full BMad workflow autonomously (YOLO mode), and present the result as a PR for review — which I can review, approve, and merge from my phone, triggering the full release pipeline,
So that the entire development lifecycle from idea to release can be driven from a mobile device with zero local setup.

**Acceptance Criteria:**

**Given** TheDev creates a GitHub issue from mobile and applies a trigger tag (e.g., `auto-bmad`)
**When** the cloud automation detects the tagged issue
**Then** it picks up the issue and starts the full BMad workflow autonomously
**And** it creates a branch, implements the fix or feature following all project rules
**And** it runs quality gates (tests, lint, type check, coverage) and iterates until passing
**And** it creates a PR linking to the issue with full template and risk assessment

**Given** the autonomous PR is ready
**When** TheDev reviews from mobile
**Then** the PR includes a clear summary of what was done and why
**And** TheDev can approve or request changes from the GitHub mobile app

**Given** TheDev approves and merges the PR from mobile
**When** the merge completes
**Then** the release pipeline triggers automatically (per Story 1.3)
**And** a GitHub Release is created with changelog
**And** the full flow from issue creation to release completes without any local terminal access

**Open questions:**
- Cloud execution environment (GitHub Actions self-hosted runner, GitHub Codespace, dedicated CI runner)?
- How to provide Claude/agent access securely in CI context?
- Cost and rate-limit implications of autonomous YOLO runs?
- Guardrails for autonomous mode (scope limits, cost caps, human-in-the-loop checkpoints)?

### Story 1.10: Lazy Logging Sweep (f-string cleanup) [DONE]

As TheAdmin,
I want all logging calls to use lazy formatting (`%s` style) instead of f-strings,
So that the codebase follows HA logging guidelines and avoids unnecessary string interpolation when log levels are disabled.

**Scope:** 169 f-string log calls across 19 files in `custom_components/quiet_solar/`. Heaviest files: `ha_model/charger.py` (69), `ha_model/home.py` (25), `home_model/load.py` (24).

**Acceptance Criteria:**

**Given** the codebase contains `_LOGGER.<level>(f"...")` calls
**When** the sweep is complete
**Then** all logging calls use `_LOGGER.<level>("... %s", var)` lazy formatting
**And** no f-string log calls remain in `custom_components/quiet_solar/`
**And** all existing tests still pass

## Epic 2: Trust-Critical Test Scenarios

The system provides a significant volume of implemented trust-critical test scenarios covering charger budgeting, constraint interactions, solver edge cases, and device orchestration gaps — building confidence in the existing codebase before any improvements begin.

### Story 2.2: Charger Budgeting Scenario Tests — **DONE** (PR #QS_19)

As TheDev,
I want implemented test scenarios covering multi-charger rebalancing sequences with intermediate state verification,
So that the trust-critical charger budgeting system is proven safe at every intermediate step.

**Acceptance Criteria:**

**Given** charger budgeting scenarios are executed
**Then** multi-charger rebalancing verifies no phase is exceeded at any intermediate state
**And** staged transition recovery is tested (crash between Phase 1 reduce and Phase 2 increase)
**And** phase switching under load is tested (1P→3P with concurrent chargers)
**And** priority inversion is tested (budget priority vs constraint priority)
**And** dampening accuracy over non-linear EV charging curves is tested
**And** all scenarios use `@pytest.mark.integration` marker

### Story 2.3: Constraint Interaction Boundary Tests [DONE]

As TheDev,
I want implemented test scenarios covering constraint type interactions under resource exhaustion,
So that the system's behavior at constraint boundaries is explicit and tested.

**Acceptance Criteria:**

**Given** constraint interaction scenarios are executed
**Then** MANDATORY_END_TIME vs exhausted `num_max_on_off` switching budget is tested
**And** mandatory constraint vs exhausted charger amp budget is tested
**And** multiple MANDATORY constraints competing for insufficient power is tested
**And** constraint type transitions under resource pressure are verified
**And** all scenarios use `@pytest.mark.integration` marker

### Story 2.4: Solver & Device Orchestration Scenario Tests [DONE]

As TheDev,
I want implemented test scenarios covering solver edge cases and device orchestration gaps identified in the architecture,
So that existing behavior is verified comprehensively before any improvements begin in Epic 4.

**Acceptance Criteria:**

**Given** solver and device orchestration scenarios are executed
**Then** solver behavior under rapid forecast changes (cloudy day) is tested
**And** off-grid mode edge cases are tested (FR19-FR20: load shedding, battery depletion thresholds)
**And** external control detection and adaptation is tested (FR10)
**And** green-only device behavior is tested (FR11)
**And** boost-only device management is tested (FR12)
**And** dynamic group capacity enforcement across nested groups is tested (FR9c)
**And** piloted device coordination is tested (FR9b)
**And** all scenarios use `@pytest.mark.integration` marker

## Epic 3: Failure Resilience & Transparency

TheAdmin can trust the system to handle failures gracefully and understand exactly what happened when things go wrong. The failure mode catalog drives the work: document what can fail, implement how the system responds, test every failure path, and surface human-readable explanations.

### Story 3.1: Failure Mode Catalog & Resilience Plan [DONE]

As TheDev,
I want a structured failure mode catalog documenting every external dependency weakness with its failure signature, planned system response, recovery path, and test coverage status,
So that resilience implementation is systematic and no dependency weakness is left unplanned.

**Acceptance Criteria:**

**Given** the failure mode catalog is created
**When** TheDev consults it
**Then** every external dependency is listed (solar forecast API, charger protocols, EV vendor APIs, HA state, numpy persistence, grid power, prediction confidence)
**And** each entry has: failure signature, fallback behavior, recovery path, implementation status, test coverage status
**And** the catalog aligns with the architecture's resilience fallback table (AR5)
**And** entries are prioritized by user impact (missed commitment > degraded optimization > reduced visibility)
**And** the catalog is a living document updated whenever a new failure mode is discovered (NFR20)

### Story 3.2: FM-006 — Numpy Persistence Hardening [DONE]

As TheDev,
I want numpy persistence failures to be caught with specific exception types and logged as warnings,
So that silent data loss is eliminated and persistence health is observable.

**Priority:** P3 — reduced visibility | **Size:** XS
**Gaps addressed:** G7
**CC Impact:**
- CC-003: `binary_sensor.qs_persistence_health`

**Acceptance Criteria:**

**Given** a corrupted `.npy` file (partial write, disk full, permission error)
**When** the system attempts to load it
**Then** specific exceptions are caught (`OSError`, `ValueError`, `pickle.UnpicklingError`) instead of bare `except:`
**And** a warning is logged identifying the corrupted file and exception type
**And** the system continues without historical data using conservative estimates

**Given** numpy persistence health is queried
**When** TheAdmin checks the dashboard
**Then** `binary_sensor.qs_persistence_health` reflects current persistence state
**And** the sensor turns off when a load/save failure is detected

### Story 3.3: FM-005 — Grid Outage Verification [DONE]

As TheAdmin,
I want grid outage handling to be fully verified — emergency broadcasts reach Magali and load shedding prioritizes essential loads,
So that the household is protected during power outages.

**Priority:** P1 — missed commitment | **Size:** S (verification of existing implementation)
**Gaps addressed:** G6
**CC Impact:**
- CC-001: Verify emergency broadcast reaches Magali with plain-language message
- CC-002: Already exists (`SELECT_OFF_GRID_MODE`)
- CC-003: Already exists (off-grid binary sensors)

**Acceptance Criteria:**

**Given** a grid outage is detected
**When** the system switches to off-grid mode
**Then** Magali receives a plain-language notification ("Grid power is out — the house is running on solar and battery")
**And** essential loads are prioritized over non-critical loads
**And** non-critical loads are shed when solar + battery capacity is insufficient

**Given** the grid is restored
**When** the system detects grid recovery
**Then** a recovery notification is sent to Magali and TheAdmin
**And** normal operation resumes automatically

### Story 3.4: FM-003 — HA Device Communication Resilience

**Foundation story** — establishes generic device resilience patterns that FM-002 (charger) and FM-008 (car API) specialize.

As TheAdmin,
I want the system to verify that devices actually respond to commands, detect stale state, reject garbage values, and alert me when a device needs manual intervention,
So that silent failures don't lead to missed commitments.

**Priority:** P1 (command failure) / P3 (stale state) | **Size:** L
**Gaps addressed:** G15, G16, G17, G18, G19
**CC Impact:**
- CC-001: Human alert when command not applied ("The pool pump didn't turn on — please check the breaker or switch it manually")
- CC-002: Force load ON/OFF switch entities per controllable load
- CC-003: `binary_sensor.qs_<device>_communication_ok` per device

**Acceptance Criteria:**

**Given** the system sends a command to a device (service call)
**When** the device state doesn't change within the expected window
**Then** the system retries once
**And** if still not applied, alerts TheAdmin/Magali with a plain-language message identifying the device and suggested manual action (CC-001)
**And** the device is marked as command-failed

**Given** a device entity's `last_updated` timestamp exceeds a configurable staleness threshold
**When** the system detects the stale state
**Then** the device is marked as stale and conservative decisions are made (FR24)
**And** `binary_sensor.qs_<device>_communication_ok` turns off (CC-003)

**Given** a device reports obviously wrong values (negative power, SOC > 100%, temperature outside physical range)
**When** the system receives the value
**Then** the value is rejected and last-known-good is used instead
**And** a warning is logged identifying the device and rejected value

**Given** TheAdmin wants to force a load ON or OFF
**When** TheAdmin uses the force switch entity (CC-002)
**Then** the device command is sent bypassing the solver
**And** the solver excludes that device from the current planning cycle

### Story 3.5: FM-002 — Charger Communication Resilience

As TheAdmin,
I want charger commands to retry with exponential backoff, chargers to be marked unavailable after persistent failure, and Magali to be notified when her car's charger loses connection,
So that transient charger issues self-heal and persistent failures are escalated.

**Priority:** P1 — missed commitment | **Size:** M
**Dependencies:** Builds on Story 3.4 (FM-003 generic device resilience)
**Gaps addressed:** G3, G4
**CC Impact:**
- CC-001: Magali notification ("Your car isn't charging because the charger lost connection — you can plug into the other charger")
- CC-002: Force-start/force-stop charge button entities per charger
- CC-003: `binary_sensor.qs_<charger>_communication_ok`

**Acceptance Criteria:**

**Given** a charger command is not ACKed within `CHARGER_STATE_REFRESH_INTERVAL_S` (14s)
**When** the system detects the failure
**Then** it retries with configurable count and exponential backoff (NFR12)
**And** after max retries, the charger is marked unavailable with an explicit state transition
**And** the solver excludes the unavailable charger from planning
**And** TheAdmin is notified (FR29)
**And** if a car was actively charging, Magali is notified with plain-language explanation and suggested action (CC-001)

**Given** a charger was marked unavailable
**When** the charger responds again
**Then** the system auto-recovers and re-includes the charger in the next solver cycle (FR23)
**And** TheAdmin is notified of recovery

**Given** TheAdmin or Magali wants to force-start or force-stop a charge
**When** they press the button entity (CC-002)
**Then** the command is sent directly, bypassing the solver
**And** the solver adjusts its plan in the next evaluation cycle

### Story 3.6: FM-004 — Solver Infeasibility Handling

As TheAdmin,
I want the solver to detect when no valid plan exists, fail safe with mandatory constraints prioritized, and notify Magali when a commitment may be missed,
So that the system never silently fails to act and household members know when to intervene.

**Priority:** P1 — missed commitment | **Size:** M
**Gaps addressed:** G5
**CC Impact:**
- CC-001: Magali notification ("Your car may not reach 80% by 7 AM — there isn't enough capacity for all requests")
- CC-002: Force-start charge / force load ON buttons (reuse from 3.4/3.5)
- CC-003: `binary_sensor.qs_solver_health`

**Acceptance Criteria:**

**Given** the solver cannot satisfy all active constraints
**When** infeasibility is detected
**Then** mandatory constraints get priority (mandatory urgent > mandatory deadline)
**And** filler constraints are dropped first, then green constraints
**And** battery is held at current SOC as a safe default (NFR11)
**And** the infeasibility cause is logged (which constraints conflicted, what resource was exhausted)

**Given** a mandatory constraint will be missed due to infeasibility
**When** the solver determines which commitments are affected
**Then** Magali is notified with plain-language explanation of what's affected and why (CC-001)
**And** TheAdmin is notified with technical details (constraint IDs, resource state)

**Given** the solver was infeasible
**When** the next evaluation triggers (state change or periodic fallback)
**Then** the solver re-attempts with updated state
**And** if feasibility is restored, normal operation resumes

### Story 3.7: FM-001 — Solar Forecast API Resilience [DONE]

As TheAdmin,
I want the system to support multiple solar forecast providers, detect stale forecasts, fall back to historical patterns, score providers against actual production, apply dampening corrections, and let me choose the active provider or let the system auto-select the best one,
So that optimization always uses the most accurate forecast available, even when APIs fail.

**Priority:** P2 — degraded optimization | **Size:** L
**Gaps addressed:** G1, G2
**CC Impact:**
- CC-002: `select.qs_solar_provider_mode` (auto / provider name), `switch.qs_solar_dampening_<provider>` per provider
- CC-003: `sensor.qs_solar_forecast_age`, `binary_sensor.qs_solar_forecast_ok`, `sensor.qs_solar_forecast_score_<provider>`, `sensor.qs_solar_forecast_score_raw_<provider>`, `sensor.qs_solar_forecast_score_dampened_<provider>`

**Acceptance Criteria:**

**Given** a solar forecast API fails (timeout, HTTP error, invalid data)
**When** the system detects the failure
**Then** it uses the last successful forecast
**And** if the forecast is stale >6h, it falls back to historical solar patterns from the numpy ring buffer (AR5)
**And** it auto-retries on the next forecast polling cycle
**And** TheAdmin is notified when forecast becomes stale

**Given** TheAdmin configures solar forecast providers
**When** setting up or reconfiguring the solar device in config flow
**Then** multiple providers can be selected simultaneously (e.g., Solcast + Open-Meteo)
**And** each provider is named for identification
**And** `select.qs_solar_provider_mode` allows choosing: "auto", or any individual provider by name
**And** in "auto" mode, the system uses the provider with the best 7-day accuracy score
**And** failed providers are re-probed periodically (not permanently removed)

**Given** forecast quality is queried
**When** TheAdmin checks the dashboard
**Then** `sensor.qs_solar_forecast_age` shows hours since last successful update from the active provider
**And** `binary_sensor.qs_solar_forecast_ok` reflects whether the active forecast is fresh (<6h)
**And** `sensor.qs_solar_forecast_score_<provider>` shows per-provider 7-day accuracy (MAE of forecast vs actual solar production)

**Given** automatic dampening is enabled for a provider (via `switch.qs_solar_dampening_<provider>`)
**When** the system recomputes dampening at midnight
**Then** for each time step k in the provider's native temporal resolution, it computes (a_k, b_k) via MOS linear regression on 7 days of (forecast_k, actual_k) data
**And** dampened forecast = `max(0, a_k * raw_forecast_k + b_k)` (clamped to non-negative)
**And** physical guards are enforced: nighttime steps use identity, a_k bounded to [0.1, 3.0], minimum 3 data points required per step
**And** `sensor.qs_solar_forecast_score_raw_<provider>` shows accuracy without dampening
**And** `sensor.qs_solar_forecast_score_dampened_<provider>` shows accuracy with dampening
**And** if dampening is disabled, the raw forecast is used directly

### Story 3.13: Add Forecast.Solar Provider Support — DONE

As TheAdmin,
I want Forecast.Solar to be available as a solar forecast provider alongside Solcast and Open-Meteo,
So that I have a third provider option and can compare its accuracy against others using the multi-provider infrastructure from Story 3.7.

**Priority:** P3 — additional provider | **Size:** S
**Dependencies:** Story 3.7 (multi-provider infrastructure must be in place)

**Acceptance Criteria:**

**Given** the Forecast.Solar HA integration is installed and configured
**When** TheAdmin configures a solar device in Quiet Solar
**Then** "Forecast.Solar" appears as an available provider in the config flow
**And** it can be selected alongside other providers (Solcast, Open-Meteo)
**And** it participates in provider scoring, dampening, and auto-selection like any other provider

**Given** the Forecast.Solar provider is active
**When** it delivers forecast data
**Then** the forecast is extracted from the coordinator's `Estimate.watts` dict (same `dict[datetime, int]` structure as Open-Meteo)
**And** the provider's native temporal resolution is detected from the data (hourly for free accounts, higher for paid)
**And** all existing resilience features (staleness detection, health monitoring, re-probing) apply

### Story 3.8: FM-007 — Prediction Confidence Resilience

As TheAdmin,
I want the system to calculate prediction confidence, adjust behavior when confidence is low, and expose confidence metrics,
So that the system charges more conservatively when it's uncertain rather than risking a missed trip.

**Priority:** P2 — degraded optimization | **Size:** M
**Gaps addressed:** G8
**CC Impact:**
- CC-003: `sensor.qs_<person>_prediction_confidence` per person

**Acceptance Criteria:**

**Given** a person's trip prediction has insufficient mileage history (sparse data, holiday period, new commute)
**When** the prediction engine calculates confidence
**Then** an explicit confidence score (0-100%) is computed based on data quality and pattern match
**And** when confidence < configurable threshold, the system falls back to conservative defaults (charge to higher SOC, e.g., use max observed mileage)
**And** the confidence score is logged for observability

**Given** prediction confidence is queried
**When** TheAdmin checks the dashboard
**Then** `sensor.qs_<person>_prediction_confidence` shows current confidence percentage
**And** low confidence is visually distinguishable (sensor attributes include data quality indicators)

### Story 3.9: FM-008 — Car/EV Vendor API Resilience

As TheAdmin,
I want the system to detect stale car API data, offer a manual car mode when the API is unreliable for days, and handle the cascading impact across charging, prediction, and person tracking,
So that an unreliable car vendor API doesn't silently degrade the entire household optimization.

**Priority:** P1 — missed commitment (cascading) | **Size:** XL
**Dependencies:** Builds on Story 3.4 (FM-003 generic device resilience)
**Gaps addressed:** G9, G10, G11, G12
**CC Impact:**
- CC-001: Magali notification ("Your Twingo's data is stale — charging will use conservative estimates. You can enter the current battery level manually.")
- CC-002: `switch.qs_<car>_manual_mode`, `number.qs_<car>_manual_soc`, `button.qs_<car>_remove_from_planning`
- CC-003: `binary_sensor.qs_<car>_api_ok`

**Acceptance Criteria:**

**Given** a car entity's data stops updating (API returns stale timestamps)
**When** the system detects staleness (configurable threshold per entity type)
**Then** the car is flagged as stale across all subsystems
**And** TheAdmin is notified with the specific stale data points
**And** Magali is notified if her car is affected, with plain-language explanation and available actions (CC-001)
**And** `binary_sensor.qs_<car>_api_ok` turns off (CC-003)

**Given** the car API has been stale for an extended period
**When** TheAdmin toggles `switch.qs_<car>_manual_mode` (CC-002)
**Then** the system bypasses the vendor API entirely
**And** TheAdmin can enter current SOC via `number.qs_<car>_manual_soc`
**And** the system tracks SOC by accumulating charger energy delivery (kWh to % using battery capacity)
**And** car presence is inferred from charger state (plugged = home, unplugged = away)
**And** person forecast for this car uses conservative defaults (always assume full charge needed)

**Given** the car API was stale and recovers (fresh timestamps detected)
**When** the system detects recovery
**Then** it suggests exiting manual mode (notification to TheAdmin)
**And** if manual mode is turned off, normal API-based operation resumes
**And** stale subsystem data is refreshed

**Given** TheAdmin wants to remove a car from planning entirely
**When** they press `button.qs_<car>_remove_from_planning` (CC-002)
**Then** the car is excluded from solver constraints and charger assignments
**And** the solver replans without that car

### Story 3.10: CC-003 — Admin Resilience Dashboard

As TheAdmin,
I want a single dashboard view showing all active failures, their severity, and available recovery actions,
So that I can see system health at a glance and take action when needed.

**Size:** M
**Dependencies:** Stories 3.2–3.9 (provide the per-FM sensor entities)
**Gaps addressed:** G20, G21, G22, G23

**Acceptance Criteria:**

**Given** per-FM health sensors exist from Stories 3.2–3.9
**When** TheAdmin opens the resilience dashboard
**Then** Active Alerts section shows current P1/P2/P3 failures with timestamp, affected device, plain-language description
**And** P1 Actions section shows one-click manual overrides (force charge, force load, manual SOC)
**And** P2 Status section shows solar forecast freshness, prediction confidence per person
**And** P3 Status section shows persistence health, entity availability
**And** Recovery Log section shows recent auto-recovered failures with duration and impact

**Given** a failure is detected
**When** notification routing runs
**Then** P1 failures notify Magali + TheAdmin
**And** P2/P3 failures notify TheAdmin only
**And** notification preferences per household member are respected (FR32)

**Given** a centralized failure registry exists
**When** a failure mode is detected or recovers
**Then** the registry tracks: FM-ID, severity, start timestamp, affected devices, current state (active/recovered)
**And** the registry is queryable for dashboard rendering and notification decisions

### Story 3.11: Failure Mode Scenario Tests

As TheDev,
I want implemented test scenarios simulating every failure mode in the catalog under degraded and crisis conditions,
So that every resilience implementation from Stories 3.2–3.9 is verified, including cascading failures.

**Dependencies:** Stories 3.2–3.9

**Acceptance Criteria:**

**Given** the failure catalog from Story 3.1
**When** failure scenarios are executed
**Then** each catalog entry has at least one test scenario exercising its failure signature
**And** fallback behavior is verified (correct degraded operation)
**And** recovery behavior is verified (automatic return to normal)
**And** cascading failures are tested (multiple dependencies failing simultaneously, e.g., car API stale + charger offline)
**And** all scenarios use `@pytest.mark.integration` marker
**And** 100% coverage is maintained

### Story 3.12: Troubleshooting Transparency

As TheAdmin,
I want human-readable explanations when commitments are missed and a decision log showing what the solver decided and why,
So that I can understand and trust the system even when things go wrong.

**Acceptance Criteria:**

**Given** a commitment is missed (e.g., car not charged to target SOC)
**When** TheAdmin checks the explanation (FR25a)
**Then** the system provides specific causes and contributing factors in human-readable language
**And** the explanation includes: what was promised, what was delivered, what went wrong (forecast error, competing priority, device failure, insufficient capacity)
**And** no internal IDs or technical jargon appear in the explanation

**Given** TheAdmin wants to understand solver behavior for a time period
**When** TheAdmin checks the decision log (FR25b)
**Then** the log shows what the solver decided and why for each solver cycle in the period
**And** decisions are presented with context (available solar, active constraints, tariff window, device states)
**And** the log is queryable by time range and device

## Epic 4: Solver Edge Case Improvements

Improvements to how the system handles edge cases in solver planning, person-car allocation, and device orchestration.

### Story 4.1: Default Charge When No Person Assigned (GitHub Issue #30) [DONE]

As Magali,
I want my car to automatically charge to its default target when plugged in but no person is assigned,
so that the car is always usefully charged even when the system has no trip forecast to plan against.

**Acceptance Criteria:**

**Given** a car is plugged in **and** the system assigns no person to it (no forecast match)
**When** the person-car allocation completes
**Then** the car's target charge is set to `car_default_charge`
**And** no departure time constraint is created

**Given** a car is plugged in **and** the user selects "Force no person for car"
**When** the person-car allocation completes
**Then** the car's target charge is set to `car_default_charge`
**And** no departure time constraint is created

**Given** a car has a user-originated charge target or charge time
**When** the system would apply the default charge for no-person
**Then** the user-originated setting is preserved — the system MUST NOT overwrite it

**Given** a car is NOT plugged in and has no person assigned
**When** the person-car allocation completes
**Then** no default charge target is set

