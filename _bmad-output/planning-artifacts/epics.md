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

### Epic 3: Failure Resilience & Transparency
TheAdmin can trust the system to handle failures gracefully and understand exactly what happened when things go wrong. Every external dependency has a documented failure path with fallback and recovery. Fed directly by Epic 2's failure catalog — Epic 2 documents what can fail, Epic 3 implements how the system responds.
**FRs:** FR21, FR22, FR23, FR24, FR25a, FR25b
**ARs:** AR5
**NFRs:** NFR8, NFR9, NFR10, NFR11, NFR12, NFR13, NFR21
**Scope:** MVP | **Dependencies:** Builds on Epic 1, 2

### Epic 4: Solver Edge Case Improvements *(stories deferred — detail when Epics 1-3 complete)*
**FRs:** FR1-FR5b | **ARs:** AR6 | **NFRs:** NFR1-NFR4 | **Scope:** MVP | **Depends on:** Epic 1, 2

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

### Story 1.8: AI-Assisted PR Review with Interactive Feedback Loop

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

### Story 1.9: Mobile-First Autonomous GitHub Flow

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

### Story 3.2: Resilience Implementation

As TheAdmin,
I want the system to detect every cataloged failure mode and execute the planned fallback and recovery behavior,
So that failures are handled gracefully without manual intervention.

**Acceptance Criteria:**

**Given** a solar forecast API failure (timeout or HTTP error)
**When** the system detects the failure
**Then** it uses the last successful forecast, falling back to historical patterns if stale > 6h (NFR13)
**And** it auto-retries on the next forecast cycle and recovers automatically (FR23)

**Given** a charger communication failure (command not ACKed)
**When** the system detects the failure
**Then** it retries with configurable count and exponential backoff (NFR12)
**And** after max retries, marks charger unavailable and notifies TheAdmin (FR29)
**And** it auto-recovers when the charger responds and re-includes it in the next solver cycle

**Given** solver infeasibility (no valid plan)
**When** the system detects infeasibility
**Then** mandatory constraints get priority, filler constraints are dropped, battery held at current SOC (NFR11)
**And** it re-solves on the next triggered evaluation

**Given** an HA restart (power failure, update, crash)
**When** HA becomes ready
**Then** the system recovers previous state from persisted sensor data (NFR8, NFR9)
**And** optimization resumes within one solver evaluation cycle without manual intervention

**Given** numpy persistence corruption (partial write, disk full)
**When** the system detects corrupted data
**Then** it operates without historical data using conservative estimates
**And** the ring buffer rebuilds over time as new data accumulates

**Given** stale device state (communication stall)
**When** the system detects staleness (FR24)
**Then** it makes conservative decisions rather than acting on outdated information

### Story 3.3: Failure Mode Scenario Tests

As TheDev,
I want implemented test scenarios simulating every failure mode in the catalog under degraded and crisis conditions,
So that every resilience implementation from Story 3.2 is verified.

**Acceptance Criteria:**

**Given** the failure catalog from Story 3.1
**When** failure scenarios are executed
**Then** each catalog entry has at least one test scenario exercising its failure signature
**And** fallback behavior is verified (correct degraded operation)
**And** recovery behavior is verified (automatic return to normal)
**And** cascading failures are tested (multiple dependencies failing simultaneously)
**And** all scenarios use `@pytest.mark.integration` marker
**And** 100% coverage is maintained

### Story 3.4: Troubleshooting Transparency

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
