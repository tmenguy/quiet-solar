---
title: Glossary
slug: glossary
kind: principle
last_verified: 2026-05-19
---

# Glossary

Domain terms that appear throughout the codebase, the concept docs,
and PR discussions. New term? Add it here so future agents don't have
to grep the entire tree to figure out what an `ack` is.

## Solver vocabulary

- **MANDATORY_AS_FAST_AS_POSSIBLE** — highest-priority constraint
  tier (score 9). The solver schedules these first, minimizing
  completion time regardless of solar production.
- **MANDATORY_END_TIME** — priority tier 7. Must complete by a
  deadline; the solver picks the cheapest power slots that finish in
  time.
- **BEFORE_BATTERY_GREEN** — priority tier 5. Runs before the battery
  switches to solar-only mode (i.e., during the morning when grid
  arbitrage still beats holding battery for evening).
- **FILLER** — priority tier 3. Uses surplus energy only; never
  triggers grid imports.
- **FILLER_AUTO** — priority tier 1, lowest. Automatic / opportunistic
  filler — runs whenever any surplus exists.
- **LoadConstraint** — a time-windowed demand on a load with progress
  tracking (initial → current → target).
- **LoadCommand** — a discrete action at a specific power level
  (one of 10 command types, ordered by score).
- **PeriodSolver** — the strategic optimization engine. Plans in
  15-minute slots. Defined in `home_model/solver.py`.
- **SOLVER_STEP_S** — 900 seconds (15 minutes). The solver's
  discretisation step. Constant in `const.py`. Do not touch.

## Command lifecycle

- **pending** — the solver has decided a command should run; it is
  queued for launch.
- **launched / running_command** — the command was dispatched to HA
  (service call sent) but not yet confirmed.
- **acked / current_command** — the device confirmed the command via
  `probe_if_command_set()`; the command is now the device's reality.
- **stacking** — when a device is busy, additional pending commands
  stack instead of overwriting.

## Device & topology

- **AbstractDevice** — base for all controllable devices (in
  `home_model/load.py`). Configuration, command lifecycle, state
  tracking, switching-cost protection.
- **AbstractLoad** — extends AbstractDevice. Adds constraint
  management and the solver interface.
- **HADeviceMixin** — the bridge layer (in `ha_model/device.py`).
  Every HA device class inherits this + a domain class.
- **QSHome** — the orchestrator (`ha_model/home.py`). Root of the
  dynamic-group tree.
- **QSDynamicGroup** — a non-leaf node in the amp budgeting tree
  (`ha_model/dynamic_group.py`).
- **PilotedDevice** — a device that pilots another device
  (e.g. a heat pump piloting an auxiliary heater).
- **External control detection** — the system noticed a state change
  it did not initiate, so a human or another integration is driving
  the device. Quiet-solar steps back.

## Charger budgeting

- **Charger Dynamic Budgeting** — the tactical layer (in
  `ha_model/charger.py`). Manages real-time power distribution within
  circuit constraints. Can override the strategic solver.
- **Adaptation window** — `CHARGER_ADAPTATION_WINDOW_S = 45` seconds.
  Chargers must be stable this long before rebalancing.
- **3P / 1P** — three-phase vs single-phase charging. Phase switching
  redistributes load across phases (`1P@32A → 3P@11A` reduces
  per-phase amps).
- **Dampening** — using real measured power (not just the target) to
  drive future estimates.
- **Staged transitions** — large budget changes split across two
  cycles to prevent transient overages.
- **charge_score** — per-charger priority. Higher score wins when
  budgets conflict.

## Battery & solar

- **SOC** — state of charge, %. Battery's current energy as a fraction
  of capacity.
- **DC-coupled** — the battery shares an inverter with the PV array;
  charging from PV avoids the DC↔AC round trip.
- **Multi-provider solar** — quiet-solar combines multiple forecast
  providers (Solcast, OpenMeteo) with dampening.
- **Off-grid mode** — automatic load shedding triggered by grid
  outage detection.

## Constraint metadata

- **load_info** — dict on every constraint tracking its
  **originator** (`user_override`, `agenda`, `prediction`, `system`).
  Used for notification routing and conflict resolution.
- **agenda** — a calendar-derived constraint (e.g., HA calendar
  entity).
- **prediction** — a constraint derived from historical patterns
  (e.g., trip prediction for a person/car pair).
- **user_override** — a constraint a human created to overrule the
  prediction or the schedule.

## Hysteresis & switching cost

- **num_max_on_off** — daily on/off budget. Defaults vary per device
  type.
- **CHANGE_ON_OFF_STATE_HYSTERESIS_S** — 10 minutes. Minimum delay
  between state changes for a single device.
- **Multi-pass adaptation** — the solver tries free transitions
  first, only spending switching budget when necessary.

## Persons & cars

- **Trip prediction** — based on GPS + mileage history (31-day
  window). Generates constraints for the person's car.
- **Person-car allocation** — which car is "assigned" to which
  person, affects prediction targeting.

## Workflow & infrastructure

- **HACS** — Home Assistant Community Store. Quiet-solar's
  distribution channel.
- **HA** — Home Assistant. The host platform.
- **FakeHass** — the lightweight test double for the HA layer
  (`tests/conftest.py` + `tests/factories.py`).
- **Quality gate** — `scripts/qs/quality_gate.py`. Runs pytest 100%
  cov + ruff + mypy + translations. Required before every commit.
- **Static agents** — agents whose body is committed to the repo
  (`.claude/agents/`, `.cursor/agents/`, `.opencode/agents/`),
  versus per-task rendering (legacy approach in `legacy/`).
- **Worktree** — a separate working directory tied to the same git
  repo. Each task runs in its own worktree (`QS_<N>` branch).
- **`covers:` frontmatter** — list of source files a doc claims to
  describe. Validated by `scripts/qs/check_doc_drift.py`.
- **Drift checker** — `scripts/qs/check_doc_drift.py`. Catches docs
  that fall out of sync with their `covers:` source.

## See also

- [index.md](index.md) — entry point to the whole `docs/agents/`
  hierarchy.
- [../workflow/project-context.md](../workflow/project-context.md)
  — the 42-rule code-style reference.
