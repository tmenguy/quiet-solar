---
title: docs/agents/ index
slug: index
kind: principle
last_verified: 2026-05-19
---

# docs/agents/ — addressable knowledge for AI agents

This tree is the **single entry point** for plan / implement / review
agents that need a slice of quiet-solar domain knowledge. Every file
is short and scoped to one topic; pull the file, get the answer,
move on. Drift between docs and code is caught by
`scripts/qs/check_doc_drift.py` (see "Maintenance contract" at the
bottom).

For broader architectural context (decisions, validation tables, CI
strategy), see [../product/architecture.md](../product/architecture.md).
For the workflow itself, see
[../workflow/overview.md](../workflow/overview.md).

## Lookup by source file

Modifying a file under `custom_components/quiet_solar/`? Pull the
docs covering it before you write the change.

| If you're modifying… | Read |
|---|---|
| `home_model/commands.py` | [concepts/commands.md](concepts/commands.md) |
| `home_model/constraints.py` | [concepts/constraints.md](concepts/constraints.md) |
| `home_model/solver.py` | [concepts/solver.md](concepts/solver.md) |
| `home_model/load.py` (AbstractDevice / AbstractLoad) | [concepts/load-base.md](concepts/load-base.md) |
| `home_model/load.py` (PilotedDevice) | [concepts/piloted-device-and-heat-pump.md](concepts/piloted-device-and-heat-pump.md) |
| `home_model/load.py` (external control + user override) | [concepts/external-control-detection.md](concepts/external-control-detection.md), [concepts/user-override.md](concepts/user-override.md) |
| `home_model/battery.py` | [concepts/home-model-battery.md](concepts/home-model-battery.md) |
| `ha_model/battery.py` | [concepts/ha-battery.md](concepts/ha-battery.md) |
| `ha_model/charger.py` | [concepts/charger-budgeting.md](concepts/charger-budgeting.md) |
| `ha_model/dynamic_group.py` | [concepts/dynamic-group-tree.md](concepts/dynamic-group-tree.md) |
| `ha_model/home.py` (cycles + QSHomeMode) | [concepts/qs-home-orchestrator.md](concepts/qs-home-orchestrator.md), [concepts/off-grid-mode.md](concepts/off-grid-mode.md), [concepts/notification-routing.md](concepts/notification-routing.md) |
| `ha_model/device.py` (HADeviceMixin) | [concepts/ha-device-mixin.md](concepts/ha-device-mixin.md) |
| `ha_model/person.py`, `ha_model/car.py` | [concepts/person-trip-prediction.md](concepts/person-trip-prediction.md) |
| `ha_model/heat_pump.py`, `ha_model/climate_controller.py` | [concepts/piloted-device-and-heat-pump.md](concepts/piloted-device-and-heat-pump.md) |
| `ha_model/bistate_duration.py`, `ha_model/on_off_duration.py`, `ha_model/pool.py` | [concepts/bistate-duration-devices.md](concepts/bistate-duration-devices.md) |
| `ha_model/solar.py` | [concepts/solar-providers.md](concepts/solar-providers.md) |
| `config_flow.py`, `__init__.py`, `data_handler.py`, `const.py` | [concepts/config-and-setup-flow.md](concepts/config-and-setup-flow.md) |
| `tests/conftest.py`, `tests/factories.py`, `tests/ha_tests/` | [concepts/testing-layers.md](concepts/testing-layers.md) |

## Lookup by concept

| Concept | Kind | File |
|---|---|---|
| LoadCommand | concept | [concepts/commands.md](concepts/commands.md) |
| LoadConstraint | concept | [concepts/constraints.md](concepts/constraints.md) |
| PeriodSolver | concept | [concepts/solver.md](concepts/solver.md) |
| AbstractDevice / AbstractLoad | concept | [concepts/load-base.md](concepts/load-base.md) |
| PilotedDevice + heat pump | concept | [concepts/piloted-device-and-heat-pump.md](concepts/piloted-device-and-heat-pump.md) |
| Battery (home_model) | concept | [concepts/home-model-battery.md](concepts/home-model-battery.md) |
| QSBattery (ha_model) | concept | [concepts/ha-battery.md](concepts/ha-battery.md) |
| Charger dynamic budgeting | concept | [concepts/charger-budgeting.md](concepts/charger-budgeting.md) |
| QSDynamicGroup amp-budget tree | concept | [concepts/dynamic-group-tree.md](concepts/dynamic-group-tree.md) |
| QSHome orchestrator | concept | [concepts/qs-home-orchestrator.md](concepts/qs-home-orchestrator.md) |
| HADeviceMixin bridge | concept | [concepts/ha-device-mixin.md](concepts/ha-device-mixin.md) |
| Person / car / trip prediction | concept | [concepts/person-trip-prediction.md](concepts/person-trip-prediction.md) |
| Off-grid mode | concept | [concepts/off-grid-mode.md](concepts/off-grid-mode.md) |
| External control detection | concept | [concepts/external-control-detection.md](concepts/external-control-detection.md) |
| Notification routing | concept | [concepts/notification-routing.md](concepts/notification-routing.md) |
| Bistate-duration devices (pool / on-off) | concept | [concepts/bistate-duration-devices.md](concepts/bistate-duration-devices.md) |
| User override | concept | [concepts/user-override.md](concepts/user-override.md) |
| Solar providers + dampening | concept | [concepts/solar-providers.md](concepts/solar-providers.md) |
| Config flow & setup | concept | [concepts/config-and-setup-flow.md](concepts/config-and-setup-flow.md) |
| Testing layers (FakeHass vs real HA) | concept | [concepts/testing-layers.md](concepts/testing-layers.md) |
| Observe → predict → optimize | principle | [principles/observe-predict-optimize.md](principles/observe-predict-optimize.md) |
| Two-layer boundary | principle | [principles/two-layer-boundary.md](principles/two-layer-boundary.md) |
| Hysteresis & switching cost | principle | [principles/hysteresis-and-switching-cost.md](principles/hysteresis-and-switching-cost.md) |
| Strategic vs tactical control | principle | [principles/strategic-tactical-control.md](principles/strategic-tactical-control.md) |
| Event-driven with periodic fallback | principle | [principles/event-driven-with-fallback.md](principles/event-driven-with-fallback.md) |

## Lookup by persona / use case

| Persona / use case | File | Cross-links |
|---|---|---|
| TheAdmin | [personas/the-admin.md](personas/the-admin.md) | [config-and-setup-flow.md](concepts/config-and-setup-flow.md), [external-override.md](use-cases/external-override.md) |
| TheDev | [personas/the-dev.md](personas/the-dev.md) | [testing-layers.md](concepts/testing-layers.md), `../workflow/overview.md` |
| Magali | [personas/magali.md](personas/magali.md) | [magali-plugs-in-car.md](use-cases/magali-plugs-in-car.md), [user-override.md](concepts/user-override.md) |
| Magali plugs in her car | [use-cases/magali-plugs-in-car.md](use-cases/magali-plugs-in-car.md) | [person-trip-prediction.md](concepts/person-trip-prediction.md), [solver.md](concepts/solver.md), [charger-budgeting.md](concepts/charger-budgeting.md) |
| Solar surplus allocation | [use-cases/solar-surplus-allocation.md](use-cases/solar-surplus-allocation.md) | [solver.md](concepts/solver.md), [solar-providers.md](concepts/solar-providers.md), [ha-battery.md](concepts/ha-battery.md) |
| Cheap-grid charging | [use-cases/cheap-grid-charging.md](use-cases/cheap-grid-charging.md) | [solver.md](concepts/solver.md), [constraints.md](concepts/constraints.md) |
| Off-grid kicks in | [use-cases/off-grid-kicks-in.md](use-cases/off-grid-kicks-in.md) | [off-grid-mode.md](concepts/off-grid-mode.md), [notification-routing.md](concepts/notification-routing.md) |
| External override | [use-cases/external-override.md](use-cases/external-override.md) | [external-control-detection.md](concepts/external-control-detection.md), [user-override.md](concepts/user-override.md) |
| Solver replans on state change | [use-cases/solver-replans-on-state-change.md](use-cases/solver-replans-on-state-change.md) | [event-driven-with-fallback.md](principles/event-driven-with-fallback.md), [qs-home-orchestrator.md](concepts/qs-home-orchestrator.md) |

## Out of tree — other docs

- [glossary.md](glossary.md) — ~40 domain terms.
- [lsp-evaluation.md](lsp-evaluation.md) — should agents pair with a
  Python LSP server? (Decision: defer.)
- [../workflow/project-rules.md](../workflow/project-rules.md) —
  the quality-gate rules and the "Doc maintenance" subsection.
- [../workflow/project-context.md](../workflow/project-context.md)
  — the 42-rule code-style reference.
- [../product/architecture.md](../product/architecture.md) — the
  decisions log + validation tables + CI strategy (everything not
  covered by this tree).

## Maintenance contract

This tree stays in sync with the code via three layered defences:

- **`covers:` frontmatter** — every concept doc lists the source
  files it claims to describe. The drift checker
  (`scripts/qs/check_doc_drift.py`) validates that every path
  exists; missing paths → exit 2.
- **Agent body steps** — the create-plan, implement-task,
  implement-setup-task, and review-task agents each invoke
  `check_doc_drift.py` at the right phase. See
  [../workflow/project-rules.md](../workflow/project-rules.md) ("Doc
  maintenance" subsection) for the contract; the agent bodies under
  `.claude/agents/`, `.cursor/agents/`, `.opencode/agents/` are the
  implementation. Parity across the three harnesses is pinned by
  `tests/qs/agents/test_doc_maintenance_parity.py`.
- **`last_verified` frontmatter** — every doc records the date the
  content was last cross-checked. A reviewer can say "still
  accurate after the last refactor" by bumping the date in the
  same PR.
