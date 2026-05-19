---
title: QSDynamicGroup (amp-budget tree)
slug: dynamic-group-tree
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/dynamic_group.py
last_verified: 2026-05-19
---

# QSDynamicGroup — hierarchical amp budgeting

## TL;DR

`QSDynamicGroup` is the tree topology that enforces phase-by-phase
amp budgets across the whole house. `QSHome` is the root; dynamic
groups are interior nodes; chargers are leaves. Budgets are always
expressed as `[phase1, phase2, phase3]` arrays. Validation is
**recursive**: a leaf can pass its local check yet still fail the
parent group's circuit limit, so every command launch walks the tree
from leaf to root.

## When you need this concept

- Adding a new circuit topology (e.g., a sub-panel with its own
  breaker).
- Modifying budget validation, phase math, or amp diff calculations.
- Debugging "the system tripped a breaker" issues — start here.
- Working on the staged-transition logic that interacts with the
  tree.

## Core idea

Two validation methods:

- `is_delta_current_acceptable()` — planning-phase check: "will this
  change fit in my budget?". Used before scheduling.
- `is_current_acceptable_and_diff()` — execution-phase check: takes
  the worst-case max of (actual measured current + estimated
  consumption). Used right before a command is launched.

Per-phase operations on amp arrays:

- `add_amps(a, b)`, `diff_amps(a, b)`, `is_amps_greater(a, b)`,
  `min_amps(a, b)`, `max_amps(a, b)`.

Tree topology:

```text
QSHome (root, phase limit = main breaker)
├── QSDynamicGroup "EV charging circuit" (limit = sub-breaker)
│   ├── QSChargerGeneric "car_1"
│   └── QSChargerOCPP "car_2"
└── QSChargerWallbox "wallbox_main"
```

Every command launch validates from leaf to root.

## Key types / structures

- `QSDynamicGroup(HADeviceMixin, AbstractDevice)` — tree node.
- `is_delta_current_acceptable(delta_amps)` — planning check.
- `is_current_acceptable_and_diff(target_amps)` — execution check
  with measured worst case.
- Phase-arithmetic helpers — all operate on length-3 arrays.

## Common mistakes

- Working with scalar amps when the rest of the system expects
  3-phase arrays. Treat amps as `[phase1, phase2, phase3]`
  everywhere.
- Skipping the recursive walk. A leaf-only validation can pass while
  the home root is already at its main-breaker limit.
- Confusing the two validation methods. Planning-phase is for "will
  this fit"; execution-phase is for "is it safe right now".
- Forgetting that phase switching (1P→3P) reduces per-phase load but
  doesn't change total power. The tree validation accounts for
  that.

## See also

- [charger-budgeting.md](charger-budgeting.md) — the algorithm that
  walks this tree.
- [qs-home-orchestrator.md](qs-home-orchestrator.md) — the root.
- [../principles/strategic-tactical-control.md](../principles/strategic-tactical-control.md)
  — why amp budgeting is tactical, not strategic.
