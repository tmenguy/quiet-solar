---
title: Two-layer boundary
slug: two-layer-boundary
kind: principle
last_verified: 2026-05-19
---

# Two-layer boundary — `home_model/` vs `ha_model/`

## TL;DR

`home_model/` is **pure-Python domain** code with **zero Home
Assistant imports**. `ha_model/` is the HA-integration layer that
bridges the two — every HA device class inherits both
`HADeviceMixin` (HA half) and a `home_model/` domain class
(pure-Python half). This is a non-negotiable architectural rule:
violations cause CI failures, force test setups to boot HA for
domain logic, and erode the project's testability over time.

## When you need this principle

- Choosing which file a new piece of logic lives in.
- Adding a new device type (you'll touch both layers).
- Writing tests — domain code uses FakeHass and factories; HA code
  uses real HA fixtures.
- Reviewing PRs — any `from homeassistant.*` import in
  `home_model/` is a must-fix.

## Core idea

**Why two layers?**

- **Testability**: `home_model/` tests run in milliseconds without
  HA boot. Without the split, every solver test would need a real
  HA stack.
- **Portability**: if quiet-solar ever needs to live outside HA
  (Home Assistant fork, alternative platform), the domain core is
  ready.
- **Cognitive load**: developers reasoning about constraints don't
  need to understand HA's entity model.

**Mechanics**:

- `home_model/` files: `commands.py`, `constraints.py`, `load.py`,
  `solver.py`, `battery.py`, `home_utils.py`. None import
  `homeassistant.*`.
- `ha_model/` files: every other device file. Each device class
  inherits both `HADeviceMixin` and a domain class — the bridge
  pattern.
- `HADeviceMixin.add_to_history()` is the **only** path HA state
  enters the domain. `execute_command()` is the **only** path
  commands leave the domain.

## Concrete implications

- **Domain code receives data as parameters**, never as HA entity
  reads.
- **HA code never knows about constraint priorities** beyond
  passing them through.
- **Test fixtures split along the boundary**: `tests/conftest.py`
  for the domain layer (FakeHass + factories); `tests/ha_tests/`
  for the integration layer (real HA fixtures).
- **A function that needs both domains lives in `ha_model/`**, not
  `home_model/`. The boundary is asymmetric: `home_model/` knows
  nothing about HA; `ha_model/` knows about both.

## Common mistakes

- Importing `homeassistant.*` in `home_model/` "just to grab a
  helper". The helper belongs in a `home_model/`-friendly module
  or in `ha_model/` as a wrapper.
- Writing logic in `ha_model/battery.py` that doesn't need HA. It
  belongs in `home_model/battery.py`.
- Adding a constant in `homeassistant.const` rather than
  `quiet_solar/const.py`. The two `const.py` files are independent
  worlds.
- Testing domain logic via real HA fixtures because "it's easier".
  The fast layer exists for a reason — using the slow layer for
  domain logic erodes the dev-loop story.

## See also

- [../concepts/load-base.md](../concepts/load-base.md) — the domain
  base.
- [../concepts/ha-device-mixin.md](../concepts/ha-device-mixin.md)
  — the bridge.
- [../concepts/testing-layers.md](../concepts/testing-layers.md) —
  the test infrastructure that mirrors the boundary.
- [../../workflow/project-rules.md](../../workflow/project-rules.md)
  — the "Architecture constraints" section pins this rule.
