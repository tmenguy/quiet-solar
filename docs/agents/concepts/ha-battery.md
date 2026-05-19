---
title: QSBattery (ha_model)
slug: ha-battery
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/battery.py
last_verified: 2026-05-19
---

# QSBattery (ha_model) — HA-facing battery integration

## TL;DR

`QSBattery` is the HA-side battery class. It inherits
`HADeviceMixin` plus the pure-domain `home_model/battery.py` model
([home-model-battery.md](home-model-battery.md)) and translates the
solver's `LoadCommand`s into HA service calls (write SOC setpoint via
an HA number entity, toggle charge/discharge enable via switch
entities). It attaches HA state probes for SOC, charge power, and
discharge power so the solver always sees fresh state.

## When you need this concept

- Integrating a new battery vendor (e.g., a new inverter brand).
- Changing how SOC / power are read from HA entities.
- Modifying the discharge-enable / charge-enable switch wiring.
- Debugging "the battery isn't following the plan" issues.

## Core idea

The `HADeviceMixin` bridge pattern: `QSBattery` extends both
`HADeviceMixin` and the domain `Battery` class. State flows in
through `HADeviceMixin.add_to_history()` (SOC, power, capacity-derived
signals); commands flow out through `execute_command()` →
`hass.services.async_call(...)`. The probe-update cycle handles
external state changes: if a human turns off discharge via the
inverter UI, `probe_if_command_set()` detects the discrepancy and the
device's `external_user_initiated_state` is updated.

## Key types / structures

- `QSBattery(HADeviceMixin, Battery)` — the bridge class.
- `execute_command(time, command)` — maps `LoadCommand` →
  `hass.services.async_call(number/switch)`.
- `probe_if_command_set(time, command)` — reads HA entity state to
  verify command landed.
- HA entities attached: SOC sensor, charge/discharge power sensors,
  charge enable + discharge enable switches, target-SOC number.

## Lifecycle

```text
Config flow → QSBattery created → attach HA probes
  ↓
update_states() reads HA → updates Battery SOC, power
  ↓
solver computes LoadCommand
  ↓
execute_command() → hass.services.async_call(...)
  ↓
probe_if_command_set() observes state change → acked
```

## Common mistakes

- Adding battery logic in `ha_model/battery.py` that doesn't need
  HA. It belongs in `home_model/battery.py`.
- Calling `hass.services.async_call` synchronously. All HA I/O is
  async; blocking calls freeze the event loop.
- Forgetting to attach a probe for a state quiet-solar needs to
  trust. If the SOC sensor isn't wired, the solver plans on stale
  data.

## See also

- [home-model-battery.md](home-model-battery.md) — the pure-domain
  counterpart.
- [ha-device-mixin.md](ha-device-mixin.md) — the bridge layer.
- [commands.md](commands.md) — the action language `execute_command`
  consumes.
- [../use-cases/solar-surplus-allocation.md](../use-cases/solar-surplus-allocation.md)
  — surplus → battery charging end-to-end.
