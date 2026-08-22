---
title: QSBattery (ha_model)
slug: ha-battery
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/battery.py
last_verified: 2026-08-20
---

# QSBattery (ha_model) — HA-facing battery integration

## TL;DR

`QSBattery` is the HA-side battery class. It inherits
`HADeviceMixin` plus the pure-domain `home_model/battery.py` model
([home-model-battery.md](home-model-battery.md)) and translates the
solver's `LoadCommand`s into HA service calls: set max-charge /
max-discharge **power** via HA number entities and toggle the single
charge-from-grid switch. For discharge-limiting commands,
`_command_to_values` emits the **outage safety floor**
(`min_discharging_power`, default `0`; QS-349) as the HA
`max_discharging_power` value written to the `max_discharge_number`
entity — `CMD_GREEN_CHARGE_ONLY` and `CMD_FORCE_CHARGE` write the floor
instead of a hard `0` (a nonzero floor *limits* discharge, it does not
disable it; the domain `Battery.max_discharging_power` attribute is
unchanged). Writes go through `_number_entity_target`, which maps the W
value to the entity's unit/min/max/step so the write, the read-back, and
the probe agree (kW-denominated or stepped entities otherwise never
confirm — eternal retry). It applies the domain clamp too, so a consign
above `max_charging_power` confirms at the clamped value (review-fix #03
T2). Snap direction is safety-directed: the discharge **floor** snaps
**up** (a minimum is never lowered to 0), while a **maximum** — the
max-discharge restore and the charge limit — snaps **down** (a limit is
never raised past its configured cap; review-fix #01 S2, #02 R1/R5, #03
T1/T3). **Snap-policy priority** (review-fix #04 U1/U2): the safety floor
(`domain_min`) still wins **upward** and the configured hardware max
(`domain_max`) wins **downward** even after the step snap — so a floor is
never snapped below itself and a max is never snapped above the configured
limit, accepting a non-step-aligned write at either bound (HA core
validates min/max, not step alignment). Non-numeric **or** mutually
inconsistent (`min > max`) entity attributes are treated as absent
(review-fix #03 T7 / #04 U6). A landed value that diverges from the
request by more than one step — in **either** direction (an entity `min`
forcing more, or an entity `max` forcing less than the safety floor) — is
logged once per distinct `(entity, landed, direction)` (review-fix #03 T8
/ #04 U3/U5). While the number entity is `unknown`/`unavailable` the write
is skipped and retried next cycle, so a momentarily-unavailable kW entity
never gets a raw-W write (U4). Because the U1/U2 policy can write a
non-step-aligned value at a domain bound, the **probe** accepts a
read-back within one entity step of the expected landed value when a step
is advertised (a step-quantizing integration echoes the aligned
neighbour) and requires exact equality otherwise — this device-echo
tolerance is distinct from the divergence-warning tolerance (review-fix
#05 V1). The probe is a pure read and never emits the divergence warning
(V6); the warning tolerance is direction-aware — a shortfall on the
snap-up path is always surfaced because ceil cannot legitimately land
below the request (V2) — and it re-warns after a confirmed probe clears
the latch (V5). There is no discharge-enable
switch and no SOC-setpoint number. It attaches HA state probes for SOC
(`charge_percent_sensor`) and the combined charge/discharge power
sensor so the solver always sees fresh state.

## When you need this concept

- Integrating a new battery vendor (e.g., a new inverter brand).
- Changing how SOC / power are read from HA entities.
- Modifying the max-power number / charge-from-grid switch wiring.
- Debugging "the battery isn't following the plan" issues.

## Core idea

The `HADeviceMixin` bridge pattern: `QSBattery` extends both
`HADeviceMixin` and the domain `Battery` class. State flows in
through `HADeviceMixin.add_to_history()` (SOC, power, capacity-derived
signals); commands flow out through `execute_command()` →
`hass.services.async_call(...)`. The probe-update cycle handles
external state changes: if a human changes the inverter settings
externally, `probe_if_command_set()` reads the HA entities and returns
whether the currently-expected command is actually in effect —
`bool | None`, where `None` means a *configured* entity's state could
not be read (HA cold start, inverter offline), i.e. "inconclusive",
not "command absent". Unconfigured entities are excluded from the
check entirely — `_command_to_values` nulls their expected value, so
they match vacuously (a fully-unconfigured battery probes `True`, not
`None`). `QSBattery` has no
`external_user_initiated_state`; that attribute belongs to
`AbstractLoad`, which `QSBattery` does not inherit from.

## Key types / structures

- `QSBattery(HADeviceMixin, Battery)` — the bridge class.
- `execute_command(time, command)` — maps `LoadCommand` →
  `hass.services.async_call(number/switch)`.
- `probe_if_command_set(time, command)` — reads HA entity state to
  verify command landed.
- HA entities wired: **probed** — `charge_percent_sensor` (SOC) and
  `charge_discharge_sensor` (combined power); **read on demand** —
  `max_charge_number` + `max_discharge_number` (max-power numbers) and
  `charge_from_grid_switch` (the only switch), bare entity IDs read
  via `hass.states.get` and never entering the probe/history
  machinery.

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
