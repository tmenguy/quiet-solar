---
title: External override
slug: external-override
kind: use-case
last_verified: 2026-05-19
---

# External override — somebody else is driving

## TL;DR

A human walks over to the charger and presses the physical "stop"
button. Or another HA integration sends a command to the same
charger. Or the device's manufacturer app interferes. Whatever the
cause, quiet-solar **detects** the discrepancy between the command
it sent and the device's observed state, and **steps back** —
existing constraints pause, no new commands are issued, automation
resumes only when control is back. This is the "embrace uncertainty"
principle made concrete: quiet-solar doesn't fight the human.

## When you need this use case

- Implementing or modifying a device's `probe_if_command_set()`.
- Designing a feature that must behave correctly under external
  control.
- Debugging "quiet-solar overrides my manual setting" issues —
  almost always a detection gap.
- Working on the cool-down before automation resumes.

## End-to-end sequence

```text
1. Solver decided: car charges at 11kW between 22:00 and 06:00.
   Charger executes — observed power matches plan.
   ↓
2. At 23:00, a human (or another integration) issues "stop
   charging" to the charger directly.
   ↓
3. Next state-polling cycle (~4s): observed charger power = 0W;
   the last quiet-solar command was "11kW" → divergence detected.
   ↓
4. probe_if_command_set() returns False; HADeviceMixin.update_states
   sets external_user_initiated_state on the load.
   ↓
5. AbstractLoad: existing constraints paused; get_for_solver_
   constraints() returns no active constraints for this load.
   ↓
6. Next solver cycle: this load is excluded from the allocation.
   The household keeps operating around it.
   ↓
7. ... cool-down window passes; state polling observes the load
   returned to a state consistent with quiet-solar control ...
   ↓
8. external_user_initiated_state cleared; constraints resume; next
   solver cycle re-includes the load in the plan.
```

## What "consistent with quiet-solar control" means

The cool-down doesn't just wait for a timer; it observes:

- The device's state hasn't changed in a way quiet-solar didn't
  initiate for `external_user_initiated_cooldown_s` seconds.
- The device is back in a state quiet-solar could have produced
  (e.g., off, idle, or accepting a fresh command).

If both hold, quiet-solar resumes.

## Difference vs user override

| | External control | User override |
|---|---|---|
| Triggered by | Human or other integration operating the device directly | Human telling quiet-solar to do something different |
| Detection | State divergence between observed and expected | UI / API call into quiet-solar |
| Effect | Pause automation; let the human drive | Update the constraint set; let the solver replan |
| Recovery | Auto-recover when device returns to expected behaviour | Override expires (e.g., next solver cycle); constraint regenerates |

See [user-override.md](../concepts/user-override.md) for the
counterpart.

## What each layer contributes

| Layer | Contribution |
|---|---|
| External control detection ([external-control-detection.md](../concepts/external-control-detection.md)) | The detection logic and the flag. |
| HADeviceMixin ([ha-device-mixin.md](../concepts/ha-device-mixin.md)) | The state-divergence check during `update_states()`. |
| AbstractLoad ([load-base.md](../concepts/load-base.md)) | Hosts `external_user_initiated_state`; gates the constraint surface. |
| Solver ([solver.md](../concepts/solver.md)) | Excludes externally-controlled loads from the plan. |

## Common mistakes when modifying this path

- Implementing a lenient `probe_if_command_set()` that returns True
  for any state. External control then goes undetected.
- Forgetting to clear the flag on recovery. The load stays
  permanently external; automation never resumes.
- Conflating with user override and routing through the override
  notification path. External is *silent* by design — the user is
  driving, no need to notify.
- Triggering a "command failed" alert instead. The flow is "user
  is driving", not "system error".

## See also

- [../concepts/external-control-detection.md](../concepts/external-control-detection.md)
  — the mechanics.
- [../concepts/user-override.md](../concepts/user-override.md) —
  the related-but-distinct flow.
- [magali-plugs-in-car.md](magali-plugs-in-car.md) — what happens
  if Magali unplugs mid-charge (a subspecies of external control).
- [../principles/observe-predict-optimize.md](../principles/observe-predict-optimize.md)
  — "embrace uncertainty" is the design philosophy.
