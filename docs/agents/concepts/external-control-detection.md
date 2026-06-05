---
title: External control detection
slug: external-control-detection
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
  - custom_components/quiet_solar/ha_model/device.py
last_verified: 2026-06-05
---

# External control detection

## TL;DR

If a device's state changes in a way quiet-solar didn't request —
because a human pressed a physical button, another HA integration
sent a command, or the device's own app issued an instruction —
quiet-solar **detects this** and steps back rather than fighting the
human. The detection logic lives in `home_model/load.py`
(`external_user_initiated_state`) and is set during
`HADeviceMixin.update_states()` in `ha_model/device.py`.

## When you need this concept

- Implementing a new device's `probe_if_command_set()` or
  `update_states()`.
- Designing a feature that interacts with externally-controlled
  devices.
- Debugging "quiet-solar keeps overriding my manual setting" issues.
- Working on the boost-only / free-solar-only flows.

## Core idea

Every cycle, `update_states()` compares observed HA state to the
state quiet-solar expected based on the most recently launched
command. If they don't match — and the mismatch can't be explained
by transient noise — the device was controlled externally.

When external control is detected:

1. `external_user_initiated_state` is set on the load.
2. The solver sees the device as effectively unavailable for
   automation: existing constraints are paused, no new commands
   are issued.
3. The behaviour persists for a configurable cool-down — long
   enough that the human knows they have the device, short enough
   that quiet-solar resumes once the external session ends.
4. Recovery is automatic: once observed state stabilises in a way
   consistent with quiet-solar control, the flag clears.

This is the structural defence behind "embrace uncertainty" —
quiet-solar doesn't assume it has exclusive control of the device.

**Detection rules (QS-256).** Three guards keep false positives out
of the bistate detection in
`QSBiStateDuration.check_load_activity_and_constraints`:

1. **Causality** — a mismatch counts only if the entity state's
   `last_changed` is NEWER than the load's
   `last_command_execution_time` (set on real service-call executions
   and anchored at storage restore when a `current_command` is
   restored). A stale state — e.g. a lagging template-switch mirror
   right after an HA restart — is not a user action. Conservative
   edge: `last_changed = None` while an anchor exists cannot prove
   freshness → no override is classified.
2. **Cooldown** — after an override resets, no new override is
   classified for `USER_OVERRIDE_STATE_BACK_DURATION_S` (180s),
   bounded by half the override window.
3. **Constraint-driven end** — an override to the idle state pushes a
   `TimeBasedHoldOffConstraint` so the solver natively sees the
   pinned-off window and the override ends through the constraint-ack
   path rather than relying purely on the timer.

## Key types / structures

- `AbstractLoad.external_user_initiated_state` — the flag.
- `HADeviceMixin.update_states()` — sets the flag based on state
  divergence.
- `probe_if_command_set()` — per-device override point for
  device-specific divergence semantics.

## Common mistakes

- Treating any state divergence as "command failed" instead of
  "external control". The two need separate handling.
- Implementing a `probe_if_command_set` that's too lenient (accepts
  any state as "command set"). External control then goes
  undetected.
- Forgetting to clear the flag on recovery. The device gets stuck
  in "external" mode and quiet-solar never resumes control.
- Conflating external control with user override. External =
  someone else is driving; user override = the user told *us* to
  drive differently. See [user-override.md](user-override.md).

## See also

- [load-base.md](load-base.md) — where the flag lives.
- [ha-device-mixin.md](ha-device-mixin.md) — where the detection
  happens.
- [user-override.md](user-override.md) — the related-but-distinct
  flow.
- [../use-cases/external-override.md](../use-cases/external-override.md)
  — end-to-end scenario.
