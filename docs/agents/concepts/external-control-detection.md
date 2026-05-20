---
title: External control detection
slug: external-control-detection
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
  - custom_components/quiet_solar/ha_model/device.py
last_verified: 2026-05-21
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
