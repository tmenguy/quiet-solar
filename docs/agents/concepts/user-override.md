---
title: User override
slug: user-override
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
last_verified: 2026-08-02
---

# User override

## TL;DR

A **user override** is a constraint a human created to overrule
quiet-solar's defaults — "charge my car to 90% tonight regardless of
solar". Overrides land as constraints with `load_info={originator:
"user_override"}`, take precedence in conflict resolution, and trigger
a confirmation notification to the user who created them. User
override is **distinct** from external control detection
([external-control-detection.md](external-control-detection.md)):
override = "user told *us* to do this differently"; external = "user
is driving the device themselves".

**Log levels (QS-306).** Override handling is unchanged; only the no-op
constraint-reset lines in `home_model/load.py` moved to DEBUG. A reset that
destroys a user-initiated flag still logs INFO — that is what the
`_has_state_to_reset()` hook ([load-base.md](load-base.md)) exists to
guarantee for subclasses such as `QSChargerGeneric`, whose override ORs in
`do_force_next_charge` / `do_next_charge_time`.

## When you need this concept

- Designing a user-facing override UI (Magali's mobile app prompt,
  TheAdmin's dashboard tile).
- Working on conflict resolution between prediction-derived and
  override-derived constraints.
- Touching the confirmation-notification flow.
- Debugging "my override didn't stick" issues.

## Core idea

Origin tracking via `load_info` is the structural foundation:

- A constraint without `load_info` is anonymous — the system can't
  explain why it exists.
- A constraint with `load_info={originator: "prediction"}` is
  automatic and can be displaced by an override.
- A constraint with `load_info={originator: "user_override"}` is
  pinned — the solver does not auto-replace it.

When a user creates an override, the workflow is:

1. UI captures the desired target (extend trip, force charge, cancel
   scheduled run).
2. Existing prediction-derived constraint on the same load is
   replaced with a `user_override` variant of the same constraint
   tier.
3. Solver re-evaluates → potential plan change.
4. Confirmation notification fires to the user who issued the
   override, citing the new target.

User overrides are still subject to physical limits (amp budgets,
SOC bounds). They don't trip breakers — they bend the plan.

**Override lifecycle (QS-256).** On bistate loads, BOTH directions of
an externally-detected override are constraint-driven:

- An override to the ON state pushes a
  `TimeBasedSimplePowerLoadConstraint`; an override to the idle/OFF
  state pushes a `TimeBasedHoldOffConstraint` (zero power, CMD_IDLE
  window). Either way the override ends through the same proven path:
  the constraint is met at its window end → acked →
  `ack_completed_constraint`'s USER_OVERRIDE branch calls
  `reset_override_state_and_set_reset_ask_time()` and arms the
  post-override cooldown.
- `LoadConstraint.score()` gives any USER_OVERRIDE-originated
  constraint a highest-order additive term (1e14) so it always wins
  allocation ordering and same-end-time cluster dedup.
- A command that conflicts with an active override is DROPPED, never
  phantom-acked: `launch_command` drops it at the drop point and
  `force_relaunch_command` drops a stale suppressed `running_command`
  (both via `is_command_suppressed_by_override`).
- **And the mirror image (QS-307): when the override ENDS, its own
  command is dropped.** Nulling `external_user_initiated_state` also
  disables the suppression drop above, so an override-aligned command
  still in flight would keep being relaunched after the override was
  over. Expiry is the one place that knows the override just ended, so
  it owns that drop — for the *aligned* command only; anything else is
  a genuine solver intent. Note this is the first command-slot mutation
  inside `check_load_activity_and_constraints`, and three of that
  method's callers run outside `_update_loads_lock` — as do the buttons
  that launch a `CMD_IDLE` of their own. So all **three** completion
  paths (`launch_command`, `check_commands` and
  `force_relaunch_command`) re-check the slot after their `await`, and
  what they check is **ownership**, not emptiness: `_slot_still_holds`
  compares the dispatch's `_running_command_generation` tag, because a
  slot that was *replaced* (rather than emptied) passes an `is None`
  test and would otherwise be acked off the previous command's result
  (QS-320). See [load-base.md](load-base.md).
- A state mismatch only classifies as a NEW override when the entity
  state is newer than `last_command_execution_time` (causality guard)
  and the 180s post-override cooldown has elapsed.
- `user_clean_and_reset` clears ALL override fields (state, time,
  reset-ask time, first-cmd-reset flag) plus the causality anchor —
  the reset button breaks any override loop.
- The legacy timer reset stays as fallback for an override without a
  constraint (e.g. restored from storage); whichever mechanism fires
  first nulls the fields, making the other a no-op.
- Restart-time limitation: storage restore evaluates stored-override
  expiry against `override_duration`, which may still hold the config
  default (the number entity restores later). If the default is
  smaller than the configured value, a still-valid override can be
  dropped on restart — accepted conservative direction (drop early
  rather than keep poison).

**An override ALWAYS expires, even on a dead load (QS-307).** The
override lifecycle's clock-driven branches — expiry, the reset-ask
follow-up, the post-override cooldown drain — run regardless of what
the command slot holds. They read no entity state, so an unresponsive
device cannot freeze them. Only *detection* is gated on
`is_load_command_set()`; see the "split gate" section of
[bistate-duration-devices.md](bistate-duration-devices.md). (A load that
cannot have an override at all is still gated out entirely, by
`support_user_override()` on the outer gate — and a stored override on
such a load is dropped at restore rather than left for a lifecycle that
will never run.)

Before QS-307 the whole block sat behind that gate, and QS-304's
saturating retry ladder meant the gate could stay shut forever: an
override on a load that stopped obeying was pinned permanently, so
`is_user_overridden()` stayed True and the load never returned to
controlled consumption or to the solver.

**The lifecycle does not end at expiry.**
`reset_override_state_and_set_reset_ask_time` nulls the override state
but arms `asked_for_reset_user_initiated_state_time`, and
`get_override_state()` reports `ASKED FOR RESET` while that is set — so
`is_user_overridden()` is still `True` and
`get_device_power_latest_possible_valid_value(ignore_auto_and_user_overridden_load=True)`
still returns `0.0`. Any change that lets the override expire but not
the cooldown has not fixed anything; it has moved the bug one step
later. When debugging a stuck override, walk all three clock branches —
they are the self-heal, and they are deliberately independent of
command state.

## Key types / structures

- `load_info` dict — `{originator: "user_override" | "agenda" |
  "prediction" | "system"}`.
- Constraint creation helpers that stamp `load_info` at the API
  level.
- The confirmation-notification path on `QSPerson`.

## Common mistakes

- Forgetting to stamp `load_info` on the override. The override
  ranks like a prediction and gets auto-replaced.
- Replacing the override constraint directly instead of creating a
  new one. The solver re-reads constraints each cycle — mutate in
  place and the cycle sees inconsistent state.
- Skipping the confirmation notification. Magali presses "extend"
  and gets no feedback → she presses it again, twice, then files
  a bug report.
- Confusing override with external control. If the user pressed
  the override button in the mobile app, it's override; if they
  walked over to the charger and unplugged it, it's external.

## See also

- [constraints.md](constraints.md) — the constraint API and
  `load_info` field.
- [external-control-detection.md](external-control-detection.md) —
  the cousin concept.
- [notification-routing.md](notification-routing.md) — the
  confirmation channel.
- [../use-cases/magali-plugs-in-car.md](../use-cases/magali-plugs-in-car.md)
  — override is the 5% case in this flow.
- [../personas/magali.md](../personas/magali.md) — the persona
  whose 5-second rule shapes the override UI.
