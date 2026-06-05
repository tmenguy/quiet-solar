---
title: User override
slug: user-override
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
last_verified: 2026-06-05
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
