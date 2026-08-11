---
title: Charger Dynamic Budgeting
slug: charger-budgeting
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/charger.py
last_verified: 2026-08-11
---

# Charger Dynamic Budgeting — the tactical layer

## TL;DR

`ha_model/charger.py` is **trust-critical**: it's where the solver's
strategic plan meets physical reality. The charger budgeting layer
operates in 45-second adaptation windows (`CHARGER_ADAPTATION_WINDOW_S`),
manages per-phase amp distribution across multiple chargers on the
same circuit, and **can override the solver** when amp budgets
conflict. Phase switching (1P↔3P), staged transitions, and dampening
(real-power measurement) all live here. A bug in the solver makes a
bad plan; a bug here trips a breaker.

**Log levels (QS-306).** The QS-306-touched per-cycle status lines in this
file are now DEBUG, or emitted at INFO only on change (or after 900 s);
**other INFO sites in this file are unchanged** — notably the start/stop
keepers and the `budgeting_algorithm_minimize_diffs` lines, which were
deliberately left out of scope. Behavior is unchanged: only log levels and
log frequency moved.

The available-power line compares against a 100 W deadband, with three
deliberate exceptions: a sign flip (export ↔ import) logs **when either side
is at least the deadband** — the floor matters, because an unbounded flip term
makes near-zero dither log every cycle — and a stuck-at-`NaN` or
stuck-at-`inf` sensor counts as *unchanged*, so a broken sensor cannot
re-inflate the log to the full cycle rate. `detach_car()` does **not** touch the
log-on-change state (QS-342 review #03 / D2): it sits on the churn path itself, so
wiping there dropped the key on every allocation change and made the throttle a
no-op in production — and it is redundant, because every key is either
car-qualified or carries the car name as its value. A disabled charger's
group-level key is still evicted so a re-enable is always announced. `QSChargerGeneric` extends
`_has_state_to_reset()` (see [load-base.md](load-base.md)) because its reset
override destroys the user-initiated `do_force_next_charge` /
`do_next_charge_time` flags.

## When you need this concept

- Implementing or modifying any charger behaviour
  (`QSChargerGeneric`, `QSChargerOCPP`, `QSChargerWallbox`).
- Touching the dynamic-group budgeting tree (see
  [dynamic-group-tree.md](dynamic-group-tree.md)).
- Working on phase switching, dampening, or adaptation-window logic.
- Anything that could affect physical safety — circuit limits,
  breaker margins, max amp per phase.

## Core idea

The budgeting algorithm (`budgeting_algorithm_minimize_diffs`):

1. **Priority check**: if the highest-priority charger isn't charging
   but a lower one is, trigger a reset allocation.
2. **Prepare budgets**: either keep current amps (minimise
   transitions) or reset to minimum (rebalance).
3. **Shave mandatory**: if the minimum amps still exceed the group
   limit, stop the lowest-score chargers first.
4. **Shave current**: try phase switching (1P→3P) for lower-score
   chargers before reducing amps.
5. **Smart allocation loop**: iteratively adjust each charger's
   budget toward the power target while respecting all constraints.

Then `apply_budget_strategy()`:

- For large changes, **stage** across two cycles: phase 1 reduces
  decreasing chargers (frees up amps), phase 2 increases other
  chargers next cycle (already validated safe).
- `remaining_budget_to_apply` persists between cycles to complete
  phase 2.

## Key types / structures

- `QSChargerGeneric` — base class. Power ramping, phase switching,
  budgeting state machine.
- `QSChargerOCPP` — OCPP variant. Adds transaction handling.
- `QSChargerWallbox` — Wallbox variant. Maps vendor status enums.
- `QSChargerGroup` — aggregates chargers on the same circuit.
- `QSChargerStatus` — per-charger state (amps, phases, real power,
  adaptation state).
- `charge_score` — per-charger priority. Higher = wins budget
  conflicts first.
- `CHARGER_ADAPTATION_WINDOW_S = 45` — stability requirement before
  rebalancing.
- `CHARGER_STATE_REFRESH_INTERVAL_S = 14` — state-polling cadence.

### Car↔charger allocation tie-break (QS-342)

`QSChargerGeneric.get_best_car` runs a greedy per-charger allocation over
all plugged chargers: each charger scores every car (`get_car_score`) and
the loop repeatedly assigns the best remaining (charger, car) pair. Two
QS-342 facts to internalise:

- **Exact score ties are a NORMAL operating case.** The dominant score
  component is the distance bump, quantised in **0.5 m buckets** over 50 m
  (deliberate — car GPS jitters by metres; finer resolution would make the
  allocation flap with noise). Cars sleeping in their usual spots between
  two wallboxes tie *every night*. Never "fix" a tie by sharpening the
  distance resolution.
- **Ties are resolved by a deterministic cascade**, evaluated over ALL
  (charger, car) pairs at the max remaining score (not just each charger's
  list head): **regret → stickiness → stable (charger name, car name)
  order**. Regret = the pair's score minus the car's best score on any
  *other unassigned* charger (no alternative → 0, i.e. regret = own score);
  it is recomputed at every greedy iteration, skipping already-assigned
  chargers (their score lists are stale by design). Stickiness prefers the
  charger the car is currently attached to, so symmetric steady states
  don't swap-flap. The stable order assumes **unique device names**
  (enforced de facto by HA config-entry titles).

**Steal semantics:** an attached car is displaced *only* by a strictly
higher score, or by an **equal score with strictly higher regret** — the
latter is required so a crossed pairing (leftover of the pre-fix
ping-pong) recovers to the regret-consistent allocation. Any attachment
state matching a regret-consistent allocation is a fixed point.

**Known limitations:**

- *Plug-time correlation is dead after an HA restart for plug sessions
  already active before the restart*: car plug probes are
  recorder-bootstrapped (3 days) but the charger's synthetic
  `is_there_a_car_plugged` probe is in-memory-only since restart, so for a
  pre-restart session the compared durations never realign and
  `plug_time_bump` is structurally 0 exactly when it is needed. A session
  that starts *after* the restart has both clocks aligned and is
  unaffected. Follow-up:
  <https://github.com/tmenguy/quiet-solar/issues/344>.
- *Residual failure mode — two unhysteresised quantisation edges.* (1) The
  0.5 m `dist_bump` bucket edge: a car GPS-jittering across it produces
  alternating **strict-score** winners that no tie-break sees. (2) The
  `dist <= 3.0` threshold that adds +1 to `plug_bump`: because `plug_bump`
  is the lowest-order term, a ±1 flip converts an **exact tie** — where
  stickiness protects the incumbent — into a strict win, which steals with
  no margin at all. The second edge therefore *bypasses* the cascade rather
  than being resolved by it, and it is reachable in the incident's own
  geometry (3.58 / 3.69 / 3.97 m). Steal-margin hysteresis stays out of
  scope until observed in the field.

`get_car_score` logs its decomposition at INFO-on-change keyed on the
quantised tuple `(plug_bump, plug_time_bump, dist_bump)` only — a
deliberate INFO exception to the QS-306 volume rules, because this line is
what makes allocation incidents diagnosable without a recorder-DB
forensic session.

**How the throttle works** (`LogOnChangeMixin.log_info_on_change`).
Per key the helper remembers the recently-*observed* values (each flagged
with whether it was actually shown) and emits a value the first time it is
seen, suppressing it while it is still remembered (`_RELOG_UNCHANGED_AFTER_S`, 900 s, which doubles as the
per-value TTL). On top of that each key carries a budget of
`_LOG_VALUES_PER_WINDOW` (4) emissions per window; further distinct values
are counted and disclosed as a single
`[+n further change(s) not shown in the last 900s]` line when the window
rolls. Both halves are load-bearing:

- **Dedup by value** kills oscillation. An A→B→A flap across a
  quantisation edge emits A once and B once per window and then goes
  quiet, so volume is `O(distinct states per window)` rather than
  `O(transitions)`.
- **The per-key budget** kills drift, which dedup cannot touch: a value
  creeping by one quantum per cycle (GPS drifting 0.55 m per ~7 s cycle)
  is a *new* value every cycle, so dedup alone satisfies "not the same
  stuff over and over" literally while leaving volume effectively
  unbounded. Capping the remembered-value map bounds memory, not volume.

**Volume bound.** Per key: **≤ budget + 1 lines per 900 s**, and this is
*data-independent* — oscillation, monotone drift and a held-constant value
all behave the same. Allocation keys use the default budget of 4 (so 5
lines per 900 s, ≈ 0.33/min); the two **telemetry** sites —
`dyn_handle`'s available-power line and the SoC callback — use
`_LOG_TELEMETRY_VALUES_PER_WINDOW` (12) instead. That exception is
deliberate (review #04 / B12+B13): the default budget assumes many distinct
values per window means churn, which is true for allocation keys and false
for telemetry, whose values advance during entirely normal operation (a
charging battery's Wh/%, a home crossing between import and export). With
the default those sites sat in permanent overflow during a normal charge
and dropped real operational events. Keys are `N·M`
(`get_car_score`) + `N` (the merged `get_best_car` winner line) + `N`
(`detach_from_other_charger`) + `2N` (`update_value_callback_soc`) +
`N + 1` (group), so the **aggregate** is ≤ `5·(N·M + 5N + 1)` per 900 s.
For N=3, M=4 that is ≤ 140 lines per 900 s ≈ 13 400/day worst case and
~2 400/day in steady state. The `2N` SoC term uses the telemetry budget, so the aggregate is
`5·(N·M + 3N + 1) + 13·2N`. Completeness holds for distinct states: only
budget *overflow* loses detail, and it is disclosed as a count.

The disclosure line is static text keyed by the throttle key — never the
caller's own message and arguments — and carries **both** the number of
distinct changes dropped and the number of dedup-suppressed repeats. That
repeat count preserves the incident's defining signal: without it a key
pinned at the 7 s cycle rate and one genuinely changing every few minutes
produce identical logs, and #342 was diagnosed from the volume being
visibly absurd.

Two caveats on "never silent":

- The disclosure is flushed by the **next call on the same key**. If a key
  goes quiet for good — charger unplugged, device disabled, car removed, HA
  restart — the final window's counts are lost. This is reachable on the
  incident path itself. Accepted rather than fixed: a flush hook would put
  logging concerns back into charger lifecycle control flow, and
  cross-session banking is what rounds #01–#03 proved unworkable.
- The window roll uses `abs()` on the elapsed time, so a caller alternating
  between two timestamps 900 s apart would roll the window every call and
  defeat the budget. Not reachable at any current call site (all pass a
  monotonically advancing `event_time`), but the "data-independent" claim
  assumes a sane clock.

`time` is normalised to UTC on entry, so a caller mixing naive and aware
datetimes under one key can neither raise nor escape the bound.

The log-on-change state deliberately **survives `detach_car()`**. There is
no session-boundary wipe, and none should be reintroduced: `detach_car()`
sits on the churn path itself (every change of the `get_best_car` value
routes through it), so wiping there dropped the key and made the throttle a
no-op in production. It is also redundant — every key is either
car-qualified (`get_car_score:{car}`) or carries the car name as its
*value* (`get_best_car`), so no key can name a stale car. The per-session
anchors are `update_power_steps`' attach line and the unplug WARNING.

A consequence (review #04 / C7): `_log_on_change_state` keys are now never
pruned, so renaming a car or charger orphans its old keys for the lifetime
of the process. Bounded by config churn — a handful of small tuples — so it
is recorded rather than fixed.

**Historical note (do not re-invent).** Review rounds #01–#03 of QS-342
each tried a *time floor* on changed values (`_CHANGED_RELOG_MIN_INTERVAL_S`,
60 s) plus a "banked suppressed-change" counter, and each round fixed the
previous round's defect while introducing the next. A time floor is
structurally incapable of completeness — any state whose whole lifetime
fits inside the window is unobservable — and the bank counted *calls*
rather than *states*, so it over-reported a held change, under-reported an
oscillation, and was destroyed outright by the `detach_car()` wipe. The
incident was never "changes happened too fast"; it was **the same value
repeated 17 791 times**, which is why the correct primitive is
de-duplication by value.

### Charge-origin tagging & `get_charge_type()` (QS-274)

Constraints carry their origin in `load_info[CONSTRAINT_ORIGINATOR_KEY]`
so the UI can distinguish *where* a charge target came from. Creation
sites stamp named constants from `const.py`:

- **manual finish time** → `CONSTRAINT_ORIGINATOR_MANUAL`
- **calendar/agenda** → `CONSTRAINT_ORIGINATOR_AGENDA` (value is exactly
  `"agenda"`, an invariant — persisted pre-QS-274 calendar constraints
  must still match)
- **person forecast** → `CONSTRAINT_ORIGINATOR_PERSON`, stamped
  *alongside* the existing `{"person": <name>}` matching key (do not
  drop that key — `_match_ct` cleanup relies on it)
- **force/override** → `CONSTRAINT_ORIGINATOR_USER_OVERRIDE` (pre-existing)

`QSChargerGeneric.get_charge_type(return_charge_errors=True)` returns
`(type, constraint)`:

- **as-fast branch:** `user_override` originator →
  `CAR_CHARGE_TYPE_MANUAL_AS_FAST_AS_POSSIBLE`, else
  `CAR_CHARGE_TYPE_AS_FAST_AS_POSSIBLE`.
- **deadline branch (precedence):** `"person"` key →
  `CAR_CHARGE_TYPE_PERSON_AUTOMATED` (person-first, keeps legacy
  `{"person": name}`-only constraints working); `elif` agenda originator
  → `CAR_CHARGE_TYPE_CALENDAR`; `else` → `CAR_CHARGE_TYPE_MANUAL`
  (untagged / `load_info=None`). The `ct.load_info and …` guard is
  mandatory to avoid `None.get(...)`.
- **`return_charge_errors=False`** skips the Faulted / No Power / Not
  Plugged short-circuits and returns the underlying type — used by the
  origin context line so a charger-error string never leaks into it.

`CAR_CHARGE_TYPE_SCHEDULE` was removed entirely; the unrelated
`WallboxChargerStatus.SCHEDULED` hardware status is a different concept
and is untouched. The `CONSTRAINT_ORIGINATOR_MANUAL/PERSON` stamps are
written for coherence/forward-compat — detection keys off the `"person"`
key and the `else`, so they are not read by `get_charge_type()` itself.

### A faulted charger leaves the solver and the budgeting group (QS-346)

`is_charger_faulted(time)` (membership of the unfiltered status in
`_unknown_state_vals` = `{STATE_UNKNOWN, STATE_UNAVAILABLE}` + the
per-type unknown values, e.g. OCPP `Faulted`/`Unavailable`) is the "a
human has to take care of it" predicate. While it holds, a charger is
removed from budgeting at **two** layers:

- **Strategic** — `QSChargerGeneric.is_load_active(time)` returns `False`
  (overriding the base "enabled and has constraints"), so the charger
  drops out of `active_loads` and the solver stops *updating* its
  constraint.
- **Tactical** — `QSChargerGroup.ensure_correct_state` skips a faulted
  charger in the same guard that skips a disabled one
  (`qs_enable_device is False or is_charger_faulted(time)`), so it never
  reaches `actionable_chargers` / `budgeting_algorithm_minimize_diffs`.

The **car and its constraint stay attached** — that is the only surface
that renders the fault red on the car card, and there is no charger
card. `is_load_active=False` only stops the solver touching the
constraint; nothing detaches. On recovery both layers revert and the
retained constraint resumes toward its target (no energy moved during
the fault, so resuming is correct — no stale-constraint handling).
"Stop commanding" needs no extra gate: with `is_load_active=False` the
loop issues `CMD_IDLE`, and on a faulted charger `check_charge_state`
returns `None`, so `_ensure_correct_state` lands in the *not charging /
don't want to charge → do nothing* branch — zero low-level commands.

### The plug-state rescue no longer needs a currently-attached car (QS-346)

`Faulted` ∈ `_unknown_state_vals`, so a faulted charger's own plug probe
answers `None` and the only compensating path is the car's own plug
sensor. Historically that path required `self.car` — but a
`qs_device_clean_and_reset` detaches the car, and with `self.car is None`
the rescue was disabled, trapping the charger in an absorbing "no car /
not plugged" state. `_last_attached_car` breaks the cycle:
`detach_car()` records the car it detaches (unconditionally, inside its
`self.car is not None` block), and `_check_plugged_val` consults it when
no car is attached — but **only while that car is not homed to another
charger** (`_last_attached_car.charger is None`), so a car re-homed to
charger B never makes charger A report itself plugged. A **physical
unplug clears the memory** (in the `is_not_plugged` branch of
`check_load_activity_and_constraints`, after the `if self.car:` block),
so a replug on a still-faulted charger does not re-attach on stale
memory — the charger correctly stops appearing available until its
status recovers.

## Lifecycle / sequence (one update cycle)

```text
All chargers stable for 45s? → dampening update (measure real power)
  ↓
Detect transitions (single charger change → record power delta)
  ↓
Budget reset opportunity? (20-min timeout or HP charger waiting)
  ↓
budgeting_algorithm_minimize_diffs()
  → priority check → prepare budgets → shave mandatory → shave current
  → smart allocation loop (iterative, phase limits)
  ↓
apply_budget_strategy()
  IF large change: phase 1 (reduce first) → phase 2 (next cycle)
  ELSE: apply directly
```

## Common mistakes

- Changing budgeting logic without integration tests that simulate
  multi-charger rebalancing sequences over time. Unit tests on a
  single charger miss the trust-critical interactions.
- Bypassing the dynamic-group tree's recursive validation. A leaf
  charger can pass its local check while violating the parent
  group's circuit limit.
- Skipping the staged-transition split for "small" changes that
  turn out to overshoot per-phase limits in transient. Apply
  staging whenever the change crosses a phase boundary.
- Treating `charge_score` as a tiebreaker. It's the primary ranking
  for budget conflicts.

## See also

- [dynamic-group-tree.md](dynamic-group-tree.md) — the topology
  budgeting walks.
- [solver.md](solver.md) — the strategic layer that produces the
  plan this layer may override.
- [../principles/strategic-tactical-control.md](../principles/strategic-tactical-control.md)
  — why the split exists.
- [../use-cases/magali-plugs-in-car.md](../use-cases/magali-plugs-in-car.md)
  — the magic-moment use case this layer enables.
- [car-soc-estimation.md](car-soc-estimation.md) — `constraint_update_value_callback_soc`
  is the **sole writer** of the car's float SOC accumulator (QS-243); while
  the car is estimating it bypasses the raw sensor and seeds the constraint
  from the effective estimate instead of a forced `0.0`. **QS-281** hoisted the
  `accumulate_soc_delta` call to exactly one per **percent** callback so the
  accumulator advances during a **healthy** charge too (return discarded on the
  non-estimating branch — the constraint value is byte-identical). The hoist is
  guarded on `is_target_percent`: the energy-mode callback never advances the
  SOC accumulator, so there is no phantom delta even if the
  `can_use_charge_percent_constraints()` ⇒ capacity coupling is ever loosened.
  It drives the
  accumulator through the car's public `soc_integration_cursor` /
  `accumulate_soc_delta` accessors (no underscore reach-in), and gates the
  zero-power hardware-fault check on `car.is_soc_sensor_distrusted()` (stale /
  no-sensor) — **not** on the broad estimation flag, so a manual override on a
  healthy car still gets fault detection. The estimate is reset on the
  genuine-plug-in branch (`do_full_reset`) and the unplug edge; a boot re-attach
  preserves it (reboot survival).
