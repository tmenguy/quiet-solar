---
title: AbstractDevice / AbstractLoad
slug: load-base
kind: concept
covers:
  - custom_components/quiet_solar/home_model/load.py
last_verified: 2026-08-02
---

# AbstractDevice & AbstractLoad

## TL;DR

`AbstractDevice` is the base for all controllable devices. It owns
configuration, the command lifecycle (pending → launched → acked,
with stacking for busy devices), and switching-cost protection
(`num_max_on_off` daily budget plus a 10-minute hysteresis). It is
3-phase aware. `AbstractLoad` extends `AbstractDevice` and adds the
constraint-management surface (`get_for_solver_constraints()` is the
solver's entry point), plus green-only mode, user override state, and
external-control detection. **Both live in `home_model/load.py` —
strict zero-HA-import boundary.**

**Log levels (QS-306).** `constraint_reset_and_reset_commands_if_needed()`
now logs INFO only when there was something to reset, and the
"no bad constraint found" line is DEBUG. No behavior change.

The "was there anything to reset?" test lives in the overridable
`_has_state_to_reset(keep_commands)` hook. **Any subclass whose reset
override clears extra state must extend it**, or that work is destroyed
while the base reports "nothing to reset" at DEBUG. The base term covers
`_constraints`, `_last_completed_constraint`, and — when
`keep_commands=False` — `current_command`, an in-flight `running_command`, or
a queued `_stacked_command`. Every access uses `getattr` with a default: the
hook runs during `AbstractDevice.__init__`, before those attributes exist.

`QSChargerGeneric` is the worked example: its override clears the
user-initiated `do_force_next_charge` / `do_next_charge_time`, so it ORs them
into the hook. Without that, a user pressing "force next charge" on a charger
holding no constraints would have the flag destroyed while the log said
"nothing to reset".

## When you need this concept

- Adding a new device type — you'll extend `AbstractLoad` (or, rarely,
  `AbstractDevice` if it doesn't participate in solving).
- Changing command lifecycle semantics.
- Touching switching-cost protection (the daily on/off budget +
  hysteresis pattern).
- Working on external-control detection or user-override handling.

## Core idea

**Behavioral contract**: `AbstractLoad` defines guarantees every
device type must honour. Device-specific tests validate *deviations*
from that contract — don't re-test the contract itself. This is a
contract-testing pattern.

Command lifecycle (`AbstractDevice`):

- `pending` — solver decided to run; queued for launch.
- `launched` / `running_command` — command dispatched via HA service
  call, ACK not yet observed.
- `acked` / `current_command` — `probe_if_command_set()` confirmed.
- Stacking: when a device is busy with a `running_command`, additional
  pending commands stack until ACK clears the slot — *unless* QS has
  lost control, in which case the newer command supersedes the stale
  one (see below).

Relaunch, escalation and supersession (`AbstractDevice`, QS-304):

- `check_and_relaunch_command(time)` is the whole retry lifecycle, and
  it is the *only* thing `QSHome.check_loads_commands` calls. Probe →
  ack-or-relaunch → escalate-or-re-arm. It returns
  `command_acked_or_good` for the caller's `all_ok` aggregation.
- **Saturating backoff, not a give-up.** `command_relaunch_delay_s()`
  returns `COMMAND_RELAUNCH_BASE_DELAY_S * min(n + 1,
  NUM_MAX_COMMAND_RELAUNCH)` — 50, 100, 150, 200, 250, 300 s and then
  300 s **forever**. Two distinct indices, deliberately named apart:
  the *delay* stops growing at `n == 5`; the *uncontrollable threshold*
  is `n >= 6` (1050 s cumulative). Before QS-304 the second index
  meant "give up", which bricked the load's command slot permanently.
- **One durable clock.** `unresponsive_since` is set once when the
  threshold is crossed (one ERROR, one push) and cleared *only* by
  `_clear_unresponsive` — the single writer, so "one line in, one line
  out" cannot drift.
- **`is_uncontrollable` is internal state, not an entity.** It decides
  supersede-vs-stack in `launch_command` and gates the one-shot
  escalation; nothing exposes it to Home Assistant. That is why a clock
  that outlives its command is a nuisance rather than a user-visible bug.
- **`is_uncontrollable` needs `running_command is not None`.** That
  conjunct is load-bearing, not defensive. Without it an
  `unresponsive_since`-only property would stay `True` forever with no
  retry left to clear it. If we are not waiting on anything, we are not
  uncontrollable. QS-307 (from #308) closed the last path that emptied
  the slot **with no successor** while keeping the clock.
- **Two clocks, not one (QS-307).** `unresponsive_since` tracks the command
  in flight — every path that empties the slot **with no successor** releases
  it, which is what lets it drive supersede-vs-stack honestly. A *supersede*
  is the deliberate exception: `abandon_running_command` preserves the clock
  and hands it to the successor, so the load keeps superseding instead of
  stacking and holds QS-304's saturated 300 s cadence. Releasing it there to
  "restore the invariant" would silently undo that. `_unresponsive_needs_ack` is per **episode**
  — "we announced a loss and it has not been *acknowledged*" — and it gates the
  ERROR log and the push, never the clock. Conflating them is what produced
  a notification every ~18 min for a device flapping between unreachable and
  answering-but-disobeying: `running_command_num_relaunch_after_invalid` is
  cumulative over a command's life, so the give-up can fire long after the
  threshold was crossed by ordinary relaunches, and each release re-opened
  the guard.

  **QS-319 gave the latch a third role: user-visible state.** It is what the
  `qs_load_lost_control` PROBLEM binary sensor reads, through the public
  `has_unacknowledged_lost_control` property — *not* `is_uncontrollable`, which
  is per-command and flickers False every time the slot empties. Note the
  semantics the name carries: **acknowledged, not resolved**. A user reset ends
  the episode while the device may well still be broken; the next ladder climb
  then opens a new one and the sensor comes back on.

  An episode ends on exactly four things, and every statement of the rule —
  here, in the code comments, and in `notification-routing.md` — must list all
  four: a **real ack**, **proven contact on a slot that changed hands**
  (`_confirmed_contact_on_disowned_slot`, QS-320 — the disowned-slot bail-outs
  in `launch_command`/`check_commands` when the result was `True`; it also
  resets the successor's inherited rung so the ladder cannot instantly
  re-announce), **explicit user remediation**
  (`_acknowledge_lost_control`), or a **process restart / config-entry reload**,
  where nothing restores the latch. The latch is now set by the *announce*
  branch of `_escalate_or_recover` too, before the await, so a notify service
  that raises cannot resurrect the storm.
- **`_clear_unresponsive(reason, contact=ContactEvidence...)`.** The argument
  says what the release tells us about the device, and only that decides
  whether an **episode** ended. It is **required** and an `enum`, so a
  forgotten kwarg is a `TypeError` and a typo an `AttributeError` — defaulting
  to the episode-ending value would let a future caller end an incident by
  omission.
  - `CONFIRMED` (three callers on the merged tree: the real ack in
    `_ack_command`, and — via `_confirmed_contact_on_disowned_slot`, QS-320 —
    the disowned-slot bail-outs in `launch_command` and `check_commands`)
    **clears** the latch, *unconditionally* — an ack is contact whether
    or not there was a clock left to release, and so is a `True` result about
    a slot that changed hands.
  - `UNREACHABLE` (the invalid-probe give-up, the only such caller)
    **latches** it — but only when it actually released a live clock. Latching
    a release that released nothing swallowed the load's first genuine push
    forever, and that is the *ordinary* ordering: the give-up fires ~70 s in,
    the escalation threshold is ~1050 s away.
  - `UNKNOWN` (`_drop_running_command`, the `keep_commands=False` wipe, and —
    since QS-319 — the empty-slot re-arm) **leaves it alone**. These are QS
    changing its mind — the user took control, the load was disabled, an
    override expired, we emptied the slot ourselves — and say nothing about
    whether the device answers.

  **The empty-slot re-arm moved from `CONFIRMED` to `UNKNOWN` (QS-319).** It
  was a **fake ack**: nobody answered, *we* emptied the slot. Harmless only
  while the give-up was the sole latch-setter (it nulls `current_command`, so
  it skipped the gate). Once the announce branch also latches, `CONFIRMED`
  there would clear the episode on every cycle that runs with an empty slot —
  which is the *production* ordering. Blast radius: **the latch and nothing
  else, not even a log string** — `CONFIRMED` and `UNKNOWN` take the same
  `else` branch and emit the identical "Lost-control state cleared for load
  %s: %s" line.

  **What the third value is for, precisely.** Measured against the parity
  oracle, a two-valued signal is enough for the flapping scenario on its own.
  `UNKNOWN` exists because QS-307 *added* a drop the codebase did not have —
  the override-expiry drop in `check_load_activity_and_constraints` — and
  without it that new drop would re-announce an already-announced incident.
  That it also quiets the two pre-existing drop paths (where `main` does
  re-announce) is a bonus, not the justification. QS-319 gave `UNKNOWN` a
  second, load-bearing caller — the empty-slot re-arm — so the third value is
  no longer marginal. Read the guarantee as "one alert per announced episode",
  never as a blanket rule over unrelated notification types: see the scope
  note below.
- **One writer for the *clock*; the *latch* has four writers and five
  triggers.** These are two different guarantees and QS-319 separated them.

  `_clear_unresponsive` is the only writer of `unresponsive_since`; it is
  reached by seven releasers — an ack, the empty-slot re-arm,
  `_drop_running_command` (**unconditionally** — an emptied slot has no owner
  for the clock whether or not a confirmed command ever existed), the
  `NUM_MAX_INVALID_PROBES_COMMANDS` give-up in `check_commands` (QS-307, the
  one `UNREACHABLE` caller), a `keep_commands=False` wipe, and — QS-320
  review fix #01/2 — the disowned-slot bail-outs in `launch_command` and
  `check_commands` when the result was `True`: proven contact ends the
  episode even though the ack is withheld. `force_relaunch_command`'s
  bail-out deliberately does NOT clear (AC1 pins `unresponsive_since`
  surviving it). All seven release
  the *clock*; which of them end an *episode* is the `contact` question above.
  It also clears the supersede anchor *ahead of* its own early return, so the
  two fields cannot desynchronise.

  `_unresponsive_needs_ack` is written from four places:

  | Writer | Effect | Trigger |
  | --- | --- | --- |
  | `_escalate_or_recover` announce branch | **sets** | ladder wall crossed, episode announced (QS-319) |
  | `_clear_unresponsive(CONFIRMED)` | clears | real ack |
  | `_clear_unresponsive(UNREACHABLE)` | sets | invalid-probe give-up (QS-307) |
  | `_acknowledge_lost_control` | clears | ← `user_clean_and_reset` (reset button, car auto-reset) |
  | `_acknowledge_lost_control` | clears | ← `qs_enable_device` setter (enable/disable transition) |

  `_acknowledge_lost_control` **early-returns when no episode is open**: both
  call sites fire on every reset press and every enable/disable transition, the
  overwhelming majority with nothing to acknowledge, and an unconditional log
  would claim an acknowledgement that never happened.

  The `qs_enable_device` clear sits **inside** the setter's
  `if enabled != self._enabled:` guard, which covers both edges and makes an
  idempotent re-write unable to reach it — load-bearing, because
  `switch.py::QSSwitchEntityWithRestore.async_added_to_hass` drives the setter
  via `async_turn_on/off(for_init=True)` on every HA startup.

  Two deliberate edges. **`user_clean_constraints` is excluded**: it keeps the
  in-flight command, so the device is still being commanded and still not
  obeying — the episode is genuinely open. And **`car.py`'s
  confirmed-departure auto-reset** (`CAR_NOT_HOME_AUTO_RESET_S`) calls
  `user_clean_and_reset()` with no user involved and cascades to attached
  chargers, so a car driving away acknowledges an open episode on its charger.
  Accepted: the failure direction is an *extra* alert, never a suppressed one.
- **Both clock comparisons go through `_seconds_since`.** It returns
  `None` for "no anchor" and for an anchor in the *future*, which means
  "treat as fully elapsed". A backwards clock step (HA booting without an
  RTC, then NTP correcting) otherwise makes the delta negative — trivially
  below any threshold — which freezes the supersede throttle *and* stops
  the relaunch ladder advancing, so the rung can never reach the
  escalation threshold: a silently deadlocked slot with no ERROR and no
  PROBLEM sensor. One primitive so the two sites cannot drift apart.
- **`launch_command` guards its probe as well as its execute.** An
  unguarded `probe_if_command_set` re-created the deadlock on the
  stack-promotion path: it consumed the intent, stamped both the throttle
  and the staleness clock, and skipped the rung — and since
  `SUPERSEDE_MIN_INTERVAL_S == COMMAND_RELAUNCH_BASE_DELAY_S *
  NUM_MAX_COMMAND_RELAUNCH`, the window and the ladder delay expired
  together so the path was re-entered forever with zero service calls. A
  raising probe is treated exactly like one returning `None`: go on and
  execute.
- **Supersession + throttle.** A *differing* command against an
  uncontrollable load calls `abandon_running_command` and executes, at
  most once per `SUPERSEDE_MIN_INTERVAL_S` (300 s); inside the window it
  becomes `_stacked_command` (last-wins). An *equal* command is absorbed
  with no service call at any rung. A differing command against a
  *healthy* load is stacked, exactly as before.
- **Two abandon flavours, and picking the wrong one is a bug.**
  `abandon_running_command` preserves `running_command_num_relaunch`
  (along with `current_command` = last *confirmed* command,
  `prev_command`, `num_on_off`, `unresponsive_since` and
  `_last_supersede_time`), because zeroing the rung would restart the
  ladder at 50 s after every supersede — ~1150 service calls/day
  instead of ~288. That is only sound **when a successor is launched in
  the same call**. When the slot is left empty, use
  `_drop_running_command`, which additionally zeroes the rung and
  releases the clock: with nothing in flight, the rung describes a
  command that no longer exists and the clock has lost its subject.
- **The supersede is an intent, committed late.** `launch_command`
  decides to supersede, then still has to pass the
  override-suppression and `current_command == command` gates. Both the
  `_last_supersede_time` stamp and the abandon therefore happen next to
  the `_install_running_command(command, time)` call, so a supersede
  that launches nothing neither burns the 300 s window nor leaves a
  spent rung behind. The two gate-returns route through
  `_drop_running_command`.
- **The clock dies with the command state it describes.**
  `constraint_reset_and_reset_commands_if_needed(keep_commands=False)`
  wipes `current_command` *and* `running_command`, so it also releases
  the clock. `keep_commands=True` touches no command state and the
  clock survives. Skipping this makes the clock ownerless, and the
  first command after a reset-button press or a disable/re-enable is
  declared uncontrollable on its very first cycle.
- **A disabled load never shouts.** The escalation branch is gated on
  `qs_enable_device`; the housekeeping half still runs. QS was
  explicitly told to leave the load alone, so it must not push.
- **`_escalate_or_recover`'s `current_command is not None` gate is kept for
  minimality, not for the latch (QS-319 rewrote this).** It was
  load-bearing under QS-307, when the release here was `CONFIRMED` and the
  give-up was the only latch-setter: reaching the housekeeping arm with
  `current_command is None` meant the give-up had run earlier in the same cycle
  and may have just latched the episode, and an unconditional `CONFIRMED` would
  have undone that one statement later.

  Both halves of that argument are now gone — the release is `UNKNOWN`, which
  touches no latch, and the announce branch sets the latch too. The gate stays
  anyway: with `UNKNOWN` it is arguably unnecessary, but removing it would
  release clocks that are not released today, and there is no evidence behind
  that behavior change. Deliberate minimality. Note `_last_supersede_time` is
  cleared *outside* the gate, on purpose — the two fields answer different
  questions, so read the comment for which sentence governs which.
- **No shape keeps the clock with an empty slot (QS-307, from #308).** The
  last one that did was the `NUM_MAX_INVALID_PROBES_COMMANDS` give-up: it
  goes through `_ack_command(time, None)`, which nulls `current_command` on
  purpose (preserving it would bill phantom consumption into the persisted
  forecast) — and that is exactly what put it out of reach of
  `_escalate_or_recover`'s `current_command is not None` re-arm. The clock
  survived describing a command that no longer existed, and the *next*
  command inherited it: `is_uncontrollable` on its first cycle, so it
  **superseded** where it should have stacked, and the once-only guard then
  silenced every genuinely new episode for the rest of the load's life.
  `check_commands` now calls `_clear_unresponsive("the probe went unavailable",
  contact=ContactEvidence.UNREACHABLE)` right after the give-up, next to the
  state destruction that motivates it. `_ack_command` itself is unchanged — the release lives at the
  caller so "acked" and "gave up" stay distinguishable in the log.
  Two things stop *this release* re-announcing, and **both** are needed. The
  rung reset (`_escalate_or_recover` zeroes `running_command_num_relaunch` on
  every emptied slot, and one give-up window
  (`NUM_MAX_INVALID_PROBES_COMMANDS` × cycle ≈ 70 s) is far too short to
  climb back to `NUM_MAX_COMMAND_RELAUNCH`) covers a rung earned *inside* a
  give-up window. The **episode latch** covers a rung earned *before* one,
  which is the case the rung argument alone misses — see "Two clocks, not
  one" above. Together they held the measured parity with `main` for the
  flapping shape: **1 alert per 105 min**, where the release without the latch
  produced 5. The supersede **anchor** was already released on that path (in
  `_escalate_or_recover`'s housekeeping arm, outside the `current_command`
  gate) so a fresh command's first legitimate supersede is not throttled.

  **Scope, restated for QS-319.** The rule is now "one alert per announced
  episode", full stop — the reachable-but-disobeying shape is covered too, at
  1 alert per 105 min against 5 on `main`, and the service-call count is
  unchanged at 41. It is still not a blanket rule over unrelated notification
  types: the tag and the latch are both scoped to lost-control, by decision.
- **"One service call per 300 s" is per command *identity*, not per
  load.** The relaunch ladder and the supersede throttle are measured on
  two independent anchors, so a relaunch immediately followed by a
  superseding command can produce two calls inside one nominal window —
  bounded to one extra per episode. Deliberately not coupled: one shared
  clock would let a supersede delay the ladder, or a relaunch delay a
  newer intent.
- **Invariant:** anything meaning "we have lost control" reads
  `is_uncontrollable`, never `running_command_num_relaunch`. The
  counter is resettable; the clock is not.
- **`check_and_relaunch_command` never lets housekeeping mask a device
  error.** The relaunch + escalation run from a `finally` via
  `_finish_command_cycle`, which isolates and logs each half — a
  secondary failure (a raising probe reached again through
  `force_relaunch_command`, or a push blowing up) must not replace the
  real device exception on its way to `QSHome`'s per-load log. The push
  is issued last, for the same reason — and it re-checks the latch at
  send time (review fix QS-319#01/6): the announce branch's latch write
  stays *before* the push (a raising notify must not resurrect the
  storm), but delivery is skipped if user remediation acknowledged the
  episode in between, so a push cannot land for an episode the sensor
  already reports as over.
- **No lock.** `QSDataHandler._update_loads_lock` guards only
  `async_update_loads`; `button.py` calls `user_clean_and_reset`,
  `user_clean_constraints`, `mark_current_constraint_has_done` and
  `async_reset_override_state` straight from a press, unlocked. So the
  command slot **can** be mutated across an `await`: emptied by the
  override-expiry drop (QS-307), or **replaced** by a press that
  launches `CMD_IDLE` of its own (QS-320). The invariant relied upon is
  therefore **ownership**, not emptiness: a completion path writes
  nothing about its own dispatch's outcome unless
  `_slot_still_holds(launched_command_name, launched_generation, site,
  ctxt)` says the slot still carries that dispatch's
  `_running_command_generation` tag. The clock is still only ever cleared alongside the command state
  it describes.

Switching-cost protection (`AbstractDevice`):

- `num_max_on_off` — daily on/off budget.
- `CHANGE_ON_OFF_STATE_HYSTERESIS_S = 600` — minimum delay (10 min)
  between state changes for the same device.
- **Multi-pass adaptation**: try free transitions first; only spend
  the daily budget on the second pass.

3-phase awareness:

- Phase configuration tracked per device.
- Power → per-phase amperage conversion for budgeting checks.

## Key types / structures

- `AbstractDevice` — base. Config, lifecycle, switching cost,
  3-phase awareness.
- `AbstractLoad(AbstractDevice)` — adds constraint surface.
- `PilotedDevice(AbstractDevice)` — devices that pilot other devices
  (e.g., heat pump with aux heater). Tracks client list and per-slot
  demand counts.
- `get_for_solver_constraints()` — the solver's entry point for
  reading active constraints.
- `push_live_constraint(...)` — runtime constraint push.
- `push_agenda_constraints(...)` — calendar / schedule push.
- `external_user_initiated_state` — set when the device state
  changes without a command quiet-solar sent.
- `is_command_suppressed_by_override(time, command)` — hook checked at
  the `launch_command` drop point (after the stacked-command clear,
  before the same-command early-return): a suppressed command is
  DROPPED before `running_command` is set — no ack, no counter
  mutation, nothing for `check_commands` / `force_relaunch_command`
  to resurrect (QS-256). `force_relaunch_command` applies the same
  hook to a stale `running_command` and drops it (clears the running
  slot and relaunch counters — `current_command` is intentionally
  preserved as the last acked command of record, so a later
  non-suppressed launch can still compare against it) instead of
  retrying it against the override (review fix QS-256#02). Default
  False; bistate loads override it.
- `_restored_utc_datetime(value)` — restore-boundary parser for the
  stored override timestamps: tz-naive isoformat strings (legacy /
  hand-edited storage) are coerced to UTC so downstream datetime
  arithmetic never raises (review fix QS-256#02). The persisted payload
  keys themselves are the `STORAGE_KEY_*` constants in `const.py`
  (review fix QS-256#05).
- `unresponsive_since` / `is_uncontrollable` /
  `command_relaunch_delay_s()` / `abandon_running_command(time, reason)`
  / `_drop_running_command(time, reason)` /
  `check_and_relaunch_command(time)` / `_finish_command_cycle(time)` /
  `_is_supersede_throttled(time)` / `_notify_unresponsive(time,
  command)` / `has_unacknowledged_lost_control` /
  `_acknowledge_lost_control(reason)` — the lost-control surface.
  `_notify_unresponsive` is a documented no-op on `AbstractDevice` (the battery
  reaches the same driver and has no notification channel) and is overridden on
  `AbstractLoad` to push one `DEVICE_STATUS_CHANGE_ERROR`, tagged with
  `NOTIFICATION_TAG_LOST_CONTROL_PREFIX + device_id` so the mobile app collapses
  the series (QS-319). `has_unacknowledged_lost_control` is the public,
  per-episode read that the `qs_load_lost_control` binary sensor exposes; it is
  readable on every `AbstractDevice`, including ones that will never expose it,
  so do not add the entity to a battery on the strength of the base-class
  property.
- `_install_running_command(command, time)` /
  `_slot_still_holds(launched_command_name, launched_generation, site,
  ctxt)` — the name is a plain `str`, only ever logged — /
  `_running_command_generation` — the QS-320 dispatch-ownership
  surface, all on `AbstractDevice`. The installer is the **one way in**
  to the command slot and returns the tag the guards compare; absorb
  deliberately bypasses it (an equal-valued object for the *same*
  dispatch must stay generation-neutral). The generation is **never
  reset** — not by `reset()`, not by `abandon_running_command` — because
  a rewind would let a tag captured before the reset compare equal to
  one issued after it. The write-site half of the convention IS now
  test-enforced: `test_every_running_command_write_site_is_sanctioned`
  fails on any `self.running_command` write outside the five sanctioned
  class-qualified sites. Only the never-reset-the-generation half
  remains convention, protected by docstring and AC13 alone.
- `last_command_execution_time` — in-memory causality anchor, set
  only on real `execute_command` successes (via the shared
  `_anchor_causality_guard_if_executed` helper called from
  `launch_command` and `force_relaunch_command`, never on the
  probe-already-set branch) and
  initialized to "now" at storage restore when a `current_command` is
  restored. Never serialized. Cleared by `user_clean_and_reset`,
  which also clears ALL user-override fields (QS-256). The helper is
  **monotonic** (QS-320 review fix #01/1): it sits above the ownership
  guard, so a superseded dispatch resuming after its replacement
  anchored a newer instant must not rewind the causality floor — a
  rewound floor classifies QS's own state change as a user override.

## Common mistakes

- Adding a device only in `ha_model/` without a `home_model/`
  counterpart. The solver can't see it — it's a ghost device.
- Calling `execute_command()` from a test and awaiting the ACK in
  the same call. ACK arrives asynchronously via
  `probe_if_command_set()`; tests need to advance time.
- Bypassing `num_max_on_off` for "important" devices. The whole
  hysteresis pattern depends on the budget being respected
  uniformly.
- Importing `homeassistant.*` into `home_model/load.py`. The two-
  layer boundary is non-negotiable.
- Keying "we have lost control" on `running_command_num_relaunch`.
  `abandon_running_command` no longer zeroes it, but `_ack_command`
  still does — read `is_uncontrollable` instead (QS-304).
- Adding a give-up that empties `running_command` without clearing
  `unresponsive_since`. The clock describes an in-flight command; with the
  slot empty it is ownerless, and the next command inherits supersede
  semantics with zero evidence of its own. "The device is failing harder,
  so keep the clock" sounds right and is wrong (QS-307, from #308) — keep
  the *reason string* honest instead; `_clear_unresponsive` is reason-led
  precisely so a release does not have to claim a recovery.
- Calling `abandon_running_command` on a path that does not go on to
  launch a successor. It preserves the rung by design; use
  `_drop_running_command` instead, or the next command inherits a spent
  ladder and is declared uncontrollable with zero relaunches of its own.
- Comparing a stored timestamp against `time` directly. Use
  `_seconds_since`, or a rewound clock silently freezes whatever the
  comparison gates.
- Adding an `await` in `launch_command` whose exception can escape. The
  slot, the throttle anchor and the rung are mutated around it, so an
  escape leaves them inconsistent — that is exactly how the deadlock was
  re-created once already.
- Using a `MagicMock` as a load in a `check_loads_commands` /
  `update_loads` test. The resulting `TypeError` is swallowed by the
  per-load `except`, `all_ok` stays `True`, and your assertions never
  fire. Use `tests.factories.attach_minimal_load_to_home`.

## See also

- [ha-device-mixin.md](ha-device-mixin.md) — the HA-side counterpart.
- [constraints.md](constraints.md) — the constraint API.
- [piloted-device-and-heat-pump.md](piloted-device-and-heat-pump.md)
  — the `PilotedDevice` subclass.
- [external-control-detection.md](external-control-detection.md) — the
  external-state detection flow.
- [user-override.md](user-override.md) — user-originated state.
- [../principles/two-layer-boundary.md](../principles/two-layer-boundary.md)
  — why `home_model/` never imports HA.
- [../principles/hysteresis-and-switching-cost.md](../principles/hysteresis-and-switching-cost.md)
  — the daily-budget + hysteresis pattern.
