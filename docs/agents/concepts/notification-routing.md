---
title: Notification routing
slug: notification-routing
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
  - custom_components/quiet_solar/ha_model/person.py
  - custom_components/quiet_solar/ha_model/device.py
  - custom_components/quiet_solar/home_model/load.py
last_verified: 2026-08-19
---

# Notification routing

## TL;DR

Notifications are routed per-`QSPerson` — each household member has
their own `mobile_app` entity, their own subscription list (daily
forecasts, constraint changes, errors, off-grid alarms), and their
own quiet-hours. `QSHome` owns the broadcast helpers; `QSPerson`
owns the per-person delivery and preferences. Constraint-change
notifications carry the `load_info` originator so the user can tell
why a decision was made.

## When you need this concept

- Adding a new notification trigger (e.g., "battery hit floor").
- Designing per-person preferences (quiet hours, channel
  selection, opt-in flags).
- Touching the broadcast helpers used by off-grid mode.
- Debugging "Magali got 50 notifications today" issues.

## Core idea

Three notification categories:

- **Per-person scheduled**: daily forecast / morning brief. Routed
  to one person's `mobile_app`.
- **Per-person event**: constraint change, override confirmation,
  command failure. Routed to the affected person.
- **Broadcast**: off-grid alarm, grid restored, system-wide error.
  Routed to *every* configured `QSPerson` at high priority.

The `load_info.originator` on a constraint propagates into the
notification body: "Your car's charge target was raised by *Magali's
override*", or "by *tomorrow's predicted commute*". Without
`load_info`, the notification is mute about the why — which erodes
trust.

Quiet hours, channel selection (alarm vs default), and rate-limiting
are per-person. The broadcast path ignores quiet-hours (an outage
alarm at 3am is supposed to wake you).

## Key types / structures

- Broadcast helper on `QSHome` (routes through every `QSPerson`).
- `QSPerson.notify(category, body, priority)` — per-person delivery.
- Per-person preferences (quiet hours, channel, opt-in flags) live
  on `QSPerson`.
- `DEVICE_STATUS_CHANGE_ERROR` producers — the newest is
  `AbstractLoad._notify_unresponsive` (QS-304), fired from
  `check_and_relaunch_command` when a load crosses the lost-control threshold
  (~1050 s of unacked relaunches). It is `AbstractLoad`-only:
  the base `AbstractDevice._notify_unresponsive` is a documented no-op,
  so an uncontrollable **battery** latches and logs but never pushes, and
  gets no `qs_load_lost_control` entity either (the dispatch arm is
  `isinstance(device, AbstractLoad)`). Accepted product consequence: from
  QS-319 its per-climb trace is the INFO "clock re-armed … not
  re-notifying" line rather than one ERROR per ~18.5 min.
  A charger with **no car and no `mobile_app`** is the mirror image: it
  announces, latches and pages nobody. Also accepted — the binary sensor is
  the surface for that case.

  **How often it fires — stated exactly, because it is easy to overclaim.**
  **One alert per announced episode** (QS-319). `_unresponsive_needs_ack` is
  set by the announce branch of `_escalate_or_recover` itself, *before* the
  await, so a notify service that raises cannot resurrect the storm. An
  episode ends on exactly four things, and every statement of the rule must
  list all four:

  1. a **real ack** (`_ack_command` → `_clear_unresponsive(...,
     CONFIRMED)`) — the device answered;
  2. **proven contact on a slot that changed hands** (QS-320,
     `_confirmed_contact_on_disowned_slot`) — a `True` result reaching the
     disowned-slot bail-out in `launch_command` or `check_commands`: the
     device answered about a dispatch that was superseded mid-await, so the
     ack is withheld but the episode still ends, and the successor's
     inherited rung resets so the ladder cannot instantly re-announce;
  3. **explicit user remediation** — `_acknowledge_lost_control`, called from
     `user_clean_and_reset` (the reset button) and from the
     `qs_enable_device` setter's `enabled != self._enabled` guard;
  4. a **process restart or config-entry reload**, where nothing restores the
     latch (see below).

  This covers both shapes. A device dropping off the network intermittently
  alerted 5× where `main` alerted once over 105 minutes (QS-307, from #308);
  a device that stays *reachable* and simply never obeys alerted 5× over the
  same window on `main` and on the QS-307 branch alike (QS-319). Both are now
  1. The service-call count is unchanged at 41 over that horizon — the fix
  changes **alerts only**, never how hard QS retries.

  The second shape needed one more change: the "command slot emptied with no
  successor" release in `_escalate_or_recover` used to claim
  `ContactEvidence.CONFIRMED`. That was a **fake ack** — *we* emptied the
  slot, nobody answered — and it is now `UNKNOWN`. It matters because it is
  the *production* ordering: `check_loads_commands` runs before the solver's
  `launch_command`, so a supersede-drop leaves the slot empty across the cycle
  boundary and the next cycle reaches that release. Without it the latch would
  be cleared roughly once per load-management cycle and the once-per-episode
  rule would do nothing in the field.

  Nor does the latch survive a restart: it is deliberately **not** persisted,
  matching `unresponsive_since`. Nothing *clears* it on restart — the device
  object is simply rebuilt with `_unresponsive_needs_ack = False` by
  `__init__`, so do not hunt for clearing code. So a permanently broken device
  announces once more per HA **restart or config-entry reload** (reloads are
  much more frequent — any options edit), and `qs_load_lost_control` reads
  `off` for the ~18 minutes the ladder takes to re-cross. That window is a
  deliberate consequence, not a bug: for those minutes QS genuinely does not
  know the device is broken, and reporting "not currently in an announced
  episode" is honest. Persisting would reintroduce a failure QS-307 already had
  to fix — a latch carried into a process where nothing has been announced
  silences the first genuine alert.

  **Delivery: the push carries a collapsing tag** (QS-319). Lost-control
  pushes set `data.tag` to
  `f"{NOTIFICATION_TAG_LOST_CONTROL_PREFIX}{device_id}"`. HA's mobile-app
  notify platform treats `tag` as a replace-key on both Android and iOS, so
  repeated alerts for one load replace each other instead of stacking. The tag
  keys on the *config-derived* `device_id`, so it is stable across restarts;
  for a charger it identifies the failing **charger**, not the recipient, so
  swapping cars mid-episode does not re-alert. Scoped to lost-control pushes
  only (a generic per-type tag would collapse unrelated constraint
  notifications), and shipped **inside** `data`, never top-level — standard
  notify platforms ignore unknown `data` keys. Two same-type loads whose
  names slugify identically share a `device_id` and therefore share the tag,
  so their pushes replace each other on the phone — inherited from the
  pre-existing (and unlikely) `device_id` slug collision, where entity
  `unique_id`s collide first; not worth its own guard.

  The nested `data` dict stays **conditional** on `mobile_app_url is not None
  or notification_tag is not None`. Creating it unconditionally would ship
  `"data": {}` on every previously-bare notification, a shape change to a path
  shared by every QS notification.

The push is issued **last** in `_escalate_or_recover`, and a failure in it
cannot skip the surrounding housekeeping or mask the device exception the
cycle was propagating — `_finish_command_cycle` isolates it. One
consequence to know when counting log lines: if the push raises, a
*second* ERROR is logged for the failed housekeeping, so the
"exactly one ERROR per episode" guarantee holds on the non-raising path
only. The once-only guard is unaffected, because `unresponsive_since` is
written before the await.

### There is no recovery push (QS-304)

Still true, and the reasoning is now the *right* one: sending an
`ERROR`-status push to say things are *fine* would be wrong. Recovery gets
one INFO log line. One line in, one line out; no heartbeat.

Two things that used to justify it are **false since QS-319** and must not be
repeated: the push is no longer id-less (it carries a `data.tag` and the
mobile app collapses the series), and the state is no longer invisible — the
`qs_load_lost_control` PROBLEM binary sensor exposes
`has_unacknowledged_lost_control` on every commandable load, which is
precisely "expose recovery as state, not as a notification". The recovery
*policy* is unchanged; only these two factual claims were wrong.

### Charger faults alert the person AND the household (QS-346)

A charger in a human-fix state (`is_charger_faulted` — OCPP
`Faulted`/`Unavailable`, Wallbox `Error`/…, or plain
`STATE_UNKNOWN`/`STATE_UNAVAILABLE`) alerts **once per fault episode**
on **both** channels at once:

- the attached car's person, via the charger's `on_device_state_change`
  override (`DEVICE_STATUS_CHANGE_ERROR`), which resolves to the car's
  `current_forecasted_person`, and is silently dropped at
  `device.py` when no person is resolvable (`mobile_app is None`); and
- the whole household, via `QSHome.async_notify_all_mobile_apps` (the
  guaranteed channel, critical-alert), title
  `"Charger error — action needed"`.

The alert is gated by a **per-cycle state machine** in
`check_load_activity_and_constraints`, not a rising edge:
`_charger_fault_since` is set on the first faulted cycle and cleared the
moment the fault clears; the alert fires (and latches
`_charger_fault_notified`) only once the fault has held **continuously**
for `CHARGER_FAULT_NOTIFY_DEBOUNCE_S` (120 s). The debounce matters
because `_unknown_state_vals` also bundles the `unknown`/`unavailable`
blips of an integration reload or brief connection loss — without it
each blip would fire a critical broadcast. Both latch fields are outside
every reset path, so a mid-episode `reset(keep_commands=True)` does not
re-alert; an HA restart re-alerts once (in-memory). The message names
the charger, the car (`self.car or _last_attached_car`) and the raw
status, and asks the user to unplug/replug; with no car it falls back to
"Please check the charger". These strings are hardcoded English (no
`strings.json`).

### Off-grid broadcast is a background task (QS-349)

On an off-grid state-change transition, `_off_grid_entity_state_changed`
schedules the mobile broadcast as a **fire-and-forget background task**
(`hass.async_create_task`) **before** it awaits
`_compute_and_apply_off_grid_state`. Push notifications route through
cloud services that may hang exactly during an outage, so they must
never sit in the battery-restore critical path; scheduling the alarm
before the apply also means a hung apply cannot delay the alarm. The
task body and the apply are each wrapped in a log-only `except Exception`
(background-task allowance, `exc_info=True`) so a raising notify handler
or a failing apply is logged with its traceback and the listener
survives. The (title, message) content
and the transition conditions (including the `FORCE_ON_GRID` override
variants and the recovery messages) are unchanged — factored into the
`_off_grid_transition_notification` helper.

## Common mistakes

- Sending a notification per cycle. The 7s load management cycle
  fires too often for human attention; batch / dedup or the user
  mutes the integration.
- Bypassing the per-person routing for "important" notifications.
  Anything that should reach everyone is a broadcast — use the
  helper.
- Forgetting `load_info` on the constraint that drives the notif.
  Trust depends on traceable "why".
- Hard-coding the recipient. Always route through `QSPerson` so
  per-person preferences apply.
- Pairing an entry push with a "recovered" push. An `ERROR`-status push
  saying things are fine is wrong — expose recovery as state, not as a
  notification (QS-304). QS-319 did exactly that: read
  `binary_sensor.<load>_lost_control`.
- Clearing a lost-control *flag* without also resetting the evidence
  behind it. QS-304: anything that clears `unresponsive_since` must also
  reset the relaunch rung — otherwise the next cycle re-crosses the
  threshold instantly and pushes again with no fresh evidence. Route it
  through `_drop_running_command`.

## See also

- [qs-home-orchestrator.md](qs-home-orchestrator.md) — the
  broadcast surface.
- [person-trip-prediction.md](person-trip-prediction.md) — the
  per-person tracking that backs routing.
- [off-grid-mode.md](off-grid-mode.md) — the canonical broadcast
  consumer.
- [constraints.md](constraints.md) — the `load_info.originator`
  that powers the "why".
- [qs-home-orchestrator.md](qs-home-orchestrator.md) "Dashboard
  sections auto-migration" — QS-194 added migration logic to
  `QSHome.add_device` but no notification-related behaviour was
  touched (re-verified under review-fix #03).
- [../personas/magali.md](../personas/magali.md) — the persona
  that judges trust by notification quality.
