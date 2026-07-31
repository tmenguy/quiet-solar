---
title: Notification routing
slug: notification-routing
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
  - custom_components/quiet_solar/ha_model/person.py
last_verified: 2026-07-31
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
  so an uncontrollable **battery** gets the ERROR log but no push.
  Accepted product consequence.

  **How often it fires — stated exactly, because it is easy to overclaim.**
  When QS gives up probing a device it cannot reach, the lost-control clock is
  released so the next command is not mis-scored (QS-307, from #308).
  `_unresponsive_needs_ack` stops that release from **re-announcing an incident
  that was already announced** — without it, a device dropping off the network
  intermittently alerted 5× where `main` alerted once over the same 105
  minutes. That is the flag's entire job.

  It does **not** deduplicate alerts for a device that stays *reachable* and
  simply never obeys. That case re-crosses the threshold about every 18.5
  minutes and alerts each time, here and on `main` alike — pre-existing,
  measured identical, and tracked in
  [#319](https://github.com/tmenguy/quiet-solar/issues/319), which is where a
  general notification policy belongs.

  Nor does it survive a restart: `_unresponsive_needs_ack` is deliberately
  **not** persisted, matching `unresponsive_since`, because both describe an
  in-flight command and a reload wipes the command slot. So a permanently
  flapping device announces once more per HA restart. Persisting it would
  reintroduce a failure QS-307 already had to fix — a latch carried into a
  process where nothing has been announced silences the first genuine
  alert — so if this ever becomes a real complaint, throttle at the channel
  instead. Noted in #319.

The push is issued **last** in `_escalate_or_recover`, and a failure in it
cannot skip the surrounding housekeeping or mask the device exception the
cycle was propagating — `_finish_command_cycle` isolates it. One
consequence to know when counting log lines: if the push raises, a
*second* ERROR is logged for the failed housekeeping, so the
"exactly one ERROR per episode" guarantee holds on the non-raising path
only. The once-only guard is unaffected, because `unresponsive_since` is
written before the await.

### There is no recovery push (QS-304)

The channel is a fire-and-forget `Platform.NOTIFY` mobile push with no
notification id, so nothing can be dismissed or updated — and sending an
`ERROR`-status push to say things are *fine* would be wrong. Recovery
gets one INFO log line. One line in, one line out; no heartbeat, and no
entity — the lost-control state is internal.

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
- Pairing an entry push with a "recovered" push. The channel cannot
  dismiss, so a second push is noise — expose recovery as state, not as
  a notification (QS-304).
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
