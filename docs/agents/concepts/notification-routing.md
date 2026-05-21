---
title: Notification routing
slug: notification-routing
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/home.py
  - custom_components/quiet_solar/ha_model/person.py
last_verified: 2026-05-21
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

## See also

- [qs-home-orchestrator.md](qs-home-orchestrator.md) — the
  broadcast surface.
- [person-trip-prediction.md](person-trip-prediction.md) — the
  per-person tracking that backs routing.
- [off-grid-mode.md](off-grid-mode.md) — the canonical broadcast
  consumer.
- [constraints.md](constraints.md) — the `load_info.originator`
  that powers the "why".
- [../personas/magali.md](../personas/magali.md) — the persona
  that judges trust by notification quality.
