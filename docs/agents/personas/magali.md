---
title: Magali
slug: magali
kind: persona
last_verified: 2026-05-19
---

# Magali

## TL;DR

Magali is the household member who lives in a quiet-solar home but
**did not install it and does not configure it**. She receives
notifications, occasionally overrides decisions, and forms her opinion
of the system based on how it handles edge cases — not the happy
path. Override must be a **5-second interaction** or trust erodes.
She is the persona that defines the "magic moment" — the day she plugs
in her car and the system silently does the right thing without her
having to think about energy.

## When you need this persona

- Designing a notification, mobile-app prompt, or any household-facing
  message.
- Building an override flow (extend trip, force charge, cancel
  scheduled run).
- Sizing the response time / number of taps for any user-facing
  action.
- Making a behaviour change that could surprise a household member
  who doesn't read changelogs.

## Core idea

Magali is **passive 95% of the time**. She:

- Plugs in her car, expects it to be ready when she needs it.
- Sees a notification only when something demands her attention.
- Overrides via a single button in the HA mobile app — never opens
  configuration, never reads logs.
- Stops thinking about energy. That's the success metric.

## Characteristic interaction path

```
Magali plugs car in → person presence + trip prediction
→ Constraint created (load_info: {originator: "prediction"})
→ Solver allocates power → charger executes
→ "Done by 7am" notification → Magali ignores it (95% case)
→ Edge case: business trip tomorrow → Magali taps "extend"
→ user_override constraint → solver re-plans → confirmation push
```

## Common mistakes

- Adding a notification per cycle. Magali will mute the integration
  within a week.
- Putting override controls behind multiple taps. The 5-second rule
  is a hard ceiling.
- Forgetting Magali doesn't know what a `LoadConstraint` is. User
  copy must speak in domain terms — car / room / pool — never in
  solver vocabulary.
- Designing for the happy path. Magali judges trust based on the
  recovery story when predictions fail.

## See also

- [the-admin.md](the-admin.md) — who configured the system Magali
  uses.
- [use-cases/magali-plugs-in-car.md](../use-cases/magali-plugs-in-car.md)
  — the end-to-end "magic moment".
- [use-cases/external-override.md](../use-cases/external-override.md)
  — what happens when Magali (or anyone) operates a device outside
  quiet-solar.
