---
title: TheAdmin
slug: the-admin
kind: persona
last_verified: 2026-05-19
---

# TheAdmin

## TL;DR

TheAdmin is the homeowner who installed quiet-solar. They're comfortable
with Home Assistant, configure devices, monitor performance, and tweak
priorities. They operate in two modes — **setup** (learning, cautious,
needs guidance) and **operational** (confident, wants efficiency). Trust
is built through transparent diagnostics when things go wrong, not by
the system being perfect.

## When you need this persona

- Designing a config-flow step or device configuration UI.
- Adding a diagnostic sensor, error notification, or troubleshooting
  dashboard tile.
- Changing default behaviour, priority ordering, or any tunable
  parameter — TheAdmin must be able to explain it.
- Writing user-facing copy in `strings.json`.

## Core idea

TheAdmin is the only persona who reads logs, opens GitHub issues, and
modifies `configuration.yaml`. They need:

- **Transparent diagnostics**: every solver decision must be traceable
  to the constraint, score, and tariff window that drove it.
- **Confident defaults**: a fresh install should "just work" — no
  required tuning to get value on day one.
- **Surgical override**: the rare power-user who wants to disable
  filler scheduling for one device, force a charge, or pin a tariff
  must be able to do so without forking the code.

## Characteristic interaction path

```text
Install via HACS → config flow steps → device-type wizard
→ Dashboard with constraint progress, command history, forecast accuracy
→ "Why didn't the car charge last night?" → diagnostic sensors + logs
→ Tune number/select entity → solver re-plans → behaviour shifts
```

## Common mistakes

- Designing a feature that requires TheAdmin to keep tweaking parameters
  every week. If it isn't self-tuning, it isn't trustworthy.
- Burying diagnostic information in logs instead of surfacing it on
  the dashboard. TheAdmin should not have to `tail -f` to understand
  the system.
- Assuming TheAdmin will read documentation. Inline help text in the
  config flow and entity descriptions matters more.

## See also

- [magali.md](magali.md) — the household member who only sees the
  notifications TheAdmin's config produced.
- [the-dev.md](the-dev.md) — the developer (often the same person)
  improving the code.
- [../concepts/dashboard-and-cards.md](../concepts/dashboard-and-cards.md)
  — the two auto-generated dashboards + four JS cards that ARE
  TheAdmin's primary UX.
- [use-cases/external-override.md](../use-cases/external-override.md)
  — what happens when TheAdmin (or anyone) controls a device
  outside quiet-solar.
