---
title: Solar providers and dampening
slug: solar-providers
kind: concept
covers:
  - custom_components/quiet_solar/ha_model/solar.py
last_verified: 2026-05-19
---

# Solar providers and dampening

## TL;DR

`QSSolar` (`ha_model/solar.py`) tracks live PV production and stitches
together forecasts from **multiple providers** (Solcast, OpenMeteo)
with dampening so transient noise doesn't whip the solver. The
forecast refresh cycle runs every ~30 seconds; failed providers fall
back to the last successful response (and, if stale > 6h, to historic
patterns from the numpy ring buffer). The solver consumes a single
unified forecast — provider arbitration lives entirely in this file.

## When you need this concept

- Integrating a new solar forecast provider.
- Modifying dampening or multi-provider arbitration.
- Working on the forecast refresh cycle.
- Debugging "the system is reacting to every passing cloud" issues.

## Core idea

The provider abstraction is a per-provider class with a uniform
interface (`async fetch_forecast(time_range) -> ProductionSeries`).
`QSSolar` calls each configured provider on the forecast-refresh
cycle, then:

- Dampens raw measurements (low-pass filter on the live PV reading).
- Combines forecasts via a weighted average (provider weight is
  tunable per config-flow entry).
- Caches the merged forecast for the solver and the dashboard.

Failure handling per the resilience strategy:

- Provider API timeout / HTTP error → use last successful response.
- Stale > 6h → fall back to historical consumption / production
  patterns from the 560-day numpy ring buffer.
- Conservative defaults (over-estimate consumption, under-estimate
  production) so the fallback doesn't promise more than the system
  can deliver.

## Key types / structures

- `QSSolar(HADeviceMixin, AbstractDevice)` — the solar tracking
  class.
- Per-provider classes (Solcast, OpenMeteo) implementing the
  fetch-forecast interface.
- Dampening filter on the live measurement path.
- Merged-forecast cache consumed by the solver.

## Common mistakes

- Adding a provider that bypasses the dampening / merging layer.
  The solver expects one forecast; provider arbitration is this
  file's job.
- Hard-coding provider URLs or API keys. Everything comes through
  the config-flow entry.
- Forgetting the stale-fallback path. The forecast layer must
  degrade gracefully — the solver can't gate on "do I have a fresh
  forecast?".
- Treating dampening as smoothing-for-display. It's smoothing-for-
  decisions: undampened raw production whips the solver into
  unstable plans.

## See also

- [solver.md](solver.md) — the consumer of the merged forecast.
- [qs-home-orchestrator.md](qs-home-orchestrator.md) — drives the
  ~30s forecast cycle.
- [../use-cases/solar-surplus-allocation.md](../use-cases/solar-surplus-allocation.md)
  — surplus → load allocation end-to-end.
