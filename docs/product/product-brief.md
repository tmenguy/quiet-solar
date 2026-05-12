---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments: []
date: 2026-03-17
author: Thomas
---

# Product Brief: quiet-solar

## Executive Summary

Quiet Solar is an open-source Home Assistant custom component that optimizes household solar energy self-consumption. It orchestrates all controllable loads in a home — EV charging, battery storage, HVAC, pool heating, hot water boilers, and more — using a constraint-based solver that makes intelligent real-time decisions about what should run and when. Quiet Solar takes a whole-house, people-aware approach: it observes household patterns, predicts upcoming needs, and optimizes energy allocation — all through a family-friendly interface that makes solar optimization accessible to entire households, not just the tech-savvy owner.

---

## Core Vision

### Problem Statement

Homeowners with solar installations waste significant energy because their household devices — EV chargers, heat pumps, pool pumps, boilers, batteries — operate independently without awareness of solar production, energy prices, or household needs. Without intelligent coordination, solar energy flows back to the grid at unfavorable rates, cars aren't charged when family members need them, and electricity bills remain unnecessarily high.

### Problem Impact

- **Financial**: Excess solar exported at low grid rates instead of being self-consumed; high electricity bills despite significant solar investment
- **Quality of life**: Cars uncharged when needed for trips; homes too cold in winter or too hot in summer due to poor timing of HVAC; constant mental load on the homeowner to manually manage device schedules
- **Wasted potential**: Solar panels underperforming their true value because consumption isn't aligned with production

### Design Philosophy

Quiet Solar follows an **observe-predict-optimize** approach. It continuously observes household state, predicts upcoming energy needs and production, and optimizes the allocation of power across all controllable devices. This is the right engineering approach because home energy is inherently unpredictable — simple rules and manual schedules break under real-world variability, while a constraint-based solver handles uncertainty gracefully and makes the best possible decision at every moment.

### Design Principles

- **Whole-house orchestration**: Manages all controllable loads together — EVs, battery, HVAC, pool, boilers — optimizing across them simultaneously rather than in isolation
- **Observe-predict-optimize**: Learns household patterns, forecasts upcoming needs, and continuously makes the best possible allocation decisions with available information
- **Embrace uncertainty**: Accepts that predictions will sometimes be wrong. The system earns trust not by being perfect, but by being transparent about its decisions and making recovery graceful — clear notifications, easy overrides, and automatic adaptation
- **Family-friendly by design**: Built for entire households — balancing deep automation with the right amount of transparency and human interaction. Household members should delight in the experience, not merely tolerate it
- **Open-source and generic**: Designed to work for any home with solar and controllable loads, not tuned to a specific setup — crafted with care to be the most elegant and enjoyable solution to this problem

---

## Target Users

- **TheAdmin** — homeowner with solar, comfortable with Home Assistant. Configures devices, monitors performance, tweaks priorities. Two modes: setup (learning, cautious, needs guidance) and operational (confident, wants efficiency). Trust is built through transparent diagnostics when things go wrong, not through being perfect.

- **TheDev** — developer (often the same person as TheAdmin) who finds joy in solving real family problems through code. Building mode: smooth pipeline, no manual GitHub ops, expressive tests. Exploring mode: agentic pairing, fast feedback. Allergic to bureaucracy.

- **Household members** (e.g., Magali) — passive 95% of the time; receive notifications and occasionally override. Override must be a 5-second interaction or trust erodes.

## Optimization Hierarchy

1. **Maximize free solar self-consumption** — solar is always free; use it for as much consumption as possible.
2. **Minimize grid energy cost** — for remaining needs, schedule grid consumption at cheapest tariff windows.
3. **Maintain comfort and reliability** — honor all commitments to household members.

## Success Metrics

- **Self-consumption rate (%)** — primary indicator
- **Grid export (kWh)** — lower is better
- **Cost savings (counterfactual)** — "what would today have cost without quiet-solar?"
- **Execution accuracy** — target 100%; commands produce expected results
- **Missed commitments** — target zero
- **User override frequency** — decreasing trend = improving predictions
- **Magali's trust indicator** — household members stop thinking about energy

## Product Scope (Shipped)

Whole-house energy awareness across solar, battery, EV chargers, cars, climate, pool, boilers. Constraint-based solver in 15-minute windows. People-aware automation with trip prediction and family member overrides. Smart device handling (physical override detection, free-solar-only mode, boost-only devices). Off-grid resilience mode. Auto-generated dashboards with custom JS cards.

100% test coverage is mandatory and non-negotiable.

## Near-term focus

1. **CI/CD & development workflow** — automated testing, structured agentic workflow, release discipline
2. **Bug fixes** through the structured pipeline
3. **Solver improvements** for edge cases
4. **Documentation & community readiness**

## Out of scope (future)

Proactive recommendations, gamification, dynamic tariff support, new device types, enhanced prediction, richer notification UX.

> Technical details about the constraint-based solver, device implementations, and internal data flow live in
> [docs/product/architecture.md](architecture.md). The original full product brief is preserved at
> `_qsprocess_opencode/product/product-brief.md` (untouched) for OpenCode pipeline compatibility.
