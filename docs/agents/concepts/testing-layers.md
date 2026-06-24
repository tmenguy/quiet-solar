---
title: Testing layers (FakeHass vs real HA)
slug: testing-layers
kind: concept
covers:
  - tests/conftest.py
  - tests/factories.py
  - tests/ha_tests/conftest.py
  - tests/ha_tests/const.py
last_verified: 2026-06-24
---

# Testing layers — FakeHass, factories, and real HA fixtures

## TL;DR

Quiet-solar has a **dual test infrastructure**. Domain / unit tests
use `FakeHass` (a lightweight in-memory HA stand-in) plus factories
that build real domain objects (no mocks). HA-integration tests use
`pytest-homeassistant-custom-component`'s real HA fixtures
(`tests/ha_tests/`). Each layer has its own `conftest.py` —
`tests/conftest.py` for the fast/lightweight layer,
`tests/ha_tests/conftest.py` for the real-HA layer. The two must
stay in sync or tests pass while testing the wrong thing.

## When you need this concept

- Writing a new test — choose the right layer.
- Touching test infrastructure (`conftest.py`, `factories.py`,
  `FakeHass`).
- Investigating "this test passed but production broke" issues —
  often the FakeHass/real-HA divergence.
- Adding a fixture or factory for a new device type.

## Core idea

**Layer 1 — domain / unit tests (`tests/`, FakeHass + factories)**:

- Use cases: solver, constraint, command lifecycle, home-model
  logic.
- Speed: fast (no HA boot).
- Fixtures: `FakeHass` (provides the minimum HA surface domain code
  touches).
- Test doubles: real domain objects built by factories — **not
  mocks**. Mocks lie; factories produce objects with the same
  invariants as production.
- Layer pattern files: `tests/conftest.py`, `tests/factories.py`.

**Layer 2 — HA-integration tests (`tests/ha_tests/`, real HA)**:

- Use cases: config flow, entity creation, service-call wiring,
  HADeviceMixin, real HA service-call behaviour.
- Speed: slow (real HA boot).
- Fixtures: provided by `pytest-homeassistant-custom-component`.
- Layer pattern files: `tests/ha_tests/conftest.py`,
  `tests/ha_tests/const.py`.

**Rule**: pick the lowest layer that exercises the code you're
testing. A solver bug is a domain test; a config-flow bug is an HA
test. Don't run the slow layer for logic that the fast layer can
cover.

## Test-impact inner loop — `--impacted` (QS-276)

Orthogonal to the two *layers* above is a third *execution mode* for
the dev loop: `python scripts/qs/quality_gate.py --impacted`. It uses
`pytest-testmon` to run **only the tests impacted by your change set**
(across both layers), under `--cov=custom_components/quiet_solar`, then
`diff-cover --fail-under=100` asserts the **lines you changed** are
100% covered. On a small diff this is seconds instead of minutes, so it
is the implement-phase default before commit/PR.

**The split that makes this safe.** Whole-repo "100% on every PR" is
NON-NEGOTIABLE — and a test-impact subset can never prove it. So the
guarantee is split: `--impacted` proves *changed lines* locally; the
**whole-repo 100% gate runs authoritatively in CI** on every PR.

**Hard limitation — read this.** `--impacted` guarantees the lines you
*changed* are covered. It does **not** detect coverage lost in
*unchanged* code — e.g. deleting a test that was the sole cover of an
untouched line drops whole-repo coverage but changes no measured line,
so `--impacted` stays green. Only CI's whole-repo gate catches that.
Never treat a green `--impacted` as a substitute for the CI gate.

Fail-safe: a corrupt / non-SQLite `.testmondata` makes the gate select
*all* tests (rebuild from scratch), never silently under-select. New
worktrees seed `.testmondata` + `.mypy_cache` from the main worktree
(`worktree-setup.sh`); `finish-task` refreshes the main baseline via
`--seed-testmon` after a merge.

## Key types / structures

- `FakeHass` — lightweight HA stand-in. Provides `services`,
  `states`, `bus`, the parts domain code touches.
- `tests/factories.py` — real-object builders for every domain
  class. The single source of truth for "how do I build a valid
  X in a test?".
- `tests/conftest.py` — shared fixtures for the fast layer.
- `tests/ha_tests/conftest.py` — shared fixtures for the real-HA
  layer (mostly pytest-homeassistant-custom-component re-exports
  + per-test bootstrap helpers).
- `tests/ha_tests/const.py` — constants specific to the HA layer
  (entity IDs, test config payloads).

## Common mistakes

- Using mocks instead of factories. Mocks pass tests against
  themselves; factories pass tests against the real object's
  invariants.
- Writing a real-HA test for logic the fast layer covers. You
  burn CI time and slow the dev loop.
- Letting FakeHass diverge from real HA. Smoke tests
  (`@pytest.mark.integration`) should verify FakeHass behaves like
  real HA for the operations the fast layer relies on.
- Hardcoding entity IDs / config payloads in tests. They live in
  `tests/ha_tests/const.py` for the HA layer or the factory call
  signatures for the fast layer.

## See also

- [../../workflow/project-rules.md](../../workflow/project-rules.md)
  — the quality-gate entry points (`--quick` for fast iteration).
- [../../workflow/project-context.md](../../workflow/project-context.md)
  — the broader test-style rules.
- [../personas/the-dev.md](../personas/the-dev.md) — the persona
  whose pipeline depends on a fast inner loop.
