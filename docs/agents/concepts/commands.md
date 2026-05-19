---
title: LoadCommand
slug: commands
kind: concept
covers:
  - custom_components/quiet_solar/home_model/commands.py
last_verified: 2026-05-19
---

# LoadCommand

## TL;DR

A `LoadCommand` is a discrete action at a specific power level — the
atom of quiet-solar's control vocabulary. 10 command types are
ordered by a score (`CMD_ON=100` down to `CMD_OFF=10`); the score
determines which command wins when two constraints overlap on the
same load. Higher score = more aggressive intervention.
`merge_commands()` combines two commands by taking the higher-scored
type and the max power.

## When you need this concept

- Adding a new command type or modifying command merge semantics.
- Implementing constraint conflict resolution.
- Working on a device's `execute_command()` translation.
- Debugging "wrong command was launched" issues.

## Core idea

A `LoadCommand` carries two fields: a `command` string (one of the
10 constants `CMD_CST_*`) and a `power_consign` float (watts). The
command string maps to a score in `commands_scores` — score ordering
is the semantic meaning of the command hierarchy. When two
constraints fire on the same load simultaneously, `merge_commands()`
picks the higher-scored command type and the higher `power_consign`.

Key distinction: `is_auto()` (green/price-aware modes that adapt to
context), `is_green()` (solar-only modes that never import from the
grid), and `is_off_or_idle()` (the device should not consume).

## Key types / structures

- `LoadCommand` — dataclass with `command: str`, `power_consign: float`.
- `commands_scores` — dict mapping command type → integer score.
- `merge_commands(cmd1, cmd2)` — merge two commands by score + max power.
- `copy_command(cmd, power_consign=None)` — copy with optional power
  override.
- `copy_command_and_change_type(cmd, new_type)` — preserve power,
  change command type.
- Singleton commands: `CMD_ON`, `CMD_OFF`, `CMD_IDLE`, `CMD_AUTO_GREEN_ONLY`,
  `CMD_AUTO_GREEN_CAP`, `CMD_AUTO_GREEN_CONSIGN`,
  `CMD_AUTO_FROM_CONSIGN`, `CMD_AUTO_PRICE`, `CMD_FORCE_CHARGE`,
  `CMD_GREEN_CHARGE_ONLY`, `CMD_GREEN_CHARGE_AND_DISCHARGE`.

## Command-score table (descending)

| Score | Constant | Meaning |
|---|---|---|
| 100 | `CMD_CST_ON` | Hard on, no constraint on power. |
| 90 | `CMD_CST_FORCE_CHARGE` | Force-charge regardless of solar / price. |
| 80 | `CMD_CST_AUTO_CONSIGN` | Auto, follows an explicit power setpoint. |
| 70 | `CMD_CST_AUTO_PRICE` | Auto, price-aware. |
| 60 | `CMD_CST_AUTO_GREEN_CONSIGN` | Auto-green with setpoint floor. |
| 50 | `CMD_CST_GREEN_CHARGE_ONLY` | Solar-only charging (battery: charge only, no discharge). |
| 40 | `CMD_CST_AUTO_GREEN` | Solar-aware (default). |
| 30 | `CMD_CST_AUTO_GREEN_CAP` | Solar-aware with a hard cap (less flexible). |
| 20 | `CMD_CST_IDLE` | Device is idle. |
| 10 | `CMD_CST_OFF` | Device is off. |

## Common mistakes

- Comparing `LoadCommand` objects by identity instead of using `==`
  (`__eq__` compares command + power_consign).
- Assuming `merge_commands(a, b) == merge_commands(b, a)` for power
  (it is, via `max`) but writing code that depends on the order of
  the command type (the merge is deterministic but the score
  ordering matters).
- Introducing a new command without updating `commands_scores` — it
  silently scores 0 (via `.get(command, 0)`) and loses every merge.

## See also

- [constraints.md](constraints.md) — constraints produce commands.
- [solver.md](solver.md) — the solver assembles commands into a
  timeline.
- [../principles/strategic-tactical-control.md](../principles/strategic-tactical-control.md)
  — the strategic vs tactical split that determines when commands
  are launched vs when they're overridden.
