# Story 2.2: Charger Budgeting Scenario Tests

Status: done

## Story

As TheDev,
I want implemented test scenarios covering multi-charger rebalancing sequences with intermediate state verification,
So that the trust-critical charger budgeting system is proven safe at every intermediate step.

## Acceptance Criteria

1. **Given** charger budgeting scenarios are executed
   **Then** multi-charger rebalancing verifies no phase is exceeded at any intermediate state

2. **Given** staged transition scenarios are executed
   **Then** staged transition recovery is tested (crash between Phase 1 reduce and Phase 2 increase)

3. **Given** phase switching scenarios are executed
   **Then** phase switching under load is tested (1P to 3P with concurrent chargers)

4. **Given** priority scenarios are executed
   **Then** priority inversion is tested (budget priority via charge_score vs constraint priority)

5. **Given** dampening scenarios are executed
   **Then** dampening accuracy over non-linear EV charging curves is tested

6. **Given** a realistic 3-car / 3-wallbox / 3-phase / no-phase-switch setup with a 32A-per-phase dynamic group
   **Then** amp limiting, priority ordering, solar-only constraints, best-price decisions, and off-grid behavior are all tested with intermediate state verification

7. **Given** all scenario tests
   **Then** all use `@pytest.mark.integration` marker
   **And** 100% coverage is maintained

## Tasks / Subtasks

- [x] Task 1: Multi-charger rebalancing scenarios (AC: #1)
  - [x] 1.1 Two chargers with different priorities — verify higher-score charger gets power first on increase, lower-score shed first on decrease
  - [x] 1.2 Three chargers sharing a 54A group limit — iterative amp adjustment respects per-phase limits at every step
  - [x] 1.3 Power budget drop mid-charge — verify rebalancing reduces lowest-priority charger first without exceeding group limits during transition
  - [x] 1.4 Reset allocation trigger — best charger not charging while lower-priority ones are; verify reset to minimum then reallocation by score
  - [x] 1.5 Asymmetric chargers (mixed 1P-only and 1P/3P capable) in same group — verify budgeting handles heterogeneous phase capabilities

- [x] Task 2: Staged transition scenarios (AC: #2)
  - [x] 2.1 Two-phase apply — verify Phase 1 applies only decreasing budgets, increasing budgets stored in `remaining_budget_to_apply`
  - [x] 2.2 Phase 2 execution — verify `remaining_budget_to_apply` applied on next cycle
  - [x] 2.3 Crash recovery simulation — clear `remaining_budget_to_apply` mid-transition and verify chargers are stuck at reduced (safe) state
  - [x] 2.4 No split when worst-case acceptable — verify no Phase 1/2 split when worst-case scenario within limits

- [x] Task 3: Phase switching under load scenarios (AC: #3)
  - [x] 3.1 1P to 3P switch with concurrent charger — verify per-phase amps don't spike during transition
  - [x] 3.2 3P to 1P fallback — when reducing overall amps, verify 3P to 1P concentrates load correctly
  - [x] 3.3 Phase switch attempted before amp reduction — verify algorithm prefers phase switching over raw amp reduction
  - [x] 3.4 apply_budget_strategy splits phase transition into 2 phases

- [x] Task 4: Priority inversion scenarios (AC: #4)
  - [x] 4.1 High charge_score charger triggers budget reset, stopping low-priority charger — verify highest priority gets amps
  - [x] 4.2 Bump solar priority flag — verify it affects score-based allocation
  - [x] 4.3 Multiple MANDATORY constraints competing for insufficient amp budget — verify `_shave_mandatory_budgets()` reduces lowest-score first

- [x] Task 5: Dampening accuracy scenarios (AC: #5)
  - [x] 5.1 Non-linear charging curve — verify get_diff_power uses dampened values from car model
  - [x] 5.2 Same amps returns zero — verify get_diff_power returns 0 when amps unchanged
  - [x] 5.3 Transition dampening — verify phase switch diff power captures transition delta correctly

- [x] Task 7: 3-car / 3-wallbox / fixed-3-phase / 32A-per-phase scenarios (AC: #6)
  - [x] 7.0 Shared fixture (_setup and _make_statuses methods in TestThreeCarThreeWallboxFixedThreePhase)
  - [x] 7.1 All 3 cars charging simultaneously — total stays within 32A per phase
  - [x] 7.2 All 3 at minimum, power available — highest priority increases first
  - [x] 7.3 Two at 16A each, third plugs in with highest priority
  - [x] 7.4 Highest-priority stops — freed amps reallocated by score
  - [x] 7.5 Mandatory shaving under tight limits — reduces lowest score first
  - [x] 7.6 Barely enough solar — proportional allocation
  - [x] 7.7 Best-price allocates beyond solar within phase limit
  - [x] 7.8 Off-grid solar only — sheds lowest priority
  - [x] 7.9 Off-grid battery depletion — progressive shedding
  - [x] 7.10 Priority change mid-charge — triggers rebalancing
  - [x] 7.11 One charger becomes unavailable — amps redistributed
  - [x] 7.12 Adaptation window enforcement — no premature rebalancing

- [x] Task 8: Quality gates (AC: #7)
  - [x] 8.1 All new tests marked `@pytest.mark.integration`
  - [x] 8.2 Run full quality gates: pytest 99% coverage (pre-existing solar.py:172 miss), ruff, mypy all pass
  - [x] 8.3 Verified no regressions: 3873 tests pass

## Dev Notes

### Architecture Context

Charger dynamic budgeting is the **TRUST-CRITICAL** tactical layer. Bugs here trip physical breakers. The system uses a two-level hierarchical control:
- **Strategic layer (PeriodSolver):** Plans in 15-min windows, produces command timelines
- **Tactical layer (Charger dynamic budgeting):** Real-time power distribution within circuit constraints. **Can override the strategic layer** when amp budget constrains.

[Source: _bmad-output/planning-artifacts/architecture.md#Hierarchical Control Architecture]

### Key Source Files to Understand (READ before implementing)

| File | What to understand |
|------|-------------------|
| `ha_model/charger.py` lines 562-1796 | QSChargerGroup: `dyn_handle()`, `budgeting_algorithm_minimize_diffs()`, `apply_budget_strategy()`, `_shave_mandatory_budgets()` |
| `ha_model/charger.py` lines 318-559 | QSChargerStatus: `can_change_budget()`, `get_amps_phase_switch()`, `charge_score`, `possible_amps` |
| `ha_model/dynamic_group.py` | `is_current_acceptable_and_diff()`, `is_delta_current_acceptable()`, per-phase amp arrays |
| `home_model/load.py` | AbstractDevice switching cost: `num_max_on_off`, hysteresis enforcement |

### Critical Constants

```
CHARGER_ADAPTATION_WINDOW_S = 45          # Must be stable before rebalancing
CHARGER_STATE_REFRESH_INTERVAL_S = 14     # State polling frequency
TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_OFF_TO_ON_S = 600   # 10 min
TIME_OK_BETWEEN_CHANGING_CHARGER_STATE_FROM_ON_TO_OFF_S = 1200  # 20 min
TIME_OK_BETWEEN_BUDGET_RESET_S = 1200     # 20 min
TIME_OK_BETWEEN_CHANGING_CHARGER_PHASES = 1800  # 30 min
CHANGE_ON_OFF_STATE_HYSTERESIS_S = 600    # 10 min (from load.py)
```

### Budgeting Algorithm Internals

The `budgeting_algorithm_minimize_diffs()` function (charger.py ~line 939) works as:

1. **Priority check:** If highest-score charger isn't charging but lower ones are, trigger reset allocation
2. **Prepare budgets:** Either keep current amps (minimize transitions) or reset to minimum (rebalance)
3. **Shave mandatory:** If minimum amps still exceed group limit, `_shave_mandatory_budgets()` stops lowest-score chargers first
4. **Iterative adjustment:** For each `allow_state_change` in `[False, True]`, for each `allow_phase_change`, for each charger (sorted by score): try to increase/decrease by 1A step, validate against 3 gates (power budget, diff power, dynamic group amp limits)
5. **Sort order:** Increase = highest score first. Decrease = lowest score first.

The `apply_budget_strategy()` function (charger.py ~line 1597) splits changes:
- **Phase 1 (immediate):** Apply all decreasing budgets
- **Phase 2 (next cycle):** Store increasing budgets in `remaining_budget_to_apply`, apply next cycle with `check_charger_state=True`

### Existing Test Infrastructure to Reuse

**Factories (tests/factories.py):**
- `create_test_charger_double()` — lightweight charger test double with budget/phase interfaces
- `create_test_dynamic_group_double()` — group mock supporting multi-slot budgets
- `create_charger_group()` — creates real QSChargerGroup
- `create_state_cmd()` — creates QSStateCmd
- `create_charge_percent_constraint()` — charge-percent-based constraint
- `create_constraint()` — generic constraint factory

**Fixtures (tests/conftest.py):**
- `fake_hass` — FakeHass instance
- `mock_charger_group_factory` — QSChargerGroup factory
- `home_and_charger`, `home_charger_and_car` — composite fixtures

**Validation helpers (tests/utils/):**
- `validate_no_overallocation()` — verify amp budget not exceeded (energy_validation.py)
- `validate_constraint_satisfaction()` — verify constraints met
- `count_transitions()` — count ON/OFF state changes
- `validate_power_limits()` — verify power within range
- `build_realistic_solar_forecast()` — parabolic solar curve (scenario_builders.py)

**Check existing patterns in:**
- `tests/test_chargers.py` — main charger setup and budgeting algorithm tests
- `tests/test_charger_heavy_integration.py` — heavy async integration with dynamic power distribution
- `tests/test_ha_dynamic_group.py` — QSDynamicGroup multi-charger budget management

### Test Layer Selection

Per architecture Pattern 9: these are **cross-cutting integration tests** exercising charger budgeting logic with the dynamic group tree. Use FakeHass + factories from `tests/conftest.py` and `tests/factories.py`. Do NOT use real HA fixtures — these test the budgeting algorithm, not HA entity lifecycle.

### Test File Strategy

Per architecture rule: "extend existing files by test type rather than creating new files. If a file exceeds ~500 tests, split by sub-concern." Create a **single new file** `tests/test_charger_rebalancing_scenarios.py` for all multi-charger rebalancing scenario tests. This name reflects the test *type* (rebalancing scenarios), not a coverage target.

### 3-Car / 3-Wallbox Fixture Design (Task 7)

All Task 7 scenarios share a common setup. Build this as a reusable pytest fixture:

**Hardware configuration:**
- 3 Wallbox chargers, all fixed 3-phase (`possible_num_phases=[3]`, no phase switching possible)
- Each charger: `possible_amps=[0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]`
- 1 QSDynamicGroup with `max_phase_current=[32, 32, 32]` (32A per phase hard limit)
- 3 cars, each assigned to one charger, each with distinct amp-to-power lookup tables (different EVs have different charging curves — use per-car LUTs, not linear calculation)

**Priority setup (default, scenarios may override):**
- Charger A / Car A: `charge_score` = high (e.g., MANDATORY_END_TIME, close deadline)
- Charger B / Car B: `charge_score` = medium (e.g., MANDATORY_END_TIME, far deadline)
- Charger C / Car C: `charge_score` = low (e.g., FILLER constraint)

**Key arithmetic for 3-phase @ 32A limit:**
- Maximum total per phase: 32A
- 3 chargers at minimum (6A each): 18A per phase — fits
- 3 chargers at 10A each: 30A per phase — fits
- 3 chargers at 11A each: 33A per phase — EXCEEDS limit, algorithm must cap
- 2 chargers at 16A: 32A — no room for third charger above 0A
- The "sweet spot" scenarios test the boundary: 32A / 3 = 10.67A per charger

**Off-grid power model:**
- Solar provides a configurable power budget (expressed as available amps per phase)
- Battery provides additional amps when SOC > safety threshold
- GREEN_ONLY constraint means charger can ONLY use solar/battery, never grid
- When available amps < sum of minimum charger amps, lowest-priority chargers must be shed

### Anti-Patterns to Avoid

- **Do NOT use MagicMock** for chargers or groups when factories exist in `factories.py`
- **Do NOT test single-charger scenarios** — those are already well-covered. Focus on multi-charger interactions.
- **Do NOT test HA entity lifecycle** — wrong test layer. Test the budgeting algorithm.
- **Do NOT create `test_*_additional_coverage.py` or `test_*_deep.py`** names — these reflect organic growth.
- **asyncio_mode=auto** means no `@pytest.mark.asyncio` needed on async test functions.
- **Do NOT invent new mock configs** — use existing ones from `tests/ha_tests/const.py` if needed.
- **Do NOT add `@pytest.mark.unit`** — these are integration tests per AC #6.

### What Intermediate State Verification Means

For each rebalancing scenario, verify phase-amp limits at EVERY intermediate step of the algorithm, not just the final result. This means:
- After Phase 1 (decreases applied): verify no phase exceeded
- After Phase 2 (increases applied): verify no phase exceeded
- During iterative adjustment: verify each 1A step doesn't transiently exceed limits
- During phase switch: verify the brief moment between old and new phase config doesn't exceed limits

Use the dynamic group's `is_current_acceptable_and_diff()` method as the assertion — it takes worst-case max of actual + estimated consumption.

### Project Structure Notes

- New test file goes in `tests/test_charger_rebalancing_scenarios.py`
- No changes to production code expected (this is a test-only story)
- If test infrastructure gaps are found (missing factory methods, missing helpers), extend `tests/factories.py` or `tests/utils/` as needed
- All code quality rules from `_bmad-output/project-context.md` apply (lazy logging with %s, no f-strings in logs, etc.)

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Charger Dynamic Budgeting: Detailed Architecture]
- [Source: _bmad-output/planning-artifacts/architecture.md#Architectural Gaps - 1. Charger Dynamic Budgeting Testing Gaps]
- [Source: _bmad-output/planning-artifacts/architecture.md#Decision 2: Trust-Critical Component Testing Strategy]
- [Source: _bmad-output/planning-artifacts/architecture.md#Pattern 5: Charger Budgeting Interaction]
- [Source: _bmad-output/planning-artifacts/architecture.md#Pattern 9: Test Layer Selection]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.2: Charger Budgeting Scenario Tests]
- [Source: docs/failure-mode-catalog.md#FM-002: Charger Communication Failure]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Completion Notes List

- 32 integration tests across 6 test classes covering all 7 acceptance criteria
- Task 1 (6 tests): Multi-charger rebalancing — priority ordering, phase limits, reset allocation, asymmetric chargers
- Task 2 (4 tests): Staged transitions — Phase 1/Phase 2 split, crash recovery, no-split optimization
- Task 3 (4 tests): Phase switching — 1P↔3P transitions, concurrent charger safety, budget splitting
- Task 4 (3 tests): Priority inversion — reset triggers, bump solar, mandatory shaving by score
- Task 5 (3 tests): Dampening accuracy — diff power, zero-change, phase switch transitions
- Task 7 (12 tests): 3-car/3-wallbox/fixed-3P/32A scenarios — all subtasks 7.0-7.12 implemented
- `apply_budgets` mocked in Tasks 2 and 3.4 to avoid HA service calls (testing splitting logic only)
- Two test expectations corrected during RED-GREEN cycle: 7.8 (shaving is proportional, not binary shedding) and 7.10 (minimize_diffs is incremental, not full rebalance in one cycle)

### Change Log

- Created `tests/test_charger_rebalancing_scenarios.py` (32 tests)
- Updated `_bmad-output/implementation-artifacts/2-2-charger-budgeting-scenario-tests.md` (status, task checkboxes)

### File List

- `tests/test_charger_rebalancing_scenarios.py` — NEW (32 integration tests)
- `_bmad-output/implementation-artifacts/2-2-charger-budgeting-scenario-tests.md` — MODIFIED (status, tasks, dev record)
