# Story 2.3: Constraint Interaction Boundary Tests

Status: done

## Story

As TheDev,
I want implemented test scenarios covering constraint type interactions under resource exhaustion,
So that the system's behavior at constraint boundaries is explicit and tested.

## Acceptance Criteria

1. MANDATORY_END_TIME vs exhausted `num_max_on_off` switching budget is tested
2. Mandatory constraint vs exhausted charger amp budget is tested
3. Multiple MANDATORY constraints competing for insufficient power is tested
4. Constraint type transitions under resource pressure are verified
5. All scenarios use `@pytest.mark.integration` marker

## Tasks / Subtasks

- [x] Task 1: Switching budget exhaustion boundary tests (AC: #1)
  - [x] 1.1 MANDATORY_END_TIME constraint when `num_on_off >= num_max_on_off` — verify constraint cannot create new ON segments
  - [x] 1.2 MANDATORY_END_TIME with budget for exactly 2 more transitions — verify it uses them (note: isolated ON segment costs 2 transitions)
  - [x] 1.3 Two-pass adaptation: verify pass 1 (free transitions only) is attempted before pass 2 (budget-spending transitions)
  - [x] 1.4 Segment extension (no switch cost) still works when budget is exhausted — verify extending adjacent ON segments is allowed
  - [x] 1.5 MANDATORY_AS_FAST_AS_POSSIBLE with exhausted budget — verify behavior differs from MANDATORY_END_TIME (both are mandatory but ASAP has higher score)

- [x] Task 2: Amp budget exhaustion with mandatory constraints (AC: #2)
  - [x] 2.1 Single mandatory constraint where minimum power step exceeds available amps — verify constraint gets no allocation
  - [x] 2.2 Mandatory constraint where only the smallest power step fits in available amps — verify it uses that step
  - [x] 2.3 Mandatory constraint with amp budget dropping mid-window (amps available in early slots but exhausted in later slots) — verify adapt_power_steps_budgeting handles cross-slot transitions
  - [x] 2.4 Two mandatory constraints on same charger group with combined minimum exceeding group amp limit — verify higher-score constraint gets priority

- [x] Task 3: Multiple MANDATORY constraints competing for insufficient power (AC: #3)
  - [x] 3.1 Two MANDATORY_END_TIME constraints with same deadline, insufficient total power — verify score-based allocation (higher score gets more power)
  - [x] 3.2 MANDATORY_AS_FAST_AS_POSSIBLE vs MANDATORY_END_TIME competing for limited solar+grid — verify ASAP allocated first
  - [x] 3.3 Three mandatory constraints where only two can be satisfied — verify lowest-score constraint is the one partially/not fulfilled
  - [x] 3.4 Two mandatory constraints with different deadlines — verify earlier deadline gets priority when both can't be fully met
  - [x] 3.5 User-originated mandatory vs system-originated mandatory — verify `from_user` flag in scoring gives user constraint priority

- [x] Task 4: Constraint type transitions under resource pressure (AC: #4)
  - [x] 4.1 Off-grid mode degrades MANDATORY_AS_FAST_AS_POSSIBLE to MANDATORY_END_TIME — verify `_degraded_type` is used
  - [x] 4.2 Constraint pushed beyond deadline repeatedly (`pushed_count > 4`) is skipped — verify `skip = True`
  - [x] 4.3 Constraint promoted to ASAP after repeated deadline misses (`pushed_count <= 4`) — verify type change to MANDATORY_AS_FAST_AS_POSSIBLE
  - [x] 4.4 Off-grid transition preserves active constraint progress — verify `current_value` continuity during grid mode change
  - [x] 4.5 Switching budget + amp budget both exhausted simultaneously — verify constraint handles dual resource exhaustion gracefully (no crash, proper degradation)

- [x] Task 5: Verify 100% coverage maintained (AC: all)
  - [x] 5.1 Run full test suite with coverage gate — 3862 passed, 100% coverage
  - [x] 5.2 Run ruff check + format — all passed
  - [x] 5.3 Run mypy — no issues found

## Dev Notes

### Architecture Context — MUST READ

This story implements trust-critical tests for **Architectural Gap #2** (Constraint Interaction Testing — HIGH RISK) from the architecture document. The architecture explicitly states: "Boundary between constraint types and physical limits needs explicit test coverage."

**Decision Map entry**: "If you're implementing a constraint change → You need to understand LoadConstraint hierarchy, solver allocation order, switching budget interaction, command score semantics → Key files: `constraints.py`, `solver.py`, `load.py`"

**Risk level**: HIGH (trust) — constraint interaction bugs cause missed commitments (car not charged, load not run). These tests verify the system's promise to household members.

### Constraint System Architecture Summary

**Priority tiers** (higher number = higher priority):
- `CONSTRAINT_TYPE_FILLER_AUTO` = 1
- `CONSTRAINT_TYPE_FILLER` = 3
- `CONSTRAINT_TYPE_BEFORE_BATTERY_GREEN` = 5
- `CONSTRAINT_TYPE_MANDATORY_END_TIME` = 7
- `CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE` = 9

**Key properties** (`constraints.py`):
- `is_mandatory`: `self.type >= CONSTRAINT_TYPE_MANDATORY_END_TIME` (type >= 7)
- `as_fast_as_possible`: `self.type >= CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE` (type >= 9)

**Scoring formula** (`constraints.py:291-313`): Score is a composite of energy_score + load_score + type_score + user_score. MANDATORY constraints always score >10^12 higher than FILLER. `from_user=True` adds another ~10^13.

### Switching Budget Mechanism (`num_max_on_off`)

**Location**: `home_model/load.py`

- `num_max_on_off`: Configurable daily maximum state transitions (forced to even number)
- `num_on_off`: Counter of transitions used today (incremented in `_ack_command()`, lines 519-530)
- `reset_daily_load_datas()` (line 452): Resets daily counter
- `get_first_unlocked_slot_index()` (lines 455-464): Uses `bisect_left()` to find first slot where transitions are allowed (hysteresis enforcement)
- `CHANGE_ON_OFF_STATE_HYSTERESIS_S` = `max(10 * 60, SOLVER_STEP_S // 2)` (~600s / 10 min)

**Two-pass adaptation** (`constraints.py:1343-1348`):
```python
if self.load.num_max_on_off is not None and not self.support_auto:
    remaining_switches = self.load.num_max_on_off - self.load.num_on_off
    num_passes = 2  # Pass 1: free transitions only, Pass 2: budget-spending
else:
    remaining_switches = None
    num_passes = 1
```

When budget is exhausted (`remaining_switches = 0`):
- Pass 1: Only transitions that cost 0 switches (segment extensions)
- Pass 2: Would allow spending switches, but with 0 remaining, still blocked
- Net effect: Only adjacent segment extensions allowed, no new ON segments

### Amp Budget Mechanism

**Location**: `ha_model/charger.py`, `ha_model/dynamic_group.py`

- `QSDynamicGroup.is_delta_current_acceptable()` (`dynamic_group.py:91-117`): Recursive validation up the group tree — checks new_amps_consumption <= dynamic_group_max_phase_current
- `is_current_acceptable_and_diff()`: Execution-phase check taking worst-case max
- `adapt_power_steps_budgeting()` (`constraints.py`): Filters power steps that exceed available amps per slot
- Per-phase operations: `add_amps()`, `diff_amps()`, `is_amps_greater()`, `min_amps()`, `max_amps()`

When amp budget is exhausted:
- `adapt_power_steps_budgeting()` filters all power steps that exceed available amps
- If minimum power step exceeds available amps, constraint gets zero allocation for that slot
- Constraint may be partially fulfilled (some slots allocated, others not)

### Solver Allocation Algorithm

**Location**: `home_model/solver.py`

1. **Mandatory constraints allocated first** (lines 856-899): Sorted by score descending, `always_use_available_power_only=False` (can use grid)
2. **Battery optimization** (lines 906-944): Adjusts battery charge/discharge
3. **Non-mandatory constraints** (lines 1047-1073): `always_use_available_power_only=True` (solar only)
4. **Forced slot protection** (`constraints.py:1781, 1895`): `_get_forced_slot_commands()` prevents hysteresis-locked slots from being overridden

When multiple MANDATORY constraints compete:
- Higher-score constraint allocated first, gets more power
- Lower-score constraint gets remaining power (may be insufficient)
- Solver does NOT fail — it logs and continues with partial satisfaction

### Constraint Type Transitions

**Off-grid degradation** (`constraints.py:194-202`):
```python
@property
def type(self) -> int:
    if self.is_off_grid():
        return self._degraded_type  # capped at MANDATORY_END_TIME
    return self._type
```

**Deadline miss promotion** (`load.py:1462-1465`):
```python
if c.pushed_count > 4:
    c.skip = True  # Give up
else:
    c.type = CONSTRAINT_TYPE_MANDATORY_AS_FAST_AS_POSSIBLE
    c.end_of_constraint = new_constraint_end
```

### Test Infrastructure — MUST REUSE

**Test layer**: Domain logic tests in `tests/` using FakeHass + factories (Pattern 9 from architecture). Do NOT use real HA fixtures for these tests.

**Existing factories** (`tests/factories.py`):
- `create_constraint()` — creates `MultiStepsPowerLoadConstraint` with configurable type, time, values
- `create_charge_percent_constraint()` — charge-specific constraints
- `MinimalTestLoad` — lightweight load with configurable `num_max_on_off`
- `create_test_car_double()` / `create_test_charger_double()` / `create_test_dynamic_group_double()` — test doubles

**Existing test utilities** (`tests/utils/`):
- `scenario_builders.py`: `build_realistic_solar_forecast()`, `build_realistic_consumption_forecast()`, `build_variable_pricing()`, `create_car_with_power_steps()`, `create_test_battery()`
- `energy_validation.py`: `validate_energy_conservation()`, `validate_constraint_satisfaction()`, `count_transitions()`, `validate_power_limits()`, `validate_no_overallocation()`

**Existing test patterns to follow** (from `test_coverage_constraints.py` and `test_solver_constraint_allocation.py`):
- `_FakeLoad` / `_FakeLoadForCoverage` classes for constraint isolation tests
- `TestLoad` for solver integration tests
- `PeriodSolver(with_self_test=True)` for solver-level validation
- numpy arrays for commands/durations/power slots

**Key existing test files with proven patterns**:
- `tests/test_coverage_constraints.py` (~172 tests) — constraint budget and adaptation tests, use similar patterns for new boundary tests
- `tests/test_solver_constraint_allocation.py` — solver-level multi-constraint priority tests, extend for competition scenarios
- `tests/test_constraints.py` (~65 tests) — core constraint type tests

### Where to Write Tests

Create new test scenarios in **existing test files** where the test type matches:
- Switching budget boundary tests → extend `tests/test_coverage_constraints.py`
- Amp budget boundary tests → extend `tests/test_coverage_constraints.py` (constraint-level) or create a new focused file if >20 tests
- Solver competition tests → extend `tests/test_solver_constraint_allocation.py`
- Type transition tests → extend `tests/test_grid_transition_constraints.py` for off-grid transitions, `tests/test_coverage_constraints.py` for budget-related transitions

If a single file would receive too many new tests (>30), create `tests/test_constraint_interaction_boundaries.py` as a dedicated boundary test file. Follow existing naming: use `@pytest.mark.integration` on all tests.

### Anti-Patterns to Avoid

- Do NOT use MagicMock for constraints or loads — use factories from `factories.py`
- Do NOT test pure domain logic through real HA fixtures — use FakeHass + factories
- Do NOT invent new mock configs — check `tests/ha_tests/const.py` first
- Do NOT duplicate existing tests — ~300 constraint tests already exist; check first
- Do NOT test solver internals directly — test through the public API (`solve()`)

### Existing Coverage Gaps (from analysis)

These are the specific gaps this story fills:
1. **Switching + amp budget simultaneous exhaustion** — not tested anywhere
2. **Multiple MANDATORY constraints with insufficient resources** — only basic priority tested, no resource exhaustion
3. **Constraint type degradation under dual resource pressure** — only individual transitions tested
4. **Cross-slot amp budget changes** — adapt_power_steps_budgeting not tested with varying per-slot budgets
5. **User vs system mandatory constraint priority** — `from_user` flag in scoring not tested under competition

### Project Structure Notes

- All constraint domain logic: `custom_components/quiet_solar/home_model/constraints.py`
- Load base with switching budget: `custom_components/quiet_solar/home_model/load.py`
- Solver: `custom_components/quiet_solar/home_model/solver.py`
- Amp budgeting: `custom_components/quiet_solar/ha_model/charger.py` + `ha_model/dynamic_group.py`
- Constants: `custom_components/quiet_solar/const.py` (CONSTRAINT_TYPE_*, CHANGE_ON_OFF_STATE_HYSTERESIS_S, SOLVER_STEP_S)
- Tests: `tests/test_coverage_constraints.py`, `tests/test_solver_constraint_allocation.py`, `tests/test_constraints.py`, `tests/test_grid_transition_constraints.py`

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Architectural Gaps] — Gap #2: Constraint Interaction Testing (HIGH RISK)
- [Source: _bmad-output/planning-artifacts/architecture.md#Decision 2] — Trust-Critical Component Testing Strategy
- [Source: _bmad-output/planning-artifacts/architecture.md#Pattern 9] — Test Layer Selection
- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.3] — Acceptance criteria
- [Source: _bmad-output/project-context.md#Testing Rules] — 100% coverage mandatory, factory patterns
- [Source: docs/failure-mode-catalog.md#FM-004] — Solver infeasibility (related: constraint competition may cause partial infeasibility)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References
None

### Completion Notes List
- Created `tests/test_constraint_interaction_boundaries.py` with 19 boundary tests across 4 test classes
- Discovered that `CMD_ON`/`CMD_IDLE` are `LoadCommand` instances, not strings — must use `CMD_CST_ON`/`CMD_CST_IDLE` when constructing new `LoadCommand` objects
- Discovered that `LoadConstraint.__init__` does NOT use the `type` setter — `_degraded_type` must be passed explicitly via `degraded_type=` parameter
- Discovered that creating an isolated ON segment costs 2 transitions (OFF→ON + ON→OFF boundaries), not 1 — adjusted test 1.2 accordingly

### File List
- `tests/test_constraint_interaction_boundaries.py` — NEW: 19 boundary tests for constraint interaction under resource exhaustion
