# Story bug-Github-#134-climate-off-override-not-respected: Climate Load Override 'Off' Not Respected

issue: 134
branch: "QS_134"

Status: ready-for-dev

## Story
As a user with a climate load in force mode,
I want my manual override back to the base state to properly cancel any active override,
so that the system respects my intent instead of reverting to a stale override constraint.

## Acceptance Criteria

1. Given a climate load in force-off mode with an active heat override constraint
   When the user manually sets the load to "off"
   Then the stale heat override constraint is NOT re-added to the constraint list
   And the override state transitions to "NO OVERRIDE"

2. Given a climate load in force-off mode with an active heat override
   When the user manually overrides to "off" (matching the force-off base state)
   Then the system resets the override entirely (`external_user_initiated_state = None`)
   And no "off" override duration timer is started
   And the load returns to the force-off base behavior immediately

3. Given a climate load in force-on mode with an active off override
   When the user manually overrides to the exact configured `_state_on` (e.g., "auto")
   Then the system resets the override entirely (`external_user_initiated_state = None`)
   And the load returns to the force-on base behavior immediately
   (Note: switching to a DIFFERENT on-mode like "cool" when `_state_on` is "auto" is a new override, not "back to normal")

4. Given a climate load in force-off mode with NO prior override
   When the user manually sets the load to "heat"
   Then a `user_override` constraint is created for `override_duration`
   And `external_user_initiated_state` is set to "heat"
   And the constraint persists for the full override duration

5. Given a climate load in auto mode with an active override
   When the user manually changes the state back to the idle state
   Then the override is NOT cancelled by "back to normal" logic
   And the override persists for its full configured duration
   And the "back to normal" detection does not fire (mode guard)

## Tasks / Subtasks

### Task 1: Fix stale `override_constraint` reference [Bug Fix] (AC: #1)

- [ ] 1.1: In `ha_model/bistate_duration.py`, inside `check_load_activity_and_constraints()`, in the `if is_command_overridden_state_changed:` block, add `override_constraint = None` immediately after the `self.constraint_reset_and_reset_commands_if_needed(keep_commands=True)` call and before the idle/non-idle branch comment.

```python
# BEFORE:
                    self.constraint_reset_and_reset_commands_if_needed(
                        keep_commands=True
                    )  # remove any constraint if any we will add it back if needed below

                    # we will create a constraint if the asked state is not idle ...

# AFTER:
                    self.constraint_reset_and_reset_commands_if_needed(
                        keep_commands=True
                    )  # remove any constraint if any we will add it back if needed below
                    override_constraint = None  # clear stale ref after constraints wiped

                    # we will create a constraint if the asked state is not idle ...
```

- [ ] 1.2: This single line prevents both stale-reference effects:
  - (a) The `do_push_constraint_after` assignment that reads `override_constraint.end_of_constraint` no longer fires
  - (b) The `if override_constraint is not None:` check in the `_bistate_mode_off` handler no longer re-adds the old heat constraint via `set_live_constraints()`

### Task 2: Add "back to normal" override cancellation for force modes [Enhancement] (AC: #2, #3)

- [ ] 2.1: In the same `if is_command_overridden_state_changed:` block, BEFORE the existing line that sets `self.external_user_initiated_state = current_state`, add a new if/else structure:

```python
                    if is_command_overridden_state_changed:
                        # NEW: "back to normal" — user overrides back to base mode state
                        if (
                            self.external_user_initiated_state is not None
                            and bistate_mode in (self._bistate_mode_off, self._bistate_mode_on)
                            and (
                                (bistate_mode == self._bistate_mode_off
                                 and current_state == self.expected_state_from_command(CMD_IDLE))
                                or
                                (bistate_mode == self._bistate_mode_on
                                 and current_state == self._state_on)
                            )
                        ):
                            _LOGGER.info(
                                "check_load_activity_and_constraints: bistate "
                                "BACK TO NORMAL %s for load %s (mode %s), "
                                "cancelling override from %s",
                                current_state, self.name, bistate_mode,
                                self.external_user_initiated_state,
                            )
                            self.reset_override_state_and_set_reset_ask_time(time)
                            self.constraint_reset_and_reset_commands_if_needed(
                                keep_commands=True
                            )
                            override_constraint = None
                            do_force_next_solve = True
                        else:
                            # EXISTING: standard override creation logic
                            _LOGGER.info(...)  # existing log line
                            self.external_user_initiated_state = current_state
                            self.external_user_initiated_state_time = time
                            self.constraint_reset_and_reset_commands_if_needed(
                                keep_commands=True
                            )
                            override_constraint = None  # Task 1 fix
                            # ... rest of existing idle/non-idle branch ...
```

- [ ] 2.2: Key design constraints for the detection logic:
  - **Precondition**: `self.external_user_initiated_state is not None` — only fires when there is an existing override to cancel
  - **Mode guard**: `bistate_mode in (self._bistate_mode_off, self._bistate_mode_on)` — never fires in auto/calendar modes
  - **Force-off detection**: `current_state == self.expected_state_from_command(CMD_IDLE)` — user went back to off (= `_state_off`)
  - **Force-on detection**: `current_state == self._state_on` — user went back to the exact configured on-state (e.g., "auto"). Switching to a different on-mode (e.g., "cool" when `_state_on` is "auto") is treated as a new override, not "back to normal"
  - After reset, `do_force_next_solve = True` to trigger solver re-evaluation
  - The `reset_override_state_and_set_reset_ask_time()` method sets `asked_for_reset_user_initiated_state_time` which triggers a 60s cooldown window. This is acceptable for "back to normal" since the user just explicitly cancelled.

### Task 3: Add tests (AC: #1, #2, #3, #4, #5)

Test file: `tests/test_ha_bistate_duration.py` using existing `ConcreteBiStateDevice` class and `bistate_setup` fixture.

- [ ] 3.1: `test_heat_to_off_override_no_stale_constraint` (AC: #1)
  - Set up `ConcreteBiStateDevice` in `_bistate_mode_off`
  - Simulate user override to "on" (heat) — verify override constraint created
  - Simulate user override to "off" — verify old heat constraint is NOT in `_constraints`
  - Verify no stale constraint re-added via `set_live_constraints`

- [ ] 3.2: `test_force_off_heat_then_off_cancels_override` (AC: #2)
  - Set up device in `_bistate_mode_off`
  - Simulate user override to "on" (heat) — override active
  - Simulate user override to "off" (matching force-off base)
  - Assert `external_user_initiated_state is None`
  - Assert `_constraints` is empty

- [ ] 3.3: `test_force_on_off_then_on_cancels_override` (AC: #3)
  - Set up device in `_bistate_mode_on` (where `_state_on = "on"`)
  - Simulate user override to "off" — override active
  - Simulate user override to "on" (exact match to `_state_on`) — back to normal
  - Assert `external_user_initiated_state is None`
  - Also test: override to "off" then to a DIFFERENT non-off state (not `_state_on`) — should be treated as a new override, NOT cancelled

- [ ] 3.4: `test_force_off_heat_override_persists_for_duration` (AC: #4)
  - Set up device in `_bistate_mode_off`
  - Simulate user override to "on" (heat)
  - Assert override constraint exists with `end_of_constraint = time + override_duration`
  - Assert `external_user_initiated_state == "on"`
  - Advance time within override_duration — assert override still active

- [ ] 3.5: `test_auto_mode_override_not_cancelled_by_back_to_normal` (AC: #5)
  - Set up device in `bistate_mode_auto`
  - Simulate user override to "on" — override active
  - Simulate user changes back to "off"
  - Assert `external_user_initiated_state` is set (NOT None) — override persists
  - Confirms the mode guard prevents "back to normal" in auto mode

### Task 4: Quality gate (AC: all)

- [ ] 4.1: Run `python scripts/qs/quality_gate.py` from repo root
  - Expect: pytest passes with 100% coverage on changed files
  - Expect: ruff reports no lint errors
  - Expect: mypy passes
  - Expect: translations complete

## Dev Notes

### Root Cause (full trace from Cursor plan, confirmed by code review)

The bug is a stale local variable `override_constraint` in `check_load_activity_and_constraints()`:

1. **Line ~404-412**: The method loops over `self._constraints` and finds the existing heat override constraint, setting the local variable `override_constraint = ct`
2. **Line ~479-480**: `external_user_initiated_state` is set to `current_state` ("off") and `external_user_initiated_state_time` is set
3. **Line ~483-485**: `constraint_reset_and_reset_commands_if_needed(keep_commands=True)` clears ALL constraints from `self._constraints` to `[]`
4. **Line ~488-491**: Since the override is to the idle state, no new constraint is created. `override_constraint` **still holds the stale reference** to the old heat constraint
5. **Line ~518-524**: `do_push_constraint_after` is set based on `external_user_initiated_state_time + override_duration`
6. **Line ~526**: `override_constraint is not None` (stale!) — `do_push_constraint_after` is **overwritten** with `override_constraint.end_of_constraint + 1s` (the old heat constraint's end time)
7. **Line ~529**: `bistate_mode == self._bistate_mode_off` — enters the force-off handler
8. **Line ~531**: `do_push_constraint_after is not None` — enters the "keep ONLY the override" branch
9. **Line ~535-546**: Loops over `self._constraints` which is `[]` (cleared in step 3) — `found_override = False`
10. **Line ~554**: `override_constraint is not None` (stale!) — calls `set_live_constraints(time, [override_constraint])` which **RE-ADDS THE OLD HEAT CONSTRAINT**
11. The old heat constraint runs to its natural completion at ~18:55, and `ack_completed_constraint` calls `reset_override_state_and_set_reset_ask_time()` which clears the "off" override after only 7 minutes instead of 2 hours

### Architecture Constraints
- Two-layer boundary: fix is entirely in `ha_model/bistate_duration.py` (integration layer)
- No changes needed in `home_model/load.py`, `home_model/commands.py`, `home_model/constraints.py`, or `ha_model/climate_controller.py` (context only)
- Override state variables (`external_user_initiated_state`, etc.) are in `home_model/load.py` but accessed via inherited methods
- Constraint lifecycle: use `push_live_constraint()` for creation, `constraint_reset_and_reset_commands_if_needed()` for removal

### Important Implementation Notes
- **Logging**: existing f-string log calls in `bistate_duration.py` are a pre-existing violation. If touching any log line, convert to lazy `%s` format per project rules. Do NOT change log lines that are not touched by this PR
- **`reset_override_state_and_set_reset_ask_time()` cooldown**: this method sets `asked_for_reset_user_initiated_state_time` which triggers a 60s cooldown window (lines ~462-471) that suppresses new override detection. This is acceptable for "back to normal" — the user just explicitly cancelled, and a 60s window before re-override is reasonable
- **`_state_on` / `_state_off`**: these are concrete-class-specific. For `ConcreteBiStateDevice` in tests: `_state_on = "on"`, `_state_off = "off"`. For climate: `_state_off = HVACMode.OFF`, `_state_on = HVACMode.AUTO` (configurable). The detection uses `expected_state_from_command(CMD_IDLE)` which returns `_state_off`, making it work for all subclasses
- **Force-on code path differs from force-off**: Force-off's stale-constraint bug is at the `_bistate_mode_off` handler (~line 529). Force-on goes through the `else` branch (~line 569) which calls `_build_mode_constraint_items()`. Task 2 handles both paths uniformly by resetting the override BEFORE the constraint creation code runs
- **Force-on detection is strict**: Only `current_state == self._state_on` (exact match to the configured on-state) triggers "back to normal". If the user switches from an "off" override to a *different* on-mode (e.g., "cool" when `_state_on` is "auto"), that is a new override, not "back to normal". This avoids incorrectly cancelling overrides when the user deliberately picks a specific HVAC mode

### Project Structure Notes
- Main fix file: `custom_components/quiet_solar/ha_model/bistate_duration.py`
- Test file: `tests/test_ha_bistate_duration.py`
- Context (no changes): `custom_components/quiet_solar/home_model/load.py`, `custom_components/quiet_solar/ha_model/climate_controller.py`

### References
- Cursor plan: `/Users/tmenguy/.cursor/plans/fix_stale_override_constraint_4f355eb7.plan.md`
- Prior art: `_bmad-output/implementation-artifacts/bug-Github-#97-climate-card-target-hours-ignores-override.md`
- GitHub issue: https://github.com/tmenguy/quiet-solar/issues/134

## Adversarial Review Notes

**Reviewers:** Critic, Concrete Planner, Dev Agent Proxy, External Challenger
**Rounds:** 1

### Key findings incorporated:
- [Critic + Concrete Planner + External Challenger] Expanded root cause trace to full 11-step chain including `found_override = False` condition and both stale-reference effects (line ~526 and ~554) → Added complete trace in Dev Notes
- [Dev Proxy + Critic] Task 2 "back to normal" gated on force modes only (`bistate_mode in (_bistate_mode_off, _bistate_mode_on)`) → Added mode guard to detection logic
- [Concrete Planner + Dev Proxy] Task 2 precondition: `external_user_initiated_state is not None` required → Added precondition check
- [Concrete Planner] Exact code diffs for Task 1 → Added before/after code block
- [Dev Proxy + Concrete Planner] Structural descriptions instead of hard line numbers → Used `~line` prefix throughout
- [Concrete Planner] Test file corrected to `tests/test_ha_bistate_duration.py` with concrete function names → Updated Task 3
- [Dev Proxy] F-string logging note added → Added to implementation notes
- [Critic] 60s cooldown from `reset_override_state_and_set_reset_ask_time` documented → Added to implementation notes

### Decisions made:
- Keep Task 1 (bug fix) and Task 2 (enhancement) in same story, clearly labeled — Rationale: both address the same user-facing issue, independently testable, lower overhead than separate stories
- Approved if/else control flow structure for "back to normal" detection — Rationale: clean separation, back-to-normal check runs first, else falls through to existing logic
- Auto-mode regression test confirms mode guard — Rationale: ensures Task 2 doesn't accidentally break auto mode overrides

### Known risks acknowledged:
- [Critic] For force-on mode, the code path after override detection differs from force-off (goes through `_build_mode_constraint_items` at ~line 569 instead of `_bistate_mode_off` handler at ~line 529). Task 2 handles this by resetting BEFORE both paths diverge, but the asymmetry is worth monitoring during implementation
- [External Challenger] The Cursor plan's assumption that "off" override should persist for override_duration was explicitly overridden by user preference. If users later request the ability to have "off" overrides persist with a timer, this decision would need revisiting
