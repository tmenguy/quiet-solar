# Story bug-Github-#131: Fix dampened solar provider scores unavailable after HA reboot

issue: 131
branch: "QS_131"

Status: dev-complete

## Story
As a Quiet Solar user with dampening enabled,
I want my dampened solar forecast scores to survive HA reboots,
so that my dashboard shows accurate values immediately after restart instead of "unavailable".

## Acceptance Criteria
1. Given a provider with dampening active and a valid `score_dampened` value persisted before reboot, When HA restarts and the first scoring cycle runs but `compute_dampened_score()` fails due to missing historical data, Then the previously restored `score_dampened` value is preserved (not cleared to None).
2. Given a provider with dampening active and a valid `score_dampened` restored from state, When `_run_scoring_cycle()` is called and `compute_dampened_score()` returns False, Then the existing `score_dampened` value remains unchanged and the sensor remains available.
3. Given a provider where dampening has never been computed (score_dampened is None), When `compute_dampened_score()` fails, Then `score_dampened` remains None (no regression -- no value to preserve).
4. Given a provider where `compute_dampened_score()` succeeds, When the scoring cycle runs normally, Then `score_dampened` is updated to the newly computed value (no change in happy path behavior).

## Tasks / Subtasks
- [x] Task 1: Remove aggressive score clearing in `_run_scoring_cycle()` (AC: #1, #2, #3)
  - [x] 1.1: In `ha_model/solar.py`, method `QSSolar._run_scoring_cycle()`, remove the `provider.score_dampened = None` assignment inside the `if not provider.compute_dampened_score(time):` block
  - [x] 1.2: Change the log from `_LOGGER.warning` to `_LOGGER.info` with message: `"Dampened score refresh failed for provider %s, keeping existing score %s", name, provider.score_dampened` (lazy `%s`, no period)
  - [x] 1.3: In the existing `_LOGGER.info` that follows the scoring block, add `provider.score_dampened` to the log output for visibility
- [x] Task 2: Update and add tests in `TestScoringCycleDampenedRefresh` (AC: #1, #2, #3, #4)
  - [x] 2.0: Modify existing `test_scoring_cycle_clears_stale_dampened_score` -- rename to `test_scoring_cycle_preserves_existing_dampened_score_on_failure`, change assertion from `assert provider.score_dampened is None` to `assert provider.score_dampened == 100.0` (preserved)
  - [x] 2.1: Add `test_scoring_cycle_preserves_none_dampened_score_on_failure` -- set `provider._dampening_coefficients = {0: (0.8, 0.0)}` (so `has_dampening` is True) but leave `score_dampened` as None, mock `compute_dampened_score` to return False, assert `score_dampened is None`
  - [x] 2.2: Enhance existing `test_scoring_cycle_refreshes_dampened_score` to verify actual value update (not just that the mock was called)
- [x] Task 3: Run quality gates (AC: all)
  - [x] 3.1: Run `python scripts/qs/quality_gate.py` -- ensure 100% coverage, ruff, mypy, translations pass

## Dev Notes
- **Root cause**: `QSSolar._run_scoring_cycle()` in `ha_model/solar.py` clears `provider.score_dampened = None` when `compute_dampened_score()` fails. After reboot, historical data isn't available yet, so the computation always fails on the first cycle, destroying the value that `QSBaseSensorSolarDampenedScoreRestore.async_added_to_hass()` (sensor.py) just restored.
- **Asymmetry with raw score**: `compute_score()` returning False does NOT clear `provider.score` -- the raw score restore survives. Only the dampened score is aggressively cleared. `reset_dampening()` is a separate intentional clearing path (user action) -- unrelated to this bug.
- **Fix approach**: Remove the `provider.score_dampened = None` assignment. Demote log to `info` since this is routine after reboot. This makes dampened score behavior match raw score behavior on failed computation.
- **Files changed**: `ha_model/solar.py` (code fix) + `tests/test_solar_dampening.py` (update existing test + add new test)
- **Layer boundary**: Fix is entirely within `ha_model/solar.py` -- no cross-layer changes needed.
- **Test class**: All test changes go in `TestScoringCycleDampenedRefresh` in `tests/test_solar_dampening.py`. Helpers `_make_solar` and `FakeConfigEntry` are already available there.
- **Logging rules**: lazy `%s`, no f-strings, no trailing periods.

### Project Structure Notes
- `ha_model/solar.py` -- `QSSolar._run_scoring_cycle()` method, target: `provider.score_dampened = None` inside the `if not provider.compute_dampened_score(time):` block
- `ha_model/solar.py` -- `QSSolarProvider.compute_dampened_score()` method
- `sensor.py` -- `QSBaseSensorSolarDampenedScoreRestore.async_added_to_hass()` (restore works correctly, no changes needed)
- `tests/test_solar_dampening.py` -- `TestScoringCycleDampenedRefresh` class

### References
- Issue #131: https://github.com/tmenguy/quiet-solar/issues/131
- Similar fix pattern: bug #84 (prober persistence), bug #113 (solar forecast score hydration)

## Adversarial Review Notes

**Reviewers:** Critic, Concrete Planner, Dev Proxy
**Rounds:** 1

### Key findings incorporated:
- [All 3] Existing test `test_scoring_cycle_clears_stale_dampened_score` must be updated (not just new tests added) -> Added Task 2.0 with explicit rename and assertion change
- [Critic + Concrete + Dev Proxy] Log should be demoted to `info` (routine post-reboot), include preserved score value -> Updated Task 1.2
- [Concrete] Task 2.1 clarified as modification of existing test, Task 2.2 as enhancement -> Restructured Task 2
- [Dev Proxy] Note both files that change -> Added to Dev Notes
- [Concrete] Add dampened score to scoring info log -> Added Task 1.3

### Decisions made:
- Stale score concern dismissed -- Rationale: raw score has same staleness after reboot; fix makes behavior symmetric, not worse
- Timing/ordering concern dismissed -- Rationale: analysis confirmed async_added_to_hass completes before first update_all_states

### Known risks acknowledged:
- After reboot, "dampened score refresh failed" will log at info level on every scoring cycle until historical data accumulates (~24h). This is expected and benign.
