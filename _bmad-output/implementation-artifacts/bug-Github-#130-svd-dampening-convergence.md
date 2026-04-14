# Story bug-Github-#130-svd-dampening-convergence: Fix SVD convergence error in 7-day dampening computation

issue: 130
branch: "QS_130"

Status: ready-for-dev

## Story
As a Quiet Solar user,
I want the solar dampening computation to gracefully handle degenerate or identical forecast data,
so that pressing the "Compute Dampening (7 Day)" button never crashes with an SVD error.

## Acceptance Criteria

1. Given all forecast values in a slot are identical or near-identical (`np.ptp(forecasts) < threshold`), When `compute_dampening` runs in 7-day mode, Then it computes `a_k = mean(actuals) / mean(forecasts)` with `b_k = 0`, clamped to [0.1, 3.0].

2. Given `np.polyfit` raises `LinAlgError` or `ValueError` for any reason, When `compute_dampening` processes a slot, Then the exception is caught, a warning is logged with slot number and point count, and identity coefficients (1.0, 0.0) are used for that slot.

3. Given `np.polyfit` returns non-finite coefficients (NaN/Inf in `a_k` or `b_k`), When `compute_dampening` processes a slot, Then identity coefficients (1.0, 0.0) are used and a warning is logged.

4. Given `compute_dampening` receives data with NaN/Inf forecast or actual values for a slot, When building per-slot point lists, Then NaN/Inf pairs are filtered out before any computation (defense-in-depth for future storage changes).

5. Given a slot has fewer than 3 valid data points after filtering, When `compute_dampening` runs in 7-day mode, Then identity coefficients (1.0, 0.0) are used for that slot.

6. Given some slots fail and others succeed, When `compute_dampening` returns, Then it returns `True` with computed coefficients for successful slots and identity for failed slots.

7. Given `compute_dampening` raises an unexpected exception for a provider, When `compute_dampening_all_providers` iterates providers, Then the exception is caught and logged, and remaining providers are still processed.

## Tasks / Subtasks

- [ ] Task 1: Add near-identical forecast handling in `compute_dampening` (AC: #1, #5)
  - [ ] 1.1: In `ha_model/solar.py`, in `compute_dampening()` inside the 7-day branch (after the `len(points) < 3` check), add a near-identical forecast check using `np.ptp(forecasts) < threshold`. When triggered, compute `a_k = np.mean(actuals_vals) / np.mean(forecasts)` with `b_k = 0.0`, clamp `a_k` to [0.1, 3.0]. Guard against `mean(forecasts) < 10` (use identity).
  - [ ] 1.2: Add tests: identical forecasts (all 1000.0), near-identical forecasts (1000.0 ± 0.001), and verify `a_k` is the mean ratio with `b_k = 0`.

- [ ] Task 2: Wrap `np.polyfit` in try/except (AC: #2, #3, #6)
  - [ ] 2.1: In `ha_model/solar.py`, in `compute_dampening()`, wrap the `np.polyfit()` call in `try/except (np.linalg.LinAlgError, ValueError)`. On exception, log warning with `%s` formatting including slot number and point count, fall back to (1.0, 0.0).
  - [ ] 2.2: After polyfit returns, validate output with `np.isfinite(a_k) and np.isfinite(b_k)`. If not finite, log warning and fall back to identity.
  - [ ] 2.3: Add tests: mock `np.polyfit` to raise `LinAlgError`, verify identity fallback and warning logged. Test near-degenerate data producing non-finite output.

- [ ] Task 3: Add NaN/Inf defense-in-depth at slot assembly (AC: #4, #5)
  - [ ] 3.1: In `ha_model/solar.py`, in `compute_dampening()`, in the loop where `(forecast_val, actual_val)` pairs are appended to `slots[slot]`, add `if np.isfinite(forecast_val) and np.isfinite(actual_val)` guard. Document in comment: defense-in-depth for future storage changes (current ring buffer is int32, cannot contain NaN).
  - [ ] 3.2: Add tests: inject NaN/Inf into forecast and actual data via test helpers, verify they are excluded from slot points.

- [ ] Task 4: Add exception safety net in `compute_dampening_all_providers` (AC: #7)
  - [ ] 4.1: In `ha_model/solar.py`, in `compute_dampening_all_providers()`, wrap the `provider.compute_dampening(time, num_days)` call in try/except to catch any unexpected exception. Log error with provider name, continue to next provider.
  - [ ] 4.2: Add test: mock `compute_dampening` to raise an unexpected exception, verify other providers still get processed.

- [ ] Task 5: Verify 1-day mode resilience (AC: #4)
  - [ ] 5.1: Verify that the 1-day ratio correction path (the `num_days == 1` branch) is protected by Task 3's NaN/Inf filter at slot assembly. The `f_val < 10` guard at line 700 already prevents division by near-zero. Add test for 1-day mode with NaN data injected via test helper.

- [ ] Task 6: Run quality gates
  - [ ] 6.1: Run `python scripts/qs/quality_gate.py` — pytest 100% coverage + ruff + mypy + translations all pass.

## Dev Notes

### Root Cause
The primary failure mode is **identical int32-quantized forecast values** across 7 days for the same slot, creating a singular Vandermonde matrix in `np.polyfit`. The ring buffer stores `np.int32` values which cannot contain NaN/Inf. When all forecasts for a slot are the same (e.g., all 1000), polyfit's SVD fails to converge because the system is underdetermined.

### Architecture Constraints
- `compute_dampening()` and `compute_dampening_all_providers()` live in `ha_model/solar.py` — both in the HA model layer
- Ring buffer accessors in `ha_model/home.py` return int32 values — NaN/Inf impossible at this layer
- Two-layer boundary: all changes are in `ha_model/`, no boundary crossing
- Per project rule: "Domain-level numerical errors must be caught at ha_model layer, never bubble into HA event loop"
- Per project rule: "Button press handlers must catch exceptions and log appropriately"

### Error Handling Pattern
- Catch `np.linalg.LinAlgError` and `ValueError` specifically around polyfit
- Use lazy `%s` logging, no f-strings, no trailing periods
- Identity coefficients (1.0, 0.0) are the safe fallback — equivalent to "no dampening adjustment"
- For the near-identical case, compute meaningful `a_k` rather than falling back to identity

### Near-Identical Forecast Logic
When `np.ptp(forecasts) < threshold` (e.g., 1.0 W):
- All forecasts are effectively the same value
- Linear regression is meaningless (slope undefined)
- Instead compute `a_k = mean(actuals) / mean(forecasts)`, `b_k = 0.0`
- This is the multi-day ratio correction: "on average, actual production is a_k times the forecast"
- Clamp `a_k` to [0.1, 3.0] as with polyfit results
- If `mean(forecasts) < 10`, use identity (nighttime or near-zero production)

### Test Patterns
- Tests in `tests/test_solar_dampening.py` — follow existing patterns with `_make_provider_with_histories` helper
- Test helper injects Python floats (not int32), so NaN/Inf tests ARE possible even though production data is int32
- Document this discrepancy in test comments
- Key test scenarios:
  - Identical forecasts → ratio-based `a_k`, `b_k = 0`
  - Near-identical forecasts (within threshold) → same behavior
  - Polyfit `LinAlgError` (mocked) → identity fallback
  - Non-finite polyfit output → identity fallback
  - NaN/Inf in input data → filtered at slot assembly
  - Mixed success/failure across slots → partial coefficients
  - Provider exception in `compute_dampening_all_providers` → other providers continue
  - 1-day mode with NaN data → protected by slot filter

### Key Files
- `custom_components/quiet_solar/ha_model/solar.py` — `compute_dampening()`, `compute_dampening_all_providers()`
- `tests/test_solar_dampening.py` — existing dampening tests

### References
- Issue #130: SVD did not converge in Linear Least Squares
- Feature #124 (PR #125): introduced dampening buttons
- Story 3.14: scoring infrastructure (in-progress)
- numpy.polyfit docs: raises LinAlgError when SVD fails to converge

## Adversarial Review Notes

**Reviewers:** Critic, Concrete Planner, Dev Proxy
**Rounds:** 1

### Key findings incorporated:
- Ring buffer uses int32 — upstream NaN/Inf filter (original Task 1) was dead code → removed
- Button handler `async_press` has no try/except for dampening call → added safety net in `compute_dampening_all_providers` (Task 4)
- Pre-check `len(set(forecasts)) == 1` is fragile for floats → replaced with `np.ptp(forecasts) < threshold` and user-requested ratio computation for near-identical data (Task 1)
- Polyfit output should be validated for finiteness → added `np.isfinite` check (Task 2.2)
- Line numbers are fragile → switched to function-signature-based task descriptions
- Test helpers inject floats not int32 → documented discrepancy

### Decisions made:
- Drop upstream NaN/Inf filter in `get_historical_data` — Rationale: int32 ring buffer cannot contain NaN/Inf, filter would be dead code
- For identical/near-identical forecasts: compute `a_k = mean(actuals)/mean(forecasts)` with `b_k = 0` instead of identity fallback — Rationale: user wants meaningful dampening even with constant forecast data
- Always return True from `compute_dampening` — Rationale: identity coefficients are safe, simpler logic, consistent with current behavior
- Keep NaN/Inf slot filter as defense-in-depth — Rationale: protects against future storage changes

### Known risks acknowledged:
- Test data path (floats) differs from production data path (int32) — NaN/Inf tests exercise code that can't fire in production
- Near-identical threshold choice (e.g., 1.0 W) is somewhat arbitrary — may need tuning based on real-world data
