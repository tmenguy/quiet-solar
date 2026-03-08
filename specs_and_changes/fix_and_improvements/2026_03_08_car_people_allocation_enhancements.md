# Car/person allocation: two-pass energy-first approach + unplugged car pre-allocation

**Date**: 2026-03-08

## Problem 1: preferred car penalty dominates energy minimization

The Hungarian algorithm cost matrix for person-to-car allocation used a
`preferred_car_penalty` of `(num_persons * E_max) + 1`. This value was
intentionally larger than the sum of all possible energy costs, so the
algorithm **always** respected preferred-car assignments regardless of how
much extra charging energy that required.

Example with 2 persons, 2 cars, E_max = 20 kWh:

- Preferred assignment: A->Car1 (20 kWh) + B->Car2 (20 kWh) = **40 kWh**
- Energy-optimal: A->Car2 (1 kWh) + B->Car1 (1 kWh) = **2 kWh**

The old code would always pick 40 kWh because the preferred-car penalty
dominated the cost function.

## Problem 2: unplugged-at-home cars treated the same as plugged cars

A car that is at home but not connected to any charger (`car.charger is None`,
`car.is_car_home(time) == True`) entered the Hungarian allocation with a
`diff_energy` cost as if it could be charged. In reality, no energy can be
delivered to that car, so assigning a person who needs charging to an
unplugged car wastes the allocation -- the person's trip won't be covered.

The right approach: if an unplugged car already covers a person's trip with
its current charge, assign them early (it's a "free" allocation). This frees
up connected/chargeable cars for persons who actually need energy delivered.

## Solution

### Part A: two-pass Hungarian with global energy threshold

Replace the single Hungarian pass with a **two-pass approach** that compares
total real charging energy between an energy-optimal assignment and a
preferred-car-biased assignment, then picks based on a configurable threshold.

#### Algorithm

1. Build a raw energy matrix (sentinels for unauthorized / covered / error
   states, positive `diff_energy` for real charging needs).
2. **Pass 1 -- energy-optimal**: Finalize the cost matrix with no preferred-car
   penalty. Run Hungarian. Compute `total_energy_optimal`.
3. **Pass 2 -- preferred-biased**: Finalize the cost matrix with the large
   preferred-car penalty (same formula as before). Run Hungarian. Compute
   `total_energy_preferred`.
4. **Decision**: If `total_energy_preferred - total_energy_optimal` is within
   `PREFERRED_CAR_ENERGY_THRESHOLD_KWH` (default 1.0 kWh), use the
   preferred-biased assignment. Otherwise, use the energy-optimal one.

#### Why two-pass instead of a small epsilon penalty

A single-pass epsilon approach operates at the per-cell level: even a 0.1 kWh
difference on one pair overrides preferred car for that pair. The two-pass
approach operates at the global level: it tolerates small per-pair differences
that accumulate as long as the total stays within the threshold. This matches
the real-world intent better.

### Part B: pre-allocation of unplugged-at-home cars

Before the Hungarian algorithm runs, a pre-allocation step handles cars that
are at home but not connected to a charger:

1. Identify unplugged-at-home cars: `car.is_car_home(time) is True` and
   `car.charger is None`.
2. For each such car, find all authorized persons (not yet covered) whose
   trip is already covered by the car's current charge (`is_covered is True`).
3. **Preferred car takes priority**: if a person whose `preferred_car` is this
   car has a covered trip, assign them immediately.
4. **Fallback**: if no preferred-car match, pick the authorized person with
   the **smallest energy margin** (closest fit between the car's current
   charge and the trip need). This maximizes the chance that other cars with
   more spare charge remain available for other persons.
5. Assign that person to that car. Add both to `covered_cars` / `covered_persons`.
6. If no person's trip is covered by the unplugged car, leave it in the pool
   for normal Hungarian allocation.

This ensures unplugged cars that can already satisfy a trip are used for that
purpose, freeing connected cars for persons who actually need charging. The
preferred-car priority at this stage is safe because there is no energy cost
difference (the car can't be charged anyway), so honoring the user's preference
is always the right choice.

Available APIs used:
- `car.is_car_home(time)` -- whether the car is at the home location
- `car.charger` -- `None` when no charger is attached
- `car.get_adapt_target_percent_soc_to_reach_range_km(mileage, time)` --
  returns `(is_covered, current_soc, needed_soc, diff_energy)`

## Changed files

### `custom_components/quiet_solar/const.py`

- Added constant `PREFERRED_CAR_ENERGY_THRESHOLD_KWH = 1.0`.

### `custom_components/quiet_solar/ha_model/home.py`

Part A (two-pass Hungarian):
- Added `_build_raw_energy_matrix()` static method: builds the raw energy
  matrix with sentinel values, returns `(costs, E_max)`.
- Added `_finalize_cost_matrix()` static method: converts sentinels into real
  costs, applies `maxi_val` for unauthorized pairs and optional preferred-car
  penalty.
- Added `_compute_assignment_energy()` static method: sums positive
  `diff_energy` values for an assignment (ignores sentinels).
- Refactored `get_best_persons_cars_allocations()` to call these helpers twice
  and select between the two assignments.

Part B (unplugged car pre-allocation):
- Added `_pre_allocate_unplugged_home_cars()` method in `get_best_persons_cars_allocations()`.
  Called after user-selected assignments and before building `p_s`/`c_s` lists.
  Identifies cars that are at home but not connected to a charger, finds
  authorized persons whose trips are already covered. Preferred-car matches
  take priority; otherwise the closest-fit person (smallest energy margin) is
  picked. Both are added to `covered_cars`/`covered_persons` so they are
  excluded from the Hungarian step.

### `tests/test_hungarian_algorithm.py`

Added `TestTwoPassAllocation` class with 11 test cases:

1. Energy wins when difference is large (40 vs 2 kWh)
2. Preferred wins when difference is small (10.5 vs 10.0 kWh)
3. Exact threshold boundary (preferred wins at <=)
4. All cars already covered (E_max = 0, preferred wins)
5. Single person, single car (trivial)
6. Unauthorized pairs never selected
7. 3x3 mixed scenario
8. Rectangular: more cars than persons
9. Rectangular: more persons than cars
10. Preferred matches energy-optimal (both agree)
11. Large energy gap on one pair (50 kWh vs 0.5 kWh)

Added `TestPreAllocateUnpluggedHomeCars` class with 6 test cases:

1. Preferred person gets priority over other covered persons
2. Fallback to closest margin when no person prefers this car
3. Plugged car is skipped (not pre-allocated)
4. Car not at home is skipped
5. Trip not covered by car's current charge is skipped
6. Preferred person wins even when another has a smaller margin

## What did not change

- The Hungarian algorithm implementation in `home_utils.py`.
- The `maxi_val` for unauthorized pairs.
- The sentinel value handling (-1.0, -2.0, -3.0).
- The post-allocation fallback logic (user selection, invited cars, etc.).
- The `get_adapt_target_percent_soc_to_reach_range_km` computation in `car.py`.

## Tuning

The threshold `PREFERRED_CAR_ENERGY_THRESHOLD_KWH` defaults to 1.0 kWh.
Increase it to favor preferred cars more aggressively; decrease it (or set to
0.0) to strictly minimize energy. This could be exposed as a user config
option in the future.
