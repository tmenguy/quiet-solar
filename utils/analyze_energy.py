import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EnergySeries:
    dates: List[str]
    grid_consumption: np.ndarray
    grid_return: np.ndarray
    battery_in: np.ndarray
    battery_out: np.ndarray
    solar_production: np.ndarray


def _to_float_list(values: List[str], target_len: int) -> List[float]:
    nums: List[float] = []
    for v in values[:target_len]:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            nums.append(0.0)
    if len(nums) < target_len:
        nums.extend([0.0] * (target_len - len(nums)))
    return nums


def load_energy_csv(csv_path: str) -> EnergySeries:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 4:
            raise ValueError("Unexpected CSV format: need at least 4 columns (entity_id,type,unit,dates...)")
        date_cols = header[3:]

        wanted = {
            "grid_consumption": None,
            "grid_return": None,
            "battery_in": None,
            "battery_out": None,
            "solar_production": None,
        }  # type: Dict[str, np.ndarray|None]

        for row in reader:
            if not row or len(row) < 3:
                continue
            t = row[1].strip()
            if t in wanted:
                values = _to_float_list(row[3:], target_len=len(date_cols))
                wanted[t] = np.asarray(values, dtype=float)

    for key, arr in wanted.items():
        if arr is None:
            raise KeyError(f"Missing '{key}' in CSV 'type' column.")

    return EnergySeries(
        dates=date_cols,
        grid_consumption=wanted["grid_consumption"],
        grid_return=wanted["grid_return"],
        battery_in=wanted["battery_in"],
        battery_out=wanted["battery_out"],
        solar_production=wanted["solar_production"],
    )


def compute_load_components(series: EnergySeries) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    solar_direct = series.solar_production - series.battery_in - series.grid_return
    solar_direct = np.maximum(solar_direct, 0.0)

    total_load = series.grid_consumption + series.battery_out + solar_direct

    daily_surplus = series.solar_production - solar_direct
    daily_deficit = total_load - solar_direct

    return total_load, solar_direct, daily_surplus, daily_deficit


def simulate_import(daily_surplus: np.ndarray, daily_deficit: np.ndarray, capacity_kwh: float) -> float:
    soc = 0.0
    total_import = 0.0
    for surplus, deficit in zip(daily_surplus, daily_deficit):
        soc = min(capacity_kwh, soc + max(0.0, surplus))
        discharge = min(soc, max(0.0, deficit))
        soc -= discharge
        total_import += max(0.0, deficit - discharge)
    return total_import


def find_optimal_capacity(daily_surplus: np.ndarray, daily_deficit: np.ndarray, tol_kwh: float = 0.01) -> Tuple[float, float]:


    import_inf = simulate_import(daily_surplus, daily_deficit, capacity_kwh=1e9)

    # by construction import_inf is the minimum one

    lo = 0.0
    hi = float(np.sum(daily_surplus)) + float(np.max(daily_deficit)) + 1.0

    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        imp = simulate_import(daily_surplus, daily_deficit, capacity_kwh=mid)
        if imp <= import_inf + tol_kwh:
            hi = mid
        else:
            lo = mid

    cap_opt = hi
    return cap_opt, import_inf


def scale_solar(series: EnergySeries, total_load: np.ndarray, solar_direct: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    solar_scaled = series.solar_production * scale
    direct_scaled = np.minimum(total_load, solar_direct * scale)
    surplus_scaled = solar_scaled - direct_scaled
    deficit_scaled = total_load - direct_scaled
    return surplus_scaled, deficit_scaled


def main():
    parser = argparse.ArgumentParser(description="Analyze energy CSV and simulate battery and solar scenarios.")
    parser.add_argument(
        "--csv",
        default="energy_09_2024-09_2025.csv",
        help="Path to the yearly energy CSV",
    )
    parser.add_argument(
        "--current_battery_kwh",
        type=float,
        default=20.0,
        help="Current battery usable capacity in kWh (default: 20)",
    )
    parser.add_argument(
        "--out",
        default="analysis_results.csv",
        help="Output CSV path for scenario results",
    )
    args = parser.parse_args()

    series = load_energy_csv(args.csv)
    total_load, solar_direct, daily_surplus, daily_deficit = compute_load_components(series)

    current_import = simulate_import(daily_surplus, daily_deficit, capacity_kwh=args.current_battery_kwh)
    cap_opt, import_inf = find_optimal_capacity(daily_surplus, daily_deficit)
    import_at_opt = simulate_import(daily_surplus, daily_deficit, capacity_kwh=cap_opt)

    print(f"Days: {len(series.dates)}")
    print(f"Current battery capacity: {args.current_battery_kwh:.2f} kWh")
    print(f"Grid import with current battery: {current_import:.2f} kWh")
    print(f"Theoretical minimum grid import (very large battery): {import_inf:.2f} kWh")
    print(f"Optimal capacity to reach theoretical minimum (within 0.01 kWh): {cap_opt:.2f} kWh")
    print(f"Grid import with optimal capacity: {import_at_opt:.2f} kWh")
    print(f"Gain vs current battery: {current_import - import_at_opt:.2f} kWh")

    rows = []
    baseline_import_current = current_import
    for pct in range(0, 110, 10):
        scale = 1.0 + (pct / 100.0)
        if pct == 0:
            surplus_s, deficit_s = daily_surplus, daily_deficit
        else:
            surplus_s, deficit_s = scale_solar(series, total_load, solar_direct, scale)

        imp_current = simulate_import(surplus_s, deficit_s, capacity_kwh=args.current_battery_kwh)
        imp_double_current = simulate_import(surplus_s, deficit_s, capacity_kwh=2*args.current_battery_kwh)
        cap_opt_s, inf_s = find_optimal_capacity(surplus_s, deficit_s)
        imp_opt_s = simulate_import(surplus_s, deficit_s, capacity_kwh=cap_opt_s)

        gain_vs_baseline_current = baseline_import_current - imp_current
        gain_vs_baseline_double_current = baseline_import_current - imp_double_current
        opt_gain_over_current = imp_current - imp_opt_s
        total_gain_vs_baseline = baseline_import_current - imp_opt_s

        rows.append({
            "solar_scale_percent": pct,
            "grid_import_kwh_at_20kwh": round(imp_current, 3),
            "gain_kwh_vs_baseline_20kwh": round(gain_vs_baseline_current, 3),
            "grid_import_kwh_at_40kwh": round(imp_double_current, 3),
            "gain_kwh_vs_baseline_40kwh": round(gain_vs_baseline_double_current, 3),
            "optimal_capacity_kwh": round(cap_opt_s, 3),
            "grid_import_kwh_at_opt": round(imp_opt_s, 3),
            "optimization_gain_kwh": round(opt_gain_over_current, 3),
            "total_gain_kwh_vs_baseline": round(total_gain_vs_baseline, 3),
        })

    fieldnames = [
        "solar_scale_percent",
        f"grid_import_kwh_at_{int(args.current_battery_kwh)}kwh",
        f"gain_kwh_vs_baseline_{int(args.current_battery_kwh)}kwh",
        f"grid_import_kwh_at_{int(2*args.current_battery_kwh)}kwh",
        f"gain_kwh_vs_baseline_{int(2*args.current_battery_kwh)}kwh",
        "optimal_capacity_kwh",
        "grid_import_kwh_at_opt",
        "optimization_gain_kwh",
        "total_gain_kwh_vs_baseline",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nSolar scale study (saved to CSV):")
    header_line = " | ".join(fieldnames)
    print(header_line)
    for r in rows:
        print(" | ".join(str(r[k]) for k in fieldnames))


if __name__ == "__main__":
    main()



