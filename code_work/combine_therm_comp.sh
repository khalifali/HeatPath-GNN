python3 - <<'PY'
import csv
import glob
import json
import math
import statistics
from pathlib import Path

columns = [
    "particles",
    "contact_edges",
    "mean_coordination",
    "thermally_solved_particles",
    "thermally_unanchored_particles",
    "largest_component_size",
    "spanning_component_count",
    "hot_wall_links",
    "cold_wall_links",
    "heat_rate_hot_W",
    "heat_rate_cold_W",
    "relative_energy_balance_error",
    "effective_conductivity_W_mK",
    "temperature_min_K",
    "temperature_max_K",
]

rows = []

for filename in sorted(glob.glob(
        "packing_seed_*/thermal_results_contact/thermal_summary.json")):

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    graph = data["graph_diagnostics"]
    case = Path(filename).parts[0]
    contacts = data["contact_edges"]
    particles = data["particles"]

    rows.append({
        "row_type": "CASE",
        "case": case,
        "particles": particles,
        "contact_edges": contacts,
        "mean_coordination": 2 * contacts / particles,
        "thermally_solved_particles":
            data["thermally_solved_particles"],
        "thermally_unanchored_particles":
            data["thermally_unanchored_particles"],
        "largest_component_size":
            graph["largest_component_size"],
        "spanning_component_count":
            graph["spanning_component_count"],
        "hot_wall_links": data["hot_wall_links"],
        "cold_wall_links": data["cold_wall_links"],
        "heat_rate_hot_W": data["heat_rate_hot_W"],
        "heat_rate_cold_W": data["heat_rate_cold_W"],
        "relative_energy_balance_error":
            data["relative_energy_balance_error"],
        "effective_conductivity_W_mK":
            data["effective_conductivity_W_mK"],
        "temperature_min_K": data["temperature_min_K"],
        "temperature_max_K": data["temperature_max_K"],
    })

if not rows:
    raise SystemExit(
        "No thermal_results_contact/thermal_summary.json files found."
    )

summary_operations = {
    "MEAN": statistics.mean,
    "STD": statistics.stdev,
    "MIN": min,
    "MAX": max,
}

summary_rows = []

for label, operation in summary_operations.items():
    row = {"row_type": label, "case": label}

    for column in columns:
        values = [float(case[column]) for case in rows]
        row[column] = (
            operation(values)
            if label != "STD" or len(values) > 1
            else 0.0
        )

    summary_rows.append(row)

cv_row = {"row_type": "CV_PERCENT", "case": "CV_PERCENT"}

for column in columns:
    values = [float(case[column]) for case in rows]
    mean = statistics.mean(values)

    cv_row[column] = (
        100 * statistics.stdev(values) / abs(mean)
        if len(values) > 1 and mean != 0
        else ""
    )

summary_rows.append(cv_row)

output = "packing_thermal_comparison.csv"
fieldnames = ["row_type", "case"] + columns

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows + summary_rows)

print(f"Wrote {output}: {len(rows)} packings + 5 summary rows")
PY
