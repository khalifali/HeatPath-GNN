#!/usr/bin/env bash
set -euo pipefail

# Run the final controlled allocation protocol on the ten accepted packings,
# independently re-solve each best assignment, verify exact agreement, and
# create one project-level comparison CSV.

project_root=$(pwd -P)
python_cmd=python3
allocation_script="$project_root/run_allocation_pilot_stratified_v2.py"
thermal_script="$project_root/solve_packing_heat_transfer.py"
allocation_output_name=allocation_stratified_p3000
thermal_output_name=thermal_results_stratified_optimized_p3000
force=0
dry_run=0
start_seed=""

seeds=(18427 29173 37811 46549 55291 64109 72869 81647 90439 99223)

usage() {
    echo "Usage: bash $0 [--force] [--dry-run] [--start-seed SEED]"
    echo "  --force            rerun and overwrite existing output files"
    echo "  --dry-run          print commands without executing them"
    echo "  --start-seed SEED  begin at this seed in the accepted seed list"
}

while (( $# > 0 )); do
    case "$1" in
        --force)
            force=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --start-seed)
            [[ $# -ge 2 ]] || { echo "Missing value after --start-seed" >&2; exit 2; }
            start_seed=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "$allocation_script" ]] || { echo "Missing: $allocation_script" >&2; exit 1; }
[[ -f "$thermal_script" ]] || { echo "Missing: $thermal_script" >&2; exit 1; }
command -v "$python_cmd" >/dev/null || { echo "python3 not found" >&2; exit 1; }

if [[ -n "$start_seed" ]]; then
    found=0
    for seed in "${seeds[@]}"; do
        [[ "$seed" == "$start_seed" ]] && found=1
    done
    (( found == 1 )) || { echo "Unknown start seed: $start_seed" >&2; exit 2; }
fi

run_command() {
    if (( dry_run == 1 )); then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

started=0
[[ -z "$start_seed" ]] && started=1

for seed in "${seeds[@]}"; do
    if (( started == 0 )); then
        [[ "$seed" == "$start_seed" ]] && started=1 || continue
    fi

    case_dir="$project_root/packing_seed_$seed"
    particle_file="$case_dir/particles_final.dump"
    contact_file="$case_dir/contacts_final.dump"
    allocation_dir="$case_dir/$allocation_output_name"
    allocation_summary="$allocation_dir/allocation_summary.json"
    optimized_ids="$allocation_dir/high_ids_optimized.txt"
    thermal_dir="$case_dir/$thermal_output_name"
    thermal_summary="$thermal_dir/thermal_summary.json"

    [[ -d "$case_dir" ]] || { echo "Missing case directory: $case_dir" >&2; exit 1; }
    [[ -s "$particle_file" ]] || { echo "Missing particle dump: $particle_file" >&2; exit 1; }
    [[ -s "$contact_file" ]] || { echo "Missing contact dump: $contact_file" >&2; exit 1; }

    echo "===== packing_seed_$seed ====="

    if [[ -s "$allocation_summary" && -s "$optimized_ids" && $force -eq 0 ]]; then
        echo "Allocation already complete; skipping: $allocation_dir"
    else
        if (( dry_run == 0 )); then
            mkdir -p "$allocation_dir"
        fi
        run_command "$python_cmd" "$allocation_script" "$case_dir" \
            --output "$allocation_dir" \
            --k-low 1.0 \
            --k-high 10.0 \
            --high-count 50 \
            --type-quotas 1:10,2:30,3:10 \
            --random-samples 100 \
            --swap-proposals 3000 \
            --optimizer-restarts 5 \
            --seed 12345
    fi

    if (( dry_run == 1 )); then
        run_command "$python_cmd" "$thermal_script" "$case_dir" \
            --output "$thermal_output_name" \
            --high-ids "$optimized_ids" \
            --k-low 1.0 --k-high 10.0 --k-fluid 0.0 \
            --t-hot 301 --t-cold 300
        continue
    fi

    [[ -s "$allocation_summary" ]] || { echo "Missing allocation summary: $allocation_summary" >&2; exit 1; }
    [[ -s "$optimized_ids" ]] || { echo "Missing optimized IDs: $optimized_ids" >&2; exit 1; }

    if [[ -s "$thermal_summary" && $force -eq 0 ]]; then
        echo "Independent thermal solve already complete; skipping: $thermal_dir"
    else
        run_command "$python_cmd" "$thermal_script" "$case_dir" \
            --output "$thermal_output_name" \
            --high-ids "$optimized_ids" \
            --k-low 1.0 --k-high 10.0 --k-fluid 0.0 \
            --t-hot 301 --t-cold 300
    fi

    "$python_cmd" - "$allocation_summary" "$thermal_summary" <<'PY'
import json
import math
import sys

allocation = json.load(open(sys.argv[1], encoding="utf-8"))
thermal = json.load(open(sys.argv[2], encoding="utf-8"))
expected = float(allocation["optimized_best_k_eff_W_mK"])
observed = float(thermal["effective_conductivity_W_mK"])
if thermal["high_conductivity_particles"] != 50:
    raise SystemExit("Independent solve did not contain exactly 50 high-k particles")
if not math.isclose(expected, observed, rel_tol=1.0e-11, abs_tol=1.0e-14):
    raise SystemExit(f"Independent k_eff mismatch: allocation={expected}, thermal={observed}")
if thermal["relative_energy_balance_error"] >= 1.0e-8:
    raise SystemExit("Independent thermal solve failed the energy-balance criterion")
print(f"Independent verification PASS: k_eff={observed:.12g}")
PY
done

if (( dry_run == 1 )); then
    echo "Dry run complete; no files were written."
    exit 0
fi

combined_csv="$project_root/allocation_all_packings_summary.csv"
"$python_cmd" - "$project_root" "$allocation_output_name" \
    "$thermal_output_name" "$combined_csv" "${seeds[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
allocation_name = sys.argv[2]
thermal_name = sys.argv[3]
output = Path(sys.argv[4])
seeds = sys.argv[5:]
rows = []
for seed in seeds:
    case = root / f"packing_seed_{seed}"
    allocation = json.loads(
        (case / allocation_name / "allocation_summary.json").read_text(encoding="utf-8")
    )
    thermal = json.loads(
        (case / thermal_name / "thermal_summary.json").read_text(encoding="utf-8")
    )
    rows.append({
        "seed": seed,
        "particles": allocation["particles"],
        "high_count": allocation["high_count"],
        "fixed_high_particle_volume_m3": allocation["fixed_high_particle_volume_m3"],
        "homogeneous_k_eff_W_mK": allocation["homogeneous_k_eff_W_mK"],
        "random_mean_k_eff_W_mK": allocation["random_mean_k_eff_W_mK"],
        "random_std_k_eff_W_mK": allocation["random_std_k_eff_W_mK"],
        "degree_k_eff_W_mK": allocation["degree_k_eff_W_mK"],
        "heat_path_k_eff_W_mK": allocation["heat_path_k_eff_W_mK"],
        "optimized_mean_k_eff_W_mK": allocation["optimized_mean_k_eff_W_mK"],
        "optimized_std_k_eff_W_mK": allocation["optimized_std_k_eff_W_mK"],
        "optimized_best_k_eff_W_mK": allocation["optimized_best_k_eff_W_mK"],
        "optimized_improvement_vs_random_mean_percent": allocation[
            "optimized_improvement_vs_random_mean_percent"
        ],
        "optimized_improvement_vs_homogeneous_percent": allocation[
            "optimized_improvement_vs_homogeneous_percent"
        ],
        "independent_k_eff_W_mK": thermal["effective_conductivity_W_mK"],
        "independent_energy_balance_error": thermal["relative_energy_balance_error"],
        "thermally_unanchored_particles": thermal["thermally_unanchored_particles"],
        "spanning_component_count": thermal["graph_diagnostics"]["spanning_component_count"],
    })

with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Combined summary written to: {output}")
PY

echo "All requested cases completed and independently verified."
