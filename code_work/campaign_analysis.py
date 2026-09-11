#!/usr/bin/env python3
"""Independent verification, mechanism diagnostics and overlap-floor sensitivity."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import run_allocation_pilot_stratified_v2 as pilot
import analyze_allocation_mechanisms as mechanism


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    case = root/args.case
    allocation = case/"allocation_final"
    summary = json.loads((allocation/"allocation_summary.json").read_text())
    if summary.get("thermal_model") != "circular_constriction_4a_v2":
        raise ValueError("Allocation was not produced with the corrected thermal model")
    quotas = {int(k):v for k,v in summary["type_quotas"].items()}
    mechanism.analyse.quotas = quotas
    solver = pilot.load_solver(Path(__file__).with_name("solve_packing_heat_transfer.py"))
    make_evaluator = lambda floor: pilot.ContactOnlyEvaluator(solver,case,1.,10.,301.,300.,floor,1e-8)
    evaluator = make_evaluator(1e-8)
    rng = np.random.default_rng(summary["seed"])
    random_samples = []
    for _ in range(summary["random_samples"]):
        ids = pilot.random_stratified(evaluator,quotas,rng)
        random_samples.append((ids,evaluator.evaluate(ids).k_eff))
    near_mean = min(random_samples,key=lambda pair:abs(pair[1]-summary["random_mean_k_eff_W_mK"]))[0]
    selections = {"random_near_mean":near_mean}
    for method, filename in [("degree","high_ids_degree.txt"),("heat_path","high_ids_heat_path.txt"),
                             ("swap_optimized","high_ids_optimized.txt")]:
        selections[method] = mechanism.read_ids(allocation/filename)
    fold = root/"ml"/args.case
    with (fold/"loocv_results.csv").open() as stream:
        ml_rows = {r["model"]:r for r in csv.DictReader(stream) if r["repeat"] == "ensemble"}
    for model in ("mlp","gnn"):
        selections[model] = mechanism.read_ids(fold/"predicted_ids"/f"{args.case}_{model}_ensemble.txt")
    output = root/"analysis"/args.case
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    for method, ids in selections.items():
        result = evaluator.evaluate(ids)
        expected = (float(ml_rows[method]["effective_conductivity_W_mK"]) if method in ml_rows else
                    summary.get({"swap_optimized":"optimized_best_k_eff_W_mK",
                                 "degree":"degree_k_eff_W_mK","heat_path":"heat_path_k_eff_W_mK"}.get(method,"")))
        if expected is not None and not math.isclose(result.k_eff,expected,rel_tol=1e-9):
            raise ValueError(f"Independent verification failed: {args.case} {method}")
        if result.balance_error > 1e-8:
            raise ValueError(f"Energy imbalance: {args.case} {method}")
        row = mechanism.analyse(method,evaluator,ids,output)
        row["case"] = args.case
        rows.append(row)
    independent = json.loads((case/"thermal_optimized"/"thermal_summary.json").read_text())
    if not math.isclose(independent["effective_conductivity_W_mK"],
                        summary["optimized_best_k_eff_W_mK"],rel_tol=1e-9):
        raise ValueError("Independent CLI thermal verification differs from search")
    write_csv(output/"mechanisms.csv",rows)
    contacts = []
    baseline = evaluator.evaluate([])
    for edge in baseline.edges:
        radius = min(evaluator.particles[edge.i].radius,evaluator.particles[edge.j].radius)
        contacts.append({"case":args.case,"id_i":int(evaluator.ids[edge.i]),
                         "id_j":int(evaluator.ids[edge.j]),"contact_radius_m":edge.contact_radius,
                         "a_over_min_radius":edge.contact_radius/radius,
                         "overlap_over_min_radius":edge.overlap/radius,
                         "at_overlap_floor":int(edge.overlap <= evaluator.overlap_floor)})
    write_csv(output/"contact_geometry.csv",contacts)
    # This is NUMERICAL regularization sensitivity, not stiffness validation.
    sensitivity = []
    nominal = {r["method"]:r["effective_conductivity_W_mK"] for r in rows}
    for floor in [1e-10,1e-8,1e-6]:
        alternate = make_evaluator(floor)
        for method,ids in selections.items():
            result = alternate.evaluate(ids)
            if result.balance_error > 1e-8:
                raise ValueError("Sensitivity solve failed conservation")
            sensitivity.append({"case":args.case,"method":method,"overlap_floor_ratio":floor,
                                "effective_conductivity_W_mK":result.k_eff,
                                "change_vs_nominal_percent":100*(result.k_eff/nominal[method]-1),
                                "energy_balance_error":result.balance_error})
    write_csv(output/"floor_sensitivity.csv",sensitivity)
    print(f"Verified {args.case}; wrote mechanisms, contact geometry and floor sensitivity")


if __name__ == "__main__":
    main()
